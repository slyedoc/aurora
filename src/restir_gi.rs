//! ReSTIR GI: reservoir resampling of the first indirect bounce at sample 0's primary vertex
//! (Ouyang et al. 2021), as passes beside ReSTIR DI (docs/plans/restir-gi.html).
//!
//! The path tracer (raygen.slang `main`) writes, for each pixel whose primary vertex is rough
//! and not glass, a candidate ([`GiCandidate`] in rt_types.slang): the secondary hit, the
//! radiance the rest of the path gathered there toward the vertex, and the direction's pdf; that
//! light leaves its colour. Per view, after the trace:
//!
//! 1. `gi_candidates` (compute): each candidate completed in place as a reservoir of one; without
//!    spatial reuse (the default) that is the final reservoir.
//! 2. `gi_spatial` (compute, `restir_gi_spatial_samples` > 0): pairwise MIS over random
//!    neighbours' candidates (reconnection shift, Jacobian-weighted).
//! 3. `gi_shade` (raygen.slang): the winner's light at the vertex through one visibility ray,
//!    added to the noisy colour and the exposure meter.
//!
//! No temporal reuse: samples held across frames read to Ray Reconstruction as detail and boil.
//! Buffers live at trace resolution per view and exist only while GI is on: the candidates and
//! the final reservoirs (64 B/px). GI uses ReSTIR DI's surface buffer when DI runs, else a
//! surface buffer of its own.

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    gpu_profiler::GpuProfiler,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_settings::RenderSettings,
};

/// `GiCandidate`, `GiReservoir` and `RestirSurface` in rt_types.slang.
const CANDIDATE_BYTES: u64 = 32;
const RESERVOIR_BYTES: u64 = 32;
const SURFACE_BYTES: u64 = 64;

/// `GiFrame` in restir_gi.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiFrameGpu {
    candidates: u64,
    surfaces: u64,
    final_: u64,
    pad_ptr: u64,
    width: u32,
    height: u32,
    frame: u32,
    spatial_samples: u32,
    spatial_radius: f32,
    boiling: f32,
    pad: [u32; 2],
}

/// `GiPush` in restir_gi.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiPush {
    frame: u64,
    base: u32,
    pad: u32,
}

/// What a view's passes need this frame.
pub struct GiInput<'a> {
    pub extent: vk::Extent2D,
    pub frame: u32,
    pub slot: usize,
    /// ReSTIR DI's surface buffer this frame (0: GI keeps its own).
    pub di_surfaces: u64,
    pub settings: &'a RenderSettings,
}

/// The addresses the RT pipeline's push constants take.
#[derive(Clone, Copy, Default)]
pub struct GiAddresses {
    /// The surfaces the primary pass writes (GI's own when DI is off; 0 when DI's are used).
    pub surfaces: u64,
    pub candidates: u64,
    pub final_: u64,
    /// The buffers were (re)made and cleared by this call: with the clears on the async compute
    /// queue, the frame runs without GI.
    pub fresh: bool,
}

#[derive(Default)]
struct GiView {
    candidates: Buffer<u8>,
    reservoirs: Buffer<u8>,
    /// Only while ReSTIR DI is off.
    surfaces: Buffer<u8>,
    frame: Buffer<GiFrameGpu>,
    pixels: u64,
    threads: u32,
    spatial: bool,
}

impl GiView {
    fn release(&mut self, rd: &RenderDevice) {
        for handle in [
            self.candidates.handle,
            self.reservoirs.handle,
            self.surfaces.handle,
            self.frame.handle,
        ] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        *self = GiView::default();
    }
}

#[derive(Resource)]
pub struct RestirGiState {
    module: Handle<ComputeModule>,
    views: [GiView; crate::MAX_VIEWS],
}

impl RestirGiState {
    /// Ensures view `input.slot`'s buffers (zero-filled inside `cmd` when (re)created), records
    /// this frame's parameter block, and returns the addresses for the RT push constants.
    pub fn prepare(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        input: &GiInput,
    ) -> GiAddresses {
        let settings = input.settings;
        let view = &mut self.views[input.slot];
        let pixels = input.extent.width as u64 * input.extent.height as u64;
        let own_surfaces = input.di_surfaces == 0;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let mut fresh = pixels != view.pixels;
        if fresh {
            view.release(rd);
            view.candidates = rd.create_device_buffer(pixels * CANDIDATE_BYTES, usage);
            view.reservoirs = rd.create_device_buffer(pixels * RESERVOIR_BYTES, usage);
            view.frame = rd.create_device_buffer(1, usage);
            view.pixels = pixels;
            log::debug!(
                "restir gi: view {} buffers for {}x{} ({} MB)",
                input.slot,
                input.extent.width,
                input.extent.height,
                pixels * (CANDIDATE_BYTES + RESERVOIR_BYTES) / (1024 * 1024)
            );
        }
        let surfaces_made = own_surfaces && view.surfaces.handle == vk::Buffer::null();
        if surfaces_made {
            view.surfaces = rd.create_device_buffer(pixels * SURFACE_BYTES, usage);
        } else if !own_surfaces && view.surfaces.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(view.surfaces.handle);
            view.surfaces = Buffer::default();
        }
        if fresh || surfaces_made {
            let mut handles = Vec::with_capacity(5);
            if fresh {
                handles.push(view.candidates.handle);
                handles.push(view.reservoirs.handle);
            }
            if surfaces_made {
                handles.push(view.surfaces.handle);
            }
            for handle in handles {
                unsafe { rd.device.cmd_fill_buffer(cmd, handle, 0, vk::WHOLE_SIZE, 0) };
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            fresh = true;
        }
        let surfaces = if own_surfaces {
            view.surfaces.address
        } else {
            input.di_surfaces
        };
        let final_ = view.reservoirs.address;
        view.spatial = settings.restir_gi_spatial_samples > 0;
        view.threads = pixels as u32;
        let gpu = GiFrameGpu {
            candidates: view.candidates.address,
            surfaces,
            final_,
            pad_ptr: 0,
            width: input.extent.width,
            height: input.extent.height,
            frame: input.frame,
            spatial_samples: settings.restir_gi_spatial_samples.min(16),
            spatial_radius: settings.restir_gi_spatial_radius.max(1.0),
            boiling: settings.restir_boiling.clamp(0.0, 1.0),
            pad: [0; 2],
        };
        // Recorded, not written through a mapping: the previous frame may still read it.
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, view.frame.handle, 0, bytemuck::bytes_of(&gpu));
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        GiAddresses {
            surfaces: if own_surfaces { surfaces } else { 0 },
            candidates: view.candidates.address,
            final_,
            fresh,
        }
    }

    /// Records the candidate pass and (when on) the spatial pass for view `slot`, after the path
    /// tracer wrote its candidates and before `gi_shade` shades the winners.
    pub fn record_passes(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        slot: usize,
        prof: &GpuProfiler,
    ) -> bool {
        let Some(module) = modules.get(&self.module) else {
            return false;
        };
        let view = &self.views[slot];
        if view.frame.address == 0 {
            return false;
        }
        let push = GiPush {
            frame: view.frame.address,
            base: 0,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(GiPush, base));
        let barrier = |src: vk::PipelineStageFlags2| {
            memory_barrier(
                rd,
                cmd,
                src,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
        };
        // The trace's candidates to compute.
        barrier(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let scope = prof.begin(rd, cmd, "gi candidates");
        record_dispatch(rd, cmd, module, "gi_candidates", &push, view.threads, base);
        prof.end(rd, cmd, scope);
        barrier(vk::PipelineStageFlags2::COMPUTE_SHADER);
        if view.spatial {
            let scope = prof.begin(rd, cmd, "gi spatial");
            record_dispatch(rd, cmd, module, "gi_spatial", &push, view.threads, base);
            prof.end(rd, cmd, scope);
            barrier(vk::PipelineStageFlags2::COMPUTE_SHADER);
        }
        true
    }

    /// Frees view `slot`'s buffers (GI off, or the view gone).
    pub fn release(&mut self, rd: &RenderDevice, slot: usize) {
        if self.views[slot].pixels != 0 {
            self.views[slot].release(rd);
        }
    }
}

fn cleanup_restir_gi(mut state: ResMut<RestirGiState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        view.release(&render_device);
    }
}

pub struct RestirGiPlugin;

impl Plugin for RestirGiPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/restir_gi.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["gi_candidates", "gi_spatial"]));
        app.insert_resource(RestirGiState {
            module,
            views: std::array::from_fn(|_| GiView::default()),
        });
        app.add_systems(TeardownSchedule, cleanup_restir_gi.before(on_shutdown));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gi_frame_layout() {
        use core::mem::offset_of;
        assert_eq!(offset_of!(GiFrameGpu, width), 32);
        assert_eq!(std::mem::size_of::<GiFrameGpu>(), 64);
    }
}
