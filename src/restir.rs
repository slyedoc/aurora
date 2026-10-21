//! ReSTIR DI for the light table at the primary vertex, as passes of its own (RTXDI's shape).
//!
//! The primary pass (raygen.slang `primary`) writes each pixel's first shaded surface
//! ([`RestirSurface`] in rt_types.slang); the path tracer leaves the light table's direct light
//! there to this module and shades the winner. Per view, in the frame's command buffer:
//!
//! 1. `restir_grid` (compute, before the primary pass): a world grid of cells around the camera,
//!    each holding K light samples resampled from the power CDF by power / distance² at the
//!    cell (ReGIR), so a pixel's candidates are mostly lamps that can reach it. The trace's own
//!    path vertices past the primary draw their light candidates from it too.
//! 2. `restir_temporal` (compute): N candidates from the pixel's cell, RIS folded to one sample,
//!    merged with last frame's final reservoir at the reprojected pixel.
//! 3. `restir_spatial` (compute): merges random neighbours' reservoirs, with a boiling filter.
//! 4. The path tracer (raygen.slang `main`, `restirWinner`) at its primary vertex: the final
//!    sample's exact contribution, one shadow ray. A winner it finds occluded leaves W = 0 in its
//!    reservoir (M kept) for the next frame's temporal pass.
//!
//! Buffers live at trace resolution per view (each eye its own) and exist only while ReSTIR is on;
//! they are zero-filled on (re)creation, which also drops history on a resize.

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

/// `Reservoir` and `RestirSurface` in rt_types.slang.
const RESERVOIR_BYTES: u64 = 40;
const SURFACE_BYTES: u64 = 64;
/// The light grid's cells (x, y, z) around the camera.
const GRID: [u32; 3] = [32, 8, 32];
/// Power-CDF candidates resampled into each grid sample. Many: with hundreds of lamps of equal
/// power, a handful of draws left the one beside a cell out of it most frames (whole cells
/// strobing as it came and went).
const GRID_CANDIDATES: u32 = 32;

/// `RestirFrame` in restir.slang: everything the passes read, one per view per frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirFrameGpu {
    surfaces: u64,
    prev: u64,
    temporal: u64,
    final_: u64,
    grid: u64,
    lights: u64,
    instances: u64,
    positions: u64,
    prev_view_proj: [f32; 16],
    grid_origin: [f32; 3],
    grid_cell: f32,
    grid_dims: [u32; 3],
    grid_k: u32,
    width: u32,
    height: u32,
    frame: u32,
    epoch: u32,
    candidates: u32,
    m_cap: f32,
    spatial_samples: u32,
    spatial_radius: f32,
    boiling: f32,
    emissive_boost: f32,
    grid_candidates: u32,
    light_entries: u32,
    /// [`RenderSettings::ab`].
    ab: u32,
    pad: u32,
    /// `AuroraDebugView::Restir`'s buffer (after the matrix: it stays 16-byte aligned).
    debug: u64,
}

/// `RestirPush` in restir.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RestirPush {
    frame: u64,
    base: u32,
    pad: u32,
}

/// What a view's passes need this frame.
pub struct RestirInput<'a> {
    pub extent: vk::Extent2D,
    pub frame: u32,
    pub slot: usize,
    pub prev_view_proj: Mat4,
    pub camera: Vec3,
    pub epoch: u32,
    pub lights: u64,
    pub instances: u64,
    pub light_entries: u32,
    /// `AuroraDebugView::Restir`'s buffer (0 unless shown).
    pub debug: u64,
    pub settings: &'a RenderSettings,
}

/// The addresses the RT pipeline's push constants take: the surfaces the raygen writes, and the
/// reservoirs (last frame's final, this frame's final) the shade pass reads.
#[derive(Clone, Copy, Default)]
pub struct RestirAddresses {
    pub surfaces: u64,
    pub prev: u64,
    pub final_: u64,
    /// The parameter block (`RestirFrame`), for the raygen's light-grid lookups.
    pub frame: u64,
    /// The buffers were (re)made and cleared by this call: with the clears on the async compute
    /// queue, the frame runs without ReSTIR (the primary pass would race them).
    pub fresh: bool,
}

#[derive(Default)]
struct RestirView {
    surfaces: Buffer<u8>,
    /// Final reservoirs, ping-pong by frame parity, then the temporal pass's scratch.
    reservoirs: [Buffer<u8>; 3],
    grid: Buffer<[u32; 2]>,
    positions: Buffer<[f32; 4]>,
    entries: u32,
    frame: Buffer<RestirFrameGpu>,
    pixels: u64,
    grid_len: u64,
    threads: u32,
    spatial: bool,
}

impl RestirView {
    fn release(&mut self, rd: &RenderDevice) {
        for handle in [
            self.surfaces.handle,
            self.reservoirs[0].handle,
            self.reservoirs[1].handle,
            self.reservoirs[2].handle,
            self.grid.handle,
            self.positions.handle,
            self.frame.handle,
        ] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        *self = RestirView::default();
    }
}

#[derive(Resource)]
pub struct RestirState {
    module: Handle<ComputeModule>,
    views: [RestirView; crate::MAX_VIEWS],
}

impl RestirState {
    /// Ensures view `input.slot`'s buffers (zero-filled inside `cmd` when (re)created), writes
    /// this frame's parameter block, and returns the addresses for the RT push constants. Call
    /// after the frame fence (the block is host-visible and rewritten in place).
    pub fn prepare(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        input: &RestirInput,
    ) -> RestirAddresses {
        let settings = input.settings;
        let view = &mut self.views[input.slot];
        let pixels = input.extent.width as u64 * input.extent.height as u64;
        let k = settings.light_grid_samples.clamp(1, 64);
        let grid_len = (GRID[0] * GRID[1] * GRID[2] * k) as u64;
        let fresh = pixels != view.pixels || grid_len != view.grid_len;
        if fresh {
            view.release(rd);
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            view.surfaces = rd.create_device_buffer(pixels * SURFACE_BYTES, usage);
            view.reservoirs =
                std::array::from_fn(|_| rd.create_device_buffer(pixels * RESERVOIR_BYTES, usage));
            view.grid = rd.create_device_buffer(grid_len, usage);
            view.frame = rd.create_device_buffer(1, usage);
            for handle in [
                view.surfaces.handle,
                view.reservoirs[0].handle,
                view.reservoirs[1].handle,
                view.reservoirs[2].handle,
                view.grid.handle,
            ] {
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
            view.pixels = pixels;
            view.grid_len = grid_len;
            log::debug!(
                "restir: view {} buffers for {}x{} ({} MB)",
                input.slot,
                input.extent.width,
                input.extent.height,
                (pixels * (SURFACE_BYTES + 3 * RESERVOIR_BYTES) + grid_len * 8) / (1024 * 1024)
            );
        }
        if input.light_entries > view.entries {
            if view.positions.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(view.positions.handle);
            }
            view.entries = input.light_entries.next_power_of_two().max(64);
            view.positions =
                rd.create_device_buffer(view.entries as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
        }
        let cur = (input.frame & 1) as usize;
        let prev = view.reservoirs[cur ^ 1].address;
        let final_ = view.reservoirs[cur].address;
        view.spatial = settings.restir_spatial_samples > 0;
        let temporal = if view.spatial {
            view.reservoirs[2].address
        } else {
            final_
        };
        let cell = settings.light_grid_cell.max(0.1);
        let half = Vec3::new(GRID[0] as f32, GRID[1] as f32, GRID[2] as f32) * 0.5;
        let origin = ((input.camera / cell).floor() - half.floor()) * cell;
        view.threads = pixels as u32;
        let gpu = RestirFrameGpu {
            surfaces: view.surfaces.address,
            prev,
            temporal,
            final_,
            grid: view.grid.address,
            lights: input.lights,
            instances: input.instances,
            positions: view.positions.address,
            prev_view_proj: input.prev_view_proj.to_cols_array(),
            grid_origin: origin.to_array(),
            grid_cell: cell,
            grid_dims: GRID,
            grid_k: k,
            width: input.extent.width,
            height: input.extent.height,
            frame: input.frame,
            epoch: input.epoch,
            candidates: settings.restir_candidates.clamp(1, 32),
            m_cap: settings.restir_history.max(1.0),
            spatial_samples: settings.restir_spatial_samples.min(16),
            spatial_radius: settings.restir_spatial_radius.max(1.0),
            boiling: settings.restir_boiling.clamp(0.0, 1.0),
            emissive_boost: settings.emissive_boost.max(0.0),
            grid_candidates: GRID_CANDIDATES,
            light_entries: input.light_entries,
            ab: input.settings.ab,
            pad: 0,
            debug: input.debug,
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
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        RestirAddresses {
            surfaces: view.surfaces.address,
            prev,
            final_,
            frame: view.frame.address,
            fresh,
        }
    }

    /// Records the light grid for view `slot` (light positions, then the cells), BEFORE the
    /// main trace: its path vertices draw light candidates from it, and the temporal pass after.
    pub fn record_grid(
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
        let push = RestirPush {
            frame: view.frame.address,
            base: 0,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(RestirPush, base));
        let barrier = |dst: vk::PipelineStageFlags2| {
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                dst,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
        };
        let scope = prof.begin(rd, cmd, "restir grid");
        record_dispatch(
            rd,
            cmd,
            module,
            "restir_positions",
            &push,
            view.entries,
            base,
        );
        barrier(vk::PipelineStageFlags2::COMPUTE_SHADER);
        record_dispatch(
            rd,
            cmd,
            module,
            "restir_grid",
            &push,
            view.grid_len as u32,
            base,
        );
        prof.end(rd, cmd, scope);
        barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        );
        true
    }

    /// Records the reuse passes (temporal, spatial) for view `slot`, after the primary pass wrote
    /// its surfaces (the grid is already built) and before the path tracer shades the winners.
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
        let push = RestirPush {
            frame: view.frame.address,
            base: 0,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(RestirPush, base));
        let compute_barrier = |src: vk::PipelineStageFlags2| {
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
        // The main trace's surfaces (and last frame's shaded reservoirs) to compute.
        compute_barrier(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        let scope = prof.begin(rd, cmd, "restir temporal");
        record_dispatch(
            rd,
            cmd,
            module,
            "restir_temporal",
            &push,
            view.threads,
            base,
        );
        prof.end(rd, cmd, scope);
        compute_barrier(vk::PipelineStageFlags2::COMPUTE_SHADER);
        if view.spatial {
            let scope = prof.begin(rd, cmd, "restir spatial");
            record_dispatch(rd, cmd, module, "restir_spatial", &push, view.threads, base);
            prof.end(rd, cmd, scope);
            compute_barrier(vk::PipelineStageFlags2::COMPUTE_SHADER);
        }
        true
    }

    /// Frees view `slot`'s buffers (ReSTIR off, or the view gone).
    pub fn release(&mut self, rd: &RenderDevice, slot: usize) {
        if self.views[slot].pixels != 0 {
            self.views[slot].release(rd);
        }
    }
}

fn cleanup_restir(mut state: ResMut<RestirState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        view.release(&render_device);
    }
}

pub struct RestirPlugin;

impl Plugin for RestirPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/restir.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "restir_positions",
                "restir_grid",
                "restir_temporal",
                "restir_spatial",
            ],
        ));
        app.insert_resource(RestirState {
            module,
            views: std::array::from_fn(|_| RestirView::default()),
        });
        app.add_systems(TeardownSchedule, cleanup_restir.before(on_shutdown));
    }
}
