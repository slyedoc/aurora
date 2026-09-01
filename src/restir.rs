//! ReSTIR DI reservoir buffers (ping-pong, one per traced pixel).
//!
//! The raygen resamples the emissive-triangle light table at the primary vertex: RIS over
//! `restir_candidates` CDF samples, merged with last frame's reservoir at the reprojected
//! pixel (validated by frame, light-table epoch, depth and normal), one visibility ray for
//! the winner whose occlusion also zeroes the stored weight (visibility reuse). The spatial
//! reuse pass is a follow-up: it needs shading split out of the raygen behind a G-buffer.
//!
//! Buffers live at trace resolution (the DLSS render size when Ray Reconstruction is on)
//! and are zero-filled on (re)creation, which also drops history on any resize/mode change.

use ash::vk;
use bevy::prelude::*;

use crate::{
    compute::memory_barrier,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Must match `Reservoir` in types.glsl.
const RESERVOIR_BYTES: u64 = 32;

/// One view's ping-pong reservoir pair. Temporal reuse reprojects against that view's own
/// history, so each rendered view (each eye in XR) owns a separate pair.
#[derive(Default)]
struct ReservoirView {
    buffers: [Buffer<u8>; 2],
    bytes: u64,
}

#[derive(Resource, Default)]
pub struct RestirState {
    views: [ReservoirView; 2],
}

impl RestirState {
    /// Ensures the view's reservoir buffers cover `extent` (zero-filling on (re)creation
    /// inside `cmd`) and returns this frame's (previous, current) buffer addresses.
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        frame: u32,
        view: usize,
    ) -> (u64, u64) {
        let view = &mut self.views[view];
        let bytes = extent.width as u64 * extent.height as u64 * RESERVOIR_BYTES;
        if bytes != view.bytes {
            for b in &view.buffers {
                rd.destroyer.destroy_buffer(b.handle);
            }
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            view.buffers = [
                rd.create_device_buffer(bytes, usage),
                rd.create_device_buffer(bytes, usage),
            ];
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, view.buffers[0].handle, 0, vk::WHOLE_SIZE, 0);
                rd.device
                    .cmd_fill_buffer(cmd, view.buffers[1].handle, 0, vk::WHOLE_SIZE, 0);
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            view.bytes = bytes;
        }
        let cur = (frame & 1) as usize;
        (view.buffers[cur ^ 1].address, view.buffers[cur].address)
    }
}

fn cleanup_restir(mut state: ResMut<RestirState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        for b in &view.buffers {
            render_device.destroyer.destroy_buffer(b.handle);
        }
        *view = ReservoirView::default();
    }
}

pub struct RestirPlugin;

impl Plugin for RestirPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RestirState>();
        app.add_systems(TeardownSchedule, cleanup_restir.before(on_shutdown));
    }
}
