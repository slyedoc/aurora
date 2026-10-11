//! The fog's light cache: per view, a froxel grid (an eighth of the trace resolution across,
//! [`FOG_GRID_SLICES`] slices out to [`Fog::grid_distance`](crate::sky::Fog::grid_distance))
//! of the light the fog scatters toward the camera. Each pixel of the raygen lights one cell a
//! frame (suns, one sky ray, one light-table sample) into last frame's reprojected value, and
//! the camera ray marches last frame's grid: its fog costs lookups, not shadow rays.

use ash::vk;
use bevy::prelude::*;

use crate::{
    compute::memory_barrier,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Trace pixels per cell across; must match `FOG_GRID_TILE` in raygen.slang.
pub const FOG_GRID_TILE: u32 = 8;
/// Depth slices; must match `FOG_GRID_SLICES` in raygen.slang.
pub const FOG_GRID_SLICES: u32 = 64;
/// One cell: in-scattered rgb and its history length.
const CELL_BYTES: u64 = 16;

#[derive(Default)]
struct GridView {
    buffers: [Buffer<u8>; 2],
    cells: u32,
    /// Created and not yet zero-filled (a frame that never traced must not leave garbage
    /// for the next one to take as history).
    clear: bool,
}

/// What the raygen gets for one view this frame.
#[derive(Clone, Copy, Default)]
pub struct FogGridFrame {
    pub prev: u64,
    pub cur: u64,
    pub cells: u32,
    /// Last frame's buffer is not this grid's history (just created).
    pub fresh: bool,
}

#[derive(Resource)]
pub struct FogGridState {
    views: [GridView; crate::MAX_VIEWS],
}

impl Default for FogGridState {
    fn default() -> Self {
        Self {
            views: std::array::from_fn(|_| GridView::default()),
        }
    }
}

impl FogGridState {
    /// The view's grid for a trace of `extent` (`on` = its fog uses one), (re)created when the
    /// size changes and freed when off.
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        extent: vk::Extent2D,
        frame: u32,
        view: usize,
        on: bool,
    ) -> FogGridFrame {
        let view = &mut self.views[view];
        let cells = if on {
            (extent.width / FOG_GRID_TILE).max(1)
                * (extent.height / FOG_GRID_TILE).max(1)
                * FOG_GRID_SLICES
        } else {
            0
        };
        let fresh = cells != view.cells;
        if fresh {
            for b in &view.buffers {
                rd.destroyer.destroy_buffer(b.handle);
            }
            view.buffers = if cells == 0 {
                Default::default()
            } else {
                let bytes = cells as u64 * CELL_BYTES;
                let usage =
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
                [
                    rd.create_device_buffer(bytes, usage),
                    rd.create_device_buffer(bytes, usage),
                ]
            };
            view.cells = cells;
            view.clear = cells != 0;
        }
        if cells == 0 {
            return FogGridFrame::default();
        }
        let cur = (frame & 1) as usize;
        FogGridFrame {
            prev: view.buffers[cur ^ 1].address,
            cur: view.buffers[cur].address,
            cells,
            fresh,
        }
    }
}

impl FogGridState {
    /// Zero-fills the view's grid in `cmd` if it was just created; before its trace.
    pub fn record_clear(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer, view: usize) {
        let view = &mut self.views[view];
        if !view.clear {
            return;
        }
        view.clear = false;
        unsafe {
            for b in &view.buffers {
                rd.device.cmd_fill_buffer(cmd, b.handle, 0, vk::WHOLE_SIZE, 0);
            }
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
    }
}

fn cleanup_fog_grid(mut state: ResMut<FogGridState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        for b in &view.buffers {
            render_device.destroyer.destroy_buffer(b.handle);
        }
        *view = GridView::default();
    }
}

pub struct FogGridPlugin;

impl Plugin for FogGridPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FogGridState>();
        app.add_systems(TeardownSchedule, cleanup_fog_grid.before(on_shutdown));
    }
}
