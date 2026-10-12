//! The fog's light cache: per camera, a froxel grid ([`Fog::grid_tile`] trace pixels per cell
//! across, [`Fog::grid_slices`] slices out to [`Fog::grid_distance`]) of the light the fog
//! scatters toward the camera. A headset's two eyes share one, laid out from their mid-point
//! over a frustum around both ([`grid_view`]). Its own raygen dispatch (`fog_grid` in
//! raygen.slang, one thread per cell) lights every cell each frame (suns, one sky ray, one
//! light-table sample) into last frame's reprojected value; then each view's camera ray marches
//! this frame's grid: its fog costs lookups, not shadow rays.
//!
//! [`Fog::grid_tile`]: crate::sky::Fog::grid_tile
//! [`Fog::grid_slices`]: crate::sky::Fog::grid_slices
//! [`Fog::grid_distance`]: crate::sky::Fog::grid_distance

use ash::vk;
use bevy::prelude::*;

use crate::{
    compute::memory_barrier,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// One cell: in-scattered rgb and its history length.
const CELL_BYTES: u64 = 16;

#[derive(Default)]
struct GridView {
    buffers: [Buffer<u8>; 2],
    cells: u32,
    /// Last frame's grid view-projection: where the update finds a cell's history.
    prev_view_proj: Option<Mat4>,
    /// Created and not yet zero-filled (a frame that never traced must not leave garbage
    /// for the next one to take as history).
    clear: bool,
}

/// What the raygen gets for one grid this frame.
#[derive(Clone, Copy, Default)]
pub struct FogGridFrame {
    pub prev: u64,
    pub cur: u64,
    pub cells: u32,
    /// Last frame's buffer is not this grid's history (just created).
    pub fresh: bool,
    /// Cells across, down and deep.
    pub size: UVec3,
    pub inverse_view: Mat4,
    pub projection: Mat4,
    pub view_proj: Mat4,
    pub prev_view_proj: Mat4,
}

/// The grid's view for a camera's views (`(inverse_view, projection)`; one, or a headset's
/// two eyes): their mid-point, the first's orientation, and an off-centre frustum just
/// around all of theirs. One view is its own grid view.
pub fn grid_view(views: &[(Mat4, Mat4)]) -> (Mat4, Mat4) {
    if views.len() <= 1 {
        return views
            .first()
            .copied()
            .unwrap_or((Mat4::IDENTITY, Mat4::IDENTITY));
    }
    let centre = views.iter().map(|(v, _)| v.w_axis.truncate()).sum::<Vec3>() / views.len() as f32;
    let (_, rotation, _) = views[0].0.to_scale_rotation_translation();
    let inverse_view = Mat4::from_rotation_translation(rotation, centre);
    let to_grid = inverse_view.inverse();
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for (view, projection) in views {
        let unproject = projection.inverse();
        for corner in [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::ONE,
        ] {
            let p = unproject * corner.extend(1.0).extend(1.0);
            let d = to_grid.transform_vector3(view.transform_vector3(p.truncate() / p.w));
            let tan = d.truncate().xy() / (-d.z).max(1.0e-6);
            lo = lo.min(tan);
            hi = hi.max(tan);
        }
    }
    // Reverse-z, infinite far, as the views' own: only x, y and w matter to the grid.
    let near = views[0].1.w_axis.z;
    let projection = Mat4::from_cols(
        Vec4::new(2.0 / (hi.x - lo.x), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / (hi.y - lo.y), 0.0, 0.0),
        Vec4::new(
            (hi.x + lo.x) / (hi.x - lo.x),
            (hi.y + lo.y) / (hi.y - lo.y),
            0.0,
            -1.0,
        ),
        Vec4::new(0.0, 0.0, near, 0.0),
    );
    (inverse_view, projection)
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
    /// Grid `slot`'s grid for a trace of `extent`, `tile` pixels per cell across and `slices`
    /// deep, laid out over `(inverse_view, projection)` (`on` = its fog uses one): (re)created
    /// when the size changes and freed when off.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        extent: vk::Extent2D,
        tile: u32,
        slices: u32,
        (inverse_view, projection): (Mat4, Mat4),
        frame: u32,
        slot: usize,
        on: bool,
    ) -> FogGridFrame {
        let view = &mut self.views[slot];
        let tile = tile.max(1);
        let size = UVec3::new(
            (extent.width / tile).max(1),
            (extent.height / tile).max(1),
            slices.max(1),
        );
        let cells = if on { size.x * size.y * size.z } else { 0 };
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
            view.prev_view_proj = None;
            return FogGridFrame::default();
        }
        let view_proj = projection * inverse_view.inverse();
        let prev_view_proj = view.prev_view_proj.replace(view_proj).unwrap_or(view_proj);
        let cur = (frame & 1) as usize;
        FogGridFrame {
            prev: view.buffers[cur ^ 1].address,
            cur: view.buffers[cur].address,
            cells,
            fresh,
            size,
            inverse_view,
            projection,
            view_proj,
            prev_view_proj,
        }
    }
}

impl FogGridState {
    /// Frees grid `slot` (no camera's grid lives there this frame).
    pub fn release(&mut self, rd: &RenderDevice, slot: usize) {
        let view = &mut self.views[slot];
        if view.cells == 0 {
            return;
        }
        for b in &view.buffers {
            rd.destroyer.destroy_buffer(b.handle);
        }
        *view = GridView::default();
    }

    /// Zero-fills the view's grid in `cmd` if it was just created; before its trace.
    pub fn record_clear(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer, view: usize) {
        let view = &mut self.views[view];
        if !view.clear {
            return;
        }
        view.clear = false;
        unsafe {
            for b in &view.buffers {
                rd.device
                    .cmd_fill_buffer(cmd, b.handle, 0, vk::WHOLE_SIZE, 0);
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
