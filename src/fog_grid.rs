//! The fog's light cache: per camera, a froxel grid ([`Fog::grid_tile`] trace pixels per cell
//! across, [`Fog::grid_slices`] slices out to [`Fog::grid_distance`]) of the light the fog
//! scatters toward the camera. A headset's two eyes share one, laid out from their mid-point
//! over a frustum around both ([`grid_view`]). Per frame: `fog_reproject` (compute,
//! fog_grid.slang) carries every cell's history to where it lies now; the raygen dispatch
//! (`fog_grid` in raygen.slang) lights one cell of every [`Fog::grid_stride`]-square block
//! (suns, one sky ray, one light-table sample) into it; `fog_soften` blurs the grid once
//! across x and y; `fog_integrate` sums each column front to back, so a camera ray's fog is
//! two lookups (where it starts and ends), not shadow rays or a march. The sums and the soft
//! grid are 3D images the compute passes write in place (heap storage volumes) and the trace
//! samples through the bindless heap: the texture unit filters each lookup.
//!
//! [`Fog::grid_tile`]: crate::sky::Fog::grid_tile
//! [`Fog::grid_slices`]: crate::sky::Fog::grid_slices
//! [`Fog::grid_distance`]: crate::sky::Fog::grid_distance
//! [`Fog::grid_stride`]: crate::sky::Fog::grid_stride

use ash::vk;
use bevy::prelude::*;

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    descriptor_heap::{DescriptorHeaps, Mapping},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_texture::{RenderTexture, create_blank_volume},
    vk_init,
};

/// One cell: in-scattered rgb and its history length.
const CELL_BYTES: u64 = 16;

#[derive(Default)]
struct GridView {
    /// The lit grid, last frame's and this frame's by parity.
    buffers: [Buffer<u8>; 2],
    /// The lit grid blurred across x and y (`fog_integrate` reads it).
    soft: Buffer<u8>,
    /// The column sums (in-scatter and transmittance from the eye to each cell's far side,
    /// behind a first slice of nothing) and the soft grid, as 3D images: GENERAL while the
    /// compute passes write them, SHADER_READ_ONLY_OPTIMAL while the trace samples them.
    volumes: Option<[RenderTexture; 2]>,
    volume_tex: [u32; 2],
    cells: u32,
    /// Last frame's grid view-projection: where the update finds a cell's history.
    prev_view_proj: Option<Mat4>,
    /// Created and not yet zero-filled (a frame that never traced must not leave garbage
    /// for the next one to take as history).
    clear: bool,
}

impl GridView {
    fn free(&mut self, rd: &RenderDevice) {
        for b in self.buffers.iter().chain([&self.soft]) {
            if b.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(b.handle);
            }
        }
        for volume in self.volumes.take().into_iter().flatten() {
            rd.unregister_bindless_texture(&volume);
            rd.destroyer.destroy_image_view(volume.image_view);
            rd.destroyer.destroy_image(volume.image);
        }
    }
}

/// What the raygen gets for one grid this frame.
#[derive(Clone, Copy, Default)]
pub struct FogGridFrame {
    pub prev: u64,
    pub cur: u64,
    pub soft: u64,
    /// Bindless indices of the sum and soft volumes (sampled by the trace).
    pub sum_tex: u32,
    pub soft_tex: u32,
    /// Their storage descriptors' first index in the heap's volumes (written by compute).
    pub volumes: u32,
    pub cells: u32,
    /// The update lights one cell of every `stride` x `stride` block.
    pub stride: u32,
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
    module: Handle<ComputeModule>,
    views: [GridView; crate::MAX_VIEWS],
}

/// `FogGridPush` in fog_grid.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FogGridPush {
    uniforms: u64,
    base: u32,
    volumes: u32,
}

fn compute_to_trace(rd: &RenderDevice, cmd: vk::CommandBuffer) {
    memory_barrier(
        rd,
        cmd,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    );
}

impl FogGridState {
    /// Grid `slot`'s grid for a trace of `extent`, `tile` pixels per cell across and `slices`
    /// deep, lit `stride` cells apart, laid out over `(inverse_view, projection)` (`on` = its fog
    /// uses one): (re)created when the size changes and freed when off.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        extent: vk::Extent2D,
        tile: u32,
        slices: u32,
        stride: u32,
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
            view.free(rd);
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
            let create = || {
                if cells == 0 {
                    Buffer::default()
                } else {
                    rd.create_device_buffer(cells as u64 * CELL_BYTES, usage)
                }
            };
            view.buffers = [create(), create()];
            view.soft = create();
            if cells != 0 {
                let volume = |depth| {
                    create_blank_volume(
                        rd,
                        vk::Format::R32G32B32A32_SFLOAT,
                        vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST,
                        vk::ImageLayout::GENERAL,
                        size.x,
                        size.y,
                        depth,
                    )
                };
                let volumes = [volume(size.z + 1), volume(size.z)];
                view.volume_tex = volumes.map(|v| rd.register_bindless_texture(&v));
                view.volumes = Some(volumes);
            }
            view.cells = cells;
            view.clear = cells != 0;
        }
        let Some(volumes) = view.volumes.filter(|_| cells != 0) else {
            view.prev_view_proj = None;
            return FogGridFrame::default();
        };
        // By frame parity: the frame still in flight keeps reading its own slots.
        let volume_base = DescriptorHeaps::volume_base((frame & 1) as usize, slot);
        rd.heaps()
            .write_volumes(volume_base, &volumes.map(|v| v.image_view));
        let view_proj = projection * inverse_view.inverse();
        let prev_view_proj = view.prev_view_proj.replace(view_proj).unwrap_or(view_proj);
        let cur = (frame & 1) as usize;
        FogGridFrame {
            prev: view.buffers[cur ^ 1].address,
            cur: view.buffers[cur].address,
            soft: view.soft.address,
            sum_tex: view.volume_tex[0],
            soft_tex: view.volume_tex[1],
            volumes: volume_base,
            cells,
            stride: stride.max(1),
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
        view.free(rd);
        *view = GridView::default();
    }

    /// `fog_reproject` over every cell of `grid` (read through the view's `uniforms`), ahead of
    /// the lighting raygen; false (nothing recorded) until the kernels are compiled.
    pub fn record_reproject(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        uniforms: u64,
        grid: &FogGridFrame,
    ) -> bool {
        let Some(module) = modules.get(&self.module) else {
            return false;
        };
        let push = FogGridPush {
            uniforms,
            base: 0,
            volumes: grid.volumes,
        };
        let base = Some(std::mem::offset_of!(FogGridPush, base));
        record_dispatch(rd, cmd, module, "fog_reproject", &push, grid.cells, base);
        compute_to_trace(rd, cmd);
        true
    }

    /// After the lighting raygen: `fog_soften`, then `fog_integrate` over the columns, both
    /// writing grid `slot`'s volumes in place.
    #[allow(clippy::too_many_arguments)]
    pub fn record_integrate(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        uniforms: u64,
        grid: &FogGridFrame,
        slot: usize,
    ) {
        let Some(module) = modules.get(&self.module) else {
            return;
        };
        let push = FogGridPush {
            uniforms,
            base: 0,
            volumes: grid.volumes,
        };
        let base = Some(std::mem::offset_of!(FogGridPush, base));
        let Some(volumes) = self.views[slot].volumes else {
            return;
        };
        // The lighting's writes before the blur reads them; the volumes into GENERAL for the
        // writes below (every texel rewritten: last frame's contents are discarded).
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        volume_layouts(
            rd,
            cmd,
            &volumes,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            (
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::NONE,
            ),
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
            ),
        );
        record_dispatch(rd, cmd, module, "fog_soften", &push, grid.cells, base);
        compute_to_trace(rd, cmd);
        record_dispatch(
            rd,
            cmd,
            module,
            "fog_integrate",
            &push,
            grid.size.x * grid.size.y,
            base,
        );
        compute_to_trace(rd, cmd);
        volume_layouts(
            rd,
            cmd,
            &volumes,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
            ),
            (
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ,
            ),
        );
    }

    /// Zero-fills the view's grid in `cmd` if it was just created; before its trace.
    pub fn record_clear(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer, view: usize) {
        let view = &mut self.views[view];
        if !view.clear {
            return;
        }
        view.clear = false;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        unsafe {
            for b in view.buffers.iter().chain([&view.soft]) {
                rd.device
                    .cmd_fill_buffer(cmd, b.handle, 0, vk::WHOLE_SIZE, 0);
            }
            for volume in view.volumes.into_iter().flatten() {
                rd.device.cmd_clear_color_image(
                    cmd,
                    volume.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue::default(),
                    std::slice::from_ref(&range),
                );
            }
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
        if let Some(volumes) = view.volumes {
            volume_layouts(
                rd,
                cmd,
                &volumes,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                (
                    vk::PipelineStageFlags2::CLEAR,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                (
                    vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                    vk::AccessFlags2::SHADER_READ,
                ),
            );
        }
    }
}

/// Both volumes from layout `from` to `to`.
fn volume_layouts(
    rd: &RenderDevice,
    cmd: vk::CommandBuffer,
    volumes: &[RenderTexture; 2],
    from: vk::ImageLayout,
    to: vk::ImageLayout,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barriers = volumes.map(|v| {
        vk_init::layout_transition2(v.image, from, to)
            .src_stage_mask(src.0)
            .src_access_mask(src.1)
            .dst_stage_mask(dst.0)
            .dst_access_mask(dst.1)
    });
    unsafe {
        rd.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );
    }
}

fn cleanup_fog_grid(mut state: ResMut<FogGridState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        view.free(&render_device);
        *view = GridView::default();
    }
}

pub struct FogGridPlugin;

impl Plugin for FogGridPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/fog_grid.slang"));
        let module = asset_server.add(ComputeModule::with_heap(
            shader,
            &["fog_reproject", "fog_soften", "fog_integrate"],
            &[Mapping::Volumes { set: 0, binding: 0 }],
        ));
        app.insert_resource(FogGridState {
            module,
            views: std::array::from_fn(|_| GridView::default()),
        });
        app.add_systems(TeardownSchedule, cleanup_fog_grid.before(on_shutdown));
    }
}
