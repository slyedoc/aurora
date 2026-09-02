//! Draws `bevy_gizmos` debug lines with this crate's Vulkan backend.
//!
//! The fork's `bevy_gizmos` feature is render-free: the immediate-mode [`Gizmos`] system param,
//! the retained [`Gizmo`] component and the per-group storages fold every group's lines into
//! `Assets<GizmoAsset>` each frame (`update_gizmo_meshes`); everything that *draws* lives
//! behind `bevy_gizmos_render`, built on `bevy_render` and therefore not compiled here. This
//! module is the missing half, shaped exactly like [`crate::ui_render`]:
//! [`extract_gizmo_lines`] drains every group's segments (immediate-mode groups via the
//! [`GizmoHandles`] map, plus every retained [`Gizmo`] entity) into a flat world-space vertex
//! list, and [`draw_gizmos`] rasterizes it as a `LINE_LIST` inside the swapchain's dynamic
//! rendering pass — over the traced scene, under the UI.
//!
//! Add `bevy::gizmos::GizmoPlugin` (and any extra config groups) in the app, exactly as on
//! `bevy_render`. Without it this module idles: every extract param is `Option`-guarded, the
//! frame stays empty, and the draw records nothing.
//!
//! Lines are constant 1 px wide, never lit and never in the acceleration structure. The
//! swapchain pass has no depth attachment, so gizmos always paint over the scene:
//! `GizmoConfig::depth_bias`, line width and `perspective` are ignored.

use ash::vk;
use bevy::{
    camera::visibility::InheritedVisibility,
    color::LinearRgba,
    ecs::system::{SystemParam, lifetimeless::SRes},
    gizmos::{
        GizmoAsset, GizmoHandles, GizmoMeshSystems, config::GizmoConfigStore,
        gizmos::GizmoBufferView, retained::Gizmo,
    },
    prelude::*,
};

use crate::{
    assets::aurora_asset,
    ray_render_plugin::{RenderSet, TeardownSchedule},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

/// Hard cap on drawable line vertices per frame (2 per segment). Overflow drops the excess and
/// warns once — a million segments is ~32 MB of upload and far past the point where a
/// wireframe reads as anything.
pub const GIZMO_MAX_VERTICES: usize = 2 << 20;

/// One line-list vertex as read by `gizmo.vert` through a buffer reference.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GizmoVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Packed RGBA8 (`r | g<<8 | b<<16 | a<<24`), linear.
    pub color: u32,
}

/// Must match `gizmo.vert`'s `Registers`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GizmoPushConstants {
    /// Unjittered clip-from-world (glam column order).
    view_proj: [f32; 16],
    vertex_buffer: u64,
}

/// Linear RGBA → packed RGBA8, clamped — gizmo colors are debug paint, not radiance, so HDR
/// values saturate.
fn pack_rgba8(c: LinearRgba) -> u32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    q(c.red) | q(c.green) << 8 | q(c.blue) << 16 | q(c.alpha) << 24
}

// ---------------------------------------------------------------------------------------------
// Pipeline asset
// ---------------------------------------------------------------------------------------------

/// The vertex/fragment shader pair that draws gizmo lines. Hot-reloads like the UI pipeline.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct GizmoPipeline {
    #[dependency]
    pub vertex_shader: Handle<crate::shader::Shader>,
    #[dependency]
    pub fragment_shader: Handle<crate::shader::Shader>,
}

pub struct CompiledGizmoPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl VulkanAsset for GizmoPipeline {
    type ExtractedAsset = (crate::shader::Shader, crate::shader::Shader);
    type ExtractParam = SRes<Assets<crate::shader::Shader>>;
    type PreparedAsset = CompiledGizmoPipeline;

    fn extract_asset(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let shaders: &Assets<crate::shader::Shader> = &**param;
        let Some(vertex_shader) = shaders.get(&self.vertex_shader) else {
            log::warn!("gizmo vertex shader not ready yet");
            return None;
        };
        let Some(fragment_shader) = shaders.get(&self.fragment_shader) else {
            log::warn!("gizmo fragment shader not ready yet");
            return None;
        };
        Some((vertex_shader.clone(), fragment_shader.clone()))
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &RenderDevice,
    ) -> Self::PreparedAsset {
        let (vertex_shader, fragment_shader) = asset;

        let push_constant_info = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<GizmoPushConstants>() as u32);

        // No descriptor sets: the vertices arrive through a buffer reference in the push
        // constants, and the fragment shader samples nothing.
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&push_constant_info));
        let pipeline_layout = unsafe {
            render_device
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let shader_stages = [
            render_device.load_shader(&vertex_shader.spirv.unwrap(), vk::ShaderStageFlags::VERTEX),
            render_device.load_shader(
                &fragment_shader.spirv.unwrap(),
                vk::ShaderStageFlags::FRAGMENT,
            ),
        ];

        // Vertices are pulled from a buffer reference, so there is no vertex input state.
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::LINE_LIST);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE);
        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Straight (non-premultiplied) alpha blending, like the UI.
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&[vk::Format::B8G8R8A8_UNORM]);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut pipeline_rendering_info);

        let pipeline = unsafe {
            render_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .unwrap()[0];

        unsafe {
            render_device.destroy_shader_module(shader_stages[0].module, None);
            render_device.destroy_shader_module(shader_stages[1].module, None);
        }

        log::debug!("gizmo pipeline compiled");
        CompiledGizmoPipeline {
            pipeline,
            pipeline_layout,
        }
    }

    fn destroy_asset(render_device: &RenderDevice, prepared_asset: &Self::PreparedAsset) {
        render_device
            .destroyer
            .destroy_pipeline_layout(prepared_asset.pipeline_layout);
        render_device
            .destroyer
            .destroy_pipeline(prepared_asset.pipeline);
    }
}

fn propagate_modified(
    pipelines: Res<Assets<GizmoPipeline>>,
    mut shader_events: MessageReader<AssetEvent<crate::shader::Shader>>,
    mut parent_events: MessageWriter<AssetEvent<GizmoPipeline>>,
) {
    for event in shader_events.read() {
        if let AssetEvent::Modified { id } = event {
            for (parent_id, pipeline) in pipelines.iter() {
                if pipeline.vertex_shader.id() == *id || pipeline.fragment_shader.id() == *id {
                    parent_events.write(AssetEvent::Modified { id: parent_id });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------------------------

/// This frame's drained gizmo lines, ready to upload — the seam between the ECS drain and the
/// frame driver, exactly as [`ExtractedUi`](crate::ui_render::ExtractedUi) is for the UI.
/// Two vertices per segment, world space.
#[derive(Resource, Default)]
pub struct GizmoLineFrame {
    pub vertices: Vec<GizmoVertex>,
    /// Warn-once: the vertex cap was hit.
    warned_overflow: bool,
}

/// `Last`, after `GizmoMeshSystems`: drain every gizmo group's lines into [`GizmoLineFrame`].
///
/// Immediate-mode lines land in the [`GizmoHandles`] map (the one place ALL groups land),
/// retained [`Gizmo`] entities carry their own asset handle plus a transform. Both stream
/// list (endpoint pairs) and strip (consecutive vertices, runs separated by NaN sentinels)
/// topologies; the finite check drops any pair touching a sentinel.
fn extract_gizmo_lines(
    mut frame: ResMut<GizmoLineFrame>,
    handles: Option<Res<GizmoHandles>>,
    assets: Option<Res<Assets<GizmoAsset>>>,
    store: Option<Res<GizmoConfigStore>>,
    retained: Query<(&Gizmo, &GlobalTransform, Option<&InheritedVisibility>)>,
) {
    frame.vertices.clear();
    let (Some(handles), Some(assets)) = (handles, assets) else {
        return;
    };

    // Immediate mode: every group's lines land in the handles map.
    for (type_id, handle) in handles.handles() {
        let Some(handle) = handle else { continue };
        if store
            .as_ref()
            .and_then(|s| s.get_config_dyn(type_id))
            .is_some_and(|(config, _)| !config.enabled)
        {
            continue;
        }
        let Some(asset) = assets.get(handle) else {
            continue;
        };
        append_buffer(&mut frame, asset.buffer().buffer(), None);
    }

    // Retained `Gizmo` components.
    for (gizmo, transform, visibility) in &retained {
        if visibility.is_some_and(|v| !v.get()) {
            continue;
        }
        let Some(asset) = assets.get(&gizmo.handle) else {
            continue;
        };
        append_buffer(&mut frame, asset.buffer().buffer(), Some(transform));
    }
}

/// Fold one [`GizmoAsset`]'s list + strip streams into `frame.vertices`, applying `transform`
/// (a retained gizmo's placement) when present.
fn append_buffer(
    frame: &mut GizmoLineFrame,
    buffer: GizmoBufferView<'_>,
    transform: Option<&GlobalTransform>,
) {
    let world = |v: Vec3| -> Option<[f32; 3]> {
        if !v.is_finite() {
            return None;
        }
        Some(
            match transform {
                Some(t) => t.transform_point(v),
                None => v,
            }
            .to_array(),
        )
    };
    let mut push = |a: Vec3, b: Vec3, ca: LinearRgba, cb: LinearRgba| {
        if frame.vertices.len() + 2 > GIZMO_MAX_VERTICES {
            if !frame.warned_overflow {
                frame.warned_overflow = true;
                log::warn!(
                    "gizmos: over {GIZMO_MAX_VERTICES} line vertices this frame -- excess dropped"
                );
            }
            return;
        }
        let (Some(a), Some(b)) = (world(a), world(b)) else {
            return;
        };
        frame.vertices.push(GizmoVertex {
            position: a,
            color: pack_rgba8(ca),
        });
        frame.vertices.push(GizmoVertex {
            position: b,
            color: pack_rgba8(cb),
        });
    };

    // Line list: consecutive endpoint PAIRS, one color per endpoint.
    let fallback = LinearRgba::WHITE;
    let list_color = |i: usize| buffer.list_colors.get(i).copied().unwrap_or(fallback);
    for (i, points) in buffer.list_positions.chunks_exact(2).enumerate() {
        push(points[0], points[1], list_color(2 * i), list_color(2 * i + 1));
    }

    // Line strips: consecutive vertices, runs separated by NaN sentinels (one is pushed after
    // every strip) -- `world` rejects the sentinel, and `push` drops any pair touching one.
    for (i, pair) in buffer.strip_positions.windows(2).enumerate() {
        let ca = buffer.strip_colors.get(i).copied().unwrap_or(fallback);
        let cb = buffer.strip_colors.get(i + 1).copied().unwrap_or(ca);
        push(pair[0], pair[1], ca, cb);
    }
}

// ---------------------------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------------------------

/// Per-frame-in-flight vertex upload buffers, grown on demand.
#[derive(Resource, Default)]
pub struct GizmoVertexBuffers {
    buffers: [Buffer<GizmoVertex>; 2],
}

/// Which pipeline draws the gizmos.
#[derive(Resource, Clone)]
pub struct GizmoRenderConfig {
    pub pipeline: Handle<GizmoPipeline>,
}

/// Everything [`draw_gizmos`] needs.
#[derive(SystemParam)]
pub struct GizmoDrawParams<'w> {
    frame: Res<'w, GizmoLineFrame>,
    buffers: ResMut<'w, GizmoVertexBuffers>,
    config: Option<Res<'w, GizmoRenderConfig>>,
    pipelines: Res<'w, VulkanAssets<GizmoPipeline>>,
}

/// Records the gizmo line draw into `cmd_buffer`. Must be called inside the swapchain's
/// dynamic rendering pass, with viewport and scissor already set to the full swapchain.
/// Records nothing when there are no lines or the pipeline is not compiled yet.
pub unsafe fn draw_gizmos(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    view_proj: Mat4,
    frame_slot: usize,
    params: &mut GizmoDrawParams,
) {
    let vertices = &params.frame.vertices;
    if vertices.is_empty() {
        return;
    }
    let Some(config) = params.config.as_ref() else {
        return;
    };
    let Some(pipeline) = params.pipelines.get(&config.pipeline) else {
        return;
    };

    // Upload into this frame's buffer, growing it if needed. The old buffer may still be in
    // flight, so it goes through the deferred destroyer.
    let buffer = &mut params.buffers.buffers[frame_slot % 2];
    if buffer.nr_elements < vertices.len() as u64 {
        if buffer.handle != vk::Buffer::null() {
            render_device.destroyer.destroy_buffer(buffer.handle);
        }
        let capacity = (vertices.len() * 2).max(4096) as u64;
        *buffer = render_device
            .create_host_buffer::<GizmoVertex>(capacity, vk::BufferUsageFlags::STORAGE_BUFFER);
    }
    {
        let mut mapped = render_device.map_buffer(buffer);
        mapped.copy_from_slice(vertices);
    }

    let push_constants = GizmoPushConstants {
        view_proj: view_proj.to_cols_array(),
        vertex_buffer: buffer.address,
    };

    unsafe {
        render_device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );
        render_device.cmd_push_constants(
            cmd_buffer,
            pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytemuck::bytes_of(&push_constants),
        );
        render_device.cmd_draw(cmd_buffer, vertices.len() as u32, 1, 0, 0);
    }
}

fn cleanup_gizmos(world: &mut World) {
    let Some(mut buffers) = world.remove_resource::<GizmoVertexBuffers>() else {
        return;
    };
    let render_device = world.resource::<RenderDevice>();
    for buffer in buffers.buffers.iter_mut() {
        if buffer.handle != vk::Buffer::null() {
            render_device.destroyer.destroy_buffer(buffer.handle);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------------------------

/// The Vulkan draw half of `bevy_gizmos`. Part of
/// [`RayDefaultPlugins`](crate::ray_default_plugins::RayDefaultPlugins); add
/// `bevy::gizmos::GizmoPlugin` yourself to switch gizmos on (without it, this idles).
pub struct GizmoRenderPlugin;

impl Plugin for GizmoRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<GizmoPipeline>();
        app.init_vulkan_asset::<GizmoPipeline>();
        app.add_systems(Update, propagate_modified);

        let asset_server = app.world().get_resource::<AssetServer>().unwrap();
        let pipeline = GizmoPipeline {
            vertex_shader: asset_server.load(aurora_asset("shaders/gizmo.vert")),
            fragment_shader: asset_server.load(aurora_asset("shaders/gizmo.frag")),
        };
        let config = GizmoRenderConfig {
            pipeline: asset_server.add(pipeline),
        };
        app.insert_resource(config);

        app.init_resource::<GizmoLineFrame>();
        app.init_resource::<GizmoVertexBuffers>();
        // After every group's `update_gizmo_meshes` (the handles map is final), with the rest
        // of the frame's ECS reads.
        app.add_systems(
            Last,
            extract_gizmo_lines
                .after(GizmoMeshSystems)
                .in_set(RenderSet::Extract),
        );
        app.add_systems(
            TeardownSchedule,
            cleanup_gizmos.before(crate::ray_render_plugin::on_shutdown),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable as _;

    /// The vertex is a tightly-packed 16 bytes -- `gizmo.vert` reads a scalar-layout
    /// `GizmoVertex[]`, so any padding here would shear every vertex after the first.
    #[test]
    fn gizmo_vertex_matches_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GizmoVertex>(), 16);
        let vertex = GizmoVertex::zeroed();
        let base = &vertex as *const GizmoVertex as usize;
        assert_eq!(vertex.position.as_ptr() as usize - base, 0);
        assert_eq!(&vertex.color as *const u32 as usize - base, 12);
    }

    /// Scalar-layout mirror of `gizmo.vert`'s `Registers`: mat4 at 0, the buffer reference
    /// behind it, 72 bytes total.
    #[test]
    fn gizmo_push_constants_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GizmoPushConstants>(), 72);
        let pc = GizmoPushConstants::zeroed();
        let base = &pc as *const GizmoPushConstants as usize;
        assert_eq!(pc.view_proj.as_ptr() as usize - base, 0);
        assert_eq!(&pc.vertex_buffer as *const u64 as usize - base, 64);
    }

    /// Packed color order is `r | g<<8 | b<<16 | a<<24` with round-to-nearest, mirrored by the
    /// shader's `unpackUnorm4x8`.
    #[test]
    fn rgba8_packing_matches_the_shader() {
        assert_eq!(pack_rgba8(LinearRgba::WHITE), 0xFFFF_FFFF);
        assert_eq!(pack_rgba8(LinearRgba::rgb(1.0, 0.0, 0.0)), 0xFF00_00FF);
        assert_eq!(pack_rgba8(LinearRgba::rgb(0.0, 1.0, 0.0)), 0xFF00_FF00);
        assert_eq!(pack_rgba8(LinearRgba::rgb(0.0, 0.0, 1.0)), 0xFFFF_0000);
        // HDR clamps, negatives clamp, and alpha rides bits 24..32.
        assert_eq!(pack_rgba8(LinearRgba::new(2.0, -1.0, 0.5, 0.0)), 0x0080_00FF);
    }
}
