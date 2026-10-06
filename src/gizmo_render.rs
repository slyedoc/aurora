//! Draws `bevy_gizmos` debug lines with this crate's Vulkan backend.
//!
//! The fork's `bevy_gizmos` feature is render-free: the immediate-mode [`Gizmos`] system param,
//! the retained [`Gizmo`] component and the per-group storages fold every group's lines into
//! `Assets<GizmoAsset>` each frame (`update_gizmo_meshes`); everything that *draws* lives
//! behind `bevy_gizmos_render`, built on `bevy_render` and therefore not compiled here. This
//! module is the missing half, shaped exactly like [`crate::ui_render`]:
//! [`extract_gizmo_lines`] drains every group's segments (immediate-mode groups via the
//! [`GizmoHandles`] map, plus every retained [`Gizmo`] entity) into a flat world-space vertex
//! list, and [`draw_gizmos`] rasterizes each segment as a soft-edged screen-space quad of its
//! `line.width` (six vertices from the segment's two) inside the swapchain's dynamic
//! rendering pass — over the traced scene, under the UI.
//!
//! Add `bevy::gizmos::GizmoPlugin` (and any extra config groups) in the app, exactly as on
//! `bevy_render`. Without it this module idles: every extract param is `Option`-guarded, the
//! frame stays empty, and the draw records nothing.
//!
//! Lines are constant 1 px wide, never lit and never in the acceleration structure; line
//! width and `perspective` are ignored. The swapchain pass has no depth attachment -- the
//! scene is traced -- so `gizmo.frag` tests each line against the raygen's linear depth guide
//! instead, with `GizmoConfig::depth_bias` (and a retained `Gizmo`'s own) carried per vertex:
//! -1 always in front, 0 the plain test, towards 1 pushed behind.

use ash::vk;
use bevy::{
    camera::visibility::{InheritedVisibility, RenderLayers},
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
    swapchain::DISPLAY_FORMAT,
    tlas_builder::layers_mask,
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
    world::{InWorld, world_mask},
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
    /// The line's `depth_bias` (bevy's convention, see the module docs).
    pub depth_bias: f32,
    /// The line's width in pixels (its group's or retained gizmo's `line.width`).
    pub width: f32,
}

/// Must match `gizmo.vert`'s `Registers`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GizmoPushConstants {
    /// Unjittered clip-from-world (glam column order).
    view_proj: [f32; 16],
    vertex_buffer: u64,
    /// 1 / swapchain size: the fragment's position to the depth guide's uv.
    inv_extent: [f32; 2],
    /// 0 when no view traced, so `scene_depth` holds a placeholder and cannot be read.
    depth_guide: u32,
    _pad: u32,
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
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// The scene depth binding, `[frame_parity * MAX_VIEWS + view_slot]`: a set already
    /// recorded into this command buffer must not be rewritten.
    pub descriptor_sets: [vk::DescriptorSet; 2 * crate::MAX_VIEWS],
    /// Bound when no view traced: binding 0 needs a live image even though the fragment
    /// shader will not sample it.
    pub depth_placeholder: crate::render_texture::RenderTexture,
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

        let depth_placeholder = crate::render_texture::load_texture_from_bytes(
            render_device,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            &[0, 0, 0, 0],
            1,
            1,
        );

        let push_constant_info = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<GizmoPushConstants>() as u32);

        // The vertices arrive through a buffer reference in the push constants; the one
        // descriptor is the scene depth the fragment shader tests against.
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .binding(0)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            render_device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .unwrap()
        };
        let descriptor_sets = {
            let descriptor_pool = render_device.descriptor_pool.lock().unwrap();
            let layouts = [descriptor_set_layout; 2 * crate::MAX_VIEWS];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*descriptor_pool)
                .set_layouts(&layouts);
            unsafe {
                render_device
                    .allocate_descriptor_sets(&alloc_info)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
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
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
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

        let mut pipeline_rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[DISPLAY_FORMAT]);

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
            descriptor_set_layout,
            descriptor_sets,
            depth_placeholder,
        }
    }

    fn destroy_asset(render_device: &RenderDevice, prepared_asset: &Self::PreparedAsset) {
        render_device
            .destroyer
            .destroy_pipeline_layout(prepared_asset.pipeline_layout);
        render_device
            .destroyer
            .destroy_pipeline(prepared_asset.pipeline);
        render_device
            .destroyer
            .destroy_descriptor_set_layout(prepared_asset.descriptor_set_layout);
        render_device
            .destroyer
            .destroy_image_view(prepared_asset.depth_placeholder.image_view);
        render_device
            .destroyer
            .destroy_image(prepared_asset.depth_placeholder.image);
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
    /// Which views draw which vertices.
    pub batches: Vec<GizmoBatch>,
    /// Warn-once: the vertex cap was hit.
    warned_overflow: bool,
}

/// A stretch of [`GizmoLineFrame::vertices`] and the world mask a view must share to draw it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GizmoBatch {
    pub first: u32,
    pub count: u32,
    pub mask: u8,
}

/// `Last`, after `GizmoMeshSystems`: drain every gizmo group's lines into [`GizmoLineFrame`].
///
/// Immediate-mode lines land in the [`GizmoHandles`] map (the one place ALL groups land),
/// retained [`Gizmo`] entities carry their own asset handle plus a transform. Both stream
/// list (endpoint pairs) and strip (consecutive vertices, runs separated by NaN sentinels)
/// topologies; the finite check drops any pair touching a sentinel.
///
/// A line drawn for an entity (`GizmoBuffer::set_owner`) shows where its owner does: in the
/// owner's world and layers, and not while the owner is hidden or gone. An unowned line takes
/// its group's `render_layers`, or its retained entity's world.
fn extract_gizmo_lines(
    mut frame: ResMut<GizmoLineFrame>,
    handles: Option<Res<GizmoHandles>>,
    assets: Option<Res<Assets<GizmoAsset>>>,
    store: Option<Res<GizmoConfigStore>>,
    retained: Query<(Entity, &Gizmo, &GlobalTransform)>,
    owners: Query<(
        Option<&RenderLayers>,
        Option<&InWorld>,
        Option<&InheritedVisibility>,
    )>,
) {
    frame.vertices.clear();
    frame.batches.clear();
    let (Some(handles), Some(assets)) = (handles, assets) else {
        return;
    };
    let owner_mask = |owner: Entity| -> Option<u8> {
        let (layers, world, visibility) = owners.get(owner).ok()?;
        if visibility.is_some_and(|v| !v.get()) {
            return None;
        }
        Some(world_mask(layers, world))
    };

    // Immediate mode: every group's lines land in the handles map.
    for (type_id, handle) in handles.handles() {
        let Some(handle) = handle else { continue };
        let config = store.as_ref().and_then(|s| s.get_config_dyn(type_id));
        if config.is_some_and(|(config, _)| !config.enabled) {
            continue;
        }
        let depth_bias = config.map_or(0.0, |(config, _)| config.depth_bias);
        let width = config.map_or(2.0, |(config, _)| config.line.width);
        let group_mask = layers_mask(config.map(|(config, _)| &config.render_layers));
        let Some(asset) = assets.get(handle) else {
            continue;
        };
        append_buffer(
            &mut frame,
            asset.buffer().buffer(),
            None,
            depth_bias,
            width,
            &|owner| owner.map_or(Some(group_mask), owner_mask),
        );
    }

    // Retained `Gizmo` components: the entity owns whatever its asset does not give an owner.
    for (entity, gizmo, transform) in &retained {
        let Some(asset) = assets.get(&gizmo.handle) else {
            continue;
        };
        append_buffer(
            &mut frame,
            asset.buffer().buffer(),
            Some(transform),
            gizmo.depth_bias,
            gizmo.line_config.width,
            &|owner| owner_mask(owner.unwrap_or(entity)),
        );
    }
}

/// Add `count` vertices from `first` drawn by views sharing `mask`, extending the last batch when
/// it is the same mask and adjacent.
fn push_batch(batches: &mut Vec<GizmoBatch>, first: u32, count: u32, mask: u8) {
    if count == 0 {
        return;
    }
    if let Some(last) = batches.last_mut()
        && last.mask == mask
        && last.first + last.count == first
    {
        last.count += count;
        return;
    }
    batches.push(GizmoBatch { first, count, mask });
}

/// Fold one [`GizmoAsset`]'s list + strip streams into `frame.vertices`, applying `transform`
/// (a retained gizmo's placement) when present. `mask_of` gives each owner's world mask, or
/// `None` to leave its lines out.
fn append_buffer(
    frame: &mut GizmoLineFrame,
    buffer: GizmoBufferView<'_>,
    transform: Option<&GlobalTransform>,
    depth_bias: f32,
    width: f32,
    mask_of: &dyn Fn(Option<Entity>) -> Option<u8>,
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
    let push = |frame: &mut GizmoLineFrame, a: Vec3, b: Vec3, ca: LinearRgba, cb: LinearRgba| {
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
            depth_bias,
            width,
        });
        frame.vertices.push(GizmoVertex {
            position: b,
            color: pack_rgba8(cb),
            depth_bias,
            width,
        });
    };

    // Line list: consecutive endpoint PAIRS, one color per endpoint.
    let fallback = LinearRgba::WHITE;
    let list_color = |i: usize| buffer.list_colors.get(i).copied().unwrap_or(fallback);
    for (start, end, owner) in buffer.list_owners.ranges(buffer.list_positions.len()) {
        let Some(mask) = mask_of(owner) else { continue };
        let first = frame.vertices.len() as u32;
        let points = &buffer.list_positions[start..end];
        for (i, pair) in points.chunks_exact(2).enumerate() {
            let at = start + 2 * i;
            push(frame, pair[0], pair[1], list_color(at), list_color(at + 1));
        }
        let count = frame.vertices.len() as u32 - first;
        push_batch(&mut frame.batches, first, count, mask);
    }

    // Line strips: consecutive vertices, runs separated by NaN sentinels (one is pushed after
    // every strip) -- `world` rejects the sentinel, and `push` drops any pair touching one. An
    // owner changes only between strips, so no pair spans two owners.
    for (start, end, owner) in buffer.strip_owners.ranges(buffer.strip_positions.len()) {
        let Some(mask) = mask_of(owner) else { continue };
        let first = frame.vertices.len() as u32;
        for (i, pair) in buffer.strip_positions[start..end].windows(2).enumerate() {
            let at = start + i;
            let ca = buffer.strip_colors.get(at).copied().unwrap_or(fallback);
            let cb = buffer.strip_colors.get(at + 1).copied().unwrap_or(ca);
            push(frame, pair[0], pair[1], ca, cb);
        }
        let count = frame.vertices.len() as u32 - first;
        push_batch(&mut frame.batches, first, count, mask);
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
    pub frame: Res<'w, GizmoLineFrame>,
    buffers: ResMut<'w, GizmoVertexBuffers>,
    config: Option<Res<'w, GizmoRenderConfig>>,
    pipelines: Res<'w, VulkanAssets<GizmoPipeline>>,
}

/// Records the gizmo line draw into `cmd_buffer`. Must be called inside the swapchain's
/// dynamic rendering pass, with viewport and scissor already set to the full swapchain
/// (`extent`). `scene_depth` is the raygen's linear view depth in SHADER_READ_ONLY_OPTIMAL,
/// covering the same window. Draws only the lines whose world mask shares a bit with
/// `camera_mask`. Records nothing when there are no such lines or the pipeline is not compiled
/// yet.
pub unsafe fn draw_gizmos(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    view_proj: Mat4,
    extent: vk::Extent2D,
    scene_depth: Option<vk::ImageView>,
    frame_slot: usize,
    view_slot: usize,
    camera_mask: u32,
    params: &mut GizmoDrawParams,
) {
    let vertices = &params.frame.vertices;
    let visible = |batch: &&GizmoBatch| u32::from(batch.mask) & camera_mask != 0;
    if !params.frame.batches.iter().any(|batch| visible(&batch)) {
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
        inv_extent: [
            1.0 / extent.width.max(1) as f32,
            1.0 / extent.height.max(1) as f32,
        ],
        depth_guide: u32::from(scene_depth.is_some()),
        _pad: 0,
    };
    let set = pipeline.descriptor_sets[(frame_slot % 2) * crate::MAX_VIEWS + view_slot];
    let depth_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(scene_depth.unwrap_or(pipeline.depth_placeholder.image_view))
        .sampler(render_device.linear_sampler);
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&depth_info))];

    unsafe {
        render_device.update_descriptor_sets(&writes, &[]);
        render_device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );
        render_device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );
        render_device.cmd_push_constants(
            cmd_buffer,
            pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&push_constants),
        );
        // One draw per stretch this view's worlds share; `gl_VertexIndex` counts from `first`.
        for batch in params.frame.batches.iter().filter(visible) {
            // Two buffer vertices per segment, six drawn: the shader reads both ends itself.
            render_device.cmd_draw(cmd_buffer, batch.count / 2 * 6, 1, batch.first / 2 * 6, 0);
        }
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
/// [`AuroraDefaultPlugins`](crate::AuroraDefaultPlugins); add
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

    /// The vertex is a tightly-packed 24 bytes -- `gizmo.vert` reads a scalar-layout
    /// `GizmoVertex[]`, so any padding here would shear every vertex after the first.
    #[test]
    fn gizmo_vertex_matches_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GizmoVertex>(), 24);
        let vertex = GizmoVertex::zeroed();
        let base = &vertex as *const GizmoVertex as usize;
        assert_eq!(vertex.position.as_ptr() as usize - base, 0);
        assert_eq!(&vertex.color as *const u32 as usize - base, 12);
        assert_eq!(&vertex.depth_bias as *const f32 as usize - base, 16);
        assert_eq!(&vertex.width as *const f32 as usize - base, 20);
    }

    /// Scalar-layout mirror of `gizmo.vert`'s `Registers`: mat4 at 0, the buffer reference
    /// behind it, the extent, then the depth-guide flag. The shader's block is 84 bytes;
    /// the struct pads to 88 for its u64 alignment, which the trailing bytes cover.
    #[test]
    fn gizmo_push_constants_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<GizmoPushConstants>(), 88);
        let pc = GizmoPushConstants::zeroed();
        let base = &pc as *const GizmoPushConstants as usize;
        assert_eq!(pc.view_proj.as_ptr() as usize - base, 0);
        assert_eq!(&pc.vertex_buffer as *const u64 as usize - base, 64);
        assert_eq!(pc.inv_extent.as_ptr() as usize - base, 72);
        assert_eq!(&pc.depth_guide as *const u32 as usize - base, 80);
    }

    /// Owned lines take their owner's mask, a hidden owner drops its lines, unowned lines
    /// take the fallback, and neighbouring stretches with one mask merge.
    #[test]
    fn owners_split_lines_into_world_batches() {
        let owner = |i: u32| Entity::from_raw_u32(i).unwrap();
        let mut asset = GizmoAsset::default();
        asset.line(Vec3::ZERO, Vec3::X, LinearRgba::WHITE);
        asset.set_owner(Some(owner(1)));
        asset.line(Vec3::ZERO, Vec3::Y, LinearRgba::WHITE);
        asset.set_owner(Some(owner(2)));
        asset.line(Vec3::ZERO, Vec3::Z, LinearRgba::WHITE);
        asset.set_owner(Some(owner(3)));
        asset.line(Vec3::ZERO, Vec3::X, LinearRgba::WHITE);
        asset.linestrip([Vec3::ZERO, Vec3::X, Vec3::Y], LinearRgba::WHITE);
        asset.set_owner(None);
        asset.line(Vec3::ZERO, Vec3::Y, LinearRgba::WHITE);

        // 1 lives in world 2, 2 is hidden, 3 shares main's bit.
        let mask_of = |o: Option<Entity>| match o {
            None => Some(0b1),
            Some(e) if e == owner(1) => Some(0b100),
            Some(e) if e == owner(2) => None,
            Some(_) => Some(0b1),
        };
        let mut frame = GizmoLineFrame::default();
        append_buffer(&mut frame, asset.buffer().buffer(), None, 0.0, 2.0, &mask_of);
        assert_eq!(
            frame.batches,
            vec![
                GizmoBatch { first: 0, count: 2, mask: 0b1 },
                GizmoBatch { first: 2, count: 2, mask: 0b100 },
                // Owner 3's line, the unowned line and owner 3's strip (strips follow every
                // list line): one mask, merged.
                GizmoBatch { first: 4, count: 8, mask: 0b1 },
            ]
        );
        assert_eq!(frame.vertices.len(), 12);
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
        assert_eq!(
            pack_rgba8(LinearRgba::new(2.0, -1.0, 0.5, 0.0)),
            0x0080_00FF
        );
    }
}
