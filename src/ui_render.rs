//! Rasterizes the `bevy_ui` node tree with this crate's Vulkan backend.
//!
//! The `bevy_feathers_core` feature gives us feathers, `bevy_ui`, `bevy_ui_widgets` and
//! `bevy_text` with layout, picking and text shaping — but no `bevy_ui_render`. This module is
//! the missing half. Every frame it extracts the laid-out node tree into quads the way
//! `bevy_ui_render` does (background, border, outline, image, text glyphs), clips them against
//! `CalculatedClip`, and draws them into the swapchain with a port of bevy's `ui.wesl` SDF
//! shader. Textures (`ImageNode` images and glyph atlases) are plain `Image` assets, which
//! [`crate::render_texture`] already uploads and registers in the bindless set.
//!
//! Gradients (`BackgroundGradient` / `BorderGradient`) are a port of `gradient.wesl`: each color
//! stop pair becomes a segment quad, interpolated in the gradient's color space by the same
//! fragment shader — the color space is a per-vertex value here instead of a pipeline
//! specialization.
//!
//! Not yet drawn: `BoxShadow`, text shadows and decorations, sliced / tiled image modes (drawn
//! stretched), `OuterColor`.

use ash::vk;
use bevy::{
    asset::AssetId,
    camera::{
        Camera, CameraProjectionPlugin, ClearColor, RenderTarget, RenderTargetInfo,
        visibility::{InheritedVisibility, VisibilityPropagatePlugin},
    },
    color::{Hsla, Hsva, LinearRgba, Okhsla, Oklaba, Oklcha, Srgba},
    ecs::system::{SystemParam, lifetimeless::SRes},
    image::{Image, TextureAtlasLayout, TextureAtlasPlugin},
    input_focus::{InputDispatchPlugin, InputFocusPlugin},
    math::{Affine2, FloatOrd, Rect, Vec2},
    picking::DefaultPickingPlugins,
    prelude::*,
    sprite::BorderRect,
    text::{ComputedTextBlock, PositionedGlyph, TextColor, TextLayoutInfo, TextPlugin},
    ui::{
        BackgroundColor, BackgroundGradient, BorderColor, BorderGradient, CalculatedClip,
        ColorStop, ComputedNode, ComputedStackIndex, ComputedUiRenderTargetInfo, ConicGradient,
        Display, Gradient, InterpolationColorSpace, LinearGradient, Node, Outline, RadialGradient,
        ResolvedBorderRadius, UiGlobalTransform, UiPlugin, UiSystems, Val, VisualBox,
        widget::{ImageNode, ImageNodeSize, NodeImageMode},
    },
    ui_widgets::UiWidgetsPlugins,
    window::{PrimaryWindow, WindowRef},
};

use std::f32::consts::{FRAC_PI_2, TAU};

use crate::{
    assets::aurora_asset,
    ray_render_plugin::{RenderSet, TeardownSchedule},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

/// Shader flags; must align with `assets/shaders/ui.frag`.
pub mod shader_flags {
    pub const UNTEXTURED: u32 = 0;
    pub const TEXTURED: u32 = 1;
    pub const RADIAL: u32 = 16;
    pub const FILL_START: u32 = 32;
    pub const FILL_END: u32 = 64;
    pub const CONIC: u32 = 128;
    pub const BORDER_LEFT: u32 = 256;
    pub const BORDER_TOP: u32 = 512;
    pub const BORDER_RIGHT: u32 = 1024;
    pub const BORDER_BOTTOM: u32 = 2048;
    pub const BORDER_ALL: u32 = BORDER_LEFT | BORDER_TOP | BORDER_RIGHT | BORDER_BOTTOM;
    pub const INVERT: u32 = 4096;
    /// This crate's own: the fragment color comes from the gradient fields.
    pub const GRADIENT: u32 = 8192;
}

/// Z offsets within a stack index; same values as `bevy_ui_render::stack_z_offsets`.
mod z_offsets {
    pub const BACKGROUND_COLOR: f32 = 0.0;
    pub const BORDER: f32 = 0.01;
    pub const GRADIENT: f32 = 0.02;
    pub const BORDER_GRADIENT: f32 = 0.03;
    pub const IMAGE: f32 = 0.04;
    pub const TEXT: f32 = 0.06;
}

const QUAD_VERTEX_POSITIONS: [Vec2; 4] = [
    Vec2::new(-0.5, -0.5),
    Vec2::new(0.5, -0.5),
    Vec2::new(0.5, 0.5),
    Vec2::new(-0.5, 0.5),
];

/// One vertex as read by `ui.vert` through a buffer reference. Must match `ui_types.glsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub flags: u32,
    pub tex_index: u32,
    pub radius_x: [f32; 4],
    pub radius_y: [f32; 4],
    pub border: [f32; 4],
    pub size: [f32; 2],
    pub point: [f32; 2],
    // gradient segment (flags & GRADIENT); colors are in `color_space`
    pub g_start: [f32; 2],
    pub g_dir: [f32; 2],
    pub start_color: [f32; 4],
    pub end_color: [f32; 4],
    pub start_len: f32,
    pub end_len: f32,
    pub hint: f32,
    pub color_space: u32,
}

impl UiVertex {
    pub const ZERO: UiVertex = UiVertex {
        position: [0.0; 2],
        uv: [0.0; 2],
        color: [0.0; 4],
        flags: 0,
        tex_index: 0,
        radius_x: [0.0; 4],
        radius_y: [0.0; 4],
        border: [0.0; 4],
        size: [0.0; 2],
        point: [0.0; 2],
        g_start: [0.0; 2],
        g_dir: [0.0; 2],
        start_color: [0.0; 4],
        end_color: [0.0; 4],
        start_len: 0.0,
        end_len: 0.0,
        hint: 0.0,
        color_space: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiPushConstants {
    vertex_buffer: u64,
    screen_size: [f32; 2],
}

// ---------------------------------------------------------------------------------------------
// Pipeline asset
// ---------------------------------------------------------------------------------------------

/// The vertex/fragment shader pair that draws UI quads. Hot-reloads like the other pipelines.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct UiPipeline {
    #[dependency]
    pub vertex_shader: Handle<crate::shader::Shader>,
    #[dependency]
    pub fragment_shader: Handle<crate::shader::Shader>,
}

pub struct CompiledUiPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl VulkanAsset for UiPipeline {
    type ExtractedAsset = (crate::shader::Shader, crate::shader::Shader);
    type ExtractParam = SRes<Assets<crate::shader::Shader>>;
    type PreparedAsset = CompiledUiPipeline;

    fn extract_asset(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let shaders: &Assets<crate::shader::Shader> = &**param;
        let Some(vertex_shader) = shaders.get(&self.vertex_shader) else {
            log::warn!("UI vertex shader not ready yet");
            return None;
        };
        let Some(fragment_shader) = shaders.get(&self.fragment_shader) else {
            log::warn!("UI fragment shader not ready yet");
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
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<UiPushConstants>() as u32);

        // Set 0 is the crate-wide bindless texture set (binding 200).
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(
                &render_device.bindless_descriptor_set_layout,
            ))
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

        // Straight (non-premultiplied) alpha blending, like bevy_ui_render.
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

        log::debug!("UI pipeline compiled");
        CompiledUiPipeline {
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
    pipelines: Res<Assets<UiPipeline>>,
    mut shader_events: MessageReader<AssetEvent<crate::shader::Shader>>,
    mut parent_events: MessageWriter<AssetEvent<UiPipeline>>,
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
// Plugin
// ---------------------------------------------------------------------------------------------

/// Which pipeline draws the UI.
#[derive(Resource, Clone)]
pub struct UiRenderConfig {
    pub pipeline: Handle<UiPipeline>,
}

/// Adds the render-free bevy UI stack (text, layout, widgets, picking, focus) and the Vulkan
/// pass that draws it. Add after [`crate::ray_default_plugins::RayDefaultPlugins`]; add
/// `bevy::feathers::FeathersPlugins` yourself if you want feathers widgets.
pub struct UiRenderPlugin;

impl Plugin for UiRenderPlugin {
    fn build(&self, app: &mut App) {
        // Only the pieces the app has not already added: several of these pull each
        // other in (TextPlugin adds TextureAtlasPlugin, for example).
        if !app.is_plugin_added::<VisibilityPropagatePlugin>() {
            // Propagation only (`Visibility` -> `InheritedVisibility`, which UI nodes read):
            // the culling half of bevy's VisibilityPlugin has no job in a ray tracer, where
            // the acceleration structure is the culling structure.
            app.init_resource::<ClearColor>();
            app.add_plugins((CameraProjectionPlugin, VisibilityPropagatePlugin));
        }
        if !app.is_plugin_added::<TextPlugin>() {
            app.add_plugins(TextPlugin);
        }
        if !app.is_plugin_added::<TextureAtlasPlugin>() {
            app.add_plugins(TextureAtlasPlugin);
        }
        if !app.is_plugin_added::<UiPlugin>() {
            app.add_plugins(UiPlugin);
        }
        if !app.is_plugin_added::<InputFocusPlugin>() {
            app.add_plugins(InputFocusPlugin);
        }
        if !app.is_plugin_added::<InputDispatchPlugin>() {
            app.add_plugins(InputDispatchPlugin);
        }
        if !app.is_plugin_added::<bevy::picking::PickingPlugin>() {
            app.add_plugins(DefaultPickingPlugins);
        }
        if !app.is_plugin_added::<bevy::ui_widgets::ButtonPlugin>() {
            app.add_plugins(UiWidgetsPlugins);
        }

        app.init_asset::<UiPipeline>();
        app.init_vulkan_asset::<UiPipeline>();
        app.add_systems(Update, propagate_modified);
        app.add_systems(
            PostUpdate,
            ui_camera_target_system.before(UiSystems::Prepare),
        );

        let asset_server = app.world().get_resource::<AssetServer>().unwrap();
        let pipeline = UiPipeline {
            vertex_shader: asset_server.load(aurora_asset("shaders/ui.vert")),
            fragment_shader: asset_server.load(aurora_asset("shaders/ui.frag")),
        };
        let config = UiRenderConfig {
            pipeline: asset_server.add(pipeline),
        };
        app.insert_resource(config);

        app.init_resource::<ExtractedUi>();
        app.init_resource::<UiVertexBuffers>();
        app.add_systems(Last, extract_ui.in_set(RenderSet::Extract));
        app.add_systems(
            TeardownSchedule,
            cleanup_ui.before(crate::ray_render_plugin::on_shutdown),
        );
    }
}

/// `bevy_ui` sizes its layout from `Camera::computed.target_info`, which is normally filled in
/// by `bevy_render`'s `camera_system`. Without `bevy_render` nothing writes it, so mirror the
/// window-target half of that system here.
fn ui_camera_target_system(
    primary_window: Query<Entity, With<PrimaryWindow>>,
    windows: Query<&Window>,
    mut cameras: Query<(&mut Camera, Option<&RenderTarget>)>,
) {
    let primary_window = primary_window.iter().next();
    for (mut camera, render_target) in &mut cameras {
        let window_entity = match render_target {
            None | Some(RenderTarget::Window(WindowRef::Primary)) => primary_window,
            Some(RenderTarget::Window(WindowRef::Entity(entity))) => Some(*entity),
            Some(_) => None,
        };
        let Some(window) = window_entity.and_then(|entity| windows.get(entity).ok()) else {
            continue;
        };
        let physical_size = window.physical_size();
        let scale_factor = window.scale_factor();
        let unchanged = camera.computed.target_info.as_ref().is_some_and(|info| {
            info.physical_size == physical_size && info.scale_factor == scale_factor
        });
        if !unchanged {
            camera.computed.target_info = Some(RenderTargetInfo {
                physical_size,
                scale_factor,
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Extract
// ---------------------------------------------------------------------------------------------

/// One drawable rectangle of UI, in physical pixels.
pub struct UiQuad {
    pub z: f32,
    /// `None` for untextured nodes.
    pub image: Option<AssetId<Image>>,
    pub clip: Option<CalculatedClip>,
    /// Node-local (centered) space to screen space.
    pub transform: Affine2,
    pub item: UiItem,
}

pub enum UiItem {
    Node {
        color: [f32; 4],
        size: Vec2,
        /// Normalized texture coordinates for the four corners of the quad.
        uvs: [Vec2; 4],
        border_radius: [[f32; 4]; 2],
        border: [f32; 4],
        flags: u32,
    },
    Glyph {
        color: [f32; 4],
        /// Glyph center relative to the text block's content box origin.
        translation: Vec2,
        size: Vec2,
        uvs: [Vec2; 4],
    },
    /// A whole gradient; drawn as one segment quad per pair of adjacent stops.
    Gradient {
        size: Vec2,
        border_radius: [[f32; 4]; 2],
        border: [f32; 4],
        /// Border edge flags (for `BorderGradient`) plus `RADIAL` / `CONIC`.
        flags: u32,
        /// Start point (node-local, centered) — linear: the corner the line starts at,
        /// radial / conic: the center.
        g_start: Vec2,
        /// Linear: unit direction; radial: (x/y ratio, _); conic: (start angle, _).
        g_dir: Vec2,
        /// (color, distance along the gradient in physical pixels, hint) — resolved stops.
        stops: Vec<(LinearRgba, f32, f32)>,
        color_space: InterpolationColorSpace,
    },
}

/// The extracted UI for this frame, rebuilt from scratch every extract.
#[derive(Resource, Default)]
pub struct ExtractedUi {
    pub quads: Vec<UiQuad>,
    /// Physical size of the primary window, the space `bevy_ui` laid the nodes out in.
    pub window_size: Vec2,
}

type UiNodeQuery = (
    Option<&'static Node>,
    &'static ComputedNode,
    &'static ComputedStackIndex,
    &'static UiGlobalTransform,
    &'static InheritedVisibility,
    &'static ComputedUiRenderTargetInfo,
    Option<&'static CalculatedClip>,
    Option<&'static BackgroundColor>,
    Option<&'static BorderColor>,
    Option<&'static Outline>,
    Option<&'static BackgroundGradient>,
    Option<&'static BorderGradient>,
    Option<(&'static ImageNode, &'static ImageNodeSize)>,
    Option<(
        &'static ComputedTextBlock,
        &'static TextColor,
        &'static TextLayoutInfo,
    )>,
);

fn extract_ui(
    mut extracted: ResMut<ExtractedUi>,
    images: Res<Assets<Image>>,
    texture_atlases: Res<Assets<TextureAtlasLayout>>,
    text_colors: Query<&TextColor>,
    windows: Query<&Window, With<PrimaryWindow>>,
    nodes: Query<UiNodeQuery>,
    all_nodes: Query<(&ComputedNode, Option<&InheritedVisibility>)>,
    mut diag_tick: Local<u32>,
) {
    extracted.window_size = windows
        .iter()
        .next()
        .map(|w| w.physical_size().as_vec2())
        .unwrap_or(Vec2::ONE);
    let quads = &mut extracted.quads;
    quads.clear();

    for (
        node,
        uinode,
        stack_index,
        transform,
        inherited_visibility,
        target,
        clip,
        background_color,
        border_color,
        outline,
        background_gradient,
        border_gradient,
        image_node,
        text,
    ) in nodes.iter()
    {
        if !inherited_visibility.get() || node.is_some_and(|node| node.display == Display::None) {
            continue;
        }
        let z = |offset: f32| stack_index.0 as f32 + offset;
        let transform = Affine2::from(*transform);

        // Background
        if let Some(background_color) = background_color
            && !background_color.0.is_fully_transparent()
            && !uinode.is_empty()
        {
            quads.push(UiQuad {
                z: z(z_offsets::BACKGROUND_COLOR),
                image: None,
                clip: clip.cloned(),
                transform,
                item: UiItem::Node {
                    color: background_color.0.to_linear().to_f32_array(),
                    size: uinode.size(),
                    uvs: UNTEXTURED_UVS,
                    border_radius: uinode.border_radius().into(),
                    border: border_array(uinode.border()),
                    flags: shader_flags::UNTEXTURED,
                },
            });
        }

        // Gradients: one item per gradient, backgrounds first then border gradients.
        for (gradients, border_flags, z_offset) in [
            (background_gradient.map(|g| &g.0), 0, z_offsets::GRADIENT),
            (
                border_gradient.map(|g| &g.0),
                shader_flags::BORDER_ALL,
                z_offsets::BORDER_GRADIENT,
            ),
        ] {
            let Some(gradients) = gradients else {
                continue;
            };
            if uinode.is_empty() {
                continue;
            }
            for gradient in gradients {
                if let Some(item) = extract_gradient(gradient, uinode, target, border_flags) {
                    quads.push(UiQuad {
                        z: z(z_offset),
                        image: None,
                        clip: clip.cloned(),
                        transform,
                        item,
                    });
                }
            }
        }

        // Image
        if let Some((image, image_size)) = image_node
            && !image.color.is_fully_transparent()
            && !uinode.is_empty()
        {
            if let Some(quad) = extract_image(uinode, image, image_size, &texture_atlases) {
                quads.push(UiQuad {
                    z: z(z_offsets::IMAGE),
                    image: Some(image.image.id()),
                    clip: clip.cloned(),
                    transform: transform * Affine2::from_translation(quad.center),
                    item: quad.item,
                });
            }
        }

        // Border: one quad per distinct edge color, edges sharing a color are merged.
        if uinode.border() != BorderRect::ZERO
            && let Some(border_color) = border_color
        {
            let colors = [
                border_color.left.to_linear(),
                border_color.top.to_linear(),
                border_color.right.to_linear(),
                border_color.bottom.to_linear(),
            ];
            const EDGE_FLAGS: [u32; 4] = [
                shader_flags::BORDER_LEFT,
                shader_flags::BORDER_TOP,
                shader_flags::BORDER_RIGHT,
                shader_flags::BORDER_BOTTOM,
            ];
            let mut completed = 0;
            for (i, color) in colors.iter().enumerate() {
                if color.is_fully_transparent() || completed & EDGE_FLAGS[i] != 0 {
                    continue;
                }
                let mut flags = EDGE_FLAGS[i];
                for j in i + 1..4 {
                    if *color == colors[j] {
                        flags |= EDGE_FLAGS[j];
                    }
                }
                completed |= flags;
                quads.push(UiQuad {
                    z: z(z_offsets::BORDER),
                    image: None,
                    clip: clip.cloned(),
                    transform,
                    item: UiItem::Node {
                        color: color.to_f32_array(),
                        size: uinode.size(),
                        uvs: UNTEXTURED_UVS,
                        border_radius: uinode.border_radius().into(),
                        border: border_array(uinode.border()),
                        flags,
                    },
                });
            }
        }

        // Outline
        if uinode.outline_width() > 0.
            && let Some(outline) = outline
            && !outline.color.is_fully_transparent()
        {
            quads.push(UiQuad {
                z: z(z_offsets::BORDER),
                image: None,
                clip: clip.cloned(),
                transform,
                item: UiItem::Node {
                    color: outline.color.to_linear().to_f32_array(),
                    size: uinode.outlined_node_size(),
                    uvs: UNTEXTURED_UVS,
                    border_radius: uinode.outline_radius().into(),
                    border: [uinode.outline_width(); 4],
                    flags: shader_flags::BORDER_ALL,
                },
            });
        }

        // Text
        if let Some((block, text_color, layout)) = text
            && !uinode.is_empty()
        {
            let transform = transform * Affine2::from_translation(uinode.content_box().min);
            let mut color = text_color.0.to_linear();
            let mut current_section = 0;
            for PositionedGlyph {
                position,
                atlas_info,
                section_index,
                ..
            } in &layout.glyphs
            {
                if current_section != *section_index
                    && let Some(section_entity) = block
                        .entities()
                        .get(*section_index as usize)
                        .map(|t| t.entity)
                {
                    color = text_colors
                        .get(section_entity)
                        .map(|c| c.0.to_linear())
                        .unwrap_or_default();
                    current_section = *section_index;
                }
                let Some(atlas_size) = images.get(atlas_info.texture).map(|i| i.size().as_vec2())
                else {
                    continue;
                };
                let glyph_color = if atlas_info.is_alpha_mask {
                    color
                } else {
                    LinearRgba::WHITE
                };
                quads.push(UiQuad {
                    z: z(z_offsets::TEXT),
                    image: Some(atlas_info.texture),
                    clip: clip.cloned(),
                    transform,
                    item: UiItem::Glyph {
                        color: glyph_color.to_f32_array(),
                        translation: *position,
                        size: atlas_info.rect.size(),
                        uvs: rect_uvs(atlas_info.rect, atlas_size, false, false),
                    },
                });
            }
        }
    }

    *diag_tick += 1;
    if *diag_tick % 120 == 1 {
        let total = all_nodes.iter().count();
        let with_vis = all_nodes.iter().filter(|(_, v)| v.is_some()).count();
        let visible = all_nodes
            .iter()
            .filter(|(_, v)| v.is_some_and(|v| v.get()))
            .count();
        let nonempty = all_nodes.iter().filter(|(n, _)| !n.is_empty()).count();
        log::debug!(
            "ui extract: {total} nodes, {with_vis} with InheritedVisibility, {visible} visible, {nonempty} non-empty -> {} quads",
            quads.len()
        );
    }
}

const UNTEXTURED_UVS: [Vec2; 4] = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y];

fn border_array(border: BorderRect) -> [f32; 4] {
    [
        border.min_inset.x,
        border.min_inset.y,
        border.max_inset.x,
        border.max_inset.y,
    ]
}

/// Corner UVs for `rect` (in texture pixels) inside a texture of `extent` pixels.
fn rect_uvs(mut rect: Rect, extent: Vec2, flip_x: bool, flip_y: bool) -> [Vec2; 4] {
    if flip_x {
        std::mem::swap(&mut rect.min.x, &mut rect.max.x);
    }
    if flip_y {
        std::mem::swap(&mut rect.min.y, &mut rect.max.y);
    }
    [
        Vec2::new(rect.min.x, rect.min.y),
        Vec2::new(rect.max.x, rect.min.y),
        Vec2::new(rect.max.x, rect.max.y),
        Vec2::new(rect.min.x, rect.max.y),
    ]
    .map(|uv| uv / extent)
}

// ---------------------------------------------------------------------------------------------
// Gradients (port of bevy_ui_render::gradient)
// ---------------------------------------------------------------------------------------------

/// Resolves one `Gradient` on a node into a [`UiItem::Gradient`].
fn extract_gradient(
    gradient: &Gradient,
    uinode: &ComputedNode,
    target: &ComputedUiRenderTargetInfo,
    border_flags: u32,
) -> Option<UiItem> {
    if gradient.is_empty() {
        return None;
    }
    let size = uinode.size();
    let scale_factor = target.scale_factor();
    let target_size = target.physical_size().as_vec2();
    let corner_points = QUAD_VERTEX_POSITIONS.map(|p| p * size);
    let mut scratch = Vec::new();

    let linear = |angle: f32, stops: &[ColorStop]| {
        let length = compute_gradient_line_length(angle, size);
        let stops = compute_color_stops(
            stops,
            scale_factor,
            length,
            target_size,
            &mut Vec::new(),
            uinode.em_size,
            uinode.rem_size,
        );
        let corner_index = (angle - FRAC_PI_2).rem_euclid(TAU) / FRAC_PI_2;
        (
            corner_points[(corner_index as usize).min(3)],
            Vec2::new(angle.sin(), -angle.cos()),
            0,
            stops,
        )
    };

    let (g_start, g_dir, g_flags, stops, color_space) = if let Some(color) = gradient.get_single() {
        let (s, d, f, stops) = linear(
            0.0,
            &[
                ColorStop::new(color, Val::Percent(0.0)),
                ColorStop::new(color, Val::Percent(100.0)),
            ],
        );
        (s, d, f, stops, gradient.get_color_space())
    } else {
        match gradient {
            Gradient::Linear(LinearGradient {
                color_space,
                angle,
                stops,
            }) => {
                let (s, d, f, stops) = linear(*angle, stops);
                (s, d, f, stops, *color_space)
            }
            Gradient::Radial(RadialGradient {
                color_space,
                position,
                shape,
                stops,
            }) => {
                let center = position.resolve(
                    scale_factor,
                    size,
                    target_size,
                    uinode.em_size,
                    uinode.rem_size,
                );
                let shape_size = shape.resolve(
                    center,
                    scale_factor,
                    size,
                    target_size,
                    uinode.em_size,
                    uinode.rem_size,
                );
                let stops = compute_color_stops(
                    stops,
                    scale_factor,
                    shape_size.x,
                    target_size,
                    &mut scratch,
                    uinode.em_size,
                    uinode.rem_size,
                );
                let ratio = if shape_size.y != 0. {
                    shape_size.x / shape_size.y
                } else {
                    1.
                };
                (
                    center,
                    Vec2::splat(ratio),
                    shader_flags::RADIAL,
                    stops,
                    *color_space,
                )
            }
            Gradient::Conic(ConicGradient {
                color_space,
                start,
                position,
                stops,
            }) => {
                let center = position.resolve(
                    scale_factor,
                    size,
                    target_size,
                    uinode.em_size,
                    uinode.rem_size,
                );
                scratch.extend(stops.iter().filter_map(|stop| {
                    stop.angle
                        .map(|angle| (stop.color.to_linear(), angle.clamp(0., TAU), stop.hint))
                }));
                scratch.sort_by_key(|(_, angle, _)| FloatOrd(*angle));
                let mut sorted = scratch.drain(..);
                let mut resolved: Vec<_> = stops
                    .iter()
                    .map(|stop| {
                        if stop.angle.is_none() {
                            (stop.color.to_linear(), f32::NAN, stop.hint)
                        } else {
                            sorted.next().unwrap()
                        }
                    })
                    .collect();
                drop(sorted);
                interpolate_color_stops(&mut resolved, 0., TAU);
                (
                    center,
                    Vec2::new(*start, 0.),
                    shader_flags::CONIC,
                    resolved,
                    *color_space,
                )
            }
        }
    };

    if stops.len() < 2 {
        return None;
    }

    Some(UiItem::Gradient {
        size,
        border_radius: uinode.border_radius().into(),
        border: border_array(uinode.border()),
        flags: border_flags | g_flags,
        g_start,
        g_dir,
        stops,
        color_space,
    })
}

/// Length of the gradient line for a linear gradient at `angle` across a box of `size`.
fn compute_gradient_line_length(angle: f32, size: Vec2) -> f32 {
    let center = 0.5 * size;
    let v = Vec2::new(angle.sin(), -angle.cos());

    let (pos_corner, neg_corner) = if v.x >= 0.0 && v.y <= 0.0 {
        (size.with_y(0.), size.with_x(0.))
    } else if v.x >= 0.0 && v.y > 0.0 {
        (size, Vec2::ZERO)
    } else if v.x < 0.0 && v.y <= 0.0 {
        (Vec2::ZERO, size)
    } else {
        (size.with_x(0.), size.with_y(0.))
    };

    let t_pos = (pos_corner - center).dot(v);
    let t_neg = (neg_corner - center).dot(v);

    (t_pos - t_neg).abs()
}

/// Fills in the positions of `Auto` stops (NaN) by spreading them evenly between neighbours.
fn interpolate_color_stops(stops: &mut [(LinearRgba, f32, f32)], min: f32, max: f32) {
    if stops[0].1.is_nan() {
        stops[0].1 = min;
    }
    if stops.last().unwrap().1.is_nan() {
        stops.last_mut().unwrap().1 = max;
    }

    let mut i = 1;
    while i < stops.len() - 1 {
        if stops[i].1.is_nan() {
            let start = i;
            let mut end = i + 1;
            while end < stops.len() - 1 && stops[end].1.is_nan() {
                end += 1;
            }
            let start_point = stops[start - 1].1;
            let end_point = stops[end].1;
            let steps = end - start;
            let step = (end_point - start_point) / (steps + 1) as f32;
            for j in 0..steps {
                stops[i + j].1 = start_point + step * (j + 1) as f32;
            }
            i = end;
        }
        i += 1;
    }
}

/// Resolves stop positions to physical distances along a gradient of `length` pixels.
fn compute_color_stops(
    stops: &[ColorStop],
    scale_factor: f32,
    length: f32,
    target_size: Vec2,
    scratch: &mut Vec<(LinearRgba, f32, f32)>,
    em_size: bevy::text::EmSize,
    rem_size: bevy::text::RemSize,
) -> Vec<(LinearRgba, f32, f32)> {
    scratch.clear();
    scratch.extend(stops.iter().filter_map(|stop| {
        stop.point
            .resolve(scale_factor, length, target_size, em_size, rem_size)
            .ok()
            .map(|physical_point| (stop.color.to_linear(), physical_point, stop.hint))
    }));
    scratch.sort_by_key(|(_, point, _)| FloatOrd(*point));

    let min = scratch
        .first()
        .map(|(_, min, _)| *min)
        .unwrap_or(0.)
        .min(0.);
    let max = scratch
        .last()
        .map(|(_, max, _)| *max)
        .unwrap_or(length)
        .max(length);

    let mut sorted = scratch.drain(..);
    let mut resolved: Vec<_> = stops
        .iter()
        .map(|stop| {
            if stop.point == Val::Auto {
                (stop.color.to_linear(), f32::NAN, stop.hint)
            } else {
                sorted.next().unwrap()
            }
        })
        .collect();
    drop(sorted);

    interpolate_color_stops(&mut resolved, min, max);
    resolved
}

/// Index the fragment shader switches on; must align with the `CS_*` constants in `ui.frag`.
fn color_space_index(space: InterpolationColorSpace) -> u32 {
    match space {
        InterpolationColorSpace::LinearRgba => 0,
        InterpolationColorSpace::Srgba => 1,
        InterpolationColorSpace::Oklaba => 2,
        InterpolationColorSpace::Oklcha => 3,
        InterpolationColorSpace::OklchaLong => 4,
        InterpolationColorSpace::Okhsla => 5,
        InterpolationColorSpace::OkhslaLong => 6,
        InterpolationColorSpace::Hsla => 7,
        InterpolationColorSpace::HslaLong => 8,
        InterpolationColorSpace::Hsva => 9,
        InterpolationColorSpace::HsvaLong => 10,
    }
}

/// Converts a stop color into the gradient's interpolation space; hues are normalized to
/// `[0, 1)` for the shader.
fn convert_color_to_space(color: LinearRgba, space: InterpolationColorSpace) -> [f32; 4] {
    match space {
        InterpolationColorSpace::Oklaba => {
            let c: Oklaba = color.into();
            [c.lightness, c.a, c.b, c.alpha]
        }
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let c: Oklcha = color.into();
            [c.lightness, c.chroma, c.hue / 360., c.alpha]
        }
        InterpolationColorSpace::Okhsla | InterpolationColorSpace::OkhslaLong => {
            let c: Okhsla = color.into();
            [c.hue / 360., c.saturation, c.lightness, c.alpha]
        }
        InterpolationColorSpace::Srgba => {
            let c: Srgba = color.into();
            [c.red, c.green, c.blue, c.alpha]
        }
        InterpolationColorSpace::LinearRgba => color.to_f32_array(),
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let c: Hsla = color.into();
            [c.hue / 360., c.saturation, c.lightness, c.alpha]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let c: Hsva = color.into();
            [c.hue / 360., c.saturation, c.value, c.alpha]
        }
    }
}

struct ImageQuad {
    center: Vec2,
    item: UiItem,
}

/// Mirrors `bevy_ui_render::extract_uinode_images`, minus sliced / tiled modes.
fn extract_image(
    uinode: &ComputedNode,
    image: &ImageNode,
    image_size: &ImageNodeSize,
    texture_atlases: &Assets<TextureAtlasLayout>,
) -> Option<ImageQuad> {
    let half = 0.5 * uinode.size();
    let visual_box = match image.visual_box {
        VisualBox::ContentBox => uinode.content_box(),
        VisualBox::PaddingBox => Rect {
            min: -half + uinode.border().min_inset,
            max: half - uinode.border().max_inset,
        },
        VisualBox::BorderBox => Rect {
            min: -half,
            max: half,
        },
    };

    let atlas_rect = image
        .texture_atlas
        .as_ref()
        .and_then(|atlas| atlas.texture_rect(texture_atlases))
        .map(|r| r.as_rect());

    let source = atlas_rect
        .map(|r| r.size())
        .or_else(|| image.rect.map(|r| r.size()))
        .unwrap_or(image_size.size().as_vec2());

    let size = match image.image_mode {
        NodeImageMode::Auto if source.x > 0. && source.y > 0. => {
            source * (visual_box.size() / source).min_element()
        }
        _ => visual_box.size(),
    };
    if size.x <= 0. || size.y <= 0. {
        return None;
    }

    let mut inset = match image.visual_box {
        VisualBox::ContentBox => uinode.content_inset(),
        VisualBox::PaddingBox => uinode.border(),
        VisualBox::BorderBox => BorderRect::ZERO,
    };
    let image_inset = 0.5 * (visual_box.size() - size);
    inset.min_inset += image_inset;
    inset.max_inset += image_inset;

    let radius = uinode.border_radius();
    let clamped_radius = ResolvedBorderRadius {
        top_left: (radius.top_left - inset.min_inset).clamp(Vec2::ZERO, 0.5 * size),
        top_right: (radius.top_right - Vec2::new(inset.max_inset.x, inset.min_inset.y))
            .clamp(Vec2::ZERO, 0.5 * size),
        bottom_right: (radius.bottom_right - inset.max_inset).clamp(Vec2::ZERO, 0.5 * size),
        bottom_left: (radius.bottom_left - Vec2::new(inset.min_inset.x, inset.max_inset.y))
            .clamp(Vec2::ZERO, 0.5 * size),
    };

    let mut rect = match (atlas_rect, image.rect) {
        (None, None) => Rect {
            min: Vec2::ZERO,
            max: size,
        },
        (None, Some(image_rect)) => image_rect,
        (Some(atlas_rect), None) => atlas_rect,
        (Some(atlas_rect), Some(mut image_rect)) => {
            image_rect.min += atlas_rect.min;
            image_rect.max += atlas_rect.min;
            image_rect
        }
    };
    let atlas_scaling = if atlas_rect.is_some() || image.rect.is_some() {
        let scaling = size / rect.size();
        rect.min *= scaling;
        rect.max *= scaling;
        Some(scaling)
    } else {
        None
    };
    let atlas_extent = atlas_scaling
        .map(|scaling| image_size.size().as_vec2() * scaling)
        .unwrap_or(rect.max);

    Some(ImageQuad {
        center: visual_box.center(),
        item: UiItem::Node {
            color: image.color.to_linear().to_f32_array(),
            size,
            uvs: rect_uvs(rect, atlas_extent, image.flip_x, image.flip_y),
            border_radius: clamped_radius.into(),
            border: [0.0; 4],
            flags: shader_flags::TEXTURED,
        },
    })
}

// ---------------------------------------------------------------------------------------------
// Clipping (port of bevy_ui_render::clipping)
// ---------------------------------------------------------------------------------------------

/// Clips a convex polygon (screen-space positions with attached attributes) against every
/// inherited clip rect. Returns an empty polygon if nothing is visible.
fn clip_polygon<T: Copy>(
    clip: Option<&CalculatedClip>,
    vertices: &[(Vec2, T)],
    interpolate: impl Fn(T, T, f32) -> T + Copy,
) -> Vec<(Vec2, T)> {
    if vertices.len() < 3 {
        return Vec::new();
    }
    let Some(clip) = clip else {
        return vertices.to_vec();
    };
    let Some(rects) = clip.rects() else {
        return Vec::new();
    };

    let mut visible = vertices.to_vec();
    let mut scratch = Vec::with_capacity(visible.len() + 4);

    for region in rects {
        if visible.len() < 3 {
            break;
        }
        for (edge, distance_normal) in [
            (-region.rect.min.x, Vec2::X),
            (region.rect.max.x, Vec2::NEG_X),
            (region.rect.max.y, Vec2::NEG_Y),
            (-region.rect.min.y, Vec2::Y),
        ] {
            if edge.is_finite() {
                edge_clip(
                    &visible,
                    &mut scratch,
                    region.world_to_clip_local,
                    edge,
                    distance_normal,
                    interpolate,
                );
                std::mem::swap(&mut visible, &mut scratch);
            }
        }
    }

    if visible.len() < 3 {
        visible.clear();
    }
    visible
}

fn edge_clip<T: Copy>(
    input: &[(Vec2, T)],
    output: &mut Vec<(Vec2, T)>,
    world_to_clip: Affine2,
    edge: f32,
    distance_normal: Vec2,
    interpolate: impl Fn(T, T, f32) -> T + Copy,
) {
    output.clear();
    let Some(mut previous) = input.last().copied() else {
        return;
    };
    let mut previous_distance = world_to_clip
        .transform_point2(previous.0)
        .dot(distance_normal)
        + edge;
    let mut previous_visible = 0. <= previous_distance;

    for &vertex in input {
        let distance = world_to_clip
            .transform_point2(vertex.0)
            .dot(distance_normal)
            + edge;
        let visible = 0. <= distance;
        if visible != previous_visible {
            let t = previous_distance / (previous_distance - distance);
            output.push((
                previous.0.lerp(vertex.0, t),
                interpolate(previous.1, vertex.1, t),
            ));
        }
        if visible {
            output.push(vertex);
        }
        previous = vertex;
        previous_distance = distance;
        previous_visible = visible;
    }
}

// ---------------------------------------------------------------------------------------------
// Draw
// ---------------------------------------------------------------------------------------------

/// Per-frame-in-flight vertex storage, grown on demand.
#[derive(Resource, Default)]
pub struct UiVertexBuffers {
    buffers: [Buffer<UiVertex>; 2],
}

/// Everything [`draw_ui`] needs.
#[derive(SystemParam)]
pub struct UiDrawParams<'w, 's> {
    extracted: Res<'w, ExtractedUi>,
    buffers: ResMut<'w, UiVertexBuffers>,
    config: Option<Res<'w, UiRenderConfig>>,
    pipelines: Res<'w, VulkanAssets<UiPipeline>>,
    textures: Res<'w, VulkanAssets<Image>>,
    vertices: Local<'s, Vec<UiVertex>>,
    diag_tick: Local<'s, u32>,
}

/// Records the UI draw into `cmd_buffer`. Must be called inside the swapchain's dynamic
/// rendering pass, with viewport and scissor already set to the full swapchain.
pub unsafe fn draw_ui(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    frame_slot: usize,
    params: &mut UiDrawParams,
) {
    *params.diag_tick += 1;
    let diag = *params.diag_tick % 120 == 1;
    let Some(config) = params.config.as_ref() else {
        if diag {
            log::debug!("ui draw: no UiRenderConfig");
        }
        return;
    };
    let Some(pipeline) = params.pipelines.get(&config.pipeline) else {
        if diag {
            log::debug!("ui draw: UI pipeline not compiled yet");
        }
        return;
    };

    let vertices = &mut *params.vertices;
    vertices.clear();

    let mut quads: Vec<&UiQuad> = params.extracted.quads.iter().collect();
    quads.sort_by(|a, b| a.z.total_cmp(&b.z));

    for quad in quads {
        let tex_index = match quad.image {
            None => 0,
            Some(id) => match params.textures.get_by_id(id) {
                Some(texture) => render_device.register_bindless_texture(texture),
                // Not uploaded yet (atlas / image still in flight); skip this frame.
                None => continue,
            },
        };

        match &quad.item {
            UiItem::Node {
                color,
                size,
                uvs,
                border_radius,
                border,
                flags,
            } => {
                let points = QUAD_VERTEX_POSITIONS.map(|p| p * *size);
                let positions = points.map(|p| quad.transform.transform_point2(p));
                let polygon = clip_polygon(
                    quad.clip.as_ref(),
                    &[
                        (positions[0], (uvs[0], points[0])),
                        (positions[1], (uvs[1], points[1])),
                        (positions[2], (uvs[2], points[2])),
                        (positions[3], (uvs[3], points[3])),
                    ],
                    |a, b, t| (a.0.lerp(b.0, t), a.1.lerp(b.1, t)),
                );
                let flags = if quad.image.is_some() {
                    *flags | shader_flags::TEXTURED
                } else {
                    *flags
                };
                push_fan(vertices, &polygon, |(position, (uv, point))| UiVertex {
                    position: position.into(),
                    uv: uv.into(),
                    color: *color,
                    flags,
                    tex_index,
                    radius_x: border_radius[0],
                    radius_y: border_radius[1],
                    border: *border,
                    size: (*size).into(),
                    point: point.into(),
                    ..UiVertex::ZERO
                });
            }
            UiItem::Glyph {
                color,
                translation,
                size,
                uvs,
            } => {
                let positions = QUAD_VERTEX_POSITIONS
                    .map(|p| quad.transform.transform_point2(*translation + p * *size));
                let polygon = clip_polygon(
                    quad.clip.as_ref(),
                    &[
                        (positions[0], uvs[0]),
                        (positions[1], uvs[1]),
                        (positions[2], uvs[2]),
                        (positions[3], uvs[3]),
                    ],
                    Vec2::lerp,
                );
                push_fan(vertices, &polygon, |(position, uv)| UiVertex {
                    position: position.into(),
                    uv: uv.into(),
                    color: *color,
                    flags: shader_flags::TEXTURED,
                    tex_index,
                    size: (*size).into(),
                    ..UiVertex::ZERO
                });
            }
            UiItem::Gradient {
                size,
                border_radius,
                border,
                flags,
                g_start,
                g_dir,
                stops,
                color_space,
            } => {
                let points = QUAD_VERTEX_POSITIONS.map(|p| p * *size);
                let positions = points.map(|p| quad.transform.transform_point2(p));
                let polygon = clip_polygon(
                    quad.clip.as_ref(),
                    &[
                        (positions[0], points[0]),
                        (positions[1], points[1]),
                        (positions[2], points[2]),
                        (positions[3], points[3]),
                    ],
                    Vec2::lerp,
                );
                if polygon.is_empty() {
                    continue;
                }
                let flags = *flags | shader_flags::GRADIENT;
                let space = color_space_index(*color_space);

                // One segment per adjacent stop pair (bevy_ui_render::prepare_gradient).
                let mut segment_count = 0;
                for stop_index in 0..stops.len() - 1 {
                    let mut start_stop = stops[stop_index];
                    let end_stop = stops[stop_index + 1];
                    if start_stop.1 == end_stop.1 {
                        if stop_index == stops.len() - 2 {
                            if 0 < segment_count {
                                start_stop.0 = LinearRgba::NONE;
                            }
                        } else {
                            continue;
                        }
                    }
                    let start_color = convert_color_to_space(start_stop.0, *color_space);
                    let end_color = convert_color_to_space(end_stop.0, *color_space);
                    let mut stop_flags = flags;
                    if 0. < start_stop.1 && (stop_index == 0 || segment_count == 0) {
                        stop_flags |= shader_flags::FILL_START;
                    }
                    if stop_index == stops.len() - 2 {
                        stop_flags |= shader_flags::FILL_END;
                    }
                    push_fan(vertices, &polygon, |(position, point)| UiVertex {
                        position: position.into(),
                        flags: stop_flags,
                        radius_x: border_radius[0],
                        radius_y: border_radius[1],
                        border: *border,
                        size: (*size).into(),
                        point: point.into(),
                        g_start: (*g_start).into(),
                        g_dir: (*g_dir).into(),
                        start_color,
                        end_color,
                        start_len: start_stop.1,
                        end_len: end_stop.1,
                        hint: start_stop.2,
                        color_space: space,
                        ..UiVertex::ZERO
                    });
                    segment_count += 1;
                }
            }
        }
    }

    if vertices.is_empty() {
        if diag {
            log::debug!(
                "ui draw: {} quads produced no vertices",
                params.extracted.quads.len()
            );
        }
        return;
    }
    if diag {
        let v = &vertices[0];
        log::debug!(
            "ui draw: {} vertices from {} quads, swapchain {}x{}, window {:?}, v0 pos {:?} color {:?} flags {} size {:?} point {:?}",
            vertices.len(),
            params.extracted.quads.len(),
            extent.width,
            extent.height,
            params.extracted.window_size,
            v.position,
            v.color,
            v.flags,
            v.size,
            v.point
        );
    }

    // Upload into this frame's buffer, growing it if needed. The old buffer may still be in
    // flight, so it goes through the deferred destroyer.
    let buffer = &mut params.buffers.buffers[frame_slot % 2];
    if buffer.nr_elements < vertices.len() as u64 {
        if buffer.handle != vk::Buffer::null() {
            render_device.destroyer.destroy_buffer(buffer.handle);
        }
        let capacity = (vertices.len() * 2).max(4096) as u64;
        *buffer = render_device
            .create_host_buffer::<UiVertex>(capacity, vk::BufferUsageFlags::STORAGE_BUFFER);
    }
    {
        let mut mapped = render_device.map_buffer(buffer);
        mapped.copy_from_slice(vertices);
    }

    // Map through the window size bevy_ui laid out against, not the swapchain extent: if the
    // swapchain lags a resize, the UI stretches with the scene instead of drifting off it.
    let screen = params.extracted.window_size.max(Vec2::ONE);
    let push_constants = UiPushConstants {
        vertex_buffer: buffer.address,
        screen_size: [screen.x, screen.y],
    };

    unsafe {
        render_device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );
        render_device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            std::slice::from_ref(&render_device.bindless_descriptor_set),
            &[],
        );
        render_device.cmd_push_constants(
            cmd_buffer,
            pipeline.pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&push_constants),
        );
        render_device.cmd_draw(cmd_buffer, vertices.len() as u32, 1, 0, 0);
    }
}

/// Fan-triangulates a convex polygon into `out`.
fn push_fan<V: Copy>(out: &mut Vec<UiVertex>, polygon: &[V], vertex: impl Fn(V) -> UiVertex) {
    for i in 1..polygon.len().saturating_sub(1) {
        out.push(vertex(polygon[0]));
        out.push(vertex(polygon[i]));
        out.push(vertex(polygon[i + 1]));
    }
}

fn cleanup_ui(world: &mut World) {
    let Some(mut buffers) = world.remove_resource::<UiVertexBuffers>() else {
        return;
    };
    let render_device = world.resource::<RenderDevice>();
    for buffer in buffers.buffers.iter_mut() {
        if buffer.handle != vk::Buffer::null() {
            render_device.destroyer.destroy_buffer(buffer.handle);
            *buffer = Buffer::default();
        }
    }
}
