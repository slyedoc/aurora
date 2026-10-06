/// How many views one frame can render.
///
/// A view owns per-view GPU state -- uniform buffer, descriptor set, ReSTIR reservoir pair,
/// DLSS feature and its temporal history -- so views do NOT share denoiser history and each
/// costs its own raygen dispatch. What they DO share is everything world-scale: the TLAS,
/// the light table, GPU transforms and the SHARC cache are built once per frame regardless.
/// So frame cost tracks total PIXELS traced, not view count.
///
/// Slots are allocated per frame in order. Four covers the cases that exist: an editor
/// viewport plus two XR eyes, with one spare for a second viewport or a preview camera.
/// Raising it costs one descriptor set per frame-in-flight per slot and nothing else --
/// reservoirs and DLSS features are allocated lazily, so an unused slot holds no memory.
pub const MAX_VIEWS: usize = 4;

pub mod aftermath;
pub mod animclip;
pub mod assets;
pub mod atmosphere;
pub mod auto_exposure;
pub mod blas;
pub mod bluenoise_plugin;
pub mod bsn;
pub mod camera_target;
pub mod collision;
pub mod color_target;
pub mod compute;
pub mod debug_view;
pub mod dev_ui;
pub mod dlss;
pub mod env_light;
pub mod environment;
pub mod gizmo_render;
pub mod gpu_transform;
pub mod lights;
pub mod material;
pub mod mesh;
pub mod omm;
pub mod picking;
pub mod pointer_picking;
pub mod portal;
pub mod post_process_filter;
pub mod procedural_mesh;
pub mod ray_render_plugin;
pub mod raytracing_pipeline;
pub mod render_buffer;
pub mod render_device;
pub mod render_env;
pub mod render_shaders;
pub mod render_texture;
pub mod restir;
pub mod sbt;
pub mod shader;
pub mod sharc;
pub mod skinning;
pub mod sky;
pub mod sphere;
pub mod surface_group;
pub mod swapchain;
pub mod terrain;
pub mod tlas_builder;
pub mod transform;
pub mod ui_panel;
pub mod ui_render;
pub mod util;
pub mod vk_init;
pub mod vk_utils;
pub mod vulkan_asset;
pub mod xr;

use bevy::{app::PluginGroupBuilder, prelude::*};

/// What an app writes against: `use bevy_aurora::prelude::*;`
///
/// Scene-facing types only -- components you spawn, resources you tune, the plugin group,
/// and the app-level helpers. The renderer's own machinery (pipelines, buffers, extracted
/// mirrors, the per-system plugins [`AuroraDefaultPlugins`] already adds) stays behind its
/// module path.
///
/// Three names are held back because they would shadow `bevy::prelude`:
/// `material::AlphaMode`, `sphere::Sphere` and `transform::TransformPlugin`. Import those by
/// path and alias them.
pub mod prelude {
    pub use crate::{
        AuroraDefaultPlugins,
        animclip::AnimationTargetsByName,
        assets::aurora_asset,
        atmosphere::{Atmosphere, CloudLayer},
        auto_exposure::{AuroraExposure, FixedExposure, ev_from_ev100, ev100_from_ev},
        collision::{CollisionMesh, CollisionShape},
        compute::{ComputeModule, ComputeModules},
        debug_view::AuroraDebugView,
        dev_ui::{DevUIPanel, DevUIPlugin, DevUIState},
        dlss::{AuroraDlss, RrPreset},
        env_light::EnvLight,
        environment::{InEnvironment, RenderEnvironments},
        material::{AuroraMaterial, AuroraMaterial3d},
        mesh::{AuroraMesh, AuroraMesh3d},
        picking::{RayCaster, RayHit, RayHits},
        portal::AuroraPortal,
        procedural_mesh::{ProceduralKernels, ProceduralMesh, ProceduralMesh3d},
        skinning::{SkinJointsByName, Wind, WindSway},
        sky::{EnvironmentSkies, GradientSky, Sky},
        ui_panel::{InspectorPanel3d, UiPanel3d},
        util::{ScreenshotExt, TimeoutAppExt},
        xr::{XrHand, XrHandState, XrInput, XrPose, XrState, XrTracked},
    };
}

pub struct AuroraDefaultPlugins;

impl PluginGroup for AuroraDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        let mut group = PluginGroupBuilder::start::<Self>();
        group = group
            // Before AssetPlugin: registers the `aurora://` source for the engine's own assets.
            .add(crate::assets::AuroraAssetSourcePlugin)
            .add(bevy::log::LogPlugin::default())
            .add(bevy::app::TaskPoolPlugin::default())
            //.add(bevy::app::TypeRegistrationPlugin)
            .add(bevy::diagnostic::FrameCountPlugin)
            .add(bevy::time::TimePlugin)
            // Root sync only: the hierarchy is propagated on the GPU (see transform.rs).
            .add(crate::transform::TransformPlugin::default())
            //            .add(bevy::hierarchy::HierarchyPlugin)
            .add(bevy::diagnostic::DiagnosticsPlugin)
            .add(bevy::input::InputPlugin)
            .add(bevy::window::WindowPlugin {
                close_when_requested: false,
                ..default()
            })
            .add(bevy::a11y::AccessibilityPlugin);

        group = group.add(bevy::asset::AssetPlugin::default());
        group = group.add(bevy::scene::ScenePlugin);
        group = group.add(bevy::bsn_asset::BsnAssetPlugin);
        // Skeletal animation drives joint `Transform`s; the tracer skins on the GPU
        // (skinning.rs). Clips arrive as baked `.animclip` (animclip.rs). No glTF loader: a
        // `.glb` is an importer input, never a runtime asset.
        group = group.add(bevy::animation::AnimationPlugin);
        group = group.add(bevy::world_serialization::WorldSerializationPlugin);
        group = group.add(crate::portal::PortalPlugin);
        group = group.add(bevy::winit::WinitPlugin::default());
        group = group.add(bevy::audio::AudioPlugin::default());

        // Before RayRenderPlugin: under the `xr` feature the render device is created through
        // the OpenXR runtime, which must be up first.
        group = group.add(crate::xr::XrPlugin);
        group = group.add(crate::ray_render_plugin::RayRenderPlugin);
        group = group.add(crate::dlss::DlssPlugin::default());
        group = group.add(crate::render_env::RenderEnvPlugin);
        group = group.add(crate::env_light::EnvLightPlugin);
        group = group.add(crate::post_process_filter::PostProcessFilterPlugin);
        group = group.add(crate::raytracing_pipeline::RaytracingPipelinePlugin);
        group = group.add(crate::shader::ShaderPlugin);
        group = group.add(crate::compute::ComputePlugin);
        group = group.add(crate::material::MaterialPlugin);
        group = group.add(crate::mesh::AuroraMeshPlugin);
        group = group.add(crate::procedural_mesh::ProceduralMeshPlugin);
        group = group.add(crate::collision::CollisionPlugin);
        group = group.add(crate::gpu_transform::GpuTransformPlugin);
        // Physics and skeletal animation are part of the stack: every world is an avian
        // physics world (world.rs), and characters play bevy_animation_graph graphs.
        group = group.add_group(avian3d::prelude::PhysicsPlugins::default());
        group = group.add(bevy_animation_graph::AnimationGraphPlugin::default());
        group = group.add(crate::environment::RenderEnvironmentPlugin);
        group = group.add(crate::tlas_builder::TLASBuilderPlugin);
        group = group.add(crate::picking::PickingPlugin);
        group = group.add(crate::pointer_picking::PointerPickingPlugin);
        group = group.add(crate::skinning::SkinningPlugin);
        group = group.add(crate::terrain::TerrainPlugin);
        group = group.add(crate::lights::LightsPlugin);
        group = group.add(crate::restir::RestirPlugin);
        group = group.add(crate::sharc::SharcPlugin);
        group = group.add(crate::auto_exposure::AutoExposurePlugin);
        group = group.add(crate::atmosphere::AtmospherePlugin);
        group = group.add(crate::debug_view::DebugViewPlugin);
        group = group.add(crate::surface_group::SurfaceGroupPlugin);
        group = group.add(crate::render_shaders::RenderShadersPlugin);
        group = group.add(crate::sbt::SBTPlugin);
        group = group.add(crate::sphere::SpherePlugin);
        group = group.add(crate::render_texture::RenderTexturePlugin);
        // Rasterizes the `bevy_ui` node tree. It belongs here rather than behind
        // `DevUIPlugin`, which is where it used to live: the dev panel is a debug tool, but
        // any app with a UI needs the renderer, and reaching it through a debug plugin
        // meant an app that did not want the F2 panel silently had no UI at all -- not even
        // `Assets<Font>`, since this is what pulls `TextPlugin` in.
        group = group.add(crate::ui_render::UiRenderPlugin);
        group = group.add(crate::camera_target::CameraTargetPlugin);
        // Draws bevy_gizmos lines (add `bevy::gizmos::GizmoPlugin` yourself to switch them on).
        group = group.add(crate::gizmo_render::GizmoRenderPlugin);
        group = group.add(crate::bluenoise_plugin::BlueNoisePlugin);
        group = group.add(crate::bsn::BsnPlugin);
        group = group.add(crate::animclip::AnimClipPlugin);

        group
    }
}
