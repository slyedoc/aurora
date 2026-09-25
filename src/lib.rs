pub mod aftermath;
pub mod animclip;
pub mod assets;
pub mod atmosphere;
pub mod auto_exposure;
pub mod blas;
pub mod bsn;
pub mod collision;
pub mod compute;
pub use aurora_cluster_mesh as cluster_mesh;
pub mod bluenoise_plugin;
pub mod debug_view;
pub mod dev_shaders;
pub mod dev_ui;
pub mod dlss;
pub mod env_light;
pub mod gizmo_render;
pub mod gltf_mesh;
pub mod gpu_transform;
pub mod lights;
pub mod material;
pub mod omm;
pub mod post_process_filter;
pub mod procedural_mesh;
pub mod ray_render_plugin;
pub mod raytracing_pipeline;
pub mod render_buffer;
pub mod render_device;
pub mod render_env;
pub mod render_texture;
pub mod restir;
pub mod sbt;
pub mod shader;
pub mod sharc;
pub mod skinning;
pub mod sky;
pub mod sphere;
pub mod swapchain;
pub mod portal;
pub mod terrain;
pub mod tlas_builder;
pub mod transform;
pub mod ui_panel;
pub mod ui_render;
pub mod util;
pub mod vk_init;
pub mod vk_utils;
pub mod vulkan_asset;
pub mod vulkan_mesh;
pub mod xr;

use bevy::{app::PluginGroupBuilder, prelude::*};

/// What an app writes against: `use bevy_aurora::prelude::*;`
///
/// Scene-facing types only -- components you spawn, resources you tune, the plugin group,
/// and the app-level helpers. The renderer's own machinery (pipelines, buffers, extracted
/// mirrors, the per-system plugins [`AuroraDefaultPlugins`] already adds) stays behind its
/// module path.
///
/// Four names are held back because they would shadow `bevy::prelude`:
/// `material::AlphaMode`, `sphere::Sphere`, `transform::TransformPlugin` and
/// `gltf_mesh::GltfPlugin`. Import those by path and alias them.
pub mod prelude {
    pub use crate::{
        assets::aurora_asset,
        atmosphere::{Atmosphere, CloudLayer},
        auto_exposure::{ev100_from_ev, ev_from_ev100, AuroraExposure, FixedExposure},
        collision::{CollisionMesh, CollisionShape},
        compute::{ComputeModule, ComputeModules},
        debug_view::AuroraDebugView,
        dev_shaders::DevShaderPlugin,
        dev_ui::{DevUIPanel, DevUIPlugin, DevUIState},
        dlss::{AuroraDlss, RrPreset},
        env_light::EnvLight,
        gltf_mesh::{GltfModel, GltfModelHandle},
        material::{AuroraMaterial, AuroraMaterial3d},
        portal::AuroraPortal,
        procedural_mesh::{ProceduralKernels, ProceduralMesh, ProceduralMesh3d},
        skinning::{SkinJointsByName, Wind, WindSway},
        sky::{LayerSkies, ProceduralSky, Sky},
        ui_panel::{InspectorPanel3d, UiPanel3d},
        util::{ScreenshotExt, TimeoutAppExt},
        xr::{XrHand, XrHandState, XrInput, XrPose, XrState, XrTracked},
        AuroraDefaultPlugins,
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
        // Skeletal animation drives joint `Transform`s; the tracer skins on the GPU
        // (skinning.rs). bevy's glTF loader yields skinned meshes + clips (render-free in the
        // fork); aurora's own `GltfModel` loader keeps the `.glb` extension for typed loads.
        group = group.add(bevy::animation::AnimationPlugin);
        group = group.add(bevy::world_serialization::WorldSerializationPlugin);
        group = group.add(bevy::gltf::GltfPlugin::default());
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
        group = group.add(crate::vulkan_mesh::VulkanMeshPlugin);
        group = group.add(crate::gltf_mesh::GltfPlugin);
        group = group.add(crate::procedural_mesh::ProceduralMeshPlugin);
        group = group.add(crate::collision::CollisionPlugin);
        group = group.add(crate::gpu_transform::GpuTransformPlugin);
        group = group.add(crate::tlas_builder::TLASBuilderPlugin);
        group = group.add(crate::skinning::SkinningPlugin);
        group = group.add(crate::terrain::TerrainPlugin);
        group = group.add(crate::lights::LightsPlugin);
        group = group.add(crate::restir::RestirPlugin);
        group = group.add(crate::sharc::SharcPlugin);
        group = group.add(crate::auto_exposure::AutoExposurePlugin);
        group = group.add(crate::atmosphere::AtmospherePlugin);
        group = group.add(crate::debug_view::DebugViewPlugin);
        group = group.add(crate::sbt::SBTPlugin);
        group = group.add(crate::sphere::SpherePlugin);
        group = group.add(crate::render_texture::RenderTexturePlugin);
        // Draws bevy_gizmos lines (add `bevy::gizmos::GizmoPlugin` yourself to switch them on).
        group = group.add(crate::gizmo_render::GizmoRenderPlugin);
        group = group.add(crate::bluenoise_plugin::BlueNoisePlugin);
        group = group.add(crate::bsn::BsnPlugin);
        group = group.add(crate::animclip::AnimClipPlugin);

        group
    }
}
