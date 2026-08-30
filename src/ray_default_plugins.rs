use bevy::{app::PluginGroupBuilder, prelude::*};

pub struct RayDefaultPlugins;

impl PluginGroup for RayDefaultPlugins {
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
            .add(bevy::transform::TransformPlugin)
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
        group = group.add(bevy::winit::WinitPlugin::default());
        group = group.add(bevy::audio::AudioPlugin::default());

        group = group.add(crate::ray_render_plugin::RayRenderPlugin);
        group = group.add(crate::render_env::RenderEnvPlugin);
        group = group.add(crate::post_process_filter::PostProcessFilterPlugin);
        group = group.add(crate::raytracing_pipeline::RaytracingPipelinePlugin);
        group = group.add(crate::shader::ShaderPlugin);
        group = group.add(crate::compute::ComputePlugin);
        group = group.add(crate::vulkan_mesh::VulkanMeshPlugin);
        group = group.add(crate::gltf_mesh::GltfPlugin);
        group = group.add(crate::gpu_transform::GpuTransformPlugin);
        group = group.add(crate::tlas_builder::TLASBuilderPlugin);
        group = group.add(crate::sbt::SBTPlugin);
        group = group.add(crate::sphere::SpherePlugin);
        group = group.add(crate::render_texture::RenderTexturePlugin);
        group = group.add(crate::bluenoise_plugin::BlueNoisePlugin);
        group = group.add(crate::bsn::BsnPlugin);

        group
    }
}
