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
        group = group.add(bevy::winit::WinitPlugin::default());
        group = group.add(bevy::audio::AudioPlugin::default());

        // Before RayRenderPlugin: under the `xr` feature the render device is created through
        // the OpenXR runtime, which must be up first.
        group = group.add(crate::xr::XrPlugin);
        group = group.add(crate::ray_render_plugin::RayRenderPlugin);
        group = group.add(crate::dlss::DlssPlugin::default());
        group = group.add(crate::render_env::RenderEnvPlugin);
        group = group.add(crate::post_process_filter::PostProcessFilterPlugin);
        group = group.add(crate::raytracing_pipeline::RaytracingPipelinePlugin);
        group = group.add(crate::shader::ShaderPlugin);
        group = group.add(crate::compute::ComputePlugin);
        group = group.add(crate::material::MaterialPlugin);
        group = group.add(crate::vulkan_mesh::VulkanMeshPlugin);
        group = group.add(crate::gltf_mesh::GltfPlugin);
        group = group.add(crate::gpu_transform::GpuTransformPlugin);
        group = group.add(crate::tlas_builder::TLASBuilderPlugin);
        group = group.add(crate::skinning::SkinningPlugin);
        group = group.add(crate::lights::LightsPlugin);
        group = group.add(crate::restir::RestirPlugin);
        group = group.add(crate::sharc::SharcPlugin);
        group = group.add(crate::auto_exposure::AutoExposurePlugin);
        group = group.add(crate::debug_view::DebugViewPlugin);
        group = group.add(crate::sbt::SBTPlugin);
        group = group.add(crate::sphere::SpherePlugin);
        group = group.add(crate::render_texture::RenderTexturePlugin);
        group = group.add(crate::bluenoise_plugin::BlueNoisePlugin);
        group = group.add(crate::bsn::BsnPlugin);

        group
    }
}
