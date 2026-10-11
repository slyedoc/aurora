use bevy::prelude::*;

use crate::{
    assets::aurora_asset, post_process_filter::PostProcessFilter, ray_render_plugin::RenderConfig,
    raytracing_pipeline::RaytracingPipeline,
};

/// Publishes `RenderConfig`; override it after the group to supply your own shaders.
pub struct RenderShadersPlugin;

impl Plugin for RenderShadersPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().get_resource::<AssetServer>().unwrap();

        let filter = PostProcessFilter {
            vertex_shader: asset_server.load(aurora_asset("shaders/quad.slang")),
            fragment_shader: asset_server.load(aurora_asset("shaders/quad.slang")),
        };

        let rtx_pipeline = RaytracingPipeline {
            raygen_shader: asset_server.load(aurora_asset("shaders/raygen.slang")),
            miss_shader: asset_server.load(aurora_asset("shaders/miss.slang")),
            hit_shader: asset_server.load(aurora_asset("shaders/closest_hit.slang")),
            any_hit_shader: asset_server.load(aurora_asset("shaders/any_hit.slang")),
            // One module, two entry points (`intersection`, `closesthit`).
            sphere_intersection_shader: asset_server.load(aurora_asset("shaders/sphere.slang")),
            sphere_hit_shader: asset_server.load(aurora_asset("shaders/sphere.slang")),
        };

        let render_config = RenderConfig {
            rtx_pipeline: asset_server.add(rtx_pipeline),
            postprocess_pipeline: asset_server.add(filter),
            ..default()
        };

        app.world_mut().insert_resource(render_config);
    }
}
