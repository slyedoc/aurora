//! Emissive materials under an HDR sky: a plane, one emissive sphere (procedural), one
//! emissive cube (mesh), and a diffuse reference sphere.
//!
//!   AUTO_SCREENSHOT_MS=9000 AURORA_EXIT_SECS=11 cargo run --example emissive

use bevy::prelude::*;
use bevy_aurora::{
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    sky::Sky,
    sphere::Sphere,
    util::{ScreenshotExt, TimeoutAppExt},
};

/// HDR sky brightness: texel value x this = nits (a clear sky sits around 2000-8000).
const SKY_SCALE_NITS: f32 = 8000.0;

/// Emitter brightness in nits (cd/m^2). The engine works in physical units: anything below
/// ~10k nits barely reads against a daylit sky (see aurora_files/lighting_units.md).
const EMISSIVE_NITS: f32 = 1_000_000.0;

fn main() {
    App::new()
        .add_plugins((RayDefaultPlugins, DevShaderPlugin, DevUIPlugin))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // HDR environment sky (equirectangular), texel x scale = nits.
    commands.insert_resource(Sky::Hdr {
        image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
        scale: SKY_SCALE_NITS,
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(50.0, 50.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.3),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    // The emissive sphere (procedural) and cube (mesh), side by side.
    commands.spawn((
        Transform::from_xyz(-1.5, 1.0, 0.0),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            emissive: LinearRgba::rgb(1.0, 0.8, 0.2) * EMISSIVE_NITS,
            ..default()
        })),
    ));
    commands.spawn((
        Transform::from_xyz(1.5, 1.0, 0.0),
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            emissive: LinearRgba::rgb(0.2, 0.8, 1.0) * EMISSIVE_NITS,
            ..default()
        })),
    ));

    // A plain diffuse reference sphere.
    commands.spawn((
        Transform::from_xyz(0.0, 1.0, -2.0),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.8, 0.2, 0.2),
            ..default()
        })),
    ));
}
