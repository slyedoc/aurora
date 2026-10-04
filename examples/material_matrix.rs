//! Every material in this scene is spelled in `.bsn`, not in Rust: the point is to prove
//! the widened `AuroraMaterial` still round-trips through reflection after the jackdaw
//! port added twelve authoring fields to it. A field that is reflected but never
//! registered fails the WHOLE scene load, not just that field, so a rendered frame here
//! is the check.
//!
//! Left to right: rough dielectric, polished metal, emissive, glass.
//!
//!   AUTO_SCREENSHOT_MS=9000 cargo run --release --example material_matrix

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::prelude::*;
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    util::{ScreenshotExt, TimeoutAppExt},
};

fn main() {
    App::new()
        .add_plugins((AuroraDefaultPlugins, FreeCameraPlugin))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 14.0)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 2.2, 7.5).looking_at(Vec3::new(0.0, 0.9, 0.0), Vec3::Y),
    ));

    // A key light, so the metal has something to reflect and the glass something to bend.
    commands.spawn((
        Name::new("key"),
        DirectionalLight {
            illuminance: 12_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground: authored in Rust so a failure to load the scene is obvious -- the floor
    // renders and the spheres do not, rather than an empty frame that could be anything.
    let ground = materials.add(AuroraMaterial {
        base_color: Color::linear_rgb(0.22, 0.23, 0.25),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Name::new("ground"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Cuboid::new(16.0, 0.2, 10.0)))),
        AuroraMaterial3d(ground),
        Transform::from_xyz(0.0, -0.1, 0.0),
    ));

    // The spheres and their materials come from the scene file.
    commands.spawn((
        Name::new("material matrix"),
        Transform::IDENTITY,
        Visibility::Visible,
        ScenePatchInstance(asset_server.load(aurora_asset("scenes/material_matrix.bsn"))),
    ));
}
