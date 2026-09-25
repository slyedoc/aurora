//! Point lights as the WoW importer builds them: a small emissive core (the visible flame)
//! plus a `PointLight` at its centre carrying the halo's flux, with `radius` as the emitter
//! size. A row of them with growing radius shows the shadows going soft behind the pillars;
//! the last pair comes from `assets/scenes/point_lights.bsn` to exercise the scene path.
//!
//!   AUTO_SCREENSHOT_MS=6000 AURORA_EXIT_SECS=8 cargo run --example point_lights

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::light::PointLight;
use bevy::prelude::*;
use bevy_aurora::{
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    AuroraDefaultPlugins,
    sky::Sky,
    sphere::Sphere,
    util::{ScreenshotExt, TimeoutAppExt},
};

/// Lumens per lantern (a bright oil lamp is ~1000; WoW glow cards come out around 5000).
const LUMENS: f32 = 6_000.0;
/// The core's radiance: what a flame looks like up close, nits.
const CORE_NITS: f32 = 200_000.0;

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut windows: Query<&mut Window>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut window = windows.single_mut().unwrap();
    window.title = "aurora — point lights".into();
    window.resolution.set_scale_factor_override(Some(1.0));
    window.resolution.set(1600.0, 900.0);

    // Night: the lanterns carry the scene.
    commands.insert_resource(Sky::Color {
        radiance: Vec3::new(2.0, 3.0, 6.0),
    });
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(-1.0, 3.5, 12.0).looking_at(Vec3::new(-1.0, 1.2, -2.0), Vec3::Y),
    ));

    let grey = materials.add(AuroraMaterial {
        base_color: Color::linear_rgb(0.5, 0.48, 0.45),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Name::new("floor"),
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        AuroraMaterial3d(grey.clone()),
    ));
    commands.spawn((
        Name::new("back wall"),
        Mesh3d(meshes.add(Cuboid::new(40.0, 6.0, 0.2))),
        AuroraMaterial3d(grey.clone()),
        Transform::from_xyz(0.0, 3.0, -4.0),
    ));

    // Lanterns along x, each with a pillar in front of it to cast a shadow on the wall.
    let core = materials.add(AuroraMaterial {
        base_color: Color::BLACK,
        emissive: LinearRgba::rgb(1.0, 0.45, 0.15) * CORE_NITS,
        ..default()
    });
    for (i, radius) in [0.0f32, 0.1, 0.3, 0.6].into_iter().enumerate() {
        let x = i as f32 * 4.0 - 8.0;
        commands.spawn((
            Name::new(format!("pillar {i}")),
            Mesh3d(meshes.add(Cuboid::new(0.3, 3.0, 0.3))),
            AuroraMaterial3d(grey.clone()),
            Transform::from_xyz(x, 1.5, -2.0),
        ));
        commands.spawn((
            Name::new(format!("core {i}")),
            Sphere,
            AuroraMaterial3d(core.clone()),
            Transform::from_xyz(x, 2.2, 0.0).with_scale(Vec3::splat(0.08)),
        ));
        commands.spawn((
            Name::new(format!("lantern light {i} (radius {radius})")),
            Transform::from_xyz(x, 2.2, 0.0),
            PointLight {
                color: Color::linear_rgb(1.0, 0.45, 0.15),
                intensity: LUMENS,
                radius,
                ..default()
            },
        ));
    }

    // The same lantern from a scene file.
    commands.spawn((
        Name::new("bsn lantern"),
        Transform::IDENTITY,
        Visibility::Visible,
        ScenePatchInstance(asset_server.load(aurora_asset("scenes/point_lights.bsn"))),
    ));
    commands.spawn((
        Name::new("bsn pillar"),
        Mesh3d(meshes.add(Cuboid::new(0.3, 3.0, 0.3))),
        AuroraMaterial3d(grey),
        Transform::from_xyz(6.0, 1.5, -2.0),
    ));
    commands.spawn((
        Name::new("bsn core"),
        Sphere,
        AuroraMaterial3d(core),
        Transform::from_xyz(6.0, 2.2, 0.0).with_scale(Vec3::splat(0.08)),
    ));
}
