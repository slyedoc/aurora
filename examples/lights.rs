//! Analytic lights: one point, one spot, one rect area light over a simple test floor.
//!
//! Bevy's `PointLight` / `SpotLight` / `RectLight` components feed the NEE light table
//! (`src/lights.rs`) next to emissive triangles; ReSTIR DI samples them all through one
//! CDF. Grab any light in the F1 world inspector and drag its transform or intensity --
//! movement re-uploads in place, so ReSTIR history survives. F4 cycles the debug views
//! (the Color view shows the noise the light sampling is there to kill).

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::light::{PointLight, RectLight, SpotLight};
use bevy::prelude::*;
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
};

fn main() {
    let mut app = App::new();
    app.add_plugins(RayDefaultPlugins);
    app.add_plugins(DevShaderPlugin);
    app.add_plugins(DevUIPlugin);
    app.add_plugins(FreeCameraPlugin::default());
    app.add_screenshot(KeyCode::F12);
    app.add_timeout_exit(None, 20.0);
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(
    mut commands: Commands,
    mut windows: Query<&mut Window>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut window = windows.single_mut().unwrap();
    window.title = "aurora — analytic lights".into();

    // A near-black night sky (nits), so the analytic lights carry the scene.
    commands.insert_resource(Sky::Color {
        radiance: Vec3::splat(20.0),
    });

    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(0.0, 4.0, 12.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        FreeCamera::default(),
    ));

    commands.spawn((
        Name::new("floor"),
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.5, 0.5, 0.5),
            perceptual_roughness: 0.8,
            ..default()
        })),
    ));

    // A row of pillars with varied roughness to catch highlights and shadows.
    for i in 0..5 {
        let x = (i as f32 - 2.0) * 2.5;
        commands.spawn((
            Name::new(format!("pillar {i}")),
            Mesh3d(meshes.add(Cuboid::new(0.8, 2.0, 0.8))),
            Transform::from_xyz(x, 1.0, 0.0),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::srgb(0.8, 0.75, 0.7),
                perceptual_roughness: 0.1 + 0.2 * i as f32,
                ..default()
            })),
        ));
    }

    commands.spawn((
        Name::new("warm point"),
        Transform::from_xyz(-3.0, 4.0, 3.0),
        PointLight {
            color: Color::srgb(1.0, 0.6, 0.3),
            intensity: 20_000_000.0,
            range: 100.0,
            ..default()
        },
    ));

    commands.spawn((
        Name::new("green spot"),
        Transform::from_xyz(3.0, 6.0, 1.0).looking_at(Vec3::new(3.0, 0.0, -1.0), Vec3::Y),
        SpotLight {
            color: Color::srgb(0.3, 1.0, 0.4),
            intensity: 60_000_000.0,
            range: 100.0,
            inner_angle: 0.25,
            outer_angle: 0.5,
            ..default()
        },
    ));

    // Faces local -Z: angled down at the floor from behind the pillars.
    commands.spawn((
        Name::new("blue rect panel"),
        Transform::from_xyz(0.0, 3.5, -4.0).looking_at(Vec3::new(0.0, 0.0, 2.0), Vec3::Y),
        RectLight {
            color: Color::srgb(0.3, 0.5, 1.0),
            intensity: 40_000_000.0,
            range: 100.0,
            width: 4.0,
            height: 1.5,
            ..default()
        },
    ));
}
