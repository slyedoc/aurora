//! An 8 km ground of two triangles in the sun, the camera 1.8 m over it: its vertices sit 4 km
//! from the mesh origin, so a hit near the camera carries the error of those numbers, not of its
//! metre-sized camera-relative position. Rays leaving the ground must clear that error
//! (rt_types.slang offsetRay), or they re-hit it in square bands around the camera. A block and
//! a ball on it for contact shadows.
//!
//!   cargo run --release --example big_ground [-- --capture DIR --size WxH --frames N --noisy K]

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    auto_exposure::{AuroraExposure, ev_from_ev100},
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    sky::Sky,
    sphere::Sphere,
    util::capture::{self, CaptureArgs},
};

const REACH: f32 = 4000.0;
const EYE: Vec3 = Vec3::new(12.0, 1.8, 7.0);

fn main() {
    let mut app = App::new();
    let capture = CaptureArgs::from_args();
    if capture.is_some() {
        app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    } else {
        app.add_plugins((AuroraDefaultPlugins, DevUIPlugin, FreeCameraPlugin));
    }
    app.add_systems(Startup, setup);
    match capture {
        Some(args) => capture::run(&mut app, &args),
        None => {
            app.run();
        }
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let environment = commands
        .spawn((Name::new("Environment"), Environment::default(), Sky::Generated))
        .id();
    commands.spawn((
        ChildOf(environment),
        DirectionalLight::default(),
        Transform::default().looking_to(Vec3::new(0.4, -0.5, 0.8), Vec3::Y),
    ));
    let target = EYE + Vec3::new(-1.0, -1.8, -3.0);
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        AuroraExposure::fixed(ev_from_ev100(14.0)),
        Transform::from_translation(EYE).looking_at(target, Vec3::Y),
    ));

    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(REACH * 2.0, REACH * 2.0),
        ))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.35, 0.33, 0.28),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Cuboid::default()))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.7, 0.12, 0.08),
            perceptual_roughness: 0.7,
            ..default()
        })),
        Transform::from_translation(target + Vec3::new(-0.6, 0.15, 0.4))
            .with_scale(Vec3::new(0.4, 0.3, 0.4)),
    ));
    commands.spawn((
        ChildOf(environment),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.8, 0.8, 0.8),
            perceptual_roughness: 0.5,
            ..default()
        })),
        Transform::from_translation(target + Vec3::new(0.7, 0.2, -0.3))
            .with_scale(Vec3::splat(0.4)),
    ));
}
