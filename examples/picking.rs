//! GPU picking: click anything to log what is under the cursor. Two fixed probe rays check
//! themselves at startup and log `probe ... PASS` / `FAIL`.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    picking::{RayCaster, RayHits},
    sphere::Sphere,
    util::TimeoutAppExt,
};

fn main() {
    App::new()
        .add_plugins((AuroraDefaultPlugins, FreeCameraPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (aim_cursor_ray, log_click, check_probes))
        .add_timeout_exit(None, 8.0)
        .run();
}

#[derive(Component)]
struct CursorRay;

/// A fixed ray and the names it must hit, nearest first.
#[derive(Component)]
struct Probe {
    expect: &'static [(&'static str, f32)],
    done: bool,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    let grey = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.5, 0.5, 0.5),
        ..default()
    });
    // The sun (the sky draws its disc; it lights the scene).
    commands.spawn((
        Name::new("Sun"),
        DirectionalLight {
            illuminance: 20_000.0,
            ..default()
        },
        Transform::from_xyz(3.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 3.0, 6.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
    ));
    commands.spawn((
        Name::new("Ground"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(20.0, 20.0),
        ))),
        AuroraMaterial3d(grey.clone()),
    ));
    commands.spawn((
        Name::new("Cube"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Cuboid::default()))),
        AuroraMaterial3d(grey.clone()),
        Transform::from_xyz(-1.5, 1.0, 0.0),
    ));
    commands.spawn((
        Name::new("Ball"),
        Sphere,
        AuroraMaterial3d(grey),
        Transform::from_xyz(1.5, 0.5, 0.0),
    ));
    commands.spawn((
        CursorRay,
        RayCaster::new(Ray3d::new(Vec3::ZERO, Dir3::NEG_Y)),
    ));
    // Straight down through the floating cube (top y = 1.5, bottom 0.5) onto the ground, and
    // onto the ball's top (y = 1).
    commands.spawn((
        Probe {
            expect: &[("Cube", 3.5), ("Cube", 4.5), ("Ground", 5.0)],
            done: false,
        },
        RayCaster::new(Ray3d::new(Vec3::new(-1.5, 5.0, 0.0), Dir3::NEG_Y)),
    ));
    commands.spawn((
        Probe {
            expect: &[("Ball", 4.0), ("Ground", 5.0)],
            done: false,
        },
        RayCaster::new(Ray3d::new(Vec3::new(1.5, 5.0, 0.0), Dir3::NEG_Y)),
    ));
}

fn aim_cursor_ray(
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut caster: Single<&mut RayCaster, With<CursorRay>>,
) {
    let (camera, transform) = *camera;
    if let Some(ray) = window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(transform, cursor).ok())
    {
        caster.ray = ray;
    }
}

fn log_click(
    mouse: Res<ButtonInput<MouseButton>>,
    hits: Single<&RayHits, With<CursorRay>>,
    names: Query<&Name>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    match hits.first() {
        Some(hit) => info!(
            "clicked {} at {:.2} (distance {:.2}, normal {:.2}, triangle {})",
            names.get(hit.entity).map_or("<unnamed>", Name::as_str),
            hit.point,
            hit.distance,
            hit.normal,
            hit.triangle
        ),
        None => info!("clicked nothing"),
    }
}

fn check_probes(mut probes: Query<(&mut Probe, &RayHits)>, names: Query<&Name>) {
    for (mut probe, hits) in &mut probes {
        if probe.done || hits.is_empty() {
            continue;
        }
        probe.done = true;
        let got: Vec<(&str, f32)> = hits
            .iter()
            .map(|hit| {
                (
                    names.get(hit.entity).map_or("?", Name::as_str),
                    hit.distance,
                )
            })
            .collect();
        // The ball's hit is its near side; a cube is hit on its top face and its bottom face.
        let pass = got.len() >= probe.expect.len()
            && probe
                .expect
                .iter()
                .zip(&got)
                .all(|((name, t), (got_name, got_t))| name == got_name && (t - got_t).abs() < 1e-3);
        let normal_up = hits[0].normal.abs_diff_eq(Vec3::Y, 1e-3);
        info!(
            "probe {:?}: {} (first normal {:.2})",
            got,
            if pass && normal_up { "PASS" } else { "FAIL" },
            hits[0].normal
        );
    }
}
