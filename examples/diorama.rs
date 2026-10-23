//! Volume portal: a live diorama. A disc under a low glass-less dome stands on a table in a
//! courtyard; inside it is another environment, an island of blocks and balls under its own warm
//! sun, at 1:20. Every ray into the dome carries on inside the island from the matching point of
//! the same shape 20 times bigger, and comes back out where it leaves that; where the big shape's
//! wall runs under the island's ground the dome shows the cut, earth over stone.
//!
//! Click anything to log what is under the cursor (through the dome: the island's entity and its
//! point in the island). A probe ray straight down into the dome checks itself at startup and logs
//! `probe ... PASS` / `FAIL`.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    window::PrimaryWindow,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    picking::{RayCaster, RayHits},
    portal::{AuroraVolumePortal, VolumeShape},
    sky::Sky,
    sphere::Sphere,
    util::{
        ScreenshotExt,
        capture::{self, CaptureArgs},
    },
};

const SKY_SCALE_NITS: f32 = 8000.0;
const TABLE_TOP: f32 = 0.8;
/// The diorama's scale: island metres per courtyard metre.
const SCALE: f32 = 20.0;
/// How far under the island's ground the big shape's floor lies (island metres).
const SINK: f32 = 2.0;

fn main() {
    let mut app = App::new();
    // `--capture DIR ...`: windowless into a fixed-size image, a reference capture, then exit
    // (bevy_aurora::util::capture).
    let capture = CaptureArgs::from_args();
    if capture.is_some() {
        app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    } else {
        app.add_plugins((AuroraDefaultPlugins, DevUIPlugin, FreeCameraPlugin))
            .add_screenshot(KeyCode::F12);
    }
    app.add_systems(Startup, setup)
        .add_systems(Update, (aim_cursor_ray, log_click, check_probe));
    match capture {
        Some(args) => capture::run(&mut app, &args),
        None => {
            app.run();
        }
    }
}

#[derive(Component)]
struct CursorRay;

#[derive(Component)]
struct Probe {
    expect: &'static str,
    distance: f32,
    done: bool,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    let mut matte = |color: Color, roughness: f32| {
        materials.add(AuroraMaterial {
            base_color: color,
            perceptual_roughness: roughness,
            ..default()
        })
    };

    // The courtyard: a floor, two walls, a table.
    let courtyard = commands
        .spawn((
            Name::new("Courtyard"),
            Environment::default(),
            Sky::Hdr {
                image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
                scale: SKY_SCALE_NITS,
            },
        ))
        .id();
    commands.spawn((
        ChildOf(courtyard),
        Name::new("Courtyard Sun"),
        DirectionalLight {
            illuminance: 20_000.0,
            ..default()
        },
        Transform::from_xyz(-3.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        ChildOf(courtyard),
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 50.0_f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(0.0, 2.3, 3.2).looking_at(Vec3::new(0.0, 0.95, 0.0), Vec3::Y),
    ));
    let stone = matte(Color::srgb(0.55, 0.53, 0.5), 0.9);
    commands.spawn((
        ChildOf(courtyard),
        Name::new("Floor"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(20.0, 20.0),
        ))),
        AuroraMaterial3d(stone.clone()),
    ));
    let plaster = matte(Color::srgb(0.75, 0.7, 0.62), 0.95);
    for (size, at) in [
        (Vec3::new(10.0, 3.5, 0.2), Vec3::new(0.0, 1.75, -4.0)),
        (Vec3::new(0.2, 3.5, 8.0), Vec3::new(-5.0, 1.75, 0.0)),
    ] {
        commands.spawn((
            ChildOf(courtyard),
            Name::new("Wall"),
            AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Cuboid::from_size(size)))),
            AuroraMaterial3d(plaster.clone()),
            Transform::from_translation(at),
        ));
    }
    let wood = matte(Color::srgb(0.32, 0.18, 0.09), 0.6);
    commands.spawn((
        ChildOf(courtyard),
        Name::new("Table"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Cuboid::new(2.6, 0.06, 2.6)))),
        AuroraMaterial3d(wood.clone()),
        Transform::from_xyz(0.0, TABLE_TOP - 0.03, 0.0),
    ));
    let leg = meshes.add(AuroraMesh::from_shape(Cuboid::new(
        0.08,
        TABLE_TOP - 0.06,
        0.08,
    )));
    for (x, z) in [(-1.2, -1.2), (1.2, -1.2), (-1.2, 1.2), (1.2, 1.2)] {
        commands.spawn((
            ChildOf(courtyard),
            Name::new("Table Leg"),
            AuroraMesh3d(leg.clone()),
            AuroraMaterial3d(wood.clone()),
            Transform::from_xyz(x, (TABLE_TOP - 0.06) / 2.0, z),
        ));
    }

    // The island, in its own environment at the same coordinates.
    let island = commands
        .spawn((
            Name::new("Island"),
            Environment::default(),
            Sky::Hdr {
                image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
                scale: SKY_SCALE_NITS * 0.6,
            },
        ))
        .id();
    commands.spawn((
        ChildOf(island),
        Name::new("Island Sun"),
        DirectionalLight {
            illuminance: 25_000.0,
            color: Color::srgb(1.0, 0.75, 0.5),
            ..default()
        },
        Transform::from_xyz(10.0, 6.0, -6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        ChildOf(island),
        Name::new("Island Ground"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(400.0, 400.0),
        ))),
        AuroraMaterial3d(matte(Color::srgb(0.22, 0.38, 0.14), 1.0)),
    ));
    let block = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    for (name, color, size, at) in [
        (
            "Red block",
            Color::srgb(0.8, 0.12, 0.08),
            3.0,
            Vec3::new(7.0, 1.5, 0.0),
        ),
        (
            "Blue block",
            Color::srgb(0.1, 0.25, 0.85),
            4.0,
            Vec3::new(-6.0, 2.0, -6.0),
        ),
        (
            "Yellow block",
            Color::srgb(0.9, 0.75, 0.1),
            2.5,
            Vec3::new(-2.0, 1.25, 9.0),
        ),
        (
            "White block",
            Color::srgb(0.85, 0.85, 0.82),
            3.5,
            Vec3::new(2.0, 1.75, -9.0),
        ),
    ] {
        commands.spawn((
            ChildOf(island),
            Name::new(name),
            AuroraMesh3d(block.clone()),
            AuroraMaterial3d(matte(color, 0.7)),
            Transform::from_translation(at)
                .with_rotation(Quat::from_rotation_y(at.x * 0.3))
                .with_scale(Vec3::splat(size)),
        ));
    }
    for (name, color, metallic, radius, at) in [
        (
            "Mirror ball",
            Color::WHITE,
            1.0,
            2.0,
            Vec3::new(-7.0, 2.0, 3.0),
        ),
        (
            "Green ball",
            Color::srgb(0.2, 0.7, 0.3),
            0.0,
            1.5,
            Vec3::new(1.0, 1.5, 4.0),
        ),
        (
            "Orange ball",
            Color::srgb(0.95, 0.45, 0.1),
            0.0,
            2.0,
            Vec3::new(8.0, 2.0, 5.0),
        ),
    ] {
        commands.spawn((
            ChildOf(island),
            Name::new(name),
            Sphere,
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: color,
                metallic,
                perceptual_roughness: if metallic > 0.0 { 0.05 } else { 0.5 },
                ..default()
            })),
            Transform::from_translation(at).with_scale(Vec3::splat(radius * 2.0)),
        ));
    }

    // The dome on the table, showing the island at 1:20 (the big shape sunk SINK under its
    // ground, so the cut shows earth under the turf).
    commands.spawn((
        ChildOf(courtyard),
        Name::new("Diorama"),
        Transform::from_xyz(0.0, TABLE_TOP, 0.0),
        AuroraVolumePortal {
            target_env: island,
            target_transform: Transform::from_xyz(0.0, -SINK, 0.0).with_scale(Vec3::splat(SCALE)),
            shape: VolumeShape {
                radius: 1.0,
                wall: 0.12,
                dome: 0.3,
            },
        },
    ));

    commands.spawn((
        ChildOf(courtyard),
        CursorRay,
        RayCaster::new(Ray3d::new(Vec3::ZERO, Dir3::NEG_Y)),
    ));
    // Straight down into the dome at x = 0.35: the red block's top (island y = 3, so 0.25 up
    // the shape: courtyard y = 1.05).
    commands.spawn((
        ChildOf(courtyard),
        Probe {
            expect: "Red block",
            distance: 3.0 - (TABLE_TOP + (3.0 + SINK) / SCALE),
            done: false,
        },
        RayCaster::new(Ray3d::new(Vec3::new(0.35, 3.0, 0.0), Dir3::NEG_Y)),
    ));
}

fn aim_cursor_ray(
    window: Option<Single<&Window, With<PrimaryWindow>>>,
    camera: Single<(&Camera, &GlobalTransform)>,
    mut caster: Single<&mut RayCaster, With<CursorRay>>,
) {
    let (camera, transform) = *camera;
    if let Some(ray) = window
        .and_then(|w| w.cursor_position())
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
    let name = |e: Entity| names.get(e).map_or("<unnamed>", Name::as_str);
    match hits.first() {
        Some(hit) => info!(
            "clicked {}{} at {:.2} (distance {:.2})",
            name(hit.entity),
            hit.through
                .map_or(String::new(), |p| format!(" through {}", name(p))),
            hit.point,
            hit.distance,
        ),
        None => info!("clicked nothing"),
    }
}

fn check_probe(time: Res<Time>, mut probes: Query<(&mut Probe, &RayHits)>, names: Query<&Name>) {
    // Once the scene has streamed in.
    if time.elapsed_secs() < 3.0 {
        return;
    }
    for (mut probe, hits) in &mut probes {
        let Some(hit) = hits.first() else {
            continue;
        };
        let name = names.get(hit.entity).map_or("?", Name::as_str);
        if probe.done {
            continue;
        }
        probe.done = true;
        let pass = name == probe.expect
            && hit.through.is_some()
            && (hit.distance - probe.distance).abs() < 0.02;
        info!(
            "probe {} {}: {} at {:.2} (distance {:.3}, expected {:.3}, through {:?})",
            probe.expect,
            if pass { "PASS" } else { "FAIL" },
            name,
            hit.point,
            hit.distance,
            probe.distance,
            hit.through.map(|e| names.get(e).map_or("?", Name::as_str)),
        );
    }
}
