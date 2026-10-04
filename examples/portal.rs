//! Ray-portal demo: two gates on a plain, paired both ways. Looking into the LEFT gate
//! shows the view out of the RIGHT gate's front (the red/blue pillar cluster), and vice
//! versa -- portals in reflections and through glass come free from the raygen redirect.
//!
//! A third gate, between and behind the first two, opens into a second WORLD (an
//! avian `PhysicsWorld`, so its own render bit) occupying the same space: a red desert under
//! a night sky with its own low orange sun and its own stones. Neither world sees the other's geometry except through
//! that gate. The camera is a `PortalTraveler`: fly through any gate and you come out of its
//! partner, in its world.

use std::f32::consts::PI;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    portal::{AuroraPortal, PortalTraveler},
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
    world::{MainPhysicsWorldEntity, PhysicsWorld},
};

const SKY_SCALE_NITS: f32 = 8000.0;

fn main() {
    App::new()
        .add_plugins((AuroraDefaultPlugins, DevUIPlugin, FreeCameraPlugin))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

fn setup(
    mut commands: Commands,
    main_world: Res<MainPhysicsWorldEntity>,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    commands.entity(main_world.0).insert(Sky::Hdr {
        image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
        scale: SKY_SCALE_NITS,
    });
    // The plain's sun: high and white.
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
        PortalTraveler::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        // Looking at gate A with gate B far off to the right: the through-view must show
        // B's pillars, not the empty plain behind A.
        Transform::from_xyz(-8.0, 1.7, 6.0).looking_at(Vec3::new(-8.0, 1.5, 0.0), Vec3::Y),
    ));

    commands.spawn((
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(200.0, 200.0),
        ))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.35, 0.4, 0.35),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    let frame_mat = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.1, 0.1, 0.12),
        perceptual_roughness: 0.4,
        metallic: 1.0,
        ..default()
    });
    let frame_mesh = meshes.add(AuroraMesh::from_shape(Cuboid::new(0.2, 3.4, 0.2)));
    let lintel_mesh = meshes.add(AuroraMesh::from_shape(Cuboid::new(2.4, 0.2, 0.2)));
    // The portal surface: a quad facing +Z (Rectangle is XY-plane, normal +Z).
    let quad = meshes.add(AuroraMesh::from_shape(Rectangle::new(2.0, 3.4)));
    let quad_mat = materials.add(AuroraMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    // Gates A (x = -8) and B (x = +8), both fronts facing +Z.
    let mut gates = Vec::new();
    for gx in [-8.0_f32, 8.0] {
        for px in [-1.1, 1.1] {
            commands.spawn((
                AuroraMesh3d(frame_mesh.clone()),
                AuroraMaterial3d(frame_mat.clone()),
                Transform::from_xyz(gx + px, 1.7, 0.0),
            ));
        }
        commands.spawn((
            AuroraMesh3d(lintel_mesh.clone()),
            AuroraMaterial3d(frame_mat.clone()),
            Transform::from_xyz(gx, 3.5, 0.0),
        ));
        gates.push(
            commands
                .spawn((
                    AuroraMesh3d(quad.clone()),
                    AuroraMaterial3d(quad_mat.clone()),
                    Transform::from_xyz(gx, 1.7, 0.0),
                ))
                .id(),
        );
    }
    commands
        .entity(gates[0])
        .insert(AuroraPortal { target: gates[1] });
    commands
        .entity(gates[1])
        .insert(AuroraPortal { target: gates[0] });

    // Gate C (main world, z = -8, front +Z) opens into the desert world at the same spot:
    // gate D there faces -Z, so looking into C carries straight on into the other world.
    let desert = commands
        .spawn((
            Name::new("Desert World"),
            PhysicsWorld,
            Sky::Hdr {
                image: asset_server.load(aurora_asset("sky/night_sky.hdr")),
                scale: SKY_SCALE_NITS,
            },
        ))
        .id();
    // The desert's own sun, low and orange: it lights only the desert, and the plain's only
    // the plain.
    commands.spawn((
        Name::new("Desert Sun"),
        ChildOf(desert),
        DirectionalLight {
            illuminance: 12_000.0,
            color: Color::srgb(1.0, 0.55, 0.25),
            ..default()
        },
        Transform::from_xyz(-6.0, 1.5, -20.0).looking_at(Vec3::new(0.0, 0.0, -12.0), Vec3::Y),
    ));
    let c = spawn_gate(
        &mut commands,
        None,
        Transform::from_xyz(0.0, 0.0, -8.0),
        &frame_mesh,
        &lintel_mesh,
        &frame_mat,
        &quad,
        &quad_mat,
    );
    let d = spawn_gate(
        &mut commands,
        Some(desert),
        Transform::from_xyz(0.0, 0.0, -8.0).with_rotation(Quat::from_rotation_y(PI)),
        &frame_mesh,
        &lintel_mesh,
        &frame_mat,
        &quad,
        &quad_mat,
    );
    commands.entity(c).insert(AuroraPortal { target: d });
    commands.entity(d).insert(AuroraPortal { target: c });

    // The desert: its own ground and stones, overlapping the plain in space.
    commands.spawn((
        ChildOf(desert),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(200.0, 200.0),
        ))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.6, 0.28, 0.15),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));
    let stone = meshes.add(AuroraMesh::from_shape(Sphere::new(0.8)));
    let stone_mat = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.85, 0.82, 0.75),
        perceptual_roughness: 0.3,
        ..default()
    });
    for (x, z) in [
        (-2.5, -12.0),
        (0.0, -15.0),
        (2.5, -12.0),
        (-5.0, -18.0),
        (5.0, -18.0),
    ] {
        commands.spawn((
            ChildOf(desert),
            AuroraMesh3d(stone.clone()),
            AuroraMaterial3d(stone_mat.clone()),
            Transform::from_xyz(x, 0.8, z),
        ));
    }

    // Pillars in front of gate B only: the tell. Seen through gate A = portals work.
    let pillar = meshes.add(AuroraMesh::from_shape(Cuboid::new(0.6, 2.6, 0.6)));
    for (dx, dz, color) in [
        (-1.5, 3.0, Color::srgb(0.9, 0.15, 0.1)),
        (0.0, 4.5, Color::srgb(0.1, 0.3, 0.9)),
        (1.5, 3.0, Color::srgb(0.95, 0.8, 0.1)),
    ] {
        commands.spawn((
            AuroraMesh3d(pillar.clone()),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: color,
                perceptual_roughness: 0.6,
                ..default()
            })),
            Transform::from_xyz(8.0 + dx, 1.3, dz),
        ));
    }
}

/// A gate (two posts, a lintel and the portal quad) at `at`, front facing its local +Z,
/// under `world` when given. Returns the quad.
#[allow(clippy::too_many_arguments)]
fn spawn_gate(
    commands: &mut Commands,
    world: Option<Entity>,
    at: Transform,
    frame_mesh: &Handle<AuroraMesh>,
    lintel_mesh: &Handle<AuroraMesh>,
    frame_mat: &Handle<AuroraMaterial>,
    quad: &Handle<AuroraMesh>,
    quad_mat: &Handle<AuroraMaterial>,
) -> Entity {
    let mut root = commands.spawn((at, Visibility::default()));
    if let Some(world) = world {
        root.insert(ChildOf(world));
    }
    let root = root.id();
    for px in [-1.1, 1.1] {
        commands.spawn((
            ChildOf(root),
            AuroraMesh3d(frame_mesh.clone()),
            AuroraMaterial3d(frame_mat.clone()),
            Transform::from_xyz(px, 1.7, 0.0),
        ));
    }
    commands.spawn((
        ChildOf(root),
        AuroraMesh3d(lintel_mesh.clone()),
        AuroraMaterial3d(frame_mat.clone()),
        Transform::from_xyz(0.0, 3.5, 0.0),
    ));
    commands
        .spawn((
            ChildOf(root),
            AuroraMesh3d(quad.clone()),
            AuroraMaterial3d(quad_mat.clone()),
            Transform::from_xyz(0.0, 1.7, 0.0),
        ))
        .id()
}
