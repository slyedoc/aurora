//! Ray-portal demo: two gates on a plain, paired both ways. Looking into the LEFT gate
//! shows the view out of the RIGHT gate's front (the red/blue pillar cluster), and vice
//! versa -- portals in reflections and through glass come free from the raygen redirect.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    portal::AuroraPortal,
    ray_default_plugins::RayDefaultPlugins,
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
};

const SKY_SCALE_NITS: f32 = 8000.0;

fn main() {
    App::new()
        .add_plugins((RayDefaultPlugins, DevShaderPlugin, DevUIPlugin, FreeCameraPlugin))
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
    commands.insert_resource(Sky::Hdr {
        image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
        scale: SKY_SCALE_NITS,
    });

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        // Looking at gate A with gate B far off to the right: the through-view must show
        // B's pillars, not the empty plain behind A.
        Transform::from_xyz(-8.0, 1.7, 6.0).looking_at(Vec3::new(-8.0, 1.5, 0.0), Vec3::Y),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(200.0, 200.0))),
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
    let frame_mesh = meshes.add(Cuboid::new(0.2, 3.4, 0.2));
    let lintel_mesh = meshes.add(Cuboid::new(2.4, 0.2, 0.2));
    // The portal surface: a quad facing +Z (Rectangle is XY-plane, normal +Z).
    let quad = meshes.add(Rectangle::new(2.0, 3.0));
    let quad_mat = materials.add(AuroraMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    // Gates A (x = -8) and B (x = +8), both fronts facing +Z.
    let mut gates = Vec::new();
    for gx in [-8.0_f32, 8.0] {
        for px in [-1.1, 1.1] {
            commands.spawn((
                Mesh3d(frame_mesh.clone()),
                AuroraMaterial3d(frame_mat.clone()),
                Transform::from_xyz(gx + px, 1.7, 0.0),
            ));
        }
        commands.spawn((
            Mesh3d(lintel_mesh.clone()),
            AuroraMaterial3d(frame_mat.clone()),
            Transform::from_xyz(gx, 3.5, 0.0),
        ));
        gates.push(
            commands
                .spawn((
                    Mesh3d(quad.clone()),
                    AuroraMaterial3d(quad_mat.clone()),
                    Transform::from_xyz(gx, 1.6, 0.0),
                ))
                .id(),
        );
    }
    commands.entity(gates[0]).insert(AuroraPortal { target: gates[1] });
    commands.entity(gates[1]).insert(AuroraPortal { target: gates[0] });

    // Pillars in front of gate B only: the tell. Seen through gate A = portals work.
    let pillar = meshes.add(Cuboid::new(0.6, 2.6, 0.6));
    for (dx, dz, color) in [
        (-1.5, 3.0, Color::srgb(0.9, 0.15, 0.1)),
        (0.0, 4.5, Color::srgb(0.1, 0.3, 0.9)),
        (1.5, 3.0, Color::srgb(0.95, 0.8, 0.1)),
    ] {
        commands.spawn((
            Mesh3d(pillar.clone()),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: color,
                perceptual_roughness: 0.6,
                ..default()
            })),
            Transform::from_xyz(8.0 + dx, 1.3, dz),
        ));
    }
}
