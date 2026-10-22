//! Weather around the camera (src/weather.rs): ground mist that follows the terrain and rain
//! that falls about the camera, over rolling hills with a hollow in the middle (where the mist
//! pools, over a pond: rain rings it), a stand of trees under a low sun (shafts through the mist)
//! and a lamp post (its glow in the mist). The panel on the left edits the `Weather` live.
//!
//!   cargo run --release --example weather

use std::any::TypeId;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{theme::ThemeBackgroundColor, tokens},
    feathers_inspector::BuildComponentInspector,
    mesh::VertexAttributeValues,
    prelude::*,
    ui_widgets::ScrollArea,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    sky::{Fog, Sky},
    sphere::Sphere,
    util::ScreenshotExt,
    water::WaterKind,
    weather::Weather,
};

/// The ground's half size (metres).
const HALF: f32 = 120.0;

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins.set(WindowPlugin {
                close_when_requested: false,
                ..default()
            }),
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_screenshot(KeyCode::F12)
        .add_systems(Startup, setup)
        .run();
}

/// Rolling hills, a hollow at the middle.
fn ground_y(x: f32, z: f32) -> f32 {
    let hills = 3.0 * (x * 0.045).sin() * (z * 0.038).cos()
        + 1.2 * (x * 0.13 + 1.0).sin() * (z * 0.11).sin();
    let hollow = -4.0 * (-(x * x + z * z) / (2.0 * 22.0 * 22.0)).exp();
    hills + hollow
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let environment = commands
        .spawn((
            Name::new("Environment"),
            Environment::default(),
            Sky::Generated,
            // A thin haze, and the fog grid that lights the mist.
            Fog {
                density: 0.002,
                ..default()
            },
            Weather::RAIN,
        ))
        .id();
    commands.spawn((
        ChildOf(environment),
        DirectionalLight::default(),
        Transform::default().looking_to(Vec3::new(0.6, -0.35, 0.5), Vec3::Y),
    ));
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(-30.0, ground_y(-30.0, -30.0) + 1.7, -30.0)
            .looking_at(Vec3::new(0.0, ground_y(0.0, 0.0) + 1.0, 0.0), Vec3::Y),
    ));

    // The ground: a subdivided plane bent into the hills.
    let mut mesh: Mesh = Plane3d::default()
        .mesh()
        .size(HALF * 2.0, HALF * 2.0)
        .subdivisions(239)
        .into();
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for p in positions.iter_mut() {
            p[1] = ground_y(p[0], p[2]);
        }
    }
    mesh.compute_normals();
    commands.spawn((
        ChildOf(environment),
        Name::new("Ground"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_mesh(&mesh).expect("the plane has positions"))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.2, 0.28, 0.12),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    // A pond in the hollow (the plane's edges lie under the slopes).
    commands.spawn((
        ChildOf(environment),
        Name::new("Pond"),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::new(Vec3::Y, Vec2::splat(18.0)).mesh(),
        ))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            perceptual_roughness: 0.0,
            specular_transmission: 1.0,
            ior: 1.33,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        WaterKind::Pond,
        Transform::from_xyz(0.0, ground_y(0.0, 0.0) + 1.8, 0.0),
    ));

    // A stand of trees: trunks and round crowns.
    let trunk = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    let bark = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.25, 0.17, 0.1),
        perceptual_roughness: 0.9,
        ..default()
    });
    let leaves = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.12, 0.25, 0.08),
        perceptual_roughness: 0.8,
        diffuse_transmission: 0.3,
        ..default()
    });
    let mut seed = 7u32;
    let mut next = || {
        seed = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    for _ in 0..40 {
        let x = 10.0 + next() * 50.0;
        let z = -35.0 + next() * 70.0;
        let y = ground_y(x, z);
        let tall = 6.0 + next() * 4.0;
        commands.spawn((
            ChildOf(environment),
            AuroraMesh3d(trunk.clone()),
            AuroraMaterial3d(bark.clone()),
            Transform::from_xyz(x, y + tall * 0.5, z).with_scale(Vec3::new(0.5, tall, 0.5)),
        ));
        commands.spawn((
            ChildOf(environment),
            Sphere,
            AuroraMaterial3d(leaves.clone()),
            Transform::from_xyz(x, y + tall, z).with_scale(Vec3::splat(4.0 + next() * 2.0)),
        ));
    }

    // A lamp post in the hollow.
    let (lx, lz) = (-6.0, 4.0);
    let ly = ground_y(lx, lz);
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(trunk.clone()),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.05, 0.05, 0.05),
            perceptual_roughness: 0.5,
            metallic: 1.0,
            ..default()
        })),
        Transform::from_xyz(lx, ly + 1.5, lz).with_scale(Vec3::new(0.12, 3.0, 0.12)),
    ));
    commands.spawn((
        ChildOf(environment),
        PointLight {
            intensity: 2000.0,
            color: Color::srgb(1.0, 0.75, 0.45),
            ..default()
        },
        Transform::from_xyz(lx, ly + 3.2, lz),
    ));

    let panel = commands
        .spawn((
            Name::new("Weather Panel"),
            Node {
                position_type: PositionType::Absolute,
                left: px(16),
                top: px(16),
                width: px(340),
                max_height: vh(90),
                padding: UiRect::all(px(8)),
                flex_direction: FlexDirection::Column,
                overflow: Overflow::scroll_y(),
                ..default()
            },
            ScrollArea,
            ThemeBackgroundColor(tokens::WINDOW_BG),
        ))
        .id();
    commands.queue(BuildComponentInspector {
        target: environment,
        type_id: TypeId::of::<Weather>(),
        panel,
    });
}
