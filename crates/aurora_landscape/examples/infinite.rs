//! An infinite landscape: a generator layer under a streaming window of pages that follows
//! the camera. Rules paint rock on steep ground and snow up high, grass grows on the rest.
//! Every page that slides into the window evaluates on the GPU; nothing is stored.
//!
//! ```text
//! cargo run -r -p aurora_landscape --example infinite
//! ```
//!
//! Fly: WASD, mouse, shift to run (free camera). F12 screenshot.

use aurora_landscape::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    util::ScreenshotExt,
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

const GRASS: u8 = 0;
const ROCK: u8 = 1;
const SNOW: u8 = 2;
const DIRT: u8 = 3;

fn main() -> AppExit {
    App::new()
        .add_plugins((AuroraDefaultPlugins, FreeCameraPlugin, LandscapePlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, place_camera)
        .add_screenshot(KeyCode::F12)
        .run()
}

/// A flat-colour palette texture.
fn swatch(images: &mut Assets<Image>, rgb: [u8; 3]) -> Handle<Image> {
    images.add(Image::new(
        Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        [rgb[0], rgb[1], rgb[2], 255].repeat(16),
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ))
}

/// A tuft of tapered blades.
fn grass_tuft(bend: f32) -> AuroraMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut h = 0x2545_f491u32;
    let mut rnd = || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        h as f32 / u32::MAX as f32
    };
    for _ in 0..9 {
        let base = Vec3::new(rnd() * 0.3 - 0.15, 0.0, rnd() * 0.3 - 0.15);
        let yaw = rnd() * std::f32::consts::TAU;
        let side = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let lean = Vec3::new(rnd() - 0.5, 0.0, rnd() - 0.5) * 0.3 + Vec3::X * bend;
        let height = 0.35 + rnd() * 0.3;
        let first = positions.len() as u32;
        for i in 0..=4 {
            let t = i as f32 / 4.0;
            let centre = base + Vec3::Y * height * t + lean * height * t * t;
            let half = side * 0.0175 * (1.0 - t);
            let n = side.cross((Vec3::Y + lean * 2.0 * t).normalize()).normalize();
            for (p, u) in [(centre - half, 0.0), (centre + half, 1.0)] {
                positions.push(p.to_array());
                normals.push(n.to_array());
                uvs.push([u, t]);
            }
        }
        for i in 0..4 {
            let a = first + i * 2;
            indices.extend_from_slice(&[a, a + 1, a + 3, a, a + 3, a + 2]);
        }
    }
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices));
    AuroraMesh::from_mesh(&mesh).expect("grass tuft")
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let mut material = |rgb: [u8; 3], roughness: f32| LandscapeMaterial {
        albedo: Some(swatch(&mut images, rgb)),
        tile_size: 4.0,
        roughness,
        ..default()
    };
    let palette = vec![
        material([70, 110, 40], 0.9),
        material([105, 98, 90], 0.8),
        material([235, 238, 245], 0.4),
        material([110, 82, 55], 0.95),
    ];
    let landscape = commands
        .spawn((
            Name::new("Infinite"),
            Landscape {
                stream_radius: 6,
                ..default()
            },
            LandscapePalette(palette),
        ))
        .id();
    commands.spawn((Name::new("Terrain"), ChildOf(landscape), GeneratorLayer::default()));
    for (name, rule) in [
        (
            "Dirt in the low ground",
            MaterialRuleLayer {
                material: DIRT,
                slope: Vec2::new(0.0, 90.0),
                height: Vec2::new(-1.0e6, 30.0),
                height_fade: 15.0,
                ..default()
            },
        ),
        (
            "Rock on steep ground",
            MaterialRuleLayer {
                material: ROCK,
                slope: Vec2::new(30.0, 90.0),
                slope_fade: 6.0,
                ..default()
            },
        ),
        (
            "Snow up high",
            MaterialRuleLayer {
                material: SNOW,
                slope: Vec2::new(0.0, 40.0),
                slope_fade: 8.0,
                height: Vec2::new(420.0, 1.0e6),
                height_fade: 60.0,
                ..default()
            },
        ),
    ] {
        commands.spawn((Name::new(name), ChildOf(landscape), rule));
    }
    commands.spawn((Name::new("Paint"), ChildOf(landscape), PaintLayer::default()));

    const POSES: usize = 12;
    let poses: Vec<Handle<AuroraMesh>> = (0..POSES)
        .map(|k| {
            let phase = k as f32 / POSES as f32 * std::f32::consts::TAU;
            meshes.add(grass_tuft(0.08 + 0.18 * phase.sin()))
        })
        .collect();
    commands.spawn((
        Name::new("Grass"),
        ChildOf(landscape),
        AuroraMesh3d(poses[0].clone()),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.22, 0.42, 0.10),
            perceptual_roughness: 0.6,
            ..default()
        })),
        ScatterSpecies {
            spacing: 0.45,
            radius: 40.0,
            grows_on: vec![(GRASS, 1.0)],
            poses,
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera {
            run_speed: 200.0,
            ..default()
        },
        LandscapeViewer,
        Transform::from_xyz(0.0, 260.0, 0.0).looking_at(Vec3::new(300.0, 120.0, 300.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 100_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.6, -0.9, 0.0)),
    ));
}

/// Once, when the first pages are in: 40 m above the ground under the camera.
fn place_camera(
    mut placed: Local<bool>,
    mut camera: Single<&mut Transform, With<LandscapeViewer>>,
    landscape: Single<(&LandscapePages, &GlobalTransform)>,
) {
    let (pages, at) = *landscape;
    if *placed || !pages.is_complete() {
        return;
    }
    *placed = true;
    let local = camera.translation - at.translation();
    camera.translation.y = at.translation().y + pages.height_at(local.xz()) + 40.0;
}
