//! A jungle valley: a generated landscape with wet rock and mud (palette clear coat), broadleaf
//! trees and fern undergrowth grown on the GPU (trunks and crowns are two species on the same
//! seed, so they share their spots), leaves that let the sun through and carry dew
//! (`diffuse_transmission`, `clearcoat`), and low fog pooled in the valleys that the sun lights
//! in shafts through the canopy. Past 140 m the same trees continue as a far tier of low-poly
//! shells on the same spots, out to 600 m, so the forest never ends at a ring.
//!
//! ```text
//! cargo run -r -p aurora_landscape --example jungle
//! ```
//!
//! Fly: WASD, mouse, shift to run. F12 screenshot.

use std::f32::consts::TAU;

use aurora_landscape::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    environment::MainPhysicsEnvironmentEntity,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    sky::Fog,
    util::ScreenshotExt,
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

const FLOOR: u8 = 0;
const ROCK: u8 = 1;
const MUD: u8 = 2;
const POSES: usize = 8;

fn main() -> AppExit {
    App::new()
        .add_plugins((AuroraDefaultPlugins, FreeCameraPlugin, LandscapePlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, place_camera)
        .add_screenshot(KeyCode::F12)
        .run()
}

/// Triangles with flat normals from (a, b, c) corners, both faces traced (two-sided).
#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl Builder {
    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        let n = (b - a).cross(d - a).normalize_or(Vec3::Y);
        let first = self.positions.len() as u32;
        for (p, uv) in [(a, [0.0, 0.0]), (b, [1.0, 0.0]), (c, [1.0, 1.0]), (d, [0.0, 1.0])] {
            self.positions.push(p.to_array());
            self.normals.push(n.to_array());
            self.uvs.push(uv);
        }
        self.indices
            .extend_from_slice(&[first, first + 1, first + 2, first, first + 2, first + 3]);
    }

    fn build(self) -> AuroraMesh {
        let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
            .with_inserted_indices(Indices::U32(self.indices));
        AuroraMesh::from_mesh(&mesh).expect("procedural mesh")
    }
}

fn rng(seed: u32) -> impl FnMut() -> f32 {
    let mut h = seed.max(1);
    move || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        h as f32 / u32::MAX as f32
    }
}

/// A slightly leaning tapered trunk, 14 m.
fn trunk() -> AuroraMesh {
    trunk_of(8, 6)
}

fn trunk_of(sides: usize, rings: usize) -> AuroraMesh {
    let mut b = Builder::default();
    let height = 14.0;
    let at = |ring: usize, side: usize| {
        let t = ring as f32 / rings as f32;
        let r = 0.45 * (1.0 - t) + 0.16 * t;
        let a = side as f32 / sides as f32 * TAU;
        Vec3::new(a.cos() * r + 0.6 * t * t, t * height, a.sin() * r)
    };
    for ring in 0..rings {
        for side in 0..sides {
            b.quad(
                at(ring, side),
                at(ring, side + 1),
                at(ring + 1, side + 1),
                at(ring + 1, side),
            );
        }
    }
    b.build()
}

/// A crown of broad drooping leaves on top of the trunk; `sway` bends every tip sideways.
fn crown(sway: f32) -> AuroraMesh {
    crown_of(70, 1.0, sway)
}

/// The far tier's crown: a dozen leaves twice as wide, the same silhouette from afar.
fn crown_far() -> AuroraMesh {
    crown_of(12, 2.2, 0.0)
}

fn crown_of(leaves: usize, widen: f32, sway: f32) -> AuroraMesh {
    let mut b = Builder::default();
    let mut rnd = rng(0x51ed_270b);
    let top = Vec3::new(0.6, 14.0, 0.0);
    for _ in 0..leaves {
        let yaw = rnd() * TAU;
        let out = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let side = Vec3::new(-yaw.sin(), 0.0, yaw.cos());
        let length = 2.5 + rnd() * 2.0;
        let lift = 0.6 + rnd() * 0.8;
        let width = (0.35 + rnd() * 0.25) * widen;
        let base = top + Vec3::new(rnd() - 0.5, rnd() * 1.5 - 1.0, rnd() - 0.5) * 0.8;
        // Three segments: up and out, then drooping, the tip pushed by the wind.
        let point = |t: f32| {
            base + out * length * t + Vec3::Y * (lift * t - 1.6 * t * t) * length * 0.5
                + Vec3::X * sway * t * t * length
        };
        let half = |t: f32| side * width * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.05);
        for s in 0..3 {
            let (t0, t1) = (s as f32 / 3.0, (s + 1) as f32 / 3.0);
            b.quad(
                point(t0) - half(t0),
                point(t0) + half(t0),
                point(t1) + half(t1),
                point(t1) - half(t1),
            );
        }
    }
    b.build()
}

/// A fern: a ring of arching fronds; `sway` leans the tips.
fn fern(sway: f32) -> AuroraMesh {
    let mut b = Builder::default();
    let mut rnd = rng(0x0fe2_77aa);
    for _ in 0..9 {
        let yaw = rnd() * TAU;
        let out = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let side = Vec3::new(-yaw.sin(), 0.0, yaw.cos());
        let length = 0.7 + rnd() * 0.5;
        let point = |t: f32| {
            out * length * t * 0.8 + Vec3::Y * (1.1 * t - 0.8 * t * t) * length
                + Vec3::X * sway * t * t * length
        };
        let half = |t: f32| side * 0.09 * (1.0 - t).max(0.05);
        for s in 0..4 {
            let (t0, t1) = (s as f32 / 4.0, (s + 1) as f32 / 4.0);
            b.quad(
                point(t0) - half(t0),
                point(t0) + half(t0),
                point(t1) + half(t1),
                point(t1) - half(t1),
            );
        }
    }
    b.build()
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

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    main: Res<MainPhysicsEnvironmentEntity>,
) {
    // Fog pooled on the forest floor: thick below 85 m (the valley floors), a tenth as thick
    // 35 m higher.
    commands.entity(main.0).insert(Fog {
        density: 0.02,
        scatter: 0.75,
        height: 85.0,
        falloff: 15.0,
    });

    let mut material = |rgb: [u8; 3], roughness: f32, clearcoat: f32| LandscapeMaterial {
        albedo: Some(swatch(&mut images, rgb)),
        tile_size: 4.0,
        roughness,
        clearcoat,
        ..default()
    };
    let palette = vec![
        material([48, 58, 26], 0.9, 0.1),
        material([72, 70, 62], 0.7, 0.6),
        material([58, 42, 28], 0.6, 0.45),
    ];
    let half = Vec2::splat(600.0);
    let landscape = commands
        .spawn((
            Name::new("Jungle"),
            Landscape::covering(-half, half, 0.5),
            LandscapePalette(palette),
        ))
        .id();
    commands.spawn((
        Name::new("Hills"),
        ChildOf(landscape),
        GeneratorLayer {
            base_height: 20.0,
            hills_frequency: 1.0 / 400.0,
            hills_amplitude: 35.0,
            mountains_frequency: 1.0 / 1_500.0,
            mountains_amplitude: 120.0,
            mountains_mask_frequency: 1.0 / 5_000.0,
            mountains_mask_power: 1.0,
            detail_amplitude: 1.5,
            ..default()
        },
    ));
    for (name, rule) in [
        (
            "Mud in the low ground",
            MaterialRuleLayer {
                material: MUD,
                slope: Vec2::new(0.0, 25.0),
                height: Vec2::new(-1.0e6, 14.0),
                height_fade: 6.0,
                ..default()
            },
        ),
        (
            "Wet rock on steep ground",
            MaterialRuleLayer {
                material: ROCK,
                slope: Vec2::new(28.0, 90.0),
                slope_fade: 6.0,
                ..default()
            },
        ),
    ] {
        commands.spawn((Name::new(name), ChildOf(landscape), rule));
    }

    let leaf = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.10, 0.32, 0.05),
        perceptual_roughness: 0.45,
        diffuse_transmission: 0.45,
        clearcoat: 0.35,
        clearcoat_perceptual_roughness: 0.08,
        double_sided: true,
        ..default()
    });
    let bark = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.22, 0.16, 0.11),
        perceptual_roughness: 0.85,
        clearcoat: 0.15,
        ..default()
    });
    let crowns: Vec<Handle<AuroraMesh>> = (0..POSES)
        .map(|k| meshes.add(crown(0.06 * (k as f32 / POSES as f32 * TAU).sin())))
        .collect();
    let far_crown = meshes.add(crown_far());
    let far_trunk = meshes.add(trunk_of(4, 2));
    let ferns: Vec<Handle<AuroraMesh>> = (0..POSES)
        .map(|k| meshes.add(fern(0.12 * (k as f32 / POSES as f32 * TAU).sin())))
        .collect();
    // Trunks and crowns: the same grid, seed and scale, so each crown sits on its trunk.
    let trees = ScatterSpecies {
        spacing: 5.0,
        radius: 150.0,
        fade: 10.0,
        grows_on: vec![(FLOOR, 0.9), (MUD, 0.5)],
        scale: Vec2::new(0.75, 1.3),
        max_slope: 30.0,
        sink: 0.3,
        wind_speed: 0.2,
        seed: 7,
        ..default()
    };
    // The far tier: the same spots (grid, seed, scale), cheap meshes, from where the near
    // tier dithers out.
    let far = ScatterSpecies {
        radius: 600.0,
        inner_radius: 140.0,
        wind_speed: 0.0,
        ..trees.clone()
    };
    commands.spawn((
        Name::new("Trunks (far)"),
        ChildOf(landscape),
        AuroraMesh3d(far_trunk),
        AuroraMaterial3d(bark.clone()),
        far.clone(),
    ));
    commands.spawn((
        Name::new("Crowns (far)"),
        ChildOf(landscape),
        AuroraMesh3d(far_crown),
        AuroraMaterial3d(leaf.clone()),
        far,
    ));
    commands.spawn((
        Name::new("Trunks"),
        ChildOf(landscape),
        AuroraMesh3d(meshes.add(trunk())),
        AuroraMaterial3d(bark),
        trees.clone(),
    ));
    commands.spawn((
        Name::new("Crowns"),
        ChildOf(landscape),
        AuroraMesh3d(crowns[0].clone()),
        AuroraMaterial3d(leaf.clone()),
        ScatterSpecies {
            poses: crowns,
            ..trees
        },
    ));
    commands.spawn((
        Name::new("Ferns"),
        ChildOf(landscape),
        AuroraMesh3d(ferns[0].clone()),
        AuroraMaterial3d(leaf),
        ScatterSpecies {
            spacing: 0.9,
            radius: 40.0,
            grows_on: vec![(FLOOR, 0.8), (MUD, 0.3)],
            scale: Vec2::new(0.7, 1.4),
            max_slope: 35.0,
            poses: ferns,
            wind_speed: 0.3,
            seed: 11,
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        LandscapeViewer,
        Transform::from_xyz(0.0, 0.0, 0.0).looking_at(Vec3::new(-60.0, 0.0, -45.0), Vec3::Y),
    ));
    // Low and raking, so it comes through the canopy in shafts.
    commands.spawn((
        DirectionalLight {
            illuminance: 100_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -2.2, -0.5, 0.0)),
    ));
}

/// Once, when the pages are in: 1.7 m above the ground, looking along it.
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
    let o = at.translation();
    let ground = |xz: Vec2| o.y + pages.height_at(xz - o.xz());
    let eye = Vec2::ZERO;
    // Toward the sun: the fog scatters forward, so its shafts show against the light.
    let look = Vec2::new(-60.0, -45.0);
    info!("ground under the camera: {:.1} m", ground(eye));
    **camera = Transform::from_xyz(eye.x, ground(eye) + 1.7, eye.y).looking_at(
        Vec3::new(look.x, ground(look) + 14.0, look.y),
        Vec3::Y,
    );
}
