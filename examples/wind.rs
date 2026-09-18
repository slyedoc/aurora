//! Wind on a field of grass cards: every card is a cutout quad cross carrying a sway weight
//! in its skin stream (`JOINT_WEIGHT.x`, 0 at the root, 1 at the tip) and a phase id
//! (`JOINT_INDEX.x`), deformed each frame by the `WindSway` path under the global `Wind`
//! resource (edit it in the F1 world inspector).
//!
//!   AUTO_SCREENSHOT_MS=6000 AURORA_EXIT_SECS=8 cargo run --example wind

use bevy::{
    asset::RenderAssetUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    image::Image,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use bevy_aurora::{
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AlphaMode, AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    skinning::{Wind, WindSway},
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

const CARDS_PER_SIDE: u32 = 40;
const SPACING: f32 = 0.35;
const HEIGHT: f32 = 0.6;

fn main() {
    App::new()
        .add_plugins((
            RayDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .insert_resource(Wind {
            speed: 6.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

/// A blade silhouette: opaque inside a tapered strip, transparent outside.
fn blade_texture() -> Image {
    const N: u32 = 64;
    let mut data = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        let t = y as f32 / (N - 1) as f32; // 0 at the top row
        let half_width = 0.42 * (1.0 - t) + 0.06 * t;
        for x in 0..N {
            let u = x as f32 / (N - 1) as f32 - 0.5;
            let inside = u.abs() < half_width;
            let shade = 0.55 + 0.45 * t;
            data.extend_from_slice(&[
                (0.35 * shade * 255.0) as u8,
                (0.62 * shade * 255.0) as u8,
                (0.18 * shade * 255.0) as u8,
                if inside { 255 } else { 0 },
            ]);
        }
    }
    Image::new(
        Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    )
}

/// One chunk of cards on a jittered grid, rooted on y = 0, as a single mesh with the sway
/// weights and phase ids the wind kernel reads.
fn field_mesh(seed: u32) -> Mesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut joints: Vec<[u16; 4]> = Vec::new();
    let mut weights: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let hash = |a: u32, b: u32| -> f32 {
        let mut n = a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663) ^ seed.wrapping_mul(83492791);
        n ^= n >> 13;
        n = n.wrapping_mul(0x5bd1e995);
        n ^= n >> 15;
        (n & 0xffff) as f32 / 65535.0
    };
    let extent = CARDS_PER_SIDE as f32 * SPACING;
    for gx in 0..CARDS_PER_SIDE {
        for gz in 0..CARDS_PER_SIDE {
            let id = gx * CARDS_PER_SIDE + gz;
            let x = gx as f32 * SPACING - extent * 0.5 + (hash(gx, gz) - 0.5) * SPACING;
            let z = gz as f32 * SPACING - extent * 0.5 + (hash(gz, gx) - 0.5) * SPACING;
            let h = HEIGHT * (0.7 + 0.6 * hash(id, 7));
            let angle = hash(id, 11) * std::f32::consts::PI;
            // Two crossed cards, three rows of vertices each (root, mid, tip).
            for card in 0..2u32 {
                let a = angle + card as f32 * std::f32::consts::FRAC_PI_2;
                let (dx, dz) = (a.cos() * 0.12, a.sin() * 0.12);
                let base = positions.len() as u32;
                for row in 0..3u32 {
                    let w = row as f32 / 2.0;
                    let y = h * w;
                    for side in 0..2u32 {
                        let s = side as f32 * 2.0 - 1.0;
                        positions.push([x + dx * s, y, z + dz * s]);
                        normals.push([0.0, 1.0, 0.0]);
                        uvs.push([side as f32, 1.0 - w]);
                        joints.push([(id & 0xffff) as u16, 0, 0, 0]);
                        weights.push([w * w, 0.0, 0.0, 0.0]);
                    }
                }
                for row in 0..2u32 {
                    let r0 = base + row * 2;
                    let r1 = r0 + 2;
                    indices.extend_from_slice(&[r0, r0 + 1, r1 + 1, r0, r1 + 1, r1]);
                }
            }
        }
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_JOINT_INDEX,
        VertexAttributeValues::Uint16x4(joints),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, weights)
    .with_inserted_indices(Indices::U32(indices))
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
) {
    commands.insert_resource(Sky::Hdr {
        image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
        scale: 8000.0,
    });
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 2.2, 9.0).looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(80.0, 80.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::linear_rgb(0.10, 0.14, 0.05),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));
    let grass = materials.add(AuroraMaterial {
        base_color_texture: Some(images.add(blade_texture())),
        perceptual_roughness: 0.9,
        alpha_mode: AlphaMode::Mask(0.5),
        ..default()
    });
    // Four chunks: two sway, two are rigid for comparison (the far pair).
    for (i, (x, z, sway)) in [
        (-7.5, 0.0, true),
        (7.5, 0.0, true),
        (-7.5, -15.0, false),
        (7.5, -15.0, false),
    ]
    .into_iter()
    .enumerate()
    {
        let mut entity = commands.spawn((
            Mesh3d(meshes.add(field_mesh(i as u32))),
            AuroraMaterial3d(grass.clone()),
            Transform::from_xyz(x, 0.0, z),
        ));
        if sway {
            entity.insert(WindSway::default());
        }
    }
}
