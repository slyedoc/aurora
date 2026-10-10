//! Northshire as a layered landscape. Each of zero's nine baked ADTs is one child in both
//! stacks: its heightmap (`Replace`) and its splat (WoW's alpha layers as per-texel weights,
//! over the tileset palette). A stamp sits on top (`Max` with falloff; a 16-bit PNG from the
//! command line, else a generated volcano), erosion baked over it, then a sculpt layer, a slope rule (rock on steep
//! ground, off at start) and a paint layer. Every layer stays editable. Grass grows on the GPU
//! from the splat's grass, flower, scrub and moss textures around the camera, swaying through
//! 12 baked wind poses.
//!
//! ```text
//! cargo run -r -p aurora_landscape --example wow_zone [stamp.png]
//! ```
//!
//! Fly: WASD, mouse (free camera). Stamp: arrows move, Z / X turn, PageUp / PageDown lift,
//! B cycles the blend, H hides it. Sculpt at the screen centre: R raise, F lower, G smooth,
//! T flatten (to the height under the crosshair when pressed); Y paint, U erase paint, N / M
//! cycle the paint material; [ ] radius. J toggles the rock rule, K re-bakes the erosion, O
//! toggles it. F5 saves the sculpt layer,
//! F9 loads it. F12 screenshot.

use std::path::Path;

use aurora_landscape::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    gizmos::GizmoPlugin,
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    render_device::RenderDevice,
    util::ScreenshotExt,
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

/// zero's baked map data (`wow_import`): `<stem>_height.png` + `<stem>_layers.json` per ADT.
const MAP: &str = "/home/slyedoc/code/p/zero/assets/wow/map";
const TILE: f32 = 1600.0 / 3.0;
const TILES_X: [i32; 3] = [31, 32, 33];
const TILES_Y: [i32; 3] = [47, 48, 49];
const ANCHOR: (i32, i32) = (32, 48);
const GROUND_H: f32 = 82.178;
const SCULPT_FILE: &str = "target/wow_zone.sculpt";
/// WoW repeats each ground texture 8 times per chunk (16 chunks per ADT).
const TEXTURE_REPEAT: f32 = TILE / 128.0;
/// `palette.json`'s rock.
const ROCK: u8 = 1;
/// Grass on `palette.json`'s grass, flowers, scrub and moss.
const GRASS_GROWS_ON: [(u8, f32); 4] = [(3, 1.0), (5, 0.8), (4, 0.6), (0, 0.5)];

fn main() -> AppExit {
    App::new()
        .add_plugins((AuroraDefaultPlugins, GizmoPlugin, FreeCameraPlugin, LandscapePlugin))
        .init_resource::<Brush>()
        .add_systems(Startup, setup)
        .add_systems(Update, (move_stamp, sculpt, save_load))
        .add_screenshot(KeyCode::F12)
        .run()
}

#[derive(Component)]
struct Stamp;

#[derive(Component)]
struct Sculpt;

#[derive(Component)]
struct Paint;

#[derive(Component)]
struct Erosion;

#[derive(Component)]
struct RockRule;

#[derive(Resource)]
struct Brush {
    radius: f32,
    flatten: Option<f32>,
    material: u8,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            radius: 12.0,
            flatten: None,
            material: 5,
        }
    }
}

/// A single-channel f32 image (aurora keeps R32 images off the GPU texture path).
fn height_image(width: u32, height: u32, samples: &[f32]) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(samples).to_vec(),
        TextureFormat::R32Float,
        RenderAssetUsages::MAIN_WORLD,
    )
}

fn load_png16(path: &Path) -> Option<Image> {
    let img = image::open(path)
        .inspect_err(|e| error!("{}: {e}", path.display()))
        .ok()?
        .into_luma16();
    let samples: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 65535.0).collect();
    Some(height_image(img.width(), img.height(), &samples))
}

/// A cone with a crater and radial ridges, 0..1.
fn volcano(n: u32) -> Image {
    let mut samples = Vec::with_capacity((n * n) as usize);
    for z in 0..n {
        for x in 0..n {
            let p = Vec2::new(x as f32, z as f32) / (n - 1) as f32 * 2.0 - 1.0;
            let r = p.length();
            let cone = (1.0 - r).max(0.0).powf(1.4);
            let ridges = 1.0 + 0.12 * (p.y.atan2(p.x) * 9.0 + r * 14.0).sin();
            let crater = 1.0 - 0.35 * (1.0 - (r / 0.18).clamp(0.0, 1.0)).powi(2);
            samples.push((cone * ridges * crater).clamp(0.0, 1.0));
        }
    }
    height_image(n, n, &samples)
}

/// A tuft of tapered blades; `bend` leans every tip toward +x (metres per metre of height).
fn grass_tuft(bend: f32) -> AuroraMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut h = 0x9e37_79b9u32;
    let mut rnd = || {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        h as f32 / u32::MAX as f32
    };
    const SEGMENTS: u32 = 4;
    for _ in 0..9 {
        let base = Vec3::new(rnd() * 0.3 - 0.15, 0.0, rnd() * 0.3 - 0.15);
        let yaw = rnd() * std::f32::consts::TAU;
        let side = Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let lean = Vec3::new(rnd() - 0.5, 0.0, rnd() - 0.5) * 0.3;
        let height = 0.35 + rnd() * 0.3;
        let width = 0.035;
        let first = positions.len() as u32;
        for i in 0..=SEGMENTS {
            let t = i as f32 / SEGMENTS as f32;
            let centre = base
                + Vec3::Y * height * t * (1.0 - 0.25 * bend.abs() * t)
                + (lean + Vec3::X * bend) * height * t * t;
            let half = side * width * 0.5 * (1.0 - t);
            let up = (Vec3::Y + (lean + Vec3::X * bend) * 2.0 * t).normalize();
            let n = side.cross(up).normalize();
            for (p, u) in [(centre - half, 0.0), (centre + half, 1.0)] {
                positions.push(p.to_array());
                normals.push(n.to_array());
                uvs.push([u, t]);
            }
        }
        for i in 0..SEGMENTS {
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

fn r32_image(width: u32, height: u32, words: &[u32]) -> Image {
    Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(words).to_vec(),
        TextureFormat::R32Uint,
        RenderAssetUsages::MAIN_WORLD,
    )
}

/// An ADT's splat as (ids, weights) words: per alpha texel the chunk's four layers, WoW's
/// sequential lerp (`base`, then layers 1..3 by the alpha's r, g, b) turned into weights.
fn adt_splat(dir: &Path, stem: &str, json: &serde_json::Value) -> Option<(u32, Vec<u32>, Vec<u32>)> {
    let alpha = image::open(dir.join(format!("{stem}_alpha.png")))
        .inspect_err(|e| error!("{stem}: {e}"))
        .ok()?
        .into_rgba8();
    let n = alpha.width();
    let cell = n / 16;
    let chunks: Vec<[u32; 4]> = json["chunks"]
        .as_array()?
        .iter()
        .map(|c| {
            let mut e = [u32::MAX; 4];
            for (i, v) in c.as_array().into_iter().flatten().take(4).enumerate() {
                e[i] = v.as_u64().unwrap_or(u32::MAX as u64) as u32;
            }
            e
        })
        .collect();
    let mut ids = Vec::with_capacity((n * n) as usize);
    let mut weights = Vec::with_capacity((n * n) as usize);
    for y in 0..n {
        for x in 0..n {
            let layers = chunks.get(((y / cell) * 16 + x / cell) as usize)?;
            let px = alpha.get_pixel(x, y).0;
            let a = |i: usize| {
                if layers[i] == u32::MAX {
                    0.0
                } else {
                    px[i - 1] as f32 / 255.0
                }
            };
            let (a1, a2, a3) = (a(1), a(2), a(3));
            let w = [
                (1.0 - a1) * (1.0 - a2) * (1.0 - a3),
                a1 * (1.0 - a2) * (1.0 - a3),
                a2 * (1.0 - a3),
                a3,
            ];
            let (mut id_word, mut weight_word) = (0u32, 0u32);
            for i in 0..4 {
                let id = if layers[i] == u32::MAX { 0 } else { layers[i] & 0xFF };
                id_word |= id << (8 * i);
                weight_word |= ((w[i] * 255.0).round() as u32) << (8 * i);
            }
            ids.push(id_word);
            weights.push(weight_word);
        }
    }
    Some((n, ids, weights))
}

/// The tileset named by `palette.json`, as landscape materials.
fn wow_palette(images: &mut Assets<Image>) -> Vec<LandscapeMaterial> {
    let dir = Path::new(MAP);
    let Some(json) = std::fs::read_to_string(dir.join("palette.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
    else {
        error!("{MAP}/palette.json missing");
        return Vec::new();
    };
    json["textures"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|name| {
            let name = name.as_str()?;
            let img = image::open(dir.join("tileset").join(name))
                .inspect_err(|e| error!("{name}: {e}"))
                .ok()?
                .into_rgba8();
            let image = Image::new(
                Extent3d {
                    width: img.width(),
                    height: img.height(),
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                img.into_raw(),
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            );
            Some(LandscapeMaterial {
                albedo: Some(images.add(image)),
                tile_size: TEXTURE_REPEAT,
                roughness: 0.9,
                ..default()
            })
        })
        .collect()
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let half = Vec2::splat(TILE * 1.5);
    let landscape = commands
        .spawn((
            Name::new("Northshire"),
            Landscape::covering(-half, half, 0.5),
            Transform::from_xyz(0.0, -GROUND_H, 0.0),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::srgb(0.32, 0.38, 0.22),
                perceptual_roughness: 0.9,
                ..default()
            })),
            LandscapePalette(wow_palette(&mut images)),
        ))
        .id();

    for ay in TILES_Y {
        for ax in TILES_X {
            let stem = format!("azeroth_{ax}_{ay}");
            let dir = Path::new(MAP);
            let Some(image) = load_png16(&dir.join(format!("{stem}_height.png"))) else {
                continue;
            };
            let json: serde_json::Value = match std::fs::read_to_string(
                dir.join(format!("{stem}_layers.json")),
            )
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            {
                Ok(json) => json,
                Err(e) => {
                    error!("{stem}: {e}");
                    continue;
                }
            };
            let mut tile = commands.spawn((
                Name::new(stem.clone()),
                ChildOf(landscape),
                HeightmapLayer {
                    image: images.add(image),
                    size: Vec2::splat(TILE),
                    height_min: json["height_min"].as_f64().unwrap_or(0.0) as f32,
                    height_max: json["height_max"].as_f64().unwrap_or(0.0) as f32,
                    falloff: 0.0,
                },
                Transform::from_xyz(
                    (ANCHOR.0 - ax) as f32 * TILE,
                    0.0,
                    (ANCHOR.1 - ay) as f32 * TILE,
                ),
            ));
            if let Some((n, ids, weights)) = adt_splat(dir, &stem, &json) {
                tile.insert(SplatmapLayer {
                    ids: images.add(r32_image(n, n, &ids)),
                    weights: images.add(r32_image(n, n, &weights)),
                    size: Vec2::splat(TILE),
                });
            }
        }
    }

    let stamp = std::env::args()
        .nth(1)
        .and_then(|p| load_png16(Path::new(&p)))
        .unwrap_or_else(|| volcano(512));
    commands.spawn((
        Name::new("Stamp"),
        Stamp,
        ChildOf(landscape),
        HeightLayer {
            blend: HeightBlend::Max,
            ..default()
        },
        HeightmapLayer {
            image: images.add(stamp),
            size: Vec2::splat(420.0),
            height_min: GROUND_H - 10.0,
            height_max: GROUND_H + 170.0,
            falloff: 0.3,
        },
        Transform::from_xyz(220.0, 0.0, -180.0),
    ));
    commands.spawn((
        Name::new("Erosion"),
        Erosion,
        ChildOf(landscape),
        ErosionLayer {
            size: Vec2::splat(600.0),
            bake: true,
            ..default()
        },
        Transform::from_xyz(220.0, 0.0, -180.0),
    ));
    commands.spawn((Name::new("Sculpt"), Sculpt, ChildOf(landscape), SculptLayer::default()));
    commands.spawn((
        Name::new("Rock on steep ground"),
        RockRule,
        ChildOf(landscape),
        MaterialLayer {
            enabled: false,
            ..default()
        },
        MaterialRuleLayer {
            material: ROCK,
            slope: Vec2::new(32.0, 90.0),
            slope_fade: 6.0,
            ..default()
        },
    ));
    commands.spawn((Name::new("Paint"), Paint, ChildOf(landscape), PaintLayer::default()));

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
            spacing: 0.4,
            radius: 35.0,
            grows_on: GRASS_GROWS_ON.to_vec(),
            poses: poses.clone(),
            ..default()
        },
    ));
    // The mid tier: a sparser grid of bigger tufts from where the near tier fades out.
    commands.spawn((
        Name::new("Grass (mid)"),
        ChildOf(landscape),
        AuroraMesh3d(poses[0].clone()),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.22, 0.42, 0.10),
            perceptual_roughness: 0.6,
            ..default()
        })),
        ScatterSpecies {
            spacing: 1.1,
            radius: 140.0,
            inner_radius: 29.0,
            fade: 8.0,
            scale: Vec2::new(1.6, 2.4),
            grows_on: GRASS_GROWS_ON.to_vec(),
            poses,
            seed: 2,
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        LandscapeViewer,
        Transform::from_xyz(0.0, 340.0, 650.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 100_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.6, -1.3, 0.0)),
    ));
    info!(
        "stamp: arrows move, Z/X turn, PgUp/PgDn lift, B blend, H hide; sculpt at the \
         crosshair: R raise, F lower, G smooth, T flatten, Y paint, U erase, N/M material, \
         [ ] radius; J rock rule, K bake erosion, O erosion on/off; F5 save, F9 load"
    );
}

fn move_stamp(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut stamp: Single<(&mut Transform, &mut HeightLayer), With<Stamp>>,
    mut erosion: Single<(&mut ErosionLayer, &mut HeightLayer), (With<Erosion>, Without<Stamp>)>,
) {
    if keys.just_pressed(KeyCode::KeyK) {
        erosion.0.bake = true;
        info!("erosion: baking");
    }
    if keys.just_pressed(KeyCode::KeyO) {
        erosion.1.enabled = !erosion.1.enabled;
        info!("erosion {}", if erosion.1.enabled { "on" } else { "off" });
    }
    let (transform, layer) = &mut *stamp;
    let dt = time.delta_secs();
    let axis = |neg, pos| keys.pressed(pos) as i32 as f32 - keys.pressed(neg) as i32 as f32;
    let step = Vec3::new(
        axis(KeyCode::ArrowLeft, KeyCode::ArrowRight),
        axis(KeyCode::PageDown, KeyCode::PageUp) * 0.5,
        axis(KeyCode::ArrowUp, KeyCode::ArrowDown),
    ) * 80.0
        * dt;
    if step != Vec3::ZERO {
        transform.translation += step;
    }
    let turn = axis(KeyCode::KeyX, KeyCode::KeyZ);
    if turn != 0.0 {
        transform.rotate_y(turn * dt);
    }
    if keys.just_pressed(KeyCode::KeyB) {
        layer.blend = match layer.blend {
            HeightBlend::Max => HeightBlend::Add,
            HeightBlend::Add => HeightBlend::Min,
            HeightBlend::Min => HeightBlend::Replace,
            HeightBlend::Replace => HeightBlend::Max,
        };
        info!("stamp blend: {:?}", layer.blend);
    }
    if keys.just_pressed(KeyCode::KeyH) {
        layer.enabled = !layer.enabled;
    }
}

fn sculpt(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut brush: ResMut<Brush>,
    mut dabs: ResMut<SculptDabs>,
    camera: Single<&GlobalTransform, With<LandscapeViewer>>,
    landscape: Single<(&LandscapePages, &GlobalTransform)>,
    layer: Single<Entity, With<Sculpt>>,
    paint: Single<Entity, With<Paint>>,
    mut rule: Single<&mut MaterialLayer, With<RockRule>>,
    palette: Single<&LandscapePalette>,
    mut gizmos: Gizmos,
) {
    let count = palette.0.len().max(1) as u8;
    if keys.just_pressed(KeyCode::KeyN) {
        brush.material = (brush.material + count - 1) % count;
        info!("paint material {}", brush.material);
    }
    if keys.just_pressed(KeyCode::KeyM) {
        brush.material = (brush.material + 1) % count;
        info!("paint material {}", brush.material);
    }
    if keys.just_pressed(KeyCode::KeyJ) {
        rule.enabled = !rule.enabled;
        info!("rock rule {}", if rule.enabled { "on" } else { "off" });
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        brush.radius = (brush.radius / 1.25).max(1.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        brush.radius = (brush.radius * 1.25).min(200.0);
    }
    let (pages, at) = *landscape;
    let origin = camera.translation() - at.translation();
    let Some(hit) = pages.raycast(origin, *camera.forward(), 3000.0) else {
        return;
    };
    gizmos.circle(
        Isometry3d::new(
            hit + at.translation() + Vec3::Y * 0.3,
            Quat::from_rotation_arc(Vec3::Z, Vec3::Y),
        ),
        brush.radius,
        Color::srgb(1.0, 0.8, 0.2),
    );
    let dt = time.delta_secs();
    let kind = if keys.pressed(KeyCode::KeyR) {
        BrushKind::Raise
    } else if keys.pressed(KeyCode::KeyF) {
        BrushKind::Lower
    } else if keys.pressed(KeyCode::KeyG) {
        BrushKind::Smooth
    } else if keys.pressed(KeyCode::KeyT) {
        BrushKind::Flatten
    } else if keys.pressed(KeyCode::KeyY) {
        BrushKind::Paint
    } else if keys.pressed(KeyCode::KeyU) {
        BrushKind::Erase
    } else {
        brush.flatten = None;
        return;
    };
    let target = *brush.flatten.get_or_insert(hit.y);
    dabs.0.push(SculptDab {
        layer: if kind.paints() { *paint } else { *layer },
        center: hit.xz(),
        radius: brush.radius,
        strength: match kind {
            BrushKind::Raise | BrushKind::Lower => 12.0 * dt,
            _ => 4.0 * dt,
        },
        kind,
        target,
        material: brush.material,
    });
}

fn save_load(
    keys: Res<ButtonInput<KeyCode>>,
    rd: Res<RenderDevice>,
    mut layer: Single<(&mut SculptPages, &mut HeightLayer), With<Sculpt>>,
) {
    let (pages, heights) = &mut *layer;
    if keys.just_pressed(KeyCode::F5) {
        match std::fs::write(SCULPT_FILE, pages.to_bytes()) {
            Ok(()) => info!("saved {SCULPT_FILE}"),
            Err(e) => error!("{SCULPT_FILE}: {e}"),
        }
    }
    if keys.just_pressed(KeyCode::F9) {
        match std::fs::read(SCULPT_FILE)
            .map_err(|e| e.to_string())
            .and_then(|b| pages.load(&rd, &b).map_err(|e| e.to_string()))
        {
            Ok(()) => {
                // Re-evaluate under the layer.
                heights.set_changed();
                info!("loaded {SCULPT_FILE}");
            }
            Err(e) => error!("{SCULPT_FILE}: {e}"),
        }
    }
}
