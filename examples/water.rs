//! A pond for water work (docs/plans/water.md): one water plane at y = 0 over a shore that
//! slopes from the bank into 3 m of water, then a flat bottom out to a far bank. White tiles lie
//! on the slope at known depths (0.25, 0.5, 1, 1.5, 2, 3 m) to read the absorption by; a
//! striped pole stands through the surface (refraction's kink, the fade along it); an orange
//! ball floats half under.
//!
//! The water is a `WaterMaterial::CLEAR_LAKE` asset; the panel on the left edits it live (F3: the
//! render settings).
//!
//!   cargo run --release --example water

use std::any::TypeId;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{theme::ThemeBackgroundColor, tokens},
    feathers_inspector::BuildAssetInspector,
    prelude::*,
    ui_widgets::ScrollArea,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    sky::Sky,
    sphere::Sphere,
    util::ScreenshotExt,
    water_volume::{WaterMaterial, WaterMaterial3d, WaterSurface, WaterVolume},
};

/// The bank's edge (where the slope starts, just above the water) and the slope's foot.
const SHORE_Z: f32 = -4.0;
const SHORE_Y: f32 = 0.2;
const FOOT_Z: f32 = 12.0;
const DEPTH: f32 = 3.0;
const FAR_Z: f32 = 40.0;
const HALF_W: f32 = 20.0;
/// The tiles' depths below the surface (metres).
const TILE_DEPTHS: [f32; 6] = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0];

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

/// The floor's height under z: the bank, the slope, the flat bottom.
fn floor_y(z: f32) -> f32 {
    if z <= SHORE_Z {
        SHORE_Y
    } else if z >= FOOT_Z {
        -DEPTH
    } else {
        SHORE_Y - (SHORE_Y + DEPTH) * (z - SHORE_Z) / (FOOT_Z - SHORE_Z)
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut water_materials: ResMut<Assets<WaterMaterial>>,
) {
    let environment = commands
        .spawn((
            Name::new("Environment"),
            Environment::default(),
            Sky::Generated,
        ))
        .id();

    // Low in the north: its glint lies on the water in front of the eye.
    commands.spawn((
        ChildOf(environment),
        DirectionalLight::default(),
        Transform::default().looking_to(Vec3::new(-0.3, -0.45, -1.0), Vec3::Y),
    ));

    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 1.7, -7.0).looking_at(Vec3::new(0.0, -1.0, 4.0), Vec3::Y),
    ));

    let cube = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    let block =
        |commands: &mut Commands, material: &Handle<AuroraMaterial>, min: Vec3, max: Vec3| {
            commands.spawn((
                ChildOf(environment),
                AuroraMesh3d(cube.clone()),
                AuroraMaterial3d(material.clone()),
                Transform::from_translation((min + max) * 0.5).with_scale(max - min),
            ));
        };

    let grass = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.22, 0.3, 0.14),
        perceptual_roughness: 1.0,
        ..default()
    });
    let mud = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.32, 0.26, 0.17),
        perceptual_roughness: 0.9,
        ..default()
    });
    let white = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.85),
        perceptual_roughness: 0.8,
        ..default()
    });
    let black = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.03, 0.03, 0.03),
        perceptual_roughness: 0.8,
        ..default()
    });

    // The near bank, the far bank and the side walls (the pond is closed, so no ray leaves
    // under the water into nothing).
    block(
        &mut commands,
        &grass,
        Vec3::new(-HALF_W, -DEPTH - 0.4, -30.0),
        Vec3::new(HALF_W, SHORE_Y, SHORE_Z),
    );
    block(
        &mut commands,
        &grass,
        Vec3::new(-HALF_W - 2.0, -DEPTH - 0.4, FAR_Z),
        Vec3::new(HALF_W + 2.0, 1.2, FAR_Z + 8.0),
    );
    for x in [-HALF_W - 2.0, HALF_W] {
        block(
            &mut commands,
            &grass,
            Vec3::new(x, -DEPTH - 0.4, SHORE_Z),
            Vec3::new(x + 2.0, 0.6, FAR_Z),
        );
    }
    // The flat bottom.
    block(
        &mut commands,
        &mud,
        Vec3::new(-HALF_W, -DEPTH - 0.4, FOOT_Z),
        Vec3::new(HALF_W, -DEPTH, FAR_Z),
    );
    // The slope: a slab tilted about x, its top face from the bank's edge to the foot.
    let run = FOOT_Z - SHORE_Z;
    let drop = SHORE_Y + DEPTH;
    let angle = (drop / run).atan();
    let tilt = Quat::from_rotation_x(angle);
    let normal = tilt * Vec3::Y;
    let thickness = 0.4;
    let mid = Vec3::new(0.0, SHORE_Y - drop * 0.5, SHORE_Z + run * 0.5);
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(cube.clone()),
        AuroraMaterial3d(mud.clone()),
        Transform::from_translation(mid - normal * thickness * 0.5)
            .with_rotation(tilt)
            .with_scale(Vec3::new(
                HALF_W * 2.0,
                thickness,
                (run * run + drop * drop).sqrt(),
            )),
    ));

    // Depth tiles on the slope (x = -2), each just proud of it.
    for depth in TILE_DEPTHS {
        let z = SHORE_Z + (SHORE_Y + depth) / drop * run;
        let on = Vec3::new(-2.0, floor_y(z), z);
        commands.spawn((
            ChildOf(environment),
            Name::new(format!("Tile {depth} m")),
            AuroraMesh3d(cube.clone()),
            AuroraMaterial3d(white.clone()),
            Transform::from_translation(on + normal * 0.03)
                .with_rotation(tilt)
                .with_scale(Vec3::new(0.7, 0.05, 0.7)),
        ));
    }

    // A pole through the surface at x = 2, white and black every 0.25 m from the floor to 1 m up.
    let (pole_x, pole_z) = (2.0, 2.0);
    let bottom = floor_y(pole_z);
    let stripes = ((1.0 - bottom) / 0.25).ceil() as i32;
    for i in 0..stripes {
        let y0 = bottom + i as f32 * 0.25;
        block(
            &mut commands,
            if i % 2 == 0 { &white } else { &black },
            Vec3::new(pole_x - 0.06, y0, pole_z - 0.06),
            Vec3::new(pole_x + 0.06, y0 + 0.25, pole_z + 0.06),
        );
    }

    // A ball floating half under, the swimmer's stand-in.
    commands.spawn((
        ChildOf(environment),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.85, 0.35, 0.08),
            perceptual_roughness: 0.6,
            ..default()
        })),
        Transform::from_xyz(4.0, 0.0, 4.0).with_scale(Vec3::splat(1.2)),
    ));

    // The water: a plane, seen from both sides, is the surface (Fresnel and refraction); the
    // volume under it, down to the bottom, is the medium, a shared material.
    let water = WaterMaterial::CLEAR_LAKE;
    let ior = water.ior;
    let water = water_materials.add(water);
    commands.spawn((
        ChildOf(environment),
        Name::new("Water"),
        AuroraMesh3d(
            meshes.add(AuroraMesh::from_shape(
                Plane3d::new(
                    Vec3::Y,
                    Vec2::new(HALF_W + 2.0, (FAR_Z - SHORE_Z) * 0.5 + 2.0),
                )
                .mesh(),
            )),
        ),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.0,
            specular_transmission: 1.0,
            ior,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        WaterSurface,
        Transform::from_xyz(0.0, 0.0, (SHORE_Z + FAR_Z) * 0.5),
    ));
    let bottom = -DEPTH - 0.4;
    commands.spawn((
        ChildOf(environment),
        Name::new("Water Volume"),
        WaterVolume {
            half_extents: Vec3::new(HALF_W, -bottom * 0.5, (FAR_Z - SHORE_Z) * 0.5),
        },
        WaterMaterial3d(water.clone()),
        Transform::from_xyz(0.0, bottom * 0.5, (SHORE_Z + FAR_Z) * 0.5),
    ));

    // The water's inspector, on the left (the dev panel, F3, keeps the right).
    let panel = commands
        .spawn((
            Name::new("Water Panel"),
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
            // Scrolls with the mouse wheel when the window is too short for it.
            ScrollArea,
            ThemeBackgroundColor(tokens::WINDOW_BG),
        ))
        .id();
    commands.queue(BuildAssetInspector {
        asset_id: water.id().untyped(),
        type_id: TypeId::of::<WaterMaterial>(),
        panel,
    });
}
