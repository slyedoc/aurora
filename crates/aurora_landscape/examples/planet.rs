//! A small planet: six landscapes, one per cube-sphere face, each a generator layer with rules
//! (snow up high, rock on steep ground, sand at the shore line) evaluated into its pages and
//! pushed out onto the sphere. The generator samples its noise at the sphere point, so the
//! faces meet without seams.
//!
//! ```text
//! cargo run -r -p aurora_landscape --example planet
//! ```
//!
//! Fly: WASD, mouse, shift to run (1.5 km/s). F12 screenshot.

use aurora_landscape::prelude::*;
use bevy::{
    asset::RenderAssetUsages,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins, environment::MainPhysicsEnvironmentEntity, sky::Fog,
    util::ScreenshotExt,
};
use wgpu_types::{Extent3d, TextureDimension, TextureFormat};

/// Six 1 km pages across each face (4 m texels).
const RADIUS: f32 = 3072.0;
const TEXEL: f32 = 4.0;

const ROCK: u8 = 1;
const SNOW: u8 = 2;
const SAND: u8 = 3;

fn main() -> AppExit {
    App::new()
        .add_plugins((AuroraDefaultPlugins, FreeCameraPlugin, LandscapePlugin))
        .add_systems(Startup, setup)
        .add_screenshot(KeyCode::F12)
        .run()
}

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
    main: Res<MainPhysicsEnvironmentEntity>,
) {
    // Space is clear: the default haze (1 km mean free path) would swallow the view from orbit.
    commands.entity(main.0).insert(Fog {
        density: 0.0,
        ..default()
    });
    let mut material = |rgb: [u8; 3], roughness: f32| LandscapeMaterial {
        albedo: Some(swatch(&mut images, rgb)),
        tile_size: 8.0,
        roughness,
        ..default()
    };
    let palette = LandscapePalette(vec![
        material([62, 98, 38], 0.9),
        material([100, 94, 86], 0.8),
        material([235, 238, 245], 0.4),
        material([196, 180, 132], 0.95),
    ]);
    let planet = commands
        .spawn((Name::new("Planet"), Transform::default(), Visibility::default()))
        .id();
    for face in 0..6u8 {
        let landscape = commands
            .spawn((
                Name::new(format!("Face {face}")),
                ChildOf(planet),
                Landscape {
                    max_lod: 5,
                    ..Landscape::planet_face(face, RADIUS, TEXEL)
                },
                palette.clone(),
            ))
            .id();
        commands.spawn((
            Name::new("Terrain"),
            ChildOf(landscape),
            GeneratorLayer {
                base_height: 20.0,
                hills_frequency: 1.0 / 900.0,
                hills_amplitude: 40.0,
                mountains_frequency: 1.0 / 2_500.0,
                mountains_amplitude: 260.0,
                mountains_mask_frequency: 1.0 / 6_000.0,
                mountains_mask_power: 1.0,
                detail_amplitude: 2.0,
                detail_frequency: 1.0 / 60.0,
                ..default()
            },
        ));
        for (name, rule) in [
            (
                "Sand at the shore",
                MaterialRuleLayer {
                    material: SAND,
                    slope: Vec2::new(0.0, 25.0),
                    height: Vec2::new(-1.0e6, 12.0),
                    height_fade: 6.0,
                    ..default()
                },
            ),
            (
                "Rock on steep ground",
                MaterialRuleLayer {
                    material: ROCK,
                    slope: Vec2::new(28.0, 90.0),
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
                    height: Vec2::new(170.0, 1.0e6),
                    height_fade: 30.0,
                    ..default()
                },
            ),
        ] {
            commands.spawn((Name::new(name), ChildOf(landscape), rule));
        }
    }

    commands.spawn((
        Camera3d::default(),
        FreeCamera {
            walk_speed: 200.0,
            run_speed: 1500.0,
            ..default()
        },
        LandscapeViewer,
        Transform::from_xyz(2500.0, 3500.0, 6500.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 100_000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.7, -0.6, 0.0)),
    ));
}
