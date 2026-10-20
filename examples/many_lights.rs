//! Many lights: 441 emissive spheres over 8000 random cubes on a mirror floor, under a black sky
//! (bevy's solari_many_lights, ported). Lamp light, bounce light and nothing else: the scene for
//! judging ReSTIR DI and GI (no sun or sky to hide the indirect).
//!
//!   cargo run --release --example many_lights [-- --no-gi] [--no-restir] [--samples N] [--ev100 E] [--fullscreen] [--raw]
//!       [--capture DIR --size WxH --frames N --noisy K]
//!
//! The flags change `RenderSettings` at startup, for comparing against a reference (e.g.
//! `--no-gi --no-restir --samples 4`).

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    auto_exposure::{AuroraExposure, ev_from_ev100},
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    render_settings::RenderSettings,
    sky::Sky,
    sphere::Sphere,
    util::{
        ScreenshotExt,
        capture::{self, CaptureArgs},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The original's lamp emission per channel, at most (nits).
const LAMP_NITS: f32 = 60_000.0;

fn main() {
    let mut app = App::new();
    // `--capture DIR ...`: windowless into a fixed-size image, a reference capture, then exit
    // (bevy_aurora::util::capture).
    let capture = CaptureArgs::from_args();
    if capture.is_some() {
        app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    } else {
        app.add_plugins((
            // `--fullscreen`: borderless on the primary monitor, so every run has its size (a
            // tiling compositor gives a window whatever space is left).
            AuroraDefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    mode: if std::env::args().any(|a| a == "--fullscreen") {
                        bevy::window::WindowMode::BorderlessFullscreen(
                            bevy::window::MonitorSelection::Primary,
                        )
                    } else {
                        bevy::window::WindowMode::Windowed
                    },
                    ..default()
                }),
                close_when_requested: false,
                ..default()
            }),
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_screenshot(KeyCode::F12);
    }
    app.add_systems(Startup, (setup, apply_flags))
        .add_systems(Update, pin_exposure);
    match capture {
        Some(args) => capture::run(&mut app, &args),
        None => {
            app.run();
        }
    }
}

/// `--ev100 E`: a fixed exposure on every camera, so variants compare at one brightness.
fn pin_exposure(
    mut commands: Commands,
    cameras: Query<(Entity, Option<&AuroraExposure>), With<Camera3d>>,
    mut want: Local<Option<Option<AuroraExposure>>>,
) {
    let want = want.get_or_insert_with(|| {
        let args: Vec<String> = std::env::args().collect();
        let at = args.iter().position(|a| a == "--ev100")?;
        let ev100: f32 = args.get(at + 1)?.parse().ok()?;
        Some(AuroraExposure::fixed(ev_from_ev100(ev100)))
    });
    let Some(want) = want.as_ref() else {
        return;
    };
    for (camera, exposure) in &cameras {
        if exposure != Some(want) {
            commands.entity(camera).insert(want.clone());
        }
    }
}

/// `--no-gi`, `--no-restir`, `--samples N`.
fn apply_flags(mut settings: ResMut<RenderSettings>) {
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--no-gi" => settings.restir_gi = false,
            "--no-restir" => settings.restir = false,
            // No post-processing: the traced image exposed and shown linearly (fog is off
            // anyway: these environments have no `Fog`).
            "--raw" => {
                settings.bloom = 0.0;
                settings.local_tonemap = 0.0;
                settings.tonemap = false;
            }
            "--samples" => {
                if let Some(n) = args.get(i + 1).and_then(|n| n.parse().ok()) {
                    settings.samples = n;
                }
            }
            _ => {}
        }
    }
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    asset_server: Res<AssetServer>,
) {
    let environment = commands
        .spawn((
            Name::new("Environment"),
            Environment::default(),
            Sky::Color {
                radiance: Vec3::ZERO,
            },
        ))
        .id();

    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera {
            walk_speed: 50.0,
            run_speed: 250.0,
            ..default()
        },
        Transform::from_translation(Vec3::new(6.11329, 166.74896, 451.8226)).with_rotation(
            Quat::from_xyzw(-0.183938, 0.009093744, 0.0017017953, 0.9828943),
        ),
    ));

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // The floor: 400 m, its UVs tiled 3x, a mirror.
    let mut floor = Plane3d::default().mesh().size(400.0, 400.0).build();
    if let Some(bevy::mesh::VertexAttributeValues::Float32x2(uvs)) =
        floor.attribute_mut(Mesh::ATTRIBUTE_UV_0)
    {
        uvs.iter_mut().flatten().for_each(|x| *x *= 3.0);
    }
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(meshes.add(AuroraMesh::from_mesh(&floor).expect("floor mesh"))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color_texture: Some(asset_server.load(aurora_asset("textures/uv_checker_bw.png"))),
            perceptual_roughness: 0.0,
            ..default()
        })),
    ));

    let cube = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    for _ in 0..8000 {
        commands.spawn((
            ChildOf(environment),
            AuroraMesh3d(cube.clone()),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::srgb(rng.r#gen(), rng.r#gen(), rng.r#gen()),
                perceptual_roughness: rng.r#gen(),
                ..default()
            })),
            Transform::from_translation(Vec3::new(
                rng.gen_range(-180.0..=180.0),
                0.2,
                rng.gen_range(-180.0..=180.0),
            ))
            .with_scale(Vec3::new(
                rng.gen_range(0.2..=2.0),
                rng.gen_range(0.2..=2.0),
                rng.gen_range(0.2..=2.0),
            )),
        ));
    }

    // 21 x 21 lamps, radius 1 m (the analytic sphere is 1 m across at scale 1).
    for x in -10..=10 {
        for z in -10..=10 {
            commands.spawn((
                ChildOf(environment),
                Sphere,
                AuroraMaterial3d(materials.add(AuroraMaterial {
                    emissive: LinearRgba::rgb(
                        rng.r#gen::<f32>() * LAMP_NITS,
                        rng.r#gen::<f32>() * LAMP_NITS,
                        rng.r#gen::<f32>() * LAMP_NITS,
                    ),
                    ..default()
                })),
                Transform::from_translation(Vec3::new((x * 20) as f32, 7.0, (z * 20) as f32))
                    .with_scale(Vec3::splat(2.0)),
            ));
        }
    }
}
