//! A closed room lit by the sun through one small window: the patch of sunlight on the floor is
//! the only direct light, everything else is bounce light -- the case 1-spp path tracing handles
//! worst and ReSTIR PT is for. A red block and a blue cabinet for colour bleed, a mirror sphere
//! and a glossy one, and a clear glass sphere.
//!
//!   cargo run --release --example sun_room [-- --no-gi] [--no-restir] [--nrc] [--no-nrc] [--nrc-lr L] [--glass R] [--no-glass] [--matte-spheres] [--fog D] [--samples N] [--ev100 E] [--fullscreen] [--raw]
//!       [--capture DIR --size WxH --frames N --noisy K]
//!       [--gi-spatial N] [--gi-radius R] [--pt-history M] [--bloom B] [--local-tonemap T]
//!       [--flat-sky] [--no-sun] [--no-ground] [--bounces N] [--camera x,y,z]
//!
//! The flags change `RenderSettings` at startup (and pin the exposure), for comparing against a
//! reference (e.g. `--no-gi --samples 4 --ev100 6`).

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    auto_exposure::{AuroraExposure, ev_from_ev100},
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    render_settings::RenderSettings,
    sky::{Fog, Sky},
    sphere::Sphere,
    util::{
        ScreenshotExt, TimeoutAppExt,
        capture::{self, CaptureArgs},
    },
};

/// The room's inside: x in [-HALF_X, HALF_X], z in [-HALF_Z, HALF_Z], floor at 0.
const HALF_X: f32 = 4.0;
const HALF_Z: f32 = 3.0;
const HEIGHT: f32 = 3.0;
const WALL: f32 = 0.2;
/// The window in the south wall (z = -HALF_Z): x in [-WINDOW_HALF, WINDOW_HALF], y in WINDOW_Y.
const WINDOW_HALF: f32 = 0.6;
const WINDOW_Y: (f32, f32) = (1.0, 2.2);

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
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0);
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

/// `--no-gi`, `--no-restir`, `--samples N`, `--gi-spatial N`.
fn apply_flags(mut settings: ResMut<RenderSettings>) {
    let args: Vec<String> = std::env::args().collect();
    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--no-gi" => settings.restir_gi = false,
            "--no-restir" => settings.restir = false,
            "--nrc" => settings.nrc = true,
            "--no-nrc" => settings.nrc = false,
            "--nrc-lr" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                    settings.nrc_learning_rate = v;
                }
            }
            "--bounces" => {
                if let Some(n) = args.get(i + 1).and_then(|n| n.parse().ok()) {
                    settings.max_bounces = n;
                }
            }
            "--di-history" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                    settings.restir_history = v;
                }
            }
            "--di-spatial" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                    settings.restir_spatial_samples = v;
                }
            }
            "--ab" => {
                if let Some(v) = args.get(i + 1).and_then(|v| v.parse().ok()) {
                    settings.ab = v;
                }
            }
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
            "--bloom" => {
                if let Some(b) = args.get(i + 1).and_then(|b| b.parse().ok()) {
                    settings.bloom = b;
                }
            }
            "--local-tonemap" => {
                if let Some(t) = args.get(i + 1).and_then(|t| t.parse().ok()) {
                    settings.local_tonemap = t;
                }
            }
            "--boiling" => {
                if let Some(b) = args.get(i + 1).and_then(|b| b.parse().ok()) {
                    settings.restir_boiling = b;
                }
            }
            "--pt-history" => {
                if let Some(m) = args.get(i + 1).and_then(|m| m.parse().ok()) {
                    settings.restir_pt_history = m;
                }
            }
            "--gi-radius" => {
                if let Some(r) = args.get(i + 1).and_then(|r| r.parse().ok()) {
                    settings.restir_gi_spatial_radius = r;
                }
            }
            "--gi-spatial" => {
                if let Some(n) = args.get(i + 1).and_then(|n| n.parse().ok()) {
                    settings.restir_gi_spatial_samples = n;
                }
            }
            _ => {}
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

fn camera_position() -> Vec3 {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == "--camera")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| {
            let c: Vec<f32> = v.split(',').filter_map(|x| x.parse().ok()).collect();
            (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
        })
        .unwrap_or(Vec3::new(3.4, 1.6, 2.5))
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
            // `--flat-sky`: a uniform sky (6000 nits) instead of the generated one.
            if std::env::args().any(|a| a == "--flat-sky") {
                Sky::Color {
                    radiance: Vec3::splat(6000.0),
                }
            } else {
                Sky::Generated
            },
        ))
        .id();
    // `--fog DENSITY`: Northshire's kind of haze (per metre, forward scattering, thinning with
    // height).
    let args: Vec<String> = std::env::args().collect();
    if let Some(density) = args
        .iter()
        .position(|a| a == "--fog")
        .and_then(|at| args.get(at + 1)?.parse::<f32>().ok())
    {
        commands.entity(environment).insert(Fog {
            density,
            scatter: 0.85,
            height: 4.0,
            falloff: 20.0,
            grid_slices: 16,
            ..default()
        });
    }

    // The sun, high in the south, its beam through the window onto the floor.
    // `--no-sun`: none (the sky alone lights the room).
    if !std::env::args().any(|a| a == "--no-sun") {
        commands.spawn((
            ChildOf(environment),
            DirectionalLight::default(),
            Transform::default().looking_to(Vec3::new(0.3, -0.64, 1.0), Vec3::Y),
        ));
    }

    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        // `--camera x,y,z`: another eye point, looking at the same spot.
        Transform::from_translation(camera_position())
            .looking_at(Vec3::new(-1.0, 0.7, -1.4), Vec3::Y),
    ));

    let cube = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    let block =
        |commands: &mut Commands, material: Handle<AuroraMaterial>, min: Vec3, max: Vec3| {
            commands.spawn((
                ChildOf(environment),
                AuroraMesh3d(cube.clone()),
                AuroraMaterial3d(material),
                Transform::from_translation((min + max) * 0.5).with_scale(max - min),
            ));
        };

    let wall = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.8, 0.78, 0.72),
        perceptual_roughness: 0.9,
        ..default()
    });
    let floor = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.45, 0.3, 0.18),
        perceptual_roughness: 0.6,
        ..default()
    });
    let ground = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.25, 0.3, 0.2),
        perceptual_roughness: 1.0,
        ..default()
    });
    let (x0, x1, z0, z1) = (-HALF_X - WALL, HALF_X + WALL, -HALF_Z - WALL, HALF_Z + WALL);

    // Outside: ground around the house, just under its floor (`--no-ground`: none).
    if !std::env::args().any(|a| a == "--no-ground") {
        block(
            &mut commands,
            ground,
            Vec3::new(-60.0, -0.3, -60.0),
            Vec3::new(60.0, -0.01, 60.0),
        );
    }
    // Floor and ceiling.
    block(
        &mut commands,
        floor,
        Vec3::new(x0, -0.2, z0),
        Vec3::new(x1, 0.0, z1),
    );
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(x0, HEIGHT, z0),
        Vec3::new(x1, HEIGHT + WALL, z1),
    );
    // North, east and west walls.
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(x0, 0.0, HALF_Z),
        Vec3::new(x1, HEIGHT, z1),
    );
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(HALF_X, 0.0, -HALF_Z),
        Vec3::new(x1, HEIGHT, HALF_Z),
    );
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(x0, 0.0, -HALF_Z),
        Vec3::new(-HALF_X, HEIGHT, HALF_Z),
    );
    // The south wall around its window.
    let (wy0, wy1) = WINDOW_Y;
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(x0, 0.0, z0),
        Vec3::new(-WINDOW_HALF, HEIGHT, -HALF_Z),
    );
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(WINDOW_HALF, 0.0, z0),
        Vec3::new(x1, HEIGHT, -HALF_Z),
    );
    block(
        &mut commands,
        wall.clone(),
        Vec3::new(-WINDOW_HALF, 0.0, z0),
        Vec3::new(WINDOW_HALF, wy0, -HALF_Z),
    );
    block(
        &mut commands,
        wall,
        Vec3::new(-WINDOW_HALF, wy1, z0),
        Vec3::new(WINDOW_HALF, HEIGHT, -HALF_Z),
    );

    // Furniture: a red block beside the sunlit patch, a blue cabinet against the west wall.
    let red = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.75, 0.08, 0.06),
        perceptual_roughness: 0.8,
        ..default()
    });
    let blue = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.08, 0.15, 0.7),
        perceptual_roughness: 0.7,
        ..default()
    });
    block(
        &mut commands,
        red,
        Vec3::new(0.8, 0.0, -1.2),
        Vec3::new(1.8, 0.9, -0.2),
    );
    block(
        &mut commands,
        blue,
        Vec3::new(-HALF_X, 0.0, 0.5),
        Vec3::new(-HALF_X + 0.6, 1.8, 2.0),
    );

    // Two metal spheres, 0.8 m across: a perfect mirror and a glossy one (roughness 0.2);
    // `--matte-spheres`: both white and diffuse.
    let matte = std::env::args().any(|a| a == "--matte-spheres");
    for (x, z, roughness) in [(-1.3, 0.6, 0.0), (-0.55, -0.25, 0.2)] {
        commands.spawn((
            ChildOf(environment),
            Sphere,
            AuroraMaterial3d(materials.add(if matte {
                AuroraMaterial {
                    base_color: Color::srgb(0.8, 0.8, 0.8),
                    metallic: 0.0,
                    perceptual_roughness: 0.9,
                    ..default()
                }
            } else {
                AuroraMaterial {
                    base_color: Color::srgb(0.95, 0.8, 0.55),
                    metallic: 1.0,
                    perceptual_roughness: roughness,
                    ..default()
                }
            })),
            Transform::from_xyz(x, 0.4, z).with_scale(Vec3::splat(0.8)),
        ));
    }

    // A clear glass sphere (IOR 1.5) on the floor in front of the box: smooth, `--glass R` for
    // roughness R, `--no-glass` to leave it out.
    if !std::env::args().any(|a| a == "--no-glass") {
        let roughness = std::env::args()
            .position(|a| a == "--glass")
            .and_then(|i| std::env::args().nth(i + 1)?.parse().ok())
            .unwrap_or(0.0);
        commands.spawn((
            ChildOf(environment),
            Sphere,
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::WHITE,
                metallic: 0.0,
                perceptual_roughness: roughness,
                specular_transmission: 1.0,
                ior: 1.5,
                ..default()
            })),
            Transform::from_xyz(0.6, 0.4, 0.9).with_scale(Vec3::splat(0.8)),
        ));
    }
}
