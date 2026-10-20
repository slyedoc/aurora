//! Bloom and glare (bevy's bloom_3d, ported): bright emissive spheres bouncing over a dark plane,
//! where the lens's scatter shows. Space toggles bloom, Q/A change its strength (also in the F3
//! panel as `RenderSettings::bloom`).
//!
//!   cargo run --release --example bloom [-- <strength>]

use bevy::prelude::*;
use bevy_aurora::{
    AuroraDefaultPlugins,
    auto_exposure::{AuroraExposure, ev_from_ev100},
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    render_settings::RenderSettings,
    sky::SkyBrightness,
    sphere::Sphere,
    util::ScreenshotExt,
};

/// A fixed exposure: auto exposure meters only the dark (emitters are left out of the meter)
/// and would lift it until everything blew out. At EV100 10 a white display is ~1200 nits.
const EV100: f32 = 10.0;

fn main() {
    App::new()
        .add_plugins((AuroraDefaultPlugins, DevUIPlugin))
        .add_systems(Startup, (setup, start_strength))
        .add_systems(Update, (controls, bounce))
        .add_screenshot(KeyCode::F12)
        .run();
}

#[derive(Component)]
struct Bouncing;

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let environment = commands
        .spawn((
            Name::new("Environment"),
            Environment::default(),
            // A black sky: the glare needs the dark around it.
            SkyBrightness(0.0),
        ))
        .id();
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        AuroraExposure::fixed(ev_from_ev100(EV100)),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        ChildOf(environment),
        Transform::from_xyz(0.0, -0.8, 0.0),
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
            Plane3d::default().mesh().size(40.0, 40.0),
        ))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.05, 0.05, 0.05),
            perceptual_roughness: 0.6,
            ..default()
        })),
    ));

    // bevy's three emitters and a black sphere, in nits: blue and red well past display white,
    // white far past it, so each one's glare reads at its own strength.
    let blue = materials.add(AuroraMaterial {
        emissive: LinearRgba::rgb(0.0, 0.0, 20_000.0),
        ..default()
    });
    let white = materials.add(AuroraMaterial {
        emissive: LinearRgba::rgb(150_000.0, 150_000.0, 150_000.0),
        ..default()
    });
    let red = materials.add(AuroraMaterial {
        emissive: LinearRgba::rgb(6_000.0, 0.0, 0.0),
        ..default()
    });
    let black = materials.add(AuroraMaterial {
        base_color: Color::BLACK,
        ..default()
    });
    for x in -5..5 {
        for z in -5..5 {
            // bevy hashes (x, z); any fixed scatter does.
            let pick = ((x * 7 + z * 13 + 100) as u32).wrapping_mul(2_654_435_761) % 6;
            let (material, scale) = match pick {
                0 => (blue.clone(), 0.5),
                1 => (white.clone(), 0.1),
                2 => (red.clone(), 1.0),
                _ => (black.clone(), 1.5),
            };
            commands.spawn((
                ChildOf(environment),
                Transform::from_xyz(x as f32 * 2.0, 0.0, z as f32 * 2.0)
                    .with_scale(Vec3::splat(scale * 0.4)),
                Sphere,
                AuroraMaterial3d(material),
                Bouncing,
            ));
        }
    }

    commands.spawn((
        Text::default(),
        Readout,
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(12),
            ..default()
        },
    ));
}

/// The optional first argument: the strength to start at (0 = off).
fn start_strength(mut settings: ResMut<RenderSettings>) {
    if let Some(strength) = std::env::args().nth(1).and_then(|a| a.parse().ok()) {
        settings.bloom = strength;
    }
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut settings: ResMut<RenderSettings>,
    mut saved: Local<Option<f32>>,
    mut text: Single<&mut Text, With<Readout>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        if settings.bloom > 0.0 {
            *saved = Some(settings.bloom);
            settings.bloom = 0.0;
        } else {
            settings.bloom = saved.take().unwrap_or(0.06);
        }
    }
    let step = 0.1 * time.delta_secs();
    if settings.bloom > 0.0 {
        if keys.pressed(KeyCode::KeyQ) {
            settings.bloom = (settings.bloom + step).min(0.3);
        }
        if keys.pressed(KeyCode::KeyA) {
            settings.bloom = (settings.bloom - step).max(0.001);
        }
    }
    text.0 = if settings.bloom > 0.0 {
        format!("Bloom (Space: off)\n(Q/A) strength: {:.3}", settings.bloom)
    } else {
        "Bloom off (Space: on)".to_string()
    };
}

fn bounce(time: Res<Time>, mut spheres: Query<&mut Transform, With<Bouncing>>) {
    for (i, mut transform) in spheres.iter_mut().enumerate() {
        let phase = i as f32 * 0.37;
        transform.translation.y = (time.elapsed_secs() * 1.5 + phase).sin().abs() * 0.6;
    }
}
