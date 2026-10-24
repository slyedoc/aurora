//! A row of working telescopes built from glass (ported from the bevy solari branch's scope
//! example), side by side and all aimed at one target: each is designed by `bevy_aurora::optics`
//! from a magnification, a length and an objective radius, its barrel sized from the lenses, and
//! labelled. Galilean scopes (a concave eyepiece) show the world upright; Keplerian ones (a
//! convex eyepiece, a wider field) upside down, with a crosshair where their objective's image
//! forms. The magnification, the image's way up, the eye-box and the edge distortion are not
//! coded anywhere: they come from rays refracting through the glass.
//!
//! The target is 10 m off, a checker patch at its upper right (it stays there through a
//! Galilean and turns to the lower left through a Keplerian). Walk up to an eyepiece and put the
//! eye in its cup (the camera has a dark disc behind it for a head, which keeps the daylight
//! behind the eye off the eyepiece); back away to watch the image shrink to the exit pupil, step
//! off the axis to watch the eye-box clip.
//!
//!   cargo run --release --example scope [-- --capture DIR --size WxH --frames N --noisy K]

use std::f32::consts::PI;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    optics::{Telescope, TelescopeKind, spawn_telescope},
    sky::Sky,
    ui_panel::{UiPanel3d, UiPanel3dRoot},
    util::{
        ScreenshotExt,
        capture::{self, CaptureArgs},
    },
};

/// Optical axis height above the ground.
const AXIS_Y: f32 = 0.30;
/// The scopes' spacing along the row.
const SPACING: f32 = 0.15;
/// What they all look at.
const TARGET: Vec3 = Vec3::new(0.0, AXIS_Y, -10.0);

/// The scopes along the row: (magnification, length between the lenses, objective radius).
fn scopes() -> [Telescope; 4] {
    [
        Telescope::galilean(2.0, 0.10, 0.02),
        Telescope::galilean(4.0, 0.144, 0.03),
        Telescope::keplerian(4.0, 0.25, 0.02),
        Telescope::keplerian(8.0, 0.40, 0.025),
    ]
}

/// A label's text, written into its panel once the panel is built.
#[derive(Component)]
struct Label(String);

fn main() {
    let mut app = App::new();
    // `--capture DIR ...`: windowless into a fixed-size image, a reference capture, then exit.
    let capture = CaptureArgs::from_args();
    if capture.is_some() {
        app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    } else {
        app.add_plugins((AuroraDefaultPlugins, DevUIPlugin, FreeCameraPlugin))
            .add_screenshot(KeyCode::F12);
    }
    app.add_systems(Startup, setup)
        .add_systems(Update, write_labels);
    match capture {
        Some(args) => capture::run(&mut app, &args),
        None => {
            app.run();
        }
    }
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
        ))
        .id();
    commands.spawn((
        ChildOf(environment),
        DirectionalLight::default(),
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.6, -0.9, 0.0)),
    ));

    let mut matte = |r: f32, g: f32, b: f32| {
        materials.add(AuroraMaterial {
            base_color: Color::srgb(r, g, b),
            perceptual_roughness: 0.9,
            ..default()
        })
    };
    let black = matte(0.02, 0.02, 0.025);
    let cream = matte(0.9, 0.9, 0.85);
    let red = matte(0.8, 0.1, 0.1);
    let dark = matte(0.1, 0.1, 0.1);
    let ground = matte(0.35, 0.36, 0.33);

    // Each scope's own frame: its middle on the row, its axis (-Z) on the target.
    let scopes = scopes();
    let roots: Vec<_> = scopes
        .iter()
        .enumerate()
        .map(|(i, scope)| {
            let x = (i as f32 - (scopes.len() - 1) as f32 / 2.0) * SPACING;
            let root = commands
                .spawn((
                    ChildOf(environment),
                    Name::new(label(scope)),
                    Transform::from_xyz(x, AXIS_Y, 0.0).looking_at(TARGET, Vec3::Y),
                    Visibility::default(),
                ))
                .id();
            (root, scope)
        })
        .collect();

    let mut cups = Vec::new();
    for &(root, scope) in &roots {
        let barrel = spawn_telescope(
            &mut commands,
            &mut meshes,
            &mut materials,
            root,
            scope,
            &black,
        );
        let foot = AXIS_Y - barrel.outer;
        commands.spawn((
            ChildOf(root),
            AuroraMesh3d(meshes.add(AuroraMesh::from_shape(
                Cuboid::new(0.02, foot, 0.02).mesh().build(),
            ))),
            AuroraMaterial3d(black.clone()),
            Transform::from_xyz(0.0, foot / 2.0 - AXIS_Y, (barrel.front + barrel.back) / 2.0),
        ));
        cups.push((root, scope, barrel.cup_end));
    }

    let mut part =
        |parent: Entity, mesh: Mesh, material: &Handle<AuroraMaterial>, at: Transform| {
            commands.spawn((
                ChildOf(parent),
                AuroraMesh3d(meshes.add(AuroraMesh::from_shape(mesh))),
                AuroraMaterial3d(material.clone()),
                at,
            ));
        };

    // The target: rings sized so the naked eye barely makes them out at 10 m, a checker patch
    // at their upper right inside every scope's field.
    let rings = [&cream, &red, &cream, &red, &dark];
    for (k, (radius, material)) in [0.50, 0.40, 0.30, 0.20, 0.10]
        .into_iter()
        .zip(rings)
        .enumerate()
    {
        part(
            environment,
            Cylinder::new(radius, 0.01).mesh().build(),
            material,
            Transform::from_translation(TARGET + Vec3::Z * k as f32 * 0.006)
                .with_rotation(Quat::from_rotation_x(PI / 2.0)),
        );
    }
    const CHECKS: i32 = 6;
    const CHECK: f32 = 0.015;
    for cx in 0..CHECKS {
        for cy in 0..CHECKS {
            let material = if (cx + cy) % 2 == 0 { &cream } else { &dark };
            part(
                environment,
                Cuboid::new(CHECK, CHECK, 0.01).mesh().build(),
                material,
                Transform::from_translation(
                    TARGET + Vec3::new(0.08 + cx as f32 * CHECK, 0.08 + cy as f32 * CHECK, 0.04),
                ),
            );
        }
    }
    part(
        environment,
        Plane3d::default().mesh().size(40.0, 40.0).build(),
        &ground,
        Transform::from_xyz(0.0, 0.0, -10.0),
    );

    // Each label stands over its eyepiece end, tipped back toward the observer.
    for (root, scope, cup_end) in cups {
        commands.spawn((
            ChildOf(root),
            Label(label(scope)),
            UiPanel3d {
                // Paper white, whatever the time of day.
                nits: 1.0,
                relative: true,
                ..UiPanel3d::sized(0.13, 2.0, 4000.0)
            },
            Transform::from_xyz(0.0, 0.07, cup_end).with_rotation(Quat::from_rotation_x(-0.35)),
        ));
    }

    // The observer, behind the row; its head a dark disc behind the eye.
    let head = meshes.add(AuroraMesh::from_shape(Cylinder::new(0.1, 0.002)));
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera {
            walk_speed: 0.25,
            run_speed: 1.5,
            ..default()
        },
        Transform::from_xyz(0.0, 0.45, 0.75).looking_at(Vec3::new(0.0, AXIS_Y, 0.0), Vec3::Y),
        children![(
            AuroraMesh3d(head),
            AuroraMaterial3d(black),
            Transform::from_xyz(0.0, 0.0, 0.04).with_rotation(Quat::from_rotation_x(PI / 2.0)),
        )],
    ));
}

/// "Galilean 4x" and what it is built of.
fn label(scope: &Telescope) -> String {
    let way_up = match scope.kind {
        TelescopeKind::Galilean => "upright",
        TelescopeKind::Keplerian => "inverted + crosshair",
    };
    format!(
        "{:?} {}x\n{:.0} mm objective\n{way_up}",
        scope.kind,
        scope.magnification,
        scope.objective.aperture * 2000.0,
    )
}

/// Each label's text, once its panel's UI root exists.
fn write_labels(
    mut commands: Commands,
    fresh: Query<(&Label, &UiPanel3dRoot), Added<UiPanel3dRoot>>,
) {
    for (label, built) in &fresh {
        commands.spawn((
            ChildOf(built.root),
            Text::new(label.0.clone()),
            TextFont {
                font_size: FontSize::Px(48.0),
                ..default()
            },
            TextColor(Color::WHITE),
        ));
    }
}
