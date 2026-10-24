//! A candle flame: a Hanabi effect (particles.rs) whose particles rise off the wick, slow, shrink,
//! redden and die, traced as emissive spheres (shadowed, reflected) beside the point light that
//! lights the room (the particles are `seen_only`: no light counted twice).
//!
//!   cargo run --release --example candle [-- --capture DIR [--size WxH] [--frames N]]

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
    winit::WinitPlugin,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    auto_exposure::AuroraExposure,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    sky::Sky,
    util::capture::{self, CaptureArgs},
};
use bevy_hanabi::{prelude::*, Gradient};

fn main() {
    let mut app = App::new();
    // `--capture DIR ...`: windowless into a fixed-size image, a reference capture, then exit.
    let capture = CaptureArgs::from_args();
    if capture.is_some() {
        app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    } else {
        app.add_plugins((AuroraDefaultPlugins, FreeCameraPlugin));
    }
    app.add_systems(Startup, setup);
    match capture {
        Some(args) => capture::run(&mut app, &args),
        None => {
            app.run();
        }
    }
}

fn flame() -> EffectAsset {
    let writer = ExprWriter::new();
    let init_pos = SetPositionSphereModifier {
        center: writer.lit(Vec3::ZERO).expr(),
        radius: writer.lit(0.004).expr(),
        dimension: ShapeDimension::Volume,
    };
    let init_vel = SetAttributeModifier::new(
        Attribute::VELOCITY,
        (writer.lit(Vec3::new(-0.01, 0.06, -0.01)).uniform(writer.lit(Vec3::new(0.01, 0.12, 0.01))))
            .expr(),
    );
    let init_age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
    let init_lifetime = SetAttributeModifier::new(
        Attribute::LIFETIME,
        writer.lit(0.25).uniform(writer.lit(0.45)).expr(),
    );
    let lift = AccelModifier::new(writer.lit(Vec3::new(0.0, 0.3, 0.0)).expr());
    let drag = LinearDragModifier::new(writer.lit(3.0).expr());
    // White-hot at the wick, through orange, to a dull red as it dies (times the material's
    // emission).
    let mut color = Gradient::new();
    color.add_key(0.0, Vec4::new(1.0, 0.9, 0.7, 1.0));
    color.add_key(0.3, Vec4::new(1.0, 0.55, 0.15, 1.0));
    color.add_key(1.0, Vec4::new(0.25, 0.04, 0.0, 1.0));
    let mut size = Gradient::new();
    size.add_key(0.0, Vec3::splat(0.012));
    size.add_key(0.4, Vec3::splat(0.009));
    size.add_key(1.0, Vec3::splat(0.001));
    EffectAsset::new(256, SpawnerSettings::rate(300.0.into()), writer.finish())
        .with_name("candle flame")
        .with_simulation_space(SimulationSpace::Global)
        .init(init_pos)
        .init(init_vel)
        .init(init_age)
        .init(init_lifetime)
        .update(lift)
        .update(drag)
        .render(ColorOverLifetimeModifier::new(color))
        .render(SizeOverLifetimeModifier {
            gradient: size,
            screen_space_size: false,
        })
}

fn setup(
    mut commands: Commands,
    mut effects: ResMut<Assets<EffectAsset>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let environment = commands
        .spawn((Name::new("Environment"), Environment::default(), Sky::Color {
            radiance: Vec3::ZERO,
        }))
        .id();
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        AuroraExposure::Auto(default()),
        Transform::from_xyz(0.0, 0.22, 0.28).looking_at(Vec3::new(0.0, 0.16, 0.0), Vec3::Y),
    ));

    let cube = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    let wood = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.25, 0.15, 0.08),
        perceptual_roughness: 0.6,
        ..default()
    });
    let wax = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.9, 0.85, 0.75),
        perceptual_roughness: 0.4,
        ..default()
    });
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(cube.clone()),
        AuroraMaterial3d(wood),
        Transform::from_xyz(0.0, -0.025, 0.0).with_scale(Vec3::new(2.0, 0.05, 2.0)),
    ));
    commands.spawn((
        ChildOf(environment),
        AuroraMesh3d(cube),
        AuroraMaterial3d(wax),
        Transform::from_xyz(0.0, 0.075, 0.0).with_scale(Vec3::new(0.04, 0.15, 0.04)),
    ));

    // The flame lights the room through its point light; its particles are what one sees.
    commands.spawn((
        ChildOf(environment),
        Name::new("Flame light"),
        PointLight {
            color: Color::srgb(1.0, 0.6, 0.3),
            intensity: 12.0,
            radius: 0.01,
            ..default()
        },
        Transform::from_xyz(0.0, 0.17, 0.0),
    ));
    commands.spawn((
        ChildOf(environment),
        Name::new("Flame"),
        ParticleEffect::new(effects.add(flame())),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::rgb(9000.0, 9000.0, 9000.0),
            instance_tint: true,
            // The point light carries its light.
            seen_only: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.155, 0.0),
    ));
}
