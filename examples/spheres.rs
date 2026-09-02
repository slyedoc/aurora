use std::f32::consts::PI;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    sky::Sky,
    sphere::Sphere,
    util::{ScreenshotExt, TimeoutAppExt},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// HDR sky brightness: texel value x this = nits (a clear sky sits around 2000-8000).
const SKY_SCALE_NITS: f32 = 8000.0;

/// Light-sphere brightness in nits (cd/m^2) -- physical units; below ~10k nits an emitter
/// disappears against a daylit sky (see aurora_files/lighting_units.md).
const EMISSIVE_NITS: f32 = 1_000_000.0;

fn main() {
    App::new()
        .add_plugins((
            RayDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
            // The render-free fill half; the engine's GizmoRenderPlugin draws it.
            bevy::gizmos::GizmoPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, debug_gizmos)
        // F12 captures to target/tmp; AUTO_SCREENSHOT_MS + the CLAUDECODE auto-exit make
        // headless agent runs self-verifying.
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // HDR environment sky (equirectangular), texel x scale = nits.
    commands.insert_resource(Sky::Hdr {
        image: asset_server.load(aurora_asset("sky/symmetrical_garden_4k.hdr")),
        scale: SKY_SCALE_NITS,
    });

    // camera
    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0 * 3.1415926 / 180.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 1.0, 7.0).looking_at(Vec3::new(2.0, 1.0, 0.0), Vec3::Y),
    ));

    // plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(100.0, 100.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.1, 0.2, 0.1),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.5, 0.0)).with_scale(Vec3::splat(3.0)),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(1.0, 0.8, 0.8),
            ..default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(3.8, 1.5, 0.0)).with_scale(Vec3::splat(3.0)),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(1.0, 1.0, 1.0),
            perceptual_roughness: 0.00,
            ior: 1.05,
            specular_transmission: 1.0,
            ..default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(-3.8, 1.5, 0.0)).with_scale(Vec3::splat(3.0)),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(1.0, 0.2, 0.2),
            perceptual_roughness: 0.001,
            metallic: 0.5,
            ..default()
        })),
    ));

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let cuboid = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    for x in -11..11 {
        for y in -11..11 {
            let dx = rng.gen_range(-0.5..0.5);
            let dy = rng.gen_range(-0.5..0.5);

            let scale = 0.5 + rng.gen_range(0.0..0.9);

            let xf = 2.0 * x as f32 + dx;
            let yf = 2.0 * y as f32 + dy;

            if xf * xf + yf * yf < 4.0 * 4.0 {
                continue;
            }

            let choose_mat: f32 = rng.r#gen();
            let mut material = AuroraMaterial::default();

            if choose_mat < 0.7 {
                // lambertian
                material.base_color = Color::linear_rgb(rng.r#gen(), rng.r#gen(), rng.r#gen());
            } else if choose_mat < 0.85 {
                // mirror
                material.base_color = Color::WHITE;
                material.perceptual_roughness = 0.01;
                material.metallic = 1.0;
            } else if choose_mat < 0.95 {
                // glass
                material.base_color = Color::WHITE;
                material.perceptual_roughness = 0.0;
                material.ior = 1.01 + 0.15 * rng.r#gen::<f32>();
                material.specular_transmission = 1.0;
            } else {
                // light source (physical nits -- see EMISSIVE_NITS)
                material.emissive =
                    EMISSIVE_NITS * LinearRgba::rgb(rng.r#gen(), rng.r#gen(), rng.r#gen());
            }

            let mut entity_builder = commands.spawn((
                Transform::from_translation(Vec3::new(xf, scale / 2.0, yf))
                    .with_scale(Vec3::splat(scale))
                    .with_rotation(Quat::from_rotation_y(rng.r#gen::<f32>() * 2.0 * PI)),
                AuroraMaterial3d(materials.add(material)),
            ));

            let choose_shape: f32 = rng.r#gen();
            if choose_shape < 0.9 {
                entity_builder.insert(Sphere);
            } else {
                entity_builder.insert(Mesh3d(cuboid.clone()));
            }
        }
    }
}

/// Exercises the gizmo overlay: world axes, a spinning cuboid, a sphere shell around the big
/// sphere.
fn debug_gizmos(mut gizmos: Gizmos, time: Res<Time>) {
    gizmos.line(Vec3::ZERO, Vec3::X * 3.0, Color::srgb(1.0, 0.2, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Y * 3.0, Color::srgb(0.2, 1.0, 0.2));
    gizmos.line(Vec3::ZERO, Vec3::Z * 3.0, Color::srgb(0.3, 0.5, 1.0));
    gizmos.cube(
        Transform::from_xyz(3.0, 0.5, 2.0)
            .with_rotation(Quat::from_rotation_y(time.elapsed_secs())),
        Color::srgb(1.0, 1.0, 0.2),
    );
    gizmos.sphere(
        Isometry3d::from_translation(Vec3::new(0.0, 1.5, 0.0)),
        1.6,
        Color::WHITE,
    );
}
