//! Visibility → TLAS instance mask, and GPU transform propagation: the yellow cube spins (its
//! green child rides along through the hierarchy, propagated on the GPU) and, with the red
//! sphere, flips `Visibility` every three seconds. Hidden entities stay in the TLAS (same
//! instance list, same topology) with a zero instance mask, so rays simply miss them.

use bevy::prelude::*;
use bevy_aurora::{
    debug_camera::{DebugCamera, DebugCameraPlugin},
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    ray_default_plugins::RayDefaultPlugins,
    ray_render_plugin::RenderConfig,
    sphere::Sphere,
};

#[derive(Component)]
struct Blinker;

fn main() {
    let mut app = App::new();
    app.add_plugins(RayDefaultPlugins);
    app.add_plugins(DevShaderPlugin);
    app.add_plugins(DevUIPlugin);
    app.add_plugins(DebugCameraPlugin);
    app.add_systems(Startup, setup);
    app.add_systems(Update, (blink, spin));
    app.run();
}

fn setup(
    mut commands: Commands,
    mut windows: Query<&mut Window>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_config: ResMut<RenderConfig>,
) {
    let mut window = windows.single_mut().unwrap();
    window.resolution.set_scale_factor_override(Some(1.0));
    window.resolution.set(1280.0, 720.0);
    render_config.skydome = None;
    render_config.sky_color = Vec4::new(0.75, 0.85, 1.0, 0.0);

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        DebugCamera::default(),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.3),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));

    // Always visible reference.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.4, 0.9),
            ..default()
        })),
        Transform::from_xyz(-2.5, 0.5, 0.0),
    ));

    // Blinkers: a mesh and a sphere, with a child cube that inherits the hide.
    commands
        .spawn((
            Blinker,
            Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.9, 0.8, 0.2),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.5, 0.0),
        ))
        .with_child((
            Mesh3d(meshes.add(Cuboid::new(0.5, 0.5, 0.5))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.9, 0.4),
                ..default()
            })),
            Transform::from_xyz(0.0, 1.0, 0.0),
        ));
    commands.spawn((
        Blinker,
        Sphere,
        Visibility::default(),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.1, 0.1),
            emissive: LinearRgba::new(4.0, 0.5, 0.5, 1.0),
            ..default()
        })),
        Transform::from_xyz(2.5, 1.0, 0.0),
    ));
}

fn spin(time: Res<Time>, mut spinners: Query<&mut Transform, (With<Blinker>, With<Mesh3d>)>) {
    for mut transform in &mut spinners {
        transform.rotation = Quat::from_rotation_y(time.elapsed_secs());
        transform.translation.x = (time.elapsed_secs() * 0.7).sin();
    }
}

fn blink(time: Res<Time>, mut blinkers: Query<&mut Visibility, With<Blinker>>) {
    let hidden = (time.elapsed_secs() / 3.0) as u32 % 2 == 1;
    let want = if hidden {
        Visibility::Hidden
    } else {
        Visibility::Visible
    };
    for mut visibility in &mut blinkers {
        if *visibility != want {
            info!("blinkers -> {want:?} at {:.2}s", time.elapsed_secs());
            *visibility = want;
        }
    }
}
