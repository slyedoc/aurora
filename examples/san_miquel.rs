use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    gltf_mesh::{GltfModel, GltfModelHandle},
    ray_default_plugins::RayDefaultPlugins,
};

fn main() {
    App::new()
        .add_plugins((
            RayDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, print_cam_pos)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, mut windows: Query<&mut Window>) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 70.0 * 3.1415926 / 180.0,
            ..default()
        }),
        Transform::from_xyz(4.98, 5.83, 1.3)
            .with_rotation(Quat::from_xyzw(-0.0941, -0.701, -0.094, 0.700).normalize()),
        FreeCamera::default(),
    ));

    commands.spawn((
        GltfModelHandle(asset_server.load::<GltfModel>("models/san_miquel.glb")),
        Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
            .with_scale(Vec3::splat(0.8)),
    ));
}

fn print_cam_pos(q: Query<&Transform, With<Camera>>, keyboard: Res<ButtonInput<KeyCode>>) {
    if keyboard.just_pressed(KeyCode::Space) {
        for t in q.iter() {
            dbg!(t);
        }
    }
}
