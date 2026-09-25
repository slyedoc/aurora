//! A surface group end to end: register a class, publish its parameters, tag an entity.
//!
//! The left sphere is the built-in opaque class. The right one is the same mesh and the
//! same `AuroraMaterial`, but carries `SurfaceClass` for a registered LAYERED group, so its
//! SBT record routes to `layered.rchit` instead -- which blends two PBR sets by the
//! surface's slope. Identical inputs, different shader: that is the whole mechanism.
//!
//!   AUTO_SCREENSHOT_MS=9000 cargo run --release --example surface_groups

use ash::vk;
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::prelude::*;
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    dev_shaders::DevShaderPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sky::Sky,
    surface_group::{SurfaceClass, SurfaceGroup, SurfaceGroupData, SurfaceGroupRegistry},
    util::{ScreenshotExt, TimeoutAppExt},
};

/// Must match `struct Layered` in `layered.rchit`, field for field: the shader reads this
/// straight out of the buffer by device address.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Layered {
    layer_color: [f32; 4],
    detail_color: [f32; 4],
    layer_roughness: f32,
    detail_roughness: f32,
    layer_metallic: f32,
    detail_metallic: f32,
    blend_start: f32,
    blend_end: f32,
    blend_contrast: f32,
    pad: f32,
}

/// Holds the buffer alive: dropping it would leave `SurfaceGroupData` pointing at freed
/// memory, which the shader would happily read.
#[derive(Resource)]
struct LayeredBuffer(#[allow(dead_code)] Buffer<Layered>);

#[derive(Resource)]
struct LayeredClass(SurfaceClass);

fn main() {
    App::new()
        .add_plugins((AuroraDefaultPlugins, DevShaderPlugin, FreeCameraPlugin))
        .add_systems(Startup, (register_layered, setup).chain())
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 14.0)
        .run();
}

/// Registration has to happen before the pipeline is built -- or rather, it may happen
/// after, and the registry's generation bump rebuilds it. Either way the class is stable.
fn register_layered(
    mut commands: Commands,
    mut registry: ResMut<SurfaceGroupRegistry>,
    mut data: ResMut<SurfaceGroupData>,
    render_device: Res<RenderDevice>,
    asset_server: Res<AssetServer>,
) {
    let class = registry.register(SurfaceGroup {
        label: "layered".to_string(),
        closest_hit: asset_server.load(aurora_asset("shaders/layered.rchit")),
        any_hit: None,
    });

    // One entry per material slot, indexed the way the shader indexes it. Two here is
    // plenty: the sphere's material sits at slot 0 and everything else reads the default.
    let mut buffer: Buffer<Layered> = render_device.create_host_buffer(
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    );
    {
        let mut mapped = render_device.map_buffer(&mut buffer);
        let entry = Layered {
            // Snow on the flats, wet slate on the slopes.
            layer_color: [1.05, 1.08, 1.15, 1.0],
            detail_color: [0.20, 0.19, 0.22, 1.0],
            layer_roughness: 1.4,
            detail_roughness: 0.25,
            layer_metallic: 0.0,
            detail_metallic: 0.0,
            blend_start: 0.25,
            blend_end: 0.75,
            blend_contrast: 1.6,
            pad: 0.0,
        };
        mapped.copy_from_slice(&[entry; 4]);
    }
    data.set(class, buffer.address);
    commands.insert_resource(LayeredBuffer(buffer));
    commands.insert_resource(LayeredClass(class));
    info!("registered layered surface group as {class:?}");
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    layered: Res<LayeredClass>,
) {
    commands.insert_resource(Sky::Procedural);

    commands.spawn((
        Name::new("camera"),
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 2.4, 6.5).looking_at(Vec3::new(0.0, 0.9, 0.0), Vec3::Y),
    ));
    commands.spawn((
        Name::new("key"),
        DirectionalLight {
            illuminance: 14_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let ground = materials.add(AuroraMaterial {
        base_color: Color::linear_rgb(0.22, 0.23, 0.25),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Name::new("ground"),
        Mesh3d(meshes.add(Cuboid::new(16.0, 0.2, 10.0))),
        AuroraMaterial3d(ground),
        Transform::from_xyz(0.0, -0.1, 0.0),
    ));

    // ONE material, shared: the only difference between these two is the class.
    let stone = materials.add(AuroraMaterial {
        base_color: Color::linear_rgb(0.55, 0.55, 0.58),
        perceptual_roughness: 0.6,
        ..default()
    });

    // MESH spheres, not aurora's `Sphere` component: that one is a PROCEDURAL hit group
    // with its own record at offset 0, so it always routes to `sphere_hit.rchit` and never
    // sees a surface class. Class routing is a property of triangle records.
    let ball = meshes.add(bevy::math::primitives::Sphere::new(1.0).mesh().uv(48, 32));
    commands.spawn((
        Name::new("opaque class"),
        Mesh3d(ball.clone()),
        AuroraMaterial3d(stone.clone()),
        Transform::from_xyz(-1.6, 1.0, 0.0),
    ));
    commands.spawn((
        Name::new("layered class"),
        Mesh3d(ball),
        AuroraMaterial3d(stone),
        layered.0,
        Transform::from_xyz(1.6, 1.0, 0.0),
    ));
}
