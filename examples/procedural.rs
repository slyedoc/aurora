//! `ProceduralMesh` smoke test: a wavy heightfield whose vertices are written by a compute
//! kernel (assets/shaders/procedural_demo.slang) and traced like any mesh, beside a plain
//! `Mesh3d` cube for reference. `AUTO_SCREENSHOT_MS=7000` captures it; exits at 12 s.

use std::sync::Arc;

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules},
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    procedural_mesh::{ProceduralKernels, ProceduralMesh, ProceduralMesh3d},
    AuroraDefaultPlugins,
    sky::Sky,
    util::{ScreenshotExt, TimeoutAppExt},
};
use bytemuck::{Pod, Zeroable};

const RES: u32 = 129;
const SIZE: f32 = 40.0;

/// The kernel's own parameters (after `ProceduralHeader`; scalar layout).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DemoParams {
    res: u32,
    size: f32,
    amplitude: f32,
    frequency: f32,
}

#[derive(Resource)]
struct DemoKernel(Handle<ComputeModule>);

#[derive(Resource, Default)]
struct Spawned(bool);

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .init_resource::<Spawned>()
        .add_systems(Startup, setup)
        .add_systems(Update, spawn_when_ready)
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    commands.insert_resource(Sky::Procedural);
    let shader = asset_server.load(aurora_asset("shaders/procedural_demo.slang"));
    let module = asset_server.add(ComputeModule::new(shader, &["demo_fill"]));
    commands.insert_resource(DemoKernel(module));

    // Reference geometry through the ordinary mesh path.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(3.0, 3.0, 3.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.85, 0.3, 0.2),
            perceptual_roughness: 0.5,
            ..default()
        })),
        Transform::from_xyz(0.0, 4.0, 0.0),
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.0, 16.0, 34.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

/// The fill kernel has to be compiled before the asset is created (see procedural_mesh.rs).
fn spawn_when_ready(
    mut commands: Commands,
    mut spawned: ResMut<Spawned>,
    kernel: Res<DemoKernel>,
    kernels: Res<ProceduralKernels>,
    modules: Res<ComputeModules>,
    mut procedural: ResMut<Assets<ProceduralMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    if spawned.0 || !kernels.ready(&modules, &kernel.0) {
        return;
    }
    spawned.0 = true;
    let indices: Vec<u32> = (0..RES - 1)
        .flat_map(|y| {
            (0..RES - 1).flat_map(move |x| {
                let a = y * RES + x;
                let b = a + RES;
                [a, b, b + 1, a, b + 1, a + 1]
            })
        })
        .collect();
    let params = DemoParams {
        res: RES,
        size: SIZE,
        amplitude: 1.5,
        frequency: 0.6,
    };
    let mesh = procedural.add(ProceduralMesh {
        vertex_count: RES * RES,
        indices: Arc::from(indices),
        module: kernel.0.clone(),
        entry: "demo_fill".into(),
        params: bytemuck::bytes_of(&params).to_vec(),
    });
    commands.spawn((
        Name::new("procedural waves"),
        ProceduralMesh3d(mesh),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.35, 0.6, 0.3),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::IDENTITY,
    ));
    info!("procedural: spawned the wave grid");
}
