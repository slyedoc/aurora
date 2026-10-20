//! Particles in the tracer: embers rising off a lava pool in a closed
//! stone room, GPU-simulated and traced as emissive spheres. A compute kernel
//! (assets/shaders/particles_demo.slang) moves them and writes their rows in an `InstanceBlock`
//! each frame, so they are geometry like anything else: occluded, in the floor's and the mirror
//! ball's reflections, lighting the walls. The panel on the left edits them live.
//!
//!   cargo run --release --example particles

use std::any::TypeId;

use ash::vk;
use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{theme::ThemeBackgroundColor, tokens},
    feathers_inspector::{BuildComponentInspector, Step},
    prelude::*,
    ui_widgets::ScrollArea,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    assets::aurora_asset,
    auto_exposure::{AuroraExposure, ev_from_ev100},
    compute::{BlockRows, ComputeModule, FrameCommands, record_dispatch},
    dev_ui::DevUIPlugin,
    environment::Environment,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sky::Sky,
    sphere::Sphere,
    tlas_builder::{InstanceBlock, InstanceBlockSlots, TLAS},
    util::ScreenshotExt,
};
use bytemuck::{Pod, Zeroable};

/// The most embers the block holds (its TLAS slots); `Embers::count` runs up to it.
const CAPACITY: u32 = 16384;
/// The room's half width and height (metres).
const ROOM: f32 = 8.0;
const HEIGHT: f32 = 7.0;

/// The ember system, edited live from the panel.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component)]
struct Embers {
    /// Live embers (at most the block's capacity).
    #[reflect(@Step(64.0))]
    count: u32,
    /// The disc they rise from (metres).
    #[reflect(@Step(0.05))]
    radius: f32,
    /// A new ember's diameter (metres); it shrinks as it cools.
    #[reflect(@Step(0.002))]
    size: f32,
    /// Upward speed at birth (m/s).
    #[reflect(@Step(0.05))]
    rise: f32,
    /// The swirl's strength.
    #[reflect(@Step(0.05))]
    swirl: f32,
    /// Mean life (seconds).
    #[reflect(@Step(0.1))]
    life: f32,
    /// Their glow (nits).
    #[reflect(@Step(500.0))]
    nits: f32,
}

impl Default for Embers {
    fn default() -> Self {
        Self {
            count: 4096,
            radius: 1.6,
            size: 0.03,
            rise: 1.2,
            swirl: 0.6,
            life: 3.0,
            nits: 1500.0,
        }
    }
}

/// One ember on the GPU (`Ember` in particles_demo.slang).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Ember {
    pos: [f32; 3],
    age: f32,
    vel: [f32; 3],
    life: f32,
}

/// `EmberPush` in particles_demo.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmberPush {
    embers: u64,
    rows: BlockRows,
    count: u32,
    frame: u32,
    dt: f32,
    time: f32,
    origin: [f32; 3],
    radius: f32,
    size: f32,
    rise: f32,
    swirl: f32,
    life: f32,
}

#[derive(Resource)]
struct EmberKernel(Handle<ComputeModule>);

/// The embers' state, zeroed at creation (every ember then starts fresh).
#[derive(Resource, Default)]
struct EmberBuffer(Option<Buffer<Ember>>);

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins.set(WindowPlugin {
                close_when_requested: false,
                ..default()
            }),
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_screenshot(KeyCode::F12)
        .register_type::<Embers>()
        .init_resource::<EmberBuffer>()
        .add_systems(Startup, setup)
        .add_systems(PostUpdate, (ember_glow, step_embers))
        .add_systems(TeardownSchedule, free_embers.before(on_shutdown))
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    asset_server: Res<AssetServer>,
) {
    let shader = asset_server.load(aurora_asset("shaders/particles_demo.slang"));
    commands.insert_resource(EmberKernel(
        asset_server.add(ComputeModule::new(shader, &["embers_step"])),
    ));

    let environment = commands
        .spawn((
            Name::new("Environment"),
            Environment::default(),
            Sky::Generated,
        ))
        .id();
    commands.spawn((
        ChildOf(environment),
        Camera3d::default(),
        FreeCamera::default(),
        // A dim interior: the lava and the embers are the only light.
        AuroraExposure::fixed(ev_from_ev100(6.0)),
        Transform::from_xyz(0.0, 2.2, -6.5).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
    ));

    let cube = meshes.add(AuroraMesh::from_shape(Cuboid::default()));
    let block =
        |commands: &mut Commands, material: &Handle<AuroraMaterial>, min: Vec3, max: Vec3| {
            commands.spawn((
                ChildOf(environment),
                AuroraMesh3d(cube.clone()),
                AuroraMaterial3d(material.clone()),
                Transform::from_translation((min + max) * 0.5).with_scale(max - min),
            ));
        };
    let stone = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.35, 0.33, 0.31),
        perceptual_roughness: 0.9,
        ..default()
    });
    // Polished dark stone: the embers' reflections.
    let floor = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.09),
        perceptual_roughness: 0.15,
        ..default()
    });
    let lava = materials.add(AuroraMaterial {
        base_color: Color::srgb(0.3, 0.05, 0.02),
        emissive: LinearRgba::rgb(800.0, 200.0, 30.0),
        perceptual_roughness: 0.8,
        ..default()
    });

    // The closed room: floor, ceiling, four walls.
    block(
        &mut commands,
        &floor,
        Vec3::new(-ROOM, -0.5, -ROOM),
        Vec3::new(ROOM, 0.0, ROOM),
    );
    block(
        &mut commands,
        &stone,
        Vec3::new(-ROOM, HEIGHT, -ROOM),
        Vec3::new(ROOM, HEIGHT + 0.5, ROOM),
    );
    for (min, max) in [
        (
            Vec3::new(-ROOM - 0.5, 0.0, -ROOM),
            Vec3::new(-ROOM, HEIGHT, ROOM),
        ),
        (
            Vec3::new(ROOM, 0.0, -ROOM),
            Vec3::new(ROOM + 0.5, HEIGHT, ROOM),
        ),
        (
            Vec3::new(-ROOM, 0.0, -ROOM - 0.5),
            Vec3::new(ROOM, HEIGHT, -ROOM),
        ),
        (
            Vec3::new(-ROOM, 0.0, ROOM),
            Vec3::new(ROOM, HEIGHT, ROOM + 0.5),
        ),
    ] {
        block(&mut commands, &stone, min, max);
    }
    // The pool: a stone rim around a lava disc's square.
    block(
        &mut commands,
        &stone,
        Vec3::new(-2.3, 0.0, -2.3),
        Vec3::new(2.3, 0.25, 2.3),
    );
    block(
        &mut commands,
        &lava,
        Vec3::new(-1.9, 0.0, -1.9),
        Vec3::new(1.9, 0.27, 1.9),
    );

    // A mirror ball beside the pool.
    commands.spawn((
        ChildOf(environment),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.95, 0.95, 0.95),
            metallic: 1.0,
            perceptual_roughness: 0.02,
            ..default()
        })),
        Transform::from_xyz(3.6, 1.0, 1.0).with_scale(Vec3::splat(2.0)),
    ));

    // The embers: a block of emissive spheres the kernel writes.
    let embers = commands
        .spawn((
            ChildOf(environment),
            Name::new("Embers"),
            Embers::default(),
            Sphere,
            InstanceBlock {
                capacity: CAPACITY,
                poses: Vec::new(),
            },
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::srgb(1.0, 0.4, 0.1),
                emissive: LinearRgba::rgb(1500.0, 450.0, 75.0),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.3, 0.0),
        ))
        .id();

    let panel = commands
        .spawn((
            Name::new("Embers Panel"),
            Node {
                position_type: PositionType::Absolute,
                left: px(16),
                top: px(16),
                width: px(340),
                max_height: vh(90),
                padding: UiRect::all(px(8)),
                flex_direction: FlexDirection::Column,
                overflow: Overflow::scroll_y(),
                ..default()
            },
            ScrollArea,
            ThemeBackgroundColor(tokens::WINDOW_BG),
        ))
        .id();
    commands.queue(BuildComponentInspector {
        target: embers,
        type_id: TypeId::of::<Embers>(),
        panel,
    });
}

/// The panel's glow onto the embers' material.
fn ember_glow(
    embers: Query<(&Embers, &AuroraMaterial3d), Changed<Embers>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    for (embers, material) in &embers {
        if let Some(mut m) = materials.get_mut(&material.0) {
            m.emissive = LinearRgba::rgb(1.0, 0.3, 0.05) * embers.nits;
        }
    }
}

/// This frame's step, recorded ahead of the TLAS build that reads the rows.
#[allow(clippy::too_many_arguments)]
fn step_embers(
    embers: Query<(&Embers, &InstanceBlockSlots, &GlobalTransform)>,
    kernel: Res<EmberKernel>,
    rd: Res<RenderDevice>,
    time: Res<Time>,
    mut buffer: ResMut<EmberBuffer>,
    mut tlas: ResMut<TLAS>,
    mut frame: ResMut<FrameCommands>,
    mut frame_index: Local<u32>,
) {
    let Ok((embers, slots, at)) = embers.single() else {
        return;
    };
    let buffer = buffer.0.get_or_insert_with(|| {
        let mut b = rd.create_host_buffer(CAPACITY as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
        let mut mapped = rd.map_buffer(&mut b);
        for i in 0..CAPACITY as usize {
            mapped[i] = Ember::zeroed();
        }
        b
    });
    *frame_index = frame_index.wrapping_add(1);
    let push = EmberPush {
        embers: buffer.address,
        rows: BlockRows::default(),
        count: embers.count.min(slots.count),
        frame: *frame_index,
        // A long frame (a hitch, a capture's held world) steps as a short one.
        dt: time.delta_secs().min(1.0 / 20.0),
        time: time.elapsed_secs_wrapped(),
        origin: at.translation().to_array(),
        radius: embers.radius,
        size: embers.size,
        rise: embers.rise,
        swirl: embers.swirl,
        life: embers.life.max(0.1),
    };
    let slots = *slots;
    let module = kernel.0.clone();
    frame.push(move |rd, modules, cmd, ctx| {
        // The block's rows as the instance arrays stand now (none while it resolves).
        let (Some(module), Some(rows)) = (modules.get(&module), ctx.block_rows(slots)) else {
            return;
        };
        let push = EmberPush { rows, ..push };
        record_dispatch(rd, cmd, module, "embers_step", &push, rows.count, None);
    });
    // Every ember moved: a fresh build, not a refit of last frame's boxes.
    tlas.request_rebuild();
}

fn free_embers(world: &mut World) {
    if let Some(buffer) = world.resource_mut::<EmberBuffer>().0.take() {
        world
            .resource::<RenderDevice>()
            .destroyer
            .destroy_buffer(buffer.handle);
    }
}
