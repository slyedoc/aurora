//! GPU particles: Hanabi effects (slyedoc/bevy_hanabi, branch `aurora`) on the tracer. Hanabi
//! compiles an effect's graph into Slang kernels; here each effect instance gets them built by
//! slangc, its GPU storage and its TLAS instance block, simulated each frame through the frame
//! commands. Particles are geometry like anything else: shadowed, reflected, in every view.
//! Their material is the entity's [`AuroraMaterial3d`] (`seen_only` when a light stands in for
//! the effect's glow, `instance_tint` for each particle's colour), their shape its
//! [`AuroraMesh3d`] or the asset's `mesh`, else an analytic sphere.
//!
//! An instance owns its storage, so a particle's slot is its index in every buffer and its row
//! in the block. Each frame one command uploads the emitter, the frame's parameters and the
//! properties, runs the init kernel over the particles to spawn (each pops a free slot) and the
//! update kernel over every slot (a live particle ages, updates and writes its instance row; a
//! dead one pushes its slot back and traces nothing).

use std::{
    borrow::Cow,
    hash::{Hash as _, Hasher as _},
    path::PathBuf,
};

use ash::vk;
use bevy::{
    camera::primitives::Frustum,
    ecs::lifecycle::Remove,
    platform::collections::HashMap,
    prelude::*,
    tasks::{block_on, futures_lite::future, AsyncComputeTaskPool, Task},
};
use bevy_hanabi::{
    CompiledParticleEffect, EffectAsset, EffectProperties, EffectSimulation, EffectSpawner,
    EffectSystems, HanabiMainCamera, HanabiPlugin, SimulationCondition,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    compute::{
        BlockRows, ComputeModule, FrameCommands, compute_to_compute_barrier, memory_barrier,
        record_dispatch,
    },
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::AuroraMesh3d,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    shader::{Shader, compile_slang},
    sphere::Sphere,
    tlas_builder::{InstanceBlock, InstanceBlockSlots, TLAS},
};

/// `SimParams` in effect_common.slang. 192 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSimParams {
    frustum: [[f32; 4]; 6],
    camera_position: [f32; 4],
    camera_axes: [[f32; 4]; 3],
    delta_time: f32,
    time: f32,
    virtual_delta_time: f32,
    virtual_time: f32,
    real_delta_time: f32,
    real_time: f32,
    num_effects: u32,
    pad: u32,
}

/// `Spawner` in effect_common.slang. 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSpawner {
    transform: [[f32; 4]; 3],
    inverse_transform: [[f32; 4]; 3],
    spawn: i32,
    seed: u32,
    capacity: u32,
    frame: u32,
}

/// `EffectState` in effect_common.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEffectState {
    free_count: i32,
    alive_count: u32,
    particle_counter: u32,
    pad: u32,
}

/// `EffectPush` in effect_common.slang. 88 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectPush {
    particles: u64,
    free_slots: u64,
    alive: u64,
    last: u64,
    state: u64,
    spawner: u64,
    sim: u64,
    properties: u64,
    rows: BlockRows,
}

/// An effect's built kernels.
#[derive(Component)]
struct EffectKernels {
    source: u64,
    init: Handle<ComputeModule>,
    update: Handle<ComputeModule>,
}

/// An effect instance's GPU storage, sized for its asset's capacity and particle layout.
#[derive(Component)]
struct EffectStorage {
    capacity: u32,
    stride: u32,
    properties_size: u32,
    particles: Buffer<u8>,
    free_slots: Buffer<u32>,
    alive: Buffer<u32>,
    last: Buffer<[f32; 12]>,
    state: Buffer<GpuEffectState>,
    spawner: Buffer<GpuSpawner>,
    sim: Buffer<GpuSimParams>,
    properties: Buffer<u8>,
}

impl EffectStorage {
    fn new(rd: &RenderDevice, capacity: u32, stride: u32, properties_size: u32) -> Self {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let upload = storage | vk::BufferUsageFlags::TRANSFER_DST;
        // Every slot free, nothing alive.
        let mut free_slots = rd.create_host_buffer::<u32>(capacity as u64, storage);
        {
            let mut mapped = rd.map_buffer(&mut free_slots);
            for slot in 0..capacity as usize {
                mapped[slot] = slot as u32;
            }
        }
        let mut alive = rd.create_host_buffer::<u32>(capacity as u64, storage);
        {
            let mut mapped = rd.map_buffer(&mut alive);
            for slot in 0..capacity as usize {
                mapped[slot] = 0;
            }
        }
        let mut state = rd.create_host_buffer::<GpuEffectState>(1, storage);
        rd.map_buffer(&mut state)[0] = GpuEffectState {
            free_count: capacity as i32,
            ..Zeroable::zeroed()
        };
        Self {
            capacity,
            stride,
            properties_size,
            particles: rd.create_device_buffer(capacity as u64 * stride as u64, storage),
            free_slots,
            alive,
            last: rd.create_device_buffer(capacity as u64, storage),
            state,
            spawner: rd.create_device_buffer(1, upload),
            sim: rd.create_device_buffer(1, upload),
            properties: rd.create_device_buffer(properties_size.max(4) as u64, upload),
        }
    }

    fn destroy(&self, rd: &RenderDevice) {
        for handle in [
            self.particles.handle,
            self.free_slots.handle,
            self.alive.handle,
            self.last.handle,
            self.state.handle,
            self.spawner.handle,
            self.sim.handle,
            self.properties.handle,
        ] {
            rd.destroyer.destroy_buffer(handle);
        }
    }
}

/// A kernel module's build: compiling, or done (`None`: slangc refused it).
enum KernelBuild {
    Compiling(Task<Result<Vec<u8>, String>>),
    Done(Option<Handle<ComputeModule>>),
}

/// Every kernel module built or building, by its source's hash: instances of one asset share.
#[derive(Resource, Default)]
struct KernelCache(HashMap<u64, KernelBuild>);

/// Hanabi, and its effects simulated and traced.
pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<HanabiPlugin>() {
            app.add_plugins(HanabiPlugin);
        }
        app.init_resource::<KernelCache>()
        .add_systems(
            PostUpdate,
            (build_kernels, allocate_storage, simulate)
                .chain()
                .in_set(EffectSystems::Simulate),
        )
            .add_observer(free_storage)
            .add_systems(TeardownSchedule, free_all_storage.before(on_shutdown));
    }
}

fn source_hash(source: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    source.hash(&mut hasher);
    hasher.finish()
}

/// slangc wants a file (and resolves aurora's `import instance_rows;` from its own include dir).
fn compile_source(hash: u64, source: String) -> Result<Vec<u8>, String> {
    let dir: PathBuf = std::env::temp_dir().join("hanabi-slang");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{hash:016x}.slang"));
    std::fs::write(&path, source).map_err(|e| e.to_string())?;
    compile_slang(&path.to_string_lossy()).map_err(|e| e.to_string())
}

/// The module for `source`'s `entry`, once built: starts its compile on first sight.
fn kernel(
    cache: &mut KernelCache,
    source: &str,
    entry: &str,
    shaders: &mut Assets<Shader>,
    modules: &mut Assets<ComputeModule>,
) -> Option<Option<Handle<ComputeModule>>> {
    let hash = source_hash(source);
    let build = cache.0.entry(hash).or_insert_with(|| {
        let source = source.to_string();
        KernelBuild::Compiling(
            AsyncComputeTaskPool::get().spawn(async move { compile_source(hash, source) }),
        )
    });
    if let KernelBuild::Compiling(task) = build {
        let done = block_on(future::poll_once(task))?;
        *build = KernelBuild::Done(match done {
            Ok(spirv) => {
                let shader = shaders.add(Shader {
                    path: format!("hanabi/{hash:016x}.slang"),
                    spirv: Some(Cow::Owned(spirv)),
                });
                Some(modules.add(ComputeModule::new(shader, &[entry])))
            }
            Err(err) => {
                error!("hanabi: the {entry} kernel failed to compile: {err}");
                None
            }
        });
    }
    match build {
        KernelBuild::Done(module) => Some(module.clone()),
        KernelBuild::Compiling(_) => None,
    }
}

fn build_kernels(
    effects: Query<(Entity, &CompiledParticleEffect, Option<&EffectKernels>)>,
    mut cache: ResMut<KernelCache>,
    mut shaders: ResMut<Assets<Shader>>,
    mut modules: ResMut<Assets<ComputeModule>>,
    mut commands: Commands,
) {
    for (entity, compiled, kernels) in &effects {
        let Some(sources) = compiled.sources() else {
            continue;
        };
        let source = source_hash(&sources.init_shader_source)
            ^ source_hash(&sources.update_shader_source).rotate_left(1);
        if kernels.is_some_and(|k| k.source == source) {
            continue;
        }
        let init = kernel(
            &mut cache,
            &sources.init_shader_source,
            "init",
            &mut shaders,
            &mut modules,
        );
        let update = kernel(
            &mut cache,
            &sources.update_shader_source,
            "update",
            &mut shaders,
            &mut modules,
        );
        if let (Some(Some(init)), Some(Some(update))) = (init, update) {
            commands.entity(entity).insert(EffectKernels {
                source,
                init,
                update,
            });
        }
    }
}

#[allow(clippy::type_complexity)]
fn allocate_storage(
    mut effects: Query<(
        Entity,
        &mut CompiledParticleEffect,
        Option<&EffectStorage>,
        Option<&EffectKernels>,
        Has<Sphere>,
        Has<AuroraMesh3d>,
        Has<AuroraMaterial3d>,
    )>,
    assets: Res<Assets<EffectAsset>>,
    asset_server: Res<AssetServer>,
    rd: Res<RenderDevice>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut default_material: Local<Option<Handle<AuroraMaterial>>>,
    mut commands: Commands,
) {
    for (entity, mut compiled, storage, kernels, sphere, mesh, material) in &mut effects {
        let Some(asset) = assets.get(compiled.asset()) else {
            continue;
        };
        let capacity = asset.capacity();
        let stride = asset.particle_layout().min_binding_size32().get();
        let property_layout = asset.property_layout();
        let properties_size = if property_layout.is_empty() {
            0
        } else {
            property_layout.min_binding_size().get() as u32
        };
        let fits = storage.is_some_and(|s| {
            s.capacity == capacity && s.stride == stride && s.properties_size == properties_size
        });
        if !fits {
            if let Some(old) = storage {
                old.destroy(&rd);
            }
            let mut entity = commands.entity(entity);
            entity.insert((
                EffectStorage::new(&rd, capacity, stride, properties_size),
                InstanceBlock {
                    capacity,
                    poses: Vec::new(),
                },
            ));
            match &asset.mesh {
                Some(path) if !mesh => {
                    entity
                        .remove::<Sphere>()
                        .insert(AuroraMesh3d(asset_server.load(path.clone())));
                }
                None if !sphere && !mesh => {
                    entity.insert(Sphere);
                }
                _ => {}
            }
            if !material {
                let handle = default_material
                    .get_or_insert_with(|| {
                        materials.add(AuroraMaterial {
                            instance_tint: true,
                            ..default()
                        })
                    })
                    .clone();
                entity.insert(AuroraMaterial3d(handle));
            }
        }
        let ready = fits && kernels.is_some();
        if compiled.is_ready() != ready {
            compiled.set_ready(ready);
        }
    }
}

/// Rows of an affine transform (3x4), for the kernels.
fn rows(m: &Mat4) -> [[f32; 4]; 3] {
    let t = m.transpose();
    [t.x_axis.to_array(), t.y_axis.to_array(), t.z_axis.to_array()]
}

/// The camera the effects face and cull against: the one marked [`HanabiMainCamera`], else
/// the first active one.
fn main_camera<'a>(
    cameras: &'a Query<(&Camera, &GlobalTransform, Option<&Frustum>, Has<HanabiMainCamera>)>,
) -> Option<(&'a GlobalTransform, Option<&'a Frustum>)> {
    cameras
        .iter()
        .find(|c| c.3)
        .or_else(|| cameras.iter().find(|c| c.0.is_active))
        .map(|(_, at, frustum, _)| (at, frustum))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity, unsafe_code)]
fn simulate(
    effects: Query<(
        &CompiledParticleEffect,
        &EffectStorage,
        &EffectKernels,
        &InstanceBlockSlots,
        &GlobalTransform,
        &InheritedVisibility,
        Option<&EffectSpawner>,
        Option<&EffectProperties>,
    )>,
    cameras: Query<(&Camera, &GlobalTransform, Option<&Frustum>, Has<HanabiMainCamera>)>,
    assets: Res<Assets<EffectAsset>>,
    time: Res<Time<EffectSimulation>>,
    real: Res<Time<Real>>,
    virt: Res<Time<Virtual>>,
    mut frame: ResMut<FrameCommands>,
    mut tlas: ResMut<TLAS>,
    mut frame_index: Local<u32>,
) {
    *frame_index = frame_index.wrapping_add(1);
    let (camera_at, frustum) = main_camera(&cameras)
        .map(|(at, frustum)| (at.to_matrix(), frustum.cloned()))
        .unwrap_or((Mat4::IDENTITY, None));
    let mut planes = [[0.0; 4]; 6];
    if let Some(frustum) = frustum {
        for (plane, half_space) in planes.iter_mut().zip(frustum.half_spaces.iter()) {
            *plane = half_space.normal_d().to_array();
        }
    }
    let num_effects = effects.iter().len() as u32;
    let mut any = false;
    for (compiled, storage, kernels, slots, at, visible, spawner, properties) in &effects {
        if !compiled.is_ready() {
            continue;
        }
        let Some(asset) = assets.get(compiled.asset()) else {
            continue;
        };
        // An effect simulated only when visible holds still (and keeps its rows) out of view.
        let running = compiled.simulation_condition() == SimulationCondition::Always
            || visible.get();
        let dt = |d: f32| if running { d } else { 0.0 };
        let sim = GpuSimParams {
            frustum: planes,
            camera_position: camera_at.w_axis.to_array(),
            camera_axes: [
                camera_at.x_axis.to_array(),
                camera_at.y_axis.to_array(),
                camera_at.z_axis.to_array(),
            ],
            delta_time: dt(time.delta_secs()),
            time: time.elapsed_secs_wrapped(),
            virtual_delta_time: dt(virt.delta_secs()),
            virtual_time: virt.elapsed_secs_wrapped(),
            real_delta_time: dt(real.delta_secs()),
            real_time: real.elapsed_secs_wrapped(),
            num_effects,
            pad: 0,
        };
        let transform = at.to_matrix();
        let gpu_spawner = GpuSpawner {
            transform: rows(&transform),
            inverse_transform: rows(&transform.inverse()),
            spawn: if running {
                spawner.map_or(0, |s| s.spawn_count) as i32
            } else {
                0
            },
            seed: compiled.prng_seed(),
            capacity: storage.capacity,
            frame: *frame_index,
        };
        let props = properties
            .map(|p| p.serialize(&asset.property_layout()))
            .filter(|bytes| !bytes.is_empty());
        let push = EffectPush {
            particles: storage.particles.address,
            free_slots: storage.free_slots.address,
            alive: storage.alive.address,
            last: storage.last.address,
            state: storage.state.address,
            spawner: storage.spawner.address,
            sim: storage.sim.address,
            properties: storage.properties.address,
            rows: BlockRows::default(),
        };
        let buffers = (
            storage.spawner.handle,
            storage.sim.handle,
            storage.properties.handle,
        );
        let (init, update) = (kernels.init.clone(), kernels.update.clone());
        let (spawn, capacity, slots) = (gpu_spawner.spawn, storage.capacity, *slots);
        frame.push(move |rd, modules, cmd, ctx| {
            let (Some(init), Some(update), Some(rows)) = (
                modules.get(&init),
                modules.get(&update),
                ctx.block_rows(slots),
            ) else {
                return;
            };
            unsafe {
                rd.device
                    .cmd_update_buffer(cmd, buffers.0, 0, bytemuck::bytes_of(&gpu_spawner));
                rd.device
                    .cmd_update_buffer(cmd, buffers.1, 0, bytemuck::bytes_of(&sim));
                if let Some(props) = &props {
                    rd.device.cmd_update_buffer(cmd, buffers.2, 0, props);
                }
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            );
            let push = EffectPush { rows, ..push };
            if spawn > 0 {
                record_dispatch(rd, cmd, init, "init", &push, spawn as u32, None);
                compute_to_compute_barrier(rd, cmd);
            }
            record_dispatch(rd, cmd, update, "update", &push, capacity, None);
        });
        any = true;
    }
    if any {
        tlas.request_rebuild();
    }
}

fn free_storage(
    remove: On<Remove<EffectStorage>>,
    storage: Query<&EffectStorage>,
    rd: Option<Res<RenderDevice>>,
) {
    if let (Ok(storage), Some(rd)) = (storage.get(remove.entity), rd) {
        storage.destroy(&rd);
    }
}

fn free_all_storage(world: &mut World) {
    let mut storage = world.query::<&EffectStorage>();
    let rd = world.resource::<RenderDevice>();
    for storage in storage.iter(world) {
        storage.destroy(rd);
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::Vec4;
    use bevy_hanabi::{Gradient, *};

    use super::*;

    /// A flame: spawned in a sphere, rising and dragged, aging out, coloured and sized over its
    /// life, facing the camera, with a property for its lift.
    fn flame(space: SimulationSpace) -> EffectAsset {
        let writer = ExprWriter::new();
        let lift = writer.add_property("lift", 1.5_f32.into());
        let init_pos = SetPositionSphereModifier {
            center: writer.lit(Vec3::ZERO).expr(),
            radius: writer.lit(0.01).expr(),
            dimension: ShapeDimension::Volume,
        };
        let init_vel = SetVelocitySphereModifier {
            center: writer.lit(Vec3::ZERO).expr(),
            speed: writer.lit(0.05).uniform(writer.lit(0.1)).expr(),
        };
        let init_age = SetAttributeModifier::new(Attribute::AGE, writer.lit(0.).expr());
        let init_lifetime = SetAttributeModifier::new(
            Attribute::LIFETIME,
            writer.lit(0.4).uniform(writer.lit(0.8)).expr(),
        );
        let accel = AccelModifier::new((writer.lit(Vec3::Y) * writer.prop(lift)).expr());
        let drag = LinearDragModifier::new(writer.lit(2.).expr());
        let kill = KillAabbModifier::new(writer.lit(Vec3::ZERO).expr(), writer.lit(Vec3::ONE).expr())
            .with_kill_inside(false);
        let mut color = Gradient::new();
        color.add_key(0.0, Vec4::new(4., 2., 0.5, 1.));
        color.add_key(1.0, Vec4::ZERO);
        let mut size = Gradient::new();
        size.add_key(0.0, Vec3::splat(0.02));
        size.add_key(1.0, Vec3::splat(0.005));
        EffectAsset::new(64, SpawnerSettings::rate(60.0.into()), writer.finish())
            .with_simulation_space(space)
            .init(init_pos)
            .init(init_vel)
            .init(init_age)
            .init(init_lifetime)
            .update(accel)
            .update(drag)
            .update(kill)
            .render(ColorOverLifetimeModifier::new(color))
            .render(SizeOverLifetimeModifier {
                gradient: size,
                screen_space_size: false,
            })
            .render(OrientModifier::new(OrientMode::FaceCameraPosition))
    }

    #[test]
    fn kernels_compile() {
        for space in [SimulationSpace::Global, SimulationSpace::Local] {
            let sources = EffectShaderSources::generate(&flame(space), None, 0).unwrap();
            for (name, source) in [
                ("init", &sources.init_shader_source),
                ("update", &sources.update_shader_source),
            ] {
                if let Err(err) = compile_source(source_hash(source), source.clone()) {
                    panic!("{space:?} {name}:\n{err}");
                }
            }
        }
    }
}
