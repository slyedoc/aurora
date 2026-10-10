//! Scatter: plants and rocks grown on the GPU from the material pages (`landscape/scatter.slang`).
//!
//! A [`ScatterSpecies`] child of a landscape, carrying the plant's `AuroraMesh3d` and
//! `AuroraMaterial3d`, becomes one aurora `InstanceBlock`: a slot per cell of a square window
//! around the viewer. The kernel writes each slot's transform (or a zero one where nothing
//! grows) straight into the TLAS instance array; nothing is read back. With `poses`, it also
//! picks each plant's wind pose every frame by swapping BLAS pointers among the baked poses.

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    compute::{ComputeModules, compute_to_compute_barrier, record_dispatch},
    mesh::AuroraMesh,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::{BlockPose, GpuInstance, InstanceBlock, InstanceBlockSlots, TLAS},
};
use bytemuck::{Pod, Zeroable};

use crate::{
    Landscape, LandscapeKernels, LandscapeViewer,
    page::{GpuPagePool, LandscapePages},
    shading::PALETTE_MAX,
    tiles::viewer_position,
};

/// Most poses a species may cycle through.
const POSES_MAX: usize = 64;

/// One plant grown over its landscape (a child of it): a jittered spot per `spacing` cell
/// between `inner_radius` and `radius` of the viewer, kept with probability = the material
/// weights there times the `grows_on` factor of each palette id, and where the slope is under
/// `max_slope`. Tiers are species: a dense near one, and a sparser, larger one whose
/// `inner_radius` starts where the near one fades out.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default)]
pub struct ScatterSpecies {
    /// Metres between candidate spots.
    pub spacing: f32,
    /// Metres around the viewer.
    pub radius: f32,
    /// Metres around the viewer left bare (a nearer tier covers it).
    pub inner_radius: f32,
    /// Metres over which each edge of the ring dithers in.
    pub fade: f32,
    /// (palette id, density factor 0..1).
    pub grows_on: Vec<(u8, f32)>,
    /// Random uniform scale range.
    pub scale: Vec2,
    /// Degrees.
    pub max_slope: f32,
    /// Metres the base sits below the ground.
    pub sink: f32,
    /// Wind poses the plant cycles through (empty = rigid).
    pub poses: Vec<Handle<AuroraMesh>>,
    /// Gust cycles per second.
    pub wind_speed: f32,
    pub seed: u32,
}

impl Default for ScatterSpecies {
    fn default() -> Self {
        Self {
            spacing: 0.75,
            radius: 40.0,
            inner_radius: 0.0,
            fade: 6.0,
            grows_on: Vec::new(),
            scale: Vec2::new(0.8, 1.2),
            max_slope: 40.0,
            sink: 0.05,
            poses: Vec::new(),
            wind_speed: 0.35,
            seed: 1,
        }
    }
}

impl ScatterSpecies {
    /// Cells per window side.
    pub fn width(&self) -> u32 {
        (2.0 * self.radius / self.spacing.max(0.05)).ceil().max(1.0) as u32
    }
}

/// `ScatterPush` in scatter.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterPush {
    instances: u64,
    materials: u64,
    poses: u64,
    factors: u64,
    pool: GpuPagePool,
    first: u32,
    width: u32,
    cell0_x: i32,
    cell0_z: i32,
    spacing: f32,
    scale_lo: f32,
    scale_hi: f32,
    max_slope: f32,
    sink: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    seed_poses: u32,
    wave: f32,
    inner_radius: f32,
    fade: f32,
}

/// A species' GPU tables and what its block last held.
#[derive(Component)]
pub struct ScatterState {
    factors: Buffer<f32>,
    poses: Buffer<BlockPose>,
    last: Option<(IVec2, u32)>,
}

impl ScatterState {
    fn destroy(&mut self, rd: &RenderDevice) {
        for handle in [self.factors.handle, self.poses.handle] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        self.factors = Buffer::default();
        self.poses = Buffer::default();
    }
}

/// A new species gets its instance block and tables; one whose window size changed gets a
/// new block (its slots are reassigned next frame).
pub fn size_blocks(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    species: Query<
        (Entity, &ScatterSpecies, Option<&InstanceBlock>, Has<ScatterState>),
        Changed<ScatterSpecies>,
    >,
) {
    for (entity, species, block, has_state) in &species {
        let capacity = species.width() * species.width();
        let poses: Vec<Handle<AuroraMesh>> =
            species.poses.iter().take(POSES_MAX).cloned().collect();
        if block.is_none_or(|b| b.capacity != capacity || b.poses != poses) {
            commands
                .entity(entity)
                .remove::<(GpuInstance, InstanceBlockSlots)>()
                .insert(InstanceBlock { capacity, poses });
        }
        if !has_state {
            commands.entity(entity).insert(ScatterState {
                factors: rd.create_host_buffer(
                    PALETTE_MAX as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                poses: rd.create_host_buffer(POSES_MAX as u64, vk::BufferUsageFlags::STORAGE_BUFFER),
                last: None,
            });
        }
    }
}

/// Grow every species around the viewer: on wind every frame, otherwise when its window
/// moved, the landscape re-evaluated, the species changed or the instance array was rebuilt.
#[allow(clippy::too_many_arguments)]
pub fn scatter_species(
    rd: Res<RenderDevice>,
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    mut tlas: ResMut<TLAS>,
    time: Res<Time>,
    viewers: Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    landscapes: Query<(&Landscape, Ref<LandscapePages>, &GlobalTransform)>,
    mut species: Query<(
        Ref<ScatterSpecies>,
        &ChildOf,
        &InstanceBlockSlots,
        &mut ScatterState,
    )>,
) {
    let Some(module) = modules.get(&kernels.scatter) else {
        return;
    };
    let Some(viewer) = viewer_position(&viewers, &cameras) else {
        return;
    };
    let generation = tlas.generation();
    let mut jobs: Vec<ScatterPush> = Vec::new();
    for (spec, child_of, slots, mut state) in &mut species {
        let Ok((landscape, pages, transform)) = landscapes.get(child_of.parent()) else {
            continue;
        };
        // Scatter places on the flat plane; planets are not supported yet.
        if landscape.sphere.is_some() {
            continue;
        }
        if !pages.is_complete() || !tlas.block_ready(*slots) {
            continue;
        }
        let width = spec.width();
        if slots.count != width * width {
            continue;
        }
        let poses = tlas.block_poses(*slots).unwrap_or(&[]);
        let windy = !poses.is_empty() && spec.wind_speed != 0.0;
        let origin = transform.translation();
        let local = (viewer - origin).xz();
        let cell0 = (local / spec.spacing).floor().as_ivec2() - IVec2::splat(width as i32 / 2);
        let moved = state.last != Some((cell0, generation));
        if !(windy || moved || spec.is_changed() || pages.is_changed()) {
            continue;
        }
        if moved || spec.is_changed() {
            let mut factors = [0.0f32; PALETTE_MAX];
            for &(id, f) in &spec.grows_on {
                factors[id as usize] = f.clamp(0.0, 1.0);
            }
            rd.map_buffer(&mut state.factors).copy_from_slice(&factors);
        }
        rd.map_buffer(&mut state.poses).copy_from_slice(poses);
        state.last = Some((cell0, generation));
        jobs.push(ScatterPush {
            instances: tlas.instances_address(),
            materials: pages.materials_address(),
            poses: state.poses.address,
            factors: state.factors.address,
            pool: pages.pool(),
            first: slots.first,
            width,
            cell0_x: cell0.x,
            cell0_z: cell0.y,
            spacing: spec.spacing,
            scale_lo: spec.scale.x,
            scale_hi: spec.scale.y,
            max_slope: spec.max_slope,
            sink: spec.sink,
            origin_x: origin.x,
            origin_y: origin.y,
            origin_z: origin.z,
            seed_poses: (spec.seed & 0x00FF_FFFF)
                | if windy { (poses.len() as u32) << 24 } else { 0 },
            wave: (time.elapsed_secs_f64() * spec.wind_speed as f64).fract() as f32,
            inner_radius: spec.inner_radius,
            fade: spec.fade,
        });
    }
    if jobs.is_empty() {
        return;
    }
    rd.run_transfer_commands(|cmd| {
        for job in &jobs {
            record_dispatch(&rd, cmd, module, "scatter", job, job.width * job.width, None);
        }
        compute_to_compute_barrier(&rd, cmd);
    });
    tlas.request_rebuild();
}

pub fn on_state_removed(
    remove: On<Remove<ScatterState>>,
    mut states: Query<&mut ScatterState>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(mut state) = states.get_mut(remove.entity)
    {
        state.destroy(&rd);
    }
}

pub fn release_states(mut states: Query<&mut ScatterState>, rd: Res<RenderDevice>) {
    for mut state in &mut states {
        state.destroy(&rd);
    }
}
