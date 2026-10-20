//! Scatter: plants and rocks grown on the GPU from the material pages (`landscape/scatter.slang`).
//!
//! A [`ScatterSpecies`] under a landscape (its nearest landscape ancestor: a child, or grouped
//! under a folder entity or an instanced clutter scene), carrying the plant's `AuroraMesh3d` and
//! `AuroraMaterial3d`, becomes one aurora `InstanceBlock` over a square window of cells around
//! the viewer. Only the cells where it grows take rows (counted, scanned and written in cell
//! order by three kernel passes), so the block holds the plants, not the window: it starts at
//! `INITIAL_ROWS` and is resized from the kept total, read back a frame or two late. The kernel
//! writes each row's transform, and last frame's from the previous gust phase (motion vectors
//! without a fixed slot per plant), straight into the TLAS instance arrays. With `poses`, it
//! also leans each plant with the wind every frame: the baked pose under its lean (a BLAS
//! pointer swap) plus a shear for the rest of the way, so the sway is continuous at any frame
//! rate.

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    compute::{
        ComputeModules, FrameCommands, compute_to_compute_barrier, memory_barrier, record_dispatch,
    },
    mesh::AuroraMesh,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::{BlockPose, GpuInstance, InstanceBlock, InstanceBlockSlots, TLAS},
};
use bytemuck::{Pod, Zeroable};

use crate::{
    Landscape, LandscapeKernels, LandscapeViewer,
    interaction::InteractionField,
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
    /// Wind poses at evenly spaced lean amounts, least first (empty = rigid).
    pub poses: Vec<Handle<AuroraMesh>>,
    /// Gust cycles per second.
    pub wind_speed: f32,
    /// The shear (x per y, plant space) standing in for one pose step: the lean between two
    /// poses. 0 = the pose steps alone.
    pub wind_shear: f32,
    pub seed: u32,
    /// Multiplies every `grows_on` factor: thins (or fills) the species without moving it.
    pub density: f32,
    /// How much plants bunch into tufts and bald spots, thick tufts standing taller (0 = an
    /// even carpet, 1 = full).
    pub clump: f32,
    /// Metres across a tuft.
    pub clump_size: f32,
    /// Scale of the plants where their ground thins out to nothing, so they shrink into a path
    /// instead of standing full height at its edge (1 = no taper).
    pub edge_scale: f32,
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
            wind_shear: 0.0,
            seed: 1,
            density: 1.0,
            clump: 1.0,
            clump_size: 7.0,
            edge_scale: 0.45,
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
    tables: u64,
    /// The interaction field (interaction.rs), 0 while nothing disturbs it.
    field: u64,
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

/// `SpeciesTables` in scatter.slang: the keep factor per palette id, and the wind poses.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpeciesTables {
    factors: [f32; PALETTE_MAX],
    poses: [BlockPose; POSES_MAX],
    wind_shear: f32,
    /// Last frame's gust phase, when there was a last frame (`prev_valid`).
    prev_wave: f32,
    prev_valid: u32,
    _pad: u32,
    prev_instances: u64,
    clump: f32,
    clump_size: f32,
    edge_scale: f32,
    /// Rows the block holds.
    capacity: u32,
    /// Per 64-cell group: its keep mask (two words).
    keep: u64,
    /// Per group: its kept count, then its first row; the total after them.
    groups: u64,
    /// The total kept, read back to size the block.
    total_out: u64,
}

/// Rows a new species' block starts with (it grows to what is kept, read back).
const INITIAL_ROWS: u32 = 4096;
/// Rows a block grows by at least, and rounds to.
const ROW_STEP: u32 = 1024;

/// A species' GPU tables, its compaction scratch, and what its block last held.
#[derive(Component)]
pub struct ScatterState {
    tables: Buffer<SpeciesTables>,
    /// Per 64-cell group: the keep mask (two words each).
    keep: Buffer<u32>,
    /// Per group: kept count, then first row; the total after them.
    groups: Buffer<u32>,
    /// The total kept (host-visible): what the block should hold.
    total: Buffer<u32>,
    /// Cells the scratch is sized for.
    cells: u32,
    /// The window and TLAS generation last grown into, and the gust phase it had.
    last: Option<(IVec2, u32)>,
    last_wave: Option<f32>,
}

impl ScatterState {
    fn new(rd: &RenderDevice, cells: u32) -> Self {
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let groups = cells.div_ceil(64) as u64;
        Self {
            tables: rd.create_device_buffer(1, storage),
            keep: rd.create_device_buffer(groups * 2, storage),
            groups: rd.create_device_buffer(groups + 1, storage),
            total: rd.create_host_buffer(1, vk::BufferUsageFlags::STORAGE_BUFFER),
            cells,
            last: None,
            last_wave: None,
        }
    }

    /// The total the GPU last kept (a frame or two old).
    fn kept(&mut self, rd: &RenderDevice) -> u32 {
        if self.total.handle == vk::Buffer::null() {
            return 0;
        }
        rd.map_buffer(&mut self.total).as_slice_mut()[0]
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for buffer in [
            self.tables.handle,
            self.keep.handle,
            self.groups.handle,
            self.total.handle,
        ] {
            if buffer != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(buffer);
            }
        }
        self.tables = Buffer::default();
        self.keep = Buffer::default();
        self.groups = Buffer::default();
        self.total = Buffer::default();
    }
}

/// The rows a block should hold for `kept` plants, given what it holds now: grown past them
/// (a quarter's headroom), shrunk once they fall under a quarter of it.
fn block_rows(kept: u32, now: u32, cells: u32) -> u32 {
    let floor = INITIAL_ROWS.min(cells);
    let target = (kept + kept / 4).max(floor).next_multiple_of(ROW_STEP).min(cells);
    if kept > now || (kept < now / 4 && now > floor) {
        target
    } else {
        now
    }
}

/// A new species gets its instance block (`INITIAL_ROWS`, grown to what it keeps) and its
/// tables; one whose window size or poses changed gets a new block (its slots are reassigned
/// next frame) and scratch sized for its window.
pub fn size_blocks(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    mut species: Query<
        (
            Entity,
            &ScatterSpecies,
            Option<&InstanceBlock>,
            Option<&mut ScatterState>,
        ),
        Changed<ScatterSpecies>,
    >,
) {
    for (entity, species, block, state) in &mut species {
        let cells = species.width() * species.width();
        let poses: Vec<Handle<AuroraMesh>> =
            species.poses.iter().take(POSES_MAX).cloned().collect();
        let capacity = block.map_or(INITIAL_ROWS, |b| b.capacity).min(cells);
        if block.is_none_or(|b| b.capacity != capacity || b.poses != poses) {
            commands
                .entity(entity)
                .remove::<(GpuInstance, InstanceBlockSlots)>()
                .insert(InstanceBlock { capacity, poses });
        }
        match state {
            Some(mut state) if state.cells != cells => {
                state.destroy(&rd);
                *state = ScatterState::new(&rd, cells);
            }
            Some(_) => {}
            None => {
                commands.entity(entity).insert(ScatterState::new(&rd, cells));
            }
        }
    }
}

/// Grow every species around the viewer: on wind every frame, otherwise when its window
/// moved, the landscape re-evaluated, the species changed or the instance array was rebuilt.
#[allow(clippy::too_many_arguments)]
pub fn scatter_species(
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    mut tlas: ResMut<TLAS>,
    time: Res<Time>,
    viewers: Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    landscapes: Query<(&Landscape, Ref<LandscapePages>, &GlobalTransform)>,
    parents: Query<&ChildOf>,
    field: Res<InteractionField>,
    mut frame: ResMut<FrameCommands>,
    rd: Res<RenderDevice>,
    mut commands: Commands,
    mut species: Query<(
        Entity,
        Ref<ScatterSpecies>,
        &ChildOf,
        &InstanceBlock,
        &InstanceBlockSlots,
        &mut ScatterState,
    )>,
) {
    if modules.get(&kernels.scatter).is_none() {
        return;
    }
    let Some(viewer) = viewer_position(&viewers, &cameras) else {
        return;
    };
    let generation = tlas.generation();
    // A disturbed field re-grows every frame, so plants spring back as it decays.
    let field = field.address(time.elapsed_secs_f64());
    let mut jobs: Vec<(vk::Buffer, SpeciesTables, ScatterPush)> = Vec::new();
    // Plants appear or vanish (the window moved, the pages or the species changed): the TLAS
    // rebuilds. Wind and the interaction field only bend them (rows moved, poses swapped): a
    // refit.
    let mut structural = false;
    let wave_now = |spec: &ScatterSpecies| {
        (time.elapsed_secs_f64() * spec.wind_speed as f64).fract() as f32
    };
    for (entity, spec, child_of, block, slots, mut state) in &mut species {
        let Some((landscape, pages, transform)) = std::iter::once(child_of.parent())
            .chain(parents.iter_ancestors(child_of.parent()))
            .find_map(|e| landscapes.get(e).ok())
        else {
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
        if slots.count != block.capacity || state.cells != width * width {
            continue;
        }
        // The block holds what grows: resized from the last total the GPU kept (its slots are
        // reassigned next frame, so this species waits a frame).
        let rows = block_rows(state.kept(&rd), block.capacity, state.cells);
        if rows != block.capacity {
            // As `size_blocks` does: without its slots the TLAS assigns the block anew.
            commands
                .entity(entity)
                .remove::<(GpuInstance, InstanceBlockSlots)>()
                .insert(InstanceBlock {
                    capacity: rows,
                    poses: block.poses.clone(),
                });
            state.last = None;
            continue;
        }
        let poses = tlas.block_poses(*slots).unwrap_or(&[]);
        let windy = !poses.is_empty() && spec.wind_speed != 0.0;
        let origin = transform.translation();
        let local = (viewer - origin).xz();
        let cell0 = (local / spec.spacing).floor().as_ivec2() - IVec2::splat(width as i32 / 2);
        let moved = state.last != Some((cell0, generation));
        if !(windy || field != 0 || moved || spec.is_changed() || pages.is_changed()) {
            continue;
        }
        structural |= moved || spec.is_changed() || pages.is_changed();
        let mut tables = SpeciesTables::zeroed();
        for &(id, f) in &spec.grows_on {
            tables.factors[id as usize] = (f * spec.density).clamp(0.0, 1.0);
        }
        tables.poses[..poses.len()].copy_from_slice(poses);
        tables.wind_shear = spec.wind_shear;
        tables.clump = spec.clump.clamp(0.0, 1.0);
        tables.clump_size = spec.clump_size.max(0.1);
        tables.edge_scale = spec.edge_scale;
        let wave = wave_now(&spec);
        if let Some(prev) = state.last_wave {
            tables.prev_wave = prev;
            tables.prev_valid = 1;
        }
        tables.prev_instances = tlas.prev_instances_address();
        tables.capacity = block.capacity;
        tables.keep = state.keep.address;
        tables.groups = state.groups.address;
        tables.total_out = state.total.address;
        state.last = Some((cell0, generation));
        state.last_wave = Some(wave);
        let push = ScatterPush {
            instances: tlas.instances_address(),
            materials: pages.materials_address(),
            tables: state.tables.address,
            field,
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
            wave,
            inner_radius: spec.inner_radius,
            fade: spec.fade,
        };
        jobs.push((state.tables.handle, tables, push));
    }
    if jobs.is_empty() {
        return;
    }
    // Recorded into the frame's command buffer, ahead of the TLAS build that reads the rows.
    let kernel = kernels.scatter.clone();
    frame.push(move |rd, modules, cmd| {
        let Some(module) = modules.get(&kernel) else {
            return;
        };
        for (buffer, tables, _) in &jobs {
            unsafe {
                rd.device
                    .cmd_update_buffer(cmd, *buffer, 0, bytemuck::bytes_of(tables));
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
        // Count every species' kept spots, scan them into rows, then write the plants.
        for (_, _, job) in &jobs {
            record_dispatch(rd, cmd, module, "scatter_count", job, job.width * job.width, None);
        }
        compute_to_compute_barrier(rd, cmd);
        for (_, _, job) in &jobs {
            record_dispatch(rd, cmd, module, "scatter_scan", job, 64, None);
        }
        compute_to_compute_barrier(rd, cmd);
        for (_, _, job) in &jobs {
            record_dispatch(rd, cmd, module, "scatter_write", job, job.width * job.width, None);
        }
    });
    if structural {
        tlas.request_rebuild();
    } else {
        tlas.request_refit();
    }
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
