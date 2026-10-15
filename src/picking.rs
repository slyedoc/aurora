//! Picking on the GPU: rays traced against the frame's own TLAS with inline ray queries.
//!
//! A [`RayCaster`] holds a world-space ray. Every frame its ray is traced inside that frame's
//! command buffer, right after the TLAS rebuild, so it sees exactly the scene the frame draws:
//! skinned meshes, landscapes and spheres included. Each caster gets its nearest
//! [`MAX_HITS`] hits, nearest first, so a consumer can skip what it does not want (editor
//! helpers, the thing being dragged) without GPU-side filters.
//!
//! When a ray is pending, the frame is submitted in two parts: transforms, skins,
//! TLAS and the pick go first in a submit of their own, the trace and present follow behind
//! its timeline point. That first part is a millisecond of work, so by the next frame's
//! [`First`] its point has passed even while the rest of the frame is still on the GPU, and the hits are
//! read in place from host-visible memory into [`RayHits`]. A ray set during a frame's
//! `Update` therefore has its hits in every system of the next frame, before anything that
//! acted on them could have been drawn.

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider, Flip},
    render_device::RenderDevice,
    tlas_builder::{GpuInstanceSlots, TLAS},
};

/// Hits kept per ray (must match pick.slang).
pub const MAX_HITS: usize = 8;

/// Traces `ray` every frame; the hits land in this entity's [`RayHits`] at the start of the
/// next frame.
#[derive(Component, Clone, Copy, Debug)]
#[require(RayHits)]
pub struct RayCaster {
    pub ray: Ray3d,
    /// Hits farther than this along the ray are ignored.
    pub max_distance: f32,
    /// Instance mask the ray tests against (render-layer bits, as the trace uses them).
    pub mask: u8,
}

impl RayCaster {
    pub fn new(ray: Ray3d) -> Self {
        Self {
            ray,
            max_distance: f32::MAX,
            mask: 0xFF,
        }
    }
}

/// One traced hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub entity: Entity,
    pub distance: f32,
    pub point: Vec3,
    /// Geometric normal of the hit triangle (or sphere), facing the ray.
    pub normal: Vec3,
    /// Triangle index within the mesh.
    pub triangle: u32,
    /// Barycentrics of the hit within the triangle (vertices 1 and 2).
    pub barycentrics: Vec2,
}

/// The last traced hits of this entity's [`RayCaster`], nearest first; empty on a miss.
#[derive(Component, Clone, Debug, Default, Deref)]
pub struct RayHits(pub Vec<RayHit>);

/// One ray (pick.slang `Ray`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuRay {
    origin: [f32; 3],
    t_max: f32,
    direction: [f32; 3],
    mask: u32,
}

/// One hit (pick.slang `Hit`); `t < 0` ends a ray's list.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuHit {
    position: [f32; 3],
    t: f32,
    normal: [f32; 3],
    instance: u32,
    barycentrics: [f32; 2],
    primitive: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PickParams {
    tlas: u64,
    rays: u64,
    hits: u64,
    count: u32,
    pad: u32,
    slots: u64,
}

/// The pick pass: this frame's rays, the buffers they trace from and into, and the early
/// submit that carries them.
#[derive(Resource)]
pub struct Picker {
    module: Handle<ComputeModule>,
    capacity: u32,
    /// The rays, read in place by the pick: one copy per frame in flight.
    rays: Flip<Buffer<GpuRay>>,
    hits: Buffer<GpuHit>,
    /// This frame's casters, in ray order.
    casters: Vec<Entity>,
    staged: Vec<GpuRay>,
    /// Casters whose hits the submitted pick is writing.
    in_flight: Vec<Entity>,
    command_buffers: [vk::CommandBuffer; 2],
    /// The pick submit\'s point on the timeline, while hits are in flight.
    pub point: crate::timeline::SubmitPoint,
}

impl Picker {
    fn new(rd: &RenderDevice, module: Handle<ComputeModule>) -> Self {
        let command_buffers = unsafe {
            rd.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(rd.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2),
            )
        }
        .unwrap();
        Self {
            module,
            capacity: 0,
            rays: Flip::default(),
            hits: Buffer::default(),
            casters: Vec::new(),
            staged: Vec::new(),
            in_flight: Vec::new(),
            command_buffers: [command_buffers[0], command_buffers[1]],
            point: crate::timeline::SubmitPoint::default(),
        }
    }

    /// Whether this frame traces rays, and so submits in two parts.
    pub fn active(&self, modules: &ComputeModules, tlas: &TLAS) -> bool {
        !self.casters.is_empty()
            && tlas.acceleration_structure.address != 0
            && modules.get(&self.module).is_some()
    }

    /// Starts the early command buffer the transforms, TLAS and pick record into.
    pub fn begin(&self, rd: &RenderDevice, frame: usize) -> vk::CommandBuffer {
        let cmd = self.command_buffers[frame % 2];
        unsafe {
            rd.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            rd.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        }
        // Submitted before the CPU waits for the previous frame: order the scene work here
        // (instance rows, TLAS rebuild) after that frame's trace on the queue.
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        cmd
    }

    /// Records the pick after the TLAS build, ends `cmd` and submits it on the timeline
    /// ([`Self::point`], which the frame's submit waits for).
    pub fn submit(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        tlas: &TLAS,
    ) {
        let count = self.staged.len() as u32;
        if count > self.capacity {
            self.grow(rd, count);
        }
        let rays = self.rays.write();
        if rays.nr_elements < self.capacity as u64 {
            rd.destroyer.destroy_buffer(rays.handle);
            *rays =
                rd.create_host_buffer(self.capacity as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
        }
        rd.map_buffer(rays).as_slice_mut()[..self.staged.len()].copy_from_slice(&self.staged);
        let module = modules
            .get(&self.module)
            .expect("active() checked the module");
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );
        record_dispatch(
            rd,
            cmd,
            module,
            "pick",
            &PickParams {
                tlas: tlas.acceleration_structure.address,
                rays: self.rays.current().address,
                hits: self.hits.address,
                count,
                pad: 0,
                slots: tlas.instance_slots_address(),
            },
            count,
            None,
        );
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
        unsafe {
            rd.end_command_buffer(cmd).unwrap();
            let queue = rd.queue.lock().unwrap();
            self.point = rd
                .timeline
                .submit(
                    rd,
                    *queue,
                    std::slice::from_ref(&cmd),
                    &[],
                    &[],
                    &[],
                    vk::Fence::null(),
                    None,
                )
                .unwrap_or_else(|e| {
                    crate::aftermath::note_device_lost(e);
                    panic!("pick submit failed: {e:?}");
                });
        }
        self.in_flight = std::mem::take(&mut self.casters);
    }

    /// Room for `count` rays: the hits now, each ray copy when it is next written.
    fn grow(&mut self, rd: &RenderDevice, count: u32) {
        rd.destroyer.destroy_buffer(self.hits.handle);
        let capacity = count.next_power_of_two().max(16);
        self.hits = rd.create_buffer(
            capacity as u64 * MAX_HITS as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuToCpu,
        );
        self.capacity = capacity;
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for rays in self.rays.copies() {
            rd.destroyer.destroy_buffer(rays.handle);
        }
        rd.destroyer.destroy_buffer(self.hits.handle);
    }
}

/// [`RenderSet::Extract`]: this frame's rays, in a stable order.
fn gather_rays(mut picker: ResMut<Picker>, casters: Query<(Entity, &RayCaster)>) {
    let picker = &mut *picker;
    picker.casters.clear();
    picker.staged.clear();
    for (entity, caster) in &casters {
        picker.casters.push(entity);
        picker.staged.push(GpuRay {
            origin: caster.ray.origin.to_array(),
            t_max: caster.max_distance,
            direction: caster.ray.direction.to_array(),
            mask: caster.mask as u32,
        });
    }
}

/// [`First`]: the previous frame's hits into [`RayHits`]. Runs before anything can change an
/// instance slot, so the slot table still names the entities the pick saw.
fn read_hits(
    rd: Option<Res<RenderDevice>>,
    mut picker: ResMut<Picker>,
    slots: Res<GpuInstanceSlots>,
    mut hits: Query<&mut RayHits>,
) {
    let Some(rd) = rd else {
        return;
    };
    if picker.in_flight.is_empty() {
        return;
    }
    let picker = &mut *picker;
    rd.timeline.wait(&rd, picker.point).unwrap_or_else(|e| {
        crate::aftermath::note_device_lost(e);
        panic!("pick wait failed: {e:?}");
    });
    let view = rd.map_buffer(&mut picker.hits);
    for (ray, entity) in picker.in_flight.drain(..).enumerate() {
        let Ok(mut out) = hits.get_mut(entity) else {
            continue;
        };
        out.0.clear();
        for i in 0..MAX_HITS {
            let hit = view[ray * MAX_HITS + i];
            if hit.t < 0.0 {
                break;
            }
            let Some(entity) = slots.entity(hit.instance) else {
                continue;
            };
            out.0.push(RayHit {
                entity,
                distance: hit.t,
                point: Vec3::from_array(hit.position),
                normal: Vec3::from_array(hit.normal),
                triangle: hit.primitive,
                barycentrics: Vec2::from_array(hit.barycentrics),
            });
        }
    }
}

fn cleanup_picker(world: &mut World) {
    world.resource_scope(|world, mut picker: Mut<Picker>| {
        picker.destroy(world.resource::<RenderDevice>());
    });
}

pub struct PickingPlugin;

impl Plugin for PickingPlugin {
    fn build(&self, app: &mut App) {
        let shader = app
            .world()
            .resource::<AssetServer>()
            .load(aurora_asset("shaders/pick.slang"));
        let module = app
            .world()
            .resource::<AssetServer>()
            .add(ComputeModule::new(shader, &["pick"]));
        let picker = Picker::new(app.world().resource::<RenderDevice>(), module);
        app.insert_resource(picker)
            .add_systems(First, read_hits.before(crate::tlas_builder::clear_freed))
            .add_systems(Last, gather_rays.in_set(RenderSet::Extract))
            .add_systems(TeardownSchedule, cleanup_picker.before(on_shutdown));
    }
}
