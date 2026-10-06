//! GPU transform hierarchy.
//!
//! Every entity with a [`Transform`] owns a [`GpuNode`] slot in a GPU-resident node table:
//! local TRS, parent slot, and an intrusive child list. The CPU mirrors only *deltas* — a
//! node's local on `Changed<Transform>` / first sight / reparent, its parent on reparent, a
//! node's child chain on `Changed<Children>` — and the GPU does the rest each frame:
//!
//! ```text
//! scatter deltas → frontier (changed nodes + all descendants, epoch-deduped, indirect)
//!               → propagate (per-node ancestor walk, frontier nodes only) → World table
//! ```
//!
//! `World` rows are `VkTransformMatrixKHR` rows, so the TLAS instance gather
//! (`tlas_builder.rs`) copies them straight into the instance buffer. Static scenes cost
//! nothing past the (empty) delta check. Kernels: `assets/shaders/transform.slang`.
//!
//! The GPU table is the source of world transforms. Every node `propagate` rewrote is copied
//! back (`readback`) in a small submit of its own, ahead of the rest of the frame, and the next
//! frame's `First` writes those rows into `GlobalTransform` -- so `Update` sees last frame's
//! world poses, as in stock bevy, without the CPU walking the hierarchy (`transform.rs`).

use std::mem::offset_of;

use ash::vk;
use bevy::{
    ecs::{
        hierarchy::{ChildOf, Children},
        lifecycle::{Remove, RemovedComponents},
        observer::On,
    },
    math::{Affine3A, Mat3A, Vec3A},
    prelude::*,
    transform::TransformSystems,
};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::{
    assets::aurora_asset,
    compute::{
        ComputeModule, ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch,
        record_dispatch_indirect,
    },
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

pub const NO_NODE: u32 = u32::MAX;
pub const ROOT_PARENT: u32 = u32::MAX;
const MAX_LEVELS: u32 = 16;
const FRONTIER_HEADER: u64 = 24;
const INDIRECT_WORDS: u64 = (MAX_LEVELS as u64 + 1) * 3;
const CONSUMER_ARGS_OFFSET: u64 = MAX_LEVELS as u64 * 3 * 4;

// ---- wire formats (must match transform.slang) --------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct NodeLocal {
    pub rotation: [f32; 4],
    pub translation: [f32; 4],
    pub scale: [f32; 4],
}

impl NodeLocal {
    pub fn from_transform(t: &Transform) -> Self {
        let r = t.rotation;
        let p = t.translation;
        let s = t.scale;
        Self {
            rotation: [r.x, r.y, r.z, r.w],
            translation: [p.x, p.y, p.z, 0.0],
            scale: [s.x, s.y, s.z, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LocalRecord {
    pub slot: u32,
    pub pad: [u32; 3],
    pub local: NodeLocal,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct U32Record {
    pub slot: u32,
    pub value: u32,
}

/// A node's world affine: three rows of a row-major 3x4 (`VkTransformMatrixKHR`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct NodeEnvironment {
    pub rows: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterLocalsParams {
    records: u64,
    locals: u64,
    count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterU32Params {
    records: u64,
    dst: u64,
    count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrontierSeedParams {
    records: u64,
    frontier: u64,
    epoch: u64,
    count: u32,
    base: u32,
    frame_id: u32,
    node_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrontierFinalizeParams {
    frontier: u64,
    indirect: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrontierExpandParams {
    first_child: u64,
    next_sibling: u64,
    frontier: u64,
    epoch: u64,
    frame_id: u32,
    node_count: u32,
}

/// One node's world rows copied back for `GlobalTransform` (must match transform.slang).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct ReadbackRecord {
    pub slot: u32,
    pub pad: [u32; 3],
    pub world: NodeEnvironment,
}

impl NodeEnvironment {
    pub fn affine(&self) -> Affine3A {
        let [r0, r1, r2] = self.rows;
        Affine3A {
            matrix3: Mat3A::from_cols(
                Vec3A::new(r0[0], r1[0], r2[0]),
                Vec3A::new(r0[1], r1[1], r2[1]),
                Vec3A::new(r0[2], r1[2], r2[2]),
            ),
            translation: Vec3A::new(r0[3], r1[3], r2[3]),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ReadbackParams {
    world: u64,
    frontier: u64,
    records: u64,
    total: u64,
    count: u32,
    base: u32,
    full_rebuild: u32,
    node_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PropagateParams {
    locals: u64,
    parent: u64,
    world: u64,
    frontier: u64,
    count: u32,
    base: u32,
    full_rebuild: u32,
    node_count: u32,
}

// ---- slots ----------------------------------------------------------------------------------

/// This entity's slot in the GPU node table. Assigned to every `Transform` entity in
/// `PostUpdate`; freed (and recycled) when the component or entity goes away.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuNode(pub u32);

#[derive(Resource, Default)]
pub struct GpuNodeSlots {
    free: Vec<u32>,
    next: u32,
    /// The entity holding each slot, for writing read-back rows into `GlobalTransform`.
    entities: Vec<Option<Entity>>,
}

impl GpuNodeSlots {
    fn alloc(&mut self, entity: Entity) -> u32 {
        let slot = self.free.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            s
        });
        if self.entities.len() <= slot as usize {
            self.entities.resize(slot as usize + 1, None);
        }
        self.entities[slot as usize] = Some(entity);
        slot
    }

    /// The entity holding `slot`.
    pub fn entity(&self, slot: u32) -> Option<Entity> {
        self.entities.get(slot as usize).copied().flatten()
    }

    /// High-water mark: every live slot is below this.
    pub fn count(&self) -> u32 {
        self.next
    }
}

fn assign_gpu_nodes(
    mut commands: Commands,
    unslotted: Query<Entity, (With<Transform>, Without<GpuNode>)>,
    mut slots: ResMut<GpuNodeSlots>,
) {
    for entity in &unslotted {
        commands.entity(entity).insert(GpuNode(slots.alloc(entity)));
    }
}

fn free_gpu_node(
    remove: On<Remove<GpuNode>>,
    nodes: Query<&GpuNode>,
    mut slots: ResMut<GpuNodeSlots>,
) {
    if let Ok(node) = nodes.get(remove.entity) {
        slots.free.push(node.0);
        if let Some(entity) = slots.entities.get_mut(node.0 as usize) {
            *entity = None;
        }
    }
}

// ---- the table ------------------------------------------------------------------------------

#[derive(Resource)]
pub struct GpuTransforms {
    module: Handle<ComputeModule>,
    // CPU mirror of the whole table: the growth path re-uploads it wholesale.
    locals: Vec<NodeLocal>,
    parent: Vec<u32>,
    first_child: Vec<u32>,
    next_sibling: Vec<u32>,
    node_count: u32,
    // This frame's deltas.
    local_records: Vec<LocalRecord>,
    parent_records: Vec<U32Record>,
    first_child_records: Vec<U32Record>,
    next_sibling_records: Vec<U32Record>,
    // Device table.
    capacity: u32,
    locals_buf: Buffer<NodeLocal>,
    parent_buf: Buffer<u32>,
    first_child_buf: Buffer<u32>,
    next_sibling_buf: Buffer<u32>,
    world_buf: Buffer<NodeEnvironment>,
    epoch_buf: Buffer<u32>,
    frontier_buf: Buffer<u32>,
    indirect_buf: Buffer<u32>,
    // Host-visible delta staging, rewritten after the in-flight fence wait.
    staging_locals: Buffer<LocalRecord>,
    staging_u32: Buffer<U32Record>,
    frame_id: u32,
    needs_full_rebuild: bool,
    warned_module: bool,
    // Readback: the rows `propagate` wrote, for `GlobalTransform` next frame.
    readback: Buffer<ReadbackRecord>,
    readback_total: Buffer<u32>,
    readback_cmd: vk::CommandBuffer,
    readback_fence: vk::Fence,
    /// A submitted readback the next `First` has to collect.
    readback_pending: bool,
    /// The build `record` just made was a full rebuild (read back every node).
    rebuilt_full: bool,
}

impl GpuTransforms {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            locals: Vec::new(),
            parent: Vec::new(),
            first_child: Vec::new(),
            next_sibling: Vec::new(),
            node_count: 0,
            local_records: Vec::new(),
            parent_records: Vec::new(),
            first_child_records: Vec::new(),
            next_sibling_records: Vec::new(),
            capacity: 0,
            locals_buf: Buffer::default(),
            parent_buf: Buffer::default(),
            first_child_buf: Buffer::default(),
            next_sibling_buf: Buffer::default(),
            world_buf: Buffer::default(),
            epoch_buf: Buffer::default(),
            frontier_buf: Buffer::default(),
            indirect_buf: Buffer::default(),
            staging_locals: Buffer::default(),
            staging_u32: Buffer::default(),
            frame_id: 0,
            needs_full_rebuild: true,
            warned_module: false,
            readback: Buffer::default(),
            readback_total: Buffer::default(),
            readback_cmd: vk::CommandBuffer::null(),
            readback_fence: vk::Fence::null(),
            readback_pending: false,
            rebuilt_full: false,
        }
    }

    /// Device address of the `World` table (0 until the first build) and its node coverage.
    pub fn world(&self) -> (u64, u32) {
        (self.world_buf.address, self.node_count)
    }

    fn ensure_node(&mut self, slot: u32) {
        let need = slot as usize + 1;
        if self.locals.len() < need {
            self.locals.resize(need, NodeLocal::default());
            self.parent.resize(need, ROOT_PARENT);
            self.first_child.resize(need, NO_NODE);
            self.next_sibling.resize(need, NO_NODE);
        }
    }

    fn push_local(&mut self, slot: u32, local: NodeLocal) {
        self.ensure_node(slot);
        self.locals[slot as usize] = local;
        self.local_records.push(LocalRecord {
            slot,
            pad: [0; 3],
            local,
        });
    }

    fn push_parent(&mut self, slot: u32, value: u32) {
        self.ensure_node(slot);
        self.parent[slot as usize] = value;
        self.parent_records.push(U32Record { slot, value });
    }

    fn push_first_child(&mut self, slot: u32, value: u32) {
        self.ensure_node(slot);
        self.first_child[slot as usize] = value;
        self.first_child_records.push(U32Record { slot, value });
    }

    fn push_next_sibling(&mut self, slot: u32, value: u32) {
        self.ensure_node(slot);
        self.next_sibling[slot as usize] = value;
        self.next_sibling_records.push(U32Record { slot, value });
    }

    /// Write a node's whole child chain: `first_child[parent]` plus each slotted child's
    /// `next_sibling`, `NO_NODE`-terminated. Unslotted children are skipped.
    fn push_chain(
        &mut self,
        parent_slot: u32,
        children: Option<&Children>,
        nodes: &Query<&GpuNode>,
    ) {
        let mut head = NO_NODE;
        let mut prev: Option<u32> = None;
        for child in children.into_iter().flat_map(|c| c.iter()) {
            let Ok(node) = nodes.get(child) else { continue };
            match prev {
                None => head = node.0,
                Some(p) => self.push_next_sibling(p, node.0),
            }
            prev = Some(node.0);
        }
        if let Some(last) = prev {
            self.push_next_sibling(last, NO_NODE);
        }
        self.push_first_child(parent_slot, head);
    }

    fn has_deltas(&self) -> bool {
        !(self.local_records.is_empty()
            && self.parent_records.is_empty()
            && self.first_child_records.is_empty()
            && self.next_sibling_records.is_empty())
    }

    fn clear_deltas(&mut self) {
        self.local_records.clear();
        self.parent_records.clear();
        self.first_child_records.clear();
        self.next_sibling_records.clear();
    }

    fn destroy_device(&mut self, rd: &RenderDevice) {
        for b in [
            self.parent_buf.handle,
            self.first_child_buf.handle,
            self.next_sibling_buf.handle,
            self.epoch_buf.handle,
            self.frontier_buf.handle,
            self.indirect_buf.handle,
            self.locals_buf.handle,
            self.world_buf.handle,
            self.readback.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
    }

    fn grow(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        let capacity = (self.node_count.max(1024)).next_power_of_two();
        log::debug!("GPU transform table: {} -> {capacity} nodes", self.capacity);
        self.destroy_device(rd);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        self.locals_buf = rd.create_device_buffer(capacity as u64, storage);
        self.parent_buf = rd.create_device_buffer(capacity as u64, storage);
        self.first_child_buf = rd.create_device_buffer(capacity as u64, storage);
        self.next_sibling_buf = rd.create_device_buffer(capacity as u64, storage);
        self.world_buf = rd.create_device_buffer(capacity as u64, storage);
        self.epoch_buf = rd.create_device_buffer(capacity as u64, storage);
        self.frontier_buf = rd.create_device_buffer(FRONTIER_HEADER + capacity as u64, storage);
        self.indirect_buf = rd.create_device_buffer(
            INDIRECT_WORDS,
            storage | vk::BufferUsageFlags::INDIRECT_BUFFER,
        );
        self.readback = rd.create_buffer(
            capacity as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuToCpu,
        );
        unsafe {
            rd.device
                .cmd_fill_buffer(cmd, self.epoch_buf.handle, 0, vk::WHOLE_SIZE, 0);
            rd.device
                .cmd_fill_buffer(cmd, self.indirect_buf.handle, 0, vk::WHOLE_SIZE, 0);
            rd.device
                .cmd_fill_buffer(cmd, self.world_buf.handle, 0, vk::WHOLE_SIZE, 0);
        }
        self.capacity = capacity;
        self.needs_full_rebuild = true;
    }

    /// Records this frame's GPU work into `cmd`. Call after the in-flight fence wait (the
    /// staging buffers and the table are rewritten in place). Returns whether the `World`
    /// table changed.
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
    ) -> bool {
        if self.node_count == 0 {
            self.clear_deltas();
            return false;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("transform.slang not compiled yet; instances wait");
                self.warned_module = true;
            }
            // Keep the deltas: the CPU mirror already holds them, and the first build after
            // the module lands is a full rebuild anyway.
            self.clear_deltas();
            self.needs_full_rebuild = true;
            return false;
        };

        if self.node_count > self.capacity {
            self.grow(rd, cmd);
        }

        if self.needs_full_rebuild {
            self.needs_full_rebuild = false;
            self.clear_deltas();
            self.upload_full(rd, cmd);
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            let params = PropagateParams {
                locals: self.locals_buf.address,
                parent: self.parent_buf.address,
                world: self.world_buf.address,
                frontier: self.frontier_buf.address,
                count: self.node_count,
                base: 0,
                full_rebuild: 1,
                node_count: self.node_count,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "propagate",
                &params,
                self.node_count,
                Some(offset_of!(PropagateParams, base)),
            );
            compute_to_compute_barrier(rd, cmd);
            log::debug!("GPU transforms: full rebuild of {} nodes", self.node_count);
            self.rebuilt_full = true;
            return true;
        }

        if !self.has_deltas() {
            return false;
        }
        self.frame_id += 1;
        self.rebuilt_full = false;

        // Staging.
        let u32_records: Vec<U32Record> = self
            .parent_records
            .iter()
            .chain(&self.first_child_records)
            .chain(&self.next_sibling_records)
            .copied()
            .collect();
        ensure_staging(rd, &mut self.staging_locals, self.local_records.len());
        ensure_staging(rd, &mut self.staging_u32, u32_records.len());
        if !self.local_records.is_empty() {
            rd.map_buffer(&mut self.staging_locals)
                .copy_from_slice(&self.local_records);
        }
        if !u32_records.is_empty() {
            rd.map_buffer(&mut self.staging_u32)
                .copy_from_slice(&u32_records);
        }

        // Frontier header reset; the scatters and the seed are mutually independent.
        unsafe {
            rd.device
                .cmd_fill_buffer(cmd, self.frontier_buf.handle, 0, FRONTIER_HEADER * 4, 0);
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );

        let n_locals = self.local_records.len() as u32;
        if n_locals > 0 {
            let params = ScatterLocalsParams {
                records: self.staging_locals.address,
                locals: self.locals_buf.address,
                count: n_locals,
                base: 0,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "scatter_locals",
                &params,
                n_locals,
                Some(offset_of!(ScatterLocalsParams, base)),
            );
        }
        let u32_stride = std::mem::size_of::<U32Record>() as u64;
        let mut offset = 0u32;
        for (records, dst) in [
            (&self.parent_records, self.parent_buf.address),
            (&self.first_child_records, self.first_child_buf.address),
            (&self.next_sibling_records, self.next_sibling_buf.address),
        ] {
            let count = records.len() as u32;
            if count > 0 {
                let params = ScatterU32Params {
                    records: self.staging_u32.address + offset as u64 * u32_stride,
                    dst,
                    count,
                    base: 0,
                };
                record_dispatch(
                    rd,
                    cmd,
                    module,
                    "scatter_u32",
                    &params,
                    count,
                    Some(offset_of!(ScatterU32Params, base)),
                );
            }
            offset += count;
        }

        // Frontier: seed with the changed locals, expand level by level, publish.
        let seed = FrontierSeedParams {
            records: self.staging_locals.address,
            frontier: self.frontier_buf.address,
            epoch: self.epoch_buf.address,
            count: n_locals,
            base: 0,
            frame_id: self.frame_id,
            node_count: self.node_count,
        };
        record_dispatch(
            rd,
            cmd,
            module,
            "frontier_seed",
            &seed,
            n_locals,
            Some(offset_of!(FrontierSeedParams, base)),
        );
        compute_to_compute_barrier(rd, cmd);

        let finalize = FrontierFinalizeParams {
            frontier: self.frontier_buf.address,
            indirect: self.indirect_buf.address,
        };
        let expand = FrontierExpandParams {
            first_child: self.first_child_buf.address,
            next_sibling: self.next_sibling_buf.address,
            frontier: self.frontier_buf.address,
            epoch: self.epoch_buf.address,
            frame_id: self.frame_id,
            node_count: self.node_count,
        };
        for level in 0..MAX_LEVELS {
            record_dispatch(rd, cmd, module, "frontier_finalize", &finalize, 1, None);
            compute_to_compute_barrier(rd, cmd);
            record_dispatch_indirect(
                rd,
                cmd,
                module,
                "frontier_expand",
                &expand,
                self.indirect_buf.handle,
                level as u64 * 12,
            );
            compute_to_compute_barrier(rd, cmd);
        }
        record_dispatch(rd, cmd, module, "frontier_finalize", &finalize, 1, None);
        compute_to_compute_barrier(rd, cmd);

        let params = PropagateParams {
            locals: self.locals_buf.address,
            parent: self.parent_buf.address,
            world: self.world_buf.address,
            frontier: self.frontier_buf.address,
            count: 0,
            base: 0,
            full_rebuild: 0,
            node_count: self.node_count,
        };
        record_dispatch_indirect(
            rd,
            cmd,
            module,
            "propagate",
            &params,
            self.indirect_buf.handle,
            CONSUMER_ARGS_OFFSET,
        );
        compute_to_compute_barrier(rd, cmd);

        self.clear_deltas();
        true
    }

    /// Records and submits this frame's transform work in a command buffer of its own, with
    /// the readback at its end, under the readback fence. Call after the in-flight fence wait,
    /// before anything that reads the `World` table is recorded; those command buffers are
    /// submitted later on the same queue and need [`world_barrier`] first. Returns whether the
    /// `World` table changed.
    pub fn submit(&mut self, rd: &RenderDevice, modules: &ComputeModules) -> bool {
        if self.readback_cmd == vk::CommandBuffer::null() {
            unsafe {
                self.readback_cmd = rd
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(rd.command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0];
                self.readback_total = rd.create_buffer(
                    1,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    MemoryLocation::GpuToCpu,
                );
                self.readback_fence = rd
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap();
            }
        }
        // Normally collected by `apply_readback` already; this only covers a skipped `First`.
        self.wait_readback(rd);
        let cmd = self.readback_cmd;
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
        let changed = self.record(rd, cmd, modules);
        if changed && let Some(module) = modules.get(&self.module) {
            let params = ReadbackParams {
                world: self.world_buf.address,
                frontier: self.frontier_buf.address,
                records: self.readback.address,
                total: self.readback_total.address,
                count: self.node_count,
                base: 0,
                full_rebuild: self.rebuilt_full as u32,
                node_count: self.node_count,
            };
            if self.rebuilt_full {
                record_dispatch(
                    rd,
                    cmd,
                    module,
                    "readback",
                    &params,
                    self.node_count,
                    Some(offset_of!(ReadbackParams, base)),
                );
            } else {
                record_dispatch_indirect(
                    rd,
                    cmd,
                    module,
                    "readback",
                    &params,
                    self.indirect_buf.handle,
                    CONSUMER_ARGS_OFFSET,
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
        }
        unsafe {
            rd.end_command_buffer(cmd).unwrap();
            let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
            let queue = rd.queue.lock().unwrap();
            rd.queue_submit(*queue, std::slice::from_ref(&submit), self.readback_fence)
                .unwrap_or_else(|e| {
                    crate::aftermath::note_device_lost(e);
                    panic!("transform submit failed: {e:?}");
                });
        }
        self.readback_pending = true;
        changed
    }

    fn wait_readback(&mut self, rd: &RenderDevice) {
        if !self.readback_pending {
            return;
        }
        unsafe {
            rd.wait_for_fences(std::slice::from_ref(&self.readback_fence), true, u64::MAX)
                .unwrap_or_else(|e| {
                    crate::aftermath::note_device_lost(e);
                    panic!("transform readback fence wait failed: {e:?}");
                });
            rd.reset_fences(std::slice::from_ref(&self.readback_fence))
                .unwrap();
        }
        self.readback_pending = false;
    }

    fn upload_full(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        let n = self.node_count as usize;
        self.ensure_node(self.node_count.saturating_sub(1));
        upload_slice(rd, cmd, &self.locals[..n], &self.locals_buf);
        upload_slice(rd, cmd, &self.parent[..n], &self.parent_buf);
        upload_slice(rd, cmd, &self.first_child[..n], &self.first_child_buf);
        upload_slice(rd, cmd, &self.next_sibling[..n], &self.next_sibling_buf);
    }
}

/// Copies `data` into `dst` through a throwaway host buffer (destroyed deferred).
pub(crate) fn upload_slice<T: Pod>(
    rd: &RenderDevice,
    cmd: vk::CommandBuffer,
    data: &[T],
    dst: &Buffer<T>,
) {
    if data.is_empty() {
        return;
    }
    let mut staging: Buffer<T> =
        rd.create_host_buffer(data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
    rd.map_buffer(&mut staging).copy_from_slice(data);
    rd.upload_buffer(cmd, &staging, dst);
    rd.destroyer.destroy_buffer(staging.handle);
}

/// Host-visible record buffer large enough for `len` entries (grown geometrically).
pub(crate) fn ensure_staging<T>(rd: &RenderDevice, buffer: &mut Buffer<T>, len: usize) {
    if len as u64 <= buffer.nr_elements {
        return;
    }
    rd.destroyer.destroy_buffer(buffer.handle);
    let capacity = (len as u64).max(256).next_power_of_two();
    *buffer = rd.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER);
}

/// Makes the transform submit's `World` writes visible to a command buffer submitted after it
/// on the same queue (a barrier's first scope covers everything earlier in submission order).
pub fn world_barrier(rd: &RenderDevice, cmd: vk::CommandBuffer) {
    memory_barrier(
        rd,
        cmd,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        vk::AccessFlags2::SHADER_READ,
    );
}

/// `First`: last frame's read-back world rows into `GlobalTransform`.
fn apply_readback(
    rd: Option<Res<RenderDevice>>,
    mut table: ResMut<GpuTransforms>,
    slots: Res<GpuNodeSlots>,
    mut globals: Query<&mut GlobalTransform>,
) {
    let Some(rd) = rd else {
        return;
    };
    if !table.readback_pending {
        return;
    }
    table.wait_readback(&rd);
    let table = &mut *table;
    let total = rd.map_buffer(&mut table.readback_total).as_slice_mut()[0] as usize;
    let mut view = rd.map_buffer(&mut table.readback);
    let records = view.as_slice_mut();
    for record in &records[..total.min(records.len())] {
        let Some(entity) = slots.entity(record.slot) else {
            continue;
        };
        if let Ok(mut global) = globals.get_mut(entity) {
            global.set_if_neq(GlobalTransform::from(record.world.affine()));
        }
    }
}

// ---- extraction -----------------------------------------------------------------------------

type ChangedNodes = Or<(Changed<Transform>, Changed<ChildOf>, Added<GpuNode>)>;

/// Mirrors this frame's changes into the table's delta lists. Each column is re-scattered only
/// when its own source changed: locals on move / first sight / reparent (the re-push seeds the
/// frontier, which recomposes the node and its descendants), parent on reparent / first sight,
/// child chains on `Changed<Children>` / first sight.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn extract_transforms(
    mut table: ResMut<GpuTransforms>,
    slots: Res<GpuNodeSlots>,
    changed: Query<
        (
            Ref<Transform>,
            Option<Ref<ChildOf>>,
            Ref<GpuNode>,
            Option<&Children>,
        ),
        ChangedNodes,
    >,
    nodes: Query<&GpuNode>,
    children_changed: Query<(Ref<GpuNode>, &Children), Changed<Children>>,
    mut children_removed: RemovedComponents<Children>,
    mut child_of_removed: RemovedComponents<ChildOf>,
    has_children: Query<(), With<Children>>,
    has_child_of: Query<(), With<ChildOf>>,
    with_transform: Query<(&Transform, &GpuNode)>,
) {
    let table = &mut *table;
    table.node_count = slots.count();

    for (transform, child_of, node, children) in changed.iter() {
        let first = node.is_added();
        let slot = node.0;
        let parent_changed = child_of.as_ref().is_some_and(Ref::is_changed);
        if first || transform.is_changed() || parent_changed {
            table.push_local(slot, NodeLocal::from_transform(&transform));
        }
        if first || parent_changed {
            let parent = child_of
                .as_ref()
                .and_then(|c| nodes.get(c.parent()).ok())
                .map_or(ROOT_PARENT, |n| n.0);
            table.push_parent(slot, parent);
        }
        if first {
            table.push_chain(slot, children, &nodes);
        }
    }

    // Child-list churn after first sight (first-sight slots just wrote theirs above; a second
    // record for one slot in a frame is a scatter race).
    for (node, children) in children_changed.iter() {
        if node.is_added() {
            continue;
        }
        table.push_chain(node.0, Some(children), &nodes);
    }
    for entity in children_removed.read() {
        if has_children.contains(entity) {
            continue;
        }
        if let Ok(node) = nodes.get(entity) {
            table.push_first_child(node.0, NO_NODE);
        }
    }
    // Orphaned (parent removed): now a root; re-push the local so the frontier recomposes it.
    for entity in child_of_removed.read() {
        if has_child_of.contains(entity) {
            continue;
        }
        if let Ok((transform, node)) = with_transform.get(entity) {
            table.push_parent(node.0, ROOT_PARENT);
            table.push_local(node.0, NodeLocal::from_transform(transform));
        }
    }
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut table: Mut<GpuTransforms>| {
        let rd = world.resource::<RenderDevice>();
        table.destroy_device(rd);
        rd.destroyer.destroy_buffer(table.staging_locals.handle);
        rd.destroyer.destroy_buffer(table.staging_u32.handle);
        rd.destroyer.destroy_buffer(table.readback_total.handle);
        table.wait_readback(rd);
        if table.readback_fence != vk::Fence::null() {
            unsafe { rd.destroy_fence(table.readback_fence, None) };
        }
    });
}

pub struct GpuTransformPlugin;

impl Plugin for GpuTransformPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuNodeSlots>();
        app.add_systems(
            PostUpdate,
            assign_gpu_nodes.before(TransformSystems::Propagate),
        );
        app.add_observer(free_gpu_node);

        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/transform.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "scatter_locals",
                "scatter_u32",
                "frontier_seed",
                "frontier_finalize",
                "frontier_expand",
                "propagate",
                "readback",
            ],
        ));

        app.insert_resource(GpuTransforms::new(module));
        app.add_systems(Last, extract_transforms.in_set(RenderSet::Extract));
        app.add_systems(First, apply_readback);
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
