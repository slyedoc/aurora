//! GPU skinning: skeletal animation for ray-traced meshes.
//!
//! An entity with `Mesh3d` + bevy's [`SkinnedMesh`] (joint entities + inverse bind poses) is
//! rendered through a per-instance deformed copy of its mesh: every frame the
//! `skin_vertices` kernel (skinning.slang) linear-blend-skins the rest-pose stream into a
//! mesh-local deformed stream, `pack_triangles` rebuilds the per-triangle shading records
//! from it, and the instance's own BLAS is refit (rebuilt every [`REBUILD_INTERVAL`] frames).
//! The instance's TLAS slot is then pointed at that BLAS, and its SBT hit record at the
//! deformed streams, through the TLAS's per-slot override.
//!
//! Joint worlds are never uploaded: joints are ordinary `Transform` entities with a slot in the
//! GPU transform table, and the kernel reads their rows straight from the `World` table. The
//! only per-instance CPU data is the joint -> node-slot palette and, per bind-pose asset, the
//! inverse bind matrices.
//!
//! Baked prefabs cannot serialize `SkinnedMesh` (its joints are entity ids): they carry a
//! [`SkinJointsByName`], resolved into the real component once the prefab's bone tree has
//! spawned (bind poses are derived from that tree, armature-local).
//!
//! Frame order: `GpuTransforms::record` -> [`Skins::record`] -> `TLAS::record` (the TLAS is
//! rebuilt every frame a skinned instance deformed). Previous-frame deformed positions stay
//! in the other half of the vertex ping-pong for the closest-hit's motion vectors.

use std::collections::HashMap;
use std::mem::offset_of;

use ash::vk;
use bevy::{
    asset::AssetEvent,
    ecs::{
        hierarchy::{ChildOf, Children},
        lifecycle::{Remove, RemovedComponents},
        message::MessageReader,
        observer::On,
    },
    math::DMat4,
    mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::{AccelerationStructure, Triangle, Vertex, allocate_acceleration_structure},
    compute::{
        ComputeModule, ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch,
    },
    gpu_transform::{GpuNode, GpuTransforms, NO_NODE},
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::{GpuInstance, InstanceOverride, TLAS, prepare_instances},
    vk_utils,
    vulkan_asset::{VulkanAssets, poll_for_asset},
};

/// Frames between full BLAS rebuilds of a skinned instance; refits in between. A refit keeps
/// the tree topology of the last build, so trace quality drifts as the pose moves away from it.
const REBUILD_INTERVAL: u32 = 16;

// ---- baked prefabs: joints by name ----------------------------------------------------------

/// Joint names of a skinned prefab part, in palette order. `resolve_skin_joints` turns this
/// into a [`SkinnedMesh`] once every name resolves in the prefab's subtree; the inverse bind
/// poses are the inverses of the joints' spawned (bind) transforms relative to the armature
/// (the first joint's parent), so the armature transform places and animates the character.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default)]
pub struct SkinJointsByName(pub Vec<String>);

fn resolve_skin_joints(
    mut commands: Commands,
    pending: Query<(Entity, &SkinJointsByName)>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
    names: Query<&Name>,
    transforms: Query<&Transform>,
    mut bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>,
) {
    for (entity, by_name) in &pending {
        if by_name.0.is_empty() {
            commands.entity(entity).remove::<SkinJointsByName>();
            continue;
        }
        let mut root = entity;
        while let Ok(parent) = parents.get(root) {
            root = parent.parent();
        }
        let mut by_name_map: HashMap<&str, Entity> = HashMap::new();
        let mut stack = vec![root];
        while let Some(e) = stack.pop() {
            if let Ok(name) = names.get(e) {
                by_name_map.entry(name.as_str()).or_insert(e);
            }
            if let Ok(c) = children.get(e) {
                stack.extend(c.iter());
            }
        }
        let Some(joints) = by_name
            .0
            .iter()
            .map(|n| by_name_map.get(n.as_str()).copied())
            .collect::<Option<Vec<Entity>>>()
        else {
            // The prefab is still streaming in; retry next frame.
            continue;
        };
        let armature = parents.get(joints[0]).ok().map(|p| p.parent());
        let inverse: Vec<Mat4> = joints
            .iter()
            .map(|&joint| {
                let mut chain = vec![joint];
                let mut e = joint;
                while let Ok(parent) = parents.get(e) {
                    if Some(parent.parent()) == armature {
                        break;
                    }
                    e = parent.parent();
                    chain.push(e);
                }
                let mut bind = DMat4::IDENTITY;
                for &e in chain.iter().rev() {
                    if let Ok(t) = transforms.get(e) {
                        bind *= t.to_matrix().as_dmat4();
                    }
                }
                bind.inverse().as_mat4()
            })
            .collect();
        let inverse_bindposes = bindposes.add(SkinnedMeshInverseBindposes::from(inverse));
        commands
            .entity(entity)
            .insert(SkinnedMesh {
                inverse_bindposes,
                joints,
            })
            .remove::<SkinJointsByName>();
    }
}

// ---- inverse bind poses on the device -------------------------------------------------------

/// Device copies of every loaded [`SkinnedMeshInverseBindposes`]: 16 floats per joint
/// (column-major, as glam lays out `Mat4`).
#[derive(Resource, Default)]
pub struct BindposeBuffers {
    buffers: HashMap<AssetId<SkinnedMeshInverseBindposes>, (Buffer<f32>, u32)>,
}

fn upload_bindposes(
    render_device: Res<RenderDevice>,
    mut events: MessageReader<AssetEvent<SkinnedMeshInverseBindposes>>,
    assets: Res<Assets<SkinnedMeshInverseBindposes>>,
    mut buffers: ResMut<BindposeBuffers>,
) {
    for event in events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                let Some(poses) = assets.get(*id) else {
                    continue;
                };
                let floats: Vec<f32> = poses.iter().flat_map(|m| m.to_cols_array()).collect();
                if floats.is_empty() {
                    continue;
                }
                let mut host: Buffer<f32> = render_device
                    .create_host_buffer(floats.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
                render_device.map_buffer(&mut host).copy_from_slice(&floats);
                let device: Buffer<f32> = render_device.create_device_buffer(
                    floats.len() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                );
                render_device.run_transfer_commands(|cmd| {
                    render_device.upload_buffer(cmd, &host, &device);
                });
                render_device.destroyer.destroy_buffer(host.handle);
                if let Some((old, _)) = buffers.buffers.insert(*id, (device, poses.len() as u32)) {
                    render_device.destroyer.destroy_buffer(old.handle);
                }
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                if let Some((old, _)) = buffers.buffers.remove(id) {
                    render_device.destroyer.destroy_buffer(old.handle);
                }
            }
        }
    }
}

// ---- skinned instances ----------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinParams {
    rest: u64,
    skin: u64,
    world: u64,
    joint_nodes: u64,
    inverse_bind: u64,
    dst: u64,
    count: u32,
    base: u32,
    joint_count: u32,
    mesh_node: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    vertices: u64,
    indices: u64,
    triangles: u64,
    count: u32,
    base: u32,
}

#[derive(Clone, Debug, PartialEq)]
struct SkinSource {
    mesh: AssetId<Mesh>,
    bindposes: AssetId<SkinnedMeshInverseBindposes>,
    joints: Vec<Entity>,
    node: u32,
}

/// The device half of a skinned instance, created once its mesh BLAS (with a skin stream) and
/// bind poses are on the GPU.
struct SkinnedGpu {
    /// The mesh BLAS this was built against; a different handle means the asset was reloaded
    /// and the rest streams below are gone.
    mesh_vertex_buffer: vk::Buffer,
    vertex_count: u32,
    triangle_count: u32,
    rest: Buffer<Vertex>,
    skin: u64,
    index_buffer: Buffer<u32>,
    geometry_to_index: u64,
    geometry_to_triangle: u64,
    /// Deformed streams, ping-ponged: this frame's and last frame's (motion vectors).
    vertices: [Buffer<Vertex>; 2],
    triangles: Buffer<Triangle>,
    /// Host-visible palette: transform-table node per joint.
    joint_nodes: Buffer<u32>,
    joint_count: u32,
    inverse_bind: u64,
    blas: AccelerationStructure,
    scratch: Buffer<u8>,
    scratch_alignment: u64,
    /// Builds recorded so far; 0 = the structure has never been built.
    builds: u32,
    hit_offset: u32,
}

impl SkinnedGpu {
    fn destroy(&self, rd: &RenderDevice) {
        self.blas.destroy(rd);
        for b in [
            self.rest.handle,
            self.index_buffer.handle,
            self.vertices[0].handle,
            self.vertices[1].handle,
            self.triangles.handle,
            self.joint_nodes.handle,
            self.scratch.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
    }

    fn geometry(&self) -> vk::AccelerationStructureGeometryKHR<'static> {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.vertices[0].address,
                    })
                    .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                    .max_vertex(self.vertex_count)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.index_buffer.address,
                    })
                    .transform_data(vk::DeviceOrHostAddressConstKHR { device_address: 0 }),
            })
    }
}

const BLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::from_raw(
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD.as_raw()
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw(),
    );

struct SkinnedInstance {
    source: SkinSource,
    /// Transform-table node per joint (`NO_NODE` while a joint has no slot yet).
    joint_nodes: Vec<u32>,
    joints_dirty: bool,
    gpu: Option<SkinnedGpu>,
}

/// What the SBT writes for a skinned instance's hit record (sbt.rs).
pub struct SkinnedHitRecord {
    pub hit_offset: u32,
    pub vertex_buffer: u64,
    pub prev_vertex_buffer: u64,
    pub triangle_buffer: u64,
    pub index_buffer: u64,
    pub geometry_to_index: u64,
    pub geometry_to_triangle: u64,
}

#[derive(Resource)]
pub struct Skins {
    module: Handle<ComputeModule>,
    instances: HashMap<u32, SkinnedInstance>,
    /// Slots whose skinned component or entity went away: drop everything.
    removed: Vec<u32>,
    /// Slots whose source changed: drop the device state, keep the record.
    rebuild: Vec<u32>,
    /// Advances once per frame in `prepare`; its parity picks the deformed stream.
    frame: u32,
    warned_module: bool,
}

impl Skins {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            instances: HashMap::new(),
            removed: Vec::new(),
            rebuild: Vec::new(),
            frame: 0,
            warned_module: false,
        }
    }

    fn current(&self) -> usize {
        (self.frame & 1) as usize
    }

    /// Hit records for every skinned instance with device state, for this frame's parity.
    pub fn hit_records(&self) -> impl Iterator<Item = SkinnedHitRecord> + '_ {
        let cur = self.current();
        self.instances.values().filter_map(move |i| {
            let gpu = i.gpu.as_ref()?;
            Some(SkinnedHitRecord {
                hit_offset: gpu.hit_offset,
                vertex_buffer: gpu.vertices[cur].address,
                prev_vertex_buffer: gpu.vertices[cur ^ 1].address,
                triangle_buffer: gpu.triangles.address,
                index_buffer: gpu.index_buffer.address,
                geometry_to_index: gpu.geometry_to_index,
                geometry_to_triangle: gpu.geometry_to_triangle,
            })
        })
    }

    /// Records this frame's deforms and BLAS refits. Call after `GpuTransforms::record` and
    /// before `TLAS::record`. Returns whether any BLAS changed (the TLAS must rebuild).
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        transforms: &GpuTransforms,
    ) -> bool {
        if self.instances.values().all(|i| i.gpu.is_none()) {
            return false;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("skinning.slang not compiled yet; skinned meshes wait");
                self.warned_module = true;
            }
            return false;
        };
        let (world, node_count) = transforms.world();
        if world == 0 {
            return false;
        }
        let cur = self.current();
        let prev = cur ^ 1;

        // Palettes (host-visible, rewritten after the fence wait) and first-frame history.
        let mut transfers = false;
        for inst in self.instances.values_mut() {
            let Some(gpu) = inst.gpu.as_mut() else {
                continue;
            };
            if inst.joints_dirty {
                rd.map_buffer(&mut gpu.joint_nodes)
                    .copy_from_slice(&inst.joint_nodes);
                inst.joints_dirty = false;
            }
            if gpu.builds == 0 {
                // Last frame's stream starts as the rest pose: no fake motion on the first frame.
                let copy = vk::BufferCopy::default()
                    .size(gpu.vertex_count as u64 * std::mem::size_of::<Vertex>() as u64);
                unsafe {
                    rd.device.cmd_copy_buffer(
                        cmd,
                        gpu.rest.handle,
                        gpu.vertices[prev].handle,
                        std::slice::from_ref(&copy),
                    );
                }
                transfers = true;
            }
        }
        if transfers {
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ,
            );
        }

        // Deform.
        let mut any = false;
        for inst in self.instances.values() {
            let Some(gpu) = inst.gpu.as_ref() else {
                continue;
            };
            if inst.source.node >= node_count {
                continue;
            }
            let params = SkinParams {
                rest: gpu.rest.address,
                skin: gpu.skin,
                world,
                joint_nodes: gpu.joint_nodes.address,
                inverse_bind: gpu.inverse_bind,
                dst: gpu.vertices[cur].address,
                count: gpu.vertex_count,
                base: 0,
                joint_count: gpu.joint_count,
                mesh_node: inst.source.node,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "skin_vertices",
                &params,
                gpu.vertex_count,
                Some(offset_of!(SkinParams, base)),
            );
            any = true;
        }
        if !any {
            return false;
        }
        compute_to_compute_barrier(rd, cmd);
        for inst in self.instances.values() {
            let Some(gpu) = inst.gpu.as_ref() else {
                continue;
            };
            if inst.source.node >= node_count {
                continue;
            }
            let params = PackParams {
                vertices: gpu.vertices[cur].address,
                indices: gpu.index_buffer.address,
                triangles: gpu.triangles.address,
                count: gpu.triangle_count,
                base: 0,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "pack_triangles",
                &params,
                gpu.triangle_count,
                Some(offset_of!(PackParams, base)),
            );
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );

        // BLAS build / refit, one submission-free batch inside the frame.
        let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> = Vec::new();
        let mut ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = Vec::new();
        let mut modes: Vec<(vk::AccelerationStructureKHR, bool, u64)> = Vec::new();
        for inst in self.instances.values_mut() {
            let Some(gpu) = inst.gpu.as_mut() else {
                continue;
            };
            if inst.source.node >= node_count {
                continue;
            }
            let mut geometry = gpu.geometry();
            geometry.geometry.triangles.vertex_data = vk::DeviceOrHostAddressConstKHR {
                device_address: gpu.vertices[cur].address,
            };
            let update = gpu.builds > 0 && gpu.builds % REBUILD_INTERVAL != 0;
            gpu.builds = gpu.builds.wrapping_add(1).max(1);
            geometries.push(geometry);
            ranges.push(
                vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(gpu.triangle_count),
            );
            modes.push((
                gpu.blas.handle,
                update,
                vk_utils::aligned_size(gpu.scratch.address, gpu.scratch_alignment),
            ));
        }
        let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = geometries
            .iter()
            .zip(&modes)
            .map(|(geometry, (handle, update, scratch))| {
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .flags(BLAS_FLAGS)
                    .dst_acceleration_structure(*handle)
                    .geometries(std::slice::from_ref(geometry))
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: *scratch,
                    });
                if *update {
                    info.mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                        .src_acceleration_structure(*handle)
                } else {
                    info.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                }
            })
            .collect();
        let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
            ranges.iter().map(std::slice::from_ref).collect();
        unsafe {
            rd.ext_acc_struct
                .cmd_build_acceleration_structures(cmd, &infos, &range_refs);
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
        true
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for inst in self.instances.values_mut() {
            if let Some(gpu) = inst.gpu.take() {
                gpu.destroy(rd);
            }
        }
        self.instances.clear();
    }
}

// ---- extraction / preparation ---------------------------------------------------------------

type ChangedSkins = Or<(
    Added<GpuInstance>,
    Changed<GpuNode>,
    Changed<Mesh3d>,
    Changed<SkinnedMesh>,
)>;

#[allow(clippy::type_complexity)]
fn extract_skins(
    mut skins: ResMut<Skins>,
    changed: Query<(&GpuInstance, &GpuNode, &Mesh3d, &SkinnedMesh), ChangedSkins>,
    mut removed_components: RemovedComponents<SkinnedMesh>,
    instances: Query<&GpuInstance>,
    nodes: Query<&GpuNode>,
) {
    for entity in removed_components.read() {
        if let Ok(instance) = instances.get(entity) {
            skins.removed.push(instance.0);
        }
    }
    for (instance, node, mesh, skin) in changed.iter() {
        let source = SkinSource {
            mesh: mesh.id(),
            bindposes: skin.inverse_bindposes.id(),
            joints: skin.joints.clone(),
            node: node.0,
        };
        match skins.instances.get_mut(&instance.0) {
            Some(existing) if existing.source == source => {}
            Some(existing) => {
                let rebuild = existing.source.mesh != source.mesh
                    || existing.source.bindposes != source.bindposes
                    || existing.source.joints.len() != source.joints.len();
                existing.source = source;
                if rebuild {
                    skins.rebuild.push(instance.0);
                }
            }
            None => {
                skins.instances.insert(
                    instance.0,
                    SkinnedInstance {
                        source,
                        joint_nodes: Vec::new(),
                        joints_dirty: true,
                        gpu: None,
                    },
                );
            }
        }
    }
    // Joint palettes: joints get their transform-table slots the frame they spawn, and can
    // be re-parented / respawned by gameplay, so re-resolve every frame (a few hundred
    // lookups per skinned instance).
    for inst in skins.instances.values_mut() {
        let palette: Vec<u32> = inst
            .source
            .joints
            .iter()
            .map(|&j| nodes.get(j).map_or(NO_NODE, |n| n.0))
            .collect();
        if palette != inst.joint_nodes {
            inst.joint_nodes = palette;
            inst.joints_dirty = true;
        }
    }
}

fn on_instance_removed(
    remove: On<Remove<GpuInstance>>,
    instances: Query<&GpuInstance>,
    mut skins: ResMut<Skins>,
) {
    if let Ok(instance) = instances.get(remove.entity) {
        skins.removed.push(instance.0);
    }
}

pub fn prepare_skins(
    render_device: Res<RenderDevice>,
    mut skins: ResMut<Skins>,
    mut tlas: ResMut<TLAS>,
    meshes: Res<VulkanAssets<Mesh>>,
    bindposes: Res<BindposeBuffers>,
) {
    let skins = &mut *skins;
    skins.frame = skins.frame.wrapping_add(1);

    // Removals and rebuilds: the slot goes back to the plain mesh BLAS meanwhile.
    for slot in skins.removed.drain(..) {
        tlas.set_override(slot, None);
        if let Some(inst) = skins.instances.remove(&slot)
            && let Some(gpu) = inst.gpu
        {
            gpu.destroy(&render_device);
            tlas.release_slot_hit_offset(slot);
        }
    }
    for slot in skins.rebuild.drain(..) {
        tlas.set_override(slot, None);
        if let Some(inst) = skins.instances.get_mut(&slot)
            && let Some(gpu) = inst.gpu.take()
        {
            gpu.destroy(&render_device);
            tlas.release_slot_hit_offset(slot);
        }
    }
    let live: Vec<u32> = skins.instances.keys().copied().collect();
    for slot in live {
        let inst = skins.instances.get_mut(&slot).unwrap();
        let mesh_blas = meshes.get_by_id(inst.source.mesh);
        // A reloaded mesh asset replaced the rest streams under us.
        if let Some(gpu) = &inst.gpu {
            let stale = mesh_blas.is_none_or(|b| b.vertex_buffer.handle != gpu.mesh_vertex_buffer);
            if stale {
                let gpu = inst.gpu.take().unwrap();
                gpu.destroy(&render_device);
                tlas.release_slot_hit_offset(slot);
                tlas.set_override(slot, None);
            }
        }
        if inst.gpu.is_none() {
            let Some(blas) = mesh_blas else { continue };
            let Some(skin) = &blas.skin_buffer else {
                // Not a skinned mesh: rendered rigid off the shared BLAS.
                continue;
            };
            let Some((inverse_bind, bind_count)) = bindposes.buffers.get(&inst.source.bindposes)
            else {
                continue;
            };
            let joint_count = (inst.source.joints.len() as u32).min(*bind_count);
            if joint_count == 0 {
                continue;
            }
            let vertex_count = blas.vertex_buffer.nr_elements as u32;
            let index_count = blas.index_buffer.nr_elements as u32;
            if vertex_count == 0 || index_count < 3 {
                continue;
            }
            // Own copies of the rest streams: the mesh asset can be reloaded or dropped while
            // this instance keeps animating.
            let rest = render_device.create_device_buffer::<Vertex>(
                vertex_count as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            );
            let index_buffer = render_device.create_device_buffer::<u32>(
                index_count as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            );
            render_device.run_transfer_commands(|cmd| unsafe {
                let vcopy = vk::BufferCopy::default()
                    .size(vertex_count as u64 * std::mem::size_of::<Vertex>() as u64);
                render_device.device.cmd_copy_buffer(
                    cmd,
                    blas.vertex_buffer.handle,
                    rest.handle,
                    std::slice::from_ref(&vcopy),
                );
                let icopy = vk::BufferCopy::default().size(index_count as u64 * 4);
                render_device.device.cmd_copy_buffer(
                    cmd,
                    blas.index_buffer.handle,
                    index_buffer.handle,
                    std::slice::from_ref(&icopy),
                );
            });
            let stream_usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
            let vertices = [
                render_device.create_device_buffer::<Vertex>(vertex_count as u64, stream_usage),
                render_device.create_device_buffer::<Vertex>(vertex_count as u64, stream_usage),
            ];
            let triangle_count = index_count / 3;
            let triangles = render_device.create_device_buffer::<Triangle>(
                triangle_count as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            let joint_nodes = render_device.create_host_buffer::<u32>(
                inst.source.joints.len().max(1) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );

            let mut gpu = SkinnedGpu {
                mesh_vertex_buffer: blas.vertex_buffer.handle,
                vertex_count,
                triangle_count,
                rest,
                skin: skin.address,
                index_buffer,
                geometry_to_index: blas.geometry_to_index.address,
                geometry_to_triangle: blas.geometry_to_triangle.address,
                vertices,
                triangles,
                joint_nodes,
                joint_count,
                inverse_bind: inverse_bind.address,
                blas: AccelerationStructure::default(),
                scratch: Buffer::default(),
                scratch_alignment: vk_utils::get_acceleration_structure_properties(&render_device)
                    .min_acceleration_structure_scratch_offset_alignment
                    as u64,
                builds: 0,
                hit_offset: tlas.slot_hit_offset(slot),
            };
            let geometry = gpu.geometry();
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                render_device
                    .ext_acc_struct
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                            .flags(BLAS_FLAGS)
                            .geometries(std::slice::from_ref(&geometry)),
                        &[triangle_count],
                        &mut sizes,
                    );
            }
            gpu.blas = allocate_acceleration_structure(
                &render_device,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                &sizes,
            );
            gpu.scratch = render_device.create_device_buffer(
                sizes.build_scratch_size.max(sizes.update_scratch_size) + gpu.scratch_alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            inst.joints_dirty = true;
            inst.gpu = Some(gpu);
            log::info!(
                "skinning: slot {slot} -> {vertex_count} vertices, {triangle_count} triangles, {joint_count} joints"
            );
        }
        // Point the TLAS slot at the deformed BLAS once it has been built (the first build
        // lands in this frame's `record`; the override takes effect from the next).
        if let Some(gpu) = &inst.gpu
            && gpu.builds > 0
        {
            tlas.set_override(
                slot,
                Some(InstanceOverride {
                    blas: gpu.blas.address,
                    hit_offset: Some(gpu.hit_offset),
                }),
            );
        }
    }
}

fn cleanup_skins(world: &mut World) {
    world.resource_scope(|world, mut skins: Mut<Skins>| {
        let rd = world.resource::<RenderDevice>();
        skins.destroy(rd);
    });
    world.resource_scope(|world, mut buffers: Mut<BindposeBuffers>| {
        let rd = world.resource::<RenderDevice>();
        for (_, (buffer, _)) in buffers.buffers.drain() {
            rd.destroyer.destroy_buffer(buffer.handle);
        }
    });
}

pub struct SkinningPlugin;

impl Plugin for SkinningPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<SkinnedMeshInverseBindposes>();
        app.register_type::<SkinJointsByName>();
        app.init_resource::<BindposeBuffers>();
        app.add_systems(PreUpdate, resolve_skin_joints);
        app.add_observer(on_instance_removed);

        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/skinning.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &["skin_vertices", "pack_triangles"],
        ));
        app.insert_resource(Skins::new(module));
        app.add_systems(
            Last,
            (
                (upload_bindposes, extract_skins).in_set(RenderSet::Extract),
                prepare_skins
                    .in_set(RenderSet::Prepare)
                    .after(poll_for_asset::<Mesh>)
                    .before(prepare_instances),
            ),
        );
        app.add_systems(TeardownSchedule, cleanup_skins.before(on_shutdown));
    }
}
