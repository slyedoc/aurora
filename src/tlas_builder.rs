//! TLAS instances, GPU-resident.
//!
//! Every ray-traced entity (`Mesh3d`, `GltfModelHandle`, `Sphere`) owns a [`GpuInstance`] slot
//! in a device-local `VkAccelerationStructureInstanceKHR` array — the buffer the TLAS is built
//! from directly. The CPU touches a slot only when its *static* half changes (BLAS arrived,
//! material resolved, visibility flipped, spawned, despawned): it scatters one
//! [`InstanceRecord`]. Transforms never cross the bus per frame: the `gather_instances` kernel
//! copies each slot's row from the GPU transform table (`gpu_transform.rs`). Free slots keep a
//! zero BLAS reference, which Vulkan treats as an inactive instance, so the slot layout is
//! stable and hidden entities (mask 0) stay in place.
//!
//! One frame in flight: the TLAS is rebuilt in place inside the frame command buffer, after
//! the fence wait, and only on frames where an instance or a transform changed.

use std::collections::{HashMap, HashSet};

use ash::vk;
use bevy::{
    asset::UntypedAssetId,
    camera::visibility::InheritedVisibility,
    ecs::{lifecycle::Remove, observer::On, query::Has},
    prelude::*,
    render::{ExtractSchedule, RenderApp},
};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::{AccelerationStructure, RTXMaterial},
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    extract::Extract,
    gltf_mesh::{GltfModel, GltfModelHandle},
    gpu_transform::{GpuNode, GpuTransforms, ensure_staging, upload_slice},
    ray_render_plugin::{Render, RenderSet, TeardownSchedule},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sphere::{Sphere, SphereBLAS},
    vk_utils,
    vulkan_asset::{VulkanAssets, poll_for_asset},
};

/// `VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR`.
const INSTANCE_FLAGS: u32 = 0b1;

/// The static half of an instance slot (must match instances.slang). 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct InstanceRecord {
    pub slot: u32,
    pub custom_and_mask: u32,
    pub sbt_and_flags: u32,
    pub node: u32,
    pub blas: u64,
    pub pad: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScatterInstancesParams {
    records: u64,
    instances: u64,
    instance_node: u64,
    count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GatherInstancesParams {
    instance_node: u64,
    world: u64,
    instances: u64,
    count: u32,
    base: u32,
    node_count: u32,
    pad: u32,
}

// ---- main world: slots ----------------------------------------------------------------------

/// This entity's slot in the GPU instance table.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuInstance(pub u32);

#[derive(Resource, Default)]
pub struct GpuInstanceSlots {
    free: Vec<u32>,
    next: u32,
    /// Slots freed since the last extraction; the render side clears them.
    freed: Vec<u32>,
}

fn assign_gpu_instances(
    mut commands: Commands,
    unslotted: Query<
        Entity,
        (
            Or<(With<Mesh3d>, With<GltfModelHandle>, With<Sphere>)>,
            With<Transform>,
            Without<GpuInstance>,
        ),
    >,
    mut slots: ResMut<GpuInstanceSlots>,
) {
    for entity in &unslotted {
        let slot = slots.free.pop().unwrap_or_else(|| {
            let s = slots.next;
            slots.next += 1;
            s
        });
        commands.entity(entity).insert(GpuInstance(slot));
    }
}

fn free_gpu_instance(
    remove: On<Remove<GpuInstance>>,
    instances: Query<&GpuInstance>,
    mut slots: ResMut<GpuInstanceSlots>,
) {
    if let Ok(instance) = instances.get(remove.entity) {
        slots.free.push(instance.0);
        slots.freed.push(instance.0);
    }
}

fn clear_freed(mut slots: ResMut<GpuInstanceSlots>) {
    slots.freed.clear();
}

// ---- render world ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Geometry {
    Mesh(AssetId<Mesh>),
    Gltf(AssetId<GltfModel>),
    Sphere,
}

#[derive(Clone, Debug, PartialEq)]
struct InstanceSource {
    geometry: Geometry,
    material: Option<AssetId<StandardMaterial>>,
    node: u32,
    mask: u8,
}

/// Material records for all instances, packed into one device buffer; an instance's
/// `custom_index` is its range's offset. Ranges are first-fit recycled.
#[derive(Default)]
struct MaterialArena {
    data: Vec<RTXMaterial>,
    ranges: Vec<Option<(u32, u32)>>,
    free: Vec<(u32, u32)>,
    dirty: Vec<(u32, u32)>,
    device: Buffer<RTXMaterial>,
    capacity: u32,
    full_upload: bool,
}

impl MaterialArena {
    fn set(&mut self, slot: u32, materials: &[RTXMaterial]) -> u32 {
        let slot = slot as usize;
        if self.ranges.len() <= slot {
            self.ranges.resize(slot + 1, None);
        }
        let len = materials.len() as u32;
        let offset = match self.ranges[slot] {
            Some((off, l)) if l == len => off,
            existing => {
                if let Some(range) = existing {
                    self.free.push(range);
                }
                let off = self.alloc(len);
                self.ranges[slot] = Some((off, len));
                off
            }
        };
        let end = (offset + len) as usize;
        if self.data.len() < end {
            self.data.resize(end, RTXMaterial::default());
        }
        self.data[offset as usize..end].copy_from_slice(materials);
        self.dirty.push((offset, len));
        offset
    }

    fn clear(&mut self, slot: u32) {
        if let Some(range) = self.ranges.get_mut(slot as usize).and_then(Option::take) {
            self.free.push(range);
        }
    }

    fn alloc(&mut self, len: u32) -> u32 {
        if let Some(i) = self.free.iter().position(|(_, l)| *l >= len) {
            let (off, l) = self.free.swap_remove(i);
            if l > len {
                self.free.push((off + len, l - len));
            }
            return off;
        }
        let off = self.data.len() as u32;
        self.data
            .resize(self.data.len() + len as usize, RTXMaterial::default());
        off
    }

    /// Returns whether anything was copied (the caller adds the transfer barrier).
    fn upload(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) -> bool {
        if self.data.len() as u32 > self.capacity {
            self.capacity = (self.data.len() as u32).max(1024).next_power_of_two();
            rd.destroyer.destroy_buffer(self.device.handle);
            self.device = rd.create_device_buffer(
                self.capacity as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            self.full_upload = true;
        }
        if self.full_upload {
            self.full_upload = false;
            self.dirty.clear();
            upload_slice(rd, cmd, &self.data, &self.device);
            return !self.data.is_empty();
        }
        if self.dirty.is_empty() {
            return false;
        }
        let total: u32 = self.dirty.iter().map(|(_, l)| l).sum();
        let mut staging: Buffer<RTXMaterial> =
            rd.create_host_buffer(total as u64, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut regions = Vec::with_capacity(self.dirty.len());
        {
            let mut view = rd.map_buffer(&mut staging);
            let dst = view.as_slice_mut();
            let mut cursor = 0usize;
            let stride = std::mem::size_of::<RTXMaterial>() as u64;
            for (off, len) in self.dirty.drain(..) {
                let n = len as usize;
                dst[cursor..cursor + n]
                    .copy_from_slice(&self.data[off as usize..off as usize + n]);
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset(cursor as u64 * stride)
                        .dst_offset(off as u64 * stride)
                        .size(len as u64 * stride),
                );
                cursor += n;
            }
        }
        unsafe {
            rd.device
                .cmd_copy_buffer(cmd, staging.handle, self.device.handle, &regions);
        }
        rd.destroyer.destroy_buffer(staging.handle);
        true
    }
}

#[derive(Resource)]
pub struct TLAS {
    module: Handle<ComputeModule>,
    pub acceleration_structure: AccelerationStructure,
    scratch_buffer: Buffer<u8>,
    handle_size: u64,
    scratch_alignment: u64,
    /// SBT hit-group record per mesh asset, stable for the asset's lifetime (0 = spheres).
    pub mesh_to_hit_offset: HashMap<UntypedAssetId, u32>,
    next_hit_offset: u32,
    // Slot state.
    sources: Vec<Option<InstanceSource>>,
    mirror: Vec<InstanceRecord>,
    count: u32,
    dirty: Vec<u32>,
    pending: HashSet<u32>,
    records: Vec<InstanceRecord>,
    materials: MaterialArena,
    // Device.
    capacity: u32,
    instances_buf: Buffer<u8>,
    instance_node_buf: Buffer<u32>,
    staging: Buffer<InstanceRecord>,
    rebuild: bool,
    warned_module: bool,
}

impl TLAS {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            acceleration_structure: AccelerationStructure::default(),
            scratch_buffer: Buffer::default(),
            handle_size: 0,
            scratch_alignment: 0,
            mesh_to_hit_offset: HashMap::new(),
            next_hit_offset: 1,
            sources: Vec::new(),
            mirror: Vec::new(),
            count: 0,
            dirty: Vec::new(),
            pending: HashSet::new(),
            records: Vec::new(),
            materials: MaterialArena::default(),
            capacity: 0,
            instances_buf: Buffer::default(),
            instance_node_buf: Buffer::default(),
            staging: Buffer::default(),
            rebuild: false,
            warned_module: false,
        }
    }

    /// Device address of the packed material buffer (`custom_index` indexes it).
    pub fn material_address(&self) -> u64 {
        self.materials.device.address
    }

    /// Number of SBT hit-group records the instances may reference.
    pub fn hit_offset_count(&self) -> u32 {
        self.next_hit_offset
    }

    fn hit_offset(&mut self, id: UntypedAssetId) -> u32 {
        *self.mesh_to_hit_offset.entry(id).or_insert_with(|| {
            let o = self.next_hit_offset;
            self.next_hit_offset += 1;
            o
        })
    }

    fn set_source(&mut self, slot: u32, source: Option<InstanceSource>) {
        let i = slot as usize;
        if self.sources.len() <= i {
            self.sources.resize(i + 1, None);
            self.mirror.resize(i + 1, InstanceRecord::default());
        }
        if self.sources[i] != source {
            self.sources[i] = source;
            self.dirty.push(slot);
        }
    }

    fn grow(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        self.capacity = self.count.max(1024).next_power_of_two();
        log::debug!("GPU instance table: {} slots", self.capacity);
        rd.destroyer.destroy_buffer(self.instances_buf.handle);
        rd.destroyer.destroy_buffer(self.instance_node_buf.handle);
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        self.instances_buf = rd.create_device_buffer(
            self.capacity as u64 * 64,
            storage | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        );
        self.instance_node_buf = rd.create_device_buffer(self.capacity as u64, storage);
        unsafe {
            rd.device
                .cmd_fill_buffer(cmd, self.instances_buf.handle, 0, vk::WHOLE_SIZE, 0);
            rd.device.cmd_fill_buffer(
                cmd,
                self.instance_node_buf.handle,
                0,
                vk::WHOLE_SIZE,
                u32::MAX,
            );
        }
        // Everything the table held has to be scattered again.
        self.records = self.mirror[..self.count as usize]
            .iter()
            .enumerate()
            .map(|(slot, r)| InstanceRecord {
                slot: slot as u32,
                ..*r
            })
            .collect();
        self.rebuild = true;
    }

    /// Records this frame's instance updates and, when anything changed, the TLAS build. Call
    /// after the transform table's `record` (and the in-flight fence wait).
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        transforms: &GpuTransforms,
        world_changed: bool,
    ) {
        if self.count == 0 {
            return;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("instances.slang not compiled yet; TLAS waits");
                self.warned_module = true;
            }
            return;
        };
        let (world, node_count) = transforms.world();
        if world == 0 {
            return;
        }

        let mut transfers = false;
        if self.count > self.capacity {
            self.grow(rd, cmd);
            transfers = true;
        }
        transfers |= self.materials.upload(rd, cmd);

        if !self.records.is_empty() {
            ensure_staging(rd, &mut self.staging, self.records.len());
            rd.map_buffer(&mut self.staging)
                .copy_from_slice(&self.records);
            if transfers {
                memory_barrier(
                    rd,
                    cmd,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                );
            }
            let n = self.records.len() as u32;
            let params = ScatterInstancesParams {
                records: self.staging.address,
                instances: self.instances_buf.address,
                instance_node: self.instance_node_buf.address,
                count: n,
                base: 0,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "scatter_instances",
                &params,
                n,
                Some(std::mem::offset_of!(ScatterInstancesParams, base)),
            );
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            self.records.clear();
            self.rebuild = true;
        }

        if !(self.rebuild || world_changed) {
            return;
        }
        self.rebuild = false;

        let params = GatherInstancesParams {
            instance_node: self.instance_node_buf.address,
            world,
            instances: self.instances_buf.address,
            count: self.count,
            base: 0,
            node_count,
            pad: 0,
        };
        record_dispatch(
            rd,
            cmd,
            module,
            "gather_instances",
            &params,
            self.count,
            Some(std::mem::offset_of!(GatherInstancesParams, base)),
        );
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );
        self.build(rd, cmd);
    }

    fn build(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        if self.scratch_alignment == 0 {
            self.scratch_alignment = vk_utils::get_acceleration_structure_properties(rd)
                .min_acceleration_structure_scratch_offset_alignment
                as u64;
        }
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.instances_buf.address,
                    }),
            });
        let flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
        let primitive_count = self.count;
        let mut build_size = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            rd.ext_acc_struct.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                    .flags(flags)
                    .geometries(std::slice::from_ref(&geometry)),
                std::slice::from_ref(&primitive_count),
                &mut build_size,
            )
        };

        // The structure and its handle persist; both are only recreated when the scene
        // outgrows them (a handle created with a larger size stays valid for smaller builds).
        if build_size.acceleration_structure_size > self.handle_size {
            rd.destroyer
                .destroy_acceleration_structure(self.acceleration_structure.handle);
            rd.destroyer
                .destroy_buffer(self.acceleration_structure.buffer.handle);
            self.acceleration_structure.buffer = rd.create_device_buffer(
                build_size.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            );
            self.acceleration_structure.handle = unsafe {
                rd.ext_acc_struct.create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .size(build_size.acceleration_structure_size)
                        .buffer(self.acceleration_structure.buffer.handle),
                    None,
                )
            }
            .unwrap();
            self.handle_size = build_size.acceleration_structure_size;
            self.acceleration_structure.address = unsafe {
                rd.ext_acc_struct
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(self.acceleration_structure.handle),
                    )
            };
        }
        let scratch_size = build_size.build_scratch_size + self.scratch_alignment;
        if scratch_size > self.scratch_buffer.nr_elements {
            rd.destroyer.destroy_buffer(self.scratch_buffer.handle);
            self.scratch_buffer =
                rd.create_device_buffer(scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER);
        }

        let build_geometry = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.acceleration_structure.handle)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: vk_utils::aligned_size(
                    self.scratch_buffer.address,
                    self.scratch_alignment,
                ),
            });
        let build_range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
        let build_range_infos = std::slice::from_ref(&build_range);
        unsafe {
            rd.ext_acc_struct.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&build_geometry),
                std::slice::from_ref(&build_range_infos),
            );
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
        );
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        rd.destroyer
            .destroy_acceleration_structure(self.acceleration_structure.handle);
        for b in [
            self.acceleration_structure.buffer.handle,
            self.scratch_buffer.handle,
            self.instances_buf.handle,
            self.instance_node_buf.handle,
            self.staging.handle,
            self.materials.device.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
    }
}

// ---- extraction -----------------------------------------------------------------------------

type ChangedInstances = Or<(
    Added<GpuInstance>,
    Changed<GpuNode>,
    Changed<Mesh3d>,
    Changed<GltfModelHandle>,
    Changed<MeshMaterial3d<StandardMaterial>>,
    Changed<InheritedVisibility>,
)>;

#[allow(clippy::type_complexity)]
fn extract_instances(
    mut tlas: ResMut<TLAS>,
    slots: Extract<Res<GpuInstanceSlots>>,
    changed: Extract<
        Query<
            (
                &GpuInstance,
                &GpuNode,
                Option<&Mesh3d>,
                Option<&GltfModelHandle>,
                Has<Sphere>,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Option<&InheritedVisibility>,
            ),
            ChangedInstances,
        >,
    >,
) {
    tlas.count = slots.next;
    for slot in &slots.freed {
        tlas.set_source(*slot, None);
    }
    for (instance, node, mesh, gltf, sphere, material, visibility) in changed.iter() {
        let geometry = if let Some(mesh) = mesh {
            Geometry::Mesh(mesh.id())
        } else if let Some(gltf) = gltf {
            Geometry::Gltf(gltf.0.id())
        } else if sphere {
            Geometry::Sphere
        } else {
            continue;
        };
        tlas.set_source(
            instance.0,
            Some(InstanceSource {
                geometry,
                material: material.map(|m| m.id()),
                node: node.0,
                mask: if visibility.is_none_or(|v| v.get()) {
                    0xFF
                } else {
                    0x00
                },
            }),
        );
    }
}

/// Resolves every dirty or still-pending slot against the loaded BLASes / materials /
/// textures and emits a record for each slot whose static half differs from the GPU's.
pub fn prepare_instances(
    render_device: Res<RenderDevice>,
    mut tlas: ResMut<TLAS>,
    meshes: Res<VulkanAssets<Mesh>>,
    gltf_meshes: Res<VulkanAssets<GltfModel>>,
    materials: Res<VulkanAssets<StandardMaterial>>,
    textures: Res<VulkanAssets<Image>>,
    sphere_blas: Res<SphereBLAS>,
) {
    let tlas = &mut *tlas;
    if tlas.dirty.is_empty() && tlas.pending.is_empty() {
        return;
    }
    let mut slots: Vec<u32> = tlas.dirty.drain(..).collect();
    slots.extend(tlas.pending.iter().copied());
    slots.sort_unstable();
    slots.dedup();

    for slot in slots {
        let Some(source) = tlas.sources.get(slot as usize).cloned().flatten() else {
            tlas.pending.remove(&slot);
            tlas.materials.clear(slot);
            let record = InstanceRecord {
                slot,
                ..Default::default()
            };
            if tlas.mirror[slot as usize] != record {
                tlas.mirror[slot as usize] = record;
                tlas.records.push(record);
            }
            continue;
        };

        let mut complete = true;
        let (blas, hit_offset, gltf_materials) = match source.geometry {
            Geometry::Mesh(id) => {
                let offset = tlas.hit_offset(id.untyped());
                match meshes.get_by_id(id) {
                    Some(b) => (b.acceleration_structure.address, offset, None),
                    None => {
                        complete = false;
                        (0, offset, None)
                    }
                }
            }
            Geometry::Gltf(id) => {
                let offset = tlas.hit_offset(id.untyped());
                match gltf_meshes.get_by_id(id) {
                    Some(b) => (
                        b.acceleration_structure.address,
                        offset,
                        b.gltf_materials.clone(),
                    ),
                    None => {
                        complete = false;
                        (0, offset, None)
                    }
                }
            }
            Geometry::Sphere => (sphere_blas.acceleration_structure.address, 0, None),
        };

        let material_slice: Vec<RTXMaterial> = match (source.material, gltf_materials) {
            (Some(id), _) => match materials.get_by_id(id) {
                Some(m) => {
                    let (resolved, all_textures) = m.resolve_checked(&render_device, &textures);
                    complete &= all_textures;
                    vec![resolved]
                }
                None => {
                    complete = false;
                    vec![RTXMaterial::default()]
                }
            },
            (None, Some(bundle)) => bundle,
            (None, None) => vec![RTXMaterial::default()],
        };
        let custom_index = tlas.materials.set(slot, &material_slice);

        let record = InstanceRecord {
            slot,
            custom_and_mask: (custom_index & 0x00FF_FFFF) | ((source.mask as u32) << 24),
            sbt_and_flags: (hit_offset & 0x00FF_FFFF) | (INSTANCE_FLAGS << 24),
            node: source.node,
            blas,
            pad: 0,
        };
        if tlas.mirror[slot as usize] != record {
            tlas.mirror[slot as usize] = record;
            tlas.records.push(record);
        }
        if complete {
            tlas.pending.remove(&slot);
        } else {
            tlas.pending.insert(slot);
        }
    }
}

fn cleanup_tlas(world: &mut World) {
    world.resource_scope(|world, mut tlas: Mut<TLAS>| {
        let rd = world.resource::<RenderDevice>();
        tlas.destroy(rd);
    });
}

pub struct TLASBuilderPlugin;

impl Plugin for TLASBuilderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuInstanceSlots>();
        app.add_systems(First, clear_freed);
        app.add_systems(PostUpdate, assign_gpu_instances);
        app.add_observer(free_gpu_instance);

        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/instances.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &["scatter_instances", "gather_instances"],
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(TLAS::new(module));
        render_app.add_systems(ExtractSchedule, extract_instances);
        render_app.add_systems(
            Render,
            prepare_instances
                .in_set(RenderSet::Prepare)
                .after(poll_for_asset::<Mesh>)
                .after(poll_for_asset::<GltfModel>)
                .after(poll_for_asset::<StandardMaterial>)
                .after(poll_for_asset::<Image>),
        );
        render_app.add_systems(TeardownSchedule, cleanup_tlas);
    }
}
