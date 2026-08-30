use std::collections::HashMap;

use crate::{
    blas::RTXMaterial,
    gltf_mesh::{GltfModel, GltfModelHandle},
    ray_render_plugin::TeardownSchedule,
    render_buffer::BufferProvider,
    sphere::SphereBLAS,
    vk_utils,
};
use ash::vk;
use bevy::{
    asset::UntypedAssetId, camera::visibility::InheritedVisibility, prelude::*, render::RenderApp,
};

use crate::{
    blas::AccelerationStructure,
    ray_render_plugin::{Render, RenderSet},
    render_buffer::Buffer,
    render_device::RenderDevice,
    vulkan_asset::VulkanAssets,
};

/// The 8-bit instance mask a ray's cull mask is tested against. Hidden entities keep their slot
/// in the TLAS -- same instance list, same topology -- but no ray can hit them.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtxInstanceMask(pub u8);

impl RtxInstanceMask {
    pub const VISIBLE: Self = Self(0xFF);
    pub const HIDDEN: Self = Self(0x00);

    /// Extracts from `InheritedVisibility` (what UI and hierarchy hiding drive); entities without
    /// the component are treated as visible.
    pub fn from_visibility(visibility: Option<&InheritedVisibility>) -> Self {
        if visibility.is_none_or(|v| v.get()) {
            Self::VISIBLE
        } else {
            Self::HIDDEN
        }
    }
}

#[derive(Default, Resource)]
pub struct TLAS {
    pub acceleration_structure: AccelerationStructure,
    pub instance_buffer: Buffer<vk::AccelerationStructureInstanceKHR>,
    pub scratch_buffer: Buffer<u8>,
    pub mesh_to_hit_offset: HashMap<UntypedAssetId, u32>,
    pub material_buffer: Buffer<RTXMaterial>,
    /// CPU mirror of what the GPU buffers hold (or will hold once `pending` is recorded). A
    /// frame whose gather matches it skips the upload and the build entirely.
    instances: Vec<vk::AccelerationStructureInstanceKHR>,
    materials: Vec<RTXMaterial>,
    /// `instances`/`materials` changed since the last recorded build.
    pending: bool,
    /// Size the current `acceleration_structure.handle` was created with (0 = none).
    handle_size: u64,
    scratch_alignment: u64,
}

fn instance_bytes(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    // repr(C), 64 bytes, no padding: 12 floats + two packed u32 + a u64 union.
    unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instances),
        )
    }
}

impl TLAS {
    /// Hands the frame's gathered instances to the TLAS. Only a change against the previous
    /// gather marks a build pending; a static scene costs nothing past the comparison.
    pub fn set_instances(
        &mut self,
        instances: Vec<vk::AccelerationStructureInstanceKHR>,
        materials: Vec<RTXMaterial>,
    ) {
        let unchanged = instance_bytes(&instances) == instance_bytes(&self.instances)
            && bytemuck::cast_slice::<_, u8>(&materials)
                == bytemuck::cast_slice::<_, u8>(&self.materials);
        if unchanged {
            return;
        }
        self.instances = instances;
        self.materials = materials;
        self.pending = true;
    }

    /// Records the pending build into the frame's command buffer, followed by the barrier that
    /// makes it visible to the trace. Must be called after the in-flight fence wait: with one
    /// frame in flight the previous trace is done, so the single TLAS and its host-visible
    /// instance/material buffers are rewritten in place.
    pub fn record_build(&mut self, render_device: &RenderDevice, cmd_buffer: vk::CommandBuffer) {
        if !self.pending || self.instances.is_empty() {
            return;
        }
        self.pending = false;

        if self.scratch_alignment == 0 {
            self.scratch_alignment = vk_utils::get_acceleration_structure_properties(render_device)
                .min_acceleration_structure_scratch_offset_alignment
                as u64;
        }

        if self.instances.len() != self.instance_buffer.nr_elements as usize {
            log::debug!(
                "Reallocating instance buffer from {} to {} elements",
                self.instance_buffer.nr_elements,
                self.instances.len()
            );
            render_device
                .destroyer
                .destroy_buffer(self.instance_buffer.handle);
            self.instance_buffer = render_device
                .create_host_buffer::<vk::AccelerationStructureInstanceKHR>(
                    self.instances.len() as u64,
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                );
        }
        if self.materials.len() != self.material_buffer.nr_elements as usize {
            log::debug!(
                "Reallocating material buffer from {} to {} elements",
                self.material_buffer.nr_elements,
                self.materials.len()
            );
            render_device
                .destroyer
                .destroy_buffer(self.material_buffer.handle);
            self.material_buffer = render_device.create_host_buffer::<RTXMaterial>(
                self.materials.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
        }
        render_device
            .map_buffer(&mut self.instance_buffer)
            .copy_from_slice(&self.instances);
        render_device
            .map_buffer(&mut self.material_buffer)
            .copy_from_slice(&self.materials);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.instance_buffer.address,
                    }),
            });
        let flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
        let primitive_count = self.instances.len() as u32;
        let mut build_size = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            render_device
                .ext_acc_struct
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .flags(flags)
                        .geometries(std::slice::from_ref(&geometry)),
                    std::slice::from_ref(&primitive_count),
                    &mut build_size,
                )
        };

        // The structure and its handle persist; both are only recreated when the scene outgrows
        // them (a handle created with a larger size stays valid for smaller builds).
        if build_size.acceleration_structure_size > self.handle_size {
            render_device
                .destroyer
                .destroy_acceleration_structure(self.acceleration_structure.handle);
            render_device
                .destroyer
                .destroy_buffer(self.acceleration_structure.buffer.handle);
            self.acceleration_structure.buffer = render_device.create_device_buffer(
                build_size.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            );
            self.acceleration_structure.handle = unsafe {
                render_device.ext_acc_struct.create_acceleration_structure(
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
                render_device
                    .ext_acc_struct
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(self.acceleration_structure.handle),
                    )
            };
        }

        let scratch_size = build_size.build_scratch_size + self.scratch_alignment;
        if scratch_size > self.scratch_buffer.nr_elements {
            render_device
                .destroyer
                .destroy_buffer(self.scratch_buffer.handle);
            self.scratch_buffer = render_device
                .create_device_buffer(scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER);
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
        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(primitive_count);
        let build_range_infos = std::slice::from_ref(&build_range);

        unsafe {
            render_device
                .ext_acc_struct
                .cmd_build_acceleration_structures(
                    cmd_buffer,
                    std::slice::from_ref(&build_geometry),
                    std::slice::from_ref(&build_range_infos),
                );
            let barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
            render_device.ext_sync2.cmd_pipeline_barrier2(
                cmd_buffer,
                &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
            );
        }
    }
}

/// Gathers this frame's instances on the CPU and hands them to the TLAS, which builds only when
/// they differ from the previous frame (see `TLAS::record_build`).
pub fn update_tlas(
    render_device: Res<RenderDevice>,
    mut tlas: ResMut<TLAS>,
    meshes: Res<VulkanAssets<Mesh>>,
    gltf_meshes: Res<VulkanAssets<GltfModel>>,
    materials: Res<VulkanAssets<StandardMaterial>>,
    textures: Res<VulkanAssets<Image>>,
    mesh_components: Query<(Entity, &Mesh3d)>,
    gltf_components: Query<(Entity, &GltfModelHandle)>,
    material_components: Query<&MeshMaterial3d<StandardMaterial>>,
    sphere_blas: Res<SphereBLAS>,
    spheres: Query<(Entity, &crate::sphere::Sphere)>,
    transforms: Query<&GlobalTransform>,
    masks: Query<&RtxInstanceMask>,
) {
    tlas.mesh_to_hit_offset.clear();
    // Reserve the first offset for the sphere hit group
    let mut hit_group_offset_gen = 1;

    let mut objects: Vec<(
        Entity,
        u32,
        GlobalTransform,
        vk::AccelerationStructureReferenceKHR,
        &Option<Vec<RTXMaterial>>,
    )> = Vec::new();
    objects.extend(mesh_components.iter().filter_map(|(e, mesh_handle)| {
        let blas = meshes.get(mesh_handle)?;
        let transform = transforms.get(e).unwrap();
        let hit_offset =
            if let Some(hit_offset) = tlas.mesh_to_hit_offset.get(&mesh_handle.id().untyped()) {
                *hit_offset
            } else {
                let old_val = hit_group_offset_gen;
                hit_group_offset_gen += 1;
                tlas.mesh_to_hit_offset
                    .insert(mesh_handle.id().untyped(), old_val);
                old_val
            };

        Some((
            e,
            hit_offset,
            transform.clone(),
            blas.acceleration_structure.get_reference(),
            &blas.gltf_materials,
        ))
    }));

    objects.extend(gltf_components.iter().filter_map(|(e, gltf_handle)| {
        let blas = gltf_meshes.get(gltf_handle)?;
        let transform = transforms.get(e).unwrap();
        let hit_offset =
            if let Some(hit_offset) = tlas.mesh_to_hit_offset.get(&gltf_handle.id().untyped()) {
                *hit_offset
            } else {
                let old_val = hit_group_offset_gen;
                hit_group_offset_gen += 1;
                tlas.mesh_to_hit_offset
                    .insert(gltf_handle.id().untyped(), old_val);
                old_val
            };

        Some((
            e,
            hit_offset,
            transform.clone(),
            blas.acceleration_structure.get_reference(),
            &blas.gltf_materials,
        ))
    }));

    for (sphere_e, _) in spheres.iter() {
        let transform = transforms.get(sphere_e).unwrap();
        objects.push((
            sphere_e,
            0,
            transform.clone(),
            sphere_blas.acceleration_structure.get_reference(),
            &None,
        ));
    }

    let mut material_offset = 0;
    let instances: Vec<(vk::AccelerationStructureInstanceKHR, Vec<RTXMaterial>)> = objects
        .iter()
        .map(|(e, hit_offset, transform, reference, mat_bundle)| {
            let columns = transform.affine().to_cols_array_2d();
            let transform = vk::TransformMatrixKHR {
                matrix: [
                    columns[0][0],
                    columns[1][0],
                    columns[2][0],
                    columns[3][0],
                    columns[0][1],
                    columns[1][1],
                    columns[2][1],
                    columns[3][1],
                    columns[0][2],
                    columns[1][2],
                    columns[2][2],
                    columns[3][2],
                ],
            };

            let mask = masks.get(*e).copied().unwrap_or(RtxInstanceMask::VISIBLE).0;
            let instance = vk::AccelerationStructureInstanceKHR {
                transform,
                instance_custom_index_and_mask: vk::Packed24_8::new(material_offset, mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    *hit_offset,
                    0b1,
                ),
                acceleration_structure_reference: *reference,
            };

            let material_slice = if let Ok(material_handle) = material_components.get(*e) {
                vec![
                    materials
                        .get(material_handle)
                        .map(|material| material.resolve(&render_device, &textures))
                        .unwrap_or_default(),
                ]
            } else {
                if let Some(gltf_materials) = mat_bundle {
                    gltf_materials.clone()
                } else {
                    log::warn!("No material found for entity {:?}", e);
                    vec![RTXMaterial::default()]
                }
            };
            material_offset += material_slice.len() as u32;

            (instance, material_slice)
        })
        .collect();

    let (instances, materials): (Vec<_>, Vec<_>) = instances.into_iter().unzip();
    tlas.set_instances(instances, materials.into_iter().flatten().collect());
}

fn cleanup_tlas(world: &mut World) {
    let tlas = world.remove_resource::<TLAS>().unwrap();
    let render_device = world.get_resource::<RenderDevice>().unwrap();
    render_device
        .destroyer
        .destroy_acceleration_structure(tlas.acceleration_structure.handle);
    render_device
        .destroyer
        .destroy_buffer(tlas.acceleration_structure.buffer.handle);
    render_device
        .destroyer
        .destroy_buffer(tlas.instance_buffer.handle);
    render_device
        .destroyer
        .destroy_buffer(tlas.scratch_buffer.handle);
    render_device
        .destroyer
        .destroy_buffer(tlas.material_buffer.handle);
}

pub struct TLASBuilderPlugin;

impl Plugin for TLASBuilderPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app.init_resource::<TLAS>();
        render_app.add_systems(Render, update_tlas.in_set(RenderSet::Prepare));
        render_app.add_systems(TeardownSchedule, cleanup_tlas);
    }
}
