use ash::vk;
use bevy::{
    asset::Asset,
    math::{Vec2, Vec3},
    pbr::StandardMaterial,
    reflect::TypePath,
};
use bytemuck::{Pod, Zeroable};
use half::f16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_env::{DEFAULT_NORMAL_TEXTURE_IDX, WHITE_TEXTURE_IDX},
    render_texture::RenderTexture,
    vk_utils,
    vulkan_asset::VulkanAsset,
};

#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Triangle {
    pub tangent: u32,
    pub normals: [u32; 3],
    pub uvs: [u32; 3],
    // We get better cache aligment by making the struct
    // 32 bytes instead of (3 + 3 + 1) * 4 = 28
    pub padding: u32,
}

impl Triangle {
    pub const fn pack_normal(n: &Vec3) -> u32 {
        let x = (n.x * 0.5 + 0.5) * 65535.0;
        let y = (n.y * 0.5 + 0.5) * 32767.0;
        let z = if n.z >= 0.0 { 0 } else { 1 };
        ((x as u32) << 16) | ((y as u32) << 1) | z
    }

    // inverse of unpackHalf2x16 in glsl
    pub fn pack_uv(uv: &Vec2) -> u32 {
        let x = f16::from_f32(uv.x).to_bits();
        let y = f16::from_f32(uv.y).to_bits();
        ((y as u32) << 16) | (x as u32)
    }
}

#[derive(Debug, Clone)]
pub struct GeometryDescr {
    pub first_vertex: usize,
    pub vertex_count: usize,
    pub first_index: usize,
    pub index_count: usize,
}

#[derive(TypePath, Asset, Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct RTXMaterial {
    pub base_color_factor: [f32; 4],
    pub base_emissive_factor: [f32; 4],
    pub base_color_texture: u32,
    pub base_emissive_texture: u32,
    pub specular_transmission_texture: u32,
    pub metallic_roughness_texture: u32,
    pub normal_texture: u32,
    pub specular_transmission_factor: f32,
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub refract_index: f32,
    /// Beer-Lambert absorption per unit distance travelled inside the surface (linear RGB).
    pub absorption: [f32; 3],
}

/// Absorption coefficients from bevy's `attenuation_color` / `attenuation_distance` pair:
/// light travelling `attenuation_distance` through the medium is tinted to `attenuation_color`.
/// An infinite (default) distance means a clear medium.
pub fn absorption_from_attenuation(color: bevy::color::LinearRgba, distance: f32) -> [f32; 3] {
    if !distance.is_finite() || distance <= 0.0 {
        return [0.0; 3];
    }
    let k = |c: f32| -(c.clamp(1e-4, 1.0)).ln() / distance;
    [k(color.red), k(color.green), k(color.blue)]
}

impl RTXMaterial {
    pub fn from_bevy_standard_material(material: &StandardMaterial) -> Self {
        RTXMaterial {
            base_color_factor: {
                let c = material.base_color.to_srgba();
                [c.red, c.green, c.blue, c.alpha]
            },
            base_emissive_factor: {
                let c = material.emissive;
                [c.red, c.green, c.blue, c.alpha]
            },
            base_color_texture: WHITE_TEXTURE_IDX,
            base_emissive_texture: WHITE_TEXTURE_IDX,
            normal_texture: DEFAULT_NORMAL_TEXTURE_IDX,
            specular_transmission_texture: WHITE_TEXTURE_IDX,
            metallic_roughness_texture: WHITE_TEXTURE_IDX,
            specular_transmission_factor: material.specular_transmission,
            roughness_factor: material.perceptual_roughness,
            metallic_factor: material.metallic,
            refract_index: material.ior,
            absorption: absorption_from_attenuation(
                material.attenuation_color.to_linear(),
                material.attenuation_distance,
            ),
        }
    }
}

impl Default for RTXMaterial {
    fn default() -> Self {
        RTXMaterial {
            base_color_factor: [0.5, 0.5, 0.5, 1.0],
            base_emissive_factor: [0.0, 0.0, 0.0, 0.0],
            base_color_texture: WHITE_TEXTURE_IDX,
            base_emissive_texture: WHITE_TEXTURE_IDX,
            normal_texture: DEFAULT_NORMAL_TEXTURE_IDX,
            specular_transmission_texture: WHITE_TEXTURE_IDX,
            metallic_roughness_texture: WHITE_TEXTURE_IDX,
            specular_transmission_factor: 0.0,
            roughness_factor: 1.0,
            metallic_factor: 0.0,
            refract_index: 1.0,
            absorption: [0.0; 3],
        }
    }
}

/// An extracted `StandardMaterial`: the shader-side record plus the images its texture slots come
/// from. The slots are resolved to bindless indices at TLAS update time, once the images have
/// uploaded — so textures apply whenever they finish loading, without re-extracting the material.
#[derive(Clone, Default)]
pub struct StandardRtxMaterial {
    pub material: RTXMaterial,
    pub base_color_texture: Option<bevy::asset::AssetId<bevy::image::Image>>,
    pub emissive_texture: Option<bevy::asset::AssetId<bevy::image::Image>>,
    pub metallic_roughness_texture: Option<bevy::asset::AssetId<bevy::image::Image>>,
    pub normal_map_texture: Option<bevy::asset::AssetId<bevy::image::Image>>,
}

impl StandardRtxMaterial {
    /// The record with every texture slot filled from `textures` (or its fallback).
    pub fn resolve(
        &self,
        render_device: &RenderDevice,
        textures: &crate::vulkan_asset::VulkanAssets<bevy::image::Image>,
    ) -> RTXMaterial {
        self.resolve_checked(render_device, textures).0
    }

    /// Like [`resolve`](Self::resolve), plus whether every referenced texture was found (a
    /// `false` means a fallback stands in and the record is worth resolving again later).
    pub fn resolve_checked(
        &self,
        render_device: &RenderDevice,
        textures: &crate::vulkan_asset::VulkanAssets<bevy::image::Image>,
    ) -> (RTXMaterial, bool) {
        let mut complete = true;
        let mut slot = |id: Option<bevy::asset::AssetId<bevy::image::Image>>, fallback: u32| {
            let Some(id) = id else { return fallback };
            match textures.get_by_id(id) {
                Some(texture) => render_device.register_bindless_texture(texture),
                None => {
                    complete = false;
                    fallback
                }
            }
        };
        let material = RTXMaterial {
            base_color_texture: slot(self.base_color_texture, WHITE_TEXTURE_IDX),
            base_emissive_texture: slot(self.emissive_texture, WHITE_TEXTURE_IDX),
            metallic_roughness_texture: slot(self.metallic_roughness_texture, WHITE_TEXTURE_IDX),
            normal_texture: slot(self.normal_map_texture, DEFAULT_NORMAL_TEXTURE_IDX),
            ..self.material
        };
        (material, complete)
    }
}

impl VulkanAsset for StandardMaterial {
    type ExtractedAsset = StandardRtxMaterial;
    type ExtractParam = ();
    type PreparedAsset = StandardRtxMaterial;

    fn extract_asset(
        &self,
        _param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        Some(StandardRtxMaterial {
            material: RTXMaterial::from_bevy_standard_material(self),
            base_color_texture: self
                .base_color_texture
                .as_ref()
                .map(bevy::asset::Handle::id),
            emissive_texture: self.emissive_texture.as_ref().map(bevy::asset::Handle::id),
            metallic_roughness_texture: self
                .metallic_roughness_texture
                .as_ref()
                .map(bevy::asset::Handle::id),
            normal_map_texture: self
                .normal_map_texture
                .as_ref()
                .map(bevy::asset::Handle::id),
        })
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        _render_device: &RenderDevice,
    ) -> Self::PreparedAsset {
        asset
    }

    fn destroy_asset(_render_device: &RenderDevice, _prepared_asset: &Self::PreparedAsset) {}
}

pub struct BLAS {
    pub acceleration_structure: AccelerationStructure,
    pub vertex_buffer: Buffer<Vertex>,
    pub triangle_buffer: Buffer<Triangle>,
    pub index_buffer: Buffer<u32>,
    pub geometry_to_index: Buffer<u32>,
    pub geometry_to_triangle: Buffer<u32>,
    pub gltf_materials: Option<Vec<RTXMaterial>>,
    pub gltf_textures: Option<Vec<RenderTexture>>,
}

impl BLAS {
    pub fn destroy(&self, render_device: &RenderDevice) {
        render_device
            .destroyer
            .destroy_acceleration_structure(self.acceleration_structure.handle);
        render_device
            .destroyer
            .destroy_buffer(self.acceleration_structure.buffer.handle);
        render_device
            .destroyer
            .destroy_buffer(self.vertex_buffer.handle);
        render_device
            .destroyer
            .destroy_buffer(self.triangle_buffer.handle);
        render_device
            .destroyer
            .destroy_buffer(self.index_buffer.handle);
        render_device
            .destroyer
            .destroy_buffer(self.geometry_to_index.handle);
        render_device
            .destroyer
            .destroy_buffer(self.geometry_to_triangle.handle);
    }
}

#[derive(Default)]
pub struct AccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: Buffer<u8>,
    pub address: u64,
}

impl AccelerationStructure {
    pub fn get_reference(&self) -> vk::AccelerationStructureReferenceKHR {
        vk::AccelerationStructureReferenceKHR {
            device_handle: self.address,
        }
    }

    pub fn destroy(&self, render_device: &RenderDevice) {
        render_device
            .destroyer
            .destroy_acceleration_structure(self.handle);
        render_device.destroyer.destroy_buffer(self.buffer.handle);
    }
}

/// Everything a BLAS build needs from the caller: host-visible vertex/index buffers already
/// filled, plus the geometry ranges inside them.
pub struct BlasBuildInput {
    pub vertex_count: usize,
    pub index_count: usize,
    pub vertex_buffer_host: Buffer<Vertex>,
    pub index_buffer_host: Buffer<u32>,
    pub geometries: Vec<GeometryDescr>,
}

/// Scratch memory a single build submission may hold at once. Cluster meshes need a few KB
/// each; this only matters when a batch carries several huge meshes.
const BLAS_BATCH_SCRATCH_BUDGET: u64 = 256 * 1024 * 1024;

pub fn build_blas_from_buffers(
    render_device: &RenderDevice,
    vertex_count: usize,
    index_count: usize,
    vertex_buffer_host: Buffer<Vertex>,
    index_buffer_host: Buffer<u32>,
    geometries: &[GeometryDescr],
) -> BLAS {
    build_blas_batch(
        render_device,
        vec![BlasBuildInput {
            vertex_count,
            index_count,
            vertex_buffer_host,
            index_buffer_host,
            geometries: geometries.to_vec(),
        }],
    )
    .pop()
    .unwrap()
}

/// Per-mesh state between the upload and build stages of a batch.
struct BlasStaging {
    vertex_buffer: Buffer<Vertex>,
    index_buffer: Buffer<u32>,
    triangle_buffer: Buffer<Triangle>,
    geom_to_index: Buffer<u32>,
    geom_to_triangle: Buffer<u32>,
    geometries: Vec<GeometryDescr>,
    vertex_count: usize,
}

/// Builds many BLASes with three queue submissions total (uploads; builds + compacted-size
/// queries; compaction copies) instead of six per mesh. Scenes made of thousands of cluster
/// meshes stream in as a handful of batches rather than one mesh every few frames, since each
/// submission has to take turns with the frame loop on the single queue.
pub fn build_blas_batch(render_device: &RenderDevice, inputs: Vec<BlasBuildInput>) -> Vec<BLAS> {
    if inputs.is_empty() {
        return Vec::new();
    }
    log::debug!("Building {} BLASes", inputs.len());

    // ---- Stage 1: pack per-triangle shading data on the CPU (parallel across meshes), create
    // the device buffers, and upload everything in one submission.
    let packed: Vec<(BlasBuildInput, Vec<Triangle>, Vec<u32>, Vec<u32>)> = inputs
        .into_iter()
        .map(|mut input| {
            let vertex_view = render_device.map_buffer(&mut input.vertex_buffer_host);
            let index_view = render_device.map_buffer(&mut input.index_buffer_host);
            (input, vertex_view, index_view)
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(input, vertex_view, index_view)| {
            let vertices =
                unsafe { std::slice::from_raw_parts(vertex_view.as_ptr(), input.vertex_count) };
            let indices =
                unsafe { std::slice::from_raw_parts(index_view.as_ptr(), input.index_count) };
            let mut geom_to_index = Vec::with_capacity(input.geometries.len());
            let mut geom_to_triangle = Vec::with_capacity(input.geometries.len());
            let mut triangles = Vec::with_capacity(input.index_count / 3);
            let mut prefix_sum = 0u32;
            for geometry in &input.geometries {
                geom_to_index.push(geometry.first_index as u32);
                geom_to_triangle.push(prefix_sum);
                prefix_sum += geometry.index_count as u32 / 3;
                for tid in 0..(geometry.index_count / 3) {
                    let v0 = vertices[indices[geometry.first_index + tid * 3] as usize];
                    let v1 = vertices[indices[geometry.first_index + tid * 3 + 1] as usize];
                    let v2 = vertices[indices[geometry.first_index + tid * 3 + 2] as usize];
                    triangles.push(pack_triangle(&v0, &v1, &v2));
                }
            }
            (input, triangles, geom_to_index, geom_to_triangle)
        })
        .collect();

    let mut host_buffers: Vec<vk::Buffer> = Vec::with_capacity(packed.len() * 5);
    let mut staging: Vec<BlasStaging> = Vec::with_capacity(packed.len());
    let mut uploads: Vec<(vk::Buffer, vk::Buffer, u64)> = Vec::with_capacity(packed.len() * 5);

    for (input, triangles, geom_to_index, geom_to_triangle) in packed {
        let BlasBuildInput {
            vertex_count,
            index_count,
            vertex_buffer_host,
            index_buffer_host,
            geometries,
        } = input;
        let mut triangle_host: Buffer<Triangle> = render_device.create_host_buffer(
            triangles.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        render_device
            .map_buffer(&mut triangle_host)
            .copy_from_slice(&triangles);
        let mut geom_to_index_host: Buffer<u32> = render_device.create_host_buffer(
            geom_to_index.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        render_device
            .map_buffer(&mut geom_to_index_host)
            .copy_from_slice(&geom_to_index);
        let mut geom_to_triangle_host: Buffer<u32> = render_device.create_host_buffer(
            geom_to_triangle.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        render_device
            .map_buffer(&mut geom_to_triangle_host)
            .copy_from_slice(&geom_to_triangle);

        let as_input = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let vertex_buffer: Buffer<Vertex> =
            render_device.create_device_buffer(vertex_count as u64, as_input);
        let index_buffer: Buffer<u32> =
            render_device.create_device_buffer(index_count as u64, as_input);
        let triangle_buffer: Buffer<Triangle> =
            render_device.create_device_buffer(triangles.len() as u64, storage);
        let geom_to_index_dev: Buffer<u32> =
            render_device.create_device_buffer(geom_to_index.len() as u64, storage);
        let geom_to_triangle_dev: Buffer<u32> =
            render_device.create_device_buffer(geom_to_triangle.len() as u64, storage);

        uploads.push((
            vertex_buffer_host.handle,
            vertex_buffer.handle,
            vertex_count as u64 * std::mem::size_of::<Vertex>() as u64,
        ));
        uploads.push((
            index_buffer_host.handle,
            index_buffer.handle,
            index_count as u64 * std::mem::size_of::<u32>() as u64,
        ));
        uploads.push((
            triangle_host.handle,
            triangle_buffer.handle,
            triangles.len() as u64 * std::mem::size_of::<Triangle>() as u64,
        ));
        uploads.push((
            geom_to_index_host.handle,
            geom_to_index_dev.handle,
            geom_to_index.len() as u64 * 4,
        ));
        uploads.push((
            geom_to_triangle_host.handle,
            geom_to_triangle_dev.handle,
            geom_to_triangle.len() as u64 * 4,
        ));
        host_buffers.extend([
            vertex_buffer_host.handle,
            index_buffer_host.handle,
            triangle_host.handle,
            geom_to_index_host.handle,
            geom_to_triangle_host.handle,
        ]);
        staging.push(BlasStaging {
            vertex_buffer,
            index_buffer,
            triangle_buffer,
            geom_to_index: geom_to_index_dev,
            geom_to_triangle: geom_to_triangle_dev,
            geometries,
            vertex_count,
        });
    }

    render_device.run_transfer_commands(|cmd_buffer| {
        for (src, dst, size) in &uploads {
            if *size == 0 {
                continue;
            }
            let copy = vk::BufferCopy::default().size(*size);
            unsafe {
                render_device.device.cmd_copy_buffer(
                    cmd_buffer,
                    *src,
                    *dst,
                    std::slice::from_ref(&copy),
                )
            };
        }
    });
    for handle in host_buffers {
        render_device.destroyer.destroy_buffer(handle);
    }

    // ---- Stage 2: build + compact, in chunks bounded by scratch memory.
    let as_properties = vk_utils::get_acceleration_structure_properties(render_device);
    let scratch_alignment =
        as_properties.min_acceleration_structure_scratch_offset_alignment as u64;
    let build_flags = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;

    let mut out: Vec<BLAS> = Vec::with_capacity(staging.len());
    let mut chunk: Vec<BlasStaging> = Vec::new();
    let mut chunk_scratch = 0u64;
    let mut queued = staging.into_iter().peekable();
    while let Some(next) = queued.next() {
        chunk_scratch += estimate_scratch(render_device, &next, build_flags) + scratch_alignment;
        chunk.push(next);
        let flush = queued.peek().is_none() || chunk_scratch >= BLAS_BATCH_SCRATCH_BUDGET;
        if flush {
            out.extend(build_chunk(
                render_device,
                std::mem::take(&mut chunk),
                scratch_alignment,
                build_flags,
            ));
            chunk_scratch = 0;
        }
    }
    out
}

fn geometry_infos(
    staging: &BlasStaging,
) -> (
    Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    Vec<u32>,
    Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
) {
    let infos = staging
        .geometries
        .iter()
        .map(|_| {
            vk::AccelerationStructureGeometryKHR::default()
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: staging.vertex_buffer.address,
                        })
                        .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                        .max_vertex(staging.vertex_count as u32)
                        .index_type(vk::IndexType::UINT32)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: staging.index_buffer.address,
                        })
                        .transform_data(vk::DeviceOrHostAddressConstKHR { device_address: 0 }),
                })
        })
        .collect();
    let counts = staging
        .geometries
        .iter()
        .map(|g| (g.index_count / 3) as u32)
        .collect();
    let ranges = staging
        .geometries
        .iter()
        .map(|g| {
            vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count((g.index_count / 3) as u32)
                // offset in bytes where the primitive data is defined
                .primitive_offset(g.first_index as u32 * std::mem::size_of::<u32>() as u32)
                .first_vertex(0)
                .transform_offset(0)
        })
        .collect();
    (infos, counts, ranges)
}

fn build_sizes(
    render_device: &RenderDevice,
    geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
    counts: &[u32],
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(build_flags)
        .geometries(geometries);
    let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        render_device
            .ext_acc_struct
            .get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                counts,
                &mut size_info,
            )
    };
    size_info
}

fn estimate_scratch(
    render_device: &RenderDevice,
    staging: &BlasStaging,
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
) -> u64 {
    let (infos, counts, _) = geometry_infos(staging);
    build_sizes(render_device, &infos, &counts, build_flags).build_scratch_size
}

fn build_chunk(
    render_device: &RenderDevice,
    chunk: Vec<BlasStaging>,
    scratch_alignment: u64,
    build_flags: vk::BuildAccelerationStructureFlagsKHR,
) -> Vec<BLAS> {
    let n = chunk.len();
    let per_mesh: Vec<_> = chunk.iter().map(geometry_infos).collect();
    let sizes: Vec<_> = per_mesh
        .iter()
        .map(|(infos, counts, _)| build_sizes(render_device, infos, counts, build_flags))
        .collect();

    let mut structures: Vec<AccelerationStructure> = Vec::with_capacity(n);
    let mut scratch_buffers: Vec<Buffer<u8>> = Vec::with_capacity(n);
    for size_info in &sizes {
        structures.push(allocate_acceleration_structure(
            render_device,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            size_info,
        ));
        // The scratch address must be a multiple of minAccelerationStructureScratchOffsetAlignment.
        // The allocator gives no such guarantee, so over-allocate by one alignment and round the
        // address up into that slack. (Same handling as the TLAS build in tlas_builder.rs.)
        scratch_buffers.push(render_device.create_device_buffer(
            size_info.build_scratch_size + scratch_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        ));
    }

    let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = (0..n)
        .map(|i| {
            vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(build_flags)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .dst_acceleration_structure(structures[i].handle)
                .geometries(&per_mesh[i].0)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: vk_utils::aligned_size(
                        scratch_buffers[i].address,
                        scratch_alignment,
                    ),
                })
        })
        .collect();
    let build_ranges: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> = per_mesh
        .iter()
        .map(|(_, _, ranges)| ranges.as_slice())
        .collect();
    let handles: Vec<vk::AccelerationStructureKHR> = structures.iter().map(|s| s.handle).collect();

    let query_pool = unsafe {
        render_device.device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(n as u32),
            None,
        )
    }
    .unwrap();

    render_device.run_transfer_commands(|cmd_buffer| unsafe {
        render_device
            .device
            .cmd_reset_query_pool(cmd_buffer, query_pool, 0, n as u32);
        render_device
            .ext_acc_struct
            .cmd_build_acceleration_structures(cmd_buffer, &build_infos, &build_ranges);
        // Builds must land before the compacted-size query reads them.
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR);
        render_device.ext_sync2.cmd_pipeline_barrier2(
            cmd_buffer,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
        );
        render_device
            .ext_acc_struct
            .cmd_write_acceleration_structures_properties(
                cmd_buffer,
                &handles,
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                query_pool,
                0,
            );
    });
    for scratch in &scratch_buffers {
        render_device.destroyer.destroy_buffer(scratch.handle);
    }

    let mut compacted_sizes = vec![0u64; n];
    unsafe {
        render_device
            .device
            .get_query_pool_results::<u64>(
                query_pool,
                0,
                &mut compacted_sizes,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )
            .unwrap();
        render_device.device.destroy_query_pool(query_pool, None);
    }

    // Compaction: copy every structure that shrinks into a right-sized buffer, in one submission.
    // A zero compacted size means the query never produced a result -- creating a buffer of that
    // size yields VK_NULL_HANDLE, which then fails AS creation and the compacting copy. Keep the
    // uncompacted structure instead of building on top of a bad query.
    let mut copies: Vec<(usize, AccelerationStructure)> = Vec::new();
    for (i, &compacted) in compacted_sizes.iter().enumerate() {
        let full = sizes[i].acceleration_structure_size;
        log::debug!(
            "BLAS compaction: {} -> {} ({}%)",
            full,
            compacted,
            (compacted as f32 / full as f32) * 100.0
        );
        if compacted > 0 && compacted < full {
            let buffer = render_device.create_device_buffer::<u8>(
                compacted,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            );
            let handle = unsafe {
                render_device.ext_acc_struct.create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .size(compacted)
                        .buffer(buffer.handle),
                    None,
                )
            }
            .unwrap();
            copies.push((
                i,
                AccelerationStructure {
                    handle,
                    buffer,
                    address: 0,
                },
            ));
        } else {
            log::warn!(
                "Skipping BLAS compaction: reported compacted size {} is not a shrink from {}",
                compacted,
                full
            );
        }
    }
    if !copies.is_empty() {
        render_device.run_transfer_commands(|cmd_buffer| unsafe {
            for (i, compacted) in &copies {
                let copy_info = vk::CopyAccelerationStructureInfoKHR::default()
                    .src(structures[*i].handle)
                    .dst(compacted.handle)
                    .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);
                render_device
                    .ext_acc_struct
                    .cmd_copy_acceleration_structure(cmd_buffer, &copy_info);
            }
        });
        for (i, mut compacted) in copies {
            let old = std::mem::replace(&mut structures[i], AccelerationStructure::default());
            render_device
                .destroyer
                .destroy_acceleration_structure(old.handle);
            render_device.destroyer.destroy_buffer(old.buffer.handle);
            compacted.address = unsafe {
                render_device
                    .ext_acc_struct
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(compacted.handle),
                    )
            };
            structures[i] = compacted;
        }
    }

    chunk
        .into_iter()
        .zip(structures)
        .map(|(s, acceleration_structure)| BLAS {
            acceleration_structure,
            vertex_buffer: s.vertex_buffer,
            triangle_buffer: s.triangle_buffer,
            index_buffer: s.index_buffer,
            geometry_to_index: s.geom_to_index,
            geometry_to_triangle: s.geom_to_triangle,
            gltf_materials: None,
            gltf_textures: None,
        })
        .collect()
}

fn pack_triangle(v0: &Vertex, v1: &Vertex, v2: &Vertex) -> Triangle {
    let edge1 = v1.position - v0.position;
    let edge2 = v2.position - v0.position;
    let delta_uv1 = v1.uv - v0.uv;
    let delta_uv2 = v2.uv - v0.uv;

    let denom = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
    let tangent = if denom.abs() < 0.0001 {
        Vec3::Z
    } else {
        let f = 1.0 / denom;
        Vec3::new(
            f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x),
            f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y),
            f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z),
        )
        .normalize()
    };

    Triangle {
        tangent: Triangle::pack_normal(&tangent),
        padding: 0,
        normals: [
            Triangle::pack_normal(&v0.normal),
            Triangle::pack_normal(&v1.normal),
            Triangle::pack_normal(&v2.normal),
        ],
        uvs: [
            Triangle::pack_uv(&v0.uv),
            Triangle::pack_uv(&v1.uv),
            Triangle::pack_uv(&v2.uv),
        ],
    }
}

pub fn allocate_acceleration_structure(
    device: &RenderDevice,
    ty: vk::AccelerationStructureTypeKHR,
    build_size: &vk::AccelerationStructureBuildSizesInfoKHR,
) -> AccelerationStructure {
    let buffer: Buffer<u8> = device.create_device_buffer(
        build_size.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
    );

    let acceleration_structure = unsafe {
        device.ext_acc_struct.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .ty(ty)
                .size(build_size.acceleration_structure_size)
                .buffer(buffer.handle),
            None,
        )
    }
    .unwrap();

    let address = unsafe {
        device
            .ext_acc_struct
            .get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(acceleration_structure),
            )
    };

    AccelerationStructure {
        handle: acceleration_structure,
        buffer,
        address,
    }
}
