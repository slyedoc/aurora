//! Opacity micromaps (`VK_EXT_opacity_micromap`): baked per-triangle opacity for alpha-cutout
//! geometry, resolved by the ray-tracing hardware instead of the any-hit shader.
//!
//! The bake is offline (the `aurora_files` importers, against the material's base-colour
//! alpha) and ships inside the `.cluster_mesh` v3 slices; [`crate::cluster_mesh::OmmSlices`]
//! re-indexes them to the emitted mesh's triangle order. This module turns those slices into a
//! `VkMicromapEXT` and the `VkAccelerationStructureTrianglesOpacityMicromapEXT` the mesh's BLAS
//! build attaches to its triangle geometry (`blas.rs`). Known opaque / transparent
//! micro-triangles never invoke the any-hit shader; `unknown` ones still do (4-state bakes), so
//! the shader-side alpha test stays exact where the bake could not decide.
//!
//! The micromap is referenced by the BLAS at trace time and lives as long as it; the build
//! inputs (array data, triangle descriptors, per-triangle index, scratch) are transient.
//! Instances can opt out per frame with `VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_EXT`
//! (the dev panel's `omm` toggle, `tlas_builder.rs`).

use std::ffi::c_void;

use ash::vk;

use crate::{
    cluster_mesh::{OmmDesc, OmmSlices, OmmUsage},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    vk_utils,
};

/// Micromap build scratch alignment. The device property is smaller, but an unaligned scratch
/// makes `vkCmdBuildMicromapsEXT` emit garbage (every micro-region "unknown") with no
/// validation error; 256 is safe on every driver.
const SCRATCH_ALIGNMENT: u64 = 256;

/// A built micromap: what the BLAS references while it is traced.
pub struct Micromap {
    pub handle: vk::MicromapEXT,
    pub buffer: Buffer<u8>,
}

impl Micromap {
    pub fn destroy(&self, rd: &RenderDevice) {
        rd.destroyer.destroy_micromap(self.handle);
        rd.destroyer.destroy_buffer(self.buffer.handle);
    }
}

/// Host buffer holding `bytes` at a 256-aligned device address (the micromap build inputs
/// `data` and `triangleArray` must be; the allocator only promises the buffer's own alignment).
fn aligned_host_input(rd: &RenderDevice, bytes: &[u8]) -> (Buffer<u8>, u64) {
    let mut buffer: Buffer<u8> = rd.create_host_buffer(
        bytes.len() as u64 + SCRATCH_ALIGNMENT,
        vk::BufferUsageFlags::MICROMAP_BUILD_INPUT_READ_ONLY_EXT,
    );
    let address = vk_utils::aligned_size(buffer.address, SCRATCH_ALIGNMENT);
    let offset = (address - buffer.address) as usize;
    let mut view = rd.map_buffer(&mut buffer);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), view.as_ptr_mut().add(offset), bytes.len());
    }
    (buffer, address)
}

/// A micromap between its creation and the end of the BLAS build that consumes it. Boxed by
/// the caller and never moved meanwhile: `geometry_ext` points into `index_usage`.
pub struct MicromapBuild {
    micromap: Micromap,
    array_data: Buffer<u8>,
    array_data_address: u64,
    descs: Buffer<u8>,
    descs_address: u64,
    index: Buffer<i32>,
    scratch: Buffer<u8>,
    usage: Vec<vk::MicromapUsageEXT>,
    index_usage: Vec<vk::MicromapUsageEXT>,
    geometry_ext: vk::AccelerationStructureTrianglesOpacityMicromapEXT<'static>,
    pub triangle_count: usize,
}

fn widen(usage: &[OmmUsage]) -> Vec<vk::MicromapUsageEXT> {
    usage
        .iter()
        .map(|u| {
            vk::MicromapUsageEXT::default()
                .count(u.count)
                .subdivision_level(u32::from(u.subdivision_level))
                .format(u32::from(u.format))
        })
        .collect()
}

impl MicromapBuild {
    /// Uploads the build inputs (host-visible, read directly by the build), sizes and creates
    /// the micromap and its scratch. `None` when the device lacks the extension.
    pub fn prepare(rd: &RenderDevice, slices: &OmmSlices) -> Option<Box<Self>> {
        let ext = rd.ext_micromap.as_ref()?;
        if slices.descs.is_empty() || slices.index.is_empty() || slices.index_usage.is_empty() {
            return None;
        }
        let (array_data, array_data_address) = aligned_host_input(rd, &slices.array_data);
        let (descs, descs_address) =
            aligned_host_input(rd, bytemuck::cast_slice::<OmmDesc, u8>(&slices.descs));
        let mut index: Buffer<i32> = rd.create_host_buffer(
            slices.index.len() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        );
        rd.map_buffer(&mut index).copy_from_slice(&slices.index);

        let usage = widen(&slices.usage);
        let index_usage = widen(&slices.index_usage);

        let size_query = vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .usage_counts(&usage);
        let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
        unsafe {
            (ext.fp().get_micromap_build_sizes_ext)(
                ext.device(),
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &size_query,
                &mut sizes,
            );
        }
        if sizes.micromap_size == 0 {
            log::warn!(
                "omm: zero-sized micromap for {} descriptors",
                slices.descs.len()
            );
            for b in [array_data.handle, descs.handle, index.handle] {
                rd.destroyer.destroy_buffer(b);
            }
            return None;
        }
        let buffer: Buffer<u8> = rd.create_device_buffer(
            sizes.micromap_size,
            vk::BufferUsageFlags::MICROMAP_STORAGE_EXT,
        );
        let mut handle = vk::MicromapEXT::null();
        let create = vk::MicromapCreateInfoEXT::default()
            .buffer(buffer.handle)
            .offset(0)
            .size(sizes.micromap_size)
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP);
        let result = unsafe {
            (ext.fp().create_micromap_ext)(ext.device(), &create, std::ptr::null(), &mut handle)
        };
        if result != vk::Result::SUCCESS {
            log::error!("omm: vkCreateMicromapEXT failed: {result:?}");
            for b in [array_data.handle, descs.handle, index.handle, buffer.handle] {
                rd.destroyer.destroy_buffer(b);
            }
            return None;
        }
        let scratch: Buffer<u8> = rd.create_device_buffer(
            sizes.build_scratch_size.max(1) + SCRATCH_ALIGNMENT,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        );

        let mut build = Box::new(Self {
            micromap: Micromap { handle, buffer },
            array_data,
            array_data_address,
            descs,
            descs_address,
            index,
            scratch,
            usage,
            index_usage,
            geometry_ext: vk::AccelerationStructureTrianglesOpacityMicromapEXT::default(),
            triangle_count: slices.index.len(),
        });
        // Pointers into the box: stable from here on.
        build.geometry_ext = vk::AccelerationStructureTrianglesOpacityMicromapEXT::default()
            .index_type(vk::IndexType::UINT32)
            .index_buffer(vk::DeviceOrHostAddressConstKHR {
                device_address: build.index.address,
            })
            .index_stride(std::mem::size_of::<i32>() as u64)
            .base_triangle(0)
            .micromap(build.micromap.handle);
        build.geometry_ext.usage_counts_count = build.index_usage.len() as u32;
        build.geometry_ext.p_usage_counts = build.index_usage.as_ptr();
        Some(build)
    }

    /// The `pNext` for the BLAS's `VkAccelerationStructureGeometryTrianglesDataKHR`.
    pub fn geometry_ext_ptr(&self) -> *const c_void {
        (&self.geometry_ext as *const vk::AccelerationStructureTrianglesOpacityMicromapEXT<'_>)
            .cast()
    }

    fn build_info(&self) -> vk::MicromapBuildInfoEXT<'_> {
        vk::MicromapBuildInfoEXT::default()
            .ty(vk::MicromapTypeEXT::OPACITY_MICROMAP)
            .flags(vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE)
            .mode(vk::BuildMicromapModeEXT::BUILD)
            .usage_counts(&self.usage)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: self.array_data_address,
            })
            .triangle_array(vk::DeviceOrHostAddressConstKHR {
                device_address: self.descs_address,
            })
            .triangle_array_stride(std::mem::size_of::<OmmDesc>() as u64)
            .dst_micromap(self.micromap.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: vk_utils::aligned_size(self.scratch.address, SCRATCH_ALIGNMENT),
            })
    }

    /// Records every micromap build plus the barrier that makes them visible to the
    /// acceleration-structure builds that follow.
    pub fn record(rd: &RenderDevice, cmd: vk::CommandBuffer, builds: &[&MicromapBuild]) {
        let Some(ext) = rd.ext_micromap.as_ref() else {
            return;
        };
        if builds.is_empty() {
            return;
        }
        let infos: Vec<vk::MicromapBuildInfoEXT> = builds.iter().map(|b| b.build_info()).collect();
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::MICROMAP_BUILD_EXT)
            .src_access_mask(vk::AccessFlags2::MICROMAP_WRITE_EXT)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags2::MICROMAP_READ_EXT,
            );
        unsafe {
            (ext.fp().cmd_build_micromaps_ext)(cmd, infos.len() as u32, infos.as_ptr());
            rd.ext_sync2.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
            );
        }
    }

    /// Releases the build inputs once the BLAS build that read them has completed.
    pub fn finish(self: Box<Self>, rd: &RenderDevice) -> Micromap {
        for b in [
            self.array_data.handle,
            self.descs.handle,
            self.index.handle,
            self.scratch.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
        self.micromap
    }
}
