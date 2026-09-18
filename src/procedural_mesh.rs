//! GPU-generated meshes: a [`ProceduralMesh`] asset whose vertices are written by the app's
//! own compute kernel (any `ComputeModule` entry point), packed and built into a BLAS on the
//! asset worker, then traced exactly like a `Mesh` -- same TLAS slot, SBT record and
//! material path (`tlas_builder.rs` `Geometry::Procedural`).
//!
//! The contract with the fill kernel is its push-constant block: it MUST begin with
//! `ProceduralHeader` (`procedural.slang`: the vertex stream pointer, the vertex count and
//! the dispatch base the engine fills in), followed by whatever the app packs into
//! [`ProceduralMesh::params`] (at most [`PROCEDURAL_PARAMS_MAX`] bytes -- anything larger,
//! a planet genome say, goes in a device buffer the params point at). One thread per vertex;
//! the index buffer is uploaded from the asset, and `pack_triangles` derives the shading
//! records.
//!
//! The kernel's module must be compiled before the asset is created (a `ComputeModule` that
//! has not landed makes the extract skip the asset for good); [`ProceduralKernels::ready`]
//! is the gate.

use std::sync::Arc;

use ash::vk;
use bevy::{
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::{BLAS, BlasDeviceInput, GeometryDescr, Triangle, Vertex, build_blas_batch_device},
    compute::{
        COMPUTE_PUSH_CONSTANT_SIZE, CompiledComputeModule, ComputeModule, memory_barrier,
        record_dispatch,
    },
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

/// The fixed head of every fill kernel's push-constant block (`ProceduralHeader` in
/// procedural.slang).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProceduralHeader {
    pub vertices: u64,
    pub vertex_count: u32,
    pub base: u32,
}

/// Bytes of app parameters a [`ProceduralMesh`] may carry after the header.
pub const PROCEDURAL_PARAMS_MAX: usize =
    COMPUTE_PUSH_CONSTANT_SIZE as usize - std::mem::size_of::<ProceduralHeader>();

#[derive(Asset, TypePath, Clone)]
pub struct ProceduralMesh {
    pub vertex_count: u32,
    /// Triangle list; shared between meshes of one topology.
    pub indices: Arc<[u32]>,
    /// The module holding the fill kernel and the entry point to run, one thread per vertex.
    pub module: Handle<ComputeModule>,
    pub entry: String,
    /// Push constants after the header (the kernel's own struct, scalar layout).
    pub params: Vec<u8>,
}

/// A ray-traced instance of a [`ProceduralMesh`] (the procedural `Mesh3d`).
#[derive(Component, Clone, Debug, Deref)]
#[require(Transform, Visibility)]
pub struct ProceduralMesh3d(pub Handle<ProceduralMesh>);

/// The engine's own procedural kernels (procedural.slang).
#[derive(Resource)]
pub struct ProceduralKernels {
    pub module: Handle<ComputeModule>,
}

impl ProceduralKernels {
    /// Both the engine's pack kernel and `module` are compiled: a [`ProceduralMesh`] using
    /// `module` may be created.
    pub fn ready(
        &self,
        modules: &VulkanAssets<ComputeModule>,
        module: &Handle<ComputeModule>,
    ) -> bool {
        modules.get(&self.module).is_some() && modules.get(module).is_some()
    }
}

pub struct ExtractedProceduralMesh {
    vertex_count: u32,
    indices: Arc<[u32]>,
    params: Vec<u8>,
    entry: String,
    fill_layout: vk::PipelineLayout,
    fill_pipeline: vk::Pipeline,
    pack_layout: vk::PipelineLayout,
    pack_pipeline: vk::Pipeline,
}

impl VulkanAsset for ProceduralMesh {
    type ExtractedAsset = ExtractedProceduralMesh;
    type ExtractParam = (SRes<VulkanAssets<ComputeModule>>, SRes<ProceduralKernels>);
    type PreparedAsset = BLAS;

    fn extract_asset(
        &self,
        (modules, kernels): &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        assert!(
            self.params.len() <= PROCEDURAL_PARAMS_MAX,
            "ProceduralMesh params: {} bytes, at most {PROCEDURAL_PARAMS_MAX}",
            self.params.len()
        );
        assert!(
            self.indices.len() % 3 == 0,
            "ProceduralMesh indices are a triangle list"
        );
        let Some(fill) = modules.get(&self.module) else {
            log::warn!("procedural mesh skipped: its fill module is not compiled yet");
            return None;
        };
        let Some(&fill_pipeline) = fill.pipelines.get(&self.entry) else {
            log::warn!("procedural mesh skipped: no entry point {}", self.entry);
            return None;
        };
        let Some(pack) = modules.get(&kernels.module) else {
            log::warn!("procedural mesh skipped: procedural.slang is not compiled yet");
            return None;
        };
        Some(ExtractedProceduralMesh {
            vertex_count: self.vertex_count,
            indices: self.indices.clone(),
            params: self.params.clone(),
            entry: self.entry.clone(),
            fill_layout: fill.pipeline_layout,
            fill_pipeline,
            pack_layout: pack.pipeline_layout,
            pack_pipeline: pack.pipeline("pack_triangles"),
        })
    }

    fn prepare_asset(asset: Self::ExtractedAsset, rd: &RenderDevice) -> Self::PreparedAsset {
        Self::prepare_batch(vec![asset], rd).pop().unwrap()
    }

    /// The whole batch's fills in ONE submission (index uploads, every fill dispatch, every
    /// pack dispatch), then one uncompacted batch build: a streaming scene's patches cost a
    /// couple of queue round trips per batch instead of three per mesh, each of which had
    /// to wait behind a frame on the single queue.
    fn prepare_batch(assets: Vec<Self::ExtractedAsset>, rd: &RenderDevice) -> Vec<BLAS> {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        struct PackParams {
            vertices: u64,
            indices: u64,
            triangles: u64,
            count: u32,
            base: u32,
        }
        struct Staged {
            asset: ExtractedProceduralMesh,
            index_host: Buffer<u32>,
            input: BlasDeviceInput,
            fill: CompiledComputeModule,
            pack: CompiledComputeModule,
            push: [u8; COMPUTE_PUSH_CONSTANT_SIZE as usize],
            pack_params: PackParams,
        }

        let as_input = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        let staged: Vec<Staged> = assets
            .into_iter()
            .map(|asset| {
                let vertex_count = asset.vertex_count as usize;
                let index_count = asset.indices.len();
                let triangle_count = index_count / 3;
                let mut index_host: Buffer<u32> = rd.create_host_buffer(
                    index_count as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                );
                rd.map_buffer(&mut index_host)
                    .copy_from_slice(&asset.indices);
                let vertex_buffer: Buffer<Vertex> =
                    rd.create_device_buffer(vertex_count as u64, as_input);
                let index_buffer: Buffer<u32> =
                    rd.create_device_buffer(index_count as u64, as_input);
                let triangle_buffer: Buffer<Triangle> =
                    rd.create_device_buffer(triangle_count as u64, storage);
                let geometry_to_index: Buffer<u32> = rd.create_device_buffer(1, storage);
                let geometry_to_triangle: Buffer<u32> = rd.create_device_buffer(1, storage);

                let fill = CompiledComputeModule {
                    pipeline_layout: asset.fill_layout,
                    pipelines: [(asset.entry.clone(), asset.fill_pipeline)]
                        .into_iter()
                        .collect(),
                };
                let pack = CompiledComputeModule {
                    pipeline_layout: asset.pack_layout,
                    pipelines: [("pack_triangles".to_string(), asset.pack_pipeline)]
                        .into_iter()
                        .collect(),
                };
                // The whole push block: header, then the app's parameters.
                let mut push = [0u8; COMPUTE_PUSH_CONSTANT_SIZE as usize];
                let header = ProceduralHeader {
                    vertices: vertex_buffer.address,
                    vertex_count: asset.vertex_count,
                    base: 0,
                };
                let header_size = std::mem::size_of::<ProceduralHeader>();
                push[..header_size].copy_from_slice(bytemuck::bytes_of(&header));
                push[header_size..header_size + asset.params.len()].copy_from_slice(&asset.params);
                let pack_params = PackParams {
                    vertices: vertex_buffer.address,
                    indices: index_buffer.address,
                    triangles: triangle_buffer.address,
                    count: triangle_count as u32,
                    base: 0,
                };
                Staged {
                    input: BlasDeviceInput {
                        vertex_buffer,
                        index_buffer,
                        triangle_buffer,
                        geometry_to_index,
                        geometry_to_triangle,
                        geometries: vec![GeometryDescr {
                            first_vertex: 0,
                            vertex_count,
                            first_index: 0,
                            index_count,
                        }],
                        vertex_count,
                    },
                    asset,
                    index_host,
                    fill,
                    pack,
                    push,
                    pack_params,
                }
            })
            .collect();

        rd.run_transfer_commands(|cmd| unsafe {
            for s in &staged {
                let copy = vk::BufferCopy::default().size(s.asset.indices.len() as u64 * 4);
                rd.device.cmd_copy_buffer(
                    cmd,
                    s.index_host.handle,
                    s.input.index_buffer.handle,
                    std::slice::from_ref(&copy),
                );
                rd.device.cmd_update_buffer(
                    cmd,
                    s.input.geometry_to_index.handle,
                    0,
                    bytemuck::bytes_of(&0u32),
                );
                rd.device.cmd_update_buffer(
                    cmd,
                    s.input.geometry_to_triangle.handle,
                    0,
                    bytemuck::bytes_of(&0u32),
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            );
            for s in &staged {
                record_dispatch(
                    rd,
                    cmd,
                    &s.fill,
                    &s.asset.entry,
                    &s.push,
                    s.asset.vertex_count,
                    Some(std::mem::offset_of!(ProceduralHeader, base)),
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            );
            for s in &staged {
                record_dispatch(
                    rd,
                    cmd,
                    &s.pack,
                    "pack_triangles",
                    &s.pack_params,
                    s.pack_params.count,
                    Some(std::mem::offset_of!(PackParams, base)),
                );
            }
            // The BLAS builds are a later submission on the same queue; make the streams
            // available to them and to the hit shaders.
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
            );
        });

        let mut inputs = Vec::with_capacity(staged.len());
        for s in staged {
            rd.destroyer.destroy_buffer(s.index_host.handle);
            inputs.push(s.input);
        }
        // Streamed geometry: skip compaction (a query round trip and a copy submission per
        // batch for memory these meshes never keep long enough to matter).
        build_blas_batch_device(rd, inputs, false)
    }

    fn destroy_asset(rd: &RenderDevice, prepared: &Self::PreparedAsset) {
        prepared.destroy(rd);
    }
}

pub struct ProceduralMeshPlugin;

impl Plugin for ProceduralMeshPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/procedural.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["pack_triangles"]));
        app.insert_resource(ProceduralKernels { module });
        app.init_asset::<ProceduralMesh>();
        app.init_vulkan_asset::<ProceduralMesh>();
    }
}
