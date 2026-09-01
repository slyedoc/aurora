use std::sync::Arc;

use bevy::{
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    mesh::Indices,
    prelude::*,
};

use crate::{
    blas::{BLAS, BlasBuildInput, GeometryDescr, SkinVertex, Vertex, build_blas_batch},
    bsn::ClusterMeshOmm,
    cluster_mesh::OmmSlices,
    render_buffer::BufferProvider,
    vulkan_asset::{VulkanAsset, VulkanAssetExt},
};
use ash::vk;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl VulkanAsset for Mesh {
    /// The mesh plus its baked opacity micromap, when it came from a `.cluster_mesh` with one.
    type ExtractedAsset = (Mesh, Option<Arc<OmmSlices>>);
    type ExtractParam = (SRes<AssetServer>, SRes<ClusterMeshOmm>);
    type PreparedAsset = BLAS;

    fn extract_asset(
        &self,
        _param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        Some((self.clone(), None))
    }

    fn extract_asset_with_id(
        &self,
        id: AssetId<Self>,
        (asset_server, registry): &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let omm = asset_server
            .get_path(id)
            .and_then(|path| registry.0.lock().unwrap().get(&path).cloned());
        Some((self.clone(), omm))
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        Self::prepare_batch(vec![asset], render_device)
            .pop()
            .unwrap()
    }

    /// Packs every mesh's vertex streams in parallel, then builds all their BLASes with a
    /// handful of shared queue submissions (see `build_blas_batch`).
    fn prepare_batch(
        assets: Vec<Self::ExtractedAsset>,
        render_device: &crate::render_device::RenderDevice,
    ) -> Vec<Self::PreparedAsset> {
        let packed: Vec<(
            (Vec<f32>, Vec<u32>, Option<Vec<SkinVertex>>),
            Option<Arc<OmmSlices>>,
        )> = assets
            .into_par_iter()
            .map(|(mesh, omm)| (pack_vertex_streams(mesh), omm))
            .collect();
        let inputs = packed
            .into_iter()
            .map(|((vertex_floats, indices, skin), omm)| {
                let vertex_count = vertex_floats.len() / 8;
                let index_count = indices.len();
                let mut vertex_buffer_host = render_device.create_host_buffer::<Vertex>(
                    vertex_count as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                );
                let mut index_buffer_host = render_device.create_host_buffer::<u32>(
                    index_count as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                );
                render_device
                    .map_buffer(&mut vertex_buffer_host)
                    .copy_from_slice(bytemuck::cast_slice(&vertex_floats));
                render_device
                    .map_buffer(&mut index_buffer_host)
                    .copy_from_slice(&indices);
                let skin_host = skin.map(|skin| {
                    let mut host = render_device.create_host_buffer::<SkinVertex>(
                        skin.len() as u64,
                        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                    );
                    render_device.map_buffer(&mut host).copy_from_slice(&skin);
                    host
                });
                BlasBuildInput {
                    vertex_count,
                    index_count,
                    vertex_buffer_host,
                    index_buffer_host,
                    geometries: vec![GeometryDescr {
                        first_vertex: 0,
                        vertex_count,
                        first_index: 0,
                        index_count,
                    }],
                    skin_host,
                    omm,
                }
            })
            .collect();
        build_blas_batch(render_device, inputs)
    }

    fn destroy_asset(
        render_device: &crate::render_device::RenderDevice,
        prepared_asset: &Self::PreparedAsset,
    ) {
        prepared_asset.destroy(render_device);
    }
}

/// The three streams the shaders' `Vertex` (types.glsl) reads -- position, normal, uv --
/// interleaved as 8 floats per vertex, plus u32 indices, plus the skinning influences when
/// the mesh carries joint attributes (other attributes, like tangents, are recomputed).
fn pack_vertex_streams(asset: Mesh) -> (Vec<f32>, Vec<u32>, Option<Vec<SkinVertex>>) {
    let vertex_count = asset.count_vertices();
    let indices: Vec<u32> = match asset.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        // Non-indexed: a triangle per three consecutive vertices.
        None => (0..vertex_count as u32).collect(),
    };

    let positions = asset
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .and_then(|a| a.as_float3())
        .expect("mesh has no positions");
    let normals: Vec<[f32; 3]> = match asset
        .attribute(Mesh::ATTRIBUTE_NORMAL)
        .and_then(|a| a.as_float3())
    {
        Some(n) if n.len() == vertex_count => n.to_vec(),
        _ => {
            let mut with_normals = asset.clone();
            with_normals.compute_normals();
            with_normals
                .attribute(Mesh::ATTRIBUTE_NORMAL)
                .and_then(|a| a.as_float3())
                .map(|n| n.to_vec())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count])
        }
    };
    let uvs: Vec<[f32; 2]> = match asset.attribute(Mesh::ATTRIBUTE_UV_0) {
        Some(bevy::mesh::VertexAttributeValues::Float32x2(uv)) if uv.len() == vertex_count => {
            uv.clone()
        }
        _ => vec![[0.0; 2]; vertex_count],
    };
    let mut vertex_floats: Vec<f32> = Vec::with_capacity(vertex_count * 8);
    for i in 0..vertex_count {
        vertex_floats.extend_from_slice(&positions[i]);
        vertex_floats.extend_from_slice(&normals[i]);
        vertex_floats.extend_from_slice(&uvs[i]);
    }
    let skin = pack_skin_stream(&asset, vertex_count);
    (vertex_floats, indices, skin)
}

/// Joint indices (u8 or u16 x4) and weights (f32 x4) per vertex, when both are present and
/// cover every vertex.
fn pack_skin_stream(asset: &Mesh, vertex_count: usize) -> Option<Vec<SkinVertex>> {
    use bevy::mesh::VertexAttributeValues as V;
    let joints: Vec<[u16; 4]> = match asset.attribute(Mesh::ATTRIBUTE_JOINT_INDEX)? {
        V::Uint16x4(j) => j.clone(),
        V::Uint8x4(j) => j.iter().map(|j| j.map(u16::from)).collect(),
        _ => return None,
    };
    let weights: Vec<[f32; 4]> = match asset.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)? {
        V::Float32x4(w) => w.clone(),
        _ => return None,
    };
    if joints.len() != vertex_count || weights.len() != vertex_count {
        return None;
    }
    Some(
        joints
            .into_iter()
            .zip(weights)
            .map(|(j, w)| SkinVertex::new(j, w))
            .collect(),
    )
}

pub struct VulkanMeshPlugin;

impl Plugin for VulkanMeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Mesh>();
        app.init_vulkan_asset::<Mesh>();
    }
}
