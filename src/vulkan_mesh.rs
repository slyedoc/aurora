use bevy::{
    prelude::*,
    render::{RenderApp, mesh::Indices},
};

use crate::{
    blas::{BLAS, BlasBuildInput, GeometryDescr, Vertex, build_blas_batch},
    extract::Extract,
    ray_render_plugin::ExtractedEntity,
    render_buffer::BufferProvider,
    vulkan_asset::{VulkanAsset, VulkanAssetExt},
};
use ash::vk;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl VulkanAsset for Mesh {
    type ExtractedAsset = Mesh;
    type ExtractParam = ();
    type PreparedAsset = BLAS;

    fn extract_asset(
        &self,
        _param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        Some(self.clone())
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        Self::prepare_batch(vec![asset], render_device).pop().unwrap()
    }

    /// Packs every mesh's vertex streams in parallel, then builds all their BLASes with a
    /// handful of shared queue submissions (see `build_blas_batch`).
    fn prepare_batch(
        assets: Vec<Self::ExtractedAsset>,
        render_device: &crate::render_device::RenderDevice,
    ) -> Vec<Self::PreparedAsset> {
        let packed: Vec<(Vec<f32>, Vec<u32>)> =
            assets.into_par_iter().map(pack_vertex_streams).collect();
        let inputs = packed
            .into_iter()
            .map(|(vertex_floats, indices)| {
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
/// interleaved as 8 floats per vertex, plus u32 indices, whatever other attributes the mesh
/// carries (tangents, joints, ...).
fn pack_vertex_streams(asset: Mesh) -> (Vec<f32>, Vec<u32>) {
    let vertex_count = asset.count_vertices();
    let indices: Vec<u32> = match asset.indices() {
        Some(Indices::U32(indices)) => indices.clone(),
        Some(Indices::U16(indices)) => indices.iter().map(|&i| i as u32).collect(),
        None => panic!("Mesh has no indices"),
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
    (vertex_floats, indices)
}

pub struct VulkanMeshPlugin;

fn extract_meshes(
    mut commands: Commands,
    meshes: Extract<
        Query<(
            &Mesh3d,
            &MeshMaterial3d<StandardMaterial>,
            &Transform,
            &GlobalTransform,
        )>,
    >,
) {
    for (mesh, mat, t, gt) in meshes.iter() {
        commands.spawn((
            ExtractedEntity,
            mesh.clone(),
            mat.clone(),
            t.clone(),
            gt.clone(),
        ));
    }
}

impl Plugin for VulkanMeshPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<Mesh>();
        app.init_vulkan_asset::<Mesh>();
        app.init_asset::<StandardMaterial>();
        app.init_vulkan_asset::<StandardMaterial>();

        let render_app = app.get_sub_app_mut(RenderApp).unwrap();
        render_app.add_systems(ExtractSchedule, extract_meshes);
    }
}
