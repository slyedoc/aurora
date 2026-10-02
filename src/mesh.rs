//! The ray tracer's mesh: the data is [`aurora_mesh`], re-exported here; this module builds
//! each [`AuroraMesh`]'s BLAS from its finest LOD, with the baked opacity micromap the mesh
//! carries.
//!
//! Bevy's `Mesh3d` is not traced. Geometry arrives as baked `.aurora_mesh` (aurora_files'
//! importers), or is built in code with [`AuroraMesh::from_shape`] / [`AuroraMesh::from_mesh`].

use std::sync::Arc;

pub use aurora_mesh::*;
use bevy::{ecs::system::SystemParamItem, prelude::*};

use crate::{
    blas::{BLAS, BlasBuildInput, GeometryDescr, SkinVertex, Vertex, build_blas_batch},
    render_buffer::BufferProvider,
    vulkan_asset::{VulkanAsset, VulkanAssetExt},
};
use ash::vk;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

impl VulkanAsset for AuroraMesh {
    type ExtractedAsset = AuroraMesh;
    type ExtractParam = ();
    type PreparedAsset = BLAS;

    fn extract_asset(
        &self,
        _param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        Some(self.clone())
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        Self::prepare_batch(vec![asset], render_device)
            .pop()
            .unwrap()
    }

    /// Flattens every mesh in parallel, then builds all their BLASes with a handful of shared
    /// queue submissions (see `build_blas_batch`).
    fn prepare_batch(
        assets: Vec<Self::ExtractedAsset>,
        render_device: &crate::render_device::RenderDevice,
    ) -> Vec<Self::PreparedAsset> {
        let packed: Vec<(
            Vec<f32>,
            Vec<u32>,
            Option<Vec<SkinVertex>>,
            Option<Arc<OmmSlices>>,
        )> = assets
            .into_par_iter()
            .map(|mesh| {
                let omm = mesh.omm_slices().map(Arc::new);
                let (vertices, indices, skin) = pack_vertex_streams(mesh.flatten());
                (vertices, indices, skin, omm)
            })
            .collect();
        let inputs = packed
            .into_iter()
            .map(|(vertex_floats, indices, skin, omm)| {
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
/// interleaved as 8 floats per vertex, plus the indices and the skinning influences.
fn pack_vertex_streams(flat: FlatMesh) -> (Vec<f32>, Vec<u32>, Option<Vec<SkinVertex>>) {
    let mut vertex_floats: Vec<f32> = Vec::with_capacity(flat.positions.len() * 8);
    for i in 0..flat.positions.len() {
        vertex_floats.extend_from_slice(&flat.positions[i].to_array());
        vertex_floats.extend_from_slice(&flat.normals[i].to_array());
        vertex_floats.extend_from_slice(&flat.uvs[i].to_array());
    }
    let skin = flat.skin.map(|(joints, weights)| {
        joints
            .into_iter()
            .zip(weights)
            .map(|(j, w)| SkinVertex::new(j, w.to_array()))
            .collect()
    });
    (vertex_floats, flat.indices, skin)
}

pub struct AuroraMeshPlugin;

impl Plugin for AuroraMeshPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<AuroraMeshTypesPlugin>() {
            app.add_plugins(AuroraMeshTypesPlugin);
        }
        app.init_vulkan_asset::<AuroraMesh>();
        // Never traced, but the ecosystem builds from it -- avian's collider cache reads
        // `AssetEvent<Mesh>` and panics without the asset type registered.
        if !app.world().contains_resource::<Assets<Mesh>>() {
            app.init_asset::<Mesh>();
        }
    }
}
