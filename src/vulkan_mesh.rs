use bevy::{
    prelude::*,
    render::{RenderApp, mesh::Indices},
};

use crate::{
    blas::{BLAS, GeometryDescr, Vertex, build_blas_from_buffers},
    extract::Extract,
    ray_render_plugin::ExtractedEntity,
    render_buffer::BufferProvider,
    vulkan_asset::{VulkanAsset, VulkanAssetExt},
};
use ash::vk;

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
        let vertex_count = asset.count_vertices();
        assert!(matches!(asset.indices(), Some(Indices::U32(_))));
        let index_count = match asset.indices() {
            Some(Indices::U32(indices)) => indices.len(),
            Some(Indices::U16(indices)) => indices.len(),
            None => panic!("Mesh has no indices"),
        };

        // Pack exactly the three streams the shaders' `Vertex` (types.glsl) reads, whatever
        // other attributes the mesh carries (tangents, joints, ...).
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
        let index_data = asset.get_index_buffer_bytes().unwrap();

        let mut vertex_buffer_host = render_device.create_host_buffer::<Vertex>(
            vertex_count as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );

        let mut index_buffer_host = render_device.create_host_buffer::<u32>(
            index_count as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );

        let mut vertex_view = render_device.map_buffer(&mut vertex_buffer_host);
        vertex_view.copy_from_slice(bytemuck::cast_slice(&vertex_floats));
        let mut index_view = render_device.map_buffer(&mut index_buffer_host);
        index_view.copy_from_slice(bytemuck::cast_slice(&index_data));

        build_blas_from_buffers(
            render_device,
            vertex_count,
            index_count,
            vertex_buffer_host,
            index_buffer_host,
            &[GeometryDescr {
                first_vertex: 0,
                vertex_count,
                first_index: 0,
                index_count,
            }],
        )
    }

    fn destroy_asset(
        render_device: &crate::render_device::RenderDevice,
        prepared_asset: &Self::PreparedAsset,
    ) {
        prepared_asset.destroy(render_device);
    }
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
