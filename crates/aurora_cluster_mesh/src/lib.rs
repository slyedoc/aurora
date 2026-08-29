//! The `.cluster_mesh` asset: aurora's baked-geometry wire format, read into a plain [`Mesh`].
//!
//! Re-exported by the engine as `bevy_aurora::cluster_mesh`; the `aurora_files` importers use this
//! crate directly.
//!
//! The format (magic `CLUSTERS`, versions 2..=4) is a fixed header followed by an lz4 frame of
//! length-prefixed POD slices: vertex positions / octahedral normals / tangents / uvs, cluster-local
//! indices, the cluster tables and LOD DAG, then optional opacity-micromap and skinning slices.
//! It is frozen — the `aurora_files` importers and thousands of shipped meshes depend on it byte
//! for byte — so the layouts here mirror `bevy_aurora_old::geometry::asset` exactly.
//!
//! The aurora renderer has no cluster acceleration structures: [`ClusterMeshData::to_mesh`] takes the
//! finest LOD's clusters and rebuilds an indexed triangle [`Mesh`] for the ordinary BLAS path, and
//! [`ClusterMeshData::from_mesh_flat`] writes a single-LOD cluster set so the importers can keep
//! emitting the same format without aurora's DAG bake (and such files still load in aurora).

use std::io::{Read, Write};
use std::sync::Arc;

use bevy::{
    asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader},
    math::{Vec2, Vec3, Vec4},
    mesh::{Indices, Mesh, PrimitiveTopology},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use thiserror::Error;

/// ASCII `"CLUSTERS"` interpreted little-endian.
const CLUSTER_MESH_ASSET_MAGIC: u64 = u64::from_le_bytes(*b"CLUSTERS");
/// Current format version (v3 added OMM slices, v4 the skin slices).
pub const CLUSTER_MESH_ASSET_VERSION: u64 = 4;
const CLUSTER_MESH_ASSET_MIN_VERSION: u64 = 2;
const CLUSTER_MESH_ASSET_OMM_VERSION: u64 = 3;
const CLUSTER_MESH_ASSET_SKIN_VERSION: u64 = 4;

/// NV cluster limits the flat writer respects, so the files stay valid cluster input for aurora.
const MAX_CLUSTER_TRIANGLES: usize = 128;
const MAX_CLUSTER_VERTICES: usize = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
pub struct Cluster {
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub triangle_count: u32,
    pub bounds_sphere: [f32; 4],
    pub local_material_id: u32,
    pub lod_level: u32,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
pub struct ClusterLodGroup {
    pub cluster_start: u32,
    pub cluster_count: u32,
    pub children_offset: u32,
    pub children_count: u32,
    pub traversal_sphere: [f32; 4],
    pub max_quadric_error: f32,
    pub parent_group: u32,
    pub lod_level: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
pub struct ClusterBvhNode {
    pub traversal_sphere: [f32; 4],
    pub max_quadric_error: f32,
    pub children_offset: u32,
    pub children_packed: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
pub struct ClusterMeshAabb {
    pub center: [f32; 4],
    pub half_extent: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
pub struct ClusterBloatAabb {
    pub min: [f32; 4],
    pub max: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct OmmDesc {
    pub offset: u32,
    pub subdivision_level: u16,
    pub format: u16,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq, Eq)]
pub struct OmmUsage {
    pub count: u32,
    pub subdivision_level: u16,
    pub format: u16,
}

/// Everything a `.cluster_mesh` file holds. Slices the backend does not use (the LOD DAG, OMMs,
/// skinning) are kept so a file round-trips unchanged.
#[derive(Clone, Debug, Default)]
pub struct ClusterMeshData {
    pub vertex_positions: Arc<[Vec3]>,
    /// Octahedral normals packed as 2x16 snorm.
    pub vertex_normals: Arc<[u32]>,
    pub vertex_tangents: Arc<[Vec4]>,
    pub vertex_uvs: Arc<[Vec2]>,
    /// Cluster-local indices, 3 per triangle.
    pub indices: Arc<[u32]>,
    pub clusters: Arc<[Cluster]>,
    pub groups: Arc<[ClusterLodGroup]>,
    pub nodes: Arc<[ClusterBvhNode]>,
    pub child_table: Arc<[u32]>,
    pub cluster_to_group: Arc<[u32]>,
    pub aabb: ClusterMeshAabb,
    pub mesh_max_error: f32,
    pub root_group_id: u32,
    pub root_node_id: u32,
    pub lod_levels: u32,
    pub omm_array_data: Arc<[u8]>,
    pub omm_descs: Arc<[OmmDesc]>,
    pub omm_index: Arc<[i32]>,
    pub omm_usage: Arc<[OmmUsage]>,
    pub omm_index_usage: Arc<[OmmUsage]>,
    pub vertex_joint_indices: Arc<[[u16; 4]]>,
    pub vertex_joint_weights: Arc<[Vec4]>,
    pub cluster_bloat_aabbs: Arc<[ClusterBloatAabb]>,
    pub inverse_bind_count: u32,
}

#[derive(Error, Debug)]
pub enum ClusterMeshError {
    #[error("file was not a ClusterMesh asset")]
    WrongFileType,
    #[error("expected asset version {CLUSTER_MESH_ASSET_VERSION} but found version {found}")]
    WrongVersion { found: u64 },
    #[error("failed to compress or decompress asset data")]
    Compression(#[from] lz4_flex::frame::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("mesh has no {0} attribute")]
    MissingAttribute(&'static str),
}

// ---------------------------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------------------------

pub fn read_cluster_mesh_sync<R: Read>(mut reader: R) -> Result<ClusterMeshData, ClusterMeshError> {
    let magic = read_u64(&mut reader)?;
    if magic != CLUSTER_MESH_ASSET_MAGIC {
        return Err(ClusterMeshError::WrongFileType);
    }
    let version = read_u64(&mut reader)?;
    if !(CLUSTER_MESH_ASSET_MIN_VERSION..=CLUSTER_MESH_ASSET_VERSION).contains(&version) {
        return Err(ClusterMeshError::WrongVersion { found: version });
    }

    let mut bytes = [0u8; size_of::<ClusterMeshAabb>()];
    reader.read_exact(&mut bytes)?;
    let aabb = bytemuck::cast(bytes);
    let mesh_max_error = f32::from_le_bytes(read_4(&mut reader)?);
    let root_group_id = u32::from_le_bytes(read_4(&mut reader)?);
    let root_node_id = u32::from_le_bytes(read_4(&mut reader)?);
    let lod_levels = u32::from_le_bytes(read_4(&mut reader)?);

    let mut reader = FrameDecoder::new(reader);
    let reader: &mut dyn Read = &mut reader;

    let vertex_positions = read_slice(reader)?;
    let vertex_normals = read_slice(reader)?;
    let vertex_tangents = read_slice(reader)?;
    let vertex_uvs = read_slice(reader)?;
    let indices = read_slice(reader)?;
    let clusters = read_slice(reader)?;
    let groups = read_slice(reader)?;
    let nodes = read_slice(reader)?;
    let child_table = read_slice(reader)?;
    let cluster_to_group = read_slice(reader)?;
    let (omm_array_data, omm_descs, omm_index, omm_usage, omm_index_usage) =
        if version >= CLUSTER_MESH_ASSET_OMM_VERSION {
            (
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
                read_slice(reader)?,
            )
        } else {
            Default::default()
        };
    let (vertex_joint_indices, vertex_joint_weights, cluster_bloat_aabbs, inverse_bind_count) =
        if version >= CLUSTER_MESH_ASSET_SKIN_VERSION {
            let joints = read_slice(reader)?;
            let weights = read_slice(reader)?;
            let bloat = read_slice(reader)?;
            let count: Arc<[u32]> = read_slice(reader)?;
            (joints, weights, bloat, count.first().copied().unwrap_or(0))
        } else {
            Default::default()
        };

    Ok(ClusterMeshData {
        vertex_positions,
        vertex_normals,
        vertex_tangents,
        vertex_uvs,
        indices,
        clusters,
        groups,
        nodes,
        child_table,
        cluster_to_group,
        aabb,
        mesh_max_error,
        root_group_id,
        root_node_id,
        lod_levels,
        omm_array_data,
        omm_descs,
        omm_index,
        omm_usage,
        omm_index_usage,
        vertex_joint_indices,
        vertex_joint_weights,
        cluster_bloat_aabbs,
        inverse_bind_count,
    })
}

/// Writes the current (v4) format, byte-identical to aurora's `write_cluster_mesh_sync`.
pub fn write_cluster_mesh_sync<W: Write>(
    asset: &ClusterMeshData,
    mut writer: W,
) -> Result<(), ClusterMeshError> {
    writer.write_all(&CLUSTER_MESH_ASSET_MAGIC.to_le_bytes())?;
    writer.write_all(&CLUSTER_MESH_ASSET_VERSION.to_le_bytes())?;
    writer.write_all(bytemuck::bytes_of(&asset.aabb))?;
    writer.write_all(&asset.mesh_max_error.to_le_bytes())?;
    writer.write_all(&asset.root_group_id.to_le_bytes())?;
    writer.write_all(&asset.root_node_id.to_le_bytes())?;
    writer.write_all(&asset.lod_levels.to_le_bytes())?;

    let mut encoder = FrameEncoder::new(writer);
    write_slice(&asset.vertex_positions, &mut encoder)?;
    write_slice(&asset.vertex_normals, &mut encoder)?;
    write_slice(&asset.vertex_tangents, &mut encoder)?;
    write_slice(&asset.vertex_uvs, &mut encoder)?;
    write_slice(&asset.indices, &mut encoder)?;
    write_slice(&asset.clusters, &mut encoder)?;
    write_slice(&asset.groups, &mut encoder)?;
    write_slice(&asset.nodes, &mut encoder)?;
    write_slice(&asset.child_table, &mut encoder)?;
    write_slice(&asset.cluster_to_group, &mut encoder)?;
    write_slice(&asset.omm_array_data, &mut encoder)?;
    write_slice(&asset.omm_descs, &mut encoder)?;
    write_slice(&asset.omm_index, &mut encoder)?;
    write_slice(&asset.omm_usage, &mut encoder)?;
    write_slice(&asset.omm_index_usage, &mut encoder)?;
    write_slice(&asset.vertex_joint_indices, &mut encoder)?;
    write_slice(&asset.vertex_joint_weights, &mut encoder)?;
    write_slice(&asset.cluster_bloat_aabbs, &mut encoder)?;
    write_slice(&[asset.inverse_bind_count], &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

fn read_u64(reader: &mut dyn Read) -> Result<u64, std::io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_4(reader: &mut dyn Read) -> Result<[u8; 4], std::io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn write_slice<T: Pod>(field: &[T], writer: &mut dyn Write) -> Result<(), std::io::Error> {
    writer.write_all(&(field.len() as u64).to_le_bytes())?;
    writer.write_all(bytemuck::cast_slice(field))?;
    Ok(())
}

fn read_slice<T: Pod>(reader: &mut dyn Read) -> Result<Arc<[T]>, std::io::Error> {
    let len = read_u64(reader)? as usize;
    let mut data: Arc<[T]> = core::iter::repeat_with(T::zeroed).take(len).collect();
    let slice = Arc::get_mut(&mut data).unwrap();
    reader.read_exact(bytemuck::cast_slice_mut(slice))?;
    Ok(data)
}

// ---------------------------------------------------------------------------------------------
// Normals: octahedral, packed 2x16 snorm (matches aurora's bake / `unpack2x16snorm` decode)
// ---------------------------------------------------------------------------------------------

fn octahedral_encode(v: Vec3) -> Vec2 {
    let n = v / (v.x.abs() + v.y.abs() + v.z.abs());
    let wrap = (1.0 - n.yx().abs())
        * Vec2::new(
            if n.x >= 0.0 { 1.0 } else { -1.0 },
            if n.y >= 0.0 { 1.0 } else { -1.0 },
        );
    if n.z >= 0.0 { n.xy() } else { wrap }
}

fn octahedral_decode(v: Vec2) -> Vec3 {
    let mut n = Vec3::new(v.x, v.y, 1.0 - v.x.abs() - v.y.abs());
    let t = (-n.z).max(0.0);
    n.x += if n.x >= 0.0 { -t } else { t };
    n.y += if n.y >= 0.0 { -t } else { t };
    n.normalize_or_zero()
}

fn pack_2x16_snorm(v: Vec2) -> u32 {
    let x = (v.x.clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16 as u32;
    let y = (v.y.clamp(-1.0, 1.0) * 32767.0).round() as i16 as u16 as u32;
    x | (y << 16)
}

fn unpack_2x16_snorm(p: u32) -> Vec2 {
    let x = (p & 0xFFFF) as u16 as i16 as f32 / 32767.0;
    let y = (p >> 16) as u16 as i16 as f32 / 32767.0;
    Vec2::new(x.clamp(-1.0, 1.0), y.clamp(-1.0, 1.0))
}

pub fn pack_normal(n: Vec3) -> u32 {
    pack_2x16_snorm(octahedral_encode(n.normalize_or_zero()))
}

pub fn unpack_normal(p: u32) -> Vec3 {
    octahedral_decode(unpack_2x16_snorm(p))
}

// ---------------------------------------------------------------------------------------------
// Mesh conversion
// ---------------------------------------------------------------------------------------------

impl ClusterMeshData {
    /// Rebuilds an indexed triangle mesh from the finest LOD (the `lod_level == 0` clusters, or
    /// every cluster if the file carries no LOD levels).
    pub fn to_mesh(&self) -> Mesh {
        let finest: Vec<&Cluster> = {
            let lod0: Vec<&Cluster> = self.clusters.iter().filter(|c| c.lod_level == 0).collect();
            if lod0.is_empty() {
                self.clusters.iter().collect()
            } else {
                lod0
            }
        };

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        let has_normals = self.vertex_normals.len() == self.vertex_positions.len();
        let has_uvs = self.vertex_uvs.len() == self.vertex_positions.len();
        let has_tangents = self.vertex_tangents.len() == self.vertex_positions.len();

        for cluster in finest {
            let base = positions.len() as u32;
            let v0 = cluster.vertex_offset as usize;
            let v1 = v0 + cluster.vertex_count as usize;
            positions.extend(self.vertex_positions[v0..v1].iter().map(|p| p.to_array()));
            if has_normals {
                normals.extend(
                    self.vertex_normals[v0..v1]
                        .iter()
                        .map(|&n| unpack_normal(n).to_array()),
                );
            }
            if has_uvs {
                uvs.extend(self.vertex_uvs[v0..v1].iter().map(|uv| uv.to_array()));
            }
            if has_tangents {
                tangents.extend(self.vertex_tangents[v0..v1].iter().map(|t| t.to_array()));
            }
            let i0 = cluster.index_offset as usize;
            let i1 = i0 + 3 * cluster.triangle_count as usize;
            indices.extend(self.indices[i0..i1].iter().map(|&i| base + i));
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        if has_normals {
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        }
        if has_uvs {
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        }
        if has_tangents {
            mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, tangents);
        }
        mesh.insert_indices(Indices::U32(indices));
        if !has_normals {
            mesh.compute_normals();
        }
        mesh
    }

    /// Chunks a triangle mesh into a single-LOD cluster set (≤128 triangles / ≤256 vertices per
    /// cluster, one root group, no BVH nodes) — a valid file for both this backend and aurora.
    pub fn from_mesh_flat(mesh: &Mesh) -> Result<Self, ClusterMeshError> {
        let positions: Vec<Vec3> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .ok_or(ClusterMeshError::MissingAttribute("position"))?
            .iter()
            .map(|p| Vec3::from(*p))
            .collect();
        let normals: Vec<Vec3> = match mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(|a| a.as_float3())
        {
            Some(n) => n.iter().map(|n| Vec3::from(*n)).collect(),
            None => {
                let mut m = mesh.clone();
                m.compute_normals();
                m.attribute(Mesh::ATTRIBUTE_NORMAL)
                    .and_then(|a| a.as_float3())
                    .ok_or(ClusterMeshError::MissingAttribute("normal"))?
                    .iter()
                    .map(|n| Vec3::from(*n))
                    .collect()
            }
        };
        let uvs: Vec<Vec2> = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(bevy::mesh::VertexAttributeValues::Float32x2(uv)) => {
                uv.iter().map(|uv| Vec2::from(*uv)).collect()
            }
            _ => vec![Vec2::ZERO; positions.len()],
        };
        let tangents: Vec<Vec4> = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
            Some(bevy::mesh::VertexAttributeValues::Float32x4(t)) => {
                t.iter().map(|t| Vec4::from(*t)).collect()
            }
            _ => vec![Vec4::new(1.0, 0.0, 0.0, 1.0); positions.len()],
        };
        let source_indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(i)) => i.clone(),
            Some(Indices::U16(i)) => i.iter().map(|&i| i as u32).collect(),
            None => (0..positions.len() as u32).collect(),
        };

        let mut out_positions = Vec::new();
        let mut out_normals = Vec::new();
        let mut out_tangents = Vec::new();
        let mut out_uvs = Vec::new();
        let mut out_indices = Vec::new();
        let mut clusters = Vec::new();

        // Greedy chunking: each cluster owns its own (deduplicated) vertex range.
        let mut tri = 0;
        let triangle_count = source_indices.len() / 3;
        while tri < triangle_count {
            let vertex_offset = out_positions.len() as u32;
            let index_offset = out_indices.len() as u32;
            let mut local: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            let mut cluster_tris = 0;
            while tri < triangle_count && cluster_tris < MAX_CLUSTER_TRIANGLES {
                let corners = &source_indices[tri * 3..tri * 3 + 3];
                let new_verts = corners.iter().filter(|c| !local.contains_key(*c)).count();
                if local.len() + new_verts > MAX_CLUSTER_VERTICES {
                    break;
                }
                for &c in corners {
                    let next = local.len() as u32;
                    let idx = *local.entry(c).or_insert_with(|| {
                        out_positions.push(positions[c as usize]);
                        out_normals.push(pack_normal(normals[c as usize]));
                        out_tangents.push(tangents[c as usize]);
                        out_uvs.push(uvs[c as usize]);
                        next
                    });
                    out_indices.push(idx);
                }
                cluster_tris += 1;
                tri += 1;
            }
            let verts = &out_positions[vertex_offset as usize..];
            clusters.push(Cluster {
                vertex_offset,
                vertex_count: verts.len() as u32,
                index_offset,
                triangle_count: cluster_tris as u32,
                bounds_sphere: bounding_sphere(verts),
                local_material_id: 0,
                lod_level: 0,
                _pad: [0; 2],
            });
        }

        let (min, max) = positions.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        let (min, max) = if positions.is_empty() {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (min, max)
        };
        let center = 0.5 * (min + max);
        let half = 0.5 * (max - min);
        let group = ClusterLodGroup {
            cluster_start: 0,
            cluster_count: clusters.len() as u32,
            children_offset: 0,
            children_count: 0,
            traversal_sphere: [center.x, center.y, center.z, half.length()],
            max_quadric_error: 0.0,
            parent_group: u32::MAX,
            lod_level: 0,
            _pad: 0,
        };
        let cluster_to_group = vec![0u32; clusters.len()];

        Ok(Self {
            vertex_positions: out_positions.into(),
            vertex_normals: out_normals.into(),
            vertex_tangents: out_tangents.into(),
            vertex_uvs: out_uvs.into(),
            indices: out_indices.into(),
            clusters: clusters.into(),
            groups: vec![group].into(),
            nodes: Arc::from(&[][..]),
            child_table: Arc::from(&[][..]),
            cluster_to_group: cluster_to_group.into(),
            aabb: ClusterMeshAabb {
                center: [center.x, center.y, center.z, 0.0],
                half_extent: [half.x, half.y, half.z, 0.0],
            },
            mesh_max_error: 0.0,
            root_group_id: 0,
            root_node_id: u32::MAX,
            lod_levels: 1,
            ..Default::default()
        })
    }
}

fn bounding_sphere(points: &[Vec3]) -> [f32; 4] {
    if points.is_empty() {
        return [0.0; 4];
    }
    let (min, max) = points.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let center = 0.5 * (min + max);
    let radius = points
        .iter()
        .map(|p| p.distance(center))
        .fold(0.0f32, f32::max);
    [center.x, center.y, center.z, radius]
}

// ---------------------------------------------------------------------------------------------
// Asset loader
// ---------------------------------------------------------------------------------------------

/// Loads `.cluster_mesh` files as [`Mesh`] assets, so `Mesh3d("x.cluster_mesh")` in a `.bsn`
/// resolves straight into the BLAS path.
#[derive(TypePath, Default)]
pub struct ClusterMeshLoader;

impl AssetLoader for ClusterMeshLoader {
    type Asset = Mesh;
    type Settings = ();
    type Error = ClusterMeshError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Mesh, ClusterMeshError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let data = read_cluster_mesh_sync(bytes.as_slice())?;
        Ok(data.to_mesh())
    }

    fn extensions(&self) -> &[&str] {
        &["cluster_mesh"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_round_trip() {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        let n = 1000usize;
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32, (i * 7 % 13) as f32, 0.0])
            .collect();
        let indices: Vec<u32> = (0..n as u32 - 2).flat_map(|i| [i, i + 1, i + 2]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.5f32, 0.25]; n]);
        mesh.insert_indices(Indices::U32(indices.clone()));

        let data = ClusterMeshData::from_mesh_flat(&mesh).unwrap();
        assert!(
            data.clusters
                .iter()
                .all(|c| c.triangle_count as usize <= MAX_CLUSTER_TRIANGLES)
        );
        assert!(
            data.clusters
                .iter()
                .all(|c| c.vertex_count as usize <= MAX_CLUSTER_VERTICES)
        );
        let total: u32 = data.clusters.iter().map(|c| c.triangle_count).sum();
        assert_eq!(total as usize, indices.len() / 3);

        let mut bytes = Vec::new();
        write_cluster_mesh_sync(&data, &mut bytes).unwrap();
        let back = read_cluster_mesh_sync(bytes.as_slice()).unwrap();
        assert_eq!(back.clusters.len(), data.clusters.len());
        assert_eq!(back.indices.len(), data.indices.len());

        let rebuilt = back.to_mesh();
        let tri_count = match rebuilt.indices() {
            Some(Indices::U32(i)) => i.len() / 3,
            _ => 0,
        };
        assert_eq!(tri_count, indices.len() / 3);
        // Every rebuilt triangle has the same positions as the source triangle.
        let rebuilt_pos = rebuilt
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        if let Some(Indices::U32(ri)) = rebuilt.indices() {
            for (t, tri) in ri.chunks(3).enumerate() {
                for k in 0..3 {
                    assert_eq!(
                        rebuilt_pos[tri[k] as usize],
                        positions[indices[t * 3 + k] as usize]
                    );
                }
            }
        }
    }

    #[test]
    fn normal_pack_round_trip() {
        for n in [
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            let back = unpack_normal(pack_normal(n));
            assert!(n.dot(back) > 0.999, "{n} -> {back}");
        }
    }
}

#[cfg(test)]
mod shipped_file_tests {
    use super::*;

    /// Reads a real aurora-baked file when `AURORA_CLUSTER_MESH_SAMPLE` points at one.
    #[test]
    fn reads_shipped_file() {
        let Ok(path) = std::env::var("AURORA_CLUSTER_MESH_SAMPLE") else {
            return;
        };
        let file = std::fs::File::open(&path).unwrap();
        let data = read_cluster_mesh_sync(std::io::BufReader::new(file)).unwrap();
        assert!(!data.clusters.is_empty(), "{path}: no clusters");
        let mesh = data.to_mesh();
        let tris = match mesh.indices() {
            Some(Indices::U32(i)) => i.len() / 3,
            _ => 0,
        };
        let lod0: u32 = data
            .clusters
            .iter()
            .filter(|c| c.lod_level == 0)
            .map(|c| c.triangle_count)
            .sum();
        assert_eq!(tris as u32, lod0, "{path}: LOD0 triangle count");
        assert!(mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some());
        assert!(mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some());
        eprintln!(
            "{path}: {} clusters, {} lod levels, {} LOD0 triangles, {} vertices",
            data.clusters.len(),
            data.lod_levels,
            tris,
            mesh.count_vertices()
        );
    }
}
