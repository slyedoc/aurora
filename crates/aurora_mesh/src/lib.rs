//! The mesh as data: the [`AuroraMesh`] asset, its `.aurora_mesh` file format, and the
//! [`AuroraMesh3d`] component that places one in the world.
//!
//! Re-exported by the engine as `bevy_aurora::mesh`, which builds the BLAS; the `aurora_files`
//! importers use this crate directly.
//!
//! The format (magic `AURAMESH`, version 1) is a fixed header followed by an lz4 frame of
//! length-prefixed POD slices: vertex positions / octahedral normals / tangents / uvs,
//! cluster-local indices, the cluster tables and LOD DAG, the opacity-micromap slices, then the
//! skin slices. The clusters respect NVIDIA's cluster acceleration structure limits when baked by
//! [`AuroraMesh::clustered`]; the renderer today traces the finest LOD as one ordinary BLAS.
//!
//! Bevy's [`Mesh`] is an import format here, not something the tracer sees: an importer bakes it
//! with [`AuroraMesh::clustered`]. A mesh built at runtime is made from its streams with
//! [`AuroraMesh::from_triangles`], or converted with [`AuroraMesh::from_mesh`].

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

use aurora_material::{AuroraMaterial3d, AuroraMaterialTypesPlugin};
use bevy::{
    asset::{AssetLoader, LoadContext, RenderAssetUsages, io::Reader},
    ecs::template::FromTemplate,
    math::{Vec2, Vec3, Vec4},
    mesh::{Indices, Mesh, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use lz4_flex::frame::{FrameDecoder, FrameEncoder};
use thiserror::Error;

/// ASCII `"AURAMESH"` interpreted little-endian.
const AURORA_MESH_MAGIC: u64 = u64::from_le_bytes(*b"AURAMESH");
pub const AURORA_MESH_VERSION: u64 = 1;

/// NV cluster limits [`AuroraMesh::clustered`] respects.
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
pub struct MeshAabb {
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

/// A ray-traced mesh: everything a `.aurora_mesh` file holds. Slices the renderer does not use
/// yet (the LOD DAG) are kept so a file round-trips unchanged.
// Reflect-opaque: never authored field by field, but it must be `Reflect` so
// `AuroraMesh3d("path")` resolves its handle from a `.bsn` string.
#[derive(Asset, Clone, Debug, Default, Reflect)]
#[reflect(opaque)]
#[type_path = "bevy_aurora::mesh"]
pub struct AuroraMesh {
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
    pub aabb: MeshAabb,
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

/// The mesh of a ray-traced entity. A material is required alongside; an entity spawned
/// without one gets `AuroraMaterial3d::default()`, whose handle holds `AuroraMaterial::default()`.
#[derive(Component, FromTemplate, Clone, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq, FromTemplate)]
#[template(reflect)]
#[require(AuroraMaterial3d, Transform, Visibility)]
#[type_path = "bevy_aurora::mesh"]
pub struct AuroraMesh3d(pub Handle<AuroraMesh>);

#[derive(Error, Debug)]
pub enum AuroraMeshError {
    #[error("file was not an AuroraMesh")]
    WrongFileType,
    #[error("expected version {AURORA_MESH_VERSION} but found version {found}")]
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

/// The header fields ahead of the compressed body.
struct Header {
    aabb: MeshAabb,
    mesh_max_error: f32,
    root_group_id: u32,
    root_node_id: u32,
    lod_levels: u32,
}

fn read_header(reader: &mut dyn Read) -> Result<Header, std::io::Error> {
    let mut bytes = [0u8; size_of::<MeshAabb>()];
    reader.read_exact(&mut bytes)?;
    Ok(Header {
        aabb: bytemuck::cast(bytes),
        mesh_max_error: f32::from_le_bytes(read_4(reader)?),
        root_group_id: u32::from_le_bytes(read_4(reader)?),
        root_node_id: u32::from_le_bytes(read_4(reader)?),
        lod_levels: u32::from_le_bytes(read_4(reader)?),
    })
}

fn read_body<R: Read>(header: Header, reader: R) -> Result<AuroraMesh, AuroraMeshError> {
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
    let omm_array_data = read_slice(reader)?;
    let omm_descs = read_slice(reader)?;
    let omm_index = read_slice(reader)?;
    let omm_usage = read_slice(reader)?;
    let omm_index_usage = read_slice(reader)?;
    let vertex_joint_indices = read_slice(reader)?;
    let vertex_joint_weights = read_slice(reader)?;
    let cluster_bloat_aabbs = read_slice(reader)?;
    let inverse_bind_count: Arc<[u32]> = read_slice(reader)?;
    let inverse_bind_count = inverse_bind_count.first().copied().unwrap_or(0);

    Ok(AuroraMesh {
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
        aabb: header.aabb,
        mesh_max_error: header.mesh_max_error,
        root_group_id: header.root_group_id,
        root_node_id: header.root_node_id,
        lod_levels: header.lod_levels,
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

pub fn read_aurora_mesh<R: Read>(mut reader: R) -> Result<AuroraMesh, AuroraMeshError> {
    if read_u64(&mut reader)? != AURORA_MESH_MAGIC {
        return Err(AuroraMeshError::WrongFileType);
    }
    let version = read_u64(&mut reader)?;
    if version != AURORA_MESH_VERSION {
        return Err(AuroraMeshError::WrongVersion { found: version });
    }
    let header = read_header(&mut reader)?;
    read_body(header, reader)
}

pub fn write_aurora_mesh<W: Write>(
    asset: &AuroraMesh,
    mut writer: W,
) -> Result<(), AuroraMeshError> {
    writer.write_all(&AURORA_MESH_MAGIC.to_le_bytes())?;
    writer.write_all(&AURORA_MESH_VERSION.to_le_bytes())?;
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
// Normals: octahedral, packed 2x16 snorm (the shaders decode with `unpack2x16snorm`)
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
// Flattening: the finest LOD as one indexed triangle list
// ---------------------------------------------------------------------------------------------

/// The finest LOD as one indexed triangle list, in the order [`AuroraMesh::flatten`] emits the
/// clusters. This is what a BLAS is built from.
#[derive(Clone, Debug, Default)]
pub struct FlatMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub tangents: Vec<Vec4>,
    /// Joint indices and weights, when the mesh carries a skin covering every vertex.
    pub skin: Option<(Vec<[u16; 4]>, Vec<Vec4>)>,
    pub indices: Vec<u32>,
}

/// A mesh's baked opacity micromap, in the triangle order [`AuroraMesh::flatten`] emits: the
/// `VkMicromapEXT` build inputs (`array_data`, `descs` = `VkMicromapTriangleEXT`s, `usage`)
/// plus the per-triangle index the BLAS geometry references it through (negative = the
/// `VK_OPACITY_MICROMAP_SPECIAL_INDEX_*` uniform states) and that index's usage histogram.
#[derive(Clone, Debug)]
pub struct OmmSlices {
    pub array_data: Arc<[u8]>,
    pub descs: Arc<[OmmDesc]>,
    pub usage: Arc<[OmmUsage]>,
    pub index: Vec<i32>,
    pub index_usage: Vec<OmmUsage>,
}

impl AuroraMesh {
    /// The clusters [`Self::flatten`] emits, in emission order: the `lod_level == 0` clusters,
    /// or every cluster if the mesh carries no LOD levels.
    fn finest_clusters(&self) -> Vec<&Cluster> {
        let lod0: Vec<&Cluster> = self.clusters.iter().filter(|c| c.lod_level == 0).collect();
        if lod0.is_empty() {
            self.clusters.iter().collect()
        } else {
            lod0
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertex_positions.len()
    }

    pub fn has_omm(&self) -> bool {
        !self.omm_array_data.is_empty() && !self.omm_descs.is_empty()
    }

    pub fn has_skin(&self) -> bool {
        self.vertex_joint_indices.len() == self.vertex_positions.len()
            && self.vertex_joint_weights.len() == self.vertex_positions.len()
            && !self.vertex_positions.is_empty()
    }

    /// The finest LOD as one triangle list. A mesh without normals gets flat ones.
    pub fn flatten(&self) -> FlatMesh {
        let n = self.vertex_positions.len();
        let has_normals = self.vertex_normals.len() == n;
        let has_uvs = self.vertex_uvs.len() == n;
        let has_tangents = self.vertex_tangents.len() == n;
        let has_skin = self.has_skin();

        let mut flat = FlatMesh::default();
        let mut joints = Vec::new();
        let mut weights = Vec::new();
        for cluster in self.finest_clusters() {
            let base = flat.positions.len() as u32;
            let v0 = cluster.vertex_offset as usize;
            let v1 = v0 + cluster.vertex_count as usize;
            flat.positions
                .extend_from_slice(&self.vertex_positions[v0..v1]);
            if has_normals {
                flat.normals.extend(
                    self.vertex_normals[v0..v1]
                        .iter()
                        .map(|&n| unpack_normal(n)),
                );
            }
            flat.uvs.extend((v0..v1).map(|v| {
                if has_uvs {
                    self.vertex_uvs[v]
                } else {
                    Vec2::ZERO
                }
            }));
            flat.tangents.extend((v0..v1).map(|v| {
                if has_tangents {
                    self.vertex_tangents[v]
                } else {
                    Vec4::new(1.0, 0.0, 0.0, 1.0)
                }
            }));
            if has_skin {
                joints.extend_from_slice(&self.vertex_joint_indices[v0..v1]);
                weights.extend_from_slice(&self.vertex_joint_weights[v0..v1]);
            }
            let i0 = cluster.index_offset as usize;
            let i1 = i0 + 3 * cluster.triangle_count as usize;
            flat.indices
                .extend(self.indices[i0..i1].iter().map(|&i| base + i));
        }
        if !has_normals {
            flat.normals = flat_normals(&flat.positions, &flat.indices);
        }
        if has_skin {
            flat.skin = Some((joints, weights));
        }
        flat
    }

    /// The baked opacity micromap re-indexed to [`Self::flatten`]'s triangle order (the file's
    /// index is per cluster triangle, in cluster order). `None` when the mesh has no OMM or no
    /// emitted triangle references one.
    pub fn omm_slices(&self) -> Option<OmmSlices> {
        if !self.has_omm() {
            return None;
        }
        let mut index = Vec::new();
        for cluster in self.finest_clusters() {
            let t0 = cluster.index_offset as usize / 3;
            let t1 = t0 + cluster.triangle_count as usize;
            if t1 > self.omm_index.len() {
                return None;
            }
            index.extend_from_slice(&self.omm_index[t0..t1]);
        }
        let mut histogram: HashMap<(u16, u16), u32> = HashMap::new();
        for &i in &index {
            if i >= 0 {
                let desc = self.omm_descs.get(i as usize)?;
                *histogram
                    .entry((desc.subdivision_level, desc.format))
                    .or_default() += 1;
            }
        }
        if histogram.is_empty() {
            return None;
        }
        let mut index_usage: Vec<OmmUsage> = histogram
            .into_iter()
            .map(|((subdivision_level, format), count)| OmmUsage {
                count,
                subdivision_level,
                format,
            })
            .collect();
        index_usage.sort_by_key(|u| (u.subdivision_level, u.format));
        Some(OmmSlices {
            array_data: Arc::clone(&self.omm_array_data),
            descs: Arc::clone(&self.omm_descs),
            usage: Arc::clone(&self.omm_usage),
            index,
            index_usage,
        })
    }

    /// Attaches a baked opacity micromap whose `index` runs over the triangles of
    /// [`Self::indices`] in order -- one entry per triangle, as [`Self::omm_slices`] returns
    /// it for a [`Self::from_mesh`] mesh. A mesh assembled at runtime from baked parts (tiled
    /// clutter) carries its parts' micromaps this way.
    pub fn with_omm(mut self, omm: OmmSlices) -> Self {
        self.omm_array_data = omm.array_data;
        self.omm_descs = omm.descs;
        self.omm_usage = omm.usage;
        self.omm_index = omm.index.into();
        self.omm_index_usage = omm.index_usage.into();
        self
    }

    /// The finest LOD as a bevy [`Mesh`], for tools that want one (physics, export).
    pub fn to_mesh(&self) -> Mesh {
        let flat = self.flatten();
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            flat.positions
                .iter()
                .map(|p| p.to_array())
                .collect::<Vec<_>>(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            flat.normals
                .iter()
                .map(|n| n.to_array())
                .collect::<Vec<_>>(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            flat.uvs.iter().map(|uv| uv.to_array()).collect::<Vec<_>>(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_TANGENT,
            flat.tangents
                .iter()
                .map(|t| t.to_array())
                .collect::<Vec<_>>(),
        );
        if let Some((joints, weights)) = flat.skin {
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_INDEX,
                VertexAttributeValues::Uint16x4(joints),
            );
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_JOINT_WEIGHT,
                weights.iter().map(|w| w.to_array()).collect::<Vec<_>>(),
            );
        }
        mesh.insert_indices(Indices::U32(flat.indices));
        mesh
    }
}

fn flat_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| positions[i as usize]);
        let n = (b - a).cross(c - a);
        for &i in tri {
            normals[i as usize] += n;
        }
    }
    normals
        .into_iter()
        .map(|n| n.try_normalize().unwrap_or(Vec3::Y))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// From bevy's Mesh
// ---------------------------------------------------------------------------------------------

/// Triangle-list streams for [`AuroraMesh::from_triangles`]. A vertex stream left empty (or not
/// one entry per position) takes its default: flat normals, zero uvs, +X tangents. `indices`
/// names every triangle; empty draws nothing.
#[derive(Clone, Debug, Default)]
pub struct Triangles {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub tangents: Vec<Vec4>,
    pub indices: Vec<u32>,
}

/// A mesh's streams, read once for either conversion.
struct SourceStreams {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    tangents: Vec<Vec4>,
    skin: Option<(Vec<[u16; 4]>, Vec<Vec4>)>,
    indices: Vec<u32>,
}

impl SourceStreams {
    fn from_triangles(t: Triangles) -> Self {
        let n = t.positions.len();
        let indices = t.indices;
        let normals = if t.normals.len() == n {
            t.normals
        } else {
            flat_normals(&t.positions, &indices)
        };
        Self {
            uvs: if t.uvs.len() == n {
                t.uvs
            } else {
                vec![Vec2::ZERO; n]
            },
            tangents: if t.tangents.len() == n {
                t.tangents
            } else {
                vec![Vec4::new(1.0, 0.0, 0.0, 1.0); n]
            },
            normals,
            positions: t.positions,
            skin: None,
            indices,
        }
    }

    fn read(mesh: &Mesh) -> Result<Self, AuroraMeshError> {
        let positions: Vec<Vec3> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(|a| a.as_float3())
            .ok_or(AuroraMeshError::MissingAttribute("position"))?
            .iter()
            .map(|p| Vec3::from(*p))
            .collect();
        let indices: Vec<u32> = match mesh.indices() {
            Some(Indices::U32(i)) => i.clone(),
            Some(Indices::U16(i)) => i.iter().map(|&i| i as u32).collect(),
            None => (0..positions.len() as u32).collect(),
        };
        let normals: Vec<Vec3> = match mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(|a| a.as_float3())
        {
            Some(n) if n.len() == positions.len() => n.iter().map(|n| Vec3::from(*n)).collect(),
            _ => flat_normals(&positions, &indices),
        };
        let uvs: Vec<Vec2> = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(uv)) if uv.len() == positions.len() => {
                uv.iter().map(|uv| Vec2::from(*uv)).collect()
            }
            _ => vec![Vec2::ZERO; positions.len()],
        };
        let tangents: Vec<Vec4> = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
            Some(VertexAttributeValues::Float32x4(t)) if t.len() == positions.len() => {
                t.iter().map(|t| Vec4::from(*t)).collect()
            }
            _ => vec![Vec4::new(1.0, 0.0, 0.0, 1.0); positions.len()],
        };
        // Both skin streams must cover every vertex or neither is kept: half a palette skins
        // garbage.
        let joints: Option<Vec<[u16; 4]>> = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
            Some(VertexAttributeValues::Uint16x4(j)) => Some(j.clone()),
            Some(VertexAttributeValues::Uint8x4(j)) => {
                Some(j.iter().map(|j| j.map(u16::from)).collect())
            }
            _ => None,
        };
        let weights: Option<Vec<Vec4>> = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) {
            Some(VertexAttributeValues::Float32x4(w)) => {
                Some(w.iter().map(|w| Vec4::from(*w)).collect())
            }
            _ => None,
        };
        let skin = match (joints, weights) {
            (Some(j), Some(w)) if j.len() == positions.len() && w.len() == positions.len() => {
                Some((j, w))
            }
            _ => None,
        };
        Ok(Self {
            positions,
            normals,
            uvs,
            tangents,
            skin,
            indices,
        })
    }
}

fn aabb_of(positions: &[Vec3]) -> MeshAabb {
    if positions.is_empty() {
        return MeshAabb::default();
    }
    let (min, max) = positions.iter().fold(
        (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let center = 0.5 * (min + max);
    let half = 0.5 * (max - min);
    MeshAabb {
        center: [center.x, center.y, center.z, 0.0],
        half_extent: [half.x, half.y, half.z, 0.0],
    }
}

fn single_root_group(cluster_count: usize, aabb: &MeshAabb) -> ClusterLodGroup {
    let half = Vec3::new(
        aabb.half_extent[0],
        aabb.half_extent[1],
        aabb.half_extent[2],
    );
    ClusterLodGroup {
        cluster_start: 0,
        cluster_count: cluster_count as u32,
        children_offset: 0,
        children_count: 0,
        traversal_sphere: [
            aabb.center[0],
            aabb.center[1],
            aabb.center[2],
            half.length(),
        ],
        max_quadric_error: 0.0,
        parent_group: u32::MAX,
        lod_level: 0,
        _pad: 0,
    }
}

impl AuroraMesh {
    /// One cluster holding the mesh exactly as given: same vertices, same order, same indices.
    /// For meshes built at runtime, whose owners may address vertices by index (terrain edits
    /// heights in place). It is not valid NV cluster input past 256 vertices; bake files with
    /// [`Self::clustered`].
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, AuroraMeshError> {
        Ok(Self::from_streams(SourceStreams::read(mesh)?))
    }

    /// [`Self::from_mesh`] for streams built in code, without a bevy [`Mesh`] in between.
    pub fn from_triangles(triangles: Triangles) -> Self {
        Self::from_streams(SourceStreams::from_triangles(triangles))
    }

    fn from_streams(s: SourceStreams) -> Self {
        let aabb = aabb_of(&s.positions);
        let cluster = Cluster {
            vertex_offset: 0,
            vertex_count: s.positions.len() as u32,
            index_offset: 0,
            triangle_count: (s.indices.len() / 3) as u32,
            bounds_sphere: bounding_sphere(&s.positions),
            local_material_id: 0,
            lod_level: 0,
            _pad: [0; 2],
        };
        let (joints, weights) = s.skin.unzip();
        Self {
            vertex_normals: s.normals.iter().map(|&n| pack_normal(n)).collect(),
            vertex_positions: s.positions.into(),
            vertex_tangents: s.tangents.into(),
            vertex_uvs: s.uvs.into(),
            vertex_joint_indices: joints.unwrap_or_default().into(),
            vertex_joint_weights: weights.unwrap_or_default().into(),
            indices: s.indices.into(),
            clusters: vec![cluster].into(),
            groups: vec![single_root_group(1, &aabb)].into(),
            cluster_to_group: vec![0u32].into(),
            aabb,
            root_node_id: u32::MAX,
            lod_levels: 1,
            ..Default::default()
        }
    }

    /// Replaces the triangles of a single-cluster mesh ([`Self::from_mesh`],
    /// [`Self::from_triangles`]) over the same vertices: a surface that re-picks which of its
    /// vertices to draw (a terrain level moving its hole).
    ///
    /// # Panics
    /// On a clustered mesh, whose indices are cluster-local.
    pub fn set_indices(&mut self, indices: Vec<u32>) {
        assert_eq!(
            self.clusters.len(),
            1,
            "set_indices needs a single-cluster mesh"
        );
        let mut clusters = self.clusters.to_vec();
        clusters[0].triangle_count = (indices.len() / 3) as u32;
        self.clusters = clusters.into();
        self.indices = indices.into();
    }

    /// [`Self::from_mesh`] for anything that builds a bevy [`Mesh`]: a primitive
    /// (`Cuboid::new(..)`), a mesh builder (`Plane3d::default().mesh().size(..)`), or a `Mesh`.
    ///
    /// # Panics
    /// If the mesh has no positions, which a shape always has.
    pub fn from_shape(shape: impl Into<Mesh>) -> Self {
        Self::from_mesh(&shape.into()).expect("a shape has positions")
    }

    /// Chunks a triangle mesh into a single-LOD cluster set within NV's cluster limits (at most
    /// 128 triangles and 256 vertices per cluster, one root group, no BVH nodes). What an
    /// importer writes to disk.
    pub fn clustered(mesh: &Mesh) -> Result<Self, AuroraMeshError> {
        let s = SourceStreams::read(mesh)?;

        let mut out_positions = Vec::new();
        let mut out_normals = Vec::new();
        let mut out_tangents = Vec::new();
        let mut out_uvs = Vec::new();
        let mut out_joints: Vec<[u16; 4]> = Vec::new();
        let mut out_weights: Vec<Vec4> = Vec::new();
        let mut out_indices = Vec::new();
        let mut clusters = Vec::new();

        // Greedy chunking: each cluster owns its own (deduplicated) vertex range.
        let mut tri = 0;
        let triangle_count = s.indices.len() / 3;
        while tri < triangle_count {
            let vertex_offset = out_positions.len() as u32;
            let index_offset = out_indices.len() as u32;
            let mut local: HashMap<u32, u32> = HashMap::new();
            let mut cluster_tris = 0;
            while tri < triangle_count && cluster_tris < MAX_CLUSTER_TRIANGLES {
                let corners = &s.indices[tri * 3..tri * 3 + 3];
                let new_verts = corners.iter().filter(|c| !local.contains_key(*c)).count();
                if local.len() + new_verts > MAX_CLUSTER_VERTICES {
                    break;
                }
                for &c in corners {
                    let next = local.len() as u32;
                    let idx = *local.entry(c).or_insert_with(|| {
                        let c = c as usize;
                        out_positions.push(s.positions[c]);
                        out_normals.push(pack_normal(s.normals[c]));
                        out_tangents.push(s.tangents[c]);
                        out_uvs.push(s.uvs[c]);
                        if let Some((j, w)) = &s.skin {
                            out_joints.push(j[c]);
                            out_weights.push(w[c]);
                        }
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

        let aabb = aabb_of(&s.positions);
        let cluster_to_group = vec![0u32; clusters.len()];
        Ok(Self {
            vertex_positions: out_positions.into(),
            vertex_normals: out_normals.into(),
            vertex_tangents: out_tangents.into(),
            vertex_uvs: out_uvs.into(),
            vertex_joint_indices: out_joints.into(),
            vertex_joint_weights: out_weights.into(),
            indices: out_indices.into(),
            groups: vec![single_root_group(clusters.len(), &aabb)].into(),
            clusters: clusters.into(),
            cluster_to_group: cluster_to_group.into(),
            aabb,
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
// Asset loader and plugin
// ---------------------------------------------------------------------------------------------

/// Loads `.aurora_mesh` files, so `AuroraMesh3d("x.aurora_mesh")` in a `.bsn` resolves.
#[derive(TypePath, Default)]
pub struct AuroraMeshLoader;

impl AssetLoader for AuroraMeshLoader {
    type Asset = AuroraMesh;
    type Settings = ();
    type Error = AuroraMeshError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<AuroraMesh, AuroraMeshError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        read_aurora_mesh(bytes.as_slice())
    }

    fn extensions(&self) -> &[&str] {
        &["aurora_mesh"]
    }
}

/// The asset, its loader and the component, so a scene naming a mesh loads. Nothing here
/// draws; `bevy_aurora::mesh::AuroraMeshPlugin` adds this and builds the BLAS.
pub struct AuroraMeshTypesPlugin;

impl Plugin for AuroraMeshTypesPlugin {
    fn build(&self, app: &mut App) {
        // `AuroraMesh3d` requires a material.
        if !app.is_plugin_added::<AuroraMaterialTypesPlugin>() {
            app.add_plugins(AuroraMaterialTypesPlugin);
        }
        app.init_asset::<AuroraMesh>()
            .register_asset_loader(AuroraMeshLoader)
            .register_type::<AuroraMesh3d>()
            .register_type::<AuroraMesh3dTemplate>()
            .register_asset_reflect::<AuroraMesh>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(n: usize) -> (Mesh, Vec<[f32; 3]>, Vec<u32>) {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        let positions: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32, (i * 7 % 13) as f32, 0.0])
            .collect();
        let indices: Vec<u32> = (0..n as u32 - 2).flat_map(|i| [i, i + 1, i + 2]).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.5f32, 0.25]; n]);
        mesh.insert_indices(Indices::U32(indices.clone()));
        (mesh, positions, indices)
    }

    #[test]
    fn clustered_round_trip() {
        let (mesh, positions, indices) = strip(1000);
        let data = AuroraMesh::clustered(&mesh).unwrap();
        assert!(
            data.clusters
                .iter()
                .all(|c| c.triangle_count as usize <= MAX_CLUSTER_TRIANGLES
                    && c.vertex_count as usize <= MAX_CLUSTER_VERTICES)
        );
        let total: u32 = data.clusters.iter().map(|c| c.triangle_count).sum();
        assert_eq!(total as usize, indices.len() / 3);

        let mut bytes = Vec::new();
        write_aurora_mesh(&data, &mut bytes).unwrap();
        let back = read_aurora_mesh(bytes.as_slice()).unwrap();
        assert_eq!(back.clusters.len(), data.clusters.len());

        // Every flattened triangle has the source triangle's positions.
        let flat = back.flatten();
        assert_eq!(flat.indices.len(), indices.len());
        for (t, tri) in flat.indices.chunks(3).enumerate() {
            for k in 0..3 {
                assert_eq!(
                    flat.positions[tri[k] as usize].to_array(),
                    positions[indices[t * 3 + k] as usize]
                );
            }
        }
    }

    /// Terrain addresses vertices by grid index, so a runtime conversion keeps them in place.
    #[test]
    fn from_mesh_keeps_vertex_order_and_count() {
        let (mesh, positions, indices) = strip(1000);
        let flat = AuroraMesh::from_mesh(&mesh).unwrap().flatten();
        assert_eq!(flat.indices, indices);
        assert_eq!(
            flat.positions
                .iter()
                .map(|p| p.to_array())
                .collect::<Vec<_>>(),
            positions
        );
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

    #[test]
    fn type_paths() {
        assert_eq!(AuroraMesh::type_path(), "bevy_aurora::mesh::AuroraMesh");
        assert_eq!(AuroraMesh3d::type_path(), "bevy_aurora::mesh::AuroraMesh3d");
    }

    #[test]
    fn triangles_keep_their_order_and_default_the_rest() {
        let mesh = AuroraMesh::from_triangles(Triangles {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(1.0, 0.0, 1.0)],
            indices: vec![0, 2, 1, 1, 2, 3],
            ..default()
        });
        assert_eq!(&*mesh.indices, &[0, 2, 1, 1, 2, 3]);
        let none = AuroraMesh::from_triangles(Triangles {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Z],
            ..default()
        });
        assert!(none.indices.is_empty(), "no indices is no triangles");
        assert_eq!(mesh.vertex_uvs.len(), 4);
        assert!(unpack_normal(mesh.vertex_normals[0]).abs_diff_eq(Vec3::Y, 1e-3));
        assert_eq!(mesh.clusters.len(), 1);
    }

    #[test]
    fn set_indices_redraws_the_same_vertices() {
        let mut mesh = AuroraMesh::from_triangles(Triangles {
            positions: vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::new(1.0, 0.0, 1.0)],
            indices: vec![0, 2, 1, 1, 2, 3],
            ..default()
        });
        mesh.set_indices(vec![1, 2, 3]);
        assert_eq!(&*mesh.indices, &[1, 2, 3]);
        assert_eq!(mesh.clusters[0].triangle_count, 1);
        assert_eq!(mesh.vertex_positions.len(), 4);
    }
}
