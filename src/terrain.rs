//! GPU-editable heightmap terrain (core's tile system, ported onto the ray tracer).
//!
//! A terrain tile is an entity with `AuroraMesh3d` (built by [`terrain_mesh`] from the tile's height
//! grid — the same center-vertex pattern core used, no folded quads for RT) plus a
//! [`TerrainTile`] component carrying the editable state: the height grid, the WoW-style
//! alphamap atlas (16×16 subchunk cells, up to 4 texture layers per chunk — layer 0 base,
//! layers 1..3 weighted by the cell's R/G/B) and the per-chunk layer table. The closest-hit
//! shader blends the palette textures per hit with those weights (WoW's sequential lerp), at
//! full texture resolution and the ray cone's mip.
//!
//! Rendering rides the skinning pattern: the tile gets its own deformed vertex/triangle
//! streams and a per-instance BLAS (`ALLOW_UPDATE`), the TLAS slot is overridden to it and
//! the SBT hit record points at the streams. Heights and alphas live in HOST-VISIBLE buffers
//! that only compute kernels (terrain.slang) write:
//!
//!   brush op  →  terrain_brush / terrain_paint  →  terrain_vertices → terrain_pack → refit
//!                                              ↘  alpha atlas, read by the closest-hit
//!
//! Ops are queued from gameplay in [`TerrainEdits`] (tile-local coordinates). After edits the
//! host buffers are mirrored back into the component two frames later (the in-flight fence
//! guarantees the GPU is done) and [`TerrainHeightsSynced`] ticks — colliders and saves hang
//! off that. The splat palette is global in [`TerrainPalette`]: its textures become ordinary
//! mipmapped images, sampled bindless by the hit shader.

use std::collections::HashMap;
use std::mem::offset_of;

use ash::vk;
use bevy::{
    asset::{AssetId, RenderAssetUsages},
    ecs::lifecycle::Remove,
    ecs::observer::On,
    mesh::{Indices, Mesh, PrimitiveTopology},
    prelude::*,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::{AccelerationStructure, Triangle, Vertex, allocate_acceleration_structure},
    compute::{
        ComputeModule, ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch,
    },
    mesh::{AuroraMesh, AuroraMesh3d},
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::{GpuInstance, InstanceOverride, TLAS, prepare_instances},
    vk_utils,
    vulkan_asset::{VulkanAssets, poll_for_asset},
};

/// No texture in a chunk-layer slot (matches terrain.slang's NO_LAYER).
pub const NO_LAYER: u32 = u32::MAX;
/// Frames between full BLAS rebuilds; refits in between (edits drift trace quality slowly).
const REBUILD_INTERVAL: u32 = 16;

// ---- authoring-side components / resources ---------------------------------------------------

/// One terrain tile's editable state. Spawn together with `AuroraMesh3d(terrain_mesh(..))` and an
/// `AuroraMaterial3d` (its roughness and factors apply; the colour is the splat).
#[derive(Component)]
pub struct TerrainTile {
    /// Height grid edge (vertices), e.g. 129.
    pub resolution: u32,
    /// Tile world size (mesh spans ±size/2 in x/z, heights are absolute y).
    pub size: f32,
    /// Height grid, row-major `z * resolution + x`. GPU-owned after spawn; mirrored back
    /// here after edits (watch [`TerrainHeightsSynced`]).
    pub heights: Vec<f32>,
    /// Per-chunk palette indices (16×16 = 256 entries), [`NO_LAYER`] = unused slot.
    /// CPU-owned: mutate and the table re-uploads.
    pub chunk_layers: Vec<[u32; 4]>,
    /// Alphamap atlas, RGBA8, `alpha_atlas`² texels: 16×16 cells of `alpha_cell`² per chunk.
    /// GPU-owned after spawn; mirrored back like `heights`.
    pub alpha: Vec<u8>,
    pub alpha_atlas: u32,
    pub alpha_cell: u32,
}

/// Ticks after edited heights/alphas have been mirrored back into [`TerrainTile`] —
/// rebuild colliders / enable saving on `Changed<TerrainHeightsSynced>`.
#[derive(Component, Default)]
pub struct TerrainHeightsSynced(pub u32);

/// One tileset texture of the splat palette, as raw RGBA bytes.
pub struct TerrainPaletteEntry {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// Texture repeats per subchunk cell edge (WoW ground textures ≈ 8).
    pub repeats: f32,
}

/// The global splat palette. Fill `entries` before spawning tiles; each becomes an image in
/// `images` (static thereafter).
#[derive(Resource, Default)]
pub struct TerrainPalette {
    pub entries: Vec<TerrainPaletteEntry>,
    /// The entries as images, made by aurora; the hit shader samples them.
    pub images: Vec<Handle<Image>>,
}

/// What a brush stroke does this frame.
#[derive(Clone, Copy, Debug)]
pub enum TerrainBrushKind {
    /// Signed height delta at full falloff (already dt-scaled by the caller).
    Raise(f32),
    /// Neighbor-average lerp factor 0..1.
    Smooth(f32),
    /// Lerp toward `target` height.
    Flatten { target: f32, amount: f32 },
    /// Paint one PALETTE TEXTURE: each chunk under the brush resolves which of its layer
    /// slots holds it (add it to `TerrainTile::chunk_layers` before the stroke).
    Paint { texture: u32, amount: f32 },
    /// Scale all splat weights down (expose the base layer).
    Erase(f32),
}

/// A brush application on one tile, in TILE-LOCAL x/z.
#[derive(Clone, Copy, Debug)]
pub struct TerrainBrushOp {
    pub center: Vec2,
    pub radius: f32,
    pub kind: TerrainBrushKind,
}

/// Queue of brush ops for this frame, written by gameplay, drained at extract.
#[derive(Resource, Default)]
pub struct TerrainEdits {
    pub ops: Vec<(Entity, TerrainBrushOp)>,
}

// ---- push-constant mirrors (terrain.slang) ----------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushParams {
    heights: u64,
    res: u32,
    size: f32,
    cx: f32,
    cz: f32,
    radius: f32,
    amount: f32,
    target: f32,
    mode: u32,
    base: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VertParams {
    heights: u64,
    dst: u64,
    res: u32,
    size: f32,
    count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PackParams {
    vertices: u64,
    indices: u64,
    triangles: u64,
    count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PaintParams {
    alpha: u64,
    chunk_layers: u64,
    atlas: u32,
    cell: u32,
    size: f32,
    cx: f32,
    cz: f32,
    radius: f32,
    amount: f32,
    tex: u32,
    base: u32,
    _pad: u32,
}

/// What the terrain closest-hit reads for a tile (`TerrainShade` in closest_hit.rchit, scalar).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainShadeGpu {
    alpha: u64,
    chunk_layers: u64,
    /// `[bindless texture index, repeats bits]` per palette entry.
    palette: u64,
    atlas: u32,
    cell: u32,
    size: f32,
    palette_len: u32,
}

// ---- device state -----------------------------------------------------------------------------

#[derive(Clone, PartialEq)]
struct TerrainSource {
    mesh: AssetId<AuroraMesh>,
    resolution: u32,
    size: f32,
    alpha_atlas: u32,
    alpha_cell: u32,
}

struct TerrainGpu {
    /// The mesh BLAS the streams were copied from; a different handle = asset reloaded.
    mesh_vertex_buffer: vk::Buffer,
    vertex_count: u32,
    triangle_count: u32,
    /// Host-visible, GPU-edited state.
    heights: Buffer<f32>,
    alpha: Buffer<u32>,
    chunk_layers: Buffer<[u32; 4]>,
    /// Deformed stream + shading records (device).
    vertices: Buffer<Vertex>,
    triangles: Buffer<Triangle>,
    index_buffer: Buffer<u32>,
    geometry_to_index: u64,
    geometry_to_triangle: u64,
    /// What the hit shader reads for this tile (host-visible, rewritten each prepare).
    shade: Buffer<TerrainShadeGpu>,
    blas: AccelerationStructure,
    scratch: Buffer<u8>,
    scratch_alignment: u64,
    builds: u32,
    hit_offset: u32,
}

impl TerrainGpu {
    fn destroy(&self, rd: &RenderDevice) {
        self.blas.destroy(rd);
        for b in [
            self.heights.handle,
            self.alpha.handle,
            self.chunk_layers.handle,
            self.vertices.handle,
            self.triangles.handle,
            self.index_buffer.handle,
            self.shade.handle,
            self.scratch.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
    }

    fn geometry(&self) -> vk::AccelerationStructureGeometryKHR<'static> {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.vertices.address,
                    })
                    .vertex_stride(std::mem::size_of::<Vertex>() as u64)
                    .max_vertex(self.vertex_count)
                    .index_type(vk::IndexType::UINT32)
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.index_buffer.address,
                    })
                    .transform_data(vk::DeviceOrHostAddressConstKHR { device_address: 0 }),
            })
    }
}

const BLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR =
    vk::BuildAccelerationStructureFlagsKHR::from_raw(
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD.as_raw()
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw()
            | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS.as_raw(),
    );

struct TerrainInstance {
    entity: Entity,
    source: TerrainSource,
    gpu: Option<TerrainGpu>,
    /// Pending kernel work.
    ops: Vec<TerrainBrushOp>,
    geo_dirty: bool,
    /// CPU chunk-layer table changed: re-map the host buffer.
    layers_dirty: bool,
    /// Mirror heights/alpha back into the component at this frame (edit frame + 2:
    /// one frame in flight means that submission's fence has been waited by then).
    sync_at: Option<u32>,
}

/// What the SBT writes for a terrain instance (terrain streams are static addresses).
pub struct TerrainHitRecord {
    pub hit_offset: u32,
    pub vertex_buffer: u64,
    pub triangle_buffer: u64,
    pub index_buffer: u64,
    pub geometry_to_index: u64,
    pub geometry_to_triangle: u64,
    /// The tile's [`TerrainShadeGpu`]; 0 until the palette's images are uploaded.
    pub surface_data: u64,
}

#[derive(Resource)]
pub struct Terrains {
    module: Handle<ComputeModule>,
    instances: HashMap<u32, TerrainInstance>,
    removed: Vec<u32>,
    /// `[texture index, repeats bits]` per palette entry, once its images are uploaded.
    palette: Option<Buffer<[u32; 2]>>,
    frame: u32,
    warned_module: bool,
}

impl Terrains {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            instances: HashMap::new(),
            removed: Vec::new(),
            palette: None,
            frame: 0,
            warned_module: false,
        }
    }

    /// Hit records for every terrain instance with device state.
    pub fn hit_records(&self) -> impl Iterator<Item = TerrainHitRecord> + '_ {
        self.instances.values().filter_map(|i| {
            let gpu = i.gpu.as_ref()?;
            Some(TerrainHitRecord {
                hit_offset: gpu.hit_offset,
                vertex_buffer: gpu.vertices.address,
                triangle_buffer: gpu.triangles.address,
                index_buffer: gpu.index_buffer.address,
                geometry_to_index: gpu.geometry_to_index,
                geometry_to_triangle: gpu.geometry_to_triangle,
                surface_data: if self.palette.is_some() {
                    gpu.shade.address
                } else {
                    0
                },
            })
        })
    }

    /// Records this frame's brush kernels, stream rebuilds and BLAS refits. Call after `Skins::record`, before `TLAS::record`. Returns whether any
    /// BLAS changed (the TLAS must rebuild).
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
    ) -> bool {
        if self.instances.values().all(|i| i.gpu.is_none()) {
            return false;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("terrain.slang not compiled yet; terrain tiles wait");
                self.warned_module = true;
            }
            return false;
        };
        // Brush / paint kernels, conservatively barriered between ops (few per frame).
        let mut any_ops = false;
        for inst in self.instances.values_mut() {
            let Some(gpu) = inst.gpu.as_ref() else {
                inst.ops.clear();
                continue;
            };
            for op in inst.ops.drain(..) {
                if any_ops {
                    compute_to_compute_barrier(rd, cmd);
                }
                any_ops = true;
                match op.kind {
                    TerrainBrushKind::Raise(_)
                    | TerrainBrushKind::Smooth(_)
                    | TerrainBrushKind::Flatten { .. } => {
                        let (mode, amount, target) = match op.kind {
                            TerrainBrushKind::Raise(a) => (0, a, 0.0),
                            TerrainBrushKind::Smooth(a) => (1, a, 0.0),
                            TerrainBrushKind::Flatten { target, amount } => (2, amount, target),
                            _ => unreachable!(),
                        };
                        let params = BrushParams {
                            heights: gpu.heights.address,
                            res: inst.source.resolution,
                            size: inst.source.size,
                            cx: op.center.x,
                            cz: op.center.y,
                            radius: op.radius,
                            amount,
                            target,
                            mode,
                            base: 0,
                            _pad: 0,
                        };
                        let texels = inst.source.resolution * inst.source.resolution;
                        record_dispatch(
                            rd,
                            cmd,
                            module,
                            "terrain_brush",
                            &params,
                            texels,
                            Some(offset_of!(BrushParams, base)),
                        );
                        inst.geo_dirty = true;
                    }
                    TerrainBrushKind::Paint { .. } | TerrainBrushKind::Erase(_) => {
                        let (tex, amount) = match op.kind {
                            TerrainBrushKind::Paint { texture, amount } => (texture, amount),
                            TerrainBrushKind::Erase(amount) => (u32::MAX - 1, amount),
                            _ => unreachable!(),
                        };
                        let params = PaintParams {
                            alpha: gpu.alpha.address,
                            chunk_layers: gpu.chunk_layers.address,
                            atlas: inst.source.alpha_atlas,
                            cell: inst.source.alpha_cell,
                            size: inst.source.size,
                            cx: op.center.x,
                            cz: op.center.y,
                            radius: op.radius,
                            amount,
                            tex,
                            base: 0,
                            _pad: 0,
                        };
                        let texels = inst.source.alpha_atlas * inst.source.alpha_atlas;
                        record_dispatch(
                            rd,
                            cmd,
                            module,
                            "terrain_paint",
                            &params,
                            texels,
                            Some(offset_of!(PaintParams, base)),
                        );
                    }
                }
                inst.sync_at = Some(self.frame.wrapping_add(2));
            }
        }
        if any_ops {
            compute_to_compute_barrier(rd, cmd);
            // Painted alphas are read by the hit shader this frame.
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ,
            );
        }

        // Stream rebuild + BLAS build/refit for geometry-dirty tiles.
        let mut deformed: Vec<u32> = Vec::new();
        for (&slot, inst) in self.instances.iter() {
            let Some(gpu) = inst.gpu.as_ref() else {
                continue;
            };
            if !(inst.geo_dirty || gpu.builds == 0) {
                continue;
            }
            let params = VertParams {
                heights: gpu.heights.address,
                dst: gpu.vertices.address,
                res: inst.source.resolution,
                size: inst.source.size,
                count: gpu.vertex_count,
                base: 0,
            };
            record_dispatch(
                rd,
                cmd,
                module,
                "terrain_vertices",
                &params,
                gpu.vertex_count,
                Some(offset_of!(VertParams, base)),
            );
            deformed.push(slot);
        }
        let mut any_blas = false;
        if !deformed.is_empty() {
            compute_to_compute_barrier(rd, cmd);
            for &slot in &deformed {
                let gpu = self.instances[&slot].gpu.as_ref().unwrap();
                let params = PackParams {
                    vertices: gpu.vertices.address,
                    indices: gpu.index_buffer.address,
                    triangles: gpu.triangles.address,
                    count: gpu.triangle_count,
                    base: 0,
                };
                record_dispatch(
                    rd,
                    cmd,
                    module,
                    "terrain_pack",
                    &params,
                    gpu.triangle_count,
                    Some(offset_of!(PackParams, base)),
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
            );

            let mut geometries: Vec<vk::AccelerationStructureGeometryKHR> = Vec::new();
            let mut ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR> = Vec::new();
            let mut modes: Vec<(vk::AccelerationStructureKHR, bool, u64)> = Vec::new();
            for &slot in &deformed {
                let inst = self.instances.get_mut(&slot).unwrap();
                let gpu = inst.gpu.as_mut().unwrap();
                let update = gpu.builds > 0 && gpu.builds % REBUILD_INTERVAL != 0;
                gpu.builds = gpu.builds.wrapping_add(1).max(1);
                inst.geo_dirty = false;
                geometries.push(gpu.geometry());
                ranges.push(
                    vk::AccelerationStructureBuildRangeInfoKHR::default()
                        .primitive_count(gpu.triangle_count),
                );
                modes.push((
                    gpu.blas.handle,
                    update,
                    vk_utils::aligned_size(gpu.scratch.address, gpu.scratch_alignment),
                ));
            }
            let infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = geometries
                .iter()
                .zip(&modes)
                .map(|(geometry, (handle, update, scratch))| {
                    let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .flags(BLAS_FLAGS)
                        .dst_acceleration_structure(*handle)
                        .geometries(std::slice::from_ref(geometry))
                        .scratch_data(vk::DeviceOrHostAddressKHR {
                            device_address: *scratch,
                        });
                    if *update {
                        info.mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                            .src_acceleration_structure(*handle)
                    } else {
                        info.mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    }
                })
                .collect();
            let range_refs: Vec<&[vk::AccelerationStructureBuildRangeInfoKHR]> =
                ranges.iter().map(std::slice::from_ref).collect();
            unsafe {
                rd.ext_acc_struct
                    .cmd_build_acceleration_structures(cmd, &infos, &range_refs);
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
                vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR,
            );
            any_blas = true;
        }

        any_blas
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for inst in self.instances.values_mut() {
            if let Some(gpu) = inst.gpu.take() {
                gpu.destroy(rd);
            }
        }
        self.instances.clear();
        if let Some(palette) = self.palette.take() {
            rd.destroyer.destroy_buffer(palette.handle);
        }
    }
}

// ---- extraction / preparation ------------------------------------------------------------------

type ChangedTerrains = Or<(
    Added<GpuInstance>,
    Changed<AuroraMesh3d>,
    Changed<TerrainTile>,
)>;

#[allow(clippy::type_complexity)]
fn extract_terrains(
    mut terrains: ResMut<Terrains>,
    mut edits: ResMut<TerrainEdits>,
    changed: Query<(Entity, &GpuInstance, &AuroraMesh3d, &TerrainTile), ChangedTerrains>,
    mut removed_components: RemovedComponents<TerrainTile>,
    instances: Query<&GpuInstance>,
) {
    for entity in removed_components.read() {
        if let Ok(instance) = instances.get(entity) {
            terrains.removed.push(instance.0);
        }
    }
    for (entity, instance, mesh, tile) in changed.iter() {
        let source = TerrainSource {
            mesh: mesh.0.id(),
            resolution: tile.resolution,
            size: tile.size,
            alpha_atlas: tile.alpha_atlas,
            alpha_cell: tile.alpha_cell,
        };
        match terrains.instances.get_mut(&instance.0) {
            Some(existing) if existing.source == source => {
                // The component was touched (a heights sync, or a chunk-table edit):
                // refresh the layer table; heights/alpha stay GPU-owned.
                existing.layers_dirty = true;
            }
            Some(existing) => {
                existing.source = source;
                terrains.removed.push(instance.0); // rebuild from scratch next prepare
            }
            None => {
                terrains.instances.insert(
                    instance.0,
                    TerrainInstance {
                        entity,
                        source,
                        gpu: None,
                        ops: Vec::new(),
                        geo_dirty: false,
                        layers_dirty: false,
                        sync_at: None,
                    },
                );
            }
        }
    }
    // Route this frame's brush ops to slots.
    for (entity, op) in edits.ops.drain(..) {
        let Ok(instance) = instances.get(entity) else {
            continue;
        };
        if let Some(inst) = terrains.instances.get_mut(&instance.0) {
            inst.ops.push(op);
        }
    }
}

fn on_instance_removed(
    remove: On<Remove<GpuInstance>>,
    instances: Query<&GpuInstance>,
    mut terrains: ResMut<Terrains>,
) {
    if let Ok(instance) = instances.get(remove.entity) {
        terrains.removed.push(instance.0);
    }
}

/// Mirror GPU-edited heights/alphas back into the component once the edit's submission is
/// certainly complete, and tick [`TerrainHeightsSynced`].
fn sync_terrain_cpu(
    mut commands: Commands,
    mut terrains: ResMut<Terrains>,
    render_device: Res<RenderDevice>,
    mut tiles: Query<(&mut TerrainTile, Option<&mut TerrainHeightsSynced>)>,
) {
    let frame = terrains.frame;
    for inst in terrains.instances.values_mut() {
        let due = inst
            .sync_at
            .is_some_and(|at| frame.wrapping_sub(at) < 0x8000_0000);
        if !due {
            continue;
        }
        let Some(gpu) = inst.gpu.as_mut() else {
            inst.sync_at = None;
            continue;
        };
        let Ok((mut tile, synced)) = tiles.get_mut(inst.entity) else {
            inst.sync_at = None;
            continue;
        };
        inst.sync_at = None;
        {
            let len = tile.heights.len();
            let view = render_device.map_buffer(&mut gpu.heights);
            let src = unsafe { std::slice::from_raw_parts(view.as_ptr(), len) };
            tile.heights.copy_from_slice(src);
        }
        {
            let len = tile.alpha.len() / 4;
            let view = render_device.map_buffer(&mut gpu.alpha);
            let words: &[u32] = unsafe { std::slice::from_raw_parts(view.as_ptr(), len) };
            tile.alpha.copy_from_slice(bytemuck::cast_slice(words));
        }
        match synced {
            Some(mut s) => s.0 = s.0.wrapping_add(1),
            None => {
                commands.entity(inst.entity).insert(TerrainHeightsSynced(1));
            }
        }
    }
}

/// The palette's entries as images (UNORM: the hit shader linearises, as for every colour
/// texture), once.
fn palette_images(mut palette: ResMut<TerrainPalette>, mut images: ResMut<Assets<Image>>) {
    if palette.entries.is_empty() || !palette.images.is_empty() {
        return;
    }
    let made: Vec<Handle<Image>> = palette
        .entries
        .iter()
        .map(|entry| {
            images.add(Image::new(
                wgpu_types::Extent3d {
                    width: entry.width,
                    height: entry.height,
                    depth_or_array_layers: 1,
                },
                wgpu_types::TextureDimension::D2,
                entry.rgba.clone(),
                wgpu_types::TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            ))
        })
        .collect();
    palette.images = made;
}

pub fn prepare_terrains(
    render_device: Res<RenderDevice>,
    mut terrains: ResMut<Terrains>,
    mut tlas: ResMut<TLAS>,
    meshes: Res<VulkanAssets<AuroraMesh>>,
    palette: Option<Res<TerrainPalette>>,
    images: Res<VulkanAssets<Image>>,
    tiles: Query<&TerrainTile>,
) {
    let terrains = &mut *terrains;
    terrains.frame = terrains.frame.wrapping_add(1);

    // The palette's table, once every image is uploaded (indices are bindless slots).
    if terrains.palette.is_none()
        && let Some(palette) = palette.as_ref()
        && !palette.images.is_empty()
        && palette.images.len() == palette.entries.len()
    {
        let table: Option<Vec<[u32; 2]>> = palette
            .images
            .iter()
            .zip(&palette.entries)
            .map(|(image, entry)| {
                let texture = images.get(image)?;
                Some([
                    render_device.register_bindless_texture(texture),
                    entry.repeats.to_bits(),
                ])
            })
            .collect();
        if let Some(table) = table {
            let mut buffer: Buffer<[u32; 2]> = render_device
                .create_host_buffer(table.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
            render_device
                .map_buffer(&mut buffer)
                .copy_from_slice(&table);
            terrains.palette = Some(buffer);
            log::info!("terrain: palette ready ({} textures)", table.len());
        }
    }
    let palette_table = terrains
        .palette
        .as_ref()
        .map_or((0, 0), |b| (b.address, b.nr_elements as u32));

    for slot in terrains.removed.drain(..) {
        tlas.set_override(slot, None);
        if let Some(inst) = terrains.instances.remove(&slot)
            && let Some(gpu) = inst.gpu
        {
            gpu.destroy(&render_device);
            tlas.release_slot_hit_offset(slot);
        }
    }

    let live: Vec<u32> = terrains.instances.keys().copied().collect();
    for slot in live {
        let inst = terrains.instances.get_mut(&slot).unwrap();
        let mesh_blas = meshes.get_by_id(inst.source.mesh);
        if let Some(gpu) = &inst.gpu {
            let stale = mesh_blas.is_none_or(|b| b.vertex_buffer.handle != gpu.mesh_vertex_buffer);
            if stale {
                let gpu = inst.gpu.take().unwrap();
                gpu.destroy(&render_device);
                tlas.release_slot_hit_offset(slot);
                tlas.set_override(slot, None);
            }
        }
        if inst.gpu.is_none() {
            let Some(blas) = mesh_blas else { continue };
            let Ok(tile) = tiles.get(inst.entity) else {
                continue;
            };
            let res = inst.source.resolution;
            let expected = res * res + (res - 1) * (res - 1);
            let vertex_count = blas.vertex_buffer.nr_elements as u32;
            if vertex_count != expected {
                log::error!(
                    "terrain: slot {slot} mesh has {vertex_count} vertices, expected {expected} \
                     (build it with terrain_mesh); tile stays static"
                );
                continue;
            }
            let index_count = blas.index_buffer.nr_elements as u32;
            if index_count < 3 {
                continue;
            }
            if tile.heights.len() != (res * res) as usize
                || tile.chunk_layers.len() != 256
                || tile.alpha.len() != (tile.alpha_atlas * tile.alpha_atlas * 4) as usize
            {
                log::error!("terrain: slot {slot} component sizes are inconsistent; skipping");
                continue;
            }

            // Host-visible editable state, seeded from the component.
            let mut heights: Buffer<f32> = render_device.create_host_buffer(
                tile.heights.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            render_device
                .map_buffer(&mut heights)
                .copy_from_slice(&tile.heights);
            let alpha_words: &[u32] = bytemuck::cast_slice(&tile.alpha);
            let mut alpha: Buffer<u32> = render_device.create_host_buffer(
                alpha_words.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            render_device
                .map_buffer(&mut alpha)
                .copy_from_slice(alpha_words);
            let mut chunk_layers: Buffer<[u32; 4]> =
                render_device.create_host_buffer(256, vk::BufferUsageFlags::STORAGE_BUFFER);
            render_device
                .map_buffer(&mut chunk_layers)
                .copy_from_slice(&tile.chunk_layers);

            // Own copy of the index stream (the mesh asset can be dropped/reloaded).
            let index_buffer = render_device.create_device_buffer::<u32>(
                index_count as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            );
            render_device.run_transfer_commands(|cmd| unsafe {
                let icopy = vk::BufferCopy::default().size(index_count as u64 * 4);
                render_device.device.cmd_copy_buffer(
                    cmd,
                    blas.index_buffer.handle,
                    index_buffer.handle,
                    std::slice::from_ref(&icopy),
                );
            });
            let stream_usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
            let vertices =
                render_device.create_device_buffer::<Vertex>(vertex_count as u64, stream_usage);
            let triangle_count = index_count / 3;
            let triangles = render_device.create_device_buffer::<Triangle>(
                triangle_count as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            let shade = render_device
                .create_host_buffer::<TerrainShadeGpu>(1, vk::BufferUsageFlags::STORAGE_BUFFER);

            let mut gpu = TerrainGpu {
                mesh_vertex_buffer: blas.vertex_buffer.handle,
                vertex_count,
                triangle_count,
                heights,
                alpha,
                chunk_layers,
                vertices,
                triangles,
                index_buffer,
                geometry_to_index: blas.geometry_to_index.address,
                geometry_to_triangle: blas.geometry_to_triangle.address,
                shade,
                blas: AccelerationStructure::default(),
                scratch: Buffer::default(),
                scratch_alignment: vk_utils::get_acceleration_structure_properties(&render_device)
                    .min_acceleration_structure_scratch_offset_alignment
                    as u64,
                builds: 0,
                hit_offset: tlas.slot_hit_offset(slot),
            };
            let geometry = gpu.geometry();
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                render_device
                    .ext_acc_struct
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &vk::AccelerationStructureBuildGeometryInfoKHR::default()
                            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                            .flags(BLAS_FLAGS)
                            .geometries(std::slice::from_ref(&geometry)),
                        &[triangle_count],
                        &mut sizes,
                    );
            }
            gpu.blas = allocate_acceleration_structure(
                &render_device,
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                &sizes,
            );
            gpu.scratch = render_device.create_device_buffer(
                sizes.build_scratch_size.max(sizes.update_scratch_size) + gpu.scratch_alignment,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );
            inst.gpu = Some(gpu);
            inst.geo_dirty = true;
            inst.layers_dirty = false;
        } else if inst.layers_dirty {
            // Chunk-table edit from the CPU (adding a layer while painting): re-map.
            if let Ok(tile) = tiles.get(inst.entity)
                && let Some(gpu) = inst.gpu.as_mut()
                && tile.chunk_layers.len() == 256
            {
                render_device
                    .map_buffer(&mut gpu.chunk_layers)
                    .copy_from_slice(&tile.chunk_layers);
            }
            inst.layers_dirty = false;
        }

        if let Some(gpu) = inst.gpu.as_mut() {
            let shade = TerrainShadeGpu {
                alpha: gpu.alpha.address,
                chunk_layers: gpu.chunk_layers.address,
                palette: palette_table.0,
                atlas: inst.source.alpha_atlas,
                cell: inst.source.alpha_cell,
                size: inst.source.size,
                palette_len: palette_table.1,
            };
            render_device
                .map_buffer(&mut gpu.shade)
                .copy_from_slice(&[shade]);
        }
        if let Some(gpu) = &inst.gpu
            && gpu.builds > 0
        {
            tlas.set_override(
                slot,
                Some(InstanceOverride {
                    blas: gpu.blas.address,
                    hit_offset: Some(gpu.hit_offset),
                }),
            );
        }
    }
}

fn cleanup_terrains(world: &mut World) {
    world.resource_scope(|world, mut terrains: Mut<Terrains>| {
        let rd = world.resource::<RenderDevice>();
        terrains.destroy(rd);
    });
}

// ---- mesh building ------------------------------------------------------------------------------

/// Build a terrain tile mesh from an absolute height grid: core's center-vertex pattern
/// (4 triangles per cell around an averaged center — no diagonal artifacts, RT-stable).
/// Heights are `resolution`² row-major; the mesh spans ±size/2 in x/z. The vertex ORDER is
/// the contract with `terrain_vertices` in terrain.slang: `res`² corners then `(res-1)`²
/// centers.
pub fn terrain_mesh(heights: &[f32], resolution: u32, size: f32) -> AuroraMesh {
    let res = resolution as usize;
    assert_eq!(heights.len(), res * res, "heights must be resolution^2");
    let cells = res - 1;
    let num_outer = res * res;
    let num_vertices = num_outer + cells * cells;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(num_vertices);
    let mut indices: Vec<u32> = Vec::with_capacity(cells * cells * 12);
    let fcells = cells as f32;

    let h = |x: usize, z: usize| heights[z.min(res - 1) * res + x.min(res - 1)];

    for z in 0..res {
        for x in 0..res {
            let u = x as f32 / fcells;
            let v = z as f32 / fcells;
            positions.push([(u - 0.5) * size, h(x, z), (v - 0.5) * size]);
            uvs.push([u, v]);
        }
    }
    for cz in 0..cells {
        for cx in 0..cells {
            let u = (cx as f32 + 0.5) / fcells;
            let v = (cz as f32 + 0.5) / fcells;
            let height = (h(cx, cz) + h(cx, cz + 1) + h(cx + 1, cz + 1) + h(cx + 1, cz)) * 0.25;
            positions.push([(u - 0.5) * size, height, (v - 0.5) * size]);
            uvs.push([u, v]);
        }
    }
    for cz in 0..cells {
        for cx in 0..cells {
            let a = (cz * res + cx) as u32;
            let b = ((cz + 1) * res + cx) as u32;
            let c = ((cz + 1) * res + cx + 1) as u32;
            let d = (cz * res + cx + 1) as u32;
            let m = (num_outer + cz * cells + cx) as u32;
            indices.extend_from_slice(&[a, b, m, b, c, m, c, d, m, d, a, m]);
        }
    }

    // Smooth normals by central difference (mirrors terrain.slang's sample_normal).
    let sample = |u: f32, v: f32| -> f32 {
        let maxi = fcells;
        let x = (u * maxi).clamp(0.0, maxi);
        let z = (v * maxi).clamp(0.0, maxi);
        let (x0, z0) = (x.floor() as usize, z.floor() as usize);
        let (x1, z1) = ((x0 + 1).min(res - 1), (z0 + 1).min(res - 1));
        let (fx, fz) = (x - x0 as f32, z - z0 as f32);
        let h0 = h(x0, z0) * (1.0 - fx) + h(x1, z0) * fx;
        let h1 = h(x0, z1) * (1.0 - fx) + h(x1, z1) * fx;
        h0 * (1.0 - fz) + h1 * fz
    };
    let texel = 1.0 / fcells;
    let step = size * texel;
    let normal_at = |u: f32, v: f32| -> [f32; 3] {
        let dx = (sample(u + texel, v) - sample(u - texel, v)) / (2.0 * step);
        let dz = (sample(u, v + texel) - sample(u, v - texel)) / (2.0 * step);
        let n = Vec3::new(-dx, 1.0, -dz).normalize();
        [n.x, n.y, n.z]
    };
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(num_vertices);
    for z in 0..res {
        for x in 0..res {
            normals.push(normal_at(x as f32 / fcells, z as f32 / fcells));
        }
    }
    for cz in 0..cells {
        for cx in 0..cells {
            normals.push(normal_at(
                (cx as f32 + 0.5) / fcells,
                (cz as f32 + 0.5) / fcells,
            ));
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    // One cluster in exactly this vertex order: the GPU edit addresses vertices by index.
    AuroraMesh::from_mesh(&mesh).expect("terrain mesh has positions")
}

// ---- plugin --------------------------------------------------------------------------------------

pub struct TerrainPlugin;

/// The brush cursor: a ring the terrain closest-hit shader draws on every terrain tile
/// around `center` (world x/z) at `radius`. A shader ring rather than a gizmo so it shows in
/// every view — gizmos only reach the spectator window, not the headset. Gameplay writes it
/// each frame; `active = false` hides it.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct TerrainCursor {
    pub center: Vec2,
    pub radius: f32,
    /// Ring emission (nits).
    pub color: LinearRgba,
    pub active: bool,
}

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TerrainEdits>();
        app.init_resource::<TerrainPalette>();
        app.init_resource::<TerrainCursor>();
        app.add_observer(on_instance_removed);

        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/terrain.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "terrain_brush",
                "terrain_vertices",
                "terrain_pack",
                "terrain_paint",
            ],
        ));
        app.insert_resource(Terrains::new(module));
        app.add_systems(PostUpdate, palette_images);
        app.add_systems(
            Last,
            (
                (sync_terrain_cpu, extract_terrains)
                    .chain()
                    .in_set(RenderSet::Extract),
                prepare_terrains
                    .in_set(RenderSet::Prepare)
                    .after(poll_for_asset::<AuroraMesh>)
                    .before(prepare_instances),
            ),
        );
        app.add_systems(TeardownSchedule, cleanup_terrains.before(on_shutdown));
    }
}
