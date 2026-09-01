//! The light table for next-event estimation: emissive triangles + analytic lights.
//!
//! Entries `[0, analytic_count)` are analytic lights extracted from bevy's `PointLight` /
//! `SpotLight` / `RectLight` components (sampled NEE-only -- they have no geometry, so BRDF
//! rays never find them). The rest: every instance whose [`AuroraMaterial`] has a non-zero
//! emissive colour contributes all of its BLAS triangles as one contiguous range. The CPU
//! only tracks *which* instances are emissive and uploads the per-instance records;
//! geometry and world transforms stay on the GPU, so the per-entry sampling weights (world
//! area x emitted luminance) and their CDF are computed by `lights.slang` inside the frame
//! command buffer, reading the live TLAS instance rows.
//!
//! The raygen samples a triangle by binary-searching the CDF and converts to a solid-angle
//! pdf with the *live* triangle area, so the estimator stays unbiased even when the CDF's
//! weights go stale (a light that moved or scaled since the last rebuild is just sampled at
//! slightly wrong rates). MIS against BRDF sampling looks an emissive hit up through
//! `slot_to_linst` + the hit's global triangle index.
//!
//! Sources are `Mesh3d` + emissive [`AuroraMaterial`] instances and `GltfModelHandle`
//! instances whose bundle has emissive primitives (emission per geometry, from the glTF
//! emissive factors). Animated cluster deformations are not accounted for (their BLAS-space
//! positions are pre-deform); sphere emissives are not in the table. Both still glow through
//! BRDF hits at full weight.

use std::collections::HashMap;

use ash::vk;
use bevy::ecs::hierarchy::ChildOf;
use bevy::ecs::lifecycle::Remove;
use bevy::ecs::observer::On;
use bevy::light::{PointLight, RectLight, SpotLight};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    compute::{
        ComputeModule, ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch,
    },
    gltf_mesh::{GltfModel, GltfModelHandle},
    gpu_transform::upload_slice,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::{GpuInstance, TLAS},
    vulkan_asset::{VulkanAssets, poll_for_asset},
};

/// One emissive instance on the GPU (must match `LightInst` in lights.slang and
/// `LightInstance` in types.glsl). 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightInstGpu {
    verts: u64,
    indices: u64,
    geom_to_index: u64,
    geom_to_triangle: u64,
    geom_count: u32,
    slot: u32,
    entry_base: u32,
    tri_count: u32,
    /// Address of this instance's per-geometry emission (3 floats each, nits).
    geom_emission: u64,
    pad: u64,
}

/// Must match `LightsHeader` in types.glsl / lights.slang. 56 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightsHeaderGpu {
    entry_count: u32,
    linst_count: u32,
    /// Written by the `light_scan` kernel; 0 until then gates the raygen off the table.
    total_power: f32,
    slot_map_count: u32,
    /// Entries `[0, analytic_count)` are analytic lights; the rest triangles.
    analytic_count: u32,
    pad1: u32,
    cdf: u64,
    linsts: u64,
    slot_to_linst: u64,
    analytics: u64,
}

/// One analytic light (point / spot / rect). Must match `AnalyticLight` in types.glsl /
/// lights.slang. 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct AnalyticLightGpu {
    position: [f32; 3],
    /// 0 point, 1 spot, 2 rect.
    kind: u32,
    direction: [f32; 3],
    radius: f32,
    tangent: [f32; 3],
    cos_inner: f32,
    /// Point/spot: luminous intensity I (nit*m^2); rect: emitted radiance L (nits).
    emission: [f32; 3],
    cos_outer: f32,
    half_extents: [f32; 2],
    /// CDF weight, same units as the triangle entries (flux / pi).
    power: f32,
    /// Bit 0: two-sided (rect).
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightPowerParams {
    linsts: u64,
    instances: u64,
    powers: u64,
    analytics: u64,
    linst_count: u32,
    entry_count: u32,
    analytic_count: u32,
    base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightScanParams {
    powers: u64,
    header: u64,
    entry_count: u32,
    pad: u32,
}

/// What an emissive instance's geometry and emission come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LightSource {
    /// A `Mesh3d` with an emissive [`AuroraMaterial`] (one emission for every geometry).
    Mesh(AssetId<Mesh>, AssetId<AuroraMaterial>),
    /// A glTF bundle; emission per geometry from its materials, resolved once the BLAS is.
    Gltf(AssetId<GltfModel>),
}

#[derive(Resource)]
pub struct LightManager {
    module: Handle<ComputeModule>,
    /// slot -> source for instances that may light the scene.
    sources: HashMap<u32, LightSource>,
    /// Mesh instances whose material asset had not loaded when they were seen.
    pending: HashMap<u32, LightSource>,
    dirty: bool,
    // Device.
    header: Buffer<LightsHeaderGpu>,
    linsts: Buffer<LightInstGpu>,
    slot_map: Buffer<u32>,
    cdf: Buffer<f32>,
    emissions: Buffer<f32>,
    analytics: Buffer<AnalyticLightGpu>,
    /// The analytic lights as last extracted (entry order); re-uploaded when they move.
    analytic_cache: Vec<AnalyticLightGpu>,
    /// The cache changed in-place (same set): re-upload in `record` without a rebuild.
    analytic_upload: bool,
    entry_count: u32,
    linst_count: u32,
    needs_power_pass: bool,
    warned_module: bool,
    logged: (u32, u32),
    /// Entries the frame uniform may expose (0 while the table is empty / not built).
    pub active_entries: u32,
    /// Bumped on every rebuild; reservoirs remember it so stale entry ids are dropped.
    pub epoch: u32,
}

impl LightManager {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            sources: HashMap::new(),
            pending: HashMap::new(),
            dirty: false,
            header: Buffer::default(),
            linsts: Buffer::default(),
            slot_map: Buffer::default(),
            cdf: Buffer::default(),
            emissions: Buffer::default(),
            analytics: Buffer::default(),
            analytic_cache: Vec::new(),
            analytic_upload: false,
            entry_count: 0,
            linst_count: 0,
            needs_power_pass: false,
            warned_module: false,
            logged: (0, 0),
            active_entries: 0,
            epoch: 1,
        }
    }

    /// Device address of the [`LightsHeaderGpu`] (0 until the first build).
    pub fn header_address(&self) -> u64 {
        self.header.address
    }

    fn destroy_buffers(&mut self, rd: &RenderDevice) {
        for b in [
            self.header.handle,
            self.linsts.handle,
            self.slot_map.handle,
            self.cdf.handle,
            self.emissions.handle,
            self.analytics.handle,
        ] {
            rd.destroyer.destroy_buffer(b);
        }
        self.header = Buffer::default();
        self.linsts = Buffer::default();
        self.slot_map = Buffer::default();
        self.cdf = Buffer::default();
        self.emissions = Buffer::default();
        self.analytics = Buffer::default();
    }

    /// Records the weight + CDF kernels into the frame command buffer. Call after
    /// `TLAS::record` (the instance rows must be gathered) and before the trace.
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        tlas: &TLAS,
    ) {
        if !self.needs_power_pass || self.entry_count == 0 {
            return;
        }
        // Triangle entries read live TLAS rows; analytic-only tables have none to read.
        if self.linst_count > 0 && tlas.instances_address() == 0 {
            return;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("lights.slang not compiled yet; light table waits");
                self.warned_module = true;
            }
            return;
        };
        // Moved / re-tuned analytic lights: refresh in place (no rebuild, no epoch bump --
        // reservoirs re-evaluate against the live data like they do for moved triangles).
        if self.analytic_upload && self.analytics.handle != vk::Buffer::null() {
            self.analytic_upload = false;
            unsafe {
                rd.device.cmd_update_buffer(
                    cmd,
                    self.analytics.handle,
                    0,
                    bytemuck::cast_slice(&self.analytic_cache),
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ,
            );
        }
        let params = LightPowerParams {
            linsts: self.linsts.address,
            instances: tlas.instances_address(),
            powers: self.cdf.address,
            analytics: self.analytics.address,
            linst_count: self.linst_count,
            entry_count: self.entry_count,
            analytic_count: self.analytic_cache.len() as u32,
            base: 0,
        };
        record_dispatch(
            rd,
            cmd,
            module,
            "light_powers",
            &params,
            self.entry_count,
            Some(std::mem::offset_of!(LightPowerParams, base)),
        );
        compute_to_compute_barrier(rd, cmd);
        let scan = LightScanParams {
            powers: self.cdf.address,
            header: self.header.address,
            entry_count: self.entry_count,
            pad: 0,
        };
        record_dispatch(rd, cmd, module, "light_scan", &scan, 1, None);
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ,
        );
        self.needs_power_pass = false;
        // The table rebuilds every frame while the scene streams in; log the milestones.
        if self.logged != (self.linst_count, self.entry_count) {
            self.logged = (self.linst_count, self.entry_count);
            log::info!(
                "light table: {} instances, {} entries ({} analytic)",
                self.linst_count,
                self.entry_count,
                self.analytic_cache.len()
            );
        }
    }
}

/// Tracks which instances may light the scene. Change-driven; a modified material asset
/// triggers a full rescan (rare: panel edits, hot reloads).
#[allow(clippy::type_complexity)]
fn track_lights(
    mut lights: ResMut<LightManager>,
    materials: Res<Assets<AuroraMaterial>>,
    mut material_events: MessageReader<AssetEvent<AuroraMaterial>>,
    changed: Query<
        (
            &GpuInstance,
            Option<&Mesh3d>,
            Option<&GltfModelHandle>,
            Option<&AuroraMaterial3d>,
        ),
        Or<(
            Added<GpuInstance>,
            Changed<AuroraMaterial3d>,
            Changed<Mesh3d>,
            Changed<GltfModelHandle>,
        )>,
    >,
    all: Query<(
        &GpuInstance,
        Option<&Mesh3d>,
        Option<&GltfModelHandle>,
        Option<&AuroraMaterial3d>,
    )>,
) {
    let mut rescan = false;
    for event in material_events.read() {
        if matches!(event, AssetEvent::Modified { .. }) {
            rescan = true;
        }
    }

    let lights = &mut *lights;
    let consider = |slot: u32,
                    mesh: Option<&Mesh3d>,
                    gltf: Option<&GltfModelHandle>,
                    material: Option<&AuroraMaterial3d>,
                    sources: &mut HashMap<u32, LightSource>,
                    pending: &mut HashMap<u32, LightSource>,
                    dirty: &mut bool| {
        // glTF bundles carry their own materials; whether any geometry is emissive is
        // resolved against the prepared BLAS in prepare_lights.
        if let Some(gltf) = gltf {
            let source = LightSource::Gltf(gltf.0.id());
            pending.remove(&slot);
            if sources.insert(slot, source) != Some(source) {
                *dirty = true;
            }
            return;
        }
        let (Some(mesh), Some(material)) = (mesh, material) else {
            return;
        };
        let source = LightSource::Mesh(mesh.id(), material.0.id());
        let Some(asset) = materials.get(&material.0) else {
            pending.insert(slot, source);
            return;
        };
        let e = asset.emissive;
        let emissive = 0.2126 * e.red + 0.7152 * e.green + 0.0722 * e.blue > 0.0;
        pending.remove(&slot);
        if emissive {
            if sources.insert(slot, source) != Some(source) {
                *dirty = true;
            }
        } else if sources.remove(&slot).is_some() {
            *dirty = true;
        }
    };

    if rescan {
        for (instance, mesh, gltf, material) in all.iter() {
            consider(
                instance.0,
                mesh,
                gltf,
                material,
                &mut lights.sources,
                &mut lights.pending,
                &mut lights.dirty,
            );
        }
    } else {
        for (instance, mesh, gltf, material) in changed.iter() {
            consider(
                instance.0,
                mesh,
                gltf,
                material,
                &mut lights.sources,
                &mut lights.pending,
                &mut lights.dirty,
            );
        }
    }

    // Mesh materials that had not loaded yet: retry until they have.
    if !lights.pending.is_empty() {
        let retry: Vec<(u32, LightSource)> = lights.pending.iter().map(|(s, v)| (*s, *v)).collect();
        for (slot, source) in retry {
            let LightSource::Mesh(_, material) = source else {
                continue;
            };
            let Some(asset) = materials.get(material) else {
                continue;
            };
            let e = asset.emissive;
            let emissive = 0.2126 * e.red + 0.7152 * e.green + 0.0722 * e.blue > 0.0;
            lights.pending.remove(&slot);
            if emissive {
                if lights.sources.insert(slot, source) != Some(source) {
                    lights.dirty = true;
                }
            } else if lights.sources.remove(&slot).is_some() {
                lights.dirty = true;
            }
        }
    }
}

/// A despawned instance leaves the light table.
fn on_light_instance_removed(
    remove: On<Remove<GpuInstance>>,
    instances: Query<&GpuInstance>,
    mut lights: ResMut<LightManager>,
) {
    if let Ok(instance) = instances.get(remove.entity) {
        lights.pending.remove(&instance.0);
        if lights.sources.remove(&instance.0).is_some() {
            lights.dirty = true;
        }
    }
}

/// Rebuilds the GPU light table when the light set changed. Instances whose BLAS has not
/// loaded yet keep the manager dirty, so the table converges as the scene streams in.
fn prepare_lights(
    render_device: Res<RenderDevice>,
    mut lights: ResMut<LightManager>,
    meshes: Res<VulkanAssets<Mesh>>,
    gltf_meshes: Res<VulkanAssets<GltfModel>>,
    materials: Res<Assets<AuroraMaterial>>,
) {
    if !lights.dirty {
        return;
    }

    let mut linsts: Vec<LightInstGpu> = Vec::with_capacity(lights.sources.len());
    // Per-geometry emissions, concatenated; each linst's geom_emission starts as a float
    // offset into this and is rebased once the device buffer exists.
    let mut emissions: Vec<f32> = Vec::new();
    // Entries [0, analytic_count) are the analytic lights; triangles follow.
    let analytic_count = lights.analytic_cache.len() as u32;
    let mut entry_base = analytic_count;
    let mut max_slot = 0u32;
    let mut waiting = false;
    let mut slots: Vec<u32> = lights.sources.keys().copied().collect();
    slots.sort_unstable();
    for slot in slots {
        let (blas, geom_emissions): (&crate::blas::BLAS, Vec<[f32; 3]>) =
            match lights.sources[&slot] {
                LightSource::Mesh(mesh, material) => {
                    let Some(blas) = meshes.get_by_id(mesh) else {
                        waiting = true;
                        continue;
                    };
                    // The emissive factor as the tracer multiplies it (linear radiance, nits).
                    let Some(emission) = materials
                        .get(material)
                        .map(|m| [m.emissive.red, m.emissive.green, m.emissive.blue])
                    else {
                        waiting = true;
                        continue;
                    };
                    let geoms = (blas.geometry_to_index.nr_elements as usize).max(1);
                    (blas, vec![emission; geoms])
                }
                LightSource::Gltf(gltf) => {
                    let Some(blas) = gltf_meshes.get_by_id(gltf) else {
                        waiting = true;
                        continue;
                    };
                    let Some(bundle) = &blas.gltf_materials else {
                        continue;
                    };
                    let geom_emissions: Vec<[f32; 3]> = bundle
                        .iter()
                        .map(|m| {
                            let e = m.base_emissive_factor;
                            [e[0], e[1], e[2]]
                        })
                        .collect();
                    // A bundle with no emissive geometry lights nothing.
                    if !geom_emissions
                        .iter()
                        .any(|e| 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2] > 0.0)
                    {
                        continue;
                    }
                    (blas, geom_emissions)
                }
            };
        let tri_count = (blas.index_buffer.nr_elements / 3) as u32;
        if tri_count == 0 {
            continue;
        }
        let offset = emissions.len() as u64;
        for e in &geom_emissions {
            emissions.extend_from_slice(e);
        }
        linsts.push(LightInstGpu {
            verts: blas.vertex_buffer.address,
            indices: blas.index_buffer.address,
            geom_to_index: blas.geometry_to_index.address,
            geom_to_triangle: blas.geometry_to_triangle.address,
            geom_count: geom_emissions.len() as u32,
            slot,
            entry_base,
            tri_count,
            geom_emission: offset,
            pad: 0,
        });
        entry_base += tri_count;
        max_slot = max_slot.max(slot);
    }
    // Stay dirty (and rebuild again next frame) while any BLAS / material is still loading.
    lights.dirty = waiting;

    lights.destroy_buffers(&render_device);
    lights.epoch = lights.epoch.wrapping_add(1).max(1);
    lights.entry_count = entry_base;
    lights.linst_count = linsts.len() as u32;
    lights.active_entries = entry_base;
    lights.needs_power_pass = entry_base > 0;
    if entry_base == 0 {
        return;
    }

    let mut slot_map = vec![u32::MAX; max_slot as usize + 1];
    for (i, li) in linsts.iter().enumerate() {
        slot_map[li.slot as usize] = i as u32;
    }

    let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
    lights.linsts = render_device.create_device_buffer(linsts.len().max(1) as u64, storage);
    lights.slot_map = render_device.create_device_buffer(slot_map.len() as u64, storage);
    lights.cdf = render_device.create_device_buffer(entry_base as u64, storage);
    lights.header = render_device.create_device_buffer(1, storage);
    lights.emissions = render_device.create_device_buffer(emissions.len().max(1) as u64, storage);
    lights.analytics =
        render_device.create_device_buffer(lights.analytic_cache.len().max(1) as u64, storage);
    lights.analytic_upload = false;
    let mut linsts = linsts;
    for li in &mut linsts {
        li.geom_emission = lights.emissions.address + li.geom_emission * 4;
    }
    let header = LightsHeaderGpu {
        entry_count: entry_base,
        linst_count: lights.linst_count,
        total_power: 0.0,
        slot_map_count: slot_map.len() as u32,
        analytic_count,
        pad1: 0,
        cdf: lights.cdf.address,
        linsts: lights.linsts.address,
        slot_to_linst: lights.slot_map.address,
        analytics: lights.analytics.address,
    };
    render_device.run_transfer_commands(|cmd| {
        if !linsts.is_empty() {
            upload_slice(&render_device, cmd, &linsts, &lights.linsts);
        }
        upload_slice(&render_device, cmd, &slot_map, &lights.slot_map);
        upload_slice(&render_device, cmd, &[header], &lights.header);
        if !emissions.is_empty() {
            upload_slice(&render_device, cmd, &emissions, &lights.emissions);
        }
        if !lights.analytic_cache.is_empty() {
            upload_slice(
                &render_device,
                cmd,
                &lights.analytic_cache,
                &lights.analytics,
            );
        }
    });
}

fn luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Extracts bevy's analytic lights (`PointLight` / `SpotLight` / `RectLight`) into the
/// table. A changed set rebuilds the table (new epoch); mere movement / re-tuning
/// re-uploads in place so ReSTIR history survives. Entry order is by entity id, stable
/// across frames.
///
/// Units: bevy `intensity` is luminous power in lumens; the tracer works in nits.
/// Point/spot store luminous intensity `I = lm / 4pi` (contribution `I / d^2`), rects the
/// emitted radiance `L = lm / (pi * area)`. CDF weights are flux / pi, matching the
/// triangle entries' `area * L`.
fn track_analytic_lights(
    mut lights: ResMut<LightManager>,
    points: Query<(Entity, &PointLight)>,
    spots: Query<(Entity, &SpotLight)>,
    rects: Query<(Entity, &RectLight)>,
    nodes: Query<(&Transform, Option<&ChildOf>)>,
) {
    use std::f32::consts::PI;
    // CPU hierarchy propagation is off (mesh transforms propagate on the GPU), so a child
    // light's GlobalTransform never updates: walk the parent chain here instead. Lights
    // are few and hierarchies shallow.
    let world_of = |entity: Entity| -> Mat4 {
        let mut m = Mat4::IDENTITY;
        let mut cur = Some(entity);
        while let Some(e) = cur {
            let Ok((t, child_of)) = nodes.get(e) else {
                break;
            };
            m = t.to_matrix() * m;
            cur = child_of.map(|c| c.parent());
        }
        m
    };
    let mut found: Vec<(Entity, AnalyticLightGpu)> = Vec::new();
    for (entity, light) in &points {
        let c = light.color.to_linear();
        let i = light.intensity / (4.0 * PI);
        let emission = [c.red * i, c.green * i, c.blue * i];
        found.push((
            entity,
            AnalyticLightGpu {
                position: world_of(entity).w_axis.truncate().to_array(),
                kind: 0,
                direction: [0.0, -1.0, 0.0],
                radius: light.radius,
                tangent: [1.0, 0.0, 0.0],
                cos_inner: 1.0,
                emission,
                cos_outer: -1.0,
                half_extents: [0.0; 2],
                power: 4.0 * luma(emission),
                flags: 0,
            },
        ));
    }
    for (entity, light) in &spots {
        let c = light.color.to_linear();
        let i = light.intensity / (4.0 * PI);
        let emission = [c.red * i, c.green * i, c.blue * i];
        let cos_outer = light.outer_angle.cos();
        let solid_angle = 2.0 * PI * (1.0 - cos_outer);
        let m = world_of(entity);
        found.push((
            entity,
            AnalyticLightGpu {
                position: m.w_axis.truncate().to_array(),
                kind: 1,
                direction: m
                    .transform_vector3(Vec3::NEG_Z)
                    .normalize_or(Vec3::NEG_Z)
                    .to_array(),
                radius: light.radius,
                tangent: [1.0, 0.0, 0.0],
                cos_inner: light.inner_angle.cos(),
                emission,
                cos_outer,
                half_extents: [0.0; 2],
                power: luma(emission) * solid_angle / PI,
                flags: 0,
            },
        ));
    }
    for (entity, light) in &rects {
        let c = light.color.to_linear();
        let m = world_of(entity);
        // The rect spans local XY, faces local -Z; world scale stretches it.
        let half_x = m.transform_vector3(Vec3::X * (light.width * 0.5));
        let half_y = m.transform_vector3(Vec3::Y * (light.height * 0.5));
        let area = (4.0 * half_x.length() * half_y.length()).max(1.0e-6);
        let l = light.intensity / (PI * area);
        let emission = [c.red * l, c.green * l, c.blue * l];
        found.push((
            entity,
            AnalyticLightGpu {
                position: m.w_axis.truncate().to_array(),
                kind: 2,
                direction: m
                    .transform_vector3(Vec3::NEG_Z)
                    .normalize_or(Vec3::NEG_Z)
                    .to_array(),
                radius: 0.0,
                tangent: half_x.normalize_or(Vec3::X).to_array(),
                cos_inner: 1.0,
                emission,
                cos_outer: -1.0,
                half_extents: [half_x.length(), half_y.length()],
                power: luma(emission) * area,
                flags: 0,
            },
        ));
    }
    found.sort_by_key(|(entity, _)| *entity);
    let found: Vec<AnalyticLightGpu> = found.into_iter().map(|(_, l)| l).collect();
    if found != lights.analytic_cache {
        for l in &found {
            log::debug!("analytic light: {l:?}");
        }
        if found.len() != lights.analytic_cache.len() {
            // The entry layout shifts: full rebuild, reservoirs dropped via the epoch.
            lights.dirty = true;
        } else {
            lights.analytic_upload = true;
            lights.needs_power_pass = true;
        }
        lights.analytic_cache = found;
    }
}

fn cleanup_lights(mut lights: ResMut<LightManager>, render_device: Res<RenderDevice>) {
    lights.destroy_buffers(&render_device);
}

pub struct LightsPlugin;

impl Plugin for LightsPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/lights.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["light_powers", "light_scan"]));
        app.insert_resource(LightManager::new(module));
        app.add_observer(on_light_instance_removed);
        app.add_systems(
            Last,
            (
                (track_lights, track_analytic_lights).in_set(RenderSet::Extract),
                prepare_lights
                    .in_set(RenderSet::Prepare)
                    .after(poll_for_asset::<Mesh>)
                    .after(poll_for_asset::<GltfModel>)
                    .after(poll_for_asset::<AuroraMaterial>),
            ),
        );
        app.add_systems(TeardownSchedule, cleanup_lights.before(on_shutdown));
    }
}
