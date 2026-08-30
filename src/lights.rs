//! Emissive-triangle lights: the sampling table for next-event estimation.
//!
//! Every instance whose [`AuroraMaterial`] has a non-zero emissive colour contributes all of
//! its BLAS triangles as one contiguous range of entries in a global light table. The CPU only
//! tracks *which* instances are emissive and uploads the per-instance records; geometry and
//! world transforms stay on the GPU, so the per-entry sampling weights (world area x emitted
//! luminance) and their CDF are computed by `lights.slang` inside the frame command buffer,
//! reading the live TLAS instance rows.
//!
//! The raygen samples a triangle by binary-searching the CDF and converts to a solid-angle
//! pdf with the *live* triangle area, so the estimator stays unbiased even when the CDF's
//! weights go stale (a light that moved or scaled since the last rebuild is just sampled at
//! slightly wrong rates). MIS against BRDF sampling looks an emissive hit up through
//! `slot_to_linst` + the hit's global triangle index.
//!
//! glTF bundle emissives are not in the table (they still glow through BRDF hits, at full
//! weight); animated cluster deformations are not accounted for either. Both are follow-ups.

use std::collections::HashMap;

use ash::vk;
use bevy::ecs::lifecycle::Remove;
use bevy::ecs::observer::On;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch},
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
    emission: [f32; 3],
    pad: f32,
}

/// Must match `LightsHeader` in types.glsl / lights.slang. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct LightsHeaderGpu {
    entry_count: u32,
    linst_count: u32,
    /// Written by the `light_scan` kernel; 0 until then gates the raygen off the table.
    total_power: f32,
    slot_map_count: u32,
    cdf: u64,
    linsts: u64,
    slot_to_linst: u64,
    pad: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightPowerParams {
    linsts: u64,
    instances: u64,
    powers: u64,
    linst_count: u32,
    entry_count: u32,
    base: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightScanParams {
    powers: u64,
    header: u64,
    entry_count: u32,
    pad: u32,
}

#[derive(Resource)]
pub struct LightManager {
    module: Handle<ComputeModule>,
    /// slot -> (material, mesh) for instances with an emissive material.
    sources: HashMap<u32, (AssetId<AuroraMaterial>, AssetId<Mesh>)>,
    /// Instances whose material asset had not loaded when they were seen.
    pending: HashMap<u32, (AssetId<AuroraMaterial>, AssetId<Mesh>)>,
    dirty: bool,
    // Device.
    header: Buffer<LightsHeaderGpu>,
    linsts: Buffer<LightInstGpu>,
    slot_map: Buffer<u32>,
    cdf: Buffer<f32>,
    entry_count: u32,
    linst_count: u32,
    needs_power_pass: bool,
    warned_module: bool,
    logged: (u32, u32),
    /// Entries the frame uniform may expose (0 while the table is empty / not built).
    pub active_entries: u32,
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
            entry_count: 0,
            linst_count: 0,
            needs_power_pass: false,
            warned_module: false,
            logged: (0, 0),
            active_entries: 0,
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
        ] {
            rd.destroyer.destroy_buffer(b);
        }
        self.header = Buffer::default();
        self.linsts = Buffer::default();
        self.slot_map = Buffer::default();
        self.cdf = Buffer::default();
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
        if tlas.instances_address() == 0 {
            return;
        }
        let Some(module) = modules.get(&self.module) else {
            if !self.warned_module {
                log::info!("lights.slang not compiled yet; light table waits");
                self.warned_module = true;
            }
            return;
        };
        let params = LightPowerParams {
            linsts: self.linsts.address,
            instances: tlas.instances_address(),
            powers: self.cdf.address,
            linst_count: self.linst_count,
            entry_count: self.entry_count,
            base: 0,
            pad: 0,
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
                "light table: {} instances, {} triangle entries",
                self.linst_count,
                self.entry_count
            );
        }
    }
}

/// Tracks which instances have an emissive material. Change-driven; a modified material
/// asset triggers a full rescan (rare: panel edits, hot reloads).
#[allow(clippy::type_complexity)]
fn track_lights(
    mut lights: ResMut<LightManager>,
    materials: Res<Assets<AuroraMaterial>>,
    mut material_events: MessageReader<AssetEvent<AuroraMaterial>>,
    changed: Query<
        (&GpuInstance, &Mesh3d, &AuroraMaterial3d),
        Or<(Added<GpuInstance>, Changed<AuroraMaterial3d>, Changed<Mesh3d>)>,
    >,
    all: Query<(&GpuInstance, &Mesh3d, &AuroraMaterial3d)>,
) {
    let mut rescan = false;
    for event in material_events.read() {
        if matches!(event, AssetEvent::Modified { .. }) {
            rescan = true;
        }
    }

    let lights = &mut *lights;
    let consider = |slot: u32,
                        mesh: AssetId<Mesh>,
                        material: AssetId<AuroraMaterial>,
                        sources: &mut HashMap<u32, (AssetId<AuroraMaterial>, AssetId<Mesh>)>,
                        pending: &mut HashMap<u32, (AssetId<AuroraMaterial>, AssetId<Mesh>)>,
                        dirty: &mut bool| {
        let Some(asset) = materials.get(material) else {
            pending.insert(slot, (material, mesh));
            return;
        };
        let e = asset.emissive;
        let emissive = 0.2126 * e.red + 0.7152 * e.green + 0.0722 * e.blue > 0.0;
        pending.remove(&slot);
        if emissive {
            if sources.insert(slot, (material, mesh)) != Some((material, mesh)) {
                *dirty = true;
            }
        } else if sources.remove(&slot).is_some() {
            *dirty = true;
        }
    };

    if rescan {
        for (instance, mesh, material) in all.iter() {
            consider(
                instance.0,
                mesh.id(),
                material.0.id(),
                &mut lights.sources,
                &mut lights.pending,
                &mut lights.dirty,
            );
        }
    } else {
        for (instance, mesh, material) in changed.iter() {
            consider(
                instance.0,
                mesh.id(),
                material.0.id(),
                &mut lights.sources,
                &mut lights.pending,
                &mut lights.dirty,
            );
        }
    }

    // Materials that had not loaded yet: retry until they have.
    if !lights.pending.is_empty() {
        let retry: Vec<(u32, (AssetId<AuroraMaterial>, AssetId<Mesh>))> =
            lights.pending.iter().map(|(s, v)| (*s, *v)).collect();
        for (slot, (material, mesh)) in retry {
            consider(
                slot,
                mesh,
                material,
                &mut lights.sources,
                &mut lights.pending,
                &mut lights.dirty,
            );
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
    materials: Res<Assets<AuroraMaterial>>,
) {
    if !lights.dirty {
        return;
    }

    let mut linsts: Vec<LightInstGpu> = Vec::with_capacity(lights.sources.len());
    let mut entry_base = 0u32;
    let mut max_slot = 0u32;
    let mut waiting = false;
    let mut slots: Vec<u32> = lights.sources.keys().copied().collect();
    slots.sort_unstable();
    for slot in slots {
        let (material, mesh) = lights.sources[&slot];
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
        let tri_count = (blas.index_buffer.nr_elements / 3) as u32;
        if tri_count == 0 {
            continue;
        }
        linsts.push(LightInstGpu {
            verts: blas.vertex_buffer.address,
            indices: blas.index_buffer.address,
            geom_to_index: blas.geometry_to_index.address,
            geom_to_triangle: blas.geometry_to_triangle.address,
            geom_count: (blas.geometry_to_index.nr_elements as u32).max(1),
            slot,
            entry_base,
            tri_count,
            emission,
            pad: 0.0,
        });
        entry_base += tri_count;
        max_slot = max_slot.max(slot);
    }
    // Stay dirty (and rebuild again next frame) while any BLAS / material is still loading.
    lights.dirty = waiting;

    lights.destroy_buffers(&render_device);
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
    lights.linsts = render_device.create_device_buffer(linsts.len() as u64, storage);
    lights.slot_map = render_device.create_device_buffer(slot_map.len() as u64, storage);
    lights.cdf = render_device.create_device_buffer(entry_base as u64, storage);
    lights.header = render_device.create_device_buffer(1, storage);
    let header = LightsHeaderGpu {
        entry_count: entry_base,
        linst_count: lights.linst_count,
        total_power: 0.0,
        slot_map_count: slot_map.len() as u32,
        cdf: lights.cdf.address,
        linsts: lights.linsts.address,
        slot_to_linst: lights.slot_map.address,
        pad: 0,
    };
    render_device.run_transfer_commands(|cmd| {
        upload_slice(&render_device, cmd, &linsts, &lights.linsts);
        upload_slice(&render_device, cmd, &slot_map, &lights.slot_map);
        upload_slice(&render_device, cmd, &[header], &lights.header);
    });
}

fn cleanup_lights(mut lights: ResMut<LightManager>, render_device: Res<RenderDevice>) {
    lights.destroy_buffers(&render_device);
}

pub struct LightsPlugin;

impl Plugin for LightsPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/lights.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &["light_powers", "light_scan"],
        ));
        app.insert_resource(LightManager::new(module));
        app.add_observer(on_light_instance_removed);
        app.add_systems(
            Last,
            (
                track_lights.in_set(RenderSet::Extract),
                prepare_lights
                    .in_set(RenderSet::Prepare)
                    .after(poll_for_asset::<Mesh>)
                    .after(poll_for_asset::<AuroraMaterial>),
            ),
        );
        app.add_systems(TeardownSchedule, cleanup_lights.before(on_shutdown));
    }
}
