//! SHaRC-style world-space radiance cache.
//!
//! A fixed-size hash table of voxel entries (voxel size scales with distance to the camera,
//! keyed by position + level + dominant face normal). The raygen deposits each traced path's
//! outgoing radiance at its diffuse vertices (delta accumulation over the sample's running
//! radiance) and, from bounce 2 on, terminates paths into entries that have converged. The
//! `sharc_resolve` kernel folds each frame's accumulation into a per-entry moving average
//! and evicts entries that have not been touched for a while.
//!
//! The cache is biased by construction, so it turns off while accumulating (Space keeps the
//! reference estimator) and the panel's `sharc` toggle A/Bs it live.

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Must match SHARC_ENTRIES in raygen.rgen.
const SHARC_ENTRIES: u32 = 1 << 20;
/// Must match SharcEntry in types.glsl / sharc.slang.
const ENTRY_BYTES: u64 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SharcResolveParams {
    entries: u64,
    entry_count: u32,
    frame: u32,
    base: u32,
    pad: u32,
}

#[derive(Resource)]
pub struct SharcState {
    module: Handle<ComputeModule>,
    buffer: Buffer<u8>,
}

impl SharcState {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            buffer: Buffer::default(),
        }
    }

    /// Device address of the entry table (0 until first enabled).
    pub fn address(&self) -> u64 {
        self.buffer.address
    }

    /// Creates the table on first use and records the per-frame resolve. Call before the
    /// trace; cross-frame visibility is covered by the in-flight fence.
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        frame: u32,
        enabled: bool,
    ) {
        if self.buffer.handle == vk::Buffer::null() {
            if !enabled {
                return;
            }
            self.buffer = rd.create_device_buffer(
                SHARC_ENTRIES as u64 * ENTRY_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, self.buffer.handle, 0, vk::WHOLE_SIZE, 0);
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            log::info!(
                "sharc: {} entries ({} MB)",
                SHARC_ENTRIES,
                SHARC_ENTRIES as u64 * ENTRY_BYTES / (1024 * 1024)
            );
        }
        let Some(module) = modules.get(&self.module) else {
            return;
        };
        let params = SharcResolveParams {
            entries: self.buffer.address,
            entry_count: SHARC_ENTRIES,
            frame,
            base: 0,
            pad: 0,
        };
        record_dispatch(
            rd,
            cmd,
            module,
            "sharc_resolve",
            &params,
            SHARC_ENTRIES,
            Some(std::mem::offset_of!(SharcResolveParams, base)),
        );
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
    }
}

fn cleanup_sharc(mut sharc: ResMut<SharcState>, render_device: Res<RenderDevice>) {
    render_device.destroyer.destroy_buffer(sharc.buffer.handle);
    sharc.buffer = Buffer::default();
}

pub struct SharcPlugin;

impl Plugin for SharcPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/sharc.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["sharc_resolve"]));
        app.insert_resource(SharcState::new(module));
        app.add_systems(TeardownSchedule, cleanup_sharc.before(on_shutdown));
    }
}
