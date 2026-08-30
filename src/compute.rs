//! Compute kernels from Slang modules.
//!
//! A [`ComputeModule`] is one `.slang` file plus the entry points to build pipelines for. Every
//! kernel shares one pipeline layout: no descriptor sets, just a [`COMPUTE_PUSH_CONSTANT_SIZE`]
//! push-constant block that carries raw buffer pointers (`SPV_KHR_physical_storage_buffer`) and
//! a few scalars. Recording is [`record_dispatch`] / [`record_dispatch_indirect`] plus the
//! barrier helpers; the module reloads with its shader like the ray-tracing pipeline does.

use std::collections::HashMap;

use ash::vk;
use bevy::{
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    prelude::*,
};

use crate::{
    render_device::RenderDevice,
    shader::Shader,
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

/// Bytes of push constants every kernel may use (the Vulkan-guaranteed minimum).
pub const COMPUTE_PUSH_CONSTANT_SIZE: u32 = 128;

/// Threads per workgroup for every kernel (`[numthreads(64, 1, 1)]`).
pub const WORKGROUP_SIZE: u32 = 64;

/// Largest 1D dispatch: `maxComputeWorkGroupCount[0]` is 65535 on every desktop driver.
pub const MAX_GROUPS: u32 = 65535;

#[derive(Asset, TypePath, Clone, Debug)]
pub struct ComputeModule {
    /// A dependency so the module is re-extracted once the shader (re)loads.
    #[dependency]
    pub shader: Handle<Shader>,
    pub entry_points: Vec<String>,
}

impl ComputeModule {
    pub fn new(shader: Handle<Shader>, entry_points: &[&str]) -> Self {
        Self {
            shader,
            entry_points: entry_points.iter().map(|s| s.to_string()).collect(),
        }
    }
}

pub struct CompiledComputeModule {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipelines: HashMap<String, vk::Pipeline>,
}

impl CompiledComputeModule {
    pub fn pipeline(&self, entry: &str) -> vk::Pipeline {
        *self
            .pipelines
            .get(entry)
            .unwrap_or_else(|| panic!("compute module has no entry point {entry}"))
    }
}

impl VulkanAsset for ComputeModule {
    type ExtractedAsset = (Shader, Vec<String>);
    type ExtractParam = SRes<Assets<Shader>>;
    type PreparedAsset = CompiledComputeModule;

    fn extract_asset(
        &self,
        param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let shaders: &Assets<crate::shader::Shader> = &**param;
        let shader = shaders.get(&self.shader)?;
        shader.spirv.as_ref()?;
        Some((shader.clone(), self.entry_points.clone()))
    }

    fn prepare_asset(
        (shader, entry_points): Self::ExtractedAsset,
        render_device: &RenderDevice,
    ) -> Self::PreparedAsset {
        let spirv = shader.spirv.as_ref().unwrap();
        let words: &[u32] =
            unsafe { std::slice::from_raw_parts(spirv.as_ptr().cast::<u32>(), spirv.len() / 4) };
        let shader_module = unsafe {
            render_device
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .unwrap()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(COMPUTE_PUSH_CONSTANT_SIZE);
        let pipeline_layout = unsafe {
            render_device
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
                    None,
                )
                .unwrap()
        };

        let names: Vec<std::ffi::CString> = entry_points
            .iter()
            .map(|e| std::ffi::CString::new(e.as_str()).unwrap())
            .collect();
        let infos: Vec<vk::ComputePipelineCreateInfo> = names
            .iter()
            .map(|name| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader_module)
                            .name(name),
                    )
                    .layout(pipeline_layout)
            })
            .collect();
        let handles = unsafe {
            render_device
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
                .unwrap_or_else(|(_, e)| {
                    panic!("compute pipelines for {} failed: {e:?}", shader.path)
                })
        };
        log::info!(
            "Compiled compute module {} ({} kernels)",
            shader.path,
            handles.len()
        );
        // Pipelines hold no reference to the module once created.
        unsafe {
            render_device
                .device
                .destroy_shader_module(shader_module, None)
        };
        CompiledComputeModule {
            pipeline_layout,
            pipelines: entry_points.into_iter().zip(handles).collect(),
        }
    }

    fn destroy_asset(render_device: &RenderDevice, prepared: &Self::PreparedAsset) {
        for pipeline in prepared.pipelines.values() {
            render_device.destroyer.destroy_pipeline(*pipeline);
        }
        render_device
            .destroyer
            .destroy_pipeline_layout(prepared.pipeline_layout);
    }
}

/// Workgroups for `threads` threads, clamped to the 1D dispatch limit.
pub fn group_count(threads: u32) -> u32 {
    threads.div_ceil(WORKGROUP_SIZE).clamp(1, MAX_GROUPS)
}

/// Binds `entry`, pushes `params`, dispatches `threads` threads (in chunks if they exceed one
/// dispatch; the kernel's `base` push-constant field, at byte offset `base_offset`, receives
/// each chunk's first thread index when `base_offset` is given).
pub fn record_dispatch<P: bytemuck::Pod>(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    module: &CompiledComputeModule,
    entry: &str,
    params: &P,
    threads: u32,
    base_offset: Option<usize>,
) {
    assert!(std::mem::size_of::<P>() as u32 <= COMPUTE_PUSH_CONSTANT_SIZE);
    if threads == 0 {
        return;
    }
    let mut bytes = bytemuck::bytes_of(params).to_vec();
    let chunk = MAX_GROUPS * WORKGROUP_SIZE;
    let mut base = 0u32;
    unsafe {
        render_device.device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            module.pipeline(entry),
        );
        while base < threads {
            if let Some(off) = base_offset {
                bytes[off..off + 4].copy_from_slice(&base.to_le_bytes());
            }
            render_device.device.cmd_push_constants(
                cmd_buffer,
                module.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &bytes,
            );
            let n = (threads - base).min(chunk);
            render_device
                .device
                .cmd_dispatch(cmd_buffer, group_count(n), 1, 1);
            base += n;
            if base_offset.is_none() {
                break;
            }
        }
    }
}

/// Binds `entry`, pushes `params`, and dispatches from `(x, y, z)` at `offset` in `buffer`.
pub fn record_dispatch_indirect<P: bytemuck::Pod>(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    module: &CompiledComputeModule,
    entry: &str,
    params: &P,
    buffer: vk::Buffer,
    offset: u64,
) {
    assert!(std::mem::size_of::<P>() as u32 <= COMPUTE_PUSH_CONSTANT_SIZE);
    unsafe {
        render_device.device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::COMPUTE,
            module.pipeline(entry),
        );
        render_device.device.cmd_push_constants(
            cmd_buffer,
            module.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(params),
        );
        render_device
            .device
            .cmd_dispatch_indirect(cmd_buffer, buffer, offset);
    }
}

/// Global memory barrier between two compute dispatches (writes of the first visible to reads
/// and writes of the second). Buffers are addressed through pointers, so a global barrier is
/// the only kind that applies.
pub fn compute_to_compute_barrier(render_device: &RenderDevice, cmd_buffer: vk::CommandBuffer) {
    memory_barrier(
        render_device,
        cmd_buffer,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::INDIRECT_COMMAND_READ,
    );
}

pub fn memory_barrier(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access);
    unsafe {
        render_device.ext_sync2.cmd_pipeline_barrier2(
            cmd_buffer,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier)),
        );
    }
}

/// Re-prepares a module when its shader is modified on disk.
fn propagate_modified(
    modules: Res<Assets<ComputeModule>>,
    mut shader_events: MessageReader<AssetEvent<Shader>>,
    mut module_events: MessageWriter<AssetEvent<ComputeModule>>,
) {
    for event in shader_events.read() {
        if let AssetEvent::Modified { id } = event {
            for (module_id, module) in modules.iter() {
                if module.shader.id() == *id {
                    module_events.write(AssetEvent::Modified { id: module_id });
                }
            }
        }
    }
}

/// Loaded compute modules, by name: `render_app.world().resource::<VulkanAssets<ComputeModule>>()`.
pub type ComputeModules = VulkanAssets<ComputeModule>;

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<ComputeModule>();
        app.init_vulkan_asset::<ComputeModule>();
        app.add_systems(Update, propagate_modified);
    }
}
