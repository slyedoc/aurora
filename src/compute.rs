//! Compute kernels from Slang modules.
//!
//! A [`ComputeModule`] is one `.slang` file plus the entry points to build pipelines for. Every
//! kernel shares one pipeline layout: no descriptor sets, just a [`COMPUTE_PUSH_CONSTANT_SIZE`]
//! push-constant block that carries raw buffer pointers (`SPV_KHR_physical_storage_buffer`) and
//! a few scalars. A module made with [`ComputeModule::with_heap`] instead reads the descriptor
//! heap through its mappings (storage images) and takes the same block as heap push data.
//! Recording is [`record_dispatch`] / [`record_dispatch_indirect`] plus the barrier helpers; the
//! module reloads with its shader like the ray-tracing pipeline does.

use std::collections::HashMap;

use ash::vk;
use bevy::{
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    prelude::*,
};

use crate::{
    descriptor_heap::{Mapping, stage_mapping},
    render_device::RenderDevice,
    shader::Shader,
    tlas_builder::{InstanceBlockSlots, TLAS},
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
    /// Descriptor-heap mappings (empty: push constants only, no heap).
    pub heap: Vec<Mapping>,
}

impl ComputeModule {
    pub fn new(shader: Handle<Shader>, entry_points: &[&str]) -> Self {
        Self::with_heap(shader, entry_points, &[])
    }

    /// Kernels that read the descriptor heap through `heap` (bound and pushed at each dispatch).
    pub fn with_heap(shader: Handle<Shader>, entry_points: &[&str], heap: &[Mapping]) -> Self {
        Self {
            shader,
            entry_points: entry_points.iter().map(|s| s.to_string()).collect(),
            heap: heap.to_vec(),
        }
    }
}

pub struct CompiledComputeModule {
    /// Null for a heap module.
    pub pipeline_layout: vk::PipelineLayout,
    pub pipelines: HashMap<String, vk::Pipeline>,
    pub heap: bool,
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
    type ExtractedAsset = (Shader, Vec<String>, Vec<Mapping>);
    type ExtractParam = SRes<Assets<Shader>>;
    type PreparedAsset = CompiledComputeModule;

    fn extract_asset(
        &self,
        param: &mut SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let shaders: &Assets<crate::shader::Shader> = &**param;
        let shader = shaders.get(&self.shader)?;
        shader.spirv.as_ref()?;
        Some((shader.clone(), self.entry_points.clone(), self.heap.clone()))
    }

    fn prepare_asset(
        (shader, entry_points, heap): Self::ExtractedAsset,
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
        let pipeline_layout = if heap.is_empty() {
            unsafe {
                render_device
                    .device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
                        None,
                    )
                    .unwrap()
            }
        } else {
            vk::PipelineLayout::null()
        };
        let mappings = if heap.is_empty() {
            Vec::new()
        } else {
            render_device.heaps().mappings(&heap)
        };
        let mapping_info = stage_mapping(&mappings);
        let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default().flags(
            vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT
                | crate::render_device::capture_statistics(),
        );
        let mut flags2_plain = vk::PipelineCreateFlags2CreateInfo::default()
            .flags(crate::render_device::capture_statistics());

        let names: Vec<std::ffi::CString> = entry_points
            .iter()
            .map(|e| std::ffi::CString::new(e.as_str()).unwrap())
            .collect();
        let infos: Vec<vk::ComputePipelineCreateInfo> = names
            .iter()
            .map(|name| {
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(shader_module)
                    .name(name);
                let info = vk::ComputePipelineCreateInfo::default().layout(pipeline_layout);
                if heap.is_empty() {
                    let mut info = info.stage(stage);
                    info.p_next = (&raw mut flags2_plain).cast();
                    info
                } else {
                    stage.p_next = (&raw const mapping_info).cast();
                    let mut info = info.stage(stage);
                    info.p_next = (&raw mut flags2).cast();
                    info
                }
            })
            .collect();
        let handles: Vec<vk::Pipeline> = infos
            .iter()
            .zip(&entry_points)
            .map(|(info, entry)| {
                let label = format!("{}#{entry}", shader.path);
                crate::pipeline_binary::create_pipeline(
                    render_device,
                    &label,
                    info,
                    |info, cache| unsafe {
                        render_device
                            .device
                            .create_compute_pipelines(cache, std::slice::from_ref(info), None)
                            .map(|p| p[0])
                            .map_err(|(_, e)| e)
                    },
                )
                .unwrap_or_else(|e| panic!("compute pipelines for {} failed: {e:?}", shader.path))
            })
            .collect();
        log::debug!(
            "Compiled compute module {} ({} kernels)",
            shader.path,
            handles.len()
        );
        for (entry, handle) in entry_points.iter().zip(&handles) {
            crate::render_device::log_executable_statistics(render_device, entry, *handle);
        }
        // Pipelines hold no reference to the module once created.
        unsafe {
            render_device
                .device
                .destroy_shader_module(shader_module, None)
        };
        CompiledComputeModule {
            pipeline_layout,
            pipelines: entry_points.into_iter().zip(handles).collect(),
            heap: !heap.is_empty(),
        }
    }

    fn destroy_asset(render_device: &RenderDevice, prepared: &Self::PreparedAsset) {
        for pipeline in prepared.pipelines.values() {
            render_device.destroyer.destroy_pipeline(*pipeline);
        }
        if prepared.pipeline_layout != vk::PipelineLayout::null() {
            render_device
                .destroyer
                .destroy_pipeline_layout(prepared.pipeline_layout);
        }
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
        if module.heap {
            render_device.heaps().bind(cmd_buffer);
        }
        while base < threads {
            if let Some(off) = base_offset {
                bytes[off..off + 4].copy_from_slice(&base.to_le_bytes());
            }
            push(render_device, cmd_buffer, module, &bytes);
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

/// The kernel's parameter block: heap push data for a heap module, push constants otherwise.
fn push(rd: &RenderDevice, cmd: vk::CommandBuffer, module: &CompiledComputeModule, bytes: &[u8]) {
    if module.heap {
        rd.heaps().push_bytes(cmd, bytes);
    } else {
        unsafe {
            rd.device.cmd_push_constants(
                cmd,
                module.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
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
        if module.heap {
            render_device.heaps().bind(cmd_buffer);
        }
        push(
            render_device,
            cmd_buffer,
            module,
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
        render_device.device.cmd_pipeline_barrier2(
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

/// Work recorded into the frame's own command buffer: a system with a small per-frame dispatch
/// queues it here instead of a blocking `run_transfer_commands`, which stalls the CPU on the GPU
/// every frame. Captured data must be by value (it runs in `Last`); whatever lives in the TLAS's
/// storage (instance rows, the acceleration structure) comes from the [`FrameContext`], resolved
/// as the command records, never captured.
pub type FrameCommand = Box<
    dyn for<'a> FnOnce(&RenderDevice, &ComputeModules, vk::CommandBuffer, &FrameContext<'a>)
        + Send
        + Sync,
>;

/// What a [`FrameCommand`] may touch of the TLAS, as it records: its instance table already
/// holds every slot handed out (`TLAS::prepare_storage` runs first).
pub struct FrameContext<'a> {
    tlas: &'a TLAS,
}

impl<'a> FrameContext<'a> {
    pub(crate) fn new(tlas: &'a TLAS) -> Self {
        Self { tlas }
    }

    /// The rows a kernel writes for `block` (instance_rows.slang), or `None` while its mesh and
    /// material are still resolving (its rows' static half is not in place yet).
    pub fn block_rows(&self, block: InstanceBlockSlots) -> Option<BlockRows> {
        self.tlas.block_rows(block)
    }

    /// The TLAS as last built before this command runs (the frame commands: last frame's;
    /// after the build: this frame's), with its material records; `None` before the first build.
    pub fn tlas(&self) -> Option<TlasRef> {
        let tlas = self.tlas.acceleration_structure.address;
        (tlas != 0).then(|| TlasRef {
            tlas,
            materials: self.tlas.material_address(),
        })
    }
}

/// An [`InstanceBlock`](crate::tlas_builder::InstanceBlock)'s rows of this frame's and last
/// frame's instance arrays, and how many (instance_rows.slang `BlockRows`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockRows {
    pub rows: u64,
    pub prev: u64,
    /// The block's per-slot tints (`TLAS::instance_tints_address`).
    pub tints: u64,
    pub count: u32,
    pub pad: u32,
}

/// A TLAS to trace inline ray queries against, and its material records.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TlasRef {
    pub tlas: u64,
    pub materials: u64,
}

/// This frame's [`FrameCommand`]s: those recorded at its start, ahead of the TLAS build (they
/// may write instance rows), and those recorded right after it (they trace this frame's TLAS).
#[derive(Resource, Default)]
pub struct FrameCommands {
    uploads: Vec<FrameUpload>,
    before_build: Vec<FrameCommand>,
    after_build: Vec<FrameCommand>,
}

/// A transfer into an existing resource (an in-place asset re-upload), recorded first.
pub type FrameUpload = Box<dyn FnOnce(&RenderDevice, vk::CommandBuffer) + Send + Sync>;

impl FrameCommands {
    /// Queues an upload ahead of everything else this frame (its own GPU scope).
    pub fn push_upload(
        &mut self,
        f: impl FnOnce(&RenderDevice, vk::CommandBuffer) + Send + Sync + 'static,
    ) {
        self.uploads.push(Box::new(f));
    }

    pub fn has_uploads(&self) -> bool {
        !self.uploads.is_empty()
    }

    /// Records the queued uploads; each leaves its own writes visible to what reads them.
    pub fn record_uploads(&mut self, render_device: &RenderDevice, cmd_buffer: vk::CommandBuffer) {
        for upload in self.uploads.drain(..) {
            upload(render_device, cmd_buffer);
        }
    }

    /// Queues `f` ahead of this frame's TLAS build.
    pub fn push(
        &mut self,
        f: impl for<'a> FnOnce(&RenderDevice, &ComputeModules, vk::CommandBuffer, &FrameContext<'a>)
        + Send
        + Sync
        + 'static,
    ) {
        self.before_build.push(Box::new(f));
    }

    /// Queues `f` right after this frame's TLAS build (dropped on a frame without one).
    pub fn push_after_tlas(
        &mut self,
        f: impl for<'a> FnOnce(&RenderDevice, &ComputeModules, vk::CommandBuffer, &FrameContext<'a>)
        + Send
        + Sync
        + 'static,
    ) {
        self.after_build.push(Box::new(f));
    }

    /// Records the commands queued ahead of the build into `cmd_buffer`, then makes their writes
    /// visible to the rest of the frame.
    pub fn record(
        &mut self,
        render_device: &RenderDevice,
        modules: &ComputeModules,
        cmd_buffer: vk::CommandBuffer,
        ctx: &FrameContext,
    ) {
        if self.before_build.is_empty() {
            return;
        }
        for command in self.before_build.drain(..) {
            command(render_device, modules, cmd_buffer, ctx);
        }
        memory_barrier(
            render_device,
            cmd_buffer,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
    }

    /// Records the commands queued after the build into `cmd_buffer`, between barriers from the
    /// build and to everything after.
    pub fn record_after_tlas(
        &mut self,
        render_device: &RenderDevice,
        modules: &ComputeModules,
        cmd_buffer: vk::CommandBuffer,
        ctx: &FrameContext,
    ) {
        if self.after_build.is_empty() {
            return;
        }
        if ctx.tlas().is_none() {
            self.after_build.clear();
            return;
        }
        memory_barrier(
            render_device,
            cmd_buffer,
            vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags2::SHADER_READ,
        );
        for command in self.after_build.drain(..) {
            command(render_device, modules, cmd_buffer, ctx);
        }
        memory_barrier(
            render_device,
            cmd_buffer,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
    }
}

/// Loaded compute modules, by name: `render_app.world().resource::<VulkanAssets<ComputeModule>>()`.
pub type ComputeModules = VulkanAssets<ComputeModule>;

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<ComputeModule>();
        app.init_resource::<FrameCommands>();
        app.init_vulkan_asset::<ComputeModule>();
        app.add_systems(Update, propagate_modified);
    }
}

#[cfg(test)]
mod tests {
    /// instance_rows.slang reads `BlockRows` by device address: two pointers, the count, a pad.
    #[test]
    fn block_rows_match_the_shader() {
        use core::mem::{offset_of, size_of};
        let source = include_str!("../assets/shaders/instance_rows.slang");
        let body = source
            .split_once("public struct BlockRows {")
            .expect("instance_rows.slang declares BlockRows")
            .1;
        let fields: Vec<&str> = body[..body.find('}').unwrap()]
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("public "))
            .collect();
        assert_eq!(
            fields,
            [
                "public Instance* rows;",
                "public Instance* prev;",
                "public float4* tints;",
                "public uint count;",
                "public uint pad;"
            ]
        );
        assert_eq!(offset_of!(super::BlockRows, prev), 8);
        assert_eq!(offset_of!(super::BlockRows, tints), 16);
        assert_eq!(offset_of!(super::BlockRows, count), 24);
        assert_eq!(size_of::<super::BlockRows>(), 32);
    }
}
