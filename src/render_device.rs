use std::{
    collections::{HashMap, VecDeque},
    ffi::{CStr, c_char},
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use ash::vk::TaggedStructure as _;
use ash::{
    ext::opacity_micromap,
    khr::{
        acceleration_structure, deferred_host_operations, ray_query, ray_tracing_pipeline,
        ray_tracing_position_fetch, surface, swapchain,
    },
    nv::low_latency2,
};
use bevy::prelude::*;
use crossbeam::channel::Sender;
use gpu_allocator::{AllocationError, MemoryLocation, vulkan::*};
use raw_window_handle::DisplayHandle;

use crate::render_texture::RenderTexture;
use crate::xr::XrContext;

pub struct AllocatorState {
    allocator: Arc<Mutex<Allocator>>,
    image_allocations: HashMap<vk::Image, Allocation>,
    buffer_allocations: HashMap<vk::Buffer, Allocation>,
}

impl AllocatorState {
    pub fn allocate(
        &mut self,
        desc: &AllocationCreateDesc<'_>,
    ) -> Result<Allocation, AllocationError> {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.allocate(desc)
    }

    pub fn register_image_allocation(&mut self, image: vk::Image, allocation: Allocation) {
        self.image_allocations.insert(image, allocation);
    }

    pub fn register_buffer_allocation(&mut self, buffer: vk::Buffer, allocation: Allocation) {
        self.buffer_allocations.insert(buffer, allocation);
    }

    pub fn get_buffer_allocation<'a>(&'a self, buffer: vk::Buffer) -> Option<&'a Allocation> {
        self.buffer_allocations.get(&buffer)
    }

    pub fn free_image_allocation(&mut self, image: vk::Image) {
        let mut allocator = self.allocator.lock().unwrap();
        if let Some(allocation) = self.image_allocations.remove(&image) {
            allocator.free(allocation).unwrap();
        }
    }

    pub fn free_buffer_allocation(&mut self, buffer: vk::Buffer) {
        let mut allocator = self.allocator.lock().unwrap();
        if let Some(allocation) = self.buffer_allocations.remove(&buffer) {
            allocator.free(allocation).unwrap();
        }
    }

    /// The returned smart pointer must not outlive the allocator itself.
    pub fn unchecked_borrow_allocator(&self) -> Arc<Mutex<Allocator>> {
        return self.allocator.clone();
    }
}

impl Drop for AllocatorState {
    fn drop(&mut self) {
        assert_eq!(
            Arc::strong_count(&self.allocator),
            1,
            "something is borrowing the allocator still :("
        );
    }
}

/// VK_KHR_pipeline_executable_properties is on: pipelines built with
/// [`capture_statistics`] report each shader's registers, binary size and stack
/// ([`log_executable_statistics`]). NVIDIA reports compute pipelines; ray tracing pipelines
/// return no executables.
pub static EXECUTABLE_STATISTICS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `CAPTURE_STATISTICS_KHR` when the extension is on.
pub fn capture_statistics() -> vk::PipelineCreateFlags2 {
    if EXECUTABLE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
        vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR
    } else {
        vk::PipelineCreateFlags2::empty()
    }
}

/// Logs (debug) the driver's statistics for every shader executable of `pipeline` (built with
/// `CAPTURE_STATISTICS_KHR`): register count, instruction count, local memory and the rest.
pub fn log_executable_statistics(rd: &RenderDevice, label: &str, pipeline: vk::Pipeline) {
    if !EXECUTABLE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let ext = ash::khr::pipeline_executable_properties::Device::load(&rd.instance, &rd.device);
    let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
    let Ok(n) = (unsafe { ext.get_pipeline_executable_properties_len(&info) }) else {
        return;
    };
    let mut executables = vec![vk::PipelineExecutablePropertiesKHR::default(); n];
    if unsafe { ext.get_pipeline_executable_properties(&info, &mut executables) }.is_err() {
        return;
    }
    for (index, exe) in executables.iter().enumerate() {
        let name = exe
            .name_as_c_str()
            .map_or("?".into(), |n| n.to_string_lossy());
        let info = vk::PipelineExecutableInfoKHR::default()
            .pipeline(pipeline)
            .executable_index(index as u32);
        let Ok(n) = (unsafe { ext.get_pipeline_executable_statistics_len(&info) }) else {
            continue;
        };
        let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); n];
        if unsafe { ext.get_pipeline_executable_statistics(&info, &mut stats) }.is_err() {
            continue;
        }
        let values: Vec<String> = stats
            .iter()
            .map(|st| {
                let n = st
                    .name_as_c_str()
                    .map_or("?".into(), |n| n.to_string_lossy());
                let v = unsafe {
                    match st.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                            st.value.b32.to_string()
                        }
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => st.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                            st.value.u64.to_string()
                        }
                        _ => format!("{:.2}", st.value.f64),
                    }
                };
                format!("{n}={v}")
            })
            .collect();
        log::debug!("shader stats {label} [{name}]: {}", values.join(", "));
    }
}

/// The queue families buffers and images are shared between (graphics, then the compute-only
/// family when the device has one; async compute, docs/plans/async-compute.html). Set once, when
/// the device is made.
static SHARING_FAMILIES: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

/// The sharing mode and families for a buffer or image both queues may touch: CONCURRENT over
/// the graphics and compute families, or EXCLUSIVE with one queue.
pub fn sharing() -> (vk::SharingMode, &'static [u32]) {
    match SHARING_FAMILIES.get().map(Vec::as_slice) {
        Some(families) if families.len() > 1 => (vk::SharingMode::CONCURRENT, families),
        _ => (vk::SharingMode::EXCLUSIVE, &[]),
    }
}

/// The compute-only queue (async compute): its family, the queue, the frame's command buffers
/// per parity, and the timeline its submits signal. Every frame's last graphics submit waits on
/// its last compute submit, so the main timeline stays the retirement clock.
pub struct ComputeQueue {
    pub family: u32,
    pub queue: Mutex<vk::Queue>,
    pub pool: vk::CommandPool,
    pub command_buffers: [vk::CommandBuffer; 2],
    /// The second compute submit per frame (ReSTIR's light grids, beside the primary pass),
    /// from a pool of its own: the view lane records it while the scene lane records the first.
    pub grid_pool: vk::CommandPool,
    pub grid_buffers: [vk::CommandBuffer; 2],
    pub timeline: crate::timeline::GpuTimeline,
}

pub struct RenderDeviceData {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub ext_surface: surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: Mutex<vk::Queue>,
    pub queue_family_idx: u32,
    pub ext_swapchain: swapchain::Device,
    pub ext_rtx_pipeline: ray_tracing_pipeline::Device,
    pub ext_acc_struct: acceleration_structure::Device,
    /// `VK_EXT_opacity_micromap`, when the device has it (omm.rs); `None` otherwise.
    pub ext_micromap: Option<opacity_micromap::Device>,
    /// `VK_NV_low_latency2` (Reflex) with `VK_KHR_present_id`, on a windowed, non-XR NVIDIA
    /// device (latency.rs); `None` otherwise.
    pub ext_low_latency: Option<low_latency2::Device>,
    pub command_pool: vk::CommandPool,
    pub bindless_descriptor_map: Mutex<BindlessMap>,
    /// The resource and sampler descriptor heaps (descriptor_heap.rs), set right after the
    /// device is made.
    heaps: std::sync::OnceLock<crate::descriptor_heap::DescriptorHeaps>,
    pub transfer_command_pool: Mutex<vk::CommandPool>,
    pub command_buffers: [vk::CommandBuffer; 2],
    /// The frame's other recorders (render_frame: the view and present lanes, the scene lane's
    /// second half -- TLAS, lights -- submitted after the async compute it waits on, and the view
    /// lane's back half -- after the first view's primary pass, waiting on the light grids), each
    /// with its own pool so they record on their own threads; `command_pool` is the scene lane's.
    /// The view lane's two halves share lane 0's thread, so lane 3 is recorded on it.
    pub lanes: [(vk::CommandPool, [vk::CommandBuffer; 2]); 4],
    /// Every submit on the queue signals the next point on it (timeline.rs).
    pub timeline: crate::timeline::GpuTimeline,
    /// The compute-only queue, when the device has one.
    pub compute: Option<ComputeQueue>,
    /// Every pipeline compiles through this; seeded from and saved to disk
    /// ([`RenderDevice::save_pipeline_cache`]) so a relaunch skips the driver's compile.
    pub pipeline_cache: vk::PipelineCache,
    /// Command-buffer labels (gpu_profiler.rs).
    pub ext_debug_utils: ash::ext::debug_utils::Device,
    pub destroyer: ManuallyDrop<VkDestroyer>,
    pub allocator_state: Arc<Mutex<ManuallyDrop<AllocatorState>>>,
    /// The validation messenger (debug / `dev` builds): destroyed right before the instance.
    debug_messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl std::ops::Deref for RenderDeviceData {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

#[derive(Resource, Deref)]
pub struct RenderDevice(Arc<RenderDeviceData>);

/// Bindless texture slots: view -> slot, plus recycled slots. Slots are only ever rebound, never
/// cleared (the set is partially bound), so a recycled slot keeps a stale descriptor until its
/// next registration — harmless, nothing samples an unregistered index.
#[derive(Default)]
pub struct BindlessMap {
    pub by_view: HashMap<vk::ImageView, u32>,
    pub free: Vec<u32>,
    /// Slots released while a frame that may sample them could still run: reusable once the
    /// timeline passes the point.
    pub retired: Vec<(crate::timeline::SubmitPoint, u32)>,
    /// Slots released since the last frame submit: the next frame submit's point retires them
    /// ([`RenderDevice::retire_released`]), however many earlier submits that frame makes.
    pub released: Vec<u32>,
    pub next: u32,
    /// Textures registered so far: grows whenever one arrives (reference.rs resets on it).
    pub registrations: u64,
}

impl Clone for RenderDevice {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl RenderDevice {
    /// Submits `cmd` on the async compute queue after the main timeline reaches `after` (the
    /// earlier submits its work reads, and the previous frame's); returns the compute timeline
    /// value it signals, which the graphics submit reading its results waits on.
    pub fn submit_compute(
        &self,
        cmd: vk::CommandBuffer,
        after: crate::timeline::SubmitPoint,
    ) -> u64 {
        let compute = self
            .compute
            .as_ref()
            .expect("submit_compute without a compute queue");
        let queue = compute.queue.lock().unwrap();
        unsafe {
            compute.timeline.submit_with(
                self,
                *queue,
                std::slice::from_ref(&cmd),
                &[],
                &[(
                    self.timeline.semaphore,
                    after.0,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                )],
                &[],
                &[],
                vk::Fence::null(),
                None,
            )
        }
        .unwrap_or_else(|e| {
            crate::aftermath::note_device_lost(e);
            panic!("compute submit failed: {e:?}");
        })
        .0
    }

    /// The device, with the window system's surface support when `display_handle` is some; a
    /// windowless device (no surface or swapchain extensions) renders only to image targets.
    pub unsafe fn from_display(
        display_handle: Option<&DisplayHandle>,
        xr: Option<&XrContext>,
    ) -> Self {
        // Before the loader is touched: Aftermath registers ahead of instance/device creation.
        crate::aftermath::enable();
        unsafe {
            let entry = ash::Entry::linked();
            let (instance, debug_messenger) = create_instance(display_handle, &entry, xr);
            let ext_surface = surface::Instance::load(&entry, &instance);
            let (physical_device, queue_family_idx) = pick_physical_device(&instance, xr);
            let compute_family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .position(|p| {
                    p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && p.timestamp_valid_bits > 0
                })
                .map(|i| i as u32);
            let (device, queue, compute_queue, micromaps, low_latency) = create_logical_device(
                &entry,
                &instance,
                physical_device,
                queue_family_idx,
                compute_family,
                xr,
                display_handle.is_some(),
            );
            let mut families = vec![queue_family_idx];
            families.extend(compute_family);
            let _ = SHARING_FAMILIES.set(families);
            log::info!("vulkan: async compute queue family: {compute_family:?}");
            let ext_swapchain = swapchain::Device::load(&instance, &device);
            let ext_rtx_pipeline = ray_tracing_pipeline::Device::load(&instance, &device);
            let ext_acc_struct = acceleration_structure::Device::load(&instance, &device);
            let ext_micromap =
                micromaps.then(|| opacity_micromap::Device::load(&instance, &device));
            let ext_low_latency =
                low_latency.then(|| low_latency2::Device::load(&instance, &device));
            let command_pool = create_command_pool(&device, queue_family_idx);
            let transfer_command_pool = Mutex::new(create_command_pool(&device, queue_family_idx));
            let command_buffers = create_command_buffers(&device, command_pool);
            let lanes = std::array::from_fn(|_| {
                let pool = create_command_pool(&device, queue_family_idx);
                (pool, create_command_buffers(&device, pool))
            });
            let pipeline_cache = create_pipeline_cache(&device);
            let ext_debug_utils = ash::ext::debug_utils::Device::load(&instance, &device);

            let allocator_state = Arc::new(Mutex::new(ManuallyDrop::new(AllocatorState {
                allocator: Arc::new(Mutex::new(
                    Allocator::new(&AllocatorCreateDesc {
                        instance: instance.clone(),
                        device: device.clone(),
                        physical_device,
                        debug_settings: Default::default(),
                        buffer_device_address: true, // Ideally, check the BufferDeviceAddressFeatures struct.
                        allocation_sizes: Default::default(),
                    })
                    .unwrap(),
                )),
                image_allocations: HashMap::new(),
                buffer_allocations: HashMap::new(),
            })));

            let destroyer =
                spawn_destroy_thread(instance.clone(), device.clone(), allocator_state.clone());

            let timeline = crate::timeline::GpuTimeline::new(&device);
            let compute = compute_family.zip(compute_queue).map(|(family, queue)| {
                let pool = create_command_pool(&device, family);
                let grid_pool = create_command_pool(&device, family);
                ComputeQueue {
                    family,
                    queue,
                    pool,
                    command_buffers: create_command_buffers(&device, pool),
                    grid_pool,
                    grid_buffers: create_command_buffers(&device, grid_pool),
                    timeline: crate::timeline::GpuTimeline::new(&device),
                }
            });
            let ret = RenderDevice(Arc::new(RenderDeviceData {
                entry,
                instance,
                ext_surface,
                physical_device,
                device,
                queue,
                queue_family_idx,
                ext_swapchain,
                ext_rtx_pipeline,
                ext_acc_struct,
                ext_micromap,
                ext_low_latency,
                command_pool,
                bindless_descriptor_map: Mutex::new(BindlessMap::default()),
                heaps: std::sync::OnceLock::new(),
                transfer_command_pool,
                command_buffers,
                lanes,
                timeline,
                compute,
                pipeline_cache,
                ext_debug_utils,
                destroyer,
                debug_messenger,
                allocator_state,
            }));
            let heaps = crate::descriptor_heap::DescriptorHeaps::new(&ret);
            let _ = ret.heaps.set(heaps);

            ret
        }
    }

    /// Writes the pipeline cache to disk; called after a pipeline compiles, not at exit, so a
    /// crashing teardown still keeps it.
    pub fn save_pipeline_cache(&self) {
        let Some(path) = pipeline_cache_path() else {
            return;
        };
        let data = match unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) } {
            Ok(data) => data,
            Err(err) => {
                log::warn!("pipeline cache: read back failed: {err:?}");
                return;
            }
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let partial = path.with_extension(format!("{}.part", std::process::id()));
        if std::fs::write(&partial, &data).is_ok() && std::fs::rename(&partial, &path).is_ok() {
            log::debug!("pipeline cache: saved {} KiB", data.len() / 1024);
        }
    }

    pub fn create_render_target(&self, image_info: &vk::ImageCreateInfo) -> vk::Image {
        let image = unsafe { self.device.create_image(image_info, None).unwrap() };
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let mut state = self.allocator_state.lock().unwrap();
        let allocation = state
            .allocate(&AllocationCreateDesc {
                name: "Image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(image),
            })
            .unwrap();

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        state.register_image_allocation(image, allocation);
        image
    }

    pub fn register_bindless_texture(&self, texture: &RenderTexture) -> u32 {
        let mut map = self.bindless_descriptor_map.lock().unwrap();
        if let Some(index) = map.by_view.get(&texture.image_view) {
            return *index;
        }

        let reached = self.timeline.last_completed(&self.device);
        let map = &mut *map;
        map.retired.retain(|&(point, slot)| {
            let done = point <= reached;
            if done {
                map.free.push(slot);
            }
            !done
        });
        let index = map.free.pop().unwrap_or_else(|| {
            let index = map.next;
            map.next += 1;
            index
        });
        // Past the bindless range the slot would land on the per-frame descriptors after it.
        assert!(
            index < crate::descriptor_heap::BINDLESS_SLOTS,
            "bindless textures exhausted ({} slots)",
            crate::descriptor_heap::BINDLESS_SLOTS
        );
        map.by_view.insert(texture.image_view, index);
        map.registrations += 1;

        self.heaps().write_bindless(index, texture.image_view);

        index
    }

    /// Textures registered so far (it grows as textures stream in).
    pub fn bindless_registrations(&self) -> u64 {
        self.bindless_descriptor_map.lock().unwrap().registrations
    }

    pub fn get_bindless_texture_index(&self, texture: &RenderTexture) -> Option<u32> {
        let map = self.bindless_descriptor_map.lock().unwrap();
        map.by_view.get(&texture.image_view).copied()
    }

    /// Forget a view's bindless slot before its deferred destruction. Vulkan reuses handle
    /// values, so a destroyed view's entry would otherwise alias whatever gets created next
    /// (the font atlas is re-created on every glyph-cache growth, so this bites early).
    pub fn unregister_bindless_texture(&self, texture: &RenderTexture) {
        let mut map = self.bindless_descriptor_map.lock().unwrap();
        if let Some(index) = map.by_view.remove(&texture.image_view) {
            // The frame being recorded may already use it and has not been submitted yet.
            map.released.push(index);
        }
    }

    /// After a frame submit at `point`: the slots released before it are free once it is done.
    pub fn retire_released(&self, point: crate::timeline::SubmitPoint) {
        let mut map = self.bindless_descriptor_map.lock().unwrap();
        let map = &mut *map;
        map.retired
            .extend(map.released.drain(..).map(|index| (point, index)));
    }

    /// The descriptor heaps (descriptor_heap.rs).
    pub fn heaps(&self) -> &crate::descriptor_heap::DescriptorHeaps {
        self.heaps
            .get()
            .expect("descriptor heaps are made with the device")
    }

    /// Creates an image view and notes how, so descriptors can be written from it.
    pub fn create_view(&self, info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        let view = unsafe { self.device.create_image_view(info, None) }.unwrap();
        if let Some(heaps) = self.heaps.get() {
            heaps.track_view(view, info);
        }
        view
    }

    pub fn load_shader(
        &self,
        spirv: &[u8],
        stage: vk::ShaderStageFlags,
    ) -> vk::PipelineShaderStageCreateInfo<'_> {
        self.load_shader_entry(spirv, stage, c"main")
    }

    /// [`Self::load_shader`] for a module whose entry point for this stage is `entry` (a Slang
    /// file holding several stages names each one).
    pub fn load_shader_entry<'a>(
        &self,
        spirv: &[u8],
        stage: vk::ShaderStageFlags,
        entry: &'a std::ffi::CStr,
    ) -> vk::PipelineShaderStageCreateInfo<'a> {
        let spirv: &[u32] =
            unsafe { std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4) };
        let shader_module = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
                .unwrap()
        };

        vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .module(shader_module)
            .name(entry)
    }

    /// Records `f` into a one-shot command buffer, submits it, and blocks until it finishes.
    ///
    /// The pool is held only while recording (command pools are externally synchronized, and the
    /// asset worker records concurrently with the render thread), and the queue only for the
    /// submit itself -- never across the fence wait, which would stall frame submission behind
    /// every asset upload.
    /// Record `f` into a one-off command buffer, submit it, and wait for it. The queue is
    /// locked only around the submit; the record and the fence wait run unlocked, so workers
    /// (asset uploads, BLAS builds) overlap freely.
    pub fn run_transfer_commands(&self, f: impl FnOnce(vk::CommandBuffer)) {
        let (cmd_buffer, fence) = self.record_transfer(f);
        {
            let queue = self.queue.lock().unwrap();
            self.submit_transfer(*queue, cmd_buffer, fence);
        }
        self.finish_transfer(cmd_buffer, fence);
    }

    /// [`run_transfer_commands`](Self::run_transfer_commands) for work that REWRITES buffers a
    /// submitted frame may still read (light tables, landscape pages): its submit waits, on
    /// the GPU, for everything already queued, so it never overlaps a frame in flight. Plain
    /// uploads into fresh buffers use `run_transfer_commands` and do not wait.
    pub fn run_transfer_commands_after_frames(&self, f: impl FnOnce(vk::CommandBuffer)) {
        let (cmd_buffer, fence) = self.record_transfer(f);
        {
            let queue = self.queue.lock().unwrap();
            let last = self.timeline.last_submitted();
            unsafe {
                self.timeline.submit(
                    &self.device,
                    *queue,
                    std::slice::from_ref(&cmd_buffer),
                    &[last],
                    &[],
                    &[],
                    fence,
                    None,
                )
            }
            .unwrap_or_else(|e| {
                crate::aftermath::note_device_lost(e);
                panic!("ordered transfer submit failed: {e:?}");
            });
        }
        self.finish_transfer(cmd_buffer, fence);
    }

    /// [`run_transfer_commands`](Self::run_transfer_commands) with the queue lock held for the
    /// whole record / submit / wait. For callers whose `f` drives a library that uses the
    /// device queue on its own -- NGX feature creation and release submit and wait internally,
    /// outside our mutex -- so no worker thread can touch the queue meanwhile. Two threads on
    /// one `VkQueue` is a data race on an externally synchronized object (seen as the GPU
    /// dropping off the bus when a texture upload landed during a DLSS rebuild).
    pub fn run_transfer_commands_exclusive(&self, f: impl FnOnce(vk::CommandBuffer)) {
        let queue = self.queue.lock().unwrap();
        let (cmd_buffer, fence) = self.record_transfer(f);
        self.submit_transfer(*queue, cmd_buffer, fence);
        self.finish_transfer(cmd_buffer, fence);
    }

    fn record_transfer(&self, f: impl FnOnce(vk::CommandBuffer)) -> (vk::CommandBuffer, vk::Fence) {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None) }.unwrap();
        let transfer_command_pool = self.transfer_command_pool.lock().unwrap();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*transfer_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buffer = unsafe { self.device.allocate_command_buffers(&alloc_info) }.unwrap()[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd_buffer, &begin_info) }.unwrap();

        f(cmd_buffer);

        unsafe { self.device.end_command_buffer(cmd_buffer) }.unwrap();
        (cmd_buffer, fence)
    }

    /// Caller holds the queue lock.
    fn submit_transfer(&self, queue: vk::Queue, cmd_buffer: vk::CommandBuffer, fence: vk::Fence) {
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buffer));
        unsafe {
            self.device
                .queue_submit(queue, std::slice::from_ref(&submit_info), fence)
                .unwrap();
        }
    }

    fn finish_transfer(&self, cmd_buffer: vk::CommandBuffer, fence: vk::Fence) {
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .unwrap_or_else(|e| {
                    crate::aftermath::note_device_lost(e);
                    panic!("transfer fence wait failed: {e:?}");
                });
            let transfer_command_pool = self.transfer_command_pool.lock().unwrap();
            self.device
                .free_command_buffers(*transfer_command_pool, std::slice::from_ref(&cmd_buffer));
            self.device.destroy_fence(fence, None);
        }
    }
}

impl Drop for RenderDeviceData {
    fn drop(&mut self) {
        log::debug!("Dropping RenderDevice");
        unsafe {
            let destroyer = ManuallyDrop::take(&mut self.destroyer);
            drop(destroyer);

            let mut alloc_state = self.allocator_state.lock().unwrap();
            if let Some(heaps) = self.heaps.get() {
                heaps.destroy_objects(&self.device);
                for buffer in heaps.buffers() {
                    alloc_state.free_buffer_allocation(buffer);
                    self.device.destroy_buffer(buffer, None);
                }
            }
            let alloc_state = ManuallyDrop::take(&mut *alloc_state);

            drop(alloc_state);

            self.destroy_pipeline_cache(self.pipeline_cache, None);
            {
                let transfer_command_pool = self.transfer_command_pool.lock().unwrap();
                self.destroy_command_pool(*transfer_command_pool, None);
            }
            self.destroy_command_pool(self.command_pool, None);
            for (pool, _) in self.lanes {
                self.destroy_command_pool(pool, None);
            }
            self.timeline.destroy(&self.device);
            if let Some(compute) = &self.compute {
                self.destroy_command_pool(compute.pool, None);
                self.destroy_command_pool(compute.grid_pool, None);
                compute.timeline.destroy(&self.device);
            }
            self.device.destroy_device(None);
            if let Some((debug_utils, messenger)) = &self.debug_messenger {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_instance(
    display_handle: Option<&DisplayHandle>,
    entry: &ash::Entry,
    xr: Option<&XrContext>,
) -> (
    ash::Instance,
    Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
) {
    unsafe {
        let app_name = CStr::from_bytes_with_nul_unchecked(b"VK RAYS\0");
        let mut layer_names: Vec<&CStr> = Vec::new();

        // Validation: on in debug builds and whenever the `dev` feature is on (which also
        // turns on synchronization validation -- the hazard checker for buffer / image access
        // races between submissions).
        let validate = cfg!(debug_assertions) || cfg!(feature = "dev");
        if validate {
            layer_names.push(CStr::from_bytes_with_nul_unchecked(
                b"VK_LAYER_KHRONOS_validation\0",
            ));
        }

        log::debug!(
            "vulkan: validation layers: {:?}",
            layer_names
                .iter()
                .map(|l| l.to_string_lossy())
                .collect::<Vec<_>>()
        );

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();
        let mut instance_extensions: Vec<*const c_char> = match display_handle {
            Some(display_handle) => {
                ash_window::enumerate_required_extensions(display_handle.as_raw())
                    .unwrap()
                    .to_vec()
            }
            None => Vec::new(),
        };
        // DLSS (NGX) asks for its own instance extensions; union them in.
        let ngx_instance_extensions = crate::dlss::instance_extensions();
        for ext in &ngx_instance_extensions {
            if !instance_extensions
                .iter()
                .any(|p| CStr::from_ptr(*p) == ext.as_c_str())
            {
                instance_extensions.push(ext.as_ptr());
            }
        }

        log::debug!(
            "vulkan: instance extensions: {:?}",
            instance_extensions
                .iter()
                .map(|e| CStr::from_ptr(*e).to_string_lossy())
                .collect::<Vec<_>>()
        );

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_4);

        let sync_validation = cfg!(feature = "dev");
        if sync_validation {
            instance_extensions.push(ash::ext::validation_features::NAME.as_ptr());
        }
        // Always: command-buffer labels name the passes in Nsight (gpu_profiler.rs), and the
        // messenger below needs it (without it the layer prints to stdout only, and the
        // sync-val tail is lost in scrollback exactly when a device loss needs it).
        instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        let enabled = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&instance_extensions);
        if sync_validation {
            instance_info = instance_info.push(&mut validation_features);
        }

        // XR: the runtime performs the create (xrCreateVulkanInstanceKHR), adding whatever
        // instance extensions its compositor interop needs on top of ours.
        let instance = match xr {
            Some(xr) => xr.create_vk_instance(entry, &instance_info),
            None => entry.create_instance(&instance_info, None).unwrap(),
        };

        // Route validation output through `log` and keep a tail for the crash reporter
        // (dump_validation_tail). The messenger is the instance's last child to go
        // (RenderDeviceData::drop), so the layer's own leak check stays quiet.
        let mut debug_messenger = None;
        if validate {
            let debug_utils = ash::ext::debug_utils::Instance::load(entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_utils_callback));
            match debug_utils.create_debug_utils_messenger(&info, None) {
                Ok(messenger) => {
                    log::info!("validation messenger armed (tail kept for device loss)");
                    debug_messenger = Some((debug_utils, messenger));
                }
                Err(err) => log::warn!("validation messenger failed: {err:?}"),
            }
        }
        (instance, debug_messenger)
    }
}

// ---- validation message capture --------------------------------------------------------------

/// Recent validation warnings/errors (newest last), kept for [`dump_validation_tail`]:
/// when the GPU falls off the bus the Aftermath dump is often truncated, and this tail is
/// the only evidence of a sync hazard building up to it.
static VALIDATION_TAIL: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
const VALIDATION_TAIL_CAP: usize = 200;

unsafe extern "system" fn debug_utils_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as S;
    let msg = unsafe { std::ffi::CStr::from_ptr((*data).p_message) }
        .to_string_lossy()
        .into_owned();
    if severity.contains(S::ERROR) {
        log::error!("vk: {msg}");
    } else if severity.contains(S::WARNING) {
        log::warn!("vk: {msg}");
    } else {
        log::debug!("vk: {msg}");
    }
    if severity.intersects(S::ERROR | S::WARNING) {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let mut tail = VALIDATION_TAIL.lock().unwrap();
        if tail.len() >= VALIDATION_TAIL_CAP {
            tail.pop_front();
        }
        tail.push_back(format!("[{ms}] {msg}"));
    }
    vk::FALSE
}

/// Write the validation tail next to the Aftermath dumps (same `AURORA_AFTERMATH_DIR` /
/// `./crash-dumps` resolution), fsynced — the box often gets hard-reset seconds after a
/// GPU loss. Called from `aftermath::note_device_lost`; a no-op when the tail is empty.
pub fn dump_validation_tail() {
    let tail: Vec<String> = VALIDATION_TAIL.lock().unwrap().iter().cloned().collect();
    if tail.is_empty() {
        return;
    }
    let dir = std::env::var_os("AURORA_AFTERMATH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_default()
                .join("crash-dumps")
        });
    let _ = std::fs::create_dir_all(&dir);
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("validation-tail-{ms}.log"));
    if let Ok(mut f) = std::fs::File::create(&path) {
        use std::io::Write as _;
        let _ = f.write_all(
            tail.join(
                "
",
            )
            .as_bytes(),
        );
        let _ = f.sync_all();
        log::error!(
            "wrote validation tail ({} messages) -> {}",
            tail.len(),
            path.display()
        );
    }
}

unsafe fn pick_physical_device(
    instance: &ash::Instance,
    xr: Option<&XrContext>,
) -> (vk::PhysicalDevice, u32) {
    unsafe {
        // XR: the device is not ours to choose — take the one driving the HMD.
        if let Some(xr) = xr {
            let physical_device = xr.vk_physical_device(instance);
            let properties = instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family_idx = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .expect("XR physical device has no graphics queue")
                as u32;
            let info = instance.get_physical_device_properties(physical_device);
            log::info!(
                "vulkan: {} (OpenXR)",
                CStr::from_ptr(info.device_name.as_ptr()).to_string_lossy()
            );
            return (physical_device, queue_family_idx);
        }
        let all_devices = instance.enumerate_physical_devices().unwrap();
        log::debug!(
            "vulkan: devices: {:?}",
            all_devices
                .iter()
                .map(|d| {
                    let info = instance.get_physical_device_properties(*d);
                    CStr::from_ptr(info.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>()
        );

        let (physical_device, queue_family_idx) = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find_map(|d| {
                let info = instance.get_physical_device_properties(d);
                if !CStr::from_ptr(info.device_name.as_ptr())
                    .to_str()
                    .unwrap()
                    .contains("NVIDIA")
                {
                    return None;
                }
                if info.api_version < vk::API_VERSION_1_4 {
                    log::warn!(
                        "vulkan: {} offers Vulkan {}.{}, aurora needs 1.4 (update the driver)",
                        CStr::from_ptr(info.device_name.as_ptr()).to_string_lossy(),
                        vk::api_version_major(info.api_version),
                        vk::api_version_minor(info.api_version)
                    );
                    return None;
                }

                let properties = instance.get_physical_device_queue_family_properties(d);
                properties.iter().enumerate().find_map(|(i, p)| {
                    if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some((d, i as u32))
                    } else {
                        None
                    }
                })
            })
            .expect("no NVIDIA device with Vulkan 1.4 and a graphics queue");

        let device_properties = instance.get_physical_device_properties(physical_device);
        log::info!(
            "vulkan: {}",
            CStr::from_ptr(device_properties.device_name.as_ptr()).to_string_lossy()
        );
        (physical_device, queue_family_idx)
    }
}

unsafe fn create_logical_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_idx: u32,
    compute_family: Option<u32>,
    xr: Option<&XrContext>,
    windowed: bool,
) -> (
    ash::Device,
    Mutex<vk::Queue>,
    Option<Mutex<vk::Queue>>,
    bool,
    bool,
) {
    unsafe {
        let mut device_extensions = vec![
            acceleration_structure::NAME.as_ptr(),
            ray_tracing_pipeline::NAME.as_ptr(),
            // Picking (picking.rs): inline ray queries from compute, reading the hit triangle's
            // vertices out of the BLAS.
            ray_query::NAME.as_ptr(),
            ray_tracing_position_fetch::NAME.as_ptr(),
            // Shader Execution Reordering: the raygen regroups its hits before shading.
            vk::NV_RAY_TRACING_INVOCATION_REORDER_NAME.as_ptr(),
            deferred_host_operations::NAME.as_ptr(),
        ];
        if windowed {
            device_extensions.push(swapchain::NAME.as_ptr());
        }
        // DLSS (NGX) device extensions (VK_NVX_binary_import, VK_NVX_image_view_handle, ...):
        // they must be present at vkCreateDevice, and a device without them still has to
        // come up -- NGX then reports the feature unavailable.
        let ngx_device_extensions =
            crate::dlss::device_extensions(instance.handle(), physical_device);
        let supported = instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default();
        for ext in &ngx_device_extensions {
            let known = supported
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == ext.as_c_str());
            let listed = device_extensions
                .iter()
                .any(|p| CStr::from_ptr(*p) == ext.as_c_str());
            if !known {
                log::warn!(
                    "dlss: device lacks {} -- DLSS will be unavailable",
                    ext.to_string_lossy()
                );
            } else if !listed {
                device_extensions.push(ext.as_ptr());
            }
        }

        // Opacity micromaps (omm.rs): optional, so a device without them still comes up.
        let micromaps = supported
            .iter()
            .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == opacity_micromap::NAME);
        if micromaps {
            device_extensions.push(opacity_micromap::NAME.as_ptr());
        } else {
            log::warn!(
                "device lacks VK_EXT_opacity_micromap -- alpha cutouts stay on the any-hit path"
            );
        }

        // Reflex (latency.rs): low-latency sleep and markers, keyed by present ids. Windowed,
        // flat only: in XR the OpenXR runtime paces the frame.
        let has = |name: &CStr| {
            supported
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == name)
        };
        let low_latency =
            windowed && xr.is_none() && has(low_latency2::NAME) && has(ash::khr::present_id::NAME);
        if low_latency {
            device_extensions.push(low_latency2::NAME.as_ptr());
            device_extensions.push(ash::khr::present_id::NAME.as_ptr());
        }

        // Nsight Aftermath (aftermath.rs): with the callbacks armed, ask the driver to
        // track resources, shader debug info, checkpoints, and shader error reporting, so
        // a crash dump names the faulting buffer / shader source line instead of a bare
        // VA ("Shader mapping: null"). NV-only extension; skipped when unsupported.
        let diagnostics = crate::aftermath::is_armed()
            && supported.iter().any(|p| {
                CStr::from_ptr(p.extension_name.as_ptr()) == vk::NV_DEVICE_DIAGNOSTICS_CONFIG_NAME
            });
        if diagnostics {
            device_extensions.push(vk::NV_DEVICE_DIAGNOSTICS_CONFIG_NAME.as_ptr());
        }

        log::debug!(
            "vulkan: device extensions: {:?}",
            device_extensions
                .iter()
                .map(|e| CStr::from_ptr(*e).to_string_lossy())
                .collect::<Vec<_>>()
        );

        // An internally synchronized queue (VK_KHR_internally_synchronized_queues): the driver
        // serializes access, so a submit that bypasses our mutex (NGX and the OpenXR runtime
        // submit on this queue on their own) can no longer race one of ours.
        let internal_sync = has(vk::KHR_INTERNALLY_SYNCHRONIZED_QUEUES_NAME) && {
            let mut query = vk::PhysicalDeviceInternallySynchronizedQueuesFeaturesKHR::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push(&mut query);
            instance.get_physical_device_features2(physical_device, &mut features);
            query.internally_synchronized_queues == vk::TRUE
        };
        if internal_sync {
            device_extensions.push(vk::KHR_INTERNALLY_SYNCHRONIZED_QUEUES_NAME.as_ptr());
        }
        let queue_flags = if internal_sync {
            vk::DeviceQueueCreateFlags::INTERNALLY_SYNCHRONIZED_KHR
        } else {
            vk::DeviceQueueCreateFlags::empty()
        };
        log::info!("vulkan: internally synchronized queue: {internal_sync}");
        // Descriptor heaps (descriptor_heap.rs) are the only binding model: no fallback.
        let heap_supported = has(ash::ext::descriptor_heap::NAME) && {
            let mut query = vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push(&mut query);
            instance.get_physical_device_features2(physical_device, &mut features);
            query.descriptor_heap == vk::TRUE
        };
        assert!(
            heap_supported,
            "aurora needs VK_EXT_descriptor_heap (NVIDIA driver 610 or newer)"
        );
        device_extensions.push(ash::ext::descriptor_heap::NAME.as_ptr());
        // The raygen times each pixel for the Cost debug view (debug_view.rs).
        device_extensions.push(ash::khr::shader_clock::NAME.as_ptr());
        let executable_stats = has(ash::khr::pipeline_executable_properties::NAME);
        if executable_stats {
            device_extensions.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        }
        EXECUTABLE_STATISTICS.store(executable_stats, std::sync::atomic::Ordering::Relaxed);
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .flags(queue_flags)
                .queue_family_index(queue_family_idx)
                .queue_priorities(&[1.0]),
        ];
        if let Some(family) = compute_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .flags(queue_flags)
                    .queue_family_index(family)
                    .queue_priorities(&[1.0]),
            );
        }

        // gpu_profiler.rs resets its timestamps from the host after reading them. 16-bit
        // arithmetic: Slang's f16tof32 / f32tof16 lower through 16-bit integer ops (core
        // shaderInt16 below), and half math proper declares Float16. 64-bit atomic adds: the
        // Cost view sums the frame's raygen nanoseconds (raygen.slang).
        let mut features_12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            // The shaders' bindless `textures[]` (mapped onto the descriptor heap).
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .host_query_reset(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .shader_buffer_int64_atomics(true)
            // 64-bit wave reductions (the Cost view's per-wave total, raygen.slang).
            .shader_subgroup_extended_types(true)
            .timeline_semaphore(true);

        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true)
            .maintenance4(true);

        let mut features_14 = vk::PhysicalDeviceVulkan14Features::default()
            .maintenance5(true)
            .maintenance6(true)
            .push_descriptor(true);

        let mut features_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut features_raytracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let mut features_ray_query =
            vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);

        let mut features_position_fetch =
            vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
                .ray_tracing_position_fetch(true);

        let mut features_reorder =
            vk::PhysicalDeviceRayTracingInvocationReorderFeaturesNV::default()
                .ray_tracing_invocation_reorder(true);

        let mut features_clock =
            vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_device_clock(true);

        let mut features_micromap =
            vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true);

        // Aftermath diagnostics: the feature bit plus what to track (the full set from
        // NVIDIA's VkHelloNsightAftermath sample; `dev`-only overhead).
        let mut features_diagnostics =
            vk::PhysicalDeviceDiagnosticsConfigFeaturesNV::default().diagnostics_config(true);
        let mut diagnostics_info = vk::DeviceDiagnosticsConfigCreateInfoNV::default().flags(
            vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_DEBUG_INFO
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_RESOURCE_TRACKING
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_AUTOMATIC_CHECKPOINTS
                | vk::DeviceDiagnosticsConfigFlagsNV::ENABLE_SHADER_ERROR_REPORTING,
        );

        // 64-bit integers: Slang kernels carry buffer addresses as `uint64_t`. 64-bit floats:
        // planet-scale procedural kernels evaluate directions in f64.
        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_float64(true)
            .shader_int16(true)
            // The post-process pass writes the exposure meter from what it displays.
            .fragment_stores_and_atomics(true);

        let device_info = vk::DeviceCreateInfo::default()
            .enabled_features(&core_features)
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push(&mut features_12)
            .push(&mut features_13)
            .push(&mut features_14)
            .push(&mut features_acceleration_structure)
            .push(&mut features_raytracing_pipeline)
            .push(&mut features_ray_query)
            .push(&mut features_position_fetch)
            .push(&mut features_reorder)
            .push(&mut features_clock);
        let mut features_heap =
            vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
        let device_info = device_info.push(&mut features_heap);
        let device_info = if micromaps {
            device_info.push(&mut features_micromap)
        } else {
            device_info
        };
        let mut features_executable =
            vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
                .pipeline_executable_info(true);
        let device_info = if executable_stats {
            device_info.push(&mut features_executable)
        } else {
            device_info
        };
        let mut features_present_id =
            vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        let mut features_internal_sync =
            vk::PhysicalDeviceInternallySynchronizedQueuesFeaturesKHR::default()
                .internally_synchronized_queues(true);
        let device_info = if internal_sync {
            device_info.push(&mut features_internal_sync)
        } else {
            device_info
        };
        let device_info = if low_latency {
            device_info.push(&mut features_present_id)
        } else {
            device_info
        };
        let device_info = if diagnostics {
            device_info
                .push(&mut features_diagnostics)
                .push(&mut diagnostics_info)
        } else {
            device_info
        };

        // XR: created through the runtime (xrCreateVulkanDeviceKHR) so it can inject its
        // interop device extensions; identical device otherwise.
        let device = match xr {
            Some(xr) => xr.create_vk_device(entry, instance, physical_device, &device_info),
            None => instance
                .create_device(physical_device, &device_info, None)
                .unwrap(),
        };
        let queue = device.get_device_queue2(
            &vk::DeviceQueueInfo2::default()
                .flags(queue_flags)
                .queue_family_index(queue_family_idx)
                .queue_index(0),
        );

        let compute_queue = compute_family.map(|family| {
            Mutex::new(
                device.get_device_queue2(
                    &vk::DeviceQueueInfo2::default()
                        .flags(queue_flags)
                        .queue_family_index(family)
                        .queue_index(0),
                ),
            )
        });

        (
            device,
            Mutex::new(queue),
            compute_queue,
            micromaps,
            low_latency,
        )
    }
}

fn create_command_pool(device: &ash::Device, queue_family_idx: u32) -> vk::CommandPool {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_idx)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe { device.create_command_pool(&pool_info, None).unwrap() }
}

fn create_command_buffers(device: &ash::Device, pool: vk::CommandPool) -> [vk::CommandBuffer; 2] {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(2);
    unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .unwrap()
            .as_slice()
            .try_into()
            .unwrap()
    }
}

#[derive(Debug)]
pub enum VkDestroyCmd {
    ImageView(vk::ImageView),
    Image(vk::Image),
    Buffer(vk::Buffer),
    Swapchain(vk::SwapchainKHR),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    AccelerationStructure(vk::AccelerationStructureKHR),
    Micromap(vk::MicromapEXT),
    Tick,
}

pub struct VkDestroyer {
    sender: Option<Sender<VkDestroyCmd>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl VkDestroyer {
    pub fn destroy_image_view(&self, view: vk::ImageView) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::ImageView(view))
            .unwrap();
    }

    pub fn destroy_image(&self, image: vk::Image) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Image(image))
            .unwrap();
    }

    pub fn destroy_buffer(&self, buffer: vk::Buffer) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Buffer(buffer))
            .unwrap();
    }

    pub fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Swapchain(swapchain))
            .unwrap();
    }

    pub fn destroy_pipeline(&self, pipeline: vk::Pipeline) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Pipeline(pipeline))
            .unwrap();
    }

    pub fn destroy_pipeline_layout(&self, layout: vk::PipelineLayout) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::PipelineLayout(layout))
            .unwrap();
    }

    pub fn destroy_acceleration_structure(
        &self,
        acceleration_structure: vk::AccelerationStructureKHR,
    ) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::AccelerationStructure(acceleration_structure))
            .unwrap();
    }

    pub fn destroy_micromap(&self, micromap: vk::MicromapEXT) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Micromap(micromap))
            .unwrap();
    }

    pub fn tick(&self) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::Tick)
            .unwrap();
    }
}

impl Drop for VkDestroyer {
    fn drop(&mut self) {
        log::debug!("Dropping connection to destroy thread");
        let sender = self.sender.take().unwrap();
        drop(sender);
        self.thread.take().unwrap().join().unwrap();
    }
}

fn spawn_destroy_thread(
    instance: ash::Instance,
    device: ash::Device,
    state: Arc<Mutex<ManuallyDrop<AllocatorState>>>,
) -> ManuallyDrop<VkDestroyer> {
    let ext_swapchain = swapchain::Device::load(&instance, &device);
    let ext_acc_struct = acceleration_structure::Device::load(&instance, &device);
    // Only ever called for micromaps created through the extension (present in that case).
    let ext_micromap = opacity_micromap::Device::load(&instance, &device);
    let (sender, receiver) = crossbeam::channel::unbounded();
    let thread = std::thread::spawn(move || {
        // Assuming 3 frames in flight
        let mut queue = VecDeque::from(vec![Vec::new(), Vec::new()]);
        while let Ok(cmd) = receiver.recv() {
            match cmd {
                VkDestroyCmd::Tick => {
                    queue.push_front(Vec::new());
                    let death_list = queue.pop_back().unwrap();
                    for event in death_list {
                        log::trace!("Executing destroy {:?}", event);
                        match event {
                            VkDestroyCmd::ImageView(view) => unsafe {
                                device.destroy_image_view(view, None);
                            },
                            VkDestroyCmd::Image(image) => unsafe {
                                let mut state = state
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                state.free_image_allocation(image);
                                device.destroy_image(image, None);
                            },
                            VkDestroyCmd::Buffer(buffer) => unsafe {
                                let mut state = state
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                state.free_buffer_allocation(buffer);
                                device.destroy_buffer(buffer, None);
                            },
                            VkDestroyCmd::Swapchain(swapchain) => unsafe {
                                ext_swapchain.destroy_swapchain(swapchain, None);
                            },
                            VkDestroyCmd::Pipeline(pipeline) => unsafe {
                                device.destroy_pipeline(pipeline, None);
                            },
                            VkDestroyCmd::PipelineLayout(layout) => unsafe {
                                device.destroy_pipeline_layout(layout, None);
                            },
                            VkDestroyCmd::AccelerationStructure(acceleration_structure) => unsafe {
                                ext_acc_struct
                                    .destroy_acceleration_structure(acceleration_structure, None);
                            },
                            VkDestroyCmd::Micromap(micromap) => unsafe {
                                (ext_micromap.fp().destroy_micromap_ext)(
                                    ext_micromap.device(),
                                    micromap,
                                    std::ptr::null(),
                                );
                            },
                            VkDestroyCmd::Tick => panic!("Tick event in death list"),
                        }
                    }
                }
                destroy_event => {
                    queue[0].push(destroy_event);
                }
            }
        }
        log::debug!("Destroy thread finished");
    });

    ManuallyDrop::new(VkDestroyer {
        sender: Some(sender),
        thread: Some(thread),
    })
}

fn pipeline_cache_path() -> Option<std::path::PathBuf> {
    crate::shader::cache_dir().map(|dir| dir.join("pipelines.bin"))
}

/// The pipeline cache, seeded from the last run's; the driver checks the blob's header (vendor,
/// device, driver UUID) and starts empty when it doesn't match.
fn create_pipeline_cache(device: &ash::Device) -> vk::PipelineCache {
    let data = pipeline_cache_path()
        .and_then(|path| std::fs::read(path).ok())
        .unwrap_or_default();
    let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
    unsafe { device.create_pipeline_cache(&info, None) }
        .or_else(|_| unsafe {
            device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
        })
        .expect("create pipeline cache")
}
