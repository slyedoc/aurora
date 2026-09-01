use std::{
    collections::{HashMap, VecDeque},
    ffi::{CStr, c_char},
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

use ash::vk;
use ash::{
    ext::{descriptor_indexing, opacity_micromap},
    khr::{
        acceleration_structure, deferred_host_operations, maintenance4, ray_tracing_pipeline,
        spirv_1_4, surface, swapchain, synchronization2,
    },
};
use bevy::prelude::*;
use crossbeam::channel::Sender;
use gpu_allocator::{AllocationError, MemoryLocation, vulkan::*};
use raw_window_handle::DisplayHandle;

use crate::render_texture::RenderTexture;
use crate::xr::XrContext;

const MAX_BINDLESS_IMAGES: u32 = 16536;

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

pub struct RenderDeviceData {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub ext_surface: surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: Mutex<vk::Queue>,
    pub queue_family_idx: u32,
    pub ext_swapchain: swapchain::Device,
    pub ext_sync2: synchronization2::Device,
    pub ext_rtx_pipeline: ray_tracing_pipeline::Device,
    pub ext_acc_struct: acceleration_structure::Device,
    /// `VK_EXT_opacity_micromap`, when the device has it (omm.rs); `None` otherwise.
    pub ext_micromap: Option<opacity_micromap::Device>,
    pub command_pool: vk::CommandPool,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub bindless_descriptor_set_layout: vk::DescriptorSetLayout,
    pub bindless_descriptor_map: Mutex<BindlessMap>,
    pub transfer_command_pool: Mutex<vk::CommandPool>,
    pub command_buffers: [vk::CommandBuffer; 2],
    pub descriptor_pool: Mutex<vk::DescriptorPool>,
    pub linear_sampler: vk::Sampler,
    pub destroyer: ManuallyDrop<VkDestroyer>,
    pub allocator_state: Arc<Mutex<ManuallyDrop<AllocatorState>>>,
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
    pub next: u32,
}

impl Clone for RenderDevice {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl RenderDevice {
    pub unsafe fn from_display(display_handle: &DisplayHandle, xr: Option<&XrContext>) -> Self {
        // Before the loader is touched: Aftermath registers ahead of instance/device creation.
        crate::aftermath::enable();
        unsafe {
            let entry = ash::Entry::linked();
            let instance = create_instance(display_handle, &entry, xr);
            let ext_surface = surface::Instance::new(&entry, &instance);
            let (physical_device, queue_family_idx) = pick_physical_device(&instance, xr);
            let (device, queue, micromaps) =
                create_logical_device(&entry, &instance, physical_device, queue_family_idx, xr);
            let ext_swapchain = swapchain::Device::new(&instance, &device);
            let ext_sync2 = synchronization2::Device::new(&instance, &device);
            let ext_rtx_pipeline = ray_tracing_pipeline::Device::new(&instance, &device);
            let ext_acc_struct = acceleration_structure::Device::new(&instance, &device);
            let ext_micromap = micromaps.then(|| opacity_micromap::Device::new(&instance, &device));
            let command_pool = create_command_pool(&device, queue_family_idx);
            let transfer_command_pool = Mutex::new(create_command_pool(&device, queue_family_idx));
            let command_buffers = create_command_buffers(&device, command_pool);
            let descriptor_pool = create_descriptor_pool(&device);
            let (bindless_descriptor_set, bindless_descriptor_set_layout) =
                create_global_descriptor(device.clone(), *descriptor_pool.lock().unwrap());
            let linear_sampler = create_linear_sampler(device.clone());

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

            let ret = RenderDevice(Arc::new(RenderDeviceData {
                entry,
                instance,
                ext_surface,
                physical_device,
                device,
                queue,
                queue_family_idx,
                ext_swapchain,
                ext_sync2,
                ext_rtx_pipeline,
                ext_acc_struct,
                ext_micromap,
                command_pool,
                bindless_descriptor_set,
                bindless_descriptor_set_layout,
                bindless_descriptor_map: Mutex::new(BindlessMap::default()),
                transfer_command_pool,
                command_buffers,
                descriptor_pool,
                linear_sampler,
                destroyer,
                allocator_state,
            }));

            ret
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

        let index = map.free.pop().unwrap_or_else(|| {
            let index = map.next;
            map.next += 1;
            index
        });
        map.by_view.insert(texture.image_view, index);

        let descriptor_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.image_view)
            .sampler(self.linear_sampler);

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor_set)
            .dst_binding(200)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&descriptor_info));

        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
        }

        index
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
            map.free.push(index);
        }
    }

    pub fn load_shader(
        &self,
        spirv: &[u8],
        stage: vk::ShaderStageFlags,
    ) -> vk::PipelineShaderStageCreateInfo {
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
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
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
        log::info!("Dropping RenderDevice");
        unsafe {
            let destroyer = ManuallyDrop::take(&mut self.destroyer);
            drop(destroyer);

            let mut alloc_state = self.allocator_state.lock().unwrap();
            let alloc_state = ManuallyDrop::take(&mut *alloc_state);

            drop(alloc_state);

            self.destroy_descriptor_set_layout(self.bindless_descriptor_set_layout, None);

            self.destroy_sampler(self.linear_sampler, None);
            {
                let transfer_command_pool = self.transfer_command_pool.lock().unwrap();
                self.destroy_command_pool(*transfer_command_pool, None);
            }
            {
                let descriptor_pool = self.descriptor_pool.lock().unwrap();
                self.destroy_descriptor_pool(*descriptor_pool, None);
            }
            self.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

unsafe fn create_instance(
    display_handle: &DisplayHandle,
    entry: &ash::Entry,
    xr: Option<&XrContext>,
) -> ash::Instance {
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

        println!("Validation layers:");
        for layer_name in layer_names.iter() {
            println!("  - {}", layer_name.to_str().unwrap());
        }

        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();
        let mut instance_extensions: Vec<*const c_char> =
            ash_window::enumerate_required_extensions(display_handle.as_raw())
                .unwrap()
                .to_vec();
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

        println!("Instance extensions:");
        for extension_name in instance_extensions.iter() {
            println!("  - {}", CStr::from_ptr(*extension_name).to_str().unwrap());
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let sync_validation = cfg!(feature = "dev");
        if sync_validation {
            instance_extensions.push(ash::ext::validation_features::NAME.as_ptr());
        }
        let enabled = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&instance_extensions);
        if sync_validation {
            instance_info = instance_info.push_next(&mut validation_features);
        }

        // XR: the runtime performs the create (xrCreateVulkanInstanceKHR), adding whatever
        // instance extensions its compositor interop needs on top of ours.
        match xr {
            Some(xr) => xr.create_vk_instance(entry, &instance_info),
            None => entry.create_instance(&instance_info, None).unwrap(),
        }
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
            println!(
                "Running on device (OpenXR): {}",
                CStr::from_ptr(info.device_name.as_ptr()).to_str().unwrap()
            );
            return (physical_device, queue_family_idx);
        }
        let all_devices = instance.enumerate_physical_devices().unwrap();
        println!("Available devices:");
        for device in all_devices.iter() {
            let info = instance.get_physical_device_properties(*device);
            println!(
                "  - {}",
                CStr::from_ptr(info.device_name.as_ptr()).to_str().unwrap()
            );
        }

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

                let properties = instance.get_physical_device_queue_family_properties(d);
                properties.iter().enumerate().find_map(|(i, p)| {
                    if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some((d, i as u32))
                    } else {
                        None
                    }
                })
            })
            .expect("Not a single device found!");

        let device_properties = instance.get_physical_device_properties(physical_device);
        println!(
            "Running on device: {}",
            CStr::from_ptr(device_properties.device_name.as_ptr())
                .to_str()
                .unwrap()
        );
        (physical_device, queue_family_idx)
    }
}

unsafe fn create_logical_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_idx: u32,
    xr: Option<&XrContext>,
) -> (ash::Device, Mutex<vk::Queue>, bool) {
    unsafe {
        let mut device_extensions = vec![
            swapchain::NAME.as_ptr(),
            synchronization2::NAME.as_ptr(),
            maintenance4::NAME.as_ptr(),
            acceleration_structure::NAME.as_ptr(),
            ray_tracing_pipeline::NAME.as_ptr(),
            deferred_host_operations::NAME.as_ptr(),
            spirv_1_4::NAME.as_ptr(),
            descriptor_indexing::NAME.as_ptr(),
        ];
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
                println!(
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
            println!(
                "device lacks VK_EXT_opacity_micromap -- alpha cutouts stay on the any-hit path"
            );
        }

        println!("Device extensions:");
        for extension_name in device_extensions.iter() {
            println!("  - {}", CStr::from_ptr(*extension_name).to_str().unwrap());
        }

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_idx)
            .queue_priorities(&[1.0]);

        let mut sync2_info =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        let mut dynamic_rendering_info =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut maintaince4_info =
            vk::PhysicalDeviceMaintenance4Features::default().maintenance4(true);

        let mut bda_info =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);

        let mut features_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_variable_descriptor_count(true);

        let mut features_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);

        let mut features_raytracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let mut features_scalar_block =
            vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);

        // 16-bit arithmetic in shaders: Slang's f16tof32 / f32tof16 lower through 16-bit
        // integer ops (SPIR-V Int16, core shaderInt16 below), and half math proper declares
        // Float16. Both are supported on every RTX-class device.
        let mut features_float16 =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);

        let mut features_micromap =
            vk::PhysicalDeviceOpacityMicromapFeaturesEXT::default().micromap(true);

        // 64-bit integers: Slang kernels carry buffer addresses as `uint64_t`.
        let core_features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_int16(true);

        let device_info = vk::DeviceCreateInfo::default()
            .enabled_features(&core_features)
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extensions)
            .push_next(&mut sync2_info)
            .push_next(&mut dynamic_rendering_info)
            .push_next(&mut maintaince4_info)
            .push_next(&mut bda_info)
            .push_next(&mut features_indexing)
            .push_next(&mut features_acceleration_structure)
            .push_next(&mut features_raytracing_pipeline)
            .push_next(&mut features_scalar_block)
            .push_next(&mut features_float16);
        let device_info = if micromaps {
            device_info.push_next(&mut features_micromap)
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
        let queue = device.get_device_queue(queue_family_idx, 0);

        (device, Mutex::new(queue), micromaps)
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

fn create_descriptor_pool(device: &ash::Device) -> Mutex<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1000,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: MAX_BINDLESS_IMAGES,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 256,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 16,
        },
    ];

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
        .pool_sizes(&pool_sizes)
        .max_sets(1000);

    Mutex::new(unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap()
    })
}

fn create_global_descriptor(
    device: ash::Device,
    descriptor_pool: vk::DescriptorPool,
) -> (vk::DescriptorSet, vk::DescriptorSetLayout) {
    const MAX_BINDLESS_IMAGES: u32 = 16536;
    let image_binding = vk::DescriptorSetLayoutBinding::default()
        .binding(200)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(MAX_BINDLESS_IMAGES)
        .stage_flags(vk::ShaderStageFlags::ALL);

    let bindless_flags = vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
        | vk::DescriptorBindingFlags::PARTIALLY_BOUND
        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
    let max_binding = MAX_BINDLESS_IMAGES - 1;

    let mut layout_info_ext = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
        .binding_flags(std::slice::from_ref(&bindless_flags));

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(std::slice::from_ref(&image_binding))
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .push_next(&mut layout_info_ext);

    let descriptor_set_layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    };

    let mut alloc_info_ext = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
        .descriptor_counts(std::slice::from_ref(&max_binding));

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_next(&mut alloc_info_ext);

    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .unwrap()
            .pop()
            .unwrap()
    };

    return (descriptor_set, descriptor_set_layout);
}

fn create_linear_sampler(device: ash::Device) -> vk::Sampler {
    let linear_sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR);
    unsafe { device.create_sampler(&linear_sampler_info, None).unwrap() }
}

#[derive(Debug)]
pub enum VkDestroyCmd {
    ImageView(vk::ImageView),
    Image(vk::Image),
    Buffer(vk::Buffer),
    Swapchain(vk::SwapchainKHR),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    DescriptorSetLayout(vk::DescriptorSetLayout),
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

    pub fn destroy_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout) {
        self.sender
            .as_ref()
            .unwrap()
            .send(VkDestroyCmd::DescriptorSetLayout(layout))
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
        log::info!("Dropping connection to destroy thread");
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
    let ext_swapchain = swapchain::Device::new(&instance, &device);
    let ext_acc_struct = acceleration_structure::Device::new(&instance, &device);
    // Only ever called for micromaps created through the extension (present in that case).
    let ext_micromap = opacity_micromap::Device::new(&instance, &device);
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
                                let mut state = state.lock().unwrap();
                                state.free_image_allocation(image);
                                device.destroy_image(image, None);
                            },
                            VkDestroyCmd::Buffer(buffer) => unsafe {
                                let mut state = state.lock().unwrap();
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
                            VkDestroyCmd::DescriptorSetLayout(layout) => unsafe {
                                device.destroy_descriptor_set_layout(layout, None);
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
        log::info!("Destroy thread finished");
    });

    ManuallyDrop::new(VkDestroyer {
        sender: Some(sender),
        thread: Some(thread),
    })
}
