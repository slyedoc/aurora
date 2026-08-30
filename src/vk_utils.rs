use ash::vk;

use crate::render_device::RenderDevice;

pub fn aligned_size(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

pub fn transition_image_layout(
    device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    image: vk::Image,
    from: vk::ImageLayout,
    to: vk::ImageLayout,
) {
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
    // Stage / access masks per transition. Without them (sync2 NONE on both sides) the layout
    // change has no execution dependency at all: the swapchain transition could run before the
    // presentation engine released the image, and the present transition before the colour
    // writes landed. Sync validation flagged both (WRITE_AFTER_READ against
    // vkAcquireNextImageKHR, WRITE_AFTER_WRITE against the attachment clear).
    let (src_stage, src_access, dst_stage, dst_access) = match (from, to) {
        // Acquired swapchain image: the submit waits on the acquire semaphore at
        // COLOR_ATTACHMENT_OUTPUT, so the transition sits behind that stage and ahead of the
        // attachment clear.
        (L::UNDEFINED, L::ATTACHMENT_OPTIMAL) => (
            S::COLOR_ATTACHMENT_OUTPUT,
            A::NONE,
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
        ),
        // Rendering finished -> present: order after the colour writes; nothing on the GPU
        // reads it afterwards (the presentation engine waits on the semaphore).
        (L::ATTACHMENT_OPTIMAL, L::PRESENT_SRC_KHR) => (
            S::COLOR_ATTACHMENT_OUTPUT,
            A::COLOR_ATTACHMENT_WRITE,
            S::NONE,
            A::NONE,
        ),
        // Fresh storage image (render target) ahead of the trace / blit.
        (L::UNDEFINED, L::GENERAL) => (
            S::NONE,
            A::NONE,
            S::ALL_COMMANDS,
            A::SHADER_STORAGE_WRITE | A::SHADER_STORAGE_READ | A::SHADER_SAMPLED_READ,
        ),
        _ => (
            S::ALL_COMMANDS,
            A::MEMORY_WRITE | A::MEMORY_READ,
            S::ALL_COMMANDS,
            A::MEMORY_WRITE | A::MEMORY_READ,
        ),
    };
    let image_barrier = crate::vk_init::layout_transition2(image, from, to)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access);
    let barrier_info =
        vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&image_barrier));
    unsafe {
        device
            .ext_sync2
            .cmd_pipeline_barrier2(cmd_buffer, &barrier_info);
    }
}

pub fn get_raytracing_properties(
    device: &RenderDevice,
) -> vk::PhysicalDeviceRayTracingPipelinePropertiesKHR {
    let mut raytracing_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut properties2 =
        vk::PhysicalDeviceProperties2KHR::default().push_next(&mut raytracing_properties);
    unsafe {
        device
            .instance
            .get_physical_device_properties2(device.physical_device, &mut properties2)
    }
    raytracing_properties
}

pub fn get_acceleration_structure_properties(
    device: &RenderDevice,
) -> vk::PhysicalDeviceAccelerationStructurePropertiesKHR {
    let mut acceleration_structure_properties =
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut properties2 = vk::PhysicalDeviceProperties2KHR::default()
        .push_next(&mut acceleration_structure_properties);
    unsafe {
        device
            .instance
            .get_physical_device_properties2(device.physical_device, &mut properties2)
    }
    acceleration_structure_properties
}
