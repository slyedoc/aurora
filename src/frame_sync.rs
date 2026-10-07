//! The frame's fence and counter. Every frame waits for the previous one before rewriting its
//! resources (TLAS, instance buffers, the per-frame command buffer) and submits under the
//! fence again, whether or not there is a window to present to.

use ash::vk;
use bevy::prelude::*;

use crate::render_device::RenderDevice;

pub const FRAMES_IN_FLIGHT: usize = 1;

#[derive(Resource)]
pub struct FrameSync {
    device: RenderDevice,
    in_flight_fences: [vk::Fence; FRAMES_IN_FLIGHT],
    /// Frames submitted so far; its parity picks the per-frame command buffer and descriptor sets.
    pub frame_count: usize,
}

impl FrameSync {
    pub fn new(device: RenderDevice) -> Self {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight_fences =
            std::array::from_fn(|_| unsafe { device.create_fence(&fence_info, None).unwrap() });
        Self {
            device,
            in_flight_fences,
            frame_count: 0,
        }
    }

    fn fence(&self) -> vk::Fence {
        self.in_flight_fences[self.frame_count % FRAMES_IN_FLIGHT]
    }

    /// Wait for the previous frame and rearm its fence.
    pub unsafe fn wait(&mut self) {
        unsafe {
            let _wait = info_span!("frame_fence_wait").entered();
            self.device
                .wait_for_fences(std::slice::from_ref(&self.fence()), true, u64::MAX)
                .unwrap_or_else(|e| {
                    // Device loss surfaces here first: let the driver finish its crash dump.
                    crate::aftermath::note_device_lost(e);
                    panic!("frame fence wait failed: {e:?}");
                });
            self.device
                .reset_fences(std::slice::from_ref(&self.fence()))
                .unwrap();
        }
    }

    /// Submit the frame's command buffer under the fence, waiting on `waits` and signalling
    /// `signals`. The caller holds the queue.
    pub unsafe fn submit(
        &self,
        queue: vk::Queue,
        cmd_buffer: vk::CommandBuffer,
        waits: &[(vk::Semaphore, vk::PipelineStageFlags)],
        signals: &[vk::Semaphore],
    ) {
        unsafe {
            let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|w| w.0).collect();
            let wait_stages: Vec<vk::PipelineStageFlags> = waits.iter().map(|w| w.1).collect();
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&cmd_buffer))
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .signal_semaphores(signals);
            let _submit = info_span!("queue_submit").entered();
            self.device
                .queue_submit(queue, std::slice::from_ref(&submit_info), self.fence())
                .unwrap_or_else(|e| {
                    crate::aftermath::note_device_lost(e);
                    panic!("frame submit failed: {e:?}");
                });
        }
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        unsafe {
            {
                let queue = self.device.queue.lock().unwrap();
                self.device.queue_wait_idle(*queue).unwrap();
            }
            for fence in self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }
        }
    }
}
