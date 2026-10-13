//! The frame's point on the GPU timeline and the frame counter. A frame records first, then waits
//! for the previous one (render_frame), and submits on the timeline whether or not there is a
//! window to present to.

use ash::vk;
use bevy::prelude::*;

use crate::render_device::RenderDevice;
use crate::timeline::SubmitPoint;

#[derive(Resource)]
pub struct FrameSync {
    device: RenderDevice,
    /// The last frame submit's point on the timeline (0 before the first).
    pub last_frame: SubmitPoint,
    /// Frames submitted so far; its parity picks the per-frame command buffers and heap blocks.
    pub frame_count: usize,
}

impl FrameSync {
    pub fn new(device: RenderDevice) -> Self {
        Self {
            device,
            last_frame: SubmitPoint::default(),
            frame_count: 0,
        }
    }

    /// Waits for the previous frame: before anything it produced is read back, and before the
    /// destroyer frees what it used.
    pub unsafe fn wait(&self) {
        let _wait = info_span!("frame_fence_wait").entered();
        self.device
            .timeline
            .wait(&self.device, self.last_frame)
            .unwrap_or_else(|e| {
                // Device loss surfaces here first: let the driver finish its crash dump.
                crate::aftermath::note_device_lost(e);
                panic!("frame wait failed: {e:?}");
            });
    }

    /// Submit the frame's command buffers, in order, on the timeline, after `points` (earlier submits this
    /// frame depends on) and `waits`, signalling `signals`. The caller holds the queue.
    pub unsafe fn submit(
        &mut self,
        queue: vk::Queue,
        cmd_buffers: &[vk::CommandBuffer],
        points: &[SubmitPoint],
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signals: &[vk::Semaphore],
        present_id: Option<u64>,
    ) {
        let _submit = info_span!("queue_submit").entered();
        let point = unsafe {
            self.device.timeline.submit(
                &self.device,
                queue,
                cmd_buffers,
                points,
                waits,
                signals,
                vk::Fence::null(),
                present_id,
            )
        }
        .unwrap_or_else(|e| {
            crate::aftermath::note_device_lost(e);
            panic!("frame submit failed: {e:?}");
        });
        self.last_frame = point;
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        let queue = self.device.queue.lock().unwrap();
        unsafe { self.device.queue_wait_idle(*queue).unwrap() };
    }
}
