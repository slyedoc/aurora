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
        unsafe { self.wait_for(self.last_frame) }
    }

    /// Waits for the frame submitted at `point` (the previous one, once this frame's submit has
    /// gone ahead of the wait).
    pub unsafe fn wait_for(&self, point: SubmitPoint) {
        let _wait = info_span!("frame_fence_wait").entered();
        self.device
            .timeline
            .wait(&self.device, point)
            .unwrap_or_else(|e| {
                // Device loss surfaces here first: let the driver finish its crash dump.
                crate::aftermath::note_device_lost(e);
                panic!("frame wait failed: {e:?}");
            });
    }

    /// Submit the frame's command buffers, in order, on the timeline, after `points` (earlier submits this
    /// frame depends on) and `waits`, signalling `signals`. The caller holds the queue.
    /// `timeline_waits`: the async compute queue's points this frame waits on (timeline.rs
    /// `submit_with`); the frame's point then covers that work too.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &mut self,
        queue: vk::Queue,
        cmd_buffers: &[vk::CommandBuffer],
        points: &[SubmitPoint],
        timeline_waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        signals: &[vk::Semaphore],
        present_id: Option<u64>,
    ) {
        let _submit = info_span!("queue_submit").entered();
        let point = unsafe {
            self.device.timeline.submit_with(
                &self.device,
                queue,
                cmd_buffers,
                points,
                timeline_waits,
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
        self.device.retire_released(point);
        self.last_frame = point;
    }
}

impl Drop for FrameSync {
    fn drop(&mut self) {
        let queue = self.device.queue.lock().unwrap();
        unsafe { self.device.queue_wait_idle(*queue).unwrap() };
    }
}
