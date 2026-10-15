//! One timeline semaphore for every submit on the queue. Each submit signals the next value
//! (a [`SubmitPoint`]); "has this work finished" is a comparison against the semaphore's
//! counter, and waiting for it is one call -- no per-subsystem fences.
//!
//! Values are allocated under the queue lock, so they rise in submission order and reaching a
//! point implies every earlier submit finished too.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use ash::vk::TaggedStructure as _;

/// The value a submit signals on the timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubmitPoint(pub u64);

pub struct GpuTimeline {
    pub semaphore: vk::Semaphore,
    /// The last value handed out.
    last: AtomicU64,
}

impl GpuTimeline {
    pub fn new(device: &ash::Device) -> Self {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push(&mut kind);
        let semaphore = unsafe { device.create_semaphore(&info, None) }.unwrap();
        Self {
            semaphore,
            last: AtomicU64::new(0),
        }
    }

    /// The most recent point handed out (finished or not).
    pub fn last_submitted(&self) -> SubmitPoint {
        SubmitPoint(self.last.load(Ordering::Acquire))
    }

    /// Submits `cmds` on `queue` (the caller holds the queue lock), after `waits` -- timeline
    /// points of earlier submits, waited at `ALL_COMMANDS` -- and `binary_waits`, signalling
    /// `binary_signals` and the returned point. `fence`, when not null, is signalled too;
    /// `present_id` ties the submit to a Reflex-tagged present.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        cmds: &[vk::CommandBuffer],
        waits: &[SubmitPoint],
        binary_waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        binary_signals: &[vk::Semaphore],
        fence: vk::Fence,
        present_id: Option<u64>,
    ) -> Result<SubmitPoint, vk::Result> {
        unsafe {
            self.submit_with(
                device,
                queue,
                cmds,
                waits,
                &[],
                binary_waits,
                binary_signals,
                fence,
                present_id,
            )
        }
    }

    /// [`Self::submit`], also waiting on other timelines: `(semaphore, value, stage)` -- the
    /// other queue's points (async compute), each at the stage that first reads its results.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit_with(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        cmds: &[vk::CommandBuffer],
        waits: &[SubmitPoint],
        timeline_waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        binary_waits: &[(vk::Semaphore, vk::PipelineStageFlags2)],
        binary_signals: &[vk::Semaphore],
        fence: vk::Fence,
        present_id: Option<u64>,
    ) -> Result<SubmitPoint, vk::Result> {
        let point = SubmitPoint(self.last.fetch_add(1, Ordering::AcqRel) + 1);
        let commands: Vec<vk::CommandBufferSubmitInfo> = cmds
            .iter()
            .map(|&cmd| vk::CommandBufferSubmitInfo::default().command_buffer(cmd))
            .collect();
        let mut wait_infos: Vec<vk::SemaphoreSubmitInfo> = binary_waits
            .iter()
            .map(|&(semaphore, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .stage_mask(stage)
            })
            .collect();
        for &(semaphore, value, stage) in timeline_waits {
            if value > 0 {
                wait_infos.push(
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(semaphore)
                        .value(value)
                        .stage_mask(stage),
                );
            }
        }
        if let Some(&latest) = waits.iter().max() {
            wait_infos.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.semaphore)
                    .value(latest.0)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let mut signal_infos: Vec<vk::SemaphoreSubmitInfo> = binary_signals
            .iter()
            .map(|&semaphore| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        signal_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.semaphore)
                .value(point.0)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
        // Reflex (latency.rs): the submit belongs to this present.
        let mut latency =
            vk::LatencySubmissionPresentIdNV::default().present_id(present_id.unwrap_or(0));
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .wait_semaphore_infos(&wait_infos)
            .signal_semaphore_infos(&signal_infos);
        let submit = if present_id.is_some() {
            submit.push(&mut latency)
        } else {
            submit
        };
        unsafe { device.queue_submit2(queue, std::slice::from_ref(&submit), fence) }?;
        Ok(point)
    }

    /// The latest point the GPU has finished.
    pub fn last_completed(&self, device: &ash::Device) -> SubmitPoint {
        SubmitPoint(unsafe { device.get_semaphore_counter_value(self.semaphore) }.unwrap_or(0))
    }

    /// Whether the work up to `point` has finished.
    pub fn reached(&self, device: &ash::Device, point: SubmitPoint) -> bool {
        unsafe { device.get_semaphore_counter_value(self.semaphore) }
            .is_ok_and(|value| value >= point.0)
    }

    /// Blocks until the work up to `point` has finished.
    pub fn wait(&self, device: &ash::Device, point: SubmitPoint) -> Result<(), vk::Result> {
        if point.0 == 0 {
            return Ok(());
        }
        let semaphores = [self.semaphore];
        let values = [point.0];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        unsafe { device.wait_semaphores(&info, u64::MAX) }
    }

    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.semaphore, None) };
    }
}
