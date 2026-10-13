//! Reflex (`VK_NV_low_latency2`): the driver paces when the CPU starts a frame so the GPU never
//! runs far behind input, and markers along the frame (simulation, render submit, present,
//! tagged with a present id) let it measure and report the latency. `First` sleeps until the
//! driver says go; the render path marks submit and present. Windowed flat only (render
//! device: `ext_low_latency`); XR is paced by the OpenXR runtime.

use ash::vk;
use ash::vk::TaggedStructure as _;
use bevy::prelude::*;

use crate::ray_render_plugin::{TeardownSchedule, on_shutdown};
use crate::render_device::RenderDevice;
use crate::render_settings::RenderSettings;
use crate::swapchain::Swapchain;

/// `RenderSettings::low_latency`.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Default, Clone, PartialEq)]
pub enum LowLatency {
    Off,
    /// Reflex: the CPU waits at the start of a frame instead of queueing ahead of the GPU.
    #[default]
    On,
    /// Also keeps the GPU clocked up when the frame is CPU-bound.
    Boost,
}

/// Frames between reads of the driver's latency reports.
const REPORT_EVERY: u64 = 30;

#[derive(Resource, Default)]
pub struct Latency {
    /// Timeline semaphore the driver signals when the sleep ends.
    semaphore: vk::Semaphore,
    sleep_value: u64,
    /// The present id of the frame being simulated and rendered (0 = untagged).
    present_id: u64,
    /// The swapchain generation and mode the sleep mode was last set for.
    applied: Option<(u64, LowLatency)>,
    /// Smoothed milliseconds from simulation start to the end of present (driver timings).
    pub latency_ms: f64,
}

impl Latency {
    /// The id this frame's submit and present carry, when Reflex is on.
    pub fn present_id(&self) -> Option<u64> {
        (self.present_id != 0).then_some(self.present_id)
    }

    /// Marks `marker` for this frame on `swapchain`.
    pub fn mark(&self, rd: &RenderDevice, swapchain: &Swapchain, marker: vk::LatencyMarkerNV) {
        let (Some(ext), Some(id)) = (rd.ext_low_latency.as_ref(), self.present_id()) else {
            return;
        };
        let info = vk::SetLatencyMarkerInfoNV::default()
            .present_id(id)
            .marker(marker);
        unsafe { ext.set_latency_marker(swapchain.swapchain, &info) };
    }
}

/// `First`: applies the mode, sleeps until the driver releases the frame, starts a new present
/// id and marks the simulation start.
fn sleep_and_start(
    rd: Option<Res<RenderDevice>>,
    swapchain: Option<Res<Swapchain>>,
    settings: Res<RenderSettings>,
    mut latency: ResMut<Latency>,
) {
    let (Some(rd), Some(swapchain)) = (rd, swapchain) else {
        return;
    };
    let Some(ext) = rd.ext_low_latency.as_ref() else {
        return;
    };
    if swapchain.swapchain == vk::SwapchainKHR::null() {
        return;
    }
    if latency.semaphore == vk::Semaphore::null() {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push(&mut kind);
        latency.semaphore = unsafe { rd.device.create_semaphore(&info, None) }.unwrap();
    }
    let mode = settings.low_latency;
    if latency.applied != Some((swapchain.generation, mode)) {
        let info = vk::LatencySleepModeInfoNV::default()
            .low_latency_mode(mode != LowLatency::Off)
            .low_latency_boost(mode == LowLatency::Boost)
            .minimum_interval_us(0);
        if let Err(e) = unsafe { ext.set_latency_sleep_mode(swapchain.swapchain, Some(&info)) } {
            warn!("reflex: setting the sleep mode failed: {e:?}");
        }
        latency.applied = Some((swapchain.generation, mode));
    }
    if mode == LowLatency::Off {
        latency.present_id = 0;
        return;
    }
    let _sleep = info_span!("reflex_sleep").entered();
    latency.sleep_value += 1;
    let sleep = vk::LatencySleepInfoNV::default()
        .signal_semaphore(latency.semaphore)
        .value(latency.sleep_value);
    if unsafe { ext.latency_sleep(swapchain.swapchain, &sleep) }.is_ok() {
        let semaphores = [latency.semaphore];
        let values = [latency.sleep_value];
        let wait = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // Bounded: a driver that never signals must not hang the app.
        let _ = unsafe { rd.device.wait_semaphores(&wait, 100_000_000) };
    }
    latency.present_id += 1;
    latency.mark(&rd, &swapchain, vk::LatencyMarkerNV::SIMULATION_START);
    if latency.present_id % REPORT_EVERY == 0 {
        read_report(&rd, ext, &swapchain, &mut latency);
    }
}

/// `Last`, before the render sets: the simulation is done.
fn simulation_end(
    rd: Option<Res<RenderDevice>>,
    swapchain: Option<Res<Swapchain>>,
    latency: Res<Latency>,
) {
    if let (Some(rd), Some(swapchain)) = (rd, swapchain) {
        latency.mark(&rd, &swapchain, vk::LatencyMarkerNV::SIMULATION_END);
    }
}

/// The newest complete driver report: simulation start to present end, smoothed.
fn read_report(
    rd: &RenderDevice,
    ext: &ash::nv::low_latency2::Device,
    swapchain: &Swapchain,
    latency: &mut Latency,
) {
    let _ = rd;
    let mut reports = [vk::LatencyTimingsFrameReportNV::default(); 64];
    let mut info = vk::GetLatencyMarkerInfoNV::default().timings(&mut reports);
    unsafe { ext.get_latency_timings(swapchain.swapchain, &mut info) };
    let count = info.timing_count as usize;
    let newest = reports[..count.min(reports.len())]
        .iter()
        .filter(|r| r.sim_start_time_us != 0 && r.present_end_time_us > r.sim_start_time_us)
        .max_by_key(|r| r.present_id);
    if let Some(r) = newest {
        let ms = (r.present_end_time_us - r.sim_start_time_us) as f64 / 1.0e3;
        latency.latency_ms = if latency.latency_ms > 0.0 {
            latency.latency_ms + (ms - latency.latency_ms) * 0.2
        } else {
            ms
        };
        if latency.present_id % (REPORT_EVERY * 20) == 0 {
            debug!("reflex: sim start -> present end {:.1} ms", latency.latency_ms);
        }
    }
}

fn cleanup(rd: Option<Res<RenderDevice>>, mut latency: ResMut<Latency>) {
    if let Some(rd) = rd
        && latency.semaphore != vk::Semaphore::null()
    {
        unsafe { rd.device.destroy_semaphore(latency.semaphore, None) };
        latency.semaphore = vk::Semaphore::null();
    }
}

pub struct LatencyPlugin;

impl Plugin for LatencyPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Latency>()
            .add_systems(First, sleep_and_start)
            .add_systems(
                Last,
                simulation_end.before(crate::ray_render_plugin::RenderSet::Extract),
            )
            .add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
