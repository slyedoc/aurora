//! Per-pass GPU time. [`GpuProfiler::begin`] / [`GpuProfiler::end`] bracket a stretch of the
//! frame's commands with a debug-utils label (Nsight's GPU Trace names the pass) and two
//! timestamps. Once the frame's fence has passed they are read back, smoothed, logged every few
//! seconds and, with `trace_tracy`, plotted in Tracy as `gpu <name> ms`.

use std::{collections::HashMap, ffi::CString};

use ash::vk;
use bevy::prelude::*;

use crate::{
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_device::RenderDevice,
};

/// Most scopes one frame records; later ones still label, but go untimed.
const MAX_SCOPES: u32 = 64;
/// Seconds between log lines.
const LOG_EVERY_SECS: f64 = 5.0;
/// Weight of the newest frame in the smoothed times.
const SMOOTHING: f64 = 0.05;

#[derive(Resource)]
pub struct GpuProfiler {
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    /// This frame's timed scopes, in begin order: scope `i` owns queries `2i` and `2i + 1`.
    recorded: Vec<&'static str>,
    labels: HashMap<&'static str, CString>,
    /// Smoothed milliseconds per scope name, in first-seen order.
    pub times: Vec<(&'static str, f64)>,
    /// The last frame read back, unsmoothed (scopes of one name summed).
    pub last_frame: Vec<(&'static str, f64)>,
    /// Frames read back so far; `last_frame` changes when this does.
    pub frames_read: u64,
    last_log: f64,
    #[cfg(feature = "trace_tracy")]
    plots: HashMap<&'static str, tracy_client::PlotName>,
}

/// An open scope, closed by [`GpuProfiler::end`].
#[must_use]
pub struct GpuScope(Option<u32>);

impl GpuProfiler {
    pub fn new(rd: &RenderDevice) -> Self {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(MAX_SCOPES * 2);
        let pool = unsafe { rd.device.create_query_pool(&info, None) }.unwrap();
        unsafe { rd.device.reset_query_pool(pool, 0, MAX_SCOPES * 2) };
        let period_ns = unsafe {
            rd.instance
                .get_physical_device_properties(rd.physical_device)
                .limits
                .timestamp_period
        } as f64;
        Self {
            pool,
            period_ns,
            recorded: Vec::new(),
            labels: HashMap::new(),
            times: Vec::new(),
            last_frame: Vec::new(),
            frames_read: 0,
            last_log: 0.0,
            #[cfg(feature = "trace_tracy")]
            plots: HashMap::new(),
        }
    }

    /// Reads the previous frame's scopes (call after its fence) and starts this frame's.
    pub fn begin_frame(&mut self, rd: &RenderDevice, now: f64) {
        let count = self.recorded.len() as u32;
        if count > 0 {
            let mut ticks = vec![0u64; (count * 2) as usize];
            let read = unsafe {
                rd.device.get_query_pool_results(
                    self.pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            unsafe { rd.device.reset_query_pool(self.pool, 0, count * 2) };
            if read.is_ok() {
                let mut frame: Vec<(&'static str, f64)> = Vec::new();
                for (i, name) in self.recorded.iter().enumerate() {
                    let ms = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * self.period_ns
                        / 1.0e6;
                    match frame.iter_mut().find(|(n, _)| n == name) {
                        Some((_, total)) => *total += ms,
                        None => frame.push((name, ms)),
                    }
                }
                self.last_frame.clone_from(&frame);
                self.frames_read += 1;
                for (name, ms) in frame {
                    match self.times.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, smooth)) => *smooth += (ms - *smooth) * SMOOTHING,
                        None => self.times.push((name, ms)),
                    }
                    #[cfg(feature = "trace_tracy")]
                    if let Some(client) = tracy_client::Client::running() {
                        let plot = *self.plots.entry(name).or_insert_with(|| {
                            tracy_client::PlotName::new_leak(format!("gpu {name} ms"))
                        });
                        client.plot(plot, ms);
                    }
                }
            }
        }
        self.recorded.clear();
        if now - self.last_log >= LOG_EVERY_SECS && !self.times.is_empty() {
            self.last_log = now;
            let line: Vec<String> = self
                .times
                .iter()
                .map(|(name, ms)| format!("{name} {ms:.2}"))
                .collect();
            debug!("gpu ms: {}", line.join(", "));
        }
    }

    /// Opens a scope named `name` on `cmd`.
    pub fn begin(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        name: &'static str,
    ) -> GpuScope {
        let label = self
            .labels
            .entry(name)
            .or_insert_with(|| CString::new(name).unwrap());
        unsafe {
            rd.ext_debug_utils.cmd_begin_debug_utils_label(
                cmd,
                &vk::DebugUtilsLabelEXT::default().label_name(label),
            );
        }
        let index = self.recorded.len() as u32;
        if index >= MAX_SCOPES {
            return GpuScope(None);
        }
        self.recorded.push(name);
        unsafe {
            rd.device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                2 * index,
            );
        }
        GpuScope(Some(index))
    }

    /// Closes `scope` on `cmd` (the buffer it was opened on).
    pub fn end(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer, scope: GpuScope) {
        if let GpuScope(Some(index)) = scope {
            unsafe {
                rd.device.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    self.pool,
                    2 * index + 1,
                );
            }
        }
        unsafe { rd.ext_debug_utils.cmd_end_debug_utils_label(cmd) };
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        if self.pool != vk::QueryPool::null() {
            unsafe { rd.device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut profiler: Mut<GpuProfiler>| {
        profiler.destroy(world.resource::<RenderDevice>());
    });
}

pub struct GpuProfilerPlugin;

impl Plugin for GpuProfilerPlugin {
    fn build(&self, app: &mut App) {
        let profiler = GpuProfiler::new(app.world().resource::<RenderDevice>());
        app.insert_resource(profiler)
            .add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
