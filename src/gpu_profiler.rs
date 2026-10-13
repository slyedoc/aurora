//! Per-pass GPU time. [`GpuProfiler::begin`] / [`GpuProfiler::end`] bracket a stretch of the
//! frame's commands with a debug-utils label (Nsight's GPU Trace names the pass) and two
//! timestamps. Once the frame's fence has passed they are read back, smoothed, logged every few
//! seconds and, with `trace_tracy`, plotted in Tracy as `gpu <name> ms`.

use std::{collections::HashMap, ffi::CString, sync::Mutex};

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
    /// One pool per frame parity: the frame being recorded writes its own while the previous
    /// frame's are still to be read.
    pools: [vk::QueryPool; 2],
    /// The parity being recorded.
    current: usize,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    /// Each parity's timed scopes, in begin order: scope `i` owns queries `2i` and `2i + 1`.
    /// Behind a lock: the frame's recorders open scopes from several threads.
    recorded: Mutex<[Vec<&'static str>; 2]>,
    labels: Mutex<HashMap<&'static str, CString>>,
    /// Smoothed milliseconds per scope name, in first-seen order.
    pub times: Vec<(&'static str, f64)>,
    /// The last frame read back, unsmoothed (scopes of one name summed).
    pub last_frame: Vec<(&'static str, f64)>,
    /// Frames read back so far; `last_frame` changes when this does.
    pub frames_read: u64,
    /// The last frame's final timestamp: the next frame's first one minus this is how long the
    /// GPU sat with nothing queued (`idle`).
    last_end: Option<u64>,
    /// Smoothed CPU milliseconds from the frame fence wait to the frame's submit: while the CPU
    /// spends them the GPU has nothing queued.
    pub cpu_record: f64,
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
        let pools = std::array::from_fn(|_| {
            let pool = unsafe { rd.device.create_query_pool(&info, None) }.unwrap();
            unsafe { rd.device.reset_query_pool(pool, 0, MAX_SCOPES * 2) };
            pool
        });
        let period_ns = unsafe {
            rd.instance
                .get_physical_device_properties(rd.physical_device)
                .limits
                .timestamp_period
        } as f64;
        Self {
            pools,
            current: 0,
            period_ns,
            recorded: Mutex::new([Vec::new(), Vec::new()]),
            labels: Mutex::new(HashMap::new()),
            times: Vec::new(),
            last_frame: Vec::new(),
            frames_read: 0,
            last_end: None,
            cpu_record: 0.0,
            last_log: 0.0,
            #[cfg(feature = "trace_tracy")]
            plots: HashMap::new(),
        }
    }

    /// Reads the previous frame's scopes (call after its fence) and starts this frame's.
    /// Starts recording frame `frame` into its parity's pool (read and reset two frames ago).
    pub fn begin_recording(&mut self, frame: usize) {
        self.current = frame % 2;
        self.recorded.get_mut().unwrap()[self.current].clear();
    }

    /// Reads back the previous frame's scopes -- once its work has finished -- and resets
    /// their pool for the frame after this one.
    pub fn read_previous(&mut self, rd: &RenderDevice, now: f64) {
        let previous = self.current ^ 1;
        let pool = self.pools[previous];
        let recorded = std::mem::take(&mut self.recorded.get_mut().unwrap()[previous]);
        let count = recorded.len() as u32;
        if count > 0 {
            let mut ticks = vec![0u64; (count * 2) as usize];
            let read = unsafe {
                rd.device
                    .get_query_pool_results(pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)
            };
            unsafe { rd.device.reset_query_pool(pool, 0, count * 2) };
            if read.is_ok() {
                let mut frame: Vec<(&'static str, f64)> = Vec::new();
                for (i, name) in recorded.iter().enumerate() {
                    let ms = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * self.period_ns
                        / 1.0e6;
                    match frame.iter_mut().find(|(n, _)| n == name) {
                        Some((_, total)) => *total += ms,
                        None => frame.push((name, ms)),
                    }
                }
                // Synthetic entries from absolute timestamps across every submit of the frame:
                // `span` first scope start to last scope end, `idle` the gap since the last frame.
                let start = (0..count as usize).map(|i| ticks[2 * i]).filter(|&t| t != 0).min();
                let end = (0..count as usize).map(|i| ticks[2 * i + 1]).max();
                if let (Some(start), Some(end)) = (start, end) {
                    let ms = |t: u64| t as f64 * self.period_ns / 1.0e6;
                    frame.push(("span", ms(end.saturating_sub(start))));
                    if let Some(last) = self.last_end {
                        frame.push(("idle", ms(start.saturating_sub(last))));
                    }
                    self.last_end = Some(end);
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

    /// The CPU's fence-to-submit time this frame (see [`Self::cpu_record`]).
    pub fn note_cpu_record(&mut self, ms: f64) {
        self.cpu_record += (ms - self.cpu_record) * SMOOTHING;
        #[cfg(feature = "trace_tracy")]
        if let Some(client) = tracy_client::Client::running() {
            client.plot(tracy_client::plot_name!("cpu record ms"), ms);
        }
    }

    /// A smoothed scope time by name.
    pub fn time(&self, name: &str) -> Option<f64> {
        self.times.iter().find(|(n, _)| *n == name).map(|(_, ms)| *ms)
    }

    /// Opens a scope named `name` on `cmd`.
    pub fn begin(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        name: &'static str,
    ) -> GpuScope {
        {
            let mut labels = self.labels.lock().unwrap();
            let label = labels
                .entry(name)
                .or_insert_with(|| CString::new(name).unwrap());
            unsafe {
                rd.ext_debug_utils.cmd_begin_debug_utils_label(
                    cmd,
                    &vk::DebugUtilsLabelEXT::default().label_name(label),
                );
            }
        }
        let index = {
            let mut recorded = self.recorded.lock().unwrap();
            let recorded = &mut recorded[self.current];
            let index = recorded.len() as u32;
            if index >= MAX_SCOPES {
                return GpuScope(None);
            }
            recorded.push(name);
            index
        };
        unsafe {
            rd.device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pools[self.current],
                2 * index,
            );
        }
        GpuScope(Some(index))
    }

    /// Closes `scope` on `cmd` (the buffer it was opened on).
    pub fn end(&self, rd: &RenderDevice, cmd: vk::CommandBuffer, scope: GpuScope) {
        if let GpuScope(Some(index)) = scope {
            unsafe {
                rd.device.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    self.pools[self.current],
                    2 * index + 1,
                );
            }
        }
        unsafe { rd.ext_debug_utils.cmd_end_debug_utils_label(cmd) };
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for pool in &mut self.pools {
            if *pool != vk::QueryPool::null() {
                unsafe { rd.device.destroy_query_pool(*pool, None) };
                *pool = vk::QueryPool::null();
            }
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
