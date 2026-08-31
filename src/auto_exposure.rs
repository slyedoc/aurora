//! GPU auto-exposure: histogram metering with percentile trimming and temporal adaptation,
//! ported from `bevy_post_process::auto_exposure` onto aurora's compute framework.
//!
//! The raygen writes each pixel's raw-radiance luminance into a buffer; next frame, before
//! the trace, `ae_histogram` + `ae_resolve` (assets/shaders/auto_exposure.slang) turn last
//! frame's buffer into a smoothed exposure that the raygen reads from a 2-float GPU state
//! buffer -- no CPU round-trip, one frame of latency that the adaptation smoothing hides.
//! With RR always ingesting mid-gray-centred colour, its internal per-frame estimator (the
//! dev-overlay flicker) has nothing left to do.
//!
//! Controlled by [`AuroraAutoExposure`] on the camera (inserted on every `Camera3d`, edit it
//! in the F1 world inspector); `enabled: false` falls back to the panel's manual
//! `exposure_ev`.

use ash::vk;
use bevy::prelude::*;

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Must match `AE_HISTOGRAM_THREADS` in auto_exposure.slang.
const HISTOGRAM_THREADS: u32 = 16384;

/// Histogram domain in log2 nits: bin 1 at 2^-5 (deep shadow) through 2^27 (the sun disk).
const MIN_LOG_LUM: f32 = -5.0;
const LOG_LUM_RANGE: f32 = 32.0;

/// Auto-exposure for the path tracer -- on every `Camera3d`. The metered, temporally-smoothed
/// exposure is applied in the raygen (before Ray Reconstruction, which needs pre-exposed
/// colour); the trimmed-percentile histogram makes it immune to fireflies.
#[derive(Component, Reflect, Clone, PartialEq, Debug)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct AuroraAutoExposure {
    /// Off = the dev panel's manual `exposure_ev`.
    pub enabled: bool,
    /// Smoothed exposure is clamped to this EV range (log2 of the linear multiplier
    /// applied to radiance in nits; sunlit exteriors sit near -15).
    #[reflect(@-30.0..=0.0_f32)]
    pub min_ev: f32,
    #[reflect(@-30.0..=0.0_f32)]
    pub max_ev: f32,
    /// Fraction of darkest samples excluded from metering.
    #[reflect(@0.0..=0.5_f32)]
    pub filter_low: f32,
    /// Fraction below which brightest samples are excluded (fireflies, the sun disk).
    #[reflect(@0.5..=1.0_f32)]
    pub filter_high: f32,
    /// Adaptation speed towards a brighter exposure, EV per second.
    #[reflect(@0.1..=20.0_f32)]
    pub speed_brighten: f32,
    #[reflect(@0.1..=20.0_f32)]
    pub speed_darken: f32,
    /// EV distance over which adaptation switches from linear to exponential.
    #[reflect(@0.01..=10.0_f32)]
    pub exponential_transition_distance: f32,
    /// Artist EV offset on the mid-gray target.
    #[reflect(@-8.0..=8.0_f32)]
    pub compensation: f32,
}

impl Default for AuroraAutoExposure {
    fn default() -> Self {
        Self {
            enabled: true,
            min_ev: -30.0,
            max_ev: 0.0,
            filter_low: 0.10,
            filter_high: 0.90,
            speed_brighten: 3.0,
            speed_darken: 1.0,
            exponential_transition_distance: 1.5,
            // Metering targets photographic mid-gray; ACES reads a stop under that as
            // "well exposed" rather than washed out.
            compensation: -2.0,
        }
    }
}

/// Must match `AeState` in auto_exposure.slang / `AeData` in types.glsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AeGpu {
    ev: f32,
    exposure: f32,
}

/// Must match `AeParams` in auto_exposure.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AeParams {
    lum: u64,
    histogram: u64,
    state: u64,
    pixel_count: u32,
    min_log_lum: f32,
    inv_log_lum_range: f32,
    log_lum_range: f32,
    low_percent: f32,
    high_percent: f32,
    speed_brighten: f32,
    speed_darken: f32,
    exp_transition: f32,
    dt: f32,
    compensation: f32,
    min_ev: f32,
    max_ev: f32,
    /// Keeps the struct free of implicit tail padding (u64 alignment) for `Pod`.
    _pad: u32,
}

#[derive(Resource)]
pub struct AutoExposureState {
    module: Handle<ComputeModule>,
    /// Per-pixel luminance at render resolution, written by the raygen.
    lum: Buffer<f32>,
    pixels: u64,
    histogram: Buffer<u32>,
    state: Buffer<AeGpu>,
    /// `lum` holds a full traced frame at the current size (metering skips until then).
    pub primed: bool,
}

impl AutoExposureState {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            lum: Buffer::default(),
            pixels: 0,
            histogram: Buffer::default(),
            state: Buffer::default(),
            primed: false,
        }
    }

    /// Buffer addresses for the push constants: (per-pixel luminance, exposure state).
    pub fn addresses(&self) -> (u64, u64) {
        (self.lum.address, self.state.address)
    }

    /// Ensures the buffers cover `extent` and records the metering for this frame (from
    /// LAST frame's luminance). When disabled or unprimed the state buffer just gets the
    /// manual exposure. Call before the trace is recorded.
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        extent: vk::Extent2D,
        settings: &AuroraAutoExposure,
        manual_ev: f32,
        dt: f32,
    ) {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        if self.state.handle == vk::Buffer::null() {
            self.histogram = rd.create_device_buffer(64, usage);
            self.state = rd.create_device_buffer(1, usage);
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, self.histogram.handle, 0, vk::WHOLE_SIZE, 0);
            }
        }
        let pixels = extent.width as u64 * extent.height as u64;
        if pixels != self.pixels {
            rd.destroyer.destroy_buffer(self.lum.handle);
            self.lum = rd.create_device_buffer(pixels.max(1), usage);
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, self.lum.handle, 0, vk::WHOLE_SIZE, 0);
            }
            self.pixels = pixels;
            self.primed = false;
        }

        let metering = settings.enabled && self.primed && modules.get(&self.module).is_some();
        if !metering {
            // Manual (or not yet meterable): the raygen still reads the state buffer.
            let manual = AeGpu {
                ev: manual_ev,
                exposure: manual_ev.exp2(),
            };
            unsafe {
                rd.device.cmd_update_buffer(
                    cmd,
                    self.state.handle,
                    0,
                    bytemuck::bytes_of(&manual),
                );
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_READ,
            );
            return;
        }
        let module = modules.get(&self.module).unwrap();

        let params = AeParams {
            lum: self.lum.address,
            histogram: self.histogram.address,
            state: self.state.address,
            pixel_count: self.pixels as u32,
            min_log_lum: MIN_LOG_LUM,
            inv_log_lum_range: 1.0 / LOG_LUM_RANGE,
            log_lum_range: LOG_LUM_RANGE,
            low_percent: settings.filter_low.clamp(0.0, 1.0),
            high_percent: settings.filter_high.clamp(0.0, 1.0),
            speed_brighten: settings.speed_brighten.max(0.0),
            speed_darken: settings.speed_darken.max(0.0),
            exp_transition: settings.exponential_transition_distance.max(1.0e-3),
            dt: dt.clamp(0.0, 0.25),
            compensation: settings.compensation,
            min_ev: settings.min_ev,
            max_ev: settings.max_ev.max(settings.min_ev),
            _pad: 0,
        };

        // Last frame's raygen wrote `lum`; this frame's raygen reads the exposure.
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        record_dispatch(
            rd,
            cmd,
            module,
            "ae_histogram",
            &params,
            HISTOGRAM_THREADS,
            None,
        );
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        record_dispatch(rd, cmd, module, "ae_resolve", &params, 1, None);
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
    }
}

/// Every 3D camera meters (so the component is always there to inspect).
fn default_auto_exposure(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<AuroraAutoExposure>)>,
) {
    for camera in &cameras {
        commands.entity(camera).insert(AuroraAutoExposure::default());
    }
}

fn cleanup(mut state: ResMut<AutoExposureState>, rd: Res<RenderDevice>) {
    rd.destroyer.destroy_buffer(state.lum.handle);
    rd.destroyer.destroy_buffer(state.histogram.handle);
    rd.destroyer.destroy_buffer(state.state.handle);
    state.lum = Buffer::default();
    state.histogram = Buffer::default();
    state.state = Buffer::default();
}

pub struct AutoExposurePlugin;

impl Plugin for AutoExposurePlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/auto_exposure.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["ae_histogram", "ae_resolve"]));
        app.insert_resource(AutoExposureState::new(module));
        app.register_type::<AuroraAutoExposure>();
        app.add_systems(Update, default_auto_exposure);
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
