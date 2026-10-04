//! GPU auto-exposure: histogram metering with percentile trimming and temporal adaptation,
//! ported from `bevy_post_process::auto_exposure` onto aurora's compute framework.
//!
//! The raygen writes each pixel's metering value into a buffer; next frame, before the trace,
//! `ae_histogram` + `ae_resolve` (assets/shaders/auto_exposure.slang) turn last frame's
//! buffer into a smoothed exposure that the raygen reads from a GPU state buffer -- no CPU
//! round-trip, one frame of latency that the adaptation smoothing hides.
//!
//! The meter is INCIDENT, not reflected: each pixel's radiance divided by its first
//! surface's albedo, so the exposure follows how much light falls on the scene rather than
//! how bright the surfaces in frame happen to be. Sky and directly seen emitters are left
//! out. A pale wall filling the screen, or a dark sky overhead, no longer moves it.
//!
//! Every camera meters for itself: each one has its own buffers and adaptation state.
//! With RR always ingesting mid-gray-centred colour, its internal per-frame estimator (the
//! dev-overlay flicker) has nothing left to do.
//!
//! Controlled by [`AuroraExposure`] on the camera (inserted on every `Camera3d`, edit it in
//! the F1 world inspector): `Auto` meters, `Fixed` locks an EV for a look that never
//! changes. Exposure has no other owner.

use ash::vk;
use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    assets::aurora_asset,
    compute::{
        CompiledComputeModule, ComputeModule, ComputeModules, memory_barrier, record_dispatch,
    },
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Must match `AE_HISTOGRAM_THREADS` in auto_exposure.slang.
const HISTOGRAM_THREADS: u32 = 16384;

/// Side of the square patch of `lum` copied back for the centre-screen nits readout.
/// A single pixel of 1-spp path radiance is mostly noise; a patch mean plus the temporal
/// smoothing below gives a number that can actually be read off the panel.
const PROBE_SIDE: u32 = 9;

/// Histogram domain in log2 nits: bin 1 at 2^-5 (deep shadow) through 2^27 (the sun disk).
const MIN_LOG_LUM: f32 = -5.0;
const LOG_LUM_RANGE: f32 = 32.0;

/// Camera exposure -- on every `Camera3d`, applied in the raygen (before Ray
/// Reconstruction, which needs pre-exposed colour). `World` (the default) takes the fixed
/// exposure of the world the camera is in; `Fixed` locks a look of its own; `Auto` meters
/// and adapts.
#[derive(Component, Reflect, Clone, PartialEq, Debug)]
#[reflect(Component, Default, Clone, PartialEq)]
pub enum AuroraExposure {
    /// Histogram metering with percentile trimming and temporal adaptation.
    Auto(AutoExposureSettings),
    /// A locked LOOK: applied in the blit, so the picture never adapts to content (Ray
    /// Reconstruction's input stays metered underneath -- changing the EV is instant and
    /// costs no denoiser history).
    Fixed(FixedExposure),
    /// The fixed exposure of the camera's world (sky.rs `WorldSky::ev100`: its
    /// `WorldExposure`, else its brightest sun's), so crossing a portal changes it.
    World,
}

/// The world's fixed exposure: a consistent look that never changes with what the camera
/// points at, set by how brightly the world is lit. `Auto` is opt-in, for scenes that move
/// between very different light levels.
impl Default for AuroraExposure {
    fn default() -> Self {
        Self::World
    }
}

/// `log2(1.2)`. Aurora's `ev` is log2 of the multiplier applied to radiance in nits;
/// EV100 is the photographic stop. Filament's mapping is `exposure = 1 / (1.2 * 2^EV100)`,
/// so `ev = -EV100 - log2(1.2)` -- which is why [`AuroraExposure::SUNLIGHT`] is -15.26 and
/// not -15. The panel speaks EV100 because that is the language of
/// aurora_files/lighting_units.md and of every exposure reference.
pub const EV100_OFFSET: f32 = 0.2630344;

/// EV100 -> aurora's internal `ev`.
pub fn ev_from_ev100(ev100: f32) -> f32 {
    -ev100 - EV100_OFFSET
}

/// Aurora's internal `ev` -> EV100.
pub fn ev100_from_ev(ev: f32) -> f32 {
    -ev - EV100_OFFSET
}

impl AuroraExposure {
    /// Bevy's `Exposure` presets (EV100, through Filament's `exp2(-ev100) / 1.2`),
    /// in aurora's convention: `ev` = log2 of the multiplier applied to radiance in nits.
    pub const SUNLIGHT: Self = Self::Fixed(FixedExposure { ev: -15.26 });
    pub const OVERCAST: Self = Self::Fixed(FixedExposure { ev: -12.26 });
    pub const INDOOR: Self = Self::Fixed(FixedExposure { ev: -7.26 });
    /// Calibrated to Blender's implicit exposure; a reasonable default look.
    pub const BLENDER: Self = Self::Fixed(FixedExposure { ev: -9.96 });

    pub fn fixed(ev: f32) -> Self {
        Self::Fixed(FixedExposure { ev })
    }

    /// `World` as the fixed exposure of a world at `world_ev100`; anything else unchanged.
    pub fn resolve(&self, world_ev100: f32) -> Self {
        match self {
            Self::World => Self::fixed(ev_from_ev100(world_ev100)),
            other => other.clone(),
        }
    }

    /// The blit's look: 0 = follow the metering, else the fixed linear exposure. `World`
    /// must be [resolved](Self::resolve) first.
    pub fn display_exposure(&self) -> f32 {
        match self {
            Self::Auto(_) => 0.0,
            Self::Fixed(fixed) => fixed.ev.exp2(),
            Self::World => ev_from_ev100(crate::sky::DEFAULT_EV100).exp2(),
        }
    }
}

#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct FixedExposure {
    /// log2 of the linear multiplier applied to radiance in nits; sunlit exteriors sit
    /// near -15.
    #[reflect(@-30.0..=0.0_f32)]
    pub ev: f32,
}

/// EV100 13: the exposure for a ~20 klux sun (EV100 = log2(lux * 100 / 250)).
impl Default for FixedExposure {
    fn default() -> Self {
        Self {
            ev: ev_from_ev100(13.0),
        }
    }
}

/// The metering: the trimmed-percentile histogram makes it immune to fireflies, the
/// adaptation makes it temporally stable.
#[derive(Reflect, Clone, PartialEq, Debug)]
#[reflect(Default, Clone, PartialEq)]
pub struct AutoExposureSettings {
    /// Smoothed exposure is clamped to this EV range.
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
    /// The look adapts around this exposure...
    #[reflect(@-30.0..=0.0_f32)]
    pub reference_ev: f32,
    /// ...by this fraction of the way to the metering: 1 = full auto (every scene lands at
    /// the same brightness), 0 = locked at `reference_ev`. In between, dim scenes stay
    /// dimmer than bright ones.
    #[reflect(@0.0..=1.0_f32)]
    pub adaptation: f32,
    /// Ray Reconstruction's input exposure follows the full metering, but its history
    /// cannot follow an exposure change: the input holds still within a quarter stop and
    /// drifts at this many EV per second beyond it (slow enough to stay unseen)...
    #[reflect(@0.0..=4.0_f32)]
    pub input_speed: f32,
    /// ...and snaps to the metering, with one visible pop, once it is this many EV off
    /// (the light changed wholesale).
    #[reflect(@0.25..=6.0_f32)]
    pub input_deadband: f32,
}

impl Default for AutoExposureSettings {
    fn default() -> Self {
        Self {
            min_ev: -30.0,
            max_ev: 0.0,
            filter_low: 0.10,
            filter_high: 0.90,
            // Snappy: a full interior<->exterior swing (~7 EV) settles in under a second,
            // with the exponential tail keeping the last stops smooth. Slow these towards
            // eye-adaptation (3.0 / 1.0) for a cinematic feel.
            speed_brighten: 10.0,
            speed_darken: 8.0,
            exponential_transition_distance: 1.5,
            // Metering targets photographic mid-gray; ACES reads a couple of stops under
            // that as "well exposed" rather than washed out.
            compensation: -2.0,
            reference_ev: -13.0,
            adaptation: 0.8,
            input_speed: 0.5,
            input_deadband: 2.0,
        }
    }
}

/// Must match `AeState` in auto_exposure.slang / `AeData` in types.glsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AeGpu {
    /// Smoothed log2 exposure (the Auto look).
    ev: f32,
    exposure: f32,
    /// What the raygen applies: `ev` quantised to whole EV steps with hysteresis, so Ray
    /// Reconstruction's history sees a still input exposure.
    input_ev: f32,
    input_exposure: f32,
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
    input_deadband: f32,
    input_speed: f32,
    reference_ev: f32,
    adaptation: f32,
    /// Keeps the struct free of implicit tail padding (u64 alignment) for `Pod`.
    _pad: u32,
}

/// One camera's metering: its own luminance buffer and adaptation state.
#[derive(Default)]
struct ViewExposure {
    /// Two floats per pixel at render resolution, written by the raygen: the metering value
    /// (incident luminance, or < 0 for pixels the meter skips) and the raw nits.
    lum: Buffer<f32>,
    pixels: u64,
    histogram: Buffer<u32>,
    state: Buffer<AeGpu>,
    /// Host-visible copy of a PROBE_SIDE^2 patch of `lum` at screen centre, for the panel
    /// readout. One frame behind, like the metering.
    probe: Buffer<f32>,
    /// Smoothed centre-screen scene luminance, NITS -- physical, pre-exposure, so it reads
    /// the same whatever the metering or the look is doing. See [`Self::probe_nits`].
    probe_nits: f32,
    /// `lum` holds a full traced frame at the current size (metering skips until then).
    primed: bool,
}

impl ViewExposure {
    /// Reads back the patch the PREVIOUS frame copied and folds it into the smoothed
    /// value. Called at the top of `record`, by which point (one frame in flight) that
    /// copy has landed.
    fn read_probe(&mut self, rd: &RenderDevice) {
        if self.probe.handle == vk::Buffer::null() || !self.primed {
            return;
        }
        let mut view = rd.map_buffer(&mut self.probe);
        // Interleaved (meter, nits) pairs; the readout is the nits.
        let nits: Vec<f32> = view
            .as_slice_mut()
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        let mean = nits.iter().sum::<f32>() / nits.len().max(1) as f32;
        // Geometric-ish smoothing: luminance spans decades, so a linear EMA would be
        // dominated by the brightest frames.
        const ALPHA: f32 = 0.15;
        self.probe_nits = if self.probe_nits > 0.0 && mean > 0.0 {
            (self.probe_nits.ln() * (1.0 - ALPHA) + mean.ln() * ALPHA).exp()
        } else {
            mean
        };
    }

    /// Copies the centre patch of `lum` into the host-visible probe buffer.
    fn record_probe(&self, rd: &RenderDevice, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
        if self.probe.handle == vk::Buffer::null() || extent.width == 0 || extent.height == 0 {
            return;
        }
        let side = PROBE_SIDE.min(extent.width).min(extent.height);
        let half = side / 2;
        let cx = (extent.width / 2).saturating_sub(half);
        let cy = (extent.height / 2).saturating_sub(half);
        // A pixel is two floats.
        let stride = 2 * std::mem::size_of::<f32>() as u64;
        // One region per row of the patch: `lum` is a flat row-major buffer, so a square
        // is not contiguous.
        let regions: Vec<vk::BufferCopy> = (0..side)
            .map(|row| {
                let src = ((cy + row) as u64 * extent.width as u64 + cx as u64) * stride;
                vk::BufferCopy {
                    src_offset: src,
                    dst_offset: row as u64 * side as u64 * stride,
                    size: side as u64 * stride,
                }
            })
            .collect();
        unsafe {
            rd.device
                .cmd_copy_buffer(cmd, self.lum.handle, self.probe.handle, &regions);
        }
    }

    /// Buffer addresses for the push constants: (per-pixel luminance, exposure state).
    fn addresses(&self) -> (u64, u64) {
        (self.lum.address, self.state.address)
    }

    /// Writes `ev` straight into the state buffer the raygen reads (input = smooth = `ev`).
    fn write_ev(&self, rd: &RenderDevice, cmd: vk::CommandBuffer, ev: f32) {
        let state = AeGpu {
            ev,
            exposure: ev.exp2(),
            input_ev: ev,
            input_exposure: ev.exp2(),
        };
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, self.state.handle, 0, bytemuck::bytes_of(&state));
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            // The raygen applies the exposure; the blit reads it for a fixed look's ratio.
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
    }

    /// Ensures the buffers cover `extent` and records the metering for this frame (from
    /// LAST frame's luminance). Metering ALWAYS runs -- it normalises what Ray
    /// Reconstruction ingests to mid-gray, keeping the denoiser and the dev overlay stable
    /// in every mode; a `Fixed` look re-exposes in the blit instead. Call before the trace
    /// is recorded.
    fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: Option<&CompiledComputeModule>,
        extent: vk::Extent2D,
        exposure: &AuroraExposure,
        dt: f32,
    ) {
        self.read_probe(rd);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        if self.state.handle == vk::Buffer::null() {
            self.histogram = rd.create_device_buffer(64, usage);
            self.state = rd.create_device_buffer(1, usage);
            self.probe = rd.create_host_buffer(
                2 * (PROBE_SIDE * PROBE_SIDE) as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
            );
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, self.histogram.handle, 0, vk::WHOLE_SIZE, 0);
            }
        }
        let pixels = extent.width as u64 * extent.height as u64;
        if pixels != self.pixels {
            rd.destroyer.destroy_buffer(self.lum.handle);
            self.lum = rd.create_device_buffer(2 * pixels.max(1), usage);
            unsafe {
                rd.device
                    .cmd_fill_buffer(cmd, self.lum.handle, 0, vk::WHOLE_SIZE, 0);
            }
            self.pixels = pixels;
            self.primed = false;
        }

        let fixed_settings;
        let settings = match exposure {
            AuroraExposure::Auto(settings) => settings,
            // A fixed look still meters the RR input; default metering does that job.
            AuroraExposure::Fixed(_) | AuroraExposure::World => {
                fixed_settings = AutoExposureSettings::default();
                &fixed_settings
            }
        };
        let (true, Some(module)) = (self.primed, module) else {
            // Not meterable yet: start adaptation from the reference exposure.
            self.write_ev(
                rd,
                cmd,
                settings
                    .reference_ev
                    .clamp(settings.min_ev, settings.max_ev),
            );
            return;
        };

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
            input_deadband: settings.input_deadband.max(0.0),
            input_speed: settings.input_speed.max(0.0),
            reference_ev: settings.reference_ev,
            adaptation: settings.adaptation.clamp(0.0, 1.0),
            _pad: 0,
        };

        // Last frame's raygen wrote `lum`; this frame's raygen reads the exposure. The
        // TRANSFER destination is the centre-patch probe copy below.
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::SHADER_READ
                | vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::TRANSFER_READ,
        );
        self.record_probe(rd, cmd, extent);
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
            // The raygen applies the exposure; the blit reads it for a fixed look's ratio.
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for handle in [
            self.lum.handle,
            self.histogram.handle,
            self.state.handle,
            self.probe.handle,
        ] {
            rd.destroyer.destroy_buffer(handle);
        }
        *self = Self::default();
    }
}

/// Every camera's metering, keyed by camera entity.
#[derive(Resource)]
pub struct AutoExposureState {
    module: Handle<ComputeModule>,
    views: HashMap<Entity, ViewExposure>,
    /// The camera the panel readout follows.
    primary: Option<Entity>,
}

impl AutoExposureState {
    fn new(module: Handle<ComputeModule>) -> Self {
        Self {
            module,
            views: HashMap::default(),
            primary: None,
        }
    }

    /// Centre-screen scene luminance in nits of the primary camera: what the surface under
    /// the crosshair is actually emitting or reflecting, independent of exposure and
    /// tonemap. Compare against the bands in aurora_files/lighting_units.md.
    pub fn probe_nits(&self) -> f32 {
        self.primary
            .and_then(|camera| self.views.get(&camera))
            .map_or(0.0, |view| view.probe_nits)
    }

    /// Records `camera`'s metering for this frame (from its LAST frame's luminance) and
    /// returns the (luminance, exposure state) addresses its trace writes and reads. Call
    /// once per camera per frame, before its trace is recorded; the first camera recorded
    /// in a frame is the primary.
    #[expect(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        camera: Entity,
        extent: vk::Extent2D,
        exposure: &AuroraExposure,
        dt: f32,
    ) -> (u64, u64) {
        let module = modules.get(&self.module);
        let view = self.views.entry(camera).or_default();
        view.record(rd, cmd, module, extent, exposure, dt);
        view.addresses()
    }

    pub fn set_primary(&mut self, camera: Option<Entity>) {
        self.primary = camera;
    }

    /// `camera`'s trace has filled its luminance buffer at the current size.
    pub fn mark_traced(&mut self, camera: Entity) {
        if let Some(view) = self.views.get_mut(&camera) {
            view.primed = true;
        }
    }

    /// Frees the metering of cameras `keep` rejects (despawned, or no longer 3D).
    pub fn retain(&mut self, rd: &RenderDevice, keep: impl Fn(Entity) -> bool) {
        self.views.retain(|camera, view| {
            let kept = keep(*camera);
            if !kept {
                view.destroy(rd);
            }
            kept
        });
    }
}

/// Every 3D camera carries an exposure (so it is always there to inspect).
fn default_exposure(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<AuroraExposure>)>,
) {
    for camera in &cameras {
        commands.entity(camera).insert(AuroraExposure::default());
    }
}

fn cleanup(mut state: ResMut<AutoExposureState>, rd: Res<RenderDevice>) {
    state.retain(&rd, |_| false);
}

pub struct AutoExposurePlugin;

impl Plugin for AutoExposurePlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/auto_exposure.slang"));
        let module = asset_server.add(ComputeModule::new(shader, &["ae_histogram", "ae_resolve"]));
        app.insert_resource(AutoExposureState::new(module));
        app.register_type::<AuroraExposure>();
        app.register_type::<AutoExposureSettings>();
        app.register_type::<FixedExposure>();
        app.add_systems(Update, default_exposure);
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
