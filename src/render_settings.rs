//! The renderer's tunables, each where it belongs: [`RenderSettings`] (a resource: path quality,
//! light sampling, caches, DLSS) and [`AuroraLens`] (a camera's depth of field and vignette). An
//! environment's fog and sky level are [`Fog`](crate::sky::Fog) and
//! [`SkyBrightness`](crate::sky::SkyBrightness) on the environment entity; a camera's exposure and
//! DLSS mode are [`AuroraExposure`](crate::auto_exposure::AuroraExposure) and
//! [`AuroraDlss`](crate::dlss::AuroraDlss).

use bevy::prelude::*;

use crate::dlss::{RrPreset, set_jitter_scale};

/// How the frame is traced. Read every frame; edit it from any inspector. `samples`,
/// `max_bounces`, `light_candidates`, `spec_hit_roughness`, `roulette_from`, `restir` and
/// `restir_gi` are built into the ray tracing pipeline (specialization constants): changing one
/// rebuilds it.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Default)]
pub struct RenderSettings {
    /// Paths per pixel per frame; RR is trained for 1 spp, so extra samples mostly buy
    /// trace time.
    #[reflect(@1.0..=4.0_f32)]
    pub samples: u32,
    /// Maximum path length (Russian roulette ends most paths well before it).
    #[reflect(@1.0..=64.0_f32)]
    pub max_bounces: u32,
    /// Uniform multiplier on every light's emission, emissive surfaces and analytic lights
    /// alike. Uniform, so NEE stays unbiased.
    #[reflect(@0.0..=100.0_f32)]
    pub emissive_boost: f32,
    /// Next-event estimation for emissive triangles (off = BRDF sampling only, the
    /// reference estimator).
    pub light_nee: bool,
    /// Light candidates resampled at every shading point (RIS); deeper bounces use a quarter.
    #[reflect(@1.0..=32.0_f32)]
    pub light_candidates: u32,
    /// Added to the ray-cone texture LOD on top of log2(render / output). The DLSS guide asks
    /// for -1; past -2 textureLod clamps to mip 0.
    #[reflect(@-2.0..=1.0_f32)]
    pub texture_lod_bias: f32,
    /// Ray Reconstruction's specular hit distance comes from one extra mirror ray at the
    /// primary vertex for surfaces up to this roughness. 0 = no guide rays. Past ~0.3 the
    /// reflection is too blurry for its motion to show, and at 0.6 the ray cost Northshire's
    /// textured stone and wood ~1.4 ms at 3840x1080.
    #[reflect(@0.0..=1.0_f32)]
    pub spec_hit_roughness: f32,
    /// Russian roulette: from this bounce on, a path continues with probability its throughput
    /// (its brightest channel, at least 5 %) and survivors are boosted to match, so dim paths end
    /// early instead of running to `max_bounces`. Unbiased; 0 = off.
    #[reflect(@0.0..=8.0_f32)]
    pub roulette_from: u32,
    /// ReSTIR DI for the light table at the primary vertex: candidates from a world light grid,
    /// temporal and spatial reuse, one shadow ray (restir.rs). On by default: at 1 spp without
    /// it Ray Reconstruction drops the rare paths that find a lamp (Northshire's lamp-lit plaza
    /// a quarter dark against a 4 spp reference; with it within ~3 %, for ~0.7 ms).
    pub restir: bool,
    /// Initial light candidates per pixel (from its grid cell).
    #[reflect(@1.0..=32.0_f32)]
    pub restir_candidates: u32,
    /// Temporal history cap, in frames: short keeps Ray Reconstruction's input decorrelated.
    #[reflect(@1.0..=64.0_f32)]
    pub restir_history: f32,
    /// Neighbours merged by the spatial pass (0 = no spatial reuse) and their disc (pixels).
    #[reflect(@0.0..=16.0_f32)]
    pub restir_spatial_samples: u32,
    #[reflect(@1.0..=64.0_f32)]
    pub restir_spatial_radius: f32,
    /// Boiling filter: a reservoir more than 1/this times its neighbourhood's is dropped
    /// (0 = off).
    #[reflect(@0.0..=1.0_f32)]
    pub restir_boiling: f32,
    /// ReSTIR PT (restir_gi.rs; docs/plans/restir-pt.html; the setting keeps ReSTIR GI's name,
    /// which it replaced): sample 0's path from its reconnection vertex on resampled across
    /// neighbouring pixels (and, with `restir_pt_history`, frames) by the hybrid shift, shaded by
    /// a pass of its own. Unbiased against the plain path tracer (sun room references).
    pub restir_gi: bool,
    /// Perceptual roughness from which a vertex may begin or end a reconnection (both ends of
    /// the reconnection segment must be at least this rough).
    #[reflect(@0.0..=1.0_f32)]
    pub restir_gi_roughness: f32,
    /// Neighbours merged by the spatial pass (0 = none) and their disc (pixels). 4 by default:
    /// sun room at DLAA, Ray Reconstruction's error against the reference 0.035 -> 0.027 and
    /// black input pixels 61 -> 21 %, for ~1.5 ms at 1920x1080.
    #[reflect(@0.0..=16.0_f32)]
    pub restir_gi_spatial_samples: u32,
    #[reflect(@1.0..=64.0_f32)]
    pub restir_gi_spatial_radius: f32,
    /// ReSTIR PT's temporal reuse: the confidence (samples) last frame's reservoir may carry into
    /// this frame's (0 = none, the default), lowered as the history moves or the suns change.
    /// Unbiased, but Ray Reconstruction takes the reused samples' correlation for signal: in a
    /// sparse-lit interior (Northshire's abbey) its error against the reference rises (spatial
    /// alone 0.113, cap 4 0.139, 9 % bright) and at 3840x1080 it costs 1.3 ms; in the sun room cap
    /// 4 cut RR's error 0.031 -> 0.025.
    #[reflect(@0.0..=32.0_f32)]
    pub restir_pt_history: f32,
    /// A/B switch for measuring a change in one run (benches/northshire.rs `--ab`): shaders
    /// under test take their old path at 0 and the new one at 1.
    pub ab: u32,
    /// The light grid's cell size (m) and samples per cell.
    #[reflect(@0.5..=32.0_f32)]
    pub light_grid_cell: f32,
    #[reflect(@1.0..=64.0_f32)]
    pub light_grid_samples: u32,
    /// Radiance cache (SHaRC-style): indirect paths end in converged voxels from the first bounce
    /// on (a few train it to the end). Off by default: the stopgap before ReSTIR PT, which now
    /// gives Ray Reconstruction the dense input (sun room with PT: RR error 0.017 with it, 0.020
    /// without, within noise) and is faster without it (trace 1.9 -> 1.6 ms at 1080p).
    pub sharc: bool,
    /// Cache voxel size at the camera (meters); doubles per distance octave past 8m.
    #[reflect(@0.05..=2.0_f32)]
    pub sharc_voxel: f32,
    /// Neural radiance cache (nrc.rs; off on a device without VK_NV_cooperative_vector): paths
    /// that have spread far end in a small network's answer, trained online from the frame's own
    /// paths (docs/plans/nrc.html). A no-cache truth for comparisons turns it off.
    pub nrc: bool,
    /// A path ends in the cache once its footprint has spread past this x the primary vertex's
    /// (larger: later, less bias, more tracing).
    #[reflect(@0.0001..=1.0_f32)]
    pub nrc_spread: f32,
    /// Share of paths traced on to teach the cache (capped near 64k records a frame).
    #[reflect(@0.0..=0.2_f32)]
    pub nrc_train_fraction: f32,
    /// The training's Adam step size.
    #[reflect(@0.0001..=0.02_f32)]
    pub nrc_learning_rate: f32,
    /// Opacity micromaps on alpha-cutout meshes that carry a bake (off = any-hit alpha test).
    pub omm: bool,
    /// Ray Reconstruction model preset; changing it rebuilds the feature.
    pub rr_preset: RrPreset,
    /// Reflex (latency.rs): On makes the CPU wait for the driver at the start of a frame
    /// instead of queueing work ahead of the GPU. Flat windows on NVIDIA only.
    pub low_latency: crate::latency::LowLatency,
    /// Sub-pixel camera jitter: 1 = the full +-0.5 traced pixel, 0 = pixel centres.
    #[reflect(@0.0..=1.0_f32)]
    pub jitter_scale: f32,
    /// Local tone mapping: how far each region's exposure moves from its surroundings' average
    /// toward the whole frame's (0 = one exposure for the frame). Lifts deep shade and holds
    /// back sunlit ground in one view; a frame dark all over keeps its exposure.
    #[reflect(@0.0..=1.0_f32)]
    pub local_tonemap: f32,
    /// Bloom and glare: the share of each pixel's light spread over its surroundings (a lens's
    /// scatter: the output's mip pyramid, levels 1-6 averaged). Energy-conserving -- it moves
    /// light, so the sun, lamps and sunlit leaves spill into what is around them and nothing
    /// brightens overall. 0 = off.
    #[reflect(@0.0..=0.3_f32)]
    pub bloom: f32,
    /// ACES filmic tone mapping in the composite; off shows the exposed image linearly, clipped
    /// at white (for judging the render itself).
    pub tonemap: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            samples: 1,
            max_bounces: 4,
            emissive_boost: 1.0,
            light_nee: true,
            light_candidates: 4,
            texture_lod_bias: -1.0,
            spec_hit_roughness: 0.3,
            roulette_from: 1,
            restir: true,
            restir_candidates: 4,
            restir_history: 8.0,
            restir_spatial_samples: 2,
            restir_spatial_radius: 30.0,
            restir_boiling: 0.0,
            restir_gi: true,
            restir_gi_roughness: 0.35,
            restir_gi_spatial_samples: 4,
            restir_gi_spatial_radius: 30.0,
            restir_pt_history: 0.0,
            ab: 0,
            light_grid_cell: 4.0,
            light_grid_samples: 16,
            sharc: false,
            sharc_voxel: 0.25,
            nrc: true,
            nrc_spread: 0.01,
            nrc_train_fraction: 0.03,
            nrc_learning_rate: 2.0e-3,
            omm: true,
            rr_preset: RrPreset::current(),
            low_latency: crate::latency::LowLatency::On,
            jitter_scale: 1.0,
            local_tonemap: 0.0,
            bloom: 0.0,
            tonemap: false,
        }
    }
}

/// A camera's lens: depth of field and vignette. Without one, a pinhole with no vignette.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Component, Default)]
pub struct AuroraLens {
    /// Aperture radius (meters); 0 = pinhole, everything sharp.
    #[reflect(@0.0..=0.02_f32)]
    pub aperture: f32,
    /// Post-process vignette strength (0 = off); aspect-corrected.
    #[reflect(@0.0..=1.0_f32)]
    pub vignette: f32,
}

pub struct RenderSettingsPlugin;

impl Plugin for RenderSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<RenderSettings>()
            .register_type::<AuroraLens>()
            .add_systems(Update, apply_dlss_settings);
    }

    // After `DlssPlugin::build` has set the preset the defaults read.
    fn finish(&self, app: &mut App) {
        app.init_resource::<RenderSettings>();
    }
}

fn apply_dlss_settings(settings: Option<Res<RenderSettings>>) {
    let Some(settings) = settings.filter(|settings| settings.is_changed()) else {
        return;
    };
    if settings.rr_preset != RrPreset::current() {
        settings.rr_preset.make_current();
    }
    set_jitter_scale(settings.jitter_scale);
}
