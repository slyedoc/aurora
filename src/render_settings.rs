//! The renderer's tunables, each where it belongs: [`RenderSettings`] (a resource: path quality,
//! light sampling, caches, DLSS) and [`AuroraLens`] (a camera's depth of field and vignette). An
//! environment's fog and sky level are [`Fog`](crate::sky::Fog) and
//! [`SkyBrightness`](crate::sky::SkyBrightness) on the environment entity; a camera's exposure and
//! DLSS mode are [`AuroraExposure`](crate::auto_exposure::AuroraExposure) and
//! [`AuroraDlss`](crate::dlss::AuroraDlss).

use bevy::prelude::*;

use crate::dlss::{RrPreset, set_jitter_scale};

/// How the frame is traced. Read every frame; edit it from any inspector.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Default)]
pub struct RenderSettings {
    /// Paths per pixel per frame; RR is trained for 1 spp, so extra samples mostly buy
    /// trace time.
    #[reflect(@1.0..=4.0_f32)]
    pub samples: u32,
    /// Maximum path length. With the firefly clamp a short path is all the denoiser needs.
    #[reflect(@1.0..=64.0_f32)]
    pub max_bounces: u32,
    /// Firefly suppression: indirect contributions are clamped to this many times the
    /// metered mid-gray luminance (0 = off). Biases bright indirect paths down; kills the
    /// speckle Ray Reconstruction would otherwise smear.
    #[reflect(@0.0..=64.0_f32)]
    pub firefly_clamp: f32,
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
    /// primary vertex for surfaces up to this roughness. 0 = no guide rays.
    #[reflect(@0.0..=1.0_f32)]
    pub spec_hit_roughness: f32,
    /// ReSTIR DI at the primary vertex (initial candidates + temporal reuse).
    pub restir: bool,
    /// Initial light candidates per pixel.
    #[reflect(@1.0..=32.0_f32)]
    pub restir_candidates: u32,
    /// Temporal history cap, in multiples of the candidate count.
    #[reflect(@0.0..=64.0_f32)]
    pub restir_history: f32,
    /// Radiance cache: paths terminate into converged voxels from bounce 2 on. Biased.
    pub sharc: bool,
    /// Cache voxel size at the camera (meters); doubles per distance octave past 8m.
    #[reflect(@0.05..=2.0_f32)]
    pub sharc_voxel: f32,
    /// Opacity micromaps on alpha-cutout meshes that carry a bake (off = any-hit alpha test).
    pub omm: bool,
    /// Ray Reconstruction model preset; changing it rebuilds the feature.
    pub rr_preset: RrPreset,
    /// Sub-pixel camera jitter: 1 = the full +-0.5 traced pixel, 0 = pixel centres.
    #[reflect(@0.0..=1.0_f32)]
    pub jitter_scale: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            samples: 1,
            max_bounces: 32,
            firefly_clamp: 8.0,
            emissive_boost: 1.0,
            light_nee: true,
            light_candidates: 8,
            texture_lod_bias: -1.0,
            spec_hit_roughness: 0.6,
            restir: false,
            restir_candidates: 8,
            restir_history: 20.0,
            sharc: false,
            sharc_voxel: 0.25,
            omm: true,
            rr_preset: RrPreset::current(),
            jitter_scale: 1.0,
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
