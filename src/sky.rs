//! The sky: what a ray that escapes the scene returns.
//!
//! Three sources, all in nits so they sit on the same scale as the importers' emitters
//! (Bistro's lamps are 20,000) and the viewers' `exposure_ev = -13`:
//! - [`Sky::Color`]: one flat radiance.
//! - [`Sky::Hdr`]: an equirectangular image (a linear `.hdr` / `.exr`), texel × `scale`.
//! - [`Sky::Procedural`]: an analytic clear sky from [`ProceduralSky`] -- zenith / horizon
//!   gradient, ground, and a soft sun disc. The default: nothing to load.
//!
//! [`DevUIState::sky_brightness`](crate::dev_ui::DevUIState) multiplies whichever is active.
//! The procedural sun is gathered by next-event estimation in the raygen (one shadow ray per
//! surface hit towards the disc), so it can be small and physically bright: the defaults
//! give ~115 klux of direct sunlight against a ~25 klux sky, i.e. real shadows.

use bevy::prelude::*;

#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub enum Sky {
    /// Flat radiance in nits.
    Color { radiance: Vec3 },
    /// Equirectangular image (linear float); texel × `scale` = nits.
    Hdr { image: Handle<Image>, scale: f32 },
    /// Analytic clear sky from the [`ProceduralSky`] resource.
    Procedural,
}

impl Default for Sky {
    fn default() -> Self {
        Self::Procedural
    }
}

impl Sky {
    /// The luminance the firefly clamp is relative to (the sky's typical radiance, nits).
    pub fn reference_luminance(&self, procedural: &ProceduralSky) -> f32 {
        match self {
            Sky::Color { radiance } => luma(*radiance),
            Sky::Hdr { scale, .. } => *scale,
            Sky::Procedural => luma(procedural.zenith_radiance()),
        }
    }
}

/// Parameters of [`Sky::Procedural`]; radiances in nits, angles in degrees.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub struct ProceduralSky {
    /// Sun height above the horizon.
    #[reflect(@-10.0..=90.0_f32)]
    pub sun_elevation: f32,
    /// Sun heading, degrees clockwise from -Z.
    #[reflect(@0.0..=360.0_f32)]
    pub sun_azimuth: f32,
    /// Angular radius of the disc (the real sun is 0.27°); softer shadows as it grows.
    #[reflect(@0.25..=20.0_f32)]
    pub sun_angular_radius: f32,
    /// Radiance of the disc (nits). Irradiance = radiance × the disc's solid angle, so a
    /// smaller disc needs a brighter one for the same light on the ground.
    #[reflect(@0.0..=1.0e8_f32)]
    pub sun_radiance: f32,
    /// Sky colour straight up (chromaticity) and its radiance (nits).
    pub zenith: Color,
    #[reflect(@0.0..=30000.0_f32)]
    pub zenith_nits: f32,
    /// Sky colour at the horizon and its radiance (nits).
    pub horizon: Color,
    #[reflect(@0.0..=30000.0_f32)]
    pub horizon_nits: f32,
    /// Below the horizon and its radiance (nits).
    pub ground: Color,
    #[reflect(@0.0..=30000.0_f32)]
    pub ground_nits: f32,
}

impl Default for ProceduralSky {
    fn default() -> Self {
        Self {
            sun_elevation: 35.0,
            sun_azimuth: 150.0,
            sun_angular_radius: 2.0,
            sun_radiance: 3.0e7,
            zenith: Color::linear_rgb(0.45, 0.62, 1.0),
            zenith_nits: 8000.0,
            horizon: Color::linear_rgb(0.80, 0.86, 0.95),
            horizon_nits: 9000.0,
            ground: Color::linear_rgb(0.30, 0.28, 0.25),
            ground_nits: 2500.0,
        }
    }
}

impl ProceduralSky {
    /// Unit vector towards the sun (y up, azimuth 0 = -Z).
    pub fn sun_direction(&self) -> Vec3 {
        let (el, az) = (
            self.sun_elevation.to_radians(),
            self.sun_azimuth.to_radians(),
        );
        Vec3::new(el.cos() * az.sin(), el.sin(), -el.cos() * az.cos()).normalize()
    }

    pub fn sun_cos_radius(&self) -> f32 {
        self.sun_angular_radius.to_radians().cos()
    }

    /// Linear radiance (nits) of the zenith / horizon / ground.
    pub fn zenith_radiance(&self) -> Vec3 {
        radiance(self.zenith, self.zenith_nits)
    }
    pub fn horizon_radiance(&self) -> Vec3 {
        radiance(self.horizon, self.horizon_nits)
    }
    pub fn ground_radiance(&self) -> Vec3 {
        radiance(self.ground, self.ground_nits)
    }
}

/// A colour (any space) times a radiance, as linear RGB nits.
fn radiance(color: Color, nits: f32) -> Vec3 {
    color.to_linear().to_vec3() * nits
}

pub fn luma(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Sky>()
            .register_type::<ProceduralSky>()
            .init_resource::<Sky>()
            .init_resource::<ProceduralSky>();
    }
}
