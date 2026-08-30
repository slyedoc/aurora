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
//! Without next-event estimation a small, physically bright sun is only ever found by chance
//! (and its indirect contribution is then firefly-clamped), so the procedural sun defaults
//! to a wide, moderate disc that still reads as a sun and lights the scene; a true 0.27° sun
//! is a job for NEE.

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
            Sky::Procedural => luma(procedural.zenith),
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
    /// Angular radius of the disc. The real sun is 0.27°; wide and soft converges without NEE.
    #[reflect(@0.25..=20.0_f32)]
    pub sun_angular_radius: f32,
    /// Radiance of the disc (nits).
    #[reflect(@0.0..=2.0e6_f32)]
    pub sun_radiance: f32,
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
}

impl Default for ProceduralSky {
    fn default() -> Self {
        Self {
            sun_elevation: 40.0,
            sun_azimuth: 150.0,
            sun_angular_radius: 6.0,
            sun_radiance: 1.2e5,
            zenith: Vec3::new(0.45, 0.62, 1.0) * 8000.0,
            horizon: Vec3::new(0.80, 0.86, 0.95) * 9000.0,
            ground: Vec3::new(0.30, 0.28, 0.25) * 2500.0,
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
