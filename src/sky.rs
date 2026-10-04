//! Each world's sky and suns: what a ray that escapes the scene returns, and the directional
//! light it is lit by.
//!
//! A world (the main world, or any other avian `PhysicsWorld`, src/world.rs) carries its own
//! [`Sky`], and [`GradientSky`] for the gradient's colours; a world without one gets the
//! defaults. A ray evaluates the sky of the world it is IN (portals swap that), so two
//! portal-linked worlds each have their own.
//!
//! Suns are bevy's `DirectionalLight`s, in the world they belong to like any other entity:
//! direction from the light's transform, `illuminance` the lux on a surface facing it, the
//! disc's size from an optional `SunDisk` (Earth's sun without one). Up to [`MAX_SUNS`] per
//! world. Every sky draws the discs of its world's suns on the paths that may see them; the
//! raygen gathers their light by next-event estimation, one shadow ray per sun per hit.
//!
//! Radiances are in nits so they sit on the same scale as the importers' emitters;
//! [`DevUIState::sky_brightness`](crate::dev_ui::DevUIState) multiplies the sky.

use bevy::{
    camera::visibility::RenderLayers,
    light::{DirectionalLight, SunDisk},
    prelude::*,
};

use bytemuck::{Pod, Zeroable};

use crate::world::{InWorld, RenderWorlds, world_mask};

/// Suns per world the GPU table holds; more are ignored.
pub const MAX_SUNS: usize = 4;

/// The exposure (EV100) of a world with no [`WorldExposure`] and no sun.
pub const DEFAULT_EV100: f32 = 13.0;

/// The exposure a world's cameras take by default (`AuroraExposure::World`), in EV100. On a
/// world entity; without one it follows the world's brightest sun.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct WorldExposure {
    #[reflect(@0.0..=20.0_f32)]
    pub ev100: f32,
}

impl Default for WorldExposure {
    fn default() -> Self {
        Self {
            ev100: DEFAULT_EV100,
        }
    }
}

/// The exposure a sun of `lux` calls for: incident-meter EV100 = log2(lux * 100 / 250).
pub fn ev100_for_lux(lux: f32) -> f32 {
    (lux.max(1.0e-3) * 100.0 / 250.0).log2()
}

/// What a ray that escapes this world returns. On a world entity.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default)]
pub enum Sky {
    /// Flat radiance in nits.
    Color { radiance: Vec3 },
    /// Equirectangular image (linear float); texel × `scale` = nits.
    Hdr { image: Handle<Image>, scale: f32 },
    /// Zenith / horizon / ground gradient from the world's [`GradientSky`].
    #[default]
    Gradient,
    /// A planet's air and cloud shell ([`crate::atmosphere::Atmosphere`] /
    /// [`crate::atmosphere::CloudLayer`]), lit by the world's first sun. One world at a time:
    /// the atmosphere's lookup tables are global.
    Atmosphere,
}

/// The colours of [`Sky::Gradient`]; radiances in nits. On a world entity.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
pub struct GradientSky {
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

impl Default for GradientSky {
    fn default() -> Self {
        Self {
            zenith: Color::linear_rgb(0.45, 0.62, 1.0),
            zenith_nits: 8000.0,
            horizon: Color::linear_rgb(0.80, 0.86, 0.95),
            horizon_nits: 9000.0,
            ground: Color::linear_rgb(0.30, 0.28, 0.25),
            ground_nits: 2500.0,
        }
    }
}

impl GradientSky {
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

/// A sun as the tracer sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Unit vector towards the sun.
    pub direction: Vec3,
    /// Cosine of the disc's angular radius.
    pub cos_radius: f32,
    /// The disc's radiance (nits): `illuminance` spread over its solid angle.
    pub radiance: Vec3,
    /// `SunDisk::intensity`: scales the disc where it is seen directly, not its light.
    pub disc: f32,
}

impl Sun {
    pub fn of(
        light: &DirectionalLight,
        disk: Option<&SunDisk>,
        transform: &GlobalTransform,
    ) -> Self {
        let disk = disk.cloned().unwrap_or(SunDisk::EARTH);
        let cos_radius = (disk.angular_size * 0.5).clamp(1.0e-4, 0.5).cos();
        let solid_angle = std::f32::consts::TAU * (1.0 - cos_radius);
        Self {
            // A directional light shines along its forward (-Z); the sun is behind it.
            direction: *transform.back(),
            cos_radius,
            radiance: light.color.to_linear().to_vec3() * (light.illuminance / solid_angle),
            disc: disk.intensity,
        }
    }

    pub fn solid_angle(&self) -> f32 {
        std::f32::consts::TAU * (1.0 - self.cos_radius)
    }
}

/// One world's sky, suns and exposure, gathered each frame.
#[derive(Clone, Debug)]
pub struct WorldSky {
    pub sky: Sky,
    pub gradient: GradientSky,
    pub suns: Vec<Sun>,
    /// The fixed exposure its cameras take by default (EV100).
    pub ev100: f32,
}

impl Default for WorldSky {
    fn default() -> Self {
        Self {
            sky: Sky::default(),
            gradient: GradientSky::default(),
            suns: Vec::new(),
            ev100: DEFAULT_EV100,
        }
    }
}

/// Every world's sky and suns, indexed by world bit; rebuilt each frame before rendering.
#[derive(Resource, Default, Debug)]
pub struct WorldSkies {
    pub worlds: [WorldSky; 8],
    /// The world bit of the first active camera (its HDR sky gets importance sampling).
    pub camera_world: u8,
    /// The one world whose sky is the atmosphere, if any.
    pub atmosphere: Option<u8>,
}

impl WorldSkies {
    pub fn camera(&self) -> &WorldSky {
        &self.worlds[self.camera_world as usize]
    }
}

#[allow(clippy::type_complexity)]
fn gather_skies(
    worlds: Option<Res<RenderWorlds>>,
    skies: Query<(Option<&Sky>, Option<&GradientSky>, Option<&WorldExposure>)>,
    suns: Query<(
        &DirectionalLight,
        Option<&SunDisk>,
        &GlobalTransform,
        Option<&RenderLayers>,
        Option<&InWorld>,
        Option<&InheritedVisibility>,
    )>,
    cameras: Query<(&Camera, Option<&RenderLayers>, Option<&InWorld>), With<Camera3d>>,
    mut out: ResMut<WorldSkies>,
) {
    let Some(worlds) = worlds else {
        return;
    };
    let mut atmosphere = None;
    let mut exposures = [None; 8];
    for bit in 0..8u8 {
        let (sky, gradient, exposure) = worlds
            .world(bit)
            .and_then(|world| skies.get(world).ok())
            .unwrap_or((None, None, None));
        exposures[bit as usize] = exposure.map(|e| e.ev100);
        let mut sky = sky.cloned().unwrap_or_default();
        if matches!(sky, Sky::Atmosphere) {
            if atmosphere.is_some() {
                warn_once!(
                    "aurora: more than one world has Sky::Atmosphere; the others get Sky::Gradient"
                );
                sky = Sky::Gradient;
            } else {
                atmosphere = Some(bit);
            }
        }
        let entry = &mut out.worlds[bit as usize];
        entry.sky = sky;
        entry.gradient = gradient.cloned().unwrap_or_default();
        entry.suns.clear();
    }
    out.atmosphere = atmosphere;

    let mut brightest = [0.0f32; 8];
    for (light, disk, transform, layers, world, visibility) in &suns {
        if visibility.is_some_and(|v| !v.get()) || light.illuminance <= 0.0 {
            continue;
        }
        let sun = Sun::of(light, disk, transform);
        let mask = world_mask(layers, world);
        for bit in 0..8 {
            let suns = &mut out.worlds[bit].suns;
            if mask & (1 << bit) != 0 && suns.len() < MAX_SUNS {
                suns.push(sun);
                brightest[bit] = brightest[bit].max(light.illuminance);
            }
        }
    }
    for bit in 0..8 {
        out.worlds[bit].ev100 = exposures[bit].unwrap_or(if brightest[bit] > 0.0 {
            ev100_for_lux(brightest[bit])
        } else {
            DEFAULT_EV100
        });
    }

    out.camera_world = cameras
        .iter()
        .filter(|(camera, ..)| camera.is_active)
        .min_by_key(|(camera, ..)| camera.order)
        .map_or(0, |(_, layers, world)| {
            world_mask(layers, world).trailing_zeros().min(7) as u8
        });
}

/// One sun in the GPU table (must match `WorldSun` in types.glsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct WorldSunGpu {
    /// xyz towards the sun, w the cosine of its angular radius.
    pub direction: [f32; 4],
    /// rgb the disc's radiance (nits), a the directly-seen disc's scale.
    pub radiance: [f32; 4],
}

/// One world's entry in the GPU table (must match `WorldEnv` in types.glsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct WorldEnvGpu {
    /// 0 flat colour, 1 HDR, 2 gradient, 3 atmosphere.
    pub mode: u32,
    /// Bindless texture: the HDR sky, or the image behind the atmosphere.
    pub tex: u32,
    pub sun_count: u32,
    pub pad: u32,
    /// Flat: the radiance. HDR: the scale. Atmosphere: the space image's scale (0 = none).
    pub color: [f32; 4],
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub ground: [f32; 4],
    pub suns: [WorldSunGpu; MAX_SUNS],
}

impl WorldEnvGpu {
    /// `tex` and `space` are the renderer's: the HDR's (or space image's) bindless index,
    /// and the space image's scale.
    pub fn new(world: &WorldSky, tex: u32, space: Vec4) -> Self {
        let (mode, color) = match &world.sky {
            Sky::Color { radiance } => (0, radiance.extend(0.0)),
            Sky::Hdr { scale, .. } => (1, Vec4::splat(*scale)),
            Sky::Gradient => (2, Vec4::ZERO),
            Sky::Atmosphere => (3, space),
        };
        let mut suns = [WorldSunGpu::default(); MAX_SUNS];
        for (gpu, sun) in suns.iter_mut().zip(&world.suns) {
            *gpu = WorldSunGpu {
                direction: sun.direction.extend(sun.cos_radius).to_array(),
                radiance: sun.radiance.extend(sun.disc).to_array(),
            };
        }
        Self {
            mode,
            tex,
            sun_count: world.suns.len().min(MAX_SUNS) as u32,
            pad: 0,
            color: color.to_array(),
            zenith: world.gradient.zenith_radiance().extend(0.0).to_array(),
            horizon: world.gradient.horizon_radiance().extend(0.0).to_array(),
            ground: world.gradient.ground_radiance().extend(0.0).to_array(),
            suns,
        }
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
            .register_type::<GradientSky>()
            .register_type::<WorldExposure>()
            .register_type::<DirectionalLight>()
            .register_type::<SunDisk>()
            .init_resource::<WorldSkies>()
            .add_systems(
                Last,
                gather_skies.in_set(crate::ray_render_plugin::RenderSet::Extract),
            );
    }
}
