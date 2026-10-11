//! Importance sampling for every environment's [`Sky::Hdr`]: one table per environment.
//!
//! An equirectangular sky lights the scene through the miss shader, but a BRDF-sampled bounce
//! rarely lands on the small, bright parts of the map (the sun), so an HDR sky alone gives
//! flat, dim direct light. Like the procedural sun, the raygen takes one next-event
//! estimation ray per surface hit towards the environment: this module builds the sampling
//! distribution it draws from -- a luminance × sin θ CDF over a [`ENV_W`]×[`ENV_H`]
//! downsample of the image, plus each texel's solid-angle pdf so the estimate can be
//! MIS-weighted against BRDF sampling (and BRDF rays that reach the sky against it).
//!
//! Device buffer (f32): `cdf[N + 1]` (`cdf[0] = 0`, `cdf[N] = 1`), then `pdf_solid[N]`,
//! row-major, `N = ENV_W * ENV_H`. A sky with a sun has it taken out first ([`HdrSun`]): the
//! table and the shaders see the sky clipped inside the sun's cone, and the sun renders as an
//! analytic sun of its environment. Each environment's entry in the GPU table
//! (`EnvironmentGpu::importance`) holds its address; rebuilt whenever that sky's image
//! changes, 0 (no sampling) for any other sky.

use ash::vk;
use bevy::prelude::*;

use crate::{
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sky::{EnvironmentSkies, Sky},
};

pub const ENV_W: u32 = 512;
pub const ENV_H: u32 = 256;

/// One HDR sky's sampling table.
#[derive(Default)]
struct Table {
    image: Option<AssetId<Image>>,
    buffer: Buffer<f32>,
    sun: Option<HdrSun>,
}

impl Table {
    fn release(&mut self, rd: &RenderDevice) {
        if self.buffer.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(self.buffer.handle);
        }
        self.buffer = Buffer::default();
        self.image = None;
        self.sun = None;
    }
}

/// The sampling table of each environment's HDR sky, by environment bit.
#[derive(Resource, Default)]
pub struct SkyImportance {
    tables: [Table; 8],
}

impl SkyImportance {
    /// Device address of environment `bit`'s CDF + pdf buffer, 0 when it has none.
    pub fn address(&self, bit: usize) -> u64 {
        self.tables[bit].buffer.address
    }

    /// The sun taken out of environment `bit`'s HDR sky, if it had one.
    pub fn sun(&self, bit: usize) -> Option<&HdrSun> {
        self.tables[bit].sun.as_ref()
    }
}

/// The sun of an HDR sky, taken out of it: rendered as analytic suns of its environment (sharp
/// shadows, one shadow ray, the sun's own next-event paths: fog shafts, translucency), while the
/// sky keeps only what is left inside its cone (`clip`). Two discs: the core, and the bright halo
/// around it (clouds lit by the sun), which left in the sky projects through every gap in the
/// foliage as a blocky image of itself. In the image's units (times the sky's scale and
/// brightness on the GPU).
#[derive(Clone, Copy, Debug)]
pub struct HdrSun {
    /// Unit vector towards the sun.
    pub direction: Vec3,
    /// Cosine of the cone the sun's texels fill (the halo's when it has one).
    pub cos_radius: f32,
    /// Luminance the sky is clipped to inside the cone.
    pub clip: f32,
    pub core: SunDisc,
    pub halo: Option<SunDisc>,
}

/// One analytic disc around [`HdrSun::direction`]: its energy above the clip spread over its cone.
#[derive(Clone, Copy, Debug)]
pub struct SunDisc {
    pub cos_radius: f32,
    pub radiance: Vec3,
}

impl HdrSun {
    pub fn discs(&self) -> impl Iterator<Item = SunDisc> {
        std::iter::once(self.core).chain(self.halo)
    }
}

/// The direction of equirect texel (x, y) (`env_uv_to_dir` in the shaders) and its solid angle.
fn texel_dir(x: usize, y: usize, w: usize, h: usize) -> (Vec3, f32) {
    let u = (x as f32 + 0.5) / w as f32;
    let v = (y as f32 + 0.5) / h as f32;
    let phi = (u + 0.5) * std::f32::consts::TAU;
    let theta = v * std::f32::consts::PI;
    let st = theta.sin();
    let dir = Vec3::new(st * phi.sin(), theta.cos(), st * phi.cos());
    let solid = (std::f32::consts::TAU / w as f32) * (std::f32::consts::PI / h as f32) * st;
    (dir, solid)
}

/// A sun is a peak at least this many times the sky's mean radiance; the texels above the
/// geometric mean of peak and sky are its core, the ones above `SUN_HALO_OVER_SKY` x the mean
/// around it its halo, within this many degrees of the peak.
const SUN_PEAK_OVER_SKY: f32 = 200.0;
const SUN_HALO_OVER_SKY: f32 = 50.0;
const SUN_MAX_DEGREES: f32 = 8.0;

/// The image's linear RGB per texel (the two layouts the texture upload path produces).
fn texels(image: &Image) -> Option<(Vec<Vec3>, usize, usize)> {
    let data = image.data.as_ref()?;
    let (w, h) = (
        image.texture_descriptor.size.width as usize,
        image.texture_descriptor.size.height as usize,
    );
    if w == 0 || h == 0 {
        return None;
    }
    let bpp = data.len() / (w * h);
    let rgb: Vec<Vec3> = match bpp {
        16 => (0..w * h)
            .map(|px| {
                let f: &[f32] = bytemuck::cast_slice(&data[px * 16..px * 16 + 12]);
                Vec3::new(f[0], f[1], f[2])
            })
            .collect(),
        4 => (0..w * h)
            .map(|px| {
                let p = &data[px * 4..px * 4 + 3];
                Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 255.0
            })
            .collect(),
        _ => return None,
    };
    Some((rgb, w, h))
}

fn luma3(c: Vec3) -> f32 {
    let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    if l.is_finite() { l.max(0.0) } else { 0.0 }
}

/// Find the sun in the sky, if it has one (an overcast or night sky has no peak to take).
fn find_sun(rgb: &[Vec3], w: usize, h: usize) -> Option<HdrSun> {
    let (mut sum, mut area, mut peak, mut peak_at) = (0.0f64, 0.0f64, 0.0f32, 0usize);
    for y in 0..h {
        let (_, solid) = texel_dir(0, y, w, h);
        for x in 0..w {
            let l = luma3(rgb[y * w + x]);
            sum += (l * solid) as f64;
            area += solid as f64;
            if l > peak {
                peak = l;
                peak_at = y * w + x;
            }
        }
    }
    let mean = (sum / area.max(1e-9)) as f32;
    if mean <= 0.0 || peak < SUN_PEAK_OVER_SKY * mean {
        return None;
    }
    let core_clip = (peak * mean).sqrt();
    let clip = core_clip.min(SUN_HALO_OVER_SKY * mean);
    let (centre, _) = texel_dir(peak_at % w, peak_at / w, w, h);
    let cos_max = SUN_MAX_DEGREES.to_radians().cos();
    let mut weighted = Vec3::ZERO;
    let mut taken = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let c = rgb[y * w + x];
            let l = luma3(c);
            if l <= clip {
                continue;
            }
            let (dir, solid) = texel_dir(x, y, w, h);
            if dir.dot(centre) < cos_max {
                continue;
            }
            if l > core_clip {
                weighted += dir * l * solid;
            }
            taken.push((dir, c * (1.0 - clip / l) * solid, l > core_clip));
        }
    }
    let direction = weighted.normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    // Each cone covers every texel it takes (a hair wider, so the clip in the shaders catches
    // the texel edges too); the core takes everything inside its cone, the halo the rest.
    let cone = |cos: f32| cos.min(0.99999) - 1.0e-5;
    let core_cos = cone(
        taken
            .iter()
            .filter(|t| t.2)
            .map(|t| t.0.dot(direction))
            .fold(1.0f32, f32::min),
    );
    let (mut core_energy, mut halo_energy, mut halo_cos) = (Vec3::ZERO, Vec3::ZERO, 1.0f32);
    for (dir, energy, _) in &taken {
        let cos = dir.dot(direction);
        if cos >= core_cos {
            core_energy += *energy;
        } else {
            halo_energy += *energy;
            halo_cos = halo_cos.min(cos);
        }
    }
    let solid = |cos: f32| std::f32::consts::TAU * (1.0 - cos);
    let core = SunDisc {
        cos_radius: core_cos,
        radiance: core_energy / solid(core_cos),
    };
    let halo = (halo_energy != Vec3::ZERO).then(|| {
        let cos_radius = cone(halo_cos);
        SunDisc {
            cos_radius,
            radiance: halo_energy / solid(cos_radius),
        }
    });
    Some(HdrSun {
        direction,
        cos_radius: halo.map_or(core.cos_radius, |h| h.cos_radius),
        clip,
        core,
        halo,
    })
}

/// The sampling grid: per-texel luminance with the sun clipped out, averaged into ENV_W x ENV_H.
fn luminance_grid(rgb: &[Vec3], w: usize, h: usize, sun: Option<&HdrSun>) -> Vec<f32> {
    let (gw, gh) = (ENV_W as usize, ENV_H as usize);
    let mut sum = vec![0.0f32; gw * gh];
    let mut count = vec![0u32; gw * gh];
    for y in 0..h {
        let gy = (y * gh / h).min(gh - 1);
        for x in 0..w {
            let gx = (x * gw / w).min(gw - 1);
            let mut l = luma3(rgb[y * w + x]);
            if let Some(sun) = sun
                && l > sun.clip
                && texel_dir(x, y, w, h).0.dot(sun.direction) >= sun.cos_radius
            {
                l = sun.clip;
            }
            sum[gy * gw + gx] += l;
            count[gy * gw + gx] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(s, c)| if *c > 0 { s / *c as f32 } else { 0.0 })
        .collect()
}

/// `cdf[N+1]` then `pdf_solid[N]` for the grid.
fn build_distribution(lum: &[f32]) -> Vec<f32> {
    let (gw, gh) = (ENV_W as usize, ENV_H as usize);
    let n = gw * gh;
    let mut weight = vec![0.0f32; n];
    for y in 0..gh {
        let theta = (y as f32 + 0.5) / gh as f32 * std::f32::consts::PI;
        let sin_theta = theta.sin();
        for x in 0..gw {
            // A floor keeps every direction reachable (unbiased even where the map is black).
            weight[y * gw + x] = (lum[y * gw + x] + 1.0e-4) * sin_theta;
        }
    }
    let total: f64 = weight.iter().map(|w| *w as f64).sum();
    let mut out = Vec::with_capacity(2 * n + 1);
    out.push(0.0);
    let mut acc = 0.0f64;
    for w in &weight {
        acc += *w as f64 / total;
        out.push(acc as f32);
    }
    out[n] = 1.0;
    // Texel solid angle: (2π / W)(π / H) sin θ.
    let texel = 2.0 * std::f32::consts::PI * std::f32::consts::PI / (gw * gh) as f32;
    for y in 0..gh {
        let sin_theta = ((y as f32 + 0.5) / gh as f32 * std::f32::consts::PI).sin();
        for x in 0..gw {
            let p = (weight[y * gw + x] as f64 / total) as f32;
            out.push(p / (texel * sin_theta.max(1.0e-4)));
        }
    }
    out
}

fn build_table(rd: &RenderDevice, image: &Image) -> Option<(Buffer<f32>, Option<HdrSun>)> {
    let (rgb, w, h) = texels(image)?;
    let sun = find_sun(&rgb, w, h);
    let table = build_distribution(&luminance_grid(&rgb, w, h, sun.as_ref()));
    let mut host: Buffer<f32> =
        rd.create_host_buffer(table.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
    rd.map_buffer(&mut host).copy_from_slice(&table);
    let device: Buffer<f32> = rd.create_device_buffer(
        table.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    );
    rd.run_transfer_commands(|cmd| rd.upload_buffer(cmd, &host, &device));
    rd.destroyer.destroy_buffer(host.handle);
    Some((device, sun))
}

fn prepare_sky_importance(
    render_device: Res<RenderDevice>,
    skies: Res<EnvironmentSkies>,
    images: Res<Assets<Image>>,
    mut importance: ResMut<SkyImportance>,
) {
    for (bit, environment) in skies.environments.iter().enumerate() {
        let table = &mut importance.tables[bit];
        let Sky::Hdr { image, .. } = &environment.sky else {
            if table.image.is_some() {
                table.release(&render_device);
            }
            continue;
        };
        let id = image.id();
        if table.image == Some(id) {
            continue;
        }
        let Some(image) = images.get(id) else {
            continue;
        };
        table.release(&render_device);
        table.image = Some(id);
        match build_table(&render_device, image) {
            Some((buffer, sun)) => {
                table.buffer = buffer;
                table.sun = sun;
                match sun {
                    Some(sun) => log::info!(
                        "sky importance: environment {bit}'s table built ({ENV_W}x{ENV_H}); sun \
                         taken out at {:.3} ({:.2} deg radius, halo {:.2} deg)",
                        sun.direction,
                        sun.core.cos_radius.acos().to_degrees(),
                        sun.halo.map_or(0.0, |h| h.cos_radius.acos().to_degrees())
                    ),
                    None => log::info!(
                        "sky importance: environment {bit}'s table built ({ENV_W}x{ENV_H}); no sun"
                    ),
                }
            }
            None => log::warn!(
                "sky importance: environment {bit}'s sky image has no CPU data / unsupported \
                 format; no importance sampling"
            ),
        }
    }
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut importance: Mut<SkyImportance>| {
        let rd = world.resource::<RenderDevice>();
        for table in &mut importance.tables {
            table.release(rd);
        }
    });
}

pub struct SkyImportancePlugin;

impl Plugin for SkyImportancePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyImportance>();
        app.add_systems(Last, prepare_sky_importance.in_set(RenderSet::Prepare));
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat sky of 1.0 with a sun of `sun` radiance inside `degrees` of `towards`.
    fn sky(w: usize, h: usize, towards: Vec3, degrees: f32, sun: f32) -> Vec<Vec3> {
        let cos = degrees.to_radians().cos();
        let mut rgb = vec![Vec3::ONE; w * h];
        for y in 0..h {
            for x in 0..w {
                if texel_dir(x, y, w, h).0.dot(towards) >= cos {
                    rgb[y * w + x] = Vec3::splat(sun);
                }
            }
        }
        rgb
    }

    #[test]
    fn the_sun_comes_out_where_it_is_and_its_energy_is_kept() {
        let (w, h) = (1024, 512);
        let towards = Vec3::new(0.4, 0.7, 0.55).normalize();
        let rgb = sky(w, h, towards, 2.0, 1.0e5);
        let sun = find_sun(&rgb, w, h).expect("a sun");
        assert!(sun.direction.dot(towards) > 0.9999, "{:?}", sun.direction);
        // What the sky loses inside the cone is what the sun carries.
        let (mut before, mut after) = (0.0f64, 0.0f64);
        for y in 0..h {
            for x in 0..w {
                let (dir, solid) = texel_dir(x, y, w, h);
                let l = luma3(rgb[y * w + x]);
                before += (l * solid) as f64;
                let kept = if l > sun.clip && dir.dot(sun.direction) >= sun.cos_radius {
                    sun.clip
                } else {
                    l
                };
                after += (kept * solid) as f64;
            }
        }
        let carried: f64 = sun
            .discs()
            .map(|d| luma3(d.radiance) as f64 * std::f64::consts::TAU * (1.0 - d.cos_radius as f64))
            .sum();
        assert!(
            ((after + carried) / before - 1.0).abs() < 1.0e-3,
            "{after} + {carried} vs {before}"
        );
    }

    #[test]
    fn the_glow_around_the_sun_is_a_halo_not_a_wider_sun() {
        let (w, h) = (1024, 512);
        let towards = Vec3::new(0.4, 0.7, 0.55).normalize();
        let mut rgb = sky(w, h, towards, 3.0, 200.0);
        let core = sky(w, h, towards, 0.5, 1.0e5);
        for (px, c) in rgb.iter_mut().zip(&core) {
            *px = px.max(*c);
        }
        let sun = find_sun(&rgb, w, h).expect("a sun");
        let degrees = |cos: f32| cos.acos().to_degrees();
        assert!(degrees(sun.core.cos_radius) < 0.8, "{sun:?}");
        let halo = sun.halo.expect("a halo");
        assert!((degrees(halo.cos_radius) - 3.0).abs() < 0.5, "{sun:?}");
        assert!(
            luma3(halo.radiance) < 1.0e-2 * luma3(sun.core.radiance),
            "{sun:?}"
        );
        assert!(sun.clip <= SUN_HALO_OVER_SKY * 200.0);
    }

    #[test]
    fn an_overcast_sky_has_no_sun() {
        let (w, h) = (256, 128);
        let rgb = sky(w, h, Vec3::Y, 10.0, 20.0);
        assert!(find_sun(&rgb, w, h).is_none());
    }
}
