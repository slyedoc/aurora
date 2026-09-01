//! Environment importance sampling for [`Sky::Hdr`].
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
//! row-major, `N = ENV_W * ENV_H`. Rebuilt whenever the sky's image changes; 0 (no
//! sampling) for any other sky.

use ash::vk;
use bevy::prelude::*;

use crate::{
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sky::Sky,
};

pub const ENV_W: u32 = 512;
pub const ENV_H: u32 = 256;

#[derive(Resource, Default)]
pub struct EnvLight {
    image: Option<AssetId<Image>>,
    buffer: Buffer<f32>,
}

impl EnvLight {
    /// Device address of the CDF + pdf buffer for the active HDR sky, 0 when there is none.
    pub fn address(&self) -> u64 {
        self.buffer.address
    }

    fn release(&mut self, rd: &RenderDevice) {
        if self.buffer.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(self.buffer.handle);
        }
        self.buffer = Buffer::default();
        self.image = None;
    }
}

/// Per-texel luminance of the source image, averaged into the sampling grid. Handles the
/// two layouts the texture upload path produces (RGBA32F, RGBA8).
fn luminance_grid(image: &Image) -> Option<Vec<f32>> {
    let data = image.data.as_ref()?;
    let (w, h) = (
        image.texture_descriptor.size.width as usize,
        image.texture_descriptor.size.height as usize,
    );
    if w == 0 || h == 0 {
        return None;
    }
    let bpp = data.len() / (w * h);
    let luma = |px: usize| -> f32 {
        match bpp {
            16 => {
                let f: &[f32] = bytemuck::cast_slice(&data[px * 16..px * 16 + 12]);
                0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]
            }
            4 => {
                let p = &data[px * 4..px * 4 + 3];
                (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0
            }
            _ => 0.0,
        }
    };
    if bpp != 16 && bpp != 4 {
        return None;
    }
    let (gw, gh) = (ENV_W as usize, ENV_H as usize);
    let mut sum = vec![0.0f32; gw * gh];
    let mut count = vec![0u32; gw * gh];
    for y in 0..h {
        let gy = (y * gh / h).min(gh - 1);
        for x in 0..w {
            let gx = (x * gw / w).min(gw - 1);
            let l = luma(y * w + x);
            if l.is_finite() {
                sum[gy * gw + gx] += l;
                count[gy * gw + gx] += 1;
            }
        }
    }
    Some(
        sum.iter()
            .zip(&count)
            .map(|(s, c)| if *c > 0 { s / *c as f32 } else { 0.0 })
            .collect(),
    )
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

fn prepare_env_light(
    render_device: Res<RenderDevice>,
    sky: Res<Sky>,
    images: Res<Assets<Image>>,
    mut env: ResMut<EnvLight>,
) {
    let Sky::Hdr { image, .. } = &*sky else {
        if env.image.is_some() {
            env.release(&render_device);
        }
        return;
    };
    let id = image.id();
    if env.image == Some(id) {
        return;
    }
    let Some(image) = images.get(id) else { return };
    let Some(lum) = luminance_grid(image) else {
        log::warn!(
            "env light: sky image has no CPU data / unsupported format; no importance sampling"
        );
        env.release(&render_device);
        env.image = Some(id);
        return;
    };
    let table = build_distribution(&lum);
    env.release(&render_device);
    let mut host: Buffer<f32> =
        render_device.create_host_buffer(table.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
    render_device.map_buffer(&mut host).copy_from_slice(&table);
    let device: Buffer<f32> = render_device.create_device_buffer(
        table.len() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
    );
    render_device.run_transfer_commands(|cmd| render_device.upload_buffer(cmd, &host, &device));
    render_device.destroyer.destroy_buffer(host.handle);
    env.buffer = device;
    env.image = Some(id);
    log::info!("env light: importance-sampling table built ({ENV_W}x{ENV_H})");
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut env: Mut<EnvLight>| {
        let rd = world.resource::<RenderDevice>();
        env.release(rd);
    });
}

pub struct EnvLightPlugin;

impl Plugin for EnvLightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<EnvLight>();
        app.add_systems(Last, prepare_env_light.in_set(RenderSet::Prepare));
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
