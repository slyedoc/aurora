//! A water volume's waves as textures (Tessendorf, "Simulating Ocean Water", 2001): two cascades,
//! each a `WAVE_N` square tile of the surface's slope, by an inverse FFT of a Phillips spectrum
//! every frame (water_fft.slang). The first tile spans [`TILE_WAVELENGTHS`] peak wavelengths, the
//! second [`CASCADE_RATIO`] times less (not a simple ratio: the two never repeat together); each
//! carries its own band of wavenumbers, so together they hold the waves from the longest down to a
//! few centimetres. A texel is the slope and its square, so the mip chain's averages give the
//! variance of the slopes a ray's footprint covers (LEAN mapping): water_surface.slang reads both
//! cascades over the footprint and turns that variance into roughness.
//!
//! The initial spectrum h0(k) is generated here when a volume's waves change, scaled so the
//! surface's mean squared slope is `steepness^2 / 2`. The FFT, the copy into the image and its
//! mip chain run in the frame's own commands ([`FrameCommands`]).

use std::f32::consts::TAU;

use ash::vk;
use bevy::{platform::collections::HashMap, prelude::*};

use crate::{
    assets::aurora_asset,
    compute::{
        ComputeModule, FrameCommands, compute_to_compute_barrier, memory_barrier, record_dispatch,
    },
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    reference::ReferenceState,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_texture::{RenderTexture, create_blank_texture_mipped, record_mip_chain},
    vk_init,
    water_volume::{WaterMaterial, WaterMaterial3d, WaterVolume},
};

/// Texels along a cascade's side (water_fft.slang `N`).
pub const WAVE_N: u32 = 256;
const WAVE_MIPS: u32 = 9;
/// The first cascade's tile, in peak wavelengths.
pub const TILE_WAVELENGTHS: f32 = 8.0;
/// The first tile over the second (water_types.slang `WATER_CASCADE_RATIO`).
pub const CASCADE_RATIO: f32 = 6.7;
/// What a wave against the wind keeps of its energy.
const UPWIND: f32 = 0.07;
/// The shortest wave a cascade holds, in its texels: bilinear filtering draws a wave of only a
/// few texels as straight facets, which reflections show as a grid.
const TEXELS_PER_WAVE: f32 = 8.0;

/// The settings h0 is generated from.
#[derive(Clone, Copy, PartialEq)]
struct WaveKey {
    direction: Vec2,
    wavelength: f32,
    steepness: f32,
}

impl WaveKey {
    fn of(volume: &WaterMaterial) -> Option<Self> {
        (volume.steepness > 0.0 && volume.wavelength > 0.0).then(|| Self {
            direction: volume.wave_direction.try_normalize().unwrap_or(Vec2::X),
            wavelength: volume.wavelength,
            steepness: volume.steepness,
        })
    }

    fn tile(&self) -> f32 {
        self.wavelength * TILE_WAVELENGTHS
    }
}

struct WaveSet {
    key: WaveKey,
    h0: [Buffer<[f32; 2]>; 2],
    work: [Buffer<[f32; 2]>; 2],
    texels: [Buffer<[u32; 2]>; 2],
    textures: [RenderTexture; 2],
    index: [u32; 2],
}

impl WaveSet {
    fn new(rd: &RenderDevice, key: WaveKey) -> Self {
        let count = (WAVE_N * WAVE_N) as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let texture = || {
            create_blank_texture_mipped(
                rd,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                WAVE_N,
                WAVE_N,
                WAVE_MIPS,
            )
        };
        let textures = [texture(), texture()];
        let index = [
            rd.register_bindless_texture(&textures[0]),
            rd.register_bindless_texture(&textures[1]),
        ];
        let mut set = Self {
            key,
            h0: [
                rd.create_host_buffer(count, storage),
                rd.create_host_buffer(count, storage),
            ],
            work: [
                rd.create_device_buffer(count, storage),
                rd.create_device_buffer(count, storage),
            ],
            texels: [
                rd.create_device_buffer(count, storage | vk::BufferUsageFlags::TRANSFER_SRC),
                rd.create_device_buffer(count, storage | vk::BufferUsageFlags::TRANSFER_SRC),
            ],
            textures,
            index,
        };
        set.write_spectrum(rd);
        set
    }

    /// New waves: h0 into fresh host buffers (a frame in flight may still read the old ones).
    fn rekey(&mut self, rd: &RenderDevice, key: WaveKey) {
        let count = (WAVE_N * WAVE_N) as u64;
        for h0 in &mut self.h0 {
            let old = std::mem::replace(
                h0,
                rd.create_host_buffer(count, vk::BufferUsageFlags::STORAGE_BUFFER),
            );
            rd.destroyer.destroy_buffer(old.handle);
        }
        self.key = key;
        self.write_spectrum(rd);
    }

    fn write_spectrum(&mut self, rd: &RenderDevice) {
        let spectra = spectrum(&self.key);
        for (buffer, spectrum) in self.h0.iter_mut().zip(spectra) {
            let mut mapped = rd.map_buffer(buffer);
            mapped.copy_from_slice(&spectrum);
        }
    }

    fn release(self, rd: &RenderDevice) {
        for texture in &self.textures {
            rd.unregister_bindless_texture(texture);
            rd.destroyer.destroy_image_view(texture.image_view);
            rd.destroyer.destroy_image(texture.image);
        }
        for handle in self
            .h0
            .iter()
            .chain(&self.work)
            .map(|b| b.handle)
            .chain(self.texels.iter().map(|b| b.handle))
        {
            rd.destroyer.destroy_buffer(handle);
        }
    }
}

/// Each volume's wave textures.
#[derive(Resource)]
pub struct WaterWaves {
    module: Handle<ComputeModule>,
    sets: HashMap<AssetId<WaterMaterial>, WaveSet>,
}

impl WaterWaves {
    /// A volume's two cascade textures (bindless indices) and its first tile's span (metres).
    pub fn textures(&self, material: AssetId<WaterMaterial>) -> Option<([u32; 2], f32)> {
        self.sets.get(&material).map(|s| (s.index, s.key.tile()))
    }
}

/// water_fft.slang `WaveParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveParams {
    h0: u64,
    work: u64,
    texels: u64,
    tile: f32,
    time: f32,
    speed: f32,
    axis: u32,
}

/// A small deterministic generator for the spectrum's Gaussian draws (the same waves every run).
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Two independent standard normals (Box-Muller).
    fn gaussian2(&mut self) -> (f32, f32) {
        let r = (-2.0 * self.next().ln()).sqrt();
        let a = TAU * self.next();
        (r * a.cos(), r * a.sin())
    }
}

/// The signed frequency of index `i`.
fn freq(i: u32) -> f32 {
    if i < WAVE_N / 2 {
        i as f32
    } else {
        i as f32 - WAVE_N as f32
    }
}

/// Both cascades' h0: a Phillips spectrum peaking at `wavelength` along `direction`, each
/// cascade's waves no shorter than [`TEXELS_PER_WAVE`] of its texels (the first's band ends where
/// the second's starts), scaled so the mean squared slope is `steepness^2 / 2`.
fn spectrum(key: &WaveKey) -> [Vec<[f32; 2]>; 2] {
    // Phillips peaks at k = 1 / (sqrt 2 L).
    let big_l = key.wavelength / (TAU * std::f32::consts::SQRT_2);
    let tile0 = key.tile();
    let limit = |tile: f32| TAU * WAVE_N as f32 / (TEXELS_PER_WAVE * tile);
    let cut = limit(tile0);
    let mut out = [Vec::new(), Vec::new()];
    let mut slope2 = 0.0f64;
    for (c, spectrum) in out.iter_mut().enumerate() {
        let tile = tile0 / CASCADE_RATIO.powi(c as i32);
        let dk = TAU / tile;
        let mut rng = SplitMix(0x5EED_0000 + c as u64);
        spectrum.reserve((WAVE_N * WAVE_N) as usize);
        for y in 0..WAVE_N {
            for x in 0..WAVE_N {
                let k = Vec2::new(freq(x), freq(y)) * dk;
                let kl = k.length();
                let (g1, g2) = rng.gaussian2();
                let in_band = if c == 0 {
                    kl <= cut
                } else {
                    kl > cut && kl <= limit(tile)
                };
                if kl <= 0.0 || !in_band {
                    spectrum.push([0.0, 0.0]);
                    continue;
                }
                let along = (k / kl).dot(key.direction);
                let mut p = (-1.0 / (kl * big_l).powi(2)).exp() / kl.powi(4) * along * along;
                if along < 0.0 {
                    p *= UPWIND;
                }
                let amp = (p * dk * dk * 0.5).sqrt();
                let h = [g1 * amp, g2 * amp];
                slope2 += 2.0 * (kl * kl) as f64 * (h[0] * h[0] + h[1] * h[1]) as f64;
                spectrum.push(h);
            }
        }
    }
    let target = 0.5 * (key.steepness as f64).powi(2);
    let scale = if slope2 > 0.0 {
        (target / slope2).sqrt() as f32
    } else {
        0.0
    };
    for spectrum in &mut out {
        for h in spectrum.iter_mut() {
            h[0] *= scale;
            h[1] *= scale;
        }
    }
    out
}

/// `RenderSet::Prepare`, before the table upload: a wave set per volume with waves, h0 kept in
/// step with its settings, and this frame's FFT queued.
fn prepare_water_waves(
    rd: Res<RenderDevice>,
    time: Res<Time>,
    reference: Res<ReferenceState>,
    volumes: Query<(&WaterMaterial3d, &InheritedVisibility), With<WaterVolume>>,
    materials: Res<Assets<WaterMaterial>>,
    mut waves: ResMut<WaterWaves>,
    mut commands: ResMut<FrameCommands>,
) {
    let waves = &mut *waves;
    // One wave set per material in use, however many volumes share it.
    let mut live: Vec<(AssetId<WaterMaterial>, WaveKey, f32)> = Vec::new();
    for (handle, visible) in &volumes {
        let id = handle.id();
        if !visible.get() || live.iter().any(|(l, _, _)| *l == id) {
            continue;
        }
        if let Some(material) = materials.get(id)
            && let Some(key) = WaveKey::of(material)
        {
            live.push((id, key, material.wave_speed));
        }
    }
    let stale: Vec<AssetId<WaterMaterial>> = waves
        .sets
        .keys()
        .filter(|e| !live.iter().any(|(l, _, _)| l == *e))
        .copied()
        .collect();
    for entity in stale {
        if let Some(set) = waves.sets.remove(&entity) {
            set.release(&rd);
        }
    }
    // The shader clock, as the frame uniform has it (a capture pins it).
    let now = if reference.capture {
        0.0
    } else {
        time.elapsed_secs_wrapped()
    };
    for (entity, key, speed) in live {
        let set = match waves.sets.get_mut(&entity) {
            Some(set) => {
                if set.key != key {
                    set.rekey(&rd, key);
                }
                set
            }
            None => waves
                .sets
                .entry(entity)
                .or_insert_with(|| WaveSet::new(&rd, key)),
        };
        let tile0 = key.tile();
        let params: [WaveParams; 2] = std::array::from_fn(|c| WaveParams {
            h0: set.h0[c].address,
            work: set.work[c].address,
            texels: set.texels[c].address,
            tile: tile0 / CASCADE_RATIO.powi(c as i32),
            time: now,
            speed,
            axis: 0,
        });
        let texels = [set.texels[0].handle, set.texels[1].handle];
        let images = [set.textures[0].image, set.textures[1].image];
        let module = waves.module.clone();
        commands.push(move |rd, modules, cmd| {
            let Some(module) = modules.get(&module) else {
                return;
            };
            let texels_n = WAVE_N * WAVE_N;
            for p in &params {
                record_dispatch(rd, cmd, module, "wave_spectrum", p, texels_n, None);
            }
            compute_to_compute_barrier(rd, cmd);
            for axis in 0..2 {
                for p in &params {
                    let p = WaveParams { axis, ..*p };
                    // One workgroup (64 threads) per line.
                    record_dispatch(rd, cmd, module, "wave_fft", &p, WAVE_N * 64, None);
                }
                compute_to_compute_barrier(rd, cmd);
            }
            for p in &params {
                record_dispatch(rd, cmd, module, "wave_texels", p, texels_n, None);
            }
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            );
            for (buffer, image) in texels.into_iter().zip(images) {
                copy_into(rd, cmd, buffer, image);
            }
        });
    }
}

/// The texels into level 0 (every level out of shader reads first: last frame's trace may still
/// be reading them), then the chain blitted down and back to shader reads.
fn copy_into(rd: &RenderDevice, cmd: vk::CommandBuffer, buffer: vk::Buffer, image: vk::Image) {
    let to_dst = vk_init::layout_transition2(
        image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )
    .subresource_range(
        vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(WAVE_MIPS)
            .layer_count(1),
    )
    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
    .src_access_mask(vk::AccessFlags2::SHADER_READ)
    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    unsafe {
        rd.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_dst)),
        );
        rd.device.cmd_copy_buffer_to_image(
            cmd,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk_init::buffer_image_copy(WAVE_N, WAVE_N)],
        );
    }
    record_mip_chain(
        rd,
        cmd,
        image,
        WAVE_N,
        WAVE_N,
        WAVE_MIPS,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut waves: Mut<WaterWaves>| {
        let rd = world.resource::<RenderDevice>();
        for (_, set) in waves.sets.drain() {
            set.release(rd);
        }
    });
}

pub struct WaterWavesPlugin;

impl Plugin for WaterWavesPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/water_fft.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &["wave_spectrum", "wave_fft", "wave_texels"],
        ));
        app.insert_resource(WaterWaves {
            module,
            sets: HashMap::default(),
        })
        .add_systems(
            Last,
            prepare_water_waves
                .in_set(RenderSet::Prepare)
                .before(crate::water_volume::upload_water_volumes),
        )
        .add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spectrum_has_the_asked_for_slope() {
        let key = WaveKey {
            direction: Vec2::X,
            wavelength: 2.0,
            steepness: 0.3,
        };
        let spectra = spectrum(&key);
        let tile0 = key.tile();
        let mut slope2 = 0.0f64;
        for (c, spectrum) in spectra.iter().enumerate() {
            let dk = TAU / (tile0 / CASCADE_RATIO.powi(c as i32));
            for (i, h) in spectrum.iter().enumerate() {
                let k = Vec2::new(freq(i as u32 % WAVE_N), freq(i as u32 / WAVE_N)) * dk;
                slope2 += 2.0 * k.length_squared() as f64 * (h[0] * h[0] + h[1] * h[1]) as f64;
            }
        }
        assert!((slope2 - 0.5 * 0.09).abs() < 1.0e-4, "{slope2}");
    }

    #[test]
    fn the_cascades_share_no_wavenumber() {
        let key = WaveKey {
            direction: Vec2::X,
            wavelength: 2.0,
            steepness: 0.3,
        };
        let [near, far] = spectrum(&key);
        let cut = TAU * WAVE_N as f32 / (TEXELS_PER_WAVE * key.tile());
        let dk1 = TAU / (key.tile() / CASCADE_RATIO);
        for (i, h) in far.iter().enumerate() {
            let k = Vec2::new(freq(i as u32 % WAVE_N), freq(i as u32 / WAVE_N)) * dk1;
            if h[0] != 0.0 || h[1] != 0.0 {
                assert!(k.length() > cut);
            }
        }
        assert!(near.iter().any(|h| h[0] != 0.0));
    }
}
