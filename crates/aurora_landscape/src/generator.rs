//! Generator layers: procedural heights everywhere (`landscape/generator.slang`, zero's
//! flat-world terrain function). The base of an infinite landscape: a streaming window of
//! pages evaluates the generator as it slides, and everything edited sits on top.

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

use crate::HeightLayer;

/// Rolling fbm hills, ridged mountain ranges in regions picked by a low-frequency mask, and
/// fine detail; metres, frequencies per metre.
#[derive(Component, Reflect, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
#[require(HeightLayer)]
pub struct GeneratorLayer {
    pub seed: f32,
    pub base_height: f32,
    pub hills_frequency: f32,
    pub hills_amplitude: f32,
    pub hills_octaves: u32,
    pub mountains_frequency: f32,
    pub mountains_amplitude: f32,
    pub mountains_octaves: u32,
    pub mountains_warp: f32,
    pub mountains_mask_frequency: f32,
    /// > 1 shrinks the mountain regions.
    pub mountains_mask_power: f32,
    pub detail_frequency: f32,
    pub detail_amplitude: f32,
    pub detail_octaves: u32,
}

impl Default for GeneratorLayer {
    fn default() -> Self {
        Self {
            seed: 3.0,
            base_height: 60.0,
            hills_frequency: 1.0 / 1_200.0,
            hills_amplitude: 45.0,
            hills_octaves: 6,
            mountains_frequency: 1.0 / 5_000.0,
            mountains_amplitude: 600.0,
            mountains_octaves: 9,
            mountains_warp: 0.35,
            mountains_mask_frequency: 1.0 / 20_000.0,
            mountains_mask_power: 2.0,
            detail_frequency: 1.0 / 40.0,
            detail_amplitude: 1.2,
            detail_octaves: 3,
        }
    }
}

/// `Genome` in generator.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GeneratorGpu {
    seed: f32,
    base_height: f32,
    hills_frequency: f32,
    hills_amplitude: f32,
    hills_octaves: u32,
    mountains_frequency: f32,
    mountains_amplitude: f32,
    mountains_octaves: u32,
    mountains_warp: f32,
    mountains_mask_frequency: f32,
    mountains_mask_power: f32,
    detail_frequency: f32,
    detail_amplitude: f32,
    detail_octaves: u32,
    pad: [u32; 2],
}

/// `GeneratorPush` in generator.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(crate) struct GeneratorPush {
    pub page: u64,
    pub genome: u64,
    pub page_x: i32,
    pub page_z: i32,
    pub texel: f32,
    pub opacity: f32,
    pub blend: u32,
    /// Planet faces: noise at the sphere point (radius 0 = flat).
    pub face: u32,
    pub radius: f32,
    pub pad: u32,
}

/// A generator layer's parameters on the GPU.
#[derive(Component)]
pub struct GeneratorSource(Buffer<GeneratorGpu>);

impl GeneratorSource {
    pub fn address(&self) -> u64 {
        self.0.address
    }
}

pub fn upload_genomes(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    mut layers: Query<
        (Entity, &GeneratorLayer, Option<&mut GeneratorSource>),
        Changed<GeneratorLayer>,
    >,
) {
    for (entity, g, source) in &mut layers {
        let gpu = GeneratorGpu {
            seed: g.seed,
            base_height: g.base_height,
            hills_frequency: g.hills_frequency,
            hills_amplitude: g.hills_amplitude,
            hills_octaves: g.hills_octaves,
            mountains_frequency: g.mountains_frequency,
            mountains_amplitude: g.mountains_amplitude,
            mountains_octaves: g.mountains_octaves,
            mountains_warp: g.mountains_warp,
            mountains_mask_frequency: g.mountains_mask_frequency,
            mountains_mask_power: g.mountains_mask_power,
            detail_frequency: g.detail_frequency,
            detail_amplitude: g.detail_amplitude,
            detail_octaves: g.detail_octaves,
            pad: [0; 2],
        };
        match source {
            Some(mut source) => rd.map_buffer(&mut source.0)[0] = gpu,
            None => {
                let mut buffer: Buffer<GeneratorGpu> =
                    rd.create_host_buffer(1, vk::BufferUsageFlags::STORAGE_BUFFER);
                rd.map_buffer(&mut buffer)[0] = gpu;
                commands.entity(entity).insert(GeneratorSource(buffer));
            }
        }
    }
}

pub fn on_genome_removed(
    remove: On<Remove<GeneratorSource>>,
    sources: Query<&GeneratorSource>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(source) = sources.get(remove.entity)
        && source.0.handle != vk::Buffer::null()
    {
        rd.destroyer.destroy_buffer(source.0.handle);
    }
}

pub fn release_genomes(mut sources: Query<&mut GeneratorSource>, rd: Res<RenderDevice>) {
    for mut source in &mut sources {
        if source.0.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(source.0.handle);
            source.0 = Buffer::default();
        }
    }
}
