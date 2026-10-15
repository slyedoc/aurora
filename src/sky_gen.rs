//! Every sky the raygen importance-samples, built on the GPU: [`Sky::Generated`], rendered from
//! the environment's [`SkyAir`] and its brightest sun into an equirectangular image the renderer
//! shades like an HDR photo, and [`Sky::Hdr`], the photo itself box-filtered down to the grid.
//!
//! Every kernel is atmosphere.slang's (Hillaire 2020): the transmittance and multiple-scattering
//! tables, rebuilt when the air changes; `sky_render` over the image's texels, then the
//! importance table built from them (`sky_rows` / `sky_offsets` / `sky_normalize` / `sky_guide`,
//! rt_types.slang `importance`), whenever the sun or the air moved; and `sun_tint` every frame, which
//! writes the sun's radiance through the air into the environment table the CPU just filled,
//! so the light and the sky agree. All of it is recorded into the frame's command buffer
//! ([`FrameCommands`]); nothing waits or reads back.

use ash::vk;
use bevy::prelude::*;

use crate::{
    assets::aurora_asset,
    atmosphere::{AtmosphereGpu, KernelParams},
    compute::{
        CompiledComputeModule, ComputeModule, ComputeModules, FrameCommands, memory_barrier,
        record_dispatch,
    },
    ray_render_plugin::{Frame, RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_texture::{RenderTexture, create_blank_texture},
    sky::{EnvironmentGpu, EnvironmentSkies, EnvironmentSunGpu, Sky, SkyAir},
    vk_init,
    vulkan_asset::VulkanAssets,
};

/// Earth: the planet's radius and where its air ends (metres).
const PLANET_RADIUS: f32 = 6_360_000.0;
const AIR_HEIGHT: f32 = 100_000.0;
/// atmosphere.slang's table sizes.
const T_W: u32 = 256;
const T_H: u32 = 64;
const MS_N: u32 = 32;
/// The importance grid, and the generated image: atmosphere.slang's SKY_W / SKY_H,
/// rt_types.slang's SKY_IMPORTANCE_W / _H.
pub const SKY_W: u32 = 512;
pub const SKY_H: u32 = 256;

/// What the scattering tables depend on.
#[derive(Clone, Copy, PartialEq)]
struct AirKey([f32; 15]);

impl AirKey {
    fn of(air: &SkyAir) -> Self {
        let g = air.ground_albedo.to_linear();
        Self([
            air.rayleigh_scatter.x,
            air.rayleigh_scatter.y,
            air.rayleigh_scatter.z,
            air.rayleigh_height,
            air.mie_scatter,
            air.mie_absorb,
            air.mie_height,
            air.mie_g,
            air.ozone_absorb.x,
            air.ozone_absorb.y,
            air.ozone_absorb.z,
            g.red,
            g.green,
            g.blue,
            air.altitude,
        ])
    }
}

/// What the image depends on beyond the air: the sun's direction, disc and brightness.
#[derive(Clone, Copy, PartialEq)]
struct SunKey(Vec3, f32, f32);

/// The night's part of the image: the moon's direction and illuminance (rgb lux), the airglow
/// floor (rgb nits).
#[derive(Clone, Copy, PartialEq, Default)]
struct NightKey {
    moon_dir: [f32; 3],
    moon_lux: [f32; 3],
    airglow: [f32; 3],
}

impl NightKey {
    /// The moon is the environment's second sun, while it is up (zero's `DayMoon`).
    fn of(environment: &crate::sky::EnvironmentSky) -> Self {
        let air = &environment.air;
        let glow = air.airglow.to_linear().to_vec3();
        let luma = (0.2126 * glow.x + 0.7152 * glow.y + 0.0722 * glow.z).max(1.0e-6);
        let mut key = NightKey {
            airglow: (glow * (air.airglow_nits / luma)).to_array(),
            ..default()
        };
        if let Some(moon) = environment.suns.get(1) {
            key.moon_dir = moon.direction.to_array();
            key.moon_lux = (moon.radiance * moon.solid_angle()).to_array();
        }
        key
    }
}

/// One environment's importance table; a generated sky's scattering tables and image too.
#[derive(Default)]
struct Slot {
    params: Buffer<AtmosphereGpu>,
    transmittance: Buffer<[f32; 4]>,
    multiscatter: Buffer<[f32; 4]>,
    radiance: Buffer<[f32; 4]>,
    table: Buffer<f32>,
    rows: Buffer<f32>,
    texture: Option<RenderTexture>,
    tex: u32,
    air: Option<AirKey>,
    sun: Option<SunKey>,
    night: Option<NightKey>,
    /// The HDR image the table was built from.
    source: Option<vk::Image>,
    built: bool,
}

impl Slot {
    fn ensure_table(&mut self, rd: &RenderDevice) {
        if self.table.handle != vk::Buffer::null() {
            return;
        }
        let n = (SKY_W * SKY_H) as u64;
        self.radiance = rd.create_device_buffer(
            n,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        self.table = rd.create_device_buffer(
            3 * n + 2,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        self.rows =
            rd.create_device_buffer(2 * SKY_H as u64 + 1, vk::BufferUsageFlags::STORAGE_BUFFER);
    }

    fn ensure_air(&mut self, rd: &RenderDevice) {
        self.ensure_table(rd);
        if self.texture.is_some() {
            return;
        }
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        self.params = rd.create_device_buffer(1, storage);
        self.transmittance = rd.create_device_buffer((T_W * T_H) as u64, storage);
        self.multiscatter = rd.create_device_buffer((MS_N * MS_N) as u64, storage);
        let texture = create_blank_texture(
            rd,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            SKY_W,
            SKY_H,
        );
        self.tex = rd.register_bindless_texture(&texture);
        self.texture = Some(texture);
        log::debug!("generated sky: {SKY_W}x{SKY_H} image, tables {T_W}x{T_H} + {MS_N}x{MS_N}");
    }

    fn release(&mut self, rd: &RenderDevice) {
        if self.table.handle == vk::Buffer::null() {
            return;
        }
        if let Some(texture) = self.texture.take() {
            rd.unregister_bindless_texture(&texture);
            rd.destroyer.destroy_image_view(texture.image_view);
            rd.destroyer.destroy_image(texture.image);
        }
        for handle in [
            self.params.handle,
            self.transmittance.handle,
            self.multiscatter.handle,
            self.radiance.handle,
            self.table.handle,
            self.rows.handle,
        ] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        *self = Slot::default();
    }
}

/// The importance-sampled skies, by environment bit.
#[derive(Resource)]
pub struct SkyGen {
    module: Handle<ComputeModule>,
    slots: [Slot; 8],
}

impl SkyGen {
    /// Environment `bit`'s importance table (generated or HDR), 0 until it is built.
    pub fn table(&self, bit: usize) -> u64 {
        let slot = &self.slots[bit];
        if slot.built { slot.table.address } else { 0 }
    }

    /// Environment `bit`'s generated sky image (bindless), once rendered.
    pub fn tex(&self, bit: usize) -> Option<u32> {
        let slot = &self.slots[bit];
        (slot.built && slot.texture.is_some()).then_some(slot.tex)
    }

    /// Environment `bit`'s generated sky image (bindless) and importance table, once it has them.
    pub fn sky(&self, bit: usize) -> Option<(u32, u64)> {
        Some((self.tex(bit)?, self.table(bit)))
    }

    /// Environment `bit`'s grid texels (row-major, `SKY_W` x `SKY_H`), read back after a
    /// blocking copy: for tests and tools, not frames.
    pub fn read_back(&self, rd: &RenderDevice, bit: usize) -> Option<Vec<Vec4>> {
        let slot = &self.slots[bit];
        if !slot.built {
            return None;
        }
        let n = (SKY_W * SKY_H) as u64;
        let mut host: Buffer<[f32; 4]> =
            rd.create_host_buffer(n, vk::BufferUsageFlags::TRANSFER_DST);
        rd.run_transfer_commands(|cmd| unsafe {
            rd.device.cmd_copy_buffer(
                cmd,
                slot.radiance.handle,
                host.handle,
                &[vk::BufferCopy::default().size(n * 16)],
            );
        });
        let texels = rd
            .map_buffer(&mut host)
            .as_slice_mut()
            .iter()
            .map(|t| Vec4::from_array(*t))
            .collect();
        rd.destroyer.destroy_buffer(host.handle);
        Some(texels)
    }
}

impl SkyGen {
    /// Environment `bit`'s importance table (rt_types.slang `importance`), read back after a
    /// blocking copy: for tests and tools, not frames.
    pub fn read_back_table(&self, rd: &RenderDevice, bit: usize) -> Option<Vec<f32>> {
        let slot = &self.slots[bit];
        if !slot.built {
            return None;
        }
        let len = 3 * (SKY_W * SKY_H) as u64 + 2;
        let mut host: Buffer<f32> = rd.create_host_buffer(len, vk::BufferUsageFlags::TRANSFER_DST);
        rd.run_transfer_commands(|cmd| unsafe {
            rd.device.cmd_copy_buffer(
                cmd,
                slot.table.handle,
                host.handle,
                &[vk::BufferCopy::default().size(len * 4)],
            );
        });
        let table = rd.map_buffer(&mut host).as_slice_mut().to_vec();
        rd.destroyer.destroy_buffer(host.handle);
        Some(table)
    }
}

fn atmosphere(slot: &Slot, air: &SkyAir, sun: SunKey) -> AtmosphereGpu {
    let mm = 1.0e-6;
    let SunKey(direction, cos_radius, lux) = sun;
    let solid_angle = std::f32::consts::TAU * (1.0 - cos_radius);
    AtmosphereGpu {
        transmittance: slot.transmittance.address,
        multiscatter: slot.multiscatter.address,
        planet_center: [0.0, -PLANET_RADIUS, 0.0],
        planet_radius: PLANET_RADIUS,
        atmosphere_height: AIR_HEIGHT,
        rayleigh_h: air.rayleigh_height.max(1.0),
        mie_h: air.mie_height.max(1.0),
        ozone_center: 25_000.0,
        ozone_width: 15_000.0,
        rayleigh_scatter: (air.rayleigh_scatter * mm).to_array(),
        mie_scatter: Vec3::splat(air.mie_scatter * mm).to_array(),
        mie_absorb: Vec3::splat(air.mie_absorb * mm).to_array(),
        ozone_absorb: (air.ozone_absorb * mm).to_array(),
        mie_g: air.mie_g.clamp(-0.99, 0.99),
        ground_albedo: air.ground_albedo.to_linear().to_vec3().to_array(),
        sun_direction: direction.to_array(),
        sun_radiance: Vec3::splat(lux / solid_angle.max(1.0e-9)).to_array(),
        sun_cos_radius: cos_radius,
        ..default()
    }
}

fn prepare_sky_gen(
    rd: Res<RenderDevice>,
    skies: Res<EnvironmentSkies>,
    modules: Res<ComputeModules>,
    frame: Res<Frame>,
    textures: Res<VulkanAssets<Image>>,
    mut gen_: ResMut<SkyGen>,
    mut commands: ResMut<FrameCommands>,
) {
    let ready = modules.get(&gen_.module).is_some();
    let module = gen_.module.clone();
    for (bit, environment) in skies.environments.iter().enumerate() {
        let slot = &mut gen_.slots[bit];
        if let Sky::Hdr { image, .. } = &environment.sky {
            if slot.texture.is_some() {
                slot.release(&rd);
            }
            if let Some(source) = textures.get(image)
                && ready
                && slot.source != Some(source.image)
            {
                filter_hdr(&rd, slot, *source, module.clone(), &mut commands);
            }
            continue;
        }
        let generated = matches!(environment.sky, Sky::Generated);
        if !generated || slot.source.is_some() {
            slot.release(&rd);
        }
        if !generated || !ready {
            continue;
        }
        slot.ensure_air(&rd);
        let air = &environment.air;
        // No sun: night, the sun straight down.
        let sun = environment
            .suns
            .first()
            .map_or(SunKey(Vec3::NEG_Y, 0.99999, 0.0), |s| {
                SunKey(s.direction, s.cos_radius, air.sun_illuminance)
            });
        let air_key = AirKey::of(air);
        let rebuild_air = slot.air != Some(air_key);
        let night = NightKey::of(environment);
        let rerender = rebuild_air || slot.sun != Some(sun) || slot.night != Some(night);
        slot.air = Some(air_key);
        slot.sun = Some(sun);
        slot.night = Some(night);
        slot.built = true;

        let gpu = atmosphere(slot, air, sun);
        let base = KernelParams {
            atmo: slot.params.address,
            sky: slot.radiance.address,
            table: slot.table.address,
            rows: slot.rows.address,
            altitude: air.altitude,
            moon_dir: night.moon_dir,
            moon_lux: night.moon_lux,
            airglow: night.airglow,
            ..default()
        };
        // The environment table's first sun, which the CPU writes this frame.
        let sun_out =
            (frame.environments.address != 0 && !environment.suns.is_empty()).then(|| {
                frame.environments.address
                    + (bit * size_of::<EnvironmentGpu>()) as u64
                    + std::mem::offset_of!(EnvironmentGpu, suns) as u64
                    + std::mem::offset_of!(EnvironmentSunGpu, radiance) as u64
            });
        let params_buffer = slot.params.handle;
        let radiance_buffer = slot.radiance.handle;
        let image = slot.texture.unwrap().image;
        let module = module.clone();
        commands.push(move |rd, modules, cmd| {
            let Some(module) = modules.get(&module) else {
                return;
            };
            let chunked = Some(std::mem::offset_of!(KernelParams, base));
            if rerender {
                unsafe {
                    rd.device
                        .cmd_update_buffer(cmd, params_buffer, 0, bytemuck::bytes_of(&gpu));
                }
                memory_barrier(
                    rd,
                    cmd,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ,
                );
                if rebuild_air {
                    record_dispatch(
                        rd,
                        cmd,
                        module,
                        "atmo_transmittance",
                        &base,
                        T_W * T_H,
                        chunked,
                    );
                    after_compute(rd, cmd);
                    record_dispatch(
                        rd,
                        cmd,
                        module,
                        "atmo_multiscatter",
                        &base,
                        MS_N * MS_N,
                        chunked,
                    );
                    after_compute(rd, cmd);
                }
                record_dispatch(rd, cmd, module, "sky_render", &base, SKY_W * SKY_H, chunked);
                after_compute(rd, cmd);
                record_importance(rd, cmd, module, &base);
                copy_to_image(rd, cmd, radiance_buffer, image);
            }
            if let Some(sun_out) = sun_out {
                let params = KernelParams { sun_out, ..base };
                record_dispatch(rd, cmd, module, "sun_tint", &params, 1, None);
            }
        });
    }
}

fn after_compute(rd: &RenderDevice, cmd: vk::CommandBuffer) {
    memory_barrier(
        rd,
        cmd,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::TRANSFER_READ,
    )
}

/// The importance table from the grid in `base.sky`: a thread per row, one over the rows, the
/// normalised cdf and pdf, then the guide.
fn record_importance(
    rd: &RenderDevice,
    cmd: vk::CommandBuffer,
    module: &CompiledComputeModule,
    base: &KernelParams,
) {
    let chunked = Some(std::mem::offset_of!(KernelParams, base));
    record_dispatch(rd, cmd, module, "sky_rows", base, SKY_H, None);
    after_compute(rd, cmd);
    record_dispatch(rd, cmd, module, "sky_offsets", base, 1, None);
    after_compute(rd, cmd);
    record_dispatch(
        rd,
        cmd,
        module,
        "sky_normalize",
        base,
        SKY_W * SKY_H,
        chunked,
    );
    after_compute(rd, cmd);
    record_dispatch(
        rd,
        cmd,
        module,
        "sky_guide",
        base,
        SKY_W * SKY_H + 1,
        chunked,
    );
}

/// An HDR image's importance table: the image box-filtered down to the grid by 2:1 linear blits
/// (each texel the exact mean of the four under it; the last blit lands on `SKY_W` x `SKY_H`),
/// copied into the grid buffer and built like a generated sky's.
fn filter_hdr(
    rd: &RenderDevice,
    slot: &mut Slot,
    source: RenderTexture,
    module: Handle<ComputeModule>,
    commands: &mut FrameCommands,
) {
    slot.ensure_table(rd);
    slot.source = Some(source.image);
    slot.built = true;
    let mut sizes = Vec::new();
    let (mut w, mut h) = (source.width, source.height);
    while w > 2 * SKY_W || h > 2 * SKY_H {
        w = (w / 2).max(SKY_W);
        h = (h / 2).max(SKY_H);
        sizes.push((w, h));
    }
    sizes.push((SKY_W, SKY_H));
    let scratch: Vec<RenderTexture> = sizes
        .iter()
        .map(|&(w, h)| {
            create_blank_texture(
                rd,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                w,
                h,
            )
        })
        .collect();
    log::debug!(
        "sky importance: {}x{} HDR filtered to {SKY_W}x{SKY_H} in {} blits",
        source.width,
        source.height,
        scratch.len()
    );
    let base = KernelParams {
        sky: slot.radiance.address,
        table: slot.table.address,
        rows: slot.rows.address,
        ..default()
    };
    let radiance = slot.radiance.handle;
    commands.push(move |rd, modules, cmd| {
        for t in &scratch {
            rd.destroyer.destroy_image_view(t.image_view);
            rd.destroyer.destroy_image(t.image);
        }
        let Some(module) = modules.get(&module) else {
            return;
        };
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let barrier = |image, from, to, src: (S, A), dst: (S, A)| {
            let b = vk_init::layout_transition2(image, from, to)
                .src_stage_mask(src.0)
                .src_access_mask(src.1)
                .dst_stage_mask(dst.0)
                .dst_access_mask(dst.1);
            unsafe {
                rd.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&b)),
                );
            }
        };
        let layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        let corner = |t: &RenderTexture| vk::Offset3D {
            x: t.width as i32,
            y: t.height as i32,
            z: 1,
        };
        barrier(
            source.image,
            L::SHADER_READ_ONLY_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
            (S::ALL_COMMANDS, A::NONE),
            (S::TRANSFER, A::TRANSFER_READ),
        );
        let mut from = source;
        for to in &scratch {
            let blit = vk::ImageBlit::default()
                .src_subresource(layers)
                .src_offsets([vk::Offset3D::default(), corner(&from)])
                .dst_subresource(layers)
                .dst_offsets([vk::Offset3D::default(), corner(to)]);
            unsafe {
                rd.device.cmd_blit_image(
                    cmd,
                    from.image,
                    L::TRANSFER_SRC_OPTIMAL,
                    to.image,
                    L::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }
            barrier(
                to.image,
                L::TRANSFER_DST_OPTIMAL,
                L::TRANSFER_SRC_OPTIMAL,
                (S::TRANSFER, A::TRANSFER_WRITE),
                (S::TRANSFER, A::TRANSFER_READ),
            );
            from = *to;
        }
        barrier(
            source.image,
            L::TRANSFER_SRC_OPTIMAL,
            L::SHADER_READ_ONLY_OPTIMAL,
            (S::TRANSFER, A::NONE),
            (S::ALL_COMMANDS, A::SHADER_READ),
        );
        unsafe {
            rd.device.cmd_copy_image_to_buffer(
                cmd,
                from.image,
                L::TRANSFER_SRC_OPTIMAL,
                radiance,
                &[vk_init::buffer_image_copy(SKY_W, SKY_H)],
            );
        }
        memory_barrier(
            rd,
            cmd,
            S::TRANSFER,
            A::TRANSFER_WRITE,
            S::COMPUTE_SHADER,
            A::SHADER_READ,
        );
        record_importance(rd, cmd, module, &base);
    });
}

/// The rendered texels into the sampled image (the frame's later barrier makes it visible).
fn copy_to_image(rd: &RenderDevice, cmd: vk::CommandBuffer, buffer: vk::Buffer, image: vk::Image) {
    let to_dst = vk_init::layout_transition2(
        image,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    )
    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
    .src_access_mask(vk::AccessFlags2::SHADER_READ)
    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    let to_read = vk_init::layout_transition2(
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )
    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
    .dst_access_mask(vk::AccessFlags2::SHADER_READ);
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
            &[vk_init::buffer_image_copy(SKY_W, SKY_H)],
        );
        rd.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_read)),
        );
    }
}

fn cleanup(world: &mut World) {
    world.resource_scope(|world, mut gen_: Mut<SkyGen>| {
        let rd = world.resource::<RenderDevice>();
        for slot in &mut gen_.slots {
            slot.release(rd);
        }
    });
}

pub struct SkyGenPlugin;

impl Plugin for SkyGenPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/atmosphere.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "atmo_transmittance",
                "atmo_multiscatter",
                "sky_render",
                "sky_rows",
                "sky_offsets",
                "sky_normalize",
                "sky_guide",
                "sun_tint",
            ],
        ));
        app.insert_resource(SkyGen {
            module,
            slots: default(),
        });
        app.add_systems(Last, prepare_sky_gen.in_set(RenderSet::Prepare));
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
