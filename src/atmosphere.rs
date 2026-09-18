//! Planetary atmosphere and cloud shell for [`Sky::Atmosphere`].
//!
//! The planet is a sphere ([`Atmosphere::planet_center`] / `planet_radius`); the air above
//! it is Hillaire 2020's model ("A Scalable and Production Ready Sky and Atmosphere
//! Rendering Technique"): three small LUTs -- transmittance, multiple scattering, and the
//! sky as seen from the camera's altitude -- rebuilt every frame by the atmosphere.slang
//! kernels (they cost well under a tenth of a millisecond, so nothing tracks parameter
//! changes). The miss shader returns the sky-view LUT (plus the sun disc and an optional
//! space image behind the air), the raygen attenuates the sun by the transmittance LUT at
//! every hit and adds aerial perspective on the primary hit.
//!
//! [`CloudLayer`] is one procedural shell between two altitudes, marched deterministically
//! in the raygen along the camera ray (assets/shaders/atmosphere.glsl `cloudMarch`) and
//! folded into the noisy colour BEFORE Ray Reconstruction -- the RTX Remix precedent for
//! volumetrics under DLSS-RR, which only holds because the march carries no per-frame
//! noise. The noise tables it samples are built once by the `cloud_noise` kernel.
//!
//! Both resources are reflected: edit them live in the F1 world inspector. The sun
//! (direction, disc size, top-of-atmosphere radiance) is the shared [`ProceduralSky`].

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sky::{ProceduralSky, Sky},
};

// LUT and table sizes; must match atmosphere.glsl.
const T_W: u32 = 256;
const T_H: u32 = 64;
const MS_N: u32 = 32;
const SV_W: u32 = 192;
const SV_H: u32 = 108;
const NOISE_N: u32 = 128;
const DETAIL_N: u32 = 32;

/// The planet and its air. Lengths in metres; the defaults are Earth's air on a planet
/// small enough for f32 world coordinates, its surface at the origin.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub struct Atmosphere {
    /// Centre of the planet sphere (put it at `(0, -radius, 0)` for a scene at the origin).
    pub planet_center: Vec3,
    #[reflect(@1000.0..=1.0e7_f32)]
    pub planet_radius: f32,
    /// Where the air ends above the surface.
    #[reflect(@100.0..=200000.0_f32)]
    pub atmosphere_height: f32,
    /// Exponential falloff heights of the molecular (Rayleigh) and aerosol (Mie) densities.
    #[reflect(@10.0..=50000.0_f32)]
    pub rayleigh_scale_height: f32,
    #[reflect(@10.0..=50000.0_f32)]
    pub mie_scale_height: f32,
    /// The ozone layer: a tent of this half-width around this altitude.
    #[reflect(@0.0..=100000.0_f32)]
    pub ozone_center: f32,
    #[reflect(@100.0..=100000.0_f32)]
    pub ozone_width: f32,
    /// Sea-level coefficients, per megametre (1e-6 / m). Earth: (5.802, 13.558, 33.1).
    pub rayleigh_scatter: Vec3,
    #[reflect(@0.0..=100.0_f32)]
    pub mie_scatter: f32,
    #[reflect(@0.0..=100.0_f32)]
    pub mie_absorb: f32,
    /// Ozone absorption at the layer's peak, per megametre. Earth: (0.65, 1.881, 0.085).
    pub ozone_absorb: Vec3,
    /// Aerosol phase asymmetry (the bright halo around the sun).
    #[reflect(@-0.99..=0.99_f32)]
    pub mie_g: f32,
    /// What the ground reflects into the sky's multiple scattering.
    pub ground_albedo: Color,
    /// An equirectangular space image (stars) behind the air, texel × `space_scale` nits.
    pub space: Option<Handle<Image>>,
    #[reflect(@0.0..=100000.0_f32)]
    pub space_scale: f32,
}

impl Default for Atmosphere {
    fn default() -> Self {
        Self {
            planet_center: Vec3::new(0.0, -200_000.0, 0.0),
            planet_radius: 200_000.0,
            atmosphere_height: 60_000.0,
            rayleigh_scale_height: 8_000.0,
            mie_scale_height: 1_200.0,
            ozone_center: 25_000.0,
            ozone_width: 15_000.0,
            rayleigh_scatter: Vec3::new(5.802, 13.558, 33.1),
            mie_scatter: 3.996,
            mie_absorb: 4.4,
            ozone_absorb: Vec3::new(0.65, 1.881, 0.085),
            mie_g: 0.8,
            ground_albedo: Color::linear_rgb(0.3, 0.28, 0.25),
            space: None,
            space_scale: 1.0,
        }
    }
}

/// The cloud shell. Altitudes in metres above the planet surface.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource)]
pub struct CloudLayer {
    pub enabled: bool,
    /// Shadow the sun on primary hits by the shell above them.
    pub shadows: bool,
    #[reflect(@0.0..=50000.0_f32)]
    pub bottom: f32,
    #[reflect(@0.0..=50000.0_f32)]
    pub top: f32,
    /// 0 = clear sky, 1 = overcast.
    #[reflect(@0.0..=1.0_f32)]
    pub coverage: f32,
    /// Size of the weather pattern (metres per feature).
    #[reflect(@1000.0..=200000.0_f32)]
    pub coverage_scale: f32,
    /// Multiplier on the shell's density.
    #[reflect(@0.0..=4.0_f32)]
    pub density: f32,
    /// Extinction per metre at full density (real cumulus cores reach 0.04; lower reads
    /// better in the march, whose density is a soft field, not a core).
    #[reflect(@0.001..=0.2_f32)]
    pub extinction: f32,
    /// How much the erosion noise eats into the edges.
    #[reflect(@0.0..=1.0_f32)]
    pub detail: f32,
    /// Metres per repeat of the base shape noise and of the erosion noise.
    #[reflect(@500.0..=50000.0_f32)]
    pub scale: f32,
    #[reflect(@50.0..=5000.0_f32)]
    pub detail_scale: f32,
    /// Wind velocity (metres per second) the shell drifts with.
    pub wind: Vec3,
    /// March samples along the camera ray and towards the sun.
    #[reflect(@8.0..=128.0_f32)]
    pub steps: u32,
    #[reflect(@1.0..=12.0_f32)]
    pub light_steps: u32,
    /// Length of the sun march (metres).
    #[reflect(@50.0..=5000.0_f32)]
    pub light_distance: f32,
    /// The march never covers more than this along a ray (metres).
    #[reflect(@1000.0..=200000.0_f32)]
    pub max_distance: f32,
    /// Sky light on the cloud, as a fraction of the zenith radiance.
    #[reflect(@0.0..=3.0_f32)]
    pub ambient: f32,
    /// Phase asymmetries of the forward and back lobes.
    #[reflect(@0.0..=0.95_f32)]
    pub forward_g: f32,
    #[reflect(@-0.95..=0.0_f32)]
    pub back_g: f32,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            enabled: true,
            shadows: true,
            bottom: 1_500.0,
            top: 2_800.0,
            coverage: 0.55,
            coverage_scale: 30_000.0,
            density: 1.2,
            extinction: 0.006,
            detail: 0.5,
            scale: 4_500.0,
            detail_scale: 700.0,
            wind: Vec3::new(8.0, 0.0, 3.0),
            steps: 64,
            light_steps: 5,
            light_distance: 800.0,
            max_distance: 40_000.0,
            ambient: 0.6,
            forward_g: 0.7,
            back_g: -0.2,
        }
    }
}

/// Must match `AtmosphereParams` in types.glsl (scalar layout: the 8-byte pointers first,
/// then 4-byte fields with no padding).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AtmosphereGpu {
    transmittance: u64,
    multiscatter: u64,
    skyview_rad: u64,
    skyview_trn: u64,
    noise: u64,
    noise_detail: u64,
    planet_center: [f32; 3],
    planet_radius: f32,
    atmosphere_height: f32,
    rayleigh_h: f32,
    mie_h: f32,
    ozone_center: f32,
    ozone_width: f32,
    rayleigh_scatter: [f32; 3],
    mie_scatter: [f32; 3],
    mie_absorb: [f32; 3],
    ozone_absorb: [f32; 3],
    mie_g: f32,
    ground_albedo: [f32; 3],
    sun_direction: [f32; 3],
    sun_radiance: [f32; 3],
    sun_cos_radius: f32,
    camera_pos: [f32; 3],
    clouds: u32,
    cloud_shadows: u32,
    cloud_steps: u32,
    cloud_light_steps: u32,
    cloud_bottom: f32,
    cloud_top: f32,
    cloud_coverage: f32,
    cloud_coverage_scale: f32,
    cloud_density: f32,
    cloud_extinction: f32,
    cloud_detail: f32,
    cloud_scale: f32,
    cloud_detail_scale: f32,
    cloud_wind: [f32; 3],
    cloud_light_dist: f32,
    cloud_max_dist: f32,
    cloud_ambient: f32,
    cloud_forward_g: f32,
    cloud_back_g: f32,
}

/// Push constants of every atmosphere kernel.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KernelParams {
    atmo: u64,
    base: u32,
    pad: u32,
}

#[derive(Resource)]
pub struct AtmosphereState {
    /// atmosphere.slang: the three LUT kernels and the noise builder.
    module: Handle<ComputeModule>,
    params: Buffer<AtmosphereGpu>,
    transmittance: Buffer<[f32; 4]>,
    multiscatter: Buffer<[f32; 4]>,
    skyview_rad: Buffer<[f32; 4]>,
    skyview_trn: Buffer<[f32; 4]>,
    noise: Buffer<f32>,
    noise_detail: Buffer<f32>,
    noise_built: bool,
    /// Accumulated wind drift (metres).
    wind_offset: Vec3,
}

impl AtmosphereState {
    /// Device address of the parameter block (0 until the atmosphere first renders).
    pub fn address(&self) -> u64 {
        self.params.address
    }

    fn ensure_buffers(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        if self.params.handle != vk::Buffer::null() {
            return;
        }
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        self.params = rd.create_device_buffer(1, storage);
        self.transmittance = rd.create_device_buffer((T_W * T_H) as u64, storage);
        self.multiscatter = rd.create_device_buffer((MS_N * MS_N) as u64, storage);
        self.skyview_rad = rd.create_device_buffer((SV_W * SV_H) as u64, storage);
        self.skyview_trn = rd.create_device_buffer((SV_W * SV_H) as u64, storage);
        self.noise = rd.create_device_buffer((NOISE_N * NOISE_N * NOISE_N) as u64, storage);
        self.noise_detail =
            rd.create_device_buffer((DETAIL_N * DETAIL_N * DETAIL_N) as u64, storage);
        // Defined contents until the kernels have run (a black sky, no clouds).
        for handle in [
            self.transmittance.handle,
            self.multiscatter.handle,
            self.skyview_rad.handle,
            self.skyview_trn.handle,
            self.noise.handle,
            self.noise_detail.handle,
        ] {
            unsafe { rd.device.cmd_fill_buffer(cmd, handle, 0, vk::WHOLE_SIZE, 0) };
        }
        log::info!(
            "atmosphere: LUTs {}x{} + {}x{} + 2x{}x{}, cloud noise {}^3 + {}^3 ({} MB)",
            T_W,
            T_H,
            MS_N,
            MS_N,
            SV_W,
            SV_H,
            NOISE_N,
            DETAIL_N,
            (NOISE_N.pow(3) + DETAIL_N.pow(3)) * 4 / (1024 * 1024)
        );
    }

    /// Uploads this frame's parameters and records the LUT kernels (and the noise tables
    /// on first use). Call before the trace; a no-op unless the sky is [`Sky::Atmosphere`].
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        sky: &Sky,
        atmosphere: &Atmosphere,
        clouds: &CloudLayer,
        sun: &ProceduralSky,
        camera_pos: Vec3,
        dt: f32,
    ) {
        if !matches!(sky, Sky::Atmosphere) {
            return;
        }
        self.ensure_buffers(rd, cmd);
        self.wind_offset += clouds.wind * dt.clamp(0.0, 0.25);

        let km = 1.0e-6;
        let gpu = AtmosphereGpu {
            transmittance: self.transmittance.address,
            multiscatter: self.multiscatter.address,
            skyview_rad: self.skyview_rad.address,
            skyview_trn: self.skyview_trn.address,
            noise: self.noise.address,
            noise_detail: self.noise_detail.address,
            planet_center: atmosphere.planet_center.to_array(),
            planet_radius: atmosphere.planet_radius.max(1.0),
            atmosphere_height: atmosphere.atmosphere_height.max(1.0),
            rayleigh_h: atmosphere.rayleigh_scale_height.max(1.0),
            mie_h: atmosphere.mie_scale_height.max(1.0),
            ozone_center: atmosphere.ozone_center,
            ozone_width: atmosphere.ozone_width.max(1.0),
            rayleigh_scatter: (atmosphere.rayleigh_scatter * km).to_array(),
            mie_scatter: Vec3::splat(atmosphere.mie_scatter * km).to_array(),
            mie_absorb: Vec3::splat(atmosphere.mie_absorb * km).to_array(),
            ozone_absorb: (atmosphere.ozone_absorb * km).to_array(),
            mie_g: atmosphere.mie_g.clamp(-0.99, 0.99),
            ground_albedo: atmosphere.ground_albedo.to_linear().to_vec3().to_array(),
            sun_direction: sun.sun_direction().to_array(),
            sun_radiance: Vec3::splat(sun.sun_radiance).to_array(),
            sun_cos_radius: sun.sun_cos_radius(),
            camera_pos: camera_pos.to_array(),
            clouds: clouds.enabled as u32,
            cloud_shadows: clouds.shadows as u32,
            cloud_steps: clouds.steps.clamp(4, 256),
            cloud_light_steps: clouds.light_steps.clamp(1, 16),
            cloud_bottom: clouds.bottom,
            cloud_top: clouds.top.max(clouds.bottom + 1.0),
            cloud_coverage: clouds.coverage.clamp(0.0, 1.0),
            cloud_coverage_scale: clouds.coverage_scale.max(1.0),
            cloud_density: clouds.density.max(0.0),
            cloud_extinction: clouds.extinction.max(1.0e-5),
            cloud_detail: clouds.detail.clamp(0.0, 1.0),
            cloud_scale: clouds.scale.max(1.0),
            cloud_detail_scale: clouds.detail_scale.max(1.0),
            cloud_wind: self.wind_offset.to_array(),
            cloud_light_dist: clouds.light_distance.max(1.0),
            cloud_max_dist: clouds.max_distance.max(1.0),
            cloud_ambient: clouds.ambient.max(0.0),
            cloud_forward_g: clouds.forward_g.clamp(0.0, 0.99),
            cloud_back_g: clouds.back_g.clamp(-0.99, 0.0),
        };
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, self.params.handle, 0, bytemuck::bytes_of(&gpu));
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER
                | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ,
        );

        let Some(module) = modules.get(&self.module) else {
            return;
        };
        let params = KernelParams {
            atmo: self.params.address,
            base: 0,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(KernelParams, base));
        let compute_barrier = |rd: &RenderDevice| {
            memory_barrier(
                rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
        };
        if !self.noise_built {
            record_dispatch(
                rd,
                cmd,
                module,
                "cloud_noise",
                &params,
                NOISE_N.pow(3) + DETAIL_N.pow(3),
                base,
            );
            compute_barrier(rd);
            self.noise_built = true;
        }
        record_dispatch(
            rd,
            cmd,
            module,
            "atmo_transmittance",
            &params,
            T_W * T_H,
            base,
        );
        compute_barrier(rd);
        record_dispatch(
            rd,
            cmd,
            module,
            "atmo_multiscatter",
            &params,
            MS_N * MS_N,
            base,
        );
        compute_barrier(rd);
        record_dispatch(rd, cmd, module, "atmo_skyview", &params, SV_W * SV_H, base);
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags2::SHADER_READ,
        );
    }
}

fn cleanup(mut state: ResMut<AtmosphereState>, rd: Res<RenderDevice>) {
    // The buffers are created together on the first atmosphere frame; none if it never came.
    if state.params.handle == vk::Buffer::null() {
        return;
    }
    for handle in [
        state.params.handle,
        state.transmittance.handle,
        state.multiscatter.handle,
        state.skyview_rad.handle,
        state.skyview_trn.handle,
        state.noise.handle,
        state.noise_detail.handle,
    ] {
        rd.destroyer.destroy_buffer(handle);
    }
    state.params = Buffer::default();
    state.transmittance = Buffer::default();
    state.multiscatter = Buffer::default();
    state.skyview_rad = Buffer::default();
    state.skyview_trn = Buffer::default();
    state.noise = Buffer::default();
    state.noise_detail = Buffer::default();
    state.noise_built = false;
}

pub struct AtmospherePlugin;

impl Plugin for AtmospherePlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/atmosphere.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "atmo_transmittance",
                "atmo_multiscatter",
                "atmo_skyview",
                "cloud_noise",
            ],
        ));
        let state = AtmosphereState {
            module,
            params: Buffer::default(),
            transmittance: Buffer::default(),
            multiscatter: Buffer::default(),
            skyview_rad: Buffer::default(),
            skyview_trn: Buffer::default(),
            noise: Buffer::default(),
            noise_detail: Buffer::default(),
            noise_built: false,
            wind_offset: Vec3::ZERO,
        };
        app.insert_resource(state);
        app.register_type::<Atmosphere>()
            .register_type::<CloudLayer>()
            .init_resource::<Atmosphere>()
            .init_resource::<CloudLayer>();
        app.add_systems(TeardownSchedule, cleanup.before(on_shutdown));
    }
}
