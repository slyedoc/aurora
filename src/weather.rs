//! Weather around the camera: ground mist and rain, WoW-style (nothing laid over the whole map).
//! A [`Weather`] on an environment keeps three parts under it: a [`FogShape::Ground`] fog volume
//! that follows the active camera (the fog grid lights it like any fog: shafts, lamps, shadows),
//! a block of rain streaks falling through a box about the camera, and their splashes. The mist
//! lies on the terrain around the camera, which a ray-query pass samples after each TLAS build
//! ([`GroundHeights`]: the nearest surface under each texel that is not an alpha-tested card, so
//! foliage is skipped). A drop casts its own path against the TLAS as it starts: it splashes
//! where that lands (droplets; on water a spreading disc as well).

use ash::vk;
use bevy::{camera::visibility::RenderLayers, feathers_inspector::Step, prelude::*};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::material_flags,
    compute::{BlockRows, ComputeModule, FrameCommands, memory_barrier, record_dispatch},
    environment::{InEnvironment, environment_mask},
    fog_volume::{FogShape, FogVolume, GroundHeights},
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    sphere::Sphere,
    surface_group::SurfaceRow,
    tlas_builder::{InstanceBlock, InstanceBlockSlots, TLAS},
};

/// Texels along the ground buffer's side, and their size (metres): 256 m about the camera.
pub const GROUND_SIZE: u32 = 256;
pub const GROUND_TEXEL: f32 = 1.0;
/// A ground texel with nothing under it (weather_ground.slang `GROUND_NONE`).
const GROUND_NONE: f32 = -1.0e30;
/// The rain's parts cast no shadow (their rows' material records carry the flag).
const NO_SHADOW: SurfaceRow = SurfaceRow {
    row: 0,
    flags: material_flags::NO_SHADOW,
};

/// Most rain streaks at once (`Weather::rain` 1).
pub const RAIN_CAPACITY: u32 = 65536;
/// Instance rows per splash (weather_rain.slang `SPLASH_PARTS`).
const SPLASH_PARTS: u32 = 3;
/// A drop's record on the GPU (weather_rain.slang `Drop`), in words.
const DROP_WORDS: usize = 20;

/// The weather of the environment it is on, around the camera.
#[derive(Component, Reflect, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Weather {
    /// Mist's extinction per metre at its thickest, on the ground (0 = none).
    #[reflect(@0.0..=1.0_f32)]
    pub mist: f32,
    /// How deep the mist lies over the terrain (metres).
    #[reflect(@Step(0.1))]
    pub mist_thickness: f32,
    /// How far about the camera it reaches (metres; at most ~120, the ground buffer's half).
    #[reflect(@Step(1.0))]
    pub mist_radius: f32,
    /// Size of its wisps (metres).
    #[reflect(@Step(0.1))]
    pub mist_noise_scale: f32,
    /// How much the wisps carve it (0 = even).
    #[reflect(@0.0..=1.0_f32)]
    pub mist_noise_strength: f32,
    /// How fast the wisps climb (metres per second).
    #[reflect(@Step(0.05))]
    pub mist_rise: f32,
    /// Henyey-Greenstein g of its light: low, as a thick layer's light has scattered many times
    /// (single forward scattering left mist dark from the sun's side).
    #[reflect(@-0.95..=0.95_f32)]
    pub mist_anisotropy: f32,
    /// What drifts the mist and slants the rain (metres per second).
    #[reflect(@Step(0.1))]
    pub wind: Vec3,
    /// Rain, 0 none to 1 a downpour.
    #[reflect(@0.0..=1.0_f32)]
    pub rain: f32,
    /// How far about the camera it falls (metres); denser near, sparser far.
    #[reflect(@Step(1.0))]
    pub rain_radius: f32,
    /// The height of the column it starts in about the camera (metres; 0.6 above, 0.4 below).
    #[reflect(@Step(1.0))]
    pub rain_height: f32,
    /// Falling speed (metres per second).
    #[reflect(@Step(0.5))]
    pub rain_speed: f32,
    /// A streak's length and width (metres).
    #[reflect(@Step(0.05))]
    pub rain_length: f32,
    #[reflect(@Step(0.001))]
    pub rain_width: f32,
    /// The streaks' colour: they scatter the light around them (lamps, windows, the sky), so
    /// this is how bright rain reads by day and by night.
    pub rain_color: Color,
}

impl Weather {
    pub const CLEAR: Self = Self {
        mist: 0.0,
        mist_thickness: 3.0,
        mist_radius: 100.0,
        mist_noise_scale: 6.0,
        mist_noise_strength: 0.7,
        mist_rise: 0.25,
        mist_anisotropy: 0.2,
        wind: Vec3::new(0.8, 0.0, 0.3),
        rain: 0.0,
        rain_radius: 100.0,
        rain_height: 30.0,
        rain_speed: 9.0,
        rain_length: 0.45,
        rain_width: 0.008,
        rain_color: Color::srgb(0.72, 0.75, 0.78),
    };

    /// A still morning: mist in the hollows, no rain.
    pub const MIST: Self = Self {
        mist: 0.12,
        mist_thickness: 4.0,
        wind: Vec3::new(0.3, 0.0, 0.1),
        ..Self::CLEAR
    };

    /// Elwynn in the rain: a knee-deep buff over the ground, steady rain.
    pub const RAIN: Self = Self {
        mist: 0.08,
        mist_thickness: 2.5,
        rain: 0.5,
        ..Self::CLEAR
    };

    /// Heavy rain, driven by the wind.
    pub const STORM: Self = Self {
        mist: 0.1,
        mist_thickness: 3.0,
        mist_rise: 0.5,
        wind: Vec3::new(4.0, 0.0, 1.5),
        rain: 1.0,
        rain_speed: 12.0,
        rain_length: 0.6,
        ..Self::CLEAR
    };
}

impl Default for Weather {
    fn default() -> Self {
        Self::RAIN
    }
}

/// The parts a [`Weather`] keeps under its environment.
#[derive(Component, Clone, Copy)]
struct WeatherParts {
    mist: Entity,
    rain: Entity,
    splash: Entity,
    ripples: Entity,
}

#[derive(Resource)]
struct WeatherKernels {
    ground: Handle<ComputeModule>,
    rain: Handle<ComputeModule>,
    rain_material: Handle<AuroraMaterial>,
    ripple_material: Handle<AuroraMaterial>,
    /// A thin unit ring in its xz plane: a ripple's crest.
    ripple_mesh: Handle<AuroraMesh>,
}

/// The ground buffers (written alternately: a frame in flight reads the other) and the rain's
/// drops.
#[derive(Resource, Default)]
struct WeatherBuffers {
    ground: [Option<Buffer<f32>>; 2],
    /// The one written last frame, and where it lies.
    last: Option<(usize, Vec2)>,
    drops: Option<Buffer<[f32; DROP_WORDS]>>,
    /// The rain's `RainRows`, rewritten by each step.
    rain_rows: Option<Buffer<RainRows>>,
    frame: u32,
}

/// `GroundPush` in weather_ground.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroundPush {
    tlas: u64,
    materials: u64,
    heights: u64,
    origin: [f32; 2],
    texel: f32,
    size: u32,
    top: f32,
    depth: f32,
    mask: u32,
    pad: u32,
}

/// The rain's blocks (weather_rain.slang `RainRows`), uploaded with each step.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RainRows {
    drops: BlockRows,
    splashes: BlockRows,
    ripples: BlockRows,
}

/// `RainPush` in weather_rain.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RainPush {
    drops: u64,
    rows: u64,
    tlas: u64,
    materials: u64,
    count: u32,
    frame: u32,
    dt: f32,
    time: f32,
    camera: [f32; 3],
    radius: f32,
    height: f32,
    speed: f32,
    wind: [f32; 2],
    length: f32,
    width: f32,
    mask: u32,
    /// The angle one output pixel spans: streaks, droplets and rings thicken with distance so
    /// each covers about a pixel (thinner, they drop out of the image a few metres off).
    pixel_angle: f32,
}

fn spawn_parts(add: On<Add<Weather>>, kernels: Res<WeatherKernels>, mut commands: Commands) {
    let env = add.entity;
    let mist = commands
        .spawn((
            Name::new("weather mist"),
            ChildOf(env),
            FogVolume {
                shape: FogShape::Ground,
                density: 0.0,
                ..default()
            },
            Visibility::default(),
        ))
        .id();
    let rain = commands
        .spawn((
            Name::new("weather rain"),
            ChildOf(env),
            Sphere,
            InstanceBlock {
                capacity: RAIN_CAPACITY,
                poses: Vec::new(),
            },
            AuroraMaterial3d(kernels.rain_material.clone()),
            NO_SHADOW,
            Transform::default(),
        ))
        .id();
    let splash = commands
        .spawn((
            Name::new("weather splashes"),
            ChildOf(env),
            Sphere,
            InstanceBlock {
                capacity: RAIN_CAPACITY * SPLASH_PARTS,
                poses: Vec::new(),
            },
            AuroraMaterial3d(kernels.rain_material.clone()),
            NO_SHADOW,
            Transform::default(),
        ))
        .id();
    let ripples = commands
        .spawn((
            Name::new("weather ripples"),
            ChildOf(env),
            AuroraMesh3d(kernels.ripple_mesh.clone()),
            InstanceBlock {
                capacity: RAIN_CAPACITY,
                poses: Vec::new(),
            },
            AuroraMaterial3d(kernels.ripple_material.clone()),
            NO_SHADOW,
            Transform::default(),
        ))
        .id();
    commands.entity(env).insert(WeatherParts {
        mist,
        rain,
        splash,
        ripples,
    });
}

fn despawn_parts(remove: On<Remove<Weather>>, parts: Query<&WeatherParts>, mut commands: Commands) {
    if let Ok(parts) = parts.get(remove.entity) {
        commands.entity(parts.mist).try_despawn();
        commands.entity(parts.rain).try_despawn();
        commands.entity(parts.splash).try_despawn();
        commands.entity(parts.ripples).try_despawn();
        commands.entity(remove.entity).try_remove::<WeatherParts>();
    }
}

/// The mist volume follows the camera; the ground under it is sampled after the TLAS build and
/// the rain stepped before it.
#[allow(clippy::too_many_arguments)]
/// The weather's rain colour onto the streaks' and splashes' material.
fn tint_rain(
    weathers: Query<&Weather, Changed<Weather>>,
    kernels: Res<WeatherKernels>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    let Some(weather) = weathers.iter().next() else {
        return;
    };
    if materials
        .get(&kernels.rain_material)
        .is_some_and(|m| m.base_color != weather.rain_color)
        && let Some(mut m) = materials.get_mut(&kernels.rain_material)
    {
        m.base_color = weather.rain_color;
    }
}

fn update_weather(
    weathers: Query<(&Weather, &WeatherParts, &GlobalTransform)>,
    mut volumes: Query<(&mut FogVolume, &mut Transform)>,
    rain_slots: Query<&InstanceBlockSlots>,
    cameras: Query<(
        &GlobalTransform,
        &Camera,
        Option<&RenderLayers>,
        Option<&InEnvironment>,
        Option<&Projection>,
    )>,
    kernels: Res<WeatherKernels>,
    rd: Res<RenderDevice>,
    time: Res<Time>,
    mut buffers: ResMut<WeatherBuffers>,
    mut ground: ResMut<GroundHeights>,
    mut tlas: ResMut<TLAS>,
    mut frame: ResMut<FrameCommands>,
) {
    let Some((camera, active, layers, world, projection)) =
        cameras.iter().find(|(_, c, ..)| c.is_active)
    else {
        return;
    };
    // The angle one output pixel spans (0: no floor on the rain's widths).
    let pixel_angle = match (projection, active.physical_viewport_size()) {
        (Some(Projection::Perspective(p)), Some(size)) if size.y > 0 => p.fov / size.y as f32,
        _ => 0.0,
    };
    let eye = camera.translation();
    let Some((weather, parts, env)) = weathers.iter().next() else {
        *ground = GroundHeights::default();
        return;
    };
    let to_env = env.affine().inverse();
    let buffers = &mut *buffers;
    buffers.frame = buffers.frame.wrapping_add(1);

    if let Ok((mut volume, mut transform)) = volumes.get_mut(parts.mist) {
        let radius = weather
            .mist_radius
            .clamp(1.0, GROUND_SIZE as f32 * GROUND_TEXEL * 0.47);
        let want = FogVolume {
            shape: FogShape::Ground,
            half_extents: Vec3::new(radius, 80.0, radius),
            density: weather.mist.max(0.0),
            edge: 0.3,
            noise_scale: weather.mist_noise_scale.max(0.1),
            noise_strength: weather.mist_noise_strength,
            wind: weather.wind,
            thickness: weather.mist_thickness,
            rise: weather.mist_rise,
        };
        if *volume != want {
            *volume = want;
        }
        let at = to_env.transform_point3(eye - Vec3::Y * 20.0);
        if transform.translation != at {
            transform.translation = at;
        }
    }

    // This frame's ground: the buffer not read by last frame, snapped to whole texels.
    let index = buffers.last.map_or(0, |(i, _)| 1 - i);
    let size = GROUND_SIZE;
    let half = (size / 2) as f32 * GROUND_TEXEL;
    let origin = ((eye.xz() - Vec2::splat(half)) / GROUND_TEXEL).floor() * GROUND_TEXEL;
    if buffers.ground[index].is_none() {
        let buffer: Buffer<f32> = rd.create_device_buffer(
            (size * size) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        // No ground anywhere until the first pass lands (device memory starts as whatever).
        let handle = buffer.handle;
        frame.push(move |rd, _, cmd, _| unsafe {
            rd.device
                .cmd_fill_buffer(cmd, handle, 0, vk::WHOLE_SIZE, GROUND_NONE.to_bits());
        });
        buffers.ground[index] = Some(buffer);
    }
    let heights = buffers.ground[index].as_ref().unwrap().address;
    *ground = GroundHeights {
        address: heights,
        origin,
        texel: GROUND_TEXEL,
        size,
    };
    let push = GroundPush {
        tlas: 0,
        materials: 0,
        heights,
        origin: origin.to_array(),
        texel: GROUND_TEXEL,
        size,
        top: eye.y + 150.0,
        depth: 450.0,
        mask: environment_mask(layers, world) as u32,
        pad: 0,
    };
    let module = kernels.ground.clone();
    frame.push_after_tlas(move |rd, modules, cmd, ctx| {
        let (Some(module), Some(at)) = (modules.get(&module), ctx.tlas()) else {
            return;
        };
        let push = GroundPush {
            tlas: at.tlas,
            materials: at.materials,
            ..push
        };
        record_dispatch(rd, cmd, module, "ground_heights", &push, size * size, None);
    });
    buffers.last = Some((index, origin));

    // Rain: each drop casts its path against the TLAS built last frame (the one standing now;
    // none yet on the first frames, when drops fall to the box's floor).
    let (Ok(&slots), Ok(&splash_slots), Ok(&ripple_slots)) = (
        rain_slots.get(parts.rain),
        rain_slots.get(parts.splash),
        rain_slots.get(parts.ripples),
    ) else {
        return;
    };
    let count = ((weather.rain.clamp(0.0, 1.0) * slots.count as f32) as u32).min(slots.count);
    if count == 0 && buffers.drops.is_none() {
        return;
    }
    let drops = buffers
        .drops
        .get_or_insert_with(|| {
            let mut b =
                rd.create_host_buffer(RAIN_CAPACITY as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
            rd.map_buffer(&mut b).as_slice_mut().fill([0.0; DROP_WORDS]);
            b
        })
        .address;
    let rows_buffer = buffers.rain_rows.get_or_insert_with(|| {
        rd.create_device_buffer(
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )
    });
    let (rows_handle, rows_address) = (rows_buffer.handle, rows_buffer.address);
    let push = RainPush {
        drops,
        rows: rows_address,
        tlas: 0,
        materials: 0,
        count,
        frame: buffers.frame,
        dt: time.delta_secs().min(1.0 / 20.0),
        time: time.elapsed_secs_wrapped(),
        camera: eye.to_array(),
        radius: weather.rain_radius.max(1.0),
        height: weather.rain_height.max(1.0),
        speed: weather.rain_speed,
        wind: weather.wind.xz().to_array(),
        length: weather.rain_length,
        width: weather.rain_width,
        mask: environment_mask(layers, world) as u32,
        pixel_angle,
    };
    let module = kernels.rain.clone();
    frame.push(move |rd, modules, cmd, ctx| {
        // The blocks' rows as the instance arrays stand now; a block still resolving waits.
        let (Some(module), Some(drops), Some(splashes), Some(ripples)) = (
            modules.get(&module),
            ctx.block_rows(slots),
            ctx.block_rows(splash_slots),
            ctx.block_rows(ripple_slots),
        ) else {
            return;
        };
        let rows = RainRows {
            drops,
            splashes,
            ripples,
        };
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, rows_handle, 0, bytemuck::bytes_of(&rows));
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        // Each drop casts its path against the TLAS as it stands (last frame's build).
        let at = ctx.tlas().unwrap_or_default();
        let push = RainPush {
            tlas: at.tlas,
            materials: at.materials,
            ..push
        };
        let threads = drops
            .count
            .min(splashes.count / SPLASH_PARTS)
            .min(ripples.count)
            .min(RAIN_CAPACITY);
        record_dispatch(rd, cmd, module, "rain_step", &push, threads, None);
    });
    tlas.request_rebuild();
}

fn free_buffers(world: &mut World) {
    let Some(mut buffers) = world.remove_resource::<WeatherBuffers>() else {
        return;
    };
    let rd = world.resource::<RenderDevice>();
    for buffer in buffers.ground.iter_mut().filter_map(Option::take) {
        rd.destroyer.destroy_buffer(buffer.handle);
    }
    if let Some(buffer) = buffers.drops.take() {
        rd.destroyer.destroy_buffer(buffer.handle);
    }
    if let Some(buffer) = buffers.rain_rows.take() {
        rd.destroyer.destroy_buffer(buffer.handle);
    }
}

pub struct WeatherPlugin;

impl Plugin for WeatherPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Weather>()
            .init_resource::<WeatherBuffers>()
            .init_resource::<GroundHeights>()
            .add_observer(spawn_parts)
            .add_observer(despawn_parts)
            .add_systems(
                PostUpdate,
                (
                    tint_rain,
                    update_weather.run_if(resource_exists::<RenderDevice>),
                ),
            )
            .add_systems(TeardownSchedule, free_buffers.before(on_shutdown));
    }

    fn finish(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>().clone();
        let ground = asset_server.add(ComputeModule::new(
            asset_server.load(aurora_asset("shaders/weather_ground.slang")),
            &["ground_heights"],
        ));
        let rain = asset_server.add(ComputeModule::new(
            asset_server.load(aurora_asset("shaders/weather_rain.slang")),
            &["rain_step"],
        ));
        let mut materials = app.world_mut().resource_mut::<Assets<AuroraMaterial>>();
        // A streak is a drop smeared over the exposure, scattering every lamp's light around it:
        // a pale translucent scatterer, not a glass rod (which shows only what lies behind it,
        // and vanishes against a dark wall).
        let rain_material = materials.add(AuroraMaterial {
            base_color: Weather::CLEAR.rain_color,
            perceptual_roughness: 0.4,
            diffuse_transmission: 0.6,
            ..default()
        });
        // A ripple reads by the sky it mirrors: clear water.
        let ripple_material = materials.add(AuroraMaterial {
            base_color: Color::srgb(0.9, 0.93, 1.0),
            perceptual_roughness: 0.1,
            specular_transmission: 0.9,
            ior: 1.33,
            ..default()
        });
        let ripple_mesh = app
            .world_mut()
            .resource_mut::<Assets<AuroraMesh>>()
            .add(AuroraMesh::from_shape(Torus::new(0.84, 1.0)));
        app.insert_resource(WeatherKernels {
            ground,
            rain,
            rain_material,
            ripple_material,
            ripple_mesh,
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn push_blocks_fit() {
        use crate::compute::COMPUTE_PUSH_CONSTANT_SIZE;
        assert!(size_of::<super::RainPush>() as u32 <= COMPUTE_PUSH_CONSTANT_SIZE);
        assert_eq!(size_of::<super::GroundPush>(), 56);
        assert_eq!(size_of::<super::RainPush>(), 96);
        assert_eq!(size_of::<super::RainRows>(), 96);
    }
}
