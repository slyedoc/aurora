//! Water volumes: the body of water under a surface (a pond, a lake, the sea).
//!
//! A [`WaterVolume`] is a box of `half_extents` in its entity's frame (scale applies; keep it
//! upright: its top face is the water's surface); its [`WaterMaterial3d`] says what the water is
//! like (a [`WaterMaterial`] asset, shared between volumes along with its waves' compute). The surface mesh is only the interface
//! (a transmissive `AuroraMaterial`: Fresnel and refraction, no attenuation of its own); the
//! volume is the medium, homogeneous, crossed analytically (no medium tracking). A path's
//! segment through it draws a free-flight collision against the water's mean extinction and
//! scatters there (Henyey-Greenstein), or carries the transmittance on; shadow rays and ReSTIR
//! PT's replay and reconnection take the transmittance. A shadow ray leaving through the top
//! takes the refracted length and the surface's Fresnel transmittance, as the light it stands for
//! came through the surface bent (the shadow ray itself goes straight through it).
//!
//! [`WaterSurface`] on the surface mesh's entity routes it to the water surface class
//! (water_surface.slang): the volume's waves as its normal (FFT cascades, water_waves.rs), the
//! waves too fine for a ray's footprint folded into its roughness.
//!
//! GPU side: a host table of [`GpuWaterVolume`] rows (world-to-unit transform, the medium, the
//! waves), address + count in the frame uniform (water_types.slang, water.slang).

use ash::vk;
use bevy::{feathers_inspector::Step, prelude::*};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::absorption_from_attenuation,
    ray_render_plugin::RenderSet,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    surface_group::{SurfaceClass, SurfaceGroup, SurfaceGroupRegistry},
    water_waves::WaterWaves,
};

/// A body of water, its surface the box's top face; what the water is like comes from its
/// [`WaterMaterial3d`] (shared: every volume of one material shares its waves' compute).
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
#[require(Transform, Visibility, WaterMaterial3d)]
pub struct WaterVolume {
    /// Half its size along local x, y, z (metres, before the transform's scale).
    #[reflect(@Step(0.1))]
    pub half_extents: Vec3,
}

impl Default for WaterVolume {
    fn default() -> Self {
        Self {
            half_extents: Vec3::splat(10.0),
        }
    }
}

/// What a body of water is like: its medium and its waves. An asset, so many volumes share one
/// (and one set of wave textures).
#[derive(Asset, Reflect, Clone, Debug, PartialEq)]
#[reflect(Default)]
pub struct WaterMaterial {
    /// Absorption: light travelling `attenuation_distance` through the water is tinted to this
    /// (scattering aside).
    pub attenuation_color: Color,
    #[reflect(@Step(0.05))]
    pub attenuation_distance: f32,
    /// Scattering per metre, linear RGB: the murk that turns light aside, and with the
    /// absorption sets the water's colour (0 = clear water that only absorbs).
    #[reflect(@Step(0.01))]
    pub scattering: Vec3,
    /// Henyey-Greenstein g of the murk's scattering (0 even, toward 1 forward).
    #[reflect(@-0.95..=0.95_f32)]
    pub anisotropy: f32,
    /// The surface's index of refraction (for the shadow rays' refracted length and Fresnel).
    #[reflect(@Step(0.001))]
    pub ior: f32,
    /// The waves' direction of travel (xz; its length is ignored).
    #[reflect(@Step(0.05))]
    pub wave_direction: Vec2,
    /// The waves' peak wavelength (metres): where the spectrum's energy peaks.
    #[reflect(@Step(0.05))]
    pub wavelength: f32,
    /// The waves' steepness (the surface's RMS slope times sqrt 2; 0 = a flat surface).
    #[reflect(@0.0..=1.0_f32)]
    pub steepness: f32,
    /// Scales the waves' deep-water speed.
    #[reflect(@Step(0.05))]
    pub wave_speed: f32,
}

impl WaterMaterial {
    /// A murky pond under a light breeze: green-brown, the bottom fading out over a few metres.
    pub const MURKY_POND: Self = Self {
        attenuation_color: Color::linear_rgb(0.2, 0.41, 0.165),
        attenuation_distance: 3.0,
        scattering: Vec3::new(0.15, 0.2, 0.1),
        anisotropy: 0.7,
        ior: 1.333,
        wave_direction: Vec2::new(1.0, 0.4),
        wavelength: 2.0,
        steepness: 0.25,
        wave_speed: 1.0,
    };

    /// A clear mountain lake: blue-green, the bottom in sight ten metres down; longer waves.
    pub const CLEAR_LAKE: Self = Self {
        attenuation_color: Color::linear_rgb(0.45, 0.75, 0.8),
        attenuation_distance: 10.0,
        scattering: Vec3::new(0.02, 0.025, 0.03),
        anisotropy: 0.8,
        wavelength: 3.0,
        steepness: 0.2,
        ..Self::MURKY_POND
    };

    /// A still swamp: brown, thick with silt, gone from sight a hand's depth down.
    pub const SWAMP: Self = Self {
        attenuation_color: Color::linear_rgb(0.3, 0.2, 0.05),
        attenuation_distance: 1.5,
        scattering: Vec3::new(0.9, 0.65, 0.25),
        anisotropy: 0.6,
        wavelength: 1.5,
        steepness: 0.1,
        wave_speed: 0.6,
        ..Self::MURKY_POND
    };
}

impl Default for WaterMaterial {
    fn default() -> Self {
        Self::MURKY_POND
    }
}

/// A [`WaterVolume`]'s material. The default handle is [`WaterMaterial::MURKY_POND`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Deref)]
#[reflect(Component, Default)]
pub struct WaterMaterial3d(pub Handle<WaterMaterial>);

/// Marks a water volume's surface mesh: its instance shades with the water surface class.
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
pub struct WaterSurface;

/// The water surface class (registered at startup).
#[derive(Resource, Clone, Copy)]
pub struct WaterSurfaceClass(pub SurfaceClass);

/// One table row; must match `WaterVolumeGpu` in water_types.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuWaterVolume {
    /// World to unit-local (rows of a 3x4 affine).
    rows: [[f32; 4]; 3],
    /// Per metre.
    absorption: [f32; 3],
    ior: f32,
    scattering: [f32; 3],
    anisotropy: f32,
    wave_direction: [f32; 2],
    wavelength: f32,
    steepness: f32,
    wave_speed: f32,
    /// The two FFT cascades' textures (bindless; 0: none, the analytic waves) and the first
    /// cascade's tile (metres; water_waves.rs).
    wave_tex: [u32; 2],
    wave_tile: f32,
}

/// The uploaded volume table; the frame uniform carries its address + count.
#[derive(Resource, Default)]
pub struct WaterVolumeTable {
    buffer: Option<Buffer<GpuWaterVolume>>,
    count: u32,
    last: Vec<GpuWaterVolume>,
}

impl WaterVolumeTable {
    pub fn address(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.buffer.as_ref().map_or(0, |b| b.address)
        }
    }
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// Most volumes a segment is tested against: the ones nearest the camera.
pub const MAX_WATER_VOLUMES: usize = 64;

/// `RenderSet::Prepare`: rebuild + upload the table when a volume moved or changed.
pub(crate) fn upload_water_volumes(
    render_device: Res<RenderDevice>,
    volumes: Query<(
        &WaterVolume,
        &WaterMaterial3d,
        &GlobalTransform,
        &InheritedVisibility,
    )>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    materials: Res<Assets<WaterMaterial>>,
    waves: Res<WaterWaves>,
    mut table: ResMut<WaterVolumeTable>,
) {
    let eye = cameras
        .iter()
        .find(|(_, camera)| camera.is_active)
        .map_or(Vec3::ZERO, |(t, _)| t.translation());
    let mut nearest: Vec<_> = volumes
        .iter()
        .filter(|(.., visible)| visible.get())
        .map(|entry| {
            let (volume, _, global, _) = entry;
            let local = global.affine().inverse().transform_point3(eye);
            let outside = (local.abs() - volume.half_extents).max(Vec3::ZERO);
            (global.affine().transform_vector3(outside).length_squared(), entry)
        })
        .collect();
    if nearest.len() > MAX_WATER_VOLUMES {
        nearest.select_nth_unstable_by(MAX_WATER_VOLUMES, |a, b| a.0.total_cmp(&b.0));
        nearest.truncate(MAX_WATER_VOLUMES);
    }
    // A stable order, so a still camera re-uploads nothing.
    nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut entries: Vec<GpuWaterVolume> = Vec::new();
    for (_, (volume, material_handle, global, _)) in nearest {
        // A material still loading: no water there yet.
        let Some(material) = materials.get(&material_handle.0) else {
            continue;
        };
        let local = global.affine()
            * bevy::math::Affine3A::from_scale(volume.half_extents.max(Vec3::splat(1.0e-3)));
        let m = Mat4::from(local.inverse());
        let (wave_tex, wave_tile) = waves
            .textures(material_handle.id())
            .unwrap_or(([0, 0], 0.0));
        entries.push(GpuWaterVolume {
            rows: [
                m.row(0).to_array(),
                m.row(1).to_array(),
                m.row(2).to_array(),
            ],
            absorption: absorption_from_attenuation(
                material.attenuation_color.to_linear(),
                material.attenuation_distance,
            ),
            ior: material.ior.max(1.0),
            scattering: material.scattering.max(Vec3::ZERO).to_array(),
            anisotropy: material.anisotropy.clamp(-0.95, 0.95),
            wave_direction: material.wave_direction.to_array(),
            wavelength: material.wavelength.max(0.0),
            steepness: material.steepness.max(0.0),
            wave_speed: material.wave_speed,
            wave_tex,
            wave_tile,
        });
    }
    if entries == table.last && (table.buffer.is_some() || entries.is_empty()) {
        return;
    }

    // A fresh buffer on every change: a frame still in flight may be reading the old one,
    // which the destroyer frees once it is done.
    if let Some(old) = table.buffer.take() {
        render_device.destroyer.destroy_buffer(old.handle);
    }
    let capacity = (entries.len().max(1) as u64).next_power_of_two();
    table.buffer =
        Some(render_device.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER));
    let buffer = table.buffer.as_mut().unwrap();
    let mut mapped = render_device.map_buffer(buffer);
    for (i, entry) in entries.iter().enumerate() {
        mapped[i] = *entry;
    }
    table.count = entries.len() as u32;
    table.last = entries;
}

pub struct WaterVolumePlugin;

impl Plugin for WaterVolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<WaterMaterial>()
            .register_asset_reflect::<WaterMaterial>()
            .register_type::<WaterVolume>()
            .register_type::<WaterMaterial3d>()
            .register_type::<WaterSurface>()
            .init_resource::<WaterVolumeTable>()
            .add_systems(Last, upload_water_volumes.in_set(RenderSet::Prepare))
            .add_observer(tag_water_surface);
    }

    /// After every plugin is built (the surface group registry and the asset server exist),
    /// before anything spawns.
    fn finish(&self, app: &mut App) {
        let closest_hit = app
            .world()
            .resource::<AssetServer>()
            .load(aurora_asset("shaders/water_surface.slang"));
        let class = app
            .world_mut()
            .resource_mut::<SurfaceGroupRegistry>()
            .register(SurfaceGroup {
                label: "water".to_string(),
                closest_hit,
                any_hit: None,
            });
        app.insert_resource(WaterSurfaceClass(class));
        // The default handle's material, so a bare `WaterVolume` is water.
        let _ = app
            .world_mut()
            .resource_mut::<Assets<WaterMaterial>>()
            .insert(
                &Handle::<WaterMaterial>::default(),
                WaterMaterial::default(),
            );
    }
}

/// A [`WaterSurface`] takes the water class.
fn tag_water_surface(
    add: On<Add<WaterSurface>>,
    class: Res<WaterSurfaceClass>,
    mut commands: Commands,
) {
    commands.entity(add.entity).insert(class.0);
}

#[cfg(test)]
mod tests {
    /// The row is read by device address field for field: the Slang struct's scalars, in order.
    #[test]
    fn gpu_water_volume_matches_the_shader() {
        let source = include_str!("../assets/shaders/water_types.slang");
        let body = source
            .split_once("public struct WaterVolumeGpu {")
            .expect("water_types.slang declares WaterVolumeGpu")
            .1;
        let body = &body[..body.find('}').unwrap()];
        let mut floats = 0;
        for line in body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("public "))
        {
            let names = line.trim_end_matches(';').split_once(' ').unwrap().1;
            let (ty, names) = names.split_once(' ').unwrap();
            let n = names.split(',').count();
            floats += match ty {
                "float4" => 4 * n,
                "float" | "uint" => n,
                other => panic!("unexpected field type {other}"),
            };
        }
        assert_eq!(std::mem::size_of::<super::GpuWaterVolume>(), floats * 4);
    }
}
