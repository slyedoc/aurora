//! Water: a [`WaterSurface`] mesh is the interface (a transmissive `AuroraMaterial`: Fresnel and
//! refraction), its [`WaterMaterial3d`] the medium below it. Paths track the medium by their
//! crossings of the surface (water.slang): refracting down through one puts them in its water,
//! up out of it takes them out; shadow rays take the nearest surface they cross.
//!
//! GPU side: one [`GpuWaterMedium`] row per material in use (address + count in the frame
//! uniform); each surface instance carries its row and the water flag in its material record
//! ([`SurfaceRow`]).

use ash::vk;
use bevy::{feathers_inspector::Step, platform::collections::HashMap, prelude::*};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::aurora_asset,
    blas::{absorption_from_attenuation, material_flags},
    ray_render_plugin::RenderSet,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    surface_group::{SurfaceClass, SurfaceGroup, SurfaceGroupRegistry, SurfaceRow},
    water_waves::WaterWaves,
};

/// What a body of water is like: its medium and its waves. An asset, so many surfaces share one
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

    /// A river: clear green, the bed in sight a few metres down; short, quick ripples.
    pub const RIVER: Self = Self {
        attenuation_color: Color::linear_rgb(0.35, 0.65, 0.5),
        attenuation_distance: 5.0,
        scattering: Vec3::new(0.05, 0.07, 0.05),
        anisotropy: 0.75,
        wave_direction: Vec2::new(1.0, 0.2),
        wavelength: 1.2,
        steepness: 0.15,
        wave_speed: 1.3,
        ..Self::MURKY_POND
    };

    /// The open sea: deep blue-green, clear; long swells.
    pub const OCEAN: Self = Self {
        attenuation_color: Color::linear_rgb(0.25, 0.6, 0.7),
        attenuation_distance: 12.0,
        scattering: Vec3::new(0.01, 0.02, 0.025),
        anisotropy: 0.85,
        wave_direction: Vec2::new(1.0, 0.6),
        wavelength: 8.0,
        steepness: 0.25,
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

/// A [`WaterSurface`]'s water. The default handle is [`WaterMaterial::MURKY_POND`].
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Deref)]
#[reflect(Component, Default)]
pub struct WaterMaterial3d(pub Handle<WaterMaterial>);

/// A water surface mesh: its instance shades with the water surface class, and the water under
/// it is its [`WaterMaterial3d`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default)]
#[reflect(Component, Default)]
#[require(WaterMaterial3d)]
pub struct WaterSurface;

/// A preset water for a surface: inserts its kind's shared [`WaterMaterial3d`].
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[reflect(Component, Default)]
#[require(WaterSurface)]
pub enum WaterKind {
    #[default]
    Lake,
    River,
    Ocean,
    Pond,
    Swamp,
}

impl WaterKind {
    pub const ALL: [Self; 5] = [Self::Lake, Self::River, Self::Ocean, Self::Pond, Self::Swamp];

    pub fn material(self) -> WaterMaterial {
        match self {
            Self::Lake => WaterMaterial::CLEAR_LAKE,
            Self::River => WaterMaterial::RIVER,
            Self::Ocean => WaterMaterial::OCEAN,
            Self::Pond => WaterMaterial::MURKY_POND,
            Self::Swamp => WaterMaterial::SWAMP,
        }
    }
}

/// One shared material per [`WaterKind`] (edit one and every surface of its kind follows).
#[derive(Resource, Clone, Debug)]
pub struct WaterPresets(pub HashMap<WaterKind, Handle<WaterMaterial>>);

impl WaterPresets {
    pub fn get(&self, kind: WaterKind) -> Handle<WaterMaterial> {
        self.0[&kind].clone()
    }
}

/// The water surface class (registered at startup).
#[derive(Resource, Clone, Copy)]
pub struct WaterSurfaceClass(pub SurfaceClass);

/// One medium table row; must match `WaterMediumGpu` in water_types.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuWaterMedium {
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

/// Most materials water can be at once: a hit names its water in a byte (row + 1).
pub const MAX_WATER_MEDIA: u32 = 255;

/// The medium table: a row per [`WaterMaterial`] in use, kept while the asset lives; the frame
/// uniform carries its address + count.
#[derive(Resource, Default)]
pub struct WaterMedia {
    rows: HashMap<AssetId<WaterMaterial>, u32>,
    free: Vec<u32>,
    buffer: Option<Buffer<GpuWaterMedium>>,
    count: u32,
    last: Vec<GpuWaterMedium>,
}

impl WaterMedia {
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

    fn row(&mut self, material: AssetId<WaterMaterial>) -> Option<u32> {
        if let Some(row) = self.rows.get(&material) {
            return Some(*row);
        }
        let row = self.free.pop().or_else(|| {
            let next = self.rows.len() as u32;
            (next < MAX_WATER_MEDIA).then_some(next)
        })?;
        self.rows.insert(material, row);
        Some(row)
    }
}

/// Every surface's instance takes its material's row and the water flag.
fn assign_water_rows(
    mut media: ResMut<WaterMedia>,
    mut events: MessageReader<AssetEvent<WaterMaterial>>,
    surfaces: Query<
        (Entity, &WaterMaterial3d, Option<&SurfaceRow>),
        Or<(Changed<WaterMaterial3d>, Added<WaterSurface>)>,
    >,
    mut commands: Commands,
) {
    for event in events.read() {
        if let AssetEvent::Unused { id } | AssetEvent::Removed { id } = event
            && let Some(row) = media.rows.remove(id)
        {
            media.free.push(row);
        }
    }
    for (entity, material, current) in &surfaces {
        let Some(row) = media.row(material.id()) else {
            warn_once!("more than {MAX_WATER_MEDIA} water materials: the rest render as glass");
            continue;
        };
        let want = SurfaceRow {
            row,
            flags: material_flags::WATER,
        };
        if current != Some(&want) {
            commands.entity(entity).insert(want);
        }
    }
}

/// A [`WaterKind`] takes its preset's material.
fn apply_water_kind(
    kinds: Query<(Entity, &WaterKind), Changed<WaterKind>>,
    presets: Res<WaterPresets>,
    mut commands: Commands,
) {
    for (entity, kind) in &kinds {
        commands
            .entity(entity)
            .insert(WaterMaterial3d(presets.get(*kind)));
    }
}

/// `RenderSet::Prepare`: rewrite + upload the table when a row's material or waves changed.
pub(crate) fn upload_water_media(
    render_device: Res<RenderDevice>,
    materials: Res<Assets<WaterMaterial>>,
    waves: Res<WaterWaves>,
    mut media: ResMut<WaterMedia>,
) {
    let count = media.rows.values().map(|r| r + 1).max().unwrap_or(0);
    let mut entries = vec![GpuWaterMedium::default(); count as usize];
    for (&id, &row) in &media.rows {
        // A material still loading: clear water there for now.
        let Some(material) = materials.get(id) else {
            continue;
        };
        let (wave_tex, wave_tile) = waves.textures(id).unwrap_or(([0, 0], 0.0));
        entries[row as usize] = GpuWaterMedium {
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
        };
    }
    if entries == media.last && (media.buffer.is_some() || entries.is_empty()) {
        return;
    }

    // A fresh buffer on every change: a frame still in flight may be reading the old one,
    // which the destroyer frees once it is done.
    if let Some(old) = media.buffer.take() {
        render_device.destroyer.destroy_buffer(old.handle);
    }
    let capacity = (entries.len().max(1) as u64).next_power_of_two();
    let mut buffer =
        render_device.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER);
    let mut mapped = render_device.map_buffer(&mut buffer);
    for (i, entry) in entries.iter().enumerate() {
        mapped[i] = *entry;
    }
    media.buffer = Some(buffer);
    media.count = entries.len() as u32;
    media.last = entries;
}

pub struct WaterVolumePlugin;

impl Plugin for WaterVolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<WaterMaterial>()
            .register_asset_reflect::<WaterMaterial>()
            .register_type::<WaterMaterial3d>()
            .register_type::<WaterSurface>()
            .register_type::<WaterKind>()
            .init_resource::<WaterMedia>()
            .add_systems(PostUpdate, (apply_water_kind, assign_water_rows).chain())
            .add_systems(Last, upload_water_media.in_set(RenderSet::Prepare))
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
        let mut materials = app.world_mut().resource_mut::<Assets<WaterMaterial>>();
        // The default handle's material, so a bare `WaterSurface` is water.
        let _ = materials.insert(
            &Handle::<WaterMaterial>::default(),
            WaterMaterial::default(),
        );
        let presets = WaterKind::ALL
            .into_iter()
            .map(|kind| (kind, materials.add(kind.material())))
            .collect();
        app.insert_resource(WaterPresets(presets));
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
    fn gpu_water_medium_matches_the_shader() {
        let source = include_str!("../assets/shaders/water_types.slang");
        let body = source
            .split_once("public struct WaterMediumGpu {")
            .expect("water_types.slang declares WaterMediumGpu")
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
        assert_eq!(std::mem::size_of::<super::GpuWaterMedium>(), floats * 4);
    }
}
