//! Fog volumes: local pockets of fog (mist over a lake, haze in a hollow, the air in a
//! forest that the sun cuts shafts through).
//!
//! A [`FogVolume`] is a box or an ellipsoid of `half_extents` in its entity's frame (scale
//! applies), adding its density to the environment's [`Fog`](crate::sky::Fog) inside it. The
//! edge fades over `edge` of the way in, and wind-blown value noise can carve it into wisps.
//! Every environment sees every volume.
//!
//! GPU side: a host table of [`GpuFogVolume`] rows (world-to-unit transform and the
//! parameters), address + count in the frame uniform. The camera ray marches the fog
//! (`fogMarch` in raygen.slang); bounces delta-track it against the summed densities of the
//! volumes they cross.

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    ray_render_plugin::RenderSet,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Volume shape: the unit cube or sphere, stretched to `half_extents`.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FogShape {
    #[default]
    Box,
    Ellipsoid,
    /// A layer on the ground across the box (`thickness` deep over the terrain, which
    /// [`GroundHeights`] samples around the camera), fading out toward the box's sides: mist.
    Ground,
}

/// A pocket of fog, added to the environment's.
#[derive(Component, Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct FogVolume {
    pub shape: FogShape,
    /// Half its size along local x, y, z (metres, before the transform's scale).
    pub half_extents: Vec3,
    /// Extinction per metre at its heart.
    #[reflect(@0.0..=0.5_f32)]
    pub density: f32,
    /// How far in from the surface the density takes to reach full, as a share of the way
    /// to the centre (0 = a hard edge).
    #[reflect(@0.0..=1.0_f32)]
    pub edge: f32,
    /// Size of the wisps (metres).
    #[reflect(@0.1..=200.0_f32)]
    pub noise_scale: f32,
    /// How much the noise carves away (0 = even, 1 = wisps and gaps).
    #[reflect(@0.0..=1.0_f32)]
    pub noise_strength: f32,
    /// The wisps drift with this (metres per second).
    pub wind: Vec3,
    /// `Ground`: how deep the layer lies over the terrain (metres).
    #[reflect(@0.1..=50.0_f32)]
    pub thickness: f32,
    /// `Ground`: how fast its wisps climb (metres per second).
    #[reflect(@0.0..=5.0_f32)]
    pub rise: f32,
}

impl Default for FogVolume {
    fn default() -> Self {
        Self {
            shape: FogShape::Box,
            half_extents: Vec3::splat(10.0),
            density: 0.02,
            edge: 0.3,
            noise_scale: 12.0,
            noise_strength: 0.0,
            wind: Vec3::new(0.6, 0.0, 0.3),
            thickness: 3.0,
            rise: 0.3,
        }
    }
}

/// One table row; must match `FogVolumeGpu` in rt_types.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuFogVolume {
    /// World to unit-local (rows of a 3x4 affine).
    rows: [[f32; 4]; 3],
    density: f32,
    edge: f32,
    noise_scale: f32,
    noise_strength: f32,
    wind: [f32; 3],
    shape: u32,
    /// `Ground`: the height buffer (0: none yet, no mist) and its mapping.
    heights: u64,
    ground_origin: [f32; 2],
    ground_texel: f32,
    ground_size: u32,
    thickness: f32,
    rise: f32,
}

/// Terrain heights around the camera for [`FogShape::Ground`] volumes: `size` x `size` texels of
/// `texel` metres, texel (0, 0)'s centre at `origin` (xz). The weather plugin keeps it.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct GroundHeights {
    pub address: u64,
    pub origin: Vec2,
    pub texel: f32,
    pub size: u32,
}

/// The uploaded volume table; the frame uniform carries its address + count.
#[derive(Resource, Default)]
pub struct FogVolumeTable {
    buffer: Option<Buffer<GpuFogVolume>>,
    capacity: u64,
    count: u32,
    last: Vec<GpuFogVolume>,
}

impl FogVolumeTable {
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

/// Most volumes the raygen walks per density lookup.
pub const MAX_FOG_VOLUMES: usize = 64;

/// `RenderSet::Prepare`: rebuild + upload the table when a volume moved or changed.
fn upload_fog_volumes(
    render_device: Res<RenderDevice>,
    volumes: Query<(&FogVolume, &GlobalTransform, &InheritedVisibility)>,
    ground: Option<Res<GroundHeights>>,
    mut table: ResMut<FogVolumeTable>,
) {
    let ground = ground.map(|g| *g).unwrap_or_default();
    let mut entries: Vec<GpuFogVolume> = Vec::new();
    for (volume, global, visible) in &volumes {
        if !visible.get() || volume.density <= 0.0 || entries.len() == MAX_FOG_VOLUMES {
            continue;
        }
        let local = global.affine()
            * bevy::math::Affine3A::from_scale(volume.half_extents.max(Vec3::splat(1.0e-3)));
        let m = Mat4::from(local.inverse());
        let rows = [
            m.row(0).to_array(),
            m.row(1).to_array(),
            m.row(2).to_array(),
        ];
        entries.push(GpuFogVolume {
            rows,
            density: volume.density,
            edge: volume.edge.clamp(0.0, 1.0),
            noise_scale: volume.noise_scale.max(0.1),
            noise_strength: volume.noise_strength.clamp(0.0, 1.0),
            wind: volume.wind.to_array(),
            shape: match volume.shape {
                FogShape::Box => 0,
                FogShape::Ellipsoid => 1,
                FogShape::Ground => 2,
            },
            heights: ground.address,
            ground_origin: ground.origin.to_array(),
            ground_texel: ground.texel,
            ground_size: ground.size,
            thickness: volume.thickness.max(0.1),
            rise: volume.rise,
        });
    }
    if entries == table.last && (table.buffer.is_some() || entries.is_empty()) {
        return;
    }

    let needed = entries.len().max(1) as u64;
    // A fresh buffer on every change: a frame still in flight may be reading the old one,
    // which the destroyer frees once it is done.
    {
        if let Some(old) = table.buffer.take() {
            render_device.destroyer.destroy_buffer(old.handle);
        }
        let capacity = needed.next_power_of_two().max(16);
        table.buffer =
            Some(render_device.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER));
        table.capacity = capacity;
    }
    let buffer = table.buffer.as_mut().unwrap();
    let mut mapped = render_device.map_buffer(buffer);
    for (i, entry) in entries.iter().enumerate() {
        mapped[i] = *entry;
    }
    table.count = entries.len() as u32;
    table.last = entries;
}

pub struct FogVolumePlugin;

impl Plugin for FogVolumePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<FogVolume>()
            .init_resource::<FogVolumeTable>()
            .add_systems(Last, upload_fog_volumes.in_set(RenderSet::Prepare));
    }
}

#[cfg(test)]
mod tests {
    /// `FogVolumeGpu` in rt_types.slang: its pointer lands 8-aligned at 80, 112 bytes in all.
    #[test]
    fn gpu_fog_volume_matches_the_shader() {
        assert_eq!(core::mem::offset_of!(super::GpuFogVolume, heights), 80);
        assert_eq!(size_of::<super::GpuFogVolume>(), 112);
    }
}
