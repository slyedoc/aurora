//! Shading: tiles of a landscape with a [`LandscapePalette`] trace in the landscape surface
//! class (`landscape/hit.slang`), which reads the material pages at the hit and blends the
//! palette's textures. One parameter row per landscape; its mirror `AuroraMaterial` carries
//! the row to every tile's SBT record. The row is rebuilt each frame and written when it
//! changed: palette textures resolve to bindless slots as aurora finishes uploading them.

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    assets::aurora_asset,
    material::{AuroraMaterial, AuroraMaterial3d},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    surface_group::{SurfaceClass, SurfaceGroup, SurfaceGroupData, SurfaceGroupRegistry},
    vulkan_asset::VulkanAssets,
};
use bytemuck::{Pod, Zeroable};

use crate::{Landscape, LandscapePages, LandscapeTile};

/// No texture in a palette slot (`NO_TEXTURE` in hit.slang).
const NO_TEXTURE: u32 = u32::MAX;
const ROWS: u64 = 64;
/// Palette ids are a byte.
pub const PALETTE_MAX: usize = 256;

/// One palette material. `tile_size` is metres per texture repeat; `clearcoat` a film of
/// water over it (wet stone, mud).
#[derive(Clone, Debug, Reflect)]
pub struct LandscapeMaterial {
    pub albedo: Option<Handle<Image>>,
    pub normal: Option<Handle<Image>>,
    pub tile_size: f32,
    pub roughness: f32,
    pub clearcoat: f32,
}

impl Default for LandscapeMaterial {
    fn default() -> Self {
        Self {
            albedo: None,
            normal: None,
            tile_size: 4.0,
            roughness: 0.9,
            clearcoat: 0.0,
        }
    }
}

/// The materials a landscape's pages name by index (id 0 is what an empty page shows).
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct LandscapePalette(pub Vec<LandscapeMaterial>);

/// `PaletteEntry` in hit.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
struct PaletteGpu {
    albedo: u32,
    normal: u32,
    tile_size: f32,
    roughness: f32,
    clearcoat: f32,
    pad: [u32; 3],
}

/// `LandscapeRow` in hit.slang.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
struct LandscapeRow {
    materials: u64,
    palette: u64,
    min_x: i32,
    min_z: i32,
    pages_x: i32,
    pages_z: i32,
    texel: f32,
    origin: [f32; 3],
    palette_len: u32,
    roughness: f32,
    /// Planet faces (`SphereFace`): face and radius, 0 = flat.
    face: u32,
    radius: f32,
}

#[derive(Resource)]
pub struct LandscapeClass {
    class: SurfaceClass,
    rows: Buffer<LandscapeRow>,
    free: Vec<u32>,
    next: u32,
}

/// A landscape's row, mirror material and palette buffer.
#[derive(Component)]
pub struct LandscapeShading {
    row: u32,
    mirror: Handle<AuroraMaterial>,
    palette: Buffer<PaletteGpu>,
    written: Option<(LandscapeRow, Vec<PaletteGpu>)>,
}

pub fn register_class(
    mut commands: Commands,
    mut registry: ResMut<SurfaceGroupRegistry>,
    mut data: ResMut<SurfaceGroupData>,
    rd: Res<RenderDevice>,
    server: Res<AssetServer>,
) {
    let class = registry.register(SurfaceGroup {
        label: "aurora_landscape".to_string(),
        closest_hit: server.load(aurora_asset("shaders/landscape/hit.slang")),
        any_hit: None,
    });
    let rows: Buffer<LandscapeRow> = rd.create_host_buffer(ROWS, vk::BufferUsageFlags::STORAGE_BUFFER);
    data.set(class, rows.address);
    commands.insert_resource(LandscapeClass {
        class,
        rows,
        free: Vec::new(),
        next: 0,
    });
}

/// Rows and palettes for every landscape with a palette and pages.
#[allow(clippy::type_complexity)]
pub fn sync_shading(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    mut class: ResMut<LandscapeClass>,
    mut data: ResMut<SurfaceGroupData>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    textures: Res<VulkanAssets<Image>>,
    mut landscapes: Query<(
        Entity,
        &Landscape,
        &LandscapePalette,
        &LandscapePages,
        &GlobalTransform,
        Option<&AuroraMaterial3d>,
        Option<&mut LandscapeShading>,
    )>,
) {
    for (entity, landscape, palette, pages, transform, base, shading) in &mut landscapes {
        let mut shading = match shading {
            Some(shading) => shading,
            None => {
                let row = class.free.pop().unwrap_or_else(|| {
                    class.next += 1;
                    class.next - 1
                });
                if row as u64 >= ROWS {
                    log::error!("aurora_landscape: more than {ROWS} shaded landscapes");
                    continue;
                }
                let roughness = base
                    .and_then(|b| materials.get(&b.0))
                    .map_or(0.9, |m| m.perceptual_roughness);
                let mirror = materials.add(AuroraMaterial {
                    perceptual_roughness: roughness,
                    ..default()
                });
                data.set_row(mirror.id(), row);
                commands.entity(entity).insert(LandscapeShading {
                    row,
                    mirror,
                    palette: rd.create_host_buffer(
                        PALETTE_MAX as u64,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                    ),
                    written: None,
                });
                continue;
            }
        };
        let index = |handle: Option<&Handle<Image>>| -> u32 {
            handle
                .and_then(|h| textures.get_by_id(h.id()))
                .map_or(NO_TEXTURE, |t| rd.register_bindless_texture(t))
        };
        let gpu_palette: Vec<PaletteGpu> = palette
            .0
            .iter()
            .take(PALETTE_MAX)
            .map(|m| PaletteGpu {
                albedo: index(m.albedo.as_ref()),
                normal: index(m.normal.as_ref()),
                tile_size: m.tile_size,
                roughness: m.roughness,
                clearcoat: m.clearcoat.clamp(0.0, 1.0),
                pad: [0; 3],
            })
            .collect();
        let pool = pages.pool();
        let origin = transform.translation();
        let row = LandscapeRow {
            materials: pages.materials_address(),
            palette: shading.palette.address,
            min_x: pool.min_x,
            min_z: pool.min_z,
            pages_x: pool.pages_x,
            pages_z: pool.pages_z,
            texel: landscape.texel_size,
            origin: origin.to_array(),
            palette_len: gpu_palette.len() as u32,
            roughness: 0.9,
            face: landscape.sphere.map_or(0, |f| f.face as u32),
            radius: landscape.sphere.map_or(0.0, |f| f.radius),
        };
        if shading
            .written
            .as_ref()
            .is_some_and(|(r, p)| *r == row && *p == gpu_palette)
        {
            continue;
        }
        let at = shading.row as usize;
        rd.map_buffer(&mut shading.palette).as_slice_mut()[..gpu_palette.len()]
            .copy_from_slice(&gpu_palette);
        rd.map_buffer(&mut class.rows).as_slice_mut()[at] = row;
        shading.written = Some((row, gpu_palette));
    }
}

/// Tiles of a shaded landscape trace with its mirror material, in the class.
pub fn wear_class(
    mut commands: Commands,
    class: Res<LandscapeClass>,
    shadings: Query<&LandscapeShading>,
    tiles: Query<(Entity, &LandscapeTile, &AuroraMaterial3d, Option<&SurfaceClass>)>,
) {
    for (entity, tile, material, worn) in &tiles {
        let Ok(shading) = shadings.get(tile.landscape) else {
            continue;
        };
        if material.0 != shading.mirror || worn != Some(&class.class) {
            commands
                .entity(entity)
                .insert((AuroraMaterial3d(shading.mirror.clone()), class.class));
        }
    }
}

pub fn on_shading_removed(
    remove: On<Remove<LandscapeShading>>,
    shadings: Query<&LandscapeShading>,
    class: Option<ResMut<LandscapeClass>>,
    mut data: ResMut<SurfaceGroupData>,
    rd: Option<Res<RenderDevice>>,
) {
    let (Ok(shading), Some(mut class)) = (shadings.get(remove.entity), class) else {
        return;
    };
    data.clear_row(shading.mirror.id());
    class.free.push(shading.row);
    if let Some(rd) = rd
        && shading.palette.handle != vk::Buffer::null()
    {
        rd.destroyer.destroy_buffer(shading.palette.handle);
    }
}

pub fn release_shading(mut shadings: Query<&mut LandscapeShading>, rd: Res<RenderDevice>) {
    for mut shading in &mut shadings {
        if shading.palette.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(shading.palette.handle);
            shading.palette = Buffer::default();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_match_the_shader() {
        assert_eq!(size_of::<LandscapeRow>(), 64);
        assert_eq!(size_of::<PaletteGpu>(), 32);
    }
}
