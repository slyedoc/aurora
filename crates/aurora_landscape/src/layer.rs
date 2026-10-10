//! The stacks' layers: children of a [`Landscape`], each stack evaluated in child order. A
//! child may sit in both (an imported tile: heights and a splatmap over the same rectangle).

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};
use wgpu_types::TextureFormat;

use crate::{
    Landscape,
    page::LandscapePages,
    sculpt::{PaintPages, SculptPages},
};

/// How a layer combines with the stack below it, scaled by its opacity and falloff.
#[derive(Reflect, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeightBlend {
    #[default]
    Replace,
    Add,
    Max,
    Min,
}

/// One entry of a landscape's height stack.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct HeightLayer {
    pub enabled: bool,
    pub opacity: f32,
    pub blend: HeightBlend,
}

impl Default for HeightLayer {
    fn default() -> Self {
        Self {
            enabled: true,
            opacity: 1.0,
            blend: HeightBlend::Replace,
        }
    }
}

/// A heightmap laid over the landscape: an imported tile (`Replace`, no falloff) or a stamp
/// (`Add` / `Max` with falloff). Centred on its `Transform` in the landscape's frame: the
/// translation's y lifts it, the yaw turns it. Image values (0..1 for unorm formats) map to
/// `height_min..height_max`.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(HeightLayer)]
pub struct HeightmapLayer {
    pub image: Handle<Image>,
    /// Extent in metres (x, z); the first and last samples sit on the edges.
    pub size: Vec2,
    pub height_min: f32,
    pub height_max: f32,
    /// Edge fade, as a fraction of the half extent (0 = hard edge).
    pub falloff: f32,
}

impl Default for HeightmapLayer {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            size: Vec2::splat(256.0),
            height_min: 0.0,
            height_max: 100.0,
            falloff: 0.0,
        }
    }
}

/// Brush strokes: a sparse set of delta pages ([`crate::SculptPages`]), always added. `file`
/// is where they are saved (a `.sculpt` asset); its contents load into the pages.
#[derive(Component, Reflect, Clone, Default, Debug)]
#[reflect(Component, Default)]
#[require(HeightLayer, SculptPages)]
pub struct SculptLayer {
    pub file: Handle<crate::sculpt::LayerPagesFile>,
}

/// A heightmap layer's samples on the GPU, as f32.
#[derive(Component)]
pub struct HeightmapSource {
    buffer: Buffer<f32>,
    pub width: u32,
    pub height: u32,
    image: AssetId<Image>,
}

impl HeightmapSource {
    pub fn address(&self) -> u64 {
        self.buffer.address
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        if self.buffer.handle != vk::Buffer::null() {
            rd.destroyer.destroy_buffer(self.buffer.handle);
            self.buffer = Buffer::default();
        }
    }
}

/// The landscape-local rectangle a layer last covered.
#[derive(Component, Clone, Copy, Debug)]
pub struct LayerFootprint(pub Rect);

/// An image's first channel as f32, unorm formats in 0..1.
pub fn image_samples(image: &Image) -> Option<Vec<f32>> {
    let data = image.data.as_ref()?;
    let n = (image.width() * image.height()) as usize;
    let unorm8 = |stride: usize| (0..n).map(|i| data[i * stride] as f32 / 255.0).collect();
    let unorm16 = |stride: usize| {
        (0..n)
            .map(|i| {
                let o = i * stride;
                u16::from_le_bytes([data[o], data[o + 1]]) as f32 / 65535.0
            })
            .collect()
    };
    let float32 = |stride: usize| {
        (0..n)
            .map(|i| {
                let o = i * stride;
                f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
            })
            .collect()
    };
    let samples = match image.texture_descriptor.format {
        TextureFormat::R8Unorm => unorm8(1),
        TextureFormat::Rg8Unorm => unorm8(2),
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => unorm8(4),
        TextureFormat::R16Unorm => unorm16(2),
        TextureFormat::Rgba16Unorm => unorm16(8),
        TextureFormat::R32Float => float32(4),
        TextureFormat::Rgba32Float => float32(16),
        other => {
            log::warn!("heightmap layer: unsupported image format {other:?}");
            return None;
        }
    };
    (data.len() >= n).then_some(samples)
}

/// Heightmap images become f32 buffers once loaded (and again when the image changes).
pub fn upload_heightmaps(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    images: Res<Assets<Image>>,
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut layers: Query<(Entity, &HeightmapLayer, Option<&mut HeightmapSource>)>,
) {
    let modified: Vec<AssetId<Image>> = image_events
        .read()
        .filter_map(|e| match e {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    for (entity, layer, source) in &mut layers {
        let id = layer.image.id();
        let stale = source
            .as_ref()
            .is_none_or(|s| s.image != id || modified.contains(&id));
        if !stale {
            continue;
        }
        let Some(image) = images.get(id) else {
            continue;
        };
        let Some(samples) = image_samples(image) else {
            continue;
        };
        let mut buffer: Buffer<f32> =
            rd.create_host_buffer(samples.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
        rd.map_buffer(&mut buffer).copy_from_slice(&samples);
        let fresh = HeightmapSource {
            buffer,
            width: image.width(),
            height: image.height(),
            image: id,
        };
        match source {
            Some(mut source) => {
                source.destroy(&rd);
                *source = fresh;
            }
            None => {
                commands.entity(entity).insert(fresh);
            }
        }
    }
}

/// The landscape-local rectangle a layer covers.
/// A rectangle of `size` centred on the transform, or everywhere (a streaming window slides
/// over it).
fn footprint(_landscape: &Landscape, transform: &Transform, size: Option<Vec2>) -> Rect {
    let Some(size) = size else {
        return Rect::from_corners(Vec2::splat(-1.0e9), Vec2::splat(1.0e9));
    };
    let half = size * 0.5;
    let corners = [
        Vec3::new(-half.x, 0.0, -half.y),
        Vec3::new(half.x, 0.0, -half.y),
        Vec3::new(-half.x, 0.0, half.y),
        Vec3::new(half.x, 0.0, half.y),
    ];
    let yaw = Quat::from_rotation_y(transform.rotation.to_euler(EulerRot::YXZ).0);
    let mut rect = Rect::from_center_size(transform.translation.xz(), Vec2::ZERO);
    for c in corners {
        rect = rect.union_point(transform.translation.xz() + (yaw * c).xz());
    }
    rect
}

/// A changed layer dirties the pages under where it was and where it is now.
#[allow(clippy::type_complexity)]
pub fn track_layers(
    mut commands: Commands,
    mut landscapes: Query<(&Landscape, &mut LandscapePages)>,
    layers: Query<
        (
            Entity,
            &ChildOf,
            &Transform,
            Option<&HeightmapLayer>,
            Option<&LayerFootprint>,
        ),
        (
            With<HeightLayer>,
            Or<(
                Changed<HeightLayer>,
                Changed<Transform>,
                Changed<HeightmapLayer>,
                Changed<HeightmapSource>,
                Changed<crate::generator::GeneratorSource>,
                Changed<ChildOf>,
            )>,
        ),
    >,
) {
    for (entity, child_of, transform, heightmap, old) in &layers {
        let Ok((landscape, mut pages)) = landscapes.get_mut(child_of.parent()) else {
            continue;
        };
        let new = footprint(landscape, transform, heightmap.map(|h| h.size));
        let texel = Vec2::splat(landscape.texel_size);
        let grow = |r: Rect| Rect::from_corners(r.min - texel, r.max + texel);
        pages.mark_dirty(landscape.pages_in(grow(new)));
        if let Some(old) = old {
            pages.mark_dirty(landscape.pages_in(grow(old.0)));
        }
        commands.entity(entity).insert(LayerFootprint(new));
    }
}

/// One entry of a landscape's material stack, evaluated after the heights.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct MaterialLayer {
    pub enabled: bool,
    pub opacity: f32,
}

impl Default for MaterialLayer {
    fn default() -> Self {
        Self {
            enabled: true,
            opacity: 1.0,
        }
    }
}

/// Per texel the top four palette ids and weights over a rectangle centred on the
/// `Transform`, like [`HeightmapLayer`]; samples at texel centres. Two images of equal size,
/// a byte per slot (`R32Uint`, or RGBA8 as in a PNG, r = slot 0): `ids` and `weights` (any
/// scale, normalised per texel).
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default)]
#[require(MaterialLayer)]
pub struct SplatmapLayer {
    pub ids: Handle<Image>,
    pub weights: Handle<Image>,
    pub size: Vec2,
}

/// One palette material where the slope (degrees) and height (landscape-local metres) fall
/// inside their windows, ramping over the fades: autoterrain as a layer.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(MaterialLayer)]
pub struct MaterialRuleLayer {
    pub material: u8,
    pub slope: Vec2,
    pub slope_fade: f32,
    pub height: Vec2,
    pub height_fade: f32,
}

impl Default for MaterialRuleLayer {
    fn default() -> Self {
        Self {
            material: 1,
            slope: Vec2::new(35.0, 90.0),
            slope_fade: 5.0,
            height: Vec2::new(-1.0e6, 1.0e6),
            height_fade: 1.0,
        }
    }
}

/// Painted materials: a sparse set of pages ([`crate::PaintPages`]) blended by coverage.
/// `file` is where they are saved (a `.paint` asset).
#[derive(Component, Reflect, Clone, Default, Debug)]
#[reflect(Component, Default)]
#[require(MaterialLayer, PaintPages)]
pub struct PaintLayer {
    pub file: Handle<crate::sculpt::LayerPagesFile>,
}

/// A splatmap layer's texels on the GPU.
#[derive(Component)]
pub struct SplatmapSource {
    ids: Buffer<u32>,
    weights: Buffer<u32>,
    pub width: u32,
    pub height: u32,
    images: (AssetId<Image>, AssetId<Image>),
}

impl SplatmapSource {
    pub fn ids_address(&self) -> u64 {
        self.ids.address
    }

    pub fn weights_address(&self) -> u64 {
        self.weights.address
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for buffer in [&mut self.ids, &mut self.weights] {
            if buffer.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(buffer.handle);
                *buffer = Buffer::default();
            }
        }
    }
}

/// The landscape-local rectangle a material layer last covered.
#[derive(Component, Clone, Copy, Debug)]
pub struct MaterialFootprint(pub Rect);

fn r32_words(image: &Image) -> Option<&[u32]> {
    if !matches!(
        image.texture_descriptor.format,
        TextureFormat::R32Uint | TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        log::warn!(
            "splatmap layer: images must be R32Uint or RGBA8, got {:?}",
            image.texture_descriptor.format
        );
        return None;
    }
    let data = image.data.as_ref()?;
    let n = (image.width() * image.height()) as usize;
    (data.len() >= n * 4).then(|| &bytemuck::cast_slice(&data[..n * 4])[..])
}

fn words_buffer(rd: &RenderDevice, words: &[u32]) -> Buffer<u32> {
    let mut buffer: Buffer<u32> =
        rd.create_host_buffer(words.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
    rd.map_buffer(&mut buffer).copy_from_slice(words);
    buffer
}

/// Splatmap images become buffers once loaded (and again when either changes).
pub fn upload_splatmaps(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    images: Res<Assets<Image>>,
    mut image_events: MessageReader<AssetEvent<Image>>,
    mut layers: Query<(Entity, &SplatmapLayer, Option<&mut SplatmapSource>)>,
) {
    let modified: Vec<AssetId<Image>> = image_events
        .read()
        .filter_map(|e| match e {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    for (entity, layer, source) in &mut layers {
        let ids = (layer.ids.id(), layer.weights.id());
        let stale = source.as_ref().is_none_or(|s| {
            s.images != ids || modified.contains(&ids.0) || modified.contains(&ids.1)
        });
        if !stale {
            continue;
        }
        let (Some(id_image), Some(weight_image)) = (images.get(ids.0), images.get(ids.1)) else {
            continue;
        };
        if id_image.size() != weight_image.size() {
            log::warn!("splatmap layer: ids and weights differ in size");
            continue;
        }
        let (Some(id_words), Some(weight_words)) = (r32_words(id_image), r32_words(weight_image))
        else {
            continue;
        };
        let fresh = SplatmapSource {
            ids: words_buffer(&rd, id_words),
            weights: words_buffer(&rd, weight_words),
            width: id_image.width(),
            height: id_image.height(),
            images: ids,
        };
        match source {
            Some(mut source) => {
                source.destroy(&rd);
                *source = fresh;
            }
            None => {
                commands.entity(entity).insert(fresh);
            }
        }
    }
}

/// A changed material layer dirties the materials under where it was and where it is now.
#[allow(clippy::type_complexity)]
pub fn track_materials(
    mut commands: Commands,
    mut landscapes: Query<(&Landscape, &mut LandscapePages)>,
    layers: Query<
        (
            Entity,
            &ChildOf,
            &Transform,
            Option<&SplatmapLayer>,
            Option<&MaterialFootprint>,
        ),
        (
            With<MaterialLayer>,
            Or<(
                Changed<MaterialLayer>,
                Changed<Transform>,
                Changed<SplatmapLayer>,
                Changed<SplatmapSource>,
                Changed<MaterialRuleLayer>,
                Changed<ChildOf>,
            )>,
        ),
    >,
) {
    for (entity, child_of, transform, splatmap, old) in &layers {
        let Ok((landscape, mut pages)) = landscapes.get_mut(child_of.parent()) else {
            continue;
        };
        let new = footprint(landscape, transform, splatmap.map(|s| s.size));
        pages.mark_materials_dirty(landscape.pages_in(new));
        if let Some(old) = old {
            pages.mark_materials_dirty(landscape.pages_in(old.0));
        }
        commands.entity(entity).insert(MaterialFootprint(new));
    }
}

pub fn on_splatmap_removed(
    remove: On<Remove<SplatmapSource>>,
    mut sources: Query<&mut SplatmapSource>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(mut source) = sources.get_mut(remove.entity)
    {
        source.destroy(&rd);
    }
}

pub fn release_splatmaps(mut sources: Query<&mut SplatmapSource>, rd: Res<RenderDevice>) {
    for mut source in &mut sources {
        source.destroy(&rd);
    }
}

pub fn on_source_removed(
    remove: On<Remove<HeightmapSource>>,
    mut sources: Query<&mut HeightmapSource>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(mut source) = sources.get_mut(remove.entity)
    {
        source.destroy(&rd);
    }
}

pub fn release_sources(mut sources: Query<&mut HeightmapSource>, rd: Res<RenderDevice>) {
    for mut source in &mut sources {
        source.destroy(&rd);
    }
}
