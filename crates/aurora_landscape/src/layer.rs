//! The stacks' layers: children of a [`Landscape`], each stack evaluated in child order. A
//! child may sit in both (an imported tile: heights and a splatmap over the same rectangle).

use ash::vk;
use bevy::{ecs::lifecycle::Remove, prelude::*};
use bevy_aurora::{
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_texture::DataImages,
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
    #[reflect(@0.0..=1.0_f32)]
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
    #[reflect(@0.0..=1.0_f32)]
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
    #[reflect(@0.0..=45.0_f32)]
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

/// A splatmap layer's texels on the GPU: a coarse copy, always held (what LOD rings read),
/// and the full texels while the layer is near the finest window ([`stream_splatmaps`]).
#[derive(Component)]
pub struct SplatmapSource {
    fine: Option<SplatTexels>,
    coarse: SplatTexels,
    images: (AssetId<Image>, AssetId<Image>),
}

/// One copy of a splatmap's ids and weights.
pub struct SplatTexels {
    ids: Buffer<u32>,
    weights: Buffer<u32>,
    pub width: u32,
    pub height: u32,
}

impl SplatTexels {
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

impl SplatmapSource {
    /// The texels a level evaluates from: the finest level the full ones when held (the
    /// coarse until they are), a ring the coarse (a ring's texel is at least as large).
    pub fn texels(&self, finest: bool) -> &SplatTexels {
        match &self.fine {
            Some(fine) if finest => fine,
            _ => &self.coarse,
        }
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        self.coarse.destroy(rd);
        if let Some(mut fine) = self.fine.take() {
            fine.destroy(rd);
        }
    }
}

/// The coarse copy keeps every `COARSE_STEP`th texel each way: the first LOD ring's texel. Ids
/// are picked, never averaged (a palette id has no mean).
const COARSE_STEP: u32 = 2;
/// Full texels load for layers within this many pages of the finest window, and go once a
/// layer is `FINE_RELEASE_PAGES` away.
const FINE_MARGIN_PAGES: f32 = 1.0;
const FINE_RELEASE_PAGES: f32 = 3.0;

fn coarse_words(words: &[u32], width: u32, height: u32) -> (Vec<u32>, u32, u32) {
    let (cw, ch) = (width.div_ceil(COARSE_STEP), height.div_ceil(COARSE_STEP));
    let mut out = Vec::with_capacity((cw * ch) as usize);
    for y in 0..ch {
        let sy = (y * COARSE_STEP + COARSE_STEP / 2).min(height - 1);
        for x in 0..cw {
            let sx = (x * COARSE_STEP + COARSE_STEP / 2).min(width - 1);
            out.push(words[(sy * width + sx) as usize]);
        }
    }
    (out, cw, ch)
}

fn fine_texels(rd: &RenderDevice, ids: &Image, weights: &Image) -> Option<SplatTexels> {
    let (id_words, weight_words) = (r32_words(ids)?, r32_words(weights)?);
    Some(SplatTexels {
        ids: words_buffer(rd, id_words),
        weights: words_buffer(rd, weight_words),
        width: ids.width(),
        height: ids.height(),
    })
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
        let (w, h) = (id_image.width(), id_image.height());
        let (coarse_ids, cw, ch) = coarse_words(id_words, w, h);
        let (coarse_weights, _, _) = coarse_words(weight_words, w, h);
        // The full texels come with `stream_splatmaps`, near the finest window.
        let fresh = SplatmapSource {
            fine: None,
            coarse: SplatTexels {
                ids: words_buffer(&rd, &coarse_ids),
                weights: words_buffer(&rd, &coarse_weights),
                width: cw,
                height: ch,
            },
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

/// A height or splatmap layer's images are data, read into buffers here: aurora never uploads
/// them as textures (a continent's splatmaps were 7 GB of unsampled mip chains).
pub fn layer_images_are_data(
    insert: On<Insert<(HeightmapLayer, SplatmapLayer)>>,
    layers: Query<(Option<&HeightmapLayer>, Option<&SplatmapLayer>)>,
    data: Option<ResMut<DataImages>>,
) {
    let (Some(mut data), Ok((heightmap, splatmap))) = (data, layers.get(insert.entity)) else {
        return;
    };
    if let Some(heightmap) = heightmap {
        data.0.insert(heightmap.image.id());
    }
    if let Some(splatmap) = splatmap {
        data.0.insert(splatmap.ids.id());
        data.0.insert(splatmap.weights.id());
    }
}

/// Full splatmap texels near the finest window: loaded for the layers within
/// `FINE_MARGIN_PAGES` of its pages (every layer, when the landscape doesn't stream), dropped
/// past `FINE_RELEASE_PAGES`. Rings read the coarse copy, so a streaming continent holds its
/// full texels only around the viewer. Loading one re-evaluates the pages under it (a change
/// of the source); dropping one does not (no page the finest level holds lies under it).
pub fn stream_splatmaps(
    rd: Res<RenderDevice>,
    images: Res<Assets<Image>>,
    landscapes: Query<&Landscape, Without<crate::ring::LandscapeRing>>,
    mut layers: Query<(
        &ChildOf,
        &SplatmapLayer,
        &MaterialFootprint,
        &mut SplatmapSource,
    )>,
) {
    for (child_of, layer, footprint, mut source) in &mut layers {
        let Ok(landscape) = landscapes.get(child_of.parent()) else {
            continue;
        };
        let page = landscape.page_size();
        let window = Rect::from_corners(
            landscape.pages_min.as_vec2() * page,
            landscape.pages_max.as_vec2() * page,
        );
        let near = |margin: f32| {
            landscape.stream_radius == 0
                || !footprint
                    .0
                    .intersect(window.inflate(margin * page))
                    .is_empty()
        };
        if source.fine.is_none() && near(FINE_MARGIN_PAGES) {
            let (Some(ids), Some(weights)) = (images.get(&layer.ids), images.get(&layer.weights))
            else {
                continue;
            };
            if let Some(fine) = fine_texels(&rd, ids, weights) {
                source.fine = Some(fine);
            }
        } else if source.fine.is_some() && !near(FINE_RELEASE_PAGES) {
            let source = source.bypass_change_detection();
            if let Some(mut fine) = source.fine.take() {
                fine.destroy(&rd);
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

/// Each stack's layers in child order, as last evaluated.
#[derive(Component, Default)]
pub struct StackOrder {
    heights: Vec<Entity>,
    materials: Vec<Entity>,
}

/// A stack whose layers were reordered or removed re-evaluates everywhere.
#[allow(clippy::type_complexity)]
pub fn track_stack_order(
    mut commands: Commands,
    mut landscapes: Query<
        (Entity, &Children, &mut LandscapePages, Option<&mut StackOrder>),
        (With<Landscape>, Changed<Children>),
    >,
    heights: Query<(), With<HeightLayer>>,
    materials: Query<(), With<MaterialLayer>>,
) {
    for (entity, children, mut pages, order) in &mut landscapes {
        let h: Vec<Entity> = children.iter().filter(|c| heights.contains(*c)).collect();
        let m: Vec<Entity> = children.iter().filter(|c| materials.contains(*c)).collect();
        let Some(mut order) = order else {
            commands.entity(entity).insert(StackOrder {
                heights: h,
                materials: m,
            });
            continue;
        };
        if order.heights != h {
            pages.dirty_all();
            order.heights = h;
        }
        if order.materials != m {
            pages.dirty_all_materials();
            order.materials = m;
        }
    }
}
