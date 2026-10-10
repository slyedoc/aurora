//! Brush layers: sparse pages written by GPU brushes (`landscape/brush.slang`). A sculpt
//! layer's pages are height deltas; a paint layer's are material sets whose weights sum to
//! the painted coverage (`landscape/materials.slang`).

use ash::vk;
use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    ecs::lifecycle::Remove,
    platform::collections::HashMap,
    prelude::*,
};
use bevy_aurora::{
    compute::{ComputeModules, compute_to_compute_barrier, record_dispatch},
    render_buffer::{Buffer, BufferProvider, BufferView},
    render_device::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    HeightLayer, Landscape, LandscapeKernels, MaterialLayer, PAGE_TEXELS, PaintLayer, SculptLayer,
    page::{GpuPagePool, LandscapePages},
};

const PAGE_LEN: usize = (PAGE_TEXELS * PAGE_TEXELS) as usize;

#[derive(Reflect, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum BrushKind {
    #[default]
    Raise,
    Lower,
    Smooth,
    Flatten,
    /// Paint `SculptDab::material` (paint layers).
    Paint,
    /// Fade a paint layer's coverage back out.
    Erase,
}

impl BrushKind {
    pub fn paints(self) -> bool {
        matches!(self, Self::Paint | Self::Erase)
    }
}

/// One brush application on a sculpt or paint layer, landscape-local. `strength` is metres
/// for raise / lower and a 0..1 rate otherwise; scale it by the frame time.
#[derive(Clone, Copy, Debug)]
pub struct SculptDab {
    pub layer: Entity,
    pub center: Vec2,
    pub radius: f32,
    pub strength: f32,
    pub kind: BrushKind,
    /// Flatten's height.
    pub target: f32,
    /// Paint's palette id.
    pub material: u8,
}

/// Dabs queued this frame; applied in `PostUpdate`.
#[derive(Resource, Default)]
pub struct SculptDabs(pub Vec<SculptDab>);

/// `BrushPush` in landscape/brush.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushPush {
    delta: u64,
    paint: u64,
    pool: GpuPagePool,
    page_x: i32,
    page_z: i32,
    center_x: f32,
    center_z: f32,
    radius: f32,
    strength: f32,
    target: f32,
    kind: u32,
}

/// What a brush layer stores per texel.
pub trait PageTexel: Pod + Send + Sync + 'static {
    const MAGIC: &'static [u8; 8];
}

impl PageTexel for f32 {
    const MAGIC: &'static [u8; 8] = b"LSCULPT1";
}

impl PageTexel for [u32; 2] {
    const MAGIC: &'static [u8; 8] = b"LPAINT01";
}

struct LayerPage<T> {
    buffer: Buffer<T>,
    view: BufferView<T>,
}

/// A brush layer's pages, host-visible.
#[derive(Component)]
pub struct LayerPages<T: PageTexel> {
    pages: HashMap<IVec2, LayerPage<T>>,
}

/// Height deltas.
pub type SculptPages = LayerPages<f32>;
/// Painted material sets.
pub type PaintPages = LayerPages<[u32; 2]>;

impl<T: PageTexel> Default for LayerPages<T> {
    fn default() -> Self {
        Self {
            pages: HashMap::default(),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PageFormatError {
    #[error("not a layer page file of this kind")]
    Magic,
    #[error("truncated layer page file")]
    Truncated,
}

impl<T: PageTexel> LayerPages<T> {
    pub fn address(&self, page: IVec2) -> Option<u64> {
        self.pages.get(&page).map(|p| p.buffer.address)
    }

    pub fn page(&self, page: IVec2) -> Option<&[T]> {
        self.pages
            .get(&page)
            .map(|p| unsafe { std::slice::from_raw_parts(p.view.as_ptr(), PAGE_LEN) })
    }

    pub fn pages(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.pages.keys().copied()
    }

    pub(crate) fn ensure(&mut self, rd: &RenderDevice, page: IVec2) -> u64 {
        self.pages
            .entry(page)
            .or_insert_with(|| {
                let mut buffer: Buffer<T> =
                    rd.create_host_buffer(PAGE_LEN as u64, vk::BufferUsageFlags::STORAGE_BUFFER);
                let mut view = rd.map_buffer(&mut buffer);
                view.as_slice_mut().fill(T::zeroed());
                LayerPage { buffer, view }
            })
            .buffer
            .address
    }

    /// Sets one page's texels (allocating it), or drops it with `None` (an undo).
    pub fn set_page(&mut self, rd: &RenderDevice, page: IVec2, texels: Option<&[T]>) {
        match texels {
            Some(texels) => {
                self.ensure(rd, page);
                let view = &mut self.pages.get_mut(&page).unwrap().view;
                view.as_slice_mut().copy_from_slice(&texels[..PAGE_LEN]);
            }
            None => {
                if let Some(old) = self.pages.remove(&page) {
                    rd.destroyer.destroy_buffer(old.buffer.handle);
                }
            }
        }
    }

    /// Every page back to zero (the pages stay allocated).
    pub(crate) fn zero(&mut self) {
        for page in self.pages.values_mut() {
            page.view.as_slice_mut().fill(T::zeroed());
        }
    }

    /// The kind's magic, page count, then per page its coordinate and 256² texels (LE).
    pub fn to_bytes(&self) -> Vec<u8> {
        let texel = size_of::<T>();
        let mut out = Vec::with_capacity(12 + self.pages.len() * (8 + PAGE_LEN * texel));
        out.extend_from_slice(T::MAGIC);
        out.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        let mut keys: Vec<IVec2> = self.pages().collect();
        keys.sort_by_key(|p| (p.y, p.x));
        for page in keys {
            out.extend_from_slice(&page.x.to_le_bytes());
            out.extend_from_slice(&page.y.to_le_bytes());
            out.extend_from_slice(bytemuck::cast_slice(self.page(page).unwrap()));
        }
        out
    }

    /// Replaces these pages with the ones in `bytes` ([`Self::to_bytes`]).
    pub fn load(&mut self, rd: &RenderDevice, bytes: &[u8]) -> Result<(), PageFormatError> {
        let loaded = Self::from_bytes(rd, bytes)?;
        self.destroy(rd);
        *self = loaded;
        Ok(())
    }

    fn from_bytes(rd: &RenderDevice, bytes: &[u8]) -> Result<Self, PageFormatError> {
        let rest = bytes.strip_prefix(T::MAGIC).ok_or(PageFormatError::Magic)?;
        let word = |at: usize| -> Result<[u8; 4], PageFormatError> {
            rest.get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .ok_or(PageFormatError::Truncated)
        };
        let count = u32::from_le_bytes(word(0)?) as usize;
        let len = PAGE_LEN * size_of::<T>();
        let mut layer = Self::default();
        let mut at = 4;
        for _ in 0..count {
            let page = IVec2::new(
                i32::from_le_bytes(word(at)?),
                i32::from_le_bytes(word(at + 4)?),
            );
            at += 8;
            let data = rest.get(at..at + len).ok_or(PageFormatError::Truncated)?;
            at += len;
            layer.ensure(rd, page);
            let dst = &mut layer.pages.get_mut(&page).unwrap().view;
            bytemuck::cast_slice_mut::<T, u8>(dst.as_slice_mut()).copy_from_slice(data);
        }
        Ok(layer)
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for (_, page) in self.pages.drain() {
            rd.destroyer.destroy_buffer(page.buffer.handle);
        }
    }
}

/// Runs the frame's dabs on the GPU and dirties the pages they touched.
pub fn apply_dabs(
    rd: Res<RenderDevice>,
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    mut dabs: ResMut<SculptDabs>,
    mut layers: Query<(
        &ChildOf,
        Option<&mut SculptPages>,
        Option<&mut PaintPages>,
    )>,
    mut landscapes: Query<(&Landscape, &mut LandscapePages)>,
) {
    if dabs.0.is_empty() {
        return;
    }
    // Dabs wait (queued) until the brush kernel has compiled.
    let Some(module) = modules.get(&kernels.brush) else {
        return;
    };
    let dabs = std::mem::take(&mut dabs.0);
    let mut jobs: Vec<BrushPush> = Vec::new();
    for dab in dabs {
        let Ok((child_of, mut sculpt, mut paint)) = layers.get_mut(dab.layer) else {
            continue;
        };
        let Ok((landscape, mut pages)) = landscapes.get_mut(child_of.parent()) else {
            continue;
        };
        if !pages.is_complete() {
            continue;
        }
        let rect = Rect::from_center_half_size(dab.center, Vec2::splat(dab.radius));
        for page in landscape.pages_in(rect) {
            let delta = match (dab.kind.paints(), sculpt.as_mut(), paint.as_mut()) {
                (false, Some(sculpt), _) => sculpt.ensure(&rd, page),
                (true, _, Some(paint)) => paint.ensure(&rd, page),
                _ => continue,
            };
            jobs.push(BrushPush {
                delta: if dab.kind.paints() { 0 } else { delta },
                paint: if dab.kind.paints() { delta } else { 0 },
                pool: pages.pool(),
                page_x: page.x,
                page_z: page.y,
                center_x: dab.center.x,
                center_z: dab.center.y,
                radius: dab.radius,
                strength: dab.strength,
                target: if dab.kind.paints() {
                    dab.material as f32
                } else {
                    dab.target
                },
                kind: dab.kind as u32,
            });
            if dab.kind.paints() {
                pages.mark_materials_dirty([page]);
            } else {
                pages.mark_dirty([page]);
            }
        }
    }
    if jobs.is_empty() {
        return;
    }
    rd.run_transfer_commands(|cmd| {
        for job in &jobs {
            record_dispatch(&rd, cmd, module, "sculpt_brush", job, PAGE_LEN as u32, None);
            compute_to_compute_barrier(&rd, cmd);
        }
    });
}

pub fn on_layer_pages_removed<T: PageTexel>(
    remove: On<Remove<LayerPages<T>>>,
    mut layers: Query<&mut LayerPages<T>>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(mut layer) = layers.get_mut(remove.entity)
    {
        layer.destroy(&rd);
    }
}

pub fn release_layer_pages<T: PageTexel>(
    mut layers: Query<&mut LayerPages<T>>,
    rd: Res<RenderDevice>,
) {
    for mut layer in &mut layers {
        layer.destroy(&rd);
    }
}

/// A saved brush layer: [`LayerPages::to_bytes`] on disk (`.sculpt` / `.paint`).
#[derive(Asset, Reflect, Debug, Clone, Default)]
#[reflect(Default)]
pub struct LayerPagesFile(pub Vec<u8>);

#[derive(Default, TypePath)]
pub struct LayerPagesLoader;

impl AssetLoader for LayerPagesLoader {
    type Asset = LayerPagesFile;
    type Settings = ();
    type Error = std::io::Error;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<LayerPagesFile, std::io::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        Ok(LayerPagesFile(bytes))
    }

    fn extensions(&self) -> &[&str] {
        &["sculpt", "paint", "erosion"]
    }
}

/// The files the layers point at, the last time each loaded into its pages.
#[derive(Component)]
pub struct LoadedPagesFile(pub AssetId<LayerPagesFile>);

/// A layer's file, once loaded (and again when it changes on disk), replaces its pages; the
/// stack re-evaluates under it.
#[allow(clippy::type_complexity)]
pub fn load_page_files(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    files: Res<Assets<LayerPagesFile>>,
    mut events: MessageReader<AssetEvent<LayerPagesFile>>,
    mut sculpts: Query<(
        Entity,
        &SculptLayer,
        &mut SculptPages,
        &mut HeightLayer,
        Option<&LoadedPagesFile>,
    )>,
    mut paints: Query<
        (
            Entity,
            &PaintLayer,
            &mut PaintPages,
            &mut MaterialLayer,
            Option<&LoadedPagesFile>,
        ),
        Without<SculptLayer>,
    >,
    mut erosions: Query<
        (
            Entity,
            &crate::ErosionLayer,
            &mut SculptPages,
            &mut HeightLayer,
            Option<&LoadedPagesFile>,
        ),
        (Without<SculptLayer>, Without<PaintLayer>),
    >,
) {
    let modified: Vec<AssetId<LayerPagesFile>> = events
        .read()
        .filter_map(|e| match e {
            AssetEvent::Modified { id } => Some(*id),
            _ => None,
        })
        .collect();
    let fresh = |id: AssetId<LayerPagesFile>, loaded: Option<&LoadedPagesFile>| {
        loaded.is_none_or(|l| l.0 != id) || modified.contains(&id)
    };
    for (entity, layer, mut pages, mut stack, loaded) in &mut sculpts {
        let id = layer.file.id();
        if layer.file == Handle::default() || !fresh(id, loaded) {
            continue;
        }
        let Some(file) = files.get(id) else {
            continue;
        };
        match pages.load(&rd, &file.0) {
            Ok(()) => stack.set_changed(),
            Err(e) => log::error!("sculpt layer {entity}: {e}"),
        }
        commands.entity(entity).insert(LoadedPagesFile(id));
    }
    for (entity, layer, mut pages, mut stack, loaded) in &mut paints {
        let id = layer.file.id();
        if layer.file == Handle::default() || !fresh(id, loaded) {
            continue;
        }
        let Some(file) = files.get(id) else {
            continue;
        };
        match pages.load(&rd, &file.0) {
            Ok(()) => stack.set_changed(),
            Err(e) => log::error!("paint layer {entity}: {e}"),
        }
        commands.entity(entity).insert(LoadedPagesFile(id));
    }
    for (entity, layer, mut pages, mut stack, loaded) in &mut erosions {
        let id = layer.file.id();
        if layer.file == Handle::default() || !fresh(id, loaded) {
            continue;
        }
        let Some(file) = files.get(id) else {
            continue;
        };
        match pages.load(&rd, &file.0) {
            Ok(()) => stack.set_changed(),
            Err(e) => log::error!("erosion layer {entity}: {e}"),
        }
        commands.entity(entity).insert(LoadedPagesFile(id));
    }
}
