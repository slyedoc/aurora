//! A landscape's evaluated pages as saved (`.pages`): heights and materials of every page in
//! the window, written when the scene saves and loaded straight into the pool, so a loaded
//! landscape never evaluates its stack. While [`LandscapeLive`] is set (an editor is up) the
//! stack evaluates as usual and edits show; a bake already seen is not applied again.

use std::io::{Read, Write};

use ash::vk;
use bevy::{
    asset::{AssetLoader, LoadContext, io::Reader},
    prelude::*,
};
use bevy_aurora::{
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

use crate::{
    Landscape, PAGE_TEXELS,
    page::{LandscapePages, MaterialsEvaluated, PagesEvaluated},
};

const MAGIC: &[u8; 4] = b"LSPG";
const VERSION: u32 = 1;
const PAGE_LEN: usize = (PAGE_TEXELS * PAGE_TEXELS) as usize;

/// Where a landscape's bake lives; the scene saves it as the file's path.
#[derive(Component, Reflect, Clone, Default, Debug)]
#[reflect(Component, Default)]
pub struct LandscapeBake {
    pub file: Handle<BakedPages>,
}

/// Evaluate the stacks instead of loading bakes (set while an editor is up).
#[derive(Resource, Default)]
pub struct LandscapeLive(pub bool);

/// A `.pages` file: the window's pages in row order (z, then x).
#[derive(Asset, Reflect, Default)]
pub struct BakedPages {
    pub texel: f32,
    pub min: IVec2,
    pub count: IVec2,
    #[reflect(ignore)]
    pub heights: Vec<f32>,
    #[reflect(ignore)]
    pub materials: Vec<[u32; 2]>,
}

#[derive(Debug, thiserror::Error)]
pub enum BakeError {
    #[error("not a .pages file")]
    Magic,
    #[error(".pages version {0}, expected {VERSION}")]
    Version(u32),
    #[error("truncated .pages file")]
    Truncated,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl BakedPages {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.texel.to_le_bytes());
        for v in [self.min.x, self.min.y, self.count.x, self.count.y] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut encoder = lz4_flex::frame::FrameEncoder::new(out);
        encoder
            .write_all(bytemuck::cast_slice(&self.heights))
            .and_then(|_| encoder.write_all(bytemuck::cast_slice(&self.materials)))
            .expect("writing to a Vec");
        encoder.finish().expect("writing to a Vec")
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BakeError> {
        let rest = bytes.strip_prefix(MAGIC).ok_or(BakeError::Magic)?;
        let word = |i: usize| -> Result<[u8; 4], BakeError> {
            rest.get(i * 4..i * 4 + 4)
                .and_then(|b| b.try_into().ok())
                .ok_or(BakeError::Truncated)
        };
        let version = u32::from_le_bytes(word(0)?);
        if version != VERSION {
            return Err(BakeError::Version(version));
        }
        let texel = f32::from_le_bytes(word(1)?);
        let int = |i| word(i).map(i32::from_le_bytes);
        let min = IVec2::new(int(2)?, int(3)?);
        let count = IVec2::new(int(4)?, int(5)?);
        let texels = (count.x.max(0) * count.y.max(0)) as usize * PAGE_LEN;
        let mut heights = vec![0.0f32; texels];
        let mut materials = vec![[0u32; 2]; texels];
        let mut decoder = lz4_flex::frame::FrameDecoder::new(&rest[24..]);
        decoder.read_exact(bytemuck::cast_slice_mut(&mut heights))?;
        decoder.read_exact(bytemuck::cast_slice_mut(&mut materials))?;
        Ok(Self {
            texel,
            min,
            count,
            heights,
            materials,
        })
    }
}

#[derive(Default, TypePath)]
pub struct BakedPagesLoader;

impl AssetLoader for BakedPagesLoader {
    type Asset = BakedPages;
    type Settings = ();
    type Error = BakeError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<BakedPages, BakeError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        BakedPages::from_bytes(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["pages"]
    }
}

/// Bakes load into their landscapes' pools (once per file); going live re-evaluates.
#[allow(clippy::type_complexity)]
pub fn apply_bakes(
    rd: Res<RenderDevice>,
    live: Res<LandscapeLive>,
    mut bakes: ResMut<Assets<BakedPages>>,
    mut landscapes: Query<(Entity, &Landscape, &LandscapeBake, &mut LandscapePages)>,
    mut evaluated: MessageWriter<PagesEvaluated>,
    mut materials_evaluated: MessageWriter<MaterialsEvaluated>,
) {
    let mut loaded = Vec::new();
    for (entity, landscape, bake, mut pages) in &mut landscapes {
        let id = bake.file.id();
        if live.0 {
            if pages.baked.is_some() {
                pages.baked = None;
                pages.dirty_all();
                pages.dirty_all_materials();
            }
            pages.bake_seen = Some(id);
            continue;
        }
        if pages.bake_seen == Some(id) {
            continue;
        }
        let Some(baked) = bakes.get(id) else {
            continue;
        };
        pages.bake_seen = Some(id);
        if landscape.stream_radius != 0
            || baked.texel != landscape.texel_size
            || baked.min != pages.min()
            || baked.count != pages.count()
        {
            log::warn!("landscape {entity}: its bake is for another window; evaluating instead");
            continue;
        }
        pages.load_bake(&rd, baked);
        pages.baked = Some(id);
        loaded.push(id);
        let all: Vec<IVec2> = pages.window().collect();
        evaluated.write(PagesEvaluated {
            landscape: entity,
            pages: all.clone(),
        });
        materials_evaluated.write(MaterialsEvaluated {
            landscape: entity,
            pages: all,
        });
    }
    // The pool and the mirrors hold it now; a second CPU copy is 12 bytes a texel.
    for id in loaded {
        if landscapes
            .iter()
            .all(|(_, _, bake, pages)| bake.file.id() != id || pages.bake_seen == Some(id))
        {
            bakes.remove(id);
        }
    }
}

impl LandscapePages {
    /// The pages hold a loaded bake, and the stacks are not evaluated.
    pub fn is_baked(&self) -> bool {
        self.baked.is_some()
    }

    /// The pages already are this bake (just written from them): don't load it over them.
    pub fn adopt_bake(&mut self, id: AssetId<BakedPages>) {
        self.bake_seen = Some(id);
    }

    /// The window as a bake, once every page has evaluated.
    pub fn bake(&self) -> Option<BakedPages> {
        if !self.complete || !self.dirty.is_empty() || !self.dirty_materials.is_empty() {
            return None;
        }
        let mut heights = Vec::with_capacity(self.texel_count());
        let mut materials = Vec::with_capacity(self.texel_count());
        for page in self.window() {
            heights.extend_from_slice(self.page(page)?);
            materials.extend_from_slice(self.material_page(page)?);
        }
        Some(BakedPages {
            texel: self.texel(),
            min: self.min(),
            count: self.count(),
            heights,
            materials,
        })
    }

    fn load_bake(&mut self, rd: &RenderDevice, bake: &BakedPages) {
        let len = self.texel_count() as u64;
        let mut heights: Buffer<f32> =
            rd.create_host_buffer(len, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut materials: Buffer<[u32; 2]> =
            rd.create_host_buffer(len, vk::BufferUsageFlags::TRANSFER_SRC);
        // Mirrors first, then write-only into the mapped (ReBAR, uncached) staging: reading it
        // back costs seconds.
        let slots: Vec<(usize, usize)> = self
            .window()
            .enumerate()
            .map(|(i, page)| (i * PAGE_LEN, self.slot(page).unwrap() * PAGE_LEN))
            .collect();
        for &(src, slot) in &slots {
            self.mirror_mut()[slot..slot + PAGE_LEN]
                .copy_from_slice(&bake.heights[src..src + PAGE_LEN]);
            self.material_mirror_mut()[slot..slot + PAGE_LEN]
                .copy_from_slice(&bake.materials[src..src + PAGE_LEN]);
        }
        rd.map_buffer(&mut heights)
            .as_slice_mut()
            .copy_from_slice(self.mirror_mut());
        rd.map_buffer(&mut materials)
            .as_slice_mut()
            .copy_from_slice(self.material_mirror_mut());
        let (pool, pool_materials) = self.device_buffers();
        rd.run_transfer_commands(|cmd| unsafe {
            rd.device.cmd_copy_buffer(
                cmd,
                heights.handle,
                pool,
                &[vk::BufferCopy::default().size(len * 4)],
            );
            rd.device.cmd_copy_buffer(
                cmd,
                materials.handle,
                pool_materials,
                &[vk::BufferCopy::default().size(len * 8)],
            );
        });
        rd.destroyer.destroy_buffer(heights.handle);
        rd.destroyer.destroy_buffer(materials.handle);
        self.dirty.clear();
        self.dirty_materials.clear();
        self.complete = true;
    }
}
