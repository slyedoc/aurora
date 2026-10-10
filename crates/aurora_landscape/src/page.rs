//! The evaluated pages: dense pools over the landscape's bounds in device memory (heights, and
//! each texel's top-4 materials), rebuilt page by page from the height and material stacks
//! whenever a layer under them changes, with a host-cached height
//! mirror each evaluated page is copied into (CPU queries and colliders read the mirror;
//! reading the GPU's upload memory back is uncached and takes seconds).

use ash::vk;
use bevy::{asset::LoadState, ecs::lifecycle::Remove, platform::collections::HashSet, prelude::*};
use bevy_aurora::{
    compute::{ComputeModules, compute_to_compute_barrier, memory_barrier, record_dispatch},
    render_buffer::{Buffer, BufferProvider, BufferView},
    render_device::RenderDevice,
};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::{
    bake::{LandscapeBake, LandscapeLive},
    Landscape, LandscapeKernels, PAGE_TEXELS,
    generator::{GeneratorLayer, GeneratorPush, GeneratorSource},
    layer::{
        HeightBlend, HeightLayer, HeightmapLayer, HeightmapSource, LayerFootprint,
        MaterialFootprint, MaterialLayer, MaterialRuleLayer, SplatmapLayer, SplatmapSource,
    },
    sculpt::{PaintPages, SculptPages},
};

const PAGE_LEN: usize = (PAGE_TEXELS * PAGE_TEXELS) as usize;
/// Pages evaluated per frame; the rest wait.
const PAGES_PER_FRAME: usize = 256;

/// `PagePool` in landscape/pages.slang (scalar).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuPagePool {
    pub heights: u64,
    pub min_x: i32,
    pub min_z: i32,
    pub pages_x: i32,
    pub pages_z: i32,
    pub texel: f32,
    pub pad: u32,
}

/// `EvalPush` in landscape/eval.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct EvalPush {
    page: u64,
    source: u64,
    page_x: i32,
    page_z: i32,
    texel: f32,
    src_w: u32,
    src_h: u32,
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    ox: f32,
    oz: f32,
    height_min: f32,
    height_max: f32,
    falloff: f32,
    opacity: f32,
    blend: u32,
}

/// `MatPush` in landscape/material_eval.slang (the whole 128-byte push block).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct MatPush {
    page: u64,
    source_ids: u64,
    source_weights: u64,
    pool: GpuPagePool,
    page_x: i32,
    page_z: i32,
    src_w: u32,
    src_h: u32,
    m00: f32,
    m01: f32,
    m10: f32,
    m11: f32,
    ox: f32,
    oz: f32,
    opacity: f32,
    material: u32,
    slope_lo: f32,
    slope_hi: f32,
    slope_fade: f32,
    height_lo: f32,
    height_hi: f32,
    height_fade: f32,
}

impl Default for GpuPagePool {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// A landscape's evaluated pages: heights and materials, each with its CPU mirror.
#[derive(Component)]
pub struct LandscapePages {
    buffer: Buffer<f32>,
    mirror: Buffer<f32>,
    materials: Buffer<[u32; 2]>,
    material_mirror: Buffer<[u32; 2]>,
    view: BufferView<f32>,
    material_view: BufferView<[u32; 2]>,
    min: IVec2,
    count: IVec2,
    texel: f32,
    /// Pages waiting for evaluation (heights and then materials).
    pub(crate) dirty: HashSet<IVec2>,
    /// Pages whose materials alone wait.
    pub(crate) dirty_materials: HashSet<IVec2>,
    /// Every page has been evaluated at least once.
    pub(crate) complete: bool,
    /// The layer orders last evaluated (child order).
    pub(crate) order: Vec<Entity>,
    pub(crate) material_order: Vec<Entity>,
    /// The bake the pages hold (the stacks are not evaluated while they do).
    pub(crate) baked: Option<AssetId<crate::bake::BakedPages>>,
    /// The last bake applied or passed over, so it is not applied again.
    pub(crate) bake_seen: Option<AssetId<crate::bake::BakedPages>>,
}

/// Pages whose heights a frame re-evaluated (tiles refill and colliders rebuild over them).
#[derive(Message, Clone, Debug)]
pub struct PagesEvaluated {
    pub landscape: Entity,
    pub pages: Vec<IVec2>,
}

/// Pages whose materials a frame re-evaluated (every height page also re-evaluates these).
#[derive(Message, Clone, Debug)]
pub struct MaterialsEvaluated {
    pub landscape: Entity,
    pub pages: Vec<IVec2>,
}

impl LandscapePages {
    pub fn pool(&self) -> GpuPagePool {
        GpuPagePool {
            heights: self.buffer.address,
            min_x: self.min.x,
            min_z: self.min.y,
            pages_x: self.count.x,
            pages_z: self.count.y,
            texel: self.texel,
            pad: 0,
        }
    }

    /// The materials pool (uint2 per texel, `materials.slang`), same layout as the heights.
    pub fn materials_address(&self) -> u64 {
        self.materials.address
    }

    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// The window's first page.
    pub fn min(&self) -> IVec2 {
        self.min
    }

    /// The window's size in pages.
    pub fn count(&self) -> IVec2 {
        self.count
    }

    pub fn texel(&self) -> f32 {
        self.texel
    }

    /// Texels in the window.
    pub(crate) fn texel_count(&self) -> usize {
        (self.count.x * self.count.y) as usize * PAGE_LEN
    }

    pub(crate) fn mirror_mut(&mut self) -> &mut [f32] {
        self.view.as_slice_mut()
    }

    pub(crate) fn material_mirror_mut(&mut self) -> &mut [[u32; 2]] {
        self.material_view.as_slice_mut()
    }

    /// The device pools: heights, materials.
    pub(crate) fn device_buffers(&self) -> (vk::Buffer, vk::Buffer) {
        (self.buffer.handle, self.materials.handle)
    }

    /// The pool slot holding `page` (`page_slot` in pages.slang), if it is in the window.
    pub(crate) fn slot(&self, page: IVec2) -> Option<usize> {
        let p = page - self.min;
        (p.x >= 0 && p.y >= 0 && p.x < self.count.x && p.y < self.count.y).then(|| {
            (page.y.rem_euclid(self.count.y) * self.count.x + page.x.rem_euclid(self.count.x))
                as usize
        })
    }

    /// The pages in the window.
    pub fn window(&self) -> impl Iterator<Item = IVec2> + use<> {
        let (min, count) = (self.min, self.count);
        (0..count.y).flat_map(move |z| (0..count.x).map(move |x| min + IVec2::new(x, z)))
    }

    pub fn in_window(&self, page: IVec2) -> bool {
        self.slot(page).is_some()
    }

    fn page_address(&self, page: IVec2) -> Option<u64> {
        self.slot(page)
            .map(|s| self.buffer.address + (s * PAGE_LEN * 4) as u64)
    }

    fn material_page_address(&self, page: IVec2) -> Option<u64> {
        self.slot(page)
            .map(|s| self.materials.address + (s * PAGE_LEN * 8) as u64)
    }

    /// One page's 256² heights, row-major (z then x).
    pub fn page(&self, page: IVec2) -> Option<&[f32]> {
        let slot = self.slot(page)?;
        let all = unsafe {
            std::slice::from_raw_parts(self.view.as_ptr(), self.buffer.nr_elements as usize)
        };
        Some(&all[slot * PAGE_LEN..(slot + 1) * PAGE_LEN])
    }

    /// One page's 256² materials (ids, weights; a byte per slot), row-major.
    pub fn material_page(&self, page: IVec2) -> Option<&[[u32; 2]]> {
        let slot = self.slot(page)?;
        let all = unsafe {
            std::slice::from_raw_parts(
                self.material_view.as_ptr(),
                self.materials.nr_elements as usize,
            )
        };
        Some(&all[slot * PAGE_LEN..(slot + 1) * PAGE_LEN])
    }

    /// The (palette id, weight 0..1) pairs at global texel `t`, clamped to the bounds.
    pub fn texel_materials(&self, t: IVec2) -> [(u8, f32); 4] {
        let n = PAGE_TEXELS as i32;
        let t = t.clamp(self.min * n, (self.min + self.count) * n - 1);
        let page = IVec2::new(t.x.div_euclid(n), t.y.div_euclid(n));
        let local = t - page * n;
        let slot = self.slot(page).unwrap_or(0);
        let [ids, weights] = self.material_view[slot * PAGE_LEN + (local.y * n + local.x) as usize];
        std::array::from_fn(|i| {
            (
                (ids >> (8 * i)) as u8,
                ((weights >> (8 * i)) & 0xFF) as f32 / 255.0,
            )
        })
    }

    /// The materials of the texel nearest a landscape-local xz.
    pub fn materials_at(&self, xz: Vec2) -> [(u8, f32); 4] {
        self.texel_materials((xz / self.texel).round().as_ivec2())
    }

    /// The heaviest material at a landscape-local xz.
    pub fn material_at(&self, xz: Vec2) -> u8 {
        self.materials_at(xz)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(0, |(id, _)| id)
    }

    /// Height at global texel `t`, clamped to the bounds (`texel_height` in pages.slang).
    pub fn texel_height(&self, t: IVec2) -> f32 {
        let n = PAGE_TEXELS as i32;
        let lo = self.min * n;
        let hi = (self.min + self.count) * n - 1;
        let t = t.clamp(lo, hi);
        let page = IVec2::new(t.x.div_euclid(n), t.y.div_euclid(n));
        let local = t - page * n;
        let slot = self.slot(page).unwrap_or(0);
        self.view[slot * PAGE_LEN + (local.y * n + local.x) as usize]
    }

    /// Bilinear height at a landscape-local xz (`sample_height` in pages.slang).
    pub fn height_at(&self, xz: Vec2) -> f32 {
        let g = xz / self.texel;
        let f = g.floor();
        let t = g - f;
        let i = f.as_ivec2();
        let h = |dx, dz| self.texel_height(i + IVec2::new(dx, dz));
        let a = h(0, 0) + (h(1, 0) - h(0, 0)) * t.x;
        let b = h(0, 1) + (h(1, 1) - h(0, 1)) * t.x;
        a + (b - a) * t.y
    }

    /// The first ground hit along a landscape-local ray, marched at half-texel steps and
    /// refined by bisection.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_distance: f32) -> Option<Vec3> {
        let dir = dir.try_normalize()?;
        let step = self.texel * 0.5;
        let above = |p: Vec3| p.y - self.height_at(p.xz());
        if above(origin) < 0.0 {
            return None;
        }
        let mut t0 = 0.0;
        let mut t = step;
        while t <= max_distance {
            if above(origin + dir * t) < 0.0 {
                let mut t1 = t;
                for _ in 0..16 {
                    let mid = (t0 + t1) * 0.5;
                    if above(origin + dir * mid) < 0.0 {
                        t1 = mid;
                    } else {
                        t0 = mid;
                    }
                }
                return Some(origin + dir * t1);
            }
            t0 = t;
            // Coarser steps far above the ground.
            t += step.max(above(origin + dir * t) * 0.5);
        }
        None
    }

    pub(crate) fn dirty_all(&mut self) {
        for z in 0..self.count.y {
            for x in 0..self.count.x {
                self.dirty.insert(self.min + IVec2::new(x, z));
            }
        }
    }

    pub(crate) fn dirty_all_materials(&mut self) {
        for z in 0..self.count.y {
            for x in 0..self.count.x {
                self.dirty_materials.insert(self.min + IVec2::new(x, z));
            }
        }
    }

    /// Re-evaluate these pages' heights (and materials) next frame.
    pub fn mark_dirty(&mut self, pages: impl IntoIterator<Item = IVec2>) {
        self.dirty.extend(pages);
    }

    /// Re-evaluate these pages' materials next frame.
    pub fn mark_materials_dirty(&mut self, pages: impl IntoIterator<Item = IVec2>) {
        self.dirty_materials.extend(pages);
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for buffer in [&mut self.buffer, &mut self.mirror] {
            if buffer.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(buffer.handle);
                *buffer = Buffer::default();
            }
        }
        for buffer in [&mut self.materials, &mut self.material_mirror] {
            if buffer.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(buffer.handle);
                *buffer = Buffer::default();
            }
        }
    }
}

/// A new or resized landscape gets its pool, every page dirty; a window that only slid
/// (streaming) keeps its pool and dirties the pages it entered, whose slots the pages it
/// left held.
pub fn allocate_pages(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    mut landscapes: Query<(Entity, &Landscape, Option<&mut LandscapePages>), Changed<Landscape>>,
) {
    for (entity, landscape, pages) in &mut landscapes {
        let count = landscape.page_count();
        if let Some(mut pages) = pages {
            if pages.count == count && pages.texel == landscape.texel_size {
                if pages.min != landscape.pages_min {
                    let old = pages.min;
                    pages.min = landscape.pages_min;
                    let entered: Vec<IVec2> = pages
                        .window()
                        .filter(|p| {
                            let q = *p - old;
                            q.x < 0 || q.y < 0 || q.x >= count.x || q.y >= count.y
                        })
                        .collect();
                    pages.dirty.retain(|p| {
                        let q = *p - landscape.pages_min;
                        q.x >= 0 && q.y >= 0 && q.x < count.x && q.y < count.y
                    });
                    pages
                        .dirty_materials
                        .retain(|p| (*p - landscape.pages_min).cmpge(IVec2::ZERO).all()
                            && (*p - landscape.pages_min).cmplt(count).all());
                    pages.mark_dirty(entered);
                }
                continue;
            }
            pages.destroy(&rd);
        }
        if count.x <= 0 || count.y <= 0 {
            commands.entity(entity).remove::<LandscapePages>();
            continue;
        }
        let len = count.x as u64 * count.y as u64 * PAGE_LEN as u64;
        let buffer: Buffer<f32> = rd.create_device_buffer(
            len,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let mut mirror: Buffer<f32> = rd.create_buffer(
            len,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuToCpu,
        );
        let view = rd.map_buffer(&mut mirror);
        let materials: Buffer<[u32; 2]> = rd.create_device_buffer(
            len,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let mut material_mirror: Buffer<[u32; 2]> = rd.create_buffer(
            len,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuToCpu,
        );
        let material_view = rd.map_buffer(&mut material_mirror);
        let mut pages = LandscapePages {
            buffer,
            mirror,
            materials,
            material_mirror,
            view,
            material_view,
            min: landscape.pages_min,
            count,
            texel: landscape.texel_size,
            dirty: HashSet::default(),
            dirty_materials: HashSet::default(),
            complete: false,
            order: Vec::new(),
            material_order: Vec::new(),
            baked: None,
            bake_seen: None,
        };
        pages.dirty_all();
        commands.entity(entity).insert(pages);
    }
}

/// One layer's dispatch recipe for a frame.
struct Pass {
    entry: &'static str,
    footprint: Rect,
    push: EvalPush,
    /// Sculpt layers: the delta page per page.
    sculpt: Option<Entity>,
    /// Generator layers (generator.slang): the genome's address.
    genome: Option<u64>,
}

/// One material layer's recipe.
struct MatPass {
    entry: &'static str,
    footprint: Rect,
    push: MatPush,
    /// Paint layers: the paint page per page.
    paint: Option<Entity>,
}

/// Landscape xz -> layer uv for a rectangle of `size` centred on `transform` (yaw only).
fn placement(transform: &Transform, size: Vec2) -> (Mat2, Vec2) {
    let (yaw, _, _) = transform.rotation.to_euler(EulerRot::YXZ);
    let (s, c) = yaw.sin_cos();
    // Layer-local = R(-yaw) (xz - centre).
    let m = Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c));
    let m = Mat2::from_diagonal(Vec2::ONE / size.max(Vec2::splat(1.0e-3))) * m;
    (m, Vec2::splat(0.5) - m.mul_vec2(transform.translation.xz()))
}

type HeightLayers<'w, 's> = Query<
    'w,
    's,
    (
        &'static HeightLayer,
        Option<&'static LayerFootprint>,
        Option<&'static HeightmapLayer>,
        Option<&'static HeightmapSource>,
        Option<&'static GeneratorSource>,
        Has<GeneratorLayer>,
        &'static Transform,
    ),
>;

type MaterialLayers<'w, 's> = Query<
    'w,
    's,
    (
        &'static MaterialLayer,
        Option<&'static MaterialFootprint>,
        Option<&'static SplatmapLayer>,
        Option<&'static SplatmapSource>,
        Option<&'static MaterialRuleLayer>,
        Has<PaintPages>,
        &'static Transform,
    ),
>;

/// The height stack's passes, or `None` while a source is still loading.
fn height_passes(
    landscape: &Landscape,
    order: &[Entity],
    layers: &HeightLayers,
    sculpts: &Query<&SculptPages>,
) -> Option<Vec<Pass>> {
    let mut passes = Vec::new();
    for &entity in order {
        let Ok((layer, footprint, heightmap, source, genome, generates, transform)) =
            layers.get(entity)
        else {
            continue;
        };
        if !layer.enabled || layer.opacity <= 0.0 {
            continue;
        }
        let footprint = footprint?;
        let base = EvalPush {
            texel: landscape.texel_size,
            opacity: layer.opacity.clamp(0.0, 1.0),
            blend: match layer.blend {
                HeightBlend::Replace => 0,
                HeightBlend::Add => 1,
                HeightBlend::Max => 2,
                HeightBlend::Min => 3,
            },
            ..default()
        };
        if let Some(heightmap) = heightmap {
            let source = source?;
            let (m, o) = placement(transform, heightmap.size);
            passes.push(Pass {
                entry: "apply_heightmap",
                footprint: footprint.0,
                push: EvalPush {
                    source: source.address(),
                    src_w: source.width,
                    src_h: source.height,
                    m00: m.x_axis.x,
                    m01: m.y_axis.x,
                    m10: m.x_axis.y,
                    m11: m.y_axis.y,
                    ox: o.x,
                    oz: o.y,
                    height_min: heightmap.height_min + transform.translation.y,
                    height_max: heightmap.height_max + transform.translation.y,
                    falloff: heightmap.falloff,
                    ..base
                },
                sculpt: None,
                genome: None,
            });
        } else if generates {
            // Its parameters reach the GPU a frame after it spawns: wait, never skip.
            let genome = genome?;
            passes.push(Pass {
                entry: "apply_generator",
                footprint: footprint.0,
                push: base,
                sculpt: None,
                genome: Some(genome.address()),
            });
        } else if sculpts.contains(entity) {
            passes.push(Pass {
                entry: "apply_sculpt",
                footprint: footprint.0,
                push: base,
                sculpt: Some(entity),
                genome: None,
            });
        }
    }
    Some(passes)
}

/// The material stack's passes, or `None` while a source is still loading.
fn material_passes(
    pool: GpuPagePool,
    order: &[Entity],
    layers: &MaterialLayers,
) -> Option<Vec<MatPass>> {
    let mut passes = Vec::new();
    for &entity in order {
        let Ok((layer, footprint, splatmap, source, rule, paints, transform)) = layers.get(entity)
        else {
            continue;
        };
        if !layer.enabled || layer.opacity <= 0.0 {
            continue;
        }
        let footprint = footprint?;
        let base = MatPush {
            pool,
            opacity: layer.opacity.clamp(0.0, 1.0),
            ..default()
        };
        if let Some(splatmap) = splatmap {
            let source = source?;
            let (m, o) = placement(transform, splatmap.size);
            passes.push(MatPass {
                entry: "apply_splatmap",
                footprint: footprint.0,
                push: MatPush {
                    source_ids: source.ids_address(),
                    source_weights: source.weights_address(),
                    src_w: source.width,
                    src_h: source.height,
                    m00: m.x_axis.x,
                    m01: m.y_axis.x,
                    m10: m.x_axis.y,
                    m11: m.y_axis.y,
                    ox: o.x,
                    oz: o.y,
                    ..base
                },
                paint: None,
            });
        } else if let Some(rule) = rule {
            passes.push(MatPass {
                entry: "apply_rule",
                footprint: footprint.0,
                push: MatPush {
                    material: rule.material as u32,
                    slope_lo: rule.slope.x,
                    slope_hi: rule.slope.y,
                    slope_fade: rule.slope_fade,
                    height_lo: rule.height.x,
                    height_hi: rule.height.y,
                    height_fade: rule.height_fade,
                    ..base
                },
                paint: None,
            });
        } else if paints {
            passes.push(MatPass {
                entry: "apply_paint",
                footprint: footprint.0,
                push: base,
                paint: Some(entity),
            });
        }
    }
    Some(passes)
}

/// Re-evaluate dirty pages: the height stack over pages whose heights changed, then the
/// material stack over every evaluated page (rules read the heights). Waits while a layer's
/// source is still loading, so a landscape never shows half a stack.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_pages(
    rd: Res<RenderDevice>,
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    mut landscapes: Query<(
        Entity,
        &Landscape,
        &mut LandscapePages,
        Option<&Children>,
        Option<&LandscapeBake>,
    )>,
    live: Res<LandscapeLive>,
    server: Res<AssetServer>,
    height_layers: HeightLayers,
    material_layers: MaterialLayers,
    sculpts: Query<&SculptPages>,
    paints: Query<&PaintPages>,
    mut evaluated: MessageWriter<PagesEvaluated>,
    mut materials_evaluated: MessageWriter<MaterialsEvaluated>,
) {
    let (Some(height_module), Some(material_module), Some(generator_module)) = (
        modules.get(&kernels.eval),
        modules.get(&kernels.material),
        modules.get(&kernels.generator),
    ) else {
        return;
    };
    for (entity, landscape, mut pages, children, bake) in &mut landscapes {
        let children: Vec<Entity> = children.into_iter().flatten().copied().collect();
        let order: Vec<Entity> = children
            .iter()
            .copied()
            .filter(|c| height_layers.contains(*c))
            .collect();
        if order != pages.order {
            pages.dirty_all();
            pages.order = order.clone();
        }
        let material_order: Vec<Entity> = children
            .iter()
            .copied()
            .filter(|c| material_layers.contains(*c))
            .collect();
        if material_order != pages.material_order {
            pages.dirty_all_materials();
            pages.material_order = material_order.clone();
        }
        // Baked pages stand in for the stacks; a bake still loading is waited for.
        let waiting = bake.is_some_and(|b| {
            !live.0
                && pages.bake_seen.is_none()
                && !matches!(server.load_state(b.file.id()), LoadState::Failed(_))
                && b.file != Handle::default()
        });
        if pages.baked.is_some() || waiting {
            if pages.baked.is_some() {
                pages.dirty.clear();
                pages.dirty_materials.clear();
            }
            continue;
        }
        if pages.dirty.is_empty() && pages.dirty_materials.is_empty() {
            continue;
        }
        let Some(passes) = height_passes(landscape, &order, &height_layers, &sculpts) else {
            continue;
        };
        let Some(mat_passes) = material_passes(pages.pool(), &material_order, &material_layers)
        else {
            continue;
        };

        let heights: Vec<IVec2> = pages.dirty.iter().copied().take(PAGES_PER_FRAME).collect();
        for p in &heights {
            pages.dirty.remove(p);
            pages.dirty_materials.remove(p);
        }
        let mut batch = heights.clone();
        let extra: Vec<IVec2> = pages
            .dirty_materials
            .iter()
            .copied()
            .take(PAGES_PER_FRAME.saturating_sub(batch.len()))
            .collect();
        for p in &extra {
            pages.dirty_materials.remove(p);
        }
        batch.extend(extra);

        let pages = &mut *pages;
        let started = std::time::Instant::now();
        rd.run_transfer_commands(|cmd| {
            if !heights.is_empty() {
                for &page in &heights {
                    let push = EvalPush {
                        page: pages.page_address(page).unwrap(),
                        ..default()
                    };
                    record_dispatch(
                        &rd,
                        cmd,
                        height_module,
                        "clear_page",
                        &push,
                        PAGE_LEN as u32,
                        None,
                    );
                }
                for pass in &passes {
                    compute_to_compute_barrier(&rd, cmd);
                    for &page in &heights {
                        if landscape.page_rect(page).intersect(pass.footprint).is_empty() {
                            continue;
                        }
                        if let Some(genome) = pass.genome {
                            let push = GeneratorPush {
                                page: pages.page_address(page).unwrap(),
                                genome,
                                page_x: page.x,
                                page_z: page.y,
                                texel: pass.push.texel,
                                opacity: pass.push.opacity,
                                blend: pass.push.blend,
                                face: landscape.sphere.map_or(0, |f| f.face as u32),
                                radius: landscape.sphere.map_or(0.0, |f| f.radius),
                                pad: 0,
                            };
                            record_dispatch(
                                &rd,
                                cmd,
                                generator_module,
                                pass.entry,
                                &push,
                                PAGE_LEN as u32,
                                None,
                            );
                            continue;
                        }
                        let mut push = pass.push;
                        push.page = pages.page_address(page).unwrap();
                        push.page_x = page.x;
                        push.page_z = page.y;
                        if let Some(layer) = pass.sculpt {
                            let Some(delta) =
                                sculpts.get(layer).ok().and_then(|s| s.address(page))
                            else {
                                continue;
                            };
                            push.source = delta;
                        }
                        record_dispatch(
                            &rd,
                            cmd,
                            height_module,
                            pass.entry,
                            &push,
                            PAGE_LEN as u32,
                            None,
                        );
                    }
                }
                memory_barrier(
                    &rd,
                    cmd,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_WRITE,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                );
                let bytes = (PAGE_LEN * 4) as u64;
                let regions: Vec<vk::BufferCopy> = heights
                    .iter()
                    .map(|&p| {
                        let offset = pages.slot(p).unwrap() as u64 * bytes;
                        vk::BufferCopy::default()
                            .src_offset(offset)
                            .dst_offset(offset)
                            .size(bytes)
                    })
                    .collect();
                unsafe {
                    rd.device.cmd_copy_buffer(
                        cmd,
                        pages.buffer.handle,
                        pages.mirror.handle,
                        &regions,
                    );
                }
                memory_barrier(
                    &rd,
                    cmd,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::HOST,
                    vk::AccessFlags2::HOST_READ,
                );
            }

            // Materials: rules read the heights just written.
            compute_to_compute_barrier(&rd, cmd);
            for &page in &batch {
                let push = MatPush {
                    page: pages.material_page_address(page).unwrap(),
                    ..default()
                };
                record_dispatch(
                    &rd,
                    cmd,
                    material_module,
                    "clear_materials",
                    &push,
                    PAGE_LEN as u32,
                    None,
                );
            }
            for pass in &mat_passes {
                compute_to_compute_barrier(&rd, cmd);
                for &page in &batch {
                    if landscape.page_rect(page).intersect(pass.footprint).is_empty() {
                        continue;
                    }
                    let mut push = pass.push;
                    push.page = pages.material_page_address(page).unwrap();
                    push.page_x = page.x;
                    push.page_z = page.y;
                    if let Some(layer) = pass.paint {
                        let Some(paint) = paints.get(layer).ok().and_then(|p| p.address(page))
                        else {
                            continue;
                        };
                        push.source_ids = paint;
                    }
                    record_dispatch(
                        &rd,
                        cmd,
                        material_module,
                        pass.entry,
                        &push,
                        PAGE_LEN as u32,
                        None,
                    );
                }
            }
            memory_barrier(
                &rd,
                cmd,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            );
            let bytes = (PAGE_LEN * 8) as u64;
            let regions: Vec<vk::BufferCopy> = batch
                .iter()
                .map(|&p| {
                    let offset = pages.slot(p).unwrap() as u64 * bytes;
                    vk::BufferCopy::default()
                        .src_offset(offset)
                        .dst_offset(offset)
                        .size(bytes)
                })
                .collect();
            unsafe {
                rd.device.cmd_copy_buffer(
                    cmd,
                    pages.materials.handle,
                    pages.material_mirror.handle,
                    &regions,
                );
            }
            memory_barrier(
                &rd,
                cmd,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ,
            );
        });
        log::debug!(
            "landscape: {} height + {} material pages, {} + {} layers in {:.1} ms",
            heights.len(),
            batch.len(),
            passes.len(),
            mat_passes.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
        if pages.dirty.is_empty() && !pages.complete {
            pages.complete = true;
            log::info!(
                "landscape: {} pages evaluated in {:.1} ms",
                batch.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        if !batch.is_empty() {
            materials_evaluated.write(MaterialsEvaluated {
                landscape: entity,
                pages: batch,
            });
        }
        if !heights.is_empty() {
            evaluated.write(PagesEvaluated {
                landscape: entity,
                pages: heights,
            });
        }
    }
}

pub fn on_pages_removed(
    remove: On<Remove<LandscapePages>>,
    mut pages: Query<&mut LandscapePages>,
    rd: Option<Res<RenderDevice>>,
) {
    if let Some(rd) = rd
        && let Ok(mut pages) = pages.get_mut(remove.entity)
    {
        pages.destroy(&rd);
    }
}

pub fn release_pages(mut pages: Query<&mut LandscapePages>, rd: Res<RenderDevice>) {
    for mut pages in &mut pages {
        pages.destroy(&rd);
    }
}
