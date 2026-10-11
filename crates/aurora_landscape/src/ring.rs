//! LOD rings: a streaming landscape's view past its window.
//!
//! A streaming [`Landscape`] with `lod_rings > 0` keeps, besides its own window of pages,
//! that many rings around it: hidden child landscapes ([`LandscapeRing`]) whose texels are
//! twice the size of the level inside, with the same number of pages. Each level's window is
//! aligned to whole pages of the next, and a ring draws no tiles where the level inside it
//! has them (its `hole`), so the levels tile the ground without overlapping. Memory and
//! evaluation grow with the number of rings, not with the square of the distance.
//!
//! Rings evaluate the parent's layers at their own texel size: the ones placed by position
//! (generators, heightmaps, splatmaps, rules). Edits (sculpt and paint layers) and everything
//! read on the CPU (colliders, scatter, height queries, bakes) are the finest level's alone.

use bevy::prelude::*;
use bevy_aurora::material::AuroraMaterial3d;

use crate::{
    Landscape, LandscapePages, LandscapePalette,
    layer::{HeightLayer, MaterialLayer},
};

/// A coarser level of `of`'s streaming window (spawned and kept by the plugin).
#[derive(Component, Clone, Copy, Debug)]
pub struct LandscapeRing {
    pub of: Entity,
    /// 1 is the ring just outside the parent's own window.
    pub level: u8,
    /// The level inside's window, in this ring's pages (min inclusive, max exclusive): its
    /// tiles cover that ground, so this ring has none there.
    pub hole: (IVec2, IVec2),
}

impl LandscapeRing {
    pub fn in_hole(&self, page: IVec2) -> bool {
        page.cmpge(self.hole.0).all() && page.cmplt(self.hole.1).all()
    }
}

/// The ring at `level` of `parent`: the parent's tiling at `2^level` times the texel size.
fn ring_landscape(parent: &Landscape, level: u8) -> Landscape {
    Landscape {
        texel_size: parent.texel_size * (1u32 << level) as f32,
        max_lod: parent.max_lod,
        patch_resolution: parent.patch_resolution,
        split_factor: parent.split_factor,
        colliders: false,
        stream_radius: parent.stream_radius,
        lod_rings: 0,
        sphere: None,
        pages_min: IVec2::ZERO,
        pages_max: IVec2::ZERO,
    }
}

/// Every streaming landscape has exactly its `lod_rings` rings, each following its tiling,
/// palette and material.
#[allow(clippy::type_complexity)]
pub fn sync_rings(
    mut commands: Commands,
    parents: Query<
        (
            Entity,
            Ref<Landscape>,
            Option<Ref<LandscapePalette>>,
            Option<Ref<AuroraMaterial3d>>,
        ),
        Without<LandscapeRing>,
    >,
    mut rings: Query<
        (
            Entity,
            &LandscapeRing,
            &mut Landscape,
            Has<LandscapePalette>,
            Has<AuroraMaterial3d>,
        ),
        With<LandscapeRing>,
    >,
) {
    let mut existing: Vec<(Entity, Entity, u8)> = rings
        .iter()
        .map(|(entity, ring, ..)| (entity, ring.of, ring.level))
        .collect();
    for (parent, landscape, palette, material) in &parents {
        let wanted = if landscape.stream_radius > 0 && landscape.sphere.is_none() {
            landscape.lod_rings
        } else {
            0
        };
        for level in 1..=wanted {
            let Some(at) = existing
                .iter()
                .position(|&(_, of, l)| of == parent && l == level)
            else {
                let mut ring = commands.spawn((
                    Name::new(format!("LandscapeRing_{level}")),
                    LandscapeRing {
                        of: parent,
                        level,
                        hole: (IVec2::ZERO, IVec2::ZERO),
                    },
                    ring_landscape(&landscape, level),
                    Transform::IDENTITY,
                    ChildOf(parent),
                ));
                if let Some(palette) = &palette {
                    ring.insert((**palette).clone());
                }
                if let Some(material) = &material {
                    ring.insert((**material).clone());
                }
                continue;
            };
            let (entity, _, _) = existing.swap_remove(at);
            let Ok((_, _, mut ring, has_palette, has_material)) = rings.get_mut(entity) else {
                continue;
            };
            let want = ring_landscape(&landscape, level);
            if ring.texel_size != want.texel_size
                || ring.max_lod != want.max_lod
                || ring.patch_resolution != want.patch_resolution
                || ring.split_factor != want.split_factor
                || ring.stream_radius != want.stream_radius
            {
                *ring = Landscape {
                    pages_min: ring.pages_min,
                    pages_max: ring.pages_max,
                    ..want
                };
            }
            if let Some(palette) = &palette
                && (palette.is_changed() || !has_palette)
            {
                commands.entity(entity).insert((**palette).clone());
            }
            if let Some(material) = &material
                && (material.is_changed() || !has_material)
            {
                commands.entity(entity).insert((**material).clone());
            }
        }
    }
    // Rings of landscapes that no longer want them, or of no landscape at all.
    for (entity, ..) in existing {
        commands.entity(entity).try_despawn();
    }
}

/// A change to a parent's height or material stack re-evaluates its rings. Edits are not
/// applied to rings, so this is a layer being added, moved, re-ordered or re-tuned: rare, and
/// a whole ring is a few dozen pages.
#[allow(clippy::type_complexity)]
pub fn track_ring_layers(
    heights: Query<
        &ChildOf,
        (
            With<HeightLayer>,
            Or<(
                Changed<HeightLayer>,
                Changed<Transform>,
                Changed<crate::layer::HeightmapLayer>,
                Changed<crate::layer::HeightmapSource>,
                Changed<crate::generator::GeneratorSource>,
                Changed<ChildOf>,
            )>,
        ),
    >,
    materials: Query<
        &ChildOf,
        (
            With<MaterialLayer>,
            Or<(
                Changed<MaterialLayer>,
                Changed<Transform>,
                Changed<crate::layer::MaterialRuleLayer>,
                Changed<crate::layer::SplatmapLayer>,
                Changed<crate::layer::SplatmapSource>,
                Changed<ChildOf>,
            )>,
        ),
    >,
    mut rings: Query<(&LandscapeRing, &mut LandscapePages)>,
) {
    let mut height_parents: Vec<Entity> = heights.iter().map(ChildOf::parent).collect();
    let mut material_parents: Vec<Entity> = materials.iter().map(ChildOf::parent).collect();
    if height_parents.is_empty() && material_parents.is_empty() {
        return;
    }
    height_parents.dedup();
    material_parents.dedup();
    for (ring, mut pages) in &mut rings {
        if height_parents.contains(&ring.of) {
            pages.dirty_all();
        } else if material_parents.contains(&ring.of) {
            pages.dirty_all_materials();
        }
    }
}

/// Window placement for a landscape with rings: `count` pages a side (even), the first page
/// even, so every level's window is whole pages of the next.
pub(crate) fn ringed_window(centre: IVec2, radius: u32) -> (IVec2, i32) {
    let n = radius as i32 + 1;
    let min = IVec2::new(
        2 * (centre.x - n).div_euclid(2),
        2 * (centre.y - n).div_euclid(2),
    );
    (min, 2 * n)
}

/// A ring's window around the level inside it (whose first page `inner_min` is even): the
/// inner window is `n` of this ring's pages from `inner_min / 2`, centred as near as an even
/// first page allows. Returns the window's first page and the hole.
pub(crate) fn ring_window(inner_min: IVec2, radius: u32) -> (IVec2, (IVec2, IVec2)) {
    let n = radius as i32 + 1;
    let lo = IVec2::new(inner_min.x.div_euclid(2), inner_min.y.div_euclid(2));
    let h = n / 2;
    let min = IVec2::new(2 * (lo.x - h).div_euclid(2), 2 * (lo.y - h).div_euclid(2));
    (min, (lo, lo + IVec2::splat(n)))
}
