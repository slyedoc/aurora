//! Heightfield colliders, one per page, rebuilt whenever the page re-evaluates or its holes
//! change (hole texels are removed cells). The heights
//! are copied on the main thread; the shape (tens of ms per page) is built on the async
//! compute pool and attached when done.

use avian3d::{
    parry::{
        shape::{HeightField, HeightFieldCellStatus, SharedShape},
        utils::Array2,
    },
    prelude::{Collider, RigidBody},
};
use bevy::{
    platform::collections::HashMap,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};

use crate::{
    Landscape, PAGE_TEXELS,
    page::{HolesChanged, LandscapePages, PagesEvaluated},
};

#[derive(Component, Clone, Copy, Debug)]
pub struct PageCollider {
    pub landscape: Entity,
    pub page: IVec2,
}

/// The shape being built for a page; a newer evaluation replaces (and so cancels) it.
#[derive(Component)]
pub struct ColliderBuild(Task<Collider>);

/// A page's heightfield: its 256 texels plus the next page's first, so pages meet. The main
/// thread copies the page and its two seams; the transpose and the shape run on the pool.
fn page_heightfield(pages: &LandscapePages, page: IVec2, size: f32) -> Task<Collider> {
    let n = PAGE_TEXELS as usize;
    let base = page * n as i32;
    let texels = pages.page(page).map(<[f32]>::to_vec);
    let holes = pages.page_holes(page).filter(|h| h.contains(&true));
    let edge_x: Vec<f32> = (0..=n as i32)
        .map(|z| pages.texel_height(base + IVec2::new(n as i32, z)))
        .collect();
    let edge_z: Vec<f32> = (0..n as i32)
        .map(|x| pages.texel_height(base + IVec2::new(x, n as i32)))
        .collect();
    AsyncComputeTaskPool::get().spawn(async move {
        // parry's Array2 is column-major over (z, x): x-major, z contiguous.
        let mut data = Vec::with_capacity((n + 1) * (n + 1));
        for x in 0..=n {
            for z in 0..=n {
                data.push(match (x == n, z == n, &texels) {
                    (true, _, _) => edge_x[z],
                    (_, true, _) => edge_z[x],
                    (_, _, Some(texels)) => texels[z * n + x],
                    (_, _, None) => 0.0,
                });
            }
        }
        let mut field = HeightField::new(Array2::new(n + 1, n + 1, data), Vec3::new(size, 1.0, size));
        if let Some(holes) = holes {
            cut_holes(&mut field, &holes, n);
        }
        SharedShape::new(field).into()
    })
}

/// Removes the cells over hole texels (`holes`, n² row-major). Cell (z, x) spans texels
/// x..x+1, z..z+1: the texel it starts at decides it, as the tiles cut a quad by the texel under
/// its centre.
fn cut_holes(field: &mut HeightField, holes: &[bool], n: usize) {
    for z in 0..n {
        for x in 0..n {
            if holes[z * n + x] {
                field.set_cell_status(z, x, HeightFieldCellStatus::CELL_REMOVED);
            }
        }
    }
}

pub fn update_colliders(
    mut commands: Commands,
    mut evaluated: MessageReader<PagesEvaluated>,
    mut holes: MessageReader<HolesChanged>,
    landscapes: Query<(&Landscape, &LandscapePages)>,
    existing: Query<(Entity, &PageCollider)>,
) {
    // Pages a streaming window left take their colliders along.
    for (entity, collider) in &existing {
        if landscapes
            .get(collider.landscape)
            .is_ok_and(|(_, pages)| !pages.in_window(collider.page))
        {
            commands.entity(entity).try_despawn();
        }
    }
    let mut by_page: Option<HashMap<(Entity, IVec2), Entity>> = None;
    let changed: Vec<(Entity, Vec<IVec2>)> = evaluated
        .read()
        .map(|m| (m.landscape, m.pages.clone()))
        .chain(holes.read().map(|m| (m.landscape, m.pages.clone())))
        .collect();
    for (landscape_entity, changed_pages) in changed {
        let Ok((landscape, pages)) = landscapes.get(landscape_entity) else {
            continue;
        };
        // Heightfields are flat: planet faces go without for now.
        if !landscape.colliders || landscape.sphere.is_some() {
            continue;
        }
        let by_page = by_page.get_or_insert_with(|| {
            existing
                .iter()
                .map(|(e, c)| ((c.landscape, c.page), e))
                .collect()
        });
        let size = landscape.page_size();
        let started = std::time::Instant::now();
        for &page in &changed_pages {
            let build = ColliderBuild(page_heightfield(pages, page, size));
            let centre = landscape.page_rect(page).center();
            let at = Transform::from_translation(Vec3::new(centre.x, 0.0, centre.y));
            match by_page.get(&(landscape_entity, page)) {
                Some(&e) => {
                    commands.entity(e).insert((build, at));
                }
                None => {
                    let e = commands
                        .spawn((
                            Name::new(format!("LandscapeCollider_{}_{}", page.x, page.y)),
                            ChildOf(landscape_entity),
                            PageCollider {
                                landscape: landscape_entity,
                                page,
                            },
                            RigidBody::Static,
                            build,
                            at,
                        ))
                        .id();
                    by_page.insert((landscape_entity, page), e);
                }
            }
        }
        log::debug!(
            "landscape: {} collider heights copied in {:.1} ms",
            changed_pages.len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Attach finished shapes.
pub fn finish_colliders(mut commands: Commands, mut builds: Query<(Entity, &mut ColliderBuild)>) {
    for (entity, mut build) in &mut builds {
        if let Some(collider) = check_ready(&mut build.0) {
            commands
                .entity(entity)
                .remove::<ColliderBuild>()
                .insert(collider);
        }
    }
}

#[cfg(test)]
mod tests {
    use avian3d::parry::query::{Ray, RayCast};

    use super::*;

    #[test]
    fn rays_fall_through_hole_cells() {
        let n = 4;
        let mut field = HeightField::new(
            Array2::new(n + 1, n + 1, vec![0.0; (n + 1) * (n + 1)]),
            Vec3::new(4.0, 1.0, 4.0),
        );
        let mut holes = vec![false; n * n];
        holes[1 * n + 2] = true;
        cut_holes(&mut field, &holes, n);
        // Cells are 1 m, centred on the origin: texel (x, z) spans x - 2 .. x - 1.
        let down = |x: f32, z: f32| {
            let ray = Ray::new(Vec3::new(x, 5.0, z), Vec3::NEG_Y);
            field.cast_local_ray(&ray, 10.0, true)
        };
        assert!(down(0.5, -0.5).is_none(), "the hole cell");
        assert!(down(-0.5, -0.5).is_some(), "its neighbour");
        assert!(down(0.5, 0.5).is_some(), "the cell below it");
    }
}
