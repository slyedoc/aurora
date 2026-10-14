//! LOD tiles: one quadtree per page, every leaf a `ProceduralMesh` filled from the pages by
//! `landscape/tile.slang`. Split only below ready tiles and retire only once the replacements
//! are ready (zero's planet rules), so LOD swaps never open holes; edges against a coarser
//! neighbour are stitched. A tile whose pages re-evaluate refills in place: the old mesh
//! stays until the new one is built.

use bevy::{
    ecs::lifecycle::Remove,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};
use bevy_aurora::{
    compute::ComputeModules,
    material::AuroraMaterial3d,
    procedural_mesh::{ProceduralKernels, ProceduralMesh, ProceduralMesh3d},
    vulkan_asset::VulkanAssets,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    Landscape, LandscapeKernels, LandscapeViewer,
    collider::PageCollider,
    page::{GpuPagePool, HolesChanged, LandscapePages, PagesEvaluated},
    quadtree::{NodeId, QuadTree},
    topology::{PatchTopologies, vertex_count},
};

/// Tiles spawned or refilled per frame.
const TILE_BUDGET: u32 = 32;

/// One page's quadtree.
#[derive(Component, Clone, Copy, Debug)]
pub struct LandscapeRoot {
    pub landscape: Entity,
    pub page: IVec2,
}

/// The roots a landscape spawned, by page, and for which pool.
#[derive(Component)]
pub struct LandscapeRoots {
    pool: u64,
    roots: HashMap<IVec2, Entity>,
}

/// `TilePush` in landscape/tile.slang, after the `ProceduralHeader`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct TileParams {
    pool: GpuPagePool,
    level: u32,
    cell_x: i32,
    cell_z: i32,
    res: u32,
    root_size: f32,
    origin_x: f32,
    origin_z: f32,
    skirt_depth: f32,
    edge_steps: u32,
    /// Planet faces (`SphereFace`): face, radius (0 = flat); origin_x/y/z planet-centred.
    face: u32,
    radius: f32,
    origin_y: f32,
    /// Edges (-z, +z, -x, +x as bits 0..3) whose skirt drops `skirt_depth`: the ones on a
    /// boundary between LOD levels, where neighbouring tiles sample different pools.
    skirt_edges: u32,
    pad: u32,
    /// The evaluated materials (`LandscapePages::materials_address`; 0 = none): holes.
    materials: u64,
}

#[derive(Component, Clone, Debug)]
pub struct LandscapeTile {
    pub landscape: Entity,
    pub level: u8,
    /// Global integer cell at `level` (`page * 2^level + local`).
    pub cell: IVec2,
    /// Per edge (-z, +z, -x, +x): the coarser neighbour's step in this tile's vertex steps.
    pub edge_steps: [u8; 4],
    /// Edges with a skirt (see `TileParams::skirt_edges`).
    pub skirt_edges: u8,
    /// Landscape-local xz.
    pub rect: Rect,
    params: TileParams,
}

/// The tile's pages changed: build a new mesh.
#[derive(Component)]
pub struct Refill;

/// The new mesh, swapped in once prepared.
#[derive(Component)]
pub struct PendingMesh(Handle<ProceduralMesh>);

pub(crate) fn viewer_position(
    viewers: &Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: &Query<(&GlobalTransform, &Camera), With<Camera3d>>,
) -> Option<Vec3> {
    viewers.iter().next().map(|t| t.translation()).or_else(|| {
        cameras
            .iter()
            .find(|(_, c)| c.is_active)
            .map(|(t, _)| t.translation())
    })
}

/// A landscape whose pages are complete gets one root per evaluated page of its window;
/// roots leave with their page (a streaming window slid) or all at once (a new pool).
pub fn spawn_roots(
    mut commands: Commands,
    mut landscapes: Query<(
        Entity,
        &LandscapePages,
        Option<&mut LandscapeRoots>,
        Option<&crate::ring::LandscapeRing>,
    )>,
) {
    // A ring's page inside its hole gives way only once the ground is up under all four of its
    // pages at the level inside: rooted there, or (where that level has its own hole) covered
    // in turn by the levels further in. Built finest first.
    let rooted: HashMap<Entity, HashSet<IVec2>> = landscapes
        .iter()
        .filter_map(|(entity, _, roots, _)| {
            roots.map(|r| (entity, r.roots.keys().copied().collect()))
        })
        .collect();
    let mut levels: Vec<(Entity, Entity, u8)> = landscapes
        .iter()
        .map(|(entity, _, _, ring)| match ring {
            Some(ring) => (entity, ring.of, ring.level),
            None => (entity, entity, 0),
        })
        .collect();
    levels.sort_by_key(|&(_, _, level)| level);
    let mut ground: HashMap<(Entity, u8), HashSet<IVec2>> = HashMap::default();
    for (entity, parent, level) in levels {
        let mut pages = rooted.get(&entity).cloned().unwrap_or_default();
        if level > 0
            && let Some(inner) = ground.get(&(parent, level - 1))
        {
            let quads: HashSet<IVec2> = inner
                .iter()
                .map(|c| IVec2::new(c.x.div_euclid(2), c.y.div_euclid(2)))
                .collect();
            pages.extend(quads.into_iter().filter(|q| {
                [IVec2::ZERO, IVec2::X, IVec2::Y, IVec2::ONE]
                    .iter()
                    .all(|d| inner.contains(&(*q * 2 + *d)))
            }));
        }
        ground.insert((parent, level), pages);
    }
    for (entity, pages, roots, ring) in &mut landscapes {
        if !pages.is_complete() {
            continue;
        }
        let inner = ring.and_then(|ring| Some((*ring, ground.get(&(ring.of, ring.level - 1))?)));
        let covered = |page: IVec2| {
            inner.is_some_and(|(ring, inner)| {
                ring.in_hole(page)
                    && [IVec2::ZERO, IVec2::X, IVec2::Y, IVec2::ONE]
                        .iter()
                        .all(|d| inner.contains(&(page * 2 + *d)))
            })
        };
        let pool = pages.pool().heights;
        let Some(mut roots) = roots else {
            commands.entity(entity).insert(LandscapeRoots {
                pool,
                roots: HashMap::default(),
            });
            continue;
        };
        if roots.pool != pool {
            // A new pool: start over.
            for (_, root) in roots.roots.drain() {
                commands.entity(root).try_despawn();
            }
            roots.pool = pool;
        }
        let gone: Vec<IVec2> = roots
            .roots
            .keys()
            .copied()
            .filter(|p| !pages.in_window(*p) || covered(*p))
            .collect();
        let missing: Vec<IVec2> = pages
            .window()
            .filter(|p| {
                // Inside the hole too while the level inside has no roots there: the coarse
                // ground stands in until the finer one is up.
                !roots.roots.contains_key(p) && !pages.dirty.contains(p) && !covered(*p)
            })
            .collect();
        if gone.is_empty() && missing.is_empty() {
            continue;
        }
        for page in gone {
            if let Some(root) = roots.roots.remove(&page) {
                commands.entity(root).try_despawn();
            }
        }
        for page in missing {
            let root = commands
                .spawn((
                    Name::new(format!("LandscapeRoot_{}_{}", page.x, page.y)),
                    ChildOf(entity),
                    LandscapeRoot {
                        landscape: entity,
                        page,
                    },
                    QuadTree::new(),
                ))
                .id();
            roots.roots.insert(page, root);
        }
    }
}

/// Tiles over re-evaluated pages, or pages whose holes changed, refill.
pub fn mark_refills(
    mut commands: Commands,
    mut evaluated: MessageReader<PagesEvaluated>,
    mut holes: MessageReader<HolesChanged>,
    landscapes: Query<&Landscape>,
    tiles: Query<(Entity, &LandscapeTile)>,
) {
    let changed = evaluated
        .read()
        .map(|m| (m.landscape, &m.pages))
        .chain(holes.read().map(|m| (m.landscape, &m.pages)));
    for (landscape_entity, pages) in changed {
        let Ok(landscape) = landscapes.get(landscape_entity) else {
            continue;
        };
        let texel = Vec2::splat(landscape.texel_size);
        let rects: Vec<Rect> = pages
            .iter()
            .map(|&p| {
                let r = landscape.page_rect(p);
                Rect::from_corners(r.min - texel, r.max + texel)
            })
            .collect();
        for (entity, tile) in &tiles {
            if tile.landscape == landscape_entity
                && rects.iter().any(|r| !r.intersect(tile.rect).is_empty())
            {
                commands.entity(entity).insert(Refill);
            }
        }
    }
}

/// Split / merge every page's tree around the viewer.
pub fn update_quadtrees(
    viewers: Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    landscapes: Query<(&Landscape, &LandscapePages, &GlobalTransform)>,
    rings: Query<&crate::ring::LandscapeRing>,
    mut roots: Query<(&LandscapeRoot, &mut QuadTree)>,
    prepared: Res<VulkanAssets<ProceduralMesh>>,
    meshes: Query<&ProceduralMesh3d>,
) {
    let Some(viewer) = viewer_position(&viewers, &cameras) else {
        return;
    };
    for (root, mut tree) in &mut roots {
        let Ok((landscape, pages, transform)) = landscapes.get(root.landscape) else {
            continue;
        };
        let mut local = viewer - transform.translation();
        if landscape.sphere.is_none() {
            // Height above the ground, not above y = 0: read on the finest level, the one with
            // a CPU mirror (a ring has none, and the viewer stands over its hole).
            let ground = match rings.get(root.landscape) {
                Ok(ring) => landscapes.get(ring.of).map(|(_, p, _)| p),
                Err(_) => Ok(pages),
            };
            if let Ok(ground) = ground {
                local.y = (local.y - ground.height_at(local.xz())).max(0.0);
            }
        }
        let size = landscape.page_size();
        let origin = root.page.as_vec2() * size;
        let sphere = landscape.sphere;
        tree.update(
            local,
            landscape.split_factor,
            landscape.max_lod,
            |e| meshes.get(e).is_ok_and(|m| prepared.get(&m.0).is_some()),
            |center, half| {
                let xz = origin + center * size;
                let at = match sphere {
                    Some(face) => face.point(xz, pages.height_at(xz)),
                    None => Vec3::new(xz.x, 0.0, xz.y),
                };
                (at, half * 2.0 * size)
            },
        );
    }
}

/// The step of each edge's coarser neighbour, from the neighbouring pages' trees.
fn edge_steps(level: u8, cell: IVec2, trees: &HashMap<IVec2, &QuadTree>) -> [u8; 4] {
    let n = 1i32 << level;
    let neighbours = [
        IVec2::new(cell.x, cell.y - 1),
        IVec2::new(cell.x, cell.y + 1),
        IVec2::new(cell.x - 1, cell.y),
        IVec2::new(cell.x + 1, cell.y),
    ];
    neighbours.map(|nc| {
        let page = IVec2::new(nc.x.div_euclid(n), nc.y.div_euclid(n));
        let Some(tree) = trees.get(&page) else {
            return 1;
        };
        let uv = (nc.as_vec2() + 0.5) / n as f32 - page.as_vec2();
        let Some(leaf) = tree.leaf_containing(uv) else {
            return 1;
        };
        let d = tree.nodes[leaf.0 as usize].depth;
        if d < level {
            (1u32 << (level - d).min(7)) as u8
        } else {
            1
        }
    })
}

/// The edges (-z, +z, -x, +x as bits) across which this landscape has no page: where the
/// tile meets another LOD level, whose tiles sample a different pool.
fn boundary_edges(level: u8, cell: IVec2, trees: &HashMap<IVec2, &QuadTree>) -> u8 {
    let n = 1i32 << level;
    [
        IVec2::new(cell.x, cell.y - 1),
        IVec2::new(cell.x, cell.y + 1),
        IVec2::new(cell.x - 1, cell.y),
        IVec2::new(cell.x + 1, cell.y),
    ]
    .iter()
    .enumerate()
    .filter(|(_, nc)| !trees.contains_key(&IVec2::new(nc.x.div_euclid(n), nc.y.div_euclid(n))))
    .fold(0u8, |bits, (i, _)| bits | (1 << i))
}

/// How far a level-boundary skirt drops, in the landscape's texels: past the gap a coarser
/// level's interpolation leaves against a finer one.
const SKIRT_TEXELS: f32 = 16.0;

fn tile_mesh(
    kernels: &LandscapeKernels,
    topologies: &mut PatchTopologies,
    params: &TileParams,
) -> ProceduralMesh {
    ProceduralMesh {
        vertex_count: vertex_count(params.res),
        indices: topologies.get(params.res),
        module: kernels.tile.clone(),
        entry: "tile_fill".into(),
        params: bytemuck::bytes_of(params).to_vec(),
    }
}

/// Spawn every unpopulated leaf, and respawn tiles whose coarser-neighbour pattern changed.
#[allow(clippy::too_many_arguments)]
pub fn spawn_tiles(
    mut commands: Commands,
    mut procedural: ResMut<Assets<ProceduralMesh>>,
    mut topologies: ResMut<PatchTopologies>,
    kernels: Res<LandscapeKernels>,
    engine_kernels: Res<ProceduralKernels>,
    modules: Res<ComputeModules>,
    landscapes: Query<(
        &Landscape,
        &LandscapePages,
        Option<&AuroraMaterial3d>,
        Has<crate::ring::LandscapeRing>,
    )>,
    mut roots: Query<(Entity, &LandscapeRoot, &mut QuadTree)>,
    tiles: Query<&LandscapeTile>,
) {
    if !engine_kernels.ready(&modules, &kernels.tile) {
        return;
    }
    struct Todo {
        root: Entity,
        leaf: NodeId,
        steps: [u8; 4],
        skirts: u8,
        respawn: bool,
    }
    let mut todo: Vec<Todo> = Vec::new();
    {
        let mut trees: HashMap<Entity, HashMap<IVec2, &QuadTree>> = HashMap::default();
        for (_, root, tree) in roots.iter() {
            trees
                .entry(root.landscape)
                .or_default()
                .insert(root.page, tree);
        }
        for (entity, root, tree) in roots.iter() {
            let trees = &trees[&root.landscape];
            // Only landscapes with LOD levels meet another level at their window's edge.
            let ringed = landscapes
                .get(root.landscape)
                .is_ok_and(|(l, _, _, ring)| ring || l.lod_rings > 0);
            for leaf in tree.leaf_node_ids() {
                let node = &tree.nodes[leaf.0 as usize];
                let n = 1i32 << node.depth;
                let (uv_min, _) = node.uv_bounds();
                let cell = root.page * n + (uv_min * n as f32).round().as_ivec2();
                let steps = edge_steps(node.depth, cell, trees);
                let skirts = if ringed {
                    boundary_edges(node.depth, cell, trees)
                } else {
                    0
                };
                let respawn = match node.patch_entity {
                    None => false,
                    Some(e)
                        if tiles
                            .get(e)
                            .is_ok_and(|t| t.edge_steps != steps || t.skirt_edges != skirts) =>
                    {
                        true
                    }
                    Some(_) => continue,
                };
                todo.push(Todo {
                    root: entity,
                    leaf,
                    steps,
                    skirts,
                    respawn,
                });
            }
        }
    }

    let mut budget = TILE_BUDGET;
    for item in todo {
        if budget == 0 {
            break;
        }
        let Ok((_, root, mut tree)) = roots.get_mut(item.root) else {
            continue;
        };
        let Ok((landscape, pages, material, _)) = landscapes.get(root.landscape) else {
            continue;
        };
        if item.respawn {
            tree.retire_leaf(item.leaf);
        }
        let node = &tree.nodes[item.leaf.0 as usize];
        let depth = node.depth;
        let n = 1i32 << depth;
        let (uv_min, uv_max) = node.uv_bounds();
        let cell = root.page * n + (uv_min * n as f32).round().as_ivec2();
        let size = landscape.page_size();
        let tile_size = size / n as f32;
        let origin = (root.page.as_vec2() + (uv_min + uv_max) * 0.5) * size;
        // Planet tiles sit at their planet-centred point; flat ones on the landscape plane.
        let at = match landscape.sphere {
            Some(face) => face.point(origin, 0.0),
            None => Vec3::new(origin.x, 0.0, origin.y),
        };
        let params = TileParams {
            pool: pages.pool(),
            level: depth as u32,
            cell_x: cell.x,
            cell_z: cell.y,
            res: landscape.patch_resolution.clamp(2, 255),
            root_size: size,
            origin_x: if landscape.sphere.is_some() {
                at.x
            } else {
                origin.x
            },
            origin_z: if landscape.sphere.is_some() {
                at.z
            } else {
                origin.y
            },
            skirt_depth: if item.skirts != 0 {
                SKIRT_TEXELS * landscape.texel_size
            } else {
                0.0
            },
            edge_steps: u32::from_le_bytes(item.steps),
            face: landscape.sphere.map_or(0, |f| f.face as u32),
            radius: landscape.sphere.map_or(0.0, |f| f.radius),
            origin_y: at.y,
            skirt_edges: item.skirts as u32,
            pad: 0,
            materials: pages.materials_address(),
        };
        let mesh = procedural.add(tile_mesh(&kernels, &mut topologies, &params));
        let entity = commands
            .spawn((
                Name::new(format!("LandscapeTile_L{}_({},{})", depth, cell.x, cell.y)),
                ProceduralMesh3d(mesh),
                material.cloned().unwrap_or_default(),
                // Under the landscape: it inherits its world (environment) and goes with it.
                ChildOf(root.landscape),
                Transform::from_translation(at),
                LandscapeTile {
                    landscape: root.landscape,
                    level: depth,
                    cell,
                    edge_steps: item.steps,
                    skirt_edges: item.skirts,
                    rect: Rect::from_center_size(origin, Vec2::splat(tile_size)),
                    params,
                },
            ))
            .id();
        tree.nodes[item.leaf.0 as usize].patch_entity = Some(entity);
        budget -= 1;
    }
}

/// Start a new mesh for each tile marked [`Refill`] that has none in flight.
pub fn refill_tiles(
    mut commands: Commands,
    mut procedural: ResMut<Assets<ProceduralMesh>>,
    mut topologies: ResMut<PatchTopologies>,
    kernels: Res<LandscapeKernels>,
    landscapes: Query<&LandscapePages>,
    mut tiles: Query<(Entity, &mut LandscapeTile), (With<Refill>, Without<PendingMesh>)>,
) {
    for (entity, mut tile) in tiles.iter_mut().take(TILE_BUDGET as usize) {
        // The window may have slid since the tile was made.
        if let Ok(pages) = landscapes.get(tile.landscape) {
            tile.params.pool = pages.pool();
        }
        let mesh = procedural.add(tile_mesh(&kernels, &mut topologies, &tile.params));
        commands
            .entity(entity)
            .remove::<Refill>()
            .insert(PendingMesh(mesh));
    }
}

/// Swap prepared refills in.
pub fn swap_refilled(
    mut commands: Commands,
    prepared: Res<VulkanAssets<ProceduralMesh>>,
    mut tiles: Query<(Entity, &PendingMesh, &mut ProceduralMesh3d)>,
) {
    for (entity, pending, mut mesh) in &mut tiles {
        if prepared.get(&pending.0).is_some() {
            mesh.0 = pending.0.clone();
            commands.entity(entity).remove::<PendingMesh>();
        }
    }
}

/// Despawn retired tiles once every replacement covering them is prepared.
pub fn despawn_covered_tiles(
    mut commands: Commands,
    mut roots: Query<&mut QuadTree, With<LandscapeRoot>>,
    prepared: Res<VulkanAssets<ProceduralMesh>>,
    meshes: Query<&ProceduralMesh3d, With<LandscapeTile>>,
) {
    for mut tree in &mut roots {
        let flushed =
            tree.flush_retired(|e| meshes.get(e).is_ok_and(|m| prepared.get(&m.0).is_some()));
        for entity in flushed {
            commands.entity(entity).try_despawn();
        }
    }
}

/// Tiles are roots of their own, so a despawned page tree takes its tiles down explicitly.
pub fn on_root_removed(
    remove: On<Remove<QuadTree>>,
    roots: Query<&QuadTree, With<LandscapeRoot>>,
    mut commands: Commands,
) {
    if let Ok(tree) = roots.get(remove.entity) {
        for e in tree.all_patch_entities() {
            commands.entity(e).try_despawn();
        }
    }
}

/// A despawned landscape takes its roots and colliders with it.
pub fn on_landscape_removed(
    remove: On<Remove<Landscape>>,
    roots: Query<(Entity, &LandscapeRoot)>,
    colliders: Query<(Entity, &PageCollider)>,
    mut commands: Commands,
) {
    for (e, root) in &roots {
        if root.landscape == remove.entity {
            commands.entity(e).try_despawn();
        }
    }
    for (e, collider) in &colliders {
        if collider.landscape == remove.entity {
            commands.entity(e).try_despawn();
        }
    }
}
