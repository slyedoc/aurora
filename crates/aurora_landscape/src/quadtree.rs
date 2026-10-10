//! A LOD quadtree over a unit square (zero's planet tree; one per landscape page). The tree knows nothing
//! about the mapping: `update` takes a closure from (uv centre, half extent) to (world
//! position, edge length) and splits where the camera is within `split_factor` edges.

use bevy::prelude::*;

/// A split region merges only when the camera retreats past split-distance × this — dead-band
/// so the boundary ring doesn't flip split↔merge per frame.
pub const MERGE_HYSTERESIS: f32 = 1.4;

/// Index into the [`QuadTree`] node arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A single node in the quadtree. Stored in an arena on [`QuadTree`].
pub struct QuadNode {
    /// Center of this node in uv space `[0,1]x[0,1]`.
    pub center: Vec2,
    /// Half the width/height of this node in uv space.
    pub half_extent: f32,
    /// Subdivision depth (0 = root).
    pub depth: u8,
    /// Child node indices, or `None` if this is a leaf.
    pub children: Option<[NodeId; 4]>,
    /// The mesh entity for this leaf, if one has been spawned.
    pub patch_entity: Option<Entity>,
}

impl QuadNode {
    fn new(center: Vec2, half_extent: f32, depth: u8) -> Self {
        Self {
            center,
            half_extent,
            depth,
            children: None,
            patch_entity: None,
        }
    }

    /// UV bounds: `(min, max)`.
    pub fn uv_bounds(&self) -> (Vec2, Vec2) {
        let min = self.center - Vec2::splat(self.half_extent);
        let max = self.center + Vec2::splat(self.half_extent);
        (min, max)
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

/// Component on each surface-cell entity (a cube face, a flat root cell). Contains the
/// quadtree as an arena.
#[derive(Component)]
pub struct QuadTree {
    pub nodes: Vec<QuadNode>,
    pub root: NodeId,
    /// Stale patches kept alive (uv center, half-extent, entity) until their replacement
    /// leaves are populated — avoids a visible hole during LOD transitions.
    retired: Vec<(Vec2, f32, Entity)>,
}

impl Default for QuadTree {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadTree {
    pub fn new() -> Self {
        let root_node = QuadNode::new(Vec2::splat(0.5), 0.5, 0);
        Self {
            nodes: vec![root_node],
            root: NodeId(0),
            retired: Vec::new(),
        }
    }

    fn alloc(&mut self, node: QuadNode) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    /// Recursively update the tree around `camera_pos` (in the space `map` returns): splits
    /// nodes whose edge length × `split_factor` exceeds their distance to the camera, merges
    /// distant ones. `map` takes a node's (uv centre, uv half extent) to its (world position,
    /// world edge length); `is_ready` says whether a patch entity is traceable.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        split_factor: f32,
        max_lod: u8,
        is_ready: impl Fn(Entity) -> bool,
        map: impl Fn(Vec2, f32) -> (Vec3, f32),
    ) {
        // Rebuild from scratch each frame — tree is small (hundreds of nodes max)
        // and this avoids complex incremental update logic for v1.
        let old_nodes = std::mem::take(&mut self.nodes);
        self.nodes.clear();

        // Re-create root
        let root_node = QuadNode::new(Vec2::splat(0.5), 0.5, 0);
        self.nodes.push(root_node);
        self.root = NodeId(0);

        // Collect old entities for potential reuse
        let mut old_entities: Vec<(Vec2, f32, Entity)> = old_nodes
            .iter()
            .filter_map(|n| n.patch_entity.map(|e| (n.center, n.half_extent, e)))
            .collect();

        // Regions whose CURRENT patch is already traceable — only these may split deeper.
        // Backpressure: on a fast descent a region otherwise retires through several
        // generations before any replacement is ready, stacking coplanar layers that
        // z-fight (patch flicker) and flooding the GPU gen pipeline with work that gets
        // thrown away.
        let mut ready_regions: std::collections::HashSet<(u32, u32, u32)> = old_entities
            .iter()
            .filter(|(_, _, e)| is_ready(*e))
            .map(|(c, h, _)| (c.x.to_bits(), c.y.to_bits(), h.to_bits()))
            .collect();
        // A region that was ALREADY split stays splittable — its interior node carries no
        // entity, but the split is in effect and its children gate themselves. Without this
        // the tree collapses to the root every frame (interior ≠ ready) and oscillates
        // between coarse LODs.
        let split_regions: std::collections::HashSet<(u32, u32, u32)> = old_nodes
            .iter()
            .filter(|n| n.children.is_some())
            .map(|n| {
                (
                    n.center.x.to_bits(),
                    n.center.y.to_bits(),
                    n.half_extent.to_bits(),
                )
            })
            .collect();
        ready_regions.extend(split_regions.iter().copied());

        // Recursively subdivide
        self.subdivide(
            self.root,
            camera_pos,
            split_factor,
            max_lod,
            &ready_regions,
            &split_regions,
            &map,
        );

        // Assign old entities to matching leaf nodes, queue non-matching for despawn
        for leaf_id in self.leaf_node_ids() {
            let node = &self.nodes[leaf_id.0 as usize];
            // Try to find a matching old entity (same center + extent)
            if let Some(idx) = old_entities.iter().position(|(c, h, _)| {
                (*c - node.center).length() < 1e-6 && (*h - node.half_extent).abs() < 1e-6
            }) {
                let (_, _, entity) = old_entities.swap_remove(idx);
                self.nodes[leaf_id.0 as usize].patch_entity = Some(entity);
            }
        }

        // Retire (don't despawn) leftover patches — flushed once their area is recovered.
        for (c, h, entity) in old_entities {
            self.retired.push((c, h, entity));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn subdivide(
        &mut self,
        node_id: NodeId,
        camera_pos: Vec3,
        split_factor: f32,
        max_lod: u8,
        ready_regions: &std::collections::HashSet<(u32, u32, u32)>,
        split_regions: &std::collections::HashSet<(u32, u32, u32)>,
        map: &impl Fn(Vec2, f32) -> (Vec3, f32),
    ) {
        let node = &self.nodes[node_id.0 as usize];
        if node.depth >= max_lod {
            return;
        }

        let key = (
            node.center.x.to_bits(),
            node.center.y.to_bits(),
            node.half_extent.to_bits(),
        );
        let should_split = {
            let (world_pos, edge) = map(node.center, node.half_extent);
            let dist = camera_pos.distance(world_pos);
            // Hysteresis: an already-split region stays split until the camera retreats
            // well past the split ring — the bare threshold flips split↔merge every frame
            // at the boundary (full regen both ways).
            let factor = if split_regions.contains(&key) {
                split_factor * MERGE_HYSTERESIS
            } else {
                split_factor
            };
            dist < edge * factor
        };

        // One generation in flight per region: split only once THIS region's own patch is
        // traceable (see `ready_regions` in [`Self::update`]).
        if !should_split || !ready_regions.contains(&key) {
            return;
        }

        let center = self.nodes[node_id.0 as usize].center;
        let child_half = self.nodes[node_id.0 as usize].half_extent * 0.5;
        let child_depth = self.nodes[node_id.0 as usize].depth + 1;

        let offsets = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(1.0, 1.0),
        ];

        let children: [NodeId; 4] = std::array::from_fn(|i| {
            let child_center = center + offsets[i] * child_half;
            self.alloc(QuadNode::new(child_center, child_half, child_depth))
        });

        self.nodes[node_id.0 as usize].children = Some(children);

        // Recurse into children
        for child_id in children {
            self.subdivide(
                child_id,
                camera_pos,
                split_factor,
                max_lod,
                ready_regions,
                split_regions,
                map,
            );
        }
    }

    /// Retired patches whose every overlapping current leaf is now populated (replacement
    /// geometry is in place). Returns them to despawn and drops them from `retired`.
    ///
    /// A leaf only counts as populated once its patch is actually TRACEABLE
    /// (`is_traceable`), not merely spawned — otherwise retiring the old patch opens a hole
    /// while the replacement's vertex data is still being generated.
    pub fn flush_retired(&mut self, is_traceable: impl Fn(Entity) -> bool) -> Vec<Entity> {
        let leaves: Vec<(Vec2, f32, bool)> = self
            .leaf_node_ids()
            .into_iter()
            .map(|id| {
                let n = &self.nodes[id.0 as usize];
                (
                    n.center,
                    n.half_extent,
                    n.patch_entity.is_some_and(&is_traceable),
                )
            })
            .collect();
        let mut ready = Vec::new();
        self.retired.retain(|&(c, h, entity)| {
            let covered = leaves
                .iter()
                .filter(|(lc, lh, _)| (lc.x - c.x).abs() < lh + h && (lc.y - c.y).abs() < lh + h)
                .all(|&(_, _, populated)| populated);
            if covered {
                ready.push(entity);
            }
            !covered
        });
        ready
    }

    /// Hand a leaf's current patch to the retired list (it stays traceable until the leaf's
    /// replacement is ready) and clear the leaf so the spawner builds a new one -- for a
    /// patch whose inputs changed (a neighbour's LOD, say) without the tree changing.
    pub fn retire_leaf(&mut self, leaf: NodeId) {
        let node = &mut self.nodes[leaf.0 as usize];
        if let Some(entity) = node.patch_entity.take() {
            self.retired.push((node.center, node.half_extent, entity));
        }
    }

    /// Patches kept alive until their replacements are ready.
    pub fn retired_count(&self) -> usize {
        self.retired.len()
    }

    /// Every patch entity the tree references: live leaves and retired ones.
    pub fn all_patch_entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.nodes
            .iter()
            .filter_map(|n| n.patch_entity)
            .chain(self.retired.iter().map(|(_, _, e)| *e))
    }

    /// Leaf node whose UV bounds contain `uv`, descending from the root.
    /// `None` if `uv` is outside `[0,1]²`.
    pub fn leaf_containing(&self, uv: Vec2) -> Option<NodeId> {
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return None;
        }
        let mut id = self.root;
        loop {
            let node = &self.nodes[id.0 as usize];
            match node.children {
                None => return Some(id),
                Some(children) => {
                    let ix = (uv.x > node.center.x) as usize;
                    let iy = (uv.y > node.center.y) as usize;
                    id = children[iy * 2 + ix];
                }
            }
        }
    }

    /// Returns all leaf node IDs.
    pub fn leaf_node_ids(&self) -> Vec<NodeId> {
        let mut leaves = Vec::new();
        self.collect_leaves(self.root, &mut leaves);
        leaves
    }

    fn collect_leaves(&self, node_id: NodeId, out: &mut Vec<NodeId>) {
        let node = &self.nodes[node_id.0 as usize];
        if let Some(children) = node.children {
            for child in children {
                self.collect_leaves(child, out);
            }
        } else {
            out.push(node_id);
        }
    }
}
