//! The patch index list: one canonical triangle list per resolution, shared by every patch
//! (`ProceduralMesh::indices` is an `Arc`). Vertex layout, mirrored by `landscape/tile.slang`:
//! `res²` grid vertices row-major, `(res-1)²` quad centres row-major, then 4 skirt rows of
//! `res` vertices (edges bottom, top, left, right) the kernel drops by the skirt depth. Each
//! quad fans four triangles from its centre (WoW's ADT pattern): no diagonal to cut across a
//! ridge, and the centre samples the heights between the corners.

use std::sync::Arc;

use bevy::{platform::collections::HashMap, prelude::*};

/// Vertices a patch of `res` has: the grid, the quad centres and the four skirt rows.
pub fn vertex_count(res: u32) -> u32 {
    res * res + (res - 1) * (res - 1) + 4 * res
}

/// Grid coord of edge `edge` (0 = bottom, 1 = top, 2 = left, 3 = right), position `i`.
fn edge_grid_coord(edge: u32, i: u32, res: u32) -> (u32, u32) {
    match edge {
        0 => (i, 0),
        1 => (i, res - 1),
        2 => (0, i),
        _ => (res - 1, i),
    }
}

/// The triangle list for a patch of `res` vertices per edge.
pub fn patch_indices(res: u32) -> Vec<u32> {
    let quads = res - 1;
    let mut indices = Vec::with_capacity((quads * quads * 12 + 4 * quads * 6) as usize);
    for y in 0..quads {
        for x in 0..quads {
            let a = y * res + x;
            let b = a + res;
            let c = b + 1;
            let d = a + 1;
            let m = res * res + y * quads + x;
            // Wound so +y faces front.
            indices.extend_from_slice(&[a, b, m, b, c, m, c, d, m, d, a, m]);
        }
    }
    // Skirts: the edge's grid vertices against its dropped row.
    let base = res * res + quads * quads;
    for edge in 0..4u32 {
        let skirt_row = base + edge * res;
        for i in 0..quads {
            let (gx0, gy0) = edge_grid_coord(edge, i, res);
            let (gx1, gy1) = edge_grid_coord(edge, i + 1, res);
            let g0 = gy0 * res + gx0;
            let g1 = gy1 * res + gx1;
            let s0 = skirt_row + i;
            let s1 = skirt_row + i + 1;
            match edge {
                0 | 3 => indices.extend_from_slice(&[g0, s1, s0, s1, g0, g1]),
                _ => indices.extend_from_slice(&[g0, s1, g1, s1, g0, s0]),
            }
        }
    }
    indices
}

/// Index lists by resolution, built once.
#[derive(Resource, Default)]
pub struct PatchTopologies {
    map: HashMap<u32, Arc<[u32]>>,
}

impl PatchTopologies {
    pub fn get(&mut self, res: u32) -> Arc<[u32]> {
        self.map
            .entry(res)
            .or_insert_with(|| Arc::from(patch_indices(res)))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_faces_up() {
        let res = 5u32;
        let idx = patch_indices(res);
        let p = |i: u32| Vec3::new((i % res) as f32, 0.0, (i / res) as f32);
        let quads = res - 1;
        let p = |i: u32| {
            if i < res * res {
                p(i)
            } else {
                let k = i - res * res;
                Vec3::new((k % quads) as f32 + 0.5, 0.0, (k / quads) as f32 + 0.5)
            }
        };
        for t in idx[..(quads * quads * 12) as usize].chunks(3) {
            let n = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            assert!(n.y > 0.0, "{t:?}");
        }
    }

    #[test]
    fn indices_stay_in_range() {
        for res in [2u32, 9, 17, 33, 65] {
            let n = vertex_count(res);
            let idx = patch_indices(res);
            assert_eq!(idx.len() % 3, 0);
            assert!(idx.iter().all(|&i| i < n), "res {res}");
            let q = res - 1;
            assert_eq!(idx.len() as u32, q * q * 12 + 4 * q * 6);
        }
    }
}
