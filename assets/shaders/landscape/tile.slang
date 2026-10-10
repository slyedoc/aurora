// Landscape tile fill: one thread per vertex of a quadtree tile, run by aurora's
// ProceduralMesh worker (push block = ProceduralHeader + TilePush, aurora_landscape tiles.rs).
// Heights come from the evaluated pages. The lattice is derived from integer (level, cell)
// inputs so same-level neighbours sample bit-identical positions on their shared edge;
// edges against a coarser neighbour are stitched onto its straight edge.
//
// Vertex layout (topology.rs): res² grid vertices row-major, (res-1)² quad centres, then 4
// skirt rows of `res` (bottom, top, left, right) dropped `skirt_depth`.
import procedural;
import pages;
import sphere;

struct TilePush {
    ProceduralHeader header;
    PagePool pool;
    uint level;
    int cell_x;
    int cell_z;
    uint res;
    float root_size;
    float origin_x;
    float origin_z;
    float skirt_depth;
    // Per edge (-z, +z, -x, +x), a byte each: the coarser neighbour's step in this tile's
    // vertex steps (1 = none).
    uint edge_steps;
    // Planet faces: the cube face and radius (0 = flat). There, origin_x/y/z are the tile's
    // planet-centred position (vertices are written relative to it), not face coordinates.
    uint face;
    float radius;
    float origin_y;
};
[[vk::push_constant]] TilePush push;

struct Sample {
    double x;
    double z;
    float h;
};

// The lattice point (li, lj) in half-steps of the vertex grid (li = 2 * gx).
Sample sample_at(int li, int lj) {
    let steps = 2u * (push.res - 1u);
    let tile = double(push.root_size) / double(1u << min(push.level, 30u));
    let step = tile / double(steps);
    Sample s;
    s.x = double(push.cell_x) * tile + double(li) * step;
    s.z = double(push.cell_z) * tile + double(lj) * step;
    s.h = sample_height(push.pool, float2(float(s.x), float(s.z)));
    return s;
}

float3 position(Sample s) {
    if (push.radius > 0.0) {
        let dir = face_direction(push.face, push.radius, float2(float(s.x), float(s.z)));
        return dir * (push.radius + s.h) - float3(push.origin_x, push.origin_y, push.origin_z);
    }
    return float3(float(s.x - double(push.origin_x)), s.h, float(s.z - double(push.origin_z)));
}

float stitched_height(uint gx, uint gy, uint res, float h) {
    let last = res - 1;
    uint step = 1;
    uint t = 0;
    bool along_x = false;
    if (gy == 0) { step = push.edge_steps & 0xffu; t = gx; along_x = true; }
    else if (gy == last) { step = (push.edge_steps >> 8) & 0xffu; t = gx; along_x = true; }
    else if (gx == 0) { step = (push.edge_steps >> 16) & 0xffu; t = gy; }
    else if (gx == last) { step = (push.edge_steps >> 24) & 0xffu; t = gy; }
    if (step <= 1 || (t % step) == 0) {
        return h;
    }
    let t0 = t - (t % step);
    let t1 = min(t0 + step, last);
    let f = float(t - t0) / float(t1 - t0);
    float h0;
    float h1;
    if (along_x) {
        h0 = sample_at(int(2 * t0), int(2 * gy)).h;
        h1 = sample_at(int(2 * t1), int(2 * gy)).h;
    } else {
        h0 = sample_at(int(2 * gx), int(2 * t0)).h;
        h1 = sample_at(int(2 * gx), int(2 * t1)).h;
    }
    return lerp(h0, h1, f);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tile_fill(uint3 gid: SV_DispatchThreadID) {
    let i = push.header.base + gid.x;
    if (i >= push.header.vertex_count) {
        return;
    }
    let res = push.res;
    let quads = res - 1;
    let centres = res * res + quads * quads;
    uint gx;
    uint gy;
    int li;
    int lj;
    var skirt = false;
    var centre = false;
    if (i < res * res) {
        gx = i % res;
        gy = i / res;
        li = int(2 * gx);
        lj = int(2 * gy);
    } else if (i < centres) {
        // Quad centres sit on the odd half-steps; never on an edge, so never stitched.
        let k = i - res * res;
        gx = k % quads;
        gy = k / quads;
        li = int(2 * gx + 1);
        lj = int(2 * gy + 1);
        centre = true;
    } else {
        let k = i - centres;
        let t = k % res;
        skirt = true;
        switch (k / res) {
            case 0: gx = t; gy = 0; break;
            case 1: gx = t; gy = res - 1; break;
            case 2: gx = 0; gy = t; break;
            default: gx = res - 1; gy = t; break;
        }
        li = int(2 * gx);
        lj = int(2 * gy);
    }

    var c = sample_at(li, lj);
    if (!centre) {
        c.h = stitched_height(gx, gy, res, c.h);
    }
    var pos = position(c);
    let px = position(sample_at(li + 1, lj));
    let nx = position(sample_at(li - 1, lj));
    let pz = position(sample_at(li, lj + 1));
    let nz = position(sample_at(li, lj - 1));
    var normal = cross(pz - nz, px - nx);
    let nlen = length(normal);
    normal = nlen > 1e-10 ? normal / nlen : float3(0.0, 1.0, 0.0);
    let up = push.radius > 0.0 ? normalize(pos + float3(push.origin_x, push.origin_y, push.origin_z))
                               : float3(0.0, 1.0, 0.0);
    if (dot(normal, up) < 0.0) {
        normal = -normal;
    }
    if (skirt) {
        pos -= up * push.skirt_depth;
    }

    Vertex out;
    out.px = pos.x;
    out.py = pos.y;
    out.pz = pos.z;
    out.nx = normal.x;
    out.ny = normal.y;
    out.nz = normal.z;
    out.u = float(li) / float(2 * quads);
    out.v = float(lj) / float(2 * quads);
    push.header.vertices[i] = out;
}
