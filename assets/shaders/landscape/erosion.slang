// Hydraulic + thermal erosion over a region of a landscape (aurora_landscape erosion.rs): the
// virtual-pipe model (Mei, Decaudin & Hu 2007) on the region's texel grid. Water rains, flows
// along pipes to the four neighbours by pressure, carries sediment up to a capacity set by its
// speed and the slope, drops it where it slows, evaporates; material steeper than the talus
// angle slips downhill. The bake's result is the change in ground height, stored in the layer's
// delta pages (faded to zero at the region's edge) and added like a sculpt layer.
import pages;

static const float G = 9.81;

// `ErosionGpu` in erosion.rs.
struct Erosion {
    float* b;
    float* b0;
    float* d;
    float* s;
    float* s2;
    float4* flux;
    float2* vel;
    // Thermal outflow: axis neighbours (-x, +x, -z, +z), then diagonals
    // ((-1,-1), (+1,-1), (-1,+1), (+1,+1)).
    float4* slip;
    float4* slip2;
    PagePool pool;
    // The grid: cell (0, 0)'s centre in landscape metres, and metres between cells.
    float x0;
    float z0;
    uint nx;
    uint nz;
    float dt;
    float rain;
    float evaporation;
    float capacity;
    float erode;
    float deposit;
    float talus;
    float thermal;
    float edge;
    float cell;
};

struct ErosionPush {
    Erosion* e;
    float* page;
    int page_x;
    int page_z;
    uint iteration;
    uint pad;
};
[[vk::push_constant]] ErosionPush push;

uint cell_of(int x, int z) {
    let e = push.e;
    return uint(clamp(z, 0, int(e.nz) - 1)) * e.nx + uint(clamp(x, 0, int(e.nx) - 1));
}

bool in_grid(uint i) {
    return i < push.e.nx * push.e.nz;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let xz = float2(e.x0, e.z0) + float2(float(i % e.nx), float(i / e.nx)) * e.cell;
    let h = sample_height(e.pool, xz);
    e.b[i] = h;
    e.b0[i] = h;
    e.d[i] = 0.0;
    e.s[i] = 0.0;
    e.s2[i] = 0.0;
    e.flux[i] = float4(0.0);
    e.vel[i] = float2(0.0);
    e.slip[i] = float4(0.0);
    e.slip2[i] = float4(0.0);
}

uint pcg(uint v) {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rain(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (in_grid(i)) {
        // Uneven rain (mean 1x) so channels form irregularly, not in lockstep.
        let r = float(pcg(i ^ pcg(push.iteration)) & 0xFFFFu) / 65535.0;
        push.e.d[i] += push.e.rain * push.e.dt * (0.25 + 1.5 * r);
    }
}

// flux = (to -x, to +x, to -z, to +z); the region's border is a wall.
[shader("compute")]
[numthreads(64, 1, 1)]
void flux(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let x = int(i % e.nx);
    let z = int(i / e.nx);
    let l = e.cell;
    let h = e.b[i] + e.d[i];
    float4 f = e.flux[i];
    let hl = x > 0 ? h - (e.b[cell_of(x - 1, z)] + e.d[cell_of(x - 1, z)]) : 0.0;
    let hr = x < int(e.nx) - 1 ? h - (e.b[cell_of(x + 1, z)] + e.d[cell_of(x + 1, z)]) : 0.0;
    let hb = z > 0 ? h - (e.b[cell_of(x, z - 1)] + e.d[cell_of(x, z - 1)]) : 0.0;
    let ht = z < int(e.nz) - 1 ? h - (e.b[cell_of(x, z + 1)] + e.d[cell_of(x, z + 1)]) : 0.0;
    f = max(float4(0.0), f + e.dt * G * float4(hl, hr, hb, ht) / l);
    if (x == 0) f.x = 0.0;
    if (x == int(e.nx) - 1) f.y = 0.0;
    if (z == 0) f.z = 0.0;
    if (z == int(e.nz) - 1) f.w = 0.0;
    let total = f.x + f.y + f.z + f.w;
    if (total > 0.0) {
        f *= min(1.0, e.d[i] * l * l / (total * e.dt));
    }
    e.flux[i] = f;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void water(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let x = int(i % e.nx);
    let z = int(i / e.nx);
    let l = e.cell;
    let f = e.flux[i];
    let from_l = x > 0 ? e.flux[cell_of(x - 1, z)].y : 0.0;
    let from_r = x < int(e.nx) - 1 ? e.flux[cell_of(x + 1, z)].x : 0.0;
    let from_b = z > 0 ? e.flux[cell_of(x, z - 1)].w : 0.0;
    let from_t = z < int(e.nz) - 1 ? e.flux[cell_of(x, z + 1)].z : 0.0;
    let inflow = from_l + from_r + from_b + from_t;
    let outflow = f.x + f.y + f.z + f.w;
    let d0 = e.d[i];
    let d1 = max(0.0, d0 + e.dt * (inflow - outflow) / (l * l));
    let depth = max((d0 + d1) * 0.5, 1.0e-4);
    let wx = (from_l - f.x + f.y - from_r) * 0.5;
    let wz = (from_b - f.z + f.w - from_t) * 0.5;
    // At most half a cell per step: faster water outruns the grid and piles sediment up.
    let v = float2(wx, wz) / (l * depth);
    let vmax = 0.5 * l / e.dt;
    let speed = length(v);
    e.vel[i] = speed > vmax ? v * (vmax / speed) : v;
    e.d[i] = d1;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void erode(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let x = int(i % e.nx);
    let z = int(i / e.nx);
    let l = e.cell;
    let dx = (e.b[cell_of(x + 1, z)] - e.b[cell_of(x - 1, z)]) / (2.0 * l);
    let dz = (e.b[cell_of(x, z + 1)] - e.b[cell_of(x, z - 1)]) / (2.0 * l);
    let tilt = sqrt(dx * dx + dz * dz) / sqrt(1.0 + dx * dx + dz * dz);
    // Shallow water carries less: capacity fades out under a few centimetres.
    let depth_limit = saturate(e.d[i] / 0.05);
    let c = e.capacity * max(tilt, 0.05) * length(e.vel[i]) * depth_limit;
    let s = e.s[i];
    if (c > s) {
        let amount = min(e.erode * (c - s) * e.dt, 0.05);
        e.b[i] -= amount;
        e.s[i] = s + amount;
    } else {
        let amount = min(e.deposit * (s - c) * e.dt, 0.05);
        e.b[i] += amount;
        e.s[i] = s - amount;
    }
}

// Semi-Lagrangian: sediment arrives from upstream.
[shader("compute")]
[numthreads(64, 1, 1)]
void advect(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let p = float2(float(i % e.nx), float(i / e.nx)) - e.vel[i] * e.dt / e.cell;
    let f = floor(p);
    let t = p - f;
    let x = int(f.x);
    let z = int(f.y);
    e.s2[i] = lerp(lerp(e.s[cell_of(x, z)], e.s[cell_of(x + 1, z)], t.x),
                   lerp(e.s[cell_of(x, z + 1)], e.s[cell_of(x + 1, z + 1)], t.x), t.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaporate(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (in_grid(i)) {
        let e = push.e;
        e.s[i] = e.s2[i];
        e.d[i] *= max(0.0, 1.0 - e.evaporation * e.dt);
    }
}

// Thermal, pass 1: how much of each cell slides to each lower neighbour (8 of them, the
// diagonals at their true distance) past the talus.
[shader("compute")]
[numthreads(64, 1, 1)]
void slip_out(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let x = int(i % e.nx);
    let z = int(i / e.nx);
    let h = e.b[i];
    let limit = e.talus * e.cell;
    let axis = float4(h - e.b[cell_of(x - 1, z)], h - e.b[cell_of(x + 1, z)],
                      h - e.b[cell_of(x, z - 1)], h - e.b[cell_of(x, z + 1)]);
    let diag = float4(h - e.b[cell_of(x - 1, z - 1)], h - e.b[cell_of(x + 1, z - 1)],
                      h - e.b[cell_of(x - 1, z + 1)], h - e.b[cell_of(x + 1, z + 1)]);
    let ea = max(axis - limit, float4(0.0));
    let ed = max(diag - limit * 1.41421356, float4(0.0)) * 0.70710678;
    let total = dot(ea, float4(1.0)) + dot(ed, float4(1.0));
    if (total <= 0.0) {
        e.slip[i] = float4(0.0);
        e.slip2[i] = float4(0.0);
        return;
    }
    let peak = max(max(max(ea.x, ea.y), max(ea.z, ea.w)), max(max(ed.x, ed.y), max(ed.z, ed.w)));
    let moved = e.thermal * peak * 0.5;
    e.slip[i] = ea / total * moved;
    e.slip2[i] = ed / total * moved;
}

// Thermal, pass 2: give and take.
[shader("compute")]
[numthreads(64, 1, 1)]
void slip_in(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (!in_grid(i)) {
        return;
    }
    let e = push.e;
    let x = int(i % e.nx);
    let z = int(i / e.nx);
    let nx = int(e.nx) - 1;
    let nz = int(e.nz) - 1;
    let a = e.slip[i];
    let d = e.slip2[i];
    var gain = 0.0;
    if (x > 0) gain += e.slip[cell_of(x - 1, z)].y;
    if (x < nx) gain += e.slip[cell_of(x + 1, z)].x;
    if (z > 0) gain += e.slip[cell_of(x, z - 1)].w;
    if (z < nz) gain += e.slip[cell_of(x, z + 1)].z;
    if (x > 0 && z > 0) gain += e.slip2[cell_of(x - 1, z - 1)].w;
    if (x < nx && z > 0) gain += e.slip2[cell_of(x + 1, z - 1)].z;
    if (x > 0 && z < nz) gain += e.slip2[cell_of(x - 1, z + 1)].y;
    if (x < nx && z < nz) gain += e.slip2[cell_of(x + 1, z + 1)].x;
    e.b[i] += gain - dot(a, float4(1.0)) - dot(d, float4(1.0));
}

float change_at(int x, int z) {
    let e = push.e;
    let c = cell_of(x, z);
    return e.b[c] - e.b0[c];
}

// One delta page: the region's change, upsampled to texels, faded out over `edge` metres at
// its border.
[shader("compute")]
[numthreads(64, 1, 1)]
void write_delta(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let e = push.e;
    let t = int2(push.page_x * PAGE + int(gid.x % PAGE), push.page_z * PAGE + int(gid.x / PAGE));
    let g = (float2(t) * e.pool.texel - float2(e.x0, e.z0)) / e.cell;
    let span = float2(float(e.nx - 1), float(e.nz - 1));
    if (any(g < 0.0) || any(g > span)) {
        push.page[gid.x] = 0.0;
        return;
    }
    let f0 = floor(g);
    let f = g - f0;
    let x = int(f0.x);
    let z = int(f0.y);
    let change = lerp(lerp(change_at(x, z), change_at(x + 1, z), f.x),
                      lerp(change_at(x, z + 1), change_at(x + 1, z + 1), f.x), f.y);
    let border = min(min(g.x, span.x - g.x), min(g.y, span.y - g.y)) * e.cell;
    push.page[gid.x] = change * smoothstep(0.0, max(e.edge, 1.0e-3), border);
}
