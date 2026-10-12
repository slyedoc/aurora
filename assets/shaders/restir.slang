// ReSTIR DI's compute passes (src/restir.rs), between the main trace and `restir_shade`:
//
//   restir_grid      ReGIR: every cell of a world grid around the camera keeps K light samples,
//                    resampled from the power CDF by power / distance^2 at the cell. The target
//                    is non-zero for every light, so each sample's contribution weight W is an
//                    unbiased 1/pdf for any light and the grid only reshapes variance.
//   restir_temporal  per pixel: RIS over N candidates from its cell (the global CDF outside
//                    the grid), folded to one sample (M = 1), merged with last frame's final
//                    reservoir at the reprojected pixel.
//   restir_spatial   merges k random neighbours' reservoirs (same-frame surfaces, rejection
//                    tested), with a boiling filter against lone hot samples.
//
// Every target is restirTarget (light_eval.slang): unshadowed, MIS-weighted against BRDF
// sampling with the CDF's pdf. The combination is the biased one (no neighbour visibility);
// the shade pass's shadow ray zeroes occluded winners for next frame (visibility reuse).
import rt_types;
import rand;
import disney;
import light_eval;

// `RestirFrameGpu` in src/restir.rs (scalar layout; pointers first).
struct RestirFrame {
    RestirSurface* surfaces;
    // Last frame's final reservoirs, this frame's temporal output, this frame's final.
    Reservoir* prev;
    Reservoir* temporal;
    Reservoir* final_;
    uint2* grid;            // (entry, asuint(W)) per cell sample; entry ~0 = none
    LightsHeader* lights;
    float4* instances;
    float4* positions;      // per light-table entry, this frame (restir_positions)
    float prev_view_proj[16];
    float grid_origin_x, grid_origin_y, grid_origin_z;
    float grid_cell;
    uint grid_x, grid_y, grid_z, grid_k;
    uint width, height, frame, epoch;
    uint candidates;        // N per pixel
    float m_cap;            // temporal history cap (frames' worth of M)
    uint spatial_samples;
    float spatial_radius;   // pixels
    float boiling;          // 0 = off
    float emissive_boost;
    uint grid_candidates;   // R per grid sample
    uint light_entries;     // 0 = light NEE off
};

struct RestirPush {
    RestirFrame* f;
    uint base;
    uint pad;
};
[[vk::push_constant]] RestirPush push;

static const uint NO_ENTRY = 0xFFFFFFFFu;

LightCtx ctxOf(RestirFrame* f) {
    LightCtx ctx;
    ctx.lights = f.lights;
    ctx.instances = f.instances;
    ctx.emissive_boost = f.emissive_boost;
    return ctx;
}

// A light-table entry's position: an analytic light's, a triangle's centroid.
float3 entryPosition(LightCtx ctx, uint e) {
    if (e < ctx.lights.analytic_count) {
        let al = ctx.lights.analytics[e];
        return float3(al.position_x, al.position_y, al.position_z);
    }
    let li = ctx.lights.linsts[lightsFindInstance(ctx, e)];
    uint g;
    float3 w0, w1, w2;
    lightTriangleWorld(ctx, li, e - li.entry_base, g, w0, w1, w2);
    return (w0 + w1 + w2) * (1.0 / 3.0);
}

// The cell holding `p`, or ~0 outside the grid.
uint gridCell(RestirFrame* f, float3 p) {
    let rel = (p - float3(f.grid_origin_x, f.grid_origin_y, f.grid_origin_z)) / f.grid_cell;
    let c = int3(floor(rel));
    if (any(c < int3(0)) || c.x >= int(f.grid_x) || c.y >= int(f.grid_y) || c.z >= int(f.grid_z)) {
        return NO_ENTRY;
    }
    return (uint(c.y) * f.grid_z + uint(c.z)) * f.grid_x + uint(c.x);
}

// Every light-table entry's position once per frame, for the grid's many candidates.
[shader("compute")]
[numthreads(64, 1, 1)]
void restir_positions(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let e = gid.x + push.base;
    if (f.light_entries == 0u || e >= f.lights.entry_count) {
        return;
    }
    f.positions[e] = float4(entryPosition(ctxOf(f), e), 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restir_grid(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    let cells = f.grid_x * f.grid_y * f.grid_z;
    if (idx >= cells * f.grid_k) {
        return;
    }
    let ctx = ctxOf(f);
    if (f.light_entries == 0u || ctx.lights.total_power <= 0.0) {
        f.grid[idx] = uint2(NO_ENTRY, 0u);
        return;
    }
    let cell = idx / f.grid_k;
    let cx = cell % f.grid_x;
    let cz = (cell / f.grid_x) % f.grid_z;
    let cy = cell / (f.grid_x * f.grid_z);
    let centre = float3(f.grid_origin_x, f.grid_origin_y, f.grid_origin_z)
        + (float3(float(cx), float(cy), float(cz)) + 0.5) * f.grid_cell;
    initRandom(uint2(cells * f.grid_k, 1u), uint2(idx, 0u), f.frame * 7u + 1u);
    let floor_d2 = f.grid_cell * f.grid_cell;
    var sel = NO_ENTRY;
    var sel_target = 0.0;
    var wsum = 0.0;
    let R = max(f.grid_candidates, 1u);
    for (uint r = 0u; r < R; r++) {
        float p_pick;
        let e = lightsPickEntry(ctx, randf() * ctx.lights.total_power, p_pick);
        if (p_pick <= 0.0) {
            continue;
        }
        let d = f.positions[e].xyz - centre;
        // power / d^2; the CDF picked by power, so the weight target / p is total / d^2.
        let target = p_pick * ctx.lights.total_power / max(dot(d, d), floor_d2);
        let w = target / p_pick;
        wsum += w;
        if (randf() * wsum < w) {
            sel = e;
            sel_target = target;
        }
    }
    let W = (sel != NO_ENTRY && sel_target > 0.0) ? wsum / (float(R) * sel_target) : 0.0;
    f.grid[idx] = uint2(W > 0.0 ? sel : NO_ENTRY, asuint(W));
}

// The pixel's reprojected position in last frame's image, or (-1, -1).
int2 reproject(RestirFrame* f, float3 prev_position) {
    float4 c;
    let m = f.prev_view_proj;
    let p = float4(prev_position, 1.0);
    // Column-major 4x4 (glam to_cols_array), as mat4MulVec in the raygen.
    c.x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
    c.y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
    c.w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    if (c.w <= 1.0e-4) {
        return int2(-1, -1);
    }
    let uv = (c.xy / c.w) * 0.5 + 0.5;
    // The raygen's pixel centre is (x, H - 1 - y) + 0.5: H - uv * H lands on row y.
    return int2(int(uv.x * float(f.width)), int(float(f.height) - uv.y * float(f.height)));
}

Reservoir emptyReservoir(RestirFrame* f, RestirSurface s) {
    Reservoir r;
    r.y = NO_ENTRY;
    r.uv_pk = 0u;
    r.W = 0.0;
    r.M = 0.0;
    r.epoch = f.epoch;
    r.normal_pk = s.normal_oct;
    r.depth = s.depth;
    r.frame = f.frame;
    return r;
}

// Whether a neighbour / history surface is close enough to reuse from.
bool similar(RestirSurface s, uint normal_oct, float depth) {
    let n0 = oct_to_float32x3(unpackHalf2(s.normal_oct));
    let n1 = oct_to_float32x3(unpackHalf2(normal_oct));
    return dot(n0, n1) > 0.9 && abs(depth - s.depth) < 0.1 * s.depth;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void restir_temporal(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let s = f.surfaces[idx];
    if (s.frame != f.frame) {
        return;
    }
    let ctx = ctxOf(f);
    let px = uint2(idx % f.width, idx / f.width);
    initRandom(uint2(f.width, f.height), px, f.frame * 7u + 2u);

    // Initial candidates: from the pixel's grid cell, else the global CDF.
    let cell = gridCell(f, s.position);
    let N = max(f.candidates, 1u);
    var sel_y = NO_ENTRY;
    var sel_uv = 0u;
    var sel_phat = 0.0;
    var wsum = 0.0;
    for (uint c = 0u; c < N; c++) {
        uint e;
        float p_src; // the candidate light's selection probability
        if (cell != NO_ENTRY) {
            let k = min(uint(randf() * float(f.grid_k)), f.grid_k - 1u);
            let gs = f.grid[cell * f.grid_k + k];
            let W = asfloat(gs.y);
            if (gs.x == NO_ENTRY || W <= 0.0) {
                continue;
            }
            e = gs.x;
            p_src = 1.0 / W;
        } else {
            e = lightsPickEntry(ctx, randf() * ctx.lights.total_power, p_src);
        }
        let uv_pk = packHalf2(float2(randf(), randf()));
        float p_area_cdf, p_cdf, dist;
        float3 dir;
        let val = restirTarget(ctx, s, e, uv_pk, p_area_cdf, p_cdf, dir, dist);
        let phat = restirLuma(val);
        if (phat <= 0.0 || p_cdf <= 0.0 || p_area_cdf <= 0.0) {
            continue;
        }
        // The point's pdf: the CDF's area pdf with the CDF pick swapped for this source's.
        let p_area = p_area_cdf / p_cdf * p_src;
        let w = phat / p_area;
        wsum += w;
        if (randf() * wsum < w) {
            sel_y = e;
            sel_uv = uv_pk;
            sel_phat = phat;
        }
    }
    // RIS over N folds into one sample: M = 1, W = wsum / (N phat).
    let W0 = sel_phat > 0.0 ? wsum / (float(N) * sel_phat) : 0.0;
    var r = emptyReservoir(f, s);
    var r_wsum = sel_phat * W0;
    var r_M = 1.0;
    var r_phat = sel_phat;
    r.y = sel_y;
    r.uv_pk = sel_uv;

    // Temporal: last frame's final reservoir at the reprojected pixel, re-evaluated here.
    let pp = reproject(f, s.prev_position);
    if (all(pp >= int2(0)) && pp.x < int(f.width) && pp.y < int(f.height)) {
        let prev = f.prev[uint(pp.y) * f.width + uint(pp.x)];
        // An occluded winner (W zeroed by the shade pass) still counts its samples: dropping
        // its M would average only the lucky, visible ones (biased bright).
        if (prev.frame + 1u == f.frame && prev.epoch == f.epoch && prev.M > 0.0
            && similar(s, prev.normal_pk, prev.depth)) {
            let Mprev = min(prev.M, f.m_cap);
            var phat = 0.0;
            if (prev.W > 0.0 && prev.y != NO_ENTRY) {
                float p_area_cdf, p_cdf, dist;
                float3 dir;
                phat = restirLuma(restirTarget(ctx, s, prev.y, prev.uv_pk, p_area_cdf, p_cdf, dir, dist));
            }
            if (phat > 0.0) {
                let w = phat * prev.W * Mprev;
                r_wsum += w;
                if (randf() * r_wsum < w) {
                    r.y = prev.y;
                    r.uv_pk = prev.uv_pk;
                    r_phat = phat;
                }
            }
            r_M += Mprev;
        }
    }
    r.M = min(r_M, f.m_cap);
    r.W = (r_phat > 0.0 && r_M > 0.0) ? r_wsum / (r_M * r_phat) : 0.0;
    if (r.W <= 0.0) {
        r.y = NO_ENTRY;
    }
    f.temporal[idx] = r;
}

// The target of sample (entry, uv) at surface `s` (0 for an empty sample), and the area density
// with which the light's sampling map put that uv's point there (the CDF pick divided out).
float targetAt(LightCtx ctx, RestirSurface s, uint e, uint uv_pk, out float density) {
    density = 0.0;
    if (e == NO_ENTRY) {
        return 0.0;
    }
    float p_area_cdf, p_cdf, dist;
    float3 dir;
    let phat = restirLuma(restirTarget(ctx, s, e, uv_pk, p_area_cdf, p_cdf, dir, dist));
    density = p_cdf > 0.0 ? p_area_cdf / p_cdf : 0.0;
    return phat;
}

float targetAt(LightCtx ctx, RestirSurface s, uint e, uint uv_pk) {
    float density;
    return targetAt(ctx, s, e, uv_pk, density);
}

static const uint MAX_SPATIAL = 16;

// Spatial reuse with pairwise MIS (Bitterli; "A Gentle Introduction to ReSTIR", 2023): each
// neighbour i pairs with this pixel (the canonical c) by balance weights over their targets and
// sample counts, a_i(x) = M_i p_i(x) / (M_i p_i(x) + M_c p_c(x)); neighbour i's sample takes
// a_i / k, the canonical sample the mean of the (1 - a_i). The weights sum to one for any x, so a
// neighbour whose light is useless here no longer pulls the result down, as the plain 1/M
// combination did (a quarter darker than the truth at night).
[shader("compute")]
[numthreads(64, 1, 1)]
void restir_spatial(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let s = f.surfaces[idx];
    if (s.frame != f.frame) {
        return;
    }
    let ctx = ctxOf(f);
    let px = int2(int(idx % f.width), int(idx / f.width));
    initRandom(uint2(f.width, f.height), uint2(px), f.frame * 7u + 3u);

    let own = f.temporal[idx];
    // Neighbours: a random disc, re-rotated every pixel and frame; those on a similar surface.
    uint picked[MAX_SPATIAL];
    var k = 0u;
    let S = min(f.spatial_samples, MAX_SPATIAL);
    for (uint i = 0u; i < S; i++) {
        let a = randf() * 2.0 * PI;
        let rad = f.spatial_radius * sqrt(randf());
        let q = px + int2(int(cos(a) * rad), int(sin(a) * rad));
        if (any(q < int2(0)) || q.x >= int(f.width) || q.y >= int(f.height) || all(q == px)) {
            continue;
        }
        let qi = uint(q.y) * f.width + uint(q.x);
        let qs = f.surfaces[qi];
        if (qs.frame != f.frame || f.temporal[qi].M <= 0.0 || !similar(s, qs.normal_oct, qs.depth)) {
            continue;
        }
        picked[k] = qi;
        k++;
    }
    if (k == 0u) {
        f.final_[idx] = own;
        return;
    }

    let Mc = max(own.M, 0.0);
    let own_valid = own.y != NO_ENTRY && own.W > 0.0;
    let pc_yc = own_valid ? targetAt(ctx, s, own.y, own.uv_pk) : 0.0;
    var r = emptyReservoir(f, s);
    var r_wsum = 0.0;
    var r_phat = 0.0;
    var r_M = Mc;
    var mc_sum = 0.0;  // sum over i of (1 - a_i(y_c))
    var neighbour_c = 0.0;
    for (uint j = 0u; j < k; j++) {
        let qi = picked[j];
        let qs = f.surfaces[qi];
        let nr = f.temporal[qi];
        let Mi = nr.M;
        r_M += Mi;
        // The canonical sample's share against this neighbour.
        if (own_valid) {
            let pi_yc = targetAt(ctx, qs, own.y, own.uv_pk);
            let den = Mi * pi_yc + Mc * pc_yc;
            mc_sum += den > 0.0 ? (Mc * pc_yc) / den : 1.0;
        }
        // The neighbour's sample.
        if (nr.y == NO_ENTRY || nr.W <= 0.0) {
            continue;
        }
        float dens_here, dens_there;
        let pc_yi = targetAt(ctx, s, nr.y, nr.uv_pk, dens_here);
        if (pc_yi <= 0.0 || dens_here <= 0.0) {
            continue;
        }
        let pi_yi = targetAt(ctx, qs, nr.y, nr.uv_pk, dens_there);
        let den = Mi * pi_yi + Mc * pc_yi;
        let ai = den > 0.0 ? (Mi * pi_yi) / den : 0.0;
        // The sample is a uv replayed here, not a fixed point: a sphere lamp's cone sampling
        // lands it elsewhere on the sphere from another surface, so its weight takes the shift's
        // Jacobian, the ratio of the two sampling densities (1 for area-sampled triangles).
        let jacobian = dens_there / dens_here;
        let w = (ai / float(k)) * pc_yi * nr.W * jacobian;
        neighbour_c += pc_yi * nr.W;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r.y = nr.y;
            r.uv_pk = nr.uv_pk;
            r_phat = pc_yi;
        }
    }
    if (own_valid && pc_yc > 0.0) {
        let w = (mc_sum / float(k)) * pc_yc * own.W;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r.y = own.y;
            r.uv_pk = own.uv_pk;
            r_phat = pc_yc;
        }
    }

    // Boiling filter: a reservoir far hotter than its neighbourhood is a lone lucky sample the
    // denoiser would hold on to; drop it rather than spread it.
    if (f.boiling > 0.0 && k >= 3u && own_valid && r.y == own.y) {
        let mean = neighbour_c / float(k);
        if (pc_yc * own.W > mean / f.boiling) {
            r_phat = 0.0;
        }
    }

    // The MIS weights sum to one: no 1/M here.
    r.M = min(r_M, f.m_cap);
    r.W = r_phat > 0.0 ? r_wsum / r_phat : 0.0;
    if (r.W <= 0.0) {
        r.y = NO_ENTRY;
    }
    f.final_[idx] = r;
}
