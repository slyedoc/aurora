// ReSTIR GI's reuse passes (src/restir_gi.rs), after the path tracer wrote this frame's
// candidates (raygen.slang `main`) and before `gi_shade` shades the winners:
//
//   gi_temporal  per pixel: its own candidate (M = 1, W = 1 / pdf), merged with last frame's
//                final reservoir near the reprojected pixel when temporal reuse is on.
//   gi_spatial   pairwise MIS over k random neighbours' temporal reservoirs (same-frame
//                surfaces, rejection tested), with the boiling filter.
//
// A sample is a secondary hit and the radiance it sends toward the vertex that found it; it moves
// to another vertex by reconnection (the hit and its radiance kept), whose solid-angle Jacobian
// is |cos_new| d_old^2 / (|cos_old| d_new^2) at the hit. Targets are giContribution's luminance
// (light_eval.slang). No neighbour visibility (biased, as DI); gi_shade traces the winner's.
import rt_types;
import rand;
import disney;
import light_eval;

// `GiFrameGpu` in src/restir_gi.rs (scalar layout; pointers first).
struct GiFrame {
    GiCandidate* candidates;
    RestirSurface* surfaces;
    // Last frame's final reservoirs, this frame's temporal output, this frame's final.
    GiReservoir* prev;
    GiReservoir* temporal;
    GiReservoir* final_;
    uint64_t pad_ptr;
    float prev_view_proj[16];
    uint width;
    uint height;
    uint frame;
    float m_cap;
    uint spatial_samples;
    float spatial_radius;
    float boiling;
    uint pad;
};

struct GiPush {
    GiFrame* f;
    uint base;
    uint pad;
};
[[vk::push_constant]] GiPush push;

// A shift is rejected when the hit's solid angle changes by more than this either way (a
// neighbour's sample from across a corner).
static const float MAX_JACOBIAN = 8.0;

GiReservoir emptyReservoir(GiFrame* f, RestirSurface s) {
    GiReservoir r;
    r.x_s = float3(0.0);
    r.n_s_oct = 0u;
    r.L = 0u;
    r.W = 0.0;
    r.M = 0.0;
    r.phat = 0.0;
    r.x_v = s.position;
    r.depth = s.depth;
    r.n_v_oct = s.normal_oct;
    r.frame = f.frame;
    r.flags = 0u;
    r.pad = 0u;
    return r;
}

// The hit's cosine toward `x` and their squared distance.
float hitCos(float3 x_s, float3 n_s, float3 x, out float d2) {
    let d = x - x_s;
    d2 = max(dot(d, d), 1.0e-8);
    return dot(n_s, d) * rsqrt(d2);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gi_temporal(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let s = restirSurfaceLoad(f.surfaces + idx);
    let cand = giCandidateLoad(f.candidates + idx);
    var r = emptyReservoir(f, s);
    // A pixel not taking part this frame: its light stayed in the path, and nothing reuses it.
    if (s.frame != f.frame || cand.frame != f.frame) {
        r.frame = f.frame - 1u;
        giReservoirStore(f.temporal + idx, r);
        return;
    }
    let px = uint2(idx % f.width, idx / f.width);
    initRandom(uint2(f.width, f.height), px, f.frame * 7u + 4u);

    // The canonical sample: this frame's candidate.
    var wsum = 0.0;
    var sel_phat = 0.0;
    var M = 1.0;
    let n_c = oct_to_float32x3(unpackHalf2(cand.n_s_oct));
    if (cand.pdf > 0.0) {
        float cos_s, d2;
        let phat = luma(giContribution(s, cand.x_s, n_c, unpackRgb9e5(cand.L), cos_s, d2));
        if (phat > 0.0) {
            wsum = phat / cand.pdf;
            sel_phat = phat;
            r.x_s = cand.x_s;
            r.n_s_oct = cand.n_s_oct;
            r.L = cand.L;
            r.flags = GI_FRESH;
        }
    }

    // Last frame's final reservoir at the reprojected pixel, shifted here.
    // Last frame's reservoir from a random pixel of the 3x3 around the reprojected one (temporal
    // permutation, Wyman & Panteleev 2021): a sample held in one pixel for frames reads to Ray
    // Reconstruction as signal and boils; wandering, it decorrelates.
    var pp = restirReproject(&f.prev_view_proj[0], f.width, f.height, s.prev_position);
    pp += int2(int(randf() * 3.0) - 1, int(randf() * 3.0) - 1);
    if (f.m_cap > 0.0 && all(pp >= int2(0)) && pp.x < int(f.width) && pp.y < int(f.height)) {
        let prev = giReservoirLoad(f.prev + (uint(pp.y) * f.width + uint(pp.x)));
        // An occluded winner (W zeroed by gi_shade) still counts its samples.
        if (prev.frame + 1u == f.frame && prev.M > 0.0 && restirSimilar(s, prev.n_v_oct, prev.depth)) {
            let Mprev = min(prev.M, f.m_cap);
            if (prev.W > 0.0 && prev.W < 1.0e20) {
                let n_p = oct_to_float32x3(unpackHalf2(prev.n_s_oct));
                float cos_new, d2_new, d2_old;
                let phat = luma(giContribution(s, prev.x_s, n_p, unpackRgb9e5(prev.L), cos_new, d2_new));
                let cos_old = hitCos(prev.x_s, n_p, prev.x_v, d2_old);
                let J = cos_old > 0.0 ? (cos_new * d2_old) / (cos_old * d2_new) : 0.0;
                if (phat > 0.0 && J > 1.0 / MAX_JACOBIAN && J < MAX_JACOBIAN) {
                    let w = phat * prev.W * Mprev * J;
                    wsum += w;
                    if (randf() * wsum < w) {
                        r.x_s = prev.x_s;
                        r.n_s_oct = prev.n_s_oct;
                        r.L = prev.L;
                        r.flags = 0u;
                        sel_phat = phat;
                    }
                }
            }
            M += Mprev;
        }
    }
    r.W = sel_phat > 0.0 ? wsum / (M * sel_phat) : 0.0;
    r.M = min(M, max(f.m_cap, 1.0));
    r.phat = r.W > 0.0 ? sel_phat : 0.0;
    giReservoirStore(f.temporal + idx, r);
}

static const uint MAX_SPATIAL = 16;

// Neighbour `i` of pixel `px` this frame: a random point of the disc, from a hash of the three
// (so the same pixels come back on a second walk, with nothing held between).
int2 neighbourPixel(GiFrame* f, int2 px, uint i) {
    let h0 = tea(uint(px.y) * f.width + uint(px.x), f.frame * 16u + i);
    let h1 = tea(h0, 0x9e3779b9u);
    let a = float(h0 & 0xFFFFFFu) * (2.0 * PI / 16777216.0);
    let rad = f.spatial_radius * sqrt(float(h1 & 0xFFFFFFu) / 16777216.0);
    return px + int2(int(cos(a) * rad), int(sin(a) * rad));
}

// The neighbour's index when it can be reused from (inside, not this pixel, taking part this
// frame on a similar surface), else ~0.
uint usableNeighbour(GiFrame* f, RestirSurface s, int2 px, uint i) {
    let q = neighbourPixel(f, px, i);
    if (any(q < int2(0)) || q.x >= int(f.width) || q.y >= int(f.height) || all(q == px)) {
        return 0xFFFFFFFFu;
    }
    let qi = uint(q.y) * f.width + uint(q.x);
    let d = (uint4*)(f.temporal + qi);
    let b = loadAligned<16>(d + 1);
    let c = loadAligned<16>(d + 2);
    let e = loadAligned<16>(d + 3);
    if (e.y != f.frame || !(asfloat(b.z) > 0.0) || !restirSimilar(s, e.x, asfloat(c.w))) {
        return 0xFFFFFFFFu;
    }
    return qi;
}

// Pairwise MIS as restir.slang's restir_spatial, every target compared in the canonical pixel's
// solid angle: a neighbour's sample shifted here weighs its own target against ours times the
// shift's Jacobian, and the canonical sample at a neighbour likewise. The neighbours are walked
// twice (counted, then merged) rather than kept in a local array.
[shader("compute")]
[numthreads(64, 1, 1)]
void gi_spatial(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let own = giReservoirLoad(f.temporal + idx);
    if (own.frame != f.frame) {
        giReservoirStore(f.final_ + idx, own);
        return;
    }
    let s = restirSurfaceLoad(f.surfaces + idx);
    let px = int2(int(idx % f.width), int(idx / f.width));
    initRandom(uint2(f.width, f.height), uint2(px), f.frame * 7u + 6u);

    let S = min(f.spatial_samples, MAX_SPATIAL);
    var k = 0u;
    for (uint i = 0u; i < S; i++) {
        if (usableNeighbour(f, s, px, i) != 0xFFFFFFFFu) {
            k++;
        }
    }
    if (k == 0u) {
        giReservoirStore(f.final_ + idx, own);
        return;
    }

    let Mc = own.M;
    let own_valid = own.W > 0.0 && own.W < 1.0e20;
    let n_c = oct_to_float32x3(unpackHalf2(own.n_s_oct));
    let L_c = unpackRgb9e5(own.L);
    float own_d2;
    let own_cos = hitCos(own.x_s, n_c, s.position, own_d2);
    let pc_yc = own_valid ? own.phat : 0.0;
    var r = emptyReservoir(f, s);
    var r_wsum = 0.0;
    var r_phat = 0.0;
    var r_M = Mc;
    var mc_sum = 0.0;
    var neighbour_c = 0.0;
    var fresh = false;
    for (uint n = 0u; n < S; n++) {
        let qi = usableNeighbour(f, s, px, n);
        if (qi == 0xFFFFFFFFu) {
            continue;
        }
        let qs = restirSurfaceLoad(f.surfaces + qi);
        let nr = giReservoirLoad(f.temporal + qi);
        r_M += nr.M;
        // The canonical sample at the neighbour, in our measure.
        if (own_valid) {
            float cos_q, d2_q;
            let phat_q = luma(giContribution(qs, own.x_s, n_c, L_c, cos_q, d2_q));
            let J = own_cos > 0.0 ? (cos_q * own_d2) / (own_cos * d2_q) : 0.0;
            let ok = phat_q > 0.0 && J > 1.0 / MAX_JACOBIAN && J < MAX_JACOBIAN;
            let pi_c = ok ? phat_q * J : 0.0;
            let den = nr.M * pi_c + Mc * pc_yc;
            mc_sum += den > 0.0 ? (Mc * pc_yc) / den : 1.0;
        }
        // The neighbour's sample, shifted here.
        if (!(nr.W > 0.0 && nr.W < 1.0e20)) {
            continue;
        }
        let n_i = oct_to_float32x3(unpackHalf2(nr.n_s_oct));
        float cos_c, d2_c, d2_i;
        let pc_yi = luma(giContribution(s, nr.x_s, n_i, unpackRgb9e5(nr.L), cos_c, d2_c));
        let cos_i = hitCos(nr.x_s, n_i, qs.position, d2_i);
        let J = cos_i > 0.0 ? (cos_c * d2_i) / (cos_i * d2_c) : 0.0;
        if (!(pc_yi > 0.0) || !(J > 1.0 / MAX_JACOBIAN && J < MAX_JACOBIAN)) {
            continue;
        }
        let den = nr.M * nr.phat + Mc * pc_yi * J;
        let ai = den > 0.0 ? (nr.M * nr.phat) / den : 0.0;
        let w = (ai / float(k)) * pc_yi * nr.W * J;
        neighbour_c += pc_yi * nr.W * J;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r.x_s = nr.x_s;
            r.n_s_oct = nr.n_s_oct;
            r.L = nr.L;
            r_phat = pc_yi;
            fresh = false;
        }
    }
    if (own_valid && pc_yc > 0.0) {
        let w = (mc_sum / float(k)) * pc_yc * own.W;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r.x_s = own.x_s;
            r.n_s_oct = own.n_s_oct;
            r.L = own.L;
            r_phat = pc_yc;
            fresh = (own.flags & GI_FRESH) != 0u;
        }
    }

    // Boiling filter (restir_spatial's): a winner far hotter than its neighbourhood is dropped.
    if (f.boiling > 0.0 && k >= 3u && own_valid && r_phat == pc_yc && all(r.x_s == own.x_s)) {
        let mean = neighbour_c / float(k);
        if (pc_yc * own.W > mean / f.boiling) {
            r_phat = 0.0;
        }
    }

    r.M = min(r_M, max(f.m_cap, 1.0));
    r.W = r_phat > 0.0 ? r_wsum / r_phat : 0.0;
    r.phat = r.W > 0.0 ? r_phat : 0.0;
    r.flags = (r.W > 0.0 && fresh) ? GI_FRESH : 0u;
    giReservoirStore(f.final_ + idx, r);
}
