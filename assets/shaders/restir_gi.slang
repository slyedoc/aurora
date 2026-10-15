// ReSTIR GI's passes (src/restir_gi.rs), after the path tracer wrote this frame's candidates
// (raygen.slang `main`) and before `gi_shade` shades the winners:
//
//   gi_candidates  per pixel: completes its candidate in place as a reservoir of one (W = 1 / pdf
//                  and its target here); without spatial reuse, that is the final reservoir.
//   gi_spatial     pairwise MIS over k random neighbours' candidates (same-frame surfaces,
//                  rejection tested), with the boiling filter.
//
// A sample is a secondary hit and the radiance it sends toward the vertex that found it; it moves
// to another vertex by reconnection (the hit and its radiance kept), whose solid-angle Jacobian
// is |cos_new| d_old^2 / (|cos_old| d_new^2) at the hit. Targets are giContribution's luminance
// (light_eval.slang). No neighbour visibility (biased, as DI); gi_shade traces the winner's.
// There is no temporal reuse: a sample held across frames reads to Ray Reconstruction as detail
// and boils (docs/plans/restir-gi-notes.md).
import rt_types;
import rand;
import disney;
import light_eval;

// `GiFrameGpu` in src/restir_gi.rs (scalar layout; pointers first).
struct GiFrame {
    GiCandidate* candidates;
    RestirSurface* surfaces;
    GiReservoir* final_;
    uint64_t pad_ptr;
    uint width;
    uint height;
    uint frame;
    uint spatial_samples;
    float spatial_radius;
    float boiling;
    uint pad[2];
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

// The hit's cosine toward `x` and their squared distance.
float hitCos(float3 x_s, float3 n_s, float3 x, out float d2) {
    let d = x - x_s;
    d2 = max(dot(d, d), 1.0e-8);
    return dot(n_s, d) * rsqrt(d2);
}

GiReservoir reservoirOf(float3 x_s, uint n_s_oct, uint L, float W, uint frame, uint flags) {
    GiReservoir r;
    r.x_s = x_s;
    r.n_s_oct = n_s_oct;
    r.L = L;
    r.W = W;
    r.frame = frame;
    r.flags = flags;
    return r;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gi_candidates(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let s = restirSurfaceLoad(f.surfaces + idx);
    let cand = giCandidateLoad(f.candidates + idx);
    // A pixel not taking part this frame: its light stayed in the path.
    let live = s.frame == f.frame && cand.frame == f.frame;
    var phat = 0.0;
    if (live && cand.pdf > 0.0) {
        float cos_s, d2;
        phat = luma(giContribution(s, cand.x_s, oct_to_float32x3(unpackHalf2(cand.n_s_oct)),
                                   unpackRgb9e5(cand.L), cos_s, d2));
    }
    let W = phat > 0.0 ? 1.0 / cand.pdf : 0.0;
    if (live) {
        let c = (uint*)(f.candidates + idx);
        c[5] = asuint(W);
        c[7] = asuint(phat);
    }
    if (f.spatial_samples == 0u) {
        giReservoirStore(f.final_ + idx, reservoirOf(cand.x_s, cand.n_s_oct, cand.L, W,
                                                     live ? f.frame : f.frame - 1u, GI_FRESH));
    }
}

static const uint MAX_SPATIAL = 16;

uint pcg(uint v) {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Neighbour `i` of pixel `px` this frame: a random point of the disc, from a hash of the three
// (so the same pixels come back on a second walk, with nothing held between).
int2 neighbourPixel(GiFrame* f, int2 px, uint i, uint seed) {
    let h0 = pcg((uint(px.y) * f.width + uint(px.x)) * MAX_SPATIAL + i + seed);
    let h1 = pcg(h0);
    let a = float(h0 >> 8u) * (2.0 * PI / 16777216.0);
    let rad = f.spatial_radius * sqrt(float(h1 >> 8u) / 16777216.0);
    return px + int2(int(cos(a) * rad), int(sin(a) * rad));
}

// The neighbour's index when it can be reused from (inside, not this pixel, taking part this
// frame on a similar surface), else ~0.
uint usableNeighbour(GiFrame* f, RestirSurface s, int2 px, uint i, uint seed) {
    let q = neighbourPixel(f, px, i, seed);
    if (any(q < int2(0)) || q.x >= int(f.width) || q.y >= int(f.height) || all(q == px)) {
        return 0xFFFFFFFFu;
    }
    let qi = uint(q.y) * f.width + uint(q.x);
    let sd = (uint4*)(f.surfaces + qi);
    let a = loadAligned<16>(sd);
    let b = loadAligned<16>(sd + 1);
    let e = loadAligned<16>(sd + 3);
    let cb = loadAligned<16>((uint4*)(f.candidates + qi) + 1);
    if (a.w != f.frame || cb.z != f.frame || !restirSimilar(s, e.x, asfloat(b.w))) {
        return 0xFFFFFFFFu;
    }
    return qi;
}

// Pairwise MIS as restir.slang's restir_spatial, every target compared in the canonical pixel's
// solid angle: a neighbour's sample shifted here weighs its own target against ours times the
// shift's Jacobian, and the canonical sample at a neighbour likewise. Each candidate is a
// reservoir of one (M = 1). The usable neighbours are a bit mask, their pixels re-derived from the
// hash: a local array of them faulted on the GPU.
[shader("compute")]
[numthreads(64, 1, 1)]
void gi_spatial(uint3 gid: SV_DispatchThreadID) {
    let f = push.f;
    let idx = gid.x + push.base;
    if (idx >= f.width * f.height) {
        return;
    }
    let s = restirSurfaceLoad(f.surfaces + idx);
    let own = giCandidateLoad(f.candidates + idx);
    if (s.frame != f.frame || own.frame != f.frame) {
        giReservoirStore(f.final_ + idx, reservoirOf(float3(0.0), 0u, 0u, 0.0, f.frame - 1u, 0u));
        return;
    }
    // After gi_candidates: `pdf` holds W, `phat` the target here.
    let own_W = own.pdf;
    let px = int2(int(idx % f.width), int(idx / f.width));
    initRandom(uint2(f.width, f.height), uint2(px), f.frame * 7u + 6u);
    let seed = pcg(f.frame);

    let S = min(f.spatial_samples, MAX_SPATIAL);
    var usable = 0u;
    for (uint i = 0u; i < S; i++) {
        if (usableNeighbour(f, s, px, i, seed) != 0xFFFFFFFFu) {
            usable |= 1u << i;
        }
    }
    let k = countbits(usable);
    let own_valid = own_W > 0.0 && own_W < 1.0e20;
    if (k == 0u) {
        giReservoirStore(f.final_ + idx, reservoirOf(own.x_s, own.n_s_oct, own.L,
                                                     own_valid ? own_W : 0.0, f.frame, GI_FRESH));
        return;
    }

    let n_c = oct_to_float32x3(unpackHalf2(own.n_s_oct));
    let L_c = unpackRgb9e5(own.L);
    float own_d2;
    let own_cos = hitCos(own.x_s, n_c, s.position, own_d2);
    let pc_yc = own_valid ? own.phat : 0.0;
    let own_shareable = giShareable(own.n_s_oct);
    var r = reservoirOf(float3(0.0), 0u, 0u, 0.0, f.frame, 0u);
    var r_wsum = 0.0;
    var r_phat = 0.0;
    var mc_sum = 0.0;
    var neighbour_c = 0.0;
    var fresh = false;
    for (var bits = usable; bits != 0u; bits &= bits - 1u) {
        let q = neighbourPixel(f, px, firstbitlow(bits), seed);
        let qi = uint(q.y) * f.width + uint(q.x);
        let qs = restirSurfaceLoad(f.surfaces + qi);
        let nr = giCandidateLoad(f.candidates + qi);
        let nr_W = nr.pdf;
        // The canonical sample at the neighbour, in our measure.
        if (own_valid) {
            float cos_q, d2_q;
            let phat_q = luma(giContribution(qs, own.x_s, n_c, L_c, cos_q, d2_q));
            let J = own_cos > 0.0 ? (cos_q * own_d2) / (own_cos * d2_q) : 0.0;
            let ok = own_shareable && phat_q > 0.0 && J > 1.0 / MAX_JACOBIAN && J < MAX_JACOBIAN;
            let pi_c = ok ? phat_q * J : 0.0;
            let den = pi_c + pc_yc;
            mc_sum += den > 0.0 ? pc_yc / den : 1.0;
        }
        // The neighbour's sample, shifted here.
        if (!(nr_W > 0.0 && nr_W < 1.0e20) || !giShareable(nr.n_s_oct)) {
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
        let den = nr.phat + pc_yi * J;
        let ai = den > 0.0 ? nr.phat / den : 0.0;
        let w = (ai / float(k)) * pc_yi * nr_W * J;
        neighbour_c += pc_yi * nr_W * J;
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
        let w = (mc_sum / float(k)) * pc_yc * own_W;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r.x_s = own.x_s;
            r.n_s_oct = own.n_s_oct;
            r.L = own.L;
            r_phat = pc_yc;
            fresh = true;
        }
    }

    // Boiling filter (restir_spatial's): a winner far hotter than its neighbourhood is dropped.
    if (f.boiling > 0.0 && k >= 3u && own_valid && fresh) {
        let mean = neighbour_c / float(k);
        if (pc_yc * own_W > mean / f.boiling) {
            r_phat = 0.0;
        }
    }

    r.W = r_phat > 0.0 ? r_wsum / r_phat : 0.0;
    r.flags = (r.W > 0.0 && fresh) ? GI_FRESH : 0u;
    giReservoirStore(f.final_ + idx, r);
}
