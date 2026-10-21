// ReSTIR PT's compute pass (src/restir_gi.rs), after the path tracer
// wrote this frame's samples (raygen.slang `main`): each completed in place as a reservoir of one
// (W = 1 / pdf, its target here). Without spatial reuse that is also the final reservoir; with it,
// raygen.slang `gi_spatial` resamples (its shift replays prefixes, so it traces), then `gi_shade`
// shades the winners.
//
// A sample's contribution at its own pixel is T L (the path's own estimate); its target is the
// luminance of T L pdf, the integrand in the measure the reconnection direction is drawn in (for
// k = 1 that is the primary vertex's BSDF, cosine and radiance: ReSTIR GI's).
import rt_types;
import rand;
import bsdf;
import light_eval;

struct GiPush {
    GiFrame* f;
    uint base;
    uint pad;
};
[[vk::push_constant]] GiPush push;

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
    var C = float3(0.0);
    if (live && cand.pdf > 0.0) {
        C = ptOwnTarget(s, cand);
    }
    let phat = luma(C);
    // The density includes the prefix's survival of Russian roulette.
    let W = phat > 0.0 && cand.q > 0.0 ? 1.0 / (cand.pdf * cand.q) : 0.0;
    if (live) {
        let c = (uint*)(f.candidates + idx);
        c[5] = asuint(W);
        c[7] = asuint(phat);
    }
    if (f.spatial_samples == 0u) {
        GiReservoir r;
        r.from_ = s.position;
        r.C = packRadiance(C);
        r.to = cand.x_s;
        r.W = W;
        r.frame = live ? f.frame : f.frame - 1u;
        r.flags = GI_FRESH;
        r.n_from = packHalf2(float32x3_to_oct(oct_to_float32x3(unpackHalf2(s.geo_normal_oct))));
        r.n_to = cand.n_s_oct;
        giReservoirStore(f.final_ + idx, r);
    }
}
