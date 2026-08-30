// Radiance-cache resolve kernel (see src/sharc.rs; SHaRC-style hash grid).
//
// The raygen deposits per-path outgoing radiance into per-entry atomic accumulators and
// queries the resolved value to terminate deep paths. This kernel runs once per frame over
// the whole table: it folds each entry's fresh accumulation into an exponential moving
// average (weighted by sample counts), clears the accumulators, and evicts entries that
// have not been touched for a while.

// Must match SharcEntry in types.glsl. 48 bytes. Resolved values are f32 bit patterns in
// uints (no 16-bit-float SPIR-V capability needed).
struct SharcEntry {
    uint tag;      // 0 = empty; hash tag of the voxel key otherwise.
    uint frame;    // Last frame the entry was deposited into or queried.
    uint acc_r;    // Fixed-point (1 nit) radiance accumulated this frame.
    uint acc_g;
    uint acc_b;
    uint acc_n;    // Samples accumulated this frame.
    uint res_r;    // Resolved radiance (f32 bits).
    uint res_g;
    uint res_b;
    uint conf;     // Confidence: samples folded in (f32 bits).
    uint pad0;
    uint pad1;
};

struct SharcResolveParams {
    SharcEntry* entries;
    uint entry_count;
    uint frame;
    uint base;
    uint pad;
};
[[vk::push_constant]] SharcResolveParams resolve_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void sharc_resolve(uint3 gid: SV_DispatchThreadID) {
    let p = resolve_params;
    let i = p.base + gid.x;
    if (i >= p.entry_count) {
        return;
    }
    var e = p.entries[i];
    if (e.tag == 0) {
        return;
    }
    // Untouched for ~10 seconds: evict.
    if (p.frame - e.frame > 600) {
        p.entries[i] = SharcEntry();
        return;
    }
    if (e.acc_n == 0) {
        return;
    }
    let fresh = float3(float(e.acc_r), float(e.acc_g), float(e.acc_b)) / float(e.acc_n);
    let old = float3(asfloat(e.res_r), asfloat(e.res_g), asfloat(e.res_b));
    let conf = asfloat(e.conf);
    // Count-weighted blend, floored so the cache keeps tracking lighting changes.
    let conf_new = min(conf + float(e.acc_n), 1024.0);
    let w = max(float(e.acc_n) / conf_new, 0.05);
    let resolved = lerp(old, fresh, conf <= 0.0 ? 1.0 : w);
    p.entries[i].res_r = asuint(resolved.x);
    p.entries[i].res_g = asuint(resolved.y);
    p.entries[i].res_b = asuint(resolved.z);
    p.entries[i].conf = asuint(conf_new);
    p.entries[i].acc_r = 0;
    p.entries[i].acc_g = 0;
    p.entries[i].acc_b = 0;
    p.entries[i].acc_n = 0;
}
