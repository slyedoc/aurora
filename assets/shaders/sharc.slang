// Radiance-cache resolve kernel (see src/sharc.rs; SHaRC-style hash grid).
//
// The raygen deposits per-path outgoing radiance into per-entry atomic accumulators and
// queries the resolved value to terminate deep paths. This kernel runs once per frame over
// the whole table: it folds each entry's fresh accumulation into an exponential moving
// average (weighted by sample counts), clears the accumulators, and evicts entries that
// have not been touched for a while.

// Must match SharcEntry in types.glsl. 32 bytes; resolved values are packed f16 pairs
// (shaderFloat16 is enabled at device creation).
struct SharcEntry {
    uint tag;      // 0 = empty; hash tag of the voxel key otherwise.
    uint frame;    // Last frame the entry was deposited into or queried.
    uint acc_r;    // Fixed-point (1 nit) radiance accumulated this frame.
    uint acc_g;
    uint acc_b;
    uint acc_n;    // Samples accumulated this frame.
    uint res_rg;   // Resolved radiance, 2xf16 (r, g).
    uint res_bc;   // Resolved radiance b + confidence (samples folded in), 2xf16.
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
    let old = float3(f16tof32(e.res_rg & 0xFFFF), f16tof32(e.res_rg >> 16), f16tof32(e.res_bc & 0xFFFF));
    let conf = f16tof32(e.res_bc >> 16);
    // Count-weighted blend, floored so the cache keeps tracking lighting changes.
    let conf_new = min(conf + float(e.acc_n), 1024.0);
    let w = max(float(e.acc_n) / conf_new, 0.05);
    let resolved = lerp(old, fresh, conf <= 0.0 ? 1.0 : w);
    p.entries[i].res_rg = f32tof16(resolved.x) | (f32tof16(resolved.y) << 16);
    p.entries[i].res_bc = f32tof16(resolved.z) | (f32tof16(conf_new) << 16);
    p.entries[i].acc_r = 0;
    p.entries[i].acc_g = 0;
    p.entries[i].acc_b = 0;
    p.entries[i].acc_n = 0;
}
