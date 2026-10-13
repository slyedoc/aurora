// Radiance-cache resolve kernel (see src/sharc.rs; SHaRC-style hash grid).
//
// The raygen deposits per-path outgoing radiance into per-entry atomic accumulators and
// queries the resolved value to terminate deep paths. This kernel runs once per frame over
// the whole table: it folds each entry's fresh accumulation into an exponential moving
// average (weighted by sample counts), clears the accumulators, and evicts entries that
// have not been touched for a while.

// Must match `SharcEntry` in rt_types.slang (56 bytes, ENTRY_BYTES in src/sharc.rs).
struct SharcEntry {
    uint tag;      // 0 = empty; hash tag of the voxel key otherwise.
    uint frame;    // Last frame the entry was deposited into or queried.
    // Radiance accumulated this frame, fixed point at SHARC_UNIT (1/65536 nit): 64-bit so a
    // moonlit voxel keeps its precision and a sunlit one cannot overflow.
    uint64_t acc_r;
    uint64_t acc_g;
    uint64_t acc_b;
    uint acc_n;    // Samples accumulated this frame.
    float conf;    // Samples folded into the resolved value.
    float res_r;   // Resolved outgoing radiance (nits).
    float res_g;
    float res_b;
    uint pad;
};

// Fixed-point unit of the accumulators, in nits.
static const float SHARC_UNIT = 1.0 / 65536.0;

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
    // Untouched for 3000 frames (15 s at 200 fps): evict. Shorter, at high frame rates whole
    // regions dropped out and relearned in waves the image pulsed with.
    if (p.frame - e.frame > 3000) {
        p.entries[i] = SharcEntry();
        return;
    }
    if (e.acc_n == 0) {
        return;
    }
    // Signed: a path's sample goes in as two parts (raygen.slang sharcOpen / sharcClose), and
    // one whose total was not finite left only its negative part.
    let sum = float3(float(int64_t(e.acc_r)), float(int64_t(e.acc_g)), float(int64_t(e.acc_b)));
    let fresh = max(sum, float3(0.0)) * (SHARC_UNIT / float(e.acc_n));
    let old = float3(e.res_r, e.res_g, e.res_b);
    // Anything not finite starts the voxel over rather than glowing until it washes out.
    let bits = asuint(float4(fresh, old.x + old.y + old.z));
    if (any((bits & 0x7f800000u) == 0x7f800000u)) {
        p.entries[i] = SharcEntry();
        return;
    }
    let conf = e.conf;
    // Count-weighted blend, floored so the cache keeps tracking lighting changes (2 %: a
    // voxel's per-frame noise stays under Ray Reconstruction's notice, a change lands in a
    // fraction of a second).
    let conf_new = min(conf + float(e.acc_n), 1024.0);
    let w = max(float(e.acc_n) / conf_new, 0.02);
    let resolved = lerp(old, fresh, conf <= 0.0 ? 1.0 : w);
    p.entries[i].res_r = resolved.x;
    p.entries[i].res_g = resolved.y;
    p.entries[i].res_b = resolved.z;
    p.entries[i].conf = conf_new;
    p.entries[i].acc_r = 0;
    p.entries[i].acc_g = 0;
    p.entries[i].acc_b = 0;
    p.entries[i].acc_n = 0;
}
