// GPU scatter: one thread per cell of a species' window around the viewer (aurora_landscape
// scatter.rs). Each cell's spot is a hash of its GLOBAL cell, so a plant stays put as the
// window moves. The spot is kept by the material weights there (times the species' factor per
// palette id) and the slope, dropped onto the heights, and written straight into the
// species' TLAS instance block; a dropped spot gets a zero transform (traces nothing). With
// poses, each instance also picks its wind pose from a gust wave travelling across the field:
// a BLAS pointer and hit record swap, no geometry rebuilt.
import pages;
import materials;

// VkAccelerationStructureInstanceKHR (instances.slang).
struct Instance {
    float4 t0;
    float4 t1;
    float4 t2;
    uint custom_and_mask;
    uint sbt_and_flags;
    uint64_t blas;
};

// `BlockPose` in aurora's tlas_builder.rs.
struct Pose {
    uint64_t blas;
    uint sbt_and_flags;
    uint pad;
};

// `ScatterPush` in scatter.rs (the whole 128-byte block).
struct ScatterPush {
    Instance* instances;
    uint2* materials;
    Pose* poses;
    float* factors;
    PagePool pool;
    uint first;
    uint width;
    int cell0_x;
    int cell0_z;
    float spacing;
    float scale_lo;
    float scale_hi;
    float max_slope;
    float sink;
    float origin_x;
    float origin_y;
    float origin_z;
    // Seed (24 bits) | pose count (8 bits).
    uint seed_poses;
    // Gust phase: time x wind speed.
    float wave;
    // Ring the species grows in around the viewer, each edge dithered over `fade` metres.
    float inner_radius;
    float fade;
};
[[vk::push_constant]] ScatterPush push;

uint pcg(uint v) {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float unit(inout uint h) {
    h = pcg(h);
    return float(h) / 4294967296.0;
}

MatSet texel_set(int2 t) {
    let p = push.pool;
    return mat_unpack(push.materials[texel_index(p.min_x, p.min_z, p.pages_x, p.pages_z, t.x, t.y)]);
}

void drop_spot(uint slot) {
    push.instances[slot].t0 = float4(0.0);
    push.instances[slot].t1 = float4(0.0);
    push.instances[slot].t2 = float4(0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= push.width * push.width) {
        return;
    }
    let slot = push.first + i;
    let cell = int2(push.cell0_x + int(i % push.width), push.cell0_z + int(i / push.width));
    let pose_count = push.seed_poses >> 24;
    var h = pcg(uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ (push.seed_poses & 0xFFFFFFu));
    let jitter = float2(unit(h), unit(h));
    let xz = (float2(cell) + jitter) * push.spacing;

    // The window is centred on the viewer.
    let centre = (float2(push.cell0_x, push.cell0_z) + float(push.width) * 0.5) * push.spacing;
    let d = length(xz - centre);
    let outer = float(push.width) * 0.5 * push.spacing;
    // One roll for both edges: a tier whose inner edge sits `fade` inside the next nearer
    // tier's outer edge takes over every spot that one lets go, and no other.
    // No inner edge at radius 0: the nearest tier keeps every spot around the viewer.
    let dither = unit(h);
    let inside = push.inner_radius > 0.0 && d < push.inner_radius + push.fade * (1.0 - dither);
    if (inside || d > outer - push.fade * dither) {
        drop_spot(slot);
        return;
    }

    let p = push.pool;
    let lo = float2(int2(p.min_x, p.min_z) * PAGE) * p.texel;
    let hi = float2((int2(p.min_x, p.min_z) + int2(p.pages_x, p.pages_z)) * PAGE) * p.texel;
    if (any(xz < lo) || any(xz >= hi)) {
        drop_spot(slot);
        return;
    }

    // Density from the nearest texel's materials.
    let s = texel_set(int2(round(xz / p.texel)));
    var density = 0.0;
    for (int k = 0; k < 4; k++) {
        density += s.w[k] * push.factors[s.ids[k]];
    }
    if (unit(h) >= density) {
        drop_spot(slot);
        return;
    }
    let ground = sample_height(p, xz);
    let dx = sample_height(p, xz + float2(p.texel, 0.0)) - sample_height(p, xz - float2(p.texel, 0.0));
    let dz = sample_height(p, xz + float2(0.0, p.texel)) - sample_height(p, xz - float2(0.0, p.texel));
    let slope = degrees(atan(length(float2(dx, dz)) / (2.0 * p.texel)));
    if (slope > push.max_slope) {
        drop_spot(slot);
        return;
    }

    let yaw = unit(h) * 6.2831853;
    let scale = lerp(push.scale_lo, push.scale_hi, unit(h));
    let c = cos(yaw) * scale;
    let sn = sin(yaw) * scale;
    let at = float3(push.origin_x + xz.x, push.origin_y + ground - push.sink, push.origin_z + xz.y);
    push.instances[slot].t0 = float4(c, 0.0, sn, at.x);
    push.instances[slot].t1 = float4(0.0, scale, 0.0, at.y);
    push.instances[slot].t2 = float4(-sn, 0.0, c, at.z);

    if (pose_count > 0) {
        // A gust front sweeping +x, slightly skewed, with a little per-plant lag.
        let wave = push.wave - (xz.x * 0.045 + xz.y * 0.02) + unit(h) * 0.15;
        let k = uint(floor(frac(wave) * float(pose_count))) % pose_count;
        let pose = push.poses[k];
        push.instances[slot].blas = pose.blas;
        push.instances[slot].sbt_and_flags = pose.sbt_and_flags;
    }
}
