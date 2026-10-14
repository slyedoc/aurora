// GPU scatter: one thread per cell of a species' window around the viewer (aurora_landscape
// scatter.rs). Each cell's spot is a hash of its GLOBAL cell, and its slot is fixed by it, so a
// plant stays put (and keeps its motion vectors) as the window moves. The spot is kept by the material weights there (times the species' factor per
// palette id) and the slope, dropped onto the heights, and written straight into the
// species' TLAS instance block; a dropped spot gets a zero transform (traces nothing). With
// poses, each instance leans with a gust wave travelling across the field: the baked pose under
// its lean (a BLAS pointer and hit record swap, no geometry rebuilt) plus a shear the rest of
// the way, so the sway is continuous at any frame rate and has motion vectors. With an interaction field, each
// instance also shears away from whatever walked through its spot (interaction.slang).
import pages;
import materials;
import interaction;

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

// `SpeciesTables` in scatter.rs: the keep factor per palette id, and the wind poses.
struct SpeciesTables {
    float factors[256];
    Pose poses[64];
    // The shear (x per y, plant space) standing in for one pose step.
    float wind_shear;
    // The window the block last held (prev_valid): cells outside it just entered.
    int prev_cell0_x;
    int prev_cell0_z;
    uint prev_valid;
    Instance* prev_instances;
    // ScatterSpecies::clump, clump_size, edge_scale.
    float clump;
    float clump_size;
    float edge_scale;
    float pad;
};

// `ScatterPush` in scatter.rs (the whole 128-byte block).
struct ScatterPush {
    Instance* instances;
    uint2* materials;
    SpeciesTables* tables;
    // The interaction field, or null while nothing disturbs it.
    FieldTexel* field;
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

// The species' keep factor at one texel: its palette factors under the texel's weights.
float texel_density(int2 t) {
    let p = push.pool;
    if (mat_is_hole(push.materials[texel_index(p.min_x, p.min_z, p.pages_x, p.pages_z, t.x, t.y)])) {
        return 0.0;
    }
    let s = texel_set(t);
    var density = 0.0;
    for (int k = 0; k < 4; k++) {
        density += s.w[k] * push.tables.factors[s.ids[k]];
    }
    return density;
}

// Value noise in [0, 1] on a unit lattice.
float value_noise(float2 x, uint seed) {
    let i = floor(x);
    let f = x - i;
    let u = f * f * (3.0 - 2.0 * f);
    let ix = int(i.x);
    let iz = int(i.y);
    float c[4];
    for (int k = 0; k < 4; k++) {
        let h = pcg(uint(ix + (k & 1)) * 73856093u ^ uint(iz + (k >> 1)) * 19349663u ^ seed);
        c[k] = float(h) / 4294967296.0;
    }
    return lerp(lerp(c[0], c[1], u.x), lerp(c[2], c[3], u.x), u.y);
}

// Where plants bunch up: 0 bald .. 1 thick. Mostly shared by every species (grass grows where
// grass grows), a little per species so flowers patch on their own.
float clump(float2 xz, float size, uint seed) {
    let shared = 0.65 * value_noise(xz / size, 0x9E3779B9u)
        + 0.35 * value_noise(xz / (size * 0.4), 0x85EBCA6Bu);
    let own = value_noise(xz / (size * 0.6), seed * 2654435761u);
    return smoothstep(0.2, 0.75, lerp(shared, own, 0.3));
}

// a mod w in [0, w) for negative a too (int % runs unsigned on some GPUs).
int2 wrapCell(int2 a, int w) {
    return a - w * int2(floor(float2(a) / float(w)));
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
    // Slots are fixed per global cell (cell mod width), so a plant keeps its slot -- and its
    // last-frame row for motion vectors -- while it stays in the window.
    let slot = push.first + i;
    let w = int(push.width);
    let cell0 = int2(push.cell0_x, push.cell0_z);
    let cell = cell0 + wrapCell(int2(int(i % push.width), int(i / push.width)) - cell0, w);
    let held = cell - int2(push.tables.prev_cell0_x, push.tables.prev_cell0_z);
    if (push.tables.prev_valid == 0u || any(held < 0) || any(held >= w)) {
        push.tables.prev_instances[slot].t0 = float4(0.0);
        push.tables.prev_instances[slot].t1 = float4(0.0);
        push.tables.prev_instances[slot].t2 = float4(0.0);
    }
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

    // Density from the materials, bilinear between texels so edges fade instead of stepping;
    // the clump noise bunches it into tufts and bald spots.
    let tx = xz / p.texel - 0.5;
    let t0 = int2(floor(tx));
    let tf = tx - floor(tx);
    let cover = lerp(lerp(texel_density(t0), texel_density(t0 + int2(1, 0)), tf.x),
                     lerp(texel_density(t0 + int2(0, 1)), texel_density(t0 + int2(1, 1)), tf.x), tf.y);
    let tables = push.tables;
    let bunch = clump(xz, tables.clump_size, push.seed_poses & 0xFFFFFFu);
    let density = cover * lerp(1.0, saturate(1.5 * bunch), tables.clump);
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
    // Thick tufts stand taller; plants thinning into a path or another ground shrink with it.
    let taper = lerp(tables.edge_scale, 1.0, smoothstep(0.1, 0.8, cover));
    let tuft = lerp(1.0, lerp(0.7, 1.1, bunch), tables.clump);
    let scale = lerp(push.scale_lo, push.scale_hi, unit(h)) * tuft * taper;
    let c = cos(yaw) * scale;
    let sn = sin(yaw) * scale;
    let at = float3(push.origin_x + xz.x, push.origin_y + ground - push.sink, push.origin_z + xz.y);
    // Lean away from what walked through: a shear about the base (local y feeds world xz), the
    // tip lowered to keep the plant's length; at most ~56 degrees (tan 1.5).
    var lean = fieldPush(push.field, at.xz);
    let lean_len = length(lean);
    if (lean_len > 1.5) {
        lean *= 1.5 / lean_len;
    }
    let tall = scale / sqrt(1.0 + dot(lean, lean));
    push.instances[slot].t0 = float4(c, lean.x * tall, sn, at.x);
    push.instances[slot].t1 = float4(0.0, tall, 0.0, at.y);
    push.instances[slot].t2 = float4(-sn, lean.y * tall, c, at.z);

    if (pose_count > 0) {
        // A gust front sweeping +x, slightly skewed, with a little per-plant lag; the lean
        // swings over the poses' range (least to most) and back.
        let wave = push.wave - (xz.x * 0.045 + xz.y * 0.02) + unit(h) * 0.15;
        let s = (0.5 + 0.5 * sin(wave * 6.2831853)) * float(pose_count - 1);
        let k = min(uint(s), max(pose_count, 2u) - 2u);
        let rest = pose_count > 1u ? s - float(k) : 0.0;
        // The rest of the way as a shear in plant space (x += g y): rotated by the yaw.
        let g = rest * push.tables.wind_shear;
        push.instances[slot].t0.y += c * g;
        push.instances[slot].t2.y -= sn * g;
        let pose = push.tables.poses[k];
        push.instances[slot].blas = pose.blas;
        push.instances[slot].sbt_and_flags = pose.sbt_and_flags;
    }
}
