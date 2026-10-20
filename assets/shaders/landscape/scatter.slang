// GPU scatter: a species' window of cells around the viewer (aurora_landscape scatter.rs), each
// cell's spot a hash of its GLOBAL cell. The spot is kept by the material weights there (times
// the species' factor per palette id) and the slope, and dropped onto the heights. Only kept
// spots take rows of the species' TLAS instance block, in cell order (three passes: count per
// group, scan, write), so the block holds the plants that grow rather than every cell, and its
// rows keep their order while the kept set holds (wind frames refit the TLAS). Last frame's row
// is written alongside, from the previous gust phase: a plant's pose is a function of its cell
// and the phase, so it has motion vectors whatever row it had last frame. With poses, each
// instance leans with a gust wave travelling across the field: the baked pose under its lean (a
// BLAS pointer and hit record swap, no geometry rebuilt) plus a shear the rest of the way, so
// the sway is continuous at any frame rate. With an interaction field, each instance also
// shears away from whatever walked through its spot (interaction.slang).
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
    // Last frame's gust phase, and whether there was a last frame (else last frame = this one).
    float prev_wave;
    uint prev_valid;
    uint pad;
    Instance* prev_instances;
    // ScatterSpecies::clump, clump_size, edge_scale.
    float clump;
    float clump_size;
    float edge_scale;
    // Rows the block holds.
    uint capacity;
    // Per 64-cell group: its keep mask (two words).
    uint* keep;
    // Per group: its kept count, then (after the scan) its first row; the total after them.
    uint* groups;
    // The total kept, for the CPU to size the block (host-visible).
    uint* total_out;
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

// A cell's spot: where it is and whether a plant grows there. `h` comes back advanced past the
// keep rolls, so `place` continues the same sequence wherever it runs.
struct Spot {
    int2 cell;
    float2 xz;
    float cover;
    float bunch;
    float ground;
    bool keep;
};

Spot spot(uint i, inout uint h) {
    Spot sp;
    sp.keep = false;
    let w = int(push.width);
    let cell0 = int2(push.cell0_x, push.cell0_z);
    sp.cell = cell0 + int2(int(i % push.width), int(i / push.width));
    h = pcg(uint(sp.cell.x) * 73856093u ^ uint(sp.cell.y) * 19349663u ^ (push.seed_poses & 0xFFFFFFu));
    let jitter = float2(unit(h), unit(h));
    sp.xz = (float2(sp.cell) + jitter) * push.spacing;

    // The window is centred on the viewer.
    let centre = (float2(cell0) + float(w) * 0.5) * push.spacing;
    let d = length(sp.xz - centre);
    let outer = float(w) * 0.5 * push.spacing;
    // One roll for both edges: a tier whose inner edge sits `fade` inside the next nearer
    // tier's outer edge takes over every spot that one lets go, and no other.
    // No inner edge at radius 0: the nearest tier keeps every spot around the viewer.
    let dither = unit(h);
    let inside = push.inner_radius > 0.0 && d < push.inner_radius + push.fade * (1.0 - dither);
    if (inside || d > outer - push.fade * dither) {
        return sp;
    }

    let p = push.pool;
    let lo = float2(int2(p.min_x, p.min_z) * PAGE) * p.texel;
    let hi = float2((int2(p.min_x, p.min_z) + int2(p.pages_x, p.pages_z)) * PAGE) * p.texel;
    if (any(sp.xz < lo) || any(sp.xz >= hi)) {
        return sp;
    }

    // Density from the materials, bilinear between texels so edges fade instead of stepping;
    // the clump noise bunches it into tufts and bald spots.
    let tx = sp.xz / p.texel - 0.5;
    let t0 = int2(floor(tx));
    let tf = tx - floor(tx);
    sp.cover = lerp(lerp(texel_density(t0), texel_density(t0 + int2(1, 0)), tf.x),
                    lerp(texel_density(t0 + int2(0, 1)), texel_density(t0 + int2(1, 1)), tf.x), tf.y);
    let tables = push.tables;
    sp.bunch = clump(sp.xz, tables.clump_size, push.seed_poses & 0xFFFFFFu);
    let density = sp.cover * lerp(1.0, saturate(1.5 * sp.bunch), tables.clump);
    if (unit(h) >= density) {
        return sp;
    }
    sp.ground = sample_height(p, sp.xz);
    let dx = sample_height(p, sp.xz + float2(p.texel, 0.0)) - sample_height(p, sp.xz - float2(p.texel, 0.0));
    let dz = sample_height(p, sp.xz + float2(0.0, p.texel)) - sample_height(p, sp.xz - float2(0.0, p.texel));
    let slope = degrees(atan(length(float2(dx, dz)) / (2.0 * p.texel)));
    sp.keep = slope <= push.max_slope;
    return sp;
}

// The pose's gust position at phase `wave`: 0 .. pose count - 1 (the lean, in pose steps).
float gust(float2 xz, float lag, float wave, uint pose_count) {
    let w = wave - (xz.x * 0.045 + xz.y * 0.02) + lag;
    return (0.5 + 0.5 * sin(w * 6.2831853)) * float(pose_count - 1);
}

// A kept spot's plant into `row`, and its pose at last frame's phase into the same row of the
// previous rows (against this frame's pose geometry, so the motion is the true step).
void place(Spot sp, inout uint h, uint row) {
    let tables = push.tables;
    let yaw = unit(h) * 6.2831853;
    // Thick tufts stand taller; plants thinning into a path or another ground shrink with it.
    let taper = lerp(tables.edge_scale, 1.0, smoothstep(0.1, 0.8, sp.cover));
    let tuft = lerp(1.0, lerp(0.7, 1.1, sp.bunch), tables.clump);
    let scale = lerp(push.scale_lo, push.scale_hi, unit(h)) * tuft * taper;
    let c = cos(yaw) * scale;
    let sn = sin(yaw) * scale;
    let at = float3(push.origin_x + sp.xz.x, push.origin_y + sp.ground - push.sink,
                    push.origin_z + sp.xz.y);
    // Lean away from what walked through: a shear about the base (local y feeds world xz), the
    // tip lowered to keep the plant's length; at most ~56 degrees (tan 1.5).
    var lean = fieldPush(push.field, at.xz);
    let lean_len = length(lean);
    if (lean_len > 1.5) {
        lean *= 1.5 / lean_len;
    }
    let tall = scale / sqrt(1.0 + dot(lean, lean));
    var r = Instance();
    r.t0 = float4(c, lean.x * tall, sn, at.x);
    r.t1 = float4(0.0, tall, 0.0, at.y);
    r.t2 = float4(-sn, lean.y * tall, c, at.z);
    var prev = r;

    let pose_count = push.seed_poses >> 24;
    if (pose_count > 0) {
        // A gust front sweeping +x, slightly skewed, with a little per-plant lag; the lean
        // swings over the poses' range (least to most) and back.
        let lag = unit(h) * 0.15;
        let s = gust(sp.xz, lag, push.wave, pose_count);
        let k = min(uint(s), max(pose_count, 2u) - 2u);
        // The rest of the way as a shear in plant space (x += g y): rotated by the yaw. Last
        // frame's lean is expressed against this pose too (one pose step ~ wind_shear, the
        // approximation the sway itself makes).
        let g = pose_count > 1u ? (s - float(k)) * tables.wind_shear : 0.0;
        let prev_s = tables.prev_valid != 0u ? gust(sp.xz, lag, tables.prev_wave, pose_count) : s;
        let prev_g = pose_count > 1u ? (prev_s - float(k)) * tables.wind_shear : 0.0;
        r.t0.y += c * g;
        r.t2.y -= sn * g;
        prev.t0.y += c * prev_g;
        prev.t2.y -= sn * prev_g;
        let pose = tables.poses[k];
        push.instances[row].blas = pose.blas;
        push.instances[row].sbt_and_flags = pose.sbt_and_flags;
    }
    push.instances[row].t0 = r.t0;
    push.instances[row].t1 = r.t1;
    push.instances[row].t2 = r.t2;
    tables.prev_instances[row].t0 = prev.t0;
    tables.prev_instances[row].t1 = prev.t1;
    tables.prev_instances[row].t2 = prev.t2;
}

// An inactive row: no transform (traces nothing), this frame and last.
void clear_row(uint row) {
    push.instances[row].t0 = float4(0.0);
    push.instances[row].t1 = float4(0.0);
    push.instances[row].t2 = float4(0.0);
    push.tables.prev_instances[row].t0 = float4(0.0);
    push.tables.prev_instances[row].t1 = float4(0.0);
    push.tables.prev_instances[row].t2 = float4(0.0);
}

groupshared uint keep_mask[2];
groupshared uint keep_count;

// Pass 1, a thread per cell: which spots grow, as each 64-cell group's mask and count.
[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_count(uint3 gid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID,
                   uint3 group: SV_GroupID) {
    if (lid.x < 2u) {
        keep_mask[lid.x] = 0u;
    }
    if (lid.x == 0u) {
        keep_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    let i = gid.x;
    var h = 0u;
    if (i < push.width * push.width && spot(i, h).keep) {
        InterlockedOr(keep_mask[lid.x >> 5], 1u << (lid.x & 31u));
        InterlockedAdd(keep_count, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0u) {
        let tables = push.tables;
        tables.keep[group.x * 2u] = keep_mask[0];
        tables.keep[group.x * 2u + 1u] = keep_mask[1];
        tables.groups[group.x] = keep_count;
    }
}

groupshared uint scan_sums[64];

// Pass 2, one group: the groups' counts become their first rows (an exclusive scan), the total
// after them and out to the CPU.
[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_scan(uint3 lid: SV_GroupThreadID) {
    let tables = push.tables;
    let group_count = (push.width * push.width + 63u) / 64u;
    let run = (group_count + 63u) / 64u;
    let first = lid.x * run;
    var sum = 0u;
    for (uint b = first; b < min(first + run, group_count); b++) {
        sum += tables.groups[b];
    }
    scan_sums[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    var offset = 0u;
    for (uint t = 0u; t < lid.x; t++) {
        offset += scan_sums[t];
    }
    for (uint b = first; b < min(first + run, group_count); b++) {
        let n = tables.groups[b];
        tables.groups[b] = offset;
        offset += n;
    }
    if (lid.x == 63u) {
        tables.groups[group_count] = offset;
        tables.total_out[0] = offset;
    }
}

// Pass 3, a thread per cell: a kept spot's plant into its row (its group's first row plus the
// kept cells before it in the group), while the rows past the total go inactive. A block too
// small for them all keeps the first `capacity` (the CPU grows it from the total).
[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_write(uint3 gid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID,
                   uint3 group: SV_GroupID) {
    let i = gid.x;
    let tables = push.tables;
    let group_count = (push.width * push.width + 63u) / 64u;
    let total = tables.groups[group_count];
    if (i < tables.capacity && i >= total) {
        clear_row(push.first + i);
    }
    if (i >= push.width * push.width) {
        return;
    }
    let lo = tables.keep[group.x * 2u];
    let hi = tables.keep[group.x * 2u + 1u];
    let lane = lid.x;
    let mask = lane < 32u ? lo : hi;
    if (((mask >> (lane & 31u)) & 1u) == 0u) {
        return;
    }
    let below = (lane & 31u) == 0u ? 0u : countbits(mask & ((1u << (lane & 31u)) - 1u));
    let rank = (lane < 32u ? 0u : countbits(lo)) + below;
    let row = tables.groups[group.x] + rank;
    if (row >= tables.capacity) {
        return;
    }
    var h = 0u;
    let sp = spot(i, h);
    place(sp, h, push.first + row);
}
