// GPU transform hierarchy — the node table's kernels (see src/gpu_transform.rs).
//
// Every entity with a `Transform` is a node: a local TRS (`NodeLocal`), a parent slot, and an
// intrusive child list (`first_child` / `next_sibling`). The CPU mirrors only what changed
// (record lists, scattered here); the frontier expands this frame's changed nodes to their
// descendants GPU-side; `propagate` then recomputes each frontier node's world affine by
// walking up its ancestors — a pure function of the node's own chain, so one thread per node
// in a single pass, writing into the persistent `World` table.
//
// Every buffer arrives as a raw pointer in push constants (no descriptor sets). All kernels are
// 64 wide; direct dispatches carry a `base` so the CPU can chunk past the 65535-group limit.
//
// Frontier layout (u32 words, must match gpu_transform.rs):
//   [0] total (atomic append cursor)   [1] total published by finalize
//   [2] current level                  [3] pad
//   [4 .. 4+MAX_LEVELS+2] level_begin  … pad to HEADER = 24
//   [HEADER ..] node slots
// Indirect args: (x,y,z) per expand level, then the consumer (propagate) args at MAX_LEVELS*3.

static const uint NO_NODE = 0xffffffff;
static const uint ROOT_PARENT = 0xffffffff;
static const uint MAX_LEVELS = 16;
static const uint HEADER = 24;
static const uint LEVEL_BEGIN = 4;
static const uint CONSUMER_ARGS = MAX_LEVELS * 3;
static const uint WORKGROUP = 64;
static const uint MAX_GROUPS = 65535;
// Safety bounds on malformed sibling chains / cyclic parent chains.
static const uint MAX_CHAIN = 4194304;
static const uint MAX_DEPTH = 64;

// A node's local transform: quaternion xyzw, translation xyz (w unused), scale xyz (w unused).
struct NodeLocal {
    float4 rotation;
    float4 translation;
    float4 scale;
};

// One changed local: the slot it lands in, then the value. 64 bytes.
struct LocalRecord {
    uint slot;
    uint pad0;
    uint pad1;
    uint pad2;
    NodeLocal local;
};

// One changed u32 column entry (parent / first_child / next_sibling).
struct U32Record {
    uint slot;
    uint value;
};

// A node's world affine as the three rows of a 3x4 matrix (row-major, translation in .w) —
// exactly VkTransformMatrixKHR, so the instance gather is a straight copy.
struct World {
    float4 r0;
    float4 r1;
    float4 r2;
};

// ---- scatter ------------------------------------------------------------------------------

struct ScatterLocalsParams {
    LocalRecord* records;
    NodeLocal* locals;
    uint count;
    uint base;
};
[[vk::push_constant]] ScatterLocalsParams scatter_locals_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_locals(uint3 gid: SV_DispatchThreadID) {
    let k = scatter_locals_params.base + gid.x;
    if (k >= scatter_locals_params.count) {
        return;
    }
    let record = scatter_locals_params.records[k];
    scatter_locals_params.locals[record.slot] = record.local;
}

struct ScatterU32Params {
    U32Record* records;
    uint* dst;
    uint count;
    uint base;
};
[[vk::push_constant]] ScatterU32Params scatter_u32_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_u32(uint3 gid: SV_DispatchThreadID) {
    let k = scatter_u32_params.base + gid.x;
    if (k >= scatter_u32_params.count) {
        return;
    }
    let record = scatter_u32_params.records[k];
    scatter_u32_params.dst[record.slot] = record.value;
}

// ---- frontier -----------------------------------------------------------------------------

// Stamp the node's epoch; if it wasn't already stamped this frame, append it.
void try_append(uint* frontier, uint* epoch, uint frame_id, uint node_count, uint slot) {
    if (slot >= node_count) {
        return;
    }
    uint previous;
    InterlockedExchange(epoch[slot], frame_id, previous);
    if (previous == frame_id) {
        return;
    }
    uint pos;
    InterlockedAdd(frontier[0], 1, pos);
    frontier[HEADER + pos] = slot;
}

struct FrontierSeedParams {
    LocalRecord* records;
    uint* frontier;
    uint* epoch;
    uint count;
    uint base;
    uint frame_id;
    uint node_count;
};
[[vk::push_constant]] FrontierSeedParams frontier_seed_params;

// Seeds: every node whose local changed this frame (the scatter's own record list).
[shader("compute")]
[numthreads(64, 1, 1)]
void frontier_seed(uint3 gid: SV_DispatchThreadID) {
    let p = frontier_seed_params;
    let k = p.base + gid.x;
    if (k >= p.count) {
        return;
    }
    try_append(p.frontier, p.epoch, p.frame_id, p.node_count, p.records[k].slot);
}

void write_args(uint* indirect, uint base, uint threads) {
    let groups = (threads + WORKGROUP - 1) / WORKGROUP;
    indirect[base] = min(max(groups, 1), MAX_GROUPS);
    indirect[base + 1] = 1;
    indirect[base + 2] = 1;
}

struct FrontierFinalizeParams {
    uint* frontier;
    uint* indirect;
};
[[vk::push_constant]] FrontierFinalizeParams frontier_finalize_params;

// One thread: close the current level (its end = the append cursor), write the next expand's
// indirect args from the level's size, publish the running total for the consumer.
[shader("compute")]
[numthreads(1, 1, 1)]
void frontier_finalize() {
    let p = frontier_finalize_params;
    let lvl = p.frontier[2];
    let total = p.frontier[0];
    p.frontier[LEVEL_BEGIN + lvl + 1] = total;
    p.frontier[2] = lvl + 1;
    if (lvl < MAX_LEVELS) {
        let begin = p.frontier[LEVEL_BEGIN + lvl];
        write_args(p.indirect, lvl * 3, total - begin);
    }
    p.frontier[1] = total;
    write_args(p.indirect, CONSUMER_ARGS, total);
}

struct FrontierExpandParams {
    uint* first_child;
    uint* next_sibling;
    uint* frontier;
    uint* epoch;
    uint frame_id;
    uint node_count;
};
[[vk::push_constant]] FrontierExpandParams frontier_expand_params;

// Expand the just-closed level: each of its nodes appends its (unstamped) children.
[shader("compute")]
[numthreads(64, 1, 1)]
void frontier_expand(uint3 gid: SV_DispatchThreadID) {
    let p = frontier_expand_params;
    let lvl = p.frontier[2] - 1;
    let begin = p.frontier[LEVEL_BEGIN + lvl];
    let end = p.frontier[LEVEL_BEGIN + lvl + 1];
    let k = begin + gid.x;
    if (k >= end) {
        return;
    }
    let node = p.frontier[HEADER + k];
    var child = p.first_child[node];
    uint guard = 0;
    while (child != NO_NODE && guard < MAX_CHAIN) {
        try_append(p.frontier, p.epoch, p.frame_id, p.node_count, child);
        child = p.next_sibling[child];
        guard = guard + 1;
    }
}

// ---- propagate ----------------------------------------------------------------------------

struct Affine {
    float3 r0;
    float3 r1;
    float3 r2;
    float3 t;
};

// TRS → affine: rotation rows with columns scaled by `scale`, plus translation.
Affine load_local(NodeLocal* locals, uint node) {
    let l = locals[node];
    let q = l.rotation;
    let s = l.scale.xyz;
    let xx = q.x * q.x; let yy = q.y * q.y; let zz = q.z * q.z;
    let xy = q.x * q.y; let xz = q.x * q.z; let yz = q.y * q.z;
    let wx = q.w * q.x; let wy = q.w * q.y; let wz = q.w * q.z;
    Affine a;
    a.r0 = float3(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)) * s;
    a.r1 = float3(2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)) * s;
    a.r2 = float3(2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)) * s;
    a.t = l.translation.xyz;
    return a;
}

struct PropagateParams {
    NodeLocal* locals;
    uint* parent;
    World* world;
    uint* frontier;
    // Threads on the full-rebuild path (node = base + thread); the frontier path takes its
    // count from the frontier header.
    uint count;
    uint base;
    uint full_rebuild;
    uint node_count;
};
[[vk::push_constant]] PropagateParams propagate_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void propagate(uint3 gid: SV_DispatchThreadID) {
    let p = propagate_params;
    uint node;
    if (p.full_rebuild != 0) {
        node = p.base + gid.x;
        if (node >= p.count) {
            return;
        }
    } else {
        let k = gid.x;
        if (k >= p.frontier[1]) {
            return;
        }
        node = p.frontier[HEADER + k];
    }
    if (node >= p.node_count) {
        return;
    }

    // M = local[node]; then left-compose each ancestor up to the root:
    // M = local[root] ∘ … ∘ local[parent] ∘ M.
    var m = load_local(p.locals, node);
    var parent = p.parent[node];
    for (uint step = 0; step < MAX_DEPTH; step++) {
        if (parent == ROOT_PARENT || parent >= p.node_count) {
            break;
        }
        let a = load_local(p.locals, parent);
        let c0 = float3(m.r0.x, m.r1.x, m.r2.x);
        let c1 = float3(m.r0.y, m.r1.y, m.r2.y);
        let c2 = float3(m.r0.z, m.r1.z, m.r2.z);
        Affine n;
        n.r0 = float3(dot(a.r0, c0), dot(a.r0, c1), dot(a.r0, c2));
        n.r1 = float3(dot(a.r1, c0), dot(a.r1, c1), dot(a.r1, c2));
        n.r2 = float3(dot(a.r2, c0), dot(a.r2, c1), dot(a.r2, c2));
        n.t = float3(dot(a.r0, m.t), dot(a.r1, m.t), dot(a.r2, m.t)) + a.t;
        m = n;
        parent = p.parent[parent];
    }

    World w;
    w.r0 = float4(m.r0, m.t.x);
    w.r1 = float4(m.r1, m.t.y);
    w.r2 = float4(m.r2, m.t.z);
    p.world[node] = w;
}
