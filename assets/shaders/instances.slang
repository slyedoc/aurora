// TLAS instance table kernels (see src/tlas_builder.rs).
//
// The instance buffer is the device-resident VkAccelerationStructureInstanceKHR array the TLAS
// is built from directly. The CPU scatters only the static half of a changed slot (BLAS
// address, SBT offset, material index, visibility mask, transform node); `gather` then writes
// every slot's transform from the GPU-propagated world table each frame.

struct World {
    float4 r0;
    float4 r1;
    float4 r2;
};

// VkAccelerationStructureInstanceKHR: 3x4 row-major transform, custom index:24 | mask:8,
// SBT record offset:24 | flags:8, BLAS device address. 64 bytes.
struct Instance {
    float4 t0;
    float4 t1;
    float4 t2;
    uint custom_and_mask;
    uint sbt_and_flags;
    uint64_t blas;
};

// The static half of one slot. 32 bytes.
struct InstanceRecord {
    uint slot;
    uint custom_and_mask;
    uint sbt_and_flags;
    uint node;
    uint64_t blas;
    // The slot changed owner (a new transform node): it has no last frame.
    uint fresh;
    uint pad;
};

struct ScatterInstancesParams {
    InstanceRecord* records;
    Instance* instances;
    Instance* prev_instances;
    uint* instance_node;
    uint count;
    uint base;
};
[[vk::push_constant]] ScatterInstancesParams scatter_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter_instances(uint3 gid: SV_DispatchThreadID) {
    let p = scatter_params;
    let k = p.base + gid.x;
    if (k >= p.count) {
        return;
    }
    let r = p.records[k];
    p.instances[r.slot].custom_and_mask = r.custom_and_mask;
    p.instances[r.slot].sbt_and_flags = r.sbt_and_flags;
    p.instances[r.slot].blas = r.blas;
    p.instance_node[r.slot] = r.node;
    if (r.fresh != 0u) {
        p.prev_instances[r.slot].t0 = float4(0.0);
        p.prev_instances[r.slot].t1 = float4(0.0);
        p.prev_instances[r.slot].t2 = float4(0.0);
    }
}

struct GatherInstancesParams {
    uint* instance_node;
    World* world;
    Instance* instances;
    uint count;
    uint base;
    uint node_count;
    uint pad;
};
[[vk::push_constant]] GatherInstancesParams gather_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void gather_instances(uint3 gid: SV_DispatchThreadID) {
    let p = gather_params;
    let i = p.base + gid.x;
    if (i >= p.count) {
        return;
    }
    let node = p.instance_node[i];
    if (node >= p.node_count) {
        return;
    }
    let w = p.world[node];
    p.instances[i].t0 = w.r0;
    p.instances[i].t1 = w.r1;
    p.instances[i].t2 = w.r2;
}

// The TLAS is built from the live slots only (src/tlas_builder.rs), packed in slot order: a
// scatter window is mostly empty cells (zero transform), and the builder costs per instance
// handed to it, live or not. Three passes over 64-slot blocks: count each block's live slots,
// scan the counts (one group), write each live row at its block's offset plus its rank in the
// block. `slots[i]` is packed row i's slot (instanceSlot() in surface.slang); rows from the
// live count up to the build's size go inactive (no BLAS).
struct LiveParams {
    Instance* instances;
    Instance* packed;
    uint* slots;
    // Per block: its live count, then (after the scan) its first packed row.
    uint* blocks;
    // The live count (word 0), read back by the CPU to size the next builds.
    uint* total;
    uint count;
    uint base;
    // Packed rows the build reads this frame.
    uint build_count;
    uint block_count;
};
[[vk::push_constant]] LiveParams live_params;

bool slotLive(Instance r) {
    let zero = all(r.t0.xyz == 0.0) && all(r.t1.xyz == 0.0) && all(r.t2.xyz == 0.0);
    return r.blas != 0 && (r.custom_and_mask >> 24) != 0u && !zero;
}

groupshared uint live_wave[2];

// The thread's rank among its block's live slots, and the block's live count.
uint blockRank(bool live, uint lane_in_group, out uint block_live) {
    let wave = lane_in_group / WaveGetLaneCount();
    let rank = WavePrefixCountBits(live);
    let in_wave = WaveActiveCountBits(live);
    if (WaveIsFirstLane()) {
        live_wave[wave] = in_wave;
    }
    GroupMemoryBarrierWithGroupSync();
    var before = 0u;
    block_live = 0u;
    for (uint w = 0u; w < 64u / WaveGetLaneCount(); w++) {
        if (w < wave) {
            before += live_wave[w];
        }
        block_live += live_wave[w];
    }
    return before + rank;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void live_count(uint3 gid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    let p = live_params;
    let slot = p.base + gid.x;
    let live = slot < p.count && slotLive(p.instances[slot]);
    uint block_live;
    blockRank(live, lid.x, block_live);
    if (lid.x == 0u) {
        p.blocks[slot / 64u] = block_live;
    }
}

groupshared uint scan_sums[64];

// One group: each thread scans a run of blocks, then the runs are offset by a scan of their
// sums. Exclusive offsets replace the counts; the total goes to `total`.
[shader("compute")]
[numthreads(64, 1, 1)]
void live_scan(uint3 lid: SV_GroupThreadID) {
    let p = live_params;
    let run = (p.block_count + 63u) / 64u;
    let first = lid.x * run;
    var sum = 0u;
    for (uint b = first; b < min(first + run, p.block_count); b++) {
        sum += p.blocks[b];
    }
    scan_sums[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    var offset = 0u;
    for (uint t = 0u; t < lid.x; t++) {
        offset += scan_sums[t];
    }
    for (uint b = first; b < min(first + run, p.block_count); b++) {
        let n = p.blocks[b];
        p.blocks[b] = offset;
        offset += n;
    }
    if (lid.x == 63u) {
        p.total[0] = offset;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void live_write(uint3 gid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    let p = live_params;
    let slot = p.base + gid.x;
    var r = Instance();
    var live = false;
    if (slot < p.count) {
        r = p.instances[slot];
        live = slotLive(r);
    }
    uint block_live;
    let rank = blockRank(live, lid.x, block_live);
    if (live) {
        let at = p.blocks[slot / 64u] + rank;
        if (at < p.build_count) {
            p.packed[at] = r;
            p.slots[at] = slot;
        }
    }
    // The build's rows past the live ones: inactive.
    let total = p.total[0];
    if (slot >= total && slot < p.build_count) {
        p.packed[slot].blas = 0;
    }
}

// A refit may not turn an instance active or inactive, nor move one to another row: it keeps
// the source build's packing. `live_total` counts the live slots now (total[1]), `live_refresh`
// re-reads each packed row from its slot and counts rows whose slot is no longer live
// (total[2]); the CPU rebuilds once either says the live set moved.
[shader("compute")]
[numthreads(64, 1, 1)]
void live_total(uint3 lid: SV_GroupThreadID) {
    let p = live_params;
    let run = (p.block_count + 63u) / 64u;
    let first = lid.x * run;
    var sum = 0u;
    for (uint b = first; b < min(first + run, p.block_count); b++) {
        sum += p.blocks[b];
    }
    scan_sums[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0u) {
        var total = 0u;
        for (uint t = 0u; t < 64u; t++) {
            total += scan_sums[t];
        }
        p.total[1] = total;
        p.total[2] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void live_refresh(uint3 gid: SV_DispatchThreadID) {
    let p = live_params;
    let row = p.base + gid.x;
    if (row >= p.build_count || row >= p.total[0]) {
        return;
    }
    let r = p.instances[p.slots[row]];
    if (!slotLive(r)) {
        InterlockedAdd(p.total[2], 1u);
    }
    // A slot whose BLAS went away is scattered, and a scatter rebuilds: never seen here.
    if (r.blas != 0) {
        p.packed[row] = r;
    } else {
        p.packed[row].t0 = float4(0.0);
        p.packed[row].t1 = float4(0.0);
        p.packed[row].t2 = float4(0.0);
    }
}
