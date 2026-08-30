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
    uint64_t pad;
};

struct ScatterInstancesParams {
    InstanceRecord* records;
    Instance* instances;
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
