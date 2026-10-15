// Emissive-triangle light table kernels (see src/lights.rs).
//
// The light table covers every triangle of every instance whose material is emissive.
// `light_powers` writes each entry's sampling weight (world-space area x emitted luminance,
// transforms read live from the TLAS instance buffer); `light_scan` turns the weights into
// the CDF the raygen binary-searches, and stores the total in the header. Both run inside
// the frame command buffer whenever the light set changed, after the instance gather.

// One emissive instance (must match LightInstGpu in lights.rs).
struct LightInst {
    float* verts;            // Vertex stream: 8 floats each, position at 0..2.
    uint* indices;
    uint* geom_to_index;     // Per geometry: first index in `indices`.
    uint* geom_to_triangle;  // Per geometry: first (global) triangle.
    uint geom_count;
    uint slot;               // TLAS instance slot (world transform).
    uint entry_base;         // First entry in the global light table.
    uint tri_count;
    float* geom_emission;    // Per geometry: emitted radiance (3 floats each, nits).
    uint64_t pad;
};

// VkAccelerationStructureInstanceKHR (as in instances.slang). 64 bytes.
struct Instance {
    float4 t0;
    float4 t1;
    float4 t2;
    uint custom_and_mask;
    uint sbt_and_flags;
    uint64_t blas;
};

// Must match LightsHeaderGpu in lights.rs.
struct LightsHeader {
    uint entry_count;
    uint linst_count;
    float total_power;
    uint slot_map_count;
    uint analytic_count;
    uint pad1;
    uint64_t cdf;
    uint64_t linsts;
    uint64_t slot_to_linst;
    uint64_t analytics;
    uint64_t entry_to_linst;
    uint64_t prepared;
};

// Must match AnalyticLightGpu in lights.rs. 80 bytes; the
// power kernel only reads the CPU-computed `power`.
struct AnalyticLight {
    float3 position;
    uint kind;
    float3 direction;
    float radius;
    float3 tangent;
    float cos_inner;
    float3 emission;
    float cos_outer;
    float2 half_extents;
    float power;
    uint flags;
};

float3 vertex_position(float* verts, uint i) {
    let b = i * 8;
    return float3(verts[b], verts[b + 1], verts[b + 2]);
}

struct LightPowerParams {
    LightInst* linsts;
    Instance* instances;
    float* powers;
    AnalyticLight* analytics;
    uint linst_count;
    uint entry_count;
    uint analytic_count;
    uint base;
};
[[vk::push_constant]] LightPowerParams power_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void light_powers(uint3 gid: SV_DispatchThreadID) {
    let p = power_params;
    let e = p.base + gid.x;
    if (e >= p.entry_count) {
        return;
    }
    // Analytic entries carry their weight from the CPU.
    if (e < p.analytic_count) {
        p.powers[e] = p.analytics[e].power;
        return;
    }
    // The light instance owning this entry: greatest entry_base <= e.
    uint lo = 0;
    uint hi = p.linst_count - 1;
    while (lo < hi) {
        let mid = (lo + hi + 1) / 2;
        if (p.linsts[mid].entry_base <= e) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    float3 w0, w1, w2, emission;
    uint mask;
    resolveTriangle(p.linsts[lo], p.instances, e, w0, w1, w2, emission, mask);
    let area = 0.5 * length(cross(w1 - w0, w2 - w0));
    p.powers[e] = area * (0.2126 * emission.r + 0.7152 * emission.g + 0.0722 * emission.b);
}

// Triangle entry `e` of light instance `li` in world space through its live instance row, its
// emitted radiance and its world mask.
void resolveTriangle(LightInst li, Instance* instances, uint e, out float3 w0, out float3 w1,
                     out float3 w2, out float3 emission, out uint mask) {
    let tri = e - li.entry_base;
    // The geometry holding the triangle (prefix offsets, few per mesh).
    uint g = 0;
    for (uint k = 1; k < li.geom_count; k++) {
        if (li.geom_to_triangle[k] <= tri) {
            g = k;
        } else {
            break;
        }
    }
    let idx = li.geom_to_index[g] + (tri - li.geom_to_triangle[g]) * 3;
    let p0 = vertex_position(li.verts, li.indices[idx + 0]);
    let p1 = vertex_position(li.verts, li.indices[idx + 1]);
    let p2 = vertex_position(li.verts, li.indices[idx + 2]);
    let inst = instances[li.slot];
    w0 = float3(dot(inst.t0, float4(p0, 1.0)), dot(inst.t1, float4(p0, 1.0)), dot(inst.t2, float4(p0, 1.0)));
    w1 = float3(dot(inst.t0, float4(p1, 1.0)), dot(inst.t1, float4(p1, 1.0)), dot(inst.t2, float4(p1, 1.0)));
    w2 = float3(dot(inst.t0, float4(p2, 1.0)), dot(inst.t1, float4(p2, 1.0)), dot(inst.t2, float4(p2, 1.0)));
    emission = float3(li.geom_emission[g * 3 + 0], li.geom_emission[g * 3 + 1], li.geom_emission[g * 3 + 2]);
    mask = inst.custom_and_mask >> 24;
}

struct LightPrepareParams {
    LightInst* linsts;
    Instance* instances;
    LightsHeader* header;
    uint* entry_to_linst;
    float* cdf;
    float4* prepared;
    uint entry_count;
    uint analytic_count;
    uint base;
    uint pad;
};
[[vk::push_constant]] LightPrepareParams prepare_params;

// Every frame, every triangle entry resolved once into four float4 (light_eval.slang reads
// them): world vertex 0 and world mask, edge 1 and pick probability, edge 2, emission. A
// sample is then one record instead of a chain of instance, geometry, index and vertex reads.
[shader("compute")]
[numthreads(64, 1, 1)]
void light_prepare(uint3 gid: SV_DispatchThreadID) {
    let p = prepare_params;
    let e = p.base + gid.x;
    if (e >= p.entry_count || e < p.analytic_count) {
        return;
    }
    float3 w0, w1, w2, emission;
    uint mask;
    resolveTriangle(p.linsts[p.entry_to_linst[e]], p.instances, e, w0, w1, w2, emission, mask);
    let below = e > 0 ? p.cdf[e - 1] : 0.0;
    let total = p.header[0].total_power;
    let p_pick = total > 0.0 ? (p.cdf[e] - below) / total : 0.0;
    let r = p.prepared + e * 4;
    r[0] = float4(w0, asfloat(mask));
    r[1] = float4(w1 - w0, p_pick);
    r[2] = float4(w2 - w0, 0.0);
    r[3] = float4(emission, 0.0);
}

struct LightScanParams {
    float* powers;
    LightsHeader* header;
    uint entry_count;
    uint pad;
};
[[vk::push_constant]] LightScanParams scan_params;

// Serial prefix sum; runs once per light-set change. Fine for tens of thousands of
// entries; a chunked scan is the upgrade if scenes ever carry millions of emissive tris.
[shader("compute")]
[numthreads(1, 1, 1)]
void light_scan(uint3 gid: SV_DispatchThreadID) {
    if (gid.x != 0) {
        return;
    }
    let p = scan_params;
    float sum = 0.0;
    for (uint i = 0; i < p.entry_count; i++) {
        sum += p.powers[i];
        p.powers[i] = sum;
    }
    p.header[0].total_power = sum;
}
