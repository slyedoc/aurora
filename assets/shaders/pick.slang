// Picking (see src/picking.rs): trace a batch of rays against the frame's TLAS and keep each
// ray's nearest MAX_HITS hits, nearest first. Every triangle reports as a candidate
// (FORCE_NON_OPAQUE) so overlapping geometry along the ray is all seen; alpha cutouts count as
// solid. The normal is the hit triangle's geometric normal, from vertices fetched out of the
// BLAS, turned to face the ray. A ray whose nearest hits reach a volume portal's small shape
// (src/portal.rs) carries on inside its target environment instead, to where it leaves the big
// shape: those hits are in the target's space, `t` still along the ray as cast, and `through` the
// small shape's instance slot + 1.
import volume_portal;

static const uint MAX_HITS = 8;
// SBT hit-group offset 0 is the analytic sphere (tlas_builder.rs); its object-space radius.
static const uint SPHERE_HIT_OFFSET = 0;
static const float SPHERE_RADIUS = 0.5;

struct Ray {
    float3 origin;
    float t_max;
    float3 direction;
    uint mask;
};

// `t < 0` ends a ray's list.
struct Hit {
    float3 position;
    float t;
    float3 normal;
    uint instance;
    float2 barycentrics;
    uint primitive;
    uint through;
};

struct PickParams {
    uint64_t tlas;
    Ray* rays;
    Hit* hits;
    uint count;
    uint pad;
    // Each TLAS row's instance slot (tlas_builder.rs packs the live slots).
    uint* slots;
    // The portal table (src/portal.rs) and the instance rows, by slot.
    uint4* portals;
    float4* instances;
    uint portal_count;
    uint pad2;
};

[[vk::push_constant]] PickParams pc;

// Ray parameters of the unit-radius-scaled sphere hits, or (-1, -1).
float2 sphere_hits(float3 origin, float3 direction) {
    float a = dot(direction, direction);
    float b = dot(origin, direction);
    float c = dot(origin, origin) - SPHERE_RADIUS * SPHERE_RADIUS;
    float disc = b * b - a * c;
    if (disc < 0.0) {
        return float2(-1.0);
    }
    float s = sqrt(disc);
    return float2((-b - s) / a, (-b + s) / a);
}

void insert(inout Hit list[MAX_HITS], inout uint len, Hit hit) {
    if (len == MAX_HITS && hit.t >= list[MAX_HITS - 1].t) {
        return;
    }
    uint i = min(len, MAX_HITS - 1);
    while (i > 0 && list[i - 1].t > hit.t) {
        list[i] = list[i - 1];
        i--;
    }
    list[i] = hit;
    len = min(len + 1, MAX_HITS);
}

float3 facing(float3 n, float3 direction) {
    n = normalize(n);
    return dot(n, direction) > 0.0 ? -n : n;
}

// The hits along `desc` within `mask`, as `t_base + t * t_scale`, into `list`.
void gather(RaytracingAccelerationStructure tlas, RayDesc desc, uint mask, float t_base, float t_scale,
            uint through, inout Hit list[MAX_HITS], inout uint len) {
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, mask & 0xFF, desc);
    while (query.Proceed()) {
        Hit hit;
        hit.instance = pc.slots[query.CandidateInstanceIndex()];
        hit.primitive = query.CandidatePrimitiveIndex();
        hit.through = through;
        float3x4 to_world = query.CandidateObjectToWorld3x4();
        float t;
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            t = query.CandidateTriangleRayT();
            hit.barycentrics = query.CandidateTriangleBarycentrics();
            float3 v[3] = query.CandidateGetIntersectionTriangleVertexPositions();
            float3 w0 = mul(to_world, float4(v[0], 1.0));
            float3 w1 = mul(to_world, float4(v[1], 1.0));
            float3 w2 = mul(to_world, float4(v[2], 1.0));
            hit.normal = facing(cross(w1 - w0, w2 - w0), desc.Direction);
        } else {
            // Procedural: only the analytic sphere has an intersection here.
            if (query.CandidateInstanceContributionToHitGroupIndex() != SPHERE_HIT_OFFSET) {
                continue;
            }
            float3 o = query.CandidateObjectRayOrigin();
            float3 d = query.CandidateObjectRayDirection();
            float2 ts = sphere_hits(o, d);
            t = ts.x >= desc.TMin ? ts.x : ts.y;
            if (t < desc.TMin || t > desc.TMax) {
                continue;
            }
            hit.barycentrics = float2(0.0);
            float3 center = mul(to_world, float4(0.0, 0.0, 0.0, 1.0));
            float3 point = desc.Origin + desc.Direction * t;
            hit.normal = facing(point - center, desc.Direction);
        }
        hit.position = desc.Origin + desc.Direction * t;
        hit.t = t_base + t * t_scale;
        insert(list, len, hit);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pick(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) {
        return;
    }
    Ray ray = pc.rays[id.x];
    RaytracingAccelerationStructure tlas = RaytracingAccelerationStructure(pc.tlas);

    RayDesc desc;
    desc.Origin = ray.origin;
    desc.Direction = ray.direction;
    desc.TMin = 0.0;
    desc.TMax = ray.t_max;

    Hit list[MAX_HITS];
    uint len = 0;
    gather(tlas, desc, ray.mask, 0.0, 1.0, 0u, list, len);

    // Through the nearest volume portal: what lies behind it here is replaced by what it shows.
    for (uint i = 0; i < len && pc.portal_count > 0u; i++) {
        let vol = volumePortalOf(pc.portals, pc.portal_count, list[i].instance);
        if (vol == 0u) {
            continue;
        }
        let v = volumePortalLoad(pc.portals, vol - 1u);
        let sb = v.slot * 4u;
        float4 s0, s1, s2;
        portalAffineInverse(pc.instances[sb], pc.instances[sb + 1u], pc.instances[sb + 2u], s0, s1, s2);
        let lp = portalXformPoint(s0, s1, s2, list[i].position);
        let ld = portalXformDir(s0, s1, s2, ray.direction);
        let d = portalXformDir(v.t0, v.t1, v.t2, ld);
        let s = length(d);
        RayDesc inner;
        inner.Direction = d / s;
        inner.Origin = portalXformPoint(v.t0, v.t1, v.t2, lp);
        inner.TMin = 1.0e-4 * s;
        inner.TMax = min(volumeExit(v, inner.Origin, inner.Direction), (ray.t_max - list[i].t) * s);
        let t_base = list[i].t;
        len = i;
        gather(tlas, inner, 1u << v.env_bit, t_base, 1.0 / s, v.slot + 1u, list, len);
        break;
    }

    uint base = id.x * MAX_HITS;
    for (uint i = 0; i < MAX_HITS; i++) {
        if (i < len) {
            pc.hits[base + i] = list[i];
        } else {
            Hit none = (Hit)0;
            none.t = -1.0;
            pc.hits[base + i] = none;
            break;
        }
    }
}
