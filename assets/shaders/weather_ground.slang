// Weather (src/weather.rs): the terrain heights around the camera that ground mist lies on and
// rain lands on.

// A material record's words the ground rays read (blas.rs `RTXMaterial`, 28 words).
static const uint MATERIAL_WORDS = 28u;
static const uint MATERIAL_ALPHA_CUTOFF_WORD = 20u;

// Nothing under this texel: mist and rain treat it as far below.
static const float GROUND_NONE = -1.0e30;

struct GroundPush {
    uint64_t tlas;
    uint* materials;
    float* heights;
    float origin_x, origin_z;
    float texel;
    uint size;
    float top;
    float depth;
    uint mask;
};
[[vk::push_constant]] GroundPush push;

// A thread per texel: one ray straight down from `top`, the nearest surface that is not an
// alpha-tested card (grass, leaves: mist and rain fall through foliage to the ground under it).
[shader("compute")]
[numthreads(64, 1, 1)]
void ground_heights(uint3 gid: SV_DispatchThreadID) {
    let n = push.size;
    if (gid.x >= n * n) {
        return;
    }
    let i = gid.x % n;
    let j = gid.x / n;
    RayDesc ray;
    ray.Origin = float3(push.origin_x + float(i) * push.texel, push.top,
                        push.origin_z + float(j) * push.texel);
    ray.Direction = float3(0.0, -1.0, 0.0);
    ray.TMin = 0.0;
    ray.TMax = push.depth;
    RaytracingAccelerationStructure tlas = RaytracingAccelerationStructure(push.tlas);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, push.mask & 0xFFu, ray);
    while (q.Proceed()) {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        let record = push.materials + (q.CandidateInstanceID() + q.CandidateGeometryIndex()) * MATERIAL_WORDS;
        if (asfloat(record[MATERIAL_ALPHA_CUTOFF_WORD]) > 0.0) {
            continue;
        }
        q.CommitNonOpaqueTriangleHit();
    }
    push.heights[gid.x] = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? push.top - q.CommittedRayT() : GROUND_NONE;
}

