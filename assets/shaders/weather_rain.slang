// Weather's rain (src/weather.rs): streaks falling through a box about the camera, and their
// splashes. A drop casts its path when it starts (and when it wraps across the box's side): where
// it lands, on what, and when. Landing, it starts its splash (three droplets thrown off the
// surface; on water a ring spreading over it too) and starts again at the top.
import instance_rows;

// A material record's words a drop's ray reads (blas.rs `RTXMaterial`, 28 words).
static const uint MATERIAL_WORDS = 28u;
static const uint MATERIAL_FLAGS_WORD = 25u;
static const uint MATERIAL_WATER_FLAG = 1u;
static const uint MATERIAL_TRANSMISSION_WORD = 13u;

// Instance rows per splash, and how long one lasts (seconds).
static const uint SPLASH_PARTS = 3u;
static const float SPLASH_LIFE = 0.3;
static const float SPLASH_SPEED = 1.6;
// A ripple ring's life (seconds), and how fast it spreads (metres per second).
static const float RIPPLE_LIFE = 0.6;
static const float RIPPLE_SPEED = 0.45;
static const float GRAVITY = 9.81;

// What a drop knows: where it is, where and when it lands (`land_in` seconds; `kind` 0 nothing in
// its way, 1 a surface, 2 water), and its last splash (`splash_at` its start, the frame clock).
struct Drop {
    float3 pos;
    uint live;
    float3 hit;
    float land_in;
    float3 normal;
    uint kind;
    float3 splash;
    float splash_at;
    float3 splash_normal;
    uint splash_kind;
};

// The rain's blocks (weather.rs `RainRows`): a row per drop, `SPLASH_PARTS` per splash, one per
// ripple.
struct RainRows {
    BlockRows drops;
    BlockRows splashes;
    BlockRows ripples;
};

struct RainPush {
    Drop* drops;
    RainRows* rows;
    // Last frame's TLAS and its material records (0: none yet, drops fall to the box's floor).
    uint64_t tlas;
    uint* materials;
    // Drops falling (the rest of the block's rows go empty).
    uint count;
    uint frame;
    float dt;
    float time;
    float cam_x, cam_y, cam_z;
    float radius;
    float height;
    float speed;
    float wind_x, wind_z;
    float length;
    float width;
    uint mask;
    uint pad;
};
[[vk::push_constant]] RainPush push;

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

float unit(inout uint s) {
    return float(pcg(s) & 0xFFFFFFu) / 16777216.0;
}

// The unit sphere stretched to `len` along `axis` and `w` across it, at `at`.
Transform stretched(float3 at, float3 axis, float len, float w) {
    let side = normalize(cross(axis, abs(axis.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    let other = cross(side, axis);
    let x = side * w;
    let y = axis * len;
    let z = other * w;
    return { float4(x.x, y.x, z.x, at.x), float4(x.y, y.y, z.y, at.y), float4(x.z, y.z, z.z, at.z) };
}

// Where a drop falling along `dir` lands, within `t_max`: the nearest surface it meets. Leaves
// and grass included: their opacity micromaps cull the cards' clear texels in the query itself
// (a card baked without one stops it as a whole quad). Clear surfaces other than water (the rain
// itself, its ripples, glass) it passes.
void castDrop(inout Drop d, float3 dir, float t_max, float speed) {
    d.kind = 0u;
    d.land_in = t_max / speed;
    if (push.tlas == 0) {
        return;
    }
    RayDesc ray;
    ray.Origin = d.pos;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    RaytracingAccelerationStructure tlas = RaytracingAccelerationStructure(push.tlas);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, push.mask & 0xFFu, ray);
    var normal = float3(0.0, 1.0, 0.0);
    var water = false;
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            // The analytic sphere (unit diameter in object space), unless it lets light through.
            let record = push.materials + q.CandidateInstanceID() * MATERIAL_WORDS;
            if (asfloat(record[MATERIAL_TRANSMISSION_WORD]) > 0.5) {
                continue;
            }
            let o = q.CandidateObjectRayOrigin();
            let od = q.CandidateObjectRayDirection();
            let b = dot(o, od);
            let a = dot(od, od);
            let disc = b * b - a * (dot(o, o) - 0.25);
            if (disc < 0.0) {
                continue;
            }
            let t = (-b - sqrt(disc)) / a;
            if (t > 0.0 && t < q.CommittedRayT()) {
                normal = normalize(mul(q.CandidateObjectToWorld3x4(), float4(o + od * t, 0.0)));
                water = false;
                q.CommitProceduralPrimitiveHit(t);
            }
            continue;
        }
        let record = push.materials + (q.CandidateInstanceID() + q.CandidateGeometryIndex()) * MATERIAL_WORDS;
        let is_water = (record[MATERIAL_FLAGS_WORD] & MATERIAL_WATER_FLAG) != 0u;
        // Clear triangles other than water (ripple rings, glass) it passes.
        if (!is_water && asfloat(record[MATERIAL_TRANSMISSION_WORD]) > 0.5) {
            continue;
        }
        // Proceed only offers candidates nearer than the committed one.
        float3x4 to_world = q.CandidateObjectToWorld3x4();
        float3 v[3] = q.CandidateGetIntersectionTriangleVertexPositions();
        let w0 = mul(to_world, float4(v[0], 1.0));
        let w1 = mul(to_world, float4(v[1], 1.0));
        let w2 = mul(to_world, float4(v[2], 1.0));
        normal = normalize(cross(w1 - w0, w2 - w0));
        water = is_water;
        q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() != COMMITTED_NOTHING) {
        let t = q.CommittedRayT();
        d.hit = d.pos + dir * t;
        d.normal = dot(normal, dir) > 0.0 ? -normal : normal;
        d.kind = water ? 2u : 1u;
        d.land_in = t / speed;
    }
}

// A water splash's ripple at `age` seconds: the ring mesh (unit radius, in its xz plane) spread
// over the surface, thinning as it goes.
Transform ripple(Drop d, float age) {
    if (age < 0.0 || age >= RIPPLE_LIFE || d.splash_kind != 2u) {
        return noTransform();
    }
    let k = age / RIPPLE_LIFE;
    let n = d.splash_normal;
    let side = normalize(cross(n, abs(n.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    let other = cross(n, side);
    let r = 0.03 + RIPPLE_SPEED * age;
    let x = side * r;
    let y = n * (0.006 * (1.0 - k));
    let z = other * r;
    let at = d.splash + n * 0.004;
    return { float4(x.x, y.x, z.x, at.x), float4(x.y, y.y, z.y, at.y), float4(x.z, y.z, z.z, at.z) };
}

// Splash part `j` at `age` seconds: droplets thrown up and out along the normal under gravity,
// shrinking (smaller off water).
Transform splashPart(Drop d, uint i, uint j, float age) {
    if (age < 0.0 || age >= SPLASH_LIFE || d.splash_kind == 0u) {
        return noTransform();
    }
    let k = age / SPLASH_LIFE;
    let n = d.splash_normal;
    let side = normalize(cross(n, abs(n.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    let other = cross(n, side);
    var s = i * 2654435761u + j * 40503u;
    let a = (float(j) + unit(s)) * 2.0943951;
    let dir = normalize(n * 1.3 + (side * cos(a) + other * sin(a)) * (0.6 + 0.4 * unit(s)));
    let v = dir * SPLASH_SPEED * (0.7 + 0.6 * unit(s));
    let at = d.splash + v * age + float3(0.0, -0.5 * GRAVITY * age * age, 0.0);
    let size = (d.splash_kind == 2u ? 0.012 : 0.018) * (1.0 - k);
    return stretched(at, normalize(v + float3(0.0, -GRAVITY * age, 0.0)), size * 1.6, size);
}

// A thread per drop: it falls with the wind, wraps across the box's sides as the camera moves,
// lands where its cast said and splashes there, then starts again at the top.
[shader("compute")]
[numthreads(64, 1, 1)]
void rain_step(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    let rows = *push.rows;
    if (i >= rows.drops.count) {
        return;
    }
    if (i >= push.count) {
        push.drops[i].live = 0u;
        push.drops[i].splash_kind = 0u;
        writeRow(rows.drops, i, noTransform(), noTransform());
        writeRow(rows.ripples, i, noTransform(), noTransform());
        for (uint j = 0u; j < SPLASH_PARTS; j++) {
            writeRow(rows.splashes, i * SPLASH_PARTS + j, noTransform(), noTransform());
        }
        return;
    }
    var s = i * 7919u + push.frame * 104729u + 3u;
    let cam = float3(push.cam_x, push.cam_y, push.cam_z);
    let vel = float3(push.wind_x, -push.speed, push.wind_z);
    let speed = length(vel);
    let dir = vel / speed;
    let top = cam.y + push.height * 0.6;
    let floor_y = cam.y - push.height * 0.4;
    var d = push.drops[i];
    var moved = true;
    var cast = false;
    if (d.live == 0u) {
        // First placed anywhere in the box, so the rain does not arrive as one sheet.
        d.pos = cam + float3((unit(s) * 2.0 - 1.0) * push.radius, (unit(s) - 0.4) * push.height,
                             (unit(s) * 2.0 - 1.0) * push.radius);
        d.live = 1u;
        d.splash_kind = 0u;
        moved = false;
        cast = true;
    }
    let before = d.pos;
    if (!cast) {
        d.pos += vel * push.dt;
        d.land_in -= push.dt;
        if (d.land_in <= 0.0) {
            // Landed: its splash starts where its cast said, and the drop again at the top.
            if (d.kind != 0u) {
                d.splash = d.hit;
                d.splash_normal = d.normal;
                d.splash_kind = d.kind;
                d.splash_at = push.time;
            }
            d.pos = float3(cam.x + (unit(s) * 2.0 - 1.0) * push.radius, top - unit(s) * push.height * 0.1,
                           cam.z + (unit(s) * 2.0 - 1.0) * push.radius);
            moved = false;
            cast = true;
        }
    }
    // Across the box's sides: in again on the far side (the camera walked away from it).
    var rel = d.pos.xz - cam.xz;
    if (abs(rel.x) > push.radius) {
        rel.x -= sign(rel.x) * 2.0 * push.radius;
        moved = false;
        cast = true;
    }
    if (abs(rel.y) > push.radius) {
        rel.y -= sign(rel.y) * 2.0 * push.radius;
        moved = false;
        cast = true;
    }
    d.pos.xz = cam.xz + rel;
    // Out through the top or the floor (the camera climbed or dropped): again from the top.
    if (d.pos.y < floor_y || d.pos.y > top + 1.0) {
        d.pos = float3(cam.x + (unit(s) * 2.0 - 1.0) * push.radius, top - unit(s) * push.height * 0.1,
                       cam.z + (unit(s) * 2.0 - 1.0) * push.radius);
        moved = false;
        cast = true;
    }
    if (cast) {
        castDrop(d, dir, max(d.pos.y - floor_y, 0.1) / max(-dir.y, 0.05), speed);
    }
    push.drops[i] = d;
    let now = stretched(d.pos, dir, push.length, push.width);
    writeRow(rows.drops, i, now, moved ? stretched(before, dir, push.length, push.width) : now);

    let age = push.time - d.splash_at;
    for (uint j = 0u; j < SPLASH_PARTS; j++) {
        let part = splashPart(d, i, j, age);
        writeRow(rows.splashes, i * SPLASH_PARTS + j, part,
                 age - push.dt >= 0.0 ? splashPart(d, i, j, age - push.dt) : part);
    }
    let ring = ripple(d, age);
    writeRow(rows.ripples, i, ring, age - push.dt >= 0.0 ? ripple(d, age - push.dt) : ring);
}
