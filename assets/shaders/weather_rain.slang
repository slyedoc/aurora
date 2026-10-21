// Weather's rain (src/weather.rs): streaks falling through a box about the camera, and their
// splashes. A drop casts its path when it starts (and when it wraps across the box's side): where
// it lands, on what, and when. Landing, it starts its splash (three droplets thrown off the
// surface; on water a ring spreading over it too) and starts again at the top.
import instance_rows;

// A material record's words a drop's ray reads (blas.rs `RTXMaterial`, 28 words).
static const uint MATERIAL_WORDS = 28u;
static const uint MATERIAL_FLAGS_WORD = 25u;
static const uint MATERIAL_WATER_FLAG = 1u;
static const uint MATERIAL_NO_SHADOW_FLAG = 2u;
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
// its way, 1 a surface, 2 water, 3 sheltered: something above it already caught it), and its
// last splash (`splash_at` its start, the frame clock).
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
    // The angle one output pixel spans (weather.rs; 0: no floor).
    float pixel_angle;
};
[[vk::push_constant]] RainPush push;

// The least width a thing at `p` takes to cover about a pixel from the camera: thinner, the
// tracer's samples miss it and it drops out of the image.
float pixelFloor(float3 p) {
    return distance(p, float3(push.cam_x, push.cam_y, push.cam_z)) * push.pixel_angle * 0.8;
}

// How far below a drop its cast reaches (metres), and how far below the camera a drop may be
// before it starts again (one that somehow outlived its landing).
static const float MAX_FALL = 1000.0;
// How far up a drop's cast starts, back along its fall line: what the rain meets first is decided
// from the sky, not from where the drop shows (it shows only about the camera).
static const float SKY_REACH = 500.0;
static const uint KIND_SHELTERED = 3u;
// More spots a drop starting under cover tries, the same frame.
static const uint SHELTER_TRIES = 4u;

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

float unit(inout uint s) {
    return float(pcg(s) & 0xFFFFFFu) / 16777216.0;
}

// Where about the camera a drop starts (xz offset): within `radius`, its density per area falling
// as 1 / distance (even on screen: dense at the feet, sparse where a drop spans a pixel), with
// a uniform disc of `RAIN_CORE` metres in the middle (no clump straight overhead).
static const float RAIN_CORE = 6.0;

float2 spawnOffset(inout uint s) {
    let r_max = max(push.radius, RAIN_CORE * 1.01);
    let core = RAIN_CORE / (2.0 * r_max - RAIN_CORE);
    let a = unit(s) * 6.2831853;
    let r = unit(s) < core ? RAIN_CORE * sqrt(unit(s)) : lerp(RAIN_CORE, r_max, unit(s));
    return float2(cos(a), sin(a)) * r;
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

// Where a drop falling along `dir` lands, within `t_max` below it: the first surface its fall
// line meets coming down from the sky (`SKY_REACH` above it), so a roof or canopy above the drop
// shelters it (`KIND_SHELTERED`) however tall the room under it. Meeting nothing (over a void, or
// the scene still loading into the TLAS) it falls only `miss_fall` and casts again. Leaves and grass included:
// their opacity micromaps cull the cards' clear texels in the query itself (a card baked without
// one stops it as a whole quad). Clear surfaces other than water (the rain itself, its ripples,
// glass) it passes.
void castDrop(inout Drop d, float3 dir, float t_max, float miss_fall, float speed) {
    d.kind = 0u;
    d.land_in = max(miss_fall, 0.0) / speed;
    if (push.tlas == 0) {
        return;
    }
    RayDesc ray;
    ray.Origin = d.pos - dir * SKY_REACH;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = SKY_REACH + t_max;
    RaytracingAccelerationStructure tlas = RaytracingAccelerationStructure(push.tlas);
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, push.mask & 0xFFu, ray);
    var normal = float3(0.0, 1.0, 0.0);
    var water = false;
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            // The analytic sphere (unit diameter in object space), unless it lets light through.
            let record = push.materials + q.CandidateInstanceID() * MATERIAL_WORDS;
            if (asfloat(record[MATERIAL_TRANSMISSION_WORD]) > 0.5
                || (record[MATERIAL_FLAGS_WORD] & MATERIAL_NO_SHADOW_FLAG) != 0u) {
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
        // Clear triangles other than water (glass) and what casts no shadow (the rain's own
        // ripple rings) it passes.
        if ((!is_water && asfloat(record[MATERIAL_TRANSMISSION_WORD]) > 0.5)
            || (record[MATERIAL_FLAGS_WORD] & MATERIAL_NO_SHADOW_FLAG) != 0u) {
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
        // From the drop's own position: below zero, the surface lies above it.
        let t = q.CommittedRayT() - SKY_REACH;
        if (t < 0.0) {
            d.kind = KIND_SHELTERED;
            d.land_in = 0.0;
            return;
        }
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
    // Seen across the water, a ring shows its height: at least a pixel of it.
    let y = n * max(0.006 * (1.0 - k), pixelFloor(d.splash));
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
    let size = max((d.splash_kind == 2u ? 0.012 : 0.018) * (1.0 - k), pixelFloor(at));
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
    // Drops start in a box about the camera and fall until their cast lands them, however far
    // below that is.
    let top = cam.y + push.height * 0.6;
    let bottom = cam.y - push.height * 0.4;
    var d = push.drops[i];
    var moved = true;
    var cast = false;
    if (d.live == 0u) {
        // First placed anywhere in the box, so the rain does not arrive as one sheet.
        d.pos = float3(cam.x, lerp(bottom, top, unit(s)), cam.z);
        d.pos.xz += spawnOffset(s);
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
            // Landed: its splash starts where its cast said, and the drop again at the top. One
            // that only gave up (sheltered, or met nothing) starts anywhere in the box: many give
            // up at once (leaving cover, the scene loading), and from the top they would fall as
            // one sheet.
            let landed = d.kind == 1u || d.kind == 2u;
            if (landed) {
                d.splash = d.hit;
                d.splash_normal = d.normal;
                d.splash_kind = d.kind;
                d.splash_at = push.time;
            }
            let y = landed ? top - unit(s) * push.height * 0.1 : lerp(bottom, top, unit(s));
            d.pos = float3(cam.x, y, cam.z);
            d.pos.xz += spawnOffset(s);
            moved = false;
            cast = true;
        }
    }
    // Out of the circle (the camera moved away from it): again somewhere inside it.
    if (length(d.pos.xz - cam.xz) > max(push.radius, RAIN_CORE * 1.01)) {
        d.pos = float3(cam.x, lerp(bottom, top, unit(s)), cam.z);
        d.pos.xz += spawnOffset(s);
        moved = false;
        cast = true;
    }
    // Far above the box (the camera dropped) or fallen far past it landing on nothing: again
    // anywhere in the box, so a camera that jumps finds rain at once, not one sheet arriving.
    if (d.pos.y < cam.y - MAX_FALL || d.pos.y > top + 1.0) {
        d.pos = float3(cam.x, lerp(bottom, top, unit(s)), cam.z);
        d.pos.xz += spawnOffset(s);
        moved = false;
        cast = true;
    }
    // A drop whose cast met nothing (the scene still loading into the TLAS, a void below) casts
    // again each frame as it falls, rather than trusting one miss for its whole fall.
    if (d.kind == 0u) {
        cast = true;
    }
    if (cast) {
        // A miss falls to the box's bottom at most, then starts again.
        castDrop(d, dir, MAX_FALL, (d.pos.y - bottom) / max(-dir.y, 0.05), speed);
        // A drop starting under cover (a roof, a canopy) tries elsewhere in the box, so the rain
        // gathers where the sky is open: under cover the open part is what the camera sees.
        for (uint k = 0u; k < SHELTER_TRIES && !moved && d.kind == KIND_SHELTERED; k++) {
            d.pos.xz = cam.xz + spawnOffset(s);
            castDrop(d, dir, MAX_FALL, (d.pos.y - bottom) / max(-dir.y, 0.05), speed);
        }
    }
    push.drops[i] = d;
    // A drop still sheltered shows nowhere; it starts again elsewhere next frame.
    let width = max(push.width, pixelFloor(d.pos));
    let now = d.kind == KIND_SHELTERED ? noTransform() : stretched(d.pos, dir, push.length, width);
    writeRow(rows.drops, i, now, moved ? stretched(before, dir, push.length, width) : now);

    let age = push.time - d.splash_at;
    for (uint j = 0u; j < SPLASH_PARTS; j++) {
        let part = splashPart(d, i, j, age);
        writeRow(rows.splashes, i * SPLASH_PARTS + j, part,
                 age - push.dt >= 0.0 ? splashPart(d, i, j, age - push.dt) : part);
    }
    let ring = ripple(d, age);
    writeRow(rows.ripples, i, ring, age - push.dt >= 0.0 ? ripple(d, age - push.dt) : ring);
}
