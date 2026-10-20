// The `particles` example's ember kernel (examples/particles.rs `EmberPush`, scalar layout): a
// thread per ember. Embers rise off a disc (the lava), drift on a swirl, cool and shrink, and
// respawn when their life runs out. Each writes its TLAS instance row (an emissive analytic
// sphere, unit diameter, scaled to the ember) and last frame's row (motion vectors).
import instance_rows;

struct Ember {
    float3 pos;
    float age;
    float3 vel;
    float life;
};

struct EmberPush {
    Ember* embers;
    // The block's rows (instance_rows.slang); a thread per row.
    BlockRows rows;
    uint count;
    uint frame;
    float dt;
    float time;
    float origin_x, origin_y, origin_z;
    float radius;
    float size;
    float rise;
    float swirl;
    float life;
};
[[vk::push_constant]] EmberPush push;

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

float unit(inout uint s) {
    return float(pcg(s) & 0xFFFFFFu) / 16777216.0;
}

Transform sphereRow(float3 at, float d) {
    return { float4(d, 0.0, 0.0, at.x), float4(0.0, d, 0.0, at.y), float4(0.0, 0.0, d, at.z) };
}

// How big an ember of this age is: it cools and shrinks over its life.
float emberSize(float age, float life) {
    let k = saturate(age / max(life, 1.0e-3));
    return push.size * sqrt(1.0 - k);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void embers_step(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= push.rows.count) {
        return;
    }
    // Past the live count: nothing there, and a fresh start when the count grows again.
    if (i >= push.count) {
        push.embers[i].life = 0.0;
        writeRow(push.rows, i, noTransform(), noTransform());
        return;
    }
    var s = i * 9781u + push.frame * 6271u + 1u;
    var e = push.embers[i];
    let origin = float3(push.origin_x, push.origin_y, push.origin_z);
    let fresh = e.life <= 0.0;
    if (fresh || e.age >= e.life) {
        // A new ember somewhere on the disc, its life drawn about the mean; the very first
        // generation starts part-way through, so they do not all rise as one sheet.
        let a = unit(s) * 6.2831853;
        let r = push.radius * sqrt(unit(s));
        e.pos = origin + float3(cos(a) * r, 0.0, sin(a) * r);
        e.vel = float3((unit(s) - 0.5) * 0.4, push.rise * (0.5 + unit(s)), (unit(s) - 0.5) * 0.4);
        e.life = push.life * (0.5 + unit(s));
        e.age = fresh ? unit(s) * e.life : 0.0;
        push.embers[i] = e;
        // Born this frame: no motion yet.
        let born = sphereRow(e.pos, emberSize(e.age, e.life));
        writeRow(push.rows, i, born, born);
        return;
    }
    let before = e.pos;
    let size_before = emberSize(e.age, e.life);
    let dt = push.dt;
    // Buoyancy fading as it cools, drag, and a swirl about the column (curl-like, from the
    // height and time) that throws embers out as they climb.
    let k = saturate(e.age / e.life);
    let up = push.rise * (1.0 - k) * 1.5;
    let rel = e.pos - origin;
    let phase = rel.y * 0.9 + push.time * 0.7 + float(i & 7u);
    let swirl = float3(sin(phase) - rel.z * 0.15, 0.0, cos(phase * 1.3) + rel.x * 0.15) * push.swirl;
    e.vel += (float3(0.0, up, 0.0) + swirl - e.vel * 0.8) * dt;
    e.pos += e.vel * dt;
    e.age += dt;
    push.embers[i] = e;
    writeRow(push.rows, i, sphereRow(e.pos, emberSize(e.age, e.life)),
             sphereRow(before, size_before));
}
