// Sculpt and paint brushes, one thread per texel of one layer page (aurora_landscape
// sculpt.rs). Sculpt writes the layer's height delta; smooth and flatten read the evaluated
// heights. Paint writes a paint page (`materials.slang`, raw weights = coverage).
import pages;
import materials;

static const uint RAISE = 0;
static const uint LOWER = 1;
static const uint SMOOTH = 2;
static const uint FLATTEN = 3;
static const uint PAINT = 4;
static const uint ERASE = 5;

// `BrushPush` in sculpt.rs. Sculpt writes `delta`, paint / erase write `paint` (a uint2 per
// texel); `target` is the painted palette id. Two typed pointers, not one cast: a reinterpreted
// pointer has no array stride, which is invalid SPIR-V.
struct BrushPush {
    float* delta;
    uint2* paint;
    PagePool pool;
    int page_x;
    int page_z;
    float center_x;
    float center_z;
    float radius;
    float strength;
    float target;
    uint kind;
};
[[vk::push_constant]] BrushPush push;

[shader("compute")]
[numthreads(64, 1, 1)]
void sculpt_brush(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let tx = push.page_x * PAGE + int(gid.x % PAGE);
    let tz = push.page_z * PAGE + int(gid.x / PAGE);
    let xz = float2(float(tx), float(tz)) * push.pool.texel;
    let d = length(xz - float2(push.center_x, push.center_z));
    if (d >= push.radius) {
        return;
    }
    // Smooth bell: full at the centre, zero at the rim.
    let w = 1.0 - smoothstep(0.0, push.radius, d);
    if (push.kind >= PAINT) {
        var s = mat_unpack(push.paint[gid.x]);
        let k = saturate(push.strength * w);
        if (push.kind == PAINT) {
            // Toward full coverage of the painted id, the rest giving way.
            mat_scale(s, 1.0 - k);
            mat_add(s, uint(push.target), k);
        } else {
            mat_scale(s, 1.0 - k);
        }
        push.paint[gid.x] = mat_pack_raw(s);
        return;
    }
    let h = texel_height(push.pool, tx, tz);
    float dh;
    switch (push.kind) {
        case RAISE: dh = push.strength * w; break;
        case LOWER: dh = -push.strength * w; break;
        case SMOOTH: {
            let r = 2;
            var sum = 0.0;
            for (int j = -r; j <= r; j++) {
                for (int i = -r; i <= r; i++) {
                    sum += texel_height(push.pool, tx + i, tz + j);
                }
            }
            dh = (sum / 25.0 - h) * saturate(push.strength * w);
            break;
        }
        default: dh = (push.target - h) * saturate(push.strength * w); break;
    }
    push.delta[gid.x] += dh;
}
