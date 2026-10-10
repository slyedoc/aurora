// Height-stack evaluation, one thread per texel of one page (aurora_landscape page.rs
// `evaluate_pages`): `clear_page`, then each enabled layer bottom to top.
import pages;

static const uint BLEND_REPLACE = 0;
static const uint BLEND_ADD = 1;
static const uint BLEND_MAX = 2;
static const uint BLEND_MIN = 3;

// `EvalPush` in page.rs.
struct EvalPush {
    float* page;
    float* source;
    int page_x;
    int page_z;
    float texel;
    uint src_w;
    uint src_h;
    // Landscape xz -> source uv: uv = m * xz + o.
    float m00;
    float m01;
    float m10;
    float m11;
    float ox;
    float oz;
    float height_min;
    float height_max;
    float falloff;
    float opacity;
    uint blend;
};
[[vk::push_constant]] EvalPush push;

uint2 texel_of(uint i) {
    return uint2(i % PAGE, i / PAGE);
}

float2 texel_xz(uint2 t) {
    return float2(float(push.page_x * PAGE + int(t.x)), float(push.page_z * PAGE + int(t.y)))
        * push.texel;
}

float source_at(int x, int y) {
    let cx = clamp(x, 0, int(push.src_w) - 1);
    let cy = clamp(y, 0, int(push.src_h) - 1);
    return push.source[cy * int(push.src_w) + cx];
}

float4 catmull_rom(float t) {
    let t2 = t * t;
    let t3 = t2 * t;
    return float4(-0.5 * t3 + t2 - 0.5 * t,
                  1.5 * t3 - 2.5 * t2 + 1.0,
                  -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                  0.5 * t3 - 0.5 * t2);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_page(uint3 gid: SV_DispatchThreadID) {
    if (gid.x < PAGE * PAGE) {
        push.page[gid.x] = 0.0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_heightmap(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let xz = texel_xz(texel_of(gid.x));
    let uv = float2(push.m00 * xz.x + push.m01 * xz.y, push.m10 * xz.x + push.m11 * xz.y)
        + float2(push.ox, push.oz);
    let eps = 1.0e-5;
    if (any(uv < -eps) || any(uv > 1.0 + eps)) {
        return;
    }
    let c = saturate(uv);
    // Corner-aligned samples: the first and last span the layer's edges.
    let g = c * float2(float(push.src_w - 1), float(push.src_h - 1));
    let f = floor(g);
    let t = g - f;
    let x = int(f.x);
    let y = int(f.y);
    // Catmull-Rom: through every sample, with a continuous slope (bilinear's slope jumps at
    // each source cell, which shows as facets and as steps in slope rules).
    let wx = catmull_rom(t.x);
    let wy = catmull_rom(t.y);
    var s = 0.0;
    for (int j = 0; j < 4; j++) {
        var row = 0.0;
        for (int i = 0; i < 4; i++) {
            row += source_at(x - 1 + i, y - 1 + j) * wx[i];
        }
        s += row * wy[j];
    }
    let v = lerp(push.height_min, push.height_max, s);

    // 0 at the layer's edge, 1 once `falloff` (of the half extent) inside it.
    let edge = min(min(c.x, 1.0 - c.x), min(c.y, 1.0 - c.y)) * 2.0;
    let fade = push.falloff > 0.0 ? smoothstep(0.0, push.falloff, edge) : 1.0;
    let a = fade * push.opacity;

    let h = push.page[gid.x];
    float r;
    switch (push.blend) {
        case BLEND_ADD: r = h + v; break;
        case BLEND_MAX: r = max(h, v); break;
        case BLEND_MIN: r = min(h, v); break;
        default: r = v; break;
    }
    push.page[gid.x] = lerp(h, r, a);
}

// A sculpt layer's delta page, aligned with this page.
[shader("compute")]
[numthreads(64, 1, 1)]
void apply_sculpt(uint3 gid: SV_DispatchThreadID) {
    if (gid.x < PAGE * PAGE) {
        push.page[gid.x] += push.source[gid.x] * push.opacity;
    }
}
