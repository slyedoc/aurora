// Material-stack evaluation, one thread per texel of one page (aurora_landscape page.rs),
// after the height stack: `clear_materials`, then each enabled material layer bottom to top.
import pages;
import materials;

// `MatPush` in page.rs (128 bytes, the whole push block).
struct MatPush {
    uint2* page;
    uint* source_ids;
    uint* source_weights;
    PagePool pool;
    int page_x;
    int page_z;
    uint src_w;
    uint src_h;
    // Landscape xz -> source uv: uv = m * xz + o.
    float m00;
    float m01;
    float m10;
    float m11;
    float ox;
    float oz;
    float opacity;
    uint material;
    float slope_lo;
    float slope_hi;
    float slope_fade;
    float height_lo;
    float height_hi;
    float height_fade;
};
[[vk::push_constant]] MatPush push;

int2 texel_of(uint i) {
    return int2(push.page_x * PAGE + int(i % PAGE), push.page_z * PAGE + int(i / PAGE));
}

// Ramps in over `fade` below `lo` and out over `fade` above `hi`.
float window(float x, float lo, float hi, float fade) {
    let f = max(fade, 1.0e-4);
    return smoothstep(lo - f, lo, x) * (1.0 - smoothstep(hi, hi + f, x));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_materials(uint3 gid: SV_DispatchThreadID) {
    if (gid.x < PAGE * PAGE) {
        // Palette material 0, fully.
        push.page[gid.x] = uint2(0u, 0xFFu);
    }
}

MatSet splat_at(int x, int y) {
    let cx = clamp(x, 0, int(push.src_w) - 1);
    let cy = clamp(y, 0, int(push.src_h) - 1);
    let i = cy * int(push.src_w) + cx;
    return mat_unpack(uint2(push.source_ids[i], push.source_weights[i]));
}

// A splatmap over a rectangle (texel centres), replacing what is below inside it.
[shader("compute")]
[numthreads(64, 1, 1)]
void apply_splatmap(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let xz = float2(texel_of(gid.x)) * push.pool.texel;
    let uv = float2(push.m00 * xz.x + push.m01 * xz.y, push.m10 * xz.x + push.m11 * xz.y)
        + float2(push.ox, push.oz);
    let eps = 1.0e-5;
    if (any(uv < -eps) || any(uv > 1.0 + eps)) {
        return;
    }
    let g = saturate(uv) * float2(float(push.src_w), float(push.src_h)) - 0.5;
    let f = floor(g);
    let t = g - f;
    let x = int(f.x);
    let y = int(f.y);
    var s = mat_empty();
    mat_merge(s, splat_at(x, y), (1.0 - t.x) * (1.0 - t.y));
    mat_merge(s, splat_at(x + 1, y), t.x * (1.0 - t.y));
    mat_merge(s, splat_at(x, y + 1), (1.0 - t.x) * t.y);
    mat_merge(s, splat_at(x + 1, y + 1), t.x * t.y);
    var below = mat_unpack(push.page[gid.x]);
    mat_scale(below, 1.0 - push.opacity);
    mat_merge(below, s, push.opacity);
    push.page[gid.x] = mat_pack(below);
}

// One material where the slope (degrees) and height fall in their windows.
[shader("compute")]
[numthreads(64, 1, 1)]
void apply_rule(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let t = texel_of(gid.x);
    let p = push.pool;
    let h = texel_height(p, t.x, t.y);
    let dx = texel_height(p, t.x + 1, t.y) - texel_height(p, t.x - 1, t.y);
    let dz = texel_height(p, t.x, t.y + 1) - texel_height(p, t.x, t.y - 1);
    let slope = degrees(atan(length(float2(dx, dz)) / (2.0 * p.texel)));
    let w = window(slope, push.slope_lo, push.slope_hi, push.slope_fade)
        * window(h, push.height_lo, push.height_hi, push.height_fade) * push.opacity;
    if (w <= 0.0) {
        return;
    }
    var s = mat_unpack(push.page[gid.x]);
    mat_scale(s, 1.0 - w);
    mat_add(s, push.material, w);
    push.page[gid.x] = mat_pack(s);
}

// A paint layer's page (aligned with this one): its weights' sum is its coverage.
[shader("compute")]
[numthreads(64, 1, 1)]
void apply_paint(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let paint = mat_unpack(uint2(push.source_ids[gid.x * 2], push.source_ids[gid.x * 2 + 1]));
    let cover = saturate(mat_total(paint)) * push.opacity;
    if (cover <= 0.0) {
        return;
    }
    var s = mat_unpack(push.page[gid.x]);
    mat_scale(s, 1.0 - cover);
    mat_merge(s, paint, push.opacity);
    push.page[gid.x] = mat_pack(s);
}
