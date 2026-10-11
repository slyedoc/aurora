// Port of bevy_ui_render's `ui.wesl` + `gradient.wesl`: rounded-box SDF backgrounds and
// borders, textured quads (images, glyph atlases) through this crate's bindless set, and
// linear / radial / conic gradient segments interpolated in a per-vertex color space.

import ui_color;

// Must match `ui_render::UiVertex` (repr(C), 168 bytes, all 4-byte fields).
struct UiVertex {
    float position_x, position_y;  // physical pixels, y down
    float uv_x, uv_y;
    float color_r, color_g, color_b, color_a;  // linear rgba
    uint flags;
    uint tex_index;  // bindless texture index (flags & TEXTURED)
    float radius_x0, radius_x1, radius_x2, radius_x3;  // top left, top right, bottom right, bottom left
    float radius_y0, radius_y1, radius_y2, radius_y3;
    float border_l, border_t, border_r, border_b;
    float size_x, size_y;
    float point_x, point_y;  // position relative to the center of the rectangle
    // gradient segment (flags & GRADIENT); colors are in `color_space`
    float g_start_x, g_start_y;
    float g_dir_x, g_dir_y;
    float start_color_r, start_color_g, start_color_b, start_color_a;
    float end_color_r, end_color_g, end_color_b, end_color_a;
    float start_len;
    float end_len;
    float hint;
    uint color_space;
};

// Must match `ui_render::UiPushConstants`.
struct UiPush {
    UiVertex* vertices;
    float2 screen_size;
};
[[vk::push_constant]] ConstantBuffer<UiPush> pc;

[[vk::binding(200, 0)]] Sampler2D textures[];

struct VertexOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] float4 color : COLOR0;
    [[vk::location(2)]] nointerpolation uint flags : TEXCOORD2;
    [[vk::location(3)]] nointerpolation uint tex : TEXCOORD3;
    [[vk::location(4)]] nointerpolation float4 radius_x : TEXCOORD4;
    [[vk::location(5)]] nointerpolation float4 radius_y : TEXCOORD5;
    [[vk::location(6)]] nointerpolation float4 border : TEXCOORD6;
    [[vk::location(7)]] nointerpolation float2 size : TEXCOORD7;
    [[vk::location(8)]] float2 point : TEXCOORD8;
    [[vk::location(9)]] nointerpolation float2 g_start : TEXCOORD9;
    [[vk::location(10)]] nointerpolation float2 g_dir : TEXCOORD10;
    [[vk::location(11)]] nointerpolation float4 start_color : TEXCOORD11;
    [[vk::location(12)]] nointerpolation float4 end_color : TEXCOORD12;
    [[vk::location(13)]] nointerpolation float3 g_lens : TEXCOORD13;  // start_len, end_len, hint
    [[vk::location(14)]] nointerpolation uint color_space : TEXCOORD14;
};

[shader("vertex")]
VertexOut vertex(uint vertex_index : SV_VulkanVertexID) {
    let v = pc.vertices[vertex_index];
    VertexOut o;
    // bevy_ui positions are physical pixels with y down; Vulkan NDC is y down too.
    let ndc = float2(v.position_x, v.position_y) / pc.screen_size * 2.0 - 1.0;
    o.position = float4(ndc, 0.0, 1.0);
    o.uv = float2(v.uv_x, v.uv_y);
    o.color = float4(v.color_r, v.color_g, v.color_b, v.color_a);
    o.flags = v.flags;
    o.tex = v.tex_index;
    o.radius_x = float4(v.radius_x0, v.radius_x1, v.radius_x2, v.radius_x3);
    o.radius_y = float4(v.radius_y0, v.radius_y1, v.radius_y2, v.radius_y3);
    o.border = float4(v.border_l, v.border_t, v.border_r, v.border_b);
    o.size = float2(v.size_x, v.size_y);
    o.point = float2(v.point_x, v.point_y);
    o.g_start = float2(v.g_start_x, v.g_start_y);
    o.g_dir = float2(v.g_dir_x, v.g_dir_y);
    o.start_color = float4(v.start_color_r, v.start_color_g, v.start_color_b, v.start_color_a);
    o.end_color = float4(v.end_color_r, v.end_color_g, v.end_color_b, v.end_color_a);
    o.g_lens = float3(v.start_len, v.end_len, v.hint);
    o.color_space = v.color_space;
    return o;
}

static const uint TEXTURED = 1u;
// must align with `ui_render::shader_flags`
static const uint RADIAL = 16u;
static const uint FILL_START = 32u;
static const uint FILL_END = 64u;
static const uint CONIC = 128u;
static const uint GRADIENT = 8192u;
static const uint BORDER_LEFT = 256u;
static const uint BORDER_TOP = 512u;
static const uint BORDER_RIGHT = 1024u;
static const uint BORDER_BOTTOM = 2048u;
static const uint BORDER_ANY = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;
static const uint INVERT = 4096u;
// The fill is a procedural pattern (feathers' material fills): kind in `color_space`, params in
// `start_color`.
static const uint PATTERN = 16384u;

bool enabled(uint flags, uint mask) {
    return (flags & mask) != 0u;
}

// One iteration of Newton's method on the 2D equation of an ellipse
// (G. Taubin, "Distance Approximations for Rasterizing Implicit Curves", §3).
float distance_to_ellipse_approx(float2 p, float2 inv_radii_sq, float scale) {
    let p_r = p * inv_radii_sq;
    let g = dot(p, p_r) - scale;
    let dG = (1.0 + scale) * p_r;
    return g * rsqrt(dot(dG, dG));
}

// Radius of the corner closest to `point`. Radii ordered x: top left, y: top right,
// z: bottom right, w: bottom left.
float2 select_corner_radius(float2 point, float4 rx, float4 ry) {
    let rxs = 0.0 < point.y ? rx.wz : rx.xy;
    let rys = 0.0 < point.y ? ry.wz : ry.xy;
    return float2(0.0 < point.x ? rxs.y : rxs.x, 0.0 < point.x ? rys.y : rys.x);
}

// Signed distance from `point` to the boundary of the rounded box (negative inside).
float sd_rounded_box(float2 point, float2 size, float4 rx, float4 ry) {
    let radius = select_corner_radius(point, rx, ry);
    let corner_to_point = abs(point) - 0.5 * size;
    let straight_distance = max(corner_to_point.x, corner_to_point.y);
    if (min(radius.x, radius.y) <= 0.0) {
        return straight_distance;
    }
    let q = corner_to_point + radius;
    let edge_distance = max(q.x - radius.x, q.y - radius.y);
    let inv_radii_sq = 1.0 / (radius * radius);
    let corner_distance = distance_to_ellipse_approx(q, inv_radii_sq, 1.0);
    return (q.x > 0.0 && q.y > 0.0) ? corner_distance : edge_distance;
}

float sd_inset_rounded_box(float2 point, float2 size, float4 radius_x, float4 radius_y, float4 inset) {
    let inner_size = size - inset.xy - inset.zw;
    let inner_center = inset.xy + 0.5 * inner_size - 0.5 * size;
    let inner_point = point - inner_center;

    var rx = radius_x;
    var ry = radius_y;

    // top left
    rx.x = rx.x - inset.x;
    ry.x = ry.x - inset.y;
    // top right
    rx.y = rx.y - inset.z;
    ry.y = ry.y - inset.y;
    // bottom right
    rx.z = rx.z - inset.z;
    ry.z = ry.z - inset.w;
    // bottom left
    rx.w = rx.w - inset.x;
    ry.w = ry.w - inset.w;

    let half_size = inner_size * 0.5;

    rx = min(max(rx, float4(0.0)), float4(half_size.x));
    ry = min(max(ry, float4(0.0)), float4(half_size.y));
    let m = min(rx, ry);
    let is_zero_radius = float4(m.x <= 0.0 ? 1.0 : 0.0, m.y <= 0.0 ? 1.0 : 0.0,
                                m.z <= 0.0 ? 1.0 : 0.0, m.w <= 0.0 ? 1.0 : 0.0);
    rx = lerp(rx, float4(0.0), is_zero_radius);
    ry = lerp(ry, float4(0.0), is_zero_radius);

    return sd_rounded_box(inner_point, inner_size, rx, ry);
}

bool nearest_border_active(float2 point_vs_mid, float2 size, float4 width, uint flags) {
    if ((flags & BORDER_ANY) == BORDER_ANY) {
        return true;
    }
    let point = clamp(point_vs_mid + size * 0.49999, float2(0.0), size);
    let left = point.x / width.x;
    let top = point.y / width.y;
    let right = (size.x - point.x) / width.z;
    let bottom = (size.y - point.y) / width.w;
    let min_dist = min(min(left, top), min(right, bottom));
    return (enabled(flags, BORDER_LEFT) && min_dist == left) ||
           (enabled(flags, BORDER_TOP) && min_dist == top) ||
           (enabled(flags, BORDER_RIGHT) && min_dist == right) ||
           (enabled(flags, BORDER_BOTTOM) && min_dist == bottom);
}

float antialias(float dist) {
    return clamp(0.5 - dist, 0.0, 1.0);
}

float4 draw_uinode_border(float4 color, float2 point, float2 size, float4 rx, float4 ry, float4 border, uint flags) {
    let external_distance = sd_rounded_box(point, size, rx, ry);
    let internal_distance = sd_inset_rounded_box(point, size, rx, ry, border);
    let border_distance = max(external_distance, -internal_distance);
    let nearest_border = nearest_border_active(point, size, border, flags) ? 1.0 : 0.0;
    // Only anti-alias where a non-zero width border is present.
    let t = external_distance < internal_distance
        ? antialias(border_distance)
        : 1.0 - step(0.0, border_distance);
    return float4(color.rgb, clamp(color.a * t * nearest_border, 0.0, 1.0));
}

float4 draw_uinode_background(float4 color, float2 point, float2 size, float4 rx, float4 ry, float4 border, uint flags) {
    let internal_distance = sd_inset_rounded_box(point, size, rx, ry, border) * (enabled(flags, INVERT) ? -1.0 : 1.0);
    let t = antialias(internal_distance);
    return float4(color.rgb, clamp(color.a * t, 0.0, 1.0));
}

// ---- gradients (port of gradient.wesl) ----

// must align with `ui_render::color_space_index`
static const uint CS_LINEAR_RGBA = 0u;
static const uint CS_SRGBA = 1u;
static const uint CS_OKLABA = 2u;
static const uint CS_OKLCHA = 3u;
static const uint CS_OKLCHA_LONG = 4u;
static const uint CS_OKHSLA = 5u;
static const uint CS_OKHSLA_LONG = 6u;
static const uint CS_HSLA = 7u;
static const uint CS_HSLA_LONG = 8u;
static const uint CS_HSVA = 9u;
static const uint CS_HSVA_LONG = 10u;

// Distance in gradient space from the start of the gradient to `point`.
float linear_distance(float2 point, float2 g_start, float2 g_dir) {
    return dot(point - g_start, g_dir);
}

float radial_distance(float2 point, float2 center, float ratio) {
    let d = point - center;
    return length(float2(d.x, d.y * ratio));
}

float conic_distance(float start, float2 point, float2 center) {
    let d = point - center;
    let angle = atan2(-d.x, d.y) + UI_PI;
    return glsl_mod(glsl_mod(angle - start, UI_PI_2) + UI_PI_2, UI_PI_2);
}

// Mix in the interpolation color space.
float3 mix_colors(float3 a, float3 b, float t, uint space) {
    switch (space) {
        case CS_OKLCHA: return mix_oklch(a, b, t);
        case CS_OKLCHA_LONG: return mix_oklch_long(a, b, t);
        case CS_HSVA: return mix_hsv(a, b, t);
        case CS_HSVA_LONG: return mix_hsv_long(a, b, t);
        case CS_HSLA: return mix_hue_short(a, b, t);
        case CS_HSLA_LONG: return mix_hue_long(a, b, t);
        case CS_OKHSLA: return mix_hue_short(a, b, t);
        case CS_OKHSLA_LONG: return mix_hue_long(a, b, t);
        // linear rgba, oklab and srgba just lerp
        default: return lerp(a, b, t);
    }
}

// Convert from the interpolation color space to linear rgba.
float4 convert_to_linear_rgba(float4 color, uint space) {
    float3 rgb;
    switch (space) {
        case CS_OKLCHA: case CS_OKLCHA_LONG: rgb = oklch_to_linear_rgb(color.xyz); break;
        case CS_OKHSLA: case CS_OKHSLA_LONG: rgb = okhsl_to_linear_rgb(color.xyz); break;
        case CS_HSVA: case CS_HSVA_LONG: rgb = hsv_to_linear_rgb(color.xyz); break;
        case CS_HSLA: case CS_HSLA_LONG: rgb = hsl_to_linear_rgb(color.xyz); break;
        case CS_OKLABA: rgb = oklab_to_linear_rgb(color.xyz); break;
        case CS_SRGBA: rgb = srgb_to_linear_rgb(color.xyz); break;
        default: rgb = color.rgb; break;
    }
    return float4(rgb, color.a);
}

float4 interpolate_gradient(
    float dist, float4 start_color, float start_distance, float4 end_color, float end_distance,
    float hint, uint flags, uint space
) {
    if (start_distance == end_distance) {
        if (dist <= start_distance && enabled(flags, FILL_START)) {
            return convert_to_linear_rgba(start_color, space);
        }
        if (start_distance <= dist && enabled(flags, FILL_END)) {
            return convert_to_linear_rgba(end_color, space);
        }
        return float4(0.0);
    }

    var t = (dist - start_distance) / (end_distance - start_distance);

    if (t < 0.0) {
        if (enabled(flags, FILL_START)) {
            return convert_to_linear_rgba(start_color, space);
        }
        return float4(0.0);
    }
    if (1.0 < t) {
        if (enabled(flags, FILL_END)) {
            return convert_to_linear_rgba(end_color, space);
        }
        return float4(0.0);
    }

    if (t < hint) {
        t = 0.5 * t / hint;
    } else {
        t = 0.5 * (1.0 + (t - hint) / (1.0 - hint));
    }

    return convert_to_linear_rgba(
        float4(mix_colors(start_color.xyz, end_color.xyz, t, space), lerp(start_color.a, end_color.a, t)),
        space
    );
}

// ---- patterns (ports of feathers' alpha_pattern / color_plane / color_wheel materials) ----

// must align with `ui_render::pattern`
static const uint PAT_CHECKER = 0u;
static const uint PAT_RG = 1u;
static const uint PAT_RB = 2u;
static const uint PAT_GB = 3u;
static const uint PAT_HS = 4u;
static const uint PAT_HL = 5u;
static const uint PAT_OKHS = 6u;
static const uint PAT_OKHL = 7u;
static const uint PAT_WHEEL = 8u;

// Must match `color_wheel.rs`.
static const float RING_WIDTH = 12.0;
static const float WHEEL_SPACING = 4.0;
static const float MIN_DIAMETER = 100.0 - 2.0 * 4.0;

float3 hwb_to_linear_rgb(float3 hwb) {
    let v = 1.0 - hwb.z;
    let s = v != 0.0 ? 1.0 - hwb.y / v : 0.0;
    return hsv_to_linear_rgb(float3(hwb.x, s, v));
}

float cross_2d(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

// Positive on the left of a -> b.
float line_distance(float2 p, float2 a, float2 b) {
    return cross_2d(b - a, p - a) / distance(a, b);
}

// Hue ring plus the whiteness/blackness triangle; `hue` in degrees, `scale` physical/logical.
float4 color_wheel(float2 uv, float2 size, float hue, float scale) {
    let min_side = max(min(size.x, size.y), MIN_DIAMETER * scale);
    let centered = (uv - float2(0.5)) / (min_side / size);
    let radial = length(centered);
    let aa = fwidth(radial);

    let inner_radius = 0.5 - RING_WIDTH * scale / min_side;
    let ring_hue = frac(atan2(centered.y, centered.x) / UI_PI_2);
    let ring_alpha = smoothstep(inner_radius - aa, inner_radius, radial)
        - smoothstep(0.5 - aa, 0.5, radial);

    let triangle_radius = 0.5 - (RING_WIDTH + 2.0 * WHEEL_SPACING) * scale / min_side;
    let hue_angle = radians(hue);
    let hue_point = float2(cos(hue_angle), sin(hue_angle)) * triangle_radius;
    let white_angle = hue_angle + UI_PI_2 / 3.0;
    let black_angle = hue_angle - UI_PI_2 / 3.0;
    let white_point = float2(cos(white_angle), sin(white_angle)) * triangle_radius;
    let black_point = float2(cos(black_angle), sin(black_angle)) * triangle_radius;

    let area = cross_2d(white_point - hue_point, black_point - hue_point);
    var whiteness = clamp(cross_2d(centered - hue_point, black_point - hue_point) / area, 0.0, 1.0);
    var blackness = clamp(cross_2d(white_point - hue_point, centered - hue_point) / area, 0.0, 1.0);
    let wb = whiteness + blackness;
    if (wb > 1.0) {
        whiteness /= wb;
        blackness /= wb;
    }

    let triangle_sd = min(
        line_distance(centered, hue_point, white_point),
        min(line_distance(centered, white_point, black_point), line_distance(centered, black_point, hue_point))
    );
    let triangle_alpha = smoothstep(-fwidth(triangle_sd), 0.0, triangle_sd);

    let alpha = min(ring_alpha + triangle_alpha, 1.0);
    let ring_color = hwb_to_linear_rgb(float3(ring_hue, 0.0, 0.0));
    let triangle_color = hwb_to_linear_rgb(float3(frac(hue / 360.0), whiteness, blackness));
    let color = (ring_color * ring_alpha + triangle_color * triangle_alpha) / max(alpha, 0.001);
    return float4(color, alpha);
}

float4 draw_pattern(uint kind, float4 params, float2 point, float2 size) {
    let uv = point / size + 0.5;
    let f = params.x;
    switch (kind) {
        case PAT_CHECKER: {
            let c = point / 16.0;
            let check = ((frac(c.x) < 0.5) != (frac(c.y) < 0.5)) ? 1.0 : 0.0;
            return float4(lerp(float3(0.2), float3(0.6), check), 1.0);
        }
        case PAT_RG: return float4(srgb_to_linear_rgb(float3(uv.x, 1.0 - uv.y, f)), 1.0);
        case PAT_RB: return float4(srgb_to_linear_rgb(float3(uv.x, f, 1.0 - uv.y)), 1.0);
        case PAT_GB: return float4(srgb_to_linear_rgb(float3(f, uv.x, 1.0 - uv.y)), 1.0);
        case PAT_HS: return float4(hsl_to_linear_rgb(float3(uv.x, 1.0 - uv.y, f)), 1.0);
        case PAT_HL: return float4(hsl_to_linear_rgb(float3(uv.x, f, 1.0 - uv.y)), 1.0);
        case PAT_OKHS: return float4(okhsl_to_linear_rgb(float3(uv.x, 1.0 - uv.y, f)), 1.0);
        case PAT_OKHL: return float4(okhsl_to_linear_rgb(float3(uv.x, f, 1.0 - uv.y)), 1.0);
        case PAT_WHEEL: return color_wheel(uv, size, params.x, params.y);
        default: return float4(1.0, 0.0, 1.0, 1.0);
    }
}

[shader("fragment")]
float4 fragment(VertexOut i) : SV_Target0 {
    float4 color;
    if (enabled(i.flags, PATTERN)) {
        color = draw_pattern(i.color_space, i.start_color, i.point, i.size);
    } else if (enabled(i.flags, GRADIENT)) {
        float g_distance;
        if (enabled(i.flags, RADIAL)) {
            g_distance = radial_distance(i.point, i.g_start, i.g_dir.x);
        } else if (enabled(i.flags, CONIC)) {
            g_distance = conic_distance(i.g_dir.x, i.point, i.g_start);
        } else {
            g_distance = linear_distance(i.point, i.g_start, i.g_dir);
        }
        color = interpolate_gradient(
            g_distance, i.start_color, i.g_lens.x, i.end_color, i.g_lens.y, i.g_lens.z,
            i.flags, i.color_space
        );
    } else {
        color = i.color;
    }

    // Linear light out: the sRGB attachment encodes on store and blends in linear space.
    if (enabled(i.flags, TEXTURED)) {
        // sRGB textures decode on sampling: the texel is linear.
        color *= textures[NonUniformResourceIndex(i.tex)].Sample(i.uv);
    }

    if (enabled(i.flags, BORDER_ANY)) {
        return draw_uinode_border(color, i.point, i.size, i.radius_x, i.radius_y, i.border, i.flags);
    }
    return draw_uinode_background(color, i.point, i.size, i.radius_x, i.radius_y, i.border, i.flags);
}
