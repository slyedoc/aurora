#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Port of bevy_ui_render's `ui.wesl` + `gradient.wesl`: rounded-box SDF backgrounds and
// borders, textured quads (images, glyph atlases) through this crate's bindless set, and
// linear / radial / conic gradient segments interpolated in a per-vertex color space.

#include "ui_color.glsl"

layout(set = 0, binding = 200) uniform sampler2D textures[];

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;
layout(location = 2) flat in uint in_flags;
layout(location = 3) flat in uint in_tex;
layout(location = 4) flat in vec4 in_radius_x;
layout(location = 5) flat in vec4 in_radius_y;
layout(location = 6) flat in vec4 in_border;
layout(location = 7) flat in vec2 in_size;
layout(location = 8) in vec2 in_point;
layout(location = 9) flat in vec2 in_g_start;
layout(location = 10) flat in vec2 in_g_dir;
layout(location = 11) flat in vec4 in_start_color;
layout(location = 12) flat in vec4 in_end_color;
layout(location = 13) flat in vec3 in_g_lens;   // start_len, end_len, hint
layout(location = 14) flat in uint in_color_space;

layout(location = 0) out vec4 out_color;

const uint TEXTURED = 1u;
// must align with `ui_render::shader_flags`
const uint RADIAL = 16u;
const uint FILL_START = 32u;
const uint FILL_END = 64u;
const uint CONIC = 128u;
const uint GRADIENT = 8192u;
const uint BORDER_LEFT = 256u;
const uint BORDER_TOP = 512u;
const uint BORDER_RIGHT = 1024u;
const uint BORDER_BOTTOM = 2048u;
const uint BORDER_ANY = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;
const uint INVERT = 4096u;

bool enabled(uint flags, uint mask) {
  return (flags & mask) != 0u;
}

// One iteration of Newton's method on the 2D equation of an ellipse
// (G. Taubin, "Distance Approximations for Rasterizing Implicit Curves", §3).
float distance_to_ellipse_approx(vec2 p, vec2 inv_radii_sq, float scale) {
  vec2 p_r = p * inv_radii_sq;
  float g = dot(p, p_r) - scale;
  vec2 dG = (1.0 + scale) * p_r;
  return g * inversesqrt(dot(dG, dG));
}

// Radius of the corner closest to `point`. Radii ordered x: top left, y: top right,
// z: bottom right, w: bottom left.
vec2 select_corner_radius(vec2 point, vec4 rx, vec4 ry) {
  vec2 rxs = 0.0 < point.y ? rx.wz : rx.xy;
  vec2 rys = 0.0 < point.y ? ry.wz : ry.xy;
  return vec2(0.0 < point.x ? rxs.y : rxs.x, 0.0 < point.x ? rys.y : rys.x);
}

// Signed distance from `point` to the boundary of the rounded box (negative inside).
float sd_rounded_box(vec2 point, vec2 size, vec4 rx, vec4 ry) {
  vec2 radius = select_corner_radius(point, rx, ry);
  vec2 corner_to_point = abs(point) - 0.5 * size;
  float straight_distance = max(corner_to_point.x, corner_to_point.y);
  if (min(radius.x, radius.y) <= 0.0) {
    return straight_distance;
  }
  vec2 q = corner_to_point + radius;
  float edge_distance = max(q.x - radius.x, q.y - radius.y);
  vec2 inv_radii_sq = 1.0 / (radius * radius);
  float corner_distance = distance_to_ellipse_approx(q, inv_radii_sq, 1.0);
  return (q.x > 0.0 && q.y > 0.0) ? corner_distance : edge_distance;
}

float sd_inset_rounded_box(vec2 point, vec2 size, vec4 radius_x, vec4 radius_y, vec4 inset) {
  vec2 inner_size = size - inset.xy - inset.zw;
  vec2 inner_center = inset.xy + 0.5 * inner_size - 0.5 * size;
  vec2 inner_point = point - inner_center;

  vec4 rx = radius_x;
  vec4 ry = radius_y;

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

  vec2 half_size = inner_size * 0.5;

  rx = min(max(rx, vec4(0.0)), vec4(half_size.x));
  ry = min(max(ry, vec4(0.0)), vec4(half_size.y));
  vec4 is_zero_radius = vec4(lessThanEqual(min(rx, ry), vec4(0.0)));
  rx = mix(rx, vec4(0.0), is_zero_radius);
  ry = mix(ry, vec4(0.0), is_zero_radius);

  return sd_rounded_box(inner_point, inner_size, rx, ry);
}

bool nearest_border_active(vec2 point_vs_mid, vec2 size, vec4 width, uint flags) {
  if ((flags & BORDER_ANY) == BORDER_ANY) {
    return true;
  }
  vec2 point = clamp(point_vs_mid + size * 0.49999, vec2(0.0), size);
  float left = point.x / width.x;
  float top = point.y / width.y;
  float right = (size.x - point.x) / width.z;
  float bottom = (size.y - point.y) / width.w;
  float min_dist = min(min(left, top), min(right, bottom));
  return (enabled(flags, BORDER_LEFT) && min_dist == left) ||
         (enabled(flags, BORDER_TOP) && min_dist == top) ||
         (enabled(flags, BORDER_RIGHT) && min_dist == right) ||
         (enabled(flags, BORDER_BOTTOM) && min_dist == bottom);
}

float antialias(float dist) {
  return clamp(0.5 - dist, 0.0, 1.0);
}

vec4 draw_uinode_border(vec4 color, vec2 point, vec2 size, vec4 rx, vec4 ry, vec4 border, uint flags) {
  float external_distance = sd_rounded_box(point, size, rx, ry);
  float internal_distance = sd_inset_rounded_box(point, size, rx, ry, border);
  float border_distance = max(external_distance, -internal_distance);
  float nearest_border = nearest_border_active(point, size, border, flags) ? 1.0 : 0.0;
  // Only anti-alias where a non-zero width border is present.
  float t = external_distance < internal_distance
      ? antialias(border_distance)
      : 1.0 - step(0.0, border_distance);
  return vec4(color.rgb, clamp(color.a * t * nearest_border, 0.0, 1.0));
}

vec4 draw_uinode_background(vec4 color, vec2 point, vec2 size, vec4 rx, vec4 ry, vec4 border, uint flags) {
  float internal_distance = sd_inset_rounded_box(point, size, rx, ry, border) * (enabled(flags, INVERT) ? -1.0 : 1.0);
  float t = antialias(internal_distance);
  return vec4(color.rgb, clamp(color.a * t, 0.0, 1.0));
}

// ---- gradients (port of gradient.wesl) ----

// must align with `ui_render::color_space_index`
const uint CS_LINEAR_RGBA = 0u;
const uint CS_SRGBA = 1u;
const uint CS_OKLABA = 2u;
const uint CS_OKLCHA = 3u;
const uint CS_OKLCHA_LONG = 4u;
const uint CS_OKHSLA = 5u;
const uint CS_OKHSLA_LONG = 6u;
const uint CS_HSLA = 7u;
const uint CS_HSLA_LONG = 8u;
const uint CS_HSVA = 9u;
const uint CS_HSVA_LONG = 10u;

// Distance in gradient space from the start of the gradient to `point`.
float linear_distance(vec2 point, vec2 g_start, vec2 g_dir) {
  return dot(point - g_start, g_dir);
}

float radial_distance(vec2 point, vec2 center, float ratio) {
  vec2 d = point - center;
  return length(vec2(d.x, d.y * ratio));
}

float conic_distance(float start, vec2 point, vec2 center) {
  vec2 d = point - center;
  float angle = atan(-d.x, d.y) + UI_PI;
  return mod(mod(angle - start, UI_PI_2) + UI_PI_2, UI_PI_2);
}

// Mix in the interpolation color space.
vec3 mix_colors(vec3 a, vec3 b, float t, uint space) {
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
    default: return mix(a, b, t);
  }
}

// Convert from the interpolation color space to linear rgba.
vec4 convert_to_linear_rgba(vec4 color, uint space) {
  vec3 rgb;
  switch (space) {
    case CS_OKLCHA: case CS_OKLCHA_LONG: rgb = oklch_to_linear_rgb(color.xyz); break;
    case CS_OKHSLA: case CS_OKHSLA_LONG: rgb = okhsl_to_linear_rgb(color.xyz); break;
    case CS_HSVA: case CS_HSVA_LONG: rgb = hsv_to_linear_rgb(color.xyz); break;
    case CS_HSLA: case CS_HSLA_LONG: rgb = hsl_to_linear_rgb(color.xyz); break;
    case CS_OKLABA: rgb = oklab_to_linear_rgb(color.xyz); break;
    case CS_SRGBA: rgb = srgb_to_linear_rgb(color.xyz); break;
    default: rgb = color.rgb; break;
  }
  return vec4(rgb, color.a);
}

vec4 interpolate_gradient(
  float dist, vec4 start_color, float start_distance, vec4 end_color, float end_distance,
  float hint, uint flags, uint space
) {
  if (start_distance == end_distance) {
    if (dist <= start_distance && enabled(flags, FILL_START)) {
      return convert_to_linear_rgba(start_color, space);
    }
    if (start_distance <= dist && enabled(flags, FILL_END)) {
      return convert_to_linear_rgba(end_color, space);
    }
    return vec4(0.0);
  }

  float t = (dist - start_distance) / (end_distance - start_distance);

  if (t < 0.0) {
    if (enabled(flags, FILL_START)) {
      return convert_to_linear_rgba(start_color, space);
    }
    return vec4(0.0);
  }
  if (1.0 < t) {
    if (enabled(flags, FILL_END)) {
      return convert_to_linear_rgba(end_color, space);
    }
    return vec4(0.0);
  }

  if (t < hint) {
    t = 0.5 * t / hint;
  } else {
    t = 0.5 * (1.0 + (t - hint) / (1.0 - hint));
  }

  return convert_to_linear_rgba(
    vec4(mix_colors(start_color.xyz, end_color.xyz, t, space), mix(start_color.a, end_color.a, t)),
    space
  );
}

void main() {
  vec4 color;
  if (enabled(in_flags, GRADIENT)) {
    float g_distance;
    if (enabled(in_flags, RADIAL)) {
      g_distance = radial_distance(in_point, in_g_start, in_g_dir.x);
    } else if (enabled(in_flags, CONIC)) {
      g_distance = conic_distance(in_g_dir.x, in_point, in_g_start);
    } else {
      g_distance = linear_distance(in_point, in_g_start, in_g_dir);
    }
    color = interpolate_gradient(
      g_distance, in_start_color, in_g_lens.x, in_end_color, in_g_lens.y, in_g_lens.z,
      in_flags, in_color_space
    );
  } else {
    color = in_color;
  }

  // The swapchain is UNORM, so linear colors from bevy get encoded here.
  color = vec4(linear_rgb_to_srgb(color.rgb), color.a);
  if (enabled(in_flags, TEXTURED)) {
    // Image bytes are already sRGB encoded (loaded as UNORM); glyph atlases are
    // white alpha masks. Apply them after encoding the tint.
    color *= texture(textures[nonuniformEXT(in_tex)], in_uv);
  }

  if (enabled(in_flags, BORDER_ANY)) {
    out_color = draw_uinode_border(color, in_point, in_size, in_radius_x, in_radius_y, in_border, in_flags);
  } else {
    out_color = draw_uinode_background(color, in_point, in_size, in_radius_x, in_radius_y, in_border, in_flags);
  }
}
