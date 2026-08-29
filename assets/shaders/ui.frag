#version 460
#extension GL_EXT_nonuniform_qualifier : enable

// Port of bevy_ui_render's `ui.wesl`: rounded-box SDF backgrounds and borders,
// textured quads (images, glyph atlases) through this crate's bindless set.

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

layout(location = 0) out vec4 out_color;

const uint TEXTURED = 1u;
// must align with `ui_render::shader_flags`
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

// The swapchain is UNORM, so linear colors from bevy get encoded here.
vec3 linear_to_srgb(vec3 c) {
  vec3 lo = 12.92 * c;
  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
  vec4 color = in_color;
  if (enabled(in_flags, TEXTURED)) {
    // Image bytes are already sRGB encoded (loaded as UNORM); glyph atlases are
    // white alpha masks. Apply them after encoding the tint.
    vec4 texel = texture(textures[nonuniformEXT(in_tex)], in_uv);
    color = vec4(linear_to_srgb(color.rgb), color.a) * texel;
  } else {
    color = vec4(linear_to_srgb(color.rgb), color.a);
  }

  if (enabled(in_flags, BORDER_ANY)) {
    out_color = draw_uinode_border(color, in_point, in_size, in_radius_x, in_radius_y, in_border, in_flags);
  } else {
    out_color = draw_uinode_background(color, in_point, in_size, in_radius_x, in_radius_y, in_border, in_flags);
  }
}
