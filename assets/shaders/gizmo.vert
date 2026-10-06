#version 460
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

// Must match `gizmo_render::GizmoVertex` (repr(C), 24 bytes).
struct GizmoVertex {
  vec3 position;     // world space
  uint color;        // packed rgba8 (r | g<<8 | b<<16 | a<<24), linear
  float depth_bias;  // bevy's GizmoConfig::depth_bias: -1 always in front .. 1 always behind
  float width;       // line width, pixels
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly restrict buffer GizmoVertices {
  GizmoVertex v[];
};

// Must match `gizmo_render::GizmoPushConstants`.
layout(push_constant, scalar) uniform Registers {
  mat4 view_proj;  // unjittered clip-from-world
  GizmoVertices vertices;
  vec2 inv_extent;  // 1 / target size
  uint depth_guide;
} pc;

layout(location = 0) out vec4 out_color;
// Linear view depth with the bias applied, and whether the line takes part in the depth
// test at all.
layout(location = 1) out float out_depth;
layout(location = 2) flat out uint out_tested;
// Signed distance from the line's centre across it, pixels; and the line's half width.
layout(location = 3) out float out_across;
layout(location = 4) flat out float out_half_width;

// One segment = two endpoints in the buffer = six vertices here: a screen-space quad around
// the line, a pixel wider than it on each side for the soft edge.
const float FEATHER = 1.0;

void main() {
  const uint segment = uint(gl_VertexIndex) / 6u;
  const uint corner = uint(gl_VertexIndex) % 6u;
  GizmoVertex a = pc.vertices.v[segment * 2u];
  GizmoVertex b = pc.vertices.v[segment * 2u + 1u];
  vec4 clip_a = pc.view_proj * vec4(a.position, 1.0);
  vec4 clip_b = pc.view_proj * vec4(b.position, 1.0);

  // Clip to just in front of the eye: the screen-space direction is meaningless behind it.
  const float near_w = 1.0e-3;
  if (clip_a.w < near_w && clip_b.w < near_w) {
    gl_Position = vec4(0.0);
    return;
  }
  if (clip_a.w < near_w) {
    clip_a = mix(clip_a, clip_b, (near_w - clip_a.w) / (clip_b.w - clip_a.w));
  } else if (clip_b.w < near_w) {
    clip_b = mix(clip_b, clip_a, (near_w - clip_b.w) / (clip_a.w - clip_b.w));
  }

  // Corners: 0,1,2 / 2,1,3 over (end, side) = (0,-) (0,+) (1,-) (1,+).
  const uint corners[6] = uint[6](0u, 1u, 2u, 2u, 1u, 3u);
  const uint c = corners[corner];
  const bool at_b = c >= 2u;
  const float side = (c & 1u) == 0u ? -1.0 : 1.0;

  const vec2 pixels = 1.0 / pc.inv_extent;
  const vec2 screen_a = clip_a.xy / clip_a.w * pixels;
  const vec2 screen_b = clip_b.xy / clip_b.w * pixels;
  vec2 dir = screen_b - screen_a;
  dir = dot(dir, dir) > 1.0e-8 ? normalize(dir) : vec2(1.0, 0.0);
  const vec2 normal = vec2(-dir.y, dir.x);

  GizmoVertex v = at_b ? b : a;
  vec4 clip = at_b ? clip_b : clip_a;
  const float half_width = max(v.width, 1.0) * 0.5;
  const float reach = half_width + FEATHER;
  // Out along the normal, and past each end by the same reach so the caps are soft too.
  const vec2 offset_px = normal * side * reach + dir * (at_b ? reach : -reach);
  // NDC spans 2 per screen: pixels to NDC is * 2 / size, then back into clip space by w.
  clip.xy += offset_px * 2.0 * pc.inv_extent * clip.w;
  gl_Position = clip;
  out_across = side * reach;
  out_half_width = half_width;

  // A perspective projection's clip w IS the linear view depth the raygen's depth guide
  // stores. The bias scales it: towards the eye below 0 (0 at -1), to infinity at 1.
  const float bias = clamp(v.depth_bias, -1.0, 1.0);
  out_depth = bias <= 0.0 ? clip.w * (1.0 + bias) : clip.w / max(1.0 - bias, 1.0e-6);
  out_tested = bias > -1.0 ? 1u : 0u;
  // The projection is GL-style (clip +Y up, matching the raygen's inverted pixel mapping);
  // Vulkan rasterization maps clip +Y down, so mirror to land on the traced scene.
  gl_Position.y = -gl_Position.y;
  out_color = unpackUnorm4x8(v.color);
}
