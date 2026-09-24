#version 460
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

// Must match `gizmo_render::GizmoVertex` (repr(C), 20 bytes).
struct GizmoVertex {
  vec3 position;     // world space
  uint color;        // packed rgba8 (r | g<<8 | b<<16 | a<<24), linear
  float depth_bias;  // bevy's GizmoConfig::depth_bias: -1 always in front .. 1 always behind
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly restrict buffer GizmoVertices {
  GizmoVertex v[];
};

// Must match `gizmo_render::GizmoPushConstants`.
layout(push_constant, scalar) uniform Registers {
  mat4 view_proj;  // unjittered clip-from-world
  GizmoVertices vertices;
  vec2 inv_extent;  // 1 / swapchain size
} pc;

layout(location = 0) out vec4 out_color;
// Linear view depth with the bias applied, and whether the line takes part in the depth
// test at all.
layout(location = 1) out float out_depth;
layout(location = 2) flat out uint out_tested;

void main() {
  GizmoVertex v = pc.vertices.v[gl_VertexIndex];
  // The hardware clipper handles segments crossing the near plane; nothing to do here.
  gl_Position = pc.view_proj * vec4(v.position, 1.0);
  // A perspective projection's clip w IS the linear view depth the raygen's depth guide
  // stores. The bias scales it: towards the eye below 0 (0 at -1), to infinity at 1.
  const float bias = clamp(v.depth_bias, -1.0, 1.0);
  out_depth = bias <= 0.0 ? gl_Position.w * (1.0 + bias) : gl_Position.w / max(1.0 - bias, 1.0e-6);
  out_tested = bias > -1.0 ? 1u : 0u;
  // The projection is GL-style (clip +Y up, matching the raygen's inverted pixel mapping);
  // Vulkan rasterization maps clip +Y down, so mirror to land on the traced scene.
  gl_Position.y = -gl_Position.y;
  out_color = unpackUnorm4x8(v.color);
}
