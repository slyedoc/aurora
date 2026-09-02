#version 460
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

// Must match `gizmo_render::GizmoVertex` (repr(C), 16 bytes).
struct GizmoVertex {
  vec3 position;  // world space
  uint color;     // packed rgba8 (r | g<<8 | b<<16 | a<<24), linear
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly restrict buffer GizmoVertices {
  GizmoVertex v[];
};

// Must match `gizmo_render::GizmoPushConstants`.
layout(push_constant, scalar) uniform Registers {
  mat4 view_proj;  // unjittered clip-from-world
  GizmoVertices vertices;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
  GizmoVertex v = pc.vertices.v[gl_VertexIndex];
  // The hardware clipper handles segments crossing the near plane; nothing to do here.
  gl_Position = pc.view_proj * vec4(v.position, 1.0);
  // The projection is GL-style (clip +Y up, matching the raygen's inverted pixel mapping);
  // Vulkan rasterization maps clip +Y down, so mirror to land on the traced scene.
  gl_Position.y = -gl_Position.y;
  out_color = unpackUnorm4x8(v.color);
}
