#version 460
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

#include "ui_types.glsl"

layout(push_constant, scalar) uniform Registers {
  UiPushConstants pc;
};

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;
layout(location = 2) flat out uint out_flags;
layout(location = 3) flat out uint out_tex;
layout(location = 4) flat out vec4 out_radius_x;
layout(location = 5) flat out vec4 out_radius_y;
layout(location = 6) flat out vec4 out_border;
layout(location = 7) flat out vec2 out_size;
layout(location = 8) out vec2 out_point;
layout(location = 9) flat out vec2 out_g_start;
layout(location = 10) flat out vec2 out_g_dir;
layout(location = 11) flat out vec4 out_start_color;
layout(location = 12) flat out vec4 out_end_color;
layout(location = 13) flat out vec3 out_g_lens;   // start_len, end_len, hint
layout(location = 14) flat out uint out_color_space;

void main() {
  UiVertex v = pc.vertices.v[gl_VertexIndex];
  // bevy_ui positions are physical pixels with y down; Vulkan NDC is y down too.
  vec2 ndc = v.position / pc.screen_size * 2.0 - 1.0;
  gl_Position = vec4(ndc, 0.0, 1.0);
  out_uv = v.uv;
  out_color = v.color;
  out_flags = v.flags;
  out_tex = v.tex_index;
  out_radius_x = v.radius_x;
  out_radius_y = v.radius_y;
  out_border = v.border;
  out_size = v.size;
  out_point = v.point;
  out_g_start = v.g_start;
  out_g_dir = v.g_dir;
  out_start_color = v.start_color;
  out_end_color = v.end_color;
  out_g_lens = vec3(v.start_len, v.end_len, v.hint);
  out_color_space = v.color_space;
}
