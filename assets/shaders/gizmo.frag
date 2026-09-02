#version 460

#include "ui_color.glsl"

layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;

void main() {
  // The swapchain holds display-encoded values (the post-process applied the OETF), so
  // encode before the straight-alpha blend -- same convention as `ui.frag`.
  out_color = vec4(linear_rgb_to_srgb(in_color.rgb), in_color.a);
}
