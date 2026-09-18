#version 460

layout(location = 0) in vec4 in_color;
layout(location = 0) out vec4 out_color;

void main() {
  // Linear light: the sRGB attachment encodes on store and blends in linear space.
  out_color = in_color;
}
