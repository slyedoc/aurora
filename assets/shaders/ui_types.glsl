#ifndef H_UI_TYPES
#define H_UI_TYPES

#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

// Must match `ui_render::UiVertex` (repr(C), 168 bytes, all 4-byte fields).
struct UiVertex {
  vec2 position;   // physical pixels, y down
  vec2 uv;
  vec4 color;      // linear rgba
  uint flags;
  uint tex_index;  // bindless texture index (flags & TEXTURED)
  vec4 radius_x;   // x: top left, y: top right, z: bottom right, w: bottom left
  vec4 radius_y;
  vec4 border;     // x: left, y: top, z: right, w: bottom
  vec2 size;
  vec2 point;      // position relative to the center of the rectangle
  // gradient segment (flags & GRADIENT); colors are in `color_space`
  vec2 g_start;
  vec2 g_dir;
  vec4 start_color;
  vec4 end_color;
  float start_len;
  float end_len;
  float hint;
  uint color_space;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) readonly restrict buffer UiVertices {
  UiVertex v[];
};

// Must match `ui_render::UiPushConstants`.
struct UiPushConstants {
  UiVertices vertices;
  vec2 screen_size;
};

#endif
