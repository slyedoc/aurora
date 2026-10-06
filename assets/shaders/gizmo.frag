#version 460
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly restrict buffer GizmoVertices {
  uint unused;
};

// Must match `gizmo_render::GizmoPushConstants`.
layout(push_constant, scalar) uniform Registers {
  mat4 view_proj;
  GizmoVertices vertices;
  vec2 inv_extent;  // 1 / target size
  uint depth_guide;
} pc;

// The raygen's linear view depth guide (render resolution, jittered; the sky is far away).
layout(set = 0, binding = 0) uniform sampler2D scene_depth;

layout(location = 0) in vec4 in_color;
layout(location = 1) in float in_depth;
layout(location = 2) flat in uint in_tested;
layout(location = 3) in float in_across;
layout(location = 4) flat in float in_half_width;
layout(location = 0) out vec4 out_color;

void main() {
  // Coverage across the line: full inside its width, falling off over the last pixel.
  float alpha = clamp(in_half_width + 0.5 - abs(in_across), 0.0, 1.0);
  if (in_tested != 0u && pc.depth_guide != 0u) {
    // There is no depth attachment -- the scene is traced -- so the test is done here. The
    // guide is coarser than the target and shifts with the sub-pixel jitter, so the line is
    // tested against each of the four texels around the fragment and keeps the share it is in
    // front of: a silhouette fades over one texel instead of blinking. The slack is how far a
    // surface's depth runs across those texels (a line lying ON a grazing surface), capped so
    // a near silhouette's spread cannot let what is behind it through.
    const vec4 depths = textureGather(scene_depth, gl_FragCoord.xy * pc.inv_extent, 0);
    const float far = max(max(depths.x, depths.y), max(depths.z, depths.w));
    const float near = min(min(depths.x, depths.y), min(depths.z, depths.w));
    const vec4 slack = min(vec4(far - near), depths * 0.05) + depths * 0.01 + 0.02;
    const vec4 passes = step(vec4(in_depth), depths + slack);
    alpha *= dot(passes, vec4(0.25));
  }
  // Linear light: the sRGB attachment encodes on store and blends in linear space.
  out_color = vec4(in_color.rgb, in_color.a * alpha);
}
