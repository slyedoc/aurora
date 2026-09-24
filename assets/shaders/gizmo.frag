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
  vec2 inv_extent;  // 1 / swapchain size
} pc;

// The raygen's linear view depth guide (render resolution, jittered; the sky is far away).
layout(set = 0, binding = 0) uniform sampler2D scene_depth;

layout(location = 0) in vec4 in_color;
layout(location = 1) in float in_depth;
layout(location = 2) flat in uint in_tested;
layout(location = 0) out vec4 out_color;

void main() {
  float visible = 1.0;
  if (in_tested != 0u) {
    // There is no depth attachment -- the scene is traced -- so the test is done here. The
    // guide is coarser than the window and shifts with the sub-pixel jitter, so the test is
    // against the FARTHEST of the four texels around the fragment (a line on a silhouette
    // holds still instead of blinking), with slack equal to their spread: that is how far
    // the surface's depth runs across one texel, which is what a line lying ON a surface
    // seen at a grazing angle is off by.
    const vec4 depths = textureGather(scene_depth, gl_FragCoord.xy * pc.inv_extent, 0);
    const float far = max(max(depths.x, depths.y), max(depths.z, depths.w));
    const float near = min(min(depths.x, depths.y), min(depths.z, depths.w));
    // Hidden = blended at zero alpha (`discard` is SPIR-V demote, a device feature this
    // pass has no other use for).
    visible = in_depth > far + (far - near) + far * 0.01 + 0.02 ? 0.0 : 1.0;
  }
  // Linear light: the sRGB attachment encodes on store and blends in linear space.
  out_color = vec4(in_color.rgb, in_color.a * visible);
}
