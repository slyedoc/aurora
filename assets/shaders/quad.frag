#version 460

#include "types.glsl"

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 out_Color;

layout (set=0, binding=0) uniform sampler2D test;

layout(push_constant, std430) uniform Registers {
  UniformData uniforms;
};

vec3 acesFilm(const vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d ) + e);
}

vec3 tonemapFilmic(const vec3 color) {
	vec3 x = max(vec3(0.0), color - 0.004);
	return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

// Aspect-corrected: 0 at the centre, 1 at the corners, so it is the same shape at any
// window size; `strength` (0 = off) is how dark the corners get.
vec3 applyVignette(vec3 color, float strength) {
    if (strength <= 0.0) { return color; }
    vec2 size = vec2(textureSize(test, 0));
    vec2 half_extent = vec2(size.x / size.y, 1.0) * 0.5;
    float dist = length((in_UV - 0.5) * vec2(size.x / size.y, 1.0)) / length(half_extent);
    float falloff = smoothstep(1.0, 0.45, dist);
    return mix(color, color * falloff, strength);
}

void main() {
  // The DLSS output: resolved, already-exposed linear HDR (the raygen applies the metered
  // exposure before reconstruction). Tonemap, then encode for the display (gamma last).
  vec3 color = texture(test, in_UV).rgb;
  color = acesFilm(color);
  color = pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(1.0/uniforms.gamma));
  color = applyVignette(color, uniforms.vignette);

  out_Color = vec4(color, 1.0);
}
