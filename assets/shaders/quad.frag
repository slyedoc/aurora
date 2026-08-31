#version 460

#include "types.glsl"

layout(location = 0) in  vec2 in_UV;
layout(location = 0) out vec4 out_Color;

// 0: the DLSS output; 1..=7 the guide images (colour, normals+roughness, diffuse,
// specular, depth, specular hit distance, motion) for the debug views. Slots the renderer
// cannot fill yet alias the output.
layout (set=0, binding=0) uniform sampler2D test[8];

layout(push_constant, std430) uniform Registers {
  UniformData uniforms;
  AeData ae;
  // The look: 0 = follow the metering (Auto); else a fixed linear exposure, applied as a
  // ratio against the metered exposure the raygen baked in.
  float display_exposure;
  // AuroraDebugView (src/debug_view.rs): 0 = the output, else a guide visualisation.
  uint debug_view;
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
    vec2 size = vec2(textureSize(test[0], 0));
    vec2 half_extent = vec2(size.x / size.y, 1.0) * 0.5;
    float dist = length((in_UV - 0.5) * vec2(size.x / size.y, 1.0)) / length(half_extent);
    float falloff = smoothstep(1.0, 0.45, dist);
    return mix(color, color * falloff, strength);
}

void main() {
  // Data views: raw encodings, no exposure or tonemap -- a static scene shows a static
  // image (matches AuroraDebugView's variant order).
  if (debug_view >= 2) {
    vec3 v;
    switch (debug_view) {
      case 2: v = texture(test[2], in_UV).rgb * 0.5 + 0.5; break;             // normals
      case 3: v = vec3(texture(test[2], in_UV).a); break;                     // roughness
      case 4: v = pow(texture(test[3], in_UV).rgb, vec3(1.0 / 2.2)); break;   // diffuse
      case 5: v = pow(texture(test[4], in_UV).rgb, vec3(1.0 / 2.2)); break;   // specular
      case 6: v = vec3(exp2(-texture(test[5], in_UV).r / 32.0)); break;       // depth
      case 7: v = vec3(exp2(-texture(test[6], in_UV).r / 32.0)); break;       // spec hit
      default: v = vec3(texture(test[7], in_UV).rg * 0.1 + 0.5, 0.5); break;  // motion
    }
    out_Color = vec4(v, 1.0);
    return;
  }

  // The DLSS output (or, view 1, its noisy colour input): resolved linear HDR at the
  // quantised input exposure (the raygen keeps RR's input near mid-gray and STILL).
  // Re-expose to the look here -- the smooth metered value (Auto) or a fixed one -- where
  // it cannot disturb RR's history. Tonemap, then encode for the display (gamma last).
  const float look = display_exposure > 0.0 ? display_exposure : ae.exposure;
  vec3 color = texture(test[debug_view], in_UV).rgb * (look / max(ae.input_exposure, 1.0e-12));
  color = acesFilm(color);
  color = pow(clamp(color, vec3(0.0), vec3(1.0)), vec3(1.0/uniforms.gamma));
  color = applyVignette(color, uniforms.vignette);

  out_Color = vec4(color, 1.0);
}
