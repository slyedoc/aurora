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

// Scene luminance in NITS, false-coloured against the authoring reference bands in
// aurora_files/lighting_units.md. One decade per band, with the boundary darkened so the
// decades are countable rather than a smooth ramp you have to eyeball.
//
//   blue   <1        deep shadow          yellow  1e3..1e4  signage / overcast sky
//   cyan    1..10    dim interior         orange  1e4..1e5  LUMINAIRE surface
//   green  10..1e2                        red     1e5..1e6
//   lime   1e2..1e3  screens              white   >1e6      sun disc
vec3 nitsFalseColor(float nits) {
  const float t = log2(max(nits, 1.0e-4)) / log2(10.0);   // log10 nits
  vec3 c;
  if      (t < 0.0) c = mix(vec3(0.02, 0.02, 0.10), vec3(0.15, 0.15, 0.65), clamp(t + 1.0, 0.0, 1.0));
  else if (t < 1.0) c = mix(vec3(0.15, 0.15, 0.65), vec3(0.00, 0.60, 0.90), t);
  else if (t < 2.0) c = mix(vec3(0.00, 0.60, 0.90), vec3(0.00, 0.85, 0.40), t - 1.0);
  else if (t < 3.0) c = mix(vec3(0.00, 0.85, 0.40), vec3(0.85, 0.95, 0.00), t - 2.0);
  else if (t < 4.0) c = mix(vec3(0.85, 0.95, 0.00), vec3(1.00, 0.60, 0.00), t - 3.0);
  else if (t < 5.0) c = mix(vec3(1.00, 0.60, 0.00), vec3(1.00, 0.20, 0.05), t - 4.0);
  else if (t < 6.0) c = mix(vec3(1.00, 0.20, 0.05), vec3(1.00, 0.00, 0.60), t - 5.0);
  else              c = mix(vec3(1.00, 0.00, 0.60), vec3(1.00, 1.00, 1.00), clamp(t - 6.0, 0.0, 1.0));

  const float f = fract(t);
  const float edge = smoothstep(0.0, 0.035, f) * smoothstep(0.0, 0.035, 1.0 - f);
  return c * (0.4 + 0.6 * edge);
}

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
  // Luminance: the one view that must UNDO the exposure. The colour guide is what Ray
  // Reconstruction ingests, pre-exposed at the quantised `input_exposure`; dividing that
  // back out recovers the scene's physical radiance in nits, independent of the look and
  // of whatever the metering is currently doing. That separation is the whole point --
  // "emitter authored too dim" and "exposure keyed to something bright" are different
  // bugs that look the same through a tonemapper.
  if (debug_view == 9) {
    const vec3 pre = texture(test[1], in_UV).rgb;
    const float nits = dot(pre, vec3(0.2126, 0.7152, 0.0722)) / max(ae.input_exposure, 1.0e-12);
    out_Color = vec4(nitsFalseColor(nits), 1.0);
    return;
  }

  // Data views: raw encodings, no exposure or tonemap -- a static scene shows a static
  // image (matches AuroraDebugView's variant order).
  if (debug_view >= 2) {
    vec3 v;
    switch (debug_view) {
      case 2: v = texture(test[2], in_UV).rgb * 0.5 + 0.5; break;             // normals
      case 3: v = vec3(texture(test[2], in_UV).a); break;                     // roughness
      case 4: v = texture(test[3], in_UV).rgb; break;                        // diffuse
      case 5: v = texture(test[4], in_UV).rgb; break;                        // specular
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
  // it cannot disturb RR's history. Tonemap to display-linear; the sRGB attachment applies
  // the transfer function on store.
  const float look = display_exposure > 0.0 ? display_exposure : ae.exposure;
  vec3 color = texture(test[debug_view], in_UV).rgb * (look / max(ae.input_exposure, 1.0e-12));
  color = clamp(acesFilm(color), vec3(0.0), vec3(1.0));
  color = applyVignette(color, uniforms.vignette);

  out_Color = vec4(color, 1.0);
}
