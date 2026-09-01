#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "types.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set=1, binding=200)         uniform sampler2D textures[];

layout(push_constant, std430) uniform Registers {
  PushConstants pc;
};

const float PI = 3.14159265359;

// Equirectangular lookup: texel (linear radiance) times the scale carried in skycolor.
vec3 hdr_sky(const vec3 d) {
  const vec2 uv = env_dir_to_uv(d);
  vec3 texel = texture(textures[pc.skydome], uv).rgb;
  // Without importance sampling a BRDF-sampled ray is the only way to the sun texel, so its
  // extreme values are clamped RELATIVE to the sky scale (a physically bright sky passes, a
  // 1e5x sun does not). With the sampler (env_light.rs) the raygen gathers the sun by
  // next-event estimation and MIS-weights these hits, so the map is used as is.
  if (pc.uniforms.env_w == 0u) { texel = min(texel, vec3(300.0)); }
  return pc.uniforms.skycolor.rgb * texel;
}

// Analytic clear sky: zenith / horizon gradient above, horizon / ground below, a soft sun
// disc with a small aureole. All inputs in nits. The disc itself is only added when the
// path may see it directly (`want_sun`); other paths gather the sun by next-event
// estimation in the raygen.
vec3 procedural_sky(const vec3 d, const bool want_sun) {
  const float up = d.y;
  vec3 col;
  if (up >= 0.0) {
    col = mix(pc.uniforms.sky_horizon, pc.uniforms.sky_zenith, pow(up, 0.6));
  } else {
    col = mix(pc.uniforms.sky_horizon, pc.uniforms.sky_ground, clamp(-up * 6.0, 0.0, 1.0));
  }
  const float c = dot(d, pc.uniforms.sun_direction);
  const float cos_r = pc.uniforms.sun_cos_radius;
  // Disc: full inside the radius, fading over the outer fifth of it.
  const float disc = smoothstep(cos_r - (1.0 - cos_r) * 0.2, cos_r, c);
  const float aureole = pow(max(c, 0.0), 48.0) * 0.35;
  if (want_sun) { col += pc.uniforms.sun_radiance * disc * step(0.0, up); }
  col += pc.uniforms.sky_horizon * aureole;
  return col;
}

void main() {
  payload.t = 0.0;
  const vec3 d = gl_WorldRayDirectionEXT;
  vec3 sky;
  switch (pc.uniforms.sky_mode) {
    case 1u: sky = hdr_sky(d); break;
    case 2u: sky = procedural_sky(d, payload.want_sun != 0u); break;
    default: sky = pc.uniforms.skycolor.rgb; break;
  }
  payload.emission = sky * pc.uniforms.sky_brightness;
}
