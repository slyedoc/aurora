#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "types.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(set=1, binding=200)         uniform sampler2D textures[];

layout(push_constant, std430) uniform Registers {
  PushConstants pc;
};

#include "atmosphere.glsl"

const float PI = 3.14159265359;

// Planetary atmosphere (atmosphere.glsl): the sky-view LUT, the sun disc through the air
// on paths that may see it, and the space image (the layer's texture, colour = scale)
// behind the atmosphere's transmittance.
vec3 atmosphere_sky(const vec3 d, const bool want_sun, const uint tex, const vec3 scale) {
  vec3 L, T;
  bool ground;
  atmoSkyView(d, L, T, ground);
  if (!ground) {
    if (want_sun) { L += atmoSunDisc(d, T); }
    if (scale != vec3(0.0)) {
      L += T * scale * min(texture(textures[tex], env_dir_to_uv(d)).rgb, vec3(300.0));
    }
  }
  return L;
}

// Equirectangular lookup: texel (linear radiance) times the layer's scale.
vec3 hdr_sky(const vec3 d, const uint tex, const vec3 scale, const bool cam_world) {
  const vec2 uv = env_dir_to_uv(d);
  vec3 texel = texture(textures[tex], uv).rgb;
  // Without importance sampling a BRDF-sampled ray is the only way to the sun texel, so its
  // extreme values are clamped RELATIVE to the sky scale (a physically bright sky passes, a
  // 1e5x sun does not). With the sampler (env_light.rs) the raygen gathers the sun by
  // next-event estimation and MIS-weights these hits -- but the env table only serves the
  // CAMERA's world; other layers' skies always clamp.
  if (pc.uniforms.env_w == 0u || !cam_world) { texel = min(texel, vec3(300.0)); }
  return scale * texel;
}

// Zenith / horizon gradient above, horizon / ground below (all nits), with a small aureole
// around each of the world's suns.
vec3 gradient_sky(const vec3 d, const WorldEnv env) {
  const float up = d.y;
  vec3 col;
  if (up >= 0.0) {
    col = mix(env.horizon.rgb, env.zenith.rgb, pow(up, 0.6));
  } else {
    col = mix(env.horizon.rgb, env.ground.rgb, clamp(-up * 6.0, 0.0, 1.0));
  }
  for (uint i = 0u; i < env.sun_count; i++) {
    col += env.horizon.rgb * pow(max(dot(d, env.suns[i].direction.xyz), 0.0), 48.0) * 0.35;
  }
  return col;
}

// The world's sun discs, on the paths that may see them directly; every other path gathers
// the suns by next-event estimation in the raygen. Full inside the radius, fading over its
// outer fifth.
vec3 sun_discs(const vec3 d, const WorldEnv env) {
  vec3 col = vec3(0.0);
  for (uint i = 0u; i < env.sun_count; i++) {
    const WorldSun sun = env.suns[i];
    const float cos_r = sun.direction.w;
    const float disc = smoothstep(cos_r - (1.0 - cos_r) * 0.2, cos_r, dot(d, sun.direction.xyz));
    col += sun.radiance.rgb * sun.radiance.a * disc;
  }
  return col;
}

void main() {
  payload.t = 0.0;
  const vec3 d = gl_WorldRayDirectionEXT;
  // The raygen packs the ray's cull mask above the want-sun bit: this miss evaluates the
  // sky of the world the ray is IN (portals swap the mask mid-path).
  const bool want_sun = (payload.want_sun & 1u) != 0u;
  const uint world = worldOf(payload.want_sun >> 1);
  const WorldEnv env = pc.uniforms.worlds.w[world];
  vec3 sky;
  switch (env.mode) {
    case 1u: sky = hdr_sky(d, env.tex, env.color.rgb, world == worldOf(pc.uniforms.camera_mask)); break;
    case 2u: sky = gradient_sky(d, env); break;
    // The atmosphere draws its own (first) sun through the air.
    case 3u: sky = atmosphere_sky(d, want_sun, env.tex, env.color.rgb); break;
    default: sky = env.color.rgb; break;
  }
  if (want_sun && env.mode != 3u) {
    sky += sun_discs(d, env);
  }
  payload.emission = sky * pc.uniforms.sky_brightness;
}
