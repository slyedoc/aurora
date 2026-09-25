#version 460
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "types.glsl"
#include "common.glsl"

#define PACKED 1
#include "surface_common.glsl"

// The OPAQUE surface class (`SurfaceClass::OPAQUE`): standard PBR out of the material
// record. Every other class is a sibling of this file -- same `surfaceHit()` and
// `surfaceWritePayload()`, a different middle.
void main() {
  const SurfaceHit hit = surfaceHit();
  const Material material = hit.material;

  payload.refract_index = material.refract_index;
  payload.absorption = material.absorption;

  payload.color = material.base_color_factor;
  payload.color *= toLinear(sampleLod(material.base_color_texture, hit.uv, hit.lod_base));
  if (material.alpha_cutoff > 0.0) {
    // A cutout's coverage comes from level 0, like the any-hit test: a blurred alpha would
    // thin foliage out with distance.
    payload.color.a = material.base_color_factor.a
        * texture(textures[material.base_color_texture], hit.uv).a;
  }
  payload.emission = material.base_emissive_factor.rgb;
  payload.emission *= toLinear(sampleLod(material.base_emissive_texture, hit.uv, hit.lod_base)).rgb;
  payload.emission *= pc.uniforms.emissive_boost;

  // Terrain records (recordFlags.x bit 1): the editor's brush ring, emissive so it reads in
  // every view (the spectator window and both eyes) without a gizmo pass.
  if ((recordFlags.x & 2u) != 0u && pc.uniforms.brush_active != 0u) {
    const vec3 world_p = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    const float d = length(world_p.xz - pc.uniforms.brush_center);
    const float width = max(pc.uniforms.brush_radius * 0.04, 0.12);
    const float ring = 1.0 - smoothstep(0.0, width, abs(d - pc.uniforms.brush_radius));
    payload.emission += pc.uniforms.brush_color * ring;
  }

  float transmission = material.specular_transmission_factor;
  transmission *= sampleLod(material.specular_transmission_texture, hit.uv, hit.lod_base).r;

  const vec4 mr = sampleLod(material.metallic_roughness_texture, hit.uv, hit.lod_base);
  const float roughness = material.roughness_factor * mr.g;
  const float metallic = material.metallic_factor * mr.b;

  surfaceWritePayload(hit, surfaceShadingNormal(hit));
  hitPayloadSetTransmission(payload, transmission);
  hitPayloadSetRoughness(payload, roughness);
  hitPayloadSetMetallic(payload, metallic);
  hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0);
}
