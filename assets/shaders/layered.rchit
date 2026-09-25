#version 460
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
// `surfaceData` is a device address carried as a uvec2; recombining it needs int64.
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "types.glsl"
#include "common.glsl"

#define PACKED 1
#include "surface_common.glsl"

// A LAYERED surface: two PBR sets blended by the surface's slope, which is the shape
// jackdaw's `LayeredSurface` wants -- rock on the steep faces, grass on the flat ones,
// without a splat texture. The blend is the surface class's own parameters, read from the
// record's `surfaceData` rather than from the shared material array: every class has a
// different parameter shape, so they cannot share one.
//
// This is the reference for what a surface group looks like. It fetches the triangle and
// writes the payload with the same two calls the opaque class uses; only the middle differs.

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer LayeredParams {
  // Indexed exactly like `pc.materials`: instance custom index + geometry index.
  vec4 layer_color[];
};

struct Layered {
  vec4 layer_color;
  vec4 detail_color;
  float layer_roughness;
  float detail_roughness;
  float layer_metallic;
  float detail_metallic;
  // Surface normal Y above which the layer wins outright; below `blend_start` the detail
  // wins outright. Between them it ramps, sharpened by `blend_contrast`.
  float blend_start;
  float blend_end;
  float blend_contrast;
  float pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer LayeredData {
  Layered entries[];
};

void main() {
  const SurfaceHit hit = surfaceHit();
  const Material material = hit.material;

  payload.refract_index = material.refract_index;
  payload.absorption = material.absorption;

  // The base colour texture still comes from the shared material, so a layered surface
  // keeps whatever albedo map it was given; the class only decides how the two sets mix.
  const vec4 albedo = material.base_color_factor
      * toLinear(sampleLod(material.base_color_texture, hit.uv, hit.lod_base));

  vec4 tint = vec4(1.0);
  float roughness = material.roughness_factor;
  float metallic = material.metallic_factor;

  // A class that published no parameter buffer shades as plain PBR rather than reading
  // address 0. `surfaceData` is zero exactly then (`SurfaceGroupData::get`).
  const uint64_t data_address = packUint2x32(surfaceData);
  if (data_address != 0) {
    LayeredData data = LayeredData(data_address);
    const Layered p = data.entries[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT];

    // Slope from the GEOMETRIC normal, not the shading normal: a normal map should change
    // how the surface catches light, not which layer it is made of.
    const float slope = clamp(hit.surface_normal.y, -1.0, 1.0);
    const float span = max(p.blend_end - p.blend_start, 1.0e-4);
    float t = clamp((slope - p.blend_start) / span, 0.0, 1.0);
    // Contrast sharpens the transition around its midpoint; 1.0 leaves it linear.
    t = pow(t, max(p.blend_contrast, 1.0e-3));
    t = t * t * (3.0 - 2.0 * t);

    tint = mix(p.detail_color, p.layer_color, t);
    roughness *= mix(p.detail_roughness, p.layer_roughness, t);
    metallic = mix(p.detail_metallic, p.layer_metallic, t);
  }

  payload.color = albedo * tint;
  payload.emission = material.base_emissive_factor.rgb;
  payload.emission *= toLinear(sampleLod(material.base_emissive_texture, hit.uv, hit.lod_base)).rgb;
  payload.emission *= pc.uniforms.emissive_boost;

  float transmission = material.specular_transmission_factor;
  transmission *= sampleLod(material.specular_transmission_texture, hit.uv, hit.lod_base).r;

  surfaceWritePayload(hit, surfaceShadingNormal(hit));
  hitPayloadSetTransmission(payload, transmission);
  hitPayloadSetRoughness(payload, clamp(roughness, 0.0, 1.0));
  hitPayloadSetMetallic(payload, clamp(metallic, 0.0, 1.0));
  hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0);
}
