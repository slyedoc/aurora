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

// A LAYERED surface: a second PBR set over the base on the faces that point up, and a detail
// set multiplied over the result. Both extra sets are sampled TRIPLANAR in world space, so
// they hold their scale across a mesh whose own UVs were laid out for the base.
//
// The blend law is `jackdaw_surface::LayerBlend::weight`, expression for expression; that
// function is the definition and carries the tests.

struct Layered {
  vec4 layer_color;
  vec4 detail_color;
  float layer_uv_scale;
  float layer_normal_strength;
  float layer_metallic;
  float layer_perceptual_roughness;
  float detail_uv_scale;
  float detail_normal_strength;
  float blend_amount;
  float blend_power;
  float blend_threshold;
  float blend_position;
  float blend_contrast;
  // Set when the mask is painted into a vertex colour channel, which this back end has no
  // vertex stream for: the shader falls back to the slope so the surface still reads.
  uint use_vertex_color;
  uint vertex_color_channel;
  // `LAYER_NORMAL_MAP` / `DETAIL_NORMAL_MAP`: an unbound colour or ORM map falls back to
  // white, which is what a set without one should contribute, but white read as a normal
  // is a slant.
  uint flags;
  uint layer_base_color_texture;
  uint layer_normal_map_texture;
  uint layer_orm_texture;
  uint detail_base_color_texture;
  uint detail_normal_map_texture;
  uint detail_orm_texture;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer LayeredData {
  Layered entries[];
};

const uint LAYER_NORMAL_MAP = 1u;
const uint DETAIL_NORMAL_MAP = 2u;
const float MIN_THRESHOLD_EXPONENT = 0.001;

// Weights for the three projections, normalized so the set contributes once.
vec3 triplanarWeights(const vec3 n) {
  vec3 w = abs(n);
  w = max(w - 0.2, vec3(0.0));
  const float sum = w.x + w.y + w.z;
  return sum > 1.0e-5 ? w / sum : vec3(0.0, 1.0, 0.0);
}

vec4 triplanar(const uint tex, const vec3 p, const vec3 w, const float lod) {
  return w.x * sampleLod(tex, p.yz, lod)
       + w.y * sampleLod(tex, p.xz, lod)
       + w.z * sampleLod(tex, p.xy, lod);
}

// Whiteout blend: each projection's tangent normal is lifted into world space by swizzling
// to that plane's axes, then summed. Cheaper than three TBNs and stable at the seams.
vec3 triplanarNormal(const uint tex, const vec3 p, const vec3 w, const vec3 n,
                     const float lod, const float strength) {
  const vec3 sx = normalTexel(sampleLod(tex, p.yz, lod));
  const vec3 sy = normalTexel(sampleLod(tex, p.xz, lod));
  const vec3 sz = normalTexel(sampleLod(tex, p.xy, lod));
  const vec3 axis = sign(n);
  const vec3 nx = vec3(sx.z * axis.x, sx.y, sx.x);
  const vec3 ny = vec3(sy.x, sy.z * axis.y, sy.y);
  const vec3 nz = vec3(sz.x, sz.y * axis.z, sz.z);
  const vec3 blended = normalize(nx * w.x + ny * w.y + nz * w.z);
  return normalize(mix(n, blended, clamp(strength, 0.0, 1.0)));
}

// `LayerBlend::weight`. `up` is the world normal's Y.
float layerWeight(const Layered p, const float up) {
  const float lifted = clamp(up + p.blend_power, 0.0, 1.0);
  const float exponent =
      MIN_THRESHOLD_EXPONENT + p.blend_threshold * (1.0 - MIN_THRESHOLD_EXPONENT);
  return p.blend_amount * pow(abs(lifted), exponent);
}

void main() {
  const SurfaceHit hit = surfaceHit();
  const Material material = hit.material;

  payload.refract_index = material.refract_index;
  payload.absorption = material.absorption;

  vec4 albedo = material.base_color_factor
      * sampleLod(material.base_color_texture, hit.uv, hit.lod_base);
  float roughness = material.roughness_factor;
  float metallic = material.metallic_factor;
  vec3 normal = surfaceShadingNormal(hit);

  // A class that published no parameter buffer shades as plain PBR rather than reading
  // address 0. `surfaceData` is zero exactly then (`SurfaceGroupData::get`).
  const uint64_t data_address = packUint2x32(surfaceData);
  if (data_address != 0) {
    LayeredData data = LayeredData(data_address);
    const Layered p = data.entries[material.surface_param_index];

    // Slope from the GEOMETRIC normal, not the shading normal: a normal map should change
    // how the surface catches light, not which layer it is made of.
    const float weight = layerWeight(p, clamp(hit.surface_normal.y, -1.0, 1.0));
    const vec3 world_p = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    const vec3 w = triplanarWeights(hit.surface_normal);

    if (weight > 0.0) {
      const vec3 lp = world_p * p.layer_uv_scale;
      vec4 layer = p.layer_color
          * triplanar(p.layer_base_color_texture, lp, w, hit.lod_base);
      // Occlusion red, roughness green, metallic blue -- the channel order glTF writes.
      const vec3 orm = triplanar(p.layer_orm_texture, lp, w, hit.lod_base).rgb;
      albedo = mix(albedo, layer, weight);
      roughness = mix(roughness, p.layer_perceptual_roughness * orm.g, weight);
      metallic = mix(metallic, p.layer_metallic * orm.b, weight);
      if ((p.flags & LAYER_NORMAL_MAP) != 0u) {
        const vec3 ln = triplanarNormal(p.layer_normal_map_texture, lp, w, hit.surface_normal,
                                        hit.lod_base, p.layer_normal_strength);
        normal = normalize(mix(normal, ln, weight));
      }
    }

    // The detail set multiplies over base and layer alike, so it is not part of the blend.
    const vec3 dp = world_p * p.detail_uv_scale;
    albedo *= p.detail_color
        * triplanar(p.detail_base_color_texture, dp, w, hit.lod_base);
    if ((p.flags & DETAIL_NORMAL_MAP) != 0u) {
      normal = triplanarNormal(p.detail_normal_map_texture, dp, w, normal, hit.lod_base,
                               p.detail_normal_strength);
    }
  }

  payload.color = albedo;
  payload.emission = material.base_emissive_factor.rgb;
  payload.emission *= sampleLod(material.base_emissive_texture, hit.uv, hit.lod_base).rgb;
  payload.emission *= pc.uniforms.emissive_boost;

  float transmission = material.specular_transmission_factor;
  transmission *= sampleLod(material.specular_transmission_texture, hit.uv, hit.lod_base).r;

  surfaceWritePayload(hit, normal);
  hitPayloadSetTransmission(payload, transmission);
  hitPayloadSetRoughness(payload, clamp(roughness, 0.0, 1.0));
  hitPayloadSetMetallic(payload, clamp(metallic, 0.0, 1.0));
  hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0);
}
