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

// Terrain tiles' splat inputs (`terrain::TerrainShadeGpu`), behind a terrain record's
// `surfaceData`: the alpha atlas (16x16 chunk cells of `cell`^2 RGBA8 weights for layers
// 1..3), the per-chunk layer table and the palette's bindless textures.
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer TerrainWords {
  uint v[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer TerrainLayers {
  uvec4 v[];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer TerrainPaletteTable {
  uvec2 v[];  // bindless texture index, repeats per chunk (float bits)
};
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer TerrainShade {
  TerrainWords alpha;
  TerrainLayers layers;
  TerrainPaletteTable palette;
  uint atlas;
  uint cell;
  float size;
  uint palette_len;
};
const uint TERRAIN_NO_LAYER = 0xffffffffu;

// Bilinear weight of layer slot `ch + 1` inside one chunk's alpha cell (edge-clamped to it).
float terrainAlpha(TerrainShade s, const uvec2 chunk, const vec2 cell_uv, const uint ch) {
  const float fcell = float(s.cell);
  const float x = clamp(cell_uv.x * fcell - 0.5, 0.0, fcell - 1.0);
  const float y = clamp(cell_uv.y * fcell - 0.5, 0.0, fcell - 1.0);
  const uint x0 = uint(x);
  const uint y0 = uint(y);
  const uint x1 = min(x0 + 1u, s.cell - 1u);
  const uint y1 = min(y0 + 1u, s.cell - 1u);
  const uint bx = chunk.x * s.cell;
  const uint by = chunk.y * s.cell;
  const uint shift = ch * 8u;
  const float w00 = float((s.alpha.v[(by + y0) * s.atlas + bx + x0] >> shift) & 0xffu);
  const float w10 = float((s.alpha.v[(by + y0) * s.atlas + bx + x1] >> shift) & 0xffu);
  const float w01 = float((s.alpha.v[(by + y1) * s.atlas + bx + x0] >> shift) & 0xffu);
  const float w11 = float((s.alpha.v[(by + y1) * s.atlas + bx + x1] >> shift) & 0xffu);
  return mix(mix(w00, w10, x - float(x0)), mix(w01, w11, x - float(x0)), y - float(y0)) / 255.0;
}

// One palette texture at the tile uv, repeated per chunk, at the ray cone's mip.
vec4 terrainLayer(TerrainShade s, const uint layer, const vec2 tile_uv, const float lod_base) {
  const uvec2 entry = s.palette.v[min(layer, s.palette_len - 1u)];
  const float scale = 16.0 * uintBitsToFloat(entry.y);
  return sampleLod(entry.x, tile_uv * scale, lod_base + log2(scale));
}

// WoW's alphamap blend: layer 0 is the base, layers 1..3 lerp over it by their weights.
vec4 terrainSplat(TerrainShade s, const vec2 tile_uv, const float lod_base) {
  const vec2 grid = clamp(tile_uv, 0.0, 0.99999) * 16.0;
  const uvec2 chunk = uvec2(grid);
  const vec2 cell_uv = grid - vec2(chunk);
  const uvec4 layers = s.layers.v[chunk.y * 16u + chunk.x];
  vec4 color = vec4(70.0, 90.0, 50.0, 255.0) / 255.0;
  if (layers.x != TERRAIN_NO_LAYER) {
    color = terrainLayer(s, layers.x, tile_uv, lod_base);
  }
  if (layers.y != TERRAIN_NO_LAYER) {
    color = mix(color, terrainLayer(s, layers.y, tile_uv, lod_base), terrainAlpha(s, chunk, cell_uv, 0u));
  }
  if (layers.z != TERRAIN_NO_LAYER) {
    color = mix(color, terrainLayer(s, layers.z, tile_uv, lod_base), terrainAlpha(s, chunk, cell_uv, 1u));
  }
  if (layers.w != TERRAIN_NO_LAYER) {
    color = mix(color, terrainLayer(s, layers.w, tile_uv, lod_base), terrainAlpha(s, chunk, cell_uv, 2u));
  }
  return vec4(color.rgb, 1.0);
}

// The OPAQUE surface class (`SurfaceClass::OPAQUE`): standard PBR out of the material
// record. Every other class is a sibling of this file -- same `surfaceHit()` and
// `surfaceWritePayload()`, a different middle.
void main() {
  const SurfaceHit hit = surfaceHit();
  const Material material = hit.material;

  payload.refract_index = material.refract_index;
  payload.absorption = material.absorption;

  payload.color = material.base_color_factor;
  const uint64_t terrain_address = packUint2x32(surfaceData);
  if ((recordFlags.x & 2u) != 0u && terrain_address != 0ul) {
    // A terrain tile: the splat, per hit. The tile uv comes from the object-space hit point
    // (the mesh spans +-size/2), not the packed vertex uv: 128 repeats a tile would show its
    // quantisation.
    TerrainShade shade = TerrainShade(terrain_address);
    const vec3 object_p = gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT;
    const vec2 tile_uv = object_p.xz / shade.size + 0.5;
    payload.color *= toLinear(terrainSplat(shade, tile_uv, hit.lod_base));
  } else {
    payload.color *= toLinear(sampleLod(material.base_color_texture, hit.uv, hit.lod_base));
  }
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
