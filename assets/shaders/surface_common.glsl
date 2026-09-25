// Everything a closest-hit shader does before it decides what the surface LOOKS like.
//
// A surface group (`surface_group.rs`) is a closest-hit shader of its own: layered terrain,
// water and foliage each shade differently, but every one of them fetches the same triangle,
// builds the same tangent frame, picks the same ray-cone texture LOD and writes the same
// motion vectors. That shared half lives here so the three shaders are three SHADING
// functions rather than three copies of this file that drift apart.
//
// A group's shader is then:
//
//   #include "surface_common.glsl"
//   void main() {
//     SurfaceHit hit = surfaceHit();
//     ... write payload.color / emission / roughness from `hit` ...
//     surfaceWritePayload(hit);
//   }
//
// The declarations below (`textures`, the shader record, push constants, the payload) are
// the contract every hit shader shares; a group that needs its own per-material parameters
// reads `surfaceData` from the record as its own buffer reference type.

#ifndef SURFACE_COMMON_GLSL
#define SURFACE_COMMON_GLSL

layout(set = 1, binding = 200) uniform sampler2D textures[];

layout(shaderRecordEXT, scalar) buffer ShaderRecord {
  VertexData vertexData;
  TriangleData triangleData;
  IndexData indexData;
  GeometryData geometries;
  GeometryData triangles;
  // Skinned instances (sbt.rs): last frame's deformed stream, flagged in recordFlags.x.
  VertexData prevVertexData;
  uvec2 recordFlags;
  // This record's surface class's parameter buffer (`surface_group::SurfaceGroupData`),
  // 0 when the class published none. The opaque class ignores it.
  uvec2 surfaceData;
};

layout(push_constant, std430) uniform Registers {
  PushConstants pc;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;

hitAttributeEXT vec2 attribs;

// The triangle this ray hit, resolved into everything shading needs.
struct SurfaceHit {
  Material material;
  vec2 uv;
  // Object space, already flipped for a back-face hit.
  vec3 object_normal;
  vec3 tangent;
  // World space.
  vec3 surface_normal;
  // Ray-cone texture LOD before the per-texture texel-count term; feed `sampleLod`.
  float lod_base;
  // The hardware's geometric answer, not a normal dot test.
  bool inside;
  vec3 bary;
};

vec3 calcTangent(const Vertex v0, const Vertex v1, const Vertex v2) {
  const vec3 edge1 = v1.position - v0.position;
  const vec3 edge2 = v2.position - v0.position;
  const vec2 deltaUV1 = v1.texcoord - v0.texcoord;
  const vec2 deltaUV2 = v2.texcoord - v0.texcoord;

  const float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
  if (abs(denom) < 0.00001f) {
    return vec3(0.0, 0.0, 1.0);
  }

  vec3 tangent;
  const float f = 1.0 / denom;
  tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
  tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
  tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

  return normalize(tangent);
}

// sRGB-encoded texels (colour textures are uploaded as UNORM) to linear; alpha untouched.
vec4 toLinear(const vec4 sRGB) {
  const bvec4 cutoff = lessThan(sRGB, vec4(0.04045));
  const vec4 higher = pow((sRGB + vec4(0.055)) / vec4(1.055), vec4(2.4));
  const vec4 lower = sRGB / vec4(12.92);

  return vec4(mix(higher, lower, cutoff).rgb, sRGB.a);
}

// One texture read at the ray-cone level: `lod_base` plus half the log2 of the texel count.
vec4 sampleLod(const uint index, const vec2 uv, const float lod_base) {
  const vec2 size = vec2(textureSize(textures[index], 0));
  return textureLod(textures[index], uv, lod_base + 0.5 * log2(size.x * size.y));
}

// The triangle, the tangent frame and the texture footprint.
SurfaceHit surfaceHit() {
  SurfaceHit hit;
  hit.bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  hit.material = pc.materials.materials[gl_InstanceCustomIndexEXT + gl_GeometryIndexEXT];

  float tri_lod;
#if PACKED
  Triangle tri = triangleData.data[triangles.index_offsets[gl_GeometryIndexEXT] + gl_PrimitiveID];
  hit.uv = mat3x2(unpackUv(tri.uvs[0]), unpackUv(tri.uvs[1]), unpackUv(tri.uvs[2])) * hit.bary;
  hit.object_normal = mat3(unpackNormal(tri.normals[0]),
                           unpackNormal(tri.normals[1]),
                           unpackNormal(tri.normals[2])) * hit.bary;
  hit.tangent = unpackNormal(tri.tangent);
  tri_lod = tri.lod;
#else
  const uint index_offset = geometries.index_offsets[gl_GeometryIndexEXT];
  const Vertex v0 = vertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 0]];
  const Vertex v1 = vertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 1]];
  const Vertex v2 = vertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 2]];
  hit.uv = v0.texcoord * hit.bary.x + v1.texcoord * hit.bary.y + v2.texcoord * hit.bary.z;
  hit.object_normal = v0.normal * hit.bary.x + v1.normal * hit.bary.y + v2.normal * hit.bary.z;
  hit.tangent = calcTangent(v0, v1, v2);
  tri_lod = 0.0;
#endif

  // Which side of the triangle was hit: the hardware's geometric answer, stable under
  // camera motion. A smooth-normal dot test here flips front-face grazing hits, which made
  // foliage normals (and the medium side for glass) swim with the camera.
  hit.inside = gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT;
  if (hit.inside) {
    hit.object_normal = -hit.object_normal;
  }
  hit.surface_normal = normalize((gl_ObjectToWorldEXT * vec4(hit.object_normal, 0.0)).xyz);

  // Texture level of detail by ray cone (Akenine-Moller et al., Ray Tracing Gems ch. 20): a
  // ray-tracing stage has no derivatives, so the footprint comes from the cone the raygen
  // carries (width at the hit), the triangle's texel density (`tri_lod`, object space, so
  // the instance scale comes off), and the incidence angle. Per texture, half the log of
  // its texel count is added (`sampleLod`). Without this every read is level 0, and under
  // sub-pixel jitter a minified texture lands on a different texel every frame.
  const float cone_width = max(payload.cone.x + payload.cone.y * gl_HitTEXT, 1.0e-7);
  const float object_scale = max(length(gl_ObjectToWorldEXT[0].xyz), 1.0e-6);
  const float incidence = max(abs(dot(hit.surface_normal, gl_WorldRayDirectionEXT)), 0.1);
  hit.lod_base = tri_lod - log2(object_scale) + log2(cone_width) - log2(incidence)
      + pc.uniforms.lod_bias;

  return hit;
}

// A tangent-space normal map sample turned into a world normal.
vec3 surfaceWorldNormal(const SurfaceHit hit, const vec3 texture_normal) {
  const vec3 bitangent = cross(hit.object_normal, hit.tangent);
  const mat3 TBN = mat3(hit.tangent, bitangent, hit.object_normal);
  return normalize(mat3(gl_ObjectToWorldEXT) * TBN * texture_normal);
}

// The material's own normal map, or the geometric normal when it has none.
vec3 surfaceShadingNormal(const SurfaceHit hit) {
  const vec3 texture_normal =
      sampleLod(hit.material.normal_texture, hit.uv, hit.lod_base).xyz * 2.0 - 1.0;
  return surfaceWorldNormal(hit, texture_normal);
}

// The half of the payload that is the same whatever the surface looks like: hit distance,
// normals, identity for the light-table MIS lookup, and last frame's position for motion
// vectors. Call it AFTER writing colour/emission/roughness; it does not touch those.
void surfaceWritePayload(const SurfaceHit hit, const vec3 world_normal) {
  payload.t = gl_HitTEXT;
  payload.surface_and_world_normal = pack2_normals(hit.surface_normal, world_normal);
  payload.slot = gl_InstanceID;
  payload.prim_tri = triangles.index_offsets[gl_GeometryIndexEXT] + gl_PrimitiveID;
  {
    // Object-space hit point through last frame's instance transform. A skinned instance
    // takes the point from last frame's deformed vertices instead (object motion).
    vec3 object_p = gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT;
    if ((recordFlags.x & 1u) != 0u) {
      const uint index_offset = geometries.index_offsets[gl_GeometryIndexEXT];
      const vec3 q0 = prevVertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 0]].position;
      const vec3 q1 = prevVertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 1]].position;
      const vec3 q2 = prevVertexData.data[indexData.data[index_offset + gl_PrimitiveID * 3 + 2]].position;
      object_p = q0 * hit.bary.x + q1 * hit.bary.y + q2 * hit.bary.z;
    }
    const vec4 p = vec4(object_p, 1.0);
    const uint base = gl_InstanceID * 4;
    const vec4 r0 = pc.prev_instances.data[base + 0];
    const vec4 r1 = pc.prev_instances.data[base + 1];
    const vec4 r2 = pc.prev_instances.data[base + 2];
    payload.prev_world_pos = vec3(dot(r0, p), dot(r1, p), dot(r2, p));
  }
  hitPayloadSetInside(payload, hit.inside);
}

#endif // SURFACE_COMMON_GLSL
