#ifndef H_TYPES
#define H_TYPES

#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : require

struct Vertex {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};

struct Triangle {
  uint tangent;
  uint normals[3];
  uint uvs[3];
  uint padding;
};

vec3 unpackNormal(uint packed) {
  float nx = float(packed >> 16) / 65535.0 * 2.0 - 1.0;
  float ny = float((packed >> 1) & 32767) / 32767.0 * 2.0 - 1.0;
  float nz = sqrt(clamp(1.0 - nx * nx - ny * ny, 0.0, 1.0)) * ((packed & 1) == 1 ? -1.0 : 1.0);
  return vec3(nx, ny, nz);
}

vec2 unpackUv(uint packed) {
  return unpackHalf2x16(packed);
}

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly restrict buffer UniformData {
  vec4 skycolor;
  mat4 inverse_view;
  mat4 inverse_projection;
  uint tick;
  uint accumulate;
  uint pull_focus_x;
  uint pull_focus_y;
  float gamma;
  float exposure;
  float aperture;
  float foginess;
  float fog_scatter;
  float sky_brightness;
  // DLSS: camera matrices for depth / motion vectors, the sub-pixel jitter (pixels), and
  // whether the guide images are bound this frame.
  mat4 view;
  mat4 view_proj;
  mat4 prev_view_proj;
  vec2 jitter;
  uint dlss;
  // Free-running frame counter (RNG seed under DLSS; `tick` is 0 unless accumulating).
  uint frame;
  // Firefly suppression: indirect path contributions are clamped to this luminance (0 = off).
  float radiance_clamp;
  // Paths per pixel this frame and their maximum length.
  uint samples;
  uint max_bounces;
  // Post-process vignette strength (0 = off).
  float vignette;
  // Sky source: 0 flat colour (skycolor), 1 equirect HDR (skycolor = scale), 2 procedural.
  uint sky_mode;
  float sun_cos_radius;
  vec3 sun_direction;
  vec3 sun_radiance;
  vec3 sky_zenith;
  vec3 sky_horizon;
  vec3 sky_ground;
  // Entries in the emissive-triangle light table (0 = no light NEE / MIS).
  uint light_entries;
  // ReSTIR DI initial candidates per pixel (0 = plain 1-sample NEE at the primary vertex).
  uint restir_candidates;
  // Light-table generation; reservoirs from another generation are dropped.
  uint light_epoch;
  // Cap on temporal history, in candidate-samples.
  float restir_m_clamp;
};

// Last frame's VkAccelerationStructureInstanceKHR array: 4 vec4 per instance, rows 0..2 are
// the 3x4 transform (row-major).
layout (buffer_reference, scalar, buffer_reference_align = 16) readonly buffer PrevInstances {
  vec4 data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) buffer restrict FocusData {
  float focal_distance;
};

layout (buffer_reference, scalar, buffer_reference_align = 16) readonly buffer VertexData {
  Vertex data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 16) readonly buffer TriangleData {
  Triangle data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer IndexData {
  uint data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer GeometryData {
  uint index_offsets[];
};

// ---- Emissive-triangle light table (src/lights.rs, lights.slang) --------------------------

// One emissive instance: its BLAS streams, its TLAS slot (world transform), and the range
// of entries its triangles occupy in the global table. 64 bytes.
struct LightInstance {
  VertexData vertices;
  IndexData indices;
  GeometryData geom_to_index;
  GeometryData geom_to_triangle;
  uint geom_count;
  uint slot;
  uint entry_base;
  uint tri_count;
  vec3 emission;
  float pad;
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer LightCdf {
  float data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer LightInstances {
  LightInstance data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer SlotToLight {
  uint data[];
};

// One ReSTIR DI reservoir (src/restir.rs allocates one per traced pixel). 32 bytes.
struct Reservoir {
  // Winning light-table entry and its (su, sv) triangle sample, packed as 2xf16.
  uint y;
  uint uv_pk;
  // Unbiased contribution weight (0 after an occluded visibility ray) and history length.
  float W;
  float M;
  // Validation: the light-table generation, the shading normal (oct, 2xf16), the view
  // depth (clip w) and the frame that wrote the reservoir.
  uint epoch;
  uint normal_pk;
  float depth;
  uint frame;
};

layout (buffer_reference, scalar, buffer_reference_align = 8) buffer ReservoirData {
  Reservoir data[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer LightsHeader {
  uint entry_count;
  uint linst_count;
  // Written by the light_scan kernel; 0 = table not ready, sampling is skipped.
  float total_power;
  uint slot_map_count;
  LightCdf cdf;
  LightInstances linsts;
  SlotToLight slot_to_linst;
  uvec2 pad2;
};

struct Material {
  vec4 base_color_factor;
  vec4 base_emissive_factor;
  uint base_color_texture;
  uint base_emissive_texture;
  uint specular_transmission_texture;
  uint metallic_roughness_texture;
  uint normal_texture;
  float specular_transmission_factor;
  float roughness_factor;
  float metallic_factor;
  float refract_index;
  // Beer-Lambert absorption per unit distance inside the surface (linear RGB).
  vec3 absorption;
  // Alpha-mask cutout threshold (0 = opaque, no any-hit test).
  float alpha_cutoff;
};

layout (buffer_reference, scalar, buffer_reference_align = 16) readonly buffer MaterialData {
  Material materials[];
};

layout (buffer_reference, scalar, buffer_reference_align = 8) readonly buffer BluenoiseData {
  uint bluenoise[];
};


struct HitPayload {
  float t;
  float refract_index;
  // r = roughness, m = metallic, t = transmission, i = inside
  int r_m_t_i;
  vec4 color;
  vec3 emission;
  vec4 surface_and_world_normal;
  vec3 absorption;
  // Where this hit point was last frame (object motion for DLSS motion vectors).
  vec3 prev_world_pos;
  // Raygen -> miss: 1 when a miss may return the procedural sun disc (camera rays and
  // pure-specular paths); 0 after a BRDF-sampled bounce, where the sun is gathered by
  // next-event estimation instead.
  uint want_sun;
  // Hit identity for the light-table MIS lookup: TLAS slot (gl_InstanceID; ~0 for
  // spheres) and the global triangle index within the BLAS.
  uint slot;
  uint prim_tri;
};

struct PushConstants {
  UniformData uniforms;
  MaterialData materials;
  BluenoiseData bluenoise;
  FocusData focus;
  uint skydome;
  uint pad0;
  PrevInstances prev_instances;
  // Emissive-triangle light table (valid whenever uniforms.light_entries > 0).
  LightsHeader lights;
  // This frame's instance rows (same layout as prev_instances), for light transforms.
  PrevInstances cur_instances;
  // ReSTIR DI reservoirs: last frame's (read) and this frame's (written).
  ReservoirData reservoirs_prev;
  ReservoirData reservoirs_cur;
};

void hitPayloadSetRoughness(inout HitPayload p, float r) {
  int v = int(r *255.0) % 256;
  p.r_m_t_i = (p.r_m_t_i & 0x00FFFFFF) | (v << 24);
}

float hitPayloadGetRoughness(const HitPayload p) {
  int v = (p.r_m_t_i >> 24) % 256;
  return v / 255.0;
}

void hitPayloadSetMetallic(inout HitPayload p, float m) {
  int v = int(m *255.0) % 256;
  p.r_m_t_i = (p.r_m_t_i & 0xFF00FFFF) | (v << 16);
}

float hitPayloadGetMetallic(const HitPayload p) {
  int v = (p.r_m_t_i >> 16) % 256;
  return v / 255.0;
}

void hitPayloadSetTransmission(inout HitPayload p, float t) {
  int v = int(t * 255.0) % 256;
  p.r_m_t_i = (p.r_m_t_i & 0xFFFF00FF) | (v << 8);
}

float hitPayloadGetTransmission(const HitPayload p) {
  int v = (p.r_m_t_i >> 8) % 256;
  return v / 255.0;
}

void hitPayloadSetInside(inout HitPayload p, bool i) {
  p.r_m_t_i = (p.r_m_t_i & 0xFFFFFF00) | (i ? 0xFF : 0);
}

bool hitPayloadGetInside(const HitPayload p) {
  int v = p.r_m_t_i % 256;
  return v != 0;
}


// Returns +/- 1
vec2 signNotZero( vec2 v )
{
    return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}

// Assume normalized input. Output is on [-1, 1] for each component.
vec2 float32x3_to_oct( in vec3 v )
{
    // Project the sphere onto the octahedron, and then onto the xy plane
    vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

vec3 oct_to_float32x3(in vec2 e )
{
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    return normalize(v);
}

vec4 pack2_normals(in vec3 lhs, in vec3 rhs) {
  return vec4(float32x3_to_oct(lhs), float32x3_to_oct(rhs));
}

#endif
