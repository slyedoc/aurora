// The closest-hit contract in Slang: what `surface_common.glsl` (+ the parts of `types.glsl` a
// hit shader touches) gives the GLSL hit shaders. A Slang surface group's shader is
//
//   import surface;
//   [shader("closesthit")]
//   void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
//     let hit = surfaceHit(payload, attribs);
//     ... payload.color / emission / roughness ...
//     surfaceWritePayload(payload, hit, world_normal);
//   }
//
// Everything read through a pointer is declared with scalars only, so its byte layout is the
// GLSL side's `scalar` layout with no vector-alignment question.
module surface;

// `HitPayload` (shared with the raygen).
__exported import hit_payload;

// `Material` in types.glsl (104 bytes, scalar).
public struct Material {
    public float base_color_r, base_color_g, base_color_b, base_color_a;
    public float emissive_r, emissive_g, emissive_b, emissive_a;
    public uint base_color_texture;
    public uint base_emissive_texture;
    public uint specular_transmission_texture;
    public uint metallic_roughness_texture;
    public uint normal_texture;
    public float specular_transmission_factor;
    public float roughness_factor;
    public float metallic_factor;
    public float refract_index;
    public float absorption_r, absorption_g, absorption_b;
    public float alpha_cutoff;
    public uint surface_param_index;
    public float diffuse_transmission;
    public float clearcoat;
    public float clearcoat_roughness;
    public uint pad;
};

// `Triangle` in types.glsl / blas.rs (32 bytes).
public struct Triangle {
    public uint tangent;
    public uint n0, n1, n2;
    public uint uv0, uv1, uv2;
    public float lod;
};

// `UniformData` is read for one field; its byte offset is pinned by a test in ray_render_plugin.rs.
static const uint UNIFORM_LOD_BIAS_FLOAT = 100;

// `PushConstants` in types.glsl, up to the last field a hit shader reads.
public struct PushConstants {
    public float* uniforms;
    public Material* materials;
    public void* bluenoise;
    public void* focus;
    public uint skydome;
    public uint pad0;
    public float4* prev_instances;
};
[[vk::push_constant]] public ConstantBuffer<PushConstants> pc;

// A u32 in a stream the record points at. Typed as a one-field struct, not `uint*`: slangc
// leaves a `uint*` reached only through the shader record without its ArrayStride (invalid
// SPIR-V, rejected by the validation layer), while struct pointers are decorated.
public struct U32 {
    public uint v;
};

// The shader record (sbt.rs): the instance's streams, then its class's parameter buffer.
public struct ShaderRecord {
    public float* vertex_data;
    public Triangle* triangle_data;
    public U32* index_data;
    public U32* geometries;
    public U32* triangles;
    public float* prev_vertex_data;
    public uint2 record_flags;
    // This record's surface class's parameter buffer, 0 when the class published none.
    public uint2 surface_data;
};
[[vk::shader_record]] public ConstantBuffer<ShaderRecord> record;

[[vk::binding(200, 1)]] public Sampler2D textures[];

// This record's class parameter buffer address.
public uint64_t surfaceData() {
    return uint64_t(record.surface_data.x) | (uint64_t(record.surface_data.y) << 32);
}

public float uniformLodBias() {
    return pc.uniforms[UNIFORM_LOD_BIAS_FLOAT];
}

public float3 unpackNormal(uint packed) {
    let nx = float(packed >> 16) / 65535.0 * 2.0 - 1.0;
    let ny = float((packed >> 1) & 32767) / 32767.0 * 2.0 - 1.0;
    let nz = sqrt(clamp(1.0 - nx * nx - ny * ny, 0.0, 1.0)) * ((packed & 1) == 1 ? -1.0 : 1.0);
    return float3(nx, ny, nz);
}

public float2 unpackUv(uint packed) {
    return float2(f16tof32(packed & 0xFFFF), f16tof32(packed >> 16));
}

// A tangent-space normal from a normal map's XY; Z is rebuilt, so two-channel maps work.
public float3 normalTexel(float4 texel) {
    let xy = texel.xy * 2.0 - 1.0;
    return float3(xy, sqrt(max(0.0, 1.0 - dot(xy, xy))));
}

// One texture read at the ray-cone level: `lod_base` plus half the log2 of the texel count.
public float4 sampleLod(uint index, float2 uv, float lod_base) {
    let t = textures[NonUniformResourceIndex(index)];
    uint w, h, levels;
    t.GetDimensions(0, w, h, levels);
    return t.SampleLevel(uv, lod_base + 0.5 * log2(float(w) * float(h)));
}

public float4 sampleLevel(uint index, float2 uv, float level) {
    return textures[NonUniformResourceIndex(index)].SampleLevel(uv, level);
}

public void hitPayloadSetRoughness(inout HitPayload p, float r) {
    let v = int(r * 255.0) % 256;
    p.r_m_t_i = (p.r_m_t_i & 0x00FFFFFF) | (v << 24);
}

public void hitPayloadSetMetallic(inout HitPayload p, float m) {
    let v = int(m * 255.0) % 256;
    p.r_m_t_i = (p.r_m_t_i & 0xFF00FFFF) | (v << 16);
}

public void hitPayloadSetTransmission(inout HitPayload p, float t) {
    let v = int(t * 255.0) % 256;
    p.r_m_t_i = (p.r_m_t_i & 0xFFFF00FF) | (v << 8);
}

public void hitPayloadSetInside(inout HitPayload p, bool i) {
    p.r_m_t_i = (p.r_m_t_i & ~0x01) | (i ? 0x01 : 0);
}

public void hitPayloadSetMasked(inout HitPayload p, bool m) {
    p.r_m_t_i = (p.r_m_t_i & ~0x02) | (m ? 0x02 : 0);
}

float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

float2 toOct(float3 v) {
    let p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    return v.z <= 0.0 ? (1.0 - abs(p.yx)) * signNotZero(p) : p;
}

// The triangle this ray hit, resolved into everything shading needs.
public struct SurfaceHit {
    public Material material;
    public float2 uv;
    // Object space, already flipped for a back-face hit.
    public float3 object_normal;
    public float3 tangent;
    // World space.
    public float3 surface_normal;
    // Ray-cone texture LOD before the per-texture texel-count term; feed `sampleLod`.
    public float lod_base;
    public bool inside;
    public float3 bary;
};

// `gl_ObjectToWorldEXT * vec4(v, 0)`.
public float3 objectToWorldDir(float3 v) {
    return mul(ObjectToWorld3x4(), float4(v, 0.0));
}

// The ray cone's width at the hit (raygen -> hit, payload.cone), never zero.
public float coneWidth(HitPayload payload) {
    return max(payload.cone.x + payload.cone.y * RayTCurrent(), 1.0e-7);
}

public SurfaceHit surfaceHit(HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    SurfaceHit hit;
    hit.bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                      attribs.barycentrics.x, attribs.barycentrics.y);
    hit.material = pc.materials[InstanceID() + GeometryIndex()];

    let tri = record.triangle_data[record.triangles[GeometryIndex()].v + PrimitiveIndex()];
    hit.uv = unpackUv(tri.uv0) * hit.bary.x + unpackUv(tri.uv1) * hit.bary.y
        + unpackUv(tri.uv2) * hit.bary.z;
    hit.object_normal = unpackNormal(tri.n0) * hit.bary.x + unpackNormal(tri.n1) * hit.bary.y
        + unpackNormal(tri.n2) * hit.bary.z;
    hit.tangent = unpackNormal(tri.tangent);

    hit.inside = HitKind() == HIT_KIND_TRIANGLE_BACK_FACE;
    if (hit.inside) {
        hit.object_normal = -hit.object_normal;
    }
    hit.surface_normal = normalize(objectToWorldDir(hit.object_normal));

    let object_scale = max(length(objectToWorldDir(float3(1.0, 0.0, 0.0))), 1.0e-6);
    let incidence = max(abs(dot(hit.surface_normal, WorldRayDirection())), 0.1);
    hit.lod_base = tri.lod - log2(object_scale) + log2(coneWidth(payload)) - log2(incidence)
        + uniformLodBias();
    return hit;
}

// A tangent-space normal map sample turned into a world normal.
public float3 surfaceWorldNormal(SurfaceHit hit, float3 texture_normal) {
    let bitangent = cross(hit.object_normal, hit.tangent);
    let object = hit.tangent * texture_normal.x + bitangent * texture_normal.y
        + hit.object_normal * texture_normal.z;
    return normalize(objectToWorldDir(object));
}

// Hit distance, normals, identity for the light-table MIS lookup, and last frame's position
// for motion vectors (`surfaceWritePayload` in surface_common.glsl). Call it after writing
// colour, emission and roughness.
public void surfaceWritePayload(inout HitPayload payload, SurfaceHit hit, float3 world_normal) {
    payload.t = RayTCurrent();
    payload.surface_and_world_normal =
        float4(toOct(hit.surface_normal), toOct(world_normal));
    payload.slot = InstanceIndex();
    payload.prim_tri = record.triangles[GeometryIndex()].v + PrimitiveIndex();

    var object_p = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
    if ((record.record_flags.x & 1u) != 0u) {
        let index_offset = record.geometries[GeometryIndex()].v;
        float3 q[3];
        for (uint k = 0; k < 3; k++) {
            let v = record.index_data[index_offset + PrimitiveIndex() * 3 + k].v * 8;
            q[k] = float3(record.prev_vertex_data[v], record.prev_vertex_data[v + 1],
                          record.prev_vertex_data[v + 2]);
        }
        object_p = q[0] * hit.bary.x + q[1] * hit.bary.y + q[2] * hit.bary.z;
    }
    let p = float4(object_p, 1.0);
    let base = InstanceIndex() * 4;
    payload.prev_world_pos = float3(dot(pc.prev_instances[base + 0], p),
                                    dot(pc.prev_instances[base + 1], p),
                                    dot(pc.prev_instances[base + 2], p));
    hitPayloadSetInside(payload, hit.inside);
    payload.lobes = packLobes(hit.material.diffuse_transmission, hit.material.clearcoat,
                              hit.material.clearcoat_roughness);
}
