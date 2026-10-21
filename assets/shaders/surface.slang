// The closest-hit contract: what every hit shader shares (the triangle fetch, tangent frame,
// ray-cone LOD, payload write). A surface group's shader is
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
// `scalar` layout the Rust side writes, with no vector-alignment question.
module surface;

// `HitPayload` (shared with the raygen).
__exported import hit_payload;

// `RTXMaterial` (104 bytes, scalar).
// The flat normal map every material without one points at (render_env.rs DEFAULT_NORMAL_TEXTURE_IDX).
public static const uint DEFAULT_NORMAL_TEXTURE = 1u;

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
    // blas.rs `material_flags`.
    public uint flags;
    public float uv_scroll_x, uv_scroll_y;
};

public static const uint MATERIAL_WATER = 1u;
public static const uint MATERIAL_NO_SHADOW = 2u;

// `Triangle` in blas.rs (32 bytes).
public struct Triangle {
    public uint tangent;
    public uint n0, n1, n2;
    public uint uv0, uv1, uv2;
    public float lod;
};

// `UniformData` fields a hit shader reads, as float indices; pinned by a test in
// ray_render_plugin.rs.
static const uint UNIFORM_LOD_BIAS_FLOAT = 94;
static const uint UNIFORM_EMISSIVE_BOOST_FLOAT = 103;
static const uint UNIFORM_CLOCK_FLOAT = 113;

// `PushConstants` in rt_types.slang, up to the last field a hit shader reads.
public struct PushConstants {
    public float* uniforms;
    public Material* materials;
    // Not void*: slangc lowers it to the uint* type, which then loses its ArrayStride (and
    // instance_slots is a uint*).
    public uint64_t focus;
    public uint pad0, pad1;
    public float4* prev_instances;
    public uint* instance_slots;
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

public float uniformEmissiveBoost() {
    return pc.uniforms[UNIFORM_EMISSIVE_BOOST_FLOAT];
}

public float4 materialBaseColor(Material m) {
    return float4(m.base_color_r, m.base_color_g, m.base_color_b, m.base_color_a);
}

public float3 materialEmissive(Material m) {
    return float3(m.emissive_r, m.emissive_g, m.emissive_b);
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

// One texture read over the hit's footprint (SurfaceHit.duv1 / duv2: the ray cone's ellipse in
// UV): the sampler filters it anisotropically, as a rasterizer's screen derivatives would.
public float4 sampleGrad(uint index, float2 uv, float2 duv1, float2 duv2) {
    return textures[NonUniformResourceIndex(index)].SampleGrad(uv, duv1, duv2);
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

// The DLSS guides stop here, smooth glass or not (water: its waves flip the dominant lobe
// between what lies under and what it reflects pixel by pixel, which the guides must not follow).
public void hitPayloadSetGuidesHere(inout HitPayload p, bool g) {
    p.r_m_t_i = (p.r_m_t_i & ~0x04) | (g ? 0x04 : 0);
}

float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

float2 toOct(float3 v) {
    let p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    return v.z <= 0.0 ? (1.0 - abs(p.yx)) * signNotZero(p) : p;
}

// Two unit normals, octahedral, as the payload carries them (surface, shading).
public float4 packNormals(float3 surface, float3 world) {
    return float4(toOct(surface), toOct(world));
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
    // The triangle's true face normal (world space), facing the ray: what a ray leaving the hit
    // offsets along (raygen offsetRay) -- vertex normals can lie in the face.
    public float3 face_normal;
    // Ray-cone texture LOD before the per-texture texel-count term; feed `sampleLod`.
    public float lod_base;
    // The ray cone's footprint on the surface as an ellipse (the bias applied): its two axes in
    // world space, and in the mesh's UV; feed `sampleGrad`.
    public float3 axis1;
    public float3 axis2;
    public float2 duv1;
    public float2 duv2;
    public bool inside;
    public float3 bary;
};

// `gl_ObjectToWorldEXT * vec4(v, 0)`.
public float3 objectToWorldDir(float3 v) {
    return mul(ObjectToWorld3x4(), float4(v, 0.0));
}

// A world-space direction in object space (`gl_WorldToObjectEXT * vec4(v, 0)`).
public float3 worldToObjectDir(float3 v) {
    return mul(WorldToObject3x4(), float4(v, 0.0));
}

// An object-space normal in world space: the inverse transpose (WorldToObject's 3x3,
// transposed). The plain transform tilts normals under non-uniform scale -- a floor scaled
// 8.4 x 0.2 x 6.4 turned a 0.3 degree tilt into 9.
public float3 objectToWorldNormal(float3 n) {
    let m = WorldToObject3x4();
    return float3(dot(n, float3(m[0][0], m[1][0], m[2][0])),
                  dot(n, float3(m[0][1], m[1][1], m[2][1])),
                  dot(n, float3(m[0][2], m[1][2], m[2][2])));
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
        + unpackUv(tri.uv2) * hit.bary.z
        + float2(hit.material.uv_scroll_x, hit.material.uv_scroll_y) * pc.uniforms[UNIFORM_CLOCK_FLOAT];
    hit.object_normal = unpackNormal(tri.n0) * hit.bary.x + unpackNormal(tri.n1) * hit.bary.y
        + unpackNormal(tri.n2) * hit.bary.z;
    hit.tangent = unpackNormal(tri.tangent);

    hit.inside = HitKind() == HIT_KIND_TRIANGLE_BACK_FACE;
    if (hit.inside) {
        hit.object_normal = -hit.object_normal;
    }
    hit.surface_normal = normalize(objectToWorldNormal(hit.object_normal));
    // From the BLAS's own positions (position fetch): world-space edges.
    let e1 = objectToWorldDir(HitTriangleVertexPosition(1) - HitTriangleVertexPosition(0));
    let e2 = objectToWorldDir(HitTriangleVertexPosition(2) - HitTriangleVertexPosition(0));
    hit.face_normal = normalize(cross(e1, e2));
    if (dot(hit.face_normal, WorldRayDirection()) > 0.0) {
        hit.face_normal = -hit.face_normal;
    }

    let object_scale = max(length(objectToWorldDir(float3(1.0, 0.0, 0.0))), 1.0e-6);
    let incidence = max(abs(dot(hit.surface_normal, WorldRayDirection())), 0.1);
    hit.lod_base = tri.lod - log2(object_scale) + log2(coneWidth(payload)) - log2(incidence)
        + uniformLodBias();

    // The footprint's ellipse (Akenine-Moller et al. 2021, "Improved Shader and Texture Level
    // of Detail Using Ray Cones"): one axis across the ray, one along it stretched by the
    // grazing angle, each the cone's width wide seen from the ray; then each axis's barycentric
    // step over the face, in UV.
    let d = WorldRayDirection();
    let n = hit.face_normal;
    let w = coneWidth(payload) * exp2(uniformLodBias());
    var a1 = d - dot(n, d) * n;
    if (dot(a1, a1) < 1.0e-8) {
        a1 = abs(n.x) < 0.9 ? cross(n, float3(1.0, 0.0, 0.0)) : cross(n, float3(0.0, 1.0, 0.0));
    }
    let p1 = a1 - dot(d, a1) * d;
    a1 *= w / max(length(p1), 1.0e-4);
    var a2 = cross(n, a1);
    let p2 = a2 - dot(d, a2) * d;
    a2 *= w / max(length(p2), 1.0e-4);
    hit.axis1 = a1;
    hit.axis2 = a2;
    let area = dot(n, cross(e1, e2));
    let duv_e1 = unpackUv(tri.uv1) - unpackUv(tri.uv0);
    let duv_e2 = unpackUv(tri.uv2) - unpackUv(tri.uv0);
    if (abs(area) > 1.0e-12) {
        let b1 = float2(dot(n, cross(a1, e2)), dot(n, cross(e1, a1))) / area;
        let b2 = float2(dot(n, cross(a2, e2)), dot(n, cross(e1, a2))) / area;
        hit.duv1 = b1.x * duv_e1 + b1.y * duv_e2;
        hit.duv2 = b2.x * duv_e1 + b2.y * duv_e2;
    } else {
        hit.duv1 = float2(0.0);
        hit.duv2 = float2(0.0);
    }
    return hit;
}

// A tangent-space normal map sample turned into a world normal: the frame is built in world
// space (the normal by the inverse transpose, the tangent as a direction, re-orthogonalised).
public float3 surfaceWorldNormal(SurfaceHit hit, float3 texture_normal) {
    let n = hit.surface_normal;
    var t = objectToWorldDir(hit.tangent);
    t = t - n * dot(t, n);
    let tl = length(t);
    if (tl < 1.0e-6) {
        return n;
    }
    t /= tl;
    return normalize(t * texture_normal.x + cross(n, t) * texture_normal.y + n * texture_normal.z);
}

// The material's own normal map, or the geometric normal when it has none (exactly: the flat
// default texel, 128/255, decodes 0.4 % off axis).
public float3 surfaceShadingNormal(SurfaceHit hit) {
    if (hit.material.normal_texture == DEFAULT_NORMAL_TEXTURE) {
        return hit.surface_normal;
    }
    return surfaceWorldNormal(
        hit, normalTexel(sampleGrad(hit.material.normal_texture, hit.uv, hit.duv1, hit.duv2)));
}

// This hit's instance slot: the TLAS is built from the live slots packed (tlas_builder.rs), so
// InstanceIndex() is a packed row.
public uint instanceSlot() {
    return pc.instance_slots[InstanceIndex()];
}

// Where an object-space point of this instance was last frame. A zero row is a slot with no
// last frame (new, reused, grown, or a scatter cell that just entered): no object motion.
public float3 prevWorldPos(float3 object_p) {
    let base = instanceSlot() * 4;
    let r0 = pc.prev_instances[base + 0];
    let r1 = pc.prev_instances[base + 1];
    let r2 = pc.prev_instances[base + 2];
    if (all(r0 == 0.0) && all(r1 == 0.0) && all(r2 == 0.0)) {
        return WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    }
    let p = float4(object_p, 1.0);
    return float3(dot(r0, p), dot(r1, p), dot(r2, p));
}

// Hit distance, normals, identity for the light-table MIS lookup, and last frame's position
// for motion vectors Call it after writing
// colour, emission and roughness.
public void surfaceWritePayload(inout HitPayload payload, SurfaceHit hit, float3 world_normal) {
    payload.t = RayTCurrent();
    payload.surface_and_world_normal = packNormals(hit.face_normal, world_normal);
    payload.slot = instanceSlot();
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
    payload.prev_world_pos = prevWorldPos(object_p);
    hitPayloadSetInside(payload, hit.inside);
    payload.lobes = packLobes(hit.material.diffuse_transmission, hit.material.clearcoat,
                              hit.material.clearcoat_roughness);
}
