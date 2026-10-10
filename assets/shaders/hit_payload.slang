// The ray payload, shared by the Slang raygen and every hit / miss shader (GLSL ones too).
module hit_payload;

// `HitPayload` in types.glsl, member for member: the payload is shared memory between stages.
public struct HitPayload {
    public float t;
    public float refract_index;
    // r = roughness, m = metallic, t = transmission, i = flags (bit 0 inside, bit 1 masked:
    // the material has an alpha cutoff, so its texture alpha means coverage).
    public int r_m_t_i;
    public float4 color;
    public float3 emission;
    public float4 surface_and_world_normal;
    public float3 absorption;
    // Where this hit point was last frame (object motion for DLSS motion vectors).
    public float3 prev_world_pos;
    // Raygen -> miss: low bit 1 when a miss may return the procedural sun disc (camera rays
    // and pure-specular paths), 0 after a BRDF-sampled bounce, where the sun is gathered by
    // next-event estimation instead; the ray's world mask above it.
    public uint want_sun;
    // Hit identity for the light-table MIS lookup: TLAS slot (InstanceIndex; ~0 for
    // spheres) and the global triangle index within the BLAS.
    public uint slot;
    public uint prim_tri;
    // Raygen -> hit: the ray cone at the ray's origin (x = width, y = spread angle, radians),
    // for the texture level of detail.
    public float2 cone;
    // Hit -> raygen: thin translucency, clear coat and its roughness, a byte each (0..255).
    public uint lobes;
};

public uint packLobes(float transmission, float clearcoat, float clearcoat_roughness) {
    return uint(saturate(transmission) * 255.0 + 0.5)
        | (uint(saturate(clearcoat) * 255.0 + 0.5) << 8)
        | (uint(saturate(clearcoat_roughness) * 255.0 + 0.5) << 16);
}
