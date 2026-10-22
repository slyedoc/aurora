// The ray payload, shared by the Slang raygen and every hit / miss shader.
module hit_payload;

// `HitPayload`: the payload is shared memory between the raygen, miss and hit stages.
public struct HitPayload {
    public float t;
    public float refract_index;
    // r = roughness, m = metallic, t = transmission, i = flags (bit 0 inside, bit 1 masked:
    // the material has an alpha cutoff, so its texture alpha means coverage; bit 2 guides here:
    // the DLSS guides stop at this surface even where it is smooth glass).
    public int r_m_t_i;
    public float4 color;
    public float3 emission;
    public float4 surface_and_world_normal;
    public float3 absorption;
    // Where this hit point was last frame (object motion for DLSS motion vectors).
    public float3 prev_world_pos;
    // Raygen -> miss: low bit 1 when a miss may return the sun discs (camera rays and
    // pure-specular paths), 0 after a BRDF-sampled bounce, where the suns are gathered by
    // next-event estimation instead; the ray's environment index above it.
    public uint miss_flags;
    // Hit identity for the light-table MIS lookup: TLAS slot (InstanceIndex; ~0 for
    // spheres) and the global triangle index within the BLAS.
    public uint slot;
    public uint prim_tri;
    // Raygen -> hit: the ray cone at the ray's origin (x = width, y = spread angle, radians),
    // for the texture level of detail.
    public float2 cone;
    // Hit -> raygen: thin translucency, clear coat and its roughness, a byte each (0..255).
    public uint lobes;
    // Hit -> raygen: the size (metres) of the numbers this hit point was computed from: its
    // primitive's object-space extent in world units, or the ray's length. Its error is a few
    // ulps of that, which a ray leaving it must clear (rt_types.slang offsetRay); 0 for a miss.
    public float extent;
};

// Shadow rays: whether the ray reached the sky (the shadow miss shader sets it), and the nearest
// water surface it passed (its water id, medium row + 1, 0 none; at `water_t`). A few words, so
// a visibility test carries none of the shading payload.
public struct ShadowPayload {
    public uint visible;
    public uint water;
    public float water_t;
};

// The water id a hit payload carries (water_surface.slang: medium row + 1; 0 not water).
public uint hitPayloadWater(HitPayload p) {
    return p.lobes >> 24;
}

public uint packLobes(float transmission, float clearcoat, float clearcoat_roughness) {
    return uint(saturate(transmission) * 255.0 + 0.5)
        | (uint(saturate(clearcoat) * 255.0 + 0.5) << 8)
        | (uint(saturate(clearcoat_roughness) * 255.0 + 0.5) << 16);
}
