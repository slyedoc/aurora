// Neural radiance cache (src/nrc.rs): what the raygen shares with the kernels (nrc.slang), with
// no cooperative vectors: a surface point's inputs, their packing into records, the cache box, and
// where a query's answer goes.
import rt_types;

// What the network sees of a surface point.
public struct NrcInput {
    // World position (the hash grid needs no box).
    public float3 position;
    // Outgoing direction (towards the viewer) and shading normal, unit.
    public float3 dir;
    public float3 normal;
    public float roughness;
    public float3 diffuse;
    public float3 specular;
};

// A query's destination (NrcRecord.target in the query buffer): none, the pixel's colour, its
// ReSTIR PT sample's light (gi_candidates L), or only a training path's targets (its tail).
public static const uint NRC_DEST_NONE = 0u;
public static const uint NRC_DEST_PIXEL = 1u;
public static const uint NRC_DEST_GI = 2u;
public static const uint NRC_DEST_TAIL = 3u;
public static const uint NRC_NO_SLOT = 0xFFFFFFFFu;

public uint packOct16(float3 v) {
    let e = saturate(float32x3_to_oct(v) * 0.5 + 0.5);
    return uint(e.x * 65535.0 + 0.5) | (uint(e.y * 65535.0 + 0.5) << 16);
}

public float3 unpackOct16(uint p) {
    let e = float2(float(p & 0xFFFFu), float(p >> 16)) * (2.0 / 65535.0) - 1.0;
    return oct_to_float32x3(e);
}

public uint packUnorm8x4(float4 v) {
    let u = uint4(saturate(v) * 255.0 + 0.5);
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

public float4 unpackUnorm8x4(uint p) {
    return float4(float(p & 0xFFu), float((p >> 8) & 0xFFu), float((p >> 16) & 0xFFu),
                  float(p >> 24)) * (1.0 / 255.0);
}

public NrcRecord nrcPackRecord(NrcInput i) {
    NrcRecord r = {};
    r.position = i.position;
    r.dir = packOct16(i.dir);
    r.normal = packOct16(i.normal);
    r.rough_diffuse = packUnorm8x4(float4(i.roughness, i.diffuse));
    r.specular = packUnorm8x4(float4(i.specular, 0.0));
    return r;
}

public NrcInput nrcUnpackRecord(NrcRecord r) {
    NrcInput i;
    i.position = r.position;
    i.dir = unpackOct16(r.dir);
    i.normal = unpackOct16(r.normal);
    let rd = unpackUnorm8x4(r.rough_diffuse);
    i.roughness = rd.x;
    i.diffuse = rd.yzw;
    i.specular = unpackUnorm8x4(r.specular).xyz;
    return i;
}

// Reflectance factorization (Müller et al. 2021, sec. 4.1): the network learns scattered
// radiance over the surface's albedo, the lighting without the texture.
public float3 nrcFactor(NrcInput i) {
    return max(i.diffuse + i.specular, float3(0.05));
}

// A query's weight: f16 per channel in the record's `a` (r, g) and `c` (b) words.
public void nrcSetWeight(NrcRecord* r, float3 w) {
    let h = min(w, float3(65504.0));
    r.a = packHalf2(h.rg);
    r.c = packHalf2(float2(h.b, 0.0));
}

public float3 nrcWeight(NrcRecord r) {
    return float3(unpackHalf2(r.a), unpackHalf2(r.c).x);
}
