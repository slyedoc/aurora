// A jackdaw terrain's splat: two texture ids per control cell, blended by comparing their
// height maps; the shading of jackdaw_terrain's `terrain_splat.wgsl`, read at the hit.
//
// The surface meshes sit at the terrain's origin, so the object-space hit point IS the
// terrain-local position and the control lookup is exact, not routed through packed UVs.
// `uv` is 0..1 across the control grid, from its origin over its size.
// A hit shader has no derivatives; every read is at the ray-cone level.
import surface;

static const float SHARPNESS_MIN = 4.0;
static const float SHARPNESS_RANGE = 60.0;
static const uint ID_MASK = 0x1Fu;
static const uint OVERLAY_SHIFT = 5u;
static const uint BLEND_SHIFT = 10u;
static const uint BLEND_MASK = 0xFFu;
static const uint MANUAL_BIT = 0x40000u;
static const uint MAX_SLOTS = 16u;

static const uint HAS_ALBEDO = 1u;
static const uint HAS_NORMAL = 2u;
static const uint FLIP_NORMAL_Y = 4u;
static const uint HAS_HEIGHT = 8u;
static const uint HAS_OCCLUSION = 16u;
static const uint HAS_ROUGHNESS = 32u;

// `SplatSlotGpu` in jackdaw_terrain's render/aurora.rs (32 bytes).
struct SplatSlot {
    float uv_scale;
    float detile;
    uint albedo;
    uint normal;
    uint height;
    uint occlusion;
    uint roughness;
    uint flags;
};

// `TerrainSplatGpu` there (592 bytes): one row per terrain.
struct TerrainSplat {
    uint* control;
    float* slope;
    float terrain_size_x;
    float terrain_size_z;
    float blend_sharpness;
    float perceptual_roughness;
    uint control_resolution;
    uint layer_count;
    uint autoterrain_enabled;
    uint autoterrain_base_slot;
    uint autoterrain_slope_slot;
    float autoterrain_slope_start;
    float autoterrain_slope_end;
    float tint_strength;
    uint tint;
    float terrain_origin_x;
    float terrain_origin_z;
    uint pad0;
    SplatSlot slots[16];
};

struct Accum {
    float3 albedo;
    float3 normal;
    float occlusion;
    float roughness;
    float total;
};

float2 tileHash(float2 tile) {
    var h = frac(float3(tile.x, tile.y, tile.x + tile.y) * float3(0.1031, 0.1030, 0.0973));
    h += dot(h, h.yzx + 33.33);
    return frac((h.xx + h.yz) * h.zy);
}

float2 turnedBy(float2 v, float2 turn) {
    return float2(turn.x * v.x - turn.y * v.y, turn.y * v.x + turn.x * v.y);
}

uint loadControl(TerrainSplat s, int2 c) {
    let last = int(max(s.control_resolution, 1u) - 1u);
    let at = clamp(c, int2(0), int2(last));
    return s.control[uint(at.y) * s.control_resolution + uint(at.x)];
}

float loadSlope(TerrainSplat s, int2 c) {
    if (s.slope == nullptr) {
        return 0.0;
    }
    let last = int(max(s.control_resolution, 1u) - 1u);
    let at = clamp(c, int2(0), int2(last));
    return s.slope[uint(at.y) * s.control_resolution + uint(at.x)];
}

// One texture id's share, summed over the corners that named it (`accumulate` in the WGSL).
// `lod` is the ray-cone level in terrain-local units; the id's UV scale is added here.
void accumulate(inout Accum a, TerrainSplat s, uint id, float4 corner_weights,
                float layer_weight, float2 local_xz, float lod) {
    let layer = min(id, min(max(s.layer_count, 1u), MAX_SLOTS) - 1u);
    let slot = s.slots[layer];
    let uv = local_xz * slot.uv_scale;
    let slot_lod = lod + log2(max(slot.uv_scale, 1.0e-6));

    // Detiling: the four tile centres around the sample, each turned and shifted rigidly by
    // its own hash and sampled in its own frame, blended with bilinear weights.
    float2 tap_uv[4];
    float2 tap_turn[4];
    float tap_weight[4];
    uint taps = 1u;
    tap_uv[0] = uv;
    tap_turn[0] = float2(1.0, 0.0);
    tap_weight[0] = 1.0;
    if (slot.detile > 0.0) {
        let p = uv - float2(0.5);
        let base = floor(p);
        let f = smoothstep(float2(0.0), float2(1.0), frac(p));
        taps = 4u;
        for (uint j = 0u; j < 2u; j++) {
            for (uint i = 0u; i < 2u; i++) {
                let corner = base + float2(float(i), float(j));
                let weight = lerp(1.0 - f.x, f.x, float(i)) * lerp(1.0 - f.y, f.y, float(j));
                let hash = tileHash(corner);
                let angle = (hash.x * 2.0 - 1.0) * slot.detile * 3.14159265;
                let shift = (hash - float2(0.5)) * slot.detile;
                let pivot = corner + float2(0.5);
                let turn = float2(cos(angle), sin(angle));
                let index = j * 2u + i;
                tap_uv[index] = pivot + turnedBy(uv - pivot, turn) + shift;
                tap_turn[index] = turn;
                tap_weight[index] = weight;
            }
        }
    }

    var albedo_sum = float3(0.0);
    var normal_sum = float3(0.0);
    var height_sum = 0.0;
    var occlusion_sum = 0.0;
    var roughness_sum = 0.0;
    var taken = 0.0;
    for (uint t = 0u; t < taps; t++) {
        let w = tap_weight[t];
        if (w <= 1.0e-4) {
            continue;
        }
        let tuv = tap_uv[t];
        albedo_sum += w * ((slot.flags & HAS_ALBEDO) != 0u
            ? sampleLod(slot.albedo, tuv, slot_lod).rgb : float3(0.2159));
        height_sum += w * ((slot.flags & HAS_HEIGHT) != 0u
            ? sampleLod(slot.height, tuv, slot_lod).r : 0.5);
        var n = float3(0.0, 0.0, 1.0);
        if ((slot.flags & HAS_NORMAL) != 0u) {
            n = normalTexel(sampleLod(slot.normal, tuv, slot_lod));
            if ((slot.flags & FLIP_NORMAL_Y) != 0u) {
                n.y = -n.y;
            }
            // The tap's UV was turned, so the normal stored in it turns back.
            n.xy = turnedBy(n.xy, float2(tap_turn[t].x, -tap_turn[t].y));
        }
        normal_sum += w * n;
        occlusion_sum += w * ((slot.flags & HAS_OCCLUSION) != 0u
            ? sampleLod(slot.occlusion, tuv, slot_lod).r : 1.0);
        roughness_sum += w * ((slot.flags & HAS_ROUGHNESS) != 0u
            ? sampleLod(slot.roughness, tuv, slot_lod).g : s.perceptual_roughness);
        taken += w;
    }
    let inv = 1.0 / max(taken, 1.0e-8);

    // The height-blend curve: a near corner gets a height advantage, a far one cannot win
    // on height alone.
    let sharpness = SHARPNESS_MIN + SHARPNESS_RANGE * s.blend_sharpness;
    let contested = float4(layer_weight + height_sum * inv);
    let weight = dot(pow(corner_weights + contested, float4(sharpness)), corner_weights);

    a.albedo += albedo_sum * inv * weight;
    a.normal += normal_sum * inv * weight;
    a.occlusion += occlusion_sum * inv * weight;
    a.roughness += roughness_sum * inv * weight;
    a.total += weight;
}

void accumulateControl(inout Accum a, TerrainSplat s, uint control, float4 corner_weights,
                       float2 local_xz, float lod) {
    let base_id = control & ID_MASK;
    let overlay_id = (control >> OVERLAY_SHIFT) & ID_MASK;
    let blend = float((control >> BLEND_SHIFT) & BLEND_MASK) / 255.0;
    if (blend < 1.0) {
        accumulate(a, s, base_id, corner_weights, 1.0 - blend, local_xz, lod);
    }
    if (blend >= 1.0 || (blend > 0.0 && overlay_id != base_id)) {
        accumulate(a, s, overlay_id, corner_weights, blend, local_xz, lod);
    }
}

// The control word a cell's own slope stands for, when no hand claimed it.
uint autoterrainControl(TerrainSplat s, float slope) {
    let t = smoothstep(s.autoterrain_slope_start, s.autoterrain_slope_end, slope);
    let blend = uint(clamp(t, 0.0, 1.0) * 255.0 + 0.5);
    return (s.autoterrain_base_slot & ID_MASK)
        | ((s.autoterrain_slope_slot & ID_MASK) << OVERLAY_SHIFT)
        | ((blend & BLEND_MASK) << BLEND_SHIFT);
}

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    payload.refract_index = hit.material.refract_index;
    payload.absorption = float3(hit.material.absorption_r, hit.material.absorption_g,
                                hit.material.absorption_b);

    var albedo = float3(0.5);
    var roughness = hit.material.roughness_factor;
    var normal = hit.surface_normal;

    let address = surfaceData();
    if (address != 0) {
        let s = ((TerrainSplat*)address)[hit.material.surface_param_index];
        if (s.control != nullptr && s.control_resolution >= 2u) {
            let local_xz = (ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection()).xz;
            let size = max(float2(s.terrain_size_x, s.terrain_size_z), float2(1.0e-6));
            let origin = float2(s.terrain_origin_x, s.terrain_origin_z);
            let uv = saturate((local_xz - origin) / size);

            // The ray-cone footprint in terrain-local units, before any texture's own terms.
            let object_scale = max(length(objectToWorldDir(float3(1.0, 0.0, 0.0))), 1.0e-6);
            let incidence = max(abs(dot(hit.surface_normal, WorldRayDirection())), 0.1);
            let lod = log2(coneWidth(payload) / object_scale) - log2(incidence) + uniformLodBias();

            let res = float(s.control_resolution);
            let g = uv * (res - 1.0);
            let corner = clamp(int2(floor(g)), int2(0), int2(int(s.control_resolution) - 2));
            let f = saturate(g - float2(corner));

            var c00 = loadControl(s, corner);
            var c10 = loadControl(s, corner + int2(1, 0));
            var c01 = loadControl(s, corner + int2(0, 1));
            var c11 = loadControl(s, corner + int2(1, 1));
            if (s.autoterrain_enabled != 0u) {
                let slope = lerp(lerp(loadSlope(s, corner), loadSlope(s, corner + int2(1, 0)), f.x),
                                 lerp(loadSlope(s, corner + int2(0, 1)),
                                      loadSlope(s, corner + int2(1, 1)), f.x), f.y);
                let unclaimed = autoterrainControl(s, slope);
                c00 = (c00 & MANUAL_BIT) != 0u ? c00 : unclaimed;
                c10 = (c10 & MANUAL_BIT) != 0u ? c10 : unclaimed;
                c01 = (c01 & MANUAL_BIT) != 0u ? c01 : unclaimed;
                c11 = (c11 & MANUAL_BIT) != 0u ? c11 : unclaimed;
            }

            let iv = float2(1.0) - f;
            let w = float4(iv.x * iv.y, f.x * iv.y, iv.x * f.y, f.x * f.y);
            Accum a;
            a.albedo = float3(0.0);
            a.normal = float3(0.0);
            a.occlusion = 0.0;
            a.roughness = 0.0;
            a.total = 0.0;
            if (c00 == c10 && c00 == c01 && c00 == c11) {
                accumulateControl(a, s, c00, w, local_xz, lod);
            } else {
                accumulateControl(a, s, c00, float4(w.x, 0.0, 0.0, 0.0), local_xz, lod);
                accumulateControl(a, s, c10, float4(0.0, w.y, 0.0, 0.0), local_xz, lod);
                accumulateControl(a, s, c01, float4(0.0, 0.0, w.z, 0.0), local_xz, lod);
                accumulateControl(a, s, c11, float4(0.0, 0.0, 0.0, w.w), local_xz, lod);
            }
            let inv_total = 1.0 / max(a.total, 1.0e-8);

            // Tint texel i sits at grid point i, so the sample lands on texel centres.
            var tint = float3(1.0);
            if (s.tint_strength > 0.0) {
                tint = sampleLevel(s.tint, (uv * (res - 1.0) + 0.5) / res, 0.0).rgb;
            }
            albedo = a.albedo * inv_total * lerp(float3(1.0), tint, s.tint_strength)
                * (a.occlusion * inv_total);
            roughness = a.roughness * inv_total;

            // No tangents on the mesh: terrain +X projected onto the normal, +Z where that
            // collapses.
            let n = normalize(hit.object_normal);
            let reference = abs(n.x) > 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            let t = normalize(reference - n * dot(reference, n));
            let tn = normalize(a.normal * inv_total);
            normal = normalize(objectToWorldDir(t * tn.x + cross(t, n) * tn.y + n * tn.z));
        }
    }

    payload.color = float4(albedo, 1.0);
    payload.emission = float3(0.0);
    surfaceWritePayload(payload, hit, normal);
    hitPayloadSetTransmission(payload, 0.0);
    hitPayloadSetRoughness(payload, saturate(roughness));
    hitPayloadSetMetallic(payload, 0.0);
    hitPayloadSetMasked(payload, false);
}
