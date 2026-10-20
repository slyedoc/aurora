// Water volumes (src/water_volume.rs): a homogeneous medium in each box, crossed analytically
// (no medium tracking). Volumes are upright: local +y is the surface's normal.
module water;

import rt_types;

// Where a ray crosses a volume's box, clipped to [0, t_max], and whether it enters (`top_in`) or
// leaves (`top_out`) through the surface. False when it misses the box.
bool waterSpan(WaterVolumeGpu v, float3 o, float3 d, float t_max, out float t0, out float t1,
               out bool top_in, out bool top_out) {
    let lo = waterLocal(v, float4(o, 1.0));
    let ld = waterLocal(v, float4(d, 0.0));
    let inv = 1.0 / select(abs(ld) < 1.0e-12, float3(1.0e-12), ld);
    let a = (float3(-1.0) - lo) * inv;
    let b = (float3(1.0) - lo) * inv;
    let n = min(a, b);
    let f = max(a, b);
    let enter = max(n.x, max(n.y, n.z));
    let leave = min(f.x, min(f.y, f.z));
    t0 = max(enter, 0.0);
    t1 = min(leave, t_max);
    top_in = ld.y < 0.0 && enter >= 0.0 && n.y >= max(n.x, n.z);
    top_out = ld.y > 0.0 && leave <= t_max && f.y <= min(f.x, f.z);
    return t1 > t0;
}

float3 waterScattering(WaterVolumeGpu v) {
    return float3(v.scattering_r, v.scattering_g, v.scattering_b);
}

// Extinction per metre: absorption plus scattering.
float3 waterExtinction(WaterVolumeGpu v) {
    return float3(v.absorption_r, v.absorption_g, v.absorption_b) + waterScattering(v);
}

// The density collisions are drawn from: the channels' mean extinction where the water scatters,
// none where it only absorbs (a collision there would carry nothing).
float waterMajorant(WaterVolumeGpu v) {
    let s = waterScattering(v);
    if (max(s.r, max(s.g, s.b)) <= 0.0) {
        return 0.0;
    }
    let e = waterExtinction(v);
    return (e.r + e.g + e.b) / 3.0;
}

// Unpolarised dielectric Fresnel reflectance, light at `cos_i` to the normal on the outside
// (index 1) of a surface of index `ior`.
float waterFresnel(float cos_i, float ior) {
    let c = saturate(cos_i);
    let s2 = (1.0 - c * c) / (ior * ior);
    if (s2 >= 1.0) {
        return 1.0;
    }
    let ct = sqrt(1.0 - s2);
    let rs = (c - ior * ct) / (c + ior * ct);
    let rp = (ior * c - ct) / (ior * c + ct);
    return 0.5 * (rs * rs + rp * rp);
}

// A traced segment from `o` along `d` for `t_max`, never scattering (ReSTIR PT's replay and
// reconnection): the path's own rays refract at the surface mesh themselves, so the length
// inside is the straight one.
public float3 waterTransmittance(UniformData* u, float3 o, float3 d, float t_max) {
    var tau = float3(0.0);
    let len = length(d);
    for (uint i = 0u; i < u.water_volume_count; i++) {
        let v = u.water_volumes[i];
        float t0, t1;
        bool top_in, top_out;
        if (waterSpan(v, o, d, t_max, t0, t1, top_in, top_out)) {
            tau += waterExtinction(v) * ((t1 - t0) * len);
        }
    }
    return exp(-tau);
}

// A shadow ray toward a light from `o` along unit `l` for `t_max`. It passes the surface mesh
// straight, but the light it stands for was bent there: a ray that crosses the surface takes
// the Fresnel transmittance and, inside, the length along the refracted direction for the same
// depth (a flat surface; the refracted path's offset is not traced).
public float3 waterLightTransmittance(UniformData* u, float3 o, float3 l, float t_max) {
    var T = float3(1.0);
    for (uint i = 0u; i < u.water_volume_count; i++) {
        let v = u.water_volumes[i];
        float t0, t1;
        bool top_in, top_out;
        if (!waterSpan(v, o, l, t_max, t0, t1, top_in, top_out)) {
            continue;
        }
        var len = t1 - t0;
        if (top_in || top_out) {
            let cos_air = abs(l.y);
            let cos_water = sqrt(max(1.0 - (1.0 - cos_air * cos_air) / (v.ior * v.ior), 1.0e-6));
            len = cos_air * len / cos_water;
            T *= 1.0 - waterFresnel(cos_air, v.ior);
        }
        T *= exp(-waterExtinction(v) * len);
    }
    return T;
}

// What a segment met in the water: a collision (`scatter`, at `t`, in a medium of phase
// anisotropy `g`) or none; `weight` is the throughput's factor either way, `survive` the
// chance of no collision (ReSTIR PT's search keeps it with its roulette odds).
public struct WaterEvent {
    public bool scatter;
    public float t;
    public float3 weight;
    public float survive;
    public float g;
};

// Free flight from `o` along `d` for `t_max` through every volume it crosses, in order. One
// density for all three channels (each volume's mean extinction): the collision is drawn against
// it, and the weight carries the channels' difference (spectral tracking in a homogeneous
// medium, exact: T / P(no collision) through, sigma_s T / p(t) at a collision). Volumes must not
// overlap.
public WaterEvent waterSegment(UniformData* u, float3 o, float3 d, float t_max, float xi) {
    WaterEvent e;
    e.scatter = false;
    e.t = t_max;
    e.weight = float3(1.0);
    e.survive = 1.0;
    e.g = 0.0;
    let count = u.water_volume_count;
    if (count == 0u) {
        return e;
    }
    let len = length(d);
    // The optical depth (in the drawn density) a collision needs.
    var depth = -log(max(xi, 1.0e-12));
    var tau = float3(0.0);
    var tau_bar = 0.0;
    var after = -1.0;
    for (uint step = 0u; step < count; step++) {
        // The next span along the ray.
        var best = 0xFFFFFFFFu;
        var b0 = 0.0;
        var b1 = 0.0;
        for (uint i = 0u; i < count; i++) {
            float t0, t1;
            bool top_in, top_out;
            if (waterSpan(u.water_volumes[i], o, d, t_max, t0, t1, top_in, top_out) && t0 > after
                && (best == 0xFFFFFFFFu || t0 < b0)) {
                best = i;
                b0 = t0;
                b1 = t1;
            }
        }
        if (best == 0xFFFFFFFFu) {
            break;
        }
        after = b0;
        let v = u.water_volumes[best];
        let sigma_t = waterExtinction(v);
        let sigma_bar = waterMajorant(v);
        let span = (b1 - b0) * len;
        if (sigma_bar > 0.0 && depth < sigma_bar * span) {
            let s = depth / sigma_bar;
            tau += sigma_t * s;
            tau_bar += sigma_bar * s;
            e.scatter = true;
            e.t = b0 + s / len;
            e.weight = exp(-(tau - tau_bar)) * waterScattering(v) / sigma_bar;
            e.survive = 0.0;
            e.g = v.anisotropy;
            return e;
        }
        depth -= sigma_bar * span;
        tau += sigma_t * span;
        tau_bar += sigma_bar * span;
    }
    e.weight = exp(-(tau - tau_bar));
    e.survive = exp(-tau_bar);
    return e;
}

// The weight a segment that did not collide carries (T over the chance of no collision): the
// primary pass's camera segment, as `main` weighs it.
public float3 waterSurvivalWeight(UniformData* u, float3 o, float3 d, float t_max) {
    var tau = float3(0.0);
    let len = length(d);
    for (uint i = 0u; i < u.water_volume_count; i++) {
        let v = u.water_volumes[i];
        float t0, t1;
        bool top_in, top_out;
        if (waterSpan(v, o, d, t_max, t0, t1, top_in, top_out)) {
            tau += (waterExtinction(v) - waterMajorant(v)) * ((t1 - t0) * len);
        }
    }
    return exp(-tau);
}

// Henyey-Greenstein, per steradian; `cos_theta` between the light's travel and the scattered
// light's (dot of the ray's direction and the direction toward the light).
public float waterPhase(float g, float cos_theta) {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PI * denom * sqrt(max(denom, 1.0e-8)));
}

// A direction drawn from waterPhase about the ray's own `incoming` (its pdf is waterPhase).
public float3 waterSamplePhase(float g, float3 incoming, float u1, float u2) {
    float cos_theta;
    if (abs(g) < 1.0e-3) {
        cos_theta = 1.0 - 2.0 * u1;
    } else {
        let term = (1.0 - g * g) / (1.0 + g * (2.0 * u1 - 1.0));
        cos_theta = (1.0 + g * g - term * term) / (2.0 * g);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * u2;
    return mul(float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta),
               orthonormalBasis(incoming));
}
