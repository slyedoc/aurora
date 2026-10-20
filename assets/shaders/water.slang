// Water media (src/water.rs): each water surface's material names a homogeneous medium below it.
// A path tracks which water it is in (a water id: medium row + 1, 0 none) by its crossings of the
// surfaces (raygen.slang); one medium at a time (no nesting), entered from above only.
module water;

import rt_types;

// The water id's medium, or false for none (or a row past the table).
bool waterMedium(UniformData* u, uint water, out WaterMediumGpu v) {
    if (water == 0u || water > u.water_medium_count) {
        v = {};
        return false;
    }
    v = u.water_media[water - 1u];
    return true;
}

float3 waterScattering(WaterMediumGpu v) {
    return float3(v.scattering_r, v.scattering_g, v.scattering_b);
}

// Extinction per metre: absorption plus scattering.
float3 waterExtinction(WaterMediumGpu v) {
    return float3(v.absorption_r, v.absorption_g, v.absorption_b) + waterScattering(v);
}

// The density collisions are drawn from: the channels' mean extinction where the water scatters,
// none where it only absorbs (a collision there would carry nothing).
float waterMajorant(WaterMediumGpu v) {
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

// The water a traced segment ran through: the hit's, when it ended on a water surface from below
// (whatever the path thought: a leak's resync), none when it ended on one from above, else the
// path's own `state`. `hit_water` is the hit's water id, `rising` whether the ray met it from
// below.
public uint waterOfSegment(uint state, uint hit_water, bool rising) {
    if (hit_water != 0u) {
        return rising ? hit_water : 0u;
    }
    return state;
}

// The water after a scatter at a vertex: through a water surface, its water when the new
// direction goes below it, none above; anywhere else the water the path is in.
public uint waterAfterScatter(uint state, uint vertex_water, bool below) {
    if (vertex_water != 0u) {
        return below ? vertex_water : 0u;
    }
    return state;
}

// A straight segment from a point in water `state` along unit `d` for `t_max` that was not traced
// to its end through the surfaces (shadow rays, ReSTIR reconnections): its ray reports the
// nearest water surface it passed (`cross` at `cross_t`, 0 none). Passing it rising, the start
// was in that water up to it; falling, the end is in it from there; passing none, the segment is
// in `state`'s water all along. `bent`: the light the segment stands for was bent at the surface
// (a shadow ray toward a light), so it takes the surface's Fresnel transmittance and, inside, the
// length along the refracted direction for the same depth (a flat surface; the refracted path's
// offset is not traced); else the straight length.
public float3 waterAlong(UniformData* u, uint state, uint cross, float cross_t, float3 d,
                         float t_max, bool bent) {
    WaterMediumGpu v;
    if (cross != 0u && waterMedium(u, cross, v)) {
        let rising = d.y > 0.0;
        var len = rising ? cross_t : max(t_max - cross_t, 0.0);
        var T = float3(1.0);
        if (bent) {
            let cos_air = abs(d.y);
            let cos_water = sqrt(max(1.0 - (1.0 - cos_air * cos_air) / (v.ior * v.ior), 1.0e-6));
            len *= cos_air / cos_water;
            T = float3(1.0 - waterFresnel(cos_air, v.ior));
        }
        return T * exp(-waterExtinction(v) * len);
    }
    if (waterMedium(u, state, v)) {
        return exp(-waterExtinction(v) * t_max);
    }
    return float3(1.0);
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

// Free flight through water `water` along a unit direction for `t_max`. One density for all three
// channels (the mean extinction): the collision is drawn against it, and the weight carries the
// channels' difference (spectral tracking in a homogeneous medium, exact: T / P(no collision)
// through, sigma_s T / p(t) at a collision).
public WaterEvent waterSegment(UniformData* u, uint water, float t_max, float xi) {
    WaterEvent e;
    e.scatter = false;
    e.t = t_max;
    e.weight = float3(1.0);
    e.survive = 1.0;
    e.g = 0.0;
    WaterMediumGpu v;
    if (!waterMedium(u, water, v)) {
        return e;
    }
    let sigma_t = waterExtinction(v);
    let sigma_bar = waterMajorant(v);
    // The optical depth (in the drawn density) a collision needs.
    let depth = -log(max(xi, 1.0e-12));
    if (sigma_bar > 0.0 && depth < sigma_bar * t_max) {
        let s = depth / sigma_bar;
        e.scatter = true;
        e.t = s;
        e.weight = exp(-(sigma_t - sigma_bar) * s) * waterScattering(v) / sigma_bar;
        e.survive = 0.0;
        e.g = v.anisotropy;
        return e;
    }
    e.weight = exp(-(sigma_t - sigma_bar) * t_max);
    e.survive = exp(-sigma_bar * t_max);
    return e;
}

// The weight a segment in water `water` that did not collide carries (T over the chance of no
// collision): the primary pass's camera segment, as `main` weighs it.
public float3 waterSurvivalWeight(UniformData* u, uint water, float t_max) {
    WaterMediumGpu v;
    if (!waterMedium(u, water, v)) {
        return float3(1.0);
    }
    return exp(-(waterExtinction(v) - waterMajorant(v)) * t_max);
}

// A traced segment that never scatters (ReSTIR PT's replay): its transmittance in `water`.
public float3 waterTransmittance(UniformData* u, uint water, float t_max) {
    WaterMediumGpu v;
    if (!waterMedium(u, water, v)) {
        return float3(1.0);
    }
    return exp(-waterExtinction(v) * t_max);
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
