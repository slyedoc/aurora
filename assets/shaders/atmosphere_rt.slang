// Planetary atmosphere + cloud shell for the raygen (atmosphere.glsl in Slang; the miss
// shader still includes the GLSL one, src/atmosphere.rs).
//
// Atmosphere: Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique". Three LUTs (transmittance, multiple scattering, sky-view) are rebuilt every
// frame by the atmosphere.slang kernels (the same medium and parametrisations); the miss
// shader reads the sky-view LUT, the raygen adds aerial perspective to the primary hit and
// attenuates the sun by the transmittance LUT.
//
// Clouds: one procedural shell between two altitudes, marched deterministically in the
// raygen along the primary ray (Nubis / Frostbite lineage: Perlin-Worley base, Worley
// erosion, Beer transmittance, the multi-octave scattering approximation). The result is
// folded into the noisy colour BEFORE Ray Reconstruction, so nothing here may be jittered
// per frame: a static Bayer offset breaks the banding instead of noise.
//
// Everything works in metres, relative to the planet centre. `pc.atmo` is the
// AtmosphereParams block from rt_types.slang; every function below reads it.
module atmosphere_rt;

import rt_types;

static const float ATM_PI = 3.14159265359;
// Must match src/atmosphere.rs.
static const uint ATM_T_W = 256u;
static const uint ATM_T_H = 64u;
static const uint ATM_MS_N = 32u;
static const uint ATM_SV_W = 192u;
static const uint ATM_SV_H = 108u;
static const uint CLOUD_NOISE_N = 128u;
static const uint CLOUD_DETAIL_N = 32u;

float3 planetCenter() {
    let a = pc.atmo;
    return float3(a.planet_center_x, a.planet_center_y, a.planet_center_z);
}
float3 rayleighScatter() {
    let a = pc.atmo;
    return float3(a.rayleigh_scatter_r, a.rayleigh_scatter_g, a.rayleigh_scatter_b);
}
float3 mieScatter() {
    let a = pc.atmo;
    return float3(a.mie_scatter_r, a.mie_scatter_g, a.mie_scatter_b);
}
float3 mieAbsorb() {
    let a = pc.atmo;
    return float3(a.mie_absorb_r, a.mie_absorb_g, a.mie_absorb_b);
}
float3 ozoneAbsorb() {
    let a = pc.atmo;
    return float3(a.ozone_absorb_r, a.ozone_absorb_g, a.ozone_absorb_b);
}
float3 groundAlbedo() {
    let a = pc.atmo;
    return float3(a.ground_albedo_r, a.ground_albedo_g, a.ground_albedo_b);
}
float3 sunDirection() {
    let a = pc.atmo;
    return float3(a.sun_direction_x, a.sun_direction_y, a.sun_direction_z);
}
float3 sunRadiance() {
    let a = pc.atmo;
    return float3(a.sun_radiance_r, a.sun_radiance_g, a.sun_radiance_b);
}
float3 cameraPos() {
    let a = pc.atmo;
    return float3(a.camera_pos_x, a.camera_pos_y, a.camera_pos_z);
}
float3 cloudWind() {
    let a = pc.atmo;
    return float3(a.cloud_wind_x, a.cloud_wind_y, a.cloud_wind_z);
}

// ---- geometry --------------------------------------------------------------------------

// Ray / sphere (centre `c`, radius `r`), numerically stable (Ray Tracing Gems ch. 7).
// Returns (near, far); far < 0 means no hit ahead.
float2 atmoRaySphere(float3 o, float3 d, float3 c, float r) {
    let f = o - c;
    let b = -dot(f, d);
    let s = f + b * d;
    let discr = r * r - dot(s, s);
    if (discr < 0.0) {
        return float2(-1.0);
    }
    let sq = sqrt(discr);
    let q = b + (b >= 0.0 ? sq : -sq);
    if (abs(q) < 1.0e-12) {
        return float2(b);
    }
    let t1 = (dot(f, f) - r * r) / q;
    let t2 = q;
    return float2(min(t1, t2), max(t1, t2));
}

// Nearest intersection ahead of the origin (-1 = none), Hillaire's raySphereIntersectNearest.
float atmoNearest(float3 o, float3 d, float r) {
    let t = atmoRaySphere(o, d, float3(0.0), r);
    if (t.y < 0.0) {
        return -1.0;
    }
    return t.x >= 0.0 ? t.x : t.y;
}

// ---- medium -----------------------------------------------------------------------------

// Per-metre scattering (rayleigh, mie) and extinction at altitude `h`.
void atmoMedium(float h, out float3 sr, out float3 sm, out float3 ext) {
    let a = pc.atmo;
    let hh = max(h, 0.0);
    let dr = exp(-hh / a.rayleigh_h);
    let dm = exp(-hh / a.mie_h);
    let doz = max(0.0, 1.0 - abs(hh - a.ozone_center) / a.ozone_width);
    sr = rayleighScatter() * dr;
    sm = mieScatter() * dm;
    ext = sr + (mieScatter() + mieAbsorb()) * dm + ozoneAbsorb() * doz;
}

float atmoPhaseRayleigh(float c) {
    return 3.0 / (16.0 * ATM_PI) * (1.0 + c * c);
}

// Cornette-Shanks.
float atmoPhaseMie(float c, float g) {
    let g2 = g * g;
    let k = 3.0 / (8.0 * ATM_PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + c * c) / pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5);
}

// Sun illuminance at the top of the atmosphere (lux): disc radiance times its solid angle.
float3 atmoSunIlluminance() {
    return sunRadiance() * (2.0 * ATM_PI * (1.0 - pc.atmo.sun_cos_radius));
}

// ---- LUT addressing ----------------------------------------------------------------------

// Bilinear fetch from a W x H float4 table, uv in [0, 1].
float4 atmoLut2D(float4* lut, uint w, uint h, float2 uv) {
    uv = clamp(uv, float2(0.0), float2(1.0));
    let p = uv * float2(float(w - 1u), float(h - 1u));
    let i0 = uint2(floor(p));
    let i1 = min(i0 + 1u, uint2(w - 1u, h - 1u));
    let f = p - float2(i0);
    let a = lut[i0.y * w + i0.x];
    let b = lut[i0.y * w + i1.x];
    let c = lut[i1.y * w + i0.x];
    let d = lut[i1.y * w + i1.x];
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Transmittance LUT parametrisation (Bruneton / Hillaire): (r, mu) -> uv.
float2 atmoTransParamsToUv(float r, float mu) {
    let Rg = pc.atmo.planet_radius;
    let Rt = Rg + pc.atmo.atmosphere_height;
    let H = sqrt(max(0.0, Rt * Rt - Rg * Rg));
    let rho = sqrt(max(0.0, r * r - Rg * Rg));
    let discr = r * r * (mu * mu - 1.0) + Rt * Rt;
    let d = max(0.0, -r * mu + sqrt(max(discr, 0.0)));
    let d_min = Rt - r;
    let d_max = rho + H;
    let x_mu = d_max > d_min ? (d - d_min) / (d_max - d_min) : 0.0;
    let x_r = H > 0.0 ? rho / H : 0.0;
    return float2(x_mu, x_r);
}

// Transmittance from radius `r` to the top of the atmosphere along a direction whose
// cosine with the local up is `mu`.
float3 atmoTransmittance(float r, float mu) {
    let Rg = pc.atmo.planet_radius;
    let Rt = Rg + pc.atmo.atmosphere_height;
    return atmoLut2D(pc.atmo.transmittance, ATM_T_W, ATM_T_H,
                     atmoTransParamsToUv(clamp(r, Rg, Rt), mu)).rgb;
}

// Multiple-scattering LUT: (sun cos zenith, altitude fraction) -> Psi_ms (per unit illuminance).
float3 atmoMultiScatter(float h, float sun_cos_zenith) {
    let uv = float2(sun_cos_zenith * 0.5 + 0.5,
                    clamp(h / pc.atmo.atmosphere_height, 0.0, 1.0));
    return atmoLut2D(pc.atmo.multiscatter, ATM_MS_N, ATM_MS_N, uv).rgb;
}

// Sky-view LUT mapping (Hillaire): v is non-linear around the horizon, u is the cosine of
// the azimuth between the view and the sun (the sky is symmetric about the sun's azimuth).
float2 atmoSkyViewParamsToUv(float r, float view_cos_zenith, float light_view_cos, bool ground) {
    let Rg = pc.atmo.planet_radius;
    let v_horizon = sqrt(max(0.0, r * r - Rg * Rg));
    let beta = acos(clamp(v_horizon / r, -1.0, 1.0));
    let zenith_horizon = ATM_PI - beta;
    let zenith = acos(clamp(view_cos_zenith, -1.0, 1.0));
    float2 uv;
    if (!ground) {
        var c = zenith / zenith_horizon;
        c = 1.0 - sqrt(max(0.0, 1.0 - c));
        uv.y = c * 0.5;
    } else {
        let c = beta > 0.0 ? (zenith - zenith_horizon) / beta : 0.0;
        uv.y = sqrt(clamp(c, 0.0, 1.0)) * 0.5 + 0.5;
    }
    uv.x = sqrt(clamp(-light_view_cos * 0.5 + 0.5, 0.0, 1.0));
    return uv;
}

// ---- single-scattering integrator (Hillaire's IntegrateScatteredLuminance) ---------------

// Marches from `p` (planet-relative) along `d` up to `t_max` metres, the atmosphere's top,
// or the ground, whichever is nearest. `L` is the in-scattered radiance (nits), `T` the
// transmittance. `use_ms` adds the multiple-scattering LUT term; `ground` adds the lit
// ground at a planet hit; `sun_e` is the sun illuminance.
void atmoIntegrate(float3 p, float3 d, float3 sun, float t_max_in, uint steps,
                   bool use_ms, bool ground, float3 sun_e,
                   out float3 L, out float3 T) {
    L = float3(0.0);
    T = float3(1.0);
    let Rg = pc.atmo.planet_radius;
    let Rt = Rg + pc.atmo.atmosphere_height;
    var t_offset = 0.0;
    if (length(p) > Rt) {
        let tt = atmoRaySphere(p, d, float3(0.0), Rt);
        if (tt.y < 0.0 || tt.x < 0.0) {
            return;
        }
        t_offset = tt.x + 1.0;
        p += d * t_offset;
    }
    let t_bottom = atmoNearest(p, d, Rg);
    let t_top = atmoNearest(p, d, Rt);
    float t_max;
    if (t_bottom < 0.0) {
        if (t_top < 0.0) {
            return;
        }
        t_max = t_top;
    } else {
        t_max = t_top > 0.0 ? min(t_top, t_bottom) : t_bottom;
    }
    let hits_ground = t_bottom > 0.0 && t_max == t_bottom;
    t_max = min(t_max, max(t_max_in - t_offset, 0.0));
    if (t_max <= 0.0) {
        return;
    }

    let c = dot(d, sun);
    let phase_r = atmoPhaseRayleigh(c);
    let phase_m = atmoPhaseMie(c, pc.atmo.mie_g);

    var t = 0.0;
    for (uint i = 0u; i < steps; i++) {
        let t_new = (float(i) + 0.3) / float(steps) * t_max;
        let dt = t_new - t;
        t = t_new;
        let x = p + d * t;
        let r = length(x);
        let up = x / r;
        float3 sr, sm, ext;
        atmoMedium(r - Rg, sr, sm, ext);
        let sun_cz = dot(up, sun);
        let t_sun = atmoTransmittance(r, sun_cz);
        // The planet's own shadow on the air.
        let shadow = atmoNearest(x, sun, Rg) >= 0.0 ? 0.0 : 1.0;
        var S = sun_e * shadow * t_sun * (sr * phase_r + sm * phase_m);
        if (use_ms) {
            S += sun_e * atmoMultiScatter(r - Rg, sun_cz) * (sr + sm);
        }
        let t_step = exp(-ext * dt);
        L += T * (S - S * t_step) / max(ext, float3(1.0e-12));
        T *= t_step;
    }
    if (ground && hits_ground && t_bottom <= t_max + 1.0) {
        let x = p + d * t_bottom;
        let up = normalize(x);
        let NoL = max(dot(up, sun), 0.0);
        let t_sun = atmoTransmittance(Rg, dot(up, sun));
        L += T * sun_e * t_sun * NoL * groundAlbedo() / ATM_PI;
    }
}

// ---- runtime lookups ---------------------------------------------------------------------

// Sky radiance (nits) and the transmittance to space along `dir` from the camera, from
// the sky-view LUT. `ground` reports whether the ray hits the planet (no sun disc, and
// the space image stays hidden).
void atmoSkyView(float3 dir, out float3 L, out float3 T, out bool ground) {
    let p = cameraPos() - planetCenter();
    let Rg = pc.atmo.planet_radius;
    let Rt = Rg + pc.atmo.atmosphere_height;
    let r = clamp(length(p), Rg + 1.0, Rt - 1.0);
    let up = normalize(p);
    let vcz = dot(dir, up);
    let sun_dir = sunDirection();
    let sun_h = sun_dir - up * dot(sun_dir, up);
    let dir_h = dir - up * vcz;
    let lh = length(dir_h) * length(sun_h);
    let lvc = lh > 1.0e-6 ? clamp(dot(dir_h, sun_h) / lh, -1.0, 1.0) : 1.0;
    ground = atmoNearest(up * r, dir, Rg) >= 0.0;
    let uv = atmoSkyViewParamsToUv(r, vcz, lvc, ground);
    L = atmoLut2D(pc.atmo.skyview_rad, ATM_SV_W, ATM_SV_H, uv).rgb;
    T = ground ? float3(0.0) : atmoLut2D(pc.atmo.skyview_trn, ATM_SV_W, ATM_SV_H, uv).rgb;
}

// Transmittance towards the sun from a world point (the atmosphere only).
public float3 atmoSunTransmittanceAt(float3 world_pos) {
    let p = world_pos - planetCenter();
    let r = length(p);
    if (atmoNearest(p, sunDirection(), pc.atmo.planet_radius) >= 0.0) {
        return float3(0.0);
    }
    return atmoTransmittance(r, dot(p / r, sunDirection()));
}

// Aerial perspective: in-scatter and transmittance over the first `dist` metres of a
// world ray.
void atmoAerial(float3 world_pos, float3 dir, float dist, uint steps, out float3 L, out float3 T) {
    atmoIntegrate(world_pos - planetCenter(), dir, sunDirection(), dist, steps,
                  true, false, atmoSunIlluminance(), L, T);
}

// ---- clouds ------------------------------------------------------------------------------

float cloudRemap(float v, float lo, float hi, float nlo, float nhi) {
    return nlo + (v - lo) / max(hi - lo, 1.0e-6) * (nhi - nlo);
}

float cloudHG(float c, float g) {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * ATM_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5));
}

float cloudNoiseAt(float* table, uint n, uint ix, uint iy, uint iz) {
    return table[(iz * n + iy) * n + ix];
}

// Trilinear fetch from a tiling N^3 float table.
float cloudNoise3D(float* table, uint n, float3 p) {
    p = p - floor(p / float(n)) * float(n);
    let i0f = floor(p);
    let f = p - i0f;
    let i0 = uint3(i0f) % n;
    let i1 = (i0 + 1u) % n;
    let c00 = lerp(cloudNoiseAt(table, n, i0.x, i0.y, i0.z), cloudNoiseAt(table, n, i1.x, i0.y, i0.z), f.x);
    let c10 = lerp(cloudNoiseAt(table, n, i0.x, i1.y, i0.z), cloudNoiseAt(table, n, i1.x, i1.y, i0.z), f.x);
    let c01 = lerp(cloudNoiseAt(table, n, i0.x, i0.y, i1.z), cloudNoiseAt(table, n, i1.x, i0.y, i1.z), f.x);
    let c11 = lerp(cloudNoiseAt(table, n, i0.x, i1.y, i1.z), cloudNoiseAt(table, n, i1.x, i1.y, i1.z), f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

float cloudHash2(float2 p) {
    let p3 = frac(float3(p.xyx) * 0.1031);
    let h = dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * h + p3.z * h);
}

float cloudValueNoise2(float2 p) {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(cloudHash2(i), cloudHash2(i + float2(1.0, 0.0)), u.x),
                lerp(cloudHash2(i + float2(0.0, 1.0)), cloudHash2(i + float2(1.0, 1.0)), u.x), u.y);
}

// The weather: coverage over the ground, 0..1 (three octaves of value noise).
float cloudCoverageAt(float2 xz) {
    let p = (xz + cloudWind().xz * 0.25) / max(pc.atmo.cloud_coverage_scale, 1.0);
    let v = 0.55 * cloudValueNoise2(p) + 0.3 * cloudValueNoise2(p * 2.3 + 7.1)
        + 0.15 * cloudValueNoise2(p * 5.1 + 3.7);
    // The coverage setting slides a soft threshold through the weather field: 0 = clear,
    // 1 = overcast.
    let edge = 1.0 - pc.atmo.cloud_coverage;
    return smoothstep(edge - 0.3, edge + 0.2, v);
}

// Cumulus profile over the shell's height fraction.
float cloudHeightGradient(float hf) {
    return smoothstep(0.0, 0.12, hf) * (1.0 - smoothstep(0.45, 1.0, hf));
}

// Extinction density (unitless 0..1 times cloud_density) at planet-relative `x`.
float cloudDensity(float3 x, float hf, bool detail) {
    let a = pc.atmo;
    let w = x + planetCenter() + cloudWind();
    let cov = cloudCoverageAt(w.xz);
    if (cov <= 0.0) {
        return 0.0;
    }
    let base = cloudNoise3D(a.noise, CLOUD_NOISE_N, w / a.cloud_scale * float(CLOUD_NOISE_N));
    let shape = base * cloudHeightGradient(hf);
    var d = clamp(cloudRemap(shape, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
    if (detail && d > 0.0) {
        let det = cloudNoise3D(a.noise_detail, CLOUD_DETAIL_N,
                               (w + cloudWind() * 0.5) / a.cloud_detail_scale
                                   * float(CLOUD_DETAIL_N));
        // Wispy at the base, billowy on top.
        let erode = lerp(det, 1.0 - det, clamp(hf * 5.0, 0.0, 1.0));
        d = clamp(cloudRemap(d, erode * a.cloud_detail, 1.0, 0.0, 1.0), 0.0, 1.0);
    }
    return d * a.cloud_density;
}

// The shell segment [t0, t1] of a planet-relative ray, false if it misses the shell.
bool cloudShell(float3 p, float3 d, out float t0, out float t1) {
    t0 = 0.0;
    t1 = 0.0;
    let r_in = pc.atmo.planet_radius + pc.atmo.cloud_bottom;
    let r_out = pc.atmo.planet_radius + pc.atmo.cloud_top;
    let r = length(p);
    let ti = atmoRaySphere(p, d, float3(0.0), r_in);
    let to = atmoRaySphere(p, d, float3(0.0), r_out);
    if (r < r_in) {
        // Below: from where the ray leaves the inner sphere to where it leaves the outer.
        if (to.y < 0.0) {
            return false;
        }
        t0 = max(ti.y, 0.0);
        t1 = to.y;
    } else if (r <= r_out) {
        // Inside the shell: to the inner sphere if the ray dips into it, else out the top.
        t0 = 0.0;
        t1 = (ti.y >= 0.0 && ti.x > 0.0) ? ti.x : to.y;
    } else {
        // Above: enter through the outer sphere.
        if (to.y < 0.0 || to.x < 0.0) {
            return false;
        }
        t0 = to.x;
        t1 = (ti.y >= 0.0 && ti.x > 0.0) ? ti.x : to.y;
    }
    return t1 > t0;
}

float cloudHeightFraction(float3 x) {
    return (length(x) - (pc.atmo.planet_radius + pc.atmo.cloud_bottom))
        / max(pc.atmo.cloud_top - pc.atmo.cloud_bottom, 1.0);
}

// Transmittance of the shell towards the sun from a world point (cloud shadows).
public float cloudShadow(float3 world_pos) {
    let a = pc.atmo;
    if (a.clouds == 0u || a.cloud_shadows == 0u) {
        return 1.0;
    }
    let p = world_pos - planetCenter();
    let sun = sunDirection();
    float t0, t1;
    if (!cloudShell(p, sun, t0, t1)) {
        return 1.0;
    }
    // A grazing sun runs a long way through the shell; three thicknesses is plenty.
    let seg = min(t1 - t0, 3.0 * (a.cloud_top - a.cloud_bottom));
    let n = 6u;
    let dt = seg / float(n);
    var tau = 0.0;
    for (uint i = 0u; i < n; i++) {
        let x = p + sun * (t0 + (float(i) + 0.5) * dt);
        let hf = cloudHeightFraction(x);
        if (hf < 0.0 || hf > 1.0) {
            continue;
        }
        // Eroded density: the base shape alone over-darkens the ground under a deck.
        tau += cloudDensity(x, hf, true) * dt;
    }
    return exp(-tau * a.cloud_extinction);
}

// Marches the shell along a world ray up to `hit_dist` (<= 0: no surface). `L` is the
// cloud radiance (nits), `T` its transmittance, `front` the distance to the first cloud
// (-1 when none) for the aerial perspective on it. `dither` in [0, 1) offsets the samples
// (static per pixel).
void cloudMarch(float3 world_pos, float3 dir, float hit_dist, float dither,
                out float3 L, out float T, out float front) {
    L = float3(0.0);
    T = 1.0;
    front = -1.0;
    let a = pc.atmo;
    if (a.clouds == 0u) {
        return;
    }
    let p = world_pos - planetCenter();
    float t0, t1;
    if (!cloudShell(p, dir, t0, t1)) {
        return;
    }
    if (hit_dist > 0.0) {
        t1 = min(t1, hit_dist);
    }
    if (t1 <= t0) {
        return;
    }
    // Two tiers: three quarters of the samples over the near range (fine steps, t ~ f^1.5),
    // the rest spread linearly out to the shell's exit so a distant deck reaches the horizon
    // instead of ending in a band of clear sky.
    let near_seg = min(t1 - t0, a.cloud_max_dist);
    let far_end = min(t1, t0 + 6.0 * a.cloud_max_dist);
    let sun = sunDirection();

    // Per-pixel lighting constants: the sun through the air at the shell, the sky above it.
    let x0 = p + dir * t0;
    let r0 = length(x0);
    let up0 = x0 / r0;
    let sun_e = atmoSunIlluminance() * atmoTransmittance(r0, dot(up0, sun));
    float3 sky_L, sky_T;
    bool sky_ground;
    atmoSkyView(up0, sky_L, sky_T, sky_ground);
    let ambient = sky_L * a.cloud_ambient;
    let cos_theta = dot(dir, sun);
    let albedo = 0.95;

    let n = a.cloud_steps;
    let n_near = max(n * 3u / 4u, 1u);
    let n_far = far_end > t0 + near_seg + 1.0 ? n - n_near : 0u;
    let ln = a.cloud_light_steps;
    let light_dt = a.cloud_light_dist / float(ln);
    var sum_t = 0.0;
    var sum_w = 0.0;
    for (uint i = 0u; i < n_near + n_far; i++) {
        float ta, tb;
        if (i < n_near) {
            let fa = float(i) / float(n_near);
            let fb = float(i + 1u) / float(n_near);
            ta = t0 + near_seg * fa * sqrt(fa);
            tb = t0 + near_seg * fb * sqrt(fb);
        } else {
            let fa = float(i - n_near) / float(n_far);
            let fb = float(i - n_near + 1u) / float(n_far);
            ta = lerp(t0 + near_seg, far_end, fa);
            tb = lerp(t0 + near_seg, far_end, fb);
        }
        let t = lerp(ta, tb, dither);
        let dt = tb - ta;
        let x = p + dir * t;
        let hf = cloudHeightFraction(x);
        if (hf < 0.0 || hf > 1.0) {
            continue;
        }
        let dens = cloudDensity(x, hf, true);
        if (dens <= 0.0) {
            continue;
        }

        // Sun: a short march towards it (base shape only).
        var tau = 0.0;
        for (uint j = 0u; j < ln; j++) {
            let xs = x + sun * ((float(j) + 0.5) * light_dt);
            let hfs = cloudHeightFraction(xs);
            if (hfs < 0.0 || hfs > 1.0) {
                break;
            }
            tau += cloudDensity(xs, hfs, false) * light_dt;
        }
        tau *= a.cloud_extinction;
        // Multi-octave scattering approximation (Frostbite): attenuated, wider octaves.
        var sun_l = 0.0;
        float oa = 1.0, ob = 1.0, oc = 1.0;
        for (uint k = 0u; k < 3u; k++) {
            let ph = lerp(cloudHG(cos_theta, a.cloud_forward_g * oc),
                          cloudHG(cos_theta, a.cloud_back_g * oc), 0.5);
            sun_l += oa * exp(-tau * ob) * ph;
            oa *= 0.5;
            ob *= 0.5;
            oc *= 0.5;
        }
        // Powder: the multiple-scatter brightening at cloud edges facing the sun.
        let powder = 1.0 - 0.5 * exp(-dens * a.cloud_extinction * 60.0);

        let sigma_t = dens * a.cloud_extinction;
        let sigma_s = sigma_t * albedo;
        let S = sigma_s * (sun_e * sun_l * powder + ambient * lerp(0.3, 1.0, hf));
        let t_step = exp(-sigma_t * dt);
        L += T * (S - S * t_step) / sigma_t;
        let w = T * (1.0 - t_step);
        sum_t += t * w;
        sum_w += w;
        T *= t_step;
        if (T < 0.005) {
            break;
        }
    }
    if (sum_w > 0.0) {
        front = sum_t / sum_w;
    }
}

// The full primary-ray composite: clouds and aerial perspective over `surface` (the path
// tracer's radiance for this pixel; `hit_dist` <= 0 means the ray reached the sky, whose
// sky-view radiance already carries its own in-scatter).
public float3 atmoComposite(float3 world_pos, float3 dir, float hit_dist, float dither, float3 surface) {
    var col = surface;
    if (hit_dist > 0.0) {
        float3 La, Ta;
        atmoAerial(world_pos, dir, hit_dist, 12u, La, Ta);
        col = La + Ta * col;
    }
    float3 Lc;
    float Tc, front;
    cloudMarch(world_pos, dir, hit_dist, dither, Lc, Tc, front);
    if (Tc < 1.0 && front > 0.0) {
        float3 Lh, Th;
        atmoAerial(world_pos, dir, front, 8u, Lh, Th);
        col = Tc * col + Th * Lc + (1.0 - Tc) * Lh;
    }
    return col;
}

// 4x4 Bayer offset in [0, 1) for a pixel: a static dither for the cloud march.
static const int ATM_BAYER[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

public float atmoBayer(int2 px) {
    return (float(ATM_BAYER[(px.y & 3) * 4 + (px.x & 3)]) + 0.5) / 16.0;
}
