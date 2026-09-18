// Atmosphere LUT builders and the cloud noise tables (src/atmosphere.rs).
//
// Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering Technique":
// `atmo_transmittance` marches the optical depth to the top of the air for every (radius,
// view zenith), `atmo_multiscatter` folds the second-order in-scatter over 64 directions
// into the infinite-order sum Psi = L2 / (1 - f_ms), and `atmo_skyview` integrates the sky
// as seen from the camera's altitude. `cloud_noise` builds the tiling Perlin-Worley base
// shape and Worley erosion tables once (Schneider, "The Real-time Volumetric Cloudscapes
// of Horizon Zero Dawn").
//
// The runtime side -- sampling these LUTs, aerial perspective, the cloud march -- lives in
// atmosphere.glsl for the GLSL raygen / miss shaders; the medium and the parametrisations
// are the same maths in both files. Lengths in metres, coefficients per metre, radiances
// in nits.

// Must match AtmosphereGpu in src/atmosphere.rs and AtmosphereParams in types.glsl
// (pointers first, then scalars only: no vector alignment questions).
struct AtmosphereParams {
    float4* transmittance;   // T_W x T_H
    float4* multiscatter;    // MS_N x MS_N
    float4* skyview_rad;     // SV_W x SV_H, radiance
    float4* skyview_trn;     // same, transmittance to space
    float* noise;            // NOISE_N^3 Perlin-Worley base shape
    float* noise_detail;     // DETAIL_N^3 Worley erosion
    float planet_center_x;
    float planet_center_y;
    float planet_center_z;
    float planet_radius;
    float atmosphere_height;
    float rayleigh_h;
    float mie_h;
    float ozone_center;
    float ozone_width;
    float rayleigh_scatter_r;
    float rayleigh_scatter_g;
    float rayleigh_scatter_b;
    float mie_scatter_r;
    float mie_scatter_g;
    float mie_scatter_b;
    float mie_absorb_r;
    float mie_absorb_g;
    float mie_absorb_b;
    float ozone_absorb_r;
    float ozone_absorb_g;
    float ozone_absorb_b;
    float mie_g;
    float ground_albedo_r;
    float ground_albedo_g;
    float ground_albedo_b;
    float sun_direction_x;
    float sun_direction_y;
    float sun_direction_z;
    float sun_radiance_r;
    float sun_radiance_g;
    float sun_radiance_b;
    float sun_cos_radius;
    float camera_pos_x;
    float camera_pos_y;
    float camera_pos_z;
    // Cloud fields follow; the kernels do not read them.
};

struct KernelParams {
    AtmosphereParams* atmo;
    uint base;
    uint pad;
};
[[vk::push_constant]] KernelParams kp;

static const float PI = 3.14159265359;
// Must match atmosphere.glsl / src/atmosphere.rs.
static const uint T_W = 256;
static const uint T_H = 64;
static const uint MS_N = 32;
static const uint SV_W = 192;
static const uint SV_H = 108;
static const uint NOISE_N = 128;
static const uint DETAIL_N = 32;

// ---- parameters ------------------------------------------------------------------------

float3 planetCenter() {
    let p = kp.atmo;
    return float3(p.planet_center_x, p.planet_center_y, p.planet_center_z);
}
float3 rayleighScatter() {
    let p = kp.atmo;
    return float3(p.rayleigh_scatter_r, p.rayleigh_scatter_g, p.rayleigh_scatter_b);
}
float3 mieScatter() {
    let p = kp.atmo;
    return float3(p.mie_scatter_r, p.mie_scatter_g, p.mie_scatter_b);
}
float3 mieAbsorb() {
    let p = kp.atmo;
    return float3(p.mie_absorb_r, p.mie_absorb_g, p.mie_absorb_b);
}
float3 ozoneAbsorb() {
    let p = kp.atmo;
    return float3(p.ozone_absorb_r, p.ozone_absorb_g, p.ozone_absorb_b);
}
float3 groundAlbedo() {
    let p = kp.atmo;
    return float3(p.ground_albedo_r, p.ground_albedo_g, p.ground_albedo_b);
}
float3 sunDirection() {
    let p = kp.atmo;
    return float3(p.sun_direction_x, p.sun_direction_y, p.sun_direction_z);
}
float3 sunRadiance() {
    let p = kp.atmo;
    return float3(p.sun_radiance_r, p.sun_radiance_g, p.sun_radiance_b);
}
float3 cameraPos() {
    let p = kp.atmo;
    return float3(p.camera_pos_x, p.camera_pos_y, p.camera_pos_z);
}

// Sun illuminance at the top of the atmosphere (lux): disc radiance times solid angle.
float3 sunIlluminance() {
    return sunRadiance() * (2.0 * PI * (1.0 - kp.atmo.sun_cos_radius));
}

// ---- geometry ----------------------------------------------------------------------------

// Ray / sphere at the origin, numerically stable (Ray Tracing Gems ch. 7). (near, far);
// far < 0 means no hit ahead.
float2 raySphere(float3 o, float3 d, float r) {
    let b = -dot(o, d);
    let s = o + b * d;
    let discr = r * r - dot(s, s);
    if (discr < 0.0) {
        return float2(-1.0, -1.0);
    }
    let sq = sqrt(discr);
    let q = b + (b >= 0.0 ? sq : -sq);
    if (abs(q) < 1.0e-12) {
        return float2(b, b);
    }
    let t1 = (dot(o, o) - r * r) / q;
    let t2 = q;
    return float2(min(t1, t2), max(t1, t2));
}

// Nearest intersection ahead of the origin (-1 = none).
float nearest(float3 o, float3 d, float r) {
    let t = raySphere(o, d, r);
    if (t.y < 0.0) {
        return -1.0;
    }
    return t.x >= 0.0 ? t.x : t.y;
}

// ---- medium ------------------------------------------------------------------------------

void medium(float h, out float3 sr, out float3 sm, out float3 ext) {
    let p = kp.atmo;
    let hh = max(h, 0.0);
    let dr = exp(-hh / p.rayleigh_h);
    let dm = exp(-hh / p.mie_h);
    let doz = max(0.0, 1.0 - abs(hh - p.ozone_center) / p.ozone_width);
    sr = rayleighScatter() * dr;
    sm = mieScatter() * dm;
    ext = sr + (mieScatter() + mieAbsorb()) * dm + ozoneAbsorb() * doz;
}

float phaseRayleigh(float c) {
    return 3.0 / (16.0 * PI) * (1.0 + c * c);
}

// Cornette-Shanks.
float phaseMie(float c, float g) {
    let g2 = g * g;
    let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + c * c) / pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5);
}

// ---- LUT addressing ----------------------------------------------------------------------

float4 lut2D(float4* lut, uint w, uint h, float2 uv) {
    uv = clamp(uv, float2(0.0, 0.0), float2(1.0, 1.0));
    let p = uv * float2(float(w - 1), float(h - 1));
    let i0 = uint2(floor(p));
    let i1 = min(i0 + 1, uint2(w - 1, h - 1));
    let f = p - float2(i0);
    let a = lut[i0.y * w + i0.x];
    let b = lut[i0.y * w + i1.x];
    let c = lut[i1.y * w + i0.x];
    let d = lut[i1.y * w + i1.x];
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// Transmittance LUT parametrisation (Bruneton / Hillaire): (r, mu) <-> uv.
void transUvToParams(float2 uv, out float r, out float mu) {
    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
    let H = sqrt(max(0.0, Rt * Rt - Rg * Rg));
    let rho = H * uv.y;
    r = sqrt(rho * rho + Rg * Rg);
    let d_min = Rt - r;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);
    mu = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

float2 transParamsToUv(float r, float mu) {
    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
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

float3 transmittance(float r, float mu) {
    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
    return lut2D(kp.atmo.transmittance, T_W, T_H, transParamsToUv(clamp(r, Rg, Rt), mu)).rgb;
}

float3 multiScatter(float h, float sun_cos_zenith) {
    let uv = float2(sun_cos_zenith * 0.5 + 0.5, saturate(h / kp.atmo.atmosphere_height));
    return lut2D(kp.atmo.multiscatter, MS_N, MS_N, uv).rgb;
}

// Sky-view LUT mapping (Hillaire): v non-linear around the horizon, u the cosine of the
// azimuth between the view and the sun.
void skyViewUvToParams(float2 uv, float r, out float view_cos_zenith, out float light_view_cos) {
    let Rg = kp.atmo.planet_radius;
    let v_horizon = sqrt(max(0.0, r * r - Rg * Rg));
    let beta = acos(clamp(v_horizon / r, -1.0, 1.0));
    let zenith_horizon = PI - beta;
    if (uv.y < 0.5) {
        var c = 1.0 - 2.0 * uv.y;
        c = 1.0 - c * c;
        view_cos_zenith = cos(zenith_horizon * c);
    } else {
        var c = uv.y * 2.0 - 1.0;
        c *= c;
        view_cos_zenith = cos(zenith_horizon + beta * c);
    }
    let c = uv.x * uv.x;
    light_view_cos = -(c * 2.0 - 1.0);
}

// ---- single-scattering integrator (Hillaire's IntegrateScatteredLuminance) ---------------

// Marches from `p` (planet-relative) along `d` to the top of the air or the ground. `L` is
// the in-scattered radiance, `T` the transmittance, `MS` the integrated scattering (the MS
// kernel's f_ms). `use_ms` adds the multiple-scattering LUT term; `mie_phase` = false is
// the isotropic variant the MS kernel wants; `sun_e` the sun illuminance.
void integrate(float3 p, float3 d, float3 sun, uint steps, bool use_ms, bool mie_phase,
               float3 sun_e, out float3 L, out float3 T, out float3 MS) {
    L = float3(0.0);
    T = float3(1.0);
    MS = float3(0.0);
    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
    if (length(p) > Rt) {
        let tt = raySphere(p, d, Rt);
        if (tt.y < 0.0 || tt.x < 0.0) {
            return;
        }
        p += d * (tt.x + 1.0);
    }
    let t_bottom = nearest(p, d, Rg);
    let t_top = nearest(p, d, Rt);
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
    if (t_max <= 0.0) {
        return;
    }

    let c = dot(d, sun);
    let phase_r = phaseRayleigh(c);
    let phase_m = mie_phase ? phaseMie(c, kp.atmo.mie_g) : 1.0 / (4.0 * PI);
    let phase_iso = 1.0 / (4.0 * PI);

    var t = 0.0;
    for (uint i = 0; i < steps; i++) {
        let t_new = (float(i) + 0.3) / float(steps) * t_max;
        let dt = t_new - t;
        t = t_new;
        let x = p + d * t;
        let r = length(x);
        let up = x / r;
        float3 sr, sm, ext;
        medium(r - Rg, sr, sm, ext);
        let sun_cz = dot(up, sun);
        let t_sun = transmittance(r, sun_cz);
        // The planet's own shadow on the air.
        let shadow = nearest(x, sun, Rg) >= 0.0 ? 0.0 : 1.0;
        var S = sun_e * shadow * t_sun
            * (mie_phase ? (sr * phase_r + sm * phase_m) : ((sr + sm) * phase_iso));
        if (use_ms) {
            S += sun_e * multiScatter(r - Rg, sun_cz) * (sr + sm);
        }
        let t_step = exp(-ext * dt);
        let safe_ext = max(ext, float3(1.0e-12));
        let ms = sr + sm;
        MS += T * (ms - ms * t_step) / safe_ext;
        L += T * (S - S * t_step) / safe_ext;
        T *= t_step;
    }
    if (hits_ground) {
        let x = p + d * t_bottom;
        let up = normalize(x);
        let NoL = max(dot(up, sun), 0.0);
        let t_sun = transmittance(Rg, dot(up, sun));
        L += T * sun_e * t_sun * NoL * groundAlbedo() / PI;
    }
}

// Optical depth to the top of the air / the ground, marched directly (no LUT).
float3 opticalDepth(float3 p, float3 d, uint steps) {
    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
    let t_bottom = nearest(p, d, Rg);
    let t_top = nearest(p, d, Rt);
    float t_max;
    if (t_bottom < 0.0) {
        if (t_top < 0.0) {
            return float3(0.0);
        }
        t_max = t_top;
    } else {
        t_max = t_top > 0.0 ? min(t_top, t_bottom) : t_bottom;
    }
    var tau = float3(0.0);
    var t = 0.0;
    for (uint i = 0; i < steps; i++) {
        let t_new = (float(i) + 0.3) / float(steps) * t_max;
        let dt = t_new - t;
        t = t_new;
        let x = p + d * t;
        float3 sr, sm, ext;
        medium(length(x) - Rg, sr, sm, ext);
        tau += ext * dt;
    }
    return tau;
}

// ---- kernels -----------------------------------------------------------------------------

[shader("compute")]
[numthreads(64, 1, 1)]
void atmo_transmittance(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x + kp.base;
    if (idx >= T_W * T_H) {
        return;
    }
    let xy = uint2(idx % T_W, idx / T_W);
    let uv = (float2(xy) + 0.5) / float2(float(T_W), float(T_H));
    float r, mu;
    transUvToParams(uv, r, mu);
    let p = float3(0.0, r, 0.0);
    let d = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    let tau = opticalDepth(p, d, 40);
    kp.atmo.transmittance[idx] = float4(exp(-tau), 1.0);
}

static const uint SQRT_SAMPLES = 8;

[shader("compute")]
[numthreads(64, 1, 1)]
void atmo_multiscatter(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x + kp.base;
    if (idx >= MS_N * MS_N) {
        return;
    }
    let xy = uint2(idx % MS_N, idx / MS_N);
    let uv = (float2(xy) + 0.5) / float(MS_N);
    let sun_cz = uv.x * 2.0 - 1.0;
    let r = kp.atmo.planet_radius + 1.0 + uv.y * (kp.atmo.atmosphere_height - 2.0);
    let p = float3(0.0, r, 0.0);
    let sun = normalize(float3(sqrt(max(0.0, 1.0 - sun_cz * sun_cz)), sun_cz, 0.0));

    var sum_L = float3(0.0);
    var sum_ms = float3(0.0);
    for (uint i = 0; i < SQRT_SAMPLES; i++) {
        for (uint j = 0; j < SQRT_SAMPLES; j++) {
            let theta = PI * (float(i) + 0.5) / float(SQRT_SAMPLES);
            let phi = acos(1.0 - 2.0 * (float(j) + 0.5) / float(SQRT_SAMPLES));
            let d = float3(cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi));
            float3 L, T, ms;
            integrate(p, d, sun, 20, false, false, float3(1.0), L, T, ms);
            sum_L += L;
            sum_ms += ms;
        }
    }
    let inv = 1.0 / float(SQRT_SAMPLES * SQRT_SAMPLES);
    let L2 = sum_L * inv;
    let f_ms = sum_ms * inv;
    let psi = L2 / max(float3(1.0) - f_ms, float3(1.0e-4));
    kp.atmo.multiscatter[idx] = float4(psi, 1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void atmo_skyview(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x + kp.base;
    if (idx >= SV_W * SV_H) {
        return;
    }
    let xy = uint2(idx % SV_W, idx / SV_W);
    let uv = (float2(xy) + 0.5) / float2(float(SV_W), float(SV_H));

    let Rg = kp.atmo.planet_radius;
    let Rt = Rg + kp.atmo.atmosphere_height;
    let cam = cameraPos() - planetCenter();
    let r = clamp(length(cam), Rg + 1.0, Rt - 1.0);
    let up = normalize(cam);
    let sun_cz = dot(up, sunDirection());

    float vcz, lvc;
    skyViewUvToParams(uv, r, vcz, lvc);
    // The abstract frame: up = +Y, the sun in the XY plane.
    let p = float3(0.0, r, 0.0);
    let sun = normalize(float3(sqrt(max(0.0, 1.0 - sun_cz * sun_cz)), sun_cz, 0.0));
    let vsz = sqrt(max(0.0, 1.0 - vcz * vcz));
    let d = float3(vsz * lvc, vcz, vsz * sqrt(max(0.0, 1.0 - lvc * lvc)));

    float3 L, T, ms;
    integrate(p, d, sun, 32, true, true, sunIlluminance(), L, T, ms);
    kp.atmo.skyview_rad[idx] = float4(L, 1.0);
    kp.atmo.skyview_trn[idx] = float4(T, 1.0);
}

// ---- cloud noise -------------------------------------------------------------------------

uint hashU(uint3 p) {
    var h = p.x * 73856093u ^ p.y * 19349663u ^ p.z * 83492791u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

// Lattice hash wrapped to `period` cells, as three unit floats.
float3 hash3(int3 p, int period) {
    p = ((p % period) + period) % period;
    let h = hashU(uint3(p) + 17u);
    return float3(float(h & 0x3FFu), float((h >> 10) & 0x3FFu), float((h >> 20) & 0x3FFu)) / 1023.0;
}

float remap(float v, float lo, float hi, float nlo, float nhi) {
    return nlo + (v - lo) / max(hi - lo, 1.0e-6) * (nhi - nlo);
}

// Tiling gradient noise, `period` cells per tile, in [0, 1].
float perlin(float3 p, int period) {
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float c[8];
    for (int k = 0; k < 8; k++) {
        let o = int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let g = normalize(hash3(int3(i) + o, period) * 2.0 - 1.0);
        c[k] = dot(g, f - float3(o));
    }
    let x0 = lerp(lerp(c[0], c[1], u.x), lerp(c[2], c[3], u.x), u.y);
    let x1 = lerp(lerp(c[4], c[5], u.x), lerp(c[6], c[7], u.x), u.y);
    return saturate(lerp(x0, x1, u.z) * 0.7 + 0.5);
}

// Tiling inverted Worley noise (1 at feature points), `period` cells per tile.
float worley(float3 p, int period) {
    let i = floor(p);
    let f = p - i;
    var best = 1.0e9;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                let o = float3(x, y, z);
                let fp = o + hash3(int3(i) + int3(x, y, z), period);
                let d = fp - f;
                best = min(best, dot(d, d));
            }
        }
    }
    return 1.0 - saturate(sqrt(best));
}

float worleyFbm(float3 uvw, int freq) {
    return worley(uvw * float(freq), freq) * 0.625 + worley(uvw * float(freq * 2), freq * 2) * 0.25
        + worley(uvw * float(freq * 4), freq * 4) * 0.125;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cloud_noise(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x + kp.base;
    let n_base = NOISE_N * NOISE_N * NOISE_N;
    let n_detail = DETAIL_N * DETAIL_N * DETAIL_N;
    if (idx < n_base) {
        let n = NOISE_N;
        let xyz = uint3(idx % n, (idx / n) % n, idx / (n * n));
        let uvw = (float3(xyz) + 0.5) / float(n);
        let pf = perlin(uvw * 4.0, 4) * 0.5 + perlin(uvw * 8.0, 8) * 0.25
            + perlin(uvw * 16.0, 16) * 0.125 + perlin(uvw * 32.0, 32) * 0.0625;
        let perl = saturate(pf / 0.9375);
        let worl = worleyFbm(uvw, 4);
        // Perlin-Worley: Worley billows where the Perlin field is low, Perlin wisps on top.
        kp.atmo.noise[idx] = saturate(remap(perl, 0.0, 1.0, worl, 1.0));
    } else if (idx < n_base + n_detail) {
        let n = DETAIL_N;
        let j = idx - n_base;
        let xyz = uint3(j % n, (j / n) % n, j / (n * n));
        let uvw = (float3(xyz) + 0.5) / float(n);
        kp.atmo.noise_detail[j] = worleyFbm(uvw, 2);
    }
}
