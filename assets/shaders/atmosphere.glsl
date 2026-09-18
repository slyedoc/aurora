#ifndef H_ATMOSPHERE
#define H_ATMOSPHERE

// Planetary atmosphere + cloud shell (src/atmosphere.rs).
//
// Atmosphere: Hillaire 2020, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique". Three LUTs (transmittance, multiple scattering, sky-view) are rebuilt every
// frame by the atmosphere.slang kernels (the same medium and parametrisations, in Slang);
// the miss shader reads the sky-view LUT, the raygen adds aerial perspective to the primary
// hit and attenuates the sun by the transmittance LUT.
//
// Clouds: one procedural shell between two altitudes, marched deterministically in the
// raygen along the primary ray (Nubis / Frostbite lineage: Perlin-Worley base, Worley
// erosion, Beer transmittance, the multi-octave scattering approximation). The result is
// folded into the noisy colour BEFORE Ray Reconstruction, so nothing here may be jittered
// per frame: a static Bayer offset breaks the banding instead of noise.
//
// Everything works in metres, relative to the planet centre. `pc.atmo` is the
// AtmosphereParams block from types.glsl; every function below reads it.

const float ATM_PI = 3.14159265359;
// Must match src/atmosphere.rs.
const uint ATM_T_W = 256u;
const uint ATM_T_H = 64u;
const uint ATM_MS_N = 32u;
const uint ATM_SV_W = 192u;
const uint ATM_SV_H = 108u;
const uint CLOUD_NOISE_N = 128u;
const uint CLOUD_DETAIL_N = 32u;

#define ATM pc.atmo

// ---- geometry --------------------------------------------------------------------------

// Ray / sphere (centre `c`, radius `r`), numerically stable (Ray Tracing Gems ch. 7).
// Returns (near, far); far < 0 means no hit ahead.
vec2 atmoRaySphere(const vec3 o, const vec3 d, const vec3 c, const float r) {
  const vec3 f = o - c;
  const float b = -dot(f, d);
  const vec3 s = f + b * d;
  const float discr = r * r - dot(s, s);
  if (discr < 0.0) { return vec2(-1.0); }
  const float sq = sqrt(discr);
  const float q = b + (b >= 0.0 ? sq : -sq);
  if (abs(q) < 1.0e-12) { return vec2(b); }
  const float t1 = (dot(f, f) - r * r) / q;
  const float t2 = q;
  return vec2(min(t1, t2), max(t1, t2));
}

// Nearest intersection ahead of the origin (-1 = none), Hillaire's raySphereIntersectNearest.
float atmoNearest(const vec3 o, const vec3 d, const float r) {
  const vec2 t = atmoRaySphere(o, d, vec3(0.0), r);
  if (t.y < 0.0) { return -1.0; }
  return t.x >= 0.0 ? t.x : t.y;
}

// ---- medium -----------------------------------------------------------------------------

// Per-metre scattering (rayleigh, mie) and extinction at altitude `h`.
void atmoMedium(const float h, out vec3 sr, out vec3 sm, out vec3 ext) {
  const float hh = max(h, 0.0);
  const float dr = exp(-hh / ATM.rayleigh_h);
  const float dm = exp(-hh / ATM.mie_h);
  const float doz = max(0.0, 1.0 - abs(hh - ATM.ozone_center) / ATM.ozone_width);
  sr = ATM.rayleigh_scatter * dr;
  sm = ATM.mie_scatter * dm;
  ext = sr + (ATM.mie_scatter + ATM.mie_absorb) * dm + ATM.ozone_absorb * doz;
}

float atmoPhaseRayleigh(const float c) {
  return 3.0 / (16.0 * ATM_PI) * (1.0 + c * c);
}

// Cornette-Shanks.
float atmoPhaseMie(const float c, const float g) {
  const float g2 = g * g;
  const float k = 3.0 / (8.0 * ATM_PI) * (1.0 - g2) / (2.0 + g2);
  return k * (1.0 + c * c) / pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5);
}

// Sun illuminance at the top of the atmosphere (lux): disc radiance times its solid angle.
vec3 atmoSunIlluminance() {
  return ATM.sun_radiance * (2.0 * ATM_PI * (1.0 - ATM.sun_cos_radius));
}

// ---- LUT addressing ----------------------------------------------------------------------

// Bilinear fetch from a W x H vec4 table, uv in [0, 1].
vec4 atmoLut2D(LutData lut, const uint w, const uint h, vec2 uv) {
  uv = clamp(uv, vec2(0.0), vec2(1.0));
  const vec2 p = uv * vec2(float(w - 1u), float(h - 1u));
  const uvec2 i0 = uvec2(floor(p));
  const uvec2 i1 = min(i0 + 1u, uvec2(w - 1u, h - 1u));
  const vec2 f = p - vec2(i0);
  const vec4 a = lut.data[i0.y * w + i0.x];
  const vec4 b = lut.data[i0.y * w + i1.x];
  const vec4 c = lut.data[i1.y * w + i0.x];
  const vec4 d = lut.data[i1.y * w + i1.x];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Transmittance LUT parametrisation (Bruneton / Hillaire): (r, mu) -> uv.
vec2 atmoTransParamsToUv(const float r, const float mu) {
  const float Rg = ATM.planet_radius;
  const float Rt = Rg + ATM.atmosphere_height;
  const float H = sqrt(max(0.0, Rt * Rt - Rg * Rg));
  const float rho = sqrt(max(0.0, r * r - Rg * Rg));
  const float discr = r * r * (mu * mu - 1.0) + Rt * Rt;
  const float d = max(0.0, -r * mu + sqrt(max(discr, 0.0)));
  const float d_min = Rt - r;
  const float d_max = rho + H;
  const float x_mu = d_max > d_min ? (d - d_min) / (d_max - d_min) : 0.0;
  const float x_r = H > 0.0 ? rho / H : 0.0;
  return vec2(x_mu, x_r);
}

// Transmittance from radius `r` to the top of the atmosphere along a direction whose
// cosine with the local up is `mu`.
vec3 atmoTransmittance(const float r, const float mu) {
  const float Rg = ATM.planet_radius;
  const float Rt = Rg + ATM.atmosphere_height;
  return atmoLut2D(ATM.transmittance, ATM_T_W, ATM_T_H,
                   atmoTransParamsToUv(clamp(r, Rg, Rt), mu)).rgb;
}

// Multiple-scattering LUT: (sun cos zenith, altitude fraction) -> Psi_ms (per unit illuminance).
vec3 atmoMultiScatter(const float h, const float sun_cos_zenith) {
  const vec2 uv = vec2(sun_cos_zenith * 0.5 + 0.5,
                       clamp(h / ATM.atmosphere_height, 0.0, 1.0));
  return atmoLut2D(ATM.multiscatter, ATM_MS_N, ATM_MS_N, uv).rgb;
}

// Sky-view LUT mapping (Hillaire): v is non-linear around the horizon, u is the cosine of
// the azimuth between the view and the sun (the sky is symmetric about the sun's azimuth).
vec2 atmoSkyViewParamsToUv(const float r, const float view_cos_zenith,
                           const float light_view_cos, const bool ground) {
  const float Rg = ATM.planet_radius;
  const float v_horizon = sqrt(max(0.0, r * r - Rg * Rg));
  const float beta = acos(clamp(v_horizon / r, -1.0, 1.0));
  const float zenith_horizon = ATM_PI - beta;
  const float zenith = acos(clamp(view_cos_zenith, -1.0, 1.0));
  vec2 uv;
  if (!ground) {
    float c = zenith / zenith_horizon;
    c = 1.0 - sqrt(max(0.0, 1.0 - c));
    uv.y = c * 0.5;
  } else {
    float c = beta > 0.0 ? (zenith - zenith_horizon) / beta : 0.0;
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
void atmoIntegrate(vec3 p, const vec3 d, const vec3 sun, const float t_max_in, const uint steps,
                   const bool use_ms, const bool ground, const vec3 sun_e,
                   out vec3 L, out vec3 T) {
  L = vec3(0.0);
  T = vec3(1.0);
  const float Rg = ATM.planet_radius;
  const float Rt = Rg + ATM.atmosphere_height;
  float t_offset = 0.0;
  if (length(p) > Rt) {
    const vec2 tt = atmoRaySphere(p, d, vec3(0.0), Rt);
    if (tt.y < 0.0 || tt.x < 0.0) { return; }
    t_offset = tt.x + 1.0;
    p += d * t_offset;
  }
  const float t_bottom = atmoNearest(p, d, Rg);
  const float t_top = atmoNearest(p, d, Rt);
  float t_max;
  if (t_bottom < 0.0) {
    if (t_top < 0.0) { return; }
    t_max = t_top;
  } else {
    t_max = t_top > 0.0 ? min(t_top, t_bottom) : t_bottom;
  }
  const bool hits_ground = t_bottom > 0.0 && t_max == t_bottom;
  t_max = min(t_max, max(t_max_in - t_offset, 0.0));
  if (t_max <= 0.0) { return; }

  const float c = dot(d, sun);
  const float phase_r = atmoPhaseRayleigh(c);
  const float phase_m = atmoPhaseMie(c, ATM.mie_g);

  float t = 0.0;
  for (uint i = 0u; i < steps; i++) {
    const float t_new = (float(i) + 0.3) / float(steps) * t_max;
    const float dt = t_new - t;
    t = t_new;
    const vec3 x = p + d * t;
    const float r = length(x);
    const vec3 up = x / r;
    vec3 sr, sm, ext;
    atmoMedium(r - Rg, sr, sm, ext);
    const float sun_cz = dot(up, sun);
    const vec3 t_sun = atmoTransmittance(r, sun_cz);
    // The planet's own shadow on the air.
    const float shadow = atmoNearest(x, sun, Rg) >= 0.0 ? 0.0 : 1.0;
    vec3 S = sun_e * shadow * t_sun * (sr * phase_r + sm * phase_m);
    if (use_ms) {
      S += sun_e * atmoMultiScatter(r - Rg, sun_cz) * (sr + sm);
    }
    const vec3 t_step = exp(-ext * dt);
    L += T * (S - S * t_step) / max(ext, vec3(1.0e-12));
    T *= t_step;
  }
  if (ground && hits_ground && t_bottom <= t_max + 1.0) {
    const vec3 x = p + d * t_bottom;
    const vec3 up = normalize(x);
    const float NoL = max(dot(up, sun), 0.0);
    const vec3 t_sun = atmoTransmittance(Rg, dot(up, sun));
    L += T * sun_e * t_sun * NoL * ATM.ground_albedo / ATM_PI;
  }
}


// ---- runtime lookups ---------------------------------------------------------------------

// Sky radiance (nits) and the transmittance to space along `dir` from the camera, from
// the sky-view LUT. `ground` reports whether the ray hits the planet (no sun disc, and
// the space image stays hidden).
void atmoSkyView(const vec3 dir, out vec3 L, out vec3 T, out bool ground) {
  const vec3 p = ATM.camera_pos - ATM.planet_center;
  const float Rg = ATM.planet_radius;
  const float Rt = Rg + ATM.atmosphere_height;
  const float r = clamp(length(p), Rg + 1.0, Rt - 1.0);
  const vec3 up = normalize(p);
  const float vcz = dot(dir, up);
  const vec3 sun_h = ATM.sun_direction - up * dot(ATM.sun_direction, up);
  const vec3 dir_h = dir - up * vcz;
  const float lh = length(dir_h) * length(sun_h);
  const float lvc = lh > 1.0e-6 ? clamp(dot(dir_h, sun_h) / lh, -1.0, 1.0) : 1.0;
  ground = atmoNearest(up * r, dir, Rg) >= 0.0;
  const vec2 uv = atmoSkyViewParamsToUv(r, vcz, lvc, ground);
  L = atmoLut2D(ATM.skyview_rad, ATM_SV_W, ATM_SV_H, uv).rgb;
  T = ground ? vec3(0.0) : atmoLut2D(ATM.skyview_trn, ATM_SV_W, ATM_SV_H, uv).rgb;
}

// The sun disc seen through the air (nits), for camera / specular paths.
vec3 atmoSunDisc(const vec3 dir, const vec3 T_space) {
  const float c = dot(dir, ATM.sun_direction);
  const float cos_r = ATM.sun_cos_radius;
  const float disc = smoothstep(cos_r - (1.0 - cos_r) * 0.2, cos_r, c);
  return ATM.sun_radiance * disc * T_space;
}

// Transmittance towards the sun from a world point (the atmosphere only).
vec3 atmoSunTransmittanceAt(const vec3 world_pos) {
  const vec3 p = world_pos - ATM.planet_center;
  const float r = length(p);
  if (atmoNearest(p, ATM.sun_direction, ATM.planet_radius) >= 0.0) { return vec3(0.0); }
  return atmoTransmittance(r, dot(p / r, ATM.sun_direction));
}

// Aerial perspective: in-scatter and transmittance over the first `dist` metres of a
// world ray.
void atmoAerial(const vec3 world_pos, const vec3 dir, const float dist, const uint steps,
                out vec3 L, out vec3 T) {
  atmoIntegrate(world_pos - ATM.planet_center, dir, ATM.sun_direction, dist, steps,
                true, false, atmoSunIlluminance(), L, T);
}

// ---- clouds ------------------------------------------------------------------------------

float cloudRemap(const float v, const float lo, const float hi, const float nlo, const float nhi) {
  return nlo + (v - lo) / max(hi - lo, 1.0e-6) * (nhi - nlo);
}

float cloudHG(const float c, const float g) {
  const float g2 = g * g;
  return (1.0 - g2) / (4.0 * ATM_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1.0e-4), 1.5));
}

// Trilinear fetch from a tiling N^3 float table.
float cloudNoise3D(NoiseData table, const uint n, vec3 p) {
  p = p - floor(p / float(n)) * float(n);
  const vec3 i0f = floor(p);
  const vec3 f = p - i0f;
  const uvec3 i0 = uvec3(i0f) % n;
  const uvec3 i1 = (i0 + 1u) % n;
#define CN(ix, iy, iz) table.data[((iz) * n + (iy)) * n + (ix)]
  const float c00 = mix(CN(i0.x, i0.y, i0.z), CN(i1.x, i0.y, i0.z), f.x);
  const float c10 = mix(CN(i0.x, i1.y, i0.z), CN(i1.x, i1.y, i0.z), f.x);
  const float c01 = mix(CN(i0.x, i0.y, i1.z), CN(i1.x, i0.y, i1.z), f.x);
  const float c11 = mix(CN(i0.x, i1.y, i1.z), CN(i1.x, i1.y, i1.z), f.x);
#undef CN
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}

float cloudHash2(const vec2 p) {
  const vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  const float h = dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * h + p3.z * h);
}

float cloudValueNoise2(const vec2 p) {
  const vec2 i = floor(p);
  const vec2 f = p - i;
  const vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(cloudHash2(i), cloudHash2(i + vec2(1.0, 0.0)), u.x),
             mix(cloudHash2(i + vec2(0.0, 1.0)), cloudHash2(i + vec2(1.0, 1.0)), u.x), u.y);
}

// The weather: coverage over the ground, 0..1 (three octaves of value noise).
float cloudCoverageAt(const vec2 xz) {
  const vec2 p = (xz + ATM.cloud_wind.xz * 0.25) / max(ATM.cloud_coverage_scale, 1.0);
  float v = 0.55 * cloudValueNoise2(p) + 0.3 * cloudValueNoise2(p * 2.3 + 7.1)
      + 0.15 * cloudValueNoise2(p * 5.1 + 3.7);
  // The coverage setting slides a soft threshold through the weather field: 0 = clear,
  // 1 = overcast.
  const float edge = 1.0 - ATM.cloud_coverage;
  return smoothstep(edge - 0.3, edge + 0.2, v);
}

// Cumulus profile over the shell's height fraction.
float cloudHeightGradient(const float hf) {
  return smoothstep(0.0, 0.12, hf) * (1.0 - smoothstep(0.45, 1.0, hf));
}

// Extinction density (unitless 0..1 times cloud_density) at planet-relative `x`.
float cloudDensity(const vec3 x, const float hf, const bool detail) {
  const vec3 w = x + ATM.planet_center + ATM.cloud_wind;
  const float cov = cloudCoverageAt(w.xz);
  if (cov <= 0.0) { return 0.0; }
  const float base = cloudNoise3D(ATM.noise, CLOUD_NOISE_N,
                                  w / ATM.cloud_scale * float(CLOUD_NOISE_N));
  const float shape = base * cloudHeightGradient(hf);
  float d = clamp(cloudRemap(shape, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
  if (detail && d > 0.0) {
    const float det = cloudNoise3D(ATM.noise_detail, CLOUD_DETAIL_N,
                                   (w + ATM.cloud_wind * 0.5) / ATM.cloud_detail_scale
                                       * float(CLOUD_DETAIL_N));
    // Wispy at the base, billowy on top.
    const float erode = mix(det, 1.0 - det, clamp(hf * 5.0, 0.0, 1.0));
    d = clamp(cloudRemap(d, erode * ATM.cloud_detail, 1.0, 0.0, 1.0), 0.0, 1.0);
  }
  return d * ATM.cloud_density;
}

// The shell segment [t0, t1] of a planet-relative ray, false if it misses the shell.
bool cloudShell(const vec3 p, const vec3 d, out float t0, out float t1) {
  const float r_in = ATM.planet_radius + ATM.cloud_bottom;
  const float r_out = ATM.planet_radius + ATM.cloud_top;
  const float r = length(p);
  const vec2 ti = atmoRaySphere(p, d, vec3(0.0), r_in);
  const vec2 to = atmoRaySphere(p, d, vec3(0.0), r_out);
  if (r < r_in) {
    // Below: from where the ray leaves the inner sphere to where it leaves the outer.
    if (to.y < 0.0) { return false; }
    t0 = max(ti.y, 0.0);
    t1 = to.y;
  } else if (r <= r_out) {
    // Inside the shell: to the inner sphere if the ray dips into it, else out the top.
    t0 = 0.0;
    t1 = (ti.y >= 0.0 && ti.x > 0.0) ? ti.x : to.y;
  } else {
    // Above: enter through the outer sphere.
    if (to.y < 0.0 || to.x < 0.0) { return false; }
    t0 = to.x;
    t1 = (ti.y >= 0.0 && ti.x > 0.0) ? ti.x : to.y;
  }
  return t1 > t0;
}

float cloudHeightFraction(const vec3 x) {
  return (length(x) - (ATM.planet_radius + ATM.cloud_bottom))
      / max(ATM.cloud_top - ATM.cloud_bottom, 1.0);
}

// Transmittance of the shell towards the sun from a world point (cloud shadows).
float cloudShadow(const vec3 world_pos) {
  if (ATM.clouds == 0u || ATM.cloud_shadows == 0u) { return 1.0; }
  const vec3 p = world_pos - ATM.planet_center;
  float t0, t1;
  if (!cloudShell(p, ATM.sun_direction, t0, t1)) { return 1.0; }
  // A grazing sun runs a long way through the shell; three thicknesses is plenty.
  const float seg = min(t1 - t0, 3.0 * (ATM.cloud_top - ATM.cloud_bottom));
  const uint n = 6u;
  const float dt = seg / float(n);
  float tau = 0.0;
  for (uint i = 0u; i < n; i++) {
    const vec3 x = p + ATM.sun_direction * (t0 + (float(i) + 0.5) * dt);
    const float hf = cloudHeightFraction(x);
    if (hf < 0.0 || hf > 1.0) { continue; }
    // Eroded density: the base shape alone over-darkens the ground under a deck.
    tau += cloudDensity(x, hf, true) * dt;
  }
  return exp(-tau * ATM.cloud_extinction);
}

// Marches the shell along a world ray up to `hit_dist` (<= 0: no surface). `L` is the
// cloud radiance (nits), `T` its transmittance, `front` the distance to the first cloud
// (-1 when none) for the aerial perspective on it. `dither` in [0, 1) offsets the samples
// (static per pixel).
void cloudMarch(const vec3 world_pos, const vec3 dir, const float hit_dist, const float dither,
                out vec3 L, out float T, out float front) {
  L = vec3(0.0);
  T = 1.0;
  front = -1.0;
  if (ATM.clouds == 0u) { return; }
  const vec3 p = world_pos - ATM.planet_center;
  float t0, t1;
  if (!cloudShell(p, dir, t0, t1)) { return; }
  if (hit_dist > 0.0) { t1 = min(t1, hit_dist); }
  if (t1 <= t0) { return; }
  // Two tiers: three quarters of the samples over the near range (fine steps, t ~ f^1.5),
  // the rest spread linearly out to the shell's exit so a distant deck reaches the horizon
  // instead of ending in a band of clear sky.
  const float near_seg = min(t1 - t0, ATM.cloud_max_dist);
  const float far_end = min(t1, t0 + 6.0 * ATM.cloud_max_dist);

  // Per-pixel lighting constants: the sun through the air at the shell, the sky above it.
  const vec3 x0 = p + dir * t0;
  const float r0 = length(x0);
  const vec3 up0 = x0 / r0;
  const vec3 sun_e = atmoSunIlluminance() * atmoTransmittance(r0, dot(up0, ATM.sun_direction));
  vec3 sky_L, sky_T;
  bool sky_ground;
  atmoSkyView(up0, sky_L, sky_T, sky_ground);
  const vec3 ambient = sky_L * ATM.cloud_ambient;
  const float cos_theta = dot(dir, ATM.sun_direction);
  const float albedo = 0.95;

  const uint n = ATM.cloud_steps;
  const uint n_near = max(n * 3u / 4u, 1u);
  const uint n_far = far_end > t0 + near_seg + 1.0 ? n - n_near : 0u;
  const uint ln = ATM.cloud_light_steps;
  const float light_dt = ATM.cloud_light_dist / float(ln);
  float sum_t = 0.0;
  float sum_w = 0.0;
  for (uint i = 0u; i < n_near + n_far; i++) {
    float ta, tb;
    if (i < n_near) {
      const float fa = float(i) / float(n_near);
      const float fb = float(i + 1u) / float(n_near);
      ta = t0 + near_seg * fa * sqrt(fa);
      tb = t0 + near_seg * fb * sqrt(fb);
    } else {
      const float fa = float(i - n_near) / float(n_far);
      const float fb = float(i - n_near + 1u) / float(n_far);
      ta = mix(t0 + near_seg, far_end, fa);
      tb = mix(t0 + near_seg, far_end, fb);
    }
    const float t = mix(ta, tb, dither);
    const float dt = tb - ta;
    const vec3 x = p + dir * t;
    const float hf = cloudHeightFraction(x);
    if (hf < 0.0 || hf > 1.0) { continue; }
    const float dens = cloudDensity(x, hf, true);
    if (dens <= 0.0) { continue; }

    // Sun: a short march towards it (base shape only).
    float tau = 0.0;
    for (uint j = 0u; j < ln; j++) {
      const vec3 xs = x + ATM.sun_direction * ((float(j) + 0.5) * light_dt);
      const float hfs = cloudHeightFraction(xs);
      if (hfs < 0.0 || hfs > 1.0) { break; }
      tau += cloudDensity(xs, hfs, false) * light_dt;
    }
    tau *= ATM.cloud_extinction;
    // Multi-octave scattering approximation (Frostbite): attenuated, wider octaves.
    float sun_l = 0.0;
    float a = 1.0, b = 1.0, c = 1.0;
    for (uint k = 0u; k < 3u; k++) {
      const float ph = mix(cloudHG(cos_theta, ATM.cloud_forward_g * c),
                           cloudHG(cos_theta, ATM.cloud_back_g * c), 0.5);
      sun_l += a * exp(-tau * b) * ph;
      a *= 0.5;
      b *= 0.5;
      c *= 0.5;
    }
    // Powder: the multiple-scatter brightening at cloud edges facing the sun.
    const float powder = 1.0 - 0.5 * exp(-dens * ATM.cloud_extinction * 60.0);

    const float sigma_t = dens * ATM.cloud_extinction;
    const float sigma_s = sigma_t * albedo;
    const vec3 S = sigma_s * (sun_e * sun_l * powder + ambient * mix(0.3, 1.0, hf));
    const float t_step = exp(-sigma_t * dt);
    L += T * (S - S * t_step) / sigma_t;
    const float w = T * (1.0 - t_step);
    sum_t += t * w;
    sum_w += w;
    T *= t_step;
    if (T < 0.005) { break; }
  }
  if (sum_w > 0.0) { front = sum_t / sum_w; }
}

// The full primary-ray composite: clouds and aerial perspective over `surface` (the path
// tracer's radiance for this pixel; `hit_dist` <= 0 means the ray reached the sky, whose
// sky-view radiance already carries its own in-scatter).
vec3 atmoComposite(const vec3 world_pos, const vec3 dir, const float hit_dist, const float dither,
                   const vec3 surface) {
  vec3 col = surface;
  if (hit_dist > 0.0) {
    vec3 La, Ta;
    atmoAerial(world_pos, dir, hit_dist, 12u, La, Ta);
    col = La + Ta * col;
  }
  vec3 Lc;
  float Tc, front;
  cloudMarch(world_pos, dir, hit_dist, dither, Lc, Tc, front);
  if (Tc < 1.0 && front > 0.0) {
    vec3 Lh, Th;
    atmoAerial(world_pos, dir, front, 8u, Lh, Th);
    col = Tc * col + Th * Lc + (1.0 - Tc) * Lh;
  }
  return col;
}

// 4x4 Bayer offset in [0, 1) for a pixel: a static dither for the cloud march.
float atmoBayer(const ivec2 px) {
  const int b[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
  return (float(b[(px.y & 3) * 4 + (px.x & 3)]) + 0.5) / 16.0;
}

#endif
