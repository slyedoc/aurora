// Noise primitives for generator layers (aurora_landscape generator.rs): simplex fbm, domain
// warp and ridged multifractal, from zero's planet terrain (simplex/fbm vendored 1:1 from
// noisy_bevy, MIT, a webgl-noise port).
module noise;

float4 permute_4_(float4 x)
{
    return ((x * 34.0 + 1.0) * x) % float4(289.0);
}

float4 taylor_inv_sqrt_4_(float4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

float3 step_3(float3 edge, float3 x)
{
    return step(edge, x);
}

// Seeded 3d simplex noise, output roughly [-1, 1]. Vendored from noisy_bevy.
public float simplex_noise_3d_seeded(float3 v, float3 seed)
{
    let C = float2(1.0 / 6.0, 1.0 / 3.0);
    let D = float4(0.0, 0.5, 1.0, 2.0);

    // first corner
    var i = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);

    // other corners
    let g = step_3(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + 1.0 * C.xxx;
    let x2 = x0 - i2 + 2.0 * C.xxx;
    let x3 = x0 - 1.0 + 3.0 * C.xxx;

    // permutations
    i = i % float3(289.0);
    let s = floor(seed + float3(0.5));
    let p = permute_4_(permute_4_(permute_4_(
        i.z + float4(0.0, i1.z, i2.z, 1.0) + s.z) +
        i.y + float4(0.0, i1.y, i2.y, 1.0) + s.y) +
        i.x + float4(0.0, i1.x, i2.x, 1.0) + s.x
    );

    // gradients (NxN points uniformly over a square, mapped onto an octahedron)
    let n_ = 1.0 / 7.0; // N=7
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z); // mod(p, N*N)

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_); // mod(j, N)

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = float4(x.xy, y.xy);
    let b1 = float4(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, float4(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = float3(a0.xy, h.x);
    var p1 = float3(a0.zw, h.y);
    var p2 = float3(a1.xy, h.z);
    var p3 = float3(a1.zw, h.w);

    // normalize gradients
    let norm = taylor_inv_sqrt_4_(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;

    // mix final noise value
    var m = 0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3));
    m = max(m, float4(0.0));
    m *= m;
    return 42.0 * dot(m * m, float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Seeded 3d fractional brownian motion. Matches noisy_bevy's
// `fbm_simplex_3d_seeded` (and the retired CPU `noise::sample_height` input).
public float fbm_simplex_3d_seeded(float3 pos, int octaves, float lacunarity, float gain, float3 seed)
{
    var sum = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        sum += simplex_noise_3d_seeded(pos * frequency, seed) * amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return sum;
}

// 3d domain warp: offset the sample position by three decorrelated FBM fields
// (iquilezles.org/articles/warp, lifted to 3d for seam-free sphere domains).
public float3 warp3d(float3 p, int octaves, float lacunarity, float gain, float amp, float seed)
{
    let wx = fbm_simplex_3d_seeded(p, octaves, lacunarity, gain, float3(seed));
    let wy = fbm_simplex_3d_seeded(p + float3(31.416, 27.183, 14.142), octaves, lacunarity, gain, float3(seed));
    let wz = fbm_simplex_3d_seeded(p + float3(-17.320, 12.247, -23.140), octaves, lacunarity, gain, float3(seed));
    return p + amp * float3(wx, wy, wz);
}

// Ridged multifractal (Musgrave): sharp crests from inverted-abs simplex,
// octave weights fed forward so ridges compound along crests.
public float ridged_multifractal_3d(float3 pos, int octaves, float lacunarity, float gain, float3 seed)
{
    var sum = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var weight = 1.0;
    for (int i = 0; i < octaves; i++)
    {
        var n = 1.0 - abs(simplex_noise_3d_seeded(pos * frequency, seed));
        n = n * n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += n * amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    return sum;
}

// ── P8: f64-domain folding — full-resolution noise at ANY octave ───────────
// The simplex above is EXACTLY periodic when its skewed cell index shifts by
// a multiple of 289 (the `i % 289` before the permutation). Folding each
// input component by 867 = 3·289 shifts the skewed lattice by exactly
// (4·289, 289, 289) — the folded value is BIT-IDENTICAL: seamless. Doing the
// multiply+fold in f64 (SHADER_F64 — solari requires it) restores full
// argument resolution at planet scale: the plain f32 path's noise argument
// quantizes above ~1e4 (hash cells collapse → stairstepped terrain). f64 is
// used ONLY here; the noise itself stays f32.

float fold867(double x)
{
    let p = double(867.0);
    return float(x - floor(x / p) * p);
}

// FBM over an f64 domain (scalar components — the compute-pass WGSL f64 is
// scalar-only), pre-scaled by the base frequency by the caller.
public float fbm_folded(double dx, double dy, double dz, int octaves, float lacunarity, float gain, float3 seed)
{
    var sum = 0.0;
    var amplitude = 1.0;
    var fx = dx;
    var fy = dy;
    var fz = dz;
    for (int i = 0; i < octaves; i++)
    {
        let p = float3(fold867(fx), fold867(fy), fold867(fz));
        sum += simplex_noise_3d_seeded(p, seed) * amplitude;
        amplitude *= gain;
        fx *= double(lacunarity);
        fy *= double(lacunarity);
        fz *= double(lacunarity);
    }
    return sum;
}

// Ridged multifractal over an f64 domain (mirrors ridged_multifractal_3d).
public float ridged_folded(double dx, double dy, double dz, int octaves, float lacunarity, float gain, float3 seed)
{
    var sum = 0.0;
    var amplitude = 0.5;
    var weight = 1.0;
    var fx = dx;
    var fy = dy;
    var fz = dz;
    for (int i = 0; i < octaves; i++)
    {
        let p = float3(fold867(fx), fold867(fy), fold867(fz));
        var n = 1.0 - abs(simplex_noise_3d_seeded(p, seed));
        n = n * n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += n * amplitude;
        amplitude *= gain;
        fx *= double(lacunarity);
        fy *= double(lacunarity);
        fz *= double(lacunarity);
    }
    return sum;
}

// Ridged octaves [first, first+count) only — the P8 shading-time detail
// bracket (octaves the mesh doesn't have but the pixel footprint resolves).
// Amplitude continues the full-series schedule; the ridge weight feed-forward
// restarts at 1 (a small approximation, invisible in a detail delta).
public float ridged_folded_from(double dx, double dy, double dz, int first, int count, float lacunarity, float gain, float3 seed)
{
    var fx = dx;
    var fy = dy;
    var fz = dz;
    var amplitude = 0.5;
    for (int i = 0; i < first; i++)
    {
        amplitude *= gain;
        fx *= double(lacunarity);
        fy *= double(lacunarity);
        fz *= double(lacunarity);
    }
    var sum = 0.0;
    var weight = 1.0;
    for (int i = 0; i < count; i++)
    {
        let p = float3(fold867(fx), fold867(fy), fold867(fz));
        var n = 1.0 - abs(simplex_noise_3d_seeded(p, seed));
        n = n * n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += n * amplitude;
        amplitude *= gain;
        fx *= double(lacunarity);
        fy *= double(lacunarity);
        fz *= double(lacunarity);
    }
    return sum;
}
