// Fog density and the fog light grid's addressing, shared by the raygen and the grid's compute
// passes (fog_grid.slang). Everything takes the frame's uniforms by pointer: the compute
// passes have no `pc`.
module fog;

import rt_types;

// Height fog: the share of a collision that scatters rather than absorbs (water droplets).
public static const float FOG_ALBEDO = 0.95;

// Fog extinction at a height: `fog_density` at and below `fog_height`, thinning by 1/e every
// `fog_falloff` metres above it.
public float fogDensity(EnvironmentGpu* env, float y) {
    return env.fog_density * exp(-max(0.0, y - env.fog_height) / env.fog_falloff);
}

// The environment's own fog at a point (uniform or height fog).
public float fogBaseDensity(EnvironmentGpu* env, float3 x) {
    return env.fog_falloff > 0.0 ? fogDensity(env, x.y) : env.fog_density;
}

// ---- fog volumes (src/fog_volume.rs) ------------------------------------------------------

float fogHash(float3 p) {
    var p3 = frac(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return frac((p3.x + p3.y) * p3.z);
}

float fogValueNoise(float3 p) {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = lerp(lerp(fogHash(i), fogHash(i + float3(1.0, 0.0, 0.0)), u.x),
                 lerp(fogHash(i + float3(0.0, 1.0, 0.0)), fogHash(i + float3(1.0, 1.0, 0.0)), u.x), u.y);
    let b = lerp(lerp(fogHash(i + float3(0.0, 0.0, 1.0)), fogHash(i + float3(1.0, 0.0, 1.0)), u.x),
                 lerp(fogHash(i + float3(0.0, 1.0, 1.0)), fogHash(i + float3(1.0, 1.0, 1.0)), u.x), u.y);
    return lerp(a, b, u.z);
}

public float3 fogVolumeLocal(FogVolumeGpu v, float4 p) {
    return float3(dot(v.row0, p), dot(v.row1, p), dot(v.row2, p));
}

// A volume's density at a world point: full inside, fading over `edge` toward the surface,
// carved by drifting noise. Never above `density` (the majorant relies on it).
public float fogVolumeDensity(FogVolumeGpu v, float3 x, float time) {
    let p = fogVolumeLocal(v, float4(x, 1.0));
    let r = v.shape == 0u ? max(abs(p.x), max(abs(p.y), abs(p.z))) : length(p);
    if (r >= 1.0) {
        return 0.0;
    }
    var d = v.density * (v.edge > 0.0 ? smoothstep(1.0, 1.0 - v.edge, r) : 1.0);
    if (v.noise_strength > 0.0) {
        let q = (x - float3(v.wind_x, v.wind_y, v.wind_z) * time) / v.noise_scale;
        let n = 0.65 * fogValueNoise(q) + 0.35 * fogValueNoise(q * 2.03 + 17.0);
        d *= lerp(1.0, smoothstep(0.3, 0.7, n), v.noise_strength);
    }
    return d;
}

// Where a ray crosses a volume's bounds, clipped to [0, t_max]; false if it misses them.
public bool fogVolumeSpan(FogVolumeGpu v, float3 o, float3 d, float t_max, out float t0, out float t1) {
    let lo = fogVolumeLocal(v, float4(o, 1.0));
    let ld = fogVolumeLocal(v, float4(d, 0.0));
    let inv = 1.0 / select(abs(ld) < 1.0e-12, float3(1.0e-12), ld);
    let a = (float3(-1.0) - lo) * inv;
    let b = (float3(1.0) - lo) * inv;
    let n = min(a, b);
    let f = max(a, b);
    t0 = max(max(n.x, n.y), max(n.z, 0.0));
    t1 = min(min(f.x, f.y), min(f.z, t_max));
    return t1 > t0;
}

// All the fog at a point: the environment's (`base`) and every volume's.
public float fogDensityAtU(UniformData* u, EnvironmentGpu* env, float3 x, bool base) {
    var d = base ? fogBaseDensity(env, x) : 0.0;
    for (uint i = 0u; i < u.fog_volume_count; i++) {
        d += fogVolumeDensity(u.fog_volumes[i], x, u.time);
    }
    return d;
}

// ---- the light grid (src/fog_grid.rs) -------------------------------------------------------

// The grid's cells across, down and deep.
public uint3 fogGridSizeU(UniformData* u) {
    return uint3(u.fog_grid_size_x, u.fog_grid_size_y, u.fog_grid_size_z);
}

// A world point's cell coordinate in the grid seen through `view_proj` (cell centres at +0.5):
// screen uv up, slices quadratic in view depth out to `dist`.
public float3 fogGridCoord(float view_proj[16], float3 x, uint3 size, float dist) {
    let clip = mat4MulVec(view_proj, float4(x, 1.0));
    let w = max(clip.w, 1.0e-4);
    let uv = clip.xy / w * 0.5 + 0.5;
    return float3(uv * float2(size.xy), sqrt(saturate(w / dist)) * float(size.z));
}

// A grid at a cell coordinate, trilinear; clamped to the grid's edges.
public float4 fogGridSample(float4* grid, uint3 size, float3 g) {
    let p = g - 0.5;
    let b = int3(floor(p));
    let f = p - float3(b);
    let hi = int3(size) - 1;
    var r = float4(0.0);
    for (uint k = 0u; k < 8u; k++) {
        let o = int3(int(k & 1u), int((k >> 1u) & 1u), int(k >> 2u));
        let q = clamp(b + o, int3(0), hi);
        let w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y) * (o.z == 1 ? f.z : 1.0 - f.z);
        r += w * grid[(uint(q.z) * size.y + uint(q.y)) * size.x + uint(q.x)];
    }
    return r;
}

// A cell's world point: (uv, slice) in [0, 1) of the grid's own view, unjittered at the centre.
public float3 fogGridPointU(UniformData* u, float2 uv, float s, float dist) {
    let target = mat4MulVec(u.fog_grid_inverse_projection, float4(uv * 2.0 - 1.0, 1.0, 1.0)).xyz;
    let view = target / max(-target.z, 1.0e-6);
    return mat4MulVec(u.fog_grid_inverse_view, float4(view * (dist * s * s), 1.0)).xyz;
}

// Which cells `fog_grid` lights this frame: one of every `stride` x `stride` block, the block's
// cell turning each frame, so every cell is lit once in stride^2 frames.
public uint2 fogGridPhase(UniformData* u) {
    let s = max(u.fog_grid_stride, 1u);
    let k = u.frame % (s * s);
    return uint2(k % s, k / s);
}
