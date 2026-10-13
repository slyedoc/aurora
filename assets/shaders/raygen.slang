// The path tracer's ray generation: camera paths with NEE (light table + ReSTIR DI, HDR
// environment, suns), the radiance cache, ray portals, the DLSS Ray Reconstruction guides,
// the atmosphere composite and the auto-exposure meter.
import rt_types;
import rand;
import disney;
import atmosphere_rt;
import light_eval;

// DLSS Ray Reconstruction guides + noisy colour, render resolution, written for the
// primary hit of sample 0.
[[vk::binding(0, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> guide_normal_roughness;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> guide_diffuse;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> guide_specular;
[[vk::binding(3, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float4> guide_depth;
[[vk::binding(4, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float4> guide_spec_hit;
[[vk::binding(5, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float4> guide_motion;
[[vk::binding(6, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> dlss_color;
// The camera ray's fog, kept out of Ray Reconstruction: rgb the in-scattered light (pre-exposed
// like the colour), a the transmittance; the composite lays it over RR's output.
[[vk::binding(7, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> guide_fog;
[[vk::binding(100, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(200, 1)]] Sampler2D textures[];

static const bool ENABLE_BLUENOISE = true;

// Features fixed when the pipeline is built (raytracing_pipeline.rs `PipelineFeatures`, from
// `RenderSettings`): off, their code is gone from the shader, registers and all. Their
// uniforms still switch them per frame within a build that has them.
[vk::constant_id(0)] const bool SHARC_ON = false;
[vk::constant_id(1)] const bool RESTIR_ON = false;
// Some environment's sky is the atmosphere (sky.rs `EnvironmentSkies::atmosphere`).
[vk::constant_id(2)] const bool ATMOSPHERE_ON = false;
// Paths per pixel and their maximum length come from the uniforms (`pc.uniforms.samples`
// / `max_bounces`); the DLSS path runs 1 spp with a short cap.

static const float g_extinction = 0.03;
// Specular hit distance of a reflection that reaches the sky: far enough to move like the
// sky does (rotation only), and still a finite half float.
static const float SPEC_HIT_FAR = 65504.0;

// gl_RayFlagsOpaqueEXT, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT.
static const uint FLAGS_OPAQUE = RAY_FLAG_FORCE_OPAQUE;
static const uint FLAGS_SHADOW = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

// traceRayEXT(topLevelAS, flags, mask, 0, 0, 0, origin, tmin, direction, tmax, 0).
void trace(uint flags, uint mask, float3 origin, float tmin, float3 direction, float tmax,
           inout HitPayload payload) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = tmin;
    ray.Direction = direction;
    ray.TMax = tmax;
    TraceRay(topLevelAS, flags, mask, 0, 0, 0, ray, payload);
}

// A shadow ray: true when nothing opaque lies between (the shadow miss shader, miss index 1,
// ran; hits use each record's shadow twin, sbt.rs). Its one-word payload keeps the shading payload out of every visibility test.
bool traceVisible(uint mask, float3 origin, float tmin, float3 direction, float tmax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = tmin;
    ray.Direction = direction;
    ray.TMax = tmax;
    ShadowPayload shadow = { 0u };
    // Hit record offset 1: every geometry's shadow twin (its any-hit takes this payload).
    TraceRay(topLevelAS, FLAGS_SHADOW, mask, 1, 0, 1, ray, shadow);
    return shadow.visible != 0u;
}

// One shadow ray for several light strategies at a vertex: each proposes its finished,
// unshadowed contribution; one is kept with probability proportional to the square root of its
// brightness and, visible, adds that contribution / its pick probability (unbiased; one ray
// instead of one per strategy). The root tempers a dominant light: in shade the sun's unshadowed
// value would otherwise win nearly every pick, its ray blocked, and the sky's fill go unsampled.
struct ShadowPick {
    float wsum;
    float w;
    float3 contribution;
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;

    [mutating]
    void propose(float3 c, float3 o, float t0, float3 d, float t1) {
        let wc = sqrt(luma(c));
        if (!(wc > 0.0)) {
            return;
        }
        wsum += wc;
        if (randf() * wsum < wc) {
            w = wc;
            contribution = c;
            origin = o;
            tmin = t0;
            direction = d;
            tmax = t1;
        }
    }

    // The kept proposal's shadow ray; its contribution over its pick probability when visible.
    float3 resolve(uint mask) {
        if (wsum <= 0.0) {
            return float3(0.0);
        }
        if (!traceVisible(mask, origin, tmin, direction, tmax)) {
            return float3(0.0);
        }
        return contribution * (wsum / w);
    }
};

// trace() through Shader Execution Reordering: the hit is found first, the warp regrouped by
// what it hit, and only then shaded, so neighbouring lanes shade the same material instead of
// diverging across the scene (Northshire: 12.2 -> 10.9 ms; a record-index hint measured the
// same as the driver's own key).
void traceReordered(uint flags, uint mask, float3 origin, float tmin, float3 direction, float tmax,
                    inout HitPayload payload) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = tmin;
    ray.Direction = direction;
    ray.TMax = tmax;
    HitObject hit = HitObject::TraceRay(topLevelAS, flags, mask, 0, 0, 0, ray, payload);
    ReorderThread(hit);
    HitObject::Invoke(topLevelAS, hit, payload);
}

// The fog of the environment a ray is in (src/sky.rs `Fog`).
float sampleDistanceInFog(EnvironmentGpu* env) {
    return -log(randf()) / env.fog_density;
}

// Fog extinction at a height: `fog_density` at and below `fog_height`, thinning by 1/e every
// `fog_falloff` metres above it.
float fogDensity(EnvironmentGpu* env, float y) {
    return env.fog_density * exp(-max(0.0, y - env.fog_height) / env.fog_falloff);
}

// The environment's own fog at a point (uniform or height fog).
float fogBaseDensity(EnvironmentGpu* env, float3 x) {
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

float3 fogVolumeLocal(FogVolumeGpu v, float4 p) {
    return float3(dot(v.row0, p), dot(v.row1, p), dot(v.row2, p));
}

// A volume's density at a world point: full inside, fading over `edge` toward the surface,
// carved by drifting noise. Never above `density` (the majorant relies on it).
float fogVolumeDensity(FogVolumeGpu v, float3 x) {
    let p = fogVolumeLocal(v, float4(x, 1.0));
    let r = v.shape == 0u ? max(abs(p.x), max(abs(p.y), abs(p.z))) : length(p);
    if (r >= 1.0) {
        return 0.0;
    }
    var d = v.density * (v.edge > 0.0 ? smoothstep(1.0, 1.0 - v.edge, r) : 1.0);
    if (v.noise_strength > 0.0) {
        let q = (x - float3(v.wind_x, v.wind_y, v.wind_z) * pc.uniforms.time) / v.noise_scale;
        let n = 0.65 * fogValueNoise(q) + 0.35 * fogValueNoise(q * 2.03 + 17.0);
        d *= lerp(1.0, smoothstep(0.3, 0.7, n), v.noise_strength);
    }
    return d;
}

// Where a ray crosses a volume's bounds, clipped to [0, t_max]; false if it misses them.
bool fogVolumeSpan(FogVolumeGpu v, float3 o, float3 d, float t_max, out float t0, out float t1) {
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
float fogDensityAt(EnvironmentGpu* env, float3 x, bool base) {
    var d = base ? fogBaseDensity(env, x) : 0.0;
    for (uint i = 0u; i < pc.uniforms.fog_volume_count; i++) {
        d += fogVolumeDensity(pc.uniforms.fog_volumes[i], x);
    }
    return d;
}

// The farthest any volume reaches along a ray (0 = it crosses none): how far a camera ray
// that reaches the sky marches.
float fogVolumesReach(float3 o, float3 d) {
    var reach = 0.0;
    for (uint i = 0u; i < pc.uniforms.fog_volume_count; i++) {
        float t0, t1;
        if (fogVolumeSpan(pc.uniforms.fog_volumes[i], o, d, 1.0e5, t0, t1)) {
            reach = max(reach, t1);
        }
    }
    return reach;
}

// Whether the ray scatters in the fog before `t_max` (0 = it reached the sky: no fog), and
// where. Uniform fog alone draws one exponential distance; otherwise delta tracking against
// a majorant (the height fog at the segment's lowest point plus every volume it crosses),
// accepting a tentative collision with probability density / majorant.
bool sampleFogScatter(EnvironmentGpu* env, float3 o, float3 d, float t_max, out float t) {
    let volumes = pc.uniforms.fog_volume_count > 0u;
    if (env.fog_falloff <= 0.0 && !volumes) {
        t = sampleDistanceInFog(env);
        return t < t_max;
    }
    t = 0.0;
    if (t_max <= 0.0) {
        return false;
    }
    let majorant = fogMajorant(env, o, d, t_max, true);
    if (majorant <= 1.0e-9) {
        return false;
    }
    for (int i = 0; i < 64; i++) {
        t += -log(randf()) / majorant;
        if (t >= t_max) {
            return false;
        }
        if (randf() * majorant < fogDensityAt(env, o + d * t, true)) {
            return true;
        }
    }
    return false;
}

// Height fog: the share of a collision that scatters rather than absorbs (water droplets).
static const float FOG_ALBEDO = 0.95;

// Henyey-Greenstein phase function, per steradian.
float fogPhase(EnvironmentGpu* env, float cos_theta) {
    return fogPhaseG(env.fog_scatter, cos_theta);
}



float calcFogTransmittance(float distance) {
    return exp(-g_extinction * distance);
}

float3 sampleFogScatterDirection(EnvironmentGpu* env, float3 incoming) {
    // Henyey-Greenstein phase function sampling
    // g is the asymmetry parameter (-1 <= g <= 1)
    // g > 0 means forward scattering
    // g < 0 means backward scattering
    // g = 0 means isotropic scattering

    let g = env.fog_scatter;
    float cosTheta;
    if (g == 0.0) {
        // isotropic
        cosTheta = 1.0 - 2.0 * randf();
    } else {
        // anisotropic
        let term = (1.0 - g * g) / (1.0 + g * (2.0 * randf() - 1.0));
        cosTheta = (1.0 + g * g - term * term) / (2.0 * g);
    }

    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 2.0 * PI * randf();

    let ret = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return mul(ret, orthonormalBasis(incoming));
}

// Optical depth per march step: a segment through faint fog takes fewer steps (down to one),
// thick fog all of `fog_steps`. Neighbouring pixels see about the same depth, so they agree
// on the count and their shadow rays stay coherent.
static const float FOG_STEP_DEPTH = 0.05;

// The densest the fog can be along a segment: the height fog at its lowest point (or the
// uniform fog) when `base`, plus every volume it crosses.
float fogMajorant(EnvironmentGpu* env, float3 o, float3 d, float t_max, bool base) {
    var majorant = 0.0;
    if (base) {
        majorant = env.fog_falloff > 0.0 ? fogDensity(env, min(o.y, o.y + d.y * t_max)) : env.fog_density;
    }
    for (uint i = 0u; i < pc.uniforms.fog_volume_count; i++) {
        float t0, t1;
        if (fogVolumeSpan(pc.uniforms.fog_volumes[i], o, d, t_max, t0, t1)) {
            majorant += pc.uniforms.fog_volumes[i].density;
        }
    }
    return majorant;
}

// The camera ray through the fog: `env.fog_steps` jittered steps over its first `seg` metres
// (closer together near the camera), each lit by the suns through a shadow ray -- the shafts
// -- and one step, picked at random, by the sky in a phase-sampled direction. Adds the
// in-scatter to `acc` and leaves the transmittance in `mask`; `base` takes the environment's
// own fog in too (not on a ray to the sky, which the old scatter never fogged either). The
// payload comes back as it went in.
void fogMarch(EnvironmentGpu* env, uint ray_env, uint ray_mask, float tmax, float3 o, float3 d,
              float seg, bool base, inout float3 mask, inout float3 acc, inout HitPayload payload) {
    let depth = fogMajorant(env, o, d, seg, base) * seg;
    if (depth <= 1.0e-6) {
        return;
    }
    let vertex = payload;
    let n = clamp(uint(ceil(depth / FOG_STEP_DEPTH)), 1u, env.fog_steps);
    let jitter = randf();
    let sky_step = min(uint(randf() * float(n)), n - 1u);
    var T = 1.0;
    var prev = 0.0;
    var L = float3(0.0);
    for (uint i = 0u; i < n; i++) {
        let f = float(i + 1u) / float(n);
        let t_end = seg * f * f;
        let fs = (float(i) + jitter) / float(n);
        let x = o + d * (seg * fs * fs);
        let dt = t_end - prev;
        prev = t_end;
        let sigma = fogDensityAt(env, x, base);
        if (sigma <= 0.0) {
            continue;
        }
        let step_t = exp(-sigma * dt);
        var Ls = fogSunLight(env, ray_mask, tmax, x, d, payload);
        if (i == sky_step) {
            // Phase-sampled, so the phase over its pdf is 1; one step in n carries it.
            let w = sampleFogScatterDirection(env, d);
            payload.t = -1.0;
            payload.miss_flags = ray_env << 1;
            trace(FLAGS_SHADOW, ray_mask, x, 0.0, w, tmax, payload);
            if (payload.t == 0.0) {
                Ls += payload.emission * float(n);
            }
        }
        L += T * FOG_ALBEDO * Ls * (1.0 - step_t);
        T *= step_t;
    }
    acc += mask * L;
    mask *= T;
    payload = vertex;
}

// The suns' light at fog point `x`, as the fog scatters it along `d`. The HDR sky's sun is
// two discs (core, halo) on one direction: one shadow ray serves every disc that shares it.
float3 fogSunLight(EnvironmentGpu* env, uint ray_mask, float tmax, float3 x, float3 d,
                   inout HitPayload payload) {
    var Ls = float3(0.0);
    var visible = false;
    var l = float3(0.0);
    for (uint si = 0u; si < env.sun_count; si++) {
        let sun = env.suns[si];
        if (si == 0u || any(sun.direction.xyz != env.suns[si - 1u].direction.xyz)) {
            l = sampleCone(sun.direction.xyz, sun.direction.w, randf(), randf());
            visible = traceVisible(ray_mask, x, 0.0, l, tmax);
        }
        if (!visible) {
            continue;
        }
        var radiance = sun.radiance.rgb;
        // Not behind ATMOSPHERE_ON like the other atmosphere sites: folding this one away hung
        // the GPU (Xid 109, driver 595, 2026-10-11).
        if (env.mode == 3u && si == 0u) {
            radiance *= atmoSunTransmittanceAt(x) * cloudShadow(x);
        }
        let solid_angle = 2.0 * PI * (1.0 - sun.direction.w);
        Ls += radiance * solid_angle * fogPhase(env, dot(d, l));
    }
    return Ls;
}

// Extracts either the first two or second two bytes from the uint
// encoded byte buffer. As a result, byteIndex must be an even number
// for this to work.
float2 TwoBnoiseSamples(uint byteIndex) {
    let uintIndex = byteIndex / 4;
    let packedUInt = pc.bluenoise[uintIndex];

    // Calculate the shift amount based on the byte index modulo 4
    let shiftAmount = (byteIndex % 4) * 8;

    // Extract the two bytes at positions byteIndex and byteIndex+1 (next byte)
    let res = uint2((packedUInt >> shiftAmount), (packedUInt >> (shiftAmount + 8)));

    return float2(float(res.x & 0xFF) / 255.0, float(res.y & 0xFF) / 255.0);
}

// Assumed to be called once per sample
float2 bnoiseBest(uint sampleIdx) {
    // 64 bluenoise slices; the seed advances every frame.
    let textureIdx = (pc.uniforms.frame * pc.uniforms.samples + sampleIdx) % 64;
    if (textureIdx >= 64) {
        let a = randf();
        let b = randf();
        return float2(a, b);
    }
    let x = DispatchRaysIndex().x % 128;
    let y = DispatchRaysIndex().y % 128;
    let byteIndex = 128 * 128 * 2 * textureIdx + 128 * 2 * y + 2 * x;
    return TwoBnoiseSamples(byteIndex);
}









// The light table as light_eval.slang reaches it.
LightCtx lightCtx() {
    LightCtx ctx;
    ctx.lights = pc.lights;
    ctx.instances = pc.cur_instances;
    ctx.emissive_boost = pc.uniforms.emissive_boost;
    return ctx;
}

// ---- Radiance cache (SHaRC-style; see src/sharc.rs) ---------------------------------------

// Must match SHARC_ENTRIES in src/sharc.rs.
static const uint SHARC_ENTRIES = 1u << 20;
// The share of paths that never query the cache (they train it).
static const float SHARC_TRAIN = 0.125;
// Cache vertices a path records (its vertices past the camera's): all of max_bounces 4's.
// Deeper paths' later vertices only read the cache. Every slot is live across every trace,
// and the raygen's stack is what the trace's time follows (4 slots: +0.17 ms at 3840x1080).
static const uint SHARC_VERTICES = 3u;

// The voxel size at a point: `sharc_voxel` near the camera, doubling per distance octave
// past 8 m.
float sharcLevel(float3 pos, float3 cam) {
    return floor(log2(max(max(length(pos - cam), 1.0e-3) / 8.0, 1.0)));
}

float sharcVoxelSize(float3 pos, float3 cam) {
    return max(pc.uniforms.sharc_voxel, 1.0e-3) * exp2(sharcLevel(pos, cam));
}

// Voxel key: position quantized at a distance-scaled size, level, dominant face normal.
uint4 sharcKey(float3 pos, float3 n, float3 cam) {
    let level = sharcLevel(pos, cam);
    let s = sharcVoxelSize(pos, cam);
    let c = int3(floor(pos / s));
    var axis = abs(n.x) > abs(n.y) ? (abs(n.x) > abs(n.z) ? 0u : 2u)
                                   : (abs(n.y) > abs(n.z) ? 1u : 2u);
    axis = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);
    return uint4(uint(c.x), uint(c.y), uint(c.z), (uint(level) << 3) | axis);
}

uint sharcHash(uint4 k) {
    var h = k.x * 0x8da6b343u ^ k.y * 0xd8163841u ^ k.z * 0xcb1ab31fu ^ k.w * 0x165667b1u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// Find-or-insert the entry for a voxel key: linear probe with a tag compare, claiming an
// empty slot atomically. Returns ~0 when the neighbourhood is full (the path just traces on).
uint sharcSlot(float3 pos, float3 n, float3 cam) {
    let key = sharcKey(pos, n, cam);
    let h = sharcHash(key);
    let tag = (sharcHash(key.wxyz) | 1u);
    for (uint i = 0u; i < 8u; i++) {
        let slot = (h + i) & (SHARC_ENTRIES - 1u);
        let t = pc.sharc[slot].tag;
        if (t == tag) {
            return slot;
        }
        if (t == 0u) {
            uint prev;
            InterlockedCompareExchange(pc.sharc[slot].tag, 0u, tag, prev);
            if (prev == 0u || prev == tag) {
                return slot;
            }
        }
    }
    return 0xFFFFFFFFu;
}

// The resolved radiance, if the entry has folded in enough samples to be trusted.
// Finite and non-negative, by bit pattern (fast-math may fold isnan away): exponent all ones
// is NaN or infinity, the sign bit is negative.
bool sharcSane(float3 v) {
    let b = asuint(v);
    return all((b & 0x7f800000u) != 0x7f800000u) && all((b & 0x80000000u) == 0u);
}

bool sharcQuery(uint slot, out float3 radiance) {
    radiance = float3(pc.sharc[slot].res_r, pc.sharc[slot].res_g, pc.sharc[slot].res_b);
    return pc.sharc[slot].conf >= 16.0 && sharcSane(radiance);
}

// One path sample of the voxel's outgoing radiance (nits; fixed point at 1/65536 nit, so a
// moonlit voxel is not rounded to black).
// A NaN from one degenerate hit (the camera inside a wall, a mirror up close) cast to an
// integer poisoned a whole voxel's channel: it glowed red, blue or yellow until it washed out.
void sharcDeposit(uint slot, float3 radiance) {
    if (!sharcSane(radiance)) {
        return;
    }
    let r = min(radiance, float3(1.0e5)) * 65536.0;
    InterlockedAdd(pc.sharc[slot].acc_r, uint64_t(r.r));
    InterlockedAdd(pc.sharc[slot].acc_g, uint64_t(r.g));
    InterlockedAdd(pc.sharc[slot].acc_b, uint64_t(r.b));
    InterlockedAdd(pc.sharc[slot].acc_n, 1u);
    pc.sharc[slot].frame = pc.uniforms.frame;
}













// The shading normal bent the least amount into the viewer's hemisphere: n' = n + v * (eps - n.v)
// leaves n untouched when it already faces v and otherwise gives n'.v = eps (before the
// renormalisation, which keeps the sign).
float3 bendTowardViewer(float3 n, float3 v) {
    let NoV = dot(n, v);
    let eps = 1.0e-3;
    return NoV >= eps ? n : normalize(n + v * (eps - NoV));
}

// DLSS-RR specular albedo: pre-integrated environment BRDF (split-sum scale/bias
// fit, Ray Tracing Gems ch. 32) -- the RR Integration Guide's EnvBRDFApprox2. RR
// demodulates specular with this, NOT raw F0. `alpha` is GGX alpha (perceptual
// roughness squared). The guide's HLSL mul(M, v) with row-major literals, verbatim.
float3 envBRDFApprox2(float3 f0, float alpha, float NoV) {
    NoV = abs(NoV);
    let X = float4(1.0, NoV, NoV * NoV, NoV * NoV * NoV);
    let Y = float4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    let M1 = float2x2(0.99044, -1.28514, 1.29678, -0.755907);
    let M2 = float3x3(1.0, 2.92338, 59.4188, 20.3225, -27.0302, 222.592, 121.563, 626.13, 316.627);
    let M3 = float2x2(0.0365463, 3.32707, 9.0632, -9.04756);
    let M4 = float3x3(1.0, 3.59685, -1.36772, 9.04401, -16.3174, 9.22949, 5.56589, 19.7886, -20.2123);
    let bias = dot(mul(M1, X.xy), Y.xy) / dot(mul(M2, X.xyw), Y.xyw)
             * clamp(f0.g * 50.0, 0.0, 1.0); // the guide's zero-reflectance hack
    let scale = dot(mul(M3, X.xy), Y.xy) / dot(mul(M4, X.xzw), Y.xyw);
    return f0 * max(0.0, scale) + max(0.0, bias);
}

// ---- HDR sky importance sampling (src/sky_importance.rs) ----------------------------------

// Whether the environment's HDR sky has a sampling table (no table: BRDF sampling only).
bool skySampled(EnvironmentGpu* env) {
    return env.importance != nullptr;
}

// Solid-angle pdf of the sky sampler picking direction `d`.
float skyPdf(EnvironmentGpu* env, float3 d) {
    let uv = env_dir_to_uv(d);
    let w = SKY_IMPORTANCE_W;
    let h = SKY_IMPORTANCE_H;
    let x = min(uint(uv.x * float(w)), w - 1u);
    let y = min(uint(uv.y * float(h)), h - 1u);
    return env.importance[w * h + 1u + y * w + x];
}

// Draws a direction from the sky's luminance CDF (binary search, jittered inside the texel);
// returns its radiance, with the direction and its solid-angle pdf.
float3 skySample(EnvironmentGpu* env, float u1, float u2, float u3, out float3 dir, out float pdf) {
    let w = SKY_IMPORTANCE_W;
    let n = w * SKY_IMPORTANCE_H;
    uint lo = 0u;
    uint hi = n;
    while (lo + 1u < hi) {
        let mid = (lo + hi) / 2u;
        if (env.importance[mid] <= u1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let i = min(lo, n - 1u);
    let uv = float2((float(i % w) + u2) / float(w), (float(i / w) + u3) / float(SKY_IMPORTANCE_H));
    dir = env_uv_to_dir(uv);
    pdf = env.importance[n + 1u + i];
    // Explicit LOD 0: a ray-tracing stage has no derivatives (glslang emits the same).
    let texel = textures[NonUniformResourceIndex(env.tex)].SampleLevel(uv, 0.0).rgb;
    return env.color.rgb * skyResidual(*env, dir, texel) * env.sky_brightness;
}

// ---- ray portals (src/portal.rs) -------------------------------------------------------
// Instance rows are the VkAccelerationStructureInstanceKHR 3x4 transform, row-major:
// r_k = row k of world_from_local. Inverse via the 3x3 adjugate, -L^-1 t for translation.
void portalAffineInverse(float4 r0, float4 r1, float4 r2,
                         out float4 o0, out float4 o1, out float4 o2) {
    let c0 = float3(r0.x, r1.x, r2.x);
    let c1 = float3(r0.y, r1.y, r2.y);
    let c2 = float3(r0.z, r1.z, r2.z);
    let t = float3(r0.w, r1.w, r2.w);
    let inv_det = 1.0 / dot(c0, cross(c1, c2));
    let i0 = cross(c1, c2) * inv_det;
    let i1 = cross(c2, c0) * inv_det;
    let i2 = cross(c0, c1) * inv_det;
    o0 = float4(i0, -dot(i0, t));
    o1 = float4(i1, -dot(i1, t));
    o2 = float4(i2, -dot(i2, t));
}

float3 portalXformPoint(float4 r0, float4 r1, float4 r2, float3 p) {
    let h = float4(p, 1.0);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

float3 portalXformDir(float4 r0, float4 r1, float4 r2, float3 v) {
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

// If `hit_slot` is a portal surface, rewrite the ray to continue from the paired portal
// (into portal-local, a half-turn about local Y, out through the target's frame) and
// return true; else leave the ray untouched.
bool portalRedirect(uint hit_slot, float3 hit_pos, inout float3 origin, inout float3 direction,
                    inout uint ray_mask, inout uint ray_env) {
    for (uint i = 0u; i < pc.uniforms.portal_count; i++) {
        let pair = pc.uniforms.portals[i];
        if (pair.z == 0u || pair.x != hit_slot) {
            continue;
        }
        let pb = pair.x * 4u;
        float4 i0, i1, i2;
        portalAffineInverse(
            pc.cur_instances[pb + 0u],
            pc.cur_instances[pb + 1u],
            pc.cur_instances[pb + 2u],
            i0, i1, i2);
        var p = portalXformPoint(i0, i1, i2, hit_pos);
        var d = portalXformDir(i0, i1, i2, direction);
        p = float3(-p.x, p.y, -p.z);
        d = float3(-d.x, d.y, -d.z);
        let tb = pair.y * 4u;
        let t0 = pc.cur_instances[tb + 0u];
        let t1 = pc.cur_instances[tb + 1u];
        let t2 = pc.cur_instances[tb + 2u];
        direction = normalize(portalXformDir(t0, t1, t2, d));
        // Nudge past the exit surface so the continued ray doesn't immediately re-hit it.
        origin = portalXformPoint(t0, t1, t2, p) + direction * 1.0e-4;
        // The ray now lives in whatever environment the exit does: adopt the exit INSTANCE's
        // render-layer mask (instance row 3 packs custom_index | mask << 24) and its environment.
        ray_mask = (asuint(pc.cur_instances[tb + 3u].x) >> 24) & 0xFFu;
        ray_env = environmentOf(ray_mask, pc.uniforms.environment_bits);
        return true;
    }
    return false;
}

// ---- Fog light grid (src/fog_grid.rs) ------------------------------------------------------

// Most slices the camera ray steps through the grid (lookups, so many more than `fog_steps`).
static const uint FOG_GRID_STEPS = 32u;

// The grid's cells across, down and deep (src/fog_grid.rs).
uint3 fogGridSize() {
    return uint3(pc.uniforms.fog_grid_size_x, pc.uniforms.fog_grid_size_y, pc.uniforms.fog_grid_size_z);
}

// A world point's cell coordinate in the grid seen through `view_proj` (cell centres at +0.5):
// screen uv up, slices quadratic in view depth out to `dist`.
float3 fogGridCoord(float view_proj[16], float3 x, uint3 size, float dist) {
    let clip = mat4MulVec(view_proj, float4(x, 1.0));
    let w = max(clip.w, 1.0e-4);
    let uv = clip.xy / w * 0.5 + 0.5;
    return float3(uv * float2(size.xy), sqrt(saturate(w / dist)) * float(size.z));
}

// A grid at a cell coordinate, trilinear; clamped to the grid's edges.
float4 fogGridSample(float4* grid, uint3 size, float3 g) {
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

// The grid across two cells (four reads half a cell apart in x and y): each cell's own noise
// (its light is a few random shadow rays) shared with its neighbours. Composited after Ray
// Reconstruction, the camera's fog no longer has it to hide a mottle at the cell size.
float4 fogGridSampleSoft(float4* grid, uint3 size, float3 g) {
    return 0.25 * (fogGridSample(grid, size, g + float3(-0.5, -0.5, 0.0))
                 + fogGridSample(grid, size, g + float3(0.5, -0.5, 0.0))
                 + fogGridSample(grid, size, g + float3(-0.5, 0.5, 0.0))
                 + fogGridSample(grid, size, g + float3(0.5, 0.5, 0.0)));
}

// A cell's world point: (uv, slice) in [0, 1) of the grid's own view, unjittered at the centre.
float3 fogGridPoint(float2 uv, float s, float dist) {
    let target = mat4MulVec(pc.uniforms.fog_grid_inverse_projection, float4(uv * 2.0 - 1.0, 1.0, 1.0)).xyz;
    let view = target / max(-target.z, 1.0e-6);
    return mat4MulVec(pc.uniforms.fog_grid_inverse_view, float4(view * (dist * s * s), 1.0)).xyz;
}

// One light-table sample at fog point `x`, as the fog scatters it along `d`: RIS over
// `light_candidates` from the power CDF, one shadow ray for the survivor, clamped.
float3 fogTableLight(EnvironmentGpu* env, uint ray_mask, float3 x, float3 d, float radiance_clamp,
                     inout HitPayload payload) {
    if (pc.uniforms.light_entries == 0u) {
        return float3(0.0);
    }
    DisneyMaterial mat = {};
    Lobes lob = {};
    let M = max(1u, pc.uniforms.light_candidates);
    var wsum = 0.0;
    var sel_val = float3(0.0);
    var sel_phat = 0.0;
    var ldir = float3(0.0);
    var ldist = 0.0;
    for (uint c = 0u; c < M; c++) {
        float p_pick;
        let e = lightsPickEntry(lightCtx(), randf() * pc.lights.total_power, p_pick);
        let su = randf();
        let sv = randf();
        float p_area, pdf_solid, cdist;
        float3 cdir;
        let val = lightSampleEval(lightCtx(), e, lightsFindInstance(lightCtx(), e), p_pick, su, sv, x, float3(0.0),
                                  float3(env.fog_scatter, 0.0, 0.0), -d, mat, lob, ray_mask, false,
                                  p_area, pdf_solid, cdir, cdist);
        let phat = luma(val);
        if (phat <= 0.0 || p_area <= 0.0) {
            continue;
        }
        let wi = phat / p_area;
        wsum += wi;
        if (randf() * wsum < wi) {
            sel_val = val;
            sel_phat = phat;
            ldir = cdir;
            ldist = cdist;
        }
    }
    if (wsum <= 0.0 || sel_phat <= 0.0) {
        return float3(0.0);
    }
    if (!traceVisible(ray_mask, x, 0.0, ldir, ldist * (1.0 - 1.0e-4))) {
        return float3(0.0);
    }
    var L = sel_val * (wsum / (float(M) * sel_phat));
    if (radiance_clamp > 0.0) {
        L *= min(1.0, radiance_clamp / max(luma(L), 1.0e-6));
    }
    return L;
}

// One thread per grid cell (src/fog_grid.rs), ahead of the views that march it: a random point
// in the cell lit by the suns, one phase-sampled sky ray and the light table, blended into
// last frame's value at the cell's reprojected centre.
[shader("raygeneration")]
void fog_grid() {
    let size = fogGridSize();
    let id = DispatchRaysIndex();
    let i = (id.z * size.y + id.y) * size.x + id.x;
    if (pc.uniforms.fog_grid_cells == 0u || i >= pc.uniforms.fog_grid_cells) {
        return;
    }
    HitPayload payload;
    initRandom(uint2(pc.uniforms.fog_grid_cells, 1u), uint2(i, 0u), pc.uniforms.frame);
    let tmax = 1.0e6;
    let radiance_clamp = pc.uniforms.firefly_clamp * 0.18 / max(pc.ae.exposure, 1.0e-20);
    let ray_env = pc.uniforms.environment;
    let ray_mask = pc.uniforms.camera_mask;
    let env = pc.uniforms.environments + ray_env;
    let dist = env.fog_grid_distance;
    let cell = float3(id);
    let grid = float3(size);
    let r = float3(randf(), randf(), randf());
    let x = fogGridPoint((cell.xy + r.xy) / grid.xy, (cell.z + r.z) / grid.z, dist);
    let cam = mat4MulVec(pc.uniforms.fog_grid_inverse_view, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let centre = fogGridPoint((cell.xy + 0.5) / grid.xy, (cell.z + 0.5) / grid.z, dist);
    let d = normalize(centre - cam);

    var L = fogSunLight(env, ray_mask, tmax, x, d, payload);
    let w = sampleFogScatterDirection(env, d);
    payload.t = -1.0;
    payload.miss_flags = ray_env << 1;
    trace(FLAGS_SHADOW, ray_mask, x, 0.0, w, tmax, payload);
    if (payload.t == 0.0) {
        L += payload.emission;
    }
    L += fogTableLight(env, ray_mask, x, d, radiance_clamp, payload);
    if (any(isnan(L)) || any(isinf(L))) {
        L = float3(0.0);
    }

    // Off last frame's grid (past its far slice moving forward, its edges turning) the
    // nearest edge cell is still the best guess: kept at a quarter of its weight rather than
    // dropped, which left those cells one raw sample every frame.
    var prev = float4(0.0);
    if (pc.uniforms.fog_grid_fresh == 0u) {
        let g = fogGridCoord(pc.uniforms.fog_grid_prev_view_proj, centre, size, dist);
        prev = fogGridSample(pc.uniforms.fog_grid_prev, size, g);
        if (any(g < float3(0.0)) || any(g > grid)) {
            prev.w = min(prev.w, env.fog_grid_history * 0.25);
        }
    }
    if (any(isnan(prev)) || any(isinf(prev)) || prev.w < 0.0) {
        prev = float4(0.0);
    }
    // Firefly clamp against the cell's own average: a sky ray through a canopy gap or a lone
    // sunlit sample in shade moves it by a bounded step instead of a flash.
    if (prev.w >= 4.0) {
        let cap = 4.0 * max(luma(prev.rgb), 1.0e-3);
        L *= min(1.0, cap / max(luma(L), 1.0e-6));
    }
    let n = min(prev.w + 1.0, env.fog_grid_history);
    pc.uniforms.fog_grid_cur[i] = float4(lerp(prev.rgb, L, 1.0 / n), n);
}

// The camera ray through the fog like fogMarch, lit from last frame's grid instead of
// shadow rays: the in-scatter goes into `acc`, the transmittance into `mask`. One lookup per
// grid slice the ray crosses (slices are quadratic in depth, so are the steps), at fixed
// positions: the fog is composited after Ray Reconstruction, which no longer averages a dither
// away, and stepping by optical depth gave thin daylight haze ONE step -- a single lookup half
// way down the ray, so a ray ending at a sunward tree sampled only its shadow (black cut-outs)
// and one past it a flat veil. Past the grid the far slice lights the rest in one step.
void fogMarchGrid(EnvironmentGpu* env, float3 o, float3 d, float seg, bool base,
                  inout float3 mask, inout float3 acc) {
    if (fogMajorant(env, o, d, seg, base) * seg <= 1.0e-6) {
        return;
    }
    let size = fogGridSize();
    let dist = env.fog_grid_distance;
    let inside = min(seg, dist);
    let n = clamp(uint(ceil(sqrt(inside / dist) * float(size.z))), 1u, FOG_GRID_STEPS);
    var T = 1.0;
    var prev = 0.0;
    var L = float3(0.0);
    for (uint i = 0u; i <= n; i++) {
        var t_end = seg;
        var t_mid = 0.5 * (prev + seg);
        if (i < n) {
            let f = float(i + 1u) / float(n);
            let fm = (float(i) + 0.5) / float(n);
            t_end = inside * f * f;
            t_mid = inside * fm * fm;
        }
        let dt = t_end - prev;
        prev = t_end;
        if (dt <= 0.0) {
            continue;
        }
        let x = o + d * t_mid;
        let sigma = fogDensityAt(env, x, base);
        if (sigma <= 0.0) {
            continue;
        }
        let step_t = exp(-sigma * dt);
        let Ls = fogGridSampleSoft(pc.uniforms.fog_grid_cur, size,
                                   fogGridCoord(pc.uniforms.fog_grid_view_proj, x, size, dist)).rgb;
        L += T * FOG_ALBEDO * Ls * (1.0 - step_t);
        T *= step_t;
    }
    acc += mask * L;
    mask *= T;
}

// One thread per grid column, after `fog_grid`: front to back through its slices (density
// at each slice's centre, light from this frame's grid as the march reads it), the running
// in-scatter and transmittance stored per cell at its far side.
[shader("raygeneration")]
void fog_integrate() {
    let size = fogGridSize();
    let id = DispatchRaysIndex().xy;
    if (pc.uniforms.fog_grid_cells == 0u || id.x >= size.x || id.y >= size.y) {
        return;
    }
    let env = pc.uniforms.environments + pc.uniforms.environment;
    let dist = env.fog_grid_distance;
    let uv = (float2(id) + 0.5) / float2(size.xy);
    let cam = fogGridPoint(uv, 0.0, dist);
    // Metres along the column per metre of view depth.
    let stretch = length(fogGridPoint(uv, 1.0, dist) - cam) / dist;
    let slices = float(size.z);
    var T = 1.0;
    var L = float3(0.0);
    var prev = 0.0;
    for (uint k = 0u; k < size.z; k++) {
        let f = float(k + 1u) / slices;
        let depth = dist * f * f;
        let dt = (depth - prev) * stretch;
        prev = depth;
        let s = (float(k) + 0.5) / slices;
        let sigma = fogDensityAt(env, fogGridPoint(uv, s, dist), true);
        if (sigma > 0.0) {
            let step_t = exp(-sigma * dt);
            let Ls = fogGridSampleSoft(pc.uniforms.fog_grid_cur, size,
                                       float3(float2(id) + 0.5, float(k) + 0.5)).rgb;
            L += T * FOG_ALBEDO * Ls * (1.0 - step_t);
            T *= step_t;
        }
        pc.uniforms.fog_grid_sum[(k * size.y + id.y) * size.x + id.x] = float4(L, T);
    }
}

// The fog from the grid's eye to `x` (in-scatter, transmittance): the column sums, bilinear
// across columns and linear between slice sides; past the grid the far slice's light and the
// density half way out cover the rest in one step.
float4 fogGridSumAt(EnvironmentGpu* env, float3 x) {
    let size = fogGridSize();
    let dist = env.fog_grid_distance;
    let clip = mat4MulVec(pc.uniforms.fog_grid_view_proj, float4(x, 1.0));
    let w = max(clip.w, 1.0e-4);
    let uv = clip.xy / w * 0.5 + 0.5;
    let z = sqrt(saturate(w / dist)) * float(size.z);
    let k0 = min(uint(z), size.z - 1u);
    let fz = z - float(k0);
    let p = uv * float2(size.xy) - 0.5;
    let b = int2(floor(p));
    let f = p - float2(b);
    let hi = int2(size.xy) - 1;
    var r = float4(0.0);
    for (uint c = 0u; c < 4u; c++) {
        let o = int2(int(c & 1u), int(c >> 1u));
        let q = clamp(b + o, int2(0), hi);
        let wxy = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        let col = uint(q.y) * size.x + uint(q.x);
        let slice = size.x * size.y;
        let near = k0 == 0u ? float4(0.0, 0.0, 0.0, 1.0) : pc.uniforms.fog_grid_sum[(k0 - 1u) * slice + col];
        let far = pc.uniforms.fog_grid_sum[k0 * slice + col];
        r += wxy * lerp(near, far, fz);
    }
    if (clip.w > dist) {
        let cam = mat4MulVec(pc.uniforms.fog_grid_inverse_view, float4(0.0, 0.0, 0.0, 1.0)).xyz;
        let xm = cam + (x - cam) * (0.5 * (dist + clip.w) / clip.w);
        let sigma = fogDensityAt(env, xm, true);
        if (sigma > 0.0) {
            let step_t = exp(-sigma * length(x - cam) * (clip.w - dist) / clip.w);
            let Ls = fogGridSampleSoft(pc.uniforms.fog_grid_cur, size,
                                       fogGridCoord(pc.uniforms.fog_grid_view_proj, xm, size, dist)).rgb;
            r.rgb += r.a * FOG_ALBEDO * Ls * (1.0 - step_t);
            r.a *= step_t;
        }
    }
    return r;
}

// A camera ray segment's fog from the column sums: what lies between its two ends, as
// fogMarchGrid leaves it (in-scatter into `acc`, transmittance into `mask`).
void fogGridSegment(EnvironmentGpu* env, float3 o, float3 d, float seg,
                    inout float3 mask, inout float3 acc) {
    if (fogMajorant(env, o, d, seg, true) * seg <= 1.0e-6) {
        return;
    }
    let a = fogGridSumAt(env, o);
    let b = fogGridSumAt(env, o + d * seg);
    let ta = max(a.a, 1.0e-6);
    acc += mask * max(b.rgb - a.rgb, float3(0.0)) / ta;
    mask *= saturate(b.a / ta);
}

// ReSTIR DI's shading (src/restir.rs), after its compute passes: the pixel's final reservoir
// gets one shadow ray; visible, its contribution goes into the noisy colour (pre-exposed like
// the main pass's). An occluded winner is NOT zeroed for reuse: carried forward, the zero
// counted shadow twice (history, then this ray) and the result came out ~15 % dark.
[shader("raygeneration")]
void restir_shade() {
    let launch_id = DispatchRaysIndex().xy;
    let launch_size = DispatchRaysDimensions().xy;
    let idx = launch_id.y * launch_size.x + launch_id.x;
    let s = pc.restir_surfaces[idx];
    if (s.frame != pc.uniforms.frame) {
        return;
    }
    var r = pc.reservoirs_cur[idx];
    if (r.y == 0xFFFFFFFFu || r.W <= 0.0) {
        return;
    }
    float p_area_cdf, p_cdf, dist;
    float3 dir;
    let val = restirTarget(lightCtx(), s, r.y, r.uv_pk, p_area_cdf, p_cdf, dir, dist);
    if (restirLuma(val) <= 0.0) {
        return;
    }
    let ns = oct_to_float32x3(unpackHalf2(s.geo_normal_oct));
    if (!traceVisible(s.ray_mask, leaveSurface(s.position, ns, dir), 0.0, dir,
                      dist * (1.0 - 1.0e-4))) {
        return;
    }
    let px = int2(launch_id);
    let add = s.throughput * val * (r.W / float(max(pc.uniforms.samples, 1u)))
        * pc.ae.input_exposure;
    let prior = dlss_color[px].rgb;
    dlss_color[px] = float4(clamp(prior + add, float3(0.0), float3(65504.0)), 1.0);
}

[shader("raygeneration")]
void main() {
    let launch_id = DispatchRaysIndex().xy;
    let launch_size = DispatchRaysDimensions().xy;
    // The one ray payload, live across every trace like the GLSL global: hit / miss shaders
    // leave fields they do not write as the previous trace left them.
    HitPayload payload;

    // Fresh noise every frame for the temporal denoiser.
    initRandom(launch_size, launch_id, pc.uniforms.frame);

    let px = int2(launch_id);

    // Rays leaving a surface start off it (`offsetRay`), so nothing needs a minimum distance.
    let tmin = 0.0;
    // Far range: the sim-scale sky domes sit kilometers out, a planet's horizon tens of km.
    let tmax = 1.0e6;

    // A known sub-pixel jitter (told to NGX); the launch y is flipped into the NDC-up space
    // here, so the image-space offset is +jitter. Row 0 is the TOP row: its centre sits half
    // a pixel below the top edge, (height - 1 - y) + 0.5 in the up-space. (An earlier
    // `height - y + 0.5` put every sample one whole pixel above its pixel: the image was
    // shifted a row against the depth/motion guides and the matrices Ray Reconstruction
    // reprojects with, and edges crawled.)
    let subpixel = float2(0.5 + pc.uniforms.jitter_x, 0.5 - pc.uniforms.jitter_y);
    let pixel_center = float2(float(launch_id.x), float(launch_size.y - 1u - launch_id.y)) + subpixel;
    let inUV = pixel_center / float2(launch_size);
    let d = inUV * 2.0 - 1.0;

    let initial_origin = mat4MulVec(pc.uniforms.inverse_view, float4(0, 0, 0, 1)).xyz;
    let target = mat4MulVec(pc.uniforms.inverse_projection, float4(d, 1, 1)).xyz;
    let initial_direction = mat4MulVec(pc.uniforms.inverse_view, float4(normalize(target), 0)).xyz;
    let focalPoint = initial_origin + initial_direction * pc.focus[0];

    if (pc.uniforms.pull_focus_x == launch_id.x && pc.uniforms.pull_focus_y == launch_id.y) {
        payload.cone = float2(0.0);
        trace(FLAGS_OPAQUE, 0xFF, initial_origin, tmin, initial_direction, tmax, payload);
        if (payload.t != 0.0) {
            pc.focus[0] = payload.t;
        }
    }

    var acc = float3(0.0);
    // The camera ray's fog for the composite, averaged over the samples, gathers in `guide_fog`
    // itself (pre-exposed light, transmittance): kept in registers it was live across every
    // trace, and the raygen's stack is what the trace's time follows. None: nothing over the
    // colour.
    guide_fog[px] = float4(0.0);
    let SAMPLES = pc.uniforms.samples;
    let fog_scale = pc.ae.input_exposure / float(SAMPLES);
    let MAX_BOUNCES = pc.uniforms.max_bounces;
    // Firefly clamp in nits: a multiple of the luminance the metering puts at 18% gray
    // (0.18 / exposure), so it tracks the scene -- a sky, or a ceiling panel in a closed room.
    // From the SMOOTH metered exposure, not the held input one: that lags the scene by up to
    // its deadband, and a clamp that many stops too loose lets every emitter speckle through.
    let radiance_clamp = pc.uniforms.firefly_clamp * 0.18 / max(pc.ae.exposure, 1.0e-20);
    // The camera ray's first hit distance (0 = sky), for the atmosphere composite below.
    var primary_t = 0.0;
    // Auto-exposure meters incident light: radiance over sample 0's guide-vertex albedo. Sky
    // and emitters seen directly are left out of the meter.
    var meter_albedo = 1.0;
    var meter_skip = false;

    for (uint sample_idx = 0; sample_idx < SAMPLES; sample_idx += 1) {
        let offsetR = sqrt(randf());
        let offsetA = randf() * 2.0 * PI;
        let offset = pc.uniforms.aperture * float2(offsetR * cos(offsetA), offsetR * sin(offsetA));

        var origin = mat4MulVec(pc.uniforms.inverse_view, float4(offset, 0, 1)).xyz;
        var direction = normalize(focalPoint - origin);
        var mask = float3(1.0);
        // Russian roulette's boost so far (1 / the survival odds): the firefly clamp allows for
        // it, or it would clip back exactly the energy that keeps roulette unbiased (the canopy's
        // indirect light came out at less than half).
        var rr_boost = 1.0;
        // The camera ray and pure-specular continuations may see the sun disc directly; after a
        // BRDF-sampled bounce the sun comes from next-event estimation only.
        var want_sun = true;
        // Solid-angle pdf of the ray being traced, when it came from BRDF sampling (0 for the
        // camera ray and delta continuations): the MIS partner of light-table sampling.
        var prev_pdf = 0.0;
        // What this ray sees and where it is: the camera's layer mask and environment until a
        // portal swaps them.
        var ray_mask = pc.uniforms.camera_mask;
        var ray_env = pc.uniforms.environment;
        // The normal of the surface `origin` sits on, for the next ray to leave it by; zero where
        // the ray does not start on a surface (the camera, a fog scatter, a portal exit).
        var surf_n = float3(0.0);
        // Still on the camera ray (through cutouts and portals, before any scatter): it skips
        // the camera's primary-hidden layers, which every other ray sees.
        var camera_segment = true;
        // Still in the view's own frustum (no portal yet): the fog grid lights only that.
        var in_view = true;
        // Radiance-cache update targets along this path: the voxel and, per vertex, snapshots
        // of the running mask / radiance so the vertex's outgoing radiance falls out as a delta
        // once the path ends.
        uint sh_slot[SHARC_VERTICES];
        float3 sh_mask[SHARC_VERTICES];
        float3 sh_acc[SHARC_VERTICES];
        var sh_n = 0u;
        // One path in SHARC_TRAIN never asks the cache: traced to its end, it is what the
        // cache learns from (answered paths only pass on what it already knows).
        let sharc_train = SHARC_ON && pc.uniforms.sharc != 0u && randf() < SHARC_TRAIN;
        // The ray being traced left a rough surface: only then may the cache answer for what it
        // hits (a mirror's or a glossy floor's reflection is traced, never the cache's cells).
        var from_rough = false;
        // The camera ray met the back of a solid surface first: the camera is inside geometry
        // (a wall it clipped through). Such a path sees unlit insides and must not teach the
        // cache -- it poisoned the voxels of the room side, which glowed until washed out.
        var cache_blind = false;
        // DLSS guides come from sample 0's first ACCEPTED vertex (a primary ray may pass through
        // alpha-cutout edges first).
        var guides_written = false;
        // The camera ray's fog transmittance so far; its light goes to `guide_fog`.
        var fog_t = 1.0;
        // Ray cone for texture level of detail (surface.slang): width at the ray's origin
        // and spread angle. The camera ray opens by one pixel; a scattered bounce opens it by the
        // lobe it was drawn from, so indirect hits read blurrier levels (less noise too).
        var cone_width = 0.0;
        // inverse_projection[1][1] (column 1, row 1).
        var cone_spread = 2.0 * pc.uniforms.inverse_projection[5] / float(launch_size.y);
        // Stable guides (RTXPT "stable planes"): the guide chain follows the analytically
        // DOMINANT Fresnel lobe through glass. When the radiance path rolls the minority lobe
        // instead, this frame's guide writes are abandoned (the images keep last frame's
        // dominant-branch values), so the guides never flicker between the reflected and the
        // refracted world -- per-pixel-random branch choices are exactly what RR cannot
        // share history across (see PathTracerStablePlanes.hlsli's seam note).
        var guide_live = true;
        // PSR-lite: when guides defer through smooth-transmission (glass) vertices, the first
        // non-delta vertex writes them at a VIRTUAL point this far along the camera ray.
        let guide_cam_origin = origin;
        let guide_cam_dir = direction;
        var guide_dist = 0.0;

        for (uint bounce_idx = 0; bounce_idx < MAX_BOUNCES; bounce_idx += 1) {
            // Low bit: may this miss return the sun discs; above it the ray's environment (the
            // miss returns that environment's sky).
            payload.miss_flags = (want_sun ? 1u : 0u) | (ray_env << 1);
            payload.cone = float2(cone_width, cone_spread);
            let trace_mask = camera_segment ? (ray_mask & ~pc.uniforms.primary_hidden) : ray_mask;
            let start = any(surf_n != float3(0.0)) ? leaveSurface(origin, surf_n, direction) : origin;
            traceReordered(FLAGS_OPAQUE, trace_mask, start, tmin, direction, tmax, payload);
            if (sample_idx == 0u && bounce_idx == 0u) {
                primary_t = payload.t;
            }
            cone_width += cone_spread * payload.t;

            let surface_normal = oct_to_float32x3(payload.surface_and_world_normal.xy);
            // The shading normal (interpolated, normal-mapped) can face away from the ray at
            // silhouettes, corners and grazing angles even though the geometric one faces it;
            // bend it the least amount into the ray's hemisphere so every BRDF term sees a
            // valid view direction. (Left alone, the sample failed and the path went on at
            // full throughput: a mirror of the sky on every edge.)
            let world_normal = bendTowardViewer(
                oct_to_float32x3(payload.surface_and_world_normal.zw), -direction);

            // Ray portals: a portal-surface hit continues from the paired portal. A delta
            // teleport, not a scatter: guides stay deferred (the far side writes them at the
            // virtual distance, PSR-style) and the next emissive hit is taken at full weight
            // (NEE cannot sample lights through a portal, so MIS must not discount them).
            if (payload.t != 0.0 && pc.uniforms.portal_count > 0u) {
                var p_origin = origin;
                var p_dir = direction;
                if (portalRedirect(payload.slot, origin + direction * payload.t, p_origin, p_dir, ray_mask, ray_env)) {
                    if (!guides_written) {
                        guide_dist += payload.t;
                    }
                    origin = p_origin;
                    surf_n = float3(0.0);
                    direction = p_dir;
                    prev_pdf = 0.0;
                    in_view = false;
                    continue;
                }
            }

            // Alpha-cutout pass-through (stochastic alpha at the edge texels of a masked material,
            // pure-black texels used as masks) applies to MASKED materials only: an opaque
            // material's texture alpha is data (WoW keeps gloss there) and its black texels are
            // black. It is decided BEFORE the DLSS guides: a primary ray that
            // continues through a leaf must not leave the leaf's albedo / normal / depth behind for
            // Ray Reconstruction to demodulate the sky behind it with -- that reads as white,
            // smeared foliage whenever something bright sits behind it.
            // (Spelled out as ifs: the coin is only drawn where GLSL's short-circuit draws it.)
            var pass_through = false;
            if (payload.t != 0.0 && all(payload.emission == float3(0.0))
                && hitPayloadGetMasked(payload)) {
                if (all(payload.color.rgb == float3(0.0))) {
                    pass_through = true;
                } else if (payload.color.a < 1.0) {
                    pass_through = randf() < 1.0 - payload.color.a;
                }
            }

            // PSR-lite: a smooth transmissive hit (glass) shows the REFRACTED world, and RR has
            // no refraction guide -- writing the glass surface here pins that content to the
            // wrong depth/motion and RR blurs it to frost. Defer: the chain's first non-delta
            // vertex writes the guides instead, at a virtual point along the camera ray.
            let guide_defer = payload.t != 0.0
                && hitPayloadGetTransmission(payload) > 0.5
                && hitPayloadGetRoughness(payload) < 0.1;

            if (sample_idx == 0 && guide_live) {
                if (!guides_written && !pass_through && !guide_defer) {
                    guides_written = true;
                    if (payload.t == 0.0) {
                        meter_skip = true;
                        guide_normal_roughness[px] = float4(0.0, 0.0, 0.0, 1.0);
                        guide_diffuse[px] = float4(0.0);
                        guide_specular[px] = float4(0.5, 0.5, 0.5, 1.0); // sky: RR guide 3.4.2
                        // Linear view depth (Depth_Type_Linear): 0 is the CAMERA plane, so the sky
                        // must sit at the far clamp -- 0 here reads as a surface on the lens and the
                        // sky/mesh silhouette boils.
                        guide_depth[px] = float4(1.0e7, 0.0, 0.0, 0.0);
                        // Sky motion: reproject the ray direction at infinity (w = 0 cancels the
                        // translation -- infinitely far points only parallax under rotation). Zero
                        // here pins the sky to the screen and it smears whenever the camera turns.
                        let sky_cur = mat4MulVec(pc.uniforms.view_proj, float4(direction, 0.0));
                        let sky_prev = mat4MulVec(pc.uniforms.prev_view_proj, float4(direction, 0.0));
                        var sky_motion = float2(0.0);
                        if (sky_cur.w > 1.0e-4 && sky_prev.w > 1.0e-4) {
                            let cuv = (sky_cur.xy / sky_cur.w) * float2(0.5, -0.5);
                            let puv = (sky_prev.xy / sky_prev.w) * float2(0.5, -0.5);
                            sky_motion = clamp(cuv - puv, float2(-2.0), float2(2.0));
                        }
                        guide_motion[px] = float4(sky_motion, 0.0, 0.0);
                        guide_spec_hit[px] = float4(0.0);
                    } else {
                        let g_roughness = hitPayloadGetRoughness(payload);
                        let g_metallic = hitPayloadGetMetallic(payload);
                        let hit_pos = origin + direction * payload.t;
                        // The end of the (possibly refracted) chain laid out along the primary ray at
                        // the accumulated distance. No glass in between => exactly hit_pos.
                        let virtual_pos = guide_cam_origin + guide_cam_dir * (guide_dist + payload.t);
                        let view_pos = mat4MulVec(pc.uniforms.view, float4(virtual_pos, 1.0));
                        let linear_depth = max(-view_pos.z, 1.0e-4);
                        let albedo = clamp(payload.color.rgb, float3(0.0), float3(1.0));
                        // Ray Reconstruction wants LINEAR roughness; the material's is perceptual.
                        let linear_roughness = clamp(g_roughness * g_roughness, 0.0, 1.0);
                        guide_normal_roughness[px] = float4(world_normal, linear_roughness);
                        // NOTE: emissive is deliberately NOT folded into this guide. A render-res albedo
                        // guide carrying emissive-texture text aliases under the DLSS jitter, and RR's
                        // history churns it into mush whenever the camera rests (tried 2026-09-02).
                        guide_diffuse[px] = float4(albedo * (1.0 - g_metallic), 1.0);
                        let g_f0 = lerp(float3(0.04), albedo, g_metallic);
                        let g_NoV = dot(world_normal, -direction);
                        let g_spec = envBRDFApprox2(g_f0, linear_roughness, g_NoV);
                        guide_specular[px] = float4(g_spec, 1.0);
                        meter_albedo = dot(float3(0.2125, 0.7154, 0.0721), albedo * (1.0 - g_metallic) + g_spec);
                        meter_skip = any(payload.emission != float3(0.0));
                        guide_depth[px] = float4(min(linear_depth, 1.0e7), 0.0, 0.0, 0.0);
                        // Motion in UV units (y down), current minus previous; NGX scales by -(w,h).
                        let cur_clip = mat4MulVec(pc.uniforms.view_proj, float4(virtual_pos, 1.0));
                        // The end vertex's own frame-to-frame motion, applied to the virtual point
                        // (identical to prev_world_pos when no glass was traversed).
                        let prev_clip = mat4MulVec(pc.uniforms.prev_view_proj,
                                                   float4(virtual_pos - (hit_pos - payload.prev_world_pos), 1.0));
                        var motion = float2(0.0);
                        if (cur_clip.w > 1.0e-4 && prev_clip.w > 1.0e-4) {
                            let cur_uv = (cur_clip.xy / cur_clip.w) * float2(0.5, -0.5);
                            let prev_uv = (prev_clip.xy / prev_clip.w) * float2(0.5, -0.5);
                            motion = clamp(cur_uv - prev_uv, float2(-2.0), float2(2.0));
                        }
                        guide_motion[px] = float4(motion, 0.0, 0.0);
                        // Specular hit distance (RR guide 3.4.9): how far the REFLECTION sits behind
                        // the surface, which RR turns into the reflection's own motion. It comes from a
                        // dedicated ray along the lobe's mirror direction, never from the path: the
                        // path leaves a dielectric on its diffuse lobe nine times in ten, and a
                        // hemisphere ray's length is per-pixel static. Past the panel's roughness
                        // cutoff the reflection is too wide to have a motion of its own: 0 pins it to
                        // the surface and the ray is saved.
                        var spec_hit = 0.0;
                        if (g_roughness <= pc.uniforms.spec_hit_roughness) {
                            var mirror = reflect(direction, world_normal);
                            if (dot(mirror, surface_normal) <= 0.0) {
                                mirror = reflect(direction, surface_normal);
                            }
                            // Its own payload: the vertex's stays put instead of riding the trace.
                            HitPayload probe = {};
                            probe.miss_flags = ray_env << 1;
                            probe.cone = float2(cone_width, cone_spread);
                            traceReordered(FLAGS_OPAQUE, ray_mask, leaveSurface(hit_pos, surface_normal, mirror), tmin, mirror, tmax, probe);
                            spec_hit = probe.t == 0.0 ? SPEC_HIT_FAR : min(probe.t, SPEC_HIT_FAR);
                        }
                        guide_spec_hit[px] = float4(spec_hit, 0.0, 0.0, 0.0);
                    }
                }
            }
            if (!guides_written) {
                guide_dist += payload.t;
            }
            let roughness = hitPayloadGetRoughness(payload);
            let metallic = hitPayloadGetMetallic(payload);
            let transmission = hitPayloadGetTransmission(payload);
            let inside = hitPayloadGetInside(payload);
            let lob = unpackLobes(payload.lobes);
            if (camera_segment && inside && transmission < 0.01 && !hitPayloadGetMasked(payload)) {
                cache_blind = true;
            }

            // fog bounce (skybox has a distance of 0 so will never hit
            // this case, having an extra check for this is slower than
            // deferring to the generic emitter check)
            float fogDistance;
            let env = pc.uniforms.environments + ray_env;
            let fog_any = env.fog_density > 0.0 || pc.uniforms.fog_volume_count > 0u;
            if (camera_segment && env.fog_steps > 0u && fog_any) {
                // The camera ray marches its fog (the shafts, without the noise of one random
                // scatter); a ray to the sky marches only as far as the volumes reach.
                let seg = payload.t > 0.0 ? payload.t : fogVolumesReach(origin, direction);
                var fog_mask = mask;
                var fog_acc = float3(0.0);
                if (seg > 0.0 && in_view && pc.uniforms.fog_grid_cells > 0u
                    && pc.uniforms.fog_grid_fresh == 0u) {
                    // A surface's segment takes the column sums; one to the sky only its
                    // volumes, which the sums (environment fog included) cannot leave out.
                    if (payload.t > 0.0) {
                        fogGridSegment(env, origin, direction, seg, fog_mask, fog_acc);
                    } else {
                        fogMarchGrid(env, origin, direction, seg, false, fog_mask, fog_acc);
                    }
                } else if (seg > 0.0) {
                    fogMarch(env, ray_env, ray_mask, tmax, origin, direction, seg,
                             payload.t > 0.0, fog_mask, fog_acc, payload);
                }
                // The surface behind reaches RR unveiled; the composite applies both.
                let seg_t = luma(mask) > 0.0 ? luma(fog_mask) / luma(mask) : 1.0;
                if (any(fog_acc > float3(0.0))) {
                    let g = guide_fog[px];
                    guide_fog[px] = float4(min(g.rgb + fog_t * fog_acc * fog_scale, float3(65504.0)), g.a);
                }
                fog_t *= seg_t;
            } else if (sampleFogScatter(env, origin, direction, payload.t, fogDistance)) {
                origin = origin + direction * fogDistance;
                surf_n = float3(0.0);
                // Uniform fog keeps its old ad-hoc loss; height fog is delta-tracked, so the
                // collision's weight is just the medium's single-scattering albedo.
                mask *= env.fog_falloff > 0.0 || pc.uniforms.fog_volume_count > 0u
                    ? FOG_ALBEDO : calcFogTransmittance(fogDistance);
                // The sun straight into the scatter point, through the phase function: the
                // shafts of light through gaps (a scattered ray would rarely find its disc).
                {
                    for (uint si = 0u; si < env.sun_count; si++) {
                        let sun = env.suns[si];
                        let r1 = randf();
                        let r2 = randf();
                        let l = sampleCone(sun.direction.xyz, sun.direction.w, r1, r2);
                        var sun_radiance = sun.radiance.rgb;
                        if (ATMOSPHERE_ON && env.mode == 3u && si == 0u) {
                            sun_radiance *= atmoSunTransmittanceAt(origin);
                        }
                        if (traceVisible(ray_mask, origin, 0.0, l, tmax)) {
                            let solid_angle = 2.0 * PI * (1.0 - sun.direction.w);
                            acc += mask * fogPhase(env, dot(direction, l)) * sun_radiance * solid_angle;
                        }
                    }
                }
                direction = sampleFogScatterDirection(env, direction);
                prev_pdf = 0.0;
                want_sun = false;
                continue;
            }

            // hit an emitter
            if (any(payload.emission != float3(0.0)) || payload.t == 0.0) {
                var contribution = mask * payload.emission;
                // MIS: a BRDF-sampled ray landing on a table light shares credit with the
                // next-event estimate taken at the previous vertex (prev_pdf 0 = camera ray or
                // delta continuation, which take the emission at full weight).
                if (payload.t != 0.0 && prev_pdf > 0.0 && pc.uniforms.light_entries > 0u
                    && payload.slot < pc.lights.slot_map_count && pc.lights.total_power > 0.0) {
                    let linst_idx = pc.lights.slot_to_linst[payload.slot];
                    if (linst_idx != 0xFFFFFFFFu && (linst_idx & LIGHT_SLOT_ANALYTIC) != 0u) {
                        // A sphere emitter: `origin` is still the vertex the ray left.
                        let pdf_light = sphereEmitterPdf(lightCtx(), linst_idx & ~LIGHT_SLOT_ANALYTIC, origin);
                        if (pdf_light > 0.0) {
                            contribution *= prev_pdf / (prev_pdf + pdf_light);
                        }
                    } else if (linst_idx != 0xFFFFFFFFu) {
                        let li = pc.lights.linsts[linst_idx];
                        if (payload.prim_tri < li.tri_count) {
                            let e = li.entry_base + payload.prim_tri;
                            let p_pick = lightsEntryPick(lightCtx(), e);
                            if (p_pick > 0.0) {
                                let pdf_light = lightTrianglePdf(lightCtx(), li, payload.prim_tri, p_pick, payload.t, direction);
                                if (pdf_light > 0.0) {
                                    contribution *= prev_pdf / (prev_pdf + pdf_light);
                                }
                            }
                        }
                    }
                }
                // MIS against the environment sampler: a BRDF-sampled ray that escapes to the HDR
                // sky shares credit with the previous vertex's environment estimate.
                if (payload.t == 0.0 && prev_pdf > 0.0 && skySampled(env)) {
                    contribution *= prev_pdf / (prev_pdf + skyPdf(env, direction));
                }
                // Firefly suppression: an indirect path that reaches an emitter far brighter than the
                // sky would take hundreds of frames to average out; clamp its luminance instead.
                if (bounce_idx > 0 && radiance_clamp > 0.0) {
                    contribution *= min(1.0, radiance_clamp * rr_boost / max(luma(contribution), 1.0e-6));
                }
                acc += contribution;
                break;
            }

            // Alpha-cutout pass-through: carry on along the same ray from the hit.
            origin = origin + direction * payload.t;
            surf_n = surface_normal;
            if (pass_through) {
                continue;
            }
            camera_segment = false;

            // refraction (glass): GGX microfacet transmission (Walter et al. 2007). The half
            // vector is sampled from the NDF, Fresnel is evaluated about it, and the ray
            // reflects/refracts THROUGH it, so rough glass scatters around the true refraction
            // direction. (The old path lerped toward a cosine sample about the OUTWARD normal,
            // which fogged and bent refraction for any nonzero roughness.)
            if (randf() < transmission) {
                // calculate the eta based on whether we are inside
                let n1 = inside ? payload.refract_index : 1.0;
                let n2 = inside ? 1.0 : payload.refract_index;
                let eta = n1 / n2;

                // Half-vector from the GGX NDF: theta_h = atan(alpha * sqrt(u / (1 - u))).
                // alpha 0 keeps H = N and the block below reduces to the smooth (delta) case.
                let t_alpha = roughness * roughness;
                var H = world_normal;
                if (t_alpha > 0.0) {
                    let u1 = randf();
                    let ct = rsqrt(1.0 + t_alpha * t_alpha * u1 / max(1.0 - u1, 1.0e-6));
                    let st = sqrt(max(0.0, 1.0 - ct * ct));
                    let phi = 2.0 * PI * randf();
                    H = normalize(mul(float3(st * cos(phi), st * sin(phi), ct), orthonormalBasis(world_normal)));
                }
                let costi = dot(H, -direction);
                if (costi <= 0.0) {
                    break;  // half-vector faces away from the ray: zero weight
                }

                let k = 1 - (eta * eta) * (1 - costi * costi);

                float pReflect;
                if (k < 0) {
                    // Total internal reflection
                    pReflect = 1;
                } else {
                    // Fresnel about H. (sin^2 = 1 - cos^2; the old code had `1 - costi - costi`,
                    // which zeroed the sine for every mid angle and skewed reflectance.)
                    let sinti = sqrt(max(0.0, 1.0 - costi * costi));
                    let costt = sqrt(1.0 - eta * eta * sinti * sinti);
                    let spol = (n1 * costi - n2 * costt) / (n1 * costi + n2 * costt);
                    let ppol = (n1 * costt - n2 * costi) / (n1 * costt + n2 * costi);
                    pReflect = 0.5 * (spol * spol + ppol * ppol);
                }

                let lobe_reflect = randf() <= pReflect;
                // Minority Fresnel lobe on the guide chain: stop writing guides this frame (see
                // `guide_live` above). The radiance stays unbiased -- only the guides sit out.
                if (sample_idx == 0 && !guides_written && lobe_reflect != (pReflect > 0.5)) {
                    guide_live = false;
                }
                float3 new_dir;
                if (lobe_reflect) {
                    new_dir = reflect(direction, H);
                    // A microfacet reflection that tunnels below the surface has zero weight.
                    if (dot(new_dir, world_normal) * dot(-direction, world_normal) < 0.0) {
                        break;
                    }
                } else {
                    new_dir = normalize(eta * direction + H * (eta * costi - sqrt(k)));
                    // A refraction that stays on the incoming side likewise.
                    if (dot(new_dir, world_normal) * dot(-direction, world_normal) > 0.0) {
                        break;
                    }
                }

                // Single-scatter Smith weight for NDF-sampled H (Walter eq. 41 with the D and pdf
                // cancelled): G1(i)·G1(o)·|i·h| / (|i·n|·|h·n|), clamped against grazing fireflies.
                if (t_alpha > 0.0) {
                    let a2 = t_alpha * t_alpha;
                    let NoV = abs(dot(world_normal, -direction));
                    let NoL = abs(dot(world_normal, new_dir));
                    let NoH = max(abs(dot(world_normal, H)), 1.0e-6);
                    let g1v = 2.0 * NoV / (NoV + sqrt(a2 + (1.0 - a2) * NoV * NoV));
                    let g1l = 2.0 * NoL / (NoL + sqrt(a2 + (1.0 - a2) * NoL * NoL));
                    mask *= clamp(g1v * g1l * costi / max(NoV * NoH, 1.0e-6), 0.0, 4.0);
                }
                direction = new_dir;
                if (inside) {
                    mask *= exp(-payload.t * payload.absorption);
                }
                prev_pdf = 0.0;
                continue;
            }

            // russian roullete
            // stopping on the first two bounces is a waste because we haven't been able
            // to gather much light yet except those directly in view of the camera, but
            // those paths would be terminated there and then anyway because they hit
            // an emitter.
            // Survival follows the path THROUGHPUT, not the local albedo: a path that has
            // already lost most of its energy in a valley of 0.4-albedo rock is not worth
            // another shadow ray, however bright the rock it just hit.
            if (bounce_idx > 1) {
                let pRussian = clamp(luma(mask), 0.03, 0.9);
                if (randf() >= pRussian) {
                    break;
                }
                mask /= pRussian;
            }

            let disney_mat = disneyMaterial(payload.color.rgb, metallic, roughness);

            // Radiance cache: diffuse-ish vertices past the primary hit take part. At the first
            // indirect hit on, a converged voxel ends the path with its cached outgoing
            // radiance -- when the ray came off a rough surface and its footprint (the cone's width
            // here) covers a voxel: a mirror's reflection or a close contact showed the cache's
            // cells as blocks and leaked light. Training paths never ask. Every participating
            // vertex of a traced path is an update target.
            if (SHARC_ON && pc.uniforms.sharc != 0u && bounce_idx > 0
                && roughness >= 0.35 && metallic <= 0.5 && transmission < 0.01) {
                let slot = sharcSlot(origin, world_normal, initial_origin);
                if (slot != 0xFFFFFFFFu) {
                    if (!sharc_train && from_rough
                        && cone_width >= sharcVoxelSize(origin, initial_origin)) {
                        float3 cached;
                        if (sharcQuery(slot, cached)) {
                            acc += mask * cached;
                            break;
                        }
                    }
                    if (sh_n < SHARC_VERTICES) {
                        sh_slot[sh_n] = slot;
                        sh_mask[sh_n] = mask;
                        sh_acc[sh_n] = acc;
                        sh_n++;
                    }
                }
            }

            // Direct light from the emissive-triangle table. At the primary vertex of sample 0,
            // with ReSTIR DI on, its passes do it (light grid, initial + temporal and spatial
            // reuse, one shadow ray: src/restir.rs); everywhere else it is RIS over CDF
            // candidates with an MIS-weighted shadow ray.
            // Positions and areas are read live, so stale CDF weights change variance, never
            // the answer.
            // From `one_shadow_ray_from` on the strategies below share one shadow ray.
            let one_ray = bounce_idx >= pc.uniforms.one_shadow_ray_from;
            ShadowPick pick = {};
            if (pc.uniforms.light_entries > 0u && pc.lights.total_power > 0.0) {
                if (RESTIR_ON && bounce_idx == 0 && sample_idx == 0 && pc.uniforms.restir_candidates > 0u) {
                    // ReSTIR DI lights this vertex from the table in its own passes (restir.slang,
                    // then `restir_shade` below): hand it the surface.
                    RestirSurface rs;
                    rs.position = origin;
                    rs.frame = pc.uniforms.frame;
                    rs.prev_position = payload.prev_world_pos;
                    rs.depth = mat4MulVec(pc.uniforms.view_proj, float4(origin, 1.0)).w;
                    rs.albedo = disney_mat.albedo;
                    rs.metallic = disney_mat.metallic;
                    rs.throughput = mask;
                    rs.roughness = disney_mat.roughness;
                    rs.normal_oct = packHalf2(float32x3_to_oct(world_normal));
                    rs.geo_normal_oct = packHalf2(float32x3_to_oct(surface_normal));
                    rs.view_oct = packHalf2(float32x3_to_oct(-direction));
                    rs.ray_mask = ray_mask;
                    rs.transmission = lob.transmission;
                    rs.clearcoat = lob.clearcoat;
                    rs.coat_alpha = lob.coat_alpha;
                    rs.pad = 0.0;
                    pc.restir_surfaces[px.y * int(launch_size.x) + px.x] = rs;
                } else {
                    // Resampled importance sampling, no reuse: M candidates, each weighted by its
                    // MIS-weighted unshadowed contribution at THIS point over its source pdf; one
                    // survives and gets the shadow ray. Candidates come from ReSTIR's light grid
                    // cell at this point when there is one (lamps resampled by power / distance^2
                    // there, so a bounce off a lit wall finds the lamp beside it), else from the
                    // power CDF, which knows nothing of distance: with hundreds of lights a single
                    // draw nearly always lands on one that contributes nothing. The MIS weight
                    // rides inside the resampled integrand against the candidate's own CDF pdf
                    // (what the emitter-hit side uses), so the two strategies still sum to one;
                    // only the RIS weight takes the grid's source pdf. M = 1 is the plain
                    // estimator.
                    let M = bounce_idx == 0u ? pc.uniforms.light_candidates
                                             : max(1u, pc.uniforms.light_candidates / 4u);
                    let grid = pc.restir_frame;
                    let cell = grid != nullptr ? gridCell(grid, origin) : 0xFFFFFFFFu;
                    var wsum = 0.0;
                    var sel_val = float3(0.0);
                    var sel_phat = 0.0;
                    var ldir = float3(0.0);
                    var ldist = 0.0;
                    for (uint c = 0u; c < M; c++) {
                        float p_pick;
                        uint e;
                        // The candidate's source pdf over its CDF pick probability.
                        var src_ratio = 1.0;
                        if (cell != 0xFFFFFFFFu) {
                            let k = min(uint(randf() * float(grid.grid_k)), grid.grid_k - 1u);
                            let gs = grid.grid[cell * grid.grid_k + k];
                            let W = asfloat(gs.y);
                            if (gs.x == 0xFFFFFFFFu || W <= 0.0) {
                                continue;
                            }
                            e = gs.x;
                            p_pick = lightsEntryPick(lightCtx(), e);
                            if (p_pick <= 0.0) {
                                continue;
                            }
                            src_ratio = 1.0 / (W * p_pick);
                        } else {
                            e = lightsPickEntry(lightCtx(), randf() * pc.lights.total_power, p_pick);
                        }
                        let su = randf();
                        let sv = randf();
                        float p_area, pdf_solid, cdist;
                        float3 cdir;
                        let val = lightSampleEval(lightCtx(), e, lightsFindInstance(lightCtx(), e), p_pick, su, sv,
                                                  origin, world_normal, surface_normal, -direction,
                                                  disney_mat, lob, ray_mask, false, p_area, pdf_solid, cdir,
                                                  cdist);
                        if (luma(val) <= 0.0 || p_area <= 0.0) {
                            continue;
                        }
                        let w = pdf_solid / (pdf_solid + layeredPdf(-direction, world_normal, cdir, disney_mat, lob));
                        let g = val * w;
                        let phat = luma(g);
                        let wi = phat / (p_area * src_ratio);
                        wsum += wi;
                        if (randf() * wsum < wi) {
                            sel_val = g;
                            sel_phat = phat;
                            ldir = cdir;
                            ldist = cdist;
                        }
                    }
                    if (wsum > 0.0 && sel_phat > 0.0) {
                        var contribution = mask * sel_val * (wsum / (float(M) * sel_phat));
                        if (bounce_idx > 0 && radiance_clamp > 0.0) {
                            contribution *= min(1.0, radiance_clamp * rr_boost / max(luma(contribution), 1.0e-6));
                        }
                        let o = leaveSurface(origin, surface_normal, ldir);
                        if (one_ray) {
                            pick.propose(contribution, o, tmin, ldir, ldist * (1.0 - 1.0e-4));
                        } else if (traceVisible(ray_mask, o, tmin, ldir, ldist * (1.0 - 1.0e-4))) {
                            acc += contribution;
                        }
                    }
                }
            }

            // Next-event estimation for an HDR environment: one importance-sampled direction
            // (luminance x sin theta CDF, sky_importance.rs), MIS-weighted against BRDF sampling; the
            // BRDF-sampled rays that reach the sky take the complementary weight on miss.
            if (skySampled(env)) {
                float3 l;
                float pdf_env;
                let u1 = randf();
                let u2 = randf();
                let u3 = randf();
                let L = skySample(env, u1, u2, u3, l, pdf_env);
                let NoL = dot(world_normal, l);
                // A thin translucent surface also takes the sky (its sun) from behind.
                let through = lob.transmission > 0.0 && NoL < 0.0 && dot(l, surface_normal) < 0.0;
                let above = NoL > 0.0 && dot(l, surface_normal) > 0.0;
                if (pdf_env > 0.0 && (above || through) && luma(L) > 0.0) {
                    let f = evalLayered(-direction, world_normal, l, disney_mat, lob, true);
                    if (any(f > float3(0.0))) {
                        let w = pdf_env / (pdf_env + layeredPdf(-direction, world_normal, l, disney_mat, lob));
                        let c = mask * f * abs(NoL) * L * (w / pdf_env);
                        let o = leaveSurface(origin, surface_normal, l);
                        if (one_ray) {
                            pick.propose(c, o, tmin, l, tmax);
                        } else if (traceVisible(ray_mask, o, tmin, l, tmax)) {
                            acc += c;
                        }
                    }
                }
            }

            // Next-event estimation for the world's suns (its DirectionalLights, src/sky.rs): one
            // shadow ray per sun towards a uniformly sampled point on its disc. Direct sunlight is
            // the dominant, low-variance term, so it is not firefly-clamped; BRDF-sampled rays that
            // reach a disc afterwards return the sky only (`want_sun` false), so nothing is counted
            // twice.
            {
                for (uint si = 0u; si < env.sun_count; si++) {
                    let sun = env.suns[si];
                    let r1 = randf();
                    let r2 = randf();
                    let l = sampleCone(sun.direction.xyz, sun.direction.w, r1, r2);
                    let NoL = dot(world_normal, l);
                    // A thin translucent surface also takes the sun from behind (a back-lit leaf).
                    let through = lob.transmission > 0.0 && NoL < 0.0 && dot(l, surface_normal) < 0.0;
                    if (!through && (NoL <= 0.0 || dot(l, surface_normal) <= 0.0)) {
                        continue;
                    }
                    let f = evalLayered(-direction, world_normal, l, disney_mat, lob, true);
                    // Under an atmosphere the first sun's radiance is top-of-atmosphere: attenuate by
                    // the air above the hit, and (primary hits only) by the cloud shell.
                    var sun_radiance = sun.radiance.rgb;
                    if (ATMOSPHERE_ON && env.mode == 3u && si == 0u) {
                        sun_radiance *= atmoSunTransmittanceAt(origin);
                        if (bounce_idx == 0u) {
                            sun_radiance *= cloudShadow(origin);
                        }
                    }
                    if (any(f > float3(0.0)) && any(sun_radiance > float3(0.0))) {
                        // Shadow ray: any accepted hit terminates without running a closest-hit
                        // shader, so the payload's t only becomes 0 when the miss shader ran. Not
                        // opaque-forced: the any-hit shader lets it through alpha-masked cutouts.
                        let solid_angle = 2.0 * PI * (1.0 - sun.direction.w);
                        let c = mask * f * abs(NoL) * sun_radiance * solid_angle;
                        let o = leaveSurface(origin, surface_normal, l);
                        if (one_ray) {
                            pick.propose(c, o, tmin, l, tmax);
                        } else if (traceVisible(ray_mask, o, tmin, l, tmax)) {
                            acc += c;
                        }
                    }
                }
            }
            if (one_ray) {
                acc += pick.resolve(ray_mask);
            }

            // Thin layers before the base: through the surface with probability `transmission`
            // (weight = albedo: a Lambertian lobe drawn by its own cosine), off the coat with
            // probability clearcoat x Fresnel (weight F G1 / that). What is left samples the
            // base, whose weight the two cancel out of. Ordinary materials draw nothing here.
            // The view the sampler below draws from (its MIS pdf needs it after `direction` moves).
            let wo = -direction;
            if (lob.transmission > 0.0 && randf() < lob.transmission) {
                let r1 = randf();
                let r2 = randf();
                direction = sampleTransmission(world_normal, r1, r2);
                mask *= disney_mat.albedo;
                prev_pdf = layeredPdf(wo, world_normal, direction, disney_mat, lob);
                want_sun = false;
                cone_spread = max(cone_spread, 0.37);
                from_rough = true;
                continue;
            }
            if (lob.clearcoat > 0.0) {
                let q = lob.clearcoat * coatFresnel(dot(world_normal, -direction));
                if (randf() < q) {
                    let r1 = randf();
                    let r2 = randf();
                    float3 l;
                    float w;
                    if (!sampleCoat(-direction, world_normal, lob.coat_alpha, r1, r2, l, w)) {
                        break;
                    }
                    direction = l;
                    mask *= lob.clearcoat * w / q;
                    prev_pdf = layeredPdf(wo, world_normal, direction, disney_mat, lob);
                    want_sun = false;
                    cone_spread = max(cone_spread, 0.02 + 0.35 * lob.coat_alpha);
                    from_rough = lob.coat_alpha >= 0.35;
                    continue;
                }
            }

            float4 noise;
            if (ENABLE_BLUENOISE && bounce_idx == 0) {
                let bnoise = bnoiseBest(sample_idx);
                // bluenoise sampling on the first bounce, since the noise values
                // are used mostly exlusively, we'll copy them for both specular
                // and diffuse input. That way consective samples are also properly
                // blue.
                noise = float4(bnoise, bnoise);
            } else {
                let n0 = randf();
                let n1 = randf();
                let n2 = randf();
                let n3 = randf();
                noise = float4(n0, n1, n2, n3);
            }

            // stratified sampling on the second bounce
            // doing it on multiple bounces like this is invalid because
            // they will become correlated.
            // This also increases raytracing performances because
            // the rays are more uniform, increasing coherence of memory
            // accesses! (effect is about 1% on fps @3 spp)
            if (bounce_idx == 0) {
                noise.x = noise.x / float(SAMPLES) + float(sample_idx) / float(SAMPLES);
                noise.z = noise.z / float(SAMPLES) + float(sample_idx) / float(SAMPLES);
            }

            let brdf = sampleDisneyBRDF(noise, -direction, world_normal, disney_mat, direction);
            want_sun = false;
            // A failed sample (direction under the surface) carries nothing: the path ends.
            if (brdf.a < 0.0) {
                break;
            }
            // brdf.a is the base lobe's pdf; 0 marks a delta (mirror) bounce whose weight is
            // applied as is. Otherwise the MIS pdf is the whole layered sampler's.
            prev_pdf = brdf.a > 0.0001 ? layeredPdf(wo, world_normal, direction, disney_mat, lob) : 0.0;

            // Prevent going into the surface (leaks light through walls)
            if (dot(direction, surface_normal) < 0) {
                break;
            }

            mask *= brdf.a > 0.0001 ? brdf.rgb / brdf.a : brdf.rgb;
            // The cone opens with the lobe: a mirror keeps it, a diffuse bounce spreads wide.
            cone_spread = max(cone_spread, 0.02 + 0.35 * roughness * roughness);
            from_rough = roughness >= 0.35;
            // Russian roulette: a dim path ends here with probability 1 - q, a survivor carries
            // 1/q more (unbiased). Paths otherwise ran to max_bounces however little they
            // carried; the indirect bounces were most of the trace.
            if (pc.uniforms.roulette_from > 0u && bounce_idx + 1u >= pc.uniforms.roulette_from) {
                let q = clamp(max(mask.r, max(mask.g, mask.b)), 0.25, 1.0);
                if (randf() >= q) {
                    break;
                }
                mask /= q;
                rr_boost /= q;
            }
        }

        // Averaged over the samples, like the colour.
        {
            let g = guide_fog[px];
            guide_fog[px] = float4(g.rgb, g.a + fog_t / float(SAMPLES));
        }

        // The path ended: everything gathered after each cache vertex, divided by its mask
        // snapshot, is one sample of that voxel's outgoing radiance (zero radiance included --
        // darkness is learned too).
        for (uint si = 0u; si < (cache_blind ? 0u : sh_n); si++) {
            let delta = max(acc - sh_acc[si], float3(0.0));
            let m = sh_mask[si];
            sharcDeposit(sh_slot[si], float3(
                m.r > 1.0e-6 ? delta.r / m.r : 0.0,
                m.g > 1.0e-6 ? delta.g / m.g : 0.0,
                m.b > 1.0e-6 ? delta.b / m.b : 0.0));
        }
    }

    var mean = acc / float(SAMPLES);
    // Planetary atmosphere: aerial perspective on the primary hit and the cloud shell,
    // folded into the colour before Ray Reconstruction (deterministic: no per-frame noise).
    if (ATMOSPHERE_ON && pc.uniforms.environments[pc.uniforms.environment].mode == 3u) {
        mean = atmoComposite(initial_origin, initial_direction, primary_t, atmoBayer(px), mean);
    }
    // Next frame's auto-exposure metering (auto_exposure.slang): the incident meter (< 0 =
    // skipped) and the raw nits for the panel readout.
    // Meter what is seen: the surface through the fog plus the fog (composited after Ray
    // Reconstruction; metered without it, the exposure ignored the veil and it blew out).
    let fog = guide_fog[px];
    let fog_out = fog.rgb / max(pc.ae.input_exposure, 1.0e-20);
    let fog_out_t = fog.a;
    let lum_nits = dot(float3(0.2125, 0.7154, 0.0721), mean * fog_out_t + fog_out);
    let lum_at = 2u * (uint(px.y) * launch_size.x + uint(px.x));
    // The incident meter: the surface's light over its albedo, the fog's as is (it reflects
    // nothing; it is light in the air).
    let fog_nits = dot(float3(0.2125, 0.7154, 0.0721), fog_out);
    pc.lum[lum_at] = meter_skip ? -1.0 : (lum_nits - fog_nits) / max(meter_albedo, 0.04) + fog_nits;
    // The pair's second slot: the albedo the meter divides by (< 0 = not metered), for the
    // post-process pass, which re-meters from Ray Reconstruction's output (quad.slang) and
    // leaves the raw nits there.
    pc.lum[lum_at + 1u] = meter_skip ? -1.0 : max(meter_albedo, 0.04);
    // Ray Reconstruction takes PRE-EXPOSED linear HDR (it has no exposure handling of its
    // own -- RR guide 3.7), so the QUANTISED metered exposure is applied here (constant
    // between rare whole-EV steps: RR's history mixes frames, and a per-frame wobble in the
    // input exposure showed up as full-screen flicker) and the blit re-exposes to the look.
    // Finite and inside f16 range.
    let radiance = clamp(mean * pc.ae.input_exposure, float3(0.0), float3(65504.0));
    dlss_color[px] = float4(radiance, 1.0);
}
