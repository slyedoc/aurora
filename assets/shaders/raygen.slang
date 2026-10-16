// The path tracer's ray generation: camera paths with NEE (light table + ReSTIR DI, HDR
// environment, suns), ReSTIR GI's candidates and shading, ray portals, the DLSS Ray
// Reconstruction guides, the atmosphere composite and the auto-exposure meter.
import rt_types;
import rand;
import bsdf;
import atmosphere_rt;
import light_eval;
import fog;
import nrc;

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
// The same bindless range read as 3D images (the fog grid's volumes, src/fog_grid.rs).
[[vk::binding(201, 1)]] Sampler3D volumes[];


// Settings fixed when the pipeline is built (raytracing_pipeline.rs `PipelineFeatures`, from
// `RenderSettings`; changing one rebuilds it): the driver folds them, so a feature that is off
// is gone from the shader, registers and all, and loops get constant bounds. ReSTIR's uniforms
// still switch it per frame within a build that has it. (Constant 0 is free.)
[vk::constant_id(1)] const bool RESTIR_ON = false;
// Some environment's sky is the atmosphere (sky.rs `EnvironmentSkies::atmosphere`).
[vk::constant_id(2)] const bool ATMOSPHERE_ON = false;
// Paths per pixel and their maximum length.
[vk::constant_id(3)] const uint SAMPLES = 1;
// The loop's vertex cap: the largest bounce budget below, plus the primary vertex.
[vk::constant_id(4)] const uint MAX_BOUNCES = 11;
// Bounce budgets by kind (Falcor's path tracer): a path ends once one is exceeded. Diffuse: a
// diffuse lobe, or reflection rougher than SPECULAR_ROUGHNESS_THRESHOLD; specular: smoother
// reflection (mirrors); transmission: through the surface (glass, thin translucency).
[vk::constant_id(9)] const uint MAX_DIFFUSE_BOUNCES = 3;
[vk::constant_id(10)] const uint MAX_SPECULAR_BOUNCES = 3;
[vk::constant_id(11)] const uint MAX_TRANSMISSION_BOUNCES = 10;
// Light candidates resampled per shading point outside ReSTIR (1 = one plain CDF sample).
[vk::constant_id(5)] const uint LIGHT_CANDIDATES = 4;
// Perceptual roughness up to which the primary vertex traces a specular hit-distance ray for
// Ray Reconstruction; rougher surfaces report 0 (0 = no guide rays at all).
[vk::constant_id(6)] const float SPEC_HIT_ROUGHNESS = 0.3;
// Russian roulette from this bounce on (0 = off).
[vk::constant_id(7)] const uint ROULETTE_FROM = 1;
// ReSTIR PT (restir_gi.rs): sample 0's path from its reconnection vertex on goes to reservoirs,
// `gi_spatial` and `gi_shade`.
[vk::constant_id(8)] const bool GI_ON = false;
// A reconnection segment shorter than this share of the primary vertex's camera distance (~20
// pixel footprints) is no reconnection: there the camera's sub-pixel jitter alone swings the
// Jacobian, and reused samples in corners and at contacts ran hot (bright and coloured flecks)
// or failed their centimetre-long visibility ray.
static const float GI_RECONNECT_MIN = 0.03;
// The deepest reconnection vertex (x_k, k counted from the primary vertex x_0): a replay traces
// up to k - 1 rays, and the search holds Russian roulette back until it ends.
static const uint PT_MAX_K = 5u;

// A vertex a reconnection may start or end at: rough (no delta lobe, the reconnection's BSDF is
// evaluated), not glass (estimated by sampling alone), no clear coat (x_k's BSDF is evaluated
// again for shifted paths from its candidate record, which holds neither).
bool ptRough(float roughness, float transmission, Lobes lob) {
    return roughness >= pc.uniforms.gi_roughness && transmission < 0.01 && lob.clearcoat <= 0.0;
}

// A primary vertex replay can start from: not glass (its packed record, RestirSurface, holds no
// glass parameters; a coat it has). Past it any vertex replays (bsdf.slang samples every lobe from
// the path's numbers): glass, clear coat and mirrors may lie between x_0 and x_k.
bool ptReplayable(float transmission, Lobes lob) {
    return transmission < 0.01;
}

static const float SPECULAR_ROUGHNESS_THRESHOLD = 0.25;

// A scatter's bounce kind as a shift into a packed counter word (diffuse bits 0-7, specular
// 8-15, transmission 16-23): from the lobe sampled and the side it left by.
uint bounceKind(uint lobe, bool through, float roughness, float coat_alpha) {
    if (through) {
        return 16u;
    }
    let kind = lobe & LOBE_MASK;
    if (kind == LOBE_DIFFUSE) {
        return 0u;
    }
    let r = kind == LOBE_COAT ? sqrt(coat_alpha) : roughness;
    return r > SPECULAR_ROUGHNESS_THRESHOLD ? 0u : 8u;
}

// Counts a scatter of kind `shift`; false once its budget is exceeded.
bool bounceWithinBudget(inout uint bounces, uint shift) {
    bounces += 1u << shift;
    let n = (bounces >> shift) & 0xFFu;
    let limit = shift == 0u ? MAX_DIFFUSE_BOUNCES
                            : (shift == 8u ? MAX_SPECULAR_BOUNCES : MAX_TRANSMISSION_BOUNCES);
    return n <= limit;
}

// A counter word in 8 bits (diffuse 2, specular 3, transmission 3, saturating): the mirror and
// glass chain in front of a primary vertex, kept in its surface record for replay.
uint packBounces(uint b) {
    return min(b & 0xFFu, 3u) | (min((b >> 8) & 0xFFu, 7u) << 2) | (min((b >> 16) & 0xFFu, 7u) << 5);
}

uint unpackBounces(uint p) {
    return (p & 3u) | (((p >> 2) & 7u) << 8) | (((p >> 5) & 7u) << 16);
}

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

// visibleInline's fallback, a real call (the shift is one too: see ptShift).
[noinline]
bool visibleTraced(uint mask, float3 origin, float3 direction, float tmax) {
    return traceVisible(mask, origin, 0.0, direction, tmax);
}

// A visibility test by ray query (inline: no shader call, nothing held across it spilled -- the
// shift's TraceRay visibility was ~85 % of gi_spatial). Opaque triangles resolve here; a candidate
// only a shader can decide (an alpha-tested or glass triangle's any-hit, a procedural primitive's
// intersection) hands the ray to traceVisible.
bool visibleInline(uint mask, float3 origin, float3 direction, float tmax) {
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.0;
    ray.Direction = direction;
    ray.TMax = tmax;
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, mask, ray);
    var shader_needed = false;
    while (q.Proceed()) {
        // The analytic sphere (hit offset 0, sphere.slang): unit diameter in object space.
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE
            && q.CandidateInstanceContributionToHitGroupIndex() == 0u) {
            let o = q.CandidateObjectRayOrigin();
            let d = q.CandidateObjectRayDirection();
            let b = dot(o, d);
            let a = dot(d, d);
            let c = dot(o, o) - 0.25;
            let disc = b * b - a * c;
            if (disc >= 0.0) {
                let sq = sqrt(disc);
                let t0 = (-b - sq) / a;
                let t1 = (-b + sq) / a;
                let t = t0 >= ray.TMin ? t0 : t1;
                if (t >= ray.TMin && t <= q.CommittedRayT()) {
                    q.CommitProceduralPrimitiveHit(t);
                }
            }
        } else {
            shader_needed = true;
        }
    }
    if (q.CommittedStatus() != COMMITTED_NOTHING) {
        return false;
    }
    return shader_needed ? visibleTraced(mask, origin, direction, tmax) : true;
}

// One shadow ray for several light strategies at a vertex: each proposes its finished,
// unshadowed contribution; one is kept with probability proportional to the square root of its
// brightness and, visible, adds that contribution / its pick probability (unbiased; one ray
// instead of one per strategy). The root tempers a dominant light: in shade the sun's unshadowed
// value would otherwise win nearly every pick, its ray blocked, and the sky's fill go unsampled.
// Every proposal's ray leaves the vertex's surface (`resolve` rebuilds its origin), so a pick
// holds only the kept contribution over its weight, its direction and its length: it is live
// through all of the vertex's light sampling, where the raygen's registers peak.
struct ShadowPick {
    float wsum;
    float3 scaled;
    float3 direction;
    float tmax;

    [mutating]
    void propose(float3 c, float3 d, float t1) {
        let wc = sqrt(luma(c));
        if (!(wc > 0.0)) {
            return;
        }
        wsum += wc;
        if (randf() * wsum < wc) {
            scaled = c / wc;
            direction = d;
            tmax = t1;
        }
    }

    // The kept proposal's shadow ray from `origin` off `normal`; its contribution over its pick
    // probability when visible.
    float3 resolve(uint mask, float3 origin, float3 normal) {
        if (wsum <= 0.0) {
            return float3(0.0);
        }
        if (!traceVisible(mask, leaveSurface(origin, normal, direction), 0.0, direction, tmax)) {
            return float3(0.0);
        }
        return scaled * wsum;
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

// All the fog at a point: the environment's (`base`) and every volume's (fog.slang).
float fogDensityAt(EnvironmentGpu* env, float3 x, bool base) {
    return fogDensityAtU(pc.uniforms, env, x, base);
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


// Henyey-Greenstein phase function, per steradian.
float fogPhase(EnvironmentGpu* env, float cos_theta) {
    return fogPhaseG(env.fog_scatter, cos_theta);
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
// own fog in too (not on a ray to the sky, which the old scatter never fogged either). Its sky
// ray has a payload of its own: the camera ray's stays as it is.
void fogMarch(EnvironmentGpu* env, uint ray_env, uint ray_mask, float tmax, float3 o, float3 d,
              float seg, bool base, inout float3 mask, inout float3 acc) {
    let depth = fogMajorant(env, o, d, seg, base) * seg;
    if (depth <= 1.0e-6) {
        return;
    }
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
        var Ls = fogSunLight(env, ray_mask, tmax, x, d);
        if (i == sky_step) {
            // Phase-sampled, so the phase over its pdf is 1; one step in n carries it. Its own
            // payload: one held across the march's sun rays doubled the caller's stack.
            let w = sampleFogScatterDirection(env, d);
            HitPayload payload = {};
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
}

// The suns' light at fog point `x`, as the fog scatters it along `d`. The HDR sky's sun is
// two discs (core, halo) on one direction: one shadow ray serves every disc that shares it.
float3 fogSunLight(EnvironmentGpu* env, uint ray_mask, float tmax, float3 x, float3 d) {
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










// The light table as light_eval.slang reaches it.
LightCtx lightCtx() {
    LightCtx ctx;
    ctx.lights = pc.lights;
    ctx.instances = pc.cur_instances;
    ctx.emissive_boost = pc.uniforms.emissive_boost;
    return ctx;
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

// ---- sky importance sampling (src/sky_gen.rs) ---------------------------------------------

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

// Draws a direction from the sky's luminance CDF (jittered inside the texel); returns its
// radiance, with the direction and its solid-angle pdf. The guide table narrows the binary
// search to the texels under u1's 1/N slice (~2), not all N: 17 dependent loads to ~1.
float3 skySample(EnvironmentGpu* env, float u1, float u2, float u3, out float3 dir, out float pdf) {
    let w = SKY_IMPORTANCE_W;
    let n = w * SKY_IMPORTANCE_H;
    let j = min(uint(u1 * float(n)), n - 1u);
    uint lo = asuint(env.importance[2u * n + 1u + j]);
    uint hi = min(asuint(env.importance[2u * n + 2u + j]) + 1u, n);
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
    return env.color.rgb * texel * env.sky_brightness;
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

uint3 fogGridSize() {
    return fogGridSizeU(pc.uniforms);
}

// A fog grid volume at normalized `uvw`, clamped to its edge texels' centres (the bindless
// sampler repeats): the texture unit's trilinear filter for eight buffer reads.
float4 fogVolume(uint tex, float3 uvw, uint3 size) {
    let lo = 0.5 / float3(size);
    return volumes[tex].SampleLevel(clamp(uvw, lo, 1.0 - lo), 0.0);
}

// The soft grid at a cell coordinate (fogGridCoord's: centres at +0.5).
float3 fogGridSoftAt(uint3 size, float3 g) {
    return fogVolume(pc.uniforms.fog_grid_soft_tex, g / float3(size), size).rgb;
}

float3 fogGridPoint(float2 uv, float s, float dist) {
    return fogGridPointU(pc.uniforms, uv, s, dist);
}

// One light-table sample at fog point `x`, as the fog scatters it along `d`: RIS over
// LIGHT_CANDIDATES from the power CDF, one shadow ray for the survivor.
float3 fogTableLight(EnvironmentGpu* env, uint ray_mask, float3 x, float3 d, inout HitPayload payload) {
    if (pc.uniforms.light_entries == 0u) {
        return float3(0.0);
    }
    SurfaceMaterial mat = {};
    Lobes lob = {};
    let M = max(1u, LIGHT_CANDIDATES);
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
    return sel_val * (wsum / (float(M) * sel_phat));
}

// One thread per cell lit this frame (one of every stride x stride block, fogGridPhase;
// src/fog_grid.rs), after `fog_reproject` carried every cell's history into this frame's grid:
// a random point in the cell lit by the suns, one phase-sampled sky ray and the light table,
// blended into that history.
[shader("raygeneration")]
void fog_grid() {
    let size = fogGridSize();
    let stride = max(pc.uniforms.fog_grid_stride, 1u);
    let id = uint3(DispatchRaysIndex().xy * stride + fogGridPhase(pc.uniforms), DispatchRaysIndex().z);
    if (pc.uniforms.fog_grid_cells == 0u || any(id >= size)) {
        return;
    }
    let i = (id.z * size.y + id.y) * size.x + id.x;
    HitPayload payload;
    initRandom(uint2(pc.uniforms.fog_grid_cells, 1u), uint2(i, 0u), pc.uniforms.frame);
    let tmax = 1.0e6;
    let ray_env = pc.uniforms.environment;
    let ray_mask = pc.uniforms.camera_mask;
    let env = pc.uniforms.environments + ray_env;
    let dist = env.fog_grid_distance;
    let cell = float3(id);
    let grid = float3(size);
    let r = float3(randf(), randf(), randf());
    let x = fogGridPoint((cell.xy + r.xy) / grid.xy, (cell.z + r.z) / grid.z, dist);
    let cam = mat4MulVecAt(&pc.uniforms.fog_grid_inverse_view[0], float4(0.0, 0.0, 0.0, 1.0)).xyz;
    let centre = fogGridPoint((cell.xy + 0.5) / grid.xy, (cell.z + 0.5) / grid.z, dist);
    let d = normalize(centre - cam);

    var L = fogSunLight(env, ray_mask, tmax, x, d);
    let w = sampleFogScatterDirection(env, d);
    payload.t = -1.0;
    payload.miss_flags = ray_env << 1;
    trace(FLAGS_SHADOW, ray_mask, x, 0.0, w, tmax, payload);
    if (payload.t == 0.0) {
        L += payload.emission;
    }
    L += fogTableLight(env, ray_mask, x, d, payload);
    if (any(isnan(L)) || any(isinf(L))) {
        L = float3(0.0);
    }

    let prev = pc.uniforms.fog_grid_cur[i];
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
        let Ls = fogGridSoftAt(size, fogGridCoord(&pc.uniforms.fog_grid_view_proj[0], x, size, dist));
        L += T * FOG_ALBEDO * Ls * (1.0 - step_t);
        T *= step_t;
    }
    acc += mask * L;
    mask *= T;
}

// The fog from the grid's eye to `x` (in-scatter, transmittance): the column sums, bilinear
// across columns and linear between slice sides; past the grid the far slice's light and the
// density half way out cover the rest in one step.
float4 fogGridSumAt(EnvironmentGpu* env, float3 x) {
    let size = fogGridSize();
    let dist = env.fog_grid_distance;
    let clip = mat4MulVecAt(&pc.uniforms.fog_grid_view_proj[0], float4(x, 1.0));
    let w = max(clip.w, 1.0e-4);
    let uv = clip.xy / w * 0.5 + 0.5;
    // Slice 0 of the sums volume is the eye; slice k + 1 the sums through slice k.
    let z = sqrt(saturate(w / dist)) * float(size.z);
    let vsize = uint3(size.xy, size.z + 1u);
    var r = fogVolume(pc.uniforms.fog_grid_sum_tex, float3(uv, (z + 0.5) / float(vsize.z)), vsize);
    if (clip.w > dist) {
        let cam = mat4MulVecAt(&pc.uniforms.fog_grid_inverse_view[0], float4(0.0, 0.0, 0.0, 1.0)).xyz;
        let xm = cam + (x - cam) * (0.5 * (dist + clip.w) / clip.w);
        let sigma = fogDensityAt(env, xm, true);
        if (sigma > 0.0) {
            let step_t = exp(-sigma * length(x - cam) * (clip.w - dist) / clip.w);
            let Ls = fogGridSoftAt(size, fogGridCoord(&pc.uniforms.fog_grid_view_proj[0], xm, size, dist));
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

// The DLSS guides at this pixel's guide vertex (sample 0's first accepted surface, or the sky)
// and the exposure meter's divisor there (< 0: not metered). `guide_dist`: how far the camera
// ray ran before this segment (smooth glass in front: the guides sit at a virtual point along it).
float writeGuides(int2 px, HitPayload payload, float3 origin, float3 direction, float3 world_normal,
                  float3 surface_normal, float2 lens, uint2 launch_id, uint2 launch_size,
                  float guide_dist, uint ray_mask, uint ray_env, float cone_width, float cone_spread) {
    let tmin = 0.0;
    let tmax = 1.0e6;
    var meter_albedo = 1.0;
    if (payload.t == 0.0) {
        meter_albedo = -1.0;
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
        let sky_cur = mat4MulVecAt(&pc.uniforms.view_proj[0], float4(direction, 0.0));
        let sky_prev = mat4MulVecAt(&pc.uniforms.prev_view_proj[0], float4(direction, 0.0));
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
        float3 cam_origin, cam_dir;
        lensRay(lens, launch_id, launch_size, cam_origin, cam_dir);
        let virtual_pos = cam_origin + cam_dir * (guide_dist + payload.t);
        let view_pos = mat4MulVecAt(&pc.uniforms.view[0], float4(virtual_pos, 1.0));
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
        meter_albedo = any(payload.emission != float3(0.0))
            ? -1.0
            : dot(float3(0.2125, 0.7154, 0.0721), albedo * (1.0 - g_metallic) + g_spec);
        guide_depth[px] = float4(min(linear_depth, 1.0e7), 0.0, 0.0, 0.0);
        // Motion in UV units (y down), current minus previous; NGX scales by -(w,h).
        let cur_clip = mat4MulVecAt(&pc.uniforms.view_proj[0], float4(virtual_pos, 1.0));
        // The end vertex's own frame-to-frame motion, applied to the virtual point
        // (identical to prev_world_pos when no glass was traversed).
        let prev_clip = mat4MulVecAt(&pc.uniforms.prev_view_proj[0],
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
        if (g_roughness <= SPEC_HIT_ROUGHNESS) {
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
    return meter_albedo;
}

// The camera ray's fog over one segment into `veil` (pre-exposed by `fog_scale`, through the
// transmittance so far), and that transmittance on. The surface behind reaches Ray
// Reconstruction unveiled; the composite applies both (`guide_fog`).
void cameraFog(EnvironmentGpu* env, uint ray_env, uint ray_mask, float3 origin, float3 direction,
               float hit_t, float3 mask, bool in_view, float fog_scale, inout float3 veil,
               inout float fog_t) {
    let tmax = 1.0e6;
    // The camera ray marches its fog (the shafts, without the noise of one random
    // scatter); a ray to the sky marches only as far as the volumes reach.
    let seg = hit_t > 0.0 ? hit_t : fogVolumesReach(origin, direction);
    var fog_mask = mask;
    var fog_acc = float3(0.0);
    if (seg > 0.0 && in_view && pc.uniforms.fog_grid_cells > 0u
        && pc.uniforms.fog_grid_fresh == 0u) {
        // A surface's segment takes the column sums; one to the sky only its
        // volumes, which the sums (environment fog included) cannot leave out.
        if (hit_t > 0.0) {
            fogGridSegment(env, origin, direction, seg, fog_mask, fog_acc);
        } else {
            fogMarchGrid(env, origin, direction, seg, false, fog_mask, fog_acc);
        }
    } else if (seg > 0.0) {
        fogMarch(env, ray_env, ray_mask, tmax, origin, direction, seg,
                 hit_t > 0.0, fog_mask, fog_acc);
    }
    // The surface behind reaches RR unveiled; the composite applies both.
    let seg_t = luma(mask) > 0.0 ? luma(fog_mask) / luma(mask) : 1.0;
    veil += fog_t * fog_acc * fog_scale;
    fog_t *= seg_t;
}

// The primary pass (docs/plans/primary-pass.html), one thread per traced pixel, ahead of ReSTIR
// and the path tracer: sample 0's camera ray through alpha cutouts and portals to its first
// accepted surface, which it records (PrimaryHit) for the path tracer to resume from; on the
// way the DLSS guides, the camera ray's fog and the exposure meter. Behind smooth glass the
// guides follow the dominant Fresnel lobe to the first surface that is not (RTXPT's stable
// planes), every frame; the radiance path makes its own lobe choice.
[shader("raygeneration")]
void primary() {
    let launch_id = DispatchRaysIndex().xy;
    let launch_size = DispatchRaysDimensions().xy;
    let px = int2(launch_id);
    let tmin = 0.0;
    let tmax = 1.0e6;
    initRandom(launch_size, launch_id, pc.uniforms.frame * 7u + 5u);

    if (pc.uniforms.pull_focus_x == launch_id.x && pc.uniforms.pull_focus_y == launch_id.y) {
        HitPayload focus = {};
        trace(FLAGS_OPAQUE, 0xFF, cameraOrigin(), tmin, pixelDirection(launch_id, launch_size), tmax,
              focus);
        if (focus.t != 0.0) {
            pc.focus[0] = focus.t;
        }
    }

    let fog_scale = pc.ae.input_exposure;
    var veil = float3(0.0);
    let offsetR = sqrt(randf());
    let offsetA = randf() * 2.0 * PI;
    let lens = pc.uniforms.aperture * float2(offsetR * cos(offsetA), offsetR * sin(offsetA));
    float3 origin, direction;
    lensRay(lens, launch_id, launch_size, origin, direction);
    let mask = float3(1.0);
    var surf_n = float3(0.0);
    var ray_mask = pc.uniforms.camera_mask;
    var ray_env = pc.uniforms.environment;
    var in_view = true;
    var guides_written = false;
    var guide_dist = 0.0;
    var fog_t = 1.0;
    var cone_width = 0.0;
    let cone_spread = 2.0 * pc.uniforms.inverse_projection[5] / float(launch_size.y);
    var meter_albedo = 1.0;
    var primary_t = 0.0;
    // The record goes to memory field by field as it is decided, never held across a trace.
    let rec = pc.primary_hits + (launch_id.y * launch_size.x + launch_id.x);
    rec.state = PRIMARY_RETRACE;
    var recorded = false;
    var glass = 0u;
    // The camera chain shares the path's bounce budget (each cutout or portal is a bounce);
    // past the record only the guides follow, a few smooth-glass vertices at most.
    for (uint bounce = 0u; bounce < MAX_BOUNCES + 4u; bounce++) {
        if (!recorded && bounce >= MAX_BOUNCES) {
            rec.state = PRIMARY_ENDED;
            rec.origin = float3(0.0);
            recorded = true;
            break;
        }
        HitPayload payload = {};
        payload.miss_flags = (recorded ? 0u : 1u) | (ray_env << 1);
        payload.cone = float2(cone_width, cone_spread);
        let trace_mask = recorded ? ray_mask : (ray_mask & ~pc.uniforms.primary_hidden);
        RayDesc ray;
        ray.Origin = any(surf_n != float3(0.0)) ? leaveSurface(origin, surf_n, direction) : origin;
        ray.Direction = direction;
        ray.TMin = tmin;
        ray.TMax = tmax;
        // No reorder: camera rays are coherent already (the sort cost a third of the pass).
        HitObject hit = HitObject::TraceRay(topLevelAS, FLAGS_OPAQUE, trace_mask, 0, 0, 0, ray, payload);
        // The hit's identity, written before it is shaded so the hit object dies here: whether
        // this segment is the one recorded is only known after shading.
        if (!recorded) {
            let kind = hit.GetHitKind();
            let triangle = hit.IsHit() && (kind == HIT_KIND_TRIANGLE_FRONT_FACE || kind == HIT_KIND_TRIANGLE_BACK_FACE);
            // The analytic sphere: hit group 0 (sbt.rs); other procedural hits re-trace.
            let sphere = hit.IsHit() && !triangle && hit.GetShaderTableIndex() == 0u;
            rec.hit_kind = triangle ? kind : (sphere ? PRIMARY_SPHERE : 0u);
            if (triangle || sphere) {
                rec.origin = ray.Origin;
                rec.direction = direction;
                rec.instance = hit.GetInstanceIndex();
                rec.geometry = hit.GetGeometryIndex();
                rec.primitive = hit.GetPrimitiveIndex();
                if (triangle) {
                    let a = hit.GetAttributes<BuiltInTriangleIntersectionAttributes>();
                    rec.bary = packUnorm16(a.barycentrics.x) | (packUnorm16(a.barycentrics.y) << 16);
                } else {
                    // The object-space point lies on the radius-0.5 sphere: its direction, octahedral.
                    let o = float32x3_to_oct(normalize(hit.GetAttributes<PrimarySphereAttributes>().point)) * 0.5 + 0.5;
                    rec.bary = packUnorm16(o.x) | (packUnorm16(o.y) << 16);
                }
                rec.cone_width = cone_width;
            }
        }
        HitObject::Invoke(topLevelAS, hit, payload);
        if (bounce == 0u) {
            primary_t = payload.t;
        }
        cone_width += cone_spread * payload.t;
        let surface_normal = oct_to_float32x3(payload.surface_and_world_normal.xy);
        let world_normal = bendTowardViewer(
            oct_to_float32x3(payload.surface_and_world_normal.zw), -direction);

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
                in_view = false;
                continue;
            }
        }

        // Alpha-cutout pass-through, as the path tracer decides it (masked materials only).
        var pass_through = false;
        if (payload.t != 0.0 && all(payload.emission == float3(0.0)) && hitPayloadGetMasked(payload)) {
            if (all(payload.color.rgb == float3(0.0))) {
                pass_through = true;
            } else if (payload.color.a < 1.0) {
                pass_through = randf() < 1.0 - payload.color.a;
            }
        }
        let guide_defer = payload.t != 0.0
            && hitPayloadGetTransmission(payload) > 0.5
            && hitPayloadGetRoughness(payload) < 0.1;
        let t = payload.t;
        let ends = any(payload.emission != float3(0.0)) || t == 0.0;
        let hit_point = origin + direction * t;
        let env = pc.uniforms.environments + ray_env;
        let fog_any = env.fog_density > 0.0 || pc.uniforms.fog_volume_count > 0u;
        let camera_segment = !recorded;
        // The record and ReSTIR's surface first, while the payload is at hand: what follows
        // traces (the spec probe, the fog's shadow rays), and the payload must not live across.
        if (!recorded) {
            if (env.fog_steps == 0u && fog_any) {
                // Random-scatter fog: the camera segment itself may scatter, which only the path
                // tracer can decide. It traces from the camera (the guides are written here).
                rec.state = PRIMARY_RETRACE;
                recorded = true;
            } else if (ends) {
                rec.state = PRIMARY_ENDED;
                rec.origin = mask * payload.emission;
                recorded = true;
            } else if (!pass_through) {
                recorded = true;
                if (rec.hit_kind != 0u) {
                    rec.t = t;
                    rec.state = PRIMARY_HIT | (bounce << 2) | ((ray_mask & 0xFFu) << 8) | ((ray_env & 7u) << 16);
                    // ReSTIR's surface: DI's passes run between this pass and the path tracer,
                    // GI's after it.
                    if ((RESTIR_ON || GI_ON) && pc.restir_surfaces != nullptr) {
                        restirSurfaceStore(pc.restir_surfaces + (launch_id.y * launch_size.x + launch_id.x), restirSurfacePack(
                            hit_point, pc.uniforms.frame, payload.prev_world_pos,
                            mat4MulVecAt(&pc.uniforms.view_proj[0], float4(hit_point, 1.0)).w, payload.color.rgb,
                            hitPayloadGetMetallic(payload), hitPayloadGetRoughness(payload), payload.lobes,
                            mask, ray_mask, world_normal, surface_normal, -direction));
                    }
                }
            }
        }
        let inside = hitPayloadGetInside(payload);
        let refract_index = payload.refract_index;
        // The payload's last use: the guides (their spec probe traces after reading it).
        if (!guides_written && !pass_through && !guide_defer) {
            guides_written = true;
            meter_albedo = writeGuides(px, payload, origin, direction, world_normal, surface_normal,
                                       lens, launch_id, launch_size, guide_dist, ray_mask, ray_env,
                                       cone_width, cone_spread);
        }
        if (!guides_written) {
            guide_dist += t;
        }
        if (camera_segment && env.fog_steps > 0u && fog_any) {
            cameraFog(env, ray_env, ray_mask, origin, direction, t, mask, in_view, fog_scale, veil,
                      fog_t);
        }
        if (ends) {
            break;
        }
        if (pass_through) {
            origin = hit_point;
            surf_n = surface_normal;
            continue;
        }
        if (guides_written || !guide_defer || ++glass > 4u) {
            break;
        }
        // The guides go on through the glass along its dominant Fresnel lobe.
        let n1 = inside ? refract_index : 1.0;
        let n2 = inside ? 1.0 : refract_index;
        let eta = n1 / n2;
        let costi = dot(world_normal, -direction);
        let k = 1.0 - eta * eta * (1.0 - costi * costi);
        var p_reflect = 1.0;
        if (k >= 0.0 && costi > 0.0) {
            let costt = sqrt(k);
            let spol = (n1 * costi - n2 * costt) / (n1 * costi + n2 * costt);
            let ppol = (n1 * costt - n2 * costi) / (n1 * costt + n2 * costi);
            p_reflect = 0.5 * (spol * spol + ppol * ppol);
        }
        direction = p_reflect > 0.5
            ? reflect(direction, world_normal)
            : normalize(eta * direction + world_normal * (eta * costi - sqrt(max(k, 0.0))));
        origin = hit_point;
        surf_n = surface_normal;
    }
    rec.meter_albedo = meter_albedo;
    rec.primary_t = primary_t;
    guide_fog[px] = float4(min(veil, float3(65504.0)), fog_t);
}

// ReSTIR PT's shading (restir_gi.rs), one thread per traced pixel after the reuse passes: the
// final reservoir's contribution (its target vector at this pixel times W), added to the noisy
// colour and the exposure meter as `main` would have.
[shader("raygeneration")]
void gi_shade() {
    let launch_id = DispatchRaysIndex().xy;
    let launch_size = DispatchRaysDimensions().xy;
    let idx = launch_id.y * launch_size.x + launch_id.x;
    if (!GI_ON || pc.gi_final == nullptr) {
        return;
    }
    let r = giReservoirLoad(pc.gi_final + idx);
    let debug = pc.gi_debug;
    if (debug != nullptr) {
        debug[2u + 2u * idx] = 0u;
        debug[3u + 2u * idx] = r.frame == pc.uniforms.frame ? GI_STATE_EMPTY : GI_STATE_OFF;
    }
    if (r.frame != pc.uniforms.frame || !(r.W > 0.0 && r.W < 1.0e20)) {
        return;
    }
    let c = unpackRadiance(r.C);
    if (!(luma(c) > 0.0)) {
        return;
    }
    // No visibility ray: every target the reuse passes resample includes the reconnection's
    // visibility (an own sample is visible by construction).
    let s = restirSurfaceLoad(pc.restir_surfaces + idx);
    var gi = c * r.W * restirThroughput(s) / float(SAMPLES);
    if (!(luma(gi) < 1.0e20)) {
        return;
    }
    if (ATMOSPHERE_ON && pc.uniforms.environments[pc.uniforms.environment].mode == 3u) {
        gi *= unpackRgb9e5(s.gi_view_t);
    }
    let px = int2(launch_id);
    let col = dlss_color[px];
    dlss_color[px] = float4(min(col.rgb + gi * pc.ae.input_exposure, float3(65504.0)), col.a);
    if (debug != nullptr) {
        debug[2u + 2u * idx] = packRgb9e5(gi * pc.ae.input_exposure);
        debug[3u + 2u * idx] = (r.flags & GI_FRESH) == 0u ? GI_STATE_NEIGHBOUR : GI_STATE_OWN;
    }
    // The meter as `main` writes it: the surface's light through the fog over its albedo.
    if (pc.lum != nullptr) {
        let meter_albedo = pc.primary_hits[idx].meter_albedo;
        if (meter_albedo >= 0.0) {
            pc.lum[2u * idx] += dot(float3(0.2125, 0.7154, 0.0721), gi) * guide_fog[px].a
                / max(meter_albedo, 0.04);
        }
    }
}

// ---- ReSTIR PT's hybrid shift ---------------------------------------------------------------

// A vertex the shift evaluates the reconnection at: position, shading and geometric normals, the
// view toward the previous vertex, the material.
struct PtVertex {
    float3 x;
    float3 n;
    float3 ng;
    float3 v;
    SurfaceMaterial mat;
    Lobes lob;
};

PtVertex ptPrimaryVertex(RestirSurface s) {
    PtVertex p;
    p.x = s.position;
    p.n = oct_to_float32x3(unpackHalf2(s.normal_oct));
    p.ng = oct_to_float32x3(unpackHalf2(s.geo_normal_oct));
    p.v = oct_to_float32x3(unpackHalf2(s.view_oct));
    p.mat = surfaceMaterial(restirAlbedo(s), restirMetallic(s), restirRoughness(s));
    p.lob = unpackLobes(s.material);
    return p;
}

// Random replay (Lin et al. 2022, RTXDI's hybrid shift): x_1 .. x_{k-1} rebuilt from primary
// vertex `s` with the sample's seed, the BSDF numbers `main` drew. `T` is the replayed prefix's
// weight (f cos / pdf per bounce: primary sample space, so the replay needs no Jacobian of its
// own); `last` is x_{k-1}. False when the replayed path could not be the sample's: a vertex it
// may not cross (glass, thin layers, alpha-tested, an emitter or the sky) or a reconnection pair
// found before k, or x_{k-1} not rough.
struct PtReplay {
    bool ok;
    PtVertex last;
    float3 T;
    // The bounce counters reaching `last` (from the primary vertex's chain on).
    uint bounces;
};

PtReplay ptReplay(RestirSurface s, uint seed, uint k, float thr) {
    var p = ptPrimaryVertex(s);
    PtReplay res;
    res.ok = false;
    res.T = float3(1.0);
    res.last = p;
    // The budgets bind the replay as they bound the path: past one, `main` would have stopped.
    var bounces = unpackBounces(s.throughput_b_mask >> 24);
    if (!ptReplayable(0.0, p.lob)) {
        return res;
    }
    var rough = ptRough(p.mat.roughness, 0.0, p.lob);
    let ray_mask = restirRayMask(s);
    for (uint i = 0u; i + 1u < k; i++) {
        // The BSDF's numbers at vertex i, as `main` drew them.
        var sg = PathSampler(seed, i);
        float3 dir;
        float bpdf;
        float3 weight;
        uint lobe;
        if (!surfaceBSDF(p.v, p.n, p.mat, p.lob).sample(sg, dir, bpdf, weight, lobe)) {
            return res;
        }
        let through = dot(dir, p.n) < 0.0;
        if (through ? dot(dir, p.ng) >= 0.0 : dot(dir, p.ng) <= 0.0) {
            return res;
        }
        if (!bounceWithinBudget(bounces, bounceKind(lobe, through, p.mat.roughness, p.lob.coat_alpha))) {
            return res;
        }
        res.T *= weight;
        HitPayload payload = {};
        payload.miss_flags = 0u;
        payload.cone = float2(0.0, 0.01);
        trace(FLAGS_OPAQUE, ray_mask, leaveSurface(p.x, p.ng, dir), 0.0, dir, 1.0e7, payload);
        if (payload.t == 0.0 || any(payload.emission != float3(0.0)) || hitPayloadGetMasked(payload)) {
            return res;
        }
        let transmission = hitPayloadGetTransmission(payload);
        let lob = unpackLobes(payload.lobes);
        let roughness = hitPayloadGetRoughness(payload);
        PtVertex q;
        q.x = p.x + dir * payload.t;
        q.ng = oct_to_float32x3(payload.surface_and_world_normal.xy);
        q.n = bendTowardViewer(oct_to_float32x3(payload.surface_and_world_normal.zw), -dir);
        if (dot(q.ng, dir) > 0.0) {
            q.ng = -q.ng;
        }
        q.v = -dir;
        // Glass as `main` saw it: eta for the side the ray came from, the medium's absorption
        // along the segment inside.
        let inside = hitPayloadGetInside(payload);
        let ior = max(payload.refract_index, 1.0e-3);
        q.mat = surfaceMaterial(payload.color.rgb, hitPayloadGetMetallic(payload), roughness,
                                transmission, inside ? ior : 1.0 / ior);
        q.lob = lob;
        if (inside && transmission > 0.0) {
            res.T *= exp(-payload.t * payload.absorption);
        }
        let q_rough = ptRough(roughness, transmission, lob);
        // `main` would have reconnected here, before k.
        if (rough && q_rough && length(q.x - p.x) > thr) {
            return res;
        }
        p = q;
        rough = q_rough;
    }
    res.last = p;
    res.bounces = bounces;
    res.ok = rough;
    return res;
}

// The sample's target vector at x_{k-1} = `p` with prefix weight `T`: the reconnection to x_k
// (`x_s`, normal `n_s`, radiance `L`) through p's BSDF and cosine, and its geometry term
// |cos theta_k| / d^2 (the Jacobian's half). Zero when x_k faces away or the segment is short.
float3 ptReconnect(PtVertex p, float3 T, GiCandidate c, float3 x_s, float3 n_s, float thr,
                   out float geo) {
    geo = 0.0;
    let d = x_s - p.x;
    let d2 = max(dot(d, d), 1.0e-8);
    let w = d * rsqrt(d2);
    let cos_s = -dot(n_s, w);
    if (cos_s <= 0.0 || sqrt(d2) <= thr) {
        return float3(0.0);
    }
    // In front, or behind a thin translucent surface (its transmission lobe).
    let NoL = dot(p.n, w);
    let front = NoL > 0.0 && dot(w, p.ng) > 0.0;
    let back = p.lob.transmission > 0.0 && NoL < 0.0 && dot(w, p.ng) < 0.0;
    if (!front && !back) {
        return float3(0.0);
    }
    geo = cos_s / d2;
    return T * surfaceEval(p.v, p.n, w, p.mat, p.lob) * ptSampleRadiance(c, -w);
}

// A sample shifted to a primary vertex: its target vector there (visibility of the reconnection
// segment included: the target is the integrand, so the reuse is unbiased), the Jacobian, and the
// replayed x_{k-1}'s position and geometric normal (the segment gi_shade tests).
struct PtShift {
    float3 f;
    float J;
    float3 from_x;
    float3 from_ng;
};

// Never inlined: inlined into gi_spatial's neighbour loop, the code came out wrong (the same shift
// of a pixel's own sample failed its facing test on 45 % of pixels inside the loop and passed
// outside it; pairwise MIS lost its pairing and reuse read ~2 % bright). A real call is right.
[noinline]
PtShift ptShift(RestirSurface s, float thr, GiCandidate c) {
    PtShift r;
    r.f = float3(0.0);
    r.J = 0.0;
    r.from_x = s.position;
    r.from_ng = float3(0.0, 1.0, 0.0);
    let k = c.k & 15u;
    if (k == 0u) {
        return r;
    }
    let rp = ptReplay(s, c.seed, k, thr);
    if (!rp.ok) {
        return r;
    }
    let from = rp.last;
    // The reconnection is x_{k-1}'s scatter: diffuse (it is rough), or through it.
    var bounces = rp.bounces;
    if (!bounceWithinBudget(bounces, dot(c.x_s - from.x, from.n) < 0.0 ? 16u : 0u)) {
        return r;
    }
    let T = rp.T;
    float geo;
    let n_s = oct_to_float32x3(unpackHalf2(c.n_s_oct));
    let f = ptReconnect(from, T, c, c.x_s, n_s, thr, geo);
    if (!(luma(f) > 0.0) || !(c.geo > 0.0)) {
        return r;
    }
    let to = offsetRay(c.x_s, n_s);
    let start = leaveSurface(from.x, from.ng, to - from.x);
    let dist = length(to - start);
    if (!(dist > 1.0e-4 && dist < 1.0e7)
        || !visibleInline(restirRayMask(s), start, (to - start) / dist, dist * 0.999)) {
        return r;
    }
    r.f = f;
    r.J = geo / c.geo;
    r.from_x = from.x;
    r.from_ng = from.ng;
    return r;
}

static const uint PT_MAX_SPATIAL = 16u;
// A shift is rejected when x_k's solid angle changes by more than this either way.
static const float PT_MAX_JACOBIAN = 8.0;

// Neighbour `i` of pixel `px` this frame: a random point of the disc, from a hash of the three
// (so the same pixels come back on a second walk, with nothing held between).
int2 ptNeighbourPixel(GiFrame* f, int2 px, uint i, uint seed) {
    let h0 = ptHash((uint(px.y) * f.width + uint(px.x)) * PT_MAX_SPATIAL + i + seed);
    let h1 = ptHash(h0);
    let a = float(h0 >> 8u) * (2.0 * PI / 16777216.0);
    let rad = f.spatial_radius * sqrt(float(h1 >> 8u) / 16777216.0);
    return px + int2(int(cos(a) * rad), int(sin(a) * rad));
}

// The neighbour's index when it can be reused from (inside, not this pixel, taking part this
// frame on a similar surface), else ~0.
uint ptUsableNeighbour(GiFrame* f, RestirSurface s, int2 px, uint i, uint seed) {
    let q = ptNeighbourPixel(f, px, i, seed);
    if (any(q < int2(0)) || q.x >= int(f.width) || q.y >= int(f.height) || all(q == px)) {
        return 0xFFFFFFFFu;
    }
    let qi = uint(q.y) * f.width + uint(q.x);
    let qs = restirSurfaceLoad(f.surfaces + qi);
    let cb = loadAligned<16>((uint4*)(f.candidates + qi) + 1);
    if (qs.frame != f.frame || cb.z != f.frame || !restirSimilar(s, qs.normal_oct, qs.depth)) {
        return 0xFFFFFFFFu;
    }
    return qi;
}

GiReservoir ptReservoir(float3 from_, float3 n_from, float3 to, float3 n_to, float3 C, float W,
                        uint frame, uint flags) {
    GiReservoir r;
    r.from_ = from_;
    r.C = packRadiance(C);
    r.to = to;
    r.W = W;
    r.frame = frame;
    r.flags = flags;
    r.n_from = packHalf2(float32x3_to_oct(n_from));
    r.n_to = packHalf2(float32x3_to_oct(n_to));
    return r;
}

// A sample as the reservoir a resampling pass writes back: W, confidence M, its target vector's
// luminance at this pixel and its geometry term from this pixel's x_{k-1}.
GiCandidate ptAsReservoir(GiCandidate c, float W, float M, float phat, float geo) {
    var r = c;
    r.pdf = W;
    r.q = M;
    r.phat = phat;
    r.geo = geo;
    return r;
}

// ReSTIR PT's temporal resampling (restir_gi.rs), one thread per traced pixel after gi_candidates:
// this pixel's own sample (confidence 1) against last frame's final reservoir at the reprojected
// pixel (confidence capped at history_cap), pairwise MIS with confidence weights: each sample's
// target in its own domain times its confidence against the other's target there times the shift's
// Jacobian. Last frame's sample moves here by the hybrid shift; ours moves to last frame's vertex for
// its weight. The result replaces this pixel's candidate in place (`pdf` = W, `q` = M, `phat` = its
// target here, `geo` from here) for the spatial pass, and its target vector goes to the final
// reservoir (the final answer when there is no spatial pass, which then also writes the history).
[shader("raygeneration")]
void gi_temporal() {
    let launch_id = DispatchRaysIndex().xy;
    let f = pc.gi_frame;
    if (!GI_ON || f == nullptr || launch_id.x >= f.width || launch_id.y >= f.height) {
        return;
    }
    let idx = launch_id.y * f.width + launch_id.x;
    let s = restirSurfaceLoad(f.surfaces + idx);
    let own = giCandidateLoad(f.candidates + idx);
    let spatial = f.spatial_samples > 0u;
    if (s.frame != f.frame || own.frame != f.frame) {
        giReservoirStore(f.final_ + idx, ptReservoir(float3(0.0), float3(0.0, 1.0, 0.0), float3(0.0),
                                                     float3(0.0, 1.0, 0.0), float3(0.0), 0.0, f.frame - 1u, 0u));
        if (!spatial) {
            ptHistoryStore(f.history + idx, ptAsReservoir(own, 0.0, 0.0, 0.0, 0.0), s, 0u, 0.0);
        }
        return;
    }
    initRandom(uint2(f.width, f.height), uint2(launch_id), f.frame * 7u + 3u);
    let thr = GI_RECONNECT_MIN * length(s.position - cameraOrigin());
    let own_W = own.pdf;
    var own_pdf = own;
    own_pdf.pdf = (own_W > 0.0 && own.q > 0.0) ? 1.0 / (own_W * own.q) : 0.0;
    let C_c = (own_W > 0.0 && own_W < 1.0e20) ? ptOwnTarget(s, own_pdf) : float3(0.0);
    let pc_c = luma(C_c);
    let own_valid = pc_c > 0.0;

    var r_sample = own;
    var r_C = own_valid ? C_c : float3(0.0);
    var r_phat = own_valid ? pc_c : 0.0;
    var r_geo = own.geo;
    var r_wsum = own_valid ? pc_c * own_W : 0.0;
    var M = 1.0;
    var fresh = true;

    var pp = restirReproject(&pc.uniforms.prev_view_proj[0], f.width, f.height, s.prev_position);
    // The further the history moved, the less it is trusted: under fast motion RR suffers from
    // reused (correlated) samples and neighbouring pixels start to share a history pixel.
    let moved = length(float2(pp - int2(launch_id)));
    let cap = f.history_cap / (1.0 + moved / 8.0);
    // Permutation sampling (DLSS-RR guide 3.5: RR assumes independent samples): the history is
    // read from the reprojected pixel XOR a per-frame mask inside 2x2 blocks, a permutation, so
    // each pixel continues a different pixel's chain every frame and no two share one.
    if (all(pp >= int2(0))) {
        let h = ptHash(f.frame * 0x2c1b3c6du + 0x297a2d39u);
        pp ^= int2(h & 1u, (h >> 1u) & 1u);
    }
    if (all(pp >= int2(0)) && pp.x < int(f.width) && pp.y < int(f.height)) {
        let hi = uint(pp.y) * f.width + uint(pp.x);
        uint hframe;
        float thr_p;
        let ps = ptHistorySurface(f.history_prev + hi, hframe, thr_p);
        let hs = ptHistorySample(f.history_prev + hi);
        let M_t = min(hs.q, cap);
        // The canonical weight counts last frame's pixel whenever it took part, whether or not its
        // sample came back empty: weighting the canonical by what the other sample happened to be
        // counted light twice (temporal alone read 6-18 % bright once the history came from
        // another pixel).
        if (hframe == f.frame - 1u && M_t > 0.0 && restirSimilar(s, ps.normal_oct, ps.depth)) {
            let hs_valid = hs.pdf > 0.0 && hs.pdf < 1.0e20 && hs.phat > 0.0;
            // Ours at last frame's vertex, in our measure.
            var m_c = 1.0;
            if (own_valid) {
                let at_p = ptShift(ps, thr_p, own);
                let ok = luma(at_p.f) > 0.0 && at_p.J > 1.0 / PT_MAX_JACOBIAN && at_p.J < PT_MAX_JACOBIAN;
                let den = pc_c + (ok ? M_t * luma(at_p.f) * at_p.J : 0.0);
                m_c = den > 0.0 ? pc_c / den : 1.0;
            }
            // Last frame's, here.
            var here = PtShift(float3(0.0), 0.0, s.position, float3(0.0, 1.0, 0.0));
            if (hs_valid) {
                here = ptShift(s, thr, hs);
            }
            let p_here = luma(here.f);
            var w_t = 0.0;
            if (p_here > 0.0 && here.J > 1.0 / PT_MAX_JACOBIAN && here.J < PT_MAX_JACOBIAN) {
                let den = M_t * hs.phat + p_here * here.J;
                let m_t = den > 0.0 ? M_t * hs.phat / den : 0.0;
                w_t = m_t * p_here * hs.pdf * here.J;
            }
            let w_c = own_valid ? m_c * pc_c * own_W : 0.0;
            r_wsum = w_c;
            r_sample = own;
            if (w_t > 0.0) {
                r_wsum += w_t;
                if (randf() * r_wsum < w_t) {
                    r_sample = hs;
                    r_C = here.f;
                    r_phat = p_here;
                    r_geo = here.J * hs.geo;
                    fresh = false;
                }
            }
            M = min(1.0 + M_t, f.history_cap + 1.0);
        }
    }
    let W = r_phat > 0.0 ? r_wsum / r_phat : 0.0;
    var res = ptAsReservoir(r_sample, W, M, r_phat, r_geo);
    // This frame's, whichever sample won (last frame's carries its own frame number).
    res.frame = f.frame;
    giCandidateStore(f.candidates + idx, res);
    giReservoirStore(f.final_ + idx, ptReservoir(s.position, float3(0.0, 1.0, 0.0), r_sample.x_s,
                                                 float3(0.0, 1.0, 0.0), r_C, W, f.frame, fresh ? GI_FRESH : 0u));
    if (!spatial) {
        ptHistoryStore(f.history + idx, res, s, f.frame, thr);
    }
}

// ReSTIR PT's spatial resampling (restir_gi.rs), one thread per traced pixel after gi_candidates:
// pairwise MIS (as restir.slang's restir_spatial) over random neighbours' samples, each moved
// here by the hybrid shift (random replay of its prefix from this pixel's vertex, reconnection
// to its x_k), every target compared in the canonical pixel's measure: a neighbour's sample here
// weighs its own target against ours times the shift's Jacobian, and the canonical sample at a
// neighbour likewise. The canonical target is the replayed one too, so every target here is the
// same function. Each candidate is a reservoir of one (M = 1). No visibility in the shift
// (gi_shade tests the winner's reconnection segment).
[shader("raygeneration")]
void gi_spatial() {
    let launch_id = DispatchRaysIndex().xy;
    let f = pc.gi_frame;
    if (!GI_ON || f == nullptr) {
        return;
    }
    let idx = launch_id.y * f.width + launch_id.x;
    if (launch_id.x >= f.width || launch_id.y >= f.height) {
        return;
    }
    let s = restirSurfaceLoad(f.surfaces + idx);
    let own = giCandidateLoad(f.candidates + idx);
    let temporal = f.history_cap > 0.0;
    if (s.frame != f.frame || own.frame != f.frame) {
        giReservoirStore(f.final_ + idx, ptReservoir(float3(0.0), float3(0.0, 1.0, 0.0), float3(0.0),
                                                     float3(0.0, 1.0, 0.0), float3(0.0), 0.0,
                                                     f.frame - 1u, 0u));
        if (temporal) {
            ptHistoryStore(f.history + idx, ptAsReservoir(own, 0.0, 0.0, 0.0, 0.0), s, 0u, 0.0);
        }
        return;
    }
    let px = int2(launch_id);
    initRandom(uint2(f.width, f.height), uint2(px), f.frame * 7u + 6u);
    let seed = ptHash(f.frame);
    let thr = GI_RECONNECT_MIN * length(s.position - cameraOrigin());
    // After gi_candidates: `pdf` holds W.
    let own_W = own.pdf;
    let n_c = oct_to_float32x3(unpackHalf2(own.n_s_oct));

    // The canonical reservoir's target here: after gi_temporal its vector is in the final
    // reservoir and its confidence in `q`; else this pixel's own sample (gi_candidates' `phat`:
    // light_eval.slang ptOwnTarget, its density recovered from W = 1 / (pdf q)).
    var C_c = float3(0.0);
    var M_c = 1.0;
    if (temporal) {
        C_c = unpackRadiance(giReservoirLoad(f.final_ + idx).C);
        M_c = max(own.q, 1.0);
    } else if (own_W > 0.0 && own_W < 1.0e20) {
        var own_pdf = own;
        own_pdf.pdf = own.q > 0.0 ? 1.0 / (own_W * own.q) : 0.0;
        C_c = ptOwnTarget(s, own_pdf);
    }
    if (!(own_W > 0.0 && own_W < 1.0e20)) {
        C_c = float3(0.0);
    }
    // Its scalar target: after gi_temporal the exact one W was divided by (C is RGB9E5-rounded).
    let pc_yc = temporal ? (luma(C_c) > 0.0 ? own.phat : 0.0) : luma(C_c);
    let own_valid = pc_yc > 0.0;

    let S = min(f.spatial_samples, PT_MAX_SPATIAL);
    var usable = 0u;
    for (uint i = 0u; i < S; i++) {
        if (ptUsableNeighbour(f, s, px, i, seed) != 0xFFFFFFFFu) {
            usable |= 1u << i;
        }
    }
    let m = countbits(usable);
    if (m == 0u) {
        giReservoirStore(f.final_ + idx, ptReservoir(s.position, n_c, own.x_s, n_c, C_c,
                                                     own_valid ? own_W : 0.0, f.frame, GI_FRESH));
        if (temporal) {
            ptHistoryStore(f.history + idx, ptAsReservoir(own, own_valid ? own_W : 0.0, M_c, pc_yc, own.geo), s,
                           f.frame, thr);
        }
        return;
    }

    var r_C = float3(0.0);
    var r_from = float3(0.0);
    var r_nfrom = float3(0.0, 1.0, 0.0);
    var r_to = float3(0.0);
    var r_nto = float3(0.0, 1.0, 0.0);
    var r_wsum = 0.0;
    var r_phat = 0.0;
    var mc_sum = 0.0;
    var neighbour_c = 0.0;
    var fresh = false;
    var M_sum = M_c;
    var r_sample = own;
    var r_geo = own.geo;
    for (var bits = usable; bits != 0u; bits &= bits - 1u) {
        let q = ptNeighbourPixel(f, px, firstbitlow(bits), seed);
        let qi = uint(q.y) * f.width + uint(q.x);
        let qs = restirSurfaceLoad(f.surfaces + qi);
        let nr = giCandidateLoad(f.candidates + qi);
        let nr_W = nr.pdf;
        let thr_q = GI_RECONNECT_MIN * length(qs.position - cameraOrigin());
        // Its confidence (after gi_temporal its reservoir's M; else 1).
        let M_i = temporal ? max(nr.q, 1.0) : 1.0;
        M_sum += (nr_W > 0.0 && nr_W < 1.0e20) ? M_i : 0.0;
        // The canonical sample at the neighbour, in our measure.
        if (own_valid) {
            let at_q = ptShift(qs, thr_q, own);
            let pi_c = luma(at_q.f);
            let J_q = at_q.J;
            let ok = pi_c > 0.0 && J_q > 1.0 / PT_MAX_JACOBIAN && J_q < PT_MAX_JACOBIAN;
            let den = (ok ? M_i * pi_c * J_q : 0.0) + M_c * pc_yc;
            mc_sum += den > 0.0 ? M_c * pc_yc / den : 1.0;
        }
        // The neighbour's sample, shifted here.
        if (!(nr_W > 0.0 && nr_W < 1.0e20) || !(nr.phat > 0.0)) {
            continue;
        }
        let here = ptShift(s, thr, nr);
        let C_i = here.f;
        let J = here.J;
        let pc_yi = luma(C_i);
        if (!(pc_yi > 0.0) || !(J > 1.0 / PT_MAX_JACOBIAN && J < PT_MAX_JACOBIAN)) {
            continue;
        }
        // Its target at its own pixel (gi_candidates).
        let pi_i = nr.phat;
        let den = M_i * pi_i + M_c * pc_yi * J;
        let ai = den > 0.0 ? M_i * pi_i / den : 0.0;
        let w = (ai / float(m)) * pc_yi * nr_W * J;
        neighbour_c += pc_yi * nr_W * J;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r_C = C_i;
            r_from = here.from_x;
            r_nfrom = here.from_ng;
            r_to = nr.x_s;
            r_nto = oct_to_float32x3(unpackHalf2(nr.n_s_oct));
            r_phat = pc_yi;
            fresh = false;
            r_sample = nr;
            r_geo = J * nr.geo;
        }
    }
    if (own_valid) {
        let w = (mc_sum / float(m)) * pc_yc * own_W;
        r_wsum += w;
        if (randf() * r_wsum < w) {
            r_C = C_c;
            // Fresh: gi_shade tests no segment.
            r_from = s.position;
            r_nfrom = n_c;
            r_to = own.x_s;
            r_nto = n_c;
            r_phat = pc_yc;
            fresh = true;
            r_sample = own;
            r_geo = own.geo;
        }
    }

    // Boiling filter (restir_spatial's): a winner far hotter than its neighbourhood is dropped.
    if (f.boiling > 0.0 && m >= 3u && own_valid && fresh) {
        let mean = neighbour_c / float(m);
        if (pc_yc * own_W > mean / f.boiling) {
            r_phat = 0.0;
        }
    }

    let W = r_phat > 0.0 ? r_wsum / r_phat : 0.0;
    giReservoirStore(f.final_ + idx, ptReservoir(r_from, r_nfrom, r_to, r_nto, r_C, W, f.frame,
                                                 (W > 0.0 && fresh) ? GI_FRESH : 0u));
    if (temporal) {
        ptHistoryStore(f.history + idx, ptAsReservoir(r_sample, W, min(M_sum, f.history_cap), r_phat, r_geo),
                       s, f.frame, thr);
    }
}

// The reference accumulation (reference.rs, docs/plans/reference-mode.html), one thread per traced
// pixel after the trace and gi_shade: this frame's noisy colour, the exposure divided back out (so
// nits), added to the running sum. A reset clears the buffer before this runs.
[shader("raygeneration")]
void accumulate() {
    if (pc.reference == nullptr) {
        return;
    }
    let launch_id = DispatchRaysIndex().xy;
    let idx = launch_id.y * DispatchRaysDimensions().x + launch_id.x;
    let c = dlss_color[int2(launch_id)].rgb / max(pc.ae.input_exposure, 1.0e-20);
    let r = pc.reference + idx;
    let sum = loadAligned<16>(r) + float4(c, 1.0);
    storeAligned<16>(r, sum);
    // Feeding: Ray Reconstruction takes the converged mean instead of this frame's sample (the
    // test of what it does with a clean input and our guides).
    if (pc.uniforms.reference_feed != 0u) {
        let px = int2(launch_id);
        dlss_color[px] = float4(min(sum.rgb / sum.a * pc.ae.input_exposure, float3(65504.0)), 1.0);
    }
}

uint packUnorm16(float x) {
    return uint(saturate(x) * 65535.0 + 0.5);
}

// ReSTIR DI's final sample for this pixel at its primary vertex: the exact contribution (layered
// BSDF, MIS-weighted against BRDF sampling, as restirTarget) times the reservoir weight, its
// direction and distance; it joins the vertex's one shadow ray.
float3 restirWinnerUnshadowed(uint idx, float3 origin, float3 n, float3 ns, float3 v,
                              SurfaceMaterial mat, Lobes lob, uint ray_mask, out float3 dir,
                              out float dist) {
    dir = float3(0.0, 1.0, 0.0);
    dist = 0.0;
    if (pc.reservoirs_cur == nullptr) {
        return float3(0.0);
    }
    let r = pc.reservoirs_cur[idx];
    if (r.y == 0xFFFFFFFFu || r.W <= 0.0) {
        return float3(0.0);
    }
    let ctx = lightCtx();
    let uv = unpackHalf2(r.uv_pk);
    let p_cdf = lightsEntryPick(ctx, r.y);
    float p_area, pdf_solid;
    var val = lightSampleEval(ctx, r.y, lightsFindInstance(ctx, r.y), p_cdf, uv.x, uv.y, origin, n, ns,
                              v, mat, lob, ray_mask, true, p_area, pdf_solid, dir, dist);
    if (pdf_solid > 0.0) {
        val *= pdf_solid / (pdf_solid + surfacePdf(v, n, dir, mat, lob));
    }
    return val * r.W;
}

// sphere.slang's hit attributes: the object-space point.
struct PrimarySphereAttributes {
    float3 point;
};

// `PrimaryHit::hit_kind` of a recorded analytic sphere (triangles keep their own kinds).
static const uint PRIMARY_SPHERE = 0x100u;

// The recorded hit rebuilt (no traversal) and shaded into `payload`.
void resumeHit(PrimaryHit rec, inout HitPayload payload) {
    RayDesc ray;
    ray.Origin = rec.origin;
    ray.Direction = rec.direction;
    ray.TMin = 0.0;
    ray.TMax = rec.t;
    if (rec.hit_kind == PRIMARY_SPHERE) {
        PrimarySphereAttributes sa;
        let o = float2(float(rec.bary & 0xFFFFu), float(rec.bary >> 16)) / 65535.0 * 2.0 - 1.0;
        sa.point = 0.5 * oct_to_float32x3(o);
        HitObject sphere = HitObject::MakeHit(topLevelAS, rec.instance, rec.geometry, rec.primitive,
                                              0u, 0, 0, ray, sa);
        HitObject::Invoke(topLevelAS, sphere, payload);
        return;
    }
    BuiltInTriangleIntersectionAttributes a;
    a.barycentrics = float2(float(rec.bary & 0xFFFFu), float(rec.bary >> 16)) / 65535.0;
    // No reorder: the records come in screen order, as coherent as the camera rays that made
    // them (sorting them cost the trace 8.6 %).
    HitObject hit = HitObject::MakeHit(topLevelAS, rec.instance, rec.geometry, rec.primitive,
                                       rec.hit_kind, 0, 0, ray, a);
    HitObject::Invoke(topLevelAS, hit, payload);
}


// The camera's position: the inverse view's translation. The camera values are recomputed
// where they are used rather than held: anything held is live through the whole path.
float3 cameraOrigin() {
    return float3(pc.uniforms.inverse_view[12], pc.uniforms.inverse_view[13], pc.uniforms.inverse_view[14]);
}

// The camera ray's direction through the launch's pixel. A known sub-pixel jitter (told to NGX);
// the launch y is flipped into the NDC-up space here, so the image-space offset is +jitter. Row 0
// is the TOP row: its centre sits half a pixel below the top edge, (height - 1 - y) + 0.5 in the
// up-space. (An earlier `height - y + 0.5` put every sample one whole pixel above its pixel: the
// image was shifted a row against the depth/motion guides and the matrices Ray Reconstruction
// reprojects with, and edges crawled.)
float3 pixelDirection(uint2 launch_id, uint2 launch_size) {
    let subpixel = float2(0.5 + pc.uniforms.jitter_x, 0.5 - pc.uniforms.jitter_y);
    let pixel_center = float2(float(launch_id.x), float(launch_size.y - 1u - launch_id.y)) + subpixel;
    let d = pixel_center / float2(launch_size) * 2.0 - 1.0;
    let target = mat4MulVecAt(&pc.uniforms.inverse_projection[0], float4(d, 1, 1)).xyz;
    return mat4MulVecAt(&pc.uniforms.inverse_view[0], float4(normalize(target), 0)).xyz;
}

// A sample's camera ray: from `lens` on the aperture toward the pixel's point in focus.
void lensRay(float2 lens, uint2 launch_id, uint2 launch_size, out float3 origin, out float3 direction) {
    let camera = cameraOrigin();
    let focal = camera + pixelDirection(launch_id, launch_size) * pc.focus[0];
    origin = mat4MulVecAt(&pc.uniforms.inverse_view[0], float4(lens, 0, 1)).xyz;
    direction = normalize(focal - origin);
}

// The camera path's flags (main's `path`).
static const uint PATH_WANT_SUN = 1u;
static const uint PATH_CAMERA = 2u;
static const uint PATH_CACHE_BLIND = 32u;
// ReSTIR PT: the primary vertex takes part (its sample is written at the path's end); the path
// is still looking for its reconnection vertex (BSDF numbers from the path seed, no roulette);
// it found one (the snapshot is live); the vertex the path just left is rough.
static const uint PATH_GI_VERTEX = 64u;
static const uint PATH_PT_SEARCH = 128u;
static const uint PATH_GI_LIVE = 256u;
static const uint PATH_PT_PREV_ROUGH = 512u;
// Neural radiance cache: the path is traced on to teach the cache (records every vertex, never
// ends in it).
static const uint PATH_NRC_TRAIN = 1024u;
// ReSTIR PT: the camera ray has crossed only mirrors and glass so far (bounces drawn with pdf 0):
// the first opaque vertex it reaches is the pixel's primary vertex x_0 for the reuse.
static const uint PATH_PT_CHAIN = 2048u;
// Vertices a training path records.
static const uint NRC_VERTICES = 4u;

[shader("raygeneration")]
void main() {
    let launch_id = DispatchRaysIndex().xy;
    let launch_size = DispatchRaysDimensions().xy;
    // The Cost view's start: the device clock (ns). The subgroup clock is per SM, and a thread
    // SER moves can end on another SM: a third of the deltas came back negative.
    let clock_start = getRealtimeClock().x;
    // The camera path's payload, fresh before each of its traces (the bounce loop).
    HitPayload payload;

    // Fresh noise every frame for the temporal denoiser.
    initRandom(launch_size, launch_id, pc.uniforms.frame);

    let px = int2(launch_id);

    // Rays leaving a surface start off it (`offsetRay`), so nothing needs a minimum distance.
    let tmin = 0.0;
    // Far range: the sim-scale sky domes sit kilometers out, a planet's horizon tens of km.
    let tmax = 1.0e6;

    // The primary pass (docs/plans/primary-pass.html) walked sample 0's camera chain: its hit
    // record, the guides, the camera ray's fog, the meter's divisor and the focus pull. The
    // record is read where needed, never held: 16 words live across every trace would spill.
    let prim_at = launch_id.y * launch_size.x + launch_id.x;

    var acc = float3(0.0);

    for (uint sample_idx = 0; sample_idx < SAMPLES; sample_idx += 1) {
        let offsetR = sqrt(randf());
        let offsetA = randf() * 2.0 * PI;
        let lens = pc.uniforms.aperture * float2(offsetR * cos(offsetA), offsetR * sin(offsetA));

        float3 origin, direction;
        lensRay(lens, launch_id, launch_size, origin, direction);
        var mask = float3(1.0);
        // The path's flags, one word (PATH_*, each explained where it would have been declared).
        var path = PATH_WANT_SUN | PATH_CAMERA | PATH_PT_CHAIN;
        // Scatters so far by kind (bounceKind's packed word).
        var bounces = 0u;
        // PATH_WANT_SUN: the camera ray and pure-specular continuations may see the sun disc directly; after a
        // BRDF-sampled bounce the sun comes from next-event estimation only.
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
        // PATH_CAMERA: still on the camera ray (through cutouts and portals, before any scatter): it skips
        // the camera's primary-hidden layers, which every other ray sees.
        // Neural radiance cache: the path's footprint (the sum of square roots of its spread so far
        // and the primary vertex's spread, Müller et al. 2021 sec. 5) and, on a training path, each
        // recorded vertex's ring slot with the mask and radiance it was reached with.
        var nrc_spread = 0.0;
        var nrc_a0 = 0.0;
        uint nrc_slot[NRC_VERTICES];
        float3 nrc_mask[NRC_VERTICES];
        float3 nrc_acc[NRC_VERTICES];
        var nrc_n = 0u;
        // A training path's pixel takes what it would have rendered had it ended in the cache
        // (its extension only teaches the cache: traced on, it is the old heavy-tailed estimate,
        // fireflies Ray Reconstruction rejected among the cached pixels).
        var nrc_render = float3(0.0);
        var nrc_ended = false;
        if (pc.nrc != nullptr && sample_idx == 0u && randf() < pc.nrc.train_fraction) {
            path |= PATH_NRC_TRAIN;
        }
        // ReSTIR PT's snapshots: the path weight reaching the reconnection vertex (the sample's
        // radiance is the rest of the path over it) and the radiance gathered before it; while
        // searching, the seed the BSDF numbers come from, the vertex just left, the reconnection
        // distance floor and the primary vertex's weight.
        var gi_mask = float3(1.0);
        var gi_acc = float3(0.0);
        var pt_seed = 0u;
        var pt_prev_pos = float3(0.0);
        var pt_thr = 0.0;
        var pt_v0_mask = float3(1.0);
        // The bounce the primary vertex was reached at (the primary pass may have used some):
        // vertex indices count from it.
        var pt_base = 0u;
        // At the reconnection vertex: its bounce, the radiance gathered once its next-event
        // sample was in, that sample's contribution and direction, the continuation's direction.
        var pt_k_bounce = 0xFFFFFFFFu;
        var pt_acc_mid = float3(0.0);
        var pt_nee = float3(0.0);
        var pt_nee_dir = float3(0.0, 1.0, 0.0);
        var pt_cont_dir = float3(0.0, 1.0, 0.0);
        var pt_nk = 0u;
        // The chance the search's prefix survived Russian roulette: the sample's density includes it,
        // its weight T does not (replay rolls no dice).
        var pt_q = 1.0;
        // PATH_CACHE_BLIND: the camera ray met the back of a solid surface first: the camera is inside geometry
        // (a wall it clipped through). Such a path sees unlit insides and must not teach the
        // neural radiance cache.
        // Ray cone for texture level of detail (surface.slang): width at the ray's origin
        // and spread angle. The camera ray opens by one pixel; a scattered bounce opens it by the
        // lobe it was drawn from, so indirect hits read blurrier levels (less noise too).
        var cone_width = 0.0;
        // inverse_projection[1][1] (column 1, row 1).
        var cone_spread = 2.0 * pc.uniforms.inverse_projection[5] / float(launch_size.y);

        // Sample 0 starts where the primary pass's chain stopped: the recorded hit, the radiance
        // it ended on, or the camera again.
        var first_bounce = 0u;
        var resume = false;
        {
            if (sample_idx == 0u) {
                let prim = pc.primary_hits[prim_at];
                let kind = primaryKind(prim);
                if (kind == PRIMARY_ENDED) {
                    acc += prim.origin;
                    first_bounce = MAX_BOUNCES;
                } else if (kind == PRIMARY_HIT) {
                    origin = prim.origin;
                    direction = prim.direction;
                    ray_mask = (prim.state >> 8) & 0xFFu;
                    ray_env = (prim.state >> 16) & 7u;
                    cone_width = prim.cone_width;
                    first_bounce = (prim.state >> 2) & 63u;
                    resume = true;
                }
            }
        }

        for (uint bounce_idx = first_bounce; bounce_idx < MAX_BOUNCES; bounce_idx += 1) {
            // A fresh payload every trace: a miss writes only t and the sky, and every hit
            // read-modify-writes the flags word, so a payload carried over was an input to the
            // next trace -- all of it live in registers through the whole loop.
            payload = {};
            // Low bit: may this miss return the sun discs; above it the ray's environment (the
            // miss returns that environment's sky).
            payload.miss_flags = (((path & PATH_WANT_SUN) != 0u) ? 1u : 0u) | (ray_env << 1);
            payload.cone = float2(cone_width, cone_spread);
            let trace_mask = ((path & PATH_CAMERA) != 0u) ? (ray_mask & ~pc.uniforms.primary_hidden) : ray_mask;
            let start = any(surf_n != float3(0.0)) ? leaveSurface(origin, surf_n, direction) : origin;
            // The recorded hit is rebuilt, not traced: the primary pass already found it.
            let resumed = resume;
            if (resume) {
                resume = false;
                resumeHit(pc.primary_hits[prim_at], payload);
            } else {
                traceReordered(FLAGS_OPAQUE, trace_mask, start, tmin, direction, tmax, payload);
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
            if (!resumed && payload.t != 0.0 && pc.uniforms.portal_count > 0u) {
                var p_origin = origin;
                var p_dir = direction;
                if (portalRedirect(payload.slot, origin + direction * payload.t, p_origin, p_dir, ray_mask, ray_env)) {
                    origin = p_origin;
                    surf_n = float3(0.0);
                    direction = p_dir;
                    prev_pdf = 0.0;
                    path &= ~PATH_PT_SEARCH;
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
            if (!resumed && payload.t != 0.0 && all(payload.emission == float3(0.0))
                && hitPayloadGetMasked(payload)) {
                if (all(payload.color.rgb == float3(0.0))) {
                    pass_through = true;
                } else if (payload.color.a < 1.0) {
                    pass_through = randf() < 1.0 - payload.color.a;
                }
            }

            let roughness = hitPayloadGetRoughness(payload);
            let metallic = hitPayloadGetMetallic(payload);
            let transmission = hitPayloadGetTransmission(payload);
            let inside = hitPayloadGetInside(payload);
            let lob = unpackLobes(payload.lobes);
            if (((path & PATH_CAMERA) != 0u) && inside && transmission < 0.01 && !hitPayloadGetMasked(payload)) {
                path |= PATH_CACHE_BLIND;
            }

            // fog bounce (skybox has a distance of 0 so will never hit
            // this case, having an extra check for this is slower than
            // deferring to the generic emitter check)
            float fogDistance;
            let env = pc.uniforms.environments + ray_env;
            let fog_any = env.fog_density > 0.0 || pc.uniforms.fog_volume_count > 0u;
            // A camera ray in marched fog: the primary pass took its veil. Without marching steps
            // the camera ray scatters at random like every other ray.
            let camera_fog = ((path & PATH_CAMERA) != 0u) && env.fog_steps > 0u && fog_any;
            if (!camera_fog && sampleFogScatter(env, origin, direction, payload.t, fogDistance)) {
                origin = origin + direction * fogDistance;
                surf_n = float3(0.0);
                // The distance was drawn from the fog's own density (exponential, or delta
                // tracked), so the collision's weight is the medium's single-scattering albedo.
                mask *= FOG_ALBEDO;
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
                path &= ~(PATH_WANT_SUN | PATH_PT_SEARCH | PATH_PT_CHAIN);
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
                acc += contribution;
                break;
            }

            // Alpha-cutout pass-through: carry on along the same ray from the hit.
            origin = origin + direction * payload.t;
            surf_n = surface_normal;
            if (pass_through) {
                // Replay cannot draw the alpha coin again.
                path &= ~PATH_PT_SEARCH;
                continue;
            }
            // The footprint: the camera ray's at the primary vertex (d^2 / (4 pi cos)), then each
            // scattered ray's sqrt(d^2 / (pdf cos)) (a delta bounce spreads nothing).
            if (pc.nrc != nullptr) {
                let cos_in = max(abs(dot(surface_normal, direction)), 1.0e-3);
                let t2 = payload.t * payload.t;
                if ((path & PATH_CAMERA) != 0u) {
                    nrc_a0 = t2 / (4.0 * PI * cos_in);
                } else if (prev_pdf > 0.0) {
                    nrc_spread += sqrt(t2 / (prev_pdf * cos_in));
                }
            }
            path &= ~PATH_CAMERA;

            // ReSTIR PT. While the path searches, each vertex it reaches is tested as the
            // reconnection vertex x_k: it and the vertex before are rough and farther apart than
            // pt_thr. The first that passes takes x_k's position, normal, the direction's pdf,
            // the seed, k, the path weight over the primary vertex's and the geometry term now;
            // its radiance at the path's end. A vertex replay could not rebuild (glass, thin
            // layers, an alpha-tested surface) or the depth limit ends the search: that path's
            // light stays here.
            if (GI_ON && (path & PATH_PT_SEARCH) != 0u) {
                let rough_here = ptRough(roughness, transmission, lob) && !hitPayloadGetMasked(payload);
                let d = length(origin - pt_prev_pos);
                if ((path & PATH_PT_PREV_ROUGH) != 0u && rough_here && d > pt_thr) {
                    path = (path & ~PATH_PT_SEARCH) | PATH_GI_LIVE;
                    gi_acc = acc;
                    gi_mask = mask;
                    let ns = dot(surface_normal, direction) > 0.0 ? -surface_normal : surface_normal;
                    let T = pt_q * float3(pt_v0_mask.r > 1.0e-6 ? mask.r / pt_v0_mask.r : 0.0,
                                          pt_v0_mask.g > 1.0e-6 ? mask.g / pt_v0_mask.g : 0.0,
                                          pt_v0_mask.b > 1.0e-6 ? mask.b / pt_v0_mask.b : 0.0);
                    let cand = (uint4*)(pc.gi_candidates + prim_at);
                    storeAligned<16>(cand, uint4(asuint(origin), packHalf2(float32x3_to_oct(ns))));
                    storeAligned<16>(cand + 2, uint4(pt_seed, bounce_idx - pt_base, packRgb9e5(T),
                                                     asuint(abs(dot(ns, direction)) / max(d * d, 1.0e-8))));
                    pc.gi_candidates[prim_at].pdf = prev_pdf;
                    // x_k's material and view: its BSDF is evaluated again for shifted paths.
                    pt_k_bounce = bounce_idx;
                    pt_acc_mid = acc;
                    pt_nk = packHalf2(float32x3_to_oct(world_normal));
                    storeAligned<16>(cand + 4, uint4(ptPackAlbedoMetallic(payload.color.rgb, metallic),
                                                     packHalf2(float2(roughness, lob.transmission)),
                                                     packHalf2(float32x3_to_oct(-direction)), asuint(pt_q)));
                } else if (bounce_idx - pt_base >= PT_MAX_K || hitPayloadGetMasked(payload)) {
                    path &= ~PATH_PT_SEARCH;
                } else {
                    path = rough_here ? (path | PATH_PT_PREV_ROUGH) : (path & ~PATH_PT_PREV_ROUGH);
                    pt_prev_pos = origin;
                }
            }
            // The pixel's primary vertex x_0 takes part unless replay could not start from it: the
            // camera ray's first hit, or the first opaque surface behind the mirrors and glass it
            // crossed (its surface record is written here then, the chain's weight its throughput:
            // gi_shade scales the reuse by it).
            if (GI_ON && sample_idx == 0u && pc.gi_candidates != nullptr
                && (path & (PATH_PT_CHAIN | PATH_GI_VERTEX)) == PATH_PT_CHAIN
                && ptReplayable(transmission, lob) && !hitPayloadGetMasked(payload)) {
                if (!resumed && pc.restir_surfaces != nullptr) {
                    var rs = restirSurfacePack(
                        origin, pc.uniforms.frame, payload.prev_world_pos,
                        mat4MulVecAt(&pc.uniforms.view_proj[0], float4(origin, 1.0)).w,
                        payload.color.rgb, metallic, roughness, payload.lobes, mask, ray_mask,
                        world_normal, surface_normal, -direction);
                    // The chain's bounces, for replay's budgets.
                    rs.throughput_b_mask |= packBounces(bounces) << 24;
                    restirSurfaceStore(pc.restir_surfaces + prim_at, rs);
                }
                path = (path | PATH_GI_VERTEX | PATH_PT_SEARCH) & ~PATH_PT_CHAIN;
                path = ptRough(roughness, transmission, lob) ? (path | PATH_PT_PREV_ROUGH)
                                                             : (path & ~PATH_PT_PREV_ROUGH);
                pt_v0_mask = mask;
                pt_base = bounce_idx;
                pt_prev_pos = origin;
                pt_thr = GI_RECONNECT_MIN * length(origin - cameraOrigin());
                pt_seed = ptHash(prim_at * 0x9e3779b9u ^ ptHash(pc.uniforms.frame + 0x632be5abu));
            }

            // Inside glass the medium absorbs along the segment that reached this vertex (the
            // interface itself is a lobe of the surface BSDF, sampled below).
            if (inside && transmission > 0.0) {
                mask *= exp(-payload.t * payload.absorption);
            }

            // Neural radiance cache (src/nrc.rs). A path whose footprint has spread past the
            // threshold ends in the network's answer, before this vertex's next-event estimate
            // (the cached radiance is all the vertex scatters). Never at or before ReSTIR PT's
            // reconnection vertex: the cache's light then reaches x_k through its continuation,
            // which shifted paths carry as is. A training path records every vertex instead.
            if (pc.nrc != nullptr && transmission < 0.01 && !hitPayloadGetMasked(payload)
                && (path & PATH_CACHE_BLIND) == 0u) {
                NrcInput q;
                q.position = nrcBoxPosition(pc.nrc, origin);
                q.dir = -direction;
                q.normal = world_normal;
                q.roughness = roughness;
                q.diffuse = payload.color.rgb * (1.0 - metallic);
                q.specular = lerp(float3(0.04), payload.color.rgb, metallic);
                // The path's last vertex ends in the cache too, training or not: what lies past
                // the bounce budgets comes back (self-training, Müller et al. 2021 sec. 5: training
                // targets then carry every bounce).
                // A path still searching at its last vertex found no x_k (its light stays in the
                // pixel), so the cache may end it too: glass in front had used up its bounces.
                let last = (bounces & 0xFFu) >= MAX_DIFFUSE_BOUNCES || bounce_idx + 1u >= MAX_BOUNCES;
                let pt_past = ((path & PATH_PT_SEARCH) == 0u || last)
                    && (!GI_ON || pt_k_bounce == 0xFFFFFFFFu || bounce_idx > pt_k_bounce);
                let spread_out = nrc_spread * nrc_spread > pc.nrc.spread * nrc_a0;
                if (nrcInBox(q.position) && (pc.nrc.flags & NRC_FLAG_INLINE) != 0u && pt_past
                    && (last || spread_out) && !nrc_ended) {
                    let cached = acc + mask * nrcQuery(pc.nrc, q) / max(pc.ae.input_exposure, 1.0e-20);
                    if ((path & PATH_NRC_TRAIN) == 0u || last) {
                        acc = cached;
                        break;
                    }
                    nrc_render = cached;
                    nrc_ended = true;
                } else if (last && nrc_ended && nrcInBox(q.position)) {
                    // The extension's own end: the cache's answer finishes its targets.
                    acc += mask * nrcQuery(pc.nrc, q) / max(pc.ae.input_exposure, 1.0e-20);
                    break;
                }
                if (nrcInBox(q.position)) {
                    if ((path & PATH_NRC_TRAIN) != 0u) {
                        if (nrc_n < NRC_VERTICES) {
                            uint slot;
                            InterlockedAdd(pc.nrc.counter[0], 1u, slot);
                            if (slot < pc.nrc.ring) {
                                pc.nrc.records[slot] = nrcPackRecord(q);
                                nrc_slot[nrc_n] = slot;
                                nrc_mask[nrc_n] = mask;
                                nrc_acc[nrc_n] = acc;
                                nrc_n++;
                            }
                        }
                    }
                }
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
                if ((path & PATH_PT_SEARCH) != 0u) {
                    pt_q *= pRussian;
                }
            }

            // Glass's eta = etaI / etaT for the side the ray came from.
            let ior = max(payload.refract_index, 1.0e-3);
            let surf_mat = surfaceMaterial(payload.color.rgb, metallic, roughness, transmission,
                                           inside ? ior : 1.0 / ior);

            // Direct light from the emissive-triangle table. At the primary vertex of sample 0,
            // with ReSTIR DI on, its passes do it (light grid, initial + temporal and spatial
            // reuse, one shadow ray: src/restir.rs); everywhere else it is RIS over CDF
            // candidates with an MIS-weighted shadow ray.
            // Positions and areas are read live, so stale CDF weights change variance, never
            // the answer.
            // The strategies below share one shadow ray: each proposes its unshadowed
            // contribution, one is picked by brightness and scaled by 1 / its pick probability.
            ShadowPick pick = {};
            if (pc.uniforms.light_entries > 0u && pc.lights.total_power > 0.0) {
                if (RESTIR_ON && resumed) {
                    // ReSTIR DI's winner for this vertex: its passes ran between the primary pass
                    // and this one, over the surface the primary pass wrote.
                    float3 wdir;
                    float wdist;
                    let c = restirWinnerUnshadowed(prim_at, origin, world_normal, surface_normal,
                                                   -direction, surf_mat, lob, ray_mask, wdir, wdist);
                    pick.propose(mask * c, wdir, wdist * (1.0 - 1.0e-4));
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
                    let M = bounce_idx == 0u ? LIGHT_CANDIDATES : max(1u, LIGHT_CANDIDATES / 4u);
                    let grid = pc.restir_frame;
                    let cell = grid != nullptr ? gridCell(grid, origin) : 0xFFFFFFFFu;
                    var wsum = 0.0;
                    // The kept candidate's value over its target (one value fewer live through
                    // the loop, the raygen's register peak).
                    var sel_scaled = float3(0.0);
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
                                                  surf_mat, lob, ray_mask, false, p_area, pdf_solid, cdir,
                                                  cdist);
                        if (luma(val) <= 0.0 || p_area <= 0.0) {
                            continue;
                        }
                        let w = pdf_solid / (pdf_solid + surfacePdf(-direction, world_normal, cdir, surf_mat, lob));
                        let g = val * w;
                        let phat = luma(g);
                        let wi = phat / (p_area * src_ratio);
                        wsum += wi;
                        if (randf() * wsum < wi) {
                            sel_scaled = g / phat;
                            ldir = cdir;
                            ldist = cdist;
                        }
                    }
                    if (wsum > 0.0) {
                        pick.propose(mask * sel_scaled * (wsum / float(M)), ldir, ldist * (1.0 - 1.0e-4));
                    }
                }
            }

            // Next-event estimation for an HDR environment: one importance-sampled direction
            // (luminance x sin theta CDF, sky_gen.rs), MIS-weighted against BRDF sampling; the
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
                    let f = surfaceEval(-direction, world_normal, l, surf_mat, lob);
                    if (any(f > float3(0.0))) {
                        let w = pdf_env / (pdf_env + surfacePdf(-direction, world_normal, l, surf_mat, lob));
                        let c = mask * f * L * (w / pdf_env);
                        pick.propose(c, l, tmax);
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
                    let f = surfaceEval(-direction, world_normal, l, surf_mat, lob);
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
                        let c = mask * f * sun_radiance * solid_angle;
                        pick.propose(c, l, tmax);
                    }
                }
            }
            let nee = pick.resolve(ray_mask, origin, surface_normal);
            acc += nee;
            if (GI_ON && bounce_idx == pt_k_bounce) {
                pt_nee = nee;
                pt_nee_dir = pick.direction;
                pt_acc_mid = acc;
            }

            // One lobe, picked by albedo, and its direction (bsdf.slang: Falcor's standard BSDF
            // under the clear coat). While ReSTIR PT searches, the numbers come from the path
            // seed, which replay draws again (ptReplay); otherwise white noise from the PCG stream:
            // Ray Reconstruction assumes independent samples (DLSS-RR Integration Guide 3.5).
            let bs = surfaceBSDF(-direction, world_normal, surf_mat, lob);
            float3 l;
            float bpdf;
            float3 weight;
            uint lobe;
            bool sampled;
            if (GI_ON && (path & PATH_PT_SEARCH) != 0u) {
                var sg = PathSampler(pt_seed, bounce_idx - pt_base);
                sampled = bs.sample(sg, l, bpdf, weight, lobe);
            } else {
                var sg = PcgSampler();
                sampled = bs.sample(sg, l, bpdf, weight, lobe);
            }
            // A failed sample carries nothing: the path ends. Nor may it leak through the surface,
            // unless it crossed the shading normal (the translucent lobe, refraction), and then it
            // must cross the face as well.
            let through = dot(l, world_normal) < 0.0;
            if (!sampled || (through ? dot(l, surface_normal) >= 0.0 : dot(l, surface_normal) < 0.0)) {
                break;
            }
            direction = l;
            if (GI_ON && bounce_idx == pt_k_bounce) {
                pt_cont_dir = direction;
            }
            // A mirror bounce or glass still sees the sun disc: neither is part of next-event
            // estimation (bpdf 0), so the disc is how the sun reaches them; any other bounce also
            // ends the camera's mirror-and-glass chain.
            if (bpdf > 0.0) {
                path &= ~(PATH_WANT_SUN | PATH_PT_CHAIN);
            }
            // The opaque lobes' pdf (0 for a delta or glass): the MIS partner of light sampling.
            prev_pdf = bpdf;
            mask *= weight;
            // The cone opens with the lobe: a mirror keeps it, a diffuse bounce spreads wide.
            if (through) {
                cone_spread = max(cone_spread, 0.37);
            } else if ((lobe & LOBE_MASK) == LOBE_COAT) {
                cone_spread = max(cone_spread, 0.02 + 0.35 * lob.coat_alpha);
            } else {
                cone_spread = max(cone_spread, 0.02 + 0.35 * roughness * roughness);
            }
            // The bounce budgets (Falcor's): past its kind's, the path ends here.
            if (!bounceWithinBudget(bounces, bounceKind(lobe, through, roughness, lob.coat_alpha))) {
                break;
            }
            // Russian roulette: a dim path ends here with probability 1 - q, a survivor carries
            // 1/q more (unbiased). Paths otherwise ran to their bounce budgets however little they
            // carried; the indirect bounces were most of the trace.
            // Not off ReSTIR PT's primary vertex: the bounce reuse shares (killed there, 60 % of
            // Northshire's paths never reached a reconnection vertex).
            let pt_first = GI_ON && (path & PATH_PT_SEARCH) != 0u && bounce_idx == pt_base;
            if (ROULETTE_FROM > 0u && bounce_idx + 1u >= ROULETTE_FROM && !pt_first) {
                let q = clamp(max(mask.r, max(mask.g, mask.b)), 0.25, 1.0);
                if (randf() >= q) {
                    break;
                }
                mask /= q;
                if ((path & PATH_PT_SEARCH) != 0u) {
                    pt_q *= q;
                }
            }
        }

        // A training path's records: each vertex's target is what the path gathered from it on,
        // over the mask that reached it, pre-exposed like the network's answers.
        for (uint ni = 0u; ni < nrc_n; ni++) {
            let delta = max(acc - nrc_acc[ni], float3(0.0));
            let m = nrc_mask[ni];
            let target = float3(m.r > 1.0e-6 ? delta.r / m.r : 0.0,
                                m.g > 1.0e-6 ? delta.g / m.g : 0.0,
                                m.b > 1.0e-6 ? delta.b / m.b : 0.0);
            pc.nrc.records[nrc_slot[ni]].target = packRadiance(target * pc.ae.input_exposure);
        }
        if (nrc_ended) {
            acc = nrc_render;
        }
        // ReSTIR PT's sample: what the path gathered from the reconnection vertex on, over the
        // weight that reached it, leaves the path for the reuse passes (null: none was found, the
        // light stays here).
        if (GI_ON && (path & PATH_GI_VERTEX) != 0u) {
            var Lc = float3(0.0);
            var Ln = float3(0.0);
            if ((path & PATH_GI_LIVE) != 0u) {
                // Each part over the weight that reached x_k: what x_k sent toward x_{k-1}.
                let cont = max(acc - pt_acc_mid, float3(0.0));
                Lc = float3(gi_mask.r > 1.0e-6 ? cont.r / gi_mask.r : 0.0,
                            gi_mask.g > 1.0e-6 ? cont.g / gi_mask.g : 0.0,
                            gi_mask.b > 1.0e-6 ? cont.b / gi_mask.b : 0.0);
                Ln = float3(gi_mask.r > 1.0e-6 ? pt_nee.r / gi_mask.r : 0.0,
                            gi_mask.g > 1.0e-6 ? pt_nee.g / gi_mask.g : 0.0,
                            gi_mask.b > 1.0e-6 ? pt_nee.b / gi_mask.b : 0.0);
                acc = gi_acc;
                storeAligned<16>((uint4*)(pc.gi_candidates + prim_at) + 3,
                                 uint4(packRadiance(Ln), packHalf2(float32x3_to_oct(pt_cont_dir)),
                                       packHalf2(float32x3_to_oct(pt_nee_dir)), pt_nk));
            } else {
                pc.gi_candidates[prim_at].pdf = 0.0;
            }
            pc.gi_candidates[prim_at].L = packRadiance(Lc);
            pc.gi_candidates[prim_at].frame = pc.uniforms.frame;
        }
    }

    var mean = acc / float(SAMPLES);
    // Planetary atmosphere: aerial perspective on the primary hit and the cloud shell,
    // folded into the colour before Ray Reconstruction (deterministic: no per-frame noise).
    if (ATMOSPHERE_ON && pc.uniforms.environments[pc.uniforms.environment].mode == 3u) {
        float3 surface_t;
        mean = atmoCompositeT(cameraOrigin(), pixelDirection(launch_id, launch_size),
                              pc.primary_hits[prim_at].primary_t,
                              atmoBayer(px), mean, surface_t);
        // gi_shade's light reaches the camera through the same air.
        if (GI_ON && pc.gi_candidates != nullptr) {
            pc.restir_surfaces[prim_at].gi_view_t = packRgb9e5(surface_t);
        }
    }
    // Next frame's auto-exposure metering (auto_exposure.slang): the incident meter (< 0 =
    // skipped) and the raw nits for the panel readout.
    // Meter what is seen: the surface through the fog plus the fog (composited after Ray
    // Reconstruction; metered without it, the exposure ignored the veil and it blew out).
    // Auto-exposure meters incident light: radiance over sample 0's guide-vertex albedo (the
    // primary pass's; < 0: the sky or an emitter, not metered).
    let meter_albedo = pc.primary_hits[prim_at].meter_albedo;
    let fog = guide_fog[px];
    let fog_out = fog.rgb / max(pc.ae.input_exposure, 1.0e-20);
    let fog_out_t = fog.a;
    let lum_nits = dot(float3(0.2125, 0.7154, 0.0721), mean * fog_out_t + fog_out);
    let lum_at = 2u * (uint(px.y) * launch_size.x + uint(px.x));
    // The incident meter: the surface's light over its albedo, the fog's as is (it reflects
    // nothing; it is light in the air).
    let fog_nits = dot(float3(0.2125, 0.7154, 0.0721), fog_out);
    // Null for a camera's second view (a headset's right eye): one view per camera meters.
    if (pc.lum != nullptr) {
        pc.lum[lum_at] = meter_albedo < 0.0 ? -1.0 : (lum_nits - fog_nits) / max(meter_albedo, 0.04) + fog_nits;
        // The pair's second slot: the albedo the meter divides by (< 0 = not metered), for the
        // post-process pass, which re-meters from Ray Reconstruction's output (quad.slang) and
        // leaves the raw nits there.
        pc.lum[lum_at + 1u] = meter_albedo < 0.0 ? -1.0 : max(meter_albedo, 0.04);
    }
    // Ray Reconstruction takes PRE-EXPOSED linear HDR (it has no exposure handling of its
    // own -- RR guide 3.7), so the input exposure is applied here (eased after the look: RR's
    // history mixes frames, and a moving input exposure reads as a transient) and the blit
    // re-exposes to the look.
    // Finite and inside f16 range.
    let radiance = clamp(mean * pc.ae.input_exposure, float3(0.0), float3(65504.0));
    dlss_color[px] = float4(radiance, 1.0);

    // The Cost view: this pixel's raygen ns, and the frame's total (one atomic per wave).
    if (pc.cost != nullptr) {
        let ns = getRealtimeClock().x - clock_start;
        pc.cost[2u + uint(px.y) * launch_size.x + uint(px.x)] = ns;
        let total = WaveActiveSum(uint64_t(ns));
        if (WaveIsFirstLane()) {
            InterlockedAdd(((uint64_t*)pc.cost)[0], total);
        }
    }
}
