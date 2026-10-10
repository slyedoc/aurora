// The path tracer's ray generation: camera paths with NEE (light table + ReSTIR DI, HDR
// environment, suns), the radiance cache, ray portals, the DLSS Ray Reconstruction guides,
// the atmosphere composite and the auto-exposure meter.
import rt_types;
import rand;
import disney;
import atmosphere_rt;

// DLSS Ray Reconstruction guides + noisy colour, render resolution, written for the
// primary hit of sample 0.
[[vk::binding(0, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> guide_normal_roughness;
[[vk::binding(1, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> guide_diffuse;
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> guide_specular;
[[vk::binding(3, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float4> guide_depth;
[[vk::binding(4, 0)]] [[vk::image_format("r32f")]] RWTexture2D<float4> guide_spec_hit;
[[vk::binding(5, 0)]] [[vk::image_format("rg16f")]] RWTexture2D<float4> guide_motion;
[[vk::binding(6, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> dlss_color;
[[vk::binding(100, 0)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(200, 1)]] Sampler2D textures[];

static const bool ENABLE_BLUENOISE = true;
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

float sampleDistanceInFog() {
    return -log(randf()) / pc.uniforms.foginess;
}

// Fog extinction at a height: `foginess` at and below `fog_height`, thinning by 1/e every
// `fog_falloff` metres above it.
float fogDensity(float y) {
    let u = pc.uniforms;
    return u.foginess * exp(-max(0.0, y - u.fog_height) / u.fog_falloff);
}

// Whether the ray scatters in the fog before `t_max` (0 = it reached the sky: no fog), and
// where. Uniform fog draws one exponential distance; height fog delta-tracks against the
// density at the segment's lowest point (the majorant), accepting a tentative collision with
// probability density / majorant.
bool sampleFogScatter(float3 o, float3 d, float t_max, out float t) {
    if (pc.uniforms.fog_falloff <= 0.0) {
        t = sampleDistanceInFog();
        return t < t_max;
    }
    t = 0.0;
    let majorant = fogDensity(min(o.y, o.y + d.y * t_max));
    if (t_max <= 0.0 || majorant <= 1.0e-9) {
        return false;
    }
    for (int i = 0; i < 64; i++) {
        t += -log(randf()) / majorant;
        if (t >= t_max) {
            return false;
        }
        if (randf() * majorant < fogDensity(o.y + d.y * t)) {
            return true;
        }
    }
    return false;
}

// Height fog: the share of a collision that scatters rather than absorbs (water droplets).
static const float FOG_ALBEDO = 0.95;

// Henyey-Greenstein phase function, per steradian.
float fogPhase(float cos_theta) {
    let g = pc.uniforms.fog_scatter;
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(denom));
}

float calcFogTransmittance(float distance) {
    return exp(-g_extinction * distance);
}

float3 sampleFogScatterDirection(float3 incoming) {
    // Henyey-Greenstein phase function sampling
    // g is the asymmetry parameter (-1 <= g <= 1)
    // g > 0 means forward scattering
    // g < 0 means backward scattering
    // g = 0 means isotropic scattering

    let g = pc.uniforms.fog_scatter;
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

float3 vertexPosition(float* verts, uint i) {
    let b = i * 8u;
    return float3(verts[b], verts[b + 1u], verts[b + 2u]);
}

// The world-space corners of triangle `tri` of light instance `li`, through the live
// TLAS instance row (3x4 row-major, as in prev_instances); `g` gets the geometry index.
void lightTriangleWorld(LightInstance li, uint tri, out uint g, out float3 w0, out float3 w1, out float3 w2) {
    g = 0u;
    for (uint k = 1u; k < li.geom_count; k++) {
        if (li.geom_to_triangle[k] <= tri) {
            g = k;
        } else {
            break;
        }
    }
    let idx = li.geom_to_index[g] + (tri - li.geom_to_triangle[g]) * 3u;
    let p0 = vertexPosition(li.vertices, li.indices[idx + 0u]);
    let p1 = vertexPosition(li.vertices, li.indices[idx + 1u]);
    let p2 = vertexPosition(li.vertices, li.indices[idx + 2u]);
    let base = li.slot * 4u;
    let r0 = pc.cur_instances[base + 0u];
    let r1 = pc.cur_instances[base + 1u];
    let r2 = pc.cur_instances[base + 2u];
    w0 = float3(dot(r0, float4(p0, 1.0)), dot(r1, float4(p0, 1.0)), dot(r2, float4(p0, 1.0)));
    w1 = float3(dot(r0, float4(p1, 1.0)), dot(r1, float4(p1, 1.0)), dot(r2, float4(p1, 1.0)));
    w2 = float3(dot(r0, float4(p2, 1.0)), dot(r1, float4(p2, 1.0)), dot(r2, float4(p2, 1.0)));
}

// Solid-angle pdf of the light table picking direction onto triangle `tri` of `li` at
// distance `t` along `dir` (for MIS against BRDF sampling). 0 when degenerate.
float lightTrianglePdf(LightInstance li, uint tri, float p_pick, float t, float3 dir) {
    uint g;
    float3 w0, w1, w2;
    lightTriangleWorld(li, tri, g, w0, w1, w2);
    let lng = cross(w1 - w0, w2 - w0);
    let area2 = length(lng);
    if (area2 <= 0.0) {
        return 0.0;
    }
    let cos_l = abs(dot(lng / area2, dir));
    if (cos_l <= 1.0e-4) {
        return 0.0;
    }
    return p_pick * t * t / (cos_l * 0.5 * area2);
}

// The pick probability of light-table entry `e` from the power CDF.
float lightsEntryPick(uint e) {
    var below = 0.0;
    if (e > 0u) {
        below = pc.lights.cdf[e - 1u];
    }
    return (pc.lights.cdf[e] - below) / pc.lights.total_power;
}

// ---- Radiance cache (SHaRC-style; see src/sharc.rs) ---------------------------------------

// Must match SHARC_ENTRIES in src/sharc.rs.
static const uint SHARC_ENTRIES = 1u << 20;

// Voxel key: position quantized at a distance-scaled size, level, dominant face normal.
uint4 sharcKey(float3 pos, float3 n, float3 cam) {
    let d = max(length(pos - cam), 1.0e-3);
    let level = floor(log2(max(d / 8.0, 1.0)));
    let s = max(pc.uniforms.sharc_voxel, 1.0e-3) * exp2(level);
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
bool sharcQuery(uint slot, out float3 radiance) {
    let rg = unpackHalf2(pc.sharc[slot].res_rg);
    let bc = unpackHalf2(pc.sharc[slot].res_bc);
    radiance = float3(rg, bc.x);
    return bc.y >= 8.0;
}

// One path sample of the voxel's outgoing radiance (nits, fixed point at 1 nit).
void sharcDeposit(uint slot, float3 radiance) {
    let r = clamp(radiance, float3(0.0), float3(1.0e6));
    InterlockedAdd(pc.sharc[slot].acc_r, uint(r.r));
    InterlockedAdd(pc.sharc[slot].acc_g, uint(r.g));
    InterlockedAdd(pc.sharc[slot].acc_b, uint(r.b));
    InterlockedAdd(pc.sharc[slot].acc_n, 1u);
    pc.sharc[slot].frame = pc.uniforms.frame;
}

// Entry from the power CDF at cumulative value x, and its pick probability.
uint lightsPickEntry(float x, out float p_pick) {
    uint lo = 0u, hi = pc.lights.entry_count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (pc.lights.cdf[mid] < x) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    p_pick = lightsEntryPick(lo);
    return lo;
}

// The light instance owning entry e (greatest entry_base <= e); 0 for analytic entries
// (which have no instance -- lightSampleEval never reads it for them).
uint lightsFindInstance(uint e) {
    if (e < pc.lights.analytic_count || pc.lights.linst_count == 0u) {
        return 0u;
    }
    uint lo = 0u, hi = pc.lights.linst_count - 1u;
    while (lo < hi) {
        let mid = (lo + hi + 1u) / 2u;
        if (pc.lights.linsts[mid].entry_base <= e) {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    return lo;
}

// Unshadowed contribution of analytic light `e` (same contract as lightSampleEval below).
// A radius-0 point or spot is a delta: `p_area` is just the pick probability and
// `pdf_solid` a huge sentinel, so the MIS weight against BRDF sampling goes to 1. Points,
// spots and rects have no geometry -- a BRDF ray never finds them, so nothing takes the
// complementary weight. A sphere emitter (kind 3) does: see sphereEmitterPdf.
float3 analyticSampleEval(uint e, float p_pick, float su, float sv,
                          float3 sp, float3 n, float3 ns, float3 v,
                          DisneyMaterial mat,
                          out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    p_area = 0.0;
    pdf_solid = 0.0;
    dir = float3(0.0);
    dist = 0.0;
    let al = pc.lights.analytics[e];
    let al_position = float3(al.position_x, al.position_y, al.position_z);
    let al_direction = float3(al.direction_x, al.direction_y, al.direction_z);
    let al_emission = float3(al.emission_r, al.emission_g, al.emission_b);
    if (al.kind == 2u) {
        // Rect: uniform point on the rectangle, evaluated exactly like a triangle entry.
        let al_tangent = float3(al.tangent_x, al.tangent_y, al.tangent_z);
        let bitangent = cross(al_direction, al_tangent);
        let point = al_position
            + al_tangent * ((su * 2.0 - 1.0) * al.half_extents_x)
            + bitangent * ((sv * 2.0 - 1.0) * al.half_extents_y);
        let to_light = point - sp;
        dist = length(to_light);
        if (dist <= 1.0e-4) {
            return float3(0.0);
        }
        dir = to_light / dist;
        var cos_l = dot(al_direction, -dir);
        if ((al.flags & 1u) != 0u) {
            cos_l = abs(cos_l);
        }
        let NoL = dot(n, dir);
        if (cos_l <= 1.0e-4 || NoL <= 0.0 || dot(dir, ns) <= 0.0) {
            return float3(0.0);
        }
        let area = 4.0 * al.half_extents_x * al.half_extents_y;
        if (area <= 0.0) {
            return float3(0.0);
        }
        p_area = p_pick / area;
        pdf_solid = p_area * dist * dist / cos_l;
        return evalDisneyBRDF(v, n, dir, mat) * NoL * al_emission * (cos_l / (dist * dist));
    }
    // Point / spot / sphere emitter. With a radius the light is a Lambertian sphere of that
    // size: a point or spot (bevy's `radius`) has no geometry and spreads its intensity over
    // the projected disc, L = I / (pi r^2), so the total is unchanged and the shadows go
    // soft; a sphere emitter (kind 3) IS an emissive `Sphere` in the scene and carries its
    // radiance directly. Radius 0 is the delta at the position.
    let to_centre = al_position - sp;
    let centre_dist = length(to_centre);
    if (centre_dist <= 1.0e-4) {
        return float3(0.0);
    }
    let centre_dir = to_centre / centre_dist;
    var atten = 1.0;
    if (al.kind == 1u) {
        // Bevy's spot falloff: smooth between the inner and outer cone (about the centre).
        let cos_a = dot(al_direction, -centre_dir);
        if (cos_a <= al.cos_outer) {
            return float3(0.0);
        }
        atten = smoothstep(al.cos_outer, max(al.cos_inner, al.cos_outer + 1.0e-4), cos_a);
    }
    if (al.radius <= 1.0e-4 || centre_dist <= al.radius) {
        // Inside a sphere emitter every direction sees it: BRDF hits carry it at full weight.
        if (al.kind == 3u) {
            return float3(0.0);
        }
        dist = centre_dist;
        dir = centre_dir;
        let NoL = dot(n, dir);
        if (NoL <= 0.0 || dot(dir, ns) <= 0.0) {
            return float3(0.0);
        }
        p_area = p_pick;
        pdf_solid = 1.0e16;
        return evalDisneyBRDF(v, n, dir, mat) * NoL * al_emission * (atten / (dist * dist));
    }
    // Uniform over the CONE the sphere subtends from here: every sample lands on the visible
    // cap, where uniform-area sampling threw the far half away and piled variance onto
    // whatever stood close to the light.
    let sin2_max = (al.radius * al.radius) / (centre_dist * centre_dist);
    let cos_max = sqrt(max(0.0, 1.0 - sin2_max));
    let one_minus_cos_max = sin2_max / (1.0 + cos_max);  // no cancellation far away
    let cos_t = 1.0 - su * one_minus_cos_max;
    let sin2_t = max(0.0, 1.0 - cos_t * cos_t);
    let phi = 2.0 * PI * sv;
    dir = normalize(mul(float3(sqrt(sin2_t) * cos(phi), sqrt(sin2_t) * sin(phi), cos_t),
                        orthonormalBasis(centre_dir)));
    let surface_dist = centre_dist * cos_t
        - sqrt(max(0.0, al.radius * al.radius - centre_dist * centre_dist * sin2_t));
    if (surface_dist <= 1.0e-4) {
        return float3(0.0);
    }
    let NoL = dot(n, dir);
    if (NoL <= 0.0 || dot(dir, ns) <= 0.0) {
        return float3(0.0);
    }
    let sn = (sp + dir * surface_dist - al_position) / al.radius;
    let cos_l = max(dot(sn, -dir), 1.0e-4);
    pdf_solid = p_pick / (2.0 * PI * one_minus_cos_max);
    p_area = pdf_solid * cos_l / (surface_dist * surface_dist);
    // The shadow ray stops short of the surface: a sphere emitter's own geometry is there.
    dist = surface_dist * 0.9995;
    let radiance = al.kind == 3u ? al_emission : al_emission / (PI * al.radius * al.radius);
    return evalDisneyBRDF(v, n, dir, mat) * NoL * radiance
        * (atten * cos_l / (surface_dist * surface_dist));
}

// Solid-angle pdf with which next-event estimation at `from` would have sampled sphere
// emitter `e` (analyticSampleEval's cone); 0 inside it. The MIS partner of a BRDF ray that
// hit the sphere.
float sphereEmitterPdf(uint e, float3 from) {
    let al = pc.lights.analytics[e];
    let to_centre = float3(al.position_x, al.position_y, al.position_z) - from;
    let d2 = dot(to_centre, to_centre);
    let r2 = al.radius * al.radius;
    if (d2 <= r2 || r2 <= 0.0) {
        return 0.0;
    }
    let p_pick = lightsEntryPick(e);
    let sin2_max = r2 / d2;
    return p_pick * (1.0 + sqrt(1.0 - sin2_max)) / (2.0 * PI * sin2_max);
}

// Unshadowed contribution of the point (su, sv) on light-table entry `e`, in AREA measure:
// f * NoL * Le * cos_l / d^2, evaluated at shading point `sp` (normal n, geometric normal
// ns, view v). Outputs the area pdf (p_pick / live area), the matching solid-angle pdf and
// the shadow-ray direction / distance; returns 0 on any rejection -- including a light in
// another world than `ray_mask`'s (the shadow ray could not see its geometry, so it would
// otherwise light through).
float3 lightSampleEval(uint e, uint li_idx, float p_pick, float su, float sv,
                       float3 sp, float3 n, float3 ns, float3 v,
                       DisneyMaterial mat, uint ray_mask,
                       out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    if (e < pc.lights.analytic_count) {
        if (((pc.lights.analytics[e].flags >> 8) & ray_mask & 0xFFu) == 0u) {
            p_area = 0.0;
            pdf_solid = 0.0;
            dir = float3(0.0);
            dist = 0.0;
            return float3(0.0);
        }
        return analyticSampleEval(e, p_pick, su, sv, sp, n, ns, v, mat,
                                  p_area, pdf_solid, dir, dist)
            * pc.uniforms.emissive_boost;
    }
    p_area = 0.0;
    pdf_solid = 0.0;
    dir = float3(0.0);
    dist = 0.0;
    let li = pc.lights.linsts[li_idx];
    // The emitter's world: its instance row's mask (packed above the custom index).
    if (((asuint(pc.cur_instances[li.slot * 4u + 3u].x) >> 24) & ray_mask & 0xFFu) == 0u) {
        return float3(0.0);
    }
    uint g;
    float3 w0, w1, w2;
    lightTriangleWorld(li, e - li.entry_base, g, w0, w1, w2);
    let lng = cross(w1 - w0, w2 - w0);
    let area2 = length(lng);
    if (area2 <= 0.0) {
        return float3(0.0);
    }
    let r = sqrt(max(su, 1.0e-6));
    let point = w0 * (1.0 - r) + w1 * (r * (1.0 - sv)) + w2 * (r * sv);
    let to_light = point - sp;
    dist = length(to_light);
    if (dist <= 1.0e-4) {
        return float3(0.0);
    }
    dir = to_light / dist;
    let cos_l = abs(dot(lng / area2, dir));
    let NoL = dot(n, dir);
    if (cos_l <= 1.0e-4 || NoL <= 0.0 || dot(dir, ns) <= 0.0) {
        return float3(0.0);
    }
    let f = evalDisneyBRDF(v, n, dir, mat);
    p_area = p_pick / (0.5 * area2);
    pdf_solid = p_area * dist * dist / cos_l;
    let emission = float3(li.geom_emission[g * 3u + 0u],
                          li.geom_emission[g * 3u + 1u],
                          li.geom_emission[g * 3u + 2u]);
    return f * NoL * emission * (pc.uniforms.emissive_boost * cos_l / (dist * dist));
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

// ---- HDR environment importance sampling (env_light.rs) ------------------------------------

// The environment sampler serves the camera world's HDR sky only (env_light.rs).
bool envSampled(uint ray_mask) {
    return pc.uniforms.env_w > 0u && environmentOf(ray_mask) == environmentOf(pc.uniforms.camera_mask);
}

// Solid-angle pdf of the environment sampler picking direction `d`.
float envPdf(float3 d) {
    let uv = env_dir_to_uv(d);
    let x = min(uint(uv.x * float(pc.uniforms.env_w)), pc.uniforms.env_w - 1u);
    let y = min(uint(uv.y * float(pc.uniforms.env_h)), pc.uniforms.env_h - 1u);
    let n = pc.uniforms.env_w * pc.uniforms.env_h;
    return pc.uniforms.env[n + 1u + y * pc.uniforms.env_w + x];
}

// Draws a direction from the environment's luminance CDF (binary search, jittered inside
// the texel); returns its radiance, with the direction and its solid-angle pdf.
float3 envSample(float u1, float u2, float u3, out float3 dir, out float pdf) {
    let w = pc.uniforms.env_w;
    let n = w * pc.uniforms.env_h;
    uint lo = 0u;
    uint hi = n;
    while (lo + 1u < hi) {
        let mid = (lo + hi) / 2u;
        if (pc.uniforms.env[mid] <= u1) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let i = min(lo, n - 1u);
    let uv = float2((float(i % w) + u2) / float(w), (float(i / w) + u3) / float(pc.uniforms.env_h));
    dir = env_uv_to_dir(uv);
    pdf = pc.uniforms.env[n + 1u + i];
    let u = pc.uniforms;
    let skycolor = float3(u.skycolor_r, u.skycolor_g, u.skycolor_b);
    // Explicit LOD 0: a ray-tracing stage has no derivatives (glslang emits the same).
    return skycolor * textures[pc.skydome].SampleLevel(uv, 0.0).rgb * u.sky_brightness;
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
bool portalRedirect(uint hit_slot, float3 hit_pos,
                    inout float3 origin, inout float3 direction, inout uint ray_mask) {
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
        // The ray now lives in whatever world the exit does: adopt the exit INSTANCE's
        // render-layer mask (instance row 3 packs custom_index | mask << 24).
        ray_mask = (asuint(pc.cur_instances[tb + 3u].x) >> 24) & 0xFFu;
        return true;
    }
    return false;
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

    let tmin = 0.0001;
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
    let SAMPLES = pc.uniforms.samples;
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
        // The camera ray and pure-specular continuations may see the sun disc directly; after a
        // BRDF-sampled bounce the sun comes from next-event estimation only.
        var want_sun = true;
        // Solid-angle pdf of the ray being traced, when it came from BRDF sampling (0 for the
        // camera ray and delta continuations): the MIS partner of light-table sampling.
        var prev_pdf = 0.0;
        // The world this ray sees: the camera's layer mask until a portal swaps it.
        var ray_mask = pc.uniforms.camera_mask;
        // Radiance-cache update targets along this path: the voxel and, per vertex, snapshots
        // of the running mask / radiance so the vertex's outgoing radiance falls out as a delta
        // once the path ends.
        uint sh_slot[4];
        float3 sh_mask[4];
        float3 sh_acc[4];
        var sh_n = 0u;
        // DLSS guides come from sample 0's first ACCEPTED vertex (a primary ray may pass through
        // alpha-cutout edges first).
        var guides_written = false;
        // Ray cone for texture level of detail (closest_hit.rchit): width at the ray's origin
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
            // Low bit: may this miss return the sun disc; high bits: the ray's world mask
            // (the miss picks that layer's sky).
            payload.want_sun = (want_sun ? 1u : 0u) | (ray_mask << 1);
            payload.cone = float2(cone_width, cone_spread);
            trace(FLAGS_OPAQUE, ray_mask, origin, tmin, direction, tmax, payload);
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
                if (portalRedirect(payload.slot, origin + direction * payload.t, p_origin, p_dir, ray_mask)) {
                    if (!guides_written) {
                        guide_dist += payload.t;
                    }
                    origin = p_origin;
                    direction = p_dir;
                    prev_pdf = 0.0;
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
                            let vertex = payload;
                            payload.want_sun = ray_mask << 1;
                            payload.cone = float2(cone_width, cone_spread);
                            trace(FLAGS_OPAQUE, ray_mask, hit_pos, tmin, mirror, tmax, payload);
                            spec_hit = payload.t == 0.0 ? SPEC_HIT_FAR : min(payload.t, SPEC_HIT_FAR);
                            payload = vertex;
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

            // fog bounce (skybox has a distance of 0 so will never hit
            // this case, having an extra check for this is slower than
            // deferring to the generic emitter check)
            float fogDistance;
            if (sampleFogScatter(origin, direction, payload.t, fogDistance)) {
                origin = origin + direction * fogDistance;
                // Uniform fog keeps its old ad-hoc loss; height fog is delta-tracked, so the
                // collision's weight is just the medium's single-scattering albedo.
                mask *= pc.uniforms.fog_falloff > 0.0 ? FOG_ALBEDO : calcFogTransmittance(fogDistance);
                // The sun straight into the scatter point, through the phase function: the
                // shafts of light through gaps (a scattered ray would rarely find its disc).
                {
                    let env = pc.uniforms.worlds[environmentOf(ray_mask)];
                    for (uint si = 0u; si < env.sun_count; si++) {
                        let sun = env.suns[si];
                        let r1 = randf();
                        let r2 = randf();
                        let l = sampleCone(sun.direction.xyz, sun.direction.w, r1, r2);
                        var sun_radiance = sun.radiance.rgb;
                        if (env.mode == 3u && si == 0u) {
                            sun_radiance *= atmoSunTransmittanceAt(origin);
                        }
                        payload.t = -1.0;
                        trace(FLAGS_SHADOW, ray_mask, origin, 0.0, l, tmax, payload);
                        if (payload.t == 0.0) {
                            let solid_angle = 2.0 * PI * (1.0 - sun.direction.w);
                            acc += mask * fogPhase(dot(direction, l)) * sun_radiance * solid_angle;
                        }
                    }
                }
                direction = sampleFogScatterDirection(direction);
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
                        let pdf_light = sphereEmitterPdf(linst_idx & ~LIGHT_SLOT_ANALYTIC, origin);
                        if (pdf_light > 0.0) {
                            contribution *= prev_pdf / (prev_pdf + pdf_light);
                        }
                    } else if (linst_idx != 0xFFFFFFFFu) {
                        let li = pc.lights.linsts[linst_idx];
                        if (payload.prim_tri < li.tri_count) {
                            let e = li.entry_base + payload.prim_tri;
                            let p_pick = lightsEntryPick(e);
                            if (p_pick > 0.0) {
                                let pdf_light = lightTrianglePdf(li, payload.prim_tri, p_pick, payload.t, direction);
                                if (pdf_light > 0.0) {
                                    contribution *= prev_pdf / (prev_pdf + pdf_light);
                                }
                            }
                        }
                    }
                }
                // MIS against the environment sampler: a BRDF-sampled ray that escapes to the HDR
                // sky shares credit with the previous vertex's environment estimate.
                if (payload.t == 0.0 && prev_pdf > 0.0 && envSampled(ray_mask)) {
                    contribution *= prev_pdf / (prev_pdf + envPdf(direction));
                }
                // Firefly suppression: an indirect path that reaches an emitter far brighter than the
                // sky would take hundreds of frames to average out; clamp its luminance instead.
                if (bounce_idx > 0 && radiance_clamp > 0.0) {
                    contribution *= min(1.0, radiance_clamp / max(luma(contribution), 1.0e-6));
                }
                acc += contribution;
                break;
            }

            // Alpha-cutout pass-through: carry on along the same ray from the hit.
            origin = origin + direction * payload.t;
            if (pass_through) {
                continue;
            }

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

            // Radiance cache: diffuse-ish vertices past the primary hit take part. From bounce
            // 2 on, a converged voxel ends the path with its cached outgoing radiance; every
            // participating vertex is also recorded as an update target, so the cache keeps
            // learning from the paths that do get traced (including ones that end in a query).
            if (pc.uniforms.sharc != 0u && bounce_idx > 0
                && roughness >= 0.35 && metallic <= 0.5 && transmission < 0.01) {
                let slot = sharcSlot(origin, world_normal, initial_origin);
                if (slot != 0xFFFFFFFFu) {
                    if (bounce_idx >= 2) {
                        float3 cached;
                        if (sharcQuery(slot, cached)) {
                            acc += mask * cached;
                            break;
                        }
                    }
                    if (sh_n < 4u) {
                        sh_slot[sh_n] = slot;
                        sh_mask[sh_n] = mask;
                        sh_acc[sh_n] = acc;
                        sh_n++;
                    }
                }
            }

            // Direct light from the emissive-triangle table. At the primary vertex of sample 0
            // this is ReSTIR DI (RIS over the CDF candidates, temporal reservoir reuse across
            // the reprojected pixel, one visibility ray whose occlusion also clears the stored
            // weight); everywhere else it is one CDF sample with an MIS-weighted shadow ray.
            // Positions and areas are read live, so stale CDF weights change variance, never
            // the answer.
            if (pc.uniforms.light_entries > 0u && pc.lights.total_power > 0.0) {
                if (bounce_idx == 0 && sample_idx == 0 && pc.uniforms.restir_candidates > 0u) {
                    var r_y = 0u;
                    var r_uv = 0u;
                    var r_wsum = 0.0;
                    var r_M = 0.0;
                    var r_phat = 0.0;
                    var r_val = float3(0.0);
                    var r_dir = float3(0.0);
                    var r_dist = 0.0;
                    // Initial candidates.
                    for (uint c = 0u; c < pc.uniforms.restir_candidates; c++) {
                        float p_pick;
                        let e = lightsPickEntry(randf() * pc.lights.total_power, p_pick);
                        let su = randf();
                        let sv = randf();
                        float p_area, pdf_solid, cdist;
                        float3 cdir;
                        let val = lightSampleEval(e, lightsFindInstance(e), p_pick, su, sv,
                                                  origin, world_normal, surface_normal,
                                                  -direction, disney_mat, ray_mask,
                                                  p_area, pdf_solid, cdir, cdist);
                        r_M += 1.0;
                        let phat = luma(val);
                        if (phat > 0.0 && p_area > 0.0) {
                            let wi = phat / p_area;
                            r_wsum += wi;
                            if (randf() * r_wsum < wi) {
                                r_y = e;
                                r_uv = packHalf2(float2(su, sv));
                                r_phat = phat;
                                r_val = val;
                                r_dir = cdir;
                                r_dist = cdist;
                            }
                        }
                    }
                    // Temporal reuse: last frame's reservoir at the reprojected pixel, revalidated
                    // and re-evaluated at this frame's shading point.
                    let prev_clip = mat4MulVec(pc.uniforms.prev_view_proj, float4(payload.prev_world_pos, 1.0));
                    if (prev_clip.w > 1.0e-4) {
                        let uv01 = (prev_clip.xy / prev_clip.w) * 0.5 + 0.5;
                        // Inverse of the forward mapping (pixel_center = (x, H - y) + subpixel): the
                        // flipped y needs +1 so truncation lands in (y, y+1) like x's (x, x+1) --
                        // without it every fetch reads the row above and the reservoir pattern
                        // scrolls down the screen one row per frame.
                        let ppx = int2(int(uv01.x * float(launch_size.x)),
                                       int(float(launch_size.y) + 1.0 - uv01.y * float(launch_size.y)));
                        if (ppx.x >= 0 && ppx.y >= 0
                            && ppx.x < int(launch_size.x) && ppx.y < int(launch_size.y)) {
                            let prev = pc.reservoirs_prev[ppx.y * int(launch_size.x) + ppx.x];
                            if (prev.frame + 1u == pc.uniforms.frame
                                && prev.epoch == pc.uniforms.light_epoch
                                && prev.M > 0.0 && prev.W > 0.0
                                && abs(prev.depth - prev_clip.w) < 0.1 * prev_clip.w
                                && dot(oct_to_float32x3(unpackHalf2(prev.normal_pk)), world_normal) > 0.9) {
                                let suv = unpackHalf2(prev.uv_pk);
                                float p_area_ign, pdf_solid_ign, cdist;
                                float3 cdir;
                                let val = lightSampleEval(prev.y, lightsFindInstance(prev.y), 1.0,
                                                          suv.x, suv.y, origin, world_normal,
                                                          surface_normal, -direction, disney_mat,
                                                          ray_mask, p_area_ign, pdf_solid_ign, cdir,
                                                          cdist);
                                let phat = luma(val);
                                let Mprev = min(prev.M, pc.uniforms.restir_m_clamp);
                                if (phat > 0.0) {
                                    let wi = phat * prev.W * Mprev;
                                    r_wsum += wi;
                                    if (randf() * r_wsum < wi) {
                                        r_y = prev.y;
                                        r_uv = prev.uv_pk;
                                        r_phat = phat;
                                        r_val = val;
                                        r_dir = cdir;
                                        r_dist = cdist;
                                    }
                                }
                                r_M += Mprev;
                            }
                        }
                    }
                    // Resolve: contribution weight, one visibility ray, shade.
                    var r_W = (r_phat > 0.0 && r_M > 0.0) ? r_wsum / (r_M * r_phat) : 0.0;
                    if (r_W > 0.0) {
                        payload.t = -1.0;
                        trace(FLAGS_SHADOW, ray_mask, origin, tmin, r_dir, r_dist - 2.0 * tmin, payload);
                        if (payload.t == 0.0) {
                            acc += mask * r_val * r_W;
                        } else {
                            // Visibility reuse: an occluded winner is forgotten, not dragged along.
                            r_W = 0.0;
                        }
                    }
                    Reservoir cur;
                    cur.y = r_y;
                    cur.uv_pk = r_uv;
                    cur.W = r_W;
                    cur.M = min(r_M, pc.uniforms.restir_m_clamp);
                    cur.epoch = pc.uniforms.light_epoch;
                    cur.normal_pk = packHalf2(float32x3_to_oct(world_normal));
                    let cur_clip = mat4MulVec(pc.uniforms.view_proj, float4(origin, 1.0));
                    cur.depth = cur_clip.w;
                    cur.frame = pc.uniforms.frame;
                    pc.reservoirs_cur[px.y * int(launch_size.x) + px.x] = cur;
                } else {
                    // Resampled importance sampling, no reuse: M candidates from the power CDF, each
                    // weighted by its MIS-weighted unshadowed contribution at THIS point over its
                    // source pdf; one survives and gets the shadow ray. The table's power knows
                    // nothing of distance, so with hundreds of lights a single sample nearly always
                    // lands on one that contributes nothing. The MIS weight rides inside the
                    // resampled integrand (against the candidate's own CDF pdf, which is what the
                    // emitter-hit side uses), so the two strategies still sum to one. M = 1 is the
                    // plain estimator.
                    let M = bounce_idx == 0u ? pc.uniforms.light_candidates
                                             : max(1u, pc.uniforms.light_candidates / 4u);
                    var wsum = 0.0;
                    var sel_val = float3(0.0);
                    var sel_phat = 0.0;
                    var ldir = float3(0.0);
                    var ldist = 0.0;
                    for (uint c = 0u; c < M; c++) {
                        float p_pick;
                        let e = lightsPickEntry(randf() * pc.lights.total_power, p_pick);
                        let su = randf();
                        let sv = randf();
                        float p_area, pdf_solid, cdist;
                        float3 cdir;
                        let val = lightSampleEval(e, lightsFindInstance(e), p_pick, su, sv,
                                                  origin, world_normal, surface_normal, -direction,
                                                  disney_mat, ray_mask, p_area, pdf_solid, cdir,
                                                  cdist);
                        if (luma(val) <= 0.0 || p_area <= 0.0) {
                            continue;
                        }
                        let w = pdf_solid / (pdf_solid + pdfDisneyBRDF(-direction, world_normal, cdir, disney_mat));
                        let g = val * w;
                        let phat = luma(g);
                        let wi = phat / p_area;
                        wsum += wi;
                        if (randf() * wsum < wi) {
                            sel_val = g;
                            sel_phat = phat;
                            ldir = cdir;
                            ldist = cdist;
                        }
                    }
                    if (wsum > 0.0 && sel_phat > 0.0) {
                        payload.t = -1.0;
                        trace(FLAGS_SHADOW, ray_mask, origin, tmin, ldir, ldist - 2.0 * tmin, payload);
                        if (payload.t == 0.0) {
                            var contribution = mask * sel_val * (wsum / (float(M) * sel_phat));
                            if (bounce_idx > 0 && radiance_clamp > 0.0) {
                                contribution *= min(1.0, radiance_clamp / max(luma(contribution), 1.0e-6));
                            }
                            acc += contribution;
                        }
                    }
                }
            }

            // Next-event estimation for an HDR environment: one importance-sampled direction
            // (luminance x sin theta CDF, env_light.rs), MIS-weighted against BRDF sampling; the
            // BRDF-sampled rays that reach the sky take the complementary weight on miss.
            if (envSampled(ray_mask)) {
                float3 l;
                float pdf_env;
                let u1 = randf();
                let u2 = randf();
                let u3 = randf();
                let L = envSample(u1, u2, u3, l, pdf_env);
                let NoL = dot(world_normal, l);
                if (pdf_env > 0.0 && NoL > 0.0 && dot(l, surface_normal) > 0.0 && luma(L) > 0.0) {
                    // The coat is left to BRDF sampling: this MIS knows only the base's pdf.
                    let f = evalLayered(-direction, world_normal, l, disney_mat, lob, false);
                    if (any(f > float3(0.0))) {
                        let w = pdf_env / (pdf_env + pdfDisneyBRDF(-direction, world_normal, l, disney_mat));
                        payload.t = -1.0;
                        trace(FLAGS_SHADOW, ray_mask, origin, tmin, l, tmax, payload);
                        if (payload.t == 0.0) {
                            acc += mask * f * NoL * L * (w / pdf_env);
                        }
                    }
                }
            }

            // Next-event estimation for the world's suns (its DirectionalLights, src/sky.rs): one
            // shadow ray per sun towards a uniformly sampled point on its disc. Direct sunlight is
            // the dominant, low-variance term, so it is not firefly-clamped; BRDF-sampled rays that
            // reach a disc afterwards return the sky only (`want_sun` = 0), so nothing is counted
            // twice.
            {
                let env = pc.uniforms.worlds[environmentOf(ray_mask)];
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
                    if (env.mode == 3u && si == 0u) {
                        sun_radiance *= atmoSunTransmittanceAt(origin);
                        if (bounce_idx == 0u) {
                            sun_radiance *= cloudShadow(origin);
                        }
                    }
                    if (any(f > float3(0.0)) && any(sun_radiance > float3(0.0))) {
                        // Shadow ray: any accepted hit terminates without running a closest-hit
                        // shader, so the payload's t only becomes 0 when the miss shader ran. Not
                        // opaque-forced: the any-hit shader lets it through alpha-masked cutouts.
                        payload.t = -1.0;
                        trace(FLAGS_SHADOW, ray_mask, origin, tmin, l, tmax, payload);
                        if (payload.t == 0.0) {
                            let solid_angle = 2.0 * PI * (1.0 - sun.direction.w);
                            acc += mask * f * abs(NoL) * sun_radiance * solid_angle;
                        }
                    }
                }
            }

            // Thin layers before the base: through the surface with probability `transmission`
            // (weight = albedo: a Lambertian lobe drawn by its own cosine), off the coat with
            // probability clearcoat x Fresnel (weight F G1 / that). What is left samples the
            // base, whose weight the two cancel out of. Ordinary materials draw nothing here.
            if (lob.transmission > 0.0 && randf() < lob.transmission) {
                let r1 = randf();
                let r2 = randf();
                direction = sampleTransmission(world_normal, r1, r2);
                mask *= disney_mat.albedo;
                prev_pdf = 0.0;
                want_sun = false;
                cone_spread = max(cone_spread, 0.37);
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
                    prev_pdf = 0.0;
                    want_sun = false;
                    cone_spread = max(cone_spread, 0.02 + 0.35 * lob.coat_alpha);
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
            // brdf.a is the lobe-weighted solid-angle pdf; 0 marks a delta (mirror) bounce whose
            // weight is applied as is.
            prev_pdf = brdf.a > 0.0001 ? brdf.a : 0.0;

            // Prevent going into the surface (leaks light through walls)
            if (dot(direction, surface_normal) < 0) {
                break;
            }

            mask *= brdf.a > 0.0001 ? brdf.rgb / brdf.a : brdf.rgb;
            // The cone opens with the lobe: a mirror keeps it, a diffuse bounce spreads wide.
            cone_spread = max(cone_spread, 0.02 + 0.35 * roughness * roughness);
        }

        // The path ended: everything gathered after each cache vertex, divided by its mask
        // snapshot, is one sample of that voxel's outgoing radiance (zero radiance included --
        // darkness is learned too).
        for (uint si = 0u; si < sh_n; si++) {
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
    if (pc.uniforms.worlds[environmentOf(pc.uniforms.camera_mask)].mode == 3u) {
        mean = atmoComposite(initial_origin, initial_direction, primary_t, atmoBayer(px), mean);
    }
    // Next frame's auto-exposure metering (auto_exposure.slang): the incident meter (< 0 =
    // skipped) and the raw nits for the panel readout.
    let lum_nits = dot(float3(0.2125, 0.7154, 0.0721), mean);
    let lum_at = 2u * (uint(px.y) * launch_size.x + uint(px.x));
    pc.lum[lum_at] = meter_skip ? -1.0 : lum_nits / max(meter_albedo, 0.04);
    pc.lum[lum_at + 1u] = lum_nits;
    // Ray Reconstruction takes PRE-EXPOSED linear HDR (it has no exposure handling of its
    // own -- RR guide 3.7), so the QUANTISED metered exposure is applied here (constant
    // between rare whole-EV steps: RR's history mixes frames, and a per-frame wobble in the
    // input exposure showed up as full-screen flicker) and the blit re-exposes to the look.
    // Finite and inside f16 range.
    let radiance = clamp(mean * pc.ae.input_exposure, float3(0.0), float3(65504.0));
    dlss_color[px] = float4(radiance, 1.0);
}
