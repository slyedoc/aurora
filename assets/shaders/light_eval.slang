// Light-table sampling and evaluation, shared by the raygen and the ReSTIR compute passes
// (restir.slang). Everything reaches the table and the live instance rows through a
// `LightCtx`, never the ray-tracing push constant, so a compute kernel can import it.
import rt_types;
import bsdf;

// Where the light table lives: its header, this frame's TLAS instance rows (3x4 transforms,
// 4 float4 per slot) and the emissive multiplier (`RenderSettings::emissive_boost`).
public struct LightCtx {
    public LightsHeader* lights;
    public float4* instances;
    public float emissive_boost;
};

// Solid-angle pdf of the light table picking direction onto triangle `tri` of `li` at
// distance `t` along `dir` (for MIS against BRDF sampling). 0 when degenerate.
public float lightTrianglePdf(LightCtx ctx, LightInstance li, uint tri, float p_pick, float t, float3 dir) {
    let r = ctx.lights.prepared + (li.entry_base + tri) * 4u;
    let lng = cross(r[1].xyz, r[2].xyz);
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

// The environment whose runs hold entry `e`.
public uint lightsEntryWorld(LightCtx ctx, uint e) {
    for (uint w = 0u; w < LIGHT_WORLDS; w++) {
        let lw = ctx.lights.worlds[w];
        if ((e >= lw.analytic_first && e < lw.analytic_end)
            || (e >= lw.triangle_first && e < lw.triangle_end)) {
            return w;
        }
    }
    return 0u;
}

// The total power of environment `world`'s lights.
public float lightsWorldPower(LightCtx ctx, uint world) {
    let lw = ctx.lights.worlds[min(world, LIGHT_WORLDS - 1u)];
    return lw.analytic_power + lw.triangle_power;
}

// The pick probability of light-table entry `e` from the power CDF, among its own
// environment's lights (the only ones a ray there samples).
public float lightsEntryPick(LightCtx ctx, uint e) {
    var below = 0.0;
    if (e > 0u) {
        below = ctx.lights.cdf[e - 1u];
    }
    let total = lightsWorldPower(ctx, lightsEntryWorld(ctx, e));
    return total > 0.0 ? (ctx.lights.cdf[e] - below) / total : 0.0;
}

// An entry of environment `world` by power, u in [0, 1), and its pick probability (0 when the
// environment has no lights). u maps into the world's analytic run, then its triangle run, as a
// point of the global CDF; the guide table narrows the search to that point's bin: the answer
// lies between its first entry and the next bin's, widened by one each way for x rounding into a
// neighbouring bin at a boundary.
public uint lightsPickEntry(LightCtx ctx, uint world, float u, out float p_pick) {
    p_pick = 0.0;
    let lw = ctx.lights.worlds[min(world, LIGHT_WORLDS - 1u)];
    let total = lw.analytic_power + lw.triangle_power;
    if (total <= 0.0) {
        return 0u;
    }
    let t = u * total;
    let analytic = t < lw.analytic_power && lw.analytic_end > lw.analytic_first;
    let x = analytic ? lw.analytic_lo + t : lw.triangle_lo + max(t - lw.analytic_power, 0.0);
    let first = analytic ? lw.analytic_first : lw.triangle_first;
    let end = analytic ? lw.analytic_end : lw.triangle_end;
    if (end <= first) {
        return 0u;
    }
    uint lo = first, hi = end - 1u;
    let bins = ctx.lights.guide_bins;
    if (bins > 0u) {
        let j = min(uint(x / ctx.lights.total_power * float(bins)), bins - 1u);
        lo = clamp(ctx.lights.guide[j], first, end - 1u);
        lo = lo > first ? lo - 1u : first;
        hi = clamp(max(ctx.lights.guide[j + 1u], lo) + 1u, lo, end - 1u);
    }
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (ctx.lights.cdf[mid] < x) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let below = lo > 0u ? ctx.lights.cdf[lo - 1u] : 0.0;
    p_pick = (ctx.lights.cdf[lo] - below) / total;
    return lo;
}

// The light instance owning entry e; 0 for analytic entries (which have no instance --
// lightSampleEval never reads it for them).
public uint lightsFindInstance(LightCtx ctx, uint e) {
    return ctx.lights.entry_to_linst[e];
}

// A surface's response to light from `dir`, cosine included: the whole layered BSDF in front
// (base, coat, less what is transmitted), and behind a thin translucent one the light coming
// through it. Zero where it takes none (and wherever the geometric normal disagrees).
public float3 thinSurface(float3 v, float3 n, float3 ns, float3 dir, SurfaceMaterial mat, Lobes lob) {
    // n = 0: a point in the fog, `ns.x` its Henyey-Greenstein g.
    if (all(n == float3(0.0))) {
        return float3(fogPhaseG(ns.x, dot(-v, dir)));
    }
    let NoL = dot(n, dir);
    let above = NoL > 0.0 && dot(dir, ns) > 0.0;
    let through = lob.transmission > 0.0 && NoL < 0.0 && dot(dir, ns) < 0.0;
    if (!above && !through) {
        return float3(0.0);
    }
    return surfaceEval(v, n, dir, mat, lob);
}

// A cheap stand-in for thinSurface, luminance only: Lambert for the diffuse share, GGX D with
// Schlick's F0 and one Smith G1 for the base and the coat, Lambert through a translucent one.
// ReSTIR resamples by it (any target that is non-zero wherever the light can contribute keeps
// it unbiased); the shade evaluates the exact layered BSDF for the winner.
public float cheapSurface(float3 v, float3 n, float3 ns, float3 dir, SurfaceMaterial mat, Lobes lob) {
    let NoL = dot(n, dir);
    let a = dot(mat.albedo, float3(0.2126, 0.7152, 0.0722));
    if (lob.transmission > 0.0 && NoL < 0.0 && dot(dir, ns) < 0.0) {
        return a * lob.transmission * (-NoL) / PI;
    }
    if (NoL <= 0.0 || dot(dir, ns) <= 0.0) {
        return 0.0;
    }
    let NoV = max(dot(n, v), 0.05);
    let NoH = saturate(dot(n, normalize(v + dir)));
    let f0 = lerp(0.04, a, mat.metallic);
    let diffuse = a * (1.0 - mat.metallic) * (1.0 - lob.transmission) * NoL / PI;
    return diffuse + f0 * cheapGgx(max(mat.roughness * mat.roughness, 0.002), NoH, NoL, NoV)
        + 0.04 * lob.clearcoat * cheapGgx(lob.coat_alpha, NoH, NoL, NoV);
}

// GGX D times one Smith G1 (at the light), over 4 NoV: a specular lobe times NoL, F left out.
float cheapGgx(float alpha, float NoH, float NoL, float NoV) {
    let a2 = alpha * alpha;
    let d = NoH * NoH * (a2 - 1.0) + 1.0;
    let g1 = 2.0 * NoL / (NoL + sqrt(a2 + (1.0 - a2) * NoL * NoL));
    return a2 * g1 / (PI * d * d * 4.0 * NoV);
}

float3 surfaceResponse(bool cheap, float3 v, float3 n, float3 ns, float3 dir, SurfaceMaterial mat, Lobes lob) {
    return cheap ? float3(cheapSurface(v, n, ns, dir, mat, lob)) : thinSurface(v, n, ns, dir, mat, lob);
}

public float3 analyticSampleEval(LightCtx ctx, uint e, float p_pick, float su, float sv,
                                 float3 sp, float3 n, float3 ns, float3 v,
                                 SurfaceMaterial mat, Lobes lob, bool fixed_point,
                                 out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    return analyticSampleEvalWith(ctx, e, p_pick, su, sv, sp, n, ns, v, mat, lob, fixed_point, false,
                                  p_area, pdf_solid, dir, dist);
}

public float3 lightSampleEval(LightCtx ctx, uint e, uint li_idx, float p_pick, float su, float sv,
                              float3 sp, float3 n, float3 ns, float3 v,
                              SurfaceMaterial mat, Lobes lob, uint ray_mask, bool fixed_point,
                              out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    return lightSampleEvalWith(ctx, e, li_idx, p_pick, su, sv, sp, n, ns, v, mat, lob, ray_mask,
                               fixed_point, false, p_area, pdf_solid, dir, dist);
}

// Unshadowed contribution of analytic light `e` (same contract as lightSampleEval below).
// A radius-0 point or spot is a delta: `p_area` is just the pick probability and
// `pdf_solid` a huge sentinel, so the MIS weight against BRDF sampling goes to 1. Points,
// spots and rects have no geometry -- a BRDF ray never finds them, so nothing takes the
// complementary weight. A sphere emitter (kind 3) does: see sphereEmitterPdf.
float3 analyticSampleEvalWith(LightCtx ctx, uint e, float p_pick, float su, float sv,
                          float3 sp, float3 n, float3 ns, float3 v,
                          SurfaceMaterial mat, Lobes lob, bool fixed_point, bool cheap,
                          out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    p_area = 0.0;
    pdf_solid = 0.0;
    dir = float3(0.0);
    dist = 0.0;
    let al = ctx.lights.analytics[e];
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
        let surf = surfaceResponse(cheap, v, n, ns, dir, mat, lob);
        if (cos_l <= 1.0e-4 || all(surf == float3(0.0))) {
            return float3(0.0);
        }
        let area = 4.0 * al.half_extents_x * al.half_extents_y;
        if (area <= 0.0) {
            return float3(0.0);
        }
        p_area = p_pick / area;
        // No geometry: a BRDF ray never finds a rect, so nothing takes the complementary MIS
        // weight; the sentinel takes it to 1.
        pdf_solid = 1.0e16;
        return surf * al_emission * (cos_l / (dist * dist));
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
        let surf = surfaceResponse(cheap, v, n, ns, dir, mat, lob);
        if (all(surf == float3(0.0))) {
            return float3(0.0);
        }
        p_area = p_pick;
        pdf_solid = 1.0e16;
        return surf * al_emission * (atten / (dist * dist));
    }
    let radiance = al.kind == 3u ? al_emission : al_emission / (PI * al.radius * al.radius);
    if (fixed_point) {
        // Uniform over the sphere's whole area (ReSTIR): (su, sv) names the same point from
        // every surface, so a reused sample needs no Jacobian; the far side just has no target.
        let z = 1.0 - 2.0 * su;
        let rxy = sqrt(max(0.0, 1.0 - z * z));
        let phi_a = 2.0 * PI * sv;
        let sn = float3(rxy * cos(phi_a), rxy * sin(phi_a), z);
        let point = al_position + sn * al.radius;
        let to_light = point - sp;
        let d = length(to_light);
        if (d <= 1.0e-4) {
            return float3(0.0);
        }
        dir = to_light / d;
        let cos_l = dot(sn, -dir);
        if (cos_l <= 1.0e-4) {
            return float3(0.0);
        }
        let surf = surfaceResponse(cheap, v, n, ns, dir, mat, lob);
        if (all(surf == float3(0.0))) {
            return float3(0.0);
        }
        p_area = p_pick / (4.0 * PI * al.radius * al.radius);
        // Only a sphere emitter is geometry a BRDF ray can find (sphereEmitterPdf is its cone
        // pdf; this area pdf is close where the cap is small); the rest take MIS weight 1.
        pdf_solid = al.kind == 3u ? p_area * d * d / cos_l : 1.0e16;
        dist = d * 0.9995;
        return surf * radiance * (atten * cos_l / (d * d));
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
    let surf = surfaceResponse(cheap, v, n, ns, dir, mat, lob);
    if (all(surf == float3(0.0))) {
        return float3(0.0);
    }
    let sn = (sp + dir * surface_dist - al_position) / al.radius;
    let cos_l = max(dot(sn, -dir), 1.0e-4);
    let cone_pdf = p_pick / (2.0 * PI * one_minus_cos_max);
    p_area = cone_pdf * cos_l / (surface_dist * surface_dist);
    // A point or spot with a radius is still not geometry: MIS weight 1 (the sentinel), or
    // its light is under-counted with nothing to make it up.
    pdf_solid = al.kind == 3u ? cone_pdf : 1.0e16;
    // The shadow ray stops short of the surface: a sphere emitter's own geometry is there.
    dist = surface_dist * 0.9995;
    return surf * radiance * (atten * cos_l / (surface_dist * surface_dist));
}

// Solid-angle pdf with which next-event estimation at `from` would have sampled sphere
// emitter `e` (analyticSampleEval's cone); 0 inside it. The MIS partner of a BRDF ray that
// hit the sphere.
public float sphereEmitterPdf(LightCtx ctx, uint e, float3 from) {
    let al = ctx.lights.analytics[e];
    let to_centre = float3(al.position_x, al.position_y, al.position_z) - from;
    let d2 = dot(to_centre, to_centre);
    let r2 = al.radius * al.radius;
    if (d2 <= r2 || r2 <= 0.0) {
        return 0.0;
    }
    let p_pick = lightsEntryPick(ctx, e);
    let sin2_max = r2 / d2;
    return p_pick * (1.0 + sqrt(1.0 - sin2_max)) / (2.0 * PI * sin2_max);
}

// Unshadowed contribution of the point (su, sv) on light-table entry `e`, in AREA measure:
// f * NoL * Le * cos_l / d^2, evaluated at shading point `sp` (normal n, geometric normal
// ns, view v). Outputs the area pdf (p_pick / live area), the matching solid-angle pdf and
// the shadow-ray direction / distance; returns 0 on any rejection -- including a light in
// another world than `ray_mask`'s (the shadow ray could not see its geometry, so it would
// otherwise light through).
float3 lightSampleEvalWith(LightCtx ctx, uint e, uint li_idx, float p_pick, float su, float sv,
                       float3 sp, float3 n, float3 ns, float3 v,
                       SurfaceMaterial mat, Lobes lob, uint ray_mask, bool fixed_point, bool cheap,
                       out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    if (e < ctx.lights.analytic_count) {
        if (((ctx.lights.analytics[e].flags >> 8) & ray_mask & 0xFFu) == 0u) {
            p_area = 0.0;
            pdf_solid = 0.0;
            dir = float3(0.0);
            dist = 0.0;
            return float3(0.0);
        }
        return analyticSampleEvalWith(ctx, e, p_pick, su, sv, sp, n, ns, v, mat, lob, fixed_point, cheap,
                                  p_area, pdf_solid, dir, dist)
            * ctx.emissive_boost;
    }
    p_area = 0.0;
    pdf_solid = 0.0;
    dir = float3(0.0);
    dist = 0.0;
    // The entry as `light_prepare` resolved it this frame; its world mask first (a light in
    // another world than the ray's must not light through).
    let rec = ctx.lights.prepared + e * 4u;
    let r0 = rec[0];
    if ((asuint(r0.w) & ray_mask & 0xFFu) == 0u) {
        return float3(0.0);
    }
    let e1 = rec[1].xyz;
    let e2 = rec[2].xyz;
    let lng = cross(e1, e2);
    let area2 = length(lng);
    if (area2 <= 0.0) {
        return float3(0.0);
    }
    let r = sqrt(max(su, 1.0e-6));
    let point = r0.xyz + e1 * (r * (1.0 - sv)) + e2 * (r * sv);
    let to_light = point - sp;
    dist = length(to_light);
    if (dist <= 1.0e-4) {
        return float3(0.0);
    }
    dir = to_light / dist;
    let cos_l = abs(dot(lng / area2, dir));
    let surf = surfaceResponse(cheap, v, n, ns, dir, mat, lob);
    if (cos_l <= 1.0e-4 || all(surf == float3(0.0))) {
        return float3(0.0);
    }
    p_area = p_pick / (0.5 * area2);
    pdf_solid = p_area * dist * dist / cos_l;
    let emission = rec[3].xyz;
    return surf * emission * (ctx.emissive_boost * cos_l / (dist * dist));
}

public float fogPhaseG(float g, float cos_theta) {
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(denom));
}

// ---- ReSTIR DI (src/restir.rs) -------------------------------------------------------------

// A surface's material, normals and view from its packed record.
public void restirSurfaceUnpack(RestirSurface s, out SurfaceMaterial mat, out Lobes lob,
                                out float3 n, out float3 ns, out float3 v) {
    mat = surfaceMaterial(restirAlbedo(s), restirMetallic(s), restirRoughness(s));
    lob = unpackLobes(s.material);
    n = oct_to_float32x3(unpackHalf2(s.normal_oct));
    ns = oct_to_float32x3(unpackHalf2(s.geo_normal_oct));
    v = oct_to_float32x3(unpackHalf2(s.view_oct));
}

// ReSTIR's target at surface `s` for light sample (entry `e`, uv packed 2xf16): the unshadowed
// contribution, MIS-weighted against BRDF sampling with the light table's own (CDF) pdf --
// the pdf a BRDF-sampled bounce that hits this emitter is weighted against, so the two
// strategies still sum to one whatever drew the sample. Also the CDF's area pdf (`p_area_cdf`)
// and pick probability, the shadow ray's direction and length.
public float3 restirTarget(LightCtx ctx, RestirSurface s, uint e, uint uv_pk,
                           out float p_area_cdf, out float p_cdf, out float3 dir, out float dist) {
    SurfaceMaterial mat;
    Lobes lob;
    float3 n, ns, v;
    restirSurfaceUnpack(s, mat, lob, n, ns, v);
    let uv = unpackHalf2(uv_pk);
    p_cdf = lightsEntryPick(ctx, e);
    float pdf_solid;
    var val = lightSampleEval(ctx, e, lightsFindInstance(ctx, e), p_cdf, uv.x, uv.y,
                              s.position, n, ns, v, mat, lob, restirRayMask(s), true,
                              p_area_cdf, pdf_solid, dir, dist);
    if (pdf_solid > 0.0) {
        val *= pdf_solid / (pdf_solid + surfacePdf(v, n, dir, mat, lob));
    }
    return val;
}

// ReSTIR's resampling target for (e, uv) at `s`: the luminance of the unshadowed contribution
// through cheapSurface, without the MIS weight (the shade applies both exactly). Also the CDF's
// area pdf and pick probability, as restirTarget.
public float restirTargetCheap(LightCtx ctx, RestirSurface s, uint e, uint uv_pk,
                               out float p_area_cdf, out float p_cdf) {
    SurfaceMaterial mat;
    Lobes lob;
    float3 n, ns, v;
    restirSurfaceUnpack(s, mat, lob, n, ns, v);
    let uv = unpackHalf2(uv_pk);
    p_cdf = lightsEntryPick(ctx, e);
    float pdf_solid, dist;
    float3 dir;
    let val = lightSampleEvalWith(ctx, e, lightsFindInstance(ctx, e), p_cdf, uv.x, uv.y,
                                  s.position, n, ns, v, mat, lob, restirRayMask(s), true, true,
                                  p_area_cdf, pdf_solid, dir, dist);
    return restirLuma(val);
}

public float restirLuma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// ReSTIR PT: the radiance a sample's reconnection vertex x_k sends toward `wo` (x_k -> x_{k-1}),
// its BSDF evaluated again for that direction over the two directions it received light along:
// the continuation and the next-event sample, each stored as what it sent toward the path's own
// direction (`c.wo`) and scaled by the BSDF's ratio between the two views. The pdfs and MIS
// weights inside stay the path's -- fixed functions of the directions, so for any `wo` this is an
// unbiased estimate of what x_k sends there. (Stored as radiance over pdf the sun's next-event
// term overflowed RGB9E5.)
public float3 ptSampleRadiance(GiCandidate c, float3 wo) {
    let n = oct_to_float32x3(unpackHalf2(c.n_k));
    let a = float3(float(c.albedo_metallic & 0xFFu), float((c.albedo_metallic >> 8) & 0xFFu),
                   float((c.albedo_metallic >> 16) & 0xFFu)) / 255.0;
    let rt = unpackHalf2(asuint(c.roughness));
    let mat = surfaceMaterial(a * a, float(c.albedo_metallic >> 24) / 255.0, rt.x);
    // x_k's thin translucency: light it took through its far side keeps its (constant) lobe.
    Lobes lob = {};
    lob.transmission = rt.y;
    let wo_src = oct_to_float32x3(unpackHalf2(c.wo));
    var L = float3(0.0);
    let Lc = unpackRadiance(c.L);
    if (any(Lc > float3(0.0))) {
        let wi = oct_to_float32x3(unpackHalf2(c.wi_c));
        let f0 = surfaceEval(wo_src, n, wi, mat, lob);
        let f1 = surfaceEval(wo, n, wi, mat, lob);
        L += float3(f0.r > 0.0 ? f1.r / f0.r : 0.0, f0.g > 0.0 ? f1.g / f0.g : 0.0,
                    f0.b > 0.0 ? f1.b / f0.b : 0.0) * Lc;
    }
    let Ln = unpackRadiance(c.L_nee);
    if (any(Ln > float3(0.0))) {
        let wi = oct_to_float32x3(unpackHalf2(c.wi_n));
        let f0 = surfaceEval(wo_src, n, wi, mat, lob);
        let f1 = surfaceEval(wo, n, wi, mat, lob);
        L += float3(f0.r > 0.0 ? f1.r / f0.r : 0.0, f0.g > 0.0 ? f1.g / f0.g : 0.0,
                    f0.b > 0.0 ? f1.b / f0.b : 0.0) * Ln;
    }
    return L;
}

// ReSTIR PT: a sample's target vector at its own primary vertex `s` -- the same function the
// shift evaluates (raygen.slang ptReconnect: the vertex's BSDF and cosine over the reconnection,
// x_k's radiance re-evaluated) with visibility 1 (the path that found it is the proof). Computed
// once per sample (gi_candidates) and read by every neighbour's pairwise weight. A sample found
// deeper (k >= 2) would need a replay here: it takes its path's own value, T L pdf, which differs
// from a replay's only by the primary vertex's stored precision.
public float3 ptOwnTarget(RestirSurface s, GiCandidate c) {
    if ((c.k & 15u) != 1u) {
        return unpackRgb9e5(c.T) * ptSampleRadiance(c, oct_to_float32x3(unpackHalf2(c.wo))) * c.pdf;
    }
    let n_s = oct_to_float32x3(unpackHalf2(c.n_s_oct));
    let d = c.x_s - s.position;
    let d2 = max(dot(d, d), 1.0e-8);
    let w = d * rsqrt(d2);
    if (-dot(n_s, w) <= 0.0) {
        return float3(0.0);
    }
    let n = oct_to_float32x3(unpackHalf2(s.normal_oct));
    let ng = oct_to_float32x3(unpackHalf2(s.geo_normal_oct));
    let lob = unpackLobes(s.material);
    let NoL = dot(n, w);
    let front = NoL > 0.0 && dot(w, ng) > 0.0;
    let back = lob.transmission > 0.0 && NoL < 0.0 && dot(w, ng) < 0.0;
    if (!front && !back) {
        return float3(0.0);
    }
    let v = oct_to_float32x3(unpackHalf2(s.view_oct));
    let mat = surfaceMaterial(restirAlbedo(s), restirMetallic(s), restirRoughness(s));
    return surfaceEval(v, n, w, mat, lob) * ptSampleRadiance(c, -w);
}
