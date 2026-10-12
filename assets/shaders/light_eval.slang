// Light-table sampling and evaluation, shared by the raygen and the ReSTIR compute passes
// (restir.slang). Everything reaches the table and the live instance rows through a
// `LightCtx`, never the ray-tracing push constant, so a compute kernel can import it.
import rt_types;
import disney;

// Where the light table lives: its header, this frame's TLAS instance rows (3x4 transforms,
// 4 float4 per slot) and the emissive multiplier (`RenderSettings::emissive_boost`).
public struct LightCtx {
    public LightsHeader* lights;
    public float4* instances;
    public float emissive_boost;
};

public float3 vertexPosition(float* verts, uint i) {
    let b = i * 8u;
    return float3(verts[b], verts[b + 1u], verts[b + 2u]);
}

// The world-space corners of triangle `tri` of light instance `li`, through the live
// TLAS instance row (3x4 row-major, as in prev_instances); `g` gets the geometry index.
public void lightTriangleWorld(LightCtx ctx, LightInstance li, uint tri, out uint g, out float3 w0, out float3 w1, out float3 w2) {
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
    let r0 = ctx.instances[base + 0u];
    let r1 = ctx.instances[base + 1u];
    let r2 = ctx.instances[base + 2u];
    w0 = float3(dot(r0, float4(p0, 1.0)), dot(r1, float4(p0, 1.0)), dot(r2, float4(p0, 1.0)));
    w1 = float3(dot(r0, float4(p1, 1.0)), dot(r1, float4(p1, 1.0)), dot(r2, float4(p1, 1.0)));
    w2 = float3(dot(r0, float4(p2, 1.0)), dot(r1, float4(p2, 1.0)), dot(r2, float4(p2, 1.0)));
}

// Solid-angle pdf of the light table picking direction onto triangle `tri` of `li` at
// distance `t` along `dir` (for MIS against BRDF sampling). 0 when degenerate.
public float lightTrianglePdf(LightCtx ctx, LightInstance li, uint tri, float p_pick, float t, float3 dir) {
    uint g;
    float3 w0, w1, w2;
    lightTriangleWorld(ctx, li, tri, g, w0, w1, w2);
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
public float lightsEntryPick(LightCtx ctx, uint e) {
    var below = 0.0;
    if (e > 0u) {
        below = ctx.lights.cdf[e - 1u];
    }
    return (ctx.lights.cdf[e] - below) / ctx.lights.total_power;
}

// Entry from the power CDF at cumulative value x, and its pick probability.
public uint lightsPickEntry(LightCtx ctx, float x, out float p_pick) {
    uint lo = 0u, hi = ctx.lights.entry_count - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (ctx.lights.cdf[mid] < x) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    p_pick = lightsEntryPick(ctx, lo);
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
public float3 thinSurface(float3 v, float3 n, float3 ns, float3 dir, DisneyMaterial mat, Lobes lob) {
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
    return evalLayered(v, n, dir, mat, lob, true) * abs(NoL);
}

// Unshadowed contribution of analytic light `e` (same contract as lightSampleEval below).
// A radius-0 point or spot is a delta: `p_area` is just the pick probability and
// `pdf_solid` a huge sentinel, so the MIS weight against BRDF sampling goes to 1. Points,
// spots and rects have no geometry -- a BRDF ray never finds them, so nothing takes the
// complementary weight. A sphere emitter (kind 3) does: see sphereEmitterPdf.
public float3 analyticSampleEval(LightCtx ctx, uint e, float p_pick, float su, float sv,
                          float3 sp, float3 n, float3 ns, float3 v,
                          DisneyMaterial mat, Lobes lob, bool fixed_point,
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
        let surf = thinSurface(v, n, ns, dir, mat, lob);
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
        let surf = thinSurface(v, n, ns, dir, mat, lob);
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
        let surf = thinSurface(v, n, ns, dir, mat, lob);
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
    let surf = thinSurface(v, n, ns, dir, mat, lob);
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
public float3 lightSampleEval(LightCtx ctx, uint e, uint li_idx, float p_pick, float su, float sv,
                       float3 sp, float3 n, float3 ns, float3 v,
                       DisneyMaterial mat, Lobes lob, uint ray_mask, bool fixed_point,
                       out float p_area, out float pdf_solid, out float3 dir, out float dist) {
    if (e < ctx.lights.analytic_count) {
        if (((ctx.lights.analytics[e].flags >> 8) & ray_mask & 0xFFu) == 0u) {
            p_area = 0.0;
            pdf_solid = 0.0;
            dir = float3(0.0);
            dist = 0.0;
            return float3(0.0);
        }
        return analyticSampleEval(ctx, e, p_pick, su, sv, sp, n, ns, v, mat, lob, fixed_point,
                                  p_area, pdf_solid, dir, dist)
            * ctx.emissive_boost;
    }
    p_area = 0.0;
    pdf_solid = 0.0;
    dir = float3(0.0);
    dist = 0.0;
    let li = ctx.lights.linsts[li_idx];
    // The emitter's world: its instance row's mask (packed above the custom index).
    if (((asuint(ctx.instances[li.slot * 4u + 3u].x) >> 24) & ray_mask & 0xFFu) == 0u) {
        return float3(0.0);
    }
    uint g;
    float3 w0, w1, w2;
    lightTriangleWorld(ctx, li, e - li.entry_base, g, w0, w1, w2);
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
    let surf = thinSurface(v, n, ns, dir, mat, lob);
    if (cos_l <= 1.0e-4 || all(surf == float3(0.0))) {
        return float3(0.0);
    }
    p_area = p_pick / (0.5 * area2);
    pdf_solid = p_area * dist * dist / cos_l;
    let emission = float3(li.geom_emission[g * 3u + 0u],
                          li.geom_emission[g * 3u + 1u],
                          li.geom_emission[g * 3u + 2u]);
    return surf * emission * (ctx.emissive_boost * cos_l / (dist * dist));
}

public float fogPhaseG(float g, float cos_theta) {
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(denom));
}

// ---- ReSTIR DI (src/restir.rs) -------------------------------------------------------------

// A surface's material, normals and view from its packed record.
public void restirSurfaceUnpack(RestirSurface s, out DisneyMaterial mat, out Lobes lob,
                                out float3 n, out float3 ns, out float3 v) {
    mat = disneyMaterial(s.albedo, s.metallic, s.roughness);
    lob.transmission = s.transmission;
    lob.clearcoat = s.clearcoat;
    lob.coat_alpha = s.coat_alpha;
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
    DisneyMaterial mat;
    Lobes lob;
    float3 n, ns, v;
    restirSurfaceUnpack(s, mat, lob, n, ns, v);
    let uv = unpackHalf2(uv_pk);
    p_cdf = lightsEntryPick(ctx, e);
    float pdf_solid;
    var val = lightSampleEval(ctx, e, lightsFindInstance(ctx, e), p_cdf, uv.x, uv.y,
                              s.position, n, ns, v, mat, lob, s.ray_mask, true,
                              p_area_cdf, pdf_solid, dir, dist);
    if (pdf_solid > 0.0) {
        val *= pdf_solid / (pdf_solid + layeredPdf(v, n, dir, mat, lob));
    }
    return val;
}

public float restirLuma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}
