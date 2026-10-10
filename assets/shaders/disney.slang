// The path tracer's BRDF (disney.glsl in Slang).
// From: https://www.shadertoy.com/view/cll3R4
module disney;

import rand;
import rt_types;

public struct DisneyMaterial {
    public float3 albedo;
    public float metallic;
    public float roughness;
};

public DisneyMaterial disneyMaterial(float3 albedo, float metallic, float roughness) {
    DisneyMaterial m;
    m.albedo = albedo;
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

// ---------------------------------------------
// Microfacet
// ---------------------------------------------
float3 F_Schlick(float3 f0, float theta) {
    return f0 + (1. - f0) * pow(1.0 - theta, 5.);
}

float F_Schlick(float f0, float f90, float theta) {
    return f0 + (f90 - f0) * pow(1.0 - theta, 5.0);
}

// GGX / GTR2 with `alpha` the microfacet width (perceptual roughness squared), the same
// alpha SampleGGXVNDF draws from. (The shadertoy original squared alpha again in the
// denominator: a lobe 1/alpha^2 too strong and far too narrow for every non-rough
// material, which showed as white blasts where the sun's next-event estimate hit.)
float D_GTR(float alpha, float NoH, float k) {
    let a2 = alpha * alpha;
    return a2 / (PI * pow((NoH * NoH) * (a2 - 1.) + 1., k));
}

// Smith GGX G1 for the same `alpha` as D_GTR.
float SmithG(float NDotV, float alpha) {
    let a = alpha * alpha;
    let b = NDotV * NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
}

// Separable Smith masking-shadowing, matching the VNDF sampler's G1.
float GeometryTerm(float NoL, float NoV, float alpha) {
    return SmithG(NoV, alpha) * SmithG(NoL, alpha);
}

float3 SampleGGXVNDF(float3 V, float ax, float ay, float r1, float r2) {
    let Vh = normalize(float3(ax * V.x, ay * V.y, V.z));

    let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    let T1 = lensq > 0. ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    let T2 = cross(Vh, T1);

    let r = sqrt(r1);
    let phi = 2.0 * PI * r2;
    let t1 = r * cos(phi);
    var t2 = r * sin(phi);
    let s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    return normalize(float3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
}

float GGXVNDFPdf(float NoH, float NoV, float alpha) {
    let D = D_GTR(alpha, NoH, 2.);
    let G1 = SmithG(NoV, alpha);
    return (D * G1) / max(0.00001, 4.0 * NoV);
}

// ---------------------------------------------
// Maths
// ---------------------------------------------

// From fizzer - https://web.archive.org/web/20170610002747/http://amietia.com/lambertnotangent.html
float3 cosineSampleHemisphere(float3 n, float r1, float r2) {
    let a = PI * 2. * r1;
    let b = 2.0 * r2 - 1.0;

    let dir = float3(sqrt(1.0 - b * b) * float2(cos(a), sin(a)), b);
    return normalize(n + dir);
}

// From pixar - https://graphics.pixar.com/library/OrthonormalB/paper.pdf
void basis(float3 n, out float3 b1, out float3 b2) {
    if (n.z < 0.) {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = float3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = float3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
}

float3 toWorld(float3 x, float3 y, float3 z, float3 v) {
    return v.x * x + v.y * y + v.z * z;
}

float3 toLocal(float3 x, float3 y, float3 z, float3 v) {
    return float3(dot(v, x), dot(v, y), dot(v, z));
}

// ---------------------------------------------
// Color
// ---------------------------------------------
public float luma(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

// ---------------------------------------------
// BRDF
// ---------------------------------------------
float3 evalDisneyDiffuse(DisneyMaterial mat, float NoL, float NoV, float LoH, float roughness) {
    let FD90 = 0.5 + 2. * roughness * pow(LoH, 2.);
    let a = F_Schlick(1., FD90, NoL);
    let b = F_Schlick(1., FD90, NoV);

    return mat.albedo * (a * b / PI);
}

float3 evalDisneySpecular(DisneyMaterial mat, float3 F, float NoH, float NoV, float NoL) {
    let alpha = pow(mat.roughness, 2.);
    let D = D_GTR(alpha, NoH, 2.);
    let G = GeometryTerm(NoL, NoV, alpha);

    let spec = D * F * G / (4. * NoL * NoV);

    return spec;
}

// Below this alpha the lobe is a delta: sampled as a mirror with weight F and pdf 0.
static const float DELTA_ALPHA = 1.0e-3;

// Returns (weight * NoL, lobe-weighted pdf). `a == 0` is a delta (mirror) bounce: apply
// the weight as is. `a < 0` is a failed sample (the direction is under the surface):
// the path carries nothing and must end.
static const float4 BRDF_SAMPLE_FAILED = float4(0.0, 0.0, 0.0, -1.0);

public float4 sampleDisneyBRDF(float4 noise, float3 v, float3 n, DisneyMaterial mat, inout float3 l) {
    let roughness = pow(mat.roughness, 2.);

    // sample microfacet normal
    float3 t, b;
    basis(n, t, b);
    let V = toLocal(t, b, n, v);
    var h = SampleGGXVNDF(V, roughness, roughness, noise.z, noise.w);
    if (h.z < 0.0) {
        h = -h;
    }
    h = toWorld(t, b, n, h);

    // fresnel
    let f0 = lerp(float3(0.04), mat.albedo, mat.metallic);
    let F = F_Schlick(f0, dot(v, h));

    // lobe weight probability
    var diffW = (1. - mat.metallic);
    var specW = luma(F);
    let invW = 1. / (diffW + specW);
    diffW *= invW;
    specW *= invW;

    var brdf = float4(0.);
    let rnd = randf();
    if (rnd < diffW) { // diffuse
        l = cosineSampleHemisphere(n, noise.x, noise.y);
        h = normalize(l + v);

        let NoL = dot(n, l);
        let NoV = dot(n, v);
        if (NoL <= 0. || NoV <= 0.) {
            return BRDF_SAMPLE_FAILED;
        }
        let LoH = dot(l, h);
        let pdf = NoL / PI;

        let diff = evalDisneyDiffuse(mat, NoL, NoV, LoH, roughness) * (1. - F);
        brdf.rgb = diff * NoL;
        brdf.a = diffW * pdf;
    } else { // specular
        l = reflect(-v, h);

        let NoL = dot(n, l);
        let NoV = dot(n, v);
        if (NoL <= 0. || NoV <= 0.) {
            return BRDF_SAMPLE_FAILED;
        }
        if (roughness < DELTA_ALPHA) {
            // Mirror: h == n, so F is the weight and there is no pdf to divide by.
            brdf.rgb = F / specW;
            brdf.a = 0.0;
            return brdf;
        }
        let NoH = min(dot(n, h), .99);
        let pdf = GGXVNDFPdf(NoH, NoV, roughness);

        let spec = evalDisneySpecular(mat, F, NoH, NoV, NoL);
        brdf.rgb = spec * NoL;
        brdf.a = specW * pdf;
    }

    return brdf;
}

// The BRDF that sampleDisneyBRDF implicitly integrates, evaluated for a given light direction
// (no cosine): Disney diffuse scaled by (1 - F) and (1 - metallic), plus the GGX specular lobe.
// Next-event estimation (the sun) uses this so its estimate matches BRDF sampling.
public float3 evalDisneyBRDF(float3 v, float3 n, float3 l, DisneyMaterial mat) {
    let NoL = dot(n, l);
    let NoV = dot(n, v);
    if (NoL <= 0.0 || NoV <= 0.0) {
        return float3(0.0);
    }
    let h = normalize(l + v);
    let NoH = min(dot(n, h), 0.99);
    let LoH = dot(l, h);
    let roughness = pow(mat.roughness, 2.);
    let f0 = lerp(float3(0.04), mat.albedo, mat.metallic);
    let F = F_Schlick(f0, dot(v, h));
    let diff = evalDisneyDiffuse(mat, NoL, NoV, LoH, roughness) * (1. - F) * (1. - mat.metallic);
    let spec = evalDisneySpecular(mat, F, NoH, NoV, NoL);
    return diff + spec;
}

// The solid-angle pdf sampleDisneyBRDF draws directions from (lobe-weighted cosine +
// GGX VNDF), for MIS against light sampling.
public float pdfDisneyBRDF(float3 v, float3 n, float3 l, DisneyMaterial mat) {
    let NoL = dot(n, l);
    let NoV = dot(n, v);
    if (NoL <= 0.0 || NoV <= 0.0) {
        return 0.0;
    }
    let h = normalize(l + v);
    let NoH = min(dot(n, h), 0.99);
    let roughness = pow(mat.roughness, 2.);
    let f0 = lerp(float3(0.04), mat.albedo, mat.metallic);
    let F = F_Schlick(f0, dot(v, h));
    var diffW = 1. - mat.metallic;
    var specW = luma(F);
    let invW = 1. / (diffW + specW);
    diffW *= invW;
    specW *= invW;
    return diffW * (NoL / PI) + specW * GGXVNDFPdf(NoH, NoV, roughness);
}

// Uniform direction inside the cone of half-angle acos(cos_r) around `dir`; pdf = 1 / solid angle.
public float3 sampleCone(float3 dir, float cos_r, float r1, float r2) {
    let cos_t = 1.0 - r1 * (1.0 - cos_r);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = 2.0 * PI * r2;
    float3 t, b;
    basis(dir, t, b);
    return normalize(t * (sin_t * cos(phi)) + b * (sin_t * sin(phi)) + dir * cos_t);
}

// ---------------------------------------------
// Thin layers over the base (leaves, dew, wet stone)
// ---------------------------------------------

// The hit's `lobes` word (hit_payload.slang `packLobes`).
public struct Lobes {
    // Fraction of the diffuse light that passes through to the far side (thin translucency).
    public float transmission;
    // Clear coat strength and its GGX alpha (perceptual roughness squared).
    public float clearcoat;
    public float coat_alpha;
};

public Lobes unpackLobes(uint w) {
    Lobes l;
    l.transmission = float(w & 0xFFu) / 255.0;
    l.clearcoat = float((w >> 8) & 0xFFu) / 255.0;
    let r = float((w >> 16) & 0xFFu) / 255.0;
    l.coat_alpha = max(r * r, 0.002);
    return l;
}

// The coat's Fresnel at the view: its share of the light, which the base below loses.
public float coatFresnel(float NoV) {
    return F_Schlick(0.04, 1.0, saturate(NoV));
}

// The coat's GGX lobe (no cosine).
float coatSpecular(float3 v, float3 n, float3 l, float alpha) {
    let NoL = dot(n, l);
    let NoV = dot(n, v);
    if (NoL <= 0.0 || NoV <= 0.0) {
        return 0.0;
    }
    let h = normalize(l + v);
    let D = D_GTR(alpha, min(dot(n, h), 0.9999), 2.0);
    let G = GeometryTerm(NoL, NoV, alpha);
    let F = F_Schlick(0.04, 1.0, dot(v, h));
    return D * G * F / max(4.0 * NoL * NoV, 1.0e-5);
}

// The whole surface for next-event estimation (no cosine): above it, the base under the coat
// (both scaled by what is not transmitted); below it, the light coming through, Lambertian,
// tinted by the albedo. `coat` false leaves the coat to BRDF sampling (the environment, whose
// MIS knows only the base's pdf).
public float3 evalLayered(float3 v, float3 n, float3 l, DisneyMaterial mat, Lobes lob, bool coat) {
    let NoL = dot(n, l);
    if (NoL <= 0.0) {
        return mat.albedo * (lob.transmission / PI);
    }
    var f = evalDisneyBRDF(v, n, l, mat);
    if (lob.clearcoat > 0.0) {
        f *= 1.0 - lob.clearcoat * coatFresnel(dot(n, v));
        if (coat) {
            f += lob.clearcoat * coatSpecular(v, n, l, lob.coat_alpha);
        }
    }
    return f * (1.0 - lob.transmission);
}

// A cosine-weighted direction through the surface (the translucent lobe).
public float3 sampleTransmission(float3 n, float r1, float r2) {
    return cosineSampleHemisphere(-n, r1, r2);
}

// A direction off the coat (GGX VNDF) and its weight F * G1(l): the coat lobe's f * cos / pdf.
// False when it would leave below the surface.
public bool sampleCoat(float3 v, float3 n, float alpha, float r1, float r2, out float3 l,
                       out float weight) {
    float3 t, b;
    basis(n, t, b);
    var h = SampleGGXVNDF(toLocal(t, b, n, v), alpha, alpha, r1, r2);
    h = toWorld(t, b, n, h);
    l = reflect(-v, h);
    let NoL = dot(n, l);
    weight = 0.0;
    if (NoL <= 0.0 || dot(n, v) <= 0.0) {
        return false;
    }
    weight = F_Schlick(0.04, 1.0, saturate(dot(v, h))) * SmithG(NoL, alpha);
    return true;
}
