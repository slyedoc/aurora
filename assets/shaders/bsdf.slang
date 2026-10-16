// The surface BSDF: NVIDIA Falcor's standard BSDF (a Lambert diffuse BRDF, a Lambert diffuse BTDF
// for thin translucency, a GGX microfacet specular BRDF: height-correlated Smith masking,
// visible-normal sampling, Schlick Fresnel), the diffuse lobes under the specular layer by its
// albedo (as Cycles' Principled BSDF: Falcor only adds them), with aurora's clear coat on top.
// Against Cycles in the sun room (flat sky, all bounces): this 1.038 / block 0.045; Falcor's
// default Frostbite diffuse 0.686 (its energy normalization), plain Lambert added 1.087. Lobes are evaluated
// in the shading frame (z = the shading normal; `wi` toward the viewer, `wo` toward the light) and
// return f * |cos(wo)|. Evaluation is differentiable; sampling draws from an `ISampler`, so random
// replay (ReSTIR PT) samples with the same code as the path that found the sample.
//
// Adapted from Falcor (Source/Falcor/Rendering/Materials: IBSDF, StandardBSDF,
// FrostbiteDiffuseBRDF, LambertDiffuseBTDF, SpecularMicrofacet, IsotropicGGX, Fresnel):
//
// Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice, this list of
//    conditions and the following disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its contributors may be used to
//    endorse or promote products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
module bsdf;

import rand;
import rt_types;

static const float kMinCosTheta = 1.0e-6;
// GGX widths below this are a mirror (a delta lobe): Falcor's threshold (roughness 0.08).
public static const float kMinGGXAlpha = 0.0064;

// ---- Random numbers ---------------------------------------------------------------------------

public interface ISampler {
    [mutating]
    float next();
}

public float2 next2<S : ISampler>(inout S s) {
    let a = s.next();
    let b = s.next();
    return float2(a, b);
}

// The path tracer's per-invocation PCG stream (rand.slang).
public struct PcgSampler : ISampler {
    [mutating]
    public float next() {
        return randf();
    }
};

public uint ptHash(uint v) {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// The path's random number `dim` at vertex `vertex`: stateless, so random replay draws the same
// numbers from another primary vertex.
public float pathRandom(uint seed, uint vertex, uint dim) {
    return float(ptHash(ptHash(ptHash(seed) + vertex * 0x9e3779b9u) + dim * 0x85ebca6bu) >> 8) / 16777216.0;
}

// ReSTIR PT's numbers at one vertex: the search and every replay of it draw the same sequence.
public struct PathSampler : ISampler {
    public uint seed;
    public uint vertex;
    public uint dim;

    public __init(uint seed, uint vertex) {
        this.seed = seed;
        this.vertex = vertex;
        this.dim = 0u;
    }

    [mutating]
    public float next() {
        let u = pathRandom(seed, vertex, dim);
        dim++;
        return u;
    }
};

// ---- Microfacet and Fresnel helpers -----------------------------------------------------------

[Differentiable]
float3 evalFresnelSchlick(float3 f0, float f90, float cosTheta) {
    return f0 + (f90 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

[Differentiable]
float evalFresnelSchlick(float f0, float f90, float cosTheta) {
    return f0 + (f90 - f0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

[Differentiable]
float evalNdfGGX(float alpha, float cosTheta) {
    let a2 = alpha * alpha;
    let d = (cosTheta * a2 - cosTheta) * cosTheta + 1.0;
    return a2 / (d * d * PI);
}

[Differentiable]
float evalG1GGX(float alphaSqr, float cosTheta) {
    if (cosTheta <= 0.0) {
        return 0.0;
    }
    let c2 = cosTheta * cosTheta;
    let t2 = max(1.0 - c2, 0.0) / c2;
    return 2.0 / (1.0 + sqrt(1.0 + alphaSqr * t2));
}

[Differentiable]
float evalLambdaGGX(float alphaSqr, float cosTheta) {
    if (cosTheta <= 0.0) {
        return 0.0;
    }
    let c2 = cosTheta * cosTheta;
    let t2 = max(1.0 - c2, 0.0) / c2;
    return 0.5 * (-1.0 + sqrt(1.0 + alphaSqr * t2));
}

// Height-correlated Smith masking-shadowing (Heitz 2014, eq. 99).
[Differentiable]
float evalMaskingSmithGGXCorrelated(float alpha, float cosThetaI, float cosThetaO) {
    let a2 = alpha * alpha;
    return 1.0 / (1.0 + evalLambdaGGX(a2, cosThetaI) + evalLambdaGGX(a2, cosThetaO));
}

// The GGX layer's directional albedo (height-correlated Smith), fitted: Ray Tracing Gems ch. 32,
// eq. 4 (Falcor's approxSpecularIntegralGGX).
[Differentiable]
float3 approxSpecularIntegralGGX(float3 f0, float alpha, float cosTheta) {
    let c = abs(cosTheta);
    let X = float4(1.0, c, c * c, c * c * c);
    let Y = float4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    let M1 = float2x2(0.995367, -1.38839, -0.24751, 1.97442);
    let M2 = float3x3(1.0, 2.68132, 52.366, 16.0932, -3.98452, 59.3013, -5.18731, 255.259, 2544.07);
    let M3 = float2x2(-0.0564526, 3.82901, 16.91, -11.0303);
    let M4 = float3x3(1.0, 4.11118, -1.37886, 19.3254, -28.9947, 16.9514, 0.545386, 96.0994, -79.4492);
    var bias = dot(mul(M1, X.xy), Y.xy) / dot(mul(M2, X.xyw), Y.xyw);
    let scale = dot(mul(M3, X.xy), Y.xy) / dot(mul(M4, X.xzw), Y.xyw);
    // Falcor's hack for a reflectance of 0.
    bias *= saturate(dot(f0, float3(1.0 / 3.0)) * 50.0);
    return f0 * max(0.0, scale) + max(0.0, bias);
}

// D_V(h) = G1(wi) D(h) max(0, wi.h) / wi.z.
float evalPdfGGX_VNDF(float alpha, float3 wi, float3 h) {
    return evalG1GGX(alpha * alpha, wi.z) * evalNdfGGX(alpha, h.z) * max(0.0, dot(wi, h)) / wi.z;
}

// Visible normals by spherical caps (Dupuy and Benyoub 2023).
float3 sampleGGX_VNDF(float alpha, float3 wi, float2 u, out float pdf) {
    let vh = normalize(float3(alpha * wi.x, alpha * wi.y, wi.z));
    let phi = 2.0 * PI * u.x;
    let z = (1.0 - u.y) * (1.0 + vh.z) - vh.z;
    let sin_theta = sqrt(saturate(1.0 - z * z));
    let nh = float3(sin_theta * cos(phi), sin_theta * sin(phi), z) + vh;
    let h = normalize(float3(alpha * nh.x, alpha * nh.y, nh.z));
    pdf = evalPdfGGX_VNDF(alpha, wi, h);
    return h;
}

// Cosine-weighted hemisphere by the concentric disc map; pdf = cos / pi.
float3 sampleCosineHemisphere(float2 u, out float pdf) {
    let p = 2.0 * u - 1.0;
    float r, phi;
    if (p.x == 0.0 && p.y == 0.0) {
        r = 0.0;
        phi = 0.0;
    } else if (abs(p.x) > abs(p.y)) {
        r = p.x;
        phi = (PI / 4.0) * (p.y / p.x);
    } else {
        r = p.y;
        phi = (PI / 2.0) - (PI / 4.0) * (p.x / p.y);
    }
    let d = r * float2(cos(phi), sin(phi));
    let z = sqrt(max(0.0, 1.0 - dot(d, d)));
    pdf = z / PI;
    return float3(d, z);
}

// ---- Lobes (Falcor's IBSDF) -------------------------------------------------------------------

public interface IBSDF {
    // f(wi, wo) |cos(wo)|.
    float3 eval(float3 wi, float3 wo);
    // A direction wo, its solid-angle pdf (0 for a delta) and weight f |cos| / pdf.
    bool sample<S : ISampler>(float3 wi, out float3 wo, out float pdf, out float3 weight, inout S sg);
    float evalPdf(float3 wi, float3 wo);
}

// Frostbite's diffuse: Disney's with an energy normalization (Lagarde and de Rousiers 2014).
public struct FrostbiteDiffuseBRDF : IBSDF, IDifferentiable {
    public float3 albedo;
    public float roughness;

    [Differentiable]
    public float3 eval(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta) {
            return float3(0.0);
        }
        return weight(wi, wo) * (wo.z / PI);
    }

    public bool sample<S : ISampler>(float3 wi, out float3 wo, out float pdf, out float3 w, inout S sg) {
        wo = sampleCosineHemisphere(next2(sg), pdf);
        if (min(wi.z, wo.z) < kMinCosTheta) {
            w = float3(0.0);
            return false;
        }
        w = weight(wi, wo);
        return true;
    }

    public float evalPdf(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta) {
            return 0.0;
        }
        return wo.z / PI;
    }

    // f(wi, wo) pi.
    [Differentiable]
    float3 weight(float3 wi, float3 wo) {
        let h = normalize(wi + wo);
        let wo_h = dot(wo, h);
        let energy_bias = lerp(0.0, 0.5, roughness);
        let energy_factor = lerp(1.0, 1.0 / 1.51, roughness);
        let fd90 = energy_bias + 2.0 * wo_h * wo_h * roughness;
        return albedo * evalFresnelSchlick(1.0, fd90, wi.z) * evalFresnelSchlick(1.0, fd90, wo.z)
            * energy_factor;
    }
};

// Lambertian diffuse reflection. `roughness` is unused (it keeps the diffuse lobes' fields alike).
public struct LambertDiffuseBRDF : IBSDF, IDifferentiable {
    public float3 albedo;
    public float roughness;

    [Differentiable]
    public float3 eval(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta) {
            return float3(0.0);
        }
        return albedo * (wo.z / PI);
    }

    public bool sample<S : ISampler>(float3 wi, out float3 wo, out float pdf, out float3 w, inout S sg) {
        wo = sampleCosineHemisphere(next2(sg), pdf);
        if (min(wi.z, wo.z) < kMinCosTheta) {
            w = float3(0.0);
            return false;
        }
        w = albedo;
        return true;
    }

    public float evalPdf(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta) {
            return 0.0;
        }
        return wo.z / PI;
    }
};

// Falcor's DiffuseBrdf choice (BSDFConfig.slangh: Lambert, Disney or Frostbite).
public typealias DiffuseBRDF = LambertDiffuseBRDF;

// Lambertian diffuse transmission: a thin translucent sheet (a leaf).
public struct LambertDiffuseBTDF : IBSDF, IDifferentiable {
    public float3 albedo;

    [Differentiable]
    public float3 eval(float3 wi, float3 wo) {
        if (min(wi.z, -wo.z) < kMinCosTheta) {
            return float3(0.0);
        }
        return albedo * (-wo.z / PI);
    }

    public bool sample<S : ISampler>(float3 wi, out float3 wo, out float pdf, out float3 w, inout S sg) {
        wo = sampleCosineHemisphere(next2(sg), pdf);
        wo.z = -wo.z;
        if (min(wi.z, -wo.z) < kMinCosTheta) {
            w = float3(0.0);
            return false;
        }
        w = albedo;
        return true;
    }

    public float evalPdf(float3 wi, float3 wo) {
        if (min(wi.z, -wo.z) < kMinCosTheta) {
            return 0.0;
        }
        return -wo.z / PI;
    }
};

// GGX microfacet reflection; alpha 0 is a mirror (a delta: no eval, pdf 0).
public struct SpecularMicrofacetBRDF : IBSDF, IDifferentiable {
    public float3 albedo;
    public float alpha;

    [Differentiable]
    public float3 eval(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta || alpha == 0.0) {
            return float3(0.0);
        }
        let h = normalize(wi + wo);
        let D = evalNdfGGX(alpha, h.z);
        let G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        let F = evalFresnelSchlick(albedo, 1.0, dot(wi, h));
        return F * D * G * 0.25 / wi.z;
    }

    public bool sample<S : ISampler>(float3 wi, out float3 wo, out float pdf, out float3 w, inout S sg) {
        wo = float3(0.0);
        w = float3(0.0);
        pdf = 0.0;
        if (wi.z < kMinCosTheta) {
            return false;
        }
        let u = next2(sg);
        if (alpha == 0.0) {
            wo = float3(-wi.x, -wi.y, wi.z);
            w = evalFresnelSchlick(albedo, 1.0, wi.z);
            return true;
        }
        let h = sampleGGX_VNDF(alpha, wi, u, pdf);
        let wi_h = dot(wi, h);
        wo = 2.0 * wi_h * h - wi;
        if (wo.z < kMinCosTheta) {
            return false;
        }
        let G = evalMaskingSmithGGXCorrelated(alpha, wi.z, wo.z);
        let G_over_G1wo = G * (1.0 + evalLambdaGGX(alpha * alpha, wi.z));
        pdf /= 4.0 * wi_h;
        w = evalFresnelSchlick(albedo, 1.0, wi_h) * G_over_G1wo;
        return true;
    }

    public float evalPdf(float3 wi, float3 wo) {
        if (min(wi.z, wo.z) < kMinCosTheta || alpha == 0.0) {
            return 0.0;
        }
        let h = normalize(wi + wo);
        return evalPdfGGX_VNDF(alpha, wi, h) / (4.0 * dot(wi, h));
    }
};

// ---- Materials --------------------------------------------------------------------------------

// What a hit reads of its material (the payload's or a packed record's).
public struct SurfaceMaterial : IDifferentiable {
    public float3 albedo;
    public float metallic;
    public float roughness;
};

public SurfaceMaterial surfaceMaterial(float3 albedo, float metallic, float roughness) {
    SurfaceMaterial m;
    m.albedo = albedo;
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

// Thin layers over the base (hit_payload.slang packLobes): diffuse translucency, a clear coat.
public struct Lobes : IDifferentiable {
    public float transmission;
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

// Which lobe a sample came from (SurfaceBSDF.sample), and whether it was a delta.
public static const uint LOBE_DIFFUSE = 0u;
public static const uint LOBE_TRANSLUCENT = 1u;
public static const uint LOBE_SPECULAR = 2u;
public static const uint LOBE_COAT = 3u;
public static const uint LOBE_DELTA = 4u;

// Falcor's standard BSDF (diffuse and translucent mixed by `transmission`, under the specular
// layer by its albedo, specular added) under
// aurora's clear coat (the base weighted by 1 - clearcoat F_coat(wi), the coat's GGX added), in
// the shading frame of normal `n`, for the view `wi` it was made with.
public struct SurfaceBSDF : IDifferentiable {
    public DiffuseBRDF diffuse;
    public LambertDiffuseBTDF translucent;
    public SpecularMicrofacetBRDF specular;
    public SpecularMicrofacetBRDF coat;
    public float diff_trans;
    public float clearcoat;
    no_diff public float3 t;
    no_diff public float3 b;
    no_diff public float3 n;
    no_diff public float3 wi;
    // Lobe selection by albedo (Falcor's), the coat's share first.
    no_diff float p_diffuse = 0.0;
    no_diff float p_translucent = 0.0;
    no_diff float p_specular = 0.0;
    no_diff float p_coat = 0.0;

    public float3 toLocal(float3 v) {
        return float3(dot(v, t), dot(v, b), dot(v, n));
    }

    public float3 toWorld(float3 v) {
        return v.x * t + v.y * b + v.z * n;
    }

    // What the specular layer leaves the diffuse lobes beneath it for the view: 1 - its albedo
    // (Cycles' Principled BSDF layers the same way; Falcor's standard BSDF only adds them).
    [Differentiable]
    float3 underSpecular() {
        return 1.0 - approxSpecularIntegralGGX(specular.albedo, specular.alpha, detach(wi.z));
    }

    // The base's weight under the coat for the view.
    [Differentiable]
    float baseWeight() {
        return 1.0 - clearcoat * evalFresnelSchlick(0.04, 1.0, detach(wi.z));
    }

    // f(v, l) |n.l| in world space, every lobe.
    [Differentiable]
    public float3 eval(no_diff float3 l) {
        let wo = no_diff toLocal(l);
        var f = ((1.0 - diff_trans) * diffuse.eval(wi, wo) + diff_trans * translucent.eval(wi, wo))
            * underSpecular() + specular.eval(wi, wo);
        if (clearcoat > 0.0) {
            f = f * baseWeight() + clearcoat * coat.eval(wi, wo);
        }
        return f;
    }

    // The sampling pdf of `l` (solid angle; 0 for what only a delta reaches).
    public float pdf(float3 l) {
        let wo = toLocal(l);
        var p = 0.0;
        if (p_diffuse > 0.0) {
            p += p_diffuse * diffuse.evalPdf(wi, wo);
        }
        if (p_translucent > 0.0) {
            p += p_translucent * translucent.evalPdf(wi, wo);
        }
        if (p_specular > 0.0) {
            p += p_specular * specular.evalPdf(wi, wo);
        }
        if (p_coat > 0.0) {
            p += p_coat * coat.evalPdf(wi, wo);
        }
        return p;
    }

    // A direction `l` from one lobe, picked by albedo: its weight f |cos| / pdf over the whole
    // mixture (the pdf every lobe gives `l`), or for a delta the lobe's own. One number picks the
    // lobe, two draw the direction.
    public bool sample<S : ISampler>(inout S sg, out float3 l, out float pdf, out float3 weight,
                                     out uint lobe) {
        l = float3(0.0);
        pdf = 0.0;
        weight = float3(0.0);
        lobe = LOBE_DIFFUSE;
        let u = sg.next();
        float3 wo;
        float lobe_pdf;
        float3 w;
        bool ok;
        float scale;
        float p;
        if (u < p_diffuse) {
            ok = diffuse.sample(wi, wo, lobe_pdf, w, sg);
            w *= underSpecular();
            scale = (1.0 - diff_trans) * baseWeight();
            p = p_diffuse;
        } else if (u < p_diffuse + p_translucent) {
            ok = translucent.sample(wi, wo, lobe_pdf, w, sg);
            w *= underSpecular();
            scale = diff_trans * baseWeight();
            p = p_translucent;
            lobe = LOBE_TRANSLUCENT;
        } else if (u < p_diffuse + p_translucent + p_specular) {
            ok = specular.sample(wi, wo, lobe_pdf, w, sg);
            scale = baseWeight();
            p = p_specular;
            lobe = LOBE_SPECULAR;
        } else {
            ok = coat.sample(wi, wo, lobe_pdf, w, sg);
            scale = clearcoat;
            p = p_coat;
            lobe = LOBE_COAT;
        }
        if (!ok || p <= 0.0) {
            return false;
        }
        l = toWorld(wo);
        if (lobe_pdf == 0.0) {
            weight = w * scale / p;
            lobe |= LOBE_DELTA;
            return true;
        }
        pdf = this.pdf(l);
        if (pdf <= 0.0) {
            return false;
        }
        weight = eval(l) / pdf;
        return true;
    }
};

// Builds `n`'s shading frame (Frisvad / Duff et al. 2017).
void frame(float3 n, out float3 t, out float3 b) {
    let s = n.z >= 0.0 ? 1.0 : -1.0;
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// The BSDF of a surface with shading normal `n` seen from `v` (unit, toward the viewer).
public SurfaceBSDF surfaceBSDF(float3 v, float3 n, SurfaceMaterial mat, Lobes lob) {
    SurfaceBSDF s;
    s.n = n;
    frame(n, s.t, s.b);
    s.wi = s.toLocal(v);
    let diffuse_albedo = mat.albedo * (1.0 - mat.metallic);
    s.diffuse.albedo = diffuse_albedo;
    s.diffuse.roughness = mat.roughness;
    s.translucent.albedo = diffuse_albedo;
    var alpha = mat.roughness * mat.roughness;
    if (alpha < kMinGGXAlpha) {
        alpha = 0.0;
    }
    s.specular.albedo = lerp(float3(0.04), mat.albedo, mat.metallic);
    s.specular.alpha = alpha;
    s.coat.albedo = float3(0.04);
    s.coat.alpha = lob.coat_alpha < kMinGGXAlpha ? 0.0 : lob.coat_alpha;
    s.diff_trans = lob.transmission;
    s.clearcoat = lob.clearcoat;
    let diffuse_weight = luminance(diffuse_albedo * s.underSpecular());
    let specular_weight = luminance(evalFresnelSchlick(s.specular.albedo, 1.0, s.wi.z));
    s.p_diffuse = diffuse_weight * (1.0 - s.diff_trans);
    s.p_translucent = diffuse_weight * s.diff_trans;
    s.p_specular = specular_weight;
    let norm = s.p_diffuse + s.p_translucent + s.p_specular;
    let q = lob.clearcoat * evalFresnelSchlick(0.04, 1.0, saturate(s.wi.z));
    let k = norm > 0.0 ? (1.0 - q) / norm : 0.0;
    s.p_diffuse *= k;
    s.p_translucent *= k;
    s.p_specular *= k;
    s.p_coat = norm > 0.0 ? q : (q > 0.0 ? 1.0 : 0.0);
    return s;
}

// f(v, l) |n.l| of a surface (every lobe): the shorthand for one evaluation.
public float3 surfaceEval(float3 v, float3 n, float3 l, SurfaceMaterial mat, Lobes lob) {
    return surfaceBSDF(v, n, mat, lob).eval(l);
}

// The sampling pdf of `l` at a surface (solid angle), the MIS partner of light sampling.
public float surfacePdf(float3 v, float3 n, float3 l, SurfaceMaterial mat, Lobes lob) {
    return surfaceBSDF(v, n, mat, lob).pdf(l);
}

// ---- Shared small helpers ---------------------------------------------------------------------

public float luma(float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

// A direction in the cone of half-angle acos(cos_r) around `dir`, uniform in solid angle.
public float3 sampleCone(float3 dir, float cos_r, float r1, float r2) {
    let cos_t = 1.0 - r1 * (1.0 - cos_r);
    let sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    let phi = 2.0 * PI * r2;
    float3 t, b;
    frame(dir, t, b);
    return normalize(t * (sin_t * cos(phi)) + b * (sin_t * sin(phi)) + dir * cos_t);
}
