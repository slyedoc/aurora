// tests/bsdf_check.rs: the surface BSDF (bsdf.slang) against itself. Per material configuration,
// 64 threads each estimate (1) the directional albedo from the sampler's weights, (2) the same
// integral of eval by uniform sphere sampling, (3) the integral of the pdf, which must equal the share of samples that succeed (a rough
// lobe's draws below the horizon are rejected); thread 0 also writes
// eval's derivatives (bwd_diff) beside finite differences, each sample's weight against eval / pdf, and whether a PathSampler replay
// repeats its sample.
import bsdf;
import rt_types;

struct Push {
    float* out;
    uint base;
    uint samples;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

struct Lcg : ISampler {
    uint s;
    [mutating]
    float next() {
        s = s * 747796405u + 2891336453u;
        let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        return float(((w >> 22u) ^ w) >> 8) * (1.0 / 16777216.0);
    }
};

static const uint CONFIGS = 64u;
static const uint OUT_PER_THREAD = 5u;

// Configuration c: roughness x metallic x thin layer x view.
void config(uint c, out SurfaceMaterial m, out Lobes lob, out float3 v) {
    let roughness = float[4](0.3, 0.5, 0.7, 1.0)[c % 4u];
    let metallic = float((c / 4u) % 2u);
    let layer = (c / 8u) % 4u;
    let cos_v = float[2](0.9, 0.35)[(c / 32u) % 2u];
    // Glass: entering from air at the first view, leaving from inside at the second (total
    // internal reflection past 42 degrees).
    let glass = layer == 3u;
    m = surfaceMaterial(float3(0.8, 0.5, 0.3), glass ? 0.0 : metallic, roughness,
                        glass ? 1.0 : 0.0, glass ? ((c / 32u) % 2u == 0u ? 1.0 / 1.5 : 1.5) : 1.0);
    lob.transmission = layer == 1u ? 0.5 : 0.0;
    lob.clearcoat = layer == 2u ? 1.0 : 0.0;
    lob.coat_alpha = 0.25;
    v = float3(sqrt(1.0 - cos_v * cos_v), 0.0, cos_v);
}

float lum(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

float evalAt(SurfaceMaterial m, Lobes lob, float3 v, float3 l) {
    return lum(surfaceBSDF(v, float3(0.0, 0.0, 1.0), m, lob).eval(l));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bsdf_check(uint3 id: SV_DispatchThreadID) {
    let i = id.x + push.base;
    let c = i / 64u;
    let t = i % 64u;
    if (c >= CONFIGS) {
        return;
    }
    SurfaceMaterial m;
    Lobes lob;
    float3 v;
    config(c, m, lob, v);
    let n = float3(0.0, 0.0, 1.0);
    let bs = surfaceBSDF(v, n, m, lob);
    var sg = Lcg(i * 9781u + 1u);
    var albedo_sampled = 0.0;
    var albedo_uniform = 0.0;
    var pdf_integral = 0.0;
    var success = 0.0;
    // Each sample's weight against f |cos| / pdf from eval and pdf themselves: the glass lobe's
    // weight is the lobe's own, which equals the mixture's when the surface is all glass.
    var worst = 0.0;
    for (uint k = 0u; k < push.samples; k++) {
        float3 l;
        float pdf;
        float3 w;
        uint lobe;
        if (bs.sample(sg, l, pdf, w, lobe)) {
            albedo_sampled += lum(w);
            success += (lobe & LOBE_DELTA) == 0u ? 1.0 : 0.0;
            if ((lobe & LOBE_DELTA) == 0u) {
                let glass_lobe = (lobe & LOBE_MASK) == LOBE_GLASS;
                if (!glass_lobe || m.transmission >= 1.0) {
                    let f = glass_lobe ? bs.eval(l) : bs.evalNee(l);
                    let q = glass_lobe ? bs.pdf(l) : bs.pdfNee(l);
                    let ref = lum(f) / q;
                    worst = max(worst, abs(lum(w) - ref) / max(ref, 1.0e-3));
                }
            }
        }
        // Uniform on the sphere: pdf 1 / 4 pi.
        let z = sg.next() * 2.0 - 1.0;
        let phi = sg.next() * 2.0 * PI;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let u = float3(r * cos(phi), r * sin(phi), z);
        albedo_uniform += lum(bs.eval(u)) * 4.0 * PI;
        pdf_integral += bs.pdf(u) * 4.0 * PI;
    }
    let at = push.out + (c * 64u + t) * OUT_PER_THREAD;
    at[0] = albedo_sampled / float(push.samples);
    at[1] = albedo_uniform / float(push.samples);
    at[2] = pdf_integral / float(push.samples);
    at[3] = success / float(push.samples);
    at[4] = worst;
    if (t == 0u) {
        // Derivatives of eval (luminance) at one direction: roughness and albedo.r.
        let l = normalize(float3(-0.3, 0.2, 0.8));
        var dp = diffPair(bs);
        bwd_diff(SurfaceBSDF.eval)(dp, l, float3(0.2126, 0.7152, 0.0722));
        let d_rough = dp.d.diffuse.roughness;
        let d_albedo = dp.d.diffuse.albedo.r + dp.d.translucent.albedo.r;
        let h = 1.0e-3;
        var bp = bs;
        var bm = bs;
        bp.diffuse.roughness += h;
        bm.diffuse.roughness -= h;
        let fd_rough = (lum(bp.eval(l)) - lum(bm.eval(l))) / (2.0 * h);
        bp = bs;
        bm = bs;
        bp.diffuse.albedo.r += h;
        bm.diffuse.albedo.r -= h;
        bp.translucent.albedo.r += h;
        bm.translucent.albedo.r -= h;
        let fd_albedo = (lum(bp.eval(l)) - lum(bm.eval(l))) / (2.0 * h);
        // Replay: the same seed and vertex give the same direction.
        var a = PathSampler(77u + c, 2u);
        var b = PathSampler(77u + c, 2u);
        float3 la, lb, wa, wb;
        float pa, pb;
        uint oa, ob;
        bs.sample(a, la, pa, wa, oa);
        bs.sample(b, lb, pb, wb, ob);
        let tail = push.out + CONFIGS * 64u * OUT_PER_THREAD + c * 8u;
        tail[0] = d_rough;
        tail[1] = fd_rough;
        tail[2] = d_albedo;
        tail[3] = fd_albedo;
        tail[4] = all(la == lb) && all(wa == wb) && oa == ob ? 1.0 : 0.0;
        tail[5] = m.roughness;
        tail[6] = m.metallic;
        tail[7] = float((c / 8u) % 4u);
    }
}
