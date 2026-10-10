// Generator layers: zero's flat-world terrain function (hills, ridged mountain regions,
// detail) evaluated per texel of one page (aurora_landscape generator.rs). It covers every
// page, so an infinite (streaming) landscape is a generator plus whatever is edited on top.
import pages;
import noise;
import sphere;

// `GeneratorGpu` in generator.rs.
struct Genome {
    float seed;
    float base_height;
    float hills_frequency;
    float hills_amplitude;
    uint hills_octaves;
    float mountains_frequency;
    float mountains_amplitude;
    uint mountains_octaves;
    float mountains_warp;
    float mountains_mask_frequency;
    float mountains_mask_power;
    float detail_frequency;
    float detail_amplitude;
    uint detail_octaves;
    uint pad0;
    uint pad1;
};

struct GeneratorPush {
    float* page;
    Genome* genome;
    int page_x;
    int page_z;
    float texel;
    float opacity;
    uint blend;
    // Planet faces: noise at the sphere point, so faces agree along their edges (0 = flat).
    uint face;
    float radius;
    uint pad;
};
[[vk::push_constant]] GeneratorPush push;

// Octaves resolvable at a sample spacing: wavelengths under ~4 spacings would alias.
uint octaves_for_spacing(float base_freq, uint octaves, float spacing) {
    var f = base_freq;
    var k = 0u;
    while (k < octaves && f * spacing <= 1.5) {
        f *= 2.0;
        k += 1;
    }
    return max(k, 1u);
}

// A 3-D position (metres) scaled into a noise domain, offset by the seed.
float3 domain(float3 p, float frequency, float seed) {
    return p * frequency + float3(0.0, seed * 0.37, 0.0);
}

// Flat worlds sample the y = 0 plane; planets the sphere point.
float terrain_height(float3 xyz, Genome g, float spacing) {
    var h = g.base_height;
    let hk = octaves_for_spacing(g.hills_frequency, min(g.hills_octaves, 16u), spacing);
    h += g.hills_amplitude
        * fbm_simplex_3d_seeded(domain(xyz, g.hills_frequency, g.seed), int(hk), 2.0, 0.5, float3(g.seed));
    // Mountain regions: a smooth mask over a very low frequency field.
    let mask_n = fbm_simplex_3d_seeded(
        domain(xyz, g.mountains_mask_frequency, g.seed + 7.0), 3, 2.0, 0.5, float3(g.seed + 7.0));
    let mask = pow(smoothstep(0.05, 0.7, mask_n), g.mountains_mask_power);
    if (mask > 0.001) {
        let mk = octaves_for_spacing(g.mountains_frequency, min(g.mountains_octaves, 16u), spacing);
        let p = domain(xyz, g.mountains_frequency, g.seed + 3.0);
        let q = warp3d(p, int(min(mk, 4u)), 2.0, 0.5, g.mountains_warp, g.seed + 91.3);
        let r = ridged_multifractal_3d(q, int(mk), 2.1, 0.5, float3(g.seed + 3.0));
        h += mask * g.mountains_amplitude * clamp(r, 0.0, 1.0);
    }
    let dk = octaves_for_spacing(g.detail_frequency, min(g.detail_octaves, 8u), spacing);
    if (g.detail_amplitude > 0.0) {
        h += g.detail_amplitude
            * fbm_simplex_3d_seeded(domain(xyz, g.detail_frequency, g.seed + 11.0), int(dk), 2.0, 0.5, float3(g.seed + 11.0));
    }
    return h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void apply_generator(uint3 gid: SV_DispatchThreadID) {
    if (gid.x >= PAGE * PAGE) {
        return;
    }
    let t = int2(push.page_x * PAGE + int(gid.x % PAGE), push.page_z * PAGE + int(gid.x / PAGE));
    let xz = float2(t) * push.texel;
    let xyz = push.radius > 0.0 ? face_direction(push.face, push.radius, xz) * push.radius
                                : float3(xz.x, 0.0, xz.y);
    let v = terrain_height(xyz, *push.genome, push.texel);
    let h = push.page[gid.x];
    float r;
    switch (push.blend) {
        case 1: r = h + v; break;
        case 2: r = max(h, v); break;
        case 3: r = min(h, v); break;
        default: r = v; break;
    }
    push.page[gid.x] = lerp(h, r, push.opacity);
}
