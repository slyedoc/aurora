// The composite's prepass (src/post_prep.rs): per view, after the output's pyramid is built, what
// quad.slang would otherwise gather at every output pixel, once at a coarser grid.
//
//   bloom_prep  bloom at the pyramid's level 1: levels 1-6, each a 9-tap tent a texel apart,
//               weighted 0.6x per level, into an RGBA16F texture (quad.slang filters it).
//   key_prep    local tone mapping's surroundings at the pyramid's level 0: the three scales'
//               smoothed log luminance into an RGBA16F texture's rgb (quad.slang `localGain`
//               blends them per pixel).
//   range_prep  at render resolution: the 3x3 texel range (min, max) of the fog veil and of the
//               sky mask, which clamp the composite's histories.

// `AeData` in rt_types.slang.
struct AeData {
    float ev;
    float exposure;
    float input_ev;
    float input_exposure;
};

struct PrepPush {
    uint bloom_width;
    uint bloom_height;
    uint base;
    uint inputs;
    uint key_width;
    uint key_height;
    uint range_width;
    uint range_height;
    uint4* fog_range;
    uint* sky_range;
    AeData* ae;
    // As QuadPush::display_exposure: 0 = the metered exposure.
    float display_exposure;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<PrepPush> push;

// 0: the output's mip pyramid; 1 the camera ray's fog; 2 the depth guide.
[[vk::binding(0, 0)]] Sampler2D prep_in[3];
// 0: local tone mapping's keys; 1: bloom (GENERAL; the composite samples them).
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> prep_out[2];

// Bloom: the lens scattering a share of every pixel's light over its surroundings. Each level a
// 9-tap tent a texel apart so the box-filtered levels upsample round and smooth, weighted 0.6x
// per level: a bright core beside a light and a wide faint skirt. Clamped to half-float range
// (the sun's disc overflows the pyramid).
[shader("compute")]
[numthreads(64, 1, 1)]
void bloom_prep(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x + push.base;
    let w = push.bloom_width;
    if (i >= w * push.bloom_height) {
        return;
    }
    let uv = (float2(float(i % w), float(i / w)) + 0.5) / float2(float(w), float(push.bloom_height));
    uint pw, ph, levels;
    prep_in[0].GetDimensions(0, pw, ph, levels);
    let pyramid = float2(float(pw), float(ph));
    let top = min(6, int(levels) - 1);
    var sum = float3(0.0);
    var n = 0.0;
    for (int l = 1; l <= top; l++) {
        let t = exp2(float(l)) / pyramid;
        let level = float(l);
        var s = float3(0.0);
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                let k = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
                s += prep_in[0].SampleLevel(uv + float2(float(x), float(y)) * t, level).rgb * k;
            }
        }
        let wl = pow(0.6, float(l - 1));
        sum += min(s / 16.0, float3(65504.0)) * wl;
        n += wl;
    }
    prep_out[1][uint2(i % w, i / w)] = float4(n > 0.0 ? sum / n : float3(0.0), 1.0);
}

uint packHalf2(float a, float b) {
    return f32tof16(a) | (f32tof16(b) << 16);
}

// quad.slang `pyramidLog`, at the look's exposure over the one the raygen baked in.
float pyramidLog(float2 uv, float level, float scale) {
    let c = prep_in[0].SampleLevel(uv, level).rgb * scale;
    return log2(max(dot(c, float3(0.2125, 0.7154, 0.0721)), 1.0e-4));
}

// A region's level at pyramid `level`, smoothed: four reads half a texel apart hide the
// bilinear diamonds of a coarse level.
float pyramidKey(float2 uv, float level, float scale, float2 size) {
    let t = exp2(level) / size * 0.5;
    return 0.25 * (pyramidLog(uv + float2(-t.x, -t.y), level, scale)
                 + pyramidLog(uv + float2(t.x, -t.y), level, scale)
                 + pyramidLog(uv + float2(-t.x, t.y), level, scale)
                 + pyramidLog(uv + float2(t.x, t.y), level, scale));
}

// Level 0 is half the output: the level whose texel spans ~3 % of the height, one 1.5 levels
// finer, and level 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void key_prep(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x + push.base;
    let w = push.key_width;
    if (i >= w * push.key_height) {
        return;
    }
    let uv = (float2(float(i % w), float(i / w)) + 0.5) / float2(float(w), float(push.key_height));
    uint pw, ph, levels;
    prep_in[0].GetDimensions(0, pw, ph, levels);
    let pyramid = float2(float(pw), float(ph));
    let look = push.display_exposure > 0.0 ? push.display_exposure : push.ae.exposure;
    let scale = look / max(push.ae.input_exposure, 1.0e-12);
    let top = float(levels) - 1.0;
    let coarse = clamp(log2(pyramid.y * 0.03), 0.0, top);
    var k = float3(0.0);
    for (int l = 0; l < 3; l++) {
        let level = l == 0 ? 0.0 : max(coarse - float(2 - l) * 1.5, 0.0);
        k[l] = pyramidKey(uv, level, scale, pyramid);
    }
    prep_out[0][uint2(i % w, i / w)] = float4(k, 0.0);
}

// quad.slang `skyAt`: the sky exemption (depth guide, faded in from 300 m to 3 km).
float skyOf(float depth) {
    return saturate(log2(depth / 300.0) / log2(10.0));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void range_prep(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x + push.base;
    let w = push.range_width;
    let h = push.range_height;
    if (i >= w * h) {
        return;
    }
    let p = int2(int(i % w), int(i / w));
    let last = int2(int(w), int(h)) - 1;
    var fog_lo = float4(65504.0);
    var fog_hi = float4(-65504.0);
    var sky_lo = 1.0;
    var sky_hi = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            let q = int3(clamp(p + int2(x, y), int2(0), last), 0);
            let f = prep_in[1].Load(q);
            fog_lo = min(fog_lo, f);
            fog_hi = max(fog_hi, f);
            let sky = skyOf(prep_in[2].Load(q).r);
            sky_lo = min(sky_lo, sky);
            sky_hi = max(sky_hi, sky);
        }
    }
    fog_lo = clamp(fog_lo, -65504.0, 65504.0);
    fog_hi = clamp(fog_hi, -65504.0, 65504.0);
    push.fog_range[i] = uint4(packHalf2(fog_lo.r, fog_lo.g), packHalf2(fog_lo.b, fog_lo.a),
                              packHalf2(fog_hi.r, fog_hi.g), packHalf2(fog_hi.b, fog_hi.a));
    push.sky_range[i] = packHalf2(sky_lo, sky_hi);
}
