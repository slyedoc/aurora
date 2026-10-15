// The composite's prepass (src/post_prep.rs): per view, after the output's pyramid is built, what
// quad.slang would otherwise gather at every output pixel, once at a coarser grid.
//
//   bloom_prep  bloom at the pyramid's level 1: levels 1-6, each a 9-tap tent a texel apart,
//               weighted 0.6x per level, into an RGBA16F texture (quad.slang filters it).
//   key_prep    local tone mapping's surroundings: a bilateral grid of log luminance (cells of
//               32 output pixels x 16 one-stop bins) into an RGBA16F texture (quad.slang
//               `localGain` reads it at each pixel's position and level).
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

// Local tone mapping's surroundings as a bilateral grid (Chen, Paris & Durand 2007; UE's local
// exposure): per cell of GRID_CELL output pixels and per bin of GRID_STOPS of log luminance, the
// mean log luminance of the pixels that fall in it, blurred over the 3x3 cells around and the
// neighbouring bins. The composite reads it at a pixel's position AND its own level, so a wall
// beside a window averages wall and the window averages window: no halo at the edge (a blur of
// the image alone pulled the wall down beside the window and lit a rim along it). Laid out as
// `GRID_BINS` tiles side by side, each `key_width / GRID_BINS` cells across: rg = (mean x
// density, density), so bilinear filtering blends the two consistently. Matches quad.slang.
static const uint GRID_CELL = 32;
static const uint GRID_BINS = 16;
static const float GRID_LO = -10.0;
static const float GRID_STOPS = 1.0;

[shader("compute")]
[numthreads(64, 1, 1)]
void key_prep(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x + push.base;
    let gw = push.key_width / GRID_BINS;
    let gh = push.key_height;
    if (i >= gw * gh) {
        return;
    }
    let cell = int2(int(i % gw), int(i / gw));
    uint pw, ph, levels;
    prep_in[0].GetDimensions(0, pw, ph, levels);
    // Level 2 of the pyramid (level 0 is half the output): a texel per 8 output pixels, 4 per
    // cell side.
    let level = min(2.0, float(levels) - 1.0);
    let lw = max(int(pw) >> int(level), 1);
    let lh = max(int(ph) >> int(level), 1);
    let per = float(lw) / float(gw);
    let look = push.display_exposure > 0.0 ? push.display_exposure : push.ae.exposure;
    let scale = look / max(push.ae.input_exposure, 1.0e-12);
    float sum[GRID_BINS];
    float wsum[GRID_BINS];
    for (uint b = 0u; b < GRID_BINS; b++) {
        sum[b] = 0.0;
        wsum[b] = 0.0;
    }
    let center = (float2(cell) + 0.5) * per;
    let lo = int2(floor(center - 1.5 * per));
    let hi = int2(ceil(center + 1.5 * per));
    for (int y = max(lo.y, 0); y < min(hi.y, lh); y++) {
        for (int x = max(lo.x, 0); x < min(hi.x, lw); x++) {
            let t = (float2(float(x), float(y)) + 0.5);
            // Tent over 1.5 cells: neighbouring cells overlap, so the grid reads smooth.
            let d = abs(t - center) / (1.5 * per);
            let ws = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
            if (ws <= 0.0) {
                continue;
            }
            let uv = t / float2(float(lw), float(lh));
            let v = pyramidLog(uv, level, scale);
            // Tent over the two nearest bins, and half as much into the next ones out.
            let p = clamp((v - GRID_LO) / GRID_STOPS - 0.5, 0.0, float(GRID_BINS - 1u));
            let b0 = int(floor(p));
            let f = p - float(b0);
            for (int k = -1; k <= 2; k++) {
                let b = b0 + k;
                if (b < 0 || b >= int(GRID_BINS)) {
                    continue;
                }
                let wb = k == 0 ? 1.0 - f : (k == 1 ? f : (k == -1 ? 0.5 * (1.0 - f) : 0.5 * f));
                sum[b] += ws * wb * v;
                wsum[b] += ws * wb;
            }
        }
    }
    var total = 0.0;
    for (uint b = 0u; b < GRID_BINS; b++) {
        total += wsum[b];
    }
    for (uint b = 0u; b < GRID_BINS; b++) {
        let density = wsum[b] / max(total, 1.0e-6);
        let mean = wsum[b] > 0.0 ? sum[b] / wsum[b] : 0.0;
        prep_out[0][uint2(b * gw + uint(cell.x), uint(cell.y))] = float4(mean * density, density, 0.0, 0.0);
    }
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
