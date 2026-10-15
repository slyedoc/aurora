// The composite: one full-screen triangle that re-exposes, tonemaps and vignettes the view's
// resolved image into the target, or shows one of its guides (AuroraDebugView).

// `AeData` in rt_types.slang: the exposure state from the metering (auto_exposure.slang). The
// raygen applies `input_exposure` (quantised, held still between rare whole-EV steps -- Ray
// Reconstruction's history needs a still input exposure); the blit maps it onto the smooth
// `exposure` (Auto look) or a fixed one.
struct AeData {
    float ev;
    float exposure;
    float input_ev;
    float input_exposure;
};

// `UniformData`'s vignette, as a float index; pinned by a test in ray_render_plugin.rs.
static const uint UNIFORM_VIGNETTE_FLOAT = 90;
// `UniformData`'s DLSS jitter (x, y; render pixels), as a float index.
static const uint UNIFORM_JITTER_FLOAT = 84;

// `PostProcessPushConstants` in post_process_filter.rs.
struct QuadPush {
    float* uniforms;
    AeData* ae;
    // 0 = follow the metering (Auto); else a fixed linear exposure, applied as a ratio
    // against the metered exposure the raygen baked in.
    float display_exposure;
    // AuroraDebugView (src/debug_view.rs): 0 = the output, else a guide visualisation.
    uint debug_view;
    // 1: a scene-radiance target (half float) -- physical radiance, no exposure or tonemap.
    uint scene_radiance;
    // 1: slot 8 holds the camera ray's fog (rgb light pre-exposed like the colour, a its
    // transmittance), composited here over Ray Reconstruction's output.
    uint fog;
    // Local tone mapping strength (RenderSettings::local_tonemap); 0 = off.
    float local_tonemap;
    // Bloom strength (RenderSettings::bloom); 0 = off.
    float bloom;
    // The view's auto-exposure meter, (meter, albedo) pairs at render resolution (null = none;
    // auto_exposure.rs): the raygen wrote its noisy meter and the surface's albedo (< 0 = not
    // metered); this pass replaces the meter with what Ray Reconstruction resolved.
    float* lum;
    uint lum_width;
    uint lum_height;
    // The Cost view's buffer at the meter's size (null = none; rt_types.slang `cost`).
    uint* cost;
    // Running averages at output resolution (null = none): local tone mapping's sky mask, last
    // frame's (read, reprojected) and this frame's (written), then the fog veil's likewise;
    // history_reset = 1 when last frame's are invalid.
    float* sky_prev;
    float* sky_cur;
    float4* fog_prev;
    float4* fog_cur;
    uint history_reset;
    uint pad_history;
    // `PrepFrame` in post_prep.rs, filled by post_prep.slang (null = none). Bloom, half4 per
    // texel at the pyramid's level 1; local tone mapping's three keys (half) at its level 0; the
    // fog veil's 3x3 range (half4 min, max) and the sky mask's (half min, max) per render pixel.
    uint2* bloom_buf;
    uint bloom_width;
    uint bloom_height;
    uint2* keys;
    uint key_width;
    uint key_height;
    uint4* fog_range;
    uint* sky_range;
    uint range_width;
    uint range_height;
};
[[vk::push_constant]] ConstantBuffer<QuadPush> push;

// 0: the DLSS output; 1..=7 the guide images (colour, normals+roughness, diffuse, specular,
// depth, specular hit distance, motion) for the debug views; 8 the camera ray's fog; 9 the
// output's mip pyramid (local tone mapping). Slots the renderer cannot fill yet alias the output.
[[vk::binding(0, 0)]] Sampler2D test[10];

// What is the same for every pixel, worked out by the three vertices instead of each pixel (the
// pass is texture-throughput bound): the pyramid's size and level count, the fog image's and
// the output's sizes, and the local tone mapping's frame average.
struct Frame {
    float2 pyramid;
    float levels;
    float frame_log;
    float2 fog;
    float2 output;
};

struct VertexOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
    [[vk::location(1)]] nointerpolation float4 pyramid : TEXCOORD1;
    [[vk::location(2)]] nointerpolation float4 sizes : TEXCOORD2;
};

// The exposure the output is shown at over the one the raygen baked in.
float lookScale() {
    let look = push.display_exposure > 0.0 ? push.display_exposure : push.ae.exposure;
    return look / max(push.ae.input_exposure, 1.0e-12);
}

[shader("vertex")]
VertexOut vertex(uint id : SV_VulkanVertexID) {
    let corners = float2[3](float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0));
    let p = corners[id];
    VertexOut o;
    o.position = float4(p, 0.0, 1.0);
    o.uv = (p + 1.0) * 0.5;
    uint w, h, levels, fw, fh, ow, oh;
    test[9].GetDimensions(0, w, h, levels);
    test[8].GetDimensions(fw, fh);
    test[0].GetDimensions(ow, oh);
    var frame_log = 0.0;
    if (push.local_tonemap > 0.0 && push.debug_view == 0u && push.fog != 0u) {
        frame_log = frameLog(lookScale(), h, levels);
    }
    o.pyramid = float4(float(w), float(h), float(levels), frame_log);
    o.sizes = float4(float(fw), float(fh), float(ow), float(oh));
    return o;
}

Frame frameOf(VertexOut v) {
    Frame f;
    f.pyramid = v.pyramid.xy;
    f.levels = v.pyramid.z;
    f.frame_log = v.pyramid.w;
    f.fog = v.sizes.xy;
    f.output = v.sizes.zw;
    return f;
}

// Scene luminance in NITS, false-coloured against the authoring reference bands in
// aurora_files/lighting_units.md. One decade per band, with the boundary darkened so the
// decades are countable rather than a smooth ramp you have to eyeball.
//
//   blue   <1        deep shadow          yellow  1e3..1e4  signage / overcast sky
//   cyan    1..10    dim interior         orange  1e4..1e5  LUMINAIRE surface
//   green  10..1e2                        red     1e5..1e6
//   lime   1e2..1e3  screens              white   >1e6      sun disc
float3 nitsFalseColor(float nits) {
    let t = log2(max(nits, 1.0e-4)) / log2(10.0);
    float3 c;
    if (t < 0.0) c = lerp(float3(0.02, 0.02, 0.10), float3(0.15, 0.15, 0.65), clamp(t + 1.0, 0.0, 1.0));
    else if (t < 1.0) c = lerp(float3(0.15, 0.15, 0.65), float3(0.00, 0.60, 0.90), t);
    else if (t < 2.0) c = lerp(float3(0.00, 0.60, 0.90), float3(0.00, 0.85, 0.40), t - 1.0);
    else if (t < 3.0) c = lerp(float3(0.00, 0.85, 0.40), float3(0.85, 0.95, 0.00), t - 2.0);
    else if (t < 4.0) c = lerp(float3(0.85, 0.95, 0.00), float3(1.00, 0.60, 0.00), t - 3.0);
    else if (t < 5.0) c = lerp(float3(1.00, 0.60, 0.00), float3(1.00, 0.20, 0.05), t - 4.0);
    else if (t < 6.0) c = lerp(float3(1.00, 0.20, 0.05), float3(1.00, 0.00, 0.60), t - 5.0);
    else c = lerp(float3(1.00, 0.00, 0.60), float3(1.00, 1.00, 1.00), clamp(t - 6.0, 0.0, 1.0));
    let f = frac(t);
    let edge = smoothstep(0.0, 0.035, f) * smoothstep(0.0, 0.035, 1.0 - f);
    return c * (0.4 + 0.6 * edge);
}

// Specular hit distance. 0 = not traced (rougher than RenderSettings::spec_hit_roughness: the reflection
// is pinned to the surface), dark purple; the far value (the probe missed) black; a distance
// grey like the depth view, white at the surface.
float3 specHitColor(float t) {
    if (t <= 0.0) {
        return float3(0.12, 0.05, 0.18);
    }
    return float3(exp2(-t / 32.0));
}

// Motion as flow colour: hue the direction, brightness the speed in render pixels per frame on a
// log scale (1 px dim, 15 px full), still pixels black. The guide holds UV units (raygen.slang).
float3 motionColor(float2 mv) {
    uint w, h;
    test[7].GetDimensions(w, h);
    let px = mv * float2(float(w), float(h));
    let speed = length(px);
    let hue = atan2(px.y, px.x) / (2.0 * 3.14159265) + 0.5;
    let rgb = saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return rgb * saturate(log2(1.0 + speed) / 4.0);
}

// A pixel's raygen cost in stops against the frame's mean, one band per stop with the edge
// darkened: blue under the mean, green at it, yellow/orange/red over it, white at 16x or more.
float3 costFalseColor(float stops) {
    let t = clamp(stops, -4.0, 4.0);
    float3 c;
    if (t < -2.0) c = lerp(float3(0.02, 0.02, 0.10), float3(0.15, 0.15, 0.65), (t + 4.0) * 0.5);
    else if (t < -1.0) c = lerp(float3(0.15, 0.15, 0.65), float3(0.00, 0.60, 0.90), t + 2.0);
    else if (t < 0.0) c = lerp(float3(0.00, 0.60, 0.90), float3(0.00, 0.85, 0.40), t + 1.0);
    else if (t < 1.0) c = lerp(float3(0.00, 0.85, 0.40), float3(0.85, 0.95, 0.00), t);
    else if (t < 2.0) c = lerp(float3(0.85, 0.95, 0.00), float3(1.00, 0.60, 0.00), t - 1.0);
    else if (t < 3.0) c = lerp(float3(1.00, 0.60, 0.00), float3(1.00, 0.10, 0.05), t - 2.0);
    else c = lerp(float3(1.00, 0.10, 0.05), float3(1.00, 1.00, 1.00), t - 3.0);
    let f = frac(t);
    let edge = smoothstep(0.0, 0.035, f) * smoothstep(0.0, 0.035, 1.0 - f);
    return saturate(c) * (0.4 + 0.6 * edge);
}

// The render-resolution inputs the composite uses unresolved (the fog veil, the sky mask) are
// traced through each frame's jitter: at an edge between very different values (a far ridge's
// haze against the sky) a pixel reads one side one frame and the other the next. Each keeps a
// running average at output resolution, reprojected by the motion guide and clamped to this
// frame's 3x3 render-pixel neighbourhood (`lo`, `hi`): inside a surface the range is tight and
// nothing ghosts; across an edge it spans both sides and the average settles to the coverage.
static const float HISTORY_BLEND = 0.1;

// This pixel's slot in the histories and last frame's slot (or -1: none / off screen).
int2 historySlots(float2 uv, float2 output) {
    if (any(uv < 0.0) || any(uv >= 1.0)) {
        return int2(-1);
    }
    let size = uint2(output);
    let p = min(uint2(uv * output), size - 1u);
    let prev_uv = uv - test[7].SampleLevel(uv, 0.0).rg;
    var prev = -1;
    if (push.history_reset == 0u && all(prev_uv >= 0.0) && all(prev_uv < 1.0)) {
        let q = min(uint2(prev_uv * output), size - 1u);
        prev = int(q.y * size.x + q.x);
    }
    return int2(int(p.y * size.x + p.x), prev);
}

float2 unpackHalf2(uint q) {
    return float2(f16tof32(q & 0xFFFFu), f16tof32(q >> 16));
}

// The prepass's buffer index of the render pixel under `uv` (post_prep.slang `range_prep`).
uint rangeIndex(float2 uv) {
    let size = uint2(push.range_width, push.range_height);
    let p = min(uint2(max(uv, 0.0) * float2(size)), size - 1u);
    return p.y * size.x + p.x;
}

// Local tone mapping's three keys (post_prep.slang `key_prep`), read bilinearly.
float3 keysTexel(int2 p) {
    let q = push.keys[uint(p.y) * push.key_width + uint(p.x)];
    return float3(unpackHalf2(q.x), unpackHalf2(q.y).x);
}

float3 keysRead(float2 uv) {
    let size = int2(int(push.key_width), int(push.key_height));
    let x = uv * float2(size) - 0.5;
    let p = int2(floor(x));
    let f = x - float2(p);
    let a = clamp(p, int2(0), size - 1);
    let b = clamp(p + 1, int2(0), size - 1);
    let top = lerp(keysTexel(a), keysTexel(int2(b.x, a.y)), f.x);
    let bottom = lerp(keysTexel(int2(a.x, b.y)), keysTexel(b), f.x);
    return lerp(top, bottom, f.y);
}

// The sky exemption at `uv` (depth guide, faded in from 300 m to 3 km).
float skyAt(float2 uv) {
    return saturate(log2(test[5].SampleLevel(uv, 0.0).r / 300.0) / log2(10.0));
}

// Local tone mapping: a region's surroundings (the output's pyramid at the level ~3 % of the
// height across, read smoothly) against the whole frame's (a 4x4 grid of log reads at a level
// ~1/16 of the height) sets a gain that moves the region `strength` of the way toward the
// frame's average, clamped to two stops either way. Low frequency only, so texture stays;
// shade under a canopy lifts, sunlit ground comes down, and a frame dark all over (night) keeps
// its exposure. A few taps of the full-resolution image instead cast faint copies of every
// sharp silhouette against the sky; the surroundings are edge-aware, and the sky is left as is.
float pyramidLog(float2 uv, float level, float scale) {
    let c = test[9].SampleLevel(uv, level).rgb * scale;
    return log2(max(dot(c, float3(0.2125, 0.7154, 0.0721)), 1.0e-4));
}

// The whole frame's level: a 4x4 grid of log reads at a level ~1/16 of the height.
float frameLog(float scale, uint h, uint levels) {
    let frame_level = clamp(log2(float(h) / 16.0), 0.0, float(levels) - 1.0);
    var frame = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            frame += pyramidLog((float2(float(x), float(y)) + 0.5) / 4.0, frame_level, scale);
        }
    }
    return frame / 16.0;
}

float3 localGain(float2 uv, float scale, float strength, Frame f, int2 slots) {
    let own = log2(max(dot(test[0].SampleLevel(uv, 0.0).rgb * scale, float3(0.2125, 0.7154, 0.0721)), 1.0e-4));
    // Edge-aware surroundings: the three scales blended by how close each is to the pixel's own
    // level (leaning coarse). Sky beside a canopy reads sky at the finer scales and keeps the
    // sky's gain -- one coarse level bled the canopy's darkness across the silhouette, a halo.
    // The three scales (post_prep.slang `key_prep`): level 0, the level whose texel spans ~3 % of
    // the height, and one 1.5 levels finer.
    let keys = keysRead(uv);
    var key = 0.0;
    var wsum = 0.0;
    for (int i = 0; i < 3; i++) {
        let k = keys[i];
        let d = (own - k) / 0.75;
        let wi = exp2(float(i)) * exp(-d * d);
        key += wi * k;
        wsum += wi;
    }
    key /= max(wsum, 1.0e-6);
    let stops = clamp((f.frame_log - key) * strength, -2.0, 2.0);
    // The sky keeps its exposure (the depth guide holds 1e7 there; its bilinear blend softens
    // the silhouette): pulled toward a darker frame it went deep blue, and any mismatch of
    // the surroundings at a silhouette showed as a halo.
    // Faded in by distance (300 m to 3 km), not sky-or-not: the depth guide is traced through
    // each frame's jitter, and at a far ridge against the sky a hard mask flipped the edge pixels
    // between the sky's gain and the ground's every frame (boil). Terrain that far keeps the
    // sky's exposure too, so the two sides of the edge agree.
    var sky = skyAt(uv);
    // A history like the fog veil's (above `historySlots`), clamped to the 3x3 render-pixel
    // range under `uv` (post_prep.slang `range_prep`).
    if (push.sky_cur != nullptr && slots.x >= 0) {
        if (slots.y >= 0 && push.sky_range != nullptr) {
            let range = unpackHalf2(push.sky_range[rangeIndex(uv)]);
            let lo = min(range.x, sky);
            let hi = max(range.y, sky);
            sky = lerp(clamp(push.sky_prev[slots.y], lo, hi), sky, HISTORY_BLEND);
        }
        push.sky_cur[slots.x] = sky;
    }
    return float3(exp2(stops * (1.0 - sky)));
}

// The exposure meter from the resolved image: at 1 spp most raw samples are exactly 0 (an
// occluded shadow ray, a bounce that found nothing) or rare bright hits, and even strip-averaged
// the raw meter read a scene up to a stop darker than it shows. Each render pixel's meter is
// written by ONE output pixel (the first whose uv lands in it, in x and y), which reads the
// raygen's albedo from the pair before overwriting it.
void meterResolved(float2 uv, float3 src, float through, float3 veil, float input) {
    // Only the image itself: a letterboxed spectator's scissor rounds its fractional viewport
    // outward, and the sliver of fragments past the edge had uv < 0 -- texel -1, an index far
    // past the buffer (a GPU page fault in XR).
    if (any(uv < 0.0) || any(uv >= 1.0)) {
        return;
    }
    let size = float2(float(push.lum_width), float(push.lum_height));
    let r = clamp(int2(floor(uv * size)), int2(0), int2(size) - 1);
    let step = float2(abs(ddx(uv.x)), abs(ddy(uv.y)));
    let before = int2(floor((uv - step) * size));
    if (before.x == r.x || before.y == r.y) {
        return;
    }
    let i = 2u * (uint(r.y) * push.lum_width + uint(r.x));
    let albedo = push.lum[i + 1u];
    if (albedo < 0.0) {
        push.lum[i] = -1.0;
        return;
    }
    let lw = float3(0.2125, 0.7154, 0.0721);
    let surface = dot(src, lw) / input * through;
    let fog = dot(veil, lw) / input;
    push.lum[i] = surface / max(albedo, 0.04) + fog;
    push.lum[i + 1u] = surface + fog;
}

// Bloom (post_prep.slang `bloom_prep`), upsampled bilinearly from the pyramid's level 1.
float3 bloomTexel(int2 p) {
    let q = push.bloom_buf[uint(p.y) * push.bloom_width + uint(p.x)];
    return float3(f16tof32(q.x & 0xFFFFu), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFFu));
}

float3 bloomRead(float2 uv) {
    let size = int2(int(push.bloom_width), int(push.bloom_height));
    let x = uv * float2(size) - 0.5;
    let p = int2(floor(x));
    let f = x - float2(p);
    let a = clamp(p, int2(0), size - 1);
    let b = clamp(p + 1, int2(0), size - 1);
    let top = lerp(bloomTexel(a), bloomTexel(int2(b.x, a.y)), f.x);
    let bottom = lerp(bloomTexel(int2(a.x, b.y)), bloomTexel(b), f.x);
    return lerp(top, bottom, f.y);
}

float3 acesFilm(float3 x) {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

// Aspect-corrected: 0 at the centre, 1 at the corners, so it is the same shape at any window
// size; `strength` (0 = off) is how dark the corners get.
float3 applyVignette(float3 color, float2 uv, float strength, float2 output) {
    if (strength <= 0.0) {
        return color;
    }
    let aspect = output.x / output.y;
    let half_extent = float2(aspect, 1.0) * 0.5;
    let dist = length((uv - 0.5) * float2(aspect, 1.0)) / length(half_extent);
    let falloff = smoothstep(1.0, 0.45, dist);
    return lerp(color, color * falloff, strength);
}

[shader("fragment")]
float4 fragment(VertexOut v) : SV_Target0 {
    let uv = v.uv;
    let frame = frameOf(v);
    let view = push.debug_view;
    // Luminance: the one view that must UNDO the exposure. The colour guide is what Ray
    // Reconstruction ingests, pre-exposed at the quantised `input_exposure`; dividing that
    // back out recovers the scene's physical radiance in nits, independent of the look and
    // of whatever the metering is currently doing.
    // Cost: the raygen's time for the render pixel under this one, against the frame's mean.
    if (view == 10u) {
        if (push.cost == nullptr || any(uv < 0.0) || any(uv >= 1.0)) {
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        let size = uint2(push.lum_width, push.lum_height);
        let r = min(uint2(uv * float2(size)), size - 1u);
        let ns = float(push.cost[2u + r.y * size.x + r.x]);
        let total = float(((uint64_t*)push.cost)[0]);
        let mean = total / float(size.x * size.y);
        return float4(costFalseColor(log2(max(ns, 1.0) / max(mean, 1.0))), 1.0);
    }
    if (view == 9u) {
        let pre = test[1].Sample(uv).rgb;
        let nits = dot(pre, float3(0.2126, 0.7152, 0.0722)) / max(push.ae.input_exposure, 1.0e-12);
        return float4(nitsFalseColor(nits), 1.0);
    }

    // Data views: raw encodings, no exposure or tonemap -- a static scene shows a static image
    // (matches AuroraDebugView's variant order).
    if (view >= 2u) {
        float3 v;
        switch (view) {
            case 2u: v = test[2].Sample(uv).rgb * 0.5 + 0.5; break;            // normals
            case 3u: v = float3(test[2].Sample(uv).a); break;                  // roughness
            case 4u: v = test[3].Sample(uv).rgb; break;                        // diffuse
            case 5u: v = test[4].Sample(uv).rgb; break;                        // specular
            case 6u: v = float3(exp2(-test[5].Sample(uv).r / 32.0)); break;    // depth
            case 7u: v = specHitColor(test[6].Sample(uv).r); break;          // spec hit
            default: v = motionColor(test[7].Sample(uv).rg); break;           // motion
        }
        return float4(v, 1.0);
    }

    // The DLSS output (or, view 1, its noisy colour input): resolved linear HDR at the
    // quantised input exposure. Re-expose to the look here -- the smooth metered value (Auto)
    // or a fixed one -- where it cannot disturb RR's history. Tonemap to display-linear; the
    // sRGB attachment applies the transfer function on store.
    let input = max(push.ae.input_exposure, 1.0e-12);
    var src = test[view].Sample(uv).rgb;
    // The camera ray's fog over it (none without the guides), kept apart so local tone
    // mapping lifts the surfaces behind the veil and not the veil.
    var veil = float3(0.0);
    var through = 1.0;
    // The fog in front of the surface, which Ray Reconstruction never saw: divided by a
    // surface's albedo there, a white veil bled colour halos around dark coloured surfaces.
    let slots = push.fog != 0u && view == 0u ? historySlots(uv, frame.output) : int2(-1);
    if (push.fog != 0u && view == 0u) {
        // One bilinear read: the camera ray's fog comes from the grid's column sums, already
        // smooth across cells (a 9-tap blur here once hid the per-cell noise of a march).
        let size = frame.fog;
        // The fog was traced through this frame's sub-pixel jitter (render texel t holds the
        // scene at t + 0.5 + jitter) and Ray Reconstruction is not there to undo it: shift
        // back, or every fog edge shimmers by a fraction of a pixel each frame.
        let jitter = float2(push.uniforms[UNIFORM_JITTER_FLOAT], push.uniforms[UNIFORM_JITTER_FLOAT + 1]);
        let at = uv - jitter / size;
        var f = test[8].Sample(at);
        if (push.fog_cur != nullptr && slots.x >= 0) {
            if (slots.y >= 0 && push.fog_range != nullptr) {
                let q = push.fog_range[rangeIndex(at)];
                let lo = min(float4(unpackHalf2(q.x), unpackHalf2(q.y)), f);
                let hi = max(float4(unpackHalf2(q.z), unpackHalf2(q.w)), f);
                f = lerp(clamp(push.fog_prev[slots.y], lo, hi), f, HISTORY_BLEND);
            }
            push.fog_cur[slots.x] = f;
        }
        veil = f.rgb;
        through = f.a;
    }
    if (push.scene_radiance != 0u) {
        return float4((src * through + veil) / input, 1.0);
    }
    let look = push.display_exposure > 0.0 ? push.display_exposure : push.ae.exposure;
    if (push.lum != nullptr && push.fog != 0u && view == 0u) {
        meterResolved(uv, src, through, veil, input);
    }
    var surface = src * through;
    if (push.local_tonemap > 0.0 && view == 0u && push.fog != 0u && push.keys != nullptr) {
        surface *= localGain(uv, look / input, push.local_tonemap, frame, slots);
    }
    var exposed = (surface + veil) * (look / input);
    if (push.bloom > 0.0 && view == 0u && push.fog != 0u && push.bloom_buf != nullptr) {
        exposed = lerp(exposed, bloomRead(uv) * (look / input), push.bloom);
    }
    var color = clamp(acesFilm(exposed), float3(0.0), float3(1.0));
    color = applyVignette(color, uv, push.uniforms[UNIFORM_VIGNETTE_FLOAT], frame.output);
    return float4(color, 1.0);
}
