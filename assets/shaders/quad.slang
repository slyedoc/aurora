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
    float pad;
    // The view's auto-exposure meter, (meter, albedo) pairs at render resolution (null = none;
    // auto_exposure.rs): the raygen wrote its noisy meter and the surface's albedo (< 0 = not
    // metered); this pass replaces the meter with what Ray Reconstruction resolved.
    float* lum;
    uint lum_width;
    uint lum_height;
};
[[vk::push_constant]] ConstantBuffer<QuadPush> push;

// 0: the DLSS output; 1..=7 the guide images (colour, normals+roughness, diffuse, specular,
// depth, specular hit distance, motion) for the debug views; 8 the camera ray's fog; 9 the
// output's mip pyramid (local tone mapping). Slots the renderer cannot fill yet alias the output.
[[vk::binding(0, 0)]] Sampler2D test[10];

struct VertexOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOut vertex(uint id : SV_VulkanVertexID) {
    let corners = float2[3](float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0));
    let p = corners[id];
    VertexOut o;
    o.position = float4(p, 0.0, 1.0);
    o.uv = (p + 1.0) * 0.5;
    return o;
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

// A region's level at pyramid `level`, smoothed: four reads half a texel apart hide the
// bilinear diamonds of a coarse level.
float pyramidKey(float2 uv, float level, float scale, uint w, uint h) {
    let t = exp2(level) / float2(float(w), float(h)) * 0.5;
    return 0.25 * (pyramidLog(uv + float2(-t.x, -t.y), level, scale)
                 + pyramidLog(uv + float2(t.x, -t.y), level, scale)
                 + pyramidLog(uv + float2(-t.x, t.y), level, scale)
                 + pyramidLog(uv + float2(t.x, t.y), level, scale));
}

float3 localGain(float2 uv, float scale, float strength) {
    uint w, h, levels;
    test[9].GetDimensions(0, w, h, levels);
    let top = float(levels) - 1.0;
    // Level 0 is half the output: the level whose texel spans ~3 % of the height, one 1.5 levels
    // finer, and level 0.
    let coarse = clamp(log2(float(h) * 0.03), 0.0, top);
    let own = log2(max(dot(test[0].SampleLevel(uv, 0.0).rgb * scale, float3(0.2125, 0.7154, 0.0721)), 1.0e-4));
    // Edge-aware surroundings: the three scales blended by how close each is to the pixel's own
    // level (leaning coarse). Sky beside a canopy reads sky at the finer scales and keeps the
    // sky's gain -- one coarse level bled the canopy's darkness across the silhouette, a halo.
    var key = 0.0;
    var wsum = 0.0;
    for (int i = 0; i < 3; i++) {
        let level = i == 0 ? 0.0 : max(coarse - float(2 - i) * 1.5, 0.0);
        let k = pyramidKey(uv, level, scale, w, h);
        let d = (own - k) / 0.75;
        let wi = exp2(float(i)) * exp(-d * d);
        key += wi * k;
        wsum += wi;
    }
    key /= max(wsum, 1.0e-6);
    let frame_level = clamp(log2(float(h) / 16.0), 0.0, top);
    var frame = 0.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            frame += pyramidLog((float2(float(x), float(y)) + 0.5) / 4.0, frame_level, scale);
        }
    }
    let stops = clamp((frame / 16.0 - key) * strength, -2.0, 2.0);
    // The sky keeps its exposure (the depth guide holds 1e7 there; its bilinear blend softens
    // the silhouette): pulled toward a darker frame it went deep blue, and any mismatch of
    // the surroundings at a silhouette showed as a halo.
    let sky = saturate(test[5].SampleLevel(uv, 0.0).r / 1.0e7);
    return float3(exp2(stops * (1.0 - sky)));
}

// The exposure meter from the resolved image: at 1 spp most raw samples are exactly 0 (an
// occluded shadow ray, a bounce that found nothing) or rare bright hits, and even strip-averaged
// the raw meter read a scene up to a stop darker than it shows. Each render pixel's meter is
// written by ONE output pixel (the first whose uv lands in it, in x and y), which reads the
// raygen's albedo from the pair before overwriting it.
void meterResolved(float2 uv, float3 src, float through, float3 veil, float input) {
    let size = float2(float(push.lum_width), float(push.lum_height));
    let r = min(int2(floor(uv * size)), int2(size) - 1);
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
float3 applyVignette(float3 color, float2 uv, float strength) {
    if (strength <= 0.0) {
        return color;
    }
    uint w, h;
    test[0].GetDimensions(w, h);
    let aspect = float(w) / float(h);
    let half_extent = float2(aspect, 1.0) * 0.5;
    let dist = length((uv - 0.5) * float2(aspect, 1.0)) / length(half_extent);
    let falloff = smoothstep(1.0, 0.45, dist);
    return lerp(color, color * falloff, strength);
}

[shader("fragment")]
float4 fragment([[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target0 {
    let view = push.debug_view;
    // Luminance: the one view that must UNDO the exposure. The colour guide is what Ray
    // Reconstruction ingests, pre-exposed at the quantised `input_exposure`; dividing that
    // back out recovers the scene's physical radiance in nits, independent of the look and
    // of whatever the metering is currently doing.
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
            case 7u: v = float3(exp2(-test[6].Sample(uv).r / 32.0)); break;    // spec hit
            default: v = float3(test[7].Sample(uv).rg * 0.1 + 0.5, 0.5); break; // motion
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
    if (push.fog != 0u && view == 0u) {
        // Nine bilinear taps 1.5 texels apart (~5x5 render texels): the fog grid's cell noise
        // smoothed out, which Ray Reconstruction used to do. Fog has no fine detail to lose.
        uint fw, fh;
        test[8].GetDimensions(fw, fh);
        let size = float2(float(fw), float(fh));
        let texel = 1.5 / size;
        // The fog was traced through this frame's sub-pixel jitter (render texel t holds the
        // scene at t + 0.5 + jitter) and Ray Reconstruction is not there to undo it: shift
        // back, or every fog edge shimmers by a fraction of a pixel each frame.
        let jitter = float2(push.uniforms[UNIFORM_JITTER_FLOAT], push.uniforms[UNIFORM_JITTER_FLOAT + 1]);
        let fog_uv = uv - jitter / size;
        var f = float4(0.0);
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                let w = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
                f += test[8].Sample(fog_uv + float2(float(x), float(y)) * texel) * w;
            }
        }
        f /= 16.0;
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
    if (push.local_tonemap > 0.0 && view == 0u && push.fog != 0u) {
        surface *= localGain(uv, look / input, push.local_tonemap);
    }
    let exposed = (surface + veil) * (look / input);
    var color = clamp(acesFilm(exposed), float3(0.0), float3(1.0));
    color = applyVignette(color, uv, push.uniforms[UNIFORM_VIGNETTE_FLOAT]);
    return float4(color, 1.0);
}
