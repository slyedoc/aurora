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
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<QuadPush> push;

// 0: the DLSS output; 1..=7 the guide images (colour, normals+roughness, diffuse, specular,
// depth, specular hit distance, motion) for the debug views. Slots the renderer cannot fill
// yet alias the output.
[[vk::binding(0, 0)]] Sampler2D test[8];

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
    let src = test[view].Sample(uv).rgb;
    if (push.scene_radiance != 0u) {
        return float4(src / input, 1.0);
    }
    let look = push.display_exposure > 0.0 ? push.display_exposure : push.ae.exposure;
    var color = clamp(acesFilm(src * (look / input)), float3(0.0), float3(1.0));
    color = applyVignette(color, uv, push.uniforms[UNIFORM_VIGNETTE_FLOAT]);
    return float4(color, 1.0);
}
