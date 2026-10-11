// Gizmo lines (gizmo_render.rs): each segment is a screen-space quad with a soft edge, depth
// tested in the fragment against the raygen's linear depth guide (there is no depth buffer --
// the scene is traced).

// Must match `gizmo_render::GizmoVertex` (repr(C), 24 bytes).
struct GizmoVertex {
    float position_x, position_y, position_z;  // world space
    uint color;                                 // packed rgba8 (r | g<<8 | b<<16 | a<<24), linear
    float depth_bias;  // bevy's GizmoConfig::depth_bias: -1 always in front .. 1 always behind
    float width;       // line width, pixels
};

// Must match `gizmo_render::GizmoPushConstants`.
struct GizmoPush {
    // Unjittered clip-from-world, column-major (glam's layout).
    float4 view_proj[4];
    GizmoVertex* vertices;
    float2 inv_extent;  // 1 / target size
    uint depth_guide;
};
[[vk::push_constant]] ConstantBuffer<GizmoPush> pc;

// The raygen's linear view depth guide (render resolution, jittered; the sky is far away).
[[vk::binding(0, 0)]] Sampler2D scene_depth;

struct VertexOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR0;
    // Linear view depth with the bias applied, and whether the line takes part in the depth
    // test at all.
    [[vk::location(1)]] float depth : TEXCOORD1;
    [[vk::location(2)]] nointerpolation uint tested : TEXCOORD2;
    // Signed distance from the line's centre across it, pixels; and the line's half width.
    [[vk::location(3)]] float across : TEXCOORD3;
    [[vk::location(4)]] nointerpolation float half_width : TEXCOORD4;
};

// One segment = two endpoints in the buffer = six vertices here: a screen-space quad around
// the line, a pixel wider than it on each side for the soft edge.
static const float FEATHER = 1.0;

float4 clipOf(float3 p) {
    return pc.view_proj[0] * p.x + pc.view_proj[1] * p.y + pc.view_proj[2] * p.z + pc.view_proj[3];
}

float4 unpackColor(uint c) {
    return float4(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu),
                  float(c >> 24)) / 255.0;
}

[shader("vertex")]
VertexOut vertex(uint vertex_index : SV_VulkanVertexID) {
    VertexOut o;
    let segment = vertex_index / 6u;
    let corner = vertex_index % 6u;
    let a = pc.vertices[segment * 2u];
    let b = pc.vertices[segment * 2u + 1u];
    var clip_a = clipOf(float3(a.position_x, a.position_y, a.position_z));
    var clip_b = clipOf(float3(b.position_x, b.position_y, b.position_z));

    // Clip to just in front of the eye: the screen-space direction is meaningless behind it.
    let near_w = 1.0e-3;
    if (clip_a.w < near_w && clip_b.w < near_w) {
        o.position = float4(0.0);
        o.color = float4(0.0);
        o.depth = 0.0;
        o.tested = 0u;
        o.across = 0.0;
        o.half_width = 0.0;
        return o;
    }
    if (clip_a.w < near_w) {
        clip_a = lerp(clip_a, clip_b, (near_w - clip_a.w) / (clip_b.w - clip_a.w));
    } else if (clip_b.w < near_w) {
        clip_b = lerp(clip_b, clip_a, (near_w - clip_b.w) / (clip_a.w - clip_b.w));
    }

    // Corners: 0,1,2 / 2,1,3 over (end, side) = (0,-) (0,+) (1,-) (1,+).
    let corners = uint[6](0u, 1u, 2u, 2u, 1u, 3u);
    let c = corners[corner];
    let at_b = c >= 2u;
    let side = (c & 1u) == 0u ? -1.0 : 1.0;

    let pixels = 1.0 / pc.inv_extent;
    let screen_a = clip_a.xy / clip_a.w * pixels;
    let screen_b = clip_b.xy / clip_b.w * pixels;
    var dir = screen_b - screen_a;
    dir = dot(dir, dir) > 1.0e-8 ? normalize(dir) : float2(1.0, 0.0);
    let normal = float2(-dir.y, dir.x);

    let v = at_b ? b : a;
    var clip = at_b ? clip_b : clip_a;
    let half_width = max(v.width, 1.0) * 0.5;
    let reach = half_width + FEATHER;
    // Out along the normal, and past each end by the same reach so the caps are soft too.
    let offset_px = normal * side * reach + dir * (at_b ? reach : -reach);
    // NDC spans 2 per screen: pixels to NDC is * 2 / size, then back into clip space by w.
    clip.xy += offset_px * 2.0 * pc.inv_extent * clip.w;
    o.across = side * reach;
    o.half_width = half_width;

    // A perspective projection's clip w IS the linear view depth the raygen's depth guide
    // stores. The bias scales it: towards the eye below 0 (0 at -1), to infinity at 1.
    let bias = clamp(v.depth_bias, -1.0, 1.0);
    o.depth = bias <= 0.0 ? clip.w * (1.0 + bias) : clip.w / max(1.0 - bias, 1.0e-6);
    o.tested = bias > -1.0 ? 1u : 0u;
    // The projection is GL-style (clip +Y up, matching the raygen's inverted pixel mapping);
    // Vulkan rasterization maps clip +Y down, so mirror to land on the traced scene.
    clip.y = -clip.y;
    o.position = clip;
    o.color = unpackColor(v.color);
    return o;
}

[shader("fragment")]
float4 fragment(VertexOut i) : SV_Target0 {
    // Coverage across the line: full inside its width, falling off over the last pixel.
    var alpha = clamp(i.half_width + 0.5 - abs(i.across), 0.0, 1.0);
    if (i.tested != 0u && pc.depth_guide != 0u) {
        // The guide is coarser than the target and shifts with the sub-pixel jitter, so the
        // line is tested against each of the four texels around the fragment and keeps the
        // share it is in front of: a silhouette fades over one texel instead of blinking. The
        // slack is how far a surface's depth runs across those texels (a line lying ON a
        // grazing surface), capped so a near silhouette's spread cannot let what is behind it
        // through.
        let depths = scene_depth.Gather(i.position.xy * pc.inv_extent);
        let far = max(max(depths.x, depths.y), max(depths.z, depths.w));
        let near = min(min(depths.x, depths.y), min(depths.z, depths.w));
        let slack = min(float4(far - near), depths * 0.05) + depths * 0.01 + 0.02;
        let passes = step(float4(i.depth), depths + slack);
        alpha *= dot(passes, float4(0.25));
    }
    // Linear light: the sRGB attachment encodes on store and blends in linear space.
    return float4(i.color.rgb, i.color.a * alpha);
}
