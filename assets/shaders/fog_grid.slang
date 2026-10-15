// The fog light grid's compute passes (src/fog_grid.rs), around raygen.slang's `fog_grid`:
//
//   fog_reproject  every cell: last frame's grid at the cell's reprojected centre, so the cells
//                  `fog_grid` does not light this frame carry their history forward.
//   fog_soften     every cell: the lit grid blurred [1 2 1]^2 / 16 across x and y. Each cell's
//                  light is a few random shadow rays; the blur shares that noise with its
//                  neighbours once here instead of in four trilinear reads at every lookup.
//   fog_integrate  every column: front to back through its slices, the running in-scatter and
//                  transmittance stored per cell at its far side (one slice on: slice 0 is the
//                  eye).
//
// The sums and the soft grid are 3D images written in place through the descriptor heap
// (heap storage volumes, src/descriptor_heap.rs) and sampled by the trace.
import rt_types;
import fog;

struct FogGridPush {
    UniformData* u;
    uint base;
    // The grid's storage volumes in `fog_volumes`: the sums, then the soft grid.
    uint volumes;
};
[[vk::push_constant]] FogGridPush push;

[[vk::binding(0, 0)]] [[vk::image_format("rgba32f")]] RWTexture3D<float4> fog_volumes[];

[shader("compute")]
[numthreads(64, 1, 1)]
void fog_reproject(uint3 gid: SV_DispatchThreadID) {
    let u = push.u;
    let i = gid.x + push.base;
    if (i >= u.fog_grid_cells) {
        return;
    }
    if (u.fog_grid_fresh != 0u) {
        u.fog_grid_cur[i] = float4(0.0);
        return;
    }
    let size = fogGridSizeU(u);
    let env = u.environments + u.environment;
    let dist = env.fog_grid_distance;
    let cell = float3(i % size.x, (i / size.x) % size.y, i / (size.x * size.y));
    let grid = float3(size);
    let centre = fogGridPointU(u, (cell.xy + 0.5) / grid.xy, (cell.z + 0.5) / grid.z, dist);
    let g = fogGridCoord(u.fog_grid_prev_view_proj, centre, size, dist);
    var prev = fogGridSample(u.fog_grid_prev, size, g);
    // Off last frame's grid (past its far slice moving forward, its edges turning) the
    // nearest edge cell is still the best guess: kept at a quarter of its weight rather than
    // dropped, which left those cells one raw sample every frame.
    if (any(g < float3(0.0)) || any(g > grid)) {
        prev.w = min(prev.w, env.fog_grid_history * 0.25);
    }
    if (any(isnan(prev)) || any(isinf(prev)) || prev.w < 0.0) {
        prev = float4(0.0);
    }
    u.fog_grid_cur[i] = prev;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fog_soften(uint3 gid: SV_DispatchThreadID) {
    let u = push.u;
    let i = gid.x + push.base;
    if (i >= u.fog_grid_cells) {
        return;
    }
    let size = fogGridSizeU(u);
    let x = int(i % size.x);
    let y = int((i / size.x) % size.y);
    let z = i / (size.x * size.y);
    let hi = int2(size.xy) - 1;
    var r = float3(0.0);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            let q = clamp(int2(x + dx, y + dy), int2(0), hi);
            let w = float((2 - abs(dx)) * (2 - abs(dy)));
            r += w * u.fog_grid_cur[(z * size.y + uint(q.y)) * size.x + uint(q.x)].rgb;
        }
    }
    let soft = float4(r * (1.0 / 16.0), 0.0);
    u.fog_grid_soft[i] = soft;
    fog_volumes[push.volumes + 1u][int3(x, y, int(z))] = soft;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void fog_integrate(uint3 gid: SV_DispatchThreadID) {
    let u = push.u;
    let size = fogGridSizeU(u);
    let c = gid.x + push.base;
    if (c >= size.x * size.y) {
        return;
    }
    let id = uint2(c % size.x, c / size.x);
    let env = u.environments + u.environment;
    let dist = env.fog_grid_distance;
    let uv = (float2(id) + 0.5) / float2(size.xy);
    let cam = fogGridPointU(u, uv, 0.0, dist);
    // Metres along the column per metre of view depth.
    let stretch = length(fogGridPointU(u, uv, 1.0, dist) - cam) / dist;
    let slices = float(size.z);
    var T = 1.0;
    var L = float3(0.0);
    var prev = 0.0;
    // Slice 0 of the sums is the eye itself: nothing in-scattered, nothing absorbed.
    fog_volumes[push.volumes][int3(int2(id), 0)] = float4(0.0, 0.0, 0.0, 1.0);
    for (uint k = 0u; k < size.z; k++) {
        let f = float(k + 1u) / slices;
        let depth = dist * f * f;
        let dt = (depth - prev) * stretch;
        prev = depth;
        let s = (float(k) + 0.5) / slices;
        let i = (k * size.y + id.y) * size.x + id.x;
        let sigma = fogDensityAtU(u, env, fogGridPointU(u, uv, s, dist), true);
        if (sigma > 0.0) {
            let step_t = exp(-sigma * dt);
            L += T * FOG_ALBEDO * u.fog_grid_soft[i].rgb * (1.0 - step_t);
            T *= step_t;
        }
        fog_volumes[push.volumes][int3(int2(id), int(k) + 1)] = float4(L, T);
    }
}
