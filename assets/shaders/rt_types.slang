// The raygen's view of the frame: the push-constant
// block and every buffer it points at.
//
// Everything read through a pointer is declared with scalars (or 16-byte-aligned vectors at
// 16-byte offsets) only, so its byte layout is the Rust side's `scalar` layout with no
// vector-alignment question. Matrices are column-major float[16] (glam / GLSL order).
module rt_types;

__exported import hit_payload;

public static const float PI = 3.141592653589793;

// Column-major 4x4 matrix in a buffer (`&uniforms.view_proj[0]`) times a column vector (GLSL
// `mat4 * vec4`), its columns read as four 16-byte loads: a by-value `float m[16]` from a
// pointer is sixteen 4-byte loads (scalar layout marks every pointer access 4-byte aligned).
// The uniform block's matrices sit on 16-byte boundaries.
public float4 mat4MulVecAt(float* m, float4 v) {
    let c = (float4*)m;
    return loadAligned<16>(c) * v.x + loadAligned<16>(c + 1) * v.y
        + loadAligned<16>(c + 2) * v.z + loadAligned<16>(c + 3) * v.w;
}

// Rows u, v, w: `mul(x, orthonormalBasis(n))` is GLSL's `mat3(u, v, w) * x`.
public float3x3 orthonormalBasis(float3 n) {
    float3 a;
    if (abs(n.x) > abs(n.z)) {
        a = float3(-n.y, n.x, 0.0);
    } else {
        a = float3(0.0, -n.z, n.y);
    }
    let w = normalize(n);
    let u = normalize(cross(a, w));
    let v = cross(w, u);
    return float3x3(u, v, w);
}

// One sun of a world (src/sky.rs EnvironmentSunGpu): xyz towards it + cos(angular radius);
// the disc's radiance (nits) + the directly-seen disc's scale.
public struct EnvironmentSun {
    public float4 direction;
    public float4 radiance;
};

// One environment: everything a ray in it sees (src/sky.rs EnvironmentGpu). mode 0 = flat
// colour (color), 1 = HDR (tex, color = scale, importance = its sampling table or null),
// 2 = gradient (zenith/horizon/ground), 3 = atmosphere (tex = the space image, color = its
// scale, 0 = none).
// One fog volume (src/fog_volume.rs `GpuFogVolume`): world to unit-local rows, then its
// parameters. Shape 0 = box, 1 = ellipsoid.
public struct FogVolumeGpu {
    public float4 row0;
    public float4 row1;
    public float4 row2;
    public float density;
    public float edge;
    public float noise_scale;
    public float noise_strength;
    public float wind_x, wind_y, wind_z;
    public uint shape;
};

public struct EnvironmentGpu {
    public uint mode;
    public uint tex;
    public uint sun_count;
    public float sky_brightness;
    // The HDR or generated sky's importance-sampling table (src/sky_gen.rs): cdf[N + 1], the
    // solid-angle pdf of each texel, then the guide (u32 bits) [N + 1]: guide[j] is the last
    // texel whose cdf <= j / N. N = SKY_IMPORTANCE_W * SKY_IMPORTANCE_H.
    public float* importance;
    // Fog: extinction per metre, Henyey-Greenstein g, base height, falloff (0 = uniform).
    public float fog_density;
    public float fog_scatter;
    public float fog_height;
    public float fog_falloff;
    public uint pad;
    // Fog: the camera ray's march steps (0 = it scatters at random like a bounce).
    public uint fog_steps;
    // Fog light grid (src/fog_grid.rs): how far it reaches (0 = off), the frames it averages,
    // trace pixels per cell across and depth slices.
    public float fog_grid_distance;
    public float fog_grid_history;
    public uint fog_grid_tile;
    public uint fog_grid_slices;
    // Stars behind a generated sky (src/sky.rs SkyAir): the rotation from a world direction
    // into the star map (quaternion), the map's bindless index and its radiance scale (nits,
    // already faded by the sun; 0 = none).
    public float4 stars_rotation;
    public uint stars_tex;
    public float stars_nits;
    public uint stars_pad0;
    public uint stars_pad1;
    public float4 color;
    public float4 zenith;
    public float4 horizon;
    public float4 ground;
    public EnvironmentSun suns[4];
};

// Where a ray leaving a surface at `p` starts: pushed off the surface along `n` (the side it
// leaves through) by a fixed number of float ulps, so the offset scales with the hit's distance
// from the world origin and the ray never re-hits the surface it left, near or far (Waechter &
// Binder, "A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing Gems ch. 6).
// Within 1/32 of the origin, where ulps get too fine, a small absolute offset instead. Rays
// that start this way trace with tmin 0.
public float3 offsetRay(float3 p, float3 n) {
    static const float ORIGIN = 1.0 / 32.0;
    static const float FLOAT_SCALE = 1.0 / 65536.0;
    static const float INT_SCALE = 256.0;
    let of_i = int3(INT_SCALE * n.x, INT_SCALE * n.y, INT_SCALE * n.z);
    let p_i = float3(
        asfloat(asint(p.x) + (p.x < 0.0 ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + (p.y < 0.0 ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + (p.z < 0.0 ? -of_i.z : of_i.z)));
    let per_axis = float3(
        abs(p.x) < ORIGIN ? p.x + FLOAT_SCALE * n.x : p_i.x,
        abs(p.y) < ORIGIN ? p.y + FLOAT_SCALE * n.y : p_i.y,
        abs(p.z) < ORIGIN ? p.z + FLOAT_SCALE * n.z : p_i.z);
    // The hit point's error scales with its LARGEST coordinate, not each one's own: ground at
    // y ~ 0 a hundred metres out got a y push of ~1e-6 m against ~1e-5 m of error, and its
    // shadow rays hit the ground they left (acne moire wherever |y| is small).
    let largest = max(max(abs(p.x), abs(p.y)), abs(p.z));
    return per_axis + n * (largest * (1.0 / 524288.0));
}

// `offsetRay` off the surface with normal `n`, to the side direction `d` leaves through.
public float3 leaveSurface(float3 p, float3 n, float3 d) {
    return offsetRay(p, dot(d, n) >= 0.0 ? n : -n);
}

// The sampling grid of every importance table (src/sky_gen.rs SKY_W / SKY_H).
public static const uint SKY_IMPORTANCE_W = 512;
public static const uint SKY_IMPORTANCE_H = 256;

// The environment a mask is in: its highest environment bit (the main one for none).
public uint environmentOf(uint mask, uint environment_bits) {
    let m = mask & (environment_bits | 1u);
    return m == 0u ? 0u : firstbithigh(m);
}

// `UniformData` in ray_render_plugin.rs (offsets pinned by a test there).
public struct UniformData {
    public float inverse_view[16];
    public float inverse_projection[16];
    public uint pull_focus_x;
    public uint pull_focus_y;
    public float aperture;
    // The environment this view is in: its camera rays start there (portals move them).
    public uint environment;
    // DLSS: camera matrices for depth / motion vectors, and the sub-pixel jitter (pixels).
    public float view[16];
    public float view_proj[16];
    public float prev_view_proj[16];
    public float jitter_x, jitter_y;
    // Free-running frame counter (the RNG seed -- fresh noise every frame for the denoiser).
    public uint frame;
    // Free: samples, max_bounces and the firefly clamp moved to raygen.slang's specialization
    // constants or went.
    public uint pad_paths[3];
    // Post-process vignette strength (0 = off).
    public float vignette;
    // Entries in the emissive-triangle light table (0 = no light NEE / MIS).
    public uint light_entries;
    public uint pad_candidates[2];
    // Added to every ray-cone texture LOD: log2(render / output width) plus the panel's bias.
    public float lod_bias;
    // ReSTIR GI: the perceptual roughness from which a secondary hit's sample is shared with
    // other pixels (raygen.slang `main`, restir_gi.slang).
    public float gi_roughness;
    // Light-table generation; reservoirs from another generation are dropped.
    public uint light_epoch;
    // Cap on temporal history, in candidate-samples.
    public float restir_m_clamp;
    // Radiance cache: 0 = off; else paths may terminate into it.
    public uint sharc;
    // Base voxel size (meters) at the camera; doubles with each distance octave past 8m.
    public float sharc_voxel;
    // Ray portals (src/portal.rs): pair table (x = instance slot, y = target slot, z = valid)
    // + entry count (0 = none).
    public uint4* portals;
    public uint portal_count;
    // Uniform multiplier on every light's emission (emissive surfaces + analytic lights).
    // Uniform across lights, so the light table's relative weights/CDF stay valid.
    public float emissive_boost;
    // The camera's render-layer cull mask; every ray starts with it (portals swap it).
    public uint camera_mask;
    // Layers the camera ray skips (AuroraPrimaryHidden); shadows and bounces still see them.
    public uint primary_hidden;
    // The mask bits that are environments; a portal ray takes the exit's from these.
    public uint environment_bits;
    // 1: the reference accumulation is written back into the noisy colour (reference.rs `feed`).
    public uint reference_feed;
    // Every environment's sky, suns and fog, by environment bit.
    public EnvironmentGpu* environments;
    // Fog volumes (src/fog_volume.rs): the table + entry count (0 = none).
    public FogVolumeGpu* fog_volumes;
    public uint fog_volume_count;
    // Seconds since startup (wrapped), for the volumes' drifting noise.
    public float time;
    // Fog light grid (src/fog_grid.rs): last frame's and this frame's cells (0 = off), the
    // cell count, and 1 when last frame's is not this grid's history.
    public float4* fog_grid_prev;
    public float4* fog_grid_cur;
    public uint fog_grid_cells;
    public uint fog_grid_fresh;
    // The grid's own view (a headset's eyes share one): where its cells are, this frame's
    // view-projection (the camera ray looks the grid up), last frame's (the update's history),
    // and its cells across, down and deep.
    public float fog_grid_inverse_view[16];
    public float fog_grid_inverse_projection[16];
    public float fog_grid_view_proj[16];
    public float fog_grid_prev_view_proj[16];
    public uint fog_grid_size_x, fog_grid_size_y, fog_grid_size_z;
    public uint pad_one_ray;
    // Bindless 3D images (raygen.slang `volumes`): the grid's columns summed front to back
    // (`fog_integrate`: per cell, the in-scatter and transmittance from the grid's eye to the
    // cell's far side, behind a first slice of nothing) and the grid blurred once across x and
    // y (`fog_soften`). A camera ray's lookups filter them in the texture unit.
    public uint fog_grid_sum_tex;
    public uint fog_grid_soft_tex;
    // The soft grid as a buffer (`fog_integrate` reads it).
    public float4* fog_grid_soft;
    // `fog_grid` lights one cell of every stride x stride block per frame (fogGridPhase).
    public uint fog_grid_stride;
    // RenderSettings::ab: 0 the old path under test, 1 the new (benches/northshire.rs --ab).
    public uint ab;
    public uint pad_shared[2];
};

// The equirectangular mapping of the HDR sky, shared by the miss shader and the environment
// sampler in the raygen. u wraps around +Z through +X, v runs from +Y (0) to -Y (1).
static const float ENV_PI = 3.14159265359;

public float2 env_dir_to_uv(float3 d) {
    let phi = atan2(d.x, d.z);
    let u = ((phi > 0.0 ? phi : (phi + 2.0 * ENV_PI)) / (2.0 * ENV_PI) - 0.5);
    let v = acos(clamp(d.y, -1.0, 1.0)) / ENV_PI;
    return float2(u < 0.0 ? u + 1.0 : u, v);
}

public float3 env_uv_to_dir(float2 uv) {
    let phi = (uv.x + 0.5) * 2.0 * ENV_PI;
    let theta = uv.y * ENV_PI;
    let st = sin(theta);
    return float3(st * sin(phi), cos(theta), st * cos(phi));
}

// ---- Emissive-triangle light table (src/lights.rs, lights.slang) --------------------------

// One emissive instance: its BLAS streams, its TLAS slot (world transform), and the range
// of entries its triangles occupy in the global table. 64 bytes.
public struct LightInstance {
    // Vertex stream: 8 floats each (position, normal, texcoord), position at 0..2.
    public float* vertices;
    public uint* indices;
    public uint* geom_to_index;
    public uint* geom_to_triangle;
    public uint geom_count;
    public uint slot;
    public uint entry_base;
    public uint tri_count;
    // Per geometry: emitted radiance (3 floats each, nits).
    public float* geom_emission;
    public uint pad2_x, pad2_y;
};

// One ReSTIR DI reservoir (src/restir.rs allocates one per traced pixel). 32 bytes.
// The primary vertex as ReSTIR DI sees it (src/restir.rs), written by the raygen at sample 0's
// first shaded surface; `frame` marks it live (a pixel that hit nothing keeps an old one).
// 64 bytes, half a cache line: the spatial pass reads neighbours' at random.
public struct RestirSurface {
    public float3 position;
    public uint frame;
    // Where the surface was last frame (object motion), and its view depth (clip w).
    public float3 prev_position;
    public float depth;
    // Albedo as rgb8 of its square root (finer steps in the darks) and metallic in the top byte.
    public uint albedo_metallic;
    // The hit's lobes word (packLobes: transmission, clear coat, coat roughness) and perceptual
    // roughness in the top byte.
    public uint material;
    // The path weight reaching this vertex (alpha pass-through, glass in front) as f16 rgb, and
    // the ray's layer mask above blue.
    public uint throughput_rg;
    public uint throughput_b_mask;
    // Octahedral 2xf16.
    public uint normal_oct;
    public uint geo_normal_oct;
    public uint view_oct;
    // ReSTIR GI: aerial perspective x cloud transmittance at the vertex (Sky::Atmosphere), RGB9E5,
    // written by the path tracer for gi_shade.
    public uint gi_view_t;
};

// A RestirSurface as four 16-byte stores / loads (the buffers' 64-byte records are 16-byte
// aligned; a struct access is sixteen 4-byte ones).
public void restirSurfaceStore(RestirSurface* p, RestirSurface s) {
    let d = (uint4*)p;
    storeAligned<16>(d, uint4(asuint(s.position), s.frame));
    storeAligned<16>(d + 1, uint4(asuint(s.prev_position), asuint(s.depth)));
    storeAligned<16>(d + 2, uint4(s.albedo_metallic, s.material, s.throughput_rg, s.throughput_b_mask));
    storeAligned<16>(d + 3, uint4(s.normal_oct, s.geo_normal_oct, s.view_oct, s.gi_view_t));
}

public RestirSurface restirSurfaceLoad(RestirSurface* p) {
    let d = (uint4*)p;
    let a = loadAligned<16>(d);
    let b = loadAligned<16>(d + 1);
    let c = loadAligned<16>(d + 2);
    let e = loadAligned<16>(d + 3);
    RestirSurface s;
    s.position = asfloat(a.xyz);
    s.frame = a.w;
    s.prev_position = asfloat(b.xyz);
    s.depth = asfloat(b.w);
    s.albedo_metallic = c.x;
    s.material = c.y;
    s.throughput_rg = c.z;
    s.throughput_b_mask = c.w;
    s.normal_oct = e.x;
    s.geo_normal_oct = e.y;
    s.view_oct = e.z;
    s.gi_view_t = e.w;
    return s;
}

// Where the path tracer resumes sample 0 (docs/plans/primary-pass.html), written by the
// primary pass per traced pixel. 64 bytes.
public struct PrimaryHit {
    // The last camera-chain segment (after pass-through and portals): its start and direction.
    public float3 origin;
    // kind (bits 0-1) | bounces the chain used (2-7) | ray mask (8-15) | environment (16-18).
    public uint state;
    public float3 direction;
    // Hit distance along that segment.
    public float t;
    // TLAS instance index, geometry and primitive index, barycentrics as 2x unorm16.
    public uint instance;
    public uint geometry;
    public uint primitive;
    public uint bary;
    // The ray cone's width at the hit; the exposure meter's divisor (< 0: not metered); the
    // first hit distance (0 = sky) for the atmosphere composite; the hit's face.
    public float cone_width;
    public float meter_albedo;
    public float primary_t;
    public uint hit_kind;
};

// RGB9E5 (shared exponent, nine-bit mantissas): non-negative radiance up to 65408 in one word.
public uint packRgb9e5(float3 c) {
    let rgb = clamp(c, float3(0.0), float3(65408.0));
    let maxc = max(rgb.r, max(rgb.g, rgb.b));
    var e = max(-16, int(floor(log2(max(maxc, 1.0e-30))))) + 16;
    var denom = exp2(float(e - 24));
    if (floor(maxc / denom + 0.5) >= 512.0) {
        denom *= 2.0;
        e += 1;
    }
    let m = uint3(min(floor(rgb / denom + 0.5), float3(511.0)));
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}

public float3 unpackRgb9e5(uint p) {
    let m = float3(float(p & 511u), float((p >> 9) & 511u), float((p >> 18) & 511u));
    return m * exp2(float(int(p >> 27) - 24));
}

// One radiance-cache entry (src/sharc.rs; resolved by sharc.slang). 32 bytes; resolved
// values are packed f16 pairs.
public struct SharcEntry {
    public uint tag;      // 0 = empty; hash tag of the voxel key otherwise.
    public uint frame;    // Last frame the entry was deposited into or queried.
    // Radiance accumulated this frame, fixed point at SHARC_UNIT (1/65536 nit): 64-bit so a
    // moonlit voxel keeps its precision and a sunlit one cannot overflow.
    public uint64_t acc_r;
    public uint64_t acc_g;
    public uint64_t acc_b;
    public uint acc_n;    // Samples accumulated this frame.
    public float conf;    // Samples folded into the resolved value.
    public float res_r;   // Resolved outgoing radiance (nits).
    public float res_g;
    public float res_b;
    public uint pad;
};
// ReSTIR PT's sample (src/restir_gi.rs; docs/plans/restir-pt.html), written by the path tracer per
// traced pixel whose primary vertex takes part and whose path found a reconnection vertex x_k: the
// first vertex k >= 1 where x_{k-1} and x_k are both rough and far apart. What x_k sends back is
// kept so its BSDF can be evaluated again for another incoming direction (a shifted x_{k-1}): its
// material and shading normal, and the light it received along its two directions -- its
// next-event sample (MIS-weighted, over its pdf) and its BSDF continuation (everything the rest of
// the path gathered, over the direction's pdf), each as radiance over f cos at the direction the
// path came from. Also the density the path drew x_{k-1} -> x_k with (0 = a null candidate), the
// seed its BSDF samples came from (random replay rebuilds x_1 .. x_{k-1} from another primary
// vertex), the path weight reaching x_k over the primary vertex's (T) and x_k's cosine toward
// x_{k-1} over their squared distance (the reconnection Jacobian's source half). 80 bytes.
// gi_candidates completes it in place: `pdf` becomes W, `phat` is filled.
public struct GiCandidate {
    public float3 x_s;
    // x_k's geometric normal, toward x_{k-1} (oct 2xf16).
    public uint n_s_oct;
    // RGB9E5: what the continuation sent from x_k toward the path's own direction (nits).
    public uint L;
    public float pdf;
    public uint frame;
    // Its target at its own vertex (gi_candidates, in place).
    public float phat;
    public uint seed;
    // k (bits 0-3).
    public uint k;
    // RGB9E5: the path weight at x_k over the primary vertex's.
    public uint T;
    // |cos theta_k| / d_k^2 on the path that found it.
    public float geo;
    // RGB9E5: what the next-event sample sent from x_k toward the path's own direction (nits).
    public uint L_nee;
    // The continuation's and the next-event sample's directions, x_k's shading normal (oct 2xf16).
    public uint wi_c;
    public uint wi_n;
    public uint n_k;
    // x_k's material: sqrt albedo (unorm8 x3) and metallic (unorm8); perceptual roughness.
    public uint albedo_metallic;
    public float roughness;
    // The direction x_k -> x_{k-1} on the path that found it (oct 2xf16).
    public uint wo;
    public uint pad;
};

public GiCandidate giCandidateLoad(GiCandidate* p) {
    let d = (uint4*)p;
    let a = loadAligned<16>(d);
    let b = loadAligned<16>(d + 1);
    let c = loadAligned<16>(d + 2);
    GiCandidate r;
    r.x_s = asfloat(a.xyz);
    r.n_s_oct = a.w;
    r.L = b.x;
    r.pdf = asfloat(b.y);
    r.frame = b.z;
    r.phat = asfloat(b.w);
    r.seed = c.x;
    r.k = c.y;
    r.T = c.z;
    r.geo = asfloat(c.w);
    let e = loadAligned<16>(d + 3);
    let g = loadAligned<16>(d + 4);
    r.L_nee = e.x;
    r.wi_c = e.y;
    r.wi_n = e.z;
    r.n_k = e.w;
    r.albedo_metallic = g.x;
    r.roughness = asfloat(g.y);
    r.wo = g.z;
    r.pad = 0u;
    return r;
}

public uint ptPackAlbedoMetallic(float3 albedo, float metallic) {
    let a = sqrt(saturate(albedo));
    return uint(a.r * 255.0 + 0.5) | (uint(a.g * 255.0 + 0.5) << 8) | (uint(a.b * 255.0 + 0.5) << 16)
        | (uint(saturate(metallic) * 255.0 + 0.5) << 24);
}

// ReSTIR PT's final reservoir, one per traced pixel: what gi_shade shades. The winner's
// contribution at this pixel (its target vector, RGB9E5), its unbiased contribution weight, and
// the reconnection segment to test (from the replayed x_{k-1} to x_k). 48 bytes.
public struct GiReservoir {
    public float3 from_;
    public uint C;
    public float3 to;
    // Unbiased contribution weight (gi_shade zeroes it when the winner is occluded).
    public float W;
    public uint frame;
    // GI_FRESH: the winner is this frame's own candidate (traced from here: no visibility ray).
    public uint flags;
    // The segment's end normals (oct 2xf16): the visibility ray leaves and arrives along them.
    public uint n_from;
    public uint n_to;
};

public static const uint GI_FRESH = 1u;

// AuroraDebugView::GiReuse's codes (gi_shade writes them, quad.slang colours them).
public static const uint GI_STATE_OFF = 0u;
public static const uint GI_STATE_EMPTY = 1u;
public static const uint GI_STATE_OWN_PRIVATE = 2u;
public static const uint GI_STATE_OWN = 3u;
public static const uint GI_STATE_NEIGHBOUR = 4u;
public static const uint GI_STATE_OCCLUDED = 5u;

// A sample's `n_s_oct` low bit: its secondary hit is rough enough (and the radiance it sends close
// enough to the same in every direction) to reconnect to another pixel's vertex. Without it the
// sample serves only the pixel that traced it. One ulp of a half, below the normal's precision.
public uint giPackHitNormal(float3 n, bool shareable) {
    return (packHalf2(float32x3_to_oct(n)) & ~1u) | (shareable ? 1u : 0u);
}

public bool giShareable(uint n_s_oct) {
    return (n_s_oct & 1u) != 0u;
}

public GiReservoir giReservoirLoad(GiReservoir* p) {
    let d = (uint4*)p;
    let a = loadAligned<16>(d);
    let b = loadAligned<16>(d + 1);
    let c = loadAligned<16>(d + 2);
    GiReservoir r;
    r.from_ = asfloat(a.xyz);
    r.C = a.w;
    r.to = asfloat(b.xyz);
    r.W = asfloat(b.w);
    r.frame = c.x;
    r.flags = c.y;
    r.n_from = c.z;
    r.n_to = c.w;
    return r;
}

public void giReservoirStore(GiReservoir* p, GiReservoir r) {
    let d = (uint4*)p;
    storeAligned<16>(d, uint4(asuint(r.from_), r.C));
    storeAligned<16>(d + 1, uint4(asuint(r.to), asuint(r.W)));
    storeAligned<16>(d + 2, uint4(r.frame, r.flags, r.n_from, r.n_to));
}

// `GiFrameGpu` in src/restir_gi.rs (scalar layout; pointers first): ReSTIR PT's per-view block.
public struct GiFrame {
    public GiCandidate* candidates;
    public RestirSurface* surfaces;
    public GiReservoir* final_;
    public uint64_t pad_ptr;
    public uint width;
    public uint height;
    public uint frame;
    public uint spatial_samples;
    public float spatial_radius;
    public float boiling;
    public uint pad[2];
};

// A world position's pixel in a frame with view-projection `m` (column-major) at `width` x
// `height`, or (-1, -1) behind it. The raygen's pixel centre is (x, H - 1 - y) + 0.5.
public int2 restirReproject(float* m, uint width, uint height, float3 p) {
    let c = mat4MulVecAt(m, float4(p, 1.0));
    if (c.w <= 1.0e-4) {
        return int2(-1, -1);
    }
    let uv = (c.xy / c.w) * 0.5 + 0.5;
    return int2(int(uv.x * float(width)), int(float(height) - uv.y * float(height)));
}

// Whether a neighbour / history surface (shading normal, view depth) is close enough to `s` to
// reuse from.
public bool restirSimilar(RestirSurface s, uint normal_oct, float depth) {
    let n0 = oct_to_float32x3(unpackHalf2(s.normal_oct));
    let n1 = oct_to_float32x3(unpackHalf2(normal_oct));
    return dot(n0, n1) > 0.9 && abs(depth - s.depth) < 0.1 * s.depth;
}

// PrimaryHit kinds: the path tracer traces from the camera as before; rebuilds the hit and
// shades it; adds the radiance in `origin` and ends.
public static const uint PRIMARY_RETRACE = 0u;
public static const uint PRIMARY_HIT = 1u;
public static const uint PRIMARY_ENDED = 2u;

public uint primaryKind(PrimaryHit h) {
    return h.state & 3u;
}

public uint packUnorm8(float x) {
    return uint(saturate(x) * 255.0 + 0.5);
}

public RestirSurface restirSurfacePack(float3 position, uint frame, float3 prev_position, float depth,
                                       float3 albedo, float metallic, float roughness, uint lobes,
                                       float3 throughput, uint ray_mask, float3 n, float3 ns, float3 v) {
    let a = sqrt(saturate(albedo));
    RestirSurface s;
    s.position = position;
    s.frame = frame;
    s.prev_position = prev_position;
    s.depth = depth;
    s.albedo_metallic = packUnorm8(a.r) | (packUnorm8(a.g) << 8) | (packUnorm8(a.b) << 16)
        | (packUnorm8(metallic) << 24);
    s.material = (lobes & 0xFFFFFFu) | (packUnorm8(roughness) << 24);
    s.throughput_rg = packHalf2(throughput.rg);
    s.throughput_b_mask = (f32tof16(throughput.b) & 0xFFFFu) | ((ray_mask & 0xFFu) << 16);
    s.normal_oct = packHalf2(float32x3_to_oct(n));
    s.geo_normal_oct = packHalf2(float32x3_to_oct(ns));
    s.view_oct = packHalf2(float32x3_to_oct(v));
    s.gi_view_t = 0u;
    return s;
}

public float3 restirAlbedo(RestirSurface s) {
    let a = float3(float(s.albedo_metallic & 0xFFu), float((s.albedo_metallic >> 8) & 0xFFu),
                   float((s.albedo_metallic >> 16) & 0xFFu)) / 255.0;
    return a * a;
}

public float restirMetallic(RestirSurface s) {
    return float(s.albedo_metallic >> 24) / 255.0;
}

public float restirRoughness(RestirSurface s) {
    return float(s.material >> 24) / 255.0;
}

public float3 restirThroughput(RestirSurface s) {
    return float3(unpackHalf2(s.throughput_rg), f16tof32(s.throughput_b_mask & 0xFFFFu));
}

public uint restirRayMask(RestirSurface s) {
    return (s.throughput_b_mask >> 16) & 0xFFu;
}

public struct Reservoir {
    // Winning light-table entry and its (su, sv) triangle sample, packed as 2xf16.
    public uint y;
    public uint uv_pk;
    // Unbiased contribution weight (0 after an occluded visibility ray) and history length.
    public float W;
    public float M;
    // Validation: the light-table generation, the shading normal (oct, 2xf16), the view
    // depth (clip w) and the frame that wrote the reservoir.
    public uint epoch;
    public uint normal_pk;
    public float depth;
    public uint frame;
    // The winner's target at this reservoir's own surface and the density its light's sampling
    // map put the point there with, so spatial reuse never re-evaluates them.
    public float phat;
    public float density;
};

// `slot_to_linst` values with this bit are a sphere emitter's ANALYTIC entry, not a light
// instance index (0xFFFFFFFF stays "not a light").
public static const uint LIGHT_SLOT_ANALYTIC = 0x80000000u;

// One analytic light (point / spot / rect / sphere emitter): table entries [0,
// analytic_count). Positions and orientations are world-space, refreshed from the CPU every
// frame they change. 80 bytes; must match AnalyticLightGpu in lights.rs.
public struct AnalyticLight {
    public float position_x, position_y, position_z;
    public uint kind;          // 0 point, 1 spot, 2 rect, 3 sphere emitter (an emissive `Sphere`)
    public float direction_x, direction_y, direction_z;  // spot axis / rect normal
    public float radius;       // point/spot/sphere emitter radius (0 = delta)
    public float tangent_x, tangent_y, tangent_z;        // rect: +X edge direction
    public float cos_inner;    // spot: full-intensity cone
    // point/spot: luminous intensity I (nit*m^2); rect/sphere: emitted radiance L (nits).
    public float emission_r, emission_g, emission_b;
    public float cos_outer;    // spot: falloff limit
    public float half_extents_x, half_extents_y;  // rect half sizes along tangent / bitangent
    public float power;        // CPU-computed CDF weight (same units as tri entries: flux / pi)
    public uint flags;         // bit 0: two-sided (rect); bits 8..15: the world mask it lights
};

public struct LightsHeader {
    public uint entry_count;
    public uint linst_count;
    // Written by the light_scan kernel; 0 = table not ready, sampling is skipped.
    public float total_power;
    public uint slot_map_count;
    // Entries [0, analytic_count) are analytic lights; the rest are emissive triangles.
    public uint analytic_count;
    // Bins in `guide` (light_eval.slang `lightsPickEntry`).
    public uint guide_bins;
    public float* cdf;
    public LightInstance* linsts;
    public uint* slot_to_linst;
    public AnalyticLight* analytics;
    // Per entry, its light instance (0 for analytic entries).
    public uint* entry_to_linst;
    // Per triangle entry, resolved each frame (lights.slang `light_prepare`): world vertex 0
    // and world mask, edge 1 and pick probability, edge 2, emission; four float4 each.
    public float4* prepared;
    // The CDF's guide table, guide_bins + 1 entries (lights.slang `light_scan`): bin j's first
    // entry whose cumulative power reaches j / guide_bins of the total.
    public uint* guide;
};

// Exposure state from the metering (auto_exposure.slang). The raygen applies
// `input_exposure` (quantised, held constant between rare whole-EV steps -- Ray
// Reconstruction's history needs a still input exposure); the blit maps it onto the smooth
// `exposure` (Auto look) or a fixed one.
public struct AeData {
    public float ev;
    public float exposure;
    public float input_ev;
    public float input_exposure;
    public float metered;
    public uint settled;
};

// Must match AtmosphereGpu in src/atmosphere.rs. Lengths in metres, coefficients per metre,
// radiances in nits; the planet is a sphere at `planet_center`.
public struct AtmosphereParams {
    public float4* transmittance;   // ATM_T_W x ATM_T_H
    public float4* multiscatter;    // ATM_MS_N x ATM_MS_N
    public float4* skyview_rad;     // ATM_SV_W x ATM_SV_H, radiance
    public float4* skyview_trn;     // same, transmittance to space
    public float* noise;            // CLOUD_NOISE_N^3 Perlin-Worley base shape
    public float* noise_detail;     // CLOUD_DETAIL_N^3 Worley erosion
    public float planet_center_x, planet_center_y, planet_center_z;
    public float planet_radius;
    public float atmosphere_height;
    public float rayleigh_h;
    public float mie_h;
    public float ozone_center;
    public float ozone_width;
    public float rayleigh_scatter_r, rayleigh_scatter_g, rayleigh_scatter_b;
    public float mie_scatter_r, mie_scatter_g, mie_scatter_b;
    public float mie_absorb_r, mie_absorb_g, mie_absorb_b;
    public float ozone_absorb_r, ozone_absorb_g, ozone_absorb_b;
    public float mie_g;
    public float ground_albedo_r, ground_albedo_g, ground_albedo_b;
    public float sun_direction_x, sun_direction_y, sun_direction_z;
    public float sun_radiance_r, sun_radiance_g, sun_radiance_b;  // top-of-atmosphere disc radiance
    public float sun_cos_radius;
    public float camera_pos_x, camera_pos_y, camera_pos_z;        // the sky-view LUT's viewpoint
    public uint clouds;
    public uint cloud_shadows;
    public uint cloud_steps;
    public uint cloud_light_steps;
    public float cloud_bottom;     // shell altitudes above the planet surface
    public float cloud_top;
    public float cloud_coverage;
    public float cloud_coverage_scale;
    public float cloud_density;
    public float cloud_extinction;
    public float cloud_detail;
    public float cloud_scale;
    public float cloud_detail_scale;
    public float cloud_wind_x, cloud_wind_y, cloud_wind_z;  // accumulated offset (wind velocity x time)
    public float cloud_light_dist;
    public float cloud_max_dist;
    public float cloud_ambient;
    public float cloud_forward_g;
    public float cloud_back_g;
};

// ReSTIR DI's per-view parameter block (`RestirFrameGpu` in restir.rs), shared with the raygen
// for the light grid.
public struct RestirFrame {
    public RestirSurface* surfaces;
    // Last frame's final reservoirs, this frame's temporal output, this frame's final.
    public Reservoir* prev;
    public Reservoir* temporal;
    public Reservoir* final_;
    public uint2* grid;            // (entry, asuint(W)) per cell sample; entry ~0 = none
    public LightsHeader* lights;
    public float4* instances;
    public float4* positions;      // per light-table entry, this frame (restir_positions)
    public float prev_view_proj[16];
    public float grid_origin_x;
    public float grid_origin_y;
    public float grid_origin_z;
    public float grid_cell;
    public uint grid_x;
    public uint grid_y;
    public uint grid_z;
    public uint grid_k;
    public uint width;
    public uint height;
    public uint frame;
    public uint epoch;
    public uint candidates;        // N per pixel
    public float m_cap;            // temporal history cap (frames' worth of M)
    public uint spatial_samples;
    public float spatial_radius;   // pixels
    public float boiling;          // 0 = off
    public float emissive_boost;
    public uint grid_candidates;   // R per grid sample
    public uint light_entries;     // 0 = light NEE off
    public uint ab;                // RenderSettings::ab: 0 the old path under test, 1 the new
    public uint pad_ab;
};

// The light-grid cell holding `p`, or ~0 outside the grid.
public uint gridCell(RestirFrame* f, float3 p) {
    let rel = (p - float3(f.grid_origin_x, f.grid_origin_y, f.grid_origin_z)) / f.grid_cell;
    let c = int3(floor(rel));
    if (any(c < int3(0)) || c.x >= int(f.grid_x) || c.y >= int(f.grid_y) || c.z >= int(f.grid_z)) {
        return 0xFFFFFFFFu;
    }
    return (uint(c.y) * f.grid_z + uint(c.z)) * f.grid_x + uint(c.x);
}

// `RaytracingPushConstants` in raytracing_pipeline.rs.
public struct PushConstants {
    public UniformData* uniforms;
    // Not void*: slangc lowers it to the uint* type, which then loses its ArrayStride.
    public uint64_t materials;
    public float* focus;
    // ReSTIR DI's primary surfaces (one per traced pixel; restir.rs).
    public RestirSurface* restir_surfaces;
    // Last frame's VkAccelerationStructureInstanceKHR array: 4 float4 per instance, rows
    // 0..2 are the 3x4 transform (row-major).
    public float4* prev_instances;
    // Each TLAS row's instance slot (the TLAS holds the live slots only, packed): what
    // InstanceIndex() means everywhere else.
    public uint* instance_slots;
    // Emissive-triangle light table (valid whenever uniforms.light_entries > 0).
    public LightsHeader* lights;
    // This frame's instance rows (same layout as prev_instances), for light transforms.
    public float4* cur_instances;
    // ReSTIR DI reservoirs: last frame's (read) and this frame's (written).
    public Reservoir* reservoirs_prev;
    public Reservoir* reservoirs_cur;
    // Auto-exposure: per-pixel (incident meter, raw nits) pairs out, smoothed exposure in
    // (auto_exposure.slang).
    public float* lum;
    public AeData* ae;
    // Atmosphere + cloud parameters and LUTs (valid whenever a world's mode is 3).
    public AtmosphereParams* atmo;
    // ReSTIR DI's parameter block, this frame's light grid already built (null when off):
    // path vertices past the primary draw their light candidates from it.
    public RestirFrame* restir_frame;
    // The Cost debug view (null unless shown): words 0-1 the frame's total ns (u64), then
    // each traced pixel's raygen time in ns (VK_KHR_shader_clock).
    public uint* cost;
    // The primary pass's hit records (null while the old frame runs).
    public PrimaryHit* primary_hits;
    // ReSTIR GI (restir_gi.rs; null when off): the candidates `main` writes, the final
    // reservoirs `gi_shade` shades.
    public GiCandidate* gi_candidates;
    public GiReservoir* gi_final;
    // AuroraDebugView::Gi / GiReuse (null unless shown): two header words, then per traced pixel
    // i gi_shade's light at 2 + 2i (RGB9E5, pre-exposed) and its state (GI_STATE_*) at 3 + 2i.
    public uint* gi_debug;
    // The reference accumulation (reference.rs; null unless on): per traced pixel the summed
    // radiance (nits) and the frame count.
    public float4* reference;
    // Radiance-cache entries (valid whenever uniforms.sharc != 0).
    public SharcEntry* sharc;
    // ReSTIR PT's per-view block (restir_gi.rs; null when off): the spatial pass's parameters.
    public GiFrame* gi_frame;
};
[[vk::push_constant]] public ConstantBuffer<PushConstants> pc;

// The packed payload fields. `(r_m_t_i >> 24) % 256` in GLSL is a floored modulo
// (OpSMod), i.e. the low byte: `& 0xFF` here, never Slang's truncating `%`.
public float hitPayloadGetRoughness(HitPayload p) {
    return float((p.r_m_t_i >> 24) & 0xFF) / 255.0;
}

public float hitPayloadGetMetallic(HitPayload p) {
    return float((p.r_m_t_i >> 16) & 0xFF) / 255.0;
}

public float hitPayloadGetTransmission(HitPayload p) {
    return float((p.r_m_t_i >> 8) & 0xFF) / 255.0;
}

public bool hitPayloadGetInside(HitPayload p) {
    return (p.r_m_t_i & 0x01) != 0;
}

public bool hitPayloadGetMasked(HitPayload p) {
    return (p.r_m_t_i & 0x02) != 0;
}

// Returns +/- 1
float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

// Assume normalized input. Output is on [-1, 1] for each component.
public float2 float32x3_to_oct(float3 v) {
    // Project the sphere onto the octahedron, and then onto the xy plane
    let p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
    // Reflect the folds of the lower hemisphere over the diagonals
    return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
}

public float3 oct_to_float32x3(float2 e) {
    var v = float3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0) {
        v.xy = (1.0 - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}

public uint packHalf2(float2 v) {
    return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

public float2 unpackHalf2(uint p) {
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}
