// The raygen's view of the frame (types.glsl + common.glsl in Slang): the push-constant
// block and every buffer it points at.
//
// Everything read through a pointer is declared with scalars (or 16-byte-aligned vectors at
// 16-byte offsets) only, so its byte layout is the GLSL side's `scalar` layout with no
// vector-alignment question. Matrices are column-major float[16] (glam / GLSL order).
module rt_types;

__exported import hit_payload;

public static const float PI = 3.141592653589793;

// Column-major 4x4 matrix times a column vector (GLSL `mat4 * vec4`).
public float4 mat4MulVec(float m[16], float4 v) {
    return float4(m[0], m[1], m[2], m[3]) * v.x + float4(m[4], m[5], m[6], m[7]) * v.y
        + float4(m[8], m[9], m[10], m[11]) * v.z + float4(m[12], m[13], m[14], m[15]) * v.w;
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

// One world's sky and suns (src/sky.rs EnvironmentGpu). mode 0 = flat colour (color),
// 1 = HDR (tex, color = scale), 2 = gradient (zenith/horizon/ground), 3 = atmosphere (tex =
// the space image, color = its scale, 0 = none).
public struct EnvironmentGpu {
    public uint mode;
    public uint tex;
    public uint sun_count;
    public uint pad;
    public float4 color;
    public float4 zenith;
    public float4 horizon;
    public float4 ground;
    public EnvironmentSun suns[4];
};

// The world a ray mask is in: its lowest set bit (the main world for an empty mask).
public uint environmentOf(uint mask) {
    return mask == 0u ? 0u : firstbitlow(mask);
}

// `UniformData` in types.glsl / ray_render_plugin.rs (offsets pinned by a test there).
public struct UniformData {
    public float skycolor_r, skycolor_g, skycolor_b, skycolor_a;
    public float inverse_view[16];
    public float inverse_projection[16];
    public uint pull_focus_x;
    public uint pull_focus_y;
    public float aperture;
    public float foginess;
    public float fog_scatter;
    public float sky_brightness;
    // DLSS: camera matrices for depth / motion vectors, and the sub-pixel jitter (pixels).
    public float view[16];
    public float view_proj[16];
    public float prev_view_proj[16];
    public float jitter_x, jitter_y;
    // Free-running frame counter (the RNG seed -- fresh noise every frame for the denoiser).
    public uint frame;
    // Firefly suppression: indirect path contributions are clamped to this multiple of the
    // metered mid-gray luminance (0 = off).
    public float firefly_clamp;
    // Paths per pixel this frame and their maximum length.
    public uint samples;
    public uint max_bounces;
    // Post-process vignette strength (0 = off).
    public float vignette;
    // Entries in the emissive-triangle light table (0 = no light NEE / MIS).
    public uint light_entries;
    // ReSTIR DI initial candidates per pixel (0 = plain 1-sample NEE at the primary vertex).
    public uint restir_candidates;
    // Light candidates resampled per shading point outside ReSTIR (1 = one plain CDF sample).
    public uint light_candidates;
    // Added to every ray-cone texture LOD: log2(render / output width) plus the panel's bias.
    public float lod_bias;
    // Perceptual roughness up to which the primary vertex traces a specular hit-distance ray
    // for Ray Reconstruction; rougher surfaces report 0 (0 = no guide rays at all).
    public float spec_hit_roughness;
    // Light-table generation; reservoirs from another generation are dropped.
    public uint light_epoch;
    // Cap on temporal history, in candidate-samples.
    public float restir_m_clamp;
    // Radiance cache: 0 = off; else paths may terminate into it.
    public uint sharc;
    // Base voxel size (meters) at the camera; doubles with each distance octave past 8m.
    public float sharc_voxel;
    // Environment importance sampling for the HDR sky (env_w = 0: none): cdf[N + 1] then
    // the solid-angle pdf of each texel, N = env_w * env_h (src/env_light.rs).
    public float* env;
    public uint env_w;
    public uint env_h;
    // Ray portals (src/portal.rs): pair table (x = instance slot, y = target slot, z = valid)
    // + entry count (0 = none).
    public uint4* portals;
    public uint portal_count;
    // Uniform multiplier on every light's emission (emissive surfaces + analytic lights).
    // Uniform across lights, so the light table's relative weights/CDF stay valid.
    public float emissive_boost;
    // The camera's render-layer cull mask; every ray starts with it (portals swap it).
    public uint camera_mask;
    public uint pad_worlds;
    // Every world's sky and suns, by world bit: a ray reads the entry of the world its mask
    // says it is in (portals swap the mask mid-path).
    public EnvironmentGpu* worlds;
    // Terrain brush ring (the closest-hit draws it); not read here.
    public float brush_center_x, brush_center_y;
    public float brush_radius;
    public uint brush_active;
    public float brush_color_r, brush_color_g, brush_color_b;
    // Height fog (sky.rs `Fog`): base height and falloff, metres (0 falloff = uniform).
    public float fog_height;
    public float fog_falloff;
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

// One radiance-cache entry (src/sharc.rs; resolved by sharc.slang). 32 bytes; resolved
// values are packed f16 pairs.
public struct SharcEntry {
    public uint tag;      // 0 = empty; hash tag of the voxel key otherwise.
    public uint frame;    // Last frame the entry was deposited into or queried.
    public uint acc_r;    // Fixed-point (1 nit) radiance accumulated this frame.
    public uint acc_g;
    public uint acc_b;
    public uint acc_n;    // Samples accumulated this frame.
    public uint res_rg;   // Resolved radiance, 2xf16 (r, g).
    public uint res_bc;   // Resolved radiance b + confidence (samples folded in), 2xf16.
};

// One ReSTIR DI reservoir (src/restir.rs allocates one per traced pixel). 32 bytes.
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
    public uint pad1;
    public float* cdf;
    public LightInstance* linsts;
    public uint* slot_to_linst;
    public AnalyticLight* analytics;
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

// `PushConstants` in types.glsl.
public struct PushConstants {
    public UniformData* uniforms;
    public void* materials;
    // 64 bluenoise slices of 128x128 byte pairs.
    public uint* bluenoise;
    public float* focus;
    public uint skydome;
    public uint pad0;
    // Last frame's VkAccelerationStructureInstanceKHR array: 4 float4 per instance, rows
    // 0..2 are the 3x4 transform (row-major).
    public float4* prev_instances;
    // Emissive-triangle light table (valid whenever uniforms.light_entries > 0).
    public LightsHeader* lights;
    // This frame's instance rows (same layout as prev_instances), for light transforms.
    public float4* cur_instances;
    // ReSTIR DI reservoirs: last frame's (read) and this frame's (written).
    public Reservoir* reservoirs_prev;
    public Reservoir* reservoirs_cur;
    // Radiance-cache entries (valid whenever uniforms.sharc != 0).
    public SharcEntry* sharc;
    // Auto-exposure: per-pixel (incident meter, raw nits) pairs out, smoothed exposure in
    // (auto_exposure.slang).
    public float* lum;
    public AeData* ae;
    // Atmosphere + cloud parameters and LUTs (valid whenever a world's mode is 3).
    public AtmosphereParams* atmo;
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
