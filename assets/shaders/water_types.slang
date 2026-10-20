// A water volume's table row and its surface's waves, shared by the raygen (rt_types, water.slang)
// and the water surface's closest-hit (water_surface.slang), which has no `rt_types`.
module water_types;

// One water volume (src/water_volume.rs `GpuWaterVolume`): world to unit-local rows, the medium
// (absorption and scattering per metre, Henyey-Greenstein g), the surface's index of refraction
// and its waves (direction of travel, longest wavelength in metres, steepness k A, speed scale).
public struct WaterVolumeGpu {
    public float4 row0;
    public float4 row1;
    public float4 row2;
    public float absorption_r, absorption_g, absorption_b;
    public float ior;
    public float scattering_r, scattering_g, scattering_b;
    public float anisotropy;
    public float wave_dir_x, wave_dir_z;
    public float wavelength;
    public float steepness;
    public float wave_speed;
    // The FFT cascades' textures (bindless, 0: none) and the first cascade's tile (metres).
    public uint wave_tex0, wave_tex1;
    public float wave_tile;
};

// The first cascade's tile over the second (water_waves.rs `CASCADE_RATIO`).
public static const float WATER_CASCADE_RATIO = 6.7;
// Texels along a cascade's side (water_waves.rs `WAVE_N`).
public static const uint WATER_WAVE_N = 256u;

public float3 waterLocal(WaterVolumeGpu v, float4 p) {
    return float3(dot(v.row0, p), dot(v.row1, p), dot(v.row2, p));
}

static const uint WATER_WAVES = 12u;
static const float WATER_G = 9.81;

// The surface's slope (dh/dx, dh/dz) at world `xz` and `time`: WATER_WAVES sines, each 0.73x
// the last's wavelength, their directions fanned about the wind by a fixed golden-angle spread,
// each moving at deep water's speed for its wavelength (omega^2 = g k) and as steep as the
// volume's steepness. A wave shorter than `footprint` (the ray cone's width there) cannot be
// resolved by one sample: it fades out of the slope into `slope_var`, the slope variance the
// surface's roughness takes instead (LEAN mapping's idea: unresolved bumps are microfacets).
public float2 waterWaveSlope(WaterVolumeGpu v, float2 xz, float time, float footprint,
                             out float slope_var) {
    slope_var = 0.0;
    var slope = float2(0.0);
    if (v.steepness <= 0.0 || v.wavelength <= 0.0) {
        return slope;
    }
    let wind = normalize(float2(v.wave_dir_x, v.wave_dir_z) + float2(1.0e-6, 0.0));
    let base = atan2(wind.y, wind.x);
    var lambda = v.wavelength;
    // Shorter waves are each a little less steep, so the sum's slope stays near `steepness`.
    var steep = v.steepness * 0.5;
    for (uint i = 0u; i < WATER_WAVES; i++) {
        // Fanned about the wind, no two directions in a simple ratio (a regular fan beats into a
        // visible grid).
        let spread = (frac(float(i) * 0.618034 + 0.3) - 0.5) * 1.6 + 0.25 * sin(float(i) * 2.3);
        let a = base + spread;
        let dir = float2(cos(a), sin(a));
        let k = 2.0 * 3.14159265 / lambda;
        let omega = sqrt(WATER_G * k) * v.wave_speed;
        // Steepness k A: the wave's peak slope.
        let peak = steep;
        let phase = k * dot(dir, xz) - omega * time + float(i) * 1.7;
        // Resolved when the footprint spans under a quarter of a wavelength.
        let resolved = saturate(2.0 - 8.0 * footprint / lambda);
        slope += dir * (peak * cos(phase) * resolved);
        slope_var += 0.5 * peak * peak * (1.0 - resolved * resolved);
        lambda *= 0.73 + 0.04 * sin(float(i) * 1.3);
        steep *= 0.9;
    }
    return slope;
}
