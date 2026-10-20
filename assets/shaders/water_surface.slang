// The WATER surface class (src/water_volume.rs `WaterSurface`): a water volume's surface mesh,
// shaded as the material's dielectric with the volume's waves as its normal. The hit finds its
// volume by position (the box whose top face it lies on); a surface over none stays flat.
import surface;
import water_types;

// `UniformData` fields this shader reads, as float indices; pinned by a test in
// ray_render_plugin.rs.
static const uint UNIFORM_WATER_VOLUMES_FLOAT = 98;
static const uint UNIFORM_TIME_FLOAT = 113;
static const uint UNIFORM_WATER_VOLUME_COUNT_FLOAT = 187;

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    let material = hit.material;

    payload.refract_index = material.refract_index;
    payload.absorption = float3(material.absorption_r, material.absorption_g, material.absorption_b);
    payload.color = materialBaseColor(material);
    payload.emission = float3(0.0);

    let world_p = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    let address = uint64_t(asuint(pc.uniforms[UNIFORM_WATER_VOLUMES_FLOAT]))
        | (uint64_t(asuint(pc.uniforms[UNIFORM_WATER_VOLUMES_FLOAT + 1])) << 32);
    let count = asuint(pc.uniforms[UNIFORM_WATER_VOLUME_COUNT_FLOAT]);
    let time = pc.uniforms[UNIFORM_TIME_FLOAT];

    var normal = hit.surface_normal;
    var roughness = material.roughness_factor;
    if (address != 0 && count > 0u) {
        let volumes = (WaterVolumeGpu*)address;
        for (uint i = 0u; i < count; i++) {
            let v = volumes[i];
            let p = waterLocal(v, float4(world_p, 1.0));
            if (abs(p.x) > 1.001 || abs(p.z) > 1.001 || abs(p.y - 1.0) > 0.05) {
                continue;
            }
            float slope_var;
            float2 slope;
            if (v.wave_tex0 != 0u) {
                // The FFT cascades (water_waves.rs), each read over the footprint's ellipse in its
                // tile: the slope, and E[s^2] - E[s]^2 over what the footprint covers.
                slope = float2(0.0);
                slope_var = 0.0;
                let tiles = float2(v.wave_tile, v.wave_tile / WATER_CASCADE_RATIO);
                let texs = uint2(v.wave_tex0, v.wave_tex1);
                for (uint c = 0u; c < 2u; c++) {
                    let a1 = hit.axis1.xz / tiles[c];
                    let a2 = hit.axis2.xz / tiles[c];
                    let t = sampleGrad(texs[c], world_p.xz / tiles[c], a1, a2);
                    slope += t.xy;
                    // Only a footprint wider than a texel hides waves. Magnified, bilinear
                    // filtering's E[s^2] - E[s]^2 is t(1 - t)(s1 - s0)^2 between texels: a
                    // roughness that peaks between texels and vanishes on them, a grid.
                    let texels = max(length(a1), length(a2)) * float(WATER_WAVE_N);
                    let hidden = saturate(log2(max(texels, 1.0e-6)));
                    slope_var += hidden * (max(t.z - t.x * t.x, 0.0) + max(t.w - t.y * t.y, 0.0));
                }
            } else {
                slope = waterWaveSlope(v, world_p.xz, time, coneWidth(payload), slope_var);
            }
            let up = normalize(float3(-slope.x, 1.0, -slope.y));
            normal = hit.inside ? -up : up;
            // A wave tilted far enough mirrors the view below the surface it lies on, which no
            // path may leave by (the far side of a crest hides that light): bend it back toward
            // the face until the mirror direction clears it.
            let d = WorldRayDirection();
            let g = hit.surface_normal;
            for (uint k = 0u; k < 6u && dot(reflect(d, normal), g) < 0.02; k++) {
                normal = normalize(normal + g * 0.5);
            }
            // Beckmann slope variance as GGX alpha^2 ~ 2 sigma^2, on top of the material's own.
            let alpha = roughness * roughness;
            roughness = sqrt(sqrt(alpha * alpha + 2.0 * slope_var));
            break;
        }
    }

    surfaceWritePayload(payload, hit, normal);
    hitPayloadSetTransmission(payload, material.specular_transmission_factor);
    hitPayloadSetRoughness(payload, saturate(roughness));
    hitPayloadSetMetallic(payload, 0.0);
    hitPayloadSetMasked(payload, false);
    hitPayloadSetGuidesHere(payload, true);
}
