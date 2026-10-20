// The WATER surface class (src/water.rs `WaterSurface`): a water surface mesh, shaded as the
// material's dielectric with its water's waves as its normal. Its water is the material record's
// medium row (`surface_param_index`, MATERIAL_WATER set), which it hands the raygen in the lobes
// word's top byte (row + 1).
import surface;
import water_types;

// `UniformData` fields this shader reads, as float indices; pinned by a test in
// ray_render_plugin.rs.
static const uint UNIFORM_WATER_MEDIA_FLOAT = 98;
static const uint UNIFORM_TIME_FLOAT = 113;
static const uint UNIFORM_WATER_MEDIUM_COUNT_FLOAT = 187;

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    let material = hit.material;

    payload.refract_index = material.refract_index;
    // The medium below is the water's (raygen.slang tracks it), not the glass's own.
    payload.absorption = float3(0.0);
    payload.color = materialBaseColor(material);
    payload.emission = float3(0.0);

    let world_p = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    let address = uint64_t(asuint(pc.uniforms[UNIFORM_WATER_MEDIA_FLOAT]))
        | (uint64_t(asuint(pc.uniforms[UNIFORM_WATER_MEDIA_FLOAT + 1])) << 32);
    let count = asuint(pc.uniforms[UNIFORM_WATER_MEDIUM_COUNT_FLOAT]);
    let time = pc.uniforms[UNIFORM_TIME_FLOAT];
    let row = material.surface_param_index;
    let water = (material.flags & MATERIAL_WATER) != 0u && address != 0 && row < count;

    var normal = hit.surface_normal;
    var roughness = material.roughness_factor;
    if (water) {
        let v = ((WaterMediumGpu*)address)[row];
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
    }

    surfaceWritePayload(payload, hit, normal);
    if (water) {
        payload.lobes |= min(row + 1u, 255u) << 24;
    }
    hitPayloadSetTransmission(payload, material.specular_transmission_factor);
    hitPayloadSetRoughness(payload, saturate(roughness));
    hitPayloadSetMetallic(payload, 0.0);
    hitPayloadSetMasked(payload, false);
    hitPayloadSetGuidesHere(payload, true);
}
