// What a ray that escapes the scene returns: the sky of the environment it is in (the
// raygen packs that environment into `miss_flags`; portals move a ray between them).
import rt_types;
import hit_payload;
import atmosphere_rt;

[[vk::binding(200, 1)]] Sampler2D textures[];

static const float PI = 3.14159265359;

float3 sampleSky(uint tex, float2 uv) {
    return textures[NonUniformResourceIndex(tex)].SampleLevel(uv, 0.0).rgb;
}

// Planetary atmosphere: the sky-view LUT, the sun disc through the air on paths that may see
// it, and the space image (colour = its scale) behind the atmosphere's transmittance.
float3 atmosphereSky(float3 d, bool want_sun, uint tex, float3 scale) {
    float3 L, T;
    bool ground;
    atmoSkyView(d, L, T, ground);
    if (!ground) {
        if (want_sun) {
            L += atmoSunDisc(d, T);
        }
        if (any(scale != float3(0.0))) {
            L += T * scale * min(sampleSky(tex, env_dir_to_uv(d)), float3(300.0));
        }
    }
    return L;
}

// Equirectangular lookup: texel (linear radiance) times the scale. Without an importance
// table a BRDF-sampled ray is the only way to the sun texel, so its extreme values are
// clamped RELATIVE to the sky scale (a physically bright sky passes, a 1e5x sun does not);
// with one, the raygen gathers the sun by next-event estimation and MIS-weights these hits.
float3 hdrSky(float3 d, EnvironmentGpu env) {
    var texel = skyResidual(env, d, sampleSky(env.tex, env_dir_to_uv(d)));
    if (env.importance == nullptr) {
        texel = min(texel, float3(300.0));
    }
    return env.color.rgb * texel;
}

// Zenith / horizon gradient above, horizon / ground below (all nits), with a small aureole
// around each of the environment's suns.
float3 gradientSky(float3 d, EnvironmentGpu env) {
    let up = d.y;
    var col = up >= 0.0
        ? lerp(env.horizon.rgb, env.zenith.rgb, pow(up, 0.6))
        : lerp(env.horizon.rgb, env.ground.rgb, clamp(-up * 6.0, 0.0, 1.0));
    for (uint i = 0u; i < env.sun_count; i++) {
        col += env.horizon.rgb * pow(max(dot(d, env.suns[i].direction.xyz), 0.0), 48.0) * 0.35;
    }
    return col;
}

// The environment's sun discs, on the paths that may see them directly; every other path
// gathers the suns by next-event estimation in the raygen. Full inside the radius, fading
// over its outer fifth.
float3 sunDiscs(float3 d, EnvironmentGpu env) {
    var col = float3(0.0);
    for (uint i = 0u; i < env.sun_count; i++) {
        let sun = env.suns[i];
        let cos_r = sun.direction.w;
        let disc = smoothstep(cos_r - (1.0 - cos_r) * 0.2, cos_r, dot(d, sun.direction.xyz));
        col += sun.radiance.rgb * sun.radiance.a * disc;
    }
    return col;
}

[shader("miss")]
void main(inout HitPayload payload) {
    payload.t = 0.0;
    let d = WorldRayDirection();
    let want_sun = (payload.miss_flags & 1u) != 0u;
    let env = pc.uniforms.environments[(payload.miss_flags >> 1) & 7u];
    float3 sky;
    switch (env.mode) {
    case 1u:
        sky = hdrSky(d, env);
        break;
    case 2u:
        sky = gradientSky(d, env);
        break;
    // The atmosphere draws its own (first) sun through the air.
    case 3u:
        sky = atmosphereSky(d, want_sun, env.tex, env.color.rgb);
        break;
    default:
        sky = env.color.rgb;
        break;
    }
    if (want_sun && env.mode != 3u) {
        sky += sunDiscs(d, env);
    }
    payload.emission = sky * env.sky_brightness;
}
