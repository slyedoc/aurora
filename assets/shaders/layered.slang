// A LAYERED surface: a second PBR set over the base on the faces that point up, and a detail
// set multiplied over the result. Both extra sets are sampled TRIPLANAR in world space, so
// they hold their scale across a mesh whose own UVs were laid out for the base.
//
// The blend law is `jackdaw_surface::LayerBlend::weight`, expression for expression; that
// function is the definition and carries the tests.
import surface;

// `LayeredParams` in surface_group.rs, field for field (a test there reads this struct).
// Scalars only: it is read through a pointer.
struct Layered {
    float layer_color_r;
    float layer_color_g;
    float layer_color_b;
    float layer_color_a;
    float detail_color_r;
    float detail_color_g;
    float detail_color_b;
    float detail_color_a;
    float layer_uv_scale;
    float layer_normal_strength;
    float layer_metallic;
    float layer_perceptual_roughness;
    float detail_uv_scale;
    float detail_normal_strength;
    float blend_amount;
    float blend_power;
    float blend_threshold;
    float blend_position;
    float blend_contrast;
    // Set when the mask is painted into a vertex colour channel, which this back end has no
    // vertex stream for: the shader falls back to the slope so the surface still reads.
    uint use_vertex_color;
    uint vertex_color_channel;
    // `LAYER_NORMAL_MAP` / `DETAIL_NORMAL_MAP`: an unbound colour or ORM map falls back to
    // white, which is what a set without one should contribute, but white read as a normal
    // is a slant.
    uint flags;
    uint layer_base_color_texture;
    uint layer_normal_map_texture;
    uint layer_orm_texture;
    uint detail_base_color_texture;
    uint detail_normal_map_texture;
    uint detail_orm_texture;
};

static const uint LAYER_NORMAL_MAP = 1u;
static const uint DETAIL_NORMAL_MAP = 2u;
static const float MIN_THRESHOLD_EXPONENT = 0.001;

// Weights for the three projections, normalized so the set contributes once.
float3 triplanarWeights(float3 n) {
    let w = max(abs(n) - 0.2, float3(0.0));
    let sum = w.x + w.y + w.z;
    return sum > 1.0e-5 ? w / sum : float3(0.0, 1.0, 0.0);
}

float4 triplanar(uint tex, float3 p, float3 w, float lod) {
    return w.x * sampleLod(tex, p.yz, lod) + w.y * sampleLod(tex, p.xz, lod)
        + w.z * sampleLod(tex, p.xy, lod);
}

// Whiteout blend: each projection's tangent normal is lifted into world space by swizzling
// to that plane's axes, then summed. Cheaper than three TBNs and stable at the seams.
float3 triplanarNormal(uint tex, float3 p, float3 w, float3 n, float lod, float strength) {
    let sx = normalTexel(sampleLod(tex, p.yz, lod));
    let sy = normalTexel(sampleLod(tex, p.xz, lod));
    let sz = normalTexel(sampleLod(tex, p.xy, lod));
    let axis = sign(n);
    let nx = float3(sx.z * axis.x, sx.y, sx.x);
    let ny = float3(sy.x, sy.z * axis.y, sy.y);
    let nz = float3(sz.x, sz.y * axis.z, sz.z);
    let blended = normalize(nx * w.x + ny * w.y + nz * w.z);
    return normalize(lerp(n, blended, clamp(strength, 0.0, 1.0)));
}

// `LayerBlend::weight`. `up` is the world normal's Y.
float layerWeight(Layered p, float up) {
    let lifted = clamp(up + p.blend_power, 0.0, 1.0);
    let exponent = MIN_THRESHOLD_EXPONENT + p.blend_threshold * (1.0 - MIN_THRESHOLD_EXPONENT);
    return p.blend_amount * pow(abs(lifted), exponent);
}

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    let material = hit.material;

    payload.refract_index = material.refract_index;
    payload.absorption = float3(material.absorption_r, material.absorption_g, material.absorption_b);

    var albedo = materialBaseColor(material)
        * sampleGrad(material.base_color_texture, hit.uv, hit.duv1, hit.duv2);
    var roughness = material.roughness_factor;
    var metallic = material.metallic_factor;
    var normal = surfaceShadingNormal(hit);

    // A class that published no parameter buffer shades as plain PBR rather than reading
    // address 0. `surfaceData` is zero exactly then (`SurfaceGroupData::get`).
    let data_address = surfaceData();
    if (data_address != 0) {
        let p = ((Layered*)data_address)[material.surface_param_index];

        // Slope from the GEOMETRIC normal, not the shading normal: a normal map should change
        // how the surface catches light, not which layer it is made of.
        let weight = layerWeight(p, clamp(hit.surface_normal.y, -1.0, 1.0));
        let world_p = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
        let w = triplanarWeights(hit.surface_normal);

        if (weight > 0.0) {
            let lp = world_p * p.layer_uv_scale;
            let layer_color = float4(p.layer_color_r, p.layer_color_g, p.layer_color_b, p.layer_color_a);
            let layer = layer_color * triplanar(p.layer_base_color_texture, lp, w, hit.lod_base);
            // Occlusion red, roughness green, metallic blue -- the channel order glTF writes.
            let orm = triplanar(p.layer_orm_texture, lp, w, hit.lod_base).rgb;
            albedo = lerp(albedo, layer, weight);
            roughness = lerp(roughness, p.layer_perceptual_roughness * orm.g, weight);
            metallic = lerp(metallic, p.layer_metallic * orm.b, weight);
            if ((p.flags & LAYER_NORMAL_MAP) != 0u) {
                let ln = triplanarNormal(p.layer_normal_map_texture, lp, w, hit.surface_normal,
                                         hit.lod_base, p.layer_normal_strength);
                normal = normalize(lerp(normal, ln, weight));
            }
        }

        // The detail set multiplies over base and layer alike, so it is not part of the blend.
        let dp = world_p * p.detail_uv_scale;
        let detail_color = float4(p.detail_color_r, p.detail_color_g, p.detail_color_b, p.detail_color_a);
        albedo *= detail_color * triplanar(p.detail_base_color_texture, dp, w, hit.lod_base);
        if ((p.flags & DETAIL_NORMAL_MAP) != 0u) {
            normal = triplanarNormal(p.detail_normal_map_texture, dp, w, normal, hit.lod_base,
                                     p.detail_normal_strength);
        }
    }

    payload.color = albedo;
    payload.emission = materialEmissive(material)
        * sampleGrad(material.base_emissive_texture, hit.uv, hit.duv1, hit.duv2).rgb
        * uniformEmissiveBoost();

    let transmission = material.specular_transmission_factor
        * sampleGrad(material.specular_transmission_texture, hit.uv, hit.duv1, hit.duv2).r;

    surfaceWritePayload(payload, hit, normal);
    hitPayloadSetTransmission(payload, transmission);
    hitPayloadSetRoughness(payload, clamp(roughness, 0.0, 1.0));
    hitPayloadSetMetallic(payload, clamp(metallic, 0.0, 1.0));
    hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0);
}
