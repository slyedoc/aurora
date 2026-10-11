// The OPAQUE surface class (`SurfaceClass::OPAQUE`): standard PBR out of the material record.
// Every other class is a sibling of this file -- same `surfaceHit` and `surfaceWritePayload`,
// a different middle.
import surface;

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    let material = hit.material;

    payload.refract_index = material.refract_index;
    payload.absorption = float3(material.absorption_r, material.absorption_g, material.absorption_b);

    let base = materialBaseColor(material);
    payload.color = base * sampleLod(material.base_color_texture, hit.uv, hit.lod_base);
    if (material.alpha_cutoff > 0.0) {
        // A cutout's coverage comes from level 0, like the any-hit test: a blurred alpha would
        // thin foliage out with distance.
        payload.color.a = base.a * sampleLevel(material.base_color_texture, hit.uv, 0.0).a;
    }
    payload.emission = materialEmissive(material)
        * sampleLod(material.base_emissive_texture, hit.uv, hit.lod_base).rgb
        * uniformEmissiveBoost();

    let transmission = material.specular_transmission_factor
        * sampleLod(material.specular_transmission_texture, hit.uv, hit.lod_base).r;
    let mr = sampleLod(material.metallic_roughness_texture, hit.uv, hit.lod_base);
    let roughness = material.roughness_factor * mr.g;
    let metallic = material.metallic_factor * mr.b;

    surfaceWritePayload(payload, hit, surfaceShadingNormal(hit));
    hitPayloadSetTransmission(payload, transmission);
    hitPayloadSetRoughness(payload, roughness);
    hitPayloadSetMetallic(payload, metallic);
    hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0);
}
