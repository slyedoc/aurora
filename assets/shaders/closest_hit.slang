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
    payload.color = base * sampleGrad(material.base_color_texture, hit.uv, hit.duv1, hit.duv2);
    if (material.alpha_cutoff > 0.0) {
        // A cutout's coverage comes from level 0, like the any-hit test: a blurred alpha would
        // thin foliage out with distance. Binary at the cutoff, the any-hit's own test: the
        // raygen's pass-through coin on a fractional edge alpha flipped leaf edges between the leaf
        // and what lies behind it every frame, guides and all, and Ray Reconstruction boiled them
        // (worst against the sky). The DLSS jitter anti-aliases the hard edge instead.
        let coverage = base.a * sampleLevel(material.base_color_texture, hit.uv, 0.0).a;
        payload.color.a = coverage >= material.alpha_cutoff ? 1.0 : 0.0;
    }
    payload.emission = materialEmissive(material)
        * sampleGrad(material.base_emissive_texture, hit.uv, hit.duv1, hit.duv2).rgb
        * uniformEmissiveBoost();
    // Unlit: an emitter everywhere, its dark texels too, so the path ends here (no gloss).
    if ((material.flags & MATERIAL_UNLIT) != 0u) {
        payload.emission = max(payload.emission, float3(1.0e-6));
    }

    let transmission = material.specular_transmission_factor
        * sampleGrad(material.specular_transmission_texture, hit.uv, hit.duv1, hit.duv2).r;
    let mr = sampleGrad(material.metallic_roughness_texture, hit.uv, hit.duv1, hit.duv2);
    let roughness = material.roughness_factor * mr.g;
    let metallic = material.metallic_factor * mr.b;

    surfaceWritePayload(payload, hit, surfaceShadingNormal(hit));
    hitPayloadSetTransmission(payload, transmission);
    hitPayloadSetRoughness(payload, roughness);
    hitPayloadSetMetallic(payload, metallic);
    // A 2-state opacity micromap already decided this hit (record flag bit 1, sbt.rs): the
    // texel's alpha is no coin to flip, the leaf is a surface like any other (ReSTIR PT replays
    // paths through it).
    hitPayloadSetMasked(payload, material.alpha_cutoff > 0.0 && (record.record_flags.x & 2u) == 0u);
}
