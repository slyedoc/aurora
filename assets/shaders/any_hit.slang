// Alpha-mask cutout for non-opaque instances (material.alpha_cutoff > 0): a candidate hit
// whose base-colour alpha is under the cutoff is ignored, so shadow rays pass through
// foliage cards instead of treating them as solid quads. Instances whose materials are all
// opaque carry VK_GEOMETRY_INSTANCE_FORCE_OPAQUE and never run this shader; primary rays
// trace opaque and keep the stochastic-alpha handling in the raygen.
import surface;

[shader("anyhit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let material = pc.materials[InstanceID() + GeometryIndex()];
    // Shadow rays (NEE / ReSTIR: terminate-on-first-hit + skip-closest-hit) pass through
    // transmissive surfaces -- clear glass barely blocks light, and counting it as a full
    // occluder painted black shadows under every wine glass. Absorption/Fresnel are ignored
    // (the standard shadow-ray approximation).
    if ((RayFlags() & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) != 0
        && material.specular_transmission_factor > 0.5) {
        IgnoreHit();
    }
    // Opaque geometry inside a mixed (non-force-opaque) instance: accept immediately.
    if (material.alpha_cutoff <= 0.0) {
        return;
    }
    let bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                      attribs.barycentrics.x, attribs.barycentrics.y);
    let tri = record.triangle_data[record.triangles[GeometryIndex()].v + PrimitiveIndex()];
    let uv = unpackUv(tri.uv0) * bary.x + unpackUv(tri.uv1) * bary.y + unpackUv(tri.uv2) * bary.z;
    // Level 0, which is what the opacity micromap bake assumes.
    let alpha = material.base_color_a * sampleLevel(material.base_color_texture, uv, 0.0).a;
    if (alpha < material.alpha_cutoff) {
        IgnoreHit();
    }
}
