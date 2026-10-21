// The analytic sphere (sphere.rs): a unit-diameter sphere in object space, its intersection
// and closest-hit shaders.
import surface;

struct SphereAttributes {
    float3 point;
};

// Ray Tracing Gems ch. 7: both ray parameters of the hits, or (-1, -1).
float2 gemsIntersections(float3 orig, float3 dir, float3 center, float radius) {
    let f = orig - center;
    let a = dot(dir, dir);
    let bi = dot(-f, dir);
    let c = dot(f, f) - radius * radius;
    let s = f + (bi / a) * dir;
    let discr = radius * radius - dot(s, s);
    var t = float2(-1.0, -1.0);
    if (discr >= 0.0) {
        let q = bi + sign(bi) * sqrt(a * discr);
        t = float2(c / q, q / a);
    }
    return t;
}

[shader("intersection")]
void intersection() {
    // Shadow rays pass clear spheres (glass marbles) and those that cast none (rain), as they
    // pass such triangles (any_hit.slang `ignoredHit`).
    if ((RayFlags() & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) != 0
        && (pc.materials[InstanceID()].specular_transmission_factor > 0.5
            || (pc.materials[InstanceID()].flags & MATERIAL_NO_SHADOW) != 0u)) {
        return;
    }
    let orig = ObjectRayOrigin();
    let dir = ObjectRayDirection();
    let t = gemsIntersections(orig, dir, float3(0.0), 0.5);
    // A ray leaving the surface (a refraction into glass, a reflection off it) starts a few ulps
    // off it, closer than the quadratic's own error: its own surface came back at t ~ 0 and the
    // ray "re-entered" (rings inside a glass sphere). Roots within 1e-4 of the diameter of the
    // origin are that surface.
    let min_t = 1.0e-4 / max(length(dir), 1.0e-20);
    SphereAttributes a;
    if (t.x > min_t) {
        a.point = orig + t.x * dir;
        ReportHit(t.x, 0, a);
    }
    if (t.y > min_t) {
        a.point = orig + t.y * dir;
        ReportHit(t.y, 0, a);
    }
}

[shader("closesthit")]
void closesthit(inout HitPayload payload, SphereAttributes attribs) {
    let material = pc.materials[InstanceID()];
    var surface_normal = normalize(objectToWorldNormal(normalize(attribs.point)));
    let inside = dot(surface_normal, WorldRayDirection()) > 0.0;
    if (inside) {
        surface_normal = -surface_normal;
    }

    payload.t = RayTCurrent();
    payload.absorption = float3(0.0);
    payload.color = materialBaseColor(material);
    payload.emission = materialEmissive(material) * uniformEmissiveBoost();
    payload.refract_index = material.refract_index;
    payload.surface_and_world_normal = packNormals(surface_normal, surface_normal);
    payload.slot = 0xFFFFFFFFu;
    payload.prim_tri = 0u;
    payload.lobes = packLobes(material.diffuse_transmission, material.clearcoat,
                              material.clearcoat_roughness);
    // Where this point of the sphere was last frame: its object-space point through last frame's
    // instance transform (as surface.slang does). The current hit point here gave a moving
    // sphere no motion, and Ray Reconstruction trailed it.
    payload.prev_world_pos = prevWorldPos(attribs.point);
    hitPayloadSetTransmission(payload, material.specular_transmission_factor);
    hitPayloadSetRoughness(payload, material.roughness_factor);
    hitPayloadSetMetallic(payload, material.metallic_factor);
    hitPayloadSetInside(payload, inside);
    hitPayloadSetMasked(payload, false);
}
