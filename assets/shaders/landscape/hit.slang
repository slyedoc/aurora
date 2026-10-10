// The landscape surface class: tiles shade from the evaluated material pages. The hit's
// landscape-local position finds its texels; their top-4 sets blend bilinearly into one set,
// whose four heaviest palette materials are sampled and mixed (aurora_landscape shading.rs).
// A hit shader has no derivatives; every read is at the ray-cone level.
import surface;
import materials;
import pages;
import sphere;

static const uint NO_TEXTURE = 0xFFFFFFFFu;

// `PaletteGpu` in shading.rs.
struct PaletteEntry {
    uint albedo;
    uint normal;
    float tile_size;
    float roughness;
    float clearcoat;
    uint pad0;
    uint pad1;
    uint pad2;
};

// `LandscapeRow` in shading.rs (64 bytes).
struct LandscapeRow {
    uint2* materials;
    PaletteEntry* palette;
    int min_x;
    int min_z;
    int pages_x;
    int pages_z;
    float texel;
    float origin_x;
    float origin_y;
    float origin_z;
    uint palette_len;
    float roughness;
    // Planet faces: the cube face and radius (0 = flat; origin = the planet's centre).
    uint face;
    float radius;
};

MatSet texel_set(LandscapeRow r, int2 t) {
    return mat_unpack(r.materials[texel_index(r.min_x, r.min_z, r.pages_x, r.pages_z, t.x, t.y)]);
}

[shader("closesthit")]
void main(inout HitPayload payload, BuiltInTriangleIntersectionAttributes attribs) {
    let hit = surfaceHit(payload, attribs);
    payload.refract_index = hit.material.refract_index;
    payload.absorption = float3(hit.material.absorption_r, hit.material.absorption_g,
                                hit.material.absorption_b);

    var albedo = float3(hit.material.base_color_r, hit.material.base_color_g,
                        hit.material.base_color_b);
    var roughness = hit.material.roughness_factor;
    var normal = hit.surface_normal;
    var clearcoat = 0.0;

    let address = surfaceData();
    if (address != 0) {
        let r = ((LandscapeRow*)address)[hit.material.surface_param_index];
        if (r.materials != nullptr && r.palette_len > 0) {
            let world = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
            var local = world - float3(r.origin_x, r.origin_y, r.origin_z);
            if (r.radius > 0.0) {
                let xz = face_coords(r.face, r.radius, normalize(local));
                local = float3(xz.x, length(local) - r.radius, xz.y);
            }
            let g = local.xz / r.texel;
            let f0 = floor(g);
            let f = g - f0;
            let t = int2(f0);
            var s = mat_empty();
            mat_merge(s, texel_set(r, t), (1.0 - f.x) * (1.0 - f.y));
            mat_merge(s, texel_set(r, t + int2(1, 0)), f.x * (1.0 - f.y));
            mat_merge(s, texel_set(r, t + int2(0, 1)), (1.0 - f.x) * f.y);
            mat_merge(s, texel_set(r, t + int2(1, 1)), f.x * f.y);

            // Ray-cone footprint in metres; each material adds its texels per metre.
            let incidence = max(abs(dot(hit.surface_normal, WorldRayDirection())), 0.1);
            let footprint = coneWidth(payload) / incidence;

            // No tangents on the tiles: +X projected onto the normal.
            let n = hit.surface_normal;
            let reference = abs(n.x) > 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            let tangent = normalize(reference - n * dot(reference, n));
            let bitangent = cross(tangent, n);

            var a = float3(0.0);
            var rough = 0.0;
            var coat = 0.0;
            var tn = float3(0.0);
            var total = 0.0;
            for (int i = 0; i < 4; i++) {
                let w = s.w[i];
                if (w <= 1.0e-3) {
                    continue;
                }
                let e = r.palette[min(s.ids[i], r.palette_len - 1)];
                let uv = local.xz / max(e.tile_size, 1.0e-3);
                let lod = log2(footprint / max(e.tile_size, 1.0e-3)) + uniformLodBias();
                if (e.albedo != NO_TEXTURE) {
                    a += sampleLod(e.albedo, uv, lod).rgb * w;
                } else {
                    a += float3(0.5) * w;
                }
                tn += (e.normal != NO_TEXTURE ? normalTexel(sampleLod(e.normal, uv, lod))
                                              : float3(0.0, 0.0, 1.0)) * w;
                rough += e.roughness * w;
                coat += e.clearcoat * w;
                total += w;
            }
            if (total > 0.0) {
                albedo = a / total;
                roughness = rough / total;
                clearcoat = coat / total;
                let m = normalize(tn);
                normal = normalize(tangent * m.x + bitangent * m.y + n * m.z);
            }
        }
    }

    payload.color = float4(albedo, 1.0);
    payload.emission = float3(0.0);
    surfaceWritePayload(payload, hit, normal);
    hitPayloadSetTransmission(payload, 0.0);
    hitPayloadSetRoughness(payload, saturate(roughness));
    hitPayloadSetMetallic(payload, 0.0);
    hitPayloadSetMasked(payload, false);
    // A wet film over the blend (palette clear coat), smooth.
    payload.lobes = packLobes(0.0, clearcoat, 0.15);
}
