// GPU linear-blend skinning for ray-traced skinned meshes (see src/skinning.rs).
//
// A skinned instance owns a deformed copy of its mesh's vertex stream and a per-instance
// BLAS that is refit from it every frame. Two kernels per instance:
//
//   skin_vertices   one thread per vertex: blends the 4 influencing joints' skin matrices
//                   (joint world from the GPU transform table * inverse bind pose), moves the
//                   rest-pose position / normal, then pre-multiplies the inverse of the
//                   instance's own world so the output is MESH-LOCAL -- the TLAS places the
//                   instance with its ordinary transform-table row, exactly like a static mesh.
//   pack_triangles  one thread per triangle: rebuilds the packed per-triangle shading record
//                   (`Triangle` in blas.rs / types.glsl) from the deformed vertices, so the
//                   closest-hit's normals and tangents follow the deformation.
//
// Joint worlds come from the `World` table (transform.slang): the frontier there expands every
// changed node to all its descendants, so a bone whose parent animates but whose own local is
// static is recomputed too. Every buffer arrives as a raw pointer in push constants.

static const uint NO_NODE = 0xffffffff;

// Rest / deformed vertex stream: `Vertex` in blas.rs (32 bytes, scalar layout).
struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

// Per-vertex joint influences: 4 x u16 joint indices packed in two uints, 4 weights (32 bytes,
// weights 16-aligned -- `SkinVertex` in blas.rs).
struct SkinVertex {
    uint2 joints;
    uint2 pad;
    float4 weights;
};

// A node's world affine: three rows of a row-major 3x4 (`NodeWorld` in gpu_transform.rs).
struct World {
    float4 r0;
    float4 r1;
    float4 r2;
};

// Per-triangle shading record: `Triangle` in blas.rs (32 bytes).
struct Triangle {
    uint tangent;
    uint n0, n1, n2;
    uint uv0, uv1, uv2;
    float lod;
};

// ---- affine helpers ---------------------------------------------------------------------

float3 affine_point(World m, float3 p) {
    return float3(dot(m.r0.xyz, p) + m.r0.w, dot(m.r1.xyz, p) + m.r1.w, dot(m.r2.xyz, p) + m.r2.w);
}

float3 affine_dir(World m, float3 v) {
    return float3(dot(m.r0.xyz, v), dot(m.r1.xyz, v), dot(m.r2.xyz, v));
}

// Normal through the cofactor (inverse-transpose) of the linear part: a blend of skin
// matrices shears, and only the cofactor keeps the normal perpendicular to the surface.
float3 affine_normal(World m, float3 n) {
    let c0 = float3(m.r0.x, m.r1.x, m.r2.x);
    let c1 = float3(m.r0.y, m.r1.y, m.r2.y);
    let c2 = float3(m.r0.z, m.r1.z, m.r2.z);
    return cross(c1, c2) * n.x + cross(c2, c0) * n.y + cross(c0, c1) * n.z;
}

World affine_inverse(World m) {
    let r0 = float3(m.r0.x, m.r1.x, m.r2.x);
    let r1 = float3(m.r0.y, m.r1.y, m.r2.y);
    let r2 = float3(m.r0.z, m.r1.z, m.r2.z);
    let t = float3(m.r0.w, m.r1.w, m.r2.w);
    let det = dot(r0, cross(r1, r2));
    let inv_det = abs(det) > 1e-12 ? 1.0 / det : 0.0;
    let i0 = cross(r1, r2) * inv_det;
    let i1 = cross(r2, r0) * inv_det;
    let i2 = cross(r0, r1) * inv_det;
    World r;
    r.r0 = float4(i0, -dot(i0, t));
    r.r1 = float4(i1, -dot(i1, t));
    r.r2 = float4(i2, -dot(i2, t));
    return r;
}

// a ∘ b: apply b, then a.
World affine_mul(World a, World b) {
    let bx = float3(b.r0.x, b.r1.x, b.r2.x);
    let by = float3(b.r0.y, b.r1.y, b.r2.y);
    let bz = float3(b.r0.z, b.r1.z, b.r2.z);
    let bt = float3(b.r0.w, b.r1.w, b.r2.w);
    World r;
    r.r0 = float4(dot(a.r0.xyz, bx), dot(a.r0.xyz, by), dot(a.r0.xyz, bz), dot(a.r0.xyz, bt) + a.r0.w);
    r.r1 = float4(dot(a.r1.xyz, bx), dot(a.r1.xyz, by), dot(a.r1.xyz, bz), dot(a.r1.xyz, bt) + a.r1.w);
    r.r2 = float4(dot(a.r2.xyz, bx), dot(a.r2.xyz, by), dot(a.r2.xyz, bz), dot(a.r2.xyz, bt) + a.r2.w);
    return r;
}

// An inverse bind pose: 16 floats, column-major (glam `Mat4`), as the three rows of a 3x4.
World load_inverse_bind(float* m, uint j) {
    let b = j * 16;
    World r;
    r.r0 = float4(m[b + 0], m[b + 4], m[b + 8], m[b + 12]);
    r.r1 = float4(m[b + 1], m[b + 5], m[b + 9], m[b + 13]);
    r.r2 = float4(m[b + 2], m[b + 6], m[b + 10], m[b + 14]);
    return r;
}

// ---- skin_vertices ----------------------------------------------------------------------

struct SkinParams {
    Vertex* rest;         // The mesh's rest-pose stream.
    SkinVertex* skin;     // Per-vertex influences, parallel to `rest`.
    World* world;         // The GPU transform table's world rows.
    uint* joint_nodes;    // Per joint: its transform-table node (NO_NODE = unresolved).
    float* inverse_bind;  // Per joint: inverse bind pose, 16 floats.
    Vertex* dst;          // This frame's deformed stream.
    uint count;
    uint base;
    uint joint_count;
    uint mesh_node;       // The instance's own node: its world is divided out.
};
[[vk::push_constant]] SkinParams skin_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void skin_vertices(uint3 gid: SV_DispatchThreadID) {
    let i = skin_params.base + gid.x;
    if (i >= skin_params.count) {
        return;
    }
    let rest = skin_params.rest[i];
    let rest_pos = float3(rest.px, rest.py, rest.pz);
    let rest_nrm = float3(rest.nx, rest.ny, rest.nz);
    let influence = skin_params.skin[i];
    let joints = uint4(
        influence.joints.x & 0xffff,
        influence.joints.x >> 16,
        influence.joints.y & 0xffff,
        influence.joints.y >> 16);
    let weights = influence.weights;

    // Blend the skin matrices: sum_k w_k * (joint_world_k * inverse_bind_k).
    World m;
    m.r0 = float4(0.0);
    m.r1 = float4(0.0);
    m.r2 = float4(0.0);
    float total = 0.0;
    for (uint k = 0; k < 4; k++) {
        let w = weights[k];
        let j = joints[k];
        if (w == 0.0 || j >= skin_params.joint_count) {
            continue;
        }
        let node = skin_params.joint_nodes[j];
        if (node == NO_NODE) {
            continue;
        }
        let skin = affine_mul(skin_params.world[node], load_inverse_bind(skin_params.inverse_bind, j));
        m.r0 += w * skin.r0;
        m.r1 += w * skin.r1;
        m.r2 += w * skin.r2;
        total += w;
    }

    float3 pos_local;
    float3 nrm_local;
    if (total <= 0.0) {
        // No usable influence: keep the rest pose.
        pos_local = rest_pos;
        nrm_local = rest_nrm;
    } else {
        let inv_instance = affine_inverse(skin_params.world[skin_params.mesh_node]);
        let pos_world = affine_point(m, rest_pos);
        let nrm_world = affine_normal(m, rest_nrm);
        pos_local = affine_point(inv_instance, pos_world);
        nrm_local = normalize(affine_normal(inv_instance, nrm_world));
    }

    Vertex out;
    out.px = pos_local.x;
    out.py = pos_local.y;
    out.pz = pos_local.z;
    out.nx = nrm_local.x;
    out.ny = nrm_local.y;
    out.nz = nrm_local.z;
    out.u = rest.u;
    out.v = rest.v;
    skin_params.dst[i] = out;
}

// ---- wind_vertices ----------------------------------------------------------------------

// `WindParams` in skinning.rs. The skin stream carries no joints here: `weights.x` is the
// sway weight (0 at the anchored base, 1 at the tip) and `joints.x & 0xffff` a per-plant
// phase id.
struct WindParams {
    Vertex* rest;
    SkinVertex* skin;
    World* world;
    Vertex* dst;
    uint count;
    uint base;
    uint mesh_node;
    uint pad;
    float4 wind;   // direction.x, direction.z (unit, the way it blows), speed m/s, time s
    float4 shape;  // amplitude (m at 10 m/s, full weight), gust, wavelength m, turbulence
};
[[vk::push_constant]] WindParams wind_params;

float hash01(uint n) {
    n = (n ^ 61u) ^ (n >> 16);
    n *= 9u;
    n ^= n >> 4;
    n *= 0x27d4eb2du;
    n ^= n >> 15;
    return float(n & 0xffffffu) / 16777216.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void wind_vertices(uint3 gid: SV_DispatchThreadID) {
    let i = wind_params.base + gid.x;
    if (i >= wind_params.count) {
        return;
    }
    let rest = wind_params.rest[i];
    let influence = wind_params.skin[i];
    let weight = clamp(influence.weights.x, 0.0, 1.0);
    let amplitude = wind_params.shape.x;
    let speed = wind_params.wind.z;
    Vertex out = rest;
    if (weight > 0.0 && amplitude > 0.0 && speed > 0.0) {
        let m = wind_params.world[wind_params.mesh_node];
        let rest_pos = float3(rest.px, rest.py, rest.pz);
        let p = affine_point(m, rest_pos);
        let dir = float2(wind_params.wind.x, wind_params.wind.y);
        let t = wind_params.wind.w;
        let k = 6.2831853 / max(wind_params.shape.z, 0.1);
        let phase_id = hash01(influence.joints.x & 0xffff) * 6.2831853;
        // Waves travel downwind at the wind speed; a faster harmonic and a per-plant flutter
        // break the sheet-like look.
        let along = dot(p.xz, dir);
        let phase = (along - t * speed) * k;
        let sway = 0.55 * sin(phase + 0.3 * phase_id)
                 + 0.30 * sin(phase * 2.17 + phase_id)
                 + 0.15 * sin(t * (2.5 + hash01(influence.joints.x ^ 0x9e37u) * 2.0) + 3.0 * phase_id);
        // A slow gust front: strength swells and fades as it rolls through.
        let gust = 1.0 + wind_params.shape.y * (0.5 + 0.5 * sin((along - t * speed * 1.4) * k * 0.11 + 0.2 * phase_id));
        let strength = amplitude * weight * (speed / 10.0) * gust;
        // Lean downwind with the oscillation on top; flutter across the wind by turbulence.
        let downwind = strength * (0.65 + 0.35 * sway);
        let across = strength * wind_params.shape.w * 0.35 * sin(t * 3.7 + 2.0 * phase_id + phase * 0.5);
        let offset_world = float3(dir.x, 0.0, dir.y) * downwind + float3(-dir.y, 0.0, dir.x) * across;
        // A bent stalk drops as it leans (d^2 / 2h for a half-metre stalk), so tips do not
        // stretch away from their bases.
        let drop = -(downwind * downwind + across * across);
        let offset_local = affine_dir(affine_inverse(m), offset_world + float3(0.0, drop, 0.0));
        out.px = rest.px + offset_local.x;
        out.py = rest.py + offset_local.y;
        out.pz = rest.pz + offset_local.z;
    }
    wind_params.dst[i] = out;
}

// ---- pack_triangles ---------------------------------------------------------------------

// `Triangle::pack_normal` in blas.rs.
uint pack_normal(float3 n) {
    let x = uint((n.x * 0.5 + 0.5) * 65535.0);
    let y = uint((n.y * 0.5 + 0.5) * 32767.0);
    let z = n.z >= 0.0 ? 0u : 1u;
    return (x << 16) | (y << 1) | z;
}

// `Triangle::pack_uv` in blas.rs: two halves, x low.
uint pack_uv(float2 uv) {
    return (f32tof16(uv.y) << 16) | f32tof16(uv.x);
}

struct PackParams {
    Vertex* vertices;     // The deformed stream.
    uint* indices;
    Triangle* triangles;  // The instance's shading records, one per triangle.
    uint count;
    uint base;
};
[[vk::push_constant]] PackParams pack_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void pack_triangles(uint3 gid: SV_DispatchThreadID) {
    let t = pack_params.base + gid.x;
    if (t >= pack_params.count) {
        return;
    }
    let v0 = pack_params.vertices[pack_params.indices[t * 3 + 0]];
    let v1 = pack_params.vertices[pack_params.indices[t * 3 + 1]];
    let v2 = pack_params.vertices[pack_params.indices[t * 3 + 2]];
    let p0 = float3(v0.px, v0.py, v0.pz);
    let p1 = float3(v1.px, v1.py, v1.pz);
    let p2 = float3(v2.px, v2.py, v2.pz);
    let uv0 = float2(v0.u, v0.v);
    let uv1 = float2(v1.u, v1.v);
    let uv2 = float2(v2.u, v2.v);

    // `pack_triangle` in blas.rs: the UV-space tangent, +Z when the UVs are degenerate.
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let denom = duv1.x * duv2.y - duv1.y * duv2.x;
    float3 tangent;
    if (abs(denom) < 0.0001) {
        tangent = float3(0.0, 0.0, 1.0);
    } else {
        let f = 1.0 / denom;
        tangent = normalize(float3(
            f * (duv2.y * edge1.x - duv1.y * edge2.x),
            f * (duv2.y * edge1.y - duv1.y * edge2.y),
            f * (duv2.y * edge1.z - duv1.y * edge2.z)));
    }

    Triangle tri;
    tri.tangent = pack_normal(tangent);
    tri.n0 = pack_normal(float3(v0.nx, v0.ny, v0.nz));
    tri.n1 = pack_normal(float3(v1.nx, v1.ny, v1.nz));
    tri.n2 = pack_normal(float3(v2.nx, v2.ny, v2.nz));
    tri.uv0 = pack_uv(uv0);
    tri.uv1 = pack_uv(uv1);
    tri.uv2 = pack_uv(uv2);
    // Texture LOD base: 0.5 * log2(uv area / surface area), both doubled (closest_hit).
    tri.lod = 0.5 * log2(max(abs(denom), 1.0e-12) / max(length(cross(edge1, edge2)), 1.0e-12));
    pack_params.triangles[t] = tri;
}
