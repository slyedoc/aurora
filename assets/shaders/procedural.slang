// Procedural mesh support kernels (src/procedural_mesh.rs).
//
// A ProceduralMesh's fill kernel (the app's own Slang, e.g. a planet's terrain function)
// writes the `Vertex` stream; `pack_triangles` then derives the per-triangle shading records
// the closest-hit reads (tangent + packed normals / uvs), exactly as skinning.slang and
// terrain.slang do for their streams. Same `Vertex` / `Triangle` layouts as blas.rs.

// Every fill kernel's push-constant block starts with this header (blas.rs
// `ProceduralHeader`): the engine fills it in, the app's own parameters follow.
public struct ProceduralHeader {
    public Vertex* vertices;
    public uint vertex_count;
    public uint base;
};

// Deformed vertex stream: `Vertex` in blas.rs (32 bytes, scalar layout).
public struct Vertex {
    public float px, py, pz;
    public float nx, ny, nz;
    public float u, v;
};

// Per-triangle shading record: `Triangle` in blas.rs (32 bytes).
public struct Triangle {
    public uint tangent;
    public uint n0, n1, n2;
    public uint uv0, uv1, uv2;
    public float lod;
};

public uint pack_normal(float3 n) {
    let x = uint((n.x * 0.5 + 0.5) * 65535.0);
    let y = uint((n.y * 0.5 + 0.5) * 32767.0);
    let z = n.z >= 0.0 ? 0u : 1u;
    return (x << 16) | (y << 1) | z;
}

public uint pack_uv(float2 uv) {
    return (f32tof16(uv.y) << 16) | f32tof16(uv.x);
}

struct PackParams {
    Vertex* vertices;
    uint* indices;
    Triangle* triangles;
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
