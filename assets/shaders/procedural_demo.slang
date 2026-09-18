// The `procedural` example's fill kernel: a wavy heightfield grid, one thread per vertex.
// Shape of every ProceduralMesh fill kernel: the push block starts with `ProceduralHeader`,
// the app's own parameters follow (`DemoParams` in examples/procedural.rs, scalar layout).
import procedural;

struct DemoPush {
    ProceduralHeader header;
    uint res;
    float size;
    float amplitude;
    float frequency;
};
[[vk::push_constant]] DemoPush push;

[shader("compute")]
[numthreads(64, 1, 1)]
void demo_fill(uint3 gid: SV_DispatchThreadID) {
    let p = push;
    let i = p.header.base + gid.x;
    if (i >= p.header.vertex_count) {
        return;
    }
    let gx = i % p.res;
    let gy = i / p.res;
    let inv = 1.0 / float(p.res - 1);
    let u = float(gx) * inv;
    let v = float(gy) * inv;
    let x = (u - 0.5) * p.size;
    let z = (v - 0.5) * p.size;
    let y = p.amplitude * sin(p.frequency * x) * cos(p.frequency * z);
    // Analytic gradient -> normal.
    let dydx = p.amplitude * p.frequency * cos(p.frequency * x) * cos(p.frequency * z);
    let dydz = -p.amplitude * p.frequency * sin(p.frequency * x) * sin(p.frequency * z);
    let n = normalize(float3(-dydx, 1.0, -dydz));

    Vertex out;
    out.px = x;
    out.py = y;
    out.pz = z;
    out.nx = n.x;
    out.ny = n.y;
    out.nz = n.z;
    out.u = u;
    out.v = v;
    p.header.vertices[i] = out;
}
