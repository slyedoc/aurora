// Neural radiance cache (src/nrc.rs): the network's input encoding and its inference, for the
// compute kernels (nrc_kernels.slang). Cooperative vectors (VK_NV_cooperative_vector): the matrix
// products run on the tensor cores. Nothing the raygen imports may use them (nrc_record.slang):
// their code in a ray tracing shader cost the whole trace 3.5x, run or not.
import rt_types;
__exported import nrc_record;

public static const uint NRC_WIDTH = 64;
// Input -> 2 hidden layers -> output (Instant-NGP's NRC network: the hash grid holds the detail):
// three 64 x 64 matrices; the output's first three rows are RGB.
public static const uint NRC_LAYERS = 3;
public static const uint NRC_MATRIX_HALVES = NRC_WIDTH * NRC_WIDTH;
// The position's multiresolution hash grid (Müller et al. 2022, Instant-NGP; its NRC task):
// NRC_GRID_LEVELS levels of NRC_GRID_FEATURES trained features, from coarse to fine.
public static const uint NRC_GRID_LEVELS = 16;
public static const uint NRC_GRID_FEATURES = 2;

// The hash grid as one pass reads it: its f16 table ([level][entry][feature]), entries per
// level (a power of two), the finest level's cell (metres) and the ratio between levels' cells.
public struct NrcGrid {
    public half* table;
    public uint size;
    public float finest;
    public float growth;
};

// One level's cell for a position: its lowest corner and where in it the position lies.
public void nrcGridCell(NrcGrid g, uint l, float3 p, out int3 corner, out float3 f) {
    let cell = g.finest * pow(g.growth, float(NRC_GRID_LEVELS - 1u - l));
    let q = p / cell;
    let base = floor(q);
    corner = int3(base);
    f = q - base;
}

// A corner's entry (Instant-NGP's spatial hash) and its trilinear weight; `k` picks the corner.
public uint nrcGridEntry(NrcGrid g, uint l, int3 corner, float3 f, uint k, out float w) {
    let o = uint3(k & 1u, (k >> 1u) & 1u, k >> 2u);
    let c = uint3(corner) + o;
    let wf = select(o != uint3(0u), f, 1.0 - f);
    w = wf.x * wf.y * wf.z;
    let h = c.x ^ (c.y * 2654435761u) ^ (c.z * 805459861u);
    return l * g.size + (h & (g.size - 1u));
}


// Four one-blob bins of a value in [0, 1] (Müller et al. 2019): Gaussian bumps of width 1/4.
void oneBlob(inout CoopVec<half, NRC_WIDTH> v, uint at, float x) {
    [ForceUnroll]
    for (uint b = 0u; b < 4u; b++) {
        let d = (x - (float(b) + 0.5) * 0.25) * 4.0;
        v[at + b] = half(exp(-0.5 * d * d));
    }
}

// 58 features, padded to 64: position by the hash grid (16 x 2, inputs 0-31), direction and
// normal octahedral and 1 - exp(-roughness) one-blob (5 x 4, 36-55), the two albedos raw (56-61).
public CoopVec<half, NRC_WIDTH> nrcEncode(NrcInput i, NrcGrid g, bool grid = true) {
    var v = CoopVec<half, NRC_WIDTH>(half(0.0));
    let table = (uint*)g.table;
    [ForceUnroll]
    for (uint l = 0u; l < NRC_GRID_LEVELS && grid; l++) {
        int3 corner;
        float3 f;
        nrcGridCell(g, l, i.position, corner, f);
        var acc = float2(0.0);
        [ForceUnroll]
        for (uint k = 0u; k < 8u; k++) {
            float w;
            let e = nrcGridEntry(g, l, corner, f, k, w);
            acc += w * unpackHalf2(table[e]);
        }
        v[l * 2u] = half(acc.x);
        v[l * 2u + 1u] = half(acc.y);
    }
    let d = float32x3_to_oct(i.dir) * 0.5 + 0.5;
    let n = float32x3_to_oct(i.normal) * 0.5 + 0.5;
    oneBlob(v, 36u, d.x);
    oneBlob(v, 40u, d.y);
    oneBlob(v, 44u, n.x);
    oneBlob(v, 48u, n.y);
    oneBlob(v, 52u, 1.0 - exp(-i.roughness));
    v[56u] = half(i.diffuse.x);
    v[57u] = half(i.diffuse.y);
    v[58u] = half(i.diffuse.z);
    v[59u] = half(i.specular.x);
    v[60u] = half(i.specular.y);
    v[61u] = half(i.specular.z);
    return v;
}

// The network's output for an encoded input: ReLU hidden layers, a linear output. The weights
// are in the inferencing-optimal layout, NRC_MATRIX_HALVES per layer (nrc.rs weights_opt).
public float3 nrcForward(half* weights, half* biases, CoopVec<half, NRC_WIDTH> x) {
    var h = x;
    [ForceUnroll]
    for (uint l = 0u; l < NRC_LAYERS; l++) {
        h = coopVecMatMulAdd<half, NRC_WIDTH>(
            h, CoopVecComponentType::Float16,
            (void*)(weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            (void*)(biases + l * NRC_WIDTH), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::InferencingOptimal, false, 0u);
        if (l + 1u < NRC_LAYERS) {
            h = max(h, CoopVec<half, NRC_WIDTH>(half(0.0)));
        }
    }
    return float3(float(h[0]), float(h[1]), float(h[2]));
}


// The cache's scattered outgoing radiance at a surface point, in the training targets' units
// (pre-exposed radiance: nits x the frame's input exposure).
public float3 nrcQuery(NrcFrame* f, NrcGrid g, NrcInput i, bool grid = true) {
    let y = max(nrcForward(f.weights, f.biases, nrcEncode(i, g, grid)), float3(0.0));
    return y * nrcFactor(i);
}
