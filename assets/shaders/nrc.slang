// Neural radiance cache (src/nrc.rs): the network's input encoding and its inference, shared by
// the raygen and the compute kernels (nrc_kernels.slang). Cooperative vectors
// (VK_NV_cooperative_vector): the matrix products run on the tensor cores.
import rt_types;

public static const uint NRC_WIDTH = 64;
// Input -> 5 hidden layers -> output: six 64 x 64 matrices; the output's first three rows are RGB.
public static const uint NRC_LAYERS = 6;
static const uint NRC_FREQS = 12u;
public static const uint NRC_MATRIX_HALVES = NRC_WIDTH * NRC_WIDTH;


// What the network sees of a surface point.
public struct NrcInput {
    // In the cache box, [0, 1]^3.
    public float3 position;
    // Outgoing direction (towards the viewer) and shading normal, unit.
    public float3 dir;
    public float3 normal;
    public float roughness;
    public float3 diffuse;
    public float3 specular;
};

// Four one-blob bins of a value in [0, 1] (Müller et al. 2019): Gaussian bumps of width 1/4.
void oneBlob(inout CoopVec<half, NRC_WIDTH> v, uint at, float x) {
    [ForceUnroll]
    for (uint b = 0u; b < 4u; b++) {
        let d = (x - (float(b) + 0.5) * 0.25) * 4.0;
        v[at + b] = half(exp(-0.5 * d * d));
    }
}

// 62 features, padded to 64 (tiny-cuda-nn's NRC encoding): position by a triangle wave at 12
// frequencies (3 x 12, periods 2 to 2^-10 of the box), direction and normal octahedral and 1 - exp(-roughness) one-blob (5 x 4),
// the two albedos raw (6).
public CoopVec<half, NRC_WIDTH> nrcEncode(NrcInput i) {
    var v = CoopVec<half, NRC_WIDTH>(half(0.0));
    [ForceUnroll]
    for (uint a = 0u; a < 3u; a++) {
        [ForceUnroll]
        for (uint o = 0u; o < NRC_FREQS; o++) {
            // Period 2 at the lowest frequency: every triangle is symmetric about its peaks, and
            // with period 1 none told x from 1 - x.
            let t = frac(i.position[a] * float(1u << o) * 0.5);
            v[a * NRC_FREQS + o] = half(abs(4.0 * t - 2.0) - 1.0);
        }
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

// The network's output for an encoded input: ReLU hidden layers, a linear output.
public float3 nrcForward(half* weights, half* biases, CoopVec<half, NRC_WIDTH> x) {
    var h = x;
    [ForceUnroll]
    for (uint l = 0u; l < NRC_LAYERS; l++) {
        h = coopVecMatMulAdd<half, NRC_WIDTH>(
            h, CoopVecComponentType::Float16,
            (void*)(weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            (void*)(biases + l * NRC_WIDTH), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::RowMajor, false, NRC_WIDTH * 2u);
        if (l + 1u < NRC_LAYERS) {
            h = max(h, CoopVec<half, NRC_WIDTH>(half(0.0)));
        }
    }
    return float3(float(h[0]), float(h[1]), float(h[2]));
}


uint packOct16(float3 v) {
    let e = saturate(float32x3_to_oct(v) * 0.5 + 0.5);
    return uint(e.x * 65535.0 + 0.5) | (uint(e.y * 65535.0 + 0.5) << 16);
}

float3 unpackOct16(uint p) {
    let e = float2(float(p & 0xFFFFu), float(p >> 16)) * (2.0 / 65535.0) - 1.0;
    return oct_to_float32x3(e);
}

uint packUnorm8x4(float4 v) {
    let u = uint4(saturate(v) * 255.0 + 0.5);
    return u.x | (u.y << 8) | (u.z << 16) | (u.w << 24);
}

float4 unpackUnorm8x4(uint p) {
    return float4(float(p & 0xFFu), float((p >> 8) & 0xFFu), float((p >> 16) & 0xFFu),
                  float(p >> 24)) * (1.0 / 255.0);
}

public NrcRecord nrcPackRecord(NrcInput i) {
    NrcRecord r = {};
    r.position = i.position;
    r.dir = packOct16(i.dir);
    r.normal = packOct16(i.normal);
    r.rough_diffuse = packUnorm8x4(float4(i.roughness, i.diffuse));
    r.specular = packUnorm8x4(float4(i.specular, 0.0));
    return r;
}

public NrcInput nrcUnpackRecord(NrcRecord r) {
    NrcInput i;
    i.position = r.position;
    i.dir = unpackOct16(r.dir);
    i.normal = unpackOct16(r.normal);
    let rd = unpackUnorm8x4(r.rough_diffuse);
    i.roughness = rd.x;
    i.diffuse = rd.yzw;
    i.specular = unpackUnorm8x4(r.specular).xyz;
    return i;
}

// Reflectance factorization (Müller et al. 2021, sec. 4.1): the network learns scattered
// radiance over the surface's albedo, the lighting without the texture.
public float3 nrcFactor(NrcInput i) {
    return max(i.diffuse + i.specular, float3(0.05));
}

// The cache's scattered outgoing radiance at a surface point, in the training targets' units
// (pre-exposed radiance: nits x the frame's input exposure).
public float3 nrcQuery(NrcFrame* f, NrcInput i) {
    let y = max(nrcForward(f.weights, f.biases, nrcEncode(i)), float3(0.0));
    return y * nrcFactor(i);
}

// A world position in the cache box, [0, 1]^3: within the radius of the anchor (f.box: its world
// position, 1 / the radius) linear, beyond contracted toward the walls (Mip-NeRF 360, by the
// L-infinity norm): resolution falls with distance, as the footprints of the paths ending there
// grow.
public float3 nrcBoxPosition(NrcFrame* f, float3 p) {
    var x = (p - f.box.xyz) * f.box.w;
    let m = max(abs(x.x), max(abs(x.y), abs(x.z)));
    if (m > 1.0) {
        x *= (2.0 - 1.0 / m) / m;
    }
    return x * 0.25 + 0.5;
}

public bool nrcInBox(float3 q) {
    return all(q >= float3(0.0)) && all(q <= float3(1.0));
}
