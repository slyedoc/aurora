// Neural radiance cache training (src/nrc.rs): a batch of records through the network
// (cooperative vectors), its weight gradients (shared-memory reduction over batch chunks) and an
// Adam step that refreshes the f16 copies the forward passes read. Plus the gym's synthetic
// records (tests/nrc_gym.rs).
import rt_types;
import bsdf;
import nrc;

static const uint NRC_PARAMS = NRC_LAYERS * (NRC_MATRIX_HALVES + NRC_WIDTH);
// Batch rows per gradient workgroup (nrc.rs NRC_CHUNK).
static const uint NRC_CHUNK = 256;

// Must match NrcNetGpu in src/nrc.rs.
struct NrcNet {
    half* weights;    // [layer][out][in], what the forward passes read
    half* biases;     // [layer][out]
    float* params;    // f32 master copy: the weights ([layer][out][in]), then the biases
    float* adam_m;
    float* adam_v;
    float* partials;  // per batch chunk, a gradient laid out like `params`
    half* acts;       // [layer][row][in]: each layer's input
    half* dz;         // [layer][row][out]: the loss gradient at each layer's output
    float* loss;      // [row]
    NrcRecord* records;
    uint* counter;    // records written into the ring (the trace's, the gym's)
    // The parameters' exponential moving average (f32, laid out like `params`) and its f16 copies,
    // what inference reads: steadier from frame to frame than the latest step (paper sec. 5).
    float* ema;
    half* weights_ema;
    half* biases_ema;
    uint ring;        // records in the ring
    uint batch;
    float learning_rate;
    float radiance_scale;
    float loss_scale;
    float ema_decay;
};

struct NrcPush {
    NrcNet* net;
    float* out;
    uint base;
    uint offset;      // the batch's first record in the ring
    uint step;        // Adam's step (from 1); the gym's seed
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<NrcPush> push;

uint ptHashKernel(uint x) {
    var h = x;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

// Where a training row keeps each layer's input (acts) and output gradient (dz) for the weight
// gradients (nrc_train_grads): byte offsets of the row and of one layer's block.
struct NrcRow {
    NrcNet* net;
    int row;
    int layer;
};

// One layer, W x + b. A cooperative-vector matrix product is not differentiable (it depends on
// layout), so it carries its own reverse rule (Slang's examples/mlp-training-coopvec pattern).
[BackwardDerivative(nrcLayerBwd)]
CoopVec<half, NRC_WIDTH> nrcLayer(NrcRow r, uint l, CoopVec<half, NRC_WIDTH> x) {
    x.store(r.net.acts, int(l) * r.layer + r.row);
    return coopVecMatMulAdd<half, NRC_WIDTH>(
        x, CoopVecComponentType::Float16,
        (void*)(r.net.weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
        (void*)(r.net.biases + l * NRC_WIDTH), CoopVecComponentType::Float16,
        CoopVecMatrixLayout::RowMajor, false, NRC_WIDTH * 2u);
}

// The layer's output gradient kept for the weight gradients; the input's is W^T dy: the same
// matrix read column-major. Layer 0's input is the encoding, which needs none.
void nrcLayerBwd(NrcRow r, uint l, inout DifferentialPair<CoopVec<half, NRC_WIDTH>> x,
                 CoopVec<half, NRC_WIDTH> dy) {
    dy.store(r.net.dz, int(l) * r.layer + r.row);
    var dx = CoopVec<half, NRC_WIDTH>(half(0.0));
    if (l > 0u) {
        dx = coopVecMatMul<half, NRC_WIDTH>(
            dy, CoopVecComponentType::Float16,
            (void*)(r.net.weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::ColumnMajor, false, NRC_WIDTH * 2u);
    }
    x = diffPair(x.p, dx);
}

[Differentiable]
float3 nrcNetwork(NrcRow r, CoopVec<half, NRC_WIDTH> x) {
    var h = x;
    [ForceUnroll]
    for (uint l = 0u; l < NRC_LAYERS; l++) {
        h = nrcLayer(r, l, h);
        if (l + 1u < NRC_LAYERS) {
            h = max(h, CoopVec<half, NRC_WIDTH>(half(0.0)));
        }
    }
    return float3(float(h[0]), float(h[1]), float(h[2]));
}

// Identity forward; backward keeps the output gradient inside f16 (it is a tensor-core operand)
// and drops a non-finite one (0 x NaN is NaN: a poisoned row must not reach Adam).
[BackwardDerivative(nrcGuardBwd)]
float3 nrcGuard(float3 p) {
    return p;
}

void nrcGuardBwd(inout DifferentialPair<float3> p, float3 d) {
    let ok = !(any(isnan(d)) || any(isinf(d)));
    p = diffPair(p.p, ok ? clamp(d, -6.0e4, 6.0e4) : float3(0.0));
}

// Relative L2: the prediction's luminance in the denominator, held constant. Its value is kept
// per row (the loss readout).
[Differentiable]
float nrcLoss(NrcRow r, CoopVec<half, NRC_WIDTH> x, no_diff float3 t, no_diff uint row_index) {
    let p = nrcGuard(nrcNetwork(r, x));
    let lp = no_diff luma(max(detach(p), float3(0.0)));
    let diff = p - t;
    let loss = dot(diff, diff) / (lp * lp + 0.01);
    r.net.loss[row_index] = isnan(detach(loss)) || isinf(detach(loss)) ? 0.0 : detach(loss);
    return loss;
}

// One record per thread: the loss's reverse pass (bwd_diff), which leaves every layer's input
// and output gradient for nrc_train_grads.
[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_train_forward(uint3 id: SV_DispatchThreadID) {
    let net = push.net;
    let row_index = id.x + push.base;
    if (row_index >= net.batch) {
        return;
    }
    // Rows draw records at random from those written (the ring fills in the order the trace's
    // warps ran: neighbours on screen, one batch would see one part of it).
    let count = min(net.counter[0], net.ring);
    if (count == 0u) {
        return;
    }
    let pick = ptHashKernel((push.offset + row_index) ^ (push.step * 0x9e3779b9u)) % count;
    let rec = net.records[pick];
    let inp = nrcUnpackRecord(rec);
    let t = unpackRadiance(rec.target) * net.radiance_scale / nrcFactor(inp);
    NrcRow r = { net, int(row_index * NRC_WIDTH * 2u), int(net.batch * NRC_WIDTH * 2u) };
    var x = diffPair(nrcEncode(inp));
    bwd_diff(nrcLoss)(r, x, t, row_index, net.loss_scale);
}

groupshared float g_dz[32][NRC_WIDTH];
groupshared float g_act[32][NRC_WIDTH];

// One workgroup per (batch chunk, layer), one thread per output: that output's row of the
// weight gradient and its bias gradient, summed over the chunk.
[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_train_grads(uint3 id: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    let net = push.net;
    let group = (id.x + push.base) / 64u;
    let l = group % NRC_LAYERS;
    let chunk = group / NRC_LAYERS;
    let o = lid.x;
    let first = chunk * NRC_CHUNK;
    let rows = min(NRC_CHUNK, net.batch - min(first, net.batch));
    let layer = l * net.batch * NRC_WIDTH;
    float acc[NRC_WIDTH];
    [ForceUnroll]
    for (uint i = 0u; i < NRC_WIDTH; i++) {
        acc[i] = 0.0;
    }
    var bias = 0.0;
    for (uint r0 = 0u; r0 < rows; r0 += 32u) {
        for (uint k = o; k < 32u * NRC_WIDTH; k += 64u) {
            let r = k / NRC_WIDTH;
            let c = k % NRC_WIDTH;
            let live = r0 + r < rows;
            let at = layer + (first + r0 + r) * NRC_WIDTH + c;
            g_dz[r][c] = live ? float(net.dz[at]) : 0.0;
            g_act[r][c] = live ? float(net.acts[at]) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0u; r < 32u; r++) {
            let dz = g_dz[r][o];
            bias += dz;
            [ForceUnroll]
            for (uint i = 0u; i < NRC_WIDTH; i++) {
                acc[i] += dz * g_act[r][i];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let base = chunk * NRC_PARAMS;
    [ForceUnroll]
    for (uint i = 0u; i < NRC_WIDTH; i++) {
        net.partials[base + l * NRC_MATRIX_HALVES + o * NRC_WIDTH + i] = acc[i];
    }
    net.partials[base + NRC_LAYERS * NRC_MATRIX_HALVES + l * NRC_WIDTH + o] = bias;
}

// One thread per parameter: the batch's mean gradient, an Adam step on the master copy, and
// the f16 copies. A non-finite gradient skips the step (the next batch heals it).
[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_adam(uint3 id: SV_DispatchThreadID) {
    let net = push.net;
    let p = id.x + push.base;
    if (p >= NRC_PARAMS) {
        return;
    }
    if (net.counter[0] == 0u) {
        return;
    }
    let chunks = (net.batch + NRC_CHUNK - 1u) / NRC_CHUNK;
    var g = 0.0;
    for (uint c = 0u; c < chunks; c++) {
        g += net.partials[c * NRC_PARAMS + p];
    }
    g /= float(net.batch) * net.loss_scale;
    if (isnan(g) || isinf(g)) {
        return;
    }
    let b1 = 0.9;
    let b2 = 0.99;
    let m = b1 * net.adam_m[p] + (1.0 - b1) * g;
    let v = b2 * net.adam_v[p] + (1.0 - b2) * g * g;
    net.adam_m[p] = m;
    net.adam_v[p] = v;
    let step = float(push.step);
    let mh = m / (1.0 - pow(b1, step));
    let vh = v / (1.0 - pow(b2, step));
    let w = clamp(net.params[p] - net.learning_rate * mh / (sqrt(vh) + 1.0e-8), -6.0e4, 6.0e4);
    net.params[p] = w;
    let e = lerp(w, net.ema[p], net.ema_decay);
    net.ema[p] = e;
    if (p < NRC_LAYERS * NRC_MATRIX_HALVES) {
        net.weights[p] = half(w);
        net.weights_ema[p] = half(e);
    } else {
        net.biases[p - NRC_LAYERS * NRC_MATRIX_HALVES] = half(w);
        net.biases_ema[p - NRC_LAYERS * NRC_MATRIX_HALVES] = half(e);
    }
}

// The network's raw output (factorized, scaled) for record `offset + i` of the ring, into
// `out` (3 floats each).
[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_eval(uint3 id: SV_DispatchThreadID) {
    let net = push.net;
    let i = id.x + push.base;
    if (i >= net.batch) {
        return;
    }
    let inp = nrcUnpackRecord(net.records[(push.offset + i) % net.ring]);
    var x = nrcEncode(inp);
    [ForceUnroll]
    for (uint l = 0u; l < NRC_LAYERS; l++) {
        x = coopVecMatMulAdd<half, NRC_WIDTH>(
            x, CoopVecComponentType::Float16,
            (void*)(net.weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            (void*)(net.biases + l * NRC_WIDTH), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::RowMajor, false, NRC_WIDTH * 2u);
        if (l + 1u < NRC_LAYERS) {
            x = max(x, CoopVec<half, NRC_WIDTH>(half(0.0)));
        }
    }
    push.out[i * 3u] = float(x[0]);
    push.out[i * 3u + 1u] = float(x[1]);
    push.out[i * 3u + 2u] = float(x[2]);
}

// The gym's field (tests/nrc_gym.rs): coloured Gaussian blobs of light in the box, each with a
// directional lobe, times the albedo; a smooth function the network must fit.
float3 gymField(NrcInput i) {
    var l = float3(0.0);
    [ForceUnroll]
    for (uint k = 0u; k < 4u; k++) {
        let fk = float(k);
        let centre = frac(float3(0.21, 0.57, 0.83) * (fk + 1.0) * 1.7);
        let colour = 0.5 + 0.5 * cos(float3(0.0, 2.1, 4.2) + fk * 1.3);
        let lobe_dir = normalize(float3(cos(fk * 2.4), 0.6, sin(fk * 2.4)));
        let d = i.position - centre;
        let lobe = 0.5 + 0.5 * dot(i.dir, lobe_dir);
        l += colour * 4.0 * exp(-dot(d, d) * 12.0) * lobe * lobe
            * (0.6 + 0.4 * saturate(dot(i.normal, lobe_dir)));
    }
    return l * (i.diffuse + i.specular);
}

float rand_f(inout uint s) {
    s = s * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return float(((word >> 22u) ^ word) >> 8) * (1.0 / 16777216.0);
}

float3 gymDirection(inout uint s) {
    let z = rand_f(s) * 2.0 - 1.0;
    let phi = rand_f(s) * 6.2831853;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

// `batch` fresh records of the gym's field at ring offset 0, seeded by `step`.
[shader("compute")]
[numthreads(64, 1, 1)]
void nrc_gym_records(uint3 id: SV_DispatchThreadID) {
    let net = push.net;
    let i = id.x + push.base;
    if (i >= net.batch) {
        return;
    }
    var s = i * 9781u + push.step * 6271u + 17u;
    NrcInput inp;
    inp.position = float3(rand_f(s), rand_f(s), rand_f(s));
    inp.dir = gymDirection(s);
    inp.normal = gymDirection(s);
    inp.roughness = rand_f(s);
    inp.diffuse = float3(rand_f(s), rand_f(s), rand_f(s)) * 0.8;
    inp.specular = float3(rand_f(s)) * 0.2;
    var r = nrcPackRecord(inp);
    // The record's own quantization is what the network sees.
    r.target = packRadiance(gymField(nrcUnpackRecord(r)) / net.radiance_scale);
    net.records[i % net.ring] = r;
    if (i == 0u) {
        net.counter[0] = net.batch;
    }
}
