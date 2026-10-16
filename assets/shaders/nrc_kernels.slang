// Neural radiance cache training (src/nrc.rs): a batch of records through the network
// (cooperative vectors), its weight gradients (shared-memory reduction over batch chunks) and an
// Adam step that refreshes the f16 copies the forward passes read. Plus the gym's synthetic
// records (tests/nrc_gym.rs).
import rt_types;
import disney;
import nrc;

static const uint NRC_PARAMS = NRC_LAYERS * (NRC_MATRIX_HALVES + NRC_WIDTH);
// Batch rows per gradient workgroup (nrc.rs NRC_CHUNK).
static const uint NRC_CHUNK = 256;

// Must match NrcNetGpu in src/nrc.rs.
struct NrcNet {
    half* weights;    // [layer][out][in], what the forward passes read
    half* weights_t;  // [layer][in][out], what backpropagation reads
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
    uint ring;        // records in the ring
    uint batch;
    float learning_rate;
    float radiance_scale;
    float loss_scale;
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

// One record per thread: forward, loss, backward to every layer's output gradient.
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
    let row = int(row_index * NRC_WIDTH * 2u);
    let layer = int(net.batch * NRC_WIDTH * 2u);
    var x = nrcEncode(inp);
    var z = x;
    [ForceUnroll]
    for (uint l = 0u; l < NRC_LAYERS; l++) {
        x.store(net.acts, int(l) * layer + row);
        z = coopVecMatMulAdd<half, NRC_WIDTH>(
            x, CoopVecComponentType::Float16,
            (void*)(net.weights + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            (void*)(net.biases + l * NRC_WIDTH), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::RowMajor, false, NRC_WIDTH * 2u);
        x = max(z, CoopVec<half, NRC_WIDTH>(half(0.0)));
    }
    let p = float3(float(z[0]), float(z[1]), float(z[2]));
    let t = unpackRadiance(rec.target) * net.radiance_scale / nrcFactor(inp);
    // Relative L2 (the prediction's luminance in the denominator, held constant).
    let lp = luma(max(p, float3(0.0)));
    let den = lp * lp + 0.01;
    let diff = p - t;
    let loss = dot(diff, diff) / den;
    let finite = !(isnan(loss) || isinf(loss));
    net.loss[row_index] = finite ? loss : 0.0;
    let g = finite ? clamp(2.0 * diff / den * net.loss_scale, -6.0e4, 6.0e4) : float3(0.0);
    var d = CoopVec<half, NRC_WIDTH>(half(0.0));
    d[0] = half(g.x);
    d[1] = half(g.y);
    d[2] = half(g.z);
    d.store(net.dz, int(NRC_LAYERS - 1u) * layer + row);
    [ForceUnroll]
    for (uint l = NRC_LAYERS - 1u; l > 0u; l--) {
        d = coopVecMatMul<half, NRC_WIDTH>(
            d, CoopVecComponentType::Float16,
            (void*)(net.weights_t + l * NRC_MATRIX_HALVES), CoopVecComponentType::Float16,
            CoopVecMatrixLayout::RowMajor, false, NRC_WIDTH * 2u);
        let a = coopVecLoad<NRC_WIDTH, half>(net.acts, int(l) * layer + row);
        [ForceUnroll]
        for (uint i = 0u; i < NRC_WIDTH; i++) {
            if (a[i] <= half(0.0)) {
                d[i] = half(0.0);
            }
        }
        d.store(net.dz, int(l - 1u) * layer + row);
    }
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
    if (p < NRC_LAYERS * NRC_MATRIX_HALVES) {
        let l = p / NRC_MATRIX_HALVES;
        let o = (p % NRC_MATRIX_HALVES) / NRC_WIDTH;
        let i = p % NRC_WIDTH;
        net.weights[p] = half(w);
        net.weights_t[l * NRC_MATRIX_HALVES + i * NRC_WIDTH + o] = half(w);
    } else {
        net.biases[p - NRC_LAYERS * NRC_MATRIX_HALVES] = half(w);
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
