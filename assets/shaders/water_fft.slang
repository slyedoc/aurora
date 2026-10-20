// Water waves by FFT (src/water_volume.rs `WaterWaves`, Tessendorf 2001): per frame, a cascade's
// initial spectrum h0(k) (generated on the CPU when the waves change) moves on to the frame's time,
// becomes its slope spectrum, and goes back to space through an inverse 2D FFT (rows, then
// columns). Both slopes ride in one complex field (dh/dx real, dh/dz imaginary: each is a real
// field, so their spectra are Hermitian and one transform carries both). The texels hold the
// slopes and their squares, so the mip chain's averages give the slope variance a footprint hides.

static const uint N = 256u;
static const uint LOG2_N = 8u;
static const float G = 9.81;
static const float TAU = 6.283185307179586;

struct WaveParams {
    float2* h0;
    float2* work;
    uint2* texels;
    // Metres the tile spans.
    float tile;
    float time;
    float speed;
    // 0: rows, 1: columns.
    uint axis;
};
[[vk::push_constant]] WaveParams wp;

float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// The signed frequency of index i (0, 1, .., N/2 - 1, -N/2, .., -1).
int freq(uint i) {
    return i < N / 2u ? int(i) : int(i) - int(N);
}

// One thread per texel: h(k, t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t}, then
// i kx h + i (i kz h).
[shader("compute")]
[numthreads(64, 1, 1)]
void wave_spectrum(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x;
    if (idx >= N * N) {
        return;
    }
    let x = idx % N;
    let y = idx / N;
    let k = float2(float(freq(x)), float(freq(y))) * (TAU / wp.tile);
    let kl = length(k);
    let w = sqrt(G * kl) * wp.speed * wp.time;
    let e = float2(cos(w), sin(w));
    let a = wp.h0[idx];
    let b = wp.h0[((N - y) % N) * N + (N - x) % N];
    let h = cmul(a, e) + cmul(float2(b.x, -b.y), float2(e.x, -e.y));
    // i kx h = (-kx h.y, kx h.x); i (i kz h) = -kz h.
    wp.work[idx] = float2(-k.x * h.y - k.y * h.x, k.x * h.x - k.y * h.y);
}

uint bitReverse(uint i) {
    return reversebits(i) >> (32u - LOG2_N);
}

groupshared float2 line[N];

// One workgroup per row (axis 0) or column (axis 1): an inverse FFT of N points in groupshared
// memory, radix 2, decimation in time from bit-reversed order.
[shader("compute")]
[numthreads(64, 1, 1)]
void wave_fft(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID) {
    let lineIdx = group.x;
    let t = thread.x;
    let base = wp.axis == 0u ? lineIdx * N : lineIdx;
    let stride = wp.axis == 0u ? 1u : N;
    for (uint i = t; i < N; i += 64u) {
        line[bitReverse(i)] = wp.work[base + i * stride];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint len = 2u; len <= N; len <<= 1u) {
        let half = len >> 1u;
        for (uint b = t; b < N / 2u; b += 64u) {
            let j = b % half;
            let i0 = (b / half) * len + j;
            let i1 = i0 + half;
            let angle = TAU * float(j) / float(len);
            let tw = cmul(float2(cos(angle), sin(angle)), line[i1]);
            let u = line[i0];
            line[i0] = u + tw;
            line[i1] = u - tw;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = t; i < N; i += 64u) {
        wp.work[base + i * stride] = line[i];
    }
}

// One thread per texel: (dh/dx, dh/dz, their squares) as four halfs, the image's level 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void wave_texels(uint3 gid: SV_DispatchThreadID) {
    let idx = gid.x;
    if (idx >= N * N) {
        return;
    }
    let s = wp.work[idx];
    wp.texels[idx] = uint2(f32tof16(s.x) | (f32tof16(s.y) << 16u),
                           f32tof16(s.x * s.x) | (f32tof16(s.y * s.y) << 16u));
}
