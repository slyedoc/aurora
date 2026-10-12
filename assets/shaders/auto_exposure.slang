// Auto-exposure metering (a port of bevy_post_process::auto_exposure; src/auto_exposure.rs).
//
// The raygen writes every pixel's (meter, nits) pair into `lum`: the meter is the incident
// luminance (radiance over the first surface's albedo), < 0 for sky and emitter pixels the
// meter skips. `ae_histogram` bins last frame's meters into 64 log2 bins (groupshared, then
// merged), and
// `ae_resolve` trims the percentile tails, averages what is left, adapts the exposure over
// time, and writes the linear exposure the NEXT frame's raygen applies.

// Fixed thread count for `ae_histogram`: 256 groups of 64, each thread striding the buffer.
static const uint AE_HISTOGRAM_THREADS = 16384;
// Neighbouring pixels (along a row) averaged into one meter reading.
static const uint AE_STRIP = 16;

// Must match AeGpu in src/auto_exposure.rs (and AeData in rt_types.slang).
struct AeState {
    float ev;              // Smoothed log2 exposure (the Auto look).
    float exposure;        // exp2(ev).
    // What the raygen applies (Ray Reconstruction's input), set by the CPU: the look's EV
    // (auto_exposure.rs); the blit maps input -> look. This kernel never writes it.
    float input_ev;
    float input_exposure;  // exp2(input_ev)
    // The full metering (log2 exposure that puts the scene at mid-gray).
    float metered;
};

// One block for both kernels (the compute layout is push constants only).
struct AeParams {
    float* lum;
    uint* histogram;
    AeState* state;
    uint pixel_count;
    // Histogram domain: bin 0 is "below min", bins 1..63 span the log2 range.
    float min_log_lum;
    float inv_log_lum_range;
    // ae_resolve: percentile trim, target, adaptation.
    float log_lum_range;
    float low_percent;
    float high_percent;
    float speed_brighten;  // EV per second towards a brighter exposure.
    float speed_darken;
    float exp_transition;  // EV distance over which adaptation goes exponential.
    float dt;
    float compensation;    // Artist EV offset on the mid-gray target.
    float min_ev;
    float max_ev;
    float reference_ev;    // The look adapts around this exposure...
    float adaptation;      // ...by this fraction of the metered difference (1 = full auto).
};
[[vk::push_constant]] AeParams p;

groupshared uint shared_bins[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void ae_histogram(uint3 gid: SV_DispatchThreadID, uint li: SV_GroupIndex) {
    shared_bins[li] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Binned per strip of AE_STRIP neighbouring pixels (their linear mean, counted once per
    // metered pixel), not per pixel: at one path per pixel many meters are exactly 0 (an
    // occluded shadow ray, a bounce that found nothing) and the log histogram read that noise
    // as darkness -- by up to seven stops once the fog lifted those zeros off the floor.
    let strips = (p.pixel_count + AE_STRIP - 1) / AE_STRIP;
    for (uint b = gid.x; b < strips; b += AE_HISTOGRAM_THREADS) {
        var sum = 0.0;
        var n = 0u;
        for (uint k = 0u; k < AE_STRIP; k++) {
            let i = b * AE_STRIP + k;
            if (i >= p.pixel_count) {
                break;
            }
            let lum = p.lum[2 * i];
            if (lum >= 0.0 && lum < 1.0e30) {
                sum += lum;
                n += 1u;
            }
        }
        if (n == 0u) {
            continue;
        }
        let lum = sum / float(n);
        uint bin = 0;
        if (lum >= exp2(p.min_log_lum)) {
            let t = saturate((log2(lum) - p.min_log_lum) * p.inv_log_lum_range);
            bin = uint(t * 62.0 + 1.0);
        }
        InterlockedAdd(shared_bins[bin], n);
    }

    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(p.histogram[li], shared_bins[li]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ae_resolve(uint3 gid: SV_DispatchThreadID) {
    if (gid.x != 0) {
        return;
    }

    // Cumulative histogram (clearing the bins for the next frame), then clamp off the
    // percentile tails so shadows and fireflies cannot drag the average.
    uint cumulative[64];
    uint sum = 0;
    for (uint i = 0; i < 64; i++) {
        sum += p.histogram[i];
        cumulative[i] = sum;
        p.histogram[i] = 0;
    }
    let first = uint(float(sum) * p.low_percent);
    let last = uint(float(sum) * p.high_percent);

    uint count = 0;
    float weighted = 0.0;
    for (uint i = 1; i < 64; i++) {
        let bin_count = clamp(cumulative[i], first, last) - clamp(cumulative[i - 1], first, last);
        weighted += float(bin_count) * float(i);
        count += bin_count;
    }

    var state = p.state[0];
    if (count > 0) {
        let avg_log_lum = weighted / (float(count) * 63.0) * p.log_lum_range + p.min_log_lum;
        // The meter reads incident light (radiance over albedo), so the exposure that shows
        // an 18% card as 0.18 is its reciprocal; plus compensation.
        let metered = clamp(p.compensation - avg_log_lum, p.min_ev, p.max_ev);

        // The LOOK goes only part of the way from the reference exposure to the metering,
        // so a night scene still reads darker than a sunlit one (a full-auto look renders
        // both the same). Linear-capped exponential approach (bevy's speed /
        // exponential_transition pair).
        let target = clamp(lerp(p.reference_ev, metered, p.adaptation), p.min_ev, p.max_ev);
        let speed = (target > state.ev) ? p.speed_brighten : p.speed_darken;
        let step_ = speed * p.dt;
        let d = target - state.ev;
        state.ev = clamp(state.ev + clamp(d * step_ / p.exp_transition, -step_, step_),
                         p.min_ev, p.max_ev);

        // Ray Reconstruction's input follows the full metering (it wants mid-gray whatever the
        // look); the CPU eases it there and tells NGX (the pre-exposure).
        state.metered = metered;
    }
    state.exposure = exp2(state.ev);
    p.state[0] = state;
}
