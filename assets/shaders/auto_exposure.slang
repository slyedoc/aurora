// Auto-exposure metering (a port of bevy_post_process::auto_exposure; src/auto_exposure.rs).
//
// The raygen writes every pixel's raw-radiance luminance (nits) into `lum`. `ae_histogram`
// bins last frame's buffer into 64 log2-luminance bins (groupshared, then merged), and
// `ae_resolve` trims the percentile tails, averages what is left, adapts the exposure over
// time, and writes the linear exposure the NEXT frame's raygen applies.

// Fixed thread count for `ae_histogram`: 256 groups of 64, each thread striding the buffer.
static const uint AE_HISTOGRAM_THREADS = 16384;

// Must match AeGpu in src/auto_exposure.rs (and AeData in types.glsl).
struct AeState {
    float ev;              // Smoothed log2 exposure (the Auto look).
    float exposure;        // exp2(ev).
    // What the raygen actually applies: `ev` quantised to whole EV steps with hysteresis.
    // Ray Reconstruction's temporal history cannot be told about per-frame exposure
    // changes, so its input exposure must sit still; the blit maps input -> look.
    float input_ev;
    float input_exposure;  // exp2(input_ev)
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
    float input_deadband;  // EV off the metering at which the input exposure snaps to it.
    float input_speed;     // EV per second the input exposure drifts towards the metering.
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

    for (uint i = gid.x; i < p.pixel_count; i += AE_HISTOGRAM_THREADS) {
        let lum = p.lum[i];
        uint bin = 0;
        if (lum >= exp2(p.min_log_lum)) {
            let t = saturate((log2(lum) - p.min_log_lum) * p.inv_log_lum_range);
            bin = uint(t * 62.0 + 1.0);
        }
        InterlockedAdd(shared_bins[bin], 1);
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
        // The exposure that puts the trimmed average at photographic mid-gray (0.18), plus
        // compensation.
        let metered = clamp(p.compensation + log2(0.18) - avg_log_lum, p.min_ev, p.max_ev);

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

        // The raygen's INPUT exposure follows the full metering -- Ray Reconstruction wants
        // mid-gray whatever the look -- but RR cannot be told about exposure changes: a
        // step rings through its temporal history as a full-screen transient (measured
        // 16% frame RMSE while tracking at look speed, 0.7% held). So it holds still
        // within a quarter stop, drifts slowly enough beyond that for the history to keep
        // up unseen, and snaps -- one honest pop, straight to the metered value -- only
        // when the light changed wholesale.
        let off = metered - state.input_ev;
        if (abs(off) > p.input_deadband) {
            state.input_ev = metered;
        } else if (abs(off) > 0.25) {
            let drift = p.input_speed * p.dt;
            state.input_ev += clamp(off, -drift, drift);
        }
    }
    state.exposure = exp2(state.ev);
    state.input_exposure = exp2(state.input_ev);
    p.state[0] = state;
}
