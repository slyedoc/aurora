// The path tracer's per-invocation RNG (rand.glsl in Slang): a PCG stream seeded by TEA.
module rand;

static uint g_seed;

public uint rand() {
    let prev = g_seed * 747796405u + 2891336453u;
    let word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
    g_seed = prev;
    return (word >> 22u) ^ word;
}

public float randf() {
    let r = rand();
    return asfloat(0x3f800000u | (r >> 9)) - 1.0;
}

// NVIDIA 2021 nvpro-samples
public uint tea(uint val0, uint val1) {
    var v0 = val0;
    var v1 = val1;
    var s0 = 0u;
    for (uint n = 0; n < 16; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
    }
    return v0;
}

public void initRandom(uint2 resolution, uint2 screenCoord, uint frame) {
    g_seed = tea(screenCoord.y * resolution.x + screenCoord.x, frame);
}
