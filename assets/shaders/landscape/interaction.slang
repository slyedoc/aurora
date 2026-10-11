// The interaction field (aurora_landscape interaction.rs): a toroidal grid of the push each
// patch of ground carries, decayed towards rest and splatted with this frame's pushers. The
// scatter kernel reads it (`fieldPush`) to shear plants away from whatever walked through.
module interaction;

public static const uint FIELD_N = 256;
public static const float FIELD_CELL = 0.25;

// `FieldTexel` in interaction.rs: the push (xz, the tangent of the lean) and the world cell it
// belongs to, packed.
public struct FieldTexel {
    public float2 push;
    public uint owner;
    public uint pad;
};

// Floored modulo: Slang's `%` truncates, so a negative cell would wrap wrong.
int wrapCell(int c) {
    return c - int(FIELD_N) * int(floor(float(c) / float(FIELD_N)));
}

uint packCell(int2 cell) {
    return (uint(cell.x) & 0xFFFFu) | ((uint(cell.y) & 0xFFFFu) << 16);
}

uint texelIndex(int2 cell) {
    return uint(wrapCell(cell.y)) * FIELD_N + uint(wrapCell(cell.x));
}

// The push at world xz (zero where the field holds another cell, or with no field).
public float2 fieldPush(FieldTexel* field, float2 xz) {
    if (field == nullptr) {
        return float2(0.0);
    }
    let cell = int2(floor(xz / FIELD_CELL));
    let texel = field[texelIndex(cell)];
    return texel.owner == packCell(cell) ? texel.push : float2(0.0);
}

// `FieldPush` in interaction.rs.
struct FieldPush {
    FieldTexel* field;
    // x, z, radius, strength per pusher.
    float4* pushers;
    uint count;
    // Per-frame decay factor: exp(-dt / recover).
    float decay;
    // The viewer's cell: the window is the N cells centred on it.
    int centre_x;
    int centre_z;
};
[[vk::push_constant]] FieldPush push;

[shader("compute")]
[numthreads(64, 1, 1)]
void field_update(uint3 gid: SV_DispatchThreadID) {
    let i = gid.x;
    if (i >= FIELD_N * FIELD_N) {
        return;
    }
    // The world cell this texel holds: the one in the window with this residue.
    let half = int(FIELD_N / 2u);
    let lo = int2(push.centre_x, push.centre_z) - half;
    let local = int2(int(i % FIELD_N), int(i / FIELD_N));
    let cell = lo + int2(wrapCell(local.x - lo.x), wrapCell(local.y - lo.y));
    var texel = push.field[i];
    if (texel.owner != packCell(cell)) {
        texel.push = float2(0.0);
        texel.owner = packCell(cell);
    }
    texel.push *= push.decay;
    let at = (float2(cell) + 0.5) * FIELD_CELL;
    for (uint p = 0u; p < push.count; p++) {
        let pusher = push.pushers[p];
        let d = at - pusher.xy;
        let dist = length(d);
        if (dist >= pusher.z) {
            continue;
        }
        // Away from the pusher, strongest at its centre (a smooth falloff to the radius).
        let f = smoothstep(pusher.z, 0.0, dist) * pusher.w;
        let away = dist > 1.0e-4 ? d / dist : float2(0.0);
        let candidate = away * f;
        if (dot(candidate, candidate) > dot(texel.push, texel.push)) {
            texel.push = candidate;
        }
    }
    push.field[i] = texel;
}
