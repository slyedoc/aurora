// A texel's materials: its top four (palette id, weight) pairs, packed as uint2 -- x the ids,
// y the weights, a byte each, weights summing to 255 once evaluated. Up to four materials
// meet at any texel (a WoW chunk's four layers import losslessly); a page holds any number.
module materials;

public struct MatSet {
    public uint ids[4];
    public float w[4];
};

public MatSet mat_empty() {
    MatSet s;
    for (int i = 0; i < 4; i++) {
        s.ids[i] = 0;
        s.w[i] = 0.0;
    }
    return s;
}

public MatSet mat_unpack(uint2 t) {
    MatSet s;
    for (int i = 0; i < 4; i++) {
        s.ids[i] = (t.x >> (8 * i)) & 0xFFu;
        s.w[i] = float((t.y >> (8 * i)) & 0xFFu) / 255.0;
    }
    return s;
}

public float mat_total(MatSet s) {
    return s.w[0] + s.w[1] + s.w[2] + s.w[3];
}

// Adds `w` of `id`, merging into its slot or replacing the lightest when it outweighs it.
public void mat_add(inout MatSet s, uint id, float w) {
    if (w <= 0.0) {
        return;
    }
    var lightest = 0;
    for (int i = 0; i < 4; i++) {
        if (s.w[i] > 0.0 && s.ids[i] == id) {
            s.w[i] += w;
            return;
        }
        if (s.w[i] < s.w[lightest]) {
            lightest = i;
        }
    }
    if (w > s.w[lightest]) {
        s.ids[lightest] = id;
        s.w[lightest] = w;
    }
}

public void mat_scale(inout MatSet s, float k) {
    for (int i = 0; i < 4; i++) {
        s.w[i] *= k;
    }
}

public void mat_merge(inout MatSet s, MatSet o, float k) {
    for (int i = 0; i < 4; i++) {
        mat_add(s, o.ids[i], o.w[i] * k);
    }
}

// Normalised to 255 (empty stays empty), rounding error left on the heaviest.
public uint2 mat_pack(MatSet s) {
    let total = mat_total(s);
    if (total <= 0.0) {
        return uint2(0, 0);
    }
    uint q[4];
    var sum = 0u;
    var heaviest = 0;
    for (int i = 0; i < 4; i++) {
        q[i] = uint(round(s.w[i] / total * 255.0));
        sum += q[i];
        if (s.w[i] > s.w[heaviest]) {
            heaviest = i;
        }
    }
    q[heaviest] = uint(clamp(int(q[heaviest]) + 255 - int(sum), 0, 255));
    uint2 t = uint2(0, 0);
    for (int i = 0; i < 4; i++) {
        t.x |= (s.ids[i] & 0xFFu) << (8 * i);
        t.y |= q[i] << (8 * i);
    }
    return t;
}

// As stored by paint pages: weights not normalised, their sum the coverage.
public uint2 mat_pack_raw(MatSet s) {
    uint2 t = uint2(0, 0);
    for (int i = 0; i < 4; i++) {
        t.x |= (s.ids[i] & 0xFFu) << (8 * i);
        t.y |= uint(clamp(round(s.w[i] * 255.0), 0.0, 255.0)) << (8 * i);
    }
    return t;
}
