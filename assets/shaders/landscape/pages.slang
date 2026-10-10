// A landscape's evaluated heights: a pool of 256² pages over its window (`GpuPagePool` in
// aurora_landscape's page.rs, scalar). Texel (tx, tz) sits at (tx, tz) * texel in the
// landscape's frame; page (px, pz) holds texels px*256.. and pz*256.. and lives in slot
// (pz mod pages_z, px mod pages_x): a streaming window slides without moving pages.
module pages;

public static const int PAGE = 256;

public struct PagePool {
    public float* heights;
    public int min_x;
    public int min_z;
    public int pages_x;
    public int pages_z;
    public float texel;
    public uint pad;
};

// v mod n, never negative. Not `%`: the Vulkan SDK's slangc compiled a negative `int % int`
// as an unsigned remainder (-1 % 14 came out 3). Page indices are far below 2^24.
int wrap_page(int v, int n) {
    return v - n * int(floor(float(v) / float(n)));
}

// The pool slot of page (px, pz).
public int page_slot(int px, int pz, int pages_x, int pages_z) {
    return wrap_page(pz, pages_z) * pages_x + wrap_page(px, pages_x);
}

// The texel's index in a pool laid out like the heights (any per-texel pool: materials too),
// clamped to the window.
public int texel_index(int min_x, int min_z, int pages_x, int pages_z, int tx, int tz) {
    let x = clamp(tx, min_x * PAGE, (min_x + pages_x) * PAGE - 1);
    let z = clamp(tz, min_z * PAGE, (min_z + pages_z) * PAGE - 1);
    return page_slot(x >> 8, z >> 8, pages_x, pages_z) * PAGE * PAGE + (z & 255) * PAGE + (x & 255);
}

// Global texel height, clamped to the pool's window.
public float texel_height(PagePool p, int tx, int tz) {
    return p.heights[texel_index(p.min_x, p.min_z, p.pages_x, p.pages_z, tx, tz)];
}

// Bilinear height at a landscape-local xz (`LandscapePages::height_at`).
public float sample_height(PagePool p, float2 xz) {
    let g = xz / p.texel;
    let f = floor(g);
    let t = g - f;
    let x = int(f.x);
    let z = int(f.y);
    let h00 = texel_height(p, x, z);
    let h10 = texel_height(p, x + 1, z);
    let h01 = texel_height(p, x, z + 1);
    let h11 = texel_height(p, x + 1, z + 1);
    return lerp(lerp(h00, h10, t.x), lerp(h01, h11, t.x), t.y);
}
