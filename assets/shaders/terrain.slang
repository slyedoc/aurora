// GPU terrain editing (see src/terrain.rs).
//
// A terrain tile is a regular height grid (`res` x `res`, WoW-style 129 = 16 subchunks x 8
// cells + 1) rendered as core's center-vertex pattern: `res^2` corner vertices then
// `(res-1)^2` cell-center vertices, 4 triangles per cell — no diagonal artifacts and no
// RT self-intersection from folded quads. All editing state lives in buffers the kernels
// own end-to-end:
//
//   terrain_brush     one thread per height texel: raise / smooth / flatten inside the brush.
//   terrain_vertices  one thread per vertex: position from the height grid (corners sample
//                     directly, centers average their 4 corners), normal by central
//                     difference — rewrites the tile's deformed vertex stream.
//   terrain_pack      one thread per triangle: per-triangle shading records from the stream
//                     (a copy of skinning.slang's pack_triangles; same Vertex/Triangle).
//   terrain_paint     one thread per alpha-atlas texel: paint one splat channel (WoW
//                     alphamaps: 16x16 chunk cells of 64x64, layers 1..3 in R/G/B, layer 0
//                     is the base).
//   terrain_compose   one thread per diffuse texel in the dirty rect: blend the palette's
//                     tileset textures with the alpha weights (sequential lerp, WoW's
//                     blending) into the tile's diffuse buffer; the host copies the rect
//                     into the material's texture image.
//
// Every buffer arrives as a raw pointer in push constants (no descriptor sets), which is
// also why the palette textures are plain RGBA byte buffers, not sampled images.

// Deformed vertex stream: `Vertex` in blas.rs (32 bytes, scalar layout).
struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

// Per-triangle shading record: `Triangle` in blas.rs (32 bytes).
struct Triangle {
    uint tangent;
    uint n0, n1, n2;
    uint uv0, uv1, uv2;
    uint padding;
};

// ---- height sampling ---------------------------------------------------------------------

float height_at(float* heights, uint res, uint x, uint z) {
    return heights[min(z, res - 1) * res + min(x, res - 1)];
}

// Bilinear height at grid-space uv in [0,1]. Mirrors core's HeightMap::sample.
float sample_height(float* heights, uint res, float2 uv) {
    let maxi = float(res - 1);
    let x = clamp(uv.x * maxi, 0.0, maxi);
    let z = clamp(uv.y * maxi, 0.0, maxi);
    let x0 = uint(floor(x));
    let z0 = uint(floor(z));
    let x1 = min(x0 + 1, res - 1);
    let z1 = min(z0 + 1, res - 1);
    let fx = x - float(x0);
    let fz = z - float(z0);
    let h00 = height_at(heights, res, x0, z0);
    let h10 = height_at(heights, res, x1, z0);
    let h01 = height_at(heights, res, x0, z1);
    let h11 = height_at(heights, res, x1, z1);
    return lerp(lerp(h00, h10, fx), lerp(h01, h11, fx), fz);
}

// Central-difference normal at uv. Mirrors core's HeightMap::sample_normal (height_scale 1).
float3 sample_normal(float* heights, uint res, float size, float2 uv) {
    let texel = 1.0 / float(res - 1);
    let hl = sample_height(heights, res, uv - float2(texel, 0.0));
    let hr = sample_height(heights, res, uv + float2(texel, 0.0));
    let hd = sample_height(heights, res, uv - float2(0.0, texel));
    let hu = sample_height(heights, res, uv + float2(0.0, texel));
    let step = size * texel;
    let dx = (hr - hl) / (2.0 * step);
    let dz = (hu - hd) / (2.0 * step);
    return normalize(float3(-dx, 1.0, -dz));
}

// Cosine-ish brush falloff: 1 at the center, 0 at the rim.
float falloff(float d, float radius) {
    let t = clamp(d / max(radius, 1e-4), 0.0, 1.0);
    return 1.0 - t * t * (3.0 - 2.0 * t);
}

// ---- terrain_brush -------------------------------------------------------------------------

static const uint BRUSH_RAISE = 0;
static const uint BRUSH_SMOOTH = 1;
static const uint BRUSH_FLATTEN = 2;

struct BrushParams {
    float* heights;
    uint res;
    float size;    // tile world size; grid (gx, gz) sits at ((gx/cells - 0.5) * size, ...)
    float cx;      // brush center, tile-local x/z
    float cz;
    float radius;
    float amount;  // raise: signed height delta; smooth/flatten: lerp factor 0..1
    float target;  // flatten target height
    uint mode;
    uint base;
};
[[vk::push_constant]] BrushParams brush_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_brush(uint3 gid: SV_DispatchThreadID) {
    let p = brush_params;
    let i = p.base + gid.x;
    if (i >= p.res * p.res) {
        return;
    }
    let gx = i % p.res;
    let gz = i / p.res;
    let cells = float(p.res - 1);
    let x = (float(gx) / cells - 0.5) * p.size;
    let z = (float(gz) / cells - 0.5) * p.size;
    let d = distance(float2(x, z), float2(p.cx, p.cz));
    if (d > p.radius) {
        return;
    }
    let w = falloff(d, p.radius);
    let h = p.heights[i];
    float out = h;
    switch (p.mode) {
    case BRUSH_RAISE:
        out = h + p.amount * w;
        break;
    case BRUSH_SMOOTH: {
        // Neighbor average; racing concurrent writes is fine for a brush.
        let hl = height_at(p.heights, p.res, gx > 0 ? gx - 1 : 0, gz);
        let hr = height_at(p.heights, p.res, gx + 1, gz);
        let hd = height_at(p.heights, p.res, gx, gz > 0 ? gz - 1 : 0);
        let hu = height_at(p.heights, p.res, gx, gz + 1);
        out = lerp(h, (hl + hr + hd + hu) * 0.25, clamp(p.amount, 0.0, 1.0) * w);
        break;
    }
    case BRUSH_FLATTEN:
        out = lerp(h, p.target, clamp(p.amount, 0.0, 1.0) * w);
        break;
    default:
        break;
    }
    p.heights[i] = out;
}

// ---- terrain_vertices ------------------------------------------------------------------------

struct VertParams {
    float* heights;
    Vertex* dst;
    uint res;
    float size;
    uint count;  // res^2 corners + (res-1)^2 centers
    uint base;
};
[[vk::push_constant]] VertParams vert_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_vertices(uint3 gid: SV_DispatchThreadID) {
    let p = vert_params;
    let i = p.base + gid.x;
    if (i >= p.count) {
        return;
    }
    let cells = p.res - 1;
    let fcells = float(cells);
    let outer = p.res * p.res;

    float2 uv;
    float y;
    if (i < outer) {
        let gx = i % p.res;
        let gz = i / p.res;
        uv = float2(float(gx) / fcells, float(gz) / fcells);
        y = p.heights[i];
    } else {
        let j = i - outer;
        let cx = j % cells;
        let cz = j / cells;
        uv = float2((float(cx) + 0.5) / fcells, (float(cz) + 0.5) / fcells);
        // Average of the 4 corners (not a bilinear sample): keeps the 4 triangles of the
        // cell planar-adjacent, the RT-stability trick from core's mesh builder.
        let ha = height_at(p.heights, p.res, cx, cz);
        let hb = height_at(p.heights, p.res, cx, cz + 1);
        let hc = height_at(p.heights, p.res, cx + 1, cz + 1);
        let hd = height_at(p.heights, p.res, cx + 1, cz);
        y = (ha + hb + hc + hd) * 0.25;
    }
    let n = sample_normal(p.heights, p.res, p.size, uv);

    Vertex out;
    out.px = (uv.x - 0.5) * p.size;
    out.py = y;
    out.pz = (uv.y - 0.5) * p.size;
    out.nx = n.x;
    out.ny = n.y;
    out.nz = n.z;
    out.u = uv.x;
    out.v = uv.y;
    p.dst[i] = out;
}

// ---- terrain_pack (pack_triangles copy; see skinning.slang) ----------------------------------

uint pack_normal(float3 n) {
    let x = uint((n.x * 0.5 + 0.5) * 65535.0);
    let y = uint((n.y * 0.5 + 0.5) * 32767.0);
    let z = n.z >= 0.0 ? 0u : 1u;
    return (x << 16) | (y << 1) | z;
}

uint pack_uv(float2 uv) {
    return (f32tof16(uv.y) << 16) | f32tof16(uv.x);
}

struct PackParams {
    Vertex* vertices;
    uint* indices;
    Triangle* triangles;
    uint count;
    uint base;
};
[[vk::push_constant]] PackParams pack_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_pack(uint3 gid: SV_DispatchThreadID) {
    let t = pack_params.base + gid.x;
    if (t >= pack_params.count) {
        return;
    }
    let v0 = pack_params.vertices[pack_params.indices[t * 3 + 0]];
    let v1 = pack_params.vertices[pack_params.indices[t * 3 + 1]];
    let v2 = pack_params.vertices[pack_params.indices[t * 3 + 2]];
    let p0 = float3(v0.px, v0.py, v0.pz);
    let p1 = float3(v1.px, v1.py, v1.pz);
    let p2 = float3(v2.px, v2.py, v2.pz);
    let uv0 = float2(v0.u, v0.v);
    let uv1 = float2(v1.u, v1.v);
    let uv2 = float2(v2.u, v2.v);

    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let denom = duv1.x * duv2.y - duv1.y * duv2.x;
    float3 tangent;
    if (abs(denom) < 0.0001) {
        tangent = float3(0.0, 0.0, 1.0);
    } else {
        let f = 1.0 / denom;
        tangent = normalize(float3(
            f * (duv2.y * edge1.x - duv1.y * edge2.x),
            f * (duv2.y * edge1.y - duv1.y * edge2.y),
            f * (duv2.y * edge1.z - duv1.y * edge2.z)));
    }

    Triangle tri;
    tri.tangent = pack_normal(tangent);
    tri.n0 = pack_normal(float3(v0.nx, v0.ny, v0.nz));
    tri.n1 = pack_normal(float3(v1.nx, v1.ny, v1.nz));
    tri.n2 = pack_normal(float3(v2.nx, v2.ny, v2.nz));
    tri.uv0 = pack_uv(uv0);
    tri.uv1 = pack_uv(uv1);
    tri.uv2 = pack_uv(uv2);
    tri.padding = 0;
    pack_params.triangles[t] = tri;
}

// ---- terrain_paint ---------------------------------------------------------------------------

// Painting is BY TEXTURE: each texel resolves which of its chunk's layer slots holds the
// palette texture and bumps that slot's weight (slot 0 is the base — painting it means
// eroding the others). `tex == PAINT_ERASE_ALL` scales every weight down instead.
static const uint PAINT_ERASE_ALL = 0xfffffffe;

struct PaintParams {
    uint* alpha;          // RGBA8 per texel; 16x16 chunk cells of `cell` px in an `atlas` atlas
    uint4* chunk_layers;  // 256 palette indices per chunk (the compose table)
    uint atlas;           // atlas edge (e.g. 1024)
    uint cell;            // px per chunk cell (e.g. 64)
    float size;           // tile world size
    float cx;
    float cz;
    float radius;
    float amount;         // weight added (0..1) at full falloff
    uint tex;             // palette index to paint, or PAINT_ERASE_ALL
    uint base;
};
[[vk::push_constant]] PaintParams paint_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_paint(uint3 gid: SV_DispatchThreadID) {
    let p = paint_params;
    let i = p.base + gid.x;
    if (i >= p.atlas * p.atlas) {
        return;
    }
    let px = i % p.atlas;
    let py = i / p.atlas;
    // Texel -> tile-local world position: chunk cell, then the fraction inside it.
    let col = px / p.cell;
    let row = py / p.cell;
    let fx = (float(px % p.cell) + 0.5) / float(p.cell);
    let fy = (float(py % p.cell) + 0.5) / float(p.cell);
    let x = ((float(col) + fx) / 16.0 - 0.5) * p.size;
    let z = ((float(row) + fy) / 16.0 - 0.5) * p.size;
    let d = distance(float2(x, z), float2(p.cx, p.cz));
    if (d > p.radius) {
        return;
    }
    let w = falloff(d, p.radius) * clamp(p.amount, 0.0, 1.0);

    let packed = p.alpha[i];
    float4 a = float4(
        float(packed & 0xff),
        float((packed >> 8) & 0xff),
        float((packed >> 16) & 0xff),
        float((packed >> 24) & 0xff));
    if (p.tex == PAINT_ERASE_ALL) {
        a.xyz *= (1.0 - w);
    } else {
        // Which of this chunk's slots carries the texture (the CPU added it to the table
        // before the stroke). Slot 0 = base: painting the base erodes the overlays.
        let layers = p.chunk_layers[row * 16 + col];
        if (layers.x == p.tex) {
            a.xyz *= (1.0 - w);
        } else if (layers.y == p.tex) {
            a.x = min(a.x + w * 255.0, 255.0);
        } else if (layers.z == p.tex) {
            a.y = min(a.y + w * 255.0, 255.0);
        } else if (layers.w == p.tex) {
            a.z = min(a.z + w * 255.0, 255.0);
        }
    }
    p.alpha[i] =
        (uint(a.w) << 24) | (uint(a.z) << 16) | (uint(a.y) << 8) | uint(a.x);
}

// ---- terrain_compose -------------------------------------------------------------------------

static const uint NO_LAYER = 0xffffffff;

// One palette (tileset) texture as a raw RGBA byte buffer. Matches `PaletteEntryGpu`.
struct PaletteEntry {
    uint* rgba;
    uint w;
    uint h;
    float repeats;  // texture repeats per chunk cell edge
    uint pad;
};

float3 palette_fetch(PaletteEntry t, uint x, uint y) {
    let p = t.rgba[y * t.w + x];
    return float3(float(p & 0xff), float((p >> 8) & 0xff), float((p >> 16) & 0xff));
}

// Bilinear, wrapping sample of a palette texture at chunk-cell uv (repeats applied).
float3 palette_sample(PaletteEntry t, float2 cell_uv) {
    let u = cell_uv.x * t.repeats;
    let v = cell_uv.y * t.repeats;
    let fw = float(t.w);
    let fh = float(t.h);
    let x = (u - floor(u)) * fw - 0.5;
    let y = (v - floor(v)) * fh - 0.5;
    let x0f = floor(x);
    let y0f = floor(y);
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = uint(int(x0f) + int(fw)) % t.w;
    let y0 = uint(int(y0f) + int(fh)) % t.h;
    let x1 = (x0 + 1) % t.w;
    let y1 = (y0 + 1) % t.h;
    let c00 = palette_fetch(t, x0, y0);
    let c10 = palette_fetch(t, x1, y0);
    let c01 = palette_fetch(t, x0, y1);
    let c11 = palette_fetch(t, x1, y1);
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

// Bilinear alpha-weight sample inside one chunk's 64x64 atlas cell (edge-clamped to the cell).
float alpha_sample(uint* alpha, uint atlas, uint cell, uint2 chunk, float2 cell_uv, uint ch) {
    let fcell = float(cell);
    let x = clamp(cell_uv.x * fcell - 0.5, 0.0, fcell - 1.0);
    let y = clamp(cell_uv.y * fcell - 0.5, 0.0, fcell - 1.0);
    let x0 = uint(floor(x));
    let y0 = uint(floor(y));
    let x1 = min(x0 + 1, cell - 1);
    let y1 = min(y0 + 1, cell - 1);
    let tx = x - float(x0);
    let ty = y - float(y0);
    let bx = chunk.x * cell;
    let by = chunk.y * cell;
    let w00 = float((alpha[(by + y0) * atlas + bx + x0] >> (ch * 8)) & 0xff);
    let w10 = float((alpha[(by + y0) * atlas + bx + x1] >> (ch * 8)) & 0xff);
    let w01 = float((alpha[(by + y1) * atlas + bx + x0] >> (ch * 8)) & 0xff);
    let w11 = float((alpha[(by + y1) * atlas + bx + x1] >> (ch * 8)) & 0xff);
    return lerp(lerp(w00, w10, tx), lerp(w01, w11, tx), ty) / 255.0;
}

struct ComposeParams {
    uint* alpha;          // the tile's alpha atlas
    uint4* chunk_layers;  // 256 palette indices per chunk, NO_LAYER = unused
    PaletteEntry* palette;
    uint* dst;            // diffuse RGBA buffer, out_px x out_px
    uint out_px;
    uint atlas;
    uint cell;
    // Dirty rect in dst pixels (x1/y1 exclusive); threads cover the rect.
    uint x0;
    uint y0;
    uint x1;
    uint y1;
    uint base;
};
[[vk::push_constant]] ComposeParams compose_params;

[shader("compute")]
[numthreads(64, 1, 1)]
void terrain_compose(uint3 gid: SV_DispatchThreadID) {
    let p = compose_params;
    let rw = p.x1 - p.x0;
    let i = p.base + gid.x;
    if (i >= rw * (p.y1 - p.y0)) {
        return;
    }
    let rx = p.x0 + i % rw;
    let ry = p.y0 + i / rw;
    let out_cell = p.out_px / 16;
    let chunk = uint2(rx / out_cell, ry / out_cell);
    let ci = chunk.y * 16 + chunk.x;
    let cell_uv = float2(
        (float(rx % out_cell) + 0.5) / float(out_cell),
        (float(ry % out_cell) + 0.5) / float(out_cell));

    let layers = p.chunk_layers[ci];
    // Sequential lerp in layer order — WoW's alphamap blending. Layer 0 is the base.
    float3 color = float3(70.0, 90.0, 50.0);
    if (layers.x != NO_LAYER) {
        color = palette_sample(p.palette[layers.x], cell_uv);
    }
    if (layers.y != NO_LAYER) {
        let w = alpha_sample(p.alpha, p.atlas, p.cell, chunk, cell_uv, 0);
        color = lerp(color, palette_sample(p.palette[layers.y], cell_uv), w);
    }
    if (layers.z != NO_LAYER) {
        let w = alpha_sample(p.alpha, p.atlas, p.cell, chunk, cell_uv, 1);
        color = lerp(color, palette_sample(p.palette[layers.z], cell_uv), w);
    }
    if (layers.w != NO_LAYER) {
        let w = alpha_sample(p.alpha, p.atlas, p.cell, chunk, cell_uv, 2);
        color = lerp(color, palette_sample(p.palette[layers.w], cell_uv), w);
    }
    p.dst[ry * p.out_px + rx] =
        0xff000000u | (uint(color.z) << 16) | (uint(color.y) << 8) | uint(color.x);
}
