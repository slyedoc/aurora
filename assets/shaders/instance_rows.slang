// The rows a GPU kernel writes into its InstanceBlock (src/compute.rs `BlockRows`, from
// `FrameContext::block_rows`): the block's own rows of this frame's and last frame's instance
// arrays, and how many it holds. Kernels index 0..count within the block and write only through
// these functions, which drop an index past it.
module instance_rows;

// VkAccelerationStructureInstanceKHR: 3x4 row-major transform, custom index:24 | mask:8, SBT
// record offset:24 | flags:8, BLAS device address. 64 bytes.
public struct Instance {
    public float4 t0;
    public float4 t1;
    public float4 t2;
    public uint custom_and_mask;
    public uint sbt_and_flags;
    public uint64_t blas;
};

// `BlockRows` in compute.rs.
public struct BlockRows {
    public Instance* rows;
    public Instance* prev;
    // Per row, the colour a material with `instance_tint` takes (writeTint).
    public float4* tints;
    public uint count;
    public uint pad;
};

// An instance transform: the rows of its 3x4 affine (all zero: the row traces nothing).
public struct Transform {
    public float4 t0;
    public float4 t1;
    public float4 t2;
};

public Transform noTransform() {
    return { float4(0.0), float4(0.0), float4(0.0) };
}

// Row `i`'s transform this frame and last (motion vectors).
public void writeRow(BlockRows b, uint i, Transform now, Transform prev) {
    if (i >= b.count) {
        return;
    }
    b.rows[i].t0 = now.t0;
    b.rows[i].t1 = now.t1;
    b.rows[i].t2 = now.t2;
    b.prev[i].t0 = prev.t0;
    b.prev[i].t1 = prev.t1;
    b.prev[i].t2 = prev.t2;
}

// Row `i`'s geometry: one of the block's poses (`BlockPose` in tlas_builder.rs).
public void writePose(BlockRows b, uint i, uint64_t blas, uint sbt_and_flags) {
    if (i >= b.count) {
        return;
    }
    b.rows[i].blas = blas;
    b.rows[i].sbt_and_flags = sbt_and_flags;
}

// Row `i`'s colour: what a material with `instance_tint` multiplies its base colour and emission
// by.
public void writeTint(BlockRows b, uint i, float4 tint) {
    if (i >= b.count) {
        return;
    }
    b.tints[i] = tint;
}
