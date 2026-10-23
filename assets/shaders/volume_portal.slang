// The ray-portal table (src/portal.rs `GpuPortal`) and the volume portal's shape, shared by the
// raygen and the pick.
module volume_portal;

// Each entry is PORTAL_STRIDE uint4s: x instance slot, y target (the exit's slot, or a volume's
// environment bit), z valid, w kind; then a volume's big-shape rows, their inverse and its shape.
public static const uint PORTAL_STRIDE = 8u;
public static const uint PORTAL_KIND_VOLUME = 1u;

public struct VolumePortal {
    public uint slot;
    public uint env_bit;
    // world_from_big, big_from_world (instance-row layout), and radius / wall / dome.
    public float4 t0, t1, t2;
    public float4 i0, i1, i2;
    public float4 shape;
};

public VolumePortal volumePortalLoad(uint4* portals, uint i) {
    let b = i * PORTAL_STRIDE;
    VolumePortal v;
    v.slot = portals[b].x;
    v.env_bit = portals[b].y;
    v.t0 = asfloat(portals[b + 1u]);
    v.t1 = asfloat(portals[b + 2u]);
    v.t2 = asfloat(portals[b + 3u]);
    v.i0 = asfloat(portals[b + 4u]);
    v.i1 = asfloat(portals[b + 5u]);
    v.i2 = asfloat(portals[b + 6u]);
    v.shape = asfloat(portals[b + 7u]);
    return v;
}

// The volume portal whose small shape is instance `slot`: its table index + 1, 0 for none.
public uint volumePortalOf(uint4* portals, uint count, uint slot) {
    for (uint i = 0u; i < count; i++) {
        let h = portals[i * PORTAL_STRIDE];
        if (h.z != 0u && h.w == PORTAL_KIND_VOLUME && h.x == slot) {
            return i + 1u;
        }
    }
    return 0u;
}

// Instance rows are the VkAccelerationStructureInstanceKHR 3x4 transform, row-major:
// r_k = row k of world_from_local. Inverse via the 3x3 adjugate, -L^-1 t for translation.
public void portalAffineInverse(float4 r0, float4 r1, float4 r2,
                                out float4 o0, out float4 o1, out float4 o2) {
    let c0 = float3(r0.x, r1.x, r2.x);
    let c1 = float3(r0.y, r1.y, r2.y);
    let c2 = float3(r0.z, r1.z, r2.z);
    let t = float3(r0.w, r1.w, r2.w);
    let inv_det = 1.0 / dot(c0, cross(c1, c2));
    let i0 = cross(c1, c2) * inv_det;
    let i1 = cross(c2, c0) * inv_det;
    let i2 = cross(c0, c1) * inv_det;
    o0 = float4(i0, -dot(i0, t));
    o1 = float4(i1, -dot(i1, t));
    o2 = float4(i2, -dot(i2, t));
}

public float3 portalXformPoint(float4 r0, float4 r1, float4 r2, float3 p) {
    let h = float4(p, 1.0);
    return float3(dot(r0, h), dot(r1, h), dot(r2, h));
}

public float3 portalXformDir(float4 r0, float4 r1, float4 r2, float3 v) {
    return float3(dot(r0.xyz, v), dot(r1.xyz, v), dot(r2.xyz, v));
}

// How far along `d` (any length: the result is in its units) a point `o` inside the shape,
// both in the shape's local space, leaves it: the nearest of the wall, the floor and the cap.
public float volumeExitLocal(float4 shape, float3 o, float3 d) {
    let r = shape.x;
    let wall = shape.y;
    let dome = shape.z;
    var t = 1.0e30;
    let a = d.x * d.x + d.z * d.z;
    if (a > 0.0) {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - r * r;
        let disc = b * b - a * c;
        if (disc >= 0.0) {
            t = min(t, (-b + sqrt(disc)) / a);
        }
    }
    if (d.y < 0.0) {
        t = min(t, -o.y / d.y);
    }
    if (dome > 1.0e-5) {
        let big = (r * r + dome * dome) / (2.0 * dome);
        let oc = o - float3(0.0, wall + dome - big, 0.0);
        let qa = dot(d, d);
        let qb = dot(oc, d);
        let disc = qb * qb - qa * (dot(oc, oc) - big * big);
        if (disc >= 0.0) {
            let far = (-qb + sqrt(disc)) / qa;
            if (far > 0.0 && o.y + d.y * far >= wall - 1.0e-5 * r) {
                t = min(t, far);
            }
        }
    } else if (d.y > 0.0) {
        t = min(t, (wall - o.y) / d.y);
    }
    return max(t, 0.0);
}

// The exit along a target-space ray (unit `d`), in target metres.
public float volumeExit(VolumePortal v, float3 o, float3 d) {
    return volumeExitLocal(v.shape, portalXformPoint(v.i0, v.i1, v.i2, o), portalXformDir(v.i0, v.i1, v.i2, d));
}

float strataHash(float3 p) {
    let q = frac(p * float3(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yzx + 33.33);
    return frac((r.x + r.y) * r.z);
}

// The cut's earth: `depth` under the ground over it, `local` the point in the shape's space
// (both in the shape's units); bands by depth as a share of the wall, wavy around the rim.
public float3 strataAlbedo(float4 shape, float depth, float3 local) {
    let scale = max(shape.y, 0.05 * shape.x);
    let around = atan2(local.z, local.x);
    let wave = 0.04 * sin(around * 5.0 + 1.3) + 0.025 * sin(around * 13.0 + local.y * 40.0);
    let f = depth / scale + wave;
    let grain = strataHash(floor(local * (400.0 / max(shape.x, 1.0e-3))));
    float3 c;
    if (f < 0.08) {
        c = float3(0.075, 0.055, 0.04);
    } else if (f < 0.32) {
        c = float3(0.21, 0.14, 0.085);
    } else if (f < 0.52) {
        c = float3(0.36, 0.25, 0.14);
    } else {
        let band = frac((f - 0.52) * 2.6 + 0.15 * sin(around * 3.0));
        c = band < 0.55 ? float3(0.31, 0.30, 0.29) : float3(0.19, 0.185, 0.18);
        if (grain > 0.93) {
            c *= 1.35;
        }
    }
    return c * (0.85 + 0.3 * grain);
}
