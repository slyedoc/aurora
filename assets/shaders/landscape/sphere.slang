// Cube-sphere faces for planet landscapes (aurora_landscape `SphereFace`). A face's
// landscape-local (x, z) lies on its cube face, half-side = radius: the point
// axis * radius + u * x + v * z, normalised, is the direction out of the planet. The bases
// mirror the flat case (u x v = -axis), so tiles wind outward the way flat ones wind up.
module sphere;

public void face_basis(uint face, out float3 axis, out float3 u, out float3 v) {
    switch (face) {
        case 0: axis = float3(1, 0, 0); u = float3(0, 0, 1); v = float3(0, 1, 0); break;
        case 1: axis = float3(-1, 0, 0); u = float3(0, 0, -1); v = float3(0, 1, 0); break;
        case 2: axis = float3(0, 1, 0); u = float3(1, 0, 0); v = float3(0, 0, 1); break;
        case 3: axis = float3(0, -1, 0); u = float3(1, 0, 0); v = float3(0, 0, -1); break;
        case 4: axis = float3(0, 0, 1); u = float3(0, 1, 0); v = float3(1, 0, 0); break;
        default: axis = float3(0, 0, -1); u = float3(1, 0, 0); v = float3(0, 1, 0); break;
    }
}

// The direction out of the planet through face-local (x, z).
public float3 face_direction(uint face, float radius, float2 xz) {
    float3 axis, u, v;
    face_basis(face, axis, u, v);
    return normalize(axis * radius + u * xz.x + v * xz.y);
}

// Face-local (x, z) of a direction (planet-centred), projected onto this face's plane.
public float2 face_coords(uint face, float radius, float3 dir) {
    float3 axis, u, v;
    face_basis(face, axis, u, v);
    let cube = dir / max(dot(dir, axis), 1.0e-6) * radius;
    return float2(dot(cube, u), dot(cube, v));
}
