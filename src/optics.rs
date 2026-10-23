//! Real glass optics for the tracer: spherical lenses lathed into meshes, and telescopes
//! designed from what you want out of them (magnification, length, aperture). Nothing is
//! faked: the meshes' normals are the optics, and the image is what rays refracting through
//! them make of the scene.
//!
//! [`Telescope::galilean`] (upright) and [`Telescope::keplerian`] (inverted, wider field) solve the
//! thick-lens system (paraxial ray-transfer matrices), so the scope is exactly afocal at exactly
//! the asked magnification on the axis; spherical surfaces add their own pincushion distortion
//! toward the field's edge, as real ones do.

use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages,
    math::{DMat2, DVec2},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

/// Ordinary crown glass.
pub const CROWN_GLASS_IOR: f32 = 1.52;

/// A spherical lens about Z, its front face toward +Z. Curvature radii are signed: positive
/// convex (bulging outward), negative concave; `aperture` is its radius, `thickness` at the axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub r_front: f32,
    pub r_back: f32,
    pub aperture: f32,
    pub thickness: f32,
}

impl Lens {
    /// Height of the front face along Z at `x` from the axis.
    pub fn z_front(&self, x: f32) -> f32 {
        self.thickness / 2.0 - sag(x, self.r_front)
    }

    /// Height of the back face along Z at `x` from the axis.
    pub fn z_back(&self, x: f32) -> f32 {
        -self.thickness / 2.0 + sag(x, self.r_back)
    }

    /// Thickness at the rim.
    pub fn edge_thickness(&self) -> f32 {
        self.z_front(self.aperture) - self.z_back(self.aperture)
    }

    /// The lens as a closed solid (front cap, edge band, back cap), so the tracer knows when a
    /// ray is inside; normals from the faces' true slope, not the triangles.
    pub fn mesh(&self) -> Mesh {
        const SEGMENTS: usize = 96;
        const RINGS: usize = 24;
        let aperture = self.aperture;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // The profile's slope: the lathe of (x, z(x)) has normal ~ (-z'(x) * radial, 1).
        let slope = |z: &dyn Fn(f32) -> f32, x: f32| {
            let eps = (aperture * 1e-3).max(1e-6);
            let lo = (x - eps).max(0.0);
            let hi = (x + eps).min(aperture);
            (z(hi) - z(lo)) / (hi - lo)
        };

        // One revolved face, rings from the axis out; `flip` -1 for the back (faces -Z).
        let mut lathe_cap = |z: &dyn Fn(f32) -> f32, flip: f32| {
            let base = positions.len() as u32;
            positions.push([0.0, 0.0, z(0.0)]);
            normals.push([0.0, 0.0, flip]);
            uvs.push([0.5, 0.5]);
            for ring in 1..=RINGS {
                let x = aperture * ring as f32 / RINGS as f32;
                let dz = slope(z, x);
                for seg in 0..=SEGMENTS {
                    let (sin_t, cos_t) = (2.0 * PI * seg as f32 / SEGMENTS as f32).sin_cos();
                    positions.push([x * cos_t, x * sin_t, z(x)]);
                    normals.push(
                        (Vec3::new(-dz * cos_t, -dz * sin_t, 1.0).normalize() * flip).to_array(),
                    );
                    uvs.push([
                        0.5 + 0.5 * (x / aperture) * cos_t,
                        0.5 + 0.5 * (x / aperture) * sin_t,
                    ]);
                }
            }
            let ring_start = |ring: usize| base + 1 + ((ring - 1) * (SEGMENTS + 1)) as u32;
            for seg in 0..SEGMENTS as u32 {
                let (a, b) = (ring_start(1) + seg, ring_start(1) + seg + 1);
                if flip > 0.0 {
                    indices.extend([base, a, b]);
                } else {
                    indices.extend([base, b, a]);
                }
            }
            for ring in 1..RINGS {
                for seg in 0..SEGMENTS as u32 {
                    let (a, b) = (ring_start(ring) + seg, ring_start(ring) + seg + 1);
                    let (c, d) = (ring_start(ring + 1) + seg, ring_start(ring + 1) + seg + 1);
                    if flip > 0.0 {
                        indices.extend([a, c, d, a, d, b]);
                    } else {
                        indices.extend([a, d, c, a, b, d]);
                    }
                }
            }
        };
        lathe_cap(&|x| self.z_front(x), 1.0);
        lathe_cap(&|x| self.z_back(x), -1.0);

        // Edge band: a cylinder at the aperture joining the two rims, facing out.
        let band = positions.len() as u32;
        for (z, v) in [(self.z_front(aperture), 0.0), (self.z_back(aperture), 1.0)] {
            for seg in 0..=SEGMENTS {
                let (sin_t, cos_t) = (2.0 * PI * seg as f32 / SEGMENTS as f32).sin_cos();
                positions.push([aperture * cos_t, aperture * sin_t, z]);
                normals.push([cos_t, sin_t, 0.0]);
                uvs.push([seg as f32 / SEGMENTS as f32, v]);
            }
        }
        let row = SEGMENTS as u32 + 1;
        for seg in 0..SEGMENTS as u32 {
            let (a, b) = (band + seg, band + seg + 1);
            let (c, d) = (band + row + seg, band + row + seg + 1);
            indices.extend([a, d, b, a, c, d]);
        }

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
    }
}

/// Which kind of two-lens telescope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelescopeKind {
    /// Biconvex objective, biconcave eyepiece inside its focus: upright image, short, a narrow
    /// field that shrinks as the magnification grows, the eye pressed close.
    Galilean,
    /// Biconvex objective and eyepiece, their foci together: a wider field and a real exit pupil
    /// some way behind the eyepiece where the eye goes, but the image upside down (an
    /// astronomical scope; a riflescope adds an erector to turn it back).
    Keplerian,
}

/// A telescope along Z, looking down -Z: the objective at `objective_z`, the eyepiece at
/// `eyepiece_z`, the eye best at `eye_z` (all relative to the scope's own origin, midway between
/// the lenses).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Telescope {
    pub kind: TelescopeKind,
    pub objective: Lens,
    pub eyepiece: Lens,
    pub objective_z: f32,
    pub eyepiece_z: f32,
    pub eye_z: f32,
    /// Where the objective forms its real image of the distance, between the lenses: a
    /// crosshair there ([`Telescope::crosshair_mesh`]) is as sharp as the target. A Galilean has
    /// none (its eyepiece takes the light before it focuses).
    pub focal_plane_z: Option<f32>,
    pub magnification: f32,
    pub ior: f32,
}

impl Telescope {
    /// A Galilean telescope ([`TelescopeKind::Galilean`]) of `magnification` times, the lenses'
    /// centres `length` apart, the objective `objective_radius` across (half its diameter), in
    /// crown glass. Panics when no such pair of lenses exists (a strong eyepiece in a short scope
    /// wants a radius smaller than its own aperture); a longer scope or a smaller magnification
    /// fixes it.
    pub fn galilean(magnification: f32, length: f32, objective_radius: f32) -> Self {
        Self::design(
            TelescopeKind::Galilean,
            magnification,
            length,
            objective_radius,
            CROWN_GLASS_IOR,
        )
    }

    /// A Keplerian telescope ([`TelescopeKind::Keplerian`], the image inverted), as
    /// [`Telescope::galilean`].
    pub fn keplerian(magnification: f32, length: f32, objective_radius: f32) -> Self {
        Self::design(
            TelescopeKind::Keplerian,
            magnification,
            length,
            objective_radius,
            CROWN_GLASS_IOR,
        )
    }

    /// Either kind, in glass of index `ior`. Solves the thick-lens system (paraxial ray-transfer
    /// matrices) for symmetric lenses: afocal, at exactly `magnification` on the axis.
    pub fn design(
        kind: TelescopeKind,
        magnification: f32,
        length: f32,
        objective_radius: f32,
        ior: f32,
    ) -> Self {
        assert!(
            magnification > 1.0,
            "a telescope magnifies: {magnification}"
        );
        assert!(length > 0.0 && objective_radius > 0.0 && ior > 1.0);
        let (m, l, a1, n) = (
            magnification as f64,
            length as f64,
            objective_radius as f64,
            ior as f64,
        );
        // Thin-lens focal lengths: |f1| + f2 = l either way, f1 / |f2| = m.
        let keplerian = kind == TelescopeKind::Keplerian;
        let f2 = if keplerian {
            l / (m + 1.0)
        } else {
            l / (m - 1.0)
        };
        let f1 = m * f2;
        // The eyepiece passes the exit pupil (a1 / m) with room for the field; a Keplerian's
        // field is the eyepiece's own, so it gets a share of its focal length too.
        let a2 = if keplerian {
            (1.2 * a1 / m).max(0.3 * f2)
        } else {
            1.6 * a1 / m
        };
        // Both lenses keep a fifth of their aperture at the thinnest (the rim of a convex one,
        // the centre of a concave one).
        let d1 = |r1: f64| 0.2 * a1 + 2.0 * sag64(a1, r1);
        let d2 = |r2: f64| {
            if keplerian {
                0.2 * a2 + 2.0 * sag64(a2, r2)
            } else {
                0.3 * a2
            }
        };
        // The eyepiece's first face (toward the objective) bulges toward it when convex.
        let s = if keplerian { 1.0 } else { -1.0 };
        // Light from the object travelling +Z; (height, angle) through the system.
        let refract = |r: f64, n1: f64, n2: f64| {
            DMat2::from_cols(
                DVec2::new(1.0, -(n2 - n1) / (n2 * r)),
                DVec2::new(0.0, n1 / n2),
            )
        };
        let travel = |t: f64| DMat2::from_cols(DVec2::new(1.0, 0.0), DVec2::new(t, 1.0));
        let system = |r1: f64, r2: f64| {
            refract(-s * r2, n, 1.0)
                * travel(d2(r2))
                * refract(s * r2, 1.0, n)
                * travel(l - d1(r1) / 2.0 - d2(r2) / 2.0)
                * refract(-r1, n, 1.0)
                * travel(d1(r1))
                * refract(r1, 1.0, n)
        };
        // Afocal (C = 0) at angular magnification D = m upright, -m inverted; Newton from the
        // thin-lens answer (f = R / (2 (n - 1)) for a symmetric lens).
        let target = if keplerian { -m } else { m };
        let residual = |x: DVec2| {
            let sys = system(x.x, x.y);
            DVec2::new(sys.x_axis.y, sys.y_axis.y - target)
        };
        let mut x = 2.0 * (n - 1.0) * DVec2::new(f1, f2);
        for _ in 0..30 {
            let f = residual(x);
            let h = x * 1e-7;
            let j = DMat2::from_cols(
                (residual(x + DVec2::new(h.x, 0.0)) - f) / h.x,
                (residual(x + DVec2::new(0.0, h.y)) - f) / h.y,
            );
            x -= j.inverse() * f;
        }
        let (r1, r2) = (x.x, x.y);
        assert!(
            residual(x).length() < 1e-6 && r1 > a1 * 1.05 && r2 > a2 * 1.05,
            "no {kind:?} telescope of {magnification}x, {length} m long, {objective_radius} m \
             objective radius: its eyepiece would need a radius of {r2:.4} m for a {a2:.4} m \
             aperture; make it longer or weaker"
        );

        let objective = Lens {
            r_front: r1 as f32,
            r_back: r1 as f32,
            aperture: a1 as f32,
            thickness: d1(r1) as f32,
        };
        let eyepiece = Lens {
            r_front: (s * r2) as f32,
            r_back: (s * r2) as f32,
            aperture: a2 as f32,
            thickness: d2(r2) as f32,
        };
        let (objective_z, eyepiece_z) = (-length / 2.0, length / 2.0);
        assert!(
            eyepiece_z - eyepiece.thickness.max(eyepiece.edge_thickness()) / 2.0
                > objective_z + objective.thickness / 2.0,
            "the lenses of a {magnification}x, {length} m {kind:?} telescope overlap; make it longer"
        );
        // The exit pupil, where the eye goes: the image of the objective, where a ray through
        // the objective's middle crosses the axis after the eyepiece. A Galilean's is virtual,
        // inside the scope: there the eye goes close.
        let chief = system(r1, r2) * DVec2::new(0.0, 1.0);
        let relief = if keplerian {
            (-chief.x / chief.y) as f32
        } else {
            2.0 * eyepiece.aperture
        };
        // The objective's image of a point at infinity: a ray parallel to the axis, through the
        // objective alone, crosses it there.
        let focal_plane_z = keplerian.then(|| {
            let ray =
                refract(-r1, n, 1.0) * travel(d1(r1)) * refract(r1, 1.0, n) * DVec2::new(1.0, 0.0);
            objective_z + objective.thickness / 2.0 + (-ray.x / ray.y) as f32
        });
        Self {
            kind,
            objective,
            eyepiece,
            objective_z,
            eyepiece_z,
            eye_z: eyepiece_z + eyepiece.thickness / 2.0 + relief,
            focal_plane_z,
            magnification,
            ior,
        }
    }

    /// A crosshair to stand at [`Telescope::focal_plane_z`] (facing along Z, centred on the
    /// axis): two wires `width` thick across a circle of `radius` (the barrel's inside there).
    /// None for a Galilean, where it could only be a blurred shadow.
    pub fn crosshair_mesh(&self, radius: f32, width: f32) -> Option<Mesh> {
        self.focal_plane_z?;
        let mut mesh = Cuboid::new(2.0 * radius, width, width).mesh().build();
        mesh.merge(&Cuboid::new(width, 2.0 * radius, width).mesh().build())
            .expect("cuboids share their attributes");
        Some(mesh)
    }
}

/// An open-ended tube about Z, `length` long and centred on the origin: outer wall, inner wall
/// and the two end rings, a closed solid. A scope's barrel, an eyecup, a baffle.
pub fn tube_mesh(inner: f32, outer: f32, length: f32) -> Mesh {
    const SEGMENTS: usize = 64;
    let h = length / 2.0;

    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Four hard-edged bands: (radii, z, normal), wound to face along their normals.
    let bands: [([f32; 2], [f32; 2], fn(f32, f32) -> [f32; 3]); 4] = [
        ([outer, outer], [-h, h], |c, s| [c, s, 0.0]),
        ([inner, inner], [h, -h], |c, s| [-c, -s, 0.0]),
        ([outer, inner], [h, h], |_, _| [0.0, 0.0, 1.0]),
        ([inner, outer], [-h, -h], |_, _| [0.0, 0.0, -1.0]),
    ];
    let row = SEGMENTS as u32 + 1;
    for (radii, zs, normal) in bands {
        let base = positions.len() as u32;
        for k in 0..2 {
            for seg in 0..=SEGMENTS {
                let (sin_t, cos_t) = (2.0 * PI * seg as f32 / SEGMENTS as f32).sin_cos();
                positions.push([radii[k] * cos_t, radii[k] * sin_t, zs[k]]);
                normals.push(normal(cos_t, sin_t));
                uvs.push([seg as f32 / SEGMENTS as f32, k as f32]);
            }
        }
        for seg in 0..SEGMENTS as u32 {
            let (a, b) = (base + seg, base + seg + 1);
            let (c, d) = (base + row + seg, base + row + seg + 1);
            indices.extend([a, b, d, a, d, c]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}

/// Depth of a spherical face of signed radius `r` at `x` from the axis (negative when concave).
fn sag(x: f32, r: f32) -> f32 {
    r.signum() * (r.abs() - (r * r - x * x).sqrt())
}

fn sag64(x: f64, r: f64) -> f64 {
    r.signum() * (r.abs() - (r * r - x * x).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The angular magnification an eye `gap` behind the eyepiece sees at `deg` off the axis,
    /// tracing the exact spherical faces (not the paraxial model the design solved).
    fn traced(t: &Telescope, gap: f32, deg: f32) -> f32 {
        let faces = |lens: &Lens, z: f32| {
            let front = (z + lens.thickness / 2.0, lens.r_front, true, lens.aperture);
            let back = (z - lens.thickness / 2.0, lens.r_back, false, lens.aperture);
            [front, back]
        };
        let mut o = Vec3::new(0.0, 0.0, t.eyepiece_z + t.eyepiece.thickness / 2.0 + gap);
        let a = deg.to_radians();
        let mut d = Vec3::new(0.0, a.sin(), -a.cos());
        let mut inside = false;
        for (vz, r, front, ap) in faces(&t.eyepiece, t.eyepiece_z)
            .into_iter()
            .chain(faces(&t.objective, t.objective_z))
        {
            // Front faces bulge to +Z when convex: their centre lies r behind the vertex.
            let c = Vec3::new(0.0, 0.0, if front { vz - r } else { vz + r });
            let oc = o - c;
            let b = oc.dot(d);
            let disc = b * b - (oc.length_squared() - r * r);
            let near = [-b - disc.sqrt(), -b + disc.sqrt()]
                .into_iter()
                .filter(|t| *t > 1e-9)
                .min_by(|p, q| {
                    ((o + d * *p).z - vz)
                        .abs()
                        .total_cmp(&((o + d * *q).z - vz).abs())
                })
                .unwrap();
            let p = o + d * near;
            assert!(p.truncate().length() <= ap, "ray left the aperture");
            let mut nrm = (p - c) / r.abs();
            if nrm.dot(d) > 0.0 {
                nrm = -nrm;
            }
            let eta = if inside { t.ior } else { 1.0 / t.ior };
            let ci = -nrm.dot(d);
            let k = 1.0 - eta * eta * (1.0 - ci * ci);
            d = (d * eta + nrm * (eta * ci - k.sqrt())).normalize();
            o = p;
            inside = !inside;
        }
        deg / d.y.atan2(-d.z).to_degrees()
    }

    #[test]
    fn galilean_magnifies_as_asked() {
        for (m, l, a) in [(4.0, 0.144, 0.03), (3.0, 0.25, 0.02), (6.0, 0.3, 0.025)] {
            let t = Telescope::galilean(m, l, a);
            for gap in [0.01, 0.03] {
                let traced = traced(&t, gap, 0.5);
                assert!((traced / m - 1.0).abs() < 0.01, "{m}x: traced {traced}");
            }
        }
    }

    #[test]
    fn keplerian_magnifies_inverted_as_asked() {
        for (m, l, a) in [(4.0, 0.25, 0.02), (8.0, 0.4, 0.025), (3.0, 0.2, 0.02)] {
            let t = Telescope::keplerian(m, l, a);
            let pupil = t.eye_z - t.eyepiece_z - t.eyepiece.thickness / 2.0;
            assert!(
                pupil > 0.0,
                "{m}x: exit pupil {pupil} not behind the eyepiece"
            );
            for gap in [pupil, pupil + 0.01] {
                let traced = traced(&t, gap, 0.5);
                assert!((traced / -m - 1.0).abs() < 0.01, "{m}x: traced {traced}");
            }
        }
    }

    #[test]
    fn crosshair_only_where_an_image_forms() {
        let k = Telescope::keplerian(4.0, 0.25, 0.02);
        let z = k.focal_plane_z.unwrap();
        assert!(
            k.objective_z < z && z < k.eyepiece_z,
            "focal plane {z} not between the lenses"
        );
        assert!(k.crosshair_mesh(0.01, 0.0002).is_some());
        assert!(
            Telescope::galilean(4.0, 0.144, 0.03)
                .crosshair_mesh(0.01, 0.0002)
                .is_none()
        );
    }

    #[test]
    #[should_panic(expected = "no Galilean telescope")]
    fn impossible_scope_says_so() {
        Telescope::galilean(4.0, 0.03, 0.03);
    }
}
