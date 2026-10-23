//! Ray portals -- surfaces that teleport rays (the solari-era feature, ported).
//!
//! Add [`AuroraPortal`] to a ray-traced surface (a quad, an arch, anything) and every ray
//! that hits it continues from the paired portal instead: into portal-local space, a
//! half-turn about local Y, out through the target's frame -- so looking INTO this surface
//! shows the view OUT of the target's front (+Z). Pair two portals by pointing their
//! `target`s at each other; a one-way portal is just an unpaired one.
//!
//! GPU side: a small host table of (instance slot, target slot) pairs, address + count in
//! the frame uniform; the raygen checks each hit's TLAS slot against it and rewrites the
//! ray (`portalRedirect` in raygen.slang), reading both transforms from the live
//! `cur_instances` rows -- portals on moving parents stay exact with no CPU reads. Because
//! the redirect happens in the continued ray, recursion is free: portals seen through
//! portals, portals in reflections, portals through glass, bounded by the bounce budgets.
//! Light is NOT transported -- portals carry the view, not next-event estimation, so each
//! side is lit by its own surroundings (and a portal surface still occludes shadow rays).
//!
//! An [`AuroraVolumePortal`] is a closed shape (a disc under a low dome) showing another
//! environment through it at a scale: a ray entering the small shape carries on inside the target
//! environment from the matching point of the same shape placed there N times bigger (a straight
//! affine map, no half-turn), and a ray leaving the big shape comes back out of the small one. What
//! the ray meets in there is lit by the target's own sky and lights. Where the big shape's wall lies
//! under the target's ground the small one's surface shows the cut (earth and stone strata).
//!
//! [`PortalTraveler`] carries an entity (a camera, a player) through a portal it crosses: the
//! same map the rays use moves it, and it joins the exit's world (a physics body by avian's
//! `TransferToEnvironment`, anything else by re-parenting under the exit's world), turns its
//! velocity with it, and triggers [`PortalCrossed`] on it. [`Portals`] finds crossings for
//! anything else that must follow a segment through (a camera boom).

use ash::vk;
use avian3d::{
    environment::TransferToEnvironment,
    prelude::{AngularVelocity, LinearVelocity, RigidBody},
};
use bevy::{
    ecs::system::SystemParam, math::Affine3A, platform::collections::HashMap, prelude::*,
    transform::TransformSystems,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    environment::{InEnvironment, RenderEnvironments},
    mesh::{AuroraMesh, AuroraMesh3d},
    ray_render_plugin::{RenderSet, TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::GpuInstance,
};

/// Marks a ray-traced surface as a portal showing the view out of `target`'s front face
/// (+Z). The surface never shades -- rays redirect on hit.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub struct AuroraPortal {
    /// The portal entity this surface looks out of.
    pub target: Entity,
}

/// The closed shape of an [`AuroraVolumePortal`], in its entity's local space: a cylinder of
/// `radius` standing on y = 0 to `wall`, capped by a spherical dome rising `dome` above that (0: a
/// flat top). Convex as long as `dome <= radius`.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
#[reflect(Default, Clone, PartialEq)]
pub struct VolumeShape {
    pub radius: f32,
    pub wall: f32,
    pub dome: f32,
}

impl Default for VolumeShape {
    fn default() -> Self {
        Self {
            radius: 1.0,
            wall: 0.15,
            dome: 0.35,
        }
    }
}

impl VolumeShape {
    /// The shape as a closed mesh (outward normals), the surface rays enter by.
    pub fn mesh(&self, segments: u32) -> Mesh {
        use bevy::{asset::RenderAssetUsages, mesh::Indices, mesh::PrimitiveTopology};
        let n = segments.max(8);
        let row = n + 1;
        let r = self.radius;
        let dome = self.dome.clamp(0.0, r);
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut ring = |radius: f32, y: f32, normal: &dyn Fn(f32) -> Vec3, v: f32| {
            for i in 0..=n {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                positions.push([radius * a.cos(), y, radius * a.sin()]);
                normals.push(normal(a).to_array());
                uvs.push([i as f32 / n as f32, v]);
            }
        };
        // The bottom (rim, then centre), the wall (foot, then top), the cap (rim to apex).
        ring(r, 0.0, &|_| Vec3::NEG_Y, 1.0);
        ring(0.0, 0.0, &|_| Vec3::NEG_Y, 0.0);
        if self.wall > 0.0 {
            let side = |a: f32| Vec3::new(a.cos(), 0.0, a.sin());
            ring(r, 0.0, &side, 0.0);
            ring(r, self.wall, &side, 1.0);
        }
        let rings = 24u32;
        let sphere = (dome > 1e-5).then(|| (r * r + dome * dome) / (2.0 * dome));
        for k in 0..=rings {
            let f = 1.0 - k as f32 / rings as f32;
            match sphere {
                Some(big) => {
                    let theta = (r / big).clamp(-1.0, 1.0).asin() * f;
                    let normal = move |a: f32| {
                        Vec3::new(theta.sin() * a.cos(), theta.cos(), theta.sin() * a.sin())
                    };
                    ring(
                        big * theta.sin(),
                        self.wall + dome - big + big * theta.cos(),
                        &normal,
                        f,
                    );
                }
                None => ring(r * f, self.wall, &|_| Vec3::Y, f),
            }
        }
        // The bottom faces down: its band runs centre to rim.
        for i in 0..n {
            let (b, t) = (row + i, i);
            indices.extend_from_slice(&[b, t, b + 1, t, t + 1, b + 1]);
        }
        // Each band from ring `o` to the next winds outward (lower / outer ring first).
        let mut band = |o: u32| {
            for i in 0..n {
                let (b, t) = (o * row + i, (o + 1) * row + i);
                indices.extend_from_slice(&[b, t, b + 1, t, t + 1, b + 1]);
            }
        };
        let mut next = 2;
        if self.wall > 0.0 {
            band(next);
            next += 2;
        }
        for k in 0..rings {
            band(next + k);
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

/// A live diorama: this entity's [`VolumeShape`] shows environment `target_env` through it,
/// where the same shape stands at `target_transform` (relative to `target_env`; its scale is the
/// diorama's, e.g. 20 for 1:20). Rays entering the shape carry on in there and come back out
/// when they leave the big one. The shape's mesh is made for it (the
/// entity's material shows only on the cut, repainted as strata).
#[derive(Component, Reflect, Clone, Copy, Debug)]
#[reflect(Component)]
pub struct AuroraVolumePortal {
    pub target_env: Entity,
    pub target_transform: Transform,
    pub shape: VolumeShape,
}

impl AuroraVolumePortal {
    /// The big shape's world pose, from its environment's.
    pub fn target_pose(&self, env: Option<&GlobalTransform>) -> Affine3A {
        env.map_or(Affine3A::IDENTITY, GlobalTransform::affine)
            * self.target_transform.compute_affine()
    }
}

/// The volume map: world space of the small shape (`source`, its entity's pose) into the
/// target environment's (`target`, the big shape's pose). Its scale is the diorama's.
pub fn volume_map(source: Affine3A, target: Affine3A) -> Affine3A {
    target * source.inverse()
}

/// One table entry; must match `PORTAL_*` in raygen.slang and pick.slang (8 x uvec4).
/// `valid = 0` covers unresolved endpoints (instances still streaming in) and the
/// zero-filled tail of the buffer -- slot 0 is real, so zeros must read as inert.
/// A volume portal (`kind = 1`) carries its target environment's bit in `target`, the big
/// shape's world pose, its inverse and the [`VolumeShape`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct GpuPortal {
    instance_slot: u32,
    target: u32,
    valid: u32,
    kind: u32,
    target_rows: [[f32; 4]; 3],
    inverse_rows: [[f32; 4]; 3],
    shape: [f32; 4],
}

impl GpuPortal {
    const INVALID: Self = Self {
        instance_slot: 0,
        target: 0,
        valid: 0,
        kind: 0,
        target_rows: [[0.0; 4]; 3],
        inverse_rows: [[0.0; 4]; 3],
        shape: [0.0; 4],
    };
}

/// The rows of an affine as the instance transform packs them (row k of world_from_local).
fn affine_rows(a: Affine3A) -> [[f32; 4]; 3] {
    let m = a.matrix3;
    let t = a.translation;
    [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x, t.x],
        [m.x_axis.y, m.y_axis.y, m.z_axis.y, t.y],
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, t.z],
    ]
}

/// The uploaded pair table; the frame uniform carries its address + count.
#[derive(Resource, Default)]
pub struct PortalTable {
    buffer: Option<Buffer<GpuPortal>>,
    capacity: u64,
    count: u32,
    /// Last-uploaded entries: the upload is skipped while nothing changed.
    last: Vec<GpuPortal>,
}

impl PortalTable {
    pub fn address(&self) -> u64 {
        if self.count == 0 {
            0
        } else {
            self.buffer.as_ref().map_or(0, |b| b.address)
        }
    }
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// `RenderSet::Prepare`: rebuild + upload the pair table when the portal set, a pairing,
/// or an endpoint's instance slot changed (slots bind late while meshes stream in, so this
/// re-resolves every frame -- the memo makes the idle cost a Vec compare).
fn upload_portals(
    render_device: Res<RenderDevice>,
    portals: Query<(Entity, &AuroraPortal)>,
    volumes: Query<(Entity, &AuroraVolumePortal)>,
    frames: Query<&GlobalTransform>,
    slots: Query<&GpuInstance>,
    environments: Res<RenderEnvironments>,
    mut table: ResMut<PortalTable>,
) {
    let mut entries: Vec<GpuPortal> = Vec::new();
    for (entity, portal) in &portals {
        entries.push(match (slots.get(entity), slots.get(portal.target)) {
            (Ok(a), Ok(b)) => GpuPortal {
                instance_slot: a.0,
                target: b.0,
                valid: 1,
                ..GpuPortal::INVALID
            },
            _ => GpuPortal::INVALID,
        });
    }
    for (entity, volume) in &volumes {
        entries.push(
            match (slots.get(entity), environments.bit(volume.target_env)) {
                (Ok(slot), Some(bit)) => {
                    let pose = volume.target_pose(frames.get(volume.target_env).ok());
                    let shape = volume.shape;
                    GpuPortal {
                        instance_slot: slot.0,
                        target: bit as u32,
                        valid: 1,
                        kind: 1,
                        target_rows: affine_rows(pose),
                        inverse_rows: affine_rows(pose.inverse()),
                        shape: [
                            shape.radius,
                            shape.wall,
                            shape.dome.clamp(0.0, shape.radius),
                            0.0,
                        ],
                    }
                }
                _ => GpuPortal::INVALID,
            },
        );
    }
    if entries == table.last && (table.buffer.is_some() || entries.is_empty()) {
        return;
    }

    let needed = entries.len().max(1) as u64;
    // A fresh buffer on every change: a frame still in flight may be reading the old one,
    // which the destroyer frees once it is done.
    {
        if let Some(old) = table.buffer.take() {
            render_device.destroyer.destroy_buffer(old.handle);
        }
        let capacity = needed.next_power_of_two().max(16);
        table.buffer =
            Some(render_device.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER));
        table.capacity = capacity;
    }
    let capacity = table.capacity;
    let buffer = table.buffer.as_mut().unwrap();
    let mut mapped = render_device.map_buffer(buffer);
    for i in 0..capacity as usize {
        mapped[i] = *entries.get(i).unwrap_or(&GpuPortal::INVALID);
    }
    table.count = entries.len() as u32;
    table.last = entries;
}

/// Moves its entity through any [`AuroraPortal`] it crosses (front to back, within the
/// portal mesh's bounds) and into the exit's world.
#[derive(Component, Reflect, Default, Clone, Copy)]
#[reflect(Component, Default)]
pub struct PortalTraveler {
    /// Last frame's world position: crossings are found on the segment from here to now.
    #[reflect(ignore)]
    last: Option<Vec3>,
}

/// The ray map: into `portal`'s frame, a half-turn about its local Y, out of `target`'s
/// (both world-space).
pub fn portal_map(portal: Affine3A, target: Affine3A) -> Affine3A {
    target * Affine3A::from_rotation_y(std::f32::consts::PI) * portal.inverse()
}

/// A crossing found by [`Portals::crossing`].
#[derive(Clone, Copy, Debug)]
pub struct Crossing {
    pub portal: Entity,
    pub target: Entity,
    /// The exit's world: the environment owning its render bit.
    pub world: Entity,
    /// World space through the portal (see [`portal_map`]).
    pub map: Affine3A,
}

/// Every portal, for finding where a segment passes through one.
#[derive(SystemParam)]
pub struct Portals<'w, 's> {
    portals: Query<
        'w,
        's,
        (
            Entity,
            &'static AuroraPortal,
            &'static GlobalTransform,
            Option<&'static AuroraMesh3d>,
        ),
    >,
    frames: Query<'w, 's, &'static GlobalTransform>,
    meshes: Res<'w, Assets<AuroraMesh>>,
    in_world: Query<'w, 's, &'static InEnvironment>,
    worlds: Res<'w, RenderEnvironments>,
}

impl Portals<'_, '_> {
    /// The environment `entity` is in.
    pub fn environment_of(&self, entity: Entity) -> Option<Entity> {
        self.worlds.environment(self.in_world.get(entity).ok()?.0)
    }

    /// The first portal in environment `world` the world-space segment `from` -> `to` enters
    /// front to back, within its mesh's bounds, and whose exit is in an environment. Portals in
    /// other environments are not there for it, wherever they stand.
    pub fn crossing(&self, world: Entity, from: Vec3, to: Vec3) -> Option<Crossing> {
        let bit = self.worlds.bit(world)?;
        self.portals
            .iter()
            .find_map(|(portal, link, global, mesh)| {
                if self.in_world.get(portal).ok().map(|w| w.0) != Some(bit) {
                    return None;
                }
                let pose = global.affine();
                let to_local = pose.inverse();
                let (a, b) = (
                    to_local.transform_point3(from),
                    to_local.transform_point3(to),
                );
                if !(a.z > 0.0 && b.z <= 0.0) {
                    return None;
                }
                let hit = a.lerp(b, a.z / (a.z - b.z));
                let half = mesh
                    .and_then(|m| self.meshes.get(&m.0))
                    .map_or(Vec2::splat(0.5), |m| {
                        Vec2::new(m.aabb.half_extent[0], m.aabb.half_extent[1])
                    });
                if hit.x.abs() > half.x || hit.y.abs() > half.y {
                    return None;
                }
                let target = self.frames.get(link.target).ok()?;
                // An exit in no environment leads nowhere.
                let Some(world) = self
                    .in_world
                    .get(link.target)
                    .ok()
                    .and_then(|w| self.worlds.environment(w.0))
                else {
                    warn_once!("aurora: portal exit {} is in no environment", link.target);
                    return None;
                };
                Some(Crossing {
                    portal,
                    target: link.target,
                    world,
                    map: portal_map(pose, target.affine()),
                })
            })
    }
}

/// Triggered on a [`PortalTraveler`] once it has been carried through a portal.
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct PortalCrossed {
    pub entity: Entity,
    pub portal: Entity,
    /// World space through the portal: what turned it, and what to turn with it.
    pub map: Affine3A,
}

#[allow(clippy::type_complexity)]
fn travel(
    mut commands: Commands,
    mut travelers: Query<(
        Entity,
        &mut PortalTraveler,
        &mut Transform,
        Option<&ChildOf>,
        Has<RigidBody>,
        Option<&mut LinearVelocity>,
        Option<&mut AngularVelocity>,
    )>,
    portals: Portals,
    frames: Query<&GlobalTransform, Without<PortalTraveler>>,
) {
    for (entity, mut traveler, mut transform, child_of, body, linear, angular) in &mut travelers {
        // This frame's pose, ahead of propagation: the teleport lands in the same frame.
        let parent_affine = child_of
            .and_then(|c| frames.get(c.parent()).ok())
            .map_or(Affine3A::IDENTITY, GlobalTransform::affine);
        let global = parent_affine * transform.compute_affine();
        let now: Vec3 = global.translation.into();
        let Some(last) = traveler.last.replace(now) else {
            continue;
        };
        let Some(world) = portals.environment_of(entity) else {
            continue;
        };
        let Some(crossing) = portals.crossing(world, last, now) else {
            continue;
        };
        let moved = crossing.map * global;
        let local = frames
            .get(crossing.world)
            .map_or(moved, |p| p.affine().inverse() * moved);
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        *transform = Transform {
            translation,
            rotation,
            scale,
        };
        traveler.last = Some(moved.translation.into());
        // Momentum turns with it.
        if let Some(mut v) = linear {
            v.0 = crossing.map.transform_vector3(v.0);
        }
        if let Some(mut w) = angular {
            w.0 = crossing.map.transform_vector3(w.0);
        }
        if body {
            commands.trigger(TransferToEnvironment {
                entity,
                world: crossing.world,
            });
        } else {
            commands.entity(entity).insert(ChildOf(crossing.world));
        }
        commands.trigger(PortalCrossed {
            entity,
            portal: crossing.portal,
            map: crossing.map,
        });
    }
}

/// The shape's mesh, made whenever the shape changes.
fn build_volume_meshes(
    mut commands: Commands,
    volumes: Query<(Entity, &AuroraVolumePortal), Changed<AuroraVolumePortal>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut built: Local<HashMap<Entity, VolumeShape>>,
) {
    for (entity, volume) in &volumes {
        if built.get(&entity) == Some(&volume.shape) {
            continue;
        }
        built.insert(entity, volume.shape);
        commands.entity(entity).insert(AuroraMesh3d(
            meshes.add(AuroraMesh::from_shape(volume.shape.mesh(96))),
        ));
    }
}

fn cleanup_portals(mut table: ResMut<PortalTable>, rd: Res<RenderDevice>) {
    if let Some(buffer) = table.buffer.take() {
        rd.destroyer.destroy_buffer(buffer.handle);
    }
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraPortal>()
            .register_type::<AuroraVolumePortal>()
            .register_type::<PortalTraveler>()
            .add_systems(PostUpdate, build_volume_meshes)
            .add_systems(PostUpdate, travel.before(TransformSystems::Propagate));
        app.init_resource::<PortalTable>();
        // The Prepare set already gates on the render device existing.
        app.add_systems(Last, upload_portals.in_set(RenderSet::Prepare))
            .add_systems(TeardownSchedule, cleanup_portals.before(on_shutdown));
    }
}

#[cfg(test)]
mod tests {
    use crate::environment::Environment;

    use std::f32::consts::PI;

    use super::*;
    use crate::environment::RenderEnvironmentPlugin;

    #[test]
    fn crossing_a_portal_moves_the_traveler_into_the_exits_world() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            // No GPU here: CPU propagation stands in for the read-back GlobalTransform.
            TransformPlugin,
            RenderEnvironmentPlugin,
        ))
        .init_resource::<Assets<AuroraMesh>>()
        .add_systems(PostUpdate, travel.before(TransformSystems::Propagate));
        let home = app.world_mut().spawn(Environment(Some(0))).id();
        let desert = app.world_mut().spawn(Environment::default()).id();
        // C faces +Z at home; D stands at the same spot in the desert, turned around, so
        // walking into C carries straight on.
        let d = app
            .world_mut()
            .spawn((
                ChildOf(desert),
                Transform::from_rotation(Quat::from_rotation_y(PI)),
            ))
            .id();
        let c = app
            .world_mut()
            .spawn((
                ChildOf(home),
                Transform::IDENTITY,
                AuroraPortal { target: d },
            ))
            .id();
        app.world_mut()
            .entity_mut(d)
            .insert(AuroraPortal { target: c });
        let walker = app
            .world_mut()
            .spawn((
                ChildOf(home),
                PortalTraveler::default(),
                Transform::from_xyz(0.1, 0.0, 1.0),
            ))
            .id();
        app.update();
        app.world_mut()
            .get_mut::<Transform>(walker)
            .unwrap()
            .translation
            .z = -1.0;
        app.update();
        app.update();

        let world = app.world();
        assert_eq!(
            world.get::<ChildOf>(walker).map(ChildOf::parent),
            Some(desert)
        );
        assert_eq!(world.get::<InEnvironment>(walker), Some(&InEnvironment(1)));
        let at = world.get::<Transform>(walker).unwrap().translation;
        assert!(at.abs_diff_eq(Vec3::new(0.1, 0.0, -1.0), 1e-4), "{at}");
    }

    #[test]
    fn the_examples_gate_c_carries_a_walker_into_the_desert() {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            // No GPU here: CPU propagation stands in for the read-back GlobalTransform.
            TransformPlugin,
            RenderEnvironmentPlugin,
        ))
        .init_resource::<Assets<AuroraMesh>>()
        .add_systems(PostUpdate, travel.before(TransformSystems::Propagate));
        let quad = app
            .world_mut()
            .resource_mut::<Assets<AuroraMesh>>()
            .add(AuroraMesh::from_shape(Rectangle::new(2.0, 3.0)));
        let home = app.world_mut().spawn(Environment(Some(0))).id();
        let desert = app.world_mut().spawn(Environment::default()).id();
        let gate = |app: &mut App, world: Entity, at: Transform| {
            let root = app
                .world_mut()
                .spawn((at, Visibility::default(), ChildOf(world)))
                .id();
            app.world_mut()
                .spawn((
                    ChildOf(root),
                    AuroraMesh3d(quad.clone()),
                    Transform::from_xyz(0.0, 1.6, 0.0),
                ))
                .id()
        };
        let c = gate(&mut app, home, Transform::from_xyz(0.0, 0.0, -8.0));
        let d = gate(
            &mut app,
            desert,
            Transform::from_xyz(0.0, 0.0, -8.0).with_rotation(Quat::from_rotation_y(PI)),
        );
        app.world_mut()
            .entity_mut(c)
            .insert(AuroraPortal { target: d });
        app.world_mut()
            .entity_mut(d)
            .insert(AuroraPortal { target: c });
        let walker = app
            .world_mut()
            .spawn((
                ChildOf(home),
                PortalTraveler::default(),
                Transform::from_xyz(0.3, 1.7, -6.0),
            ))
            .id();
        for _ in 0..60 {
            app.world_mut()
                .get_mut::<Transform>(walker)
                .unwrap()
                .translation
                .z -= 0.05;
            app.update();
        }
        assert_eq!(
            app.world().get::<ChildOf>(walker).map(ChildOf::parent),
            Some(desert)
        );
        // Under the world its pose still reaches GlobalTransform (what the camera renders from).
        app.world_mut()
            .get_mut::<Transform>(walker)
            .unwrap()
            .translation
            .z = -12.0;
        app.update();
        let global = app
            .world()
            .get::<GlobalTransform>(walker)
            .unwrap()
            .translation();
        assert!(
            global.abs_diff_eq(Vec3::new(0.3, 1.7, -12.0), 1e-4),
            "{global}"
        );
    }
}
