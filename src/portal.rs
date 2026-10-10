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
//! portals, portals in reflections, portals through glass, bounded by `max_bounces`.
//! Light is NOT transported -- portals carry the view, not next-event estimation, so each
//! side is lit by its own surroundings (and a portal surface still occludes shadow rays).
//!
//! [`PortalTraveler`] carries an entity (a camera, a player) through a portal it crosses: the
//! same map the rays use moves it, and it joins the exit's world (a physics body by avian's
//! `TransferToEnvironment`, anything else by re-parenting under the exit's world).

use ash::vk;
use avian3d::{
    environment::{MainPhysicsEnvironmentEntity, TransferToEnvironment},
    prelude::RigidBody,
};
use bevy::{math::Affine3A, prelude::*, transform::TransformSystems};
use bytemuck::{Pod, Zeroable};

use crate::{
    environment::{InEnvironment, RenderEnvironments},
    mesh::{AuroraMesh, AuroraMesh3d},
    ray_render_plugin::RenderSet,
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

/// One table entry; must match the `uvec4` unpack in raygen.slang's `portalRedirect`.
/// `valid = 0` covers unresolved endpoints (instances still streaming in) and the
/// zero-filled tail of the buffer -- slot 0 is real, so zeros must read as inert.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct GpuPortal {
    instance_slot: u32,
    target_slot: u32,
    valid: u32,
    pad: u32,
}

impl GpuPortal {
    const INVALID: Self = Self {
        instance_slot: 0,
        target_slot: 0,
        valid: 0,
        pad: 0,
    };
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
    slots: Query<&GpuInstance>,
    mut table: ResMut<PortalTable>,
) {
    let mut entries: Vec<GpuPortal> = Vec::new();
    for (entity, portal) in &portals {
        entries.push(match (slots.get(entity), slots.get(portal.target)) {
            (Ok(a), Ok(b)) => GpuPortal {
                instance_slot: a.0,
                target_slot: b.0,
                valid: 1,
                pad: 0,
            },
            _ => GpuPortal::INVALID,
        });
    }
    if entries == table.last && (table.buffer.is_some() || entries.is_empty()) {
        return;
    }

    let needed = entries.len().max(1) as u64;
    if table.buffer.is_none() || table.capacity < needed {
        if let Some(old) = table.buffer.take() {
            // The destroyer defers the free past the frames in flight.
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

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn travel(
    mut commands: Commands,
    mut travelers: Query<(
        Entity,
        &mut PortalTraveler,
        &mut Transform,
        Option<&ChildOf>,
        Has<RigidBody>,
    )>,
    portals: Query<(&AuroraPortal, &GlobalTransform, Option<&AuroraMesh3d>)>,
    frames: Query<&GlobalTransform, Without<PortalTraveler>>,
    meshes: Res<Assets<AuroraMesh>>,
    in_world: Query<&InEnvironment>,
    worlds: Res<RenderEnvironments>,
    main_world: Option<Res<MainPhysicsEnvironmentEntity>>,
) {
    for (entity, mut traveler, mut transform, child_of, body) in &mut travelers {
        // This frame's pose, ahead of propagation: the teleport lands in the same frame.
        let parent_affine = child_of
            .and_then(|c| frames.get(c.parent()).ok())
            .map_or(Affine3A::IDENTITY, GlobalTransform::affine);
        let global = parent_affine * transform.compute_affine();
        let now: Vec3 = global.translation.into();
        let Some(last) = traveler.last.replace(now) else {
            continue;
        };
        for (portal, portal_global, mesh) in &portals {
            let portal_pose = portal_global.affine();
            let to_local = portal_pose.inverse();
            let (a, b) = (
                to_local.transform_point3(last),
                to_local.transform_point3(now),
            );
            if !(a.z > 0.0 && b.z <= 0.0) {
                continue;
            }
            let hit = a.lerp(b, a.z / (a.z - b.z));
            let half = mesh
                .and_then(|m| meshes.get(&m.0))
                .map_or(Vec2::splat(0.5), |m| {
                    Vec2::new(m.aabb.half_extent[0], m.aabb.half_extent[1])
                });
            if hit.x.abs() > half.x || hit.y.abs() > half.y {
                continue;
            }
            let Ok(target) = frames.get(portal.target) else {
                continue;
            };
            let moved = portal_map(portal_pose, target.affine()) * global;
            // The exit's world: its render bit's owner, else the main world.
            let world = in_world
                .get(portal.target)
                .ok()
                .and_then(|w| worlds.environment(w.0))
                .or(main_world.as_ref().map(|m| m.0));
            let parent = world.and_then(|w| frames.get(w).ok());
            let local = parent.map_or(moved, |p| p.affine().inverse() * moved);
            let (scale, rotation, translation) = local.to_scale_rotation_translation();
            *transform = Transform {
                translation,
                rotation,
                scale,
            };
            traveler.last = Some(moved.translation.into());
            match world {
                Some(world) if body => commands.trigger(TransferToEnvironment { entity, world }),
                Some(world) => {
                    commands.entity(entity).insert(ChildOf(world));
                }
                None => {
                    commands.entity(entity).remove::<ChildOf>();
                }
            }
            break;
        }
    }
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraPortal>()
            .register_type::<PortalTraveler>()
            .add_systems(PostUpdate, travel.before(TransformSystems::Propagate));
        app.init_resource::<PortalTable>();
        // The Prepare set already gates on the render device existing.
        app.add_systems(Last, upload_portals.in_set(RenderSet::Prepare));
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
        let desert = app.world_mut().spawn(Environment::default()).id();
        // C faces +Z in the main world; D stands at the same spot in the desert, turned
        // around, so walking into C carries straight on.
        let d = app
            .world_mut()
            .spawn((
                ChildOf(desert),
                Transform::from_rotation(Quat::from_rotation_y(PI)),
            ))
            .id();
        let c = app
            .world_mut()
            .spawn((Transform::IDENTITY, AuroraPortal { target: d }))
            .id();
        app.world_mut()
            .entity_mut(d)
            .insert(AuroraPortal { target: c });
        let walker = app
            .world_mut()
            .spawn((
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
        let desert = app.world_mut().spawn(Environment::default()).id();
        let gate = |app: &mut App, world: Option<Entity>, at: Transform| {
            let mut root = app.world_mut().spawn((at, Visibility::default()));
            if let Some(world) = world {
                root.insert(ChildOf(world));
            }
            let root = root.id();
            app.world_mut()
                .spawn((
                    ChildOf(root),
                    AuroraMesh3d(quad.clone()),
                    Transform::from_xyz(0.0, 1.6, 0.0),
                ))
                .id()
        };
        let c = gate(&mut app, None, Transform::from_xyz(0.0, 0.0, -8.0));
        let d = gate(
            &mut app,
            Some(desert),
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
