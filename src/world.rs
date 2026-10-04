//! Render worlds: every avian [`PhysicsWorld`] is also a render world.
//!
//! An entity belongs to the world of its nearest [`PhysicsWorld`] ancestor (avian's rule), and
//! renders with that world's bit in the TLAS instance mask, so rays only see their own world
//! (portals swap a ray's mask). The mask is 8 bits, so at most 8 worlds are live at once:
//! [`MainPhysicsWorld`] and everything outside any world is bit 0, other worlds take the lowest
//! free bit. An explicit `RenderLayers` on an entity overrides its world's bit.

pub use avian3d::world::{MainPhysicsWorld, MainPhysicsWorldEntity, PhysicsWorld};
use bevy::{camera::visibility::RenderLayers, platform::collections::HashSet, prelude::*};

/// Which world owns each of the 8 instance-mask bits.
#[derive(Resource, Default, Debug)]
pub struct RenderWorlds {
    owners: [Option<Entity>; 8],
}

impl RenderWorlds {
    /// The mask bit of `world`, if it is a live render world.
    pub fn bit(&self, world: Entity) -> Option<u8> {
        self.owners
            .iter()
            .position(|owner| *owner == Some(world))
            .map(|bit| bit as u8)
    }

    /// The world that owns `bit`.
    pub fn world(&self, bit: u8) -> Option<Entity> {
        self.owners.get(bit as usize).copied().flatten()
    }
}

/// The world bit an entity renders in, from its nearest [`PhysicsWorld`] ancestor. Absent means
/// the main world (bit 0); set by [`propagate_worlds`], not by hand. Not reflected: derived
/// state, never saved into a scene.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub struct InWorld(pub u8);

/// The 8-bit instance / ray mask of an entity: its explicit `RenderLayers` if it has them,
/// else its world's bit.
pub fn world_mask(layers: Option<&RenderLayers>, world: Option<&InWorld>) -> u8 {
    match layers {
        Some(_) => crate::tlas_builder::layers_mask(layers),
        None => 1 << world.map_or(0, |w| w.0.min(7)),
    }
}

fn assign_bit(
    add: On<Add<PhysicsWorld>>,
    main: Query<(), With<MainPhysicsWorld>>,
    mut worlds: ResMut<RenderWorlds>,
) {
    let world = add.entity;
    let bit = if main.contains(world) {
        Some(0)
    } else {
        (1..8).find(|&bit| worlds.owners[bit].is_none())
    };
    match bit {
        Some(bit) => worlds.owners[bit] = Some(world),
        None => warn!("aurora: more than 8 physics worlds; {world} renders in the main world"),
    }
}

fn free_bit(remove: On<Remove<PhysicsWorld>>, mut worlds: ResMut<RenderWorlds>) {
    for owner in &mut worlds.owners {
        if *owner == Some(remove.entity) {
            *owner = None;
        }
    }
}

/// Re-resolves the world of every entity whose place in the hierarchy changed, and of its
/// subtree. Change-driven: a still hierarchy costs nothing.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn propagate_worlds(
    mut commands: Commands,
    worlds: Res<RenderWorlds>,
    moved: Query<Entity, Or<(Changed<ChildOf>, Added<PhysicsWorld>)>>,
    mut unparented: RemovedComponents<ChildOf>,
    mut unworlded: RemovedComponents<PhysicsWorld>,
    is_world: Query<(), With<PhysicsWorld>>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
    current: Query<Option<&InWorld>>,
) {
    let mut roots: Vec<Entity> = moved.iter().collect();
    roots.extend(unparented.read());
    roots.extend(unworlded.read());
    if roots.is_empty() {
        return;
    }
    let bit_of = |world: Entity| worlds.bit(world).unwrap_or(0);
    let resolve = |mut entity: Entity| loop {
        if is_world.contains(entity) {
            return bit_of(entity);
        }
        match parents.get(entity) {
            Ok(parent) => entity = parent.parent(),
            Err(_) => return 0,
        }
    };

    let mut visited = HashSet::new();
    let mut stack = Vec::new();
    for root in roots {
        let Ok(_) = current.get(root) else {
            continue;
        };
        if visited.contains(&root) {
            continue;
        }
        stack.push((root, resolve(root)));
        while let Some((entity, inherited)) = stack.pop() {
            if !visited.insert(entity) {
                continue;
            }
            let bit = if is_world.contains(entity) {
                bit_of(entity)
            } else {
                inherited
            };
            // Never removed once set: a move back to the main world writes 0, so the change
            // reaches everything watching `Changed<InWorld>`.
            match current.get(entity).ok().flatten() {
                Some(had) if had.0 == bit => {}
                None if bit == 0 => {}
                _ => {
                    commands.entity(entity).try_insert(InWorld(bit));
                }
            }
            if let Ok(kids) = children.get(entity) {
                stack.extend(kids.iter().map(|kid| (kid, bit)));
            }
        }
    }
}

pub struct RenderWorldPlugin;

impl Plugin for RenderWorldPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RenderWorlds>()
            .add_observer(assign_bit)
            .add_observer(free_bit)
            .add_systems(PostUpdate, propagate_worlds);
    }

    /// Without avian's physics plugins nothing spawns the main world, yet it is where the
    /// main world's sky lives: spawn it here (after every plugin has built, so avian's own
    /// is seen first).
    fn finish(&self, app: &mut App) {
        if !app.world().contains_resource::<MainPhysicsWorldEntity>() {
            let main = app.world_mut().spawn(MainPhysicsWorld).id();
            app.insert_resource(MainPhysicsWorldEntity(main));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, RenderWorldPlugin));
        app.world_mut().spawn(MainPhysicsWorld);
        app
    }

    fn bit(app: &App, entity: Entity) -> u8 {
        app.world().get::<InWorld>(entity).map_or(0, |w| w.0)
    }

    #[test]
    fn descendants_take_their_worlds_bit() {
        let mut app = app();
        let outside = app.world_mut().spawn_empty().id();
        let world = app.world_mut().spawn(PhysicsWorld).id();
        let child = app.world_mut().spawn(ChildOf(world)).id();
        let grandchild = app.world_mut().spawn(ChildOf(child)).id();
        app.update();
        assert_eq!(app.world().resource::<RenderWorlds>().bit(world), Some(1));
        assert_eq!(bit(&app, world), 1);
        assert_eq!(bit(&app, child), 1);
        assert_eq!(bit(&app, grandchild), 1);
        assert_eq!(bit(&app, outside), 0);
    }

    #[test]
    fn a_reparented_subtree_follows_its_new_world() {
        let mut app = app();
        let a = app.world_mut().spawn(PhysicsWorld).id();
        let b = app.world_mut().spawn(PhysicsWorld).id();
        let body = app.world_mut().spawn(ChildOf(a)).id();
        let part = app.world_mut().spawn(ChildOf(body)).id();
        app.update();
        assert_eq!((bit(&app, body), bit(&app, part)), (1, 1));

        app.world_mut().entity_mut(body).insert(ChildOf(b));
        app.update();
        assert_eq!((bit(&app, body), bit(&app, part)), (2, 2));

        app.world_mut().entity_mut(body).remove::<ChildOf>();
        app.update();
        assert_eq!((bit(&app, body), bit(&app, part)), (0, 0));
    }

    #[test]
    fn a_despawned_world_frees_its_bit() {
        let mut app = app();
        let a = app.world_mut().spawn(PhysicsWorld).id();
        app.update();
        app.world_mut().despawn(a);
        let b = app.world_mut().spawn(PhysicsWorld).id();
        app.update();
        assert_eq!(app.world().resource::<RenderWorlds>().bit(b), Some(1));
    }

    #[test]
    fn a_ninth_world_renders_in_the_main_world() {
        let mut app = app();
        let worlds: Vec<Entity> = (0..8)
            .map(|_| app.world_mut().spawn(PhysicsWorld).id())
            .collect();
        let child = app.world_mut().spawn(ChildOf(worlds[7])).id();
        app.update();
        assert_eq!(
            app.world().resource::<RenderWorlds>().bit(worlds[6]),
            Some(7)
        );
        assert_eq!(app.world().resource::<RenderWorlds>().bit(worlds[7]), None);
        assert_eq!(bit(&app, child), 0);
    }

    #[test]
    fn explicit_render_layers_override_the_world() {
        let layers = RenderLayers::layer(3);
        assert_eq!(world_mask(Some(&layers), Some(&InWorld(1))), 0b1000);
        assert_eq!(world_mask(None, Some(&InWorld(2))), 0b100);
        assert_eq!(world_mask(None, None), 0b1);
    }
}
