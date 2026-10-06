//! Environments: every avian [`PhysicsEnvironment`] renders apart from the others.
//!
//! An entity belongs to the environment of its nearest [`PhysicsEnvironment`] ancestor (avian's rule), and
//! renders with that environment's bit in the TLAS instance mask, so rays only see their own environment
//! (portals swap a ray's mask). The mask is 8 bits, so at most 8 environments are live at once:
//! [`MainPhysicsEnvironment`] and everything outside any environment is bit 0, other environments take the lowest
//! free bit. An explicit `RenderLayers` on an entity overrides its environment's bit.

pub use avian3d::environment::{
    MainPhysicsEnvironment, MainPhysicsEnvironmentEntity, PhysicsEnvironment,
};
use bevy::{camera::visibility::RenderLayers, platform::collections::HashSet, prelude::*};

/// An environment that renders: its own sky, suns and exposure, apart from the others. It brings
/// its [`PhysicsEnvironment`] with it. The settings are separate components on the same entity
/// ([`Sky`](crate::sky::Sky), [`GradientSky`](crate::sky::GradientSky),
/// [`Atmosphere`](crate::atmosphere::Atmosphere), [`CloudLayer`](crate::atmosphere::CloudLayer),
/// [`EnvironmentExposure`](crate::sky::EnvironmentExposure)), and a scene whose root carries it
/// is an environment scene. A physics environment without one renders in the environment above.
#[derive(Component, Reflect, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
#[require(PhysicsEnvironment)]
pub struct Environment;

/// Which environment owns each of the 8 instance-mask bits.
#[derive(Resource, Default, Debug)]
pub struct RenderEnvironments {
    owners: [Option<Entity>; 8],
}

impl RenderEnvironments {
    /// The mask bit of `environment`, if it is a live environment.
    pub fn bit(&self, world: Entity) -> Option<u8> {
        self.owners
            .iter()
            .position(|owner| *owner == Some(world))
            .map(|bit| bit as u8)
    }

    /// The environment that owns `bit`.
    pub fn environment(&self, bit: u8) -> Option<Entity> {
        self.owners.get(bit as usize).copied().flatten()
    }
}

/// The environment bit an entity renders in, from its nearest [`PhysicsEnvironment`] ancestor. Absent means
/// the main environment (bit 0); set by [`propagate_environments`], not by hand. Not reflected: derived
/// state, never saved into a scene.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub struct InEnvironment(pub u8);

/// The 8-bit instance / ray mask of an entity: its explicit `RenderLayers` if it has them,
/// else its environment's bit.
pub fn environment_mask(layers: Option<&RenderLayers>, world: Option<&InEnvironment>) -> u8 {
    match layers {
        Some(_) => crate::tlas_builder::layers_mask(layers),
        None => 1 << world.map_or(0, |w| w.0.min(7)),
    }
}

/// The environment a view of `mask` belongs to: its highest bit, so a view of {main, environment} is that
/// environment's (its sky, suns and exposure).
pub fn view_environment(mask: u8) -> u8 {
    (7 - mask.leading_zeros().min(7)) as u8
}

fn assign_bit(
    add: On<Add<Environment>>,
    main: Query<(), With<MainPhysicsEnvironment>>,
    mut environments: ResMut<RenderEnvironments>,
) {
    let environment = add.entity;
    let bit = if main.contains(environment) {
        Some(0)
    } else {
        (1..8).find(|&bit| environments.owners[bit].is_none())
    };
    match bit {
        Some(bit) => environments.owners[bit] = Some(environment),
        None => {
            warn!("aurora: more than 8 environments; {environment} renders in the main environment")
        }
    }
}

fn free_bit(remove: On<Remove<Environment>>, mut environments: ResMut<RenderEnvironments>) {
    for owner in &mut environments.owners {
        if *owner == Some(remove.entity) {
            *owner = None;
        }
    }
}

/// Re-resolves the environment of every entity whose place in the hierarchy changed, and of its
/// subtree. Change-driven: a still hierarchy costs nothing.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn propagate_environments(
    mut commands: Commands,
    worlds: Res<RenderEnvironments>,
    moved: Query<Entity, Or<(Changed<ChildOf>, Added<Environment>)>>,
    mut unparented: RemovedComponents<ChildOf>,
    mut unworlded: RemovedComponents<Environment>,
    is_world: Query<(), With<Environment>>,
    parents: Query<&ChildOf>,
    children: Query<&Children>,
    current: Query<Option<&InEnvironment>>,
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
            // Never removed once set: a move back to the main environment writes 0, so the change
            // reaches everything watching `Changed<InEnvironment>`.
            match current.get(entity).ok().flatten() {
                Some(had) if had.0 == bit => {}
                None if bit == 0 => {}
                _ => {
                    commands.entity(entity).try_insert(InEnvironment(bit));
                }
            }
            if let Ok(kids) = children.get(entity) {
                stack.extend(kids.iter().map(|kid| (kid, bit)));
            }
        }
    }
}

pub struct RenderEnvironmentPlugin;

impl Plugin for RenderEnvironmentPlugin {
    fn build(&self, app: &mut App) {
        avian3d::environment::register_environment_types(app);
        app.register_type::<Environment>()
            .init_resource::<RenderEnvironments>()
            .add_observer(assign_bit)
            .add_observer(free_bit)
            .add_systems(PostUpdate, propagate_environments);
    }

    /// Without avian's physics plugins nothing spawns the main environment, yet it is where the
    /// main environment's sky lives: spawn it here (after every plugin has built, so avian's own
    /// is seen first). Either way it renders.
    fn finish(&self, app: &mut App) {
        if !app
            .world()
            .contains_resource::<MainPhysicsEnvironmentEntity>()
        {
            let main = app.world_mut().spawn(MainPhysicsEnvironment).id();
            app.insert_resource(MainPhysicsEnvironmentEntity(main));
        }
        // The main environment renders.
        let main = app.world().resource::<MainPhysicsEnvironmentEntity>().0;
        app.world_mut().entity_mut(main).insert(Environment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, RenderEnvironmentPlugin));
        app.world_mut().spawn((MainPhysicsEnvironment, Environment));
        app
    }

    fn bit(app: &App, entity: Entity) -> u8 {
        app.world().get::<InEnvironment>(entity).map_or(0, |w| w.0)
    }

    #[test]
    fn descendants_take_their_environments_bit() {
        let mut app = app();
        let outside = app.world_mut().spawn_empty().id();
        let environment = app.world_mut().spawn(Environment).id();
        let child = app.world_mut().spawn(ChildOf(environment)).id();
        let grandchild = app.world_mut().spawn(ChildOf(child)).id();
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderEnvironments>()
                .bit(environment),
            Some(1)
        );
        assert_eq!(bit(&app, environment), 1);
        assert_eq!(bit(&app, child), 1);
        assert_eq!(bit(&app, grandchild), 1);
        assert_eq!(bit(&app, outside), 0);
    }

    #[test]
    fn a_reparented_subtree_follows_its_new_environment() {
        let mut app = app();
        let a = app.world_mut().spawn(Environment).id();
        let b = app.world_mut().spawn(Environment).id();
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
    fn a_physics_only_environment_renders_in_the_one_above() {
        let mut app = app();
        let outer = app.world_mut().spawn(Environment).id();
        let physics = app
            .world_mut()
            .spawn((PhysicsEnvironment, ChildOf(outer)))
            .id();
        let body = app.world_mut().spawn(ChildOf(physics)).id();
        app.update();
        let bits = app.world().resource::<RenderEnvironments>();
        assert_eq!((bits.bit(outer), bits.bit(physics)), (Some(1), None));
        assert_eq!(bit(&app, body), 1);
    }

    #[test]
    fn a_despawned_environment_frees_its_bit() {
        let mut app = app();
        let a = app.world_mut().spawn(Environment).id();
        app.update();
        app.world_mut().despawn(a);
        let b = app.world_mut().spawn(Environment).id();
        app.update();
        assert_eq!(app.world().resource::<RenderEnvironments>().bit(b), Some(1));
    }

    #[test]
    fn a_ninth_environment_renders_in_the_main_environment() {
        let mut app = app();
        let environments: Vec<Entity> = (0..8)
            .map(|_| app.world_mut().spawn(Environment).id())
            .collect();
        let child = app.world_mut().spawn(ChildOf(environments[7])).id();
        app.update();
        assert_eq!(
            app.world()
                .resource::<RenderEnvironments>()
                .bit(environments[6]),
            Some(7)
        );
        assert_eq!(
            app.world()
                .resource::<RenderEnvironments>()
                .bit(environments[7]),
            None
        );
        assert_eq!(bit(&app, child), 0);
    }

    #[test]
    fn explicit_render_layers_override_the_world() {
        let layers = RenderLayers::layer(3);
        assert_eq!(
            environment_mask(Some(&layers), Some(&InEnvironment(1))),
            0b1000
        );
        assert_eq!(environment_mask(None, Some(&InEnvironment(2))), 0b100);
        assert_eq!(environment_mask(None, None), 0b1);
    }
}
