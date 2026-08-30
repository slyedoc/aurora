//! `.bsn` scene support: the components a baked scene names, and the plugin that makes the
//! loader, the `.cluster_mesh` meshes and the reflected `StandardMaterial` all resolvable.
//!
//! A baked entity looks like
//!
//! ```text
//! bevy_transform::components::transform::Transform { translation: glam::Vec3 { .. } }
//! bevy_mesh::components::Mesh3d("bistro/meshes/mesh12_0.cluster_mesh")
//! bevy_aurora::bsn::RaytracingMaterial3d(bevy_pbr::pbr_material::StandardMaterial { metallic: 1.0, .. }),
//! ```
//!
//! [`RaytracingMaterial3d`] exists because the `.bsn` grammar has no generics, so
//! `MeshMaterial3d<StandardMaterial>` cannot be spelled; a system mirrors it into the
//! `MeshMaterial3d` the BLAS / TLAS builders read. Meshes without any material get a default one.

use bevy::{asset::AssetApp, ecs::template::FromTemplate, prelude::*};

use crate::cluster_mesh::ClusterMeshLoader;

/// Material for a ray-traced mesh entity, as authored in `.bsn` (an inline `StandardMaterial`
/// literal or an asset path). Mirrored into `MeshMaterial3d<StandardMaterial>` every frame it
/// changes.
#[derive(Component, FromTemplate, Clone, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct RaytracingMaterial3d(pub Handle<StandardMaterial>);

/// The material given to `Mesh3d` entities that carry neither material component.
#[derive(Resource)]
pub struct DefaultRaytracingMaterial(pub Handle<StandardMaterial>);

impl FromWorld for DefaultRaytracingMaterial {
    fn from_world(world: &mut World) -> Self {
        let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
        Self(materials.add(StandardMaterial::default()))
    }
}

pub struct BsnPlugin;

impl Plugin for BsnPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<ClusterMeshLoader>();
        app.register_type::<RaytracingMaterial3d>();
        app.register_type::<Mesh3d>();
        app.register_type::<StandardMaterial>();
        app.register_asset_reflect::<StandardMaterial>();
        app.register_asset_reflect::<Mesh>();
        app.register_asset_reflect::<Image>();
        // `Name("…")` in a .bsn: the loader needs a String -> HashedStr conversion.
        app.register_type_conversion::<String, bevy::ecs::name::HashedStr, _>(|s| Ok(s.into()));
        app.init_resource::<DefaultRaytracingMaterial>();
        app.add_systems(PostUpdate, (sync_raytracing_material, default_material));
    }
}

fn sync_raytracing_material(
    mut commands: Commands,
    changed: Query<(Entity, &RaytracingMaterial3d), Changed<RaytracingMaterial3d>>,
) {
    for (entity, material) in &changed {
        commands
            .entity(entity)
            .insert(MeshMaterial3d(material.0.clone()));
    }
}

fn default_material(
    mut commands: Commands,
    default: Res<DefaultRaytracingMaterial>,
    bare: Query<
        Entity,
        (
            With<Mesh3d>,
            Without<MeshMaterial3d<StandardMaterial>>,
            Without<RaytracingMaterial3d>,
        ),
    >,
) {
    for entity in &bare {
        commands
            .entity(entity)
            .insert(MeshMaterial3d(default.0.clone()));
    }
}
