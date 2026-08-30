//! `.bsn` scene support: the plugin that makes the loader, the `.cluster_mesh` meshes and the
//! reflected material all resolvable.
//!
//! A baked entity looks like
//!
//! ```text
//! bevy_transform::components::transform::Transform { translation: glam::Vec3 { .. } }
//! bevy_mesh::components::Mesh3d("bistro/meshes/mesh12_0.cluster_mesh")
//! bevy_aurora::material::AuroraMaterial3d(bevy_aurora::material::AuroraMaterial { metallic: 1.0, .. }),
//! ```

use bevy::{asset::AssetApp, prelude::*};

use crate::cluster_mesh::ClusterMeshLoader;

pub struct BsnPlugin;

impl Plugin for BsnPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset_loader::<ClusterMeshLoader>();
        app.register_type::<Mesh3d>();
        app.register_asset_reflect::<Mesh>();
        app.register_asset_reflect::<Image>();
        // `Name("…")` in a .bsn: the loader needs a String -> HashedStr conversion.
        app.register_type_conversion::<String, bevy::ecs::name::HashedStr, _>(|s| Ok(s.into()));
    }
}
