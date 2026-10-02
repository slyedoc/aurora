//! `.bsn` scene support: the reflection a baked scene needs beyond what each asset's own
//! plugin registers.
//!
//! A baked entity looks like
//!
//! ```text
//! bevy_transform::components::transform::Transform { translation: glam::Vec3 { .. } }
//! bevy_aurora::mesh::AuroraMesh3d("bistro/meshes/mesh12_0.aurora_mesh")
//! bevy_aurora::material::AuroraMaterial3d(bevy_aurora::material::AuroraMaterial { metallic: 1.0, .. }),
//! ```
//!
//! `AuroraMesh3d` and `AuroraMaterial3d` are registered by their own plugins.

use bevy::{asset::AssetApp, prelude::*};

pub struct BsnPlugin;

impl Plugin for BsnPlugin {
    fn build(&self, app: &mut App) {
        app.register_asset_reflect::<Image>();
        // `Name("…")` in a .bsn: the loader needs a String -> HashedStr conversion.
        app.register_type_conversion::<String, bevy::ecs::name::HashedStr, _>(|s| Ok(s.into()));
    }
}
