//! Transforms without CPU hierarchy propagation.
//!
//! World transforms of ray-traced entities come from the GPU node table
//! (`gpu_transform.rs`), which propagates the hierarchy itself from local `Transform` deltas.
//! Walking the tree on the CPU as well (`propagate_parent_transforms` over every node, every
//! frame something moved) is the single largest CPU cost in a large scene and produces a
//! `GlobalTransform` nothing here reads. So by default only `sync_simple_transforms` runs:
//! root entities without children (cameras, lights, gameplay actors) get
//! `GlobalTransform = Transform`; anything inside a hierarchy keeps whatever `GlobalTransform`
//! it was spawned with.
//!
//! Gameplay that needs CPU world transforms of *children* can turn the propagation back on
//! (`TransformPlugin { propagate_on_cpu: true }`); the GPU path is unaffected either way.

use bevy::{
    app::ValidateParentHasComponentPlugin,
    ecs::{schedule::ScheduleConfigs, system::ScheduleSystem},
    prelude::*,
    transform::{
        TransformSystems,
        systems::{
            StaticTransformOptimizations, mark_dirty_trees, propagate_parent_transforms,
            sync_simple_transforms,
        },
    },
};

fn full_propagation() -> ScheduleConfigs<ScheduleSystem> {
    (
        mark_dirty_trees,
        propagate_parent_transforms,
        sync_simple_transforms,
    )
        .chain()
        .in_set(TransformSystems::Propagate)
}

#[derive(Default)]
pub struct TransformPlugin {
    /// Also run bevy's full hierarchy propagation on the CPU (`GlobalTransform` for every
    /// descendant). Off by default: the renderer never reads it.
    pub propagate_on_cpu: bool,
}

impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ValidateParentHasComponentPlugin::<GlobalTransform>::default())
            .init_resource::<StaticTransformOptimizations>();
        if self.propagate_on_cpu {
            app.add_systems(PostStartup, full_propagation())
                .add_systems(PostUpdate, full_propagation());
        } else {
            app.add_systems(
                PostStartup,
                sync_simple_transforms.in_set(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                sync_simple_transforms.in_set(TransformSystems::Propagate),
            );
        }
    }
}
