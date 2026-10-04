//! Transforms without CPU hierarchy propagation.
//!
//! World transforms come from the GPU node table (`gpu_transform.rs`), which propagates the
//! hierarchy from local `Transform` deltas and reads the rows it rewrote back into
//! `GlobalTransform` at the next `First`. The CPU only fills in what has to be exact in the
//! frame being drawn:
//! - root entities without children: `GlobalTransform = Transform` (a copy, no walk);
//! - cameras inside a hierarchy (one that crossed a portal into a world): composed from
//!   their ancestors' `Transform`s, since the view is built before this frame's GPU rows
//!   could come back.
//!
//! A host with no GPU (headless tests) has no readback: it uses bevy's own `TransformPlugin`
//! and propagates on the CPU instead.

use bevy::{
    app::ValidateParentHasComponentPlugin,
    prelude::*,
    transform::{
        TransformSystems,
        systems::{StaticTransformOptimizations, sync_simple_transforms},
    },
};

/// Cameras inside a hierarchy: this frame's pose from their ancestors' `Transform`s.
#[allow(clippy::type_complexity)]
fn sync_child_cameras(
    mut cameras: Query<(&Transform, &ChildOf, &mut GlobalTransform), With<Camera>>,
    frames: Query<(&Transform, Option<&ChildOf>), Without<Camera>>,
) {
    for (local, child_of, mut global) in &mut cameras {
        let mut pose = local.compute_affine();
        let mut parent = Some(child_of.parent());
        while let Some(entity) = parent {
            let Ok((transform, up)) = frames.get(entity) else {
                break;
            };
            pose = transform.compute_affine() * pose;
            parent = up.map(ChildOf::parent);
        }
        global.set_if_neq(GlobalTransform::from(pose));
    }
}

#[derive(Default)]
pub struct TransformPlugin;

impl Plugin for TransformPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ValidateParentHasComponentPlugin::<GlobalTransform>::default())
            .init_resource::<StaticTransformOptimizations>()
            .add_systems(
                PostStartup,
                sync_simple_transforms.in_set(TransformSystems::Propagate),
            )
            .add_systems(
                PostUpdate,
                (sync_simple_transforms, sync_child_cameras).in_set(TransformSystems::Propagate),
            );
    }
}
