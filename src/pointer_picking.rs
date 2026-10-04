//! A bevy_picking backend on the GPU ray caster: `Pointer<Click>` and friends on anything the
//! tracer draws, with no CPU mesh copy.
//!
//! Every pointer ray bevy_picking computes (its `RayMap`: one per pointer and camera whose
//! viewport the pointer is over) gets a [`RayCaster`] in the camera's world, and its hits come
//! back as `PointerHits`. Like every GPU pick, the hits trail the pointer by one frame
//! (picking.rs). `Pickable` is honoured: an entity that is not hoverable is skipped, and one
//! that blocks lower stops the list.
//!
//! Active only when bevy_picking's `PickingPlugin` is in the app.

use bevy::{
    camera::visibility::RenderLayers,
    picking::{
        Pickable, PickingSystems,
        backend::{
            HitData, PointerHits,
            ray::{RayId, RayMap},
        },
    },
    platform::collections::HashMap,
    prelude::*,
};

use crate::{
    picking::{RayCaster, RayHits},
    world::{InWorld, world_mask},
};

/// The caster tracing one pointer ray.
#[derive(Component)]
struct PointerRay(RayId);

/// Casters by the pointer ray they trace.
#[derive(Resource, Default)]
struct PointerCasters(HashMap<RayId, Entity>);

/// Last frame's hits of every pointer ray, as `PointerHits`.
fn report_hits(
    casters: Query<(&PointerRay, &RayHits)>,
    cameras: Query<&Camera>,
    pickables: Query<&Pickable>,
    mut out: MessageWriter<PointerHits>,
) {
    for (PointerRay(id), hits) in &casters {
        let Ok(camera) = cameras.get(id.camera) else {
            continue;
        };
        let mut picks = Vec::new();
        for hit in hits.iter() {
            let pickable = pickables.get(hit.entity).ok();
            if pickable.is_none_or(|p| p.is_hoverable) {
                picks.push((
                    hit.entity,
                    HitData::new(id.camera, hit.distance, Some(hit.point), Some(hit.normal)),
                ));
            }
            if pickable.is_some_and(|p| p.should_block_lower) {
                break;
            }
        }
        out.write(PointerHits::new(id.pointer, picks, camera.order as f32));
    }
}

/// One caster per pointer ray, aimed along it in its camera's world; casters whose ray is
/// gone (the pointer left the viewport) despawn.
fn aim_pointer_rays(
    mut commands: Commands,
    rays: Res<RayMap>,
    mut casters: ResMut<PointerCasters>,
    mut aimed: Query<&mut RayCaster, With<PointerRay>>,
    cameras: Query<(Option<&RenderLayers>, Option<&InWorld>)>,
) {
    casters.0.retain(|id, entity| {
        let live = rays.map.contains_key(id);
        if !live {
            commands.entity(*entity).despawn();
        }
        live
    });
    for (&id, &ray) in rays.iter() {
        let (layers, world) = cameras.get(id.camera).unwrap_or((None, None));
        let mask = world_mask(layers, world);
        match casters.0.get(&id).and_then(|e| aimed.get_mut(*e).ok()) {
            Some(mut caster) => {
                caster.ray = ray;
                caster.mask = mask;
            }
            None => {
                let mut caster = RayCaster::new(ray);
                caster.mask = mask;
                let entity = commands
                    .spawn((Name::new("Pointer ray"), PointerRay(id), caster))
                    .id();
                casters.0.insert(id, entity);
            }
        }
    }
}

pub struct PointerPickingPlugin;

impl Plugin for PointerPickingPlugin {
    fn build(&self, _app: &mut App) {}

    /// After every plugin has built, so bevy_picking's resources are there if it is.
    fn finish(&self, app: &mut App) {
        if !app.world().contains_resource::<RayMap>() {
            return;
        }
        app.init_resource::<PointerCasters>().add_systems(
            PreUpdate,
            (report_hits, aim_pointer_rays)
                .chain()
                .in_set(PickingSystems::Backend),
        );
    }
}
