//! A bevy_picking backend on the GPU ray caster: `Pointer<Click>` and friends on anything the
//! tracer draws, with no CPU mesh copy.
//!
//! Every pointer ray bevy_picking computes (its `RayMap`: one per pointer and camera whose
//! viewport the pointer is over) gets a [`RayCaster`] in the camera's world, and its hits come
//! back as `PointerHits`. Like every GPU pick, the hits trail the pointer by one frame
//! (picking.rs). `Pickable` is honoured: an entity that is not hoverable is skipped, and one
//! that blocks lower stops the list.
//!
//! Every [`UiPointerSource`] but the window cursor (the mouse is already a `RayMap` pointer)
//! traces its ray too: an XR laser off the panels hovers and clicks what it points at.
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
        pointer::PointerId,
    },
    platform::collections::HashMap,
    prelude::*,
};

use crate::{
    environment::{AuroraPrimaryHidden, InEnvironment, environment_mask},
    picking::{RayCaster, RayHit, RayHits},
    ui_render::{UiPointerSource, UiPointerTrack, WindowCursorPointer},
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
        let picks = picks(hits, id.camera, &pickables);
        out.write(PointerHits::new(id.pointer, picks, camera.order as f32));
    }
}

/// `hits` as picks seen from `camera`, honouring `Pickable`.
fn picks(hits: &[RayHit], camera: Entity, pickables: &Query<&Pickable>) -> Vec<(Entity, HitData)> {
    let mut picks = Vec::new();
    for hit in hits {
        let pickable = pickables.get(hit.entity).ok();
        if pickable.is_none_or(|p| p.is_hoverable) {
            picks.push((
                hit.entity,
                HitData::new(camera, hit.distance, Some(hit.point), Some(hit.normal)),
            ));
        }
        if pickable.is_some_and(|p| p.should_block_lower) {
            break;
        }
    }
    picks
}

/// The caster tracing a [`UiPointerSource`]'s ray.
#[derive(Component)]
struct SourceRay;

/// Each pointer source's ray, traced in its own world. Its hits drive the source's pointer
/// off the panels (`drive_ui_pointers`).
#[allow(clippy::type_complexity)]
fn aim_source_rays(
    mut commands: Commands,
    mut sources: Query<
        (
            &Transform,
            Option<&ChildOf>,
            &UiPointerSource,
            &mut UiPointerTrack,
            Option<&InEnvironment>,
        ),
        Without<WindowCursorPointer>,
    >,
    frames: Query<(&Transform, Option<&ChildOf>)>,
    mut casters: Query<&mut RayCaster, With<SourceRay>>,
) {
    for (local, parent, source, mut track, world) in &mut sources {
        let pose =
            crate::transform::pose_from_ancestors(local, parent.map(ChildOf::parent), &frames);
        let Ok(dir) = Dir3::new(pose.transform_vector3(Vec3::NEG_Z)) else {
            continue;
        };
        let mut ray = RayCaster::new(Ray3d::new(pose.translation.into(), dir));
        ray.max_distance = source.max_distance;
        ray.mask = environment_mask(None, world);
        match track.world.and_then(|e| casters.get_mut(e).ok()) {
            Some(mut caster) => *caster = ray,
            None => {
                track.world = Some(
                    commands
                        .spawn((Name::new("Pointer source ray"), SourceRay, ray))
                        .id(),
                );
            }
        }
    }
}

/// What each source points at off the panels, as `PointerHits` (none while on one, so the
/// world behind a panel is never hovered through it).
fn report_source_hits(
    mut sources: Query<
        (&mut UiPointerSource, &UiPointerTrack, &PointerId),
        Without<WindowCursorPointer>,
    >,
    hits: Query<&RayHits>,
    cameras: Query<(Entity, &Camera), With<Camera3d>>,
    pickables: Query<&Pickable>,
    mut out: MessageWriter<PointerHits>,
) {
    let Some((camera, cam)) = cameras.iter().find(|(_, c)| c.is_active) else {
        return;
    };
    for (mut source, track, id) in &mut sources {
        let world_hit = track
            .world
            .and_then(|caster| hits.get(caster).ok())
            .and_then(|h| {
                h.iter()
                    .find(|hit| pickables.get(hit.entity).map_or(true, |p| p.is_hoverable))
                    .copied()
            });
        if source.world_hit != world_hit {
            source.world_hit = world_hit;
        }
        let picks = match (source.active && source.hit.is_none(), track.world) {
            (true, Some(caster)) => hits
                .get(caster)
                .map(|h| picks(h, camera, &pickables))
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        out.write(PointerHits::new(*id, picks, cam.order as f32));
    }
}

/// One caster per pointer ray, aimed along it in its camera's world; casters whose ray is
/// gone (the pointer left the viewport) despawn.
fn aim_pointer_rays(
    mut commands: Commands,
    rays: Res<RayMap>,
    mut casters: ResMut<PointerCasters>,
    mut aimed: Query<&mut RayCaster, With<PointerRay>>,
    cameras: Query<(
        Option<&RenderLayers>,
        Option<&InEnvironment>,
        Option<&AuroraPrimaryHidden>,
    )>,
) {
    casters.0.retain(|id, entity| {
        let live = rays.map.contains_key(id);
        if !live {
            commands.entity(*entity).despawn();
        }
        live
    });
    for (&id, &ray) in rays.iter() {
        let (layers, world, hidden) = cameras.get(id.camera).unwrap_or((None, None, None));
        // The mouse picks what the camera's rays see: not the layers they skip (a first-person
        // body's own head).
        let mask = environment_mask(layers, world) & !hidden.map_or(0, AuroraPrimaryHidden::mask);
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
            (report_hits, report_source_hits, aim_pointer_rays, aim_source_rays)
                .chain()
                .in_set(PickingSystems::Backend),
        );
    }
}
