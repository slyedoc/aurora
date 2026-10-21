//! A windowless app that keeps running: built without `WinitPlugin`, its frames driven by the
//! schedule runner, the first active `Camera3d` rendering into a fixed-size display image
//! ([`HeadlessTarget`]), and drawing the UI over it as a window would (it is the default UI
//! camera: there is no window for bevy_ui to find one by). It makes no sound: the main bus is held
//! silent (a headless app is driven by tools, and nobody is listening). Screenshots read that image back
//! ([`ScreenshotRequests::request_image`](crate::util::ScreenshotRequests::request_image)).
//!
//! `AuroraDefaultPlugins` sets it up itself under `--headless [WxH]`.

use std::time::Duration;

use bevy::{app::ScheduleRunnerPlugin, camera::RenderTarget, prelude::*, ui::IsDefaultUiCamera};
use bevy_seedling::prelude::{MainBus, Volume, VolumeNode};

use crate::camera_target::camera_target_placeholder;

/// `--headless [WxH]` on the command line: the image's size (1280x720 without one).
pub fn headless_size() -> Option<UVec2> {
    let args: Vec<String> = std::env::args().collect();
    let at = args.iter().position(|a| a == "--headless")?;
    let size = args
        .get(at + 1)
        .and_then(|v| v.split_once('x'))
        .and_then(|(w, h)| Some(UVec2::new(w.parse().ok()?, h.parse().ok()?)));
    Some(size.unwrap_or(UVec2::new(1280, 720)))
}

pub struct Headless {
    pub size: UVec2,
}

/// The image a headless app's camera renders into.
#[derive(Resource, Clone)]
pub struct HeadlessTarget {
    pub image: Handle<Image>,
    pub size: UVec2,
}

impl Plugin for Headless {
    fn build(&self, app: &mut App) {
        app.add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
            .add_systems(PostUpdate, retarget)
            .add_systems(Last, mute);
    }

    fn finish(&self, app: &mut App) {
        let image = app
            .world_mut()
            .resource_mut::<Assets<Image>>()
            .add(camera_target_placeholder(self.size));
        app.insert_resource(HeadlessTarget {
            image,
            size: self.size,
        });
    }
}

/// Holds the main bus silent, whatever the app sets it to.
fn mute(mut main: Query<&mut VolumeNode, With<MainBus>>) {
    for mut bus in &mut main {
        if bus.volume != Volume::SILENT {
            bus.volume = Volume::SILENT;
        }
    }
}

/// Keeps the target, and the UI, on the first active `Camera3d` (a game may replace its camera
/// as it loads).
fn retarget(
    mut commands: Commands,
    target: Res<HeadlessTarget>,
    cameras: Query<(Entity, &Camera, Option<&RenderTarget>), With<Camera3d>>,
    ui_cameras: Query<Entity, With<IsDefaultUiCamera>>,
) {
    let mut active: Vec<_> = cameras.iter().filter(|(_, c, _)| c.is_active).collect();
    active.sort_by_key(|(e, _, _)| *e);
    let Some((camera, _, current)) = active.first().copied() else {
        return;
    };
    let ours = matches!(current, Some(RenderTarget::Image(t)) if t.handle == target.image);
    if !ours {
        commands
            .entity(camera)
            .insert(RenderTarget::Image(target.image.clone().into()));
    }
    for other in ui_cameras.iter().filter(|e| *e != camera) {
        commands.entity(other).remove::<IsDefaultUiCamera>();
    }
    if !ui_cameras.contains(camera) {
        commands.entity(camera).insert(IsDefaultUiCamera);
    }
}
