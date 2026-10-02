//! GPU skinning: bevy's animated fox, baked with its clips by aurora_files' `prop_import
//! --hierarchy --scale 0.01` (authored in centimetres, baked in metres), path-traced through
//! per-instance deformed BLASes (src/skinning.rs).
//!
//!   cargo run --release --example skinning
//!
//! Number keys 1-3 switch clips.

use bevy::{
    animation::{
        AnimationPlayer,
        graph::{AnimationGraph, AnimationGraphHandle, AnimationNodeIndex},
    },
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    prelude::*,
};
use bevy_aurora::{
    AuroraDefaultPlugins,
    animclip::AnimationTargetsByName,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    mesh::{AuroraMesh, AuroraMesh3d},
    util::{ScreenshotExt, TimeoutAppExt},
};

const CLIPS: [&str; 3] = ["fox/survey.animclip", "fox/walk.animclip", "fox/run.animclip"];

#[derive(Resource)]
struct FoxClips(Vec<AnimationNodeIndex>);

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins,
            DevUIPlugin,
            FreeCameraPlugin::default(),
        ))
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .add_systems(Startup, setup)
        .add_systems(Update, switch_clips)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
) {
    let mut graph = AnimationGraph::new();
    let clips: Vec<AnimationNodeIndex> = CLIPS
        .iter()
        .map(|path| graph.add_clip(asset_server.load(*path), 1.0, graph.root))
        .collect();
    let mut player = AnimationPlayer::default();
    player.play(clips[2]).repeat();
    commands.insert_resource(FoxClips(clips));

    // The player on the scene root; `AnimationTargetsByName` binds the bones once they spawn.
    commands.spawn((
        Name::new("fox"),
        ScenePatchInstance(asset_server.load("fox/fox.bsn")),
        Transform::default(),
        Visibility::Visible,
        AnimationTargetsByName,
        player,
        AnimationGraphHandle(graphs.add(graph)),
    ));

    commands.spawn((
        AuroraMesh3d(meshes.add(AuroraMesh::from_shape(Plane3d::default().mesh().size(40.0, 40.0)))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.35, 0.33, 0.3),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 50.0f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(2.5, 1.4, 3.0).looking_at(Vec3::new(0.0, 0.6, 0.0), Vec3::Y),
    ));
}

fn switch_clips(
    input: Res<ButtonInput<KeyCode>>,
    clips: Res<FoxClips>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let wanted = [KeyCode::Digit1, KeyCode::Digit2, KeyCode::Digit3]
        .into_iter()
        .position(|k| input.just_pressed(k));
    let Some(i) = wanted else { return };
    for mut player in &mut players {
        player.stop_all();
        player.play(clips.0[i]).repeat();
    }
}
