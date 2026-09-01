//! GPU skinning: bevy's animated Fox.glb (three clips) path-traced through per-instance
//! deformed BLASes (src/skinning.rs).
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
    gltf::GltfAssetLabel,
    prelude::*,
    world_serialization::WorldAssetRoot,
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    util::screenshot::ScreenshotExt,
};

#[derive(Resource)]
struct FoxClips {
    graph: Handle<AnimationGraph>,
    clips: Vec<AnimationNodeIndex>,
}

fn main() {
    App::new()
        .add_plugins((
            RayDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin::default(),
        ))
        .add_screenshot(KeyCode::F12)
        .add_systems(Startup, setup)
        .add_systems(Update, (start_clips, switch_clips))
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut graph = AnimationGraph::new();
    let clips: Vec<AnimationNodeIndex> = (0..3)
        .map(|i| {
            graph.add_clip(
                asset_server.load(GltfAssetLabel::Animation(i).from_asset("models/Fox.glb")),
                1.0,
                graph.root,
            )
        })
        .collect();
    commands.insert_resource(FoxClips {
        graph: graphs.add(graph),
        clips,
    });

    // The fox is authored in centimetres.
    commands.spawn((
        Name::new("fox"),
        WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/Fox.glb"))),
        Transform::from_scale(Vec3::splat(0.01)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(40.0, 40.0))),
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

/// The glTF loader puts an `AnimationPlayer` on the fox's armature root: hook the graph up and
/// start the run clip once it appears.
fn start_clips(
    mut commands: Commands,
    clips: Res<FoxClips>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    for (entity, mut player) in &mut players {
        player.play(clips.clips[2]).repeat();
        commands
            .entity(entity)
            .insert(AnimationGraphHandle(clips.graph.clone()));
    }
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
        player.play(clips.clips[i]).repeat();
    }
}
