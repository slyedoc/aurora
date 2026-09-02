//! The dev panel: a feathers inspector over the renderer's tunables.
//!
//! [`DevUIState`] is a reflected main-world resource; `bevy_feathers_inspector` generates the
//! sliders from its `#[reflect(@range)]` attributes and writes edits back through reflection.
//! The frame reads the tunables straight from the resource
//! into the frame uniform. Drawn by [`crate::ui_render`], so no wgpu / egui anywhere.
//!
//! Keys: `F2` toggles this panel, `F1` the world inspector.

use std::any::TypeId;

use bevy::{
    feathers::{
        FeathersCorePlugin, FeathersPlugins,
        dark_theme::create_dark_theme,
        display::caption,
        theme::{ThemeBackgroundColor, UiTheme},
        tokens,
    },
    feathers_inspector::{
        DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins, InspectorCollapsed, InspectorRoot,
        build_resource_inspector,
    },
    prelude::*,
    ui::Display,
};

use crate::{
    dlss::{AuroraDlss, RrPreset},
    sky::ProceduralSky,
    ui_render::UiRenderPlugin,
};

/// Renderer tunables edited from the dev panel; the frame reads them directly.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Default)]
pub struct DevUIState {
    #[reflect(@1.5..=3.0_f32)]
    pub gamma: f32,
    #[reflect(@0.0..=0.02_f32)]
    pub aperture: f32,
    #[reflect(@0.0..=0.2_f32)]
    pub foginess: f32,
    #[reflect(@-1.0..=1.0_f32)]
    pub fog_scatter: f32,
    #[reflect(@0.0..=1.0_f32)]
    pub sky_brightness: f32,
    /// Firefly suppression: indirect contributions are clamped to this many times the sky's
    /// luminance (0 = off). Biases bright indirect paths down; kills the speckle Ray
    /// Reconstruction would otherwise smear.
    #[reflect(@0.0..=64.0_f32)]
    pub firefly_clamp: f32,
    /// Paths per pixel per frame; RR is trained for 1 spp, so extra samples mostly buy
    /// trace time.
    #[reflect(@1.0..=4.0_f32)]
    pub samples: u32,
    /// Maximum path length. With the firefly clamp a short path is all the denoiser needs.
    #[reflect(@1.0..=64.0_f32)]
    pub max_bounces: u32,
    /// Post-process vignette strength (0 = off); aspect-corrected, darkens towards the corners.
    #[reflect(@0.0..=1.0_f32)]
    pub vignette: f32,
    /// Next-event estimation for emissive triangles (off = BRDF sampling only, the
    /// reference estimator).
    pub light_nee: bool,
    /// ReSTIR DI at the primary vertex (initial candidates + temporal reuse). Off while
    /// accumulating, so Space stays the uncorrelated reference.
    pub restir: bool,
    /// Initial light candidates per pixel.
    #[reflect(@1.0..=32.0_f32)]
    pub restir_candidates: u32,
    /// Temporal history cap, in multiples of the candidate count.
    #[reflect(@0.0..=64.0_f32)]
    pub restir_history: f32,
    /// Radiance cache: paths terminate into converged voxels from bounce 2 on. Biased by
    /// construction; off while accumulating.
    pub sharc: bool,
    /// Cache voxel size at the camera (meters); doubles per distance octave past 8m.
    #[reflect(@0.05..=2.0_f32)]
    pub sharc_voxel: f32,
    /// Opacity micromaps on alpha-cutout meshes that carry a bake (off = every instance
    /// traces through the any-hit alpha test, for A/B).
    pub omm: bool,
    /// DLSS Ray Reconstruction mode; mirrors the camera's [`AuroraDlss`] component both ways.
    pub dlss: AuroraDlss,
    /// Ray Reconstruction model preset; changing it rebuilds the feature.
    pub rr_preset: RrPreset,
}

impl Default for DevUIState {
    fn default() -> Self {
        Self {
            gamma: 2.4,
            aperture: 0.0,
            foginess: 0.001,
            fog_scatter: 0.9,
            sky_brightness: 1.0,
            firefly_clamp: 8.0,
            samples: 1,
            max_bounces: 32,
            vignette: 0.0,
            light_nee: true,
            restir: false, // TODO
            restir_candidates: 8,
            restir_history: 20.0,
            sharc: false, // TODO
            sharc_voxel: 0.25,
            omm: true,
            dlss: AuroraDlss::from_env(),
            rr_preset: RrPreset::current(),
        }
    }
}

/// The panel root; `F2` flips its `Display`.
#[derive(Component, Default, Clone)]
struct DevUIPanel;

/// The live stats line (fps).
#[derive(Component, Default, Clone)]
struct DevUIStats;

/// The node the resource inspector is built under.
#[derive(Component, Default, Clone)]
struct DevUIInspectorHost;

/// The node the [`ProceduralSky`] inspector is built under.
#[derive(Component, Default, Clone)]
struct DevUISkyHost;

pub struct DevUIPlugin;

impl Plugin for DevUIPlugin {
    fn build(&self, app: &mut App) {
        // FeathersCorePlugin inits an EMPTY UiTheme (every token resolves to magenta), so
        // decide on the theme before it runs: keep the app's own theme if it set one.
        if !app.world().contains_resource::<UiTheme>() {
            app.insert_resource(UiTheme(create_dark_theme()));
        }
        if !app.is_plugin_added::<UiRenderPlugin>() {
            app.add_plugins(UiRenderPlugin);
        }
        if !app.is_plugin_added::<FeathersCorePlugin>() {
            app.add_plugins(FeathersPlugins);
        }
        if !app.is_plugin_added::<DefaultInspectorWidgetsPlugin>() {
            app.add_plugins(FeathersInspectorPlugins);
        }
        // if !app.is_plugin_added::<WorldInspectorPlugin>() {
        //     app.add_plugins(WorldInspectorPlugin::new().with_toggle_key(KeyCode::F1));
        // }

        app.register_type::<DevUIState>();
        app.init_resource::<DevUIState>();
        app.add_systems(Startup, spawn_panel);
        app.add_systems(
            Update,
            (toggle_panel, update_stats, sync_dlss_mode, sync_rr_preset),
        );
    }
}

/// Keeps the panel's `dlss` field and the camera's [`AuroraDlss`] component equal: whichever
/// side moved last (the panel, or F3 cycling the component) wins.
fn sync_dlss_mode(
    mut state: ResMut<DevUIState>,
    mut cameras: Query<&mut AuroraDlss, With<Camera3d>>,
    mut agreed: Local<Option<AuroraDlss>>,
) {
    let last = agreed.unwrap_or(state.dlss);
    if state.dlss != last {
        for mut mode in &mut cameras {
            if *mode != state.dlss {
                *mode = state.dlss;
            }
        }
        *agreed = Some(state.dlss);
        return;
    }
    if let Some(mode) = cameras.iter().find(|m| **m != last) {
        state.dlss = *mode;
        *agreed = Some(*mode);
        return;
    }
    *agreed = Some(last);
}

/// Applies the panel's preset row; the renderer rebuilds the feature on the next frame.
fn sync_rr_preset(state: Res<DevUIState>) {
    if state.rr_preset != RrPreset::current() {
        state.rr_preset.make_current();
    }
}

fn spawn_panel(world: &mut World) {
    let panel = world
        .spawn_scene(bsn! {
            Node {
                position_type: PositionType::Absolute,
                left: px(16),
                top: px(16),
                width: px(340),
                padding: UiRect::all(px(10)),
                flex_direction: FlexDirection::Column,
                row_gap: px(6),
                border_radius: BorderRadius::all(px(6)),
            }
            ThemeBackgroundColor(tokens::WINDOW_BG)
            DevUIPanel
            Children [
                caption("aurora  (F2: panel, F1: world inspector)"),
                (caption("fps: -") DevUIStats),
                (
                    Node {
                        flex_direction: FlexDirection::Column,
                        align_self: AlignSelf::Stretch,
                    }
                    DevUIInspectorHost
                ),
                caption("sky (procedural)"),
                (
                    Node {
                        flex_direction: FlexDirection::Column,
                        align_self: AlignSelf::Stretch,
                    }
                    DevUISkyHost
                ),
            ]
        })
        .expect("dev panel spawns")
        .id();
    world.flush();

    // Both cards start collapsed (expanding is one click; the panel stays compact).
    {
        let mut collapsed = world.resource_mut::<InspectorCollapsed>();
        for type_id in [TypeId::of::<DevUIState>(), TypeId::of::<ProceduralSky>()] {
            collapsed.set(&InspectorRoot::Resource { type_id }, "", true);
        }
    }

    let host = world
        .query_filtered::<Entity, With<DevUIInspectorHost>>()
        .iter(world)
        .find(|_| true)
        .unwrap_or(panel);
    build_resource_inspector(world, TypeId::of::<DevUIState>(), host);
    let sky_host = world
        .query_filtered::<Entity, With<DevUISkyHost>>()
        .iter(world)
        .find(|_| true)
        .unwrap_or(panel);
    build_resource_inspector(world, TypeId::of::<ProceduralSky>(), sky_host);
}

fn toggle_panel(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut panels: Query<&mut Node, With<DevUIPanel>>,
) {
    if keyboard.just_pressed(KeyCode::F2) {
        for mut node in &mut panels {
            node.display = if node.display == Display::None {
                Display::Flex
            } else {
                Display::None
            };
        }
    }
}

fn update_stats(
    time: Res<Time>,
    mut stats: Query<&mut Text, With<DevUIStats>>,
    mut fps_avg: Local<f32>,
) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        *fps_avg = 0.95 * *fps_avg + 0.05 * (1.0 / dt);
    }
    for mut text in &mut stats {
        text.0 = format!("fps: {:.1}", *fps_avg);
    }
}
