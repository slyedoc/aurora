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
        DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins, WorldInspectorPlugin,
        build_resource_inspector,
    },
    prelude::*,
    ui::Display,
};

use crate::{dlss::AuroraDlss, ray_render_plugin::RenderConfig, ui_render::UiRenderPlugin};

/// Renderer tunables edited from the dev panel; the frame reads them directly.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Default)]
pub struct DevUIState {
    #[reflect(@1.5..=3.0_f32)]
    pub gamma: f32,
    /// Exposure in stops: the frame is multiplied by `2^exposure_ev` before tonemapping.
    #[reflect(@-20.0..=20.0_f32)]
    pub exposure_ev: f32,
    #[reflect(@0.0..=0.02_f32)]
    pub aperture: f32,
    #[reflect(@0.0..=0.2_f32)]
    pub foginess: f32,
    #[reflect(@-1.0..=1.0_f32)]
    pub fog_scatter: f32,
    #[reflect(@0.0..=1.0_f32)]
    pub sky_brightness: f32,
    /// Firefly suppression: indirect contributions are clamped to this many times the sky's
    /// luminance (0 = off). Biases bright indirect paths down; kills speckle for accumulation
    /// and Ray Reconstruction alike.
    #[reflect(@0.0..=64.0_f32)]
    pub firefly_clamp: f32,
    /// Paths per pixel per frame for the plain / accumulating path.
    #[reflect(@1.0..=8.0_f32)]
    pub samples: u32,
    /// Maximum path length for the plain / accumulating path.
    #[reflect(@1.0..=64.0_f32)]
    pub max_bounces: u32,
    /// Paths per pixel while Ray Reconstruction is on; RR is trained for 1 spp, so extra
    /// samples mostly buy trace time.
    #[reflect(@1.0..=4.0_f32)]
    pub dlss_samples: u32,
    /// Maximum path length while Ray Reconstruction is on. With the firefly clamp a short
    /// path is all the denoiser needs.
    #[reflect(@1.0..=64.0_f32)]
    pub dlss_max_bounces: u32,
    /// Post-process vignette strength (0 = off); aspect-corrected, darkens towards the corners.
    #[reflect(@0.0..=1.0_f32)]
    pub vignette: f32,
    /// DLSS Ray Reconstruction mode; mirrors the camera's [`AuroraDlss`] component both ways.
    pub dlss: AuroraDlss,
}

impl Default for DevUIState {
    fn default() -> Self {
        Self {
            gamma: 2.4,
            exposure_ev: 0.0,
            aperture: 0.0,
            foginess: 0.001,
            fog_scatter: 0.9,
            sky_brightness: 1.0,
            firefly_clamp: 8.0,
            samples: 2,
            max_bounces: 64,
            dlss_samples: 1,
            dlss_max_bounces: 8,
            vignette: 0.0,
            dlss: AuroraDlss::from_env(),
        }
    }
}

/// The panel root; `F2` flips its `Display`.
#[derive(Component, Default, Clone)]
struct DevUIPanel;

/// The live stats line (fps, accumulation ticks).
#[derive(Component, Default, Clone)]
struct DevUIStats;

/// The node the resource inspector is built under.
#[derive(Component, Default, Clone)]
struct DevUIInspectorHost;

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
        if !app.is_plugin_added::<WorldInspectorPlugin>() {
            app.add_plugins(WorldInspectorPlugin::new().with_toggle_key(KeyCode::F1));
        }

        app.register_type::<DevUIState>();
        app.init_resource::<DevUIState>();
        app.add_systems(Startup, spawn_panel);
        app.add_systems(Update, (toggle_panel, update_stats, sync_dlss_mode));
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
            ]
        })
        .expect("dev panel spawns")
        .id();
    world.flush();

    let host = world
        .query_filtered::<Entity, With<DevUIInspectorHost>>()
        .iter(world)
        .find(|_| true)
        .unwrap_or(panel);
    build_resource_inspector(world, TypeId::of::<DevUIState>(), host);
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
    render_config: Res<RenderConfig>,
    mut stats: Query<&mut Text, With<DevUIStats>>,
    mut fps_avg: Local<f32>,
    mut ticks: Local<u32>,
    mut since_log: Local<f32>,
) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        *fps_avg = 0.95 * *fps_avg + 0.05 * (1.0 / dt);
    }
    // Also to the log every 5 s, so headless / scripted runs get the number without the panel.
    *since_log += dt;
    if *since_log >= 5.0 {
        *since_log = 0.0;
        log::info!("fps: {:.1}", *fps_avg);
    }
    // Mirrors the frame's accumulation counter: counts while accumulating, else 0.
    *ticks = if render_config.accumulate {
        *ticks + 1
    } else {
        0
    };
    for mut text in &mut stats {
        text.0 = format!("fps: {:.1}    ticks: {}", *fps_avg, *ticks);
    }
}
