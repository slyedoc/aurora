//! The dev panel, for apps without an editor: fps and the centre-screen exposure probe over
//! feathers inspectors for [`RenderSettings`] and the first camera's lens, exposure and DLSS
//! mode. It holds no state of its own; an editor edits the same resource and components.
//!
//! Keys: `F2` toggles this panel.

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
        BuildComponentInspector, DefaultInspectorWidgetsPlugin, FeathersInspectorPlugins,
        InspectorCollapsed, InspectorRoot, build_resource_inspector,
    },
    prelude::*,
    ui::{Display, GlobalZIndex},
};

use crate::{
    auto_exposure::{AuroraExposure, ev100_from_ev},
    dlss::AuroraDlss,
    render_settings::{AuroraLens, RenderSettings},
    ui_render::UiRenderPlugin,
};

/// The panel root; `F2` flips its `Display`. Public so apps can hang their own sections
/// under it (a `Node` child + `BuildComponentInspector` / `BuildResourceInspector`).
#[derive(Component, Default, Clone)]
pub struct DevUIPanel;

/// The live stats line (fps).
#[derive(Component, Default, Clone)]
struct DevUIStats;

/// The exposure/luminance readout. Its OWN line on purpose: the panel is a fixed 340 px
/// and text wider than that relayouts every frame, which reads as the whole panel
/// flickering. Both lines are padded to a stable width for the same reason -- a readout
/// that changes length as the number changes digits is the same bug, just intermittent.
#[derive(Component, Default, Clone)]
struct DevUIProbe;

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
        // `AuroraDefaultPlugins` carries `UiRenderPlugin` now; this keeps the panel
        // standalone for an app that builds its own group.
        if !app.is_plugin_added::<UiRenderPlugin>() {
            app.add_plugins(UiRenderPlugin);
        }
        if !app.is_plugin_added::<FeathersCorePlugin>() {
            app.add_plugins(FeathersPlugins);
        }
        if !app.is_plugin_added::<DefaultInspectorWidgetsPlugin>() {
            app.add_plugins(FeathersInspectorPlugins);
        }

        app.add_systems(Startup, spawn_panel);
        app.add_systems(Update, (toggle_panel, update_stats, inspect_camera));
    }
}

/// The first 3D camera's lens, exposure and DLSS mode join the panel once it exists.
fn inspect_camera(
    cameras: Query<(Entity, Has<AuroraLens>, Has<AuroraExposure>, Has<AuroraDlss>), With<Camera3d>>,
    host: Single<Entity, With<DevUIInspectorHost>>,
    mut shown: Local<Option<Entity>>,
    mut commands: Commands,
) {
    let Some((camera, lens, exposure, dlss)) = cameras.iter().next() else {
        return;
    };
    if *shown == Some(camera) {
        return;
    }
    *shown = Some(camera);
    let mut entity = commands.entity(camera);
    if !lens {
        entity.insert(AuroraLens::default());
    }
    if !exposure {
        entity.insert(AuroraExposure::default());
    }
    if !dlss {
        entity.insert(AuroraDlss::default());
    }
    for type_id in [
        TypeId::of::<AuroraLens>(),
        TypeId::of::<AuroraExposure>(),
        TypeId::of::<AuroraDlss>(),
    ] {
        commands.queue(BuildComponentInspector {
            target: camera,
            type_id,
            panel: *host,
        });
    }
}

/// Stacking order for aurora's own debug overlays: above any app UI, which sits at 0 unless it
/// says otherwise.
pub const DEV_UI_Z: i32 = 1_000;

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
            // Above whatever the app puts on screen. UI with no z-index stacks by SPAWN ORDER,
            // so a debug overlay would otherwise win or lose the race depending on which
            // startup system ran last -- and an app whose own panel covers this one leaves no
            // way to reach the controls that would tell you why. `GlobalZIndex` escapes the
            // local stacking context, so nesting cannot bury it either.
            GlobalZIndex({DEV_UI_Z})
            ThemeBackgroundColor(tokens::WINDOW_BG)
            DevUIPanel
            Children [
                @caption("aurora  (F2: panel)")
                --
                @caption("fps: -") DevUIStats
                --
                @caption("centre: -") DevUIProbe
                --
                Node {
                    flex_direction: FlexDirection::Column,
                    align_self: AlignSelf::Stretch,
                }
                DevUIInspectorHost
            ]
        })
        .expect("dev panel spawns")
        .id();
    world.flush();

    // The card starts collapsed (expanding is one click; the panel stays compact).
    world.resource_mut::<InspectorCollapsed>().set(
        &InspectorRoot::Resource {
            type_id: TypeId::of::<RenderSettings>(),
        },
        "",
        true,
    );

    let host = world
        .query_filtered::<Entity, With<DevUIInspectorHost>>()
        .iter(world)
        .find(|_| true)
        .unwrap_or(panel);
    build_resource_inspector(world, TypeId::of::<RenderSettings>(), host);
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
    ae: Option<Res<crate::auto_exposure::AutoExposureState>>,
    cameras: Query<&AuroraExposure, With<Camera3d>>,
    skies: Option<Res<crate::sky::EnvironmentSkies>>,
    mut stats: Query<&mut Text, (With<DevUIStats>, Without<DevUIProbe>)>,
    mut probe: Query<&mut Text, (With<DevUIProbe>, Without<DevUIStats>)>,
    mut fps_avg: Local<f32>,
) {
    let dt = time.delta_secs();
    if dt > 0.0 {
        *fps_avg = 0.95 * *fps_avg + 0.05 * (1.0 / dt);
    }
    // Centre-screen luminance in NITS and the EV100 in force. Physical and
    // exposure-independent, so an emitter authored too dim and a look keyed to something
    // bright stop being the same symptom.
    let nits = ae.as_ref().map_or(0.0, |ae| ae.probe_nits());
    let ev100 = cameras.iter().next().map(|exposure| match exposure {
        AuroraExposure::Fixed(fixed) => (ev100_from_ev(fixed.ev), 'L'),
        AuroraExposure::Environment => (skies.as_ref().map_or(f32::NAN, |s| s.camera().ev100), 'W'),
        AuroraExposure::Auto(_) => (f32::NAN, 'A'),
    });
    for mut text in &mut stats {
        text.0 = format!("fps: {:>6.1}", *fps_avg);
    }
    // Both halves padded to a FIXED width, and the whole line kept inside the panel's
    // 340 px: a line that grows as the numbers change digits relayouts the panel every
    // frame, which reads as a flicker.
    let exposure = match ev100 {
        Some((_, 'A')) => "EV100 auto".to_string(),
        Some((ev, tag)) => format!("EV100 {ev:.1}{tag}"),
        None => String::new(),
    };
    for mut text in &mut probe {
        text.0 = format!("{:<11}{:>11}", fmt_nits(nits), exposure);
    }
}

/// Nits with a magnitude suffix and the reference class from
/// aurora_files/lighting_units.md, so the number is readable without the table to hand.
fn fmt_nits(nits: f32) -> String {
    if !nits.is_finite() || nits <= 0.0 {
        return "-".to_string();
    }
    let class = match nits {
        n if n < 1.0 => "shdw",
        n if n < 100.0 => "dim",
        n if n < 600.0 => "scrn",
        n if n < 8_000.0 => "sky",
        n if n < 100_000.0 => "lum",
        _ => "sun",
    };
    if nits >= 1000.0 {
        format!("{:.0}k nt {}", nits / 1000.0, class)
    } else {
        format!("{nits:.0} nt {class}")
    }
}
