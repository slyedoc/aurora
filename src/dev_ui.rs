//! The dev panel, for apps without an editor: the centre-screen exposure probe over feathers
//! inspectors for [`RenderSettings`] and the first camera's lens, exposure and DLSS mode. It
//! holds no state of its own; an editor edits the same resource and components.
//!
//! Keys: `F3` toggles this panel.

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

/// The panel root; `F3` flips its `Display`. Public so apps can hang their own sections
/// under it (a `Node` child + `BuildComponentInspector` / `BuildResourceInspector`).
#[derive(Component, Default, Clone)]
pub struct DevUIPanel;

/// The exposure/luminance readout, kept inside the panel's fixed 340 px: text wider than that
/// relayouts every frame, which reads as the whole panel flickering. It is padded to a stable
/// width for the same reason -- a readout that changes length as the number changes digits is
/// the same bug, just intermittent.
#[derive(Component, Default, Clone)]
struct DevUIProbe;

/// The frame timing line under the probe.
#[derive(Component, Default, Clone)]
struct DevUIFrame;

/// The node the inspectors are built under, each in a section of its own (an inspector owns
/// its whole panel: building one into a shared node clears the others).
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
        app.add_systems(
            Update,
            (toggle_panel, update_probe, update_frame_readout, inspect_camera),
        );
    }
}

/// The first 3D camera's lens, exposure and DLSS mode join the panel once it exists.
fn inspect_camera(
    cameras: Query<
        (
            Entity,
            Has<AuroraLens>,
            Has<AuroraExposure>,
            Has<AuroraDlss>,
        ),
        With<Camera3d>,
    >,
    host: Single<Entity, With<DevUIInspectorHost>>,
    mut shown: Local<Option<Entity>>,
    mut sections: Local<Vec<Entity>>,
    mut commands: Commands,
) {
    let Some((camera, lens, exposure, dlss)) = cameras.iter().next() else {
        return;
    };
    if *shown == Some(camera) {
        return;
    }
    *shown = Some(camera);
    for section in sections.drain(..) {
        if let Ok(mut section) = commands.get_entity(section) {
            section.despawn();
        }
    }
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
        let section = commands.spawn((section_node(), ChildOf(*host))).id();
        sections.push(section);
        // The camera's cards start closed; the settings above are what the panel is for.
        commands.queue(move |world: &mut World| {
            world.resource_mut::<InspectorCollapsed>().set(
                &InspectorRoot::Component {
                    entity: camera,
                    type_id,
                },
                "",
                true,
            );
        });
        commands.queue(BuildComponentInspector {
            target: camera,
            type_id,
            panel: section,
        });
    }
}

/// Stacking order for aurora's own debug overlays: above any app UI, which sits at 0 unless it
/// says otherwise.
pub const DEV_UI_Z: i32 = 1_000;

fn spawn_panel(world: &mut World) {
    let panel = world
        .spawn_scene(bsn! {
            // Starts closed; F3 opens it.
            Node {
                display: Display::None,
                position_type: PositionType::Absolute,
                left: px(16),
                // Below the app's fps readout in the top-left corner.
                top: px(44),
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
                Node {
                    flex_direction: FlexDirection::Column,
                    align_self: AlignSelf::Stretch,
                }
                DevUIInspectorHost
                --
                @caption("centre: -") DevUIProbe
                @caption("gpu: -") DevUIFrame
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
    let section = world.spawn((section_node(), ChildOf(host))).id();
    build_resource_inspector(world, TypeId::of::<RenderSettings>(), section);
}

/// One inspector's own node in the host's column.
fn section_node() -> Node {
    Node {
        flex_direction: FlexDirection::Column,
        align_self: AlignSelf::Stretch,
        ..default()
    }
}

fn toggle_panel(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut panels: Query<&mut Node, With<DevUIPanel>>,
) {
    if keyboard.just_pressed(KeyCode::F3) {
        for mut node in &mut panels {
            node.display = if node.display == Display::None {
                Display::Flex
            } else {
                Display::None
            };
        }
    }
}

fn update_probe(
    ae: Option<Res<crate::auto_exposure::AutoExposureState>>,
    cameras: Query<
        (
            &AuroraExposure,
            Option<&bevy::camera::visibility::RenderLayers>,
            Option<&crate::environment::InEnvironment>,
        ),
        With<Camera3d>,
    >,
    skies: Option<Res<crate::sky::EnvironmentSkies>>,
    mut probe: Query<&mut Text, With<DevUIProbe>>,
) {
    // Centre-screen luminance in NITS and the EV100 in force. Physical and
    // exposure-independent, so an emitter authored too dim and a look keyed to something
    // bright stop being the same symptom.
    let nits = ae.as_ref().map_or(0.0, |ae| ae.probe_nits());
    let ev100 = cameras
        .iter()
        .next()
        .map(|(exposure, layers, world)| match exposure {
            AuroraExposure::Fixed(fixed) => (ev100_from_ev(fixed.ev), 'L'),
            AuroraExposure::Environment => (
                skies.as_ref().map_or(f32::NAN, |s| {
                    s.of_view(crate::environment::environment_mask(layers, world))
                        .ev100
                }),
                'W',
            ),
            AuroraExposure::Auto(_) => (f32::NAN, 'A'),
        });
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

/// The frame's GPU timeline: span (first to last timestamp), the idle gap before it, the CPU's
/// wait-to-submit time, and Reflex's simulation-to-present latency (0 without Reflex). Fixed
/// width, like the probe line.
fn update_frame_readout(
    profiler: Option<Res<crate::gpu_profiler::GpuProfiler>>,
    latency: Option<Res<crate::latency::Latency>>,
    mut line: Query<&mut Text, With<DevUIFrame>>,
) {
    let Some(p) = profiler else { return };
    let ms = |name| p.time(name).unwrap_or(0.0);
    let lat = latency.map_or(0.0, |l| l.latency_ms);
    for mut text in &mut line {
        text.0 = format!(
            "gpu {:5.2} idle {:4.2} rec {:4.2} lat {:5.1}",
            ms("span"),
            ms("idle"),
            p.cpu_record,
            lat
        );
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
