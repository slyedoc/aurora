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
    auto_exposure::{AuroraExposure, ev100_from_ev, ev_from_ev100},
    dlss::{AuroraDlss, RrPreset, set_jitter_scale},
    sky::ProceduralSky,
    ui_render::UiRenderPlugin,
};

/// Renderer tunables edited from the dev panel; the frame reads them directly.
#[derive(Resource, Reflect, Clone, Debug)]
#[reflect(Resource, Default)]
pub struct DevUIState {
    #[reflect(@0.0..=0.02_f32)]
    pub aperture: f32,
    #[reflect(@0.0..=0.2_f32)]
    pub foginess: f32,
    #[reflect(@-1.0..=1.0_f32)]
    pub fog_scatter: f32,
    #[reflect(@0.0..=1.0_f32)]
    pub sky_brightness: f32,
    /// Uniform multiplier on every light's emission -- emissive surfaces and analytic
    /// lights alike (the sky has `sky_brightness`). Uniform across lights, so NEE stays
    /// unbiased and the light table's sampling distribution is unchanged.
    #[reflect(@0.0..=100.0_f32)]
    pub emissive_boost: f32,
    /// Firefly suppression: indirect contributions are clamped to this many times the
    /// metered mid-gray luminance, i.e. what the exposure shows as 18% gray (0 = off), so it
    /// follows the scene whether a sky or an emitter lights it. Biases bright indirect paths
    /// down; kills the speckle Ray Reconstruction would otherwise smear.
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
    /// Added to the ray-cone texture level of detail, on top of the automatic
    /// log2(render / output) term. The DLSS guide asks for -1 (sharper: accumulation over
    /// jittered frames resolves the extra detail); 0 is the unbiased footprint, positive blurs.
    #[reflect(@-3.0..=3.0_f32)]
    pub texture_lod_bias: f32,
    /// Ray Reconstruction's specular hit distance comes from one extra mirror-direction ray
    /// at the primary vertex, for surfaces up to this perceptual roughness; rougher ones
    /// report 0 (the reflection moves with the surface). 0 = no guide rays.
    #[reflect(@0.0..=1.0_f32)]
    pub spec_hit_roughness: f32,
    /// Light candidates resampled at every shading point (RIS): each is drawn from the
    /// power-weighted table, weighted by what it would contribute HERE, one survives and
    /// gets the shadow ray. 1 = a single table sample, which with hundreds of lights almost
    /// always lands on one too far away to matter. Deeper bounces use a quarter.
    #[reflect(@1.0..=32.0_f32)]
    pub light_candidates: u32,
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
    /// Lock exposure to `ev100` instead of metering. The metering keeps running
    /// underneath either way -- it is what normalises Ray Reconstruction's input -- so
    /// this changes the LOOK only, instantly and at no cost to denoiser history.
    pub ev100_lock: bool,
    /// The locked exposure, EV100: the photographic stop, the same number
    /// aurora_files/lighting_units.md uses. ~14-16 daylight exterior, ~5-9 interior.
    /// Only read when `ev100_lock` is on.
    #[reflect(@0.0..=20.0_f32)]
    pub ev100: f32,
    /// Sub-pixel camera jitter amplitude: 1 = the full +-0.5 traced pixel, 0 = pixel centres
    /// every frame. Lower is calmer -- the raw guide views hop less (they are shown
    /// unresolved; at ultra-performance half a traced pixel is one and a half screen pixels)
    /// -- and gives Ray Reconstruction less sub-pixel coverage for anti-aliased edges and
    /// upscaled detail. NGX is always told the same scaled offset.
    #[reflect(@0.0..=1.0_f32)]
    pub jitter_scale: f32,
}

impl DevUIState {
    /// The defaults with `$AURORA_DEV_UI` applied: `field=value` pairs separated by commas
    /// (`sky_brightness=0,emissive_boost=1,restir=true`), for headless runs that cannot
    /// reach the panel. Unknown fields and unparsable values are logged and skipped.
    pub fn from_env() -> Self {
        let mut state = Self::default();
        let Ok(overrides) = std::env::var("AURORA_DEV_UI") else {
            return state;
        };
        for pair in overrides.split(',').filter(|pair| !pair.trim().is_empty()) {
            let applied = pair.split_once('=').is_some_and(|(name, value)| {
                let value = value.trim();
                let Some(field) = state.field_mut(name.trim()) else {
                    return false;
                };
                if let Some(field) = field.try_downcast_mut::<f32>() {
                    value.parse().map(|v| *field = v).is_ok()
                } else if let Some(field) = field.try_downcast_mut::<u32>() {
                    value.parse().map(|v| *field = v).is_ok()
                } else if let Some(field) = field.try_downcast_mut::<bool>() {
                    value.parse().map(|v| *field = v).is_ok()
                } else {
                    false
                }
            });
            if !applied {
                warn!("AURORA_DEV_UI: cannot apply `{pair}`");
            }
        }
        state
    }
}

impl Default for DevUIState {
    fn default() -> Self {
        Self {
            aperture: 0.0,
            foginess: 0.001,
            fog_scatter: 0.9,
            sky_brightness: 1.0,
            emissive_boost: 1.0,
            firefly_clamp: 8.0,
            samples: 1,
            max_bounces: 32,
            vignette: 0.0,
            light_nee: true,
            restir: false, // TODO
            texture_lod_bias: -1.0,
            spec_hit_roughness: 0.6,
            light_candidates: 8,
            restir_candidates: 8,
            restir_history: 20.0,
            sharc: false, // TODO
            sharc_voxel: 0.25,
            omm: true,
            dlss: AuroraDlss::from_env(),
            rr_preset: RrPreset::current(),
            ev100_lock: false,
            // Filament's indoor preset; matches AuroraExposure::INDOOR.
            ev100: 7.0,
            jitter_scale: 1.0,
        }
    }
}

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

/// The node the [`ProceduralSky`] inspector is built under (parked with the panel's sky
/// section; see `spawn_panel`).
#[allow(dead_code)]
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
        app.insert_resource(DevUIState::from_env());
        app.add_systems(Startup, spawn_panel);
        app.add_systems(
            Update,
            (
                toggle_panel,
                update_stats,
                sync_dlss_mode,
                sync_rr_preset,
                sync_exposure,
            ),
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

/// Keeps the panel's EV100 lock and the camera's [`AuroraExposure`] equal, in both
/// directions, so a lock set here shows up in the F1 inspector and vice versa.
///
/// The panel speaks EV100; `AuroraExposure` stores log2 of the radiance multiplier. The
/// two differ by `log2(1.2)` (Filament), which is why the presets are -15.26 rather than
/// -15 -- see [`ev_from_ev100`].
fn sync_exposure(
    mut state: ResMut<DevUIState>,
    mut cameras: Query<&mut AuroraExposure, With<Camera3d>>,
    mut agreed: Local<Option<(bool, f32)>>,
) {
    let want = (state.ev100_lock, state.ev100);
    let Some(last) = *agreed else {
        // First run. The camera starts on Auto while the panel may already say "locked"
        // (DevUIState::from_env), so seeding `agreed` from the PANEL would make the two
        // look agreed, and the pull-back branch below would then quietly clobber the lock
        // off. The panel is authoritative on frame one; push it out.
        for mut exposure in &mut cameras {
            *exposure = if want.0 {
                AuroraExposure::fixed(ev_from_ev100(want.1))
            } else {
                AuroraExposure::default()
            };
        }
        *agreed = Some(want);
        return;
    };

    if want != last {
        // Panel moved: push it out.
        for mut exposure in &mut cameras {
            *exposure = if want.0 {
                AuroraExposure::fixed(ev_from_ev100(want.1))
            } else {
                AuroraExposure::default()
            };
        }
        *agreed = Some(want);
        return;
    }

    // Component moved (F1 inspector, or an app setting it): pull it back.
    if let Some(exposure) = cameras.iter().next() {
        let now = match exposure {
            AuroraExposure::Fixed(fixed) => (true, ev100_from_ev(fixed.ev)),
            AuroraExposure::Auto(_) => (false, state.ev100),
        };
        if now.0 != state.ev100_lock || (now.0 && (now.1 - state.ev100).abs() > 1.0e-3) {
            state.ev100_lock = now.0;
            state.ev100 = now.1;
            *agreed = Some(now);
            return;
        }
    }
    *agreed = Some(want);
}

/// Applies the panel's preset row; the renderer rebuilds the feature on the next frame.
fn sync_rr_preset(state: Res<DevUIState>) {
    if state.rr_preset != RrPreset::current() {
        state.rr_preset.make_current();
    }
    set_jitter_scale(state.jitter_scale);
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
                (caption("centre: -") DevUIProbe),
                (
                    Node {
                        flex_direction: FlexDirection::Column,
                        align_self: AlignSelf::Stretch,
                    }
                    DevUIInspectorHost
                ),
                // The procedural-sky section is parked while the examples run HDR skies;
                // uncomment (with the inspector block below) to get it back.
                // caption("sky (procedural)"),
                // (
                //     Node {
                //         flex_direction: FlexDirection::Column,
                //         align_self: AlignSelf::Stretch,
                //     }
                //     DevUISkyHost
                // ),
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
    // Parked with the panel section above.
    // let sky_host = world
    //     .query_filtered::<Entity, With<DevUISkyHost>>()
    //     .iter(world)
    //     .find(|_| true)
    //     .unwrap_or(panel);
    // build_resource_inspector(world, TypeId::of::<ProceduralSky>(), sky_host);
    let _ = TypeId::of::<ProceduralSky>();
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
        AuroraExposure::Fixed(fixed) => (ev100_from_ev(fixed.ev), true),
        AuroraExposure::Auto(_) => (f32::NAN, false),
    });
    for mut text in &mut stats {
        text.0 = format!("fps: {:>6.1}", *fps_avg);
    }
    // Both halves padded to a FIXED width, and the whole line kept inside the panel's
    // 340 px: a line that grows as the numbers change digits relayouts the panel every
    // frame, which reads as a flicker.
    let exposure = match ev100 {
        Some((ev, true)) => format!("EV100 {ev:.1}L", ev = ev),
        Some((_, false)) => "EV100 auto".to_string(),
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
