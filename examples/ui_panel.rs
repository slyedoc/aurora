//! **`bevy_ui` on a world quad — the UI-surface lane.**
//!
//! A ray-traced scene where a floating panel's EMISSIVE TEXTURE is a live `bevy_ui` tree:
//! world-space UI that is a real light source (the XR panel posture — the window HUD doesn't
//! exist in a headset, a glowing quad does).
//!
//! ```text
//!   app                  a placeholder Image (data: None) + Camera { RenderTarget::Image }
//!   sync_ui_surfaces     camera scan → target_info = texture size (taffy lays out at 1024x576)
//!   extract_ui           per-node routing: panel quads → surface bucket, HUD → window
//!   prepare_ui_surfaces  B8G8R8A8 COLOR_ATTACHMENT|SAMPLED target, parked in VulkanAssets<Image>
//!   draw_ui_surfaces     rasterized before the trace — the rays sample this frame's UI
//!   RT trace             the panel's material samples the target as its emissive texture
//! ```
//!
//! Run `AUTO_SCREENSHOT_MS=6000 cargo run --example ui_panel` and check `target/tmp/`:
//! the panel must show the rounded card with heading + live uptime counter (not gray, not
//! garbage), its glow must tint the ground, and the top-left window HUD must coexist.

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{
        controls::{
            ButtonVariant, FeathersButton, FeathersCheckbox, FeathersSlider, FeathersToggleSwitch,
        },
        display::caption,
        rounded_corners::RoundedCorners,
    },
    prelude::*,
    ui::Checked,
    ui_widgets::{Activate, SliderValue, ValueChange},
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::{DevUIPlugin, DevUIState},
    material::{AuroraMaterial, AuroraMaterial3d},
    AuroraDefaultPlugins,
    sky::Sky,
    sphere::Sphere as RtSphere,
    ui_panel::{InspectorPanel3d, UiPanel3d, UiPanel3dRoot},
    util::{ScreenshotExt, TimeoutAppExt},
};

/// The offscreen UI target's resolution: 16:9, matching the quad's 2.4 x 1.35 world aspect
/// so a UI pixel stays square on the mesh.
const TARGET_SIZE: UVec2 = UVec2::new(1024, 576);

/// Panel emission in nits — a bright indoor display against the dim gray sky below.
const PANEL_NITS: f32 = 4000.0;

fn main() {
    App::new()
        .add_plugins((
            AuroraDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (populate_panel, tick_counter))
        // The widgets are headless: they EMIT events, the app applies the state. These three
        // observers make the panel's controls live (and prove clicks arrive from the world).
        .add_observer(|activate: On<Activate>| {
            info!("panel: {:?} activated", activate.entity);
        })
        .add_observer(|change: On<ValueChange<f32>>, mut commands: Commands| {
            // SliderValue is an immutable component: replace it.
            commands
                .entity(change.source)
                .insert(SliderValue(change.value));
        })
        .add_observer(|change: On<ValueChange<bool>>, mut commands: Commands| {
            if change.value {
                commands.entity(change.source).insert(Checked);
            } else {
                commands.entity(change.source).remove::<Checked>();
            }
        })
        .add_screenshot(KeyCode::F12)
        .add_timeout_exit(None, 12.0)
        .run();
}

/// The live element on the panel: proof the surface re-renders per frame.
#[derive(Component, Default, Clone)]
struct UptimeText;

/// The hand-built panel; its UI tree is spawned once the panel's root exists.
#[derive(Component)]
struct DemoPanel;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    // Dim overcast void: bright enough to see the scene, dim enough that the panel's glow
    // reads on the ground.
    commands.insert_resource(Sky::Color {
        radiance: Vec3::splat(250.0),
    });

    // ---- the scene --------------------------------------------------------------------
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(30.0, 30.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.45, 0.45, 0.48),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    // THE PANEL: one component. `UiPanel3d` mints the offscreen target, the routing camera
    // (UI roots pointed at it lay out at the TEXTURE's resolution), the flipped-V quad with
    // the target as its emissive texture, and the `UiSurfacePanel` every pointer source
    // (window mouse, VR controller) operates. The surface stores straight-alpha sRGB with a
    // transparent background, so only lit UI pixels emit.
    commands.spawn((
        Name::new("panel"),
        DemoPanel,
        UiPanel3d {
            size: Vec2::new(2.4, 1.35),
            px: TARGET_SIZE,
            nits: PANEL_NITS,
            thickness: 0.06,
            // The root below styles itself.
            opaque: false,
            ..default()
        },
        Transform::from_xyz(0.0, 1.6, 0.0),
    ));

    // A second panel: the renderer's dev tunables as a feathers inspector, built from the
    // reflected resource with no widget code at all (the XR wrist-panel posture).
    commands.spawn((
        Name::new("inspector panel"),
        InspectorPanel3d::resource::<DevUIState>(),
        UiPanel3d {
            size: Vec2::new(1.2, 1.6),
            px: UVec2::new(600, 800),
            scale: 1.4,
            nits: PANEL_NITS,
            ..default()
        },
        Transform::from_xyz(-1.4, 1.5, 1.8).with_rotation(Quat::from_rotation_y(0.45)),
    ));

    // A chrome sphere beside the panel: its reflection shows the UI too.
    commands.spawn((
        Transform::from_xyz(2.4, 0.8, -0.6).with_scale(Vec3::splat(1.6)),
        RtSphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.75, 0.78, 0.80),
            perceptual_roughness: 0.05,
            metallic: 1.0,
            ..default()
        })),
    ));

    commands.spawn((
        Camera3d::default(),
        FreeCamera::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0_f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(0.6, 2.0, 4.2).looking_at(Vec3::new(0.0, 1.5, 0.0), Vec3::Y),
    ));

    // ---- the window HUD (screen lane, unchanged) ---------------------------------------
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(16.0),
            bottom: Val::Px(12.0),
            ..default()
        },
        Text::new("screen lane: this text is ON THE GLASS, not the panel"),
        TextFont::from_font_size(18.0),
        TextColor(Color::WHITE),
    ));
}

/// The demo panel's UI (offscreen), spawned under the root `UiPanel3d` built for it.
fn populate_panel(
    mut commands: Commands,
    fresh: Query<&UiPanel3dRoot, (Added<UiPanel3dRoot>, With<DemoPanel>)>,
) {
    let Some(built) = fresh.iter().next() else {
        return;
    };
    commands
        .spawn_scene(bsn! {
            Node {
                width: percent(100),
                height: percent(100),
                padding: UiRect::all(px(36)),
                flex_direction: FlexDirection::Column,
                row_gap: px(18),
                border: UiRect::all(px(6)),
                border_radius: BorderRadius::all(px(30)),
            }
            BackgroundColor(Color::srgba(0.05, 0.08, 0.14, 0.96))
            BorderColor::all(Color::srgb(0.35, 0.62, 0.95))
            Children [
                (
                    Text("[ Aurora ]")
                    TextFont { font_size: FontSize::Px(72.0) }
                    TextColor(Color::srgb(0.65, 0.85, 1.0))
                ),
                (
                    Text("UI surface lane armed.")
                    TextFont { font_size: FontSize::Px(40.0) }
                    TextColor(Color::WHITE)
                ),
                (
                    Text("uptime 0.0 s")
                    TextFont { font_size: FontSize::Px(40.0) }
                    TextColor(Color::srgb(0.55, 0.95, 0.65))
                    UptimeText
                ),
                // A row of REAL feathers widgets on the same surface: same theme, same
                // extraction as the window lane, and the mouse operates them through the
                // panel's pointer bridge.
                (
                    Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        column_gap: px(18),
                        margin: UiRect::top(px(12)),
                    }
                    Children [
                        (
                            @FeathersButton {
                                @caption: bsn! { caption("Engage") },
                                @variant: ButtonVariant::Primary,
                                @corners: RoundedCorners::All,
                            }
                        ),
                        (
                            @FeathersCheckbox {
                                @caption: bsn! { caption("shields") },
                            }
                        ),
                        (@FeathersToggleSwitch),
                        (
                            Node {
                                width: px(260),
                            }
                            Children [
                                (
                                    @FeathersSlider {
                                        @min: 0.0,
                                        @max: 100.0,
                                    }
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        })
        .insert(ChildOf(built.root));
}

/// One text write per ~100 ms — visibly live without re-shaping glyphs every frame.
fn tick_counter(
    time: Res<Time>,
    mut text: Query<&mut Text, With<UptimeText>>,
    mut last: Local<f32>,
) {
    let now = time.elapsed_secs();
    if now - *last < 0.1 {
        return;
    }
    *last = now;
    for mut text in &mut text {
        text.0 = format!("uptime {now:.1} s");
    }
}
