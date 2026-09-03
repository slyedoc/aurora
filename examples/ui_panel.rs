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
    camera::{Camera, RenderTarget},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    feathers::{
        controls::{
            ButtonBundleProps, ButtonVariant, FeathersSliderProps, button_bundle,
            checkbox_bundle, slider_bundle, toggle_switch_bundle,
        },
        rounded_corners::RoundedCorners,
    },
    prelude::*,
    ui::{Checked, UiTargetCamera},
    ui_widgets::{Activate, SliderValue, ValueChange},
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    sky::Sky,
    sphere::Sphere as RtSphere,
    ui_render::{UiSurfacePanel, ui_target_placeholder},
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
            RayDefaultPlugins,
            DevShaderPlugin,
            DevUIPlugin,
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, tick_counter)
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
#[derive(Component)]
struct UptimeText;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Dim overcast void: bright enough to see the scene, dim enough that the panel's glow
    // reads on the ground.
    commands.insert_resource(Sky::Color {
        radiance: Vec3::splat(250.0),
    });

    // ---- the offscreen UI target ------------------------------------------------------
    let target = images.add(ui_target_placeholder(TARGET_SIZE));

    // The routing/layout anchor: a camera whose render target is the image. It renders
    // nothing itself — UI roots pointed at it lay out at the TEXTURE's resolution.
    let panel_camera = commands
        .spawn((
            Name::new("panel surface"),
            Camera::default(),
            RenderTarget::Image(target.clone().into()),
        ))
        .id();

    // ---- the scene --------------------------------------------------------------------
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(30.0, 30.0))),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::srgb(0.45, 0.45, 0.48),
            perceptual_roughness: 0.9,
            ..default()
        })),
    ));

    // THE PANEL: the UI texture as emissive. The surface stores straight-alpha sRGB with a
    // transparent background, so only lit UI pixels emit.
    // The UI target follows image convention (v=0 at the TOP); bevy's `Cuboid` maps its +Z
    // face with v=0 at the bottom — flip V or the card reads upside down.
    let mut panel: Mesh = Mesh::from(Cuboid::new(2.4, 1.35, 0.06));
    if let Some(bevy::mesh::VertexAttributeValues::Float32x2(uvs)) =
        panel.attribute_mut(Mesh::ATTRIBUTE_UV_0)
    {
        for uv in uvs.iter_mut() {
            uv[1] = 1.0 - uv[1];
        }
    }
    commands.spawn((
        Name::new("panel"),
        // The mouse bridge: cursor ray -> this plane -> texture-space pointer. Hover,
        // click, and drag on the panel's widgets work like window UI.
        UiSurfacePanel {
            target: target.clone(),
            size: Vec2::new(2.4, 1.35),
        },
        Mesh3d(meshes.add(panel)),
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::BLACK,
            emissive: LinearRgba::WHITE * PANEL_NITS,
            emissive_texture: Some(target),
            ..default()
        })),
        Transform::from_xyz(0.0, 1.6, 0.0),
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

    // ---- the panel's UI (offscreen) ----------------------------------------------------
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            padding: UiRect::all(Val::Px(36.0)),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(18.0),
            border: UiRect::all(Val::Px(6.0)),
            border_radius: BorderRadius::all(Val::Px(30.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.05, 0.08, 0.14, 0.96)),
        BorderColor::all(Color::srgb(0.35, 0.62, 0.95)),
        UiTargetCamera(panel_camera),
        children![
            (
                Text::new("[ Aurora ]"),
                TextFont::from_font_size(72.0),
                TextColor(Color::srgb(0.65, 0.85, 1.0)),
            ),
            (
                Text::new("UI surface lane armed."),
                TextFont::from_font_size(40.0),
                TextColor(Color::WHITE),
            ),
            (
                Text::new("uptime 0.0 s"),
                TextFont::from_font_size(40.0),
                TextColor(Color::srgb(0.55, 0.95, 0.65)),
                UptimeText,
            ),
        ],
    ))
    // A row of REAL feathers widgets on the same surface: same theme, same extraction as
    // the window lane. (They render; pointer interaction on world panels needs the
    // ray->UV pointer bridge, not yet ported.)
    .with_child({
        #[allow(deprecated)]
        (
            Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(18.0),
                margin: UiRect::top(Val::Px(12.0)),
                ..default()
            },
            children![
                button_bundle(
                    ButtonBundleProps {
                        variant: ButtonVariant::Primary,
                        corners: RoundedCorners::All,
                    },
                    (),
                    Spawn((Text::new("Engage"),)),
                ),
                checkbox_bundle((), Spawn((Text::new("shields"),))),
                toggle_switch_bundle(()),
                (
                    Node {
                        width: Val::Px(260.0),
                        ..default()
                    },
                    children![slider_bundle(
                        FeathersSliderProps {
                            min: 0.0,
                            max: 100.0,
                        },
                        (),
                    )],
                ),
            ],
        )
    });

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
