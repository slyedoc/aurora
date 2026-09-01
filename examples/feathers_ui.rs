//! bevy_feathers widgets drawn by this crate's Vulkan backend (no bevy_render / wgpu in the app).
//!
//! The UI stack is `bevy_feathers_core`: bevy lays the nodes out, `ui_render` rasterizes them.

use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::{
    feathers::{
        controls::{FeathersButton, FeathersCheckbox, FeathersSlider, FeathersToggleSwitch},
        display::caption,
        theme::ThemeBackgroundColor,
        tokens,
    },
    prelude::*,
    ui::Checked,
    ui_widgets::{Activate, SliderValue, ValueChange},
};
use bevy_aurora::{
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    material::{AuroraMaterial, AuroraMaterial3d},
    ray_default_plugins::RayDefaultPlugins,
    sphere::Sphere,
};

#[derive(Resource, Default)]
struct Counter(i32);

#[derive(Component, Default, Clone)]
struct CounterText;

fn main() {
    let mut app = App::new();
    app.add_plugins(RayDefaultPlugins);
    app.add_plugins(DevShaderPlugin);
    // DevUIPlugin brings the UI stack (UiRenderPlugin, FeathersPlugins) and the dark theme.
    app.add_plugins(DevUIPlugin);
    app.add_plugins(FreeCameraPlugin::default());
    app.init_resource::<Counter>();
    app.add_systems(Startup, (setup, panel.spawn()));
    app.add_systems(
        Update,
        update_counter_text.run_if(resource_changed::<Counter>),
    );
    app.run();
}

fn setup(
    mut commands: Commands,
    mut windows: Query<&mut Window>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    let mut window = windows.single_mut().unwrap();
    window.resolution.set_scale_factor_override(Some(1.0));
    window.resolution.set(1600.0, 900.0);

    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: 60.0 * std::f32::consts::PI / 180.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 1.5, 6.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        FreeCamera::default(),
    ));

    for (i, color) in [
        Color::srgb(0.9, 0.2, 0.2),
        Color::srgb(0.2, 0.9, 0.2),
        Color::srgb(0.2, 0.2, 0.9),
    ]
    .into_iter()
    .enumerate()
    {
        commands.spawn((
            Transform::from_xyz(i as f32 * 2.5 - 2.5, 1.0, 0.0),
            Sphere,
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: color,
                perceptual_roughness: 0.3,
                ..default()
            })),
        ));
    }

    // glowing sphere
    commands.spawn((
        Transform::from_xyz(0.0, 6.0, -2.0).with_scale(Vec3::splat(2.0)),
        Sphere,
        AuroraMaterial3d(materials.add(AuroraMaterial {
            base_color: Color::WHITE,
            emissive: LinearRgba::new(10.0, 9.0, 8.0, 1.0),
            ..default()
        })),
    ));
}

fn panel() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            right: px(16),
            top: px(16),
            width: px(320),
            padding: UiRect::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            border_radius: BorderRadius::all(px(6)),
        }
        ThemeBackgroundColor(tokens::WINDOW_BG)
        Children [
            caption("feathers on raw Vulkan"),
            (
                @FeathersButton
                on(|_: On<Activate>, mut counter: ResMut<Counter>| {
                    counter.0 += 1;
                })
                Children [ caption("Click me") ]
            ),
            (caption("clicks: 0") CounterText),
            (
                @FeathersCheckbox {
                    @caption: bsn! { caption("Checkbox") }
                }
                Checked
                on(|change: On<ValueChange<bool>>| {
                    info!("checkbox -> {}", change.value);
                })
            ),
            (
                @FeathersToggleSwitch
                on(|change: On<ValueChange<bool>>| {
                    info!("toggle -> {}", change.value);
                })
            ),
            (
                @FeathersSlider {
                    @max: 100.0,
                }
                SliderValue(35.0)
                on(|change: On<ValueChange<f32>>| {
                    info!("slider -> {}", change.value);
                })
            ),
        ]
    }
}

fn update_counter_text(
    counter: Res<Counter>,
    mut counter_text: Single<&mut Text, With<CounterText>>,
) {
    counter_text.0 = format!("clicks: {}", counter.0);
}
