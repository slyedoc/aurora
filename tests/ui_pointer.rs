//! A world UI panel operated by a ray: a `UiPointerSource` aimed at a panel's node hovers and
//! clicks it, the way a VR controller's aim pose or the window mouse does.

use bevy::{
    camera::RenderTarget,
    picking::events::{PointerClick, PointerOver},
    prelude::*,
    winit::WinitPlugin,
};
use bevy::{ui::UiGlobalTransform, ui_widgets::Checkbox};
use bevy_aurora::{
    AuroraDefaultPlugins,
    camera_target::camera_target_placeholder,
    dev_ui::DevUIPlugin,
    environment::Environment,
    ui_panel::{InspectorPanel3d, UiPanel3d, UiPanel3dRoot},
    ui_render::UiPointerSource,
};

#[derive(Resource, Default)]
struct Seen {
    over: u32,
    clicks: u32,
}

fn populate(mut commands: Commands, fresh: Query<&UiPanel3dRoot, Added<UiPanel3dRoot>>) {
    for built in &fresh {
        commands
            .spawn((
                Node {
                    width: percent(100),
                    height: percent(100),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.2, 0.4, 0.8)),
                ChildOf(built.root),
            ))
            .observe(|_: On<PointerOver>, mut seen: ResMut<Seen>| seen.over += 1)
            .observe(|_: On<PointerClick>, mut seen: ResMut<Seen>| seen.clicks += 1);
    }
}

#[test]
fn a_ray_clicks_a_world_panel() {
    let mut app = App::new();
    app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>())
        .init_resource::<Seen>()
        .add_systems(Update, populate);
    app.finish();
    app.cleanup();

    let environment = app.world_mut().spawn(Environment::default()).id();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(320, 180)));
    app.world_mut().spawn((
        ChildOf(environment),
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::from_xyz(0.0, 0.0, 3.0),
    ));
    app.world_mut().spawn((
        ChildOf(environment),
        UiPanel3d {
            size: Vec2::new(1.0, 0.5),
            px: UVec2::new(400, 200),
            ..default()
        },
        Transform::default(),
    ));
    // Two metres in front of the panel, pointing at its middle.
    let ray = app
        .world_mut()
        .spawn((
            UiPointerSource::default(),
            Transform::from_xyz(0.1, 0.05, 2.0),
        ))
        .id();

    step(&mut app, ray, false, 60);
    step(&mut app, ray, true, 3);
    step(&mut app, ray, false, 10);

    let hit = app.world().get::<UiPointerSource>(ray).unwrap().hit;
    let seen = app.world().resource::<Seen>();
    let (over, clicks) = (seen.over, seen.clicks);
    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
    assert!(hit.is_some(), "the ray lands on the panel");
    assert!(over > 0, "the node under the ray is hovered");
    assert_eq!(clicks, 1, "a press and release on it is one click");
}

#[derive(Resource, Reflect, Default)]
#[reflect(Resource)]
struct Probe {
    flag: bool,
}

/// Steps `frames` updates with the ray's primary button `down`.
fn step(app: &mut App, ray: Entity, down: bool, frames: usize) {
    app.world_mut()
        .get_mut::<UiPointerSource>(ray)
        .unwrap()
        .buttons[0] = down;
    for _ in 0..frames {
        app.update();
    }
}

#[test]
fn a_ray_ticks_an_inspector_checkbox_on_a_world_panel() {
    let mut app = App::new();
    app.add_plugins((
        AuroraDefaultPlugins.build().disable::<WinitPlugin>(),
        DevUIPlugin,
    ))
    .init_resource::<Probe>()
    .register_type::<Probe>();
    app.finish();
    app.cleanup();

    let environment = app.world_mut().spawn(Environment::default()).id();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(320, 180)));
    app.world_mut().spawn((
        ChildOf(environment),
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::from_xyz(0.0, 0.0, 3.0),
    ));
    let (size, px) = (Vec2::new(1.0, 0.8), UVec2::new(500, 400));
    app.world_mut().spawn((
        ChildOf(environment),
        InspectorPanel3d::resource::<Probe>(),
        UiPanel3d {
            size,
            px,
            // Not 1: pointers must land in the panel camera's logical pixels.
            scale: 1.6,
            ..default()
        },
        Transform::default(),
    ));
    let ray = app
        .world_mut()
        .spawn((
            UiPointerSource::default(),
            Transform::from_xyz(0.0, 0.0, 2.0),
        ))
        .id();
    step(&mut app, ray, false, 60);

    // The field's checkbox on this panel (the lowest: a section header may carry its own),
    // centre in target pixels, as a point on the panel.
    let root = app
        .world_mut()
        .query::<&UiPanel3dRoot>()
        .single(app.world())
        .unwrap()
        .root;
    let world = app.world();
    let mut stack = vec![root];
    let mut centres = Vec::new();
    while let Some(e) = stack.pop() {
        if world.get::<Checkbox>(e).is_some() {
            centres.extend(world.get::<UiGlobalTransform>(e).map(|t| t.translation));
        }
        stack.extend(world.get::<Children>(e).into_iter().flatten());
    }
    let centre = centres
        .into_iter()
        .max_by(|a, b| a.y.total_cmp(&b.y))
        .expect("the inspector built a checkbox for `flag`");
    let uv = centre / px.as_vec2();
    let on_panel = Vec3::new((uv.x - 0.5) * size.x, (0.5 - uv.y) * size.y, 2.0);
    app.world_mut()
        .entity_mut(ray)
        .insert(Transform::from_translation(on_panel));
    step(&mut app, ray, false, 5);
    step(&mut app, ray, true, 3);
    step(&mut app, ray, false, 10);

    let flag = app.world().resource::<Probe>().flag;
    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
    assert!(flag, "clicking the checkbox writes the resource");
}
