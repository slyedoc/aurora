//! `Sky::Generated` on the GPU: noon is blue overhead, a low sun reddens the horizon under it,
//! and with the sun well below the horizon the sky is dark.

use bevy::{camera::RenderTarget, prelude::*, winit::WinitPlugin};
use bevy_aurora::{
    AuroraDefaultPlugins,
    camera_target::camera_target_placeholder,
    environment::{Environment, RenderEnvironments},
    render_device::RenderDevice,
    sky::{Sky, SkyAir},
    sky_gen::SkyGen,
    sky_importance::{ENV_H, ENV_W},
};

/// The texel looking along `d` (rt_types.slang `env_dir_to_uv`).
fn texel(sky: &[Vec4], d: Vec3) -> Vec3 {
    let d = d.normalize();
    let phi = d.x.atan2(d.z);
    let mut u = (if phi > 0.0 { phi } else { phi + std::f32::consts::TAU })
        / std::f32::consts::TAU
        - 0.5;
    if u < 0.0 {
        u += 1.0;
    }
    let v = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    let x = ((u * ENV_W as f32) as u32).min(ENV_W - 1);
    let y = ((v * ENV_H as f32) as u32).min(ENV_H - 1);
    sky[(y * ENV_W + x) as usize].truncate()
}

fn luma(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The light direction of a sun `elevation` degrees up in the +X direction.
fn sun_at(elevation: f32) -> Transform {
    let e = elevation.to_radians();
    let towards = Vec3::new(e.cos(), e.sin(), 0.0);
    Transform::default().looking_to(-towards, Vec3::Z)
}

fn render(app: &mut App, environment: Entity, sun: Entity, elevation: f32) -> Vec<Vec4> {
    app.world_mut()
        .entity_mut(sun)
        .insert(sun_at(elevation));
    // Until the kernels are compiled and have run, then a few more frames.
    let start = std::time::Instant::now();
    let bit = loop {
        app.update();
        let bit = app
            .world()
            .resource::<RenderEnvironments>()
            .bit(environment)
            .map(usize::from);
        if let Some(bit) = bit
            && app.world().resource::<SkyGen>().sky(bit).is_some()
        {
            break bit;
        }
        assert!(start.elapsed().as_secs() < 30, "the generated sky was never built");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    for _ in 0..4 {
        app.update();
    }
    let world = app.world();
    world
        .resource::<SkyGen>()
        .read_back(world.resource::<RenderDevice>(), bit)
        .expect("the generated sky was built")
}

#[test]
fn the_generated_sky_is_blue_at_noon_red_at_sunset_and_dark_at_night() {
    let mut app = App::new();
    app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    app.finish();
    app.cleanup();

    let environment = app
        .world_mut()
        .spawn((Environment::default(), Sky::Generated, SkyAir::default()))
        .id();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(64, 64)));
    app.world_mut().spawn((
        ChildOf(environment),
        Camera3d::default(),
        RenderTarget::Image(target.into()),
        Transform::default(),
    ));
    let sun = app
        .world_mut()
        .spawn((
            ChildOf(environment),
            DirectionalLight {
                illuminance: 100_000.0,
                ..default()
            },
            sun_at(60.0),
        ))
        .id();

    let noon = render(&mut app, environment, sun, 60.0);
    let zenith = texel(&noon, Vec3::Y);
    let horizon_away = texel(&noon, Vec3::new(-1.0, 0.05, 0.0));
    println!("noon: zenith {zenith}, horizon {horizon_away}");
    assert!(zenith.z > zenith.x * 1.5, "a blue zenith: {zenith}");
    assert!(luma(zenith) > 500.0 && luma(zenith) < 50_000.0, "{zenith}");
    assert!(
        horizon_away.x / horizon_away.z > zenith.x / zenith.z,
        "the horizon paler than the zenith: {horizon_away} vs {zenith}"
    );

    let sunset = render(&mut app, environment, sun, 2.0);
    let under_sun = texel(&sunset, Vec3::new(1.0, 0.03, 0.0));
    let zenith = texel(&sunset, Vec3::Y);
    println!("sunset: under the sun {under_sun}, zenith {zenith}");
    assert!(under_sun.x > under_sun.z, "a red horizon under the sun: {under_sun}");
    assert!(luma(zenith) < luma(texel(&noon, Vec3::Y)), "dimmer than noon");

    let night = render(&mut app, environment, sun, -20.0);
    let brightest = night.iter().map(|t| luma(t.truncate())).fold(0.0, f32::max);
    println!("night: brightest {brightest}");
    assert!(brightest < 1.0, "a dark night: {brightest}");

    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
}
