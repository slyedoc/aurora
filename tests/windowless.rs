//! The whole stack with no window: `AuroraDefaultPlugins` without `WinitPlugin` traces into image
//! targets and presents nothing.

use bevy::{camera::RenderTarget, prelude::*, winit::WinitPlugin};
use bevy_aurora::{
    AuroraDefaultPlugins, camera_target::CameraTargets, camera_target::camera_target_placeholder,
    frame_sync::FrameSync, sphere::Sphere,
};

#[test]
fn a_windowless_app_renders_its_image_targets() {
    let mut app = App::new();
    app.add_plugins(AuroraDefaultPlugins.build().disable::<WinitPlugin>());
    app.finish();
    app.cleanup();

    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(UVec2::new(320, 180)));
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(target.clone().into()),
        Transform::from_xyz(0.0, 1.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    app.world_mut().spawn((Sphere, Transform::default()));

    for _ in 0..120 {
        app.update();
    }

    assert!(
        app.world().resource::<FrameSync>().frame_count >= 100,
        "every update submits a frame"
    );
    assert!(
        app.world()
            .resource::<CameraTargets>()
            .map
            .contains_key(&target.id()),
        "the camera's image target was built"
    );
    assert!(
        app.world()
            .get_resource::<bevy_aurora::swapchain::Swapchain>()
            .is_none(),
        "and no swapchain"
    );

    // Dropped without exiting, so the renderer is torn down first.
    bevy_aurora::ray_render_plugin::shutdown_renderer(app.world_mut());
}
