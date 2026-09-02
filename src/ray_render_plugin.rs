use bevy::{
    app::AppExit,
    ecs::{message::Messages, schedule::ScheduleLabel},
    prelude::*,
    window::{RawHandleWrapperHolder, WindowCloseRequested},
    winit::DisplayHandleWrapper,
};
use raw_window_handle::HasDisplayHandle;

use ash::vk;

use crate::sky::{ProceduralSky, Sky};
use crate::{
    bluenoise_plugin::BlueNoiseBuffer,
    post_process_filter::PostProcessFilter,
    raytracing_pipeline::{RaytracingPipeline, RaytracingPushConstants},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_env::WHITE_TEXTURE_IDX,
    sbt::SBT,
    tlas_builder::TLAS,
    util::screenshot::ScreenshotRequests,
    vk_utils,
    vulkan_asset::VulkanAssets,
};

#[derive(Resource, Clone, Default)]
pub struct RenderConfig {
    pub rtx_pipeline: Handle<RaytracingPipeline>,
    pub postprocess_pipeline: Handle<PostProcessFilter>,
    pub pull_focus: Option<(u32, u32)>,
}

#[repr(C)]
pub struct UniformData {
    sky_color: Vec4,
    inverse_view: Mat4,
    inverse_projection: Mat4,
    pull_focus_x: u32,
    pull_focus_y: u32,
    gamma: f32,
    aperture: f32,
    foginess: f32,
    fog_scatter: f32,
    sky_brightness: f32,
    // Flat arrays, not `Mat4`: the GLSL side is `scalar` layout (4-byte aligned), and glam's
    // 16-byte alignment would pad here.
    view: [f32; 16],
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    jitter: [f32; 2],
    /// Free-running frame counter: the RNG seed, so every frame's noise is new for the
    /// temporal denoiser.
    frame: u32,
    /// Indirect path contributions are clamped to this luminance (0 = off).
    radiance_clamp: f32,
    /// Paths per pixel this frame and their maximum length (from [`DevUIState`]).
    samples: u32,
    max_bounces: u32,
    /// Post-process vignette strength (0 = off).
    vignette: f32,
    /// Sky source: 0 flat colour, 1 equirect HDR (`sky_color` = scale), 2 procedural.
    sky_mode: u32,
    sun_cos_radius: f32,
    sun_direction: [f32; 3],
    sun_radiance: [f32; 3],
    sky_zenith: [f32; 3],
    sky_horizon: [f32; 3],
    sky_ground: [f32; 3],
    /// Entries in the emissive-triangle light table (0 = no light NEE / MIS).
    light_entries: u32,
    /// ReSTIR DI initial candidates per pixel (0 = plain 1-sample NEE at the primary vertex).
    restir_candidates: u32,
    /// Light-table generation; reservoirs from another generation are dropped.
    light_epoch: u32,
    /// Cap on temporal history, in candidate-samples.
    restir_m_clamp: f32,
    /// Radiance cache: 0 = off (also while accumulating).
    sharc: u32,
    /// Base cache voxel size (meters) at the camera.
    sharc_voxel: f32,
    /// Environment importance sampling (env_light.rs): CDF + pdf buffer, 0 = none.
    env: u64,
    env_w: u32,
    env_h: u32,
    /// Uniform multiplier on every light's emission (from [`DevUIState`]).
    emissive_boost: f32,
}

#[repr(C)]
pub struct FocusData {
    focal_distance: f32,
}

/// One rendered view this frame: XR renders one per eye, flat renders one for the window.
/// `slot` indexes the per-view resources (uniform buffer, DLSS feature, reservoirs,
/// descriptor sets).
struct ViewFrame {
    slot: usize,
    inverse_view: Mat4,
    view_matrix: Mat4,
    projection: Mat4,
    view_proj: Mat4,
    last_view_proj: Mat4,
    plan: Option<crate::dlss::DlssPlan>,
}

/// The renderer's teardown: every plugin that owns Vulkan objects destroys them here, and
/// [`on_shutdown`] (ordered last) drops the device. Run once by [`shutdown`].
#[derive(ScheduleLabel, PartialEq, Eq, Debug, Clone, Hash)]
pub struct TeardownSchedule;

/// The frame's stages, chained in [`Last`] after the simulation. One world: the systems in
/// [`RenderSet::Extract`] read the ECS directly (transform deltas, instance changes, UI quads,
/// asset events), [`RenderSet::Prepare`] consumes the asset worker's results, and
/// [`RenderSet::Render`] waits for the previous frame, records, and submits.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderSet {
    Shutdown,
    Extract,
    Prepare,
    Render,
}

pub struct RayRenderPlugin;

impl Plugin for RayRenderPlugin {
    fn build(&self, app: &mut App) {
        let display_handle = app.world().resource::<DisplayHandleWrapper>();
        // Present when XrPlugin (added before this plugin) brought a runtime up: instance
        // and device creation then go through it (XR_KHR_vulkan_enable2).
        let xr_context = app.world().get_resource::<crate::xr::XrContext>();
        let render_device = unsafe {
            crate::render_device::RenderDevice::from_display(
                &display_handle.0.display_handle().unwrap(),
                xr_context,
            )
        };
        let xr_state = xr_context.map(|context| {
            crate::xr::XrState::new(context, &render_device).expect("OpenXR session")
        });
        let sphere_blas = unsafe { crate::sphere::SphereBLAS::new(&render_device) };
        if let Some(xr_state) = xr_state {
            app.insert_resource(xr_state);
        }
        app.insert_resource(render_device);
        app.insert_resource(sphere_blas);
        app.init_resource::<Frame>();
        app.init_resource::<RenderConfig>();
        app.init_resource::<ScreenshotRequests>();

        let mut teardown = Schedule::new(TeardownSchedule);
        teardown.add_systems(on_shutdown);
        app.add_schedule(teardown);

        app.configure_sets(
            Last,
            (
                RenderSet::Shutdown,
                RenderSet::Extract.run_if(run_if_render_device_exists),
                RenderSet::Prepare.run_if(run_if_render_device_exists),
                RenderSet::Render.run_if(run_if_render_device_exists),
            )
                .chain(),
        );
        app.add_plugins(crate::sky::SkyPlugin);
        app.add_systems(Update, set_focus_pulling);
        app.add_systems(
            Last,
            (
                shutdown.in_set(RenderSet::Shutdown),
                update_render_window.in_set(RenderSet::Extract),
                render_frame.in_set(RenderSet::Render),
            ),
        );
    }
}

/// Tears the renderer down (queue idle, then [`TeardownSchedule`]) when the window is closing
/// or the app is exiting, and only then closes the window: the swapchain has to go before its
/// surface does. Messages are peeked, not consumed, so bevy's own exit handling still sees them.
fn shutdown(world: &mut World) {
    if !world.contains_resource::<RenderDevice>() {
        return;
    }
    let closing: Vec<Entity> = {
        let messages = world.resource::<Messages<WindowCloseRequested>>();
        messages
            .get_cursor()
            .read(messages)
            .map(|m| m.window)
            .collect()
    };
    let exiting = {
        let messages = world.resource::<Messages<AppExit>>();
        messages.get_cursor().read(messages).next().is_some()
    };
    if closing.is_empty() && !exiting {
        return;
    }
    log::info!("Shutting down the renderer");
    {
        let render_device = world.resource::<RenderDevice>();
        let queue = render_device.queue.lock().unwrap();
        unsafe { render_device.queue_wait_idle(*queue).unwrap() };
    }
    world.run_schedule(TeardownSchedule);
    for window in closing {
        if let Ok(entity) = world.get_entity_mut(window) {
            entity.despawn();
        }
    }
}

/// The primary window's size, as the frame and the swapchain see it.
#[derive(Resource)]
pub struct RenderWindow {
    pub width: u32,
    pub height: u32,
}

fn update_render_window(
    windows: Query<(&Window, &RawHandleWrapperHolder)>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    swapchain: Option<Res<crate::swapchain::Swapchain>>,
) {
    let Ok((window, handle_holder)) = windows.single() else {
        return;
    };

    // initialize the swapchain if it isn't already
    if swapchain.is_none() {
        let handle_holder = handle_holder.0.lock().unwrap();
        if let Some(handles) = &*handle_holder {
            commands.insert_resource(unsafe {
                crate::swapchain::Swapchain::from_window(render_device.clone(), &handles)
            });
        }
    }

    // Physical pixels: the swapchain is recreated whenever this changes (Wayland never
    // reports OUT_OF_DATE; the compositor would silently scale a stale-size image).
    commands.insert_resource(RenderWindow {
        width: window.resolution.physical_width().max(1),
        height: window.resolution.physical_height().max(1),
    });
}

fn set_focus_pulling(
    windows: Query<&Window>,
    mut render_config: ResMut<RenderConfig>,
    mouse: Res<ButtonInput<MouseButton>>,
) {
    // The window can already be gone on the app's last frames (close requested).
    let Ok(window) = windows.single() else {
        return;
    };
    render_config.pull_focus = None;

    if let Some(mouse_pos) = window.physical_cursor_position() {
        let x = mouse_pos.x as u32;
        let y = mouse_pos.y as u32;
        if mouse.pressed(MouseButton::Left) {
            render_config.pull_focus = Some((x, y));
        }
    }
}

#[derive(Resource, Default)]
pub struct Frame {
    pub swapchain_image: vk::Image,
    pub swapchain_view: vk::ImageView,
    /// One per rendered view (XR eyes; flat uses `[0]`): every trace dispatch in the frame
    /// needs its own matrices.
    pub uniform_buffers: [Buffer<UniformData>; 2],
    pub focus_data: Buffer<FocusData>,
}

fn render_frame(
    render_device: Res<crate::render_device::RenderDevice>,
    window: Res<RenderWindow>,
    swapchain: Option<ResMut<crate::swapchain::Swapchain>>,
    dev_ui_stuff: (
        Option<Res<crate::dev_ui::DevUIState>>,
        crate::ui_render::UiDrawParams,
        Res<Sky>,
        Res<ProceduralSky>,
        Res<crate::env_light::EnvLight>,
        crate::gizmo_render::GizmoDrawParams,
    ),
    mut frame: ResMut<Frame>,
    render_config: Res<RenderConfig>,
    rtx_pipelines: Res<VulkanAssets<RaytracingPipeline>>,
    textures: Res<VulkanAssets<bevy::prelude::Image>>,
    postprocess_filters: Res<VulkanAssets<PostProcessFilter>>,
    bluenoise_buffer: Res<BlueNoiseBuffer>,
    mut tlas: ResMut<TLAS>,
    gpu: (
        ResMut<crate::gpu_transform::GpuTransforms>,
        Res<crate::compute::ComputeModules>,
        Res<SBT>,
        ResMut<crate::lights::LightManager>,
        ResMut<crate::restir::RestirState>,
        ResMut<crate::sharc::SharcState>,
        ResMut<ScreenshotRequests>,
        ResMut<crate::auto_exposure::AutoExposureState>,
        Res<Time>,
        ResMut<crate::skinning::Skins>,
    ),
    camera: Query<
        (
            &Projection,
            &GlobalTransform,
            Option<&crate::dlss::AuroraDlss>,
            Option<&crate::auto_exposure::AuroraExposure>,
            Option<&crate::debug_view::AuroraDebugView>,
        ),
        With<Camera3d>,
    >,
    mut frame_counter: Local<u32>,
    dlss_stuff: (
        ResMut<crate::dlss::DlssState>,
        Local<[Option<Mat4>; 2]>,
        Local<bool>,
    ),
    mut xr: Option<ResMut<crate::xr::XrState>>,
) {
    let Some(mut swapchain) = swapchain else {
        return;
    };
    // XR frame pacing + swapchain image, when a session is live. None also covers the
    // compositor's "don't render" frames — the window path just runs alone then.
    let xr_frame = xr
        .as_deref_mut()
        .and_then(|s| s.begin_frame(&render_device));
    let (
        mut transforms,
        modules,
        sbt,
        mut lights,
        mut restir,
        mut sharc,
        mut screenshots,
        mut ae,
        time,
        mut skins,
    ) = gpu;
    let (mut dlss, mut prev_view_proj, mut dlss_was_active) = dlss_stuff;

    let (dev_ui_state, mut ui, sky, procedural, env_light, mut gizmos) = dev_ui_stuff;
    let dev_ui_state = dev_ui_state.map(|state| state.clone()).unwrap_or_default();
    *frame_counter = frame_counter.wrapping_add(1);
    let camera = camera.single().unwrap();
    // XR: the camera entity's transform anchors the headset's LOCAL space in the world (the
    // fly-cam still moves it); each eye's pose and asymmetric fov come from the runtime.
    // Flat (and an XR session that produced no frame): the entity transform is the whole
    // camera and the window is the one view.
    let anchor = camera.1.to_matrix();
    let (output_extent, cameras): (vk::Extent2D, Vec<(usize, Mat4, Mat4)>) =
        match (&xr_frame, xr.as_deref(), camera.0) {
            (Some(xr_frame), Some(xr_state), projection) => {
                let near = match projection {
                    Projection::Perspective(perspective) => perspective.near,
                    _ => 0.1,
                };
                (
                    xr_state.extent,
                    (0..2)
                        .map(|eye| {
                            let (head, projection) = xr_frame.eye_camera(eye, near);
                            (eye, anchor * head, projection)
                        })
                        .collect(),
                )
            }
            (_, _, Projection::Perspective(perspective)) => (
                swapchain.swapchain_extent,
                vec![(
                    0,
                    anchor,
                    Mat4::perspective_infinite_reverse_rh(
                        perspective.fov,
                        (window.width as f32) / (window.height as f32),
                        perspective.near,
                    ),
                )],
            ),
            (_, _, Projection::Orthographic(_)) => todo!("orthographic camera"),
            (_, _, Projection::Custom(_)) => todo!("custom_projection"),
        };

    // DLSS: settle each view's trace resolution and jitter before recording (feature
    // creation, when the mode or output size changed, submits and waits on its own).
    let dlss_mode = camera.2.copied().unwrap_or_default();
    let reset_requested = std::mem::take(&mut dlss.reset_requested);
    let views: Vec<ViewFrame> = cameras
        .into_iter()
        .map(|(slot, inverse_view, projection)| {
            let plan = dlss
                .renderer
                .as_mut()
                .and_then(|r| r.prepare(&render_device, slot, output_extent, dlss_mode));
            let view_matrix = inverse_view.inverse();
            let view_proj = projection * view_matrix;
            let last_view_proj = prev_view_proj[slot].unwrap_or(view_proj);
            prev_view_proj[slot] = Some(view_proj);
            ViewFrame {
                slot,
                inverse_view,
                view_matrix,
                projection,
                view_proj,
                last_view_proj,
                plan,
            }
        })
        .collect();
    let plan = views.first().and_then(|v| v.plan);
    let dlss_reset = plan.is_some() && (!*dlss_was_active || reset_requested);
    *dlss_was_active = plan.is_some();
    let trace_extent = plan.map_or(output_extent, |p| p.render);
    // Set once an evaluate has been recorded this frame: only then does a blit read a DLSS
    // output (before the RT pipeline is compiled nothing has written them).
    let mut dlss_ran = false;

    // Ensure the per-view uniform buffers exist
    for buffer in &mut frame.uniform_buffers {
        if buffer.handle == vk::Buffer::null() {
            *buffer = render_device.create_host_buffer(1, vk::BufferUsageFlags::UNIFORM_BUFFER);
        }
    }

    // Ensure the focus_data buffer exists
    if frame.focus_data.handle == vk::Buffer::null() {
        let mut staging_buffer: Buffer<FocusData> = render_device.create_host_buffer(
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );

        let initial_data = FocusData {
            focal_distance: 100.0,
        };

        {
            let mut mapped = render_device.map_buffer(&mut staging_buffer);
            mapped.copy_from_slice(&[initial_data]);
        }

        frame.focus_data = render_device.create_device_buffer(
            1,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );

        render_device.run_transfer_commands(|cmd_buffer| {
            render_device.upload_buffer(cmd_buffer, &staging_buffer, &frame.focus_data);
        });

        render_device
            .destroyer
            .destroy_buffer(staging_buffer.handle);
    }

    // Update each view's uniform buffer
    for view in &views {
        let data = UniformData {
            sky_color: match &*sky {
                Sky::Color { radiance } => radiance.extend(0.0),
                Sky::Hdr { scale, .. } => Vec4::splat(*scale),
                Sky::Procedural => Vec4::ONE,
            },
            inverse_view: view.inverse_view,
            inverse_projection: view.projection.inverse(),
            pull_focus_x: render_config
                .pull_focus
                .map(|(x, _)| x)
                .unwrap_or(0xFFFFFFFF),
            pull_focus_y: render_config
                .pull_focus
                .map(|(_, y)| y)
                .unwrap_or(0xFFFFFFFF),
            gamma: dev_ui_state.gamma,
            aperture: dev_ui_state.aperture,
            foginess: dev_ui_state.foginess,
            fog_scatter: dev_ui_state.fog_scatter,
            sky_brightness: dev_ui_state.sky_brightness,
            view: view.view_matrix.to_cols_array(),
            view_proj: view.view_proj.to_cols_array(),
            prev_view_proj: view.last_view_proj.to_cols_array(),
            jitter: view.plan.map_or([0.0; 2], |p| p.jitter),
            frame: *frame_counter,
            radiance_clamp: {
                let sky_luma = sky.reference_luminance(&procedural) * dev_ui_state.sky_brightness;
                dev_ui_state.firefly_clamp * sky_luma.max(1e-3)
            },
            samples: dev_ui_state.samples.max(1),
            max_bounces: dev_ui_state.max_bounces.max(1),
            vignette: dev_ui_state.vignette,
            sky_mode: match &*sky {
                Sky::Color { .. } => 0,
                Sky::Hdr { .. } => 1,
                Sky::Procedural => 2,
            },
            sun_cos_radius: procedural.sun_cos_radius(),
            sun_direction: procedural.sun_direction().to_array(),
            sun_radiance: Vec3::splat(procedural.sun_radiance).to_array(),
            sky_zenith: procedural.zenith_radiance().to_array(),
            sky_horizon: procedural.horizon_radiance().to_array(),
            sky_ground: procedural.ground_radiance().to_array(),
            light_entries: if dev_ui_state.light_nee {
                lights.active_entries
            } else {
                0
            },
            restir_candidates: if dev_ui_state.restir {
                dev_ui_state.restir_candidates.clamp(1, 32)
            } else {
                0
            },
            light_epoch: lights.epoch,
            restir_m_clamp: (dev_ui_state.restir_candidates as f32 * dev_ui_state.restir_history)
                .max(1.0),
            sharc: dev_ui_state.sharc as u32,
            sharc_voxel: dev_ui_state.sharc_voxel.max(0.01),
            env: env_light.address(),
            env_w: if env_light.address() != 0 {
                crate::env_light::ENV_W
            } else {
                0
            },
            env_h: if env_light.address() != 0 {
                crate::env_light::ENV_H
            } else {
                0
            },
            emissive_boost: dev_ui_state.emissive_boost.max(0.0),
        };

        let mut mapped = render_device.map_buffer(&mut frame.uniform_buffers[view.slot]);
        mapped.copy_from_slice(&[data]);
    }

    // A dev-snippet debug hotkey is (or is about to be) reallocating NGX-internal
    // visualisation resources inside evaluate: run these frames fully serialized so the
    // realloc never overlaps in-flight work (Xid 79 on this hardware otherwise).
    let dev_drain = dlss.dev_drain > 0;
    if dev_drain {
        dlss.dev_drain -= 1;
        let _ = unsafe { render_device.device.device_wait_idle() };
    }

    unsafe {
        let (swapchain_image, swapchain_view) = swapchain.aquire_next_image(&window);
        render_device.destroyer.tick();
        let cmd_buffer = render_device.command_buffers[swapchain.frame_count % 2];

        frame.swapchain_image = swapchain_image;
        frame.swapchain_view = swapchain_view;

        render_device
            .reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty())
            .unwrap();

        render_device
            .begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        // The in-flight fence was waited in aquire_next_image, so the previous trace is done:
        // propagate this frame's transform deltas on the GPU, refresh the instance table from
        // them, and rebuild the single TLAS in place -- all inside this command buffer.
        let world_changed = transforms.record(&render_device, cmd_buffer, &modules);
        let skinned = skins.record(&render_device, cmd_buffer, &modules, &transforms);
        tlas.record(
            &render_device,
            cmd_buffer,
            &modules,
            &transforms,
            world_changed || skinned,
        );
        // The light table's weight/CDF kernels, whenever the light set changed (they read
        // the instance rows the gather above wrote).
        lights.record(&render_device, cmd_buffer, &modules, &tlas);
        // The radiance cache's per-frame resolve (and its first-use allocation).
        sharc.record(
            &render_device,
            cmd_buffer,
            &modules,
            *frame_counter,
            dev_ui_state.sharc,
        );
        // Exposure: meter last frame's luminance into this frame's exposure (the raygen
        // reads it), or write the camera's fixed EV.
        let exposure = camera.3.cloned().unwrap_or_default();
        ae.record(
            &render_device,
            cmd_buffer,
            &modules,
            trace_extent,
            &exposure,
            time.delta_secs(),
        );

        if let Some(rtx_pipeline) = rtx_pipelines.get(&render_config.rtx_pipeline) {
            if tlas.acceleration_structure.handle != vk::AccelerationStructureKHR::null()
                && sbt.data.address != 0
            {
                let sky_texture = match &*sky {
                    Sky::Hdr { image, .. } => textures.get(image).map_or(WHITE_TEXTURE_IDX, |t| {
                        render_device.register_bindless_texture(&t)
                    }),
                    _ => WHITE_TEXTURE_IDX,
                };
                render_device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rtx_pipeline.pipeline,
                );

                for view in &views {
                    // Each view's own descriptor set (a set already recorded into this
                    // command buffer must not be rewritten).
                    let mut ac_binding = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(std::slice::from_ref(
                            &tlas.acceleration_structure.handle,
                        ));
                    let set =
                        rtx_pipeline.descriptor_sets[(swapchain.frame_count % 2) * 2 + view.slot];
                    let mut writes = vec![
                        vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(100)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .push_next(&mut ac_binding),
                    ];
                    // DLSS guide images + noisy colour, this view's feature.
                    let guide_infos: Vec<vk::DescriptorImageInfo> = dlss
                        .renderer
                        .as_ref()
                        .filter(|_| view.plan.is_some())
                        .and_then(|r| r.guide_views(view.slot))
                        .map(|g| {
                            [
                                g.normal_roughness,
                                g.diffuse,
                                g.specular,
                                g.depth,
                                g.spec_hit,
                                g.motion,
                                g.color,
                            ]
                            .into_iter()
                            .map(|view| {
                                vk::DescriptorImageInfo::default()
                                    .image_layout(vk::ImageLayout::GENERAL)
                                    .image_view(view)
                            })
                            .collect()
                        })
                        .unwrap_or_default();
                    for (i, info) in guide_infos.iter().enumerate() {
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(i as u32)
                                .descriptor_count(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(std::slice::from_ref(info)),
                        );
                    }

                    render_device.update_descriptor_sets(&writes, &[]);

                    render_device.cmd_bind_descriptor_sets(
                        cmd_buffer,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        rtx_pipeline.pipeline_layout,
                        0,
                        &[set, render_device.bindless_descriptor_set],
                        &[],
                    );

                    let (reservoirs_prev, reservoirs_cur) = restir.ensure(
                        &render_device,
                        cmd_buffer,
                        trace_extent,
                        *frame_counter,
                        view.slot,
                    );
                    let push_constants = RaytracingPushConstants {
                        uniform_buffer: frame.uniform_buffers[view.slot].address,
                        material_buffer: tlas.material_address(),
                        bluenoise_buffer2: bluenoise_buffer.0.address,
                        focus_buffer: frame.focus_data.address,
                        sky_texture,
                        padding: [0; 1],
                        prev_instances: tlas.prev_instances_address(),
                        lights: lights.header_address(),
                        instances: tlas.instances_address(),
                        reservoirs_prev,
                        reservoirs_cur,
                        sharc: sharc.address(),
                        lum_buffer: ae.addresses().0,
                        auto_exposure: ae.addresses().1,
                    };

                    render_device.cmd_push_constants(
                        cmd_buffer,
                        rtx_pipeline.pipeline_layout,
                        vk::ShaderStageFlags::ALL,
                        0,
                        bytemuck::cast_slice(&[push_constants]),
                    );

                    if let (Some(plan), Some(renderer)) = (view.plan, dlss.renderer.as_mut()) {
                        renderer.record_pre_trace(&render_device, cmd_buffer, view.slot);
                        render_device.ext_rtx_pipeline.cmd_trace_rays(
                            cmd_buffer,
                            &sbt.raygen_region,
                            &sbt.miss_region,
                            &sbt.hit_region,
                            &vk::StridedDeviceAddressRegionKHR::default(),
                            plan.render.width,
                            plan.render.height,
                            1,
                        );
                        renderer.record_evaluate(
                            &render_device,
                            cmd_buffer,
                            view.slot,
                            view.view_matrix,
                            view.projection,
                            plan.jitter,
                            dlss_reset,
                            time.delta_secs() * 1000.0,
                        );
                        dlss_ran = true;
                        // The raygen just filled the luminance buffer at this size.
                        ae.primed = true;
                    }
                }
            }
        }

        let parity = swapchain.frame_count % 2;
        let postprocess = postprocess_filters.get(&render_config.postprocess_pipeline);
        let debug_view = camera.4.copied().unwrap_or_default().shader_index();

        // XR: resolve each eye through the post-process into its target and hand it to the
        // compositor layer, before the window's own pass.
        if let (Some(xr_state), Some(xr_frame), Some(pipeline), true) =
            (xr.as_deref(), xr_frame.as_ref(), postprocess, dlss_ran)
        {
            for eye in 0..2 {
                let (eye_image, eye_view) = xr_state.eye_target(eye);
                vk_utils::transition_image_layout(
                    &render_device,
                    cmd_buffer,
                    eye_image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::ATTACHMENT_OPTIMAL,
                );
                let render_area = vk::Rect2D::default().extent(xr_state.extent);
                let attachment_info = vk::RenderingAttachmentInfo::default()
                    .image_view(eye_view)
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE);
                let render_info = vk::RenderingInfo::default()
                    .layer_count(1)
                    .render_area(render_area)
                    .color_attachments(std::slice::from_ref(&attachment_info));
                render_device.cmd_begin_rendering(cmd_buffer, &render_info);
                render_device.cmd_set_scissor(cmd_buffer, 0, std::slice::from_ref(&render_area));
                render_device.cmd_set_viewport(
                    cmd_buffer,
                    0,
                    std::slice::from_ref(
                        &vk::Viewport::default()
                            .width(xr_state.extent.width as f32)
                            .height(xr_state.extent.height as f32)
                            .min_depth(0.0)
                            .max_depth(1.0),
                    ),
                );
                if let Some(source_view) = dlss.renderer.as_ref().and_then(|r| r.output_view(eye)) {
                    record_post_draw(
                        &render_device,
                        cmd_buffer,
                        pipeline,
                        pipeline.descriptor_sets[parity * 3 + eye],
                        source_view,
                        dlss.renderer.as_ref().and_then(|r| r.guide_views(eye)),
                        &crate::post_process_filter::PostProcessPushConstants {
                            uniforms: frame.uniform_buffers[eye].address,
                            auto_exposure: ae.addresses().1,
                            display_exposure: exposure.display_exposure(),
                            debug_view,
                        },
                    );
                }
                render_device.cmd_end_rendering(cmd_buffer);
                xr_state.record_eye_to_layer(&render_device, cmd_buffer, xr_frame, eye);
            }
        }

        // The window: the flat render, or the XR spectator (left eye, aspect-fit).
        vk_utils::transition_image_layout(
            &render_device,
            cmd_buffer,
            swapchain_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
        );

        let render_area = vk::Rect2D::default().extent(swapchain.swapchain_extent);

        let attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_view)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE);

        let render_info = vk::RenderingInfo::default()
            .layer_count(1)
            .render_area(render_area)
            .color_attachments(std::slice::from_ref(&attachment_info));

        render_device.cmd_begin_rendering(cmd_buffer, &render_info);
        render_device.cmd_set_scissor(cmd_buffer, 0, std::slice::from_ref(&render_area));

        // Until the RT pipeline compiles and the first evaluate lands there is nothing to
        // sample -- leave the clear (the UI still draws).
        let source_view = dlss
            .renderer
            .as_ref()
            .filter(|_| dlss_ran)
            .and_then(|r| r.output_view(0));
        if let (Some(pipeline), Some(source_view)) = (postprocess, source_view) {
            let (window_w, window_h) = (
                swapchain.swapchain_extent.width as f32,
                swapchain.swapchain_extent.height as f32,
            );
            // Spectator: letterbox the eye image instead of stretching it.
            let viewport = if xr_frame.is_some() {
                let source_aspect = output_extent.width as f32 / output_extent.height as f32;
                let (w, h) = if source_aspect > window_w / window_h {
                    (window_w, window_w / source_aspect)
                } else {
                    (window_h * source_aspect, window_h)
                };
                vk::Viewport::default()
                    .x((window_w - w) * 0.5)
                    .y((window_h - h) * 0.5)
                    .width(w)
                    .height(h)
            } else {
                vk::Viewport::default().width(window_w).height(window_h)
            }
            .min_depth(0.0)
            .max_depth(1.0);
            render_device.cmd_set_viewport(cmd_buffer, 0, std::slice::from_ref(&viewport));

            let target = if xr_frame.is_some() { 2 } else { 0 };
            record_post_draw(
                &render_device,
                cmd_buffer,
                pipeline,
                pipeline.descriptor_sets[parity * 3 + target],
                source_view,
                dlss.renderer.as_ref().and_then(|r| r.guide_views(0)),
                &crate::post_process_filter::PostProcessPushConstants {
                    uniforms: frame.uniform_buffers[0].address,
                    auto_exposure: ae.addresses().1,
                    display_exposure: exposure.display_exposure(),
                    debug_view,
                },
            );
        }

        // bevy_ui / feathers (including the dev panel), drawn over the scene, full-window.
        render_device.cmd_set_viewport(
            cmd_buffer,
            0,
            std::slice::from_ref(
                &vk::Viewport::default()
                    .width(swapchain.swapchain_extent.width as f32)
                    .height(swapchain.swapchain_extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0),
            ),
        );
        // Debug gizmo lines (bevy_gizmos), world-space, over the scene and under the UI.
        // Skipped under XR: the spectator's letterboxed blit does not match views[0]'s
        // full-window projection.
        if xr_frame.is_none() {
            crate::gizmo_render::draw_gizmos(
                &render_device,
                cmd_buffer,
                views[0].view_proj,
                swapchain.frame_count % 2,
                &mut gizmos,
            );
        }
        crate::ui_render::draw_ui(
            &render_device,
            cmd_buffer,
            swapchain.swapchain_extent,
            swapchain.frame_count % 2,
            &mut ui,
        );

        render_device.cmd_end_rendering(cmd_buffer);

        // A pending screenshot copies the finished frame (UI included) out through a host
        // buffer on its way to present.
        // The capture reads the WINDOW swapchain (UI included), so it must use the
        // swapchain extent -- under XR `output_extent` is the eye extent, and sizing the
        // copy with it reads out of the swapchain image's bounds (F12 crashed XR runs).
        let capture_extent = swapchain.swapchain_extent;
        let capture = screenshots.take_next().map(|path| {
            let size = capture_extent.width as u64 * capture_extent.height as u64 * 4;
            let buffer: Buffer<u8> =
                render_device.create_host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST);
            vk_utils::transition_image_layout(
                &render_device,
                cmd_buffer,
                frame.swapchain_image,
                vk::ImageLayout::ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            // bufferRowLength 0 packs rows tightly: pitch is exactly width * 4.
            let region = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: capture_extent.width,
                    height: capture_extent.height,
                    depth: 1,
                });
            render_device.cmd_copy_image_to_buffer(
                cmd_buffer,
                frame.swapchain_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.handle,
                std::slice::from_ref(&region),
            );
            (path, buffer)
        });

        // Make swapchain available for present
        vk_utils::transition_image_layout(
            &render_device,
            cmd_buffer,
            frame.swapchain_image,
            if capture.is_some() {
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
            } else {
                vk::ImageLayout::ATTACHMENT_OPTIMAL
            },
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        render_device.end_command_buffer(cmd_buffer).unwrap();
        swapchain.submit_presentation(&window, cmd_buffer);
        // The XR side of the submit: release the image and hand the compositor its layer.
        if let (Some(xr_state), Some(xr_frame)) = (xr.as_deref_mut(), xr_frame) {
            xr_state.end_frame(&render_device, xr_frame);
        }
        if dev_drain {
            let _ = render_device.device.device_wait_idle();
        }

        if let Some((path, mut buffer)) = capture {
            // Debug tool: one hitch per capture beats plumbing a fence through the frame.
            let _ = render_device.device.device_wait_idle();
            let mut view = render_device.map_buffer(&mut buffer);
            crate::util::screenshot::save_png(
                view.as_slice_mut(),
                swapchain.swapchain_format,
                capture_extent,
                &path,
            );
            render_device.destroyer.destroy_buffer(buffer.handle);
        }
    }
}

/// One post-process draw into the current render pass: slot 0 of `set` samples
/// `source_view`, slots 1..=7 the guides for the debug views (aliasing the output until the
/// renderer has them). Guides sit in SHADER_READ_ONLY after the evaluate; the output stays
/// GENERAL. Viewport and scissor are the caller's.
unsafe fn record_post_draw(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    pipeline: &crate::post_process_filter::CompiledPostProcessFilter,
    set: vk::DescriptorSet,
    source_view: vk::ImageView,
    guides: Option<crate::dlss::GuideViews>,
    push_constants: &crate::post_process_filter::PostProcessPushConstants,
) {
    unsafe {
        render_device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline,
        );
        render_device.cmd_push_constants(
            cmd_buffer,
            pipeline.pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(push_constants),
        );

        let output_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(source_view)
            .sampler(render_device.linear_sampler);
        let mut image_infos = [output_info; 8];
        if let Some(g) = guides {
            for (slot, view) in [
                (1, g.color),
                (2, g.normal_roughness),
                (3, g.diffuse),
                (4, g.specular),
                (5, g.depth),
                (6, g.spec_hit),
                (7, g.motion),
            ] {
                image_infos[slot] = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(view)
                    .sampler(render_device.linear_sampler);
            }
        }

        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)];
        render_device.update_descriptor_sets(&writes, &[]);

        render_device.cmd_bind_descriptor_sets(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline_layout,
            0,
            std::slice::from_ref(&set),
            &[],
        );

        render_device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
    }
}

pub(crate) fn on_shutdown(world: &mut World) {
    // XR session and instance go before the VkDevice they wrap.
    let xr_state = world.remove_resource::<crate::xr::XrState>();
    world.remove_resource::<crate::xr::XrContext>();

    let render_device = world
        .remove_resource::<crate::render_device::RenderDevice>()
        .unwrap();
    if let Some(xr_state) = xr_state {
        xr_state.destroy(&render_device);
    }

    let frame = world.remove_resource::<Frame>().unwrap();

    for buffer in &frame.uniform_buffers {
        render_device.destroyer.destroy_buffer(buffer.handle);
    }
    render_device
        .destroyer
        .destroy_buffer(frame.focus_data.handle);
    let sphere_blas = world
        .remove_resource::<crate::sphere::SphereBLAS>()
        .unwrap();
    render_device
        .destroyer
        .destroy_buffer(sphere_blas.aabb_buffer.handle);
    sphere_blas.acceleration_structure.destroy(&render_device);

    render_device.destroyer.tick();
    render_device.destroyer.tick();
    render_device.destroyer.tick();
    world.remove_resource::<crate::swapchain::Swapchain>();
}

fn run_if_render_device_exists(device: Option<Res<RenderDevice>>) -> bool {
    device.is_some()
}
