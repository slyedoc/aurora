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
    vk_init, vk_utils,
    vulkan_asset::VulkanAssets,
};

#[derive(Resource, Clone)]
pub struct RenderConfig {
    pub rtx_pipeline: Handle<RaytracingPipeline>,
    pub postprocess_pipeline: Handle<PostProcessFilter>,
    pub accumulate: bool,
    pub pull_focus: Option<(u32, u32)>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            rtx_pipeline: Default::default(),
            postprocess_pipeline: Default::default(),
            // `AURORA_ACCUMULATE=1` starts with accumulation on (Space toggles it) -- a
            // converged reference for headless comparisons.
            accumulate: std::env::var_os("AURORA_ACCUMULATE").is_some(),
            pull_focus: Default::default(),
        }
    }
}

#[repr(C)]
pub struct UniformData {
    sky_color: Vec4,
    inverse_view: Mat4,
    inverse_projection: Mat4,
    tick: u32,
    accumulate: u32,
    pull_focus_x: u32,
    pull_focus_y: u32,
    gamma: f32,
    exposure: f32,
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
    dlss: u32,
    /// Free-running frame counter: the RNG seed while DLSS is on, so every frame's noise is
    /// new for the temporal denoiser (`tick` resets to 0 whenever accumulation is off).
    frame: u32,
    /// Indirect path contributions are clamped to this luminance (0 = off).
    radiance_clamp: f32,
    /// Paths per pixel this frame and their maximum length (from [`DevUIState`], the DLSS
    /// pair while Ray Reconstruction runs).
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
}

#[repr(C)]
pub struct FocusData {
    focal_distance: f32,
}

fn handle_input(keyboard: Res<ButtonInput<KeyCode>>, mut render_config: ResMut<RenderConfig>) {
    if keyboard.just_pressed(KeyCode::Space) {
        render_config.accumulate = !render_config.accumulate;
    }
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
        let render_device = unsafe {
            crate::render_device::RenderDevice::from_display(
                &display_handle.0.display_handle().unwrap(),
            )
        };
        let sphere_blas = unsafe { crate::sphere::SphereBLAS::new(&render_device) };
        app.insert_resource(render_device);
        app.insert_resource(sphere_blas);
        app.init_resource::<Frame>();
        app.init_resource::<RenderConfig>();

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
        app.add_systems(Update, (handle_input, set_focus_pulling));
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
    let window = windows.single().unwrap();
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
    pub render_frame_buffers: RenderFrameBuffers,
    pub uniform_buffer: Buffer<UniformData>,
    pub focus_data: Buffer<FocusData>,
}

#[derive(Default)]
pub struct RenderFrameBuffers {
    pub main: (vk::Image, vk::ImageView),
}

impl RenderFrameBuffers {
    pub unsafe fn prepare(
        &mut self,
        render_device: &RenderDevice,
        swapchain: &crate::swapchain::Swapchain,
        cmd_buffer: vk::CommandBuffer,
    ) {
        unsafe {
            // (Re)create the render target if needed
            if self.main.0 == vk::Image::null() || swapchain.resized {
                log::trace!("(Re)creating render target");
                render_device.destroyer.destroy_image_view(self.main.1);
                render_device.destroyer.destroy_image(self.main.0);
                let image_info = vk_init::image_info(
                    swapchain.swapchain_extent.width,
                    swapchain.swapchain_extent.height,
                    vk::Format::R32G32B32A32_SFLOAT,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                );
                self.main.0 = render_device.create_render_target(&image_info);

                let view_info = vk_init::image_view_info(self.main.0, image_info.format);
                self.main.1 = render_device.create_image_view(&view_info, None).unwrap();

                // Transition to render target to general
                vk_utils::transition_image_layout(
                    &render_device,
                    cmd_buffer,
                    self.main.0,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                );
            }
        }
    }

    pub fn destroy(&mut self, render_device: &RenderDevice) {
        render_device.destroyer.destroy_image_view(self.main.1);
        render_device.destroyer.destroy_image(self.main.0);
    }
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
    ),
    camera: Query<
        (
            &Projection,
            &GlobalTransform,
            Option<&crate::dlss::AuroraDlss>,
        ),
        With<Camera3d>,
    >,
    mut tick: Local<u32>,
    mut frame_counter: Local<u32>,
    dlss_stuff: (
        ResMut<crate::dlss::DlssState>,
        Local<Option<Mat4>>,
        Local<bool>,
    ),
) {
    let Some(mut swapchain) = swapchain else {
        return;
    };
    let (mut transforms, modules, sbt, mut lights, mut restir, mut sharc) = gpu;
    let (mut dlss, mut prev_view_proj, mut dlss_was_active) = dlss_stuff;

    let (dev_ui_state, mut ui, sky, procedural) = dev_ui_stuff;
    let dev_ui_state = dev_ui_state.map(|state| state.clone()).unwrap_or_default();

    *tick += 1;
    if !render_config.accumulate {
        *tick = 0;
    }
    *frame_counter = frame_counter.wrapping_add(1);
    let camera = camera.single().unwrap();
    let inverse_view = camera.1.to_matrix();
    let projection_matrix = match camera.0 {
        Projection::Perspective(perspective) => Mat4::perspective_infinite_reverse_rh(
            perspective.fov,
            (window.width as f32) / (window.height as f32),
            perspective.near,
        ),
        Projection::Orthographic(_) => todo!("orthographic camera"),
        Projection::Custom(_) => todo!("custom_projection"),
    };
    let inverse_projection = projection_matrix.inverse();
    let view_matrix = inverse_view.inverse();
    let view_proj = projection_matrix * view_matrix;
    let last_view_proj = prev_view_proj.unwrap_or(view_proj);
    *prev_view_proj = Some(view_proj);

    // DLSS: settle the trace resolution and jitter before recording (feature creation, when
    // the mode or window changed, submits and waits on its own).
    let output_extent = swapchain.swapchain_extent;
    let dlss_mode = camera.2.copied().unwrap_or_default();
    let plan = dlss
        .renderer
        .as_mut()
        .and_then(|r| r.prepare(&render_device, output_extent, dlss_mode));
    let reset_requested = std::mem::take(&mut dlss.reset_requested);
    let dlss_reset = plan.is_some() && (!*dlss_was_active || reset_requested);
    *dlss_was_active = plan.is_some();
    let trace_extent = plan.map_or(output_extent, |p| p.render);
    // Set once the evaluate has been recorded this frame: only then does the blit read the
    // DLSS output (before the RT pipeline is compiled nothing has written it).
    let mut dlss_ran = false;
    let rtx_ready = rtx_pipelines.get(&render_config.rtx_pipeline).is_some()
        && sbt.data.address != 0
        && tlas.acceleration_structure.handle != vk::AccelerationStructureKHR::null();

    // Ensure the uniform_buffer exists
    if frame.uniform_buffer.handle == vk::Buffer::null() {
        frame.uniform_buffer =
            render_device.create_host_buffer(1, vk::BufferUsageFlags::UNIFORM_BUFFER);
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

    // Update the uniform buffer
    {
        let dlss_on = plan.is_some() && rtx_ready;
        let data = UniformData {
            sky_color: match &*sky {
                Sky::Color { radiance } => radiance.extend(0.0),
                Sky::Hdr { scale, .. } => Vec4::splat(*scale),
                Sky::Procedural => Vec4::ONE,
            },
            inverse_view,
            inverse_projection,
            tick: *tick,
            accumulate: if render_config.accumulate { 1 } else { 0 },
            pull_focus_x: render_config
                .pull_focus
                .map(|(x, _)| x)
                .unwrap_or(0xFFFFFFFF),
            pull_focus_y: render_config
                .pull_focus
                .map(|(_, y)| y)
                .unwrap_or(0xFFFFFFFF),
            gamma: dev_ui_state.gamma,
            exposure: dev_ui_state.exposure_ev.exp2(),
            aperture: dev_ui_state.aperture,
            foginess: dev_ui_state.foginess,
            fog_scatter: dev_ui_state.fog_scatter,
            sky_brightness: dev_ui_state.sky_brightness,
            view: view_matrix.to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            prev_view_proj: last_view_proj.to_cols_array(),
            jitter: plan.map_or([0.0; 2], |p| p.jitter),
            dlss: dlss_on as u32,
            frame: *frame_counter,
            radiance_clamp: {
                let sky_luma = sky.reference_luminance(&procedural) * dev_ui_state.sky_brightness;
                dev_ui_state.firefly_clamp * sky_luma.max(1e-3)
            },
            samples: if dlss_on {
                dev_ui_state.dlss_samples
            } else {
                dev_ui_state.samples
            }
            .max(1),
            max_bounces: if dlss_on {
                dev_ui_state.dlss_max_bounces
            } else {
                dev_ui_state.max_bounces
            }
            .max(1),
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
            // Accumulation stays the uncorrelated reference estimator.
            restir_candidates: if dev_ui_state.restir && !render_config.accumulate {
                dev_ui_state.restir_candidates.clamp(1, 32)
            } else {
                0
            },
            light_epoch: lights.epoch,
            restir_m_clamp: (dev_ui_state.restir_candidates as f32 * dev_ui_state.restir_history)
                .max(1.0),
            sharc: (dev_ui_state.sharc && !render_config.accumulate) as u32,
            sharc_voxel: dev_ui_state.sharc_voxel.max(0.01),
        };

        let mut mapped = render_device.map_buffer(&mut frame.uniform_buffer);
        mapped.copy_from_slice(&[data]);
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

        frame
            .render_frame_buffers
            .prepare(&render_device, &swapchain, cmd_buffer);

        // The in-flight fence was waited in aquire_next_image, so the previous trace is done:
        // propagate this frame's transform deltas on the GPU, refresh the instance table from
        // them, and rebuild the single TLAS in place -- all inside this command buffer.
        let world_changed = transforms.record(&render_device, cmd_buffer, &modules);
        tlas.record(
            &render_device,
            cmd_buffer,
            &modules,
            &transforms,
            world_changed,
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
            dev_ui_state.sharc && !render_config.accumulate,
        );

        if let Some(rtx_pipeline) = rtx_pipelines.get(&render_config.rtx_pipeline) {
            if tlas.acceleration_structure.handle != vk::AccelerationStructureKHR::null()
                && sbt.data.address != 0
            {
                // Ensure the descriptor set is up to date
                let render_target_main_binding = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(frame.render_frame_buffers.main.1);

                let mut ac_binding = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(std::slice::from_ref(
                        &tlas.acceleration_structure.handle,
                    ));

                let set = rtx_pipeline.descriptor_sets[swapchain.frame_count % 2];
                let mut writes = vec![
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(&render_target_main_binding)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(100)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .push_next(&mut ac_binding),
                ];
                // DLSS guide images (partially bound: only written while DLSS is on).
                let guide_infos: Vec<vk::DescriptorImageInfo> = dlss
                    .renderer
                    .as_ref()
                    .filter(|_| plan.is_some())
                    .and_then(|r| r.guide_views())
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
                            .dst_binding(1 + i as u32)
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
                    &[
                        rtx_pipeline.descriptor_sets[swapchain.frame_count % 2],
                        render_device.bindless_descriptor_set,
                    ],
                    &[],
                );

                render_device.cmd_bind_pipeline(
                    cmd_buffer,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rtx_pipeline.pipeline,
                );

                let (reservoirs_prev, reservoirs_cur) =
                    restir.ensure(&render_device, cmd_buffer, trace_extent, *frame_counter);
                let push_constants = RaytracingPushConstants {
                    uniform_buffer: frame.uniform_buffer.address,
                    material_buffer: tlas.material_address(),
                    bluenoise_buffer2: bluenoise_buffer.0.address,
                    focus_buffer: frame.focus_data.address,
                    sky_texture: match &*sky {
                        Sky::Hdr { image, .. } => {
                            textures.get(image).map_or(WHITE_TEXTURE_IDX, |t| {
                                render_device.register_bindless_texture(&t)
                            })
                        }
                        _ => WHITE_TEXTURE_IDX,
                    },
                    padding: [0; 1],
                    prev_instances: tlas.prev_instances_address(),
                    lights: lights.header_address(),
                    instances: tlas.instances_address(),
                    reservoirs_prev,
                    reservoirs_cur,
                    sharc: sharc.address(),
                };

                render_device.cmd_push_constants(
                    cmd_buffer,
                    rtx_pipeline.pipeline_layout,
                    vk::ShaderStageFlags::ALL,
                    0,
                    bytemuck::cast_slice(&[push_constants]),
                );

                if let (Some(plan), Some(renderer)) = (plan, dlss.renderer.as_mut()) {
                    renderer.record_pre_trace(&render_device, cmd_buffer);
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
                        view_matrix,
                        projection_matrix,
                        plan.jitter,
                        dlss_reset,
                    );
                    dlss_ran = true;
                } else {
                    render_device.ext_rtx_pipeline.cmd_trace_rays(
                        cmd_buffer,
                        &sbt.raygen_region,
                        &sbt.miss_region,
                        &sbt.hit_region,
                        &vk::StridedDeviceAddressRegionKHR::default(),
                        trace_extent.width,
                        trace_extent.height,
                        1,
                    );
                }
            }
        }

        // Make swapchain available for rendering
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

        if let Some(pipeline) = postprocess_filters.get(&render_config.postprocess_pipeline) {
            render_device.cmd_bind_pipeline(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );

            let push_constants = frame.uniform_buffer.address;
            render_device.cmd_push_constants(
                cmd_buffer,
                pipeline.pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::cast_slice(&[push_constants]),
            );

            // Ensure the descriptor set is up to date: the DLSS output when it ran this
            // frame, else the accumulation target.
            let source_view = dlss
                .renderer
                .as_ref()
                .filter(|_| dlss_ran)
                .and_then(|r| r.output_view())
                .unwrap_or(frame.render_frame_buffers.main.1);
            let render_target_main_binding = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(source_view)
                .sampler(render_device.linear_sampler);

            let writes = [vk::WriteDescriptorSet::default()
                .dst_set(pipeline.descriptor_sets[swapchain.frame_count % 2])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&render_target_main_binding))];

            render_device.update_descriptor_sets(&writes, &[]);

            render_device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline_layout,
                0,
                std::slice::from_ref(&pipeline.descriptor_sets[swapchain.frame_count % 2]),
                &[],
            );

            render_device.cmd_draw(cmd_buffer, 3, 1, 0, 0);
        }

        // bevy_ui / feathers (including the dev panel), drawn over the scene
        crate::ui_render::draw_ui(
            &render_device,
            cmd_buffer,
            swapchain.swapchain_extent,
            swapchain.frame_count % 2,
            &mut ui,
        );

        render_device.cmd_end_rendering(cmd_buffer);

        // Make swapchain available for present
        vk_utils::transition_image_layout(
            &render_device,
            cmd_buffer,
            frame.swapchain_image,
            vk::ImageLayout::ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );

        render_device.end_command_buffer(cmd_buffer).unwrap();
        swapchain.submit_presentation(&window, cmd_buffer);
    }
}

pub(crate) fn on_shutdown(world: &mut World) {
    let render_device = world
        .remove_resource::<crate::render_device::RenderDevice>()
        .unwrap();

    let mut frame = world.remove_resource::<Frame>().unwrap();
    frame.render_frame_buffers.destroy(&render_device);

    render_device
        .destroyer
        .destroy_buffer(frame.uniform_buffer.handle);
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
