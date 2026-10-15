use bevy::tasks::{ComputeTaskPool, TaskPool};
use bevy::{
    app::AppExit,
    ecs::{message::Messages, schedule::ScheduleLabel},
    prelude::*,
    window::{RawHandleWrapperHolder, WindowCloseRequested},
    winit::DisplayHandleWrapper,
};
use raw_window_handle::HasDisplayHandle;

use ash::vk;

use crate::color_target::TargetKind;
use crate::descriptor_heap::DescriptorHeaps;

use crate::sky::{EnvironmentGpu, EnvironmentSkies, Sky};
use crate::{
    atmosphere::{Atmosphere, AtmosphereState, CloudLayer},
    bluenoise_plugin::BlueNoiseBuffer,
    post_process_filter::PostProcessFilter,
    raytracing_pipeline::{RaytracingPipeline, RaytracingPushConstants, RaytracingPushData},
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

const _: () = assert!(std::mem::size_of::<UniformData>() % 4 == 0);

#[repr(C)]
pub struct UniformData {
    inverse_view: Mat4,
    inverse_projection: Mat4,
    pull_focus_x: u32,
    pull_focus_y: u32,
    aperture: f32,
    /// The environment this view is in: its camera rays start there (portals move them).
    environment: u32,
    // Flat arrays, not `Mat4`: the Slang side reads it scalar (4-byte aligned), and glam's
    // 16-byte alignment would pad here.
    view: [f32; 16],
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    jitter: [f32; 2],
    /// Free-running frame counter: the RNG seed, so every frame's noise is new for the
    /// temporal denoiser.
    frame: u32,
    /// Free: samples, max_bounces and the firefly clamp moved to the pipeline's specialization
    /// constants or went.
    pad_paths: [u32; 3],
    /// Post-process vignette strength (0 = off).
    vignette: f32,
    /// Entries in the emissive-triangle light table (0 = no light NEE / MIS).
    light_entries: u32,
    /// Free (restir_candidates, light_candidates).
    pad_candidates: [u32; 2],
    /// Added to every ray-cone texture LOD: log2(render / output width) + the panel's bias.
    lod_bias: f32,
    /// ReSTIR GI's sharing threshold (`RenderSettings::restir_gi_roughness`).
    gi_roughness: f32,
    /// Light-table generation; reservoirs from another generation are dropped.
    light_epoch: u32,
    /// Cap on temporal history, in candidate-samples.
    restir_m_clamp: f32,
    /// Free (the radiance cache went: ReSTIR GI does its job).
    pad_cache: [u32; 2],
    /// Ray-portal pair table (src/portal.rs): buffer address + entry count (0 = none).
    portals: u64,
    portal_count: u32,
    /// Uniform multiplier on every light's emission (from [`RenderSettings`]).
    emissive_boost: f32,
    /// The camera's render-layer cull mask (tlas_builder::layers_mask); rays start with
    /// it and portals swap it to the exit instance's mask.
    camera_mask: u32,
    /// Layers the camera ray skips ([`AuroraPrimaryHidden`](crate::environment::AuroraPrimaryHidden)).
    primary_hidden: u32,
    /// The mask bits that are environments; a portal ray takes the exit's from these.
    environment_bits: u32,
    /// Free (roulette_from).
    pad_roulette: u32,
    /// Every environment's sky, suns and fog (sky.rs `EnvironmentGpu`, 8 entries by bit): a
    /// ray reads the entry of the environment it is in.
    environments: u64,
    /// Fog volume table (src/fog_volume.rs): buffer address + entry count (0 = none).
    fog_volumes: u64,
    fog_volume_count: u32,
    /// Seconds since startup, for the fog volumes' drifting noise.
    time: f32,
    /// Fog light grid (src/fog_grid.rs): last frame's and this frame's cells (0 = off), the
    /// cell count, and 1 when last frame's is not this grid's history.
    fog_grid_prev: u64,
    fog_grid_cur: u64,
    fog_grid_cells: u32,
    fog_grid_fresh: u32,
    /// The grid's own view (a headset's eyes share one): where its cells are, this frame's
    /// view-projection (the camera ray looks the grid up) and last frame's (the update's
    /// history), and its cells across, down and deep.
    fog_grid_inverse_view: [f32; 16],
    fog_grid_inverse_projection: [f32; 16],
    fog_grid_view_proj: [f32; 16],
    fog_grid_prev_view_proj: [f32; 16],
    fog_grid_size: [u32; 3],
    /// Free (one_shadow_ray_from).
    pad_one_ray: u32,
    /// Bindless 3D images: the grid's columns summed front to back (in-scatter and
    /// transmittance per cell, behind a first slice of nothing) and the grid blurred once
    /// across x and y. A camera ray's lookups filter them in the texture unit.
    fog_grid_sum_tex: u32,
    fog_grid_soft_tex: u32,
    /// The soft grid as a buffer (`fog_integrate` reads it).
    fog_grid_soft: u64,
    /// The update lights one cell of every stride x stride block per frame ([`Fog::grid_stride`](crate::sky::Fog::grid_stride)).
    fog_grid_stride: u32,
    /// [`RenderSettings::ab`].
    ab: u32,
    pad_shared: [u32; 2],
}

#[repr(C)]
pub struct FocusData {
    focal_distance: f32,
}

/// What the view lane reads of the TLAS, taken while the scene lane may be rebuilding it.
#[derive(Clone, Copy)]
struct TlasRefs {
    /// A structure is built and the SBT is ready: the views can trace.
    traceable: bool,
    /// The structure's device address (the raygen's TLAS comes from push data).
    address: u64,
    instances: u64,
    prev_instances: u64,
    materials: u64,
    instance_slots: u64,
}

impl TlasRefs {
    fn of(tlas: &TLAS, sbt_ready: bool) -> Self {
        let handle = tlas.acceleration_structure.handle;
        Self {
            traceable: handle != vk::AccelerationStructureKHR::null() && sbt_ready,
            address: tlas.acceleration_structure.address,
            instances: tlas.instances_address(),
            prev_instances: tlas.prev_instances_address(),
            materials: tlas.material_address(),
            instance_slots: tlas.instance_slots_address(),
        }
    }
}

/// One rendered view this frame: XR renders one per eye, a flat frame renders one per
/// active camera in `Camera::order`. `slot` indexes the per-view resources (uniform buffer,
/// DLSS feature, reservoirs, its descriptor-heap blocks).
struct ViewFrame {
    slot: usize,
    /// Where this view's result goes.
    target: ViewTarget,
    /// The view's final resolution. Per view, not per frame: two cameras can have targets
    /// of completely different sizes, and this drives the reservoir extent, the DLSS plan
    /// and the texture LOD bias.
    output_extent: vk::Extent2D,
    /// The resolution rays are actually traced at -- below `output_extent` when DLSS is
    /// upscaling, equal to it otherwise.
    trace_extent: vk::Extent2D,
    /// `RenderLayers` as a bitmask: which instances this view can see.
    camera_mask: u32,
    /// Of those, the layers its camera ray skips.
    primary_hidden: u32,
    /// `AuroraDebugView`'s shader index for this view.
    debug_view: u32,
    inverse_view: Mat4,
    view_matrix: Mat4,
    projection: Mat4,
    view_proj: Mat4,
    last_view_proj: Mat4,
    plan: Option<crate::dlss::DlssPlan>,
    /// The camera this view traces for; its exposure meters under this entity.
    camera: Entity,
    exposure: crate::auto_exposure::AuroraExposure,
    lens: crate::render_settings::AuroraLens,
    /// This camera's (luminance, exposure state) addresses, set when its metering records.
    ae: (u64, u64),
    /// The camera's first view, the one that writes and resolves its meter (`ae.0`): a
    /// headset's two eyes share one, and a second resolve re-read the first one's output as
    /// albedo -- the meter read ~1 nit and the exposure opened up all the way.
    meters: bool,
    /// The Cost view's buffer (handle, address); null unless this view shows it.
    cost: (vk::Buffer, u64),
    /// The primary pass's hit records (`Frame::primary_hits`).
    primary_hits: u64,
    /// The composite's histories (`Frame::histories`): last frame's and this frame's sky mask,
    /// last frame's and this frame's fog veil, and whether last frame's are invalid.
    history: ([u64; 4], u32),
    /// The composite prepass's outputs (src/post_prep.rs).
    prep: crate::post_prep::PrepFrame,
    prep_textures: Option<crate::post_prep::PrepTextures>,
}

/// Where a view's composited image lands.
#[derive(Clone, Copy)]
enum ViewTarget {
    /// The window, at this rect. Unset `Camera::viewport` gives the whole window; a docked
    /// editor viewport gives its panel's rect.
    Window { rect: vk::Rect2D },
    /// An offscreen image (`RenderTarget::Image`). The asset id resolves to a texture in
    /// `VulkanAssets<Image>`, so whatever samples the handle sees this view's result.
    Image { asset: AssetId<Image> },
}

/// The renderer's teardown: every plugin that owns Vulkan objects destroys them here, and
/// [`on_shutdown`] (ordered last) drops the device. Run once by [`shutdown`].
#[derive(ScheduleLabel, PartialEq, Eq, Debug, Clone, Hash)]
pub struct TeardownSchedule;

/// The frame's stages, chained in [`Last`] after the simulation. One world: the systems in
/// [`RenderSet::Extract`] read the ECS directly (transform deltas, instance changes, UI quads,
/// asset events), [`RenderSet::Prepare`] consumes the asset worker's results, and
/// [`RenderSet::Render`] records, waits for the previous frame, latches and submits. All of
/// it runs while the previous frame may still be on the GPU, so nothing here writes memory a
/// submitted frame reads: per-frame host buffers come in parity pairs or `Flip`s, small data
/// goes through update-buffer commands, rewrites of shared tables go through
/// `run_transfer_commands_after_frames`, and results read back from the GPU are consumed only
/// after the wait (docs/plans/frame-pipeline.html). `Fence` is kept, empty, for ordering.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum RenderSet {
    Shutdown,
    Extract,
    Fence,
    Prepare,
    Render,
}

pub struct RayRenderPlugin;

impl Plugin for RayRenderPlugin {
    fn build(&self, app: &mut App) {
        // Absent without `WinitPlugin`: a windowless app renders to image targets only.
        let display_handle = app.world().get_resource::<DisplayHandleWrapper>();
        // Present when XrPlugin (added before this plugin) brought a runtime up: instance
        // and device creation then go through it (XR_KHR_vulkan_enable2).
        let xr_context = app.world().get_resource::<crate::xr::XrContext>();
        let render_device = unsafe {
            crate::render_device::RenderDevice::from_display(
                display_handle
                    .map(|handle| handle.0.display_handle().unwrap())
                    .as_ref(),
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
        app.insert_resource(crate::frame_sync::FrameSync::new(render_device.clone()));
        app.insert_resource(render_device);
        app.insert_resource(sphere_blas);
        app.init_resource::<Frame>();
        app.init_resource::<RenderConfig>();
        app.init_resource::<ScreenshotRequests>()
            .init_resource::<crate::util::video::VideoRecorder>();

        let mut teardown = Schedule::new(TeardownSchedule);
        teardown.add_systems(on_shutdown);
        app.add_schedule(teardown);

        app.configure_sets(
            Last,
            (
                RenderSet::Shutdown,
                RenderSet::Extract.run_if(render_device_exists),
                RenderSet::Fence.run_if(render_device_exists),
                RenderSet::Prepare.run_if(render_device_exists),
                RenderSet::Render.run_if(render_device_exists),
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
/// or the app is exiting. The swapchain has to go before its surface does, so the window
/// plugin's own close handling is off (`close_when_requested: false`) and a close request ends
/// here as an `AppExit` written in the same frame: winit exits right after this update, the
/// window entity is never despawned early, and no frame runs without a [`RenderDevice`].
/// (Despawning the window instead left it alive but unregistered for a frame, and every cursor
/// event the compositor still delivered to it logged a bevy_winit "unknown window" warning.)
/// Messages are peeked, not consumed, so bevy's own exit handling still sees them.
fn shutdown(world: &mut World) {
    if !world.contains_resource::<RenderDevice>() {
        return;
    }
    let closing = {
        let messages = world.resource::<Messages<WindowCloseRequested>>();
        messages.get_cursor().read(messages).next().is_some()
    };
    let exiting = {
        let messages = world.resource::<Messages<AppExit>>();
        messages.get_cursor().read(messages).next().is_some()
    };
    if !closing && !exiting {
        return;
    }
    shutdown_renderer(world);
    if !exiting {
        world.write_message(AppExit::Success);
    }
}

/// Tear the renderer down in order: the queue drains, then [`TeardownSchedule`] frees what every
/// plugin holds before the device goes. An app's exit does this; an app dropped without exiting
/// (a test's) calls it first, or the device is destroyed with its resources still alive.
pub fn shutdown_renderer(world: &mut World) {
    if !world.contains_resource::<RenderDevice>() {
        return;
    }
    log::debug!("Shutting down the renderer");
    {
        let render_device = world.resource::<RenderDevice>();
        let queue = render_device.queue.lock().unwrap();
        unsafe { render_device.queue_wait_idle(*queue).unwrap() };
    }
    world.run_schedule(TeardownSchedule);
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

/// The window rect a flat view occupies, in physical pixels.
///
/// `Camera::viewport` set means the host is placing this view somewhere specific -- an
/// editor's docked viewport, one pane of a split screen. Unset means the whole window.
/// The rect is clamped to the swapchain, because a UI-driven rect can lag a resize by a
/// frame and a viewport past the swapchain is a validation error.
fn view_rect(camera: Option<&Camera>, window: vk::Extent2D) -> vk::Rect2D {
    let full = vk::Rect2D::default().extent(window);
    let Some(viewport) = camera.and_then(|c| c.viewport.as_ref()) else {
        return full;
    };
    let max = UVec2::new(window.width, window.height);
    let origin = viewport.physical_position.min(max);
    let size = viewport.physical_size.min(max - origin);
    if size.x == 0 || size.y == 0 {
        return full;
    }
    vk::Rect2D::default()
        .offset(vk::Offset2D {
            x: origin.x as i32,
            y: origin.y as i32,
        })
        .extent(vk::Extent2D {
            width: size.x,
            height: size.y,
        })
}

#[derive(Resource, Default)]
pub struct Frame {
    pub swapchain_image: vk::Image,
    pub swapchain_view: vk::ImageView,
    /// One per view slot ([`crate::MAX_VIEWS`]): every trace dispatch in the frame needs
    /// its own matrices.
    pub uniform_buffers: [Buffer<UniformData>; crate::MAX_VIEWS],
    pub focus_data: Buffer<FocusData>,
    /// The per-environment sky / sun / fog table, rewritten each frame.
    pub environments: Buffer<EnvironmentGpu>,
    /// Per view slot, the Cost debug view's buffer: the frame's total ns (two words), then
    /// each traced pixel's. Made the first time a view shows it.
    pub cost_buffers: [Buffer<u32>; crate::MAX_VIEWS],
    /// Per view slot, the primary pass's hit record per traced pixel (`PrimaryHit` in
    /// rt_types.slang, 64 bytes): where the path tracer resumes sample 0.
    pub primary_hits: [Buffer<[u32; 16]>; crate::MAX_VIEWS],
    /// Per view slot, by frame parity, the composite's running averages at output resolution:
    /// local tone mapping's sky mask and the camera ray's fog veil. Each frame reads the other
    /// parity's (reprojected) and writes its own.
    pub sky_masks: [[Buffer<f32>; 2]; crate::MAX_VIEWS],
    pub fog_veils: [[Buffer<[f32; 4]>; 2]; crate::MAX_VIEWS],
}

fn render_frame(
    render_device: Res<crate::render_device::RenderDevice>,
    // The window and its swapchain are absent in a windowless app: the frame then traces and
    // composites into image targets and presents nothing.
    present: (
        Option<Res<RenderWindow>>,
        Option<ResMut<crate::swapchain::Swapchain>>,
        ResMut<crate::frame_sync::FrameSync>,
        Res<crate::latency::Latency>,
    ),
    dev_ui_stuff: (
        Option<Res<crate::render_settings::RenderSettings>>,
        crate::ui_render::UiDrawParams,
        Res<EnvironmentSkies>,
        crate::gizmo_render::GizmoDrawParams,
        Res<Atmosphere>,
        Res<CloudLayer>,
        Res<crate::sky_gen::SkyGen>,
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
        ResMut<ScreenshotRequests>,
        ResMut<crate::auto_exposure::AutoExposureState>,
        Res<Time>,
        ResMut<crate::skinning::Skins>,
        Res<crate::portal::PortalTable>,
        ResMut<AtmosphereState>,
        ResMut<crate::picking::Picker>,
        Res<crate::fog_volume::FogVolumeTable>,
        ResMut<crate::fog_grid::FogGridState>,
        ResMut<crate::compute::FrameCommands>,
    ),
    camera: Query<
        (
            Entity,
            &Projection,
            &GlobalTransform,
            Option<&crate::dlss::AuroraDlss>,
            Option<&crate::auto_exposure::AuroraExposure>,
            Option<&crate::debug_view::AuroraDebugView>,
            Option<&bevy::camera::visibility::RenderLayers>,
            // `Camera::viewport` is how a host says "put this view in that rect of the
            // window" -- an editor's docked viewport, or a split screen. None means the
            // whole window, which is what a plain app wants. `order` sorts the views and
            // `is_active` skips them.
            Option<&Camera>,
            Option<&crate::environment::InEnvironment>,
            Option<&crate::render_settings::AuroraLens>,
            Option<&crate::environment::AuroraPrimaryHidden>,
        ),
        With<Camera3d>,
    >,
    mut frame_counter: Local<u32>,
    dlss_stuff: (
        ResMut<crate::dlss::DlssState>,
        Local<[Option<Mat4>; crate::MAX_VIEWS]>,
        Local<bool>,
        // Rides along here because a system's parameter list stops at 16.
        Res<crate::camera_target::CameraTargets>,
        ResMut<crate::gpu_profiler::GpuProfiler>,
        ResMut<crate::util::video::VideoRecorder>,
        ResMut<crate::post_prep::PostPrepState>,
        ResMut<crate::restir_gi::RestirGiState>,
    ),
    mut xr: Option<ResMut<crate::xr::XrState>>,
) {
    let (window, mut swapchain, mut sync, latency) = present;
    if let Some(swapchain) = swapchain.as_deref() {
        latency.mark(
            &render_device,
            swapchain,
            vk::LatencyMarkerNV::RENDERSUBMIT_START,
        );
    }
    let present_id = latency.present_id().filter(|_| swapchain.is_some());
    // Last frame's window size: a view planned against it is clamped again at acquire.
    let window_extent = swapchain
        .as_ref()
        .map_or(vk::Extent2D::default(), |s| s.swapchain_extent);
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
        mut screenshots,
        mut ae,
        time,
        mut skins,
        portal_table,
        mut atmo,
        mut picker,
        fog_volumes,
        mut fog_grid,
        mut frame_commands,
    ) = gpu;
    let (
        mut dlss,
        mut prev_view_proj,
        mut dlss_was_active,
        camera_targets,
        mut profiler,
        mut video,
        mut post_prep,
        mut restir_gi,
    ) = dlss_stuff;

    let (settings, mut ui, skies, mut gizmos, atmosphere, clouds, sky_gen) = dev_ui_stuff;
    let local_tonemap = settings.as_ref().map_or(0.0, |s| s.local_tonemap);
    let bloom = settings.as_ref().map_or(0.0, |s| s.bloom);
    let tonemap = settings.as_ref().is_none_or(|s| s.tonemap);
    let settings = settings
        .map(|settings| settings.clone())
        .unwrap_or_default();
    *frame_counter = frame_counter.wrapping_add(1);
    // Every active Camera3d becomes a view, lowest `Camera::order` first -- so a camera
    // whose target another camera samples renders before it, which is what `order` means
    // everywhere else in bevy. Entity breaks ties because query iteration order is not
    // stable and a flickering slot assignment would throw away temporal history.
    // MAX_VIEWS caps it; the rest are dropped with a warning rather than silently.
    // Inactive cameras keep their metering; only a despawned (or no longer 3D) one drops it.
    let camera_entities: Vec<Entity> = camera.iter().map(|c| c.0).collect();
    let mut cameras: Vec<_> = camera
        .iter()
        .filter(|c| c.7.is_none_or(|cam| cam.is_active))
        .collect();
    cameras.sort_by_key(|c| (c.7.map_or(0, |cam| cam.order), c.0));
    if cameras.len() > crate::MAX_VIEWS {
        log::warn!(
            "{} active cameras but MAX_VIEWS is {}; rendering the {} lowest-order",
            cameras.len(),
            crate::MAX_VIEWS,
            crate::MAX_VIEWS,
        );
        cameras.truncate(crate::MAX_VIEWS);
    }
    // NO early return when there is no camera. An editor's project picker, a main menu and
    // a loading screen are UI with no scene to trace, and this function owns the whole
    // frame -- swapchain acquire, the UI pass, gizmos, present. Returning here drew
    // nothing at all, which looks exactly like a hung window. `views` simply ends up empty
    // and every scene-only stage below skips itself.
    let camera = cameras.first().copied();
    // XR: the camera entity's transform anchors the headset's LOCAL space in the world (the
    // fly-cam still moves it); each eye's pose and asymmetric fov come from the runtime.
    // Flat (and an XR session that produced no frame): the entity transform is the whole
    // camera and the window is the one view.
    let anchor = camera.map_or(Mat4::IDENTITY, |c| c.2.to_matrix());
    /// One camera's contribution before DLSS has had its say.
    struct PlannedView {
        slot: usize,
        inverse_view: Mat4,
        projection: Mat4,
        output_extent: vk::Extent2D,
        target: ViewTarget,
        dlss_mode: crate::dlss::AuroraDlss,
        camera_mask: u32,
        primary_hidden: u32,
        debug_view: u32,
        camera: Entity,
        exposure: crate::auto_exposure::AuroraExposure,
        lens: crate::render_settings::AuroraLens,
    }

    // `AuroraExposure::Environment`: the fixed exposure of the world the camera is in.
    let world_ev100 = |layers, world| {
        skies
            .of_view(crate::environment::environment_mask(layers, world))
            .ev100
    };
    let planned: Vec<PlannedView> = match (&xr_frame, xr.as_deref()) {
        // A headset with no camera has nothing to anchor to.
        (Some(_), _) if camera.is_none() => Vec::new(),
        // A headset is one camera's two eyes, not one view per camera: the primary camera
        // anchors the headset's local space and the runtime supplies each eye's pose.
        (Some(xr_frame), Some(xr_state)) => {
            let camera = camera.expect("guarded by the arm above");
            let near = match camera.1 {
                Projection::Perspective(perspective) => perspective.near,
                _ => 0.1,
            };
            (0..2)
                .map(|eye| {
                    let (head, projection) = xr_frame.eye_camera(eye, near);
                    PlannedView {
                        slot: eye,
                        inverse_view: anchor * head,
                        projection,
                        output_extent: xr_state.extent,
                        // The window shows a letterboxed eye; the headset's own swapchain
                        // is written separately by the XR path.
                        target: ViewTarget::Window {
                            rect: vk::Rect2D::default().extent(window_extent),
                        },
                        dlss_mode: camera.3.copied().unwrap_or_default(),
                        camera_mask: crate::environment::environment_mask(camera.6, camera.8)
                            as u32,
                        primary_hidden: camera.10.map_or(0, |h| h.mask() as u32),
                        debug_view: camera.5.copied().unwrap_or_default().shader_index(),
                        camera: camera.0,
                        exposure: camera
                            .4
                            .cloned()
                            .unwrap_or_default()
                            .resolve(world_ev100(camera.6, camera.8)),
                        lens: camera.9.copied().unwrap_or_default(),
                    }
                })
                .collect()
        }
        _ => cameras
            .iter()
            .enumerate()
            .filter_map(|(slot, c)| {
                let Projection::Perspective(perspective) = c.1 else {
                    // Orthographic and custom projections have no raygen yet; skipping is
                    // better than the `todo!()` that used to take the whole frame down.
                    log::warn!(
                        "aurora: only perspective cameras render; skipping {:?}",
                        c.0
                    );
                    return None;
                };
                // An offscreen camera is sized by its target texture; a window camera by
                // its rect. Tracing at the rect's size rather than the window's is the
                // whole point of the latter: a docked viewport costs its own pixels, not
                // the window's, and never pays for a copy into a UI texture.
                let (target, extent) = match camera_targets.by_camera.get(&c.0) {
                    Some((asset, size, _)) => (
                        ViewTarget::Image { asset: *asset },
                        vk::Extent2D {
                            width: size.x,
                            height: size.y,
                        },
                    ),
                    // No window to put it in: a windowless app renders image targets only.
                    None if swapchain.is_none() => return None,
                    None => {
                        let rect = view_rect(c.7, window_extent);
                        (ViewTarget::Window { rect }, rect.extent)
                    }
                };
                Some(PlannedView {
                    slot,
                    inverse_view: c.2.to_matrix(),
                    projection: bevy::math::proj::perspective_infinite_reverse(
                        perspective.fov,
                        extent.width.max(1) as f32 / extent.height.max(1) as f32,
                        perspective.near,
                    ),
                    output_extent: extent,
                    target,
                    dlss_mode: c.3.copied().unwrap_or_default(),
                    camera_mask: crate::environment::environment_mask(c.6, c.8) as u32,
                    primary_hidden: c.10.map_or(0, |h| h.mask() as u32),
                    debug_view: c.5.copied().unwrap_or_default().shader_index(),
                    camera: c.0,
                    exposure: c
                        .4
                        .cloned()
                        .unwrap_or_default()
                        .resolve(world_ev100(c.6, c.8)),
                    lens: c.9.copied().unwrap_or_default(),
                })
            })
            .collect(),
    };

    // DLSS: settle each view's trace resolution and jitter before recording (feature
    // creation, when the mode or output size changed, submits and waits on its own).
    let reset_requested = std::mem::take(&mut dlss.reset_requested);
    let mut views: Vec<ViewFrame> = planned
        .into_iter()
        .map(|p| {
            let PlannedView {
                slot,
                inverse_view,
                projection,
                output_extent,
                target,
                dlss_mode,
                camera_mask,
                primary_hidden,
                debug_view,
                camera,
                exposure,
                lens,
            } = p;
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
                target,
                output_extent,
                trace_extent: plan.map_or(output_extent, |p| p.render),
                camera_mask,
                primary_hidden,
                debug_view,
                inverse_view,
                view_matrix,
                projection,
                view_proj,
                last_view_proj,
                plan,
                camera,
                exposure,
                lens,
                ae: (0, 0),
                meters: true,
                cost: (vk::Buffer::null(), 0),
                primary_hits: 0,
                history: ([0; 4], 1),
                prep: Default::default(),
                prep_textures: None,
            }
        })
        .collect();
    let plan = views.first().and_then(|v| v.plan);
    let dlss_reset = plan.is_some() && (!*dlss_was_active || reset_requested);
    *dlss_was_active = plan.is_some();

    // Ensure the per-view uniform buffers exist: device-local, written by update-buffer commands
    // recorded at the head of the frame (so recording never touches memory the GPU reads).
    for buffer in &mut frame.uniform_buffers {
        if buffer.handle == vk::Buffer::null() {
            *buffer = render_device.create_device_buffer(
                1,
                vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
        }
    }

    // The Cost and GI views' buffers (one word per traced pixel), at their view's trace size.
    for view in &mut views {
        use crate::debug_view::AuroraDebugView;
        if ![
            AuroraDebugView::Cost,
            AuroraDebugView::Gi,
            AuroraDebugView::GiReuse,
        ]
        .iter()
        .any(|v| v.shader_index() == view.debug_view)
        {
            continue;
        }
        let buffer = &mut frame.cost_buffers[view.slot];
        // Two words per pixel: the GI views' light and state (the Cost view uses the first half).
        let len = 2 + 2 * view.trace_extent.width as u64 * view.trace_extent.height as u64;
        if buffer.nr_elements != len {
            render_device.destroyer.destroy_buffer(buffer.handle);
            *buffer = render_device.create_device_buffer(
                len,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
        }
        view.cost = (buffer.handle, buffer.address);
    }

    // The primary pass's hit records, at each view's trace size (docs/plans/primary-pass.html).
    for view in &mut views {
        let buffer = &mut frame.primary_hits[view.slot];
        let len = view.trace_extent.width as u64 * view.trace_extent.height as u64;
        if buffer.nr_elements != len {
            render_device.destroyer.destroy_buffer(buffer.handle);
            *buffer = render_device.create_device_buffer(len, vk::BufferUsageFlags::STORAGE_BUFFER);
        }
        view.primary_hits = buffer.address;
    }

    // The composite's histories, at each view's output size.
    for view in &mut views {
        let len = view.output_extent.width as u64 * view.output_extent.height as u64;
        let mut reset = 0;
        for buffer in frame.sky_masks[view.slot].iter_mut() {
            if buffer.nr_elements != len {
                render_device.destroyer.destroy_buffer(buffer.handle);
                *buffer =
                    render_device.create_device_buffer(len, vk::BufferUsageFlags::STORAGE_BUFFER);
                reset = 1;
            }
        }
        for buffer in frame.fog_veils[view.slot].iter_mut() {
            if buffer.nr_elements != len {
                render_device.destroyer.destroy_buffer(buffer.handle);
                *buffer =
                    render_device.create_device_buffer(len, vk::BufferUsageFlags::STORAGE_BUFFER);
                reset = 1;
            }
        }
        let cur = (*frame_counter & 1) as usize;
        let (sky, fog) = (&frame.sky_masks[view.slot], &frame.fog_veils[view.slot]);
        view.history = (
            [
                sky[1 - cur].address,
                sky[cur].address,
                fog[1 - cur].address,
                fog[cur].address,
            ],
            reset,
        );
    }

    for view in &mut views {
        let (prep, textures) = post_prep.ensure(
            &render_device,
            view.slot,
            view.output_extent,
            view.trace_extent,
        );
        view.prep = prep;
        view.prep_textures = Some(textures);
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

    // Every environment's sky, suns and fog, by environment bit.
    if frame.environments.handle == vk::Buffer::null() {
        frame.environments = render_device.create_device_buffer(
            8,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
    }
    let texture_index = |image: &Handle<Image>| {
        textures.get(image).map_or(WHITE_TEXTURE_IDX, |t| {
            render_device.register_bindless_texture(t)
        })
    };
    let envs: Vec<EnvironmentGpu> = skies
        .environments
        .iter()
        .enumerate()
        .map(|(bit, environment)| {
            let importance_address = sky_gen.table(bit);
            let (tex, space) = match &environment.sky {
                Sky::Hdr { image, .. } => (texture_index(image), Vec4::ZERO),
                Sky::Generated => (sky_gen.tex(bit).unwrap_or(WHITE_TEXTURE_IDX), Vec4::ZERO),
                // The space image behind the air (colour = its scale; 0 = none).
                Sky::Atmosphere => match &atmosphere.space {
                    Some(image) => (texture_index(image), Vec4::splat(atmosphere.space_scale)),
                    None => (WHITE_TEXTURE_IDX, Vec4::ZERO),
                },
                _ => (WHITE_TEXTURE_IDX, Vec4::ZERO),
            };
            EnvironmentGpu::new(
                environment,
                tex,
                space,
                importance_address,
                environment
                    .air
                    .stars
                    .as_ref()
                    .and_then(|stars| textures.get(stars))
                    .map(|t| render_device.register_bindless_texture(t)),
            )
        })
        .collect();
    // One fog grid per camera, in its first view's slot: a headset's two eyes share theirs.
    let mut grids: Vec<(usize, Entity, crate::fog_grid::FogGridFrame)> = Vec::new();
    for view in &views {
        if grids.iter().any(|(_, camera, _)| *camera == view.camera) {
            continue;
        }
        let environment = skies.view_environment(view.camera_mask as u8);
        let fog = skies.environments[environment as usize].fog;
        let eyes: Vec<(Mat4, Mat4)> = views
            .iter()
            .filter(|v| v.camera == view.camera)
            .map(|v| (v.inverse_view, v.projection))
            .collect();
        let grid = fog_grid.ensure(
            &render_device,
            view.trace_extent,
            fog.grid_tile,
            fog.grid_slices,
            fog.grid_stride,
            crate::fog_grid::grid_view(&eyes),
            *frame_counter,
            view.slot,
            fog.grid_distance > 0.0
                && fog.steps > 0
                && (fog.density > 0.0 || fog_volumes.count() > 0),
        );
        grids.push((view.slot, view.camera, grid));
    }
    for slot in 0..crate::MAX_VIEWS {
        if !grids.iter().any(|(owner, _, _)| *owner == slot) {
            fog_grid.release(&render_device, slot);
        }
    }

    // Update each view's uniform buffer
    let mut uniform_uploads: Vec<(usize, UniformData)> = Vec::with_capacity(views.len());
    for view in &views {
        let environment = skies.view_environment(view.camera_mask as u8);
        let grid = grids
            .iter()
            .find(|(_, camera, _)| *camera == view.camera)
            .map(|(_, _, grid)| *grid)
            .unwrap_or_default();
        let data = UniformData {
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
            aperture: view.lens.aperture,
            environment: environment as u32,
            view: view.view_matrix.to_cols_array(),
            view_proj: view.view_proj.to_cols_array(),
            prev_view_proj: view.last_view_proj.to_cols_array(),
            jitter: view.plan.map_or([0.0; 2], |p| p.jitter),
            frame: *frame_counter,
            pad_paths: [0; 3],
            vignette: view.lens.vignette,
            light_entries: if settings.light_nee {
                lights.active_entries
            } else {
                0
            },
            pad_candidates: [0; 2],
            // DLSS guide 3.5: textures are picked for the OUTPUT resolution, not the trace
            // resolution, so the upscaled image keeps its detail.
            lod_bias: (view.trace_extent.width.max(1) as f32
                / view.output_extent.width.max(1) as f32)
                .log2()
                + settings.texture_lod_bias,
            gi_roughness: settings.restir_gi_roughness.clamp(0.0, 1.0),
            light_epoch: lights.epoch,
            restir_m_clamp: (settings.restir_candidates as f32 * settings.restir_history).max(1.0),
            pad_cache: [0; 2],
            emissive_boost: settings.emissive_boost.max(0.0),
            portals: portal_table.address(),
            portal_count: portal_table.count(),
            camera_mask: view.camera_mask,
            primary_hidden: view.primary_hidden,
            environment_bits: skies.environment_bits as u32,
            pad_roulette: 0,
            environments: frame.environments.address,
            fog_volumes: fog_volumes.address(),
            fog_volume_count: fog_volumes.count(),
            time: time.elapsed_secs_wrapped(),
            fog_grid_prev: grid.prev,
            fog_grid_cur: grid.cur,
            fog_grid_cells: grid.cells,
            fog_grid_fresh: grid.fresh as u32,
            fog_grid_inverse_view: grid.inverse_view.to_cols_array(),
            fog_grid_inverse_projection: grid.projection.inverse().to_cols_array(),
            fog_grid_view_proj: grid.view_proj.to_cols_array(),
            fog_grid_prev_view_proj: grid.prev_view_proj.to_cols_array(),
            fog_grid_size: grid.size.to_array(),
            pad_one_ray: 0,
            fog_grid_sum_tex: grid.sum_tex,
            fog_grid_soft_tex: grid.soft_tex,
            fog_grid_soft: grid.soft,
            fog_grid_stride: grid.stride,
            ab: settings.ab,
            pad_shared: [0; 2],
        };

        uniform_uploads.push((view.slot, data));
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
        // Record first, wait after: this frame's commands are recorded while the previous frame
        // still runs on the GPU. Recording writes only memory no submitted frame reads (parity
        // copies, update-buffer commands, fresh buffers); anything computed from the previous
        // frame's results is latched after the wait below, so nothing read back gets older.
        profiler.begin_recording(sync.frame_count);
        // The window's image for this frame, with its size and format, when there is one.
        let presenting = match (swapchain.as_deref_mut(), window.as_deref()) {
            (Some(swapchain), Some(window)) => {
                let (image, view) = swapchain.aquire_next_image(window, sync.frame_count);
                Some((
                    image,
                    view,
                    swapchain.swapchain_extent,
                    swapchain.swapchain_format,
                ))
            }
            _ => None,
        };
        // Three command buffers, recorded on three threads and submitted in this order: the
        // scene (main pool: UI surfaces, the transform-driven chain, lights), the views
        // (exposure, atmosphere, each view's trace and DLSS) and the present pass.
        let parity = sync.frame_count % 2;
        let frame_count = sync.frame_count;
        let cmd_buffer = render_device.command_buffers[parity];
        let view_cmd = render_device.lanes[0].1[parity];
        let present_cmd = render_device.lanes[1].1[parity];
        let tlas_cmd = render_device.lanes[2].1[parity];
        let view_back_cmd = render_device.lanes[3].1[parity];

        if let Some((image, view, _, _)) = presenting {
            frame.swapchain_image = image;
            frame.swapchain_view = view;
        }
        let record = info_span!("record_frame").entered();

        for cmd in [cmd_buffer, view_cmd, present_cmd, tlas_cmd, view_back_cmd] {
            render_device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            render_device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        let prof = &*profiler;

        // Offscreen UI surfaces (world panels) render first, so this frame's trace samples
        // this frame's UI.
        // Submitted ahead of the CPU's wait for the previous frame: nothing here starts before
        // that frame is done on the queue (uniforms, the radiance cache, the guides are shared).
        crate::compute::memory_barrier(
            &render_device,
            cmd_buffer,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        let frame_scope = prof.begin(&render_device, cmd_buffer, "frame");
        let section = info_span!("record_ui_surfaces").entered();
        let scope = prof.begin(&render_device, cmd_buffer, "ui surfaces");
        crate::ui_render::draw_ui_surfaces(&render_device, cmd_buffer, parity, &mut ui);
        prof.end(&render_device, cmd_buffer, scope);
        drop(section);
        // Transforms go first, in a submit of their own: their rows are read back into
        // `GlobalTransform` at the next `First` (gpu_transform.rs).
        let section = info_span!("submit_transforms").entered();
        let world_changed = transforms.submit(&render_device, &modules);
        drop(section);
        // With rays to pick, the instance table and TLAS go into their own command buffer,
        // submitted ahead of the rest with the pick at its end (picking.rs).
        let pick = picker.active(&modules, &tlas);
        // Async compute (docs/plans/async-compute.html): skins with their BLAS refits and the
        // radiance cache resolve run on the compute queue beside the frame commands; the TLAS,
        // lights and everything after wait for them. A pick submits the TLAS ahead of the frame,
        // so with one everything stays on the graphics queue.
        let compute_cmd = render_device
            .compute
            .as_ref()
            .map(|c| c.command_buffers[parity]);
        if let Some(cmd) = compute_cmd {
            render_device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            render_device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        // ReSTIR's light grids on the compute queue, beside the first view's primary pass.
        let grid_cmd = render_device
            .compute
            .as_ref()
            .map(|c| c.grid_buffers[parity]);
        if let Some(cmd) = grid_cmd {
            render_device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            render_device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        // The scene lane's second half: after the async compute when there is one.
        let after_compute = if compute_cmd.is_some() {
            tlas_cmd
        } else {
            cmd_buffer
        };
        // Exposure: each camera meters its own last frame into this frame's exposure (the
        // raygen reads it). A headset's two eyes are one camera and share one meter. Recorded
        // here, at the head of the view lane, because the present lane needs the addresses.
        let section = info_span!("record_exposure_atmo").entered();
        let exposure_cmd = compute_cmd.unwrap_or(view_cmd);
        let readers = if compute_cmd.is_some() {
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
        } else {
            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
        };
        let scope = prof.begin(&render_device, exposure_cmd, "exposure");
        let mut metered: Vec<(Entity, (u64, u64))> = Vec::new();
        for view in &mut views {
            if let Some((_, addresses)) = metered.iter().find(|(c, _)| *c == view.camera) {
                view.ae = *addresses;
                view.meters = false;
                continue;
            }
            view.ae = ae.record(
                &render_device,
                exposure_cmd,
                &modules,
                view.camera,
                view.trace_extent,
                &view.exposure,
                time.delta_secs(),
                readers,
            );
            metered.push((view.camera, view.ae));
        }
        prof.end(&render_device, exposure_cmd, scope);
        let scope = prof.begin(&render_device, view_cmd, "atmosphere");
        ae.set_primary(views.first().map(|v| v.camera));
        ae.retain(&render_device, |e| camera_entities.contains(&e));
        // The atmosphere's LUTs for this frame's camera altitude and sun (the raygen and
        // the miss shader read them); nothing unless the sky is the atmosphere.
        atmo.record(
            &render_device,
            view_cmd,
            &modules,
            &skies,
            &atmosphere,
            &clouds,
            camera.map_or(Vec3::ZERO, |c| c.2.translation()),
            time.delta_secs(),
        );
        prof.end(&render_device, view_cmd, scope);
        drop(section);

        // What the lanes read of each other's state, taken before they start.
        let light_entries = lights.active_entries;
        let light_epoch = lights.epoch;
        let light_header = lights.header_address();
        let atmo_address = atmo.address();
        let rtx_pipeline = rtx_pipelines.get(&render_config.rtx_pipeline);
        let sbt_ready = sbt.data().address != 0;
        let dlss_outputs: [(Option<vk::ImageView>, Option<crate::dlss::GuideViews>);
            crate::MAX_VIEWS] = std::array::from_fn(|slot| {
            dlss.renderer
                .as_ref()
                .map_or((None, None), |r| (r.output_view(slot), r.guide_views(slot)))
        });
        let has_dlss = dlss.renderer.is_some();
        let heaps = render_device.heaps();
        let delta_secs = time.delta_secs();
        let frame: &Frame = &frame;
        let views = &views;
        let grids = &grids;
        let settings = &settings;

        // The scene lane: this frame's per-view uniforms and environment table, the queued
        // dispatches, skins, the instance table and TLAS (and the pick), then the light table
        // and the radiance cache resolve.
        let mut scene = |tlas: &mut TLAS| {
            let _lane = info_span!("record_scene").entered();
            let scene_cmd = if pick {
                picker.begin(&render_device, frame_count)
            } else {
                cmd_buffer
            };
            // Ahead of everything that reads them (sky generation's `sun_tint` then writes the
            // sun into the environment table).
            render_device.cmd_update_buffer(
                scene_cmd,
                frame.environments.handle,
                0,
                bytemuck::cast_slice(&envs),
            );
            for (slot, data) in &uniform_uploads {
                render_device.cmd_update_buffer(
                    scene_cmd,
                    frame.uniform_buffers[*slot].handle,
                    0,
                    // SAFETY: `UniformData` is `repr(C)` plain data; update-buffer needs a size
                    // that is a multiple of 4.
                    std::slice::from_raw_parts(
                        (data as *const UniformData).cast::<u8>(),
                        std::mem::size_of::<UniformData>(),
                    ),
                );
            }
            crate::compute::memory_barrier(
                &render_device,
                scene_cmd,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::UNIFORM_READ,
            );
            // Last frame's instance rows kept for motion vectors, then the queued per-frame
            // dispatches (landscape scatter writes instance rows), ahead of the TLAS build.
            let section = info_span!("record_frame_commands").entered();
            let scope = prof.begin(&render_device, scene_cmd, "frame commands");
            tlas.snapshot_prev(&render_device, scene_cmd);
            frame_commands.record(&render_device, &modules, scene_cmd);
            prof.end(&render_device, scene_cmd, scope);
            drop(section);
            crate::gpu_transform::world_barrier(&render_device, scene_cmd);
            let section = info_span!("record_skins").entered();
            let skin_cmd = compute_cmd.unwrap_or(scene_cmd);
            let scope = prof.begin(&render_device, skin_cmd, "skins");
            let skinned = skins.record(&render_device, skin_cmd, &modules, &transforms);
            prof.end(&render_device, skin_cmd, scope);
            drop(section);
            if let Some(compute) = compute_cmd {
                render_device.end_command_buffer(compute).unwrap();
            }
            // With a pick the early submit splits: its head (frame commands) beside the compute
            // work, its tail (TLAS, pick) after it. Without one the frame's own submits do.
            let mut compute_wait = Vec::new();
            let scene_cmd = match compute_cmd {
                Some(compute) if pick => {
                    let before = render_device.timeline.last_submitted();
                    let tail = picker.split(&render_device, scene_cmd, frame_count);
                    let point = render_device.submit_compute(compute, before);
                    compute_wait.push((
                        render_device.compute.as_ref().unwrap().timeline.semaphore,
                        point,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                    ));
                    tail
                }
                Some(_) => after_compute,
                None => scene_cmd,
            };
            let section = info_span!("record_tlas").entered();
            let scope = prof.begin(&render_device, scene_cmd, "tlas");
            tlas.record(
                &render_device,
                scene_cmd,
                &modules,
                &transforms,
                world_changed || skinned,
                true,
            );
            prof.end(&render_device, scene_cmd, scope);
            drop(section);
            if pick {
                let _submit = info_span!("submit_pick").entered();
                picker.submit(&render_device, scene_cmd, &modules, tlas, &compute_wait);
            }
            // The light table's weight/CDF kernels, whenever the light set changed (they read
            // the instance rows the gather above wrote).
            let section = info_span!("record_lights").entered();
            let scope = prof.begin(&render_device, after_compute, "lights");
            lights.record(&render_device, after_compute, &modules, tlas);
            prof.end(&render_device, after_compute, scope);
            drop(section);
            render_device.end_command_buffer(cmd_buffer).unwrap();
            render_device.end_command_buffer(tlas_cmd).unwrap();
        };

        // The view lane: each view's guide descriptors, ReSTIR, fog grid, trace and DLSS.
        // Returns the cameras whose trace filled their luminance buffer.
        let mut view_lane = |tlas: TlasRefs| -> (Vec<Entity>, bool) {
            let _lane = info_span!("record_trace_dlss").entered();
            let mut traced = Vec::new();
            // The view lane's command buffer: its front half until the first view's primary
            // pass, then the back half that waits on the light grids (async compute).
            let cmd = std::cell::Cell::new(view_cmd);
            let mut grid_split = false;
            if let (Some(rtx_pipeline), true) = (rtx_pipeline, tlas.traceable) {
                static TRACED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !TRACED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::info!("first frame traced");
                }
                // ReSTIR DI's buffers, parameters and light grid, every view's first: on the async
                // compute queue (a submit of its own, beside the first view's fog grid and primary
                // pass) when there is one. A view whose buffers were just made runs without ReSTIR
                // this frame there: their clears would race the primary pass's writes.
                let mut restir_views =
                    [(false, crate::restir::RestirAddresses::default(), false); crate::MAX_VIEWS];
                for view in views {
                    let restir_on = settings.restir
                        && settings.light_nee
                        && light_entries > 0
                        && view.plan.is_some();
                    if !restir_on {
                        restir.release(&render_device, view.slot);
                        continue;
                    }
                    let addresses = restir.prepare(
                        &render_device,
                        grid_cmd.unwrap_or(view_cmd),
                        &crate::restir::RestirInput {
                            extent: view.trace_extent,
                            frame: *frame_counter,
                            slot: view.slot,
                            prev_view_proj: view.last_view_proj,
                            camera: view.inverse_view.w_axis.truncate(),
                            epoch: light_epoch,
                            lights: light_header,
                            instances: tlas.instances,
                            light_entries,
                            settings,
                        },
                    );
                    grid_split |= grid_cmd.is_some();
                    let on = !(addresses.fresh && grid_cmd.is_some());
                    let built = on
                        && restir.record_grid(
                            &render_device,
                            grid_cmd.unwrap_or(view_cmd),
                            &modules,
                            view.slot,
                            prof,
                        );
                    restir_views[view.slot] = (on, addresses, built);
                }
                // ReSTIR GI's buffers and parameters, beside DI's (same queue, same rule: buffers
                // just made skip GI this frame). It shares DI's surfaces when DI runs.
                let mut gi_views =
                    [(false, crate::restir_gi::GiAddresses::default()); crate::MAX_VIEWS];
                for view in views {
                    if !(settings.restir_gi && view.plan.is_some()) {
                        restir_gi.release(&render_device, view.slot);
                        continue;
                    }
                    let (di_on, di_addresses, _) = restir_views[view.slot];
                    let addresses = restir_gi.prepare(
                        &render_device,
                        grid_cmd.unwrap_or(view_cmd),
                        &crate::restir_gi::GiInput {
                            extent: view.trace_extent,
                            frame: *frame_counter,
                            slot: view.slot,
                            di_surfaces: if di_on { di_addresses.surfaces } else { 0 },
                            settings,
                        },
                    );
                    grid_split |= grid_cmd.is_some();
                    gi_views[view.slot] = (!(addresses.fresh && grid_cmd.is_some()), addresses);
                }
                render_device.cmd_bind_pipeline(
                    view_cmd,
                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                    rtx_pipeline.pipeline,
                );

                for view in views {
                    // This view's guides, in its block of the heap for this frame parity (a
                    // block already recorded this frame must not be rewritten).
                    let guides = DescriptorHeaps::guide_block(parity, view.slot);
                    if let Some(g) = dlss_outputs[view.slot].1.filter(|_| view.plan.is_some()) {
                        heaps.write_guides(
                            guides,
                            &[
                                g.normal_roughness,
                                g.diffuse,
                                g.specular,
                                g.depth,
                                g.spec_hit,
                                g.motion,
                                g.color,
                                g.fog,
                            ],
                        );
                    }
                    // Heaps and push data, again after anything that cleared them (compute
                    // dispatches push constants; NGX binds its own sets).
                    let push_trace = |constants: RaytracingPushConstants| {
                        heaps.bind(cmd.get());
                        heaps.push(
                            cmd.get(),
                            &RaytracingPushData {
                                constants,
                                tlas: tlas.address,
                                guides,
                                pad: 0,
                            },
                        );
                    };

                    // ReSTIR DI's buffers and this frame's parameters (none while it is off).
                    let (restir_on, restir_addresses, grid_built) = restir_views[view.slot];
                    let restir_addresses = if restir_on {
                        restir_addresses
                    } else {
                        crate::restir::RestirAddresses::default()
                    };
                    let (gi_on, gi_addresses) = gi_views[view.slot];
                    let gi_addresses = if gi_on {
                        gi_addresses
                    } else {
                        crate::restir_gi::GiAddresses::default()
                    };
                    fog_grid.record_clear(&render_device, cmd.get(), view.slot);
                    // The Cost view's frame total, summed into by the raygen; the GI views'
                    // whole buffer (a frame without GI shows nothing, not a stale one).
                    if view.cost.1 != 0 {
                        let gi_view = view.debug_view
                            >= crate::debug_view::AuroraDebugView::Gi.shader_index();
                        render_device.device.cmd_fill_buffer(
                            cmd.get(),
                            view.cost.0,
                            0,
                            if gi_view { vk::WHOLE_SIZE } else { 8 },
                            0,
                        );
                        crate::compute::memory_barrier(
                            &render_device,
                            cmd.get(),
                            vk::PipelineStageFlags2::ALL_TRANSFER,
                            vk::AccessFlags2::TRANSFER_WRITE,
                            vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                        );
                    }
                    let push_constants = RaytracingPushConstants {
                        uniform_buffer: frame.uniform_buffers[view.slot].address,
                        material_buffer: tlas.materials,
                        bluenoise_buffer2: bluenoise_buffer.0.address,
                        focus_buffer: frame.focus_data.address,
                        restir_surfaces: if restir_addresses.surfaces != 0 {
                            restir_addresses.surfaces
                        } else {
                            gi_addresses.surfaces
                        },
                        prev_instances: tlas.prev_instances,
                        lights: light_header,
                        instances: tlas.instances,
                        reservoirs_prev: restir_addresses.prev,
                        reservoirs_cur: restir_addresses.final_,
                        lum_buffer: if view.meters { view.ae.0 } else { 0 },
                        auto_exposure: view.ae.1,
                        atmo: atmo_address,
                        restir_frame: restir_addresses.frame,
                        instance_slots: tlas.instance_slots,
                        cost: if view.debug_view
                            == crate::debug_view::AuroraDebugView::Cost.shader_index()
                        {
                            view.cost.1
                        } else {
                            0
                        },
                        primary_hits: view.primary_hits,
                        gi_candidates: gi_addresses.candidates,
                        gi_final: gi_addresses.final_,
                        gi_debug: if view.debug_view
                            >= crate::debug_view::AuroraDebugView::Gi.shader_index()
                        {
                            view.cost.1
                        } else {
                            0
                        },
                    };

                    push_trace(push_constants);

                    // Guides into GENERAL before the first dispatch: the fog grid runs with
                    // this set bound, so validation checks their layout there too.
                    if let (Some(_), Some(renderer)) = (view.plan, dlss.renderer.as_mut()) {
                        renderer.record_pre_trace(&render_device, cmd.get(), view.slot);
                    }

                    // The camera's fog grid, by the first of its views: lit before any of them
                    // marches it (a headset's second eye reuses it). History carried forward
                    // (compute), this frame's cells lit (raygen), blurred and summed (compute).
                    if let Some((_, _, grid)) = grids
                        .iter()
                        .find(|(owner, _, grid)| *owner == view.slot && grid.cells > 0)
                    {
                        let uniforms = frame.uniform_buffers[view.slot].address;
                        let scope = prof.begin(&render_device, cmd.get(), "fog grid");
                        if fog_grid.record_reproject(
                            &render_device,
                            cmd.get(),
                            &modules,
                            uniforms,
                            grid,
                        ) {
                            push_trace(push_constants);
                            let stride = grid.stride.max(1);
                            // Its own ray stack (the driver's default sizes every dispatch for the largest raygen).
                            render_device
                                .ext_rtx_pipeline
                                .cmd_set_ray_tracing_pipeline_stack_size(
                                    cmd.get(),
                                    rtx_pipeline.stack_fog_grid,
                                );
                            render_device.ext_rtx_pipeline.cmd_trace_rays(
                                cmd.get(),
                                &sbt.fog_grid_region,
                                &sbt.miss_region,
                                &sbt.hit_region,
                                &vk::StridedDeviceAddressRegionKHR::default(),
                                grid.size.x.div_ceil(stride),
                                grid.size.y.div_ceil(stride),
                                grid.size.z,
                            );
                            fog_grid.record_integrate(
                                &render_device,
                                cmd.get(),
                                &modules,
                                uniforms,
                                grid,
                                view.slot,
                            );
                            push_trace(push_constants);
                        }
                        prof.end(&render_device, cmd.get(), scope);
                    }

                    if let (Some(plan), Some(renderer)) = (view.plan, dlss.renderer.as_mut()) {
                        // The light grid the trace's path vertices (and ReSTIR's passes) draw
                        // from; without it the vertices keep the power CDF.
                        // A grid that did not build leaves the vertices on the power CDF.
                        let trace_constants = RaytracingPushConstants {
                            restir_frame: if grid_built {
                                push_constants.restir_frame
                            } else {
                                0
                            },
                            ..push_constants
                        };
                        push_trace(trace_constants);
                        // The primary pass: sample 0's camera chain and its hit record, which the
                        // trace resumes from (docs/plans/primary-pass.html).
                        if view.primary_hits != 0 {
                            let scope = prof.begin(&render_device, cmd.get(), "primary");
                            // Its own ray stack (the driver's default sizes every dispatch for the largest raygen).
                            render_device
                                .ext_rtx_pipeline
                                .cmd_set_ray_tracing_pipeline_stack_size(
                                    cmd.get(),
                                    rtx_pipeline.stack_primary,
                                );
                            render_device.ext_rtx_pipeline.cmd_trace_rays(
                                cmd.get(),
                                &sbt.primary_region,
                                &sbt.miss_region,
                                &sbt.hit_region,
                                &vk::StridedDeviceAddressRegionKHR::default(),
                                plan.render.width,
                                plan.render.height,
                                1,
                            );
                            prof.end(&render_device, cmd.get(), scope);
                            crate::compute::memory_barrier(
                                &render_device,
                                cmd.get(),
                                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                                vk::AccessFlags2::SHADER_WRITE,
                                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
                                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                            );
                            // The rest waits on the light grids (async compute): the back half.
                            if grid_split && cmd.get() == view_cmd {
                                cmd.set(view_back_cmd);
                                render_device.cmd_bind_pipeline(
                                    view_back_cmd,
                                    vk::PipelineBindPoint::RAY_TRACING_KHR,
                                    rtx_pipeline.pipeline,
                                );
                                push_trace(trace_constants);
                            }
                            // ReSTIR DI between the two: its passes over the primary pass's
                            // surfaces; the trace shades the winners.
                            if restir_on
                                && restir.record_passes(
                                    &render_device,
                                    cmd.get(),
                                    &modules,
                                    view.slot,
                                    prof,
                                )
                            {
                                push_trace(trace_constants);
                            }
                        }
                        let scope = prof.begin(&render_device, cmd.get(), "trace");
                        // Its own ray stack (the driver's default sizes every dispatch for the largest raygen).
                        render_device
                            .ext_rtx_pipeline
                            .cmd_set_ray_tracing_pipeline_stack_size(
                                cmd.get(),
                                rtx_pipeline.stack_main,
                            );
                        render_device.ext_rtx_pipeline.cmd_trace_rays(
                            cmd.get(),
                            &sbt.raygen_region,
                            &sbt.miss_region,
                            &sbt.hit_region,
                            &vk::StridedDeviceAddressRegionKHR::default(),
                            plan.render.width,
                            plan.render.height,
                            1,
                        );
                        prof.end(&render_device, cmd.get(), scope);
                        // ReSTIR GI: the trace's candidates through its reuse passes, then the
                        // winners shaded into the colour.
                        if gi_on
                            && restir_gi.record_passes(
                                &render_device,
                                cmd.get(),
                                &modules,
                                view.slot,
                                prof,
                            )
                        {
                            push_trace(trace_constants);
                            let scope = prof.begin(&render_device, cmd.get(), "gi shade");
                            render_device
                                .ext_rtx_pipeline
                                .cmd_set_ray_tracing_pipeline_stack_size(
                                    cmd.get(),
                                    rtx_pipeline.stack_gi_shade,
                                );
                            render_device.ext_rtx_pipeline.cmd_trace_rays(
                                cmd.get(),
                                &sbt.gi_shade_region,
                                &sbt.miss_region,
                                &sbt.hit_region,
                                &vk::StridedDeviceAddressRegionKHR::default(),
                                plan.render.width,
                                plan.render.height,
                                1,
                            );
                            prof.end(&render_device, cmd.get(), scope);
                        }
                        // NGX knows nothing of heaps (descriptor_heap.rs `leave`).
                        heaps.leave(&render_device, cmd.get());
                        let scope = prof.begin(&render_device, cmd.get(), "dlss");
                        renderer.record_evaluate(
                            &render_device,
                            cmd.get(),
                            view.slot,
                            view.view_matrix,
                            view.projection,
                            plan.jitter,
                            dlss_reset,
                            delta_secs * 1000.0,
                        );
                        prof.end(&render_device, cmd.get(), scope);
                        if let (Some(guides), Some(textures)) =
                            (renderer.guide_views(view.slot), view.prep_textures.as_ref())
                        {
                            let scope = prof.begin(&render_device, cmd.get(), "post prep");
                            post_prep.record(
                                &render_device,
                                cmd.get(),
                                &modules,
                                &view.prep,
                                textures,
                                view.output_extent,
                                DescriptorHeaps::prep_block(parity, view.slot),
                                &crate::post_prep::PrepInputs {
                                    pyramid: guides.pyramid,
                                    fog: guides.fog,
                                    depth: guides.depth,
                                    auto_exposure: view.ae.1,
                                    display_exposure: view.exposure.display_exposure(),
                                    bloom: bloom > 0.0,
                                    local_tonemap: local_tonemap > 0.0,
                                },
                            );
                            prof.end(&render_device, cmd.get(), scope);
                        }
                        // The raygen just filled the luminance buffer at this size.
                        traced.push(view.camera);
                    }
                }
            }
            if let Some(grid) = grid_cmd {
                render_device.end_command_buffer(grid).unwrap();
            }
            render_device.end_command_buffer(view_cmd).unwrap();
            render_device.end_command_buffer(view_back_cmd).unwrap();
            (traced, grid_split)
        };

        // The present lane: the XR eyes, offscreen camera targets and the window (composite,
        // gizmos, UI, screenshot and video copies).
        let present_lane = |dlss_ran: bool| {
            let _lane = info_span!("record_present_pass").entered();
            let present_scope = prof.begin(&render_device, present_cmd, "present pass");
            let cmd_buffer = present_cmd;
            let postprocess = postprocess_filters.get(&render_config.postprocess_pipeline);
            let debug_view = camera
                .and_then(|c| c.5.copied())
                .unwrap_or_default()
                .shader_index();

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
                    render_device.cmd_set_scissor(
                        cmd_buffer,
                        0,
                        std::slice::from_ref(&render_area),
                    );
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
                    if let Some(source_view) = dlss_outputs[eye].0 {
                        record_post_draw(
                            &render_device,
                            cmd_buffer,
                            pipeline,
                            TargetKind::Display,
                            DescriptorHeaps::post_block(parity, eye),
                            source_view,
                            dlss_outputs[eye].1,
                            views[eye].prep_textures,
                            &crate::post_process_filter::PostProcessPushConstants {
                                uniforms: frame.uniform_buffers[eye].address,
                                auto_exposure: views[eye].ae.1,
                                display_exposure: views[eye].exposure.display_exposure(),
                                debug_view,
                                scene_radiance: 0,
                                fog: 0,
                                local_tonemap,
                                bloom,
                                lum: if views[eye].meters {
                                    views[eye].ae.0
                                } else {
                                    0
                                },
                                lum_width: views[eye].trace_extent.width,
                                lum_height: views[eye].trace_extent.height,
                                cost: views[eye].cost.1,
                                history: views[eye].history.0,
                                history_reset: views[eye].history.1,
                                tonemap: tonemap as u32,
                                prep: views[eye].prep,
                            },
                        );
                    }
                    render_device.cmd_end_rendering(cmd_buffer);
                    xr_state.record_eye_to_layer(&render_device, cmd_buffer, xr_frame, eye);
                }
            }

            // Offscreen camera targets, each in its own pass BEFORE the window's. Ordering
            // matters twice over: this pass cannot share the swapchain's attachment, and a
            // lower-`Camera::order` view whose target a later view samples must land first.
            // Each image ends back in SHADER_READ_ONLY_OPTIMAL, which is where anything
            // resolving the asset -- a UI node, a material, a hit shader -- expects it.
            // The pass runs even with nothing to composite: an untraceable scene still owes its
            // target a clear, which a window camera gets free from the window's own pass.
            for view in views {
                let ViewTarget::Image { asset } = view.target else {
                    continue;
                };
                let Some((texture, extent, kind)) = camera_targets.map.get(&asset) else {
                    continue;
                };
                // `dlss_ran` is frame-wide; the view's own plan says whether its slot holds
                // an output.
                let composite = postprocess.filter(|_| dlss_ran).and_then(|pipeline| {
                    dlss_outputs[view.slot]
                        .0
                        .filter(|_| view.plan.is_some())
                        .map(|source_view| (pipeline, source_view))
                });
                vk_utils::transition_image_layout(
                    &render_device,
                    cmd_buffer,
                    texture.image,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::ImageLayout::ATTACHMENT_OPTIMAL,
                );
                let area = vk::Rect2D::default().extent(*extent);
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(texture.image_view)
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    });
                let info = vk::RenderingInfo::default()
                    .layer_count(1)
                    .render_area(area)
                    .color_attachments(std::slice::from_ref(&attachment));
                render_device.cmd_begin_rendering(cmd_buffer, &info);
                render_device.cmd_set_scissor(cmd_buffer, 0, std::slice::from_ref(&area));
                render_device.cmd_set_viewport(
                    cmd_buffer,
                    0,
                    std::slice::from_ref(
                        &vk::Viewport::default()
                            .width(extent.width as f32)
                            .height(extent.height as f32)
                            .min_depth(0.0)
                            .max_depth(1.0),
                    ),
                );
                if let Some((pipeline, source_view)) = composite {
                    record_post_draw(
                        &render_device,
                        cmd_buffer,
                        pipeline,
                        *kind,
                        DescriptorHeaps::post_block(parity, view.slot),
                        source_view,
                        dlss_outputs[view.slot].1,
                        view.prep_textures,
                        &crate::post_process_filter::PostProcessPushConstants {
                            uniforms: frame.uniform_buffers[view.slot].address,
                            auto_exposure: view.ae.1,
                            display_exposure: view.exposure.display_exposure(),
                            debug_view: view.debug_view,
                            scene_radiance: 0,
                            fog: 0,
                            local_tonemap,
                            bloom,
                            lum: if view.meters { view.ae.0 } else { 0 },
                            lum_width: view.trace_extent.width,
                            lum_height: view.trace_extent.height,
                            cost: view.cost.1,
                            history: view.history.0,
                            history_reset: view.history.1,
                            tonemap: tonemap as u32,
                            prep: view.prep,
                        },
                    );
                }
                // Gizmos belong to the surface their camera rendered into; drawn
                // full-window the UI pass covers them. No depth guide means nothing traced,
                // and then nothing can occlude them either.
                // A radiance target is lit scenery, not a screen: no overlays.
                if *kind == TargetKind::Display {
                    crate::gizmo_render::draw_gizmos(
                        &render_device,
                        cmd_buffer,
                        view.view_proj,
                        *extent,
                        dlss_outputs[view.slot]
                            .1
                            .filter(|_| composite.is_some())
                            .map(|g| g.depth),
                        parity,
                        view.slot,
                        view.camera_mask,
                        &mut gizmos,
                    );
                }
                render_device.cmd_end_rendering(cmd_buffer);
                vk_utils::transition_image_layout(
                    &render_device,
                    cmd_buffer,
                    texture.image,
                    vk::ImageLayout::ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }

            // The window, when there is one: the flat render or the XR spectator (left eye,
            // aspect-fit), the gizmos and the UI. A windowless frame stops at the image targets.
            if let Some((swapchain_image, swapchain_view, window_size, window_format)) = presenting
            {
                // The XR spectator: the left eye's finished image (TRANSFER_SRC after its layer
                // blit) letterboxed into the window by a blit, not a third run of the post-process.
                let spectator = match (xr.as_deref(), xr_frame.as_ref()) {
                    (Some(xr_state), Some(_)) if dlss_ran && postprocess.is_some() => {
                        Some(xr_state)
                    }
                    _ => None,
                };
                if let Some(xr_state) = spectator {
                    record_spectator(
                        &render_device,
                        cmd_buffer,
                        xr_state,
                        swapchain_image,
                        window_size,
                    );
                } else {
                    vk_utils::transition_image_layout(
                        &render_device,
                        cmd_buffer,
                        swapchain_image,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::ATTACHMENT_OPTIMAL,
                    );
                }

                let render_area = vk::Rect2D::default().extent(window_size);

                let attachment_info = vk::RenderingAttachmentInfo::default()
                    .image_view(swapchain_view)
                    .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                    .load_op(if spectator.is_some() {
                        vk::AttachmentLoadOp::LOAD
                    } else {
                        vk::AttachmentLoadOp::CLEAR
                    })
                    .store_op(vk::AttachmentStoreOp::STORE);

                let render_info = vk::RenderingInfo::default()
                    .layer_count(1)
                    .render_area(render_area)
                    .color_attachments(std::slice::from_ref(&attachment_info));

                render_device.cmd_begin_rendering(cmd_buffer, &render_info);
                render_device.cmd_set_scissor(cmd_buffer, 0, std::slice::from_ref(&render_area));

                // Until the RT pipeline compiles and the first evaluate lands there is nothing
                // to sample -- leave the clear (the UI still draws).
                if let Some(pipeline) = postprocess.filter(|_| dlss_ran && spectator.is_none()) {
                    let (window_w, window_h) =
                        (window_size.width as f32, window_size.height as f32);
                    // XR draws ONE letterboxed eye as the spectator image; a flat frame
                    // composites every window-targeted view into its own rect.
                    let composited: &[ViewFrame] = if xr_frame.is_some() {
                        &views[..1.min(views.len())]
                    } else {
                        &views[..]
                    };
                    for view in composited {
                        let Some(source_view) = dlss_outputs[view.slot].0 else {
                            continue;
                        };
                        let ViewTarget::Window { rect } = view.target else {
                            // Image targets are composited in their own pass, outside this
                            // one: this render pass has the swapchain attached.
                            continue;
                        };
                        let viewport = if xr_frame.is_some() {
                            // Spectator: letterbox the eye image instead of stretching it.
                            let source_aspect =
                                view.output_extent.width as f32 / view.output_extent.height as f32;
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
                            // Land the traced image on the rect it was traced for. Unset
                            // `Camera::viewport` gives the whole window, so a plain app is
                            // unchanged.
                            vk::Viewport::default()
                                .x(rect.offset.x as f32)
                                .y(rect.offset.y as f32)
                                .width(rect.extent.width as f32)
                                .height(rect.extent.height as f32)
                        }
                        .min_depth(0.0)
                        .max_depth(1.0);
                        render_device.cmd_set_viewport(
                            cmd_buffer,
                            0,
                            std::slice::from_ref(&viewport),
                        );
                        // The composite is an OVERSIZED fullscreen triangle -- NDC
                        // (-1,-1)(3,-1)(-1,3) in quad.slang -- so a viewport smaller than the
                        // window is not enough on its own: the vertices past NDC map past the
                        // viewport rect and would paint over roughly twice it, straight across
                        // the editor's UI (or over the view composited before it). Clip to the
                        // rect; the UI pass below puts the full scissor back.
                        let scissor = vk::Rect2D::default()
                            .offset(vk::Offset2D {
                                x: viewport.x as i32,
                                y: viewport.y as i32,
                            })
                            .extent(vk::Extent2D {
                                width: viewport.width as u32,
                                height: viewport.height as u32,
                            });
                        render_device.cmd_set_scissor(
                            cmd_buffer,
                            0,
                            std::slice::from_ref(&scissor),
                        );

                        // The spectator gets its own set; a flat view uses its slot's.
                        let target = if xr_frame.is_some() {
                            crate::MAX_VIEWS
                        } else {
                            view.slot
                        };
                        record_post_draw(
                            &render_device,
                            cmd_buffer,
                            pipeline,
                            TargetKind::Display,
                            DescriptorHeaps::post_block(parity, target),
                            source_view,
                            dlss_outputs[view.slot].1,
                            view.prep_textures,
                            &crate::post_process_filter::PostProcessPushConstants {
                                uniforms: frame.uniform_buffers[view.slot].address,
                                auto_exposure: view.ae.1,
                                display_exposure: view.exposure.display_exposure(),
                                debug_view: view.debug_view,
                                scene_radiance: 0,
                                fog: 0,
                                local_tonemap,
                                bloom,
                                lum: if view.meters { view.ae.0 } else { 0 },
                                lum_width: view.trace_extent.width,
                                lum_height: view.trace_extent.height,
                                cost: view.cost.1,
                                history: view.history.0,
                                history_reset: view.history.1,
                                tonemap: tonemap as u32,
                                prep: view.prep,
                            },
                        );
                    }
                }

                // bevy_ui / feathers (including the dev panel), drawn over the scene,
                // full-window. Restore the full scissor: the composite above narrowed it.
                render_device.cmd_set_scissor(cmd_buffer, 0, std::slice::from_ref(&render_area));
                render_device.cmd_set_viewport(
                    cmd_buffer,
                    0,
                    std::slice::from_ref(
                        &vk::Viewport::default()
                            .width(window_size.width as f32)
                            .height(window_size.height as f32)
                            .min_depth(0.0)
                            .max_depth(1.0),
                    ),
                );
                // Debug gizmo lines (bevy_gizmos), world-space, over the scene and under the
                // UI, depth-tested against the traced frame's depth guide (so not before there
                // is one). Skipped under XR: the spectator's letterboxed blit does not match
                // views[0]'s full-window projection.
                // Window-targeted views only; an image target drew its own above. `.find()`
                // because the lowest-order view may be an image target, or there may be none.
                let window_view = views
                    .iter()
                    .find(|v| matches!(v.target, ViewTarget::Window { .. }));
                let scene_depth = window_view
                    .filter(|_| dlss_ran)
                    .and_then(|view| dlss_outputs[view.slot].1.map(|g| g.depth));
                if let (None, Some(view)) = (&xr_frame, window_view) {
                    crate::gizmo_render::draw_gizmos(
                        &render_device,
                        cmd_buffer,
                        view.view_proj,
                        window_size,
                        scene_depth,
                        parity,
                        view.slot,
                        view.camera_mask,
                        &mut gizmos,
                    );
                }
                crate::ui_render::draw_ui(&render_device, cmd_buffer, window_size, parity, &mut ui);

                render_device.cmd_end_rendering(cmd_buffer);

                // A pending screenshot copies the finished frame (UI included) out through a
                // host buffer on its way to present.
                // The capture reads the WINDOW swapchain (UI included), so it must use the
                // swapchain extent -- under XR `output_extent` is the eye extent, and sizing
                // the copy with it reads out of the swapchain image's bounds (F12 crashed XR).
                let capture_extent = window_size;
                // A recording copies its frames out the same way (util/video.rs).
                let copy_out = screenshots.has_pending() || video.wants_frame();
                if copy_out {
                    vk_utils::transition_image_layout(
                        &render_device,
                        cmd_buffer,
                        frame.swapchain_image,
                        vk::ImageLayout::ATTACHMENT_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    );
                    video.record(
                        &render_device,
                        cmd_buffer,
                        frame.swapchain_image,
                        capture_extent,
                        window_format,
                        frame_count,
                    );
                }
                if let Some(path) = screenshots.take_next() {
                    let size = capture_extent.width as u64 * capture_extent.height as u64 * 4;
                    // Read back on the CPU: host-cached memory, not the upload kind.
                    let buffer: Buffer<u8> = render_device.create_buffer(
                        size,
                        vk::BufferUsageFlags::TRANSFER_DST,
                        gpu_allocator::MemoryLocation::GpuToCpu,
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
                    screenshots.in_flight(path, buffer, capture_extent, window_format, frame_count);
                }

                // Make swapchain available for present
                vk_utils::transition_image_layout(
                    &render_device,
                    cmd_buffer,
                    frame.swapchain_image,
                    if copy_out {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else {
                        vk::ImageLayout::ATTACHMENT_OPTIMAL
                    },
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            }

            prof.end(&render_device, cmd_buffer, present_scope);
            prof.end(&render_device, cmd_buffer, frame_scope);
            render_device.end_command_buffer(cmd_buffer).unwrap();
        };

        // A frame that grows the instance table or the TLAS records the scene lane first (the
        // view lane reads the new handle and addresses); every other frame runs all three at
        // once.
        let dlss_ran_with = |tlas: &TlasRefs| {
            rtx_pipeline.is_some()
                && tlas.traceable
                && sbt_ready
                && has_dlss
                && views.iter().any(|v| v.plan.is_some())
        };
        let pool = ComputeTaskPool::get_or_init(TaskPool::default);
        let mut traced = (Vec::new(), false);
        let dlss_ran;
        if tlas.stable() {
            let refs = TlasRefs::of(&tlas, sbt_ready);
            dlss_ran = dlss_ran_with(&refs);
            let tlas = &mut *tlas;
            let traced = &mut traced;
            pool.scope(|s| {
                s.spawn(async move { scene(tlas) });
                s.spawn(async move { *traced = view_lane(refs) });
                s.spawn_on_scope(async move { present_lane(dlss_ran) });
            });
        } else {
            scene(&mut tlas);
            let refs = TlasRefs::of(&tlas, sbt_ready);
            dlss_ran = dlss_ran_with(&refs);
            let traced = &mut traced;
            pool.scope(|s| {
                s.spawn(async move { *traced = view_lane(refs) });
                s.spawn_on_scope(async move { present_lane(dlss_ran) });
            });
        }
        let (traced, grid_split) = traced;
        for camera in traced {
            ae.mark_traced(camera);
        }
        drop(record);
        let prof = &mut *profiler;
        // The frame is submitted before the CPU waits for the previous one: its commands start
        // the moment that frame ends (a barrier at its head keeps the two from overlapping), with
        // no CPU latency between. What needs the previous frame's results -- the readbacks, the
        // exposure latch the NEXT frame copies, the destroyer -- comes after the wait, below.
        let previous = sync.last_frame;
        let done = sync.frame_count.checked_sub(1);
        let waited = std::time::Instant::now();
        {
            let mut waits = Vec::new();
            let mut signals = Vec::new();
            let mut points = Vec::new();
            if let (Some(swapchain), Some(_)) = (swapchain.as_deref(), presenting) {
                let (available, finished) = swapchain.frame_semaphores();
                waits.push((available, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT));
                signals.push(finished);
            }
            // The pick's scene work, submitted ahead of this frame (picking.rs).
            if pick {
                points.push(picker.point);
            }
            let queue_lock = info_span!("queue_lock").entered();
            let queue = render_device.queue.lock().unwrap();
            drop(queue_lock);
            let traced = crate::nsight::begin_frame(*queue);
            // The frame's submits on the graphics queue, with the async compute between them
            // (docs/plans/async-compute.html): without a pick, the scene lane's first half beside
            // skins and the radiance cache resolve; then the TLAS and lights; the light grids
            // (compute) beside the first view's fog grid and primary pass; the rest after them.
            // The last submit waits (directly or through an earlier one) on every compute submit,
            // so the frame's point covers them all.
            let graphics =
                |cmds: &[vk::CommandBuffer],
                 points: &[crate::timeline::SubmitPoint],
                 timeline_waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)]| {
                    render_device
                        .timeline
                        .submit_with(
                            &render_device,
                            *queue,
                            cmds,
                            points,
                            timeline_waits,
                            &[],
                            &[],
                            vk::Fence::null(),
                            None,
                        )
                        .unwrap_or_else(|e| {
                            crate::aftermath::note_device_lost(e);
                            panic!("frame submit failed: {e:?}");
                        })
                };
            let compute_wait = |point: u64| {
                let semaphore = render_device.compute.as_ref().unwrap().timeline.semaphore;
                vec![(semaphore, point, vk::PipelineStageFlags2::ALL_COMMANDS)]
            };
            let mut head = Vec::new();
            let mut head_points = points.clone();
            let mut head_waits = Vec::new();
            match compute_cmd {
                Some(compute_cmd) if !pick => {
                    let before = render_device.timeline.last_submitted();
                    graphics(&[cmd_buffer], &head_points, &[]);
                    head_points.clear();
                    head_waits = compute_wait(render_device.submit_compute(compute_cmd, before));
                }
                _ => head.push(cmd_buffer),
            }
            head.push(tlas_cmd);
            let (tail, tail_points, tail_waits) = match grid_cmd.filter(|_| grid_split) {
                Some(grid_cmd) => {
                    let lights = graphics(&head, &head_points, &head_waits);
                    let grids = render_device.submit_compute(grid_cmd, lights);
                    graphics(&[view_cmd], &[], &[]);
                    (
                        vec![view_back_cmd, present_cmd],
                        Vec::new(),
                        compute_wait(grids),
                    )
                }
                None => {
                    head.extend([view_cmd, view_back_cmd, present_cmd]);
                    (head, head_points, head_waits)
                }
            };
            sync.submit(
                *queue,
                &tail,
                &tail_points,
                &tail_waits,
                &waits,
                &signals,
                present_id,
            );
            if traced {
                crate::nsight::end_frame(*queue);
            }
            if let Some(swapchain) = swapchain.as_deref() {
                latency.mark(
                    &render_device,
                    swapchain,
                    vk::LatencyMarkerNV::RENDERSUBMIT_END,
                );
                latency.mark(
                    &render_device,
                    swapchain,
                    vk::LatencyMarkerNV::PRESENT_START,
                );
            }
            prof.note_cpu_record(waited.elapsed().as_secs_f64() * 1.0e3);
            let presented = match (swapchain.as_deref(), presenting) {
                (Some(swapchain), Some(_)) => Some(swapchain.present(*queue, present_id)),
                _ => None,
            };
            if let Some(swapchain) = swapchain.as_deref() {
                latency.mark(&render_device, swapchain, vk::LatencyMarkerNV::PRESENT_END);
            }
            drop(queue);
            if let (Some(result), Some(swapchain), Some(window)) =
                (presented, swapchain.as_deref_mut(), window.as_deref())
            {
                swapchain.after_present(result, window);
            }
            sync.frame_count += 1;
        }
        // The XR side of the submit: release the image and hand the compositor its layer.
        if let (Some(xr_state), Some(xr_frame)) = (xr.as_deref_mut(), xr_frame) {
            xr_state.end_frame(&render_device, xr_frame);
        }
        // The previous frame: wait for it, then latch what the next frame computes from its
        // results (auto_exposure.rs double-buffers what both touch).
        sync.wait_for(previous);
        render_device.destroyer.tick();
        prof.read_previous(&render_device, time.elapsed_secs_f64());
        ae.latch(&render_device);
        // The previous frame's screenshot and video copies have landed: off to their encoders.
        if let Some(done) = done {
            screenshots.collect(&render_device, done);
            video.latch(&render_device, done);
        }
        // Whole-device drains take the queue lock: the asset workers submit their builds
        // through the same VkQueue, and a drain racing a submit is a threading violation.
        if dev_drain {
            let _queue = render_device.queue.lock().unwrap();
            let _ = render_device.device.device_wait_idle();
        }
    }
}

/// One post-process draw into the current render pass: input 0 of `block` samples
/// `source_view`, inputs 1..=9 the guides for the debug views (aliasing the output until the
/// renderer has them). Guides sit in SHADER_READ_ONLY after the evaluate; the output stays
/// GENERAL. Viewport and scissor are the caller's.
#[allow(clippy::too_many_arguments)]
/// The XR spectator into the acquired swapchain image: cleared, the left eye's finished image
/// (TRANSFER_SRC_OPTIMAL after its layer blit) blitted aspect-fit into the middle, left in
/// ATTACHMENT_OPTIMAL for the window's pass (gizmos, UI) to load.
fn record_spectator(
    rd: &RenderDevice,
    cmd: vk::CommandBuffer,
    xr: &crate::xr::XrState,
    image: vk::Image,
    window: vk::Extent2D,
) {
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
    let barrier = |from, to, src_stage, src_access, dst_stage, dst_access| {
        let b = crate::vk_init::layout_transition2(image, from, to)
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);
        unsafe {
            rd.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&b)),
            )
        };
    };
    // Behind the acquire semaphore's wait (COLOR_ATTACHMENT_OUTPUT), as the attachment path.
    barrier(
        L::UNDEFINED,
        L::TRANSFER_DST_OPTIMAL,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::NONE,
        S::CLEAR | S::BLIT,
        A::TRANSFER_WRITE,
    );
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let (eye, _) = xr.eye_target(0);
    let (src_w, src_h) = (xr.extent.width as f32, xr.extent.height as f32);
    let (win_w, win_h) = (window.width as f32, window.height as f32);
    let (w, h) = if src_w / src_h > win_w / win_h {
        (win_w, win_w * src_h / src_w)
    } else {
        (win_h * src_w / src_h, win_h)
    };
    let (x, y) = (((win_w - w) * 0.5) as i32, ((win_h - h) * 0.5) as i32);
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    let region = vk::ImageBlit::default()
        .src_subresource(layers)
        .src_offsets([
            vk::Offset3D::default(),
            vk::Offset3D {
                x: xr.extent.width as i32,
                y: xr.extent.height as i32,
                z: 1,
            },
        ])
        .dst_subresource(layers)
        .dst_offsets([
            vk::Offset3D { x, y, z: 0 },
            vk::Offset3D {
                x: x + w as i32,
                y: y + h as i32,
                z: 1,
            },
        ]);
    unsafe {
        rd.device.cmd_clear_color_image(
            cmd,
            image,
            L::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            std::slice::from_ref(&range),
        );
        // The clear and the blit both write the image: in order.
        let b = vk::MemoryBarrier2::default()
            .src_stage_mask(S::CLEAR)
            .src_access_mask(A::TRANSFER_WRITE)
            .dst_stage_mask(S::BLIT)
            .dst_access_mask(A::TRANSFER_WRITE);
        rd.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&b)),
        );
        rd.device.cmd_blit_image(
            cmd,
            eye,
            L::TRANSFER_SRC_OPTIMAL,
            image,
            L::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
            vk::Filter::LINEAR,
        );
    }
    barrier(
        L::TRANSFER_DST_OPTIMAL,
        L::ATTACHMENT_OPTIMAL,
        S::BLIT,
        A::TRANSFER_WRITE,
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE,
    );
}

unsafe fn record_post_draw(
    render_device: &RenderDevice,
    cmd_buffer: vk::CommandBuffer,
    pipeline: &crate::post_process_filter::CompiledPostProcessFilter,
    kind: TargetKind,
    block: u32,
    source_view: vk::ImageView,
    guides: Option<crate::dlss::GuideViews>,
    prep: Option<crate::post_prep::PrepTextures>,
    push_constants: &crate::post_process_filter::PostProcessPushConstants,
) {
    let Some(format_pipeline) = pipeline.pipelines.get(kind.format()) else {
        return;
    };
    let prep = prep.filter(|_| guides.is_some());
    let mut constants = crate::post_process_filter::PostProcessPushConstants {
        scene_radiance: (kind == TargetKind::SceneRadiance) as u32,
        fog: guides.is_some() as u32,
        ..*push_constants
    };
    constants.prep.textures = prep.is_some() as u32;
    let heaps = render_device.heaps();
    let mut inputs =
        [(source_view, vk::ImageLayout::GENERAL); crate::descriptor_heap::POST_INPUTS as usize];
    if let Some(p) = prep {
        inputs[10] = (p.keys, vk::ImageLayout::GENERAL);
        inputs[11] = (p.bloom, vk::ImageLayout::GENERAL);
    }
    if let Some(g) = guides {
        for (slot, view) in [
            (1, g.color),
            (2, g.normal_roughness),
            (3, g.diffuse),
            (4, g.specular),
            (5, g.depth),
            (6, g.spec_hit),
            (7, g.motion),
            (8, g.fog),
            (9, g.pyramid),
        ] {
            inputs[slot] = (view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
    }
    for (i, (view, layout)) in inputs.into_iter().enumerate() {
        heaps.write_post(block, i as u32, view, layout);
    }
    unsafe {
        render_device.cmd_bind_pipeline(
            cmd_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            format_pipeline,
        );
    }
    heaps.bind(cmd_buffer);
    heaps.push(
        cmd_buffer,
        &crate::post_process_filter::PostProcessPushData {
            constants,
            inputs: block,
            pad: 0,
        },
    );
    unsafe { render_device.cmd_draw(cmd_buffer, 3, 1, 0, 0) };
}

pub fn on_shutdown(world: &mut World) {
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

    for buffer in (frame.uniform_buffers.iter().map(|b| b.handle))
        .chain(frame.cost_buffers.iter().map(|b| b.handle))
        .chain(frame.primary_hits.iter().map(|b| b.handle))
        .chain(frame.sky_masks.iter().flatten().map(|b| b.handle))
        .chain(frame.fog_veils.iter().flatten().map(|b| b.handle))
    {
        render_device.destroyer.destroy_buffer(buffer);
    }
    render_device
        .destroyer
        .destroy_buffer(frame.focus_data.handle);
    render_device
        .destroyer
        .destroy_buffer(frame.environments.handle);
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
    world.remove_resource::<crate::frame_sync::FrameSync>();
}

/// Run condition for app systems that read [`RenderDevice`]: the device is removed by
/// [`on_shutdown`] a frame before bevy's exit lands, and `Update` still runs that frame.
pub fn render_device_exists(device: Option<Res<RenderDevice>>) -> bool {
    device.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::camera::Viewport;

    fn window() -> vk::Extent2D {
        vk::Extent2D {
            width: 1920,
            height: 1080,
        }
    }

    fn with_viewport(position: UVec2, size: UVec2) -> Camera {
        Camera {
            viewport: Some(Viewport {
                physical_position: position,
                physical_size: size,
                ..default()
            }),
            ..default()
        }
    }

    #[test]
    fn no_camera_or_no_viewport_fills_the_window() {
        assert_eq!(view_rect(None, window()).extent, window());
        let camera = Camera::default();
        assert_eq!(view_rect(Some(&camera), window()).extent, window());
    }

    #[test]
    fn a_viewport_becomes_its_own_rect() {
        let camera = with_viewport(UVec2::new(300, 40), UVec2::new(1200, 900));
        let rect = view_rect(Some(&camera), window());
        assert_eq!((rect.offset.x, rect.offset.y), (300, 40));
        assert_eq!((rect.extent.width, rect.extent.height), (1200, 900));
    }

    #[test]
    fn a_rect_past_the_window_is_clamped_not_left_overhanging() {
        // A UI-driven rect can lag a window resize by a frame; a viewport past the
        // swapchain is a validation error, not a stretched image.
        let camera = with_viewport(UVec2::new(1800, 1000), UVec2::new(800, 600));
        let rect = view_rect(Some(&camera), window());
        assert_eq!(rect.offset.x + rect.extent.width as i32, 1920);
        assert_eq!(rect.offset.y + rect.extent.height as i32, 1080);
    }

    #[test]
    fn a_collapsed_dock_falls_back_to_the_window() {
        // Zero width would be a validation error too, and a collapsed panel is a normal
        // thing for a dock to report for a frame.
        let camera = with_viewport(UVec2::new(10, 10), UVec2::ZERO);
        assert_eq!(view_rect(Some(&camera), window()).extent, window());
    }

    #[test]
    fn an_origin_past_the_window_does_not_underflow() {
        // `max - origin` would wrap if the origin were not clamped first.
        let camera = with_viewport(UVec2::new(4000, 4000), UVec2::new(100, 100));
        let rect = view_rect(Some(&camera), window());
        assert_eq!(rect.extent, window());
    }

    /// `surface.slang` reads `lod_bias` by its float index into the uniform block.
    #[test]
    fn lod_bias_sits_where_the_slang_hit_shaders_read_it() {
        assert_eq!(core::mem::offset_of!(super::UniformData, lod_bias), 94 * 4);
        assert_eq!(
            core::mem::offset_of!(super::UniformData, emissive_boost),
            103 * 4
        );
    }

    /// `rt_types.slang` mirrors `UniformData` with scalars only; its offsets are these.
    #[test]
    fn uniform_data_matches_the_slang_raygen() {
        use super::UniformData as U;
        use core::mem::offset_of;
        assert_eq!(offset_of!(U, inverse_projection), 64);
        assert_eq!(offset_of!(U, environment), 140);
        assert_eq!(offset_of!(U, view), 144);
        assert_eq!(offset_of!(U, prev_view_proj), 272);
        assert_eq!(offset_of!(U, frame), 344);
        assert_eq!(offset_of!(U, pad_cache), 392);
        assert_eq!(offset_of!(U, portals), 400);
        assert_eq!(offset_of!(U, camera_mask), 416);
        assert_eq!(offset_of!(U, primary_hidden), 420);
        assert_eq!(offset_of!(U, environment_bits), 424);
        assert_eq!(offset_of!(U, environments), 432);
        assert_eq!(offset_of!(U, fog_volumes), 440);
        assert_eq!(offset_of!(U, time), 452);
        assert_eq!(offset_of!(U, fog_grid_prev), 456);
        assert_eq!(offset_of!(U, fog_grid_cells), 472);
        assert_eq!(offset_of!(U, fog_grid_inverse_view), 480);
        assert_eq!(offset_of!(U, fog_grid_size), 736);
        assert_eq!(offset_of!(U, fog_grid_sum_tex), 752);
        assert_eq!(offset_of!(U, fog_grid_soft), 760);
        assert_eq!(offset_of!(U, fog_grid_stride), 768);
        assert_eq!(offset_of!(U, pad_shared), 776);
    }
}
