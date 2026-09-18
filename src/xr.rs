//! OpenXR (VR) support, behind the `xr` cargo feature; without it nothing here runs and the
//! OpenXR loader is never dlopen'd.
//!
//! First slice: the session rides alongside the window. The finished window image (post
//! process + UI, tone-mapped) is blitted into both layers of the XR swapchain each frame, so
//! the headset shows the flat render while the real per-eye path is built up. The runtime is
//! whatever `XR_RUNTIME_JSON` / the active-runtime manifest points at — Monado's simulated
//! HMD for headset-free dev, WiVRn for the Quest.
//!
//! Vulkan interop is `XR_KHR_vulkan_enable2`: the XR runtime creates (wraps) the VkInstance
//! and VkDevice and picks the VkPhysicalDevice, so [`XrContext`] must exist before
//! [`crate::render_device::RenderDevice`] — [`XrPlugin`] is added just before
//! [`crate::ray_render_plugin::RayRenderPlugin`] for that reason.

use std::ffi::c_void;
use std::mem::transmute;

use ash::vk;
use ash::vk::Handle;
use bevy::input::InputSystems;
use bevy::prelude::*;
use bevy::transform::TransformSystems;
use openxr as xr;

use crate::render_device::RenderDevice;
use crate::vk_init;

/// Which controller / hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Reflect)]
pub enum XrHand {
    Left,
    Right,
}

impl XrHand {
    pub const ALL: [XrHand; 2] = [XrHand::Left, XrHand::Right];

    fn index(self) -> usize {
        self as usize
    }
}

/// A pose the runtime tracks, mirrored into the ECS as a root entity whose `Transform` is
/// `rig * pose` — the rig being the single `Camera3d` entity, exactly as the render path
/// anchors the eyes. Roots, not children of the camera: with the default
/// [`crate::transform::TransformPlugin`] (no CPU propagation) a camera that gained children
/// would stop getting its `GlobalTransform` synced. Apps parent their own visuals (laser,
/// wrist panel, controller model) under these; they persist across game states.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum XrTracked {
    Head,
    /// Where the controller physically is (OpenXR grip pose): attach hand-held visuals here.
    Grip(XrHand),
    /// The pointing ray (OpenXR aim pose): -Z forward, like a camera.
    Aim(XrHand),
}

/// The tracked pose in rig (OpenXR LOCAL) space, as last located; `valid` is false while the
/// runtime has no tracking for it (controller asleep, out of view), in which case
/// `Transform` keeps its last good value.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct XrPose {
    pub local: Transform,
    pub valid: bool,
}

/// One controller's inputs, as synced this frame. Analog values are 0..1 / -1..1.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XrHandState {
    /// The controller is bound and its inputs are live.
    pub active: bool,
    pub trigger: f32,
    pub squeeze: f32,
    pub thumbstick: Vec2,
    pub thumbstick_click: bool,
    /// A / X.
    pub primary: bool,
    /// B / Y.
    pub secondary: bool,
    /// Menu (left Touch controller; both hands on the simple profile).
    pub menu: bool,
}

/// Raw XR controller input, written by [`XrSystems::Poll`] each frame. Exists (all zero)
/// even without the `xr` feature so apps need no `Option`. Apps feed this into their own
/// action layer; the engine reads none of it.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct XrInput {
    pub left: XrHandState,
    pub right: XrHandState,
    /// The session has input focus (a runtime overlay/menu steals it; inputs are zero then).
    pub focused: bool,
}

impl XrInput {
    pub fn hand(&self, hand: XrHand) -> &XrHandState {
        match hand {
            XrHand::Left => &self.left,
            XrHand::Right => &self.right,
        }
    }
}

/// `PreUpdate`, before [`InputSystems`]: `Poll` syncs actions, locates the tracked poses and
/// writes [`XrInput`] + the [`XrTracked`] entities' transforms.
#[derive(SystemSet, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum XrSystems {
    Poll,
}

/// Pre-device XR state: loader, instance, system. Present only under the `xr` feature, and
/// then only when the runtime came up. [`crate::render_device::RenderDevice::from_display`] consumes it to
/// create the Vulkan objects through the runtime.
#[derive(Resource)]
pub struct XrContext {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
}

/// A live XR session (created once the [`RenderDevice`] exists): frame loop objects, the
/// stereo swapchain, and the reference space.
#[derive(Resource)]
pub struct XrState {
    instance: xr::Instance,
    session: xr::Session<xr::Vulkan>,
    frame_waiter: xr::FrameWaiter,
    frame_stream: xr::FrameStream<xr::Vulkan>,
    space: xr::Space,
    swapchain: xr::Swapchain<xr::Vulkan>,
    images: Vec<vk::Image>,
    /// Per-eye pixel size of the XR swapchain (runtime's recommendation).
    pub extent: vk::Extent2D,
    /// Per-eye render targets: the post-process draws each eye into these (the pipeline's
    /// B8G8R8A8_UNORM), then [`Self::record_eye_to_layer`] blits into the XR image layer,
    /// whatever format the runtime picked.
    eye_targets: [EyeTarget; 2],
    blend_mode: xr::EnvironmentBlendMode,
    /// Session is between Begin and End (READY seen, STOPPING not yet).
    running: bool,
    /// Session state FOCUSED: actions sync to real values.
    focused: bool,
    /// The last `xrWaitFrame` predicted display time; poses are located at it.
    predicted_time: Option<xr::Time>,
    /// VIEW reference space, located in `space` for the head pose.
    view_space: xr::Space,
    actions: XrActions,
}

struct EyeTarget {
    image: vk::Image,
    view: vk::ImageView,
}

/// The engine's one action set, attached to the session at creation. Bound for Quest Touch
/// (WiVRn), Valve Index, and the KHR simple profile as the fallback every runtime accepts.
struct XrActions {
    set: xr::ActionSet,
    hands: [xr::Path; 2],
    aim: xr::Action<xr::Posef>,
    /// Only read through `grip_spaces`, but dropping the action would destroy them.
    _grip: xr::Action<xr::Posef>,
    trigger: xr::Action<f32>,
    squeeze: xr::Action<f32>,
    thumbstick: xr::Action<xr::Vector2f>,
    thumbstick_click: xr::Action<bool>,
    primary: xr::Action<bool>,
    secondary: xr::Action<bool>,
    menu: xr::Action<bool>,
    aim_spaces: [xr::Space; 2],
    grip_spaces: [xr::Space; 2],
}

impl XrActions {
    fn new(
        instance: &xr::Instance,
        session: &xr::Session<xr::Vulkan>,
    ) -> Result<Self, xr::sys::Result> {
        let set = instance.create_action_set("aurora", "Aurora", 0)?;
        let hands = [
            instance.string_to_path("/user/hand/left")?,
            instance.string_to_path("/user/hand/right")?,
        ];
        let aim = set.create_action("aim", "Aim pose", &hands)?;
        let grip = set.create_action("grip", "Grip pose", &hands)?;
        let trigger = set.create_action("trigger", "Trigger", &hands)?;
        let squeeze = set.create_action("squeeze", "Squeeze", &hands)?;
        let thumbstick = set.create_action("thumbstick", "Thumbstick", &hands)?;
        let thumbstick_click = set.create_action("thumbstick_click", "Thumbstick click", &hands)?;
        let primary = set.create_action("primary", "Primary (A/X)", &hands)?;
        let secondary = set.create_action("secondary", "Secondary (B/Y)", &hands)?;
        let menu = set.create_action("menu", "Menu", &hands)?;

        fn bind<'a, T: xr::ActionTy>(
            instance: &xr::Instance,
            action: &'a xr::Action<T>,
            hand: &str,
            input: &str,
        ) -> Result<xr::Binding<'a>, xr::sys::Result> {
            let path = instance.string_to_path(&format!("/user/hand/{hand}/input/{input}"))?;
            Ok(xr::Binding::new(action, path))
        }
        fn both<'a, T: xr::ActionTy>(
            instance: &xr::Instance,
            action: &'a xr::Action<T>,
            input: &str,
        ) -> Result<[xr::Binding<'a>; 2], xr::sys::Result> {
            Ok([
                bind(instance, action, "left", input)?,
                bind(instance, action, "right", input)?,
            ])
        }
        // Suggestions are validated against the spec's profile tables, not the connected
        // hardware; a rejection here is a typo'd path, so log it and keep the others.
        let suggest = |profile: &str, bindings: Vec<xr::Binding>| match instance
            .string_to_path(profile)
            .and_then(|p| instance.suggest_interaction_profile_bindings(p, &bindings))
        {
            Ok(()) => debug!("xr: suggested {} bindings for {profile}", bindings.len()),
            Err(e) => warn!("xr: binding suggestion for {profile} rejected: {e:?}"),
        };

        let mut touch = Vec::new();
        touch.extend(both(instance, &aim, "aim/pose")?);
        touch.extend(both(instance, &grip, "grip/pose")?);
        touch.extend(both(instance, &trigger, "trigger/value")?);
        touch.extend(both(instance, &squeeze, "squeeze/value")?);
        touch.extend(both(instance, &thumbstick, "thumbstick")?);
        touch.extend(both(instance, &thumbstick_click, "thumbstick/click")?);
        touch.push(bind(instance, &primary, "left", "x/click")?);
        touch.push(bind(instance, &primary, "right", "a/click")?);
        touch.push(bind(instance, &secondary, "left", "y/click")?);
        touch.push(bind(instance, &secondary, "right", "b/click")?);
        touch.push(bind(instance, &menu, "left", "menu/click")?);
        suggest("/interaction_profiles/oculus/touch_controller", touch);

        let mut index = Vec::new();
        index.extend(both(instance, &aim, "aim/pose")?);
        index.extend(both(instance, &grip, "grip/pose")?);
        index.extend(both(instance, &trigger, "trigger/value")?);
        index.extend(both(instance, &squeeze, "squeeze/value")?);
        index.extend(both(instance, &thumbstick, "thumbstick")?);
        index.extend(both(instance, &thumbstick_click, "thumbstick/click")?);
        index.extend(both(instance, &primary, "a/click")?);
        index.extend(both(instance, &secondary, "b/click")?);
        suggest("/interaction_profiles/valve/index_controller", index);

        let mut simple = Vec::new();
        simple.extend(both(instance, &aim, "aim/pose")?);
        simple.extend(both(instance, &grip, "grip/pose")?);
        // A boolean input bound to a float action reads as 0/1 per the spec.
        simple.extend(both(instance, &trigger, "select/click")?);
        simple.extend(both(instance, &menu, "menu/click")?);
        suggest("/interaction_profiles/khr/simple_controller", simple);

        session.attach_action_sets(&[&set])?;
        let aim_spaces = [
            aim.create_space(session, hands[0], xr::Posef::IDENTITY)?,
            aim.create_space(session, hands[1], xr::Posef::IDENTITY)?,
        ];
        let grip_spaces = [
            grip.create_space(session, hands[0], xr::Posef::IDENTITY)?,
            grip.create_space(session, hands[1], xr::Posef::IDENTITY)?,
        ];
        Ok(Self {
            set,
            hands,
            aim,
            _grip: grip,
            trigger,
            squeeze,
            thumbstick,
            thumbstick_click,
            primary,
            secondary,
            menu,
            aim_spaces,
            grip_spaces,
        })
    }

    fn hand_state(&self, session: &xr::Session<xr::Vulkan>, hand: usize) -> XrHandState {
        let p = self.hands[hand];
        let f = |a: &xr::Action<f32>| a.state(session, p).map(|s| s.current_state).unwrap_or(0.0);
        let b = |a: &xr::Action<bool>| {
            a.state(session, p)
                .map(|s| s.current_state)
                .unwrap_or(false)
        };
        let stick = self
            .thumbstick
            .state(session, p)
            .map(|s| Vec2::new(s.current_state.x, s.current_state.y))
            .unwrap_or(Vec2::ZERO);
        let active = self.aim.is_active(session, p).unwrap_or(false);
        XrHandState {
            active,
            trigger: f(&self.trigger),
            squeeze: f(&self.squeeze),
            thumbstick: stick,
            thumbstick_click: b(&self.thumbstick_click),
            primary: b(&self.primary),
            secondary: b(&self.secondary),
            menu: b(&self.menu),
        }
    }
}

/// Located pose → rig-space transform, if the runtime vouches for both halves.
fn locate(space: &xr::Space, base: &xr::Space, time: xr::Time) -> Option<Transform> {
    let loc = space.locate(base, time).ok()?;
    let want = xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID;
    if !loc.location_flags.contains(want) {
        return None;
    }
    let o = loc.pose.orientation;
    let p = loc.pose.position;
    Some(Transform {
        translation: Vec3::new(p.x, p.y, p.z),
        rotation: Quat::from_xyzw(o.x, o.y, o.z, o.w),
        scale: Vec3::ONE,
    })
}

/// One in-flight XR frame: produced by [`XrState::begin_frame`], consumed by
/// [`XrState::end_frame`] after the queue submit.
pub struct XrFrame {
    frame_state: xr::FrameState,
    /// The acquired-and-waited swapchain image (both eye layers).
    pub image: vk::Image,
    /// Per-eye pose + fov located at the predicted display time, in [`XrState::space`].
    pub views: Vec<xr::View>,
}

impl XrFrame {
    /// One eye's camera: its pose as a local-space camera matrix, and its asymmetric-fov
    /// infinite-reverse-z projection. The app's camera entity transform anchors this in the
    /// world (multiply on the left); the two eyes' poses differ by the wearer's IPD.
    pub fn eye_camera(&self, eye: usize, near: f32) -> (Mat4, Mat4) {
        let view = &self.views[eye];
        (pose_matrix(view.pose), projection(view.fov, near))
    }
}

/// XR pose (LOCAL space, meters, -Z forward — same handedness as bevy) as a camera-to-world
/// matrix.
fn pose_matrix(pose: xr::Posef) -> Mat4 {
    let o = pose.orientation;
    let p = pose.position;
    Mat4::from_rotation_translation(
        Quat::from_xyzw(o.x, o.y, o.z, o.w),
        Vec3::new(p.x, p.y, p.z),
    )
}

/// Asymmetric-fov projection with infinite reverse z — the XR sibling of
/// `bevy::math::proj::perspective_infinite_reverse` (fov angles are signed, left/down negative).
fn projection(fov: xr::Fovf, near: f32) -> Mat4 {
    let left = fov.angle_left.tan();
    let right = fov.angle_right.tan();
    let down = fov.angle_down.tan();
    let up = fov.angle_up.tan();
    let width = right - left;
    let height = up - down;
    Mat4::from_cols(
        Vec4::new(2.0 / width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / height, 0.0, 0.0),
        Vec4::new((right + left) / width, (up + down) / height, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

pub struct XrPlugin;

impl Plugin for XrPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<XrInput>();
        if !cfg!(feature = "xr") {
            return;
        }
        match XrContext::new() {
            Ok(context) => {
                // The spectator window must not pace the frame loop -- xrWaitFrame is the
                // governor, and FIFO would drag the headset down to the monitor's rate.
                // IMMEDIATE, not MAILBOX: mailbox on this driver is the resize Xid 79 (see
                // swapchain.rs). Before winit/render threads spawn readers.
                if std::env::var("AURORA_PRESENT_MODE").is_err() {
                    unsafe { std::env::set_var("AURORA_PRESENT_MODE", "immediate") };
                }
                app.insert_resource(context);
            }
            Err(err) => {
                error!("xr feature on but OpenXR init failed, running flat: {err}");
                return;
            }
        }
        app.configure_sets(PreUpdate, XrSystems::Poll.before(InputSystems))
            .add_systems(
                PreUpdate,
                (poll_input, follow_rig)
                    .chain()
                    .in_set(XrSystems::Poll)
                    .run_if(resource_exists::<XrState>),
            )
            .add_systems(
                PostUpdate,
                (level_rig, follow_rig)
                    .chain()
                    .before(TransformSystems::Propagate)
                    .run_if(resource_exists::<XrState>),
            );
    }
}

/// Syncs actions, mirrors poses into the [`XrTracked`] entities (spawning them on first
/// use), and publishes [`XrInput`].
fn poll_input(
    mut commands: Commands,
    mut state: ResMut<XrState>,
    mut input: ResMut<XrInput>,
    mut tracked: Query<(&XrTracked, &mut XrPose)>,
) {
    let (new_input, poses) = state.poll_input();
    if *input != new_input {
        trace!("xr input: {new_input:?}");
        *input = new_input;
    }
    if tracked.is_empty() {
        for kind in [
            XrTracked::Head,
            XrTracked::Aim(XrHand::Left),
            XrTracked::Aim(XrHand::Right),
            XrTracked::Grip(XrHand::Left),
            XrTracked::Grip(XrHand::Right),
        ] {
            commands.spawn((
                Name::new(format!("xr {kind:?}")),
                kind,
                XrPose::default(),
                Transform::default(),
                Visibility::default(),
            ));
        }
        return;
    }
    for (kind, mut pose) in &mut tracked {
        let Some((_, located)) = poses.iter().find(|(k, _)| k == kind) else {
            continue;
        };
        match located {
            Some(local) => {
                pose.local = *local;
                pose.valid = true;
            }
            None => pose.valid = false,
        }
    }
}

/// `Transform = rig * pose` for every tracked entity, the rig being the `Camera3d`'s own
/// transform (a root, so that is its world transform). Runs after the poll so `PreUpdate`
/// readers (pointer rays) see this frame's poses, and again before propagation so the render
/// sees the rig where gameplay left it this frame.
fn follow_rig(
    rig: Option<Single<&Transform, (With<Camera3d>, Without<XrTracked>)>>,
    mut tracked: Query<(&XrPose, &mut Transform), With<XrTracked>>,
) {
    let Some(rig) = rig else { return };
    let rig = **rig;
    for (pose, mut tf) in &mut tracked {
        let world = rig * pose.local;
        if *tf != world {
            *tf = world;
        }
    }
}

/// Strips pitch and roll from the rig: the headset supplies those, and any tilt on the
/// anchor tilts the whole world for the wearer. Camera controllers keep their own pitch
/// state and re-write the transform each frame, so leveling after them is harmless.
fn level_rig(rig: Option<Single<&mut Transform, With<Camera3d>>>) {
    let Some(mut rig) = rig else { return };
    let (yaw, pitch, roll) = rig.rotation.to_euler(EulerRot::YXZ);
    if pitch != 0.0 || roll != 0.0 {
        rig.rotation = Quat::from_rotation_y(yaw);
    }
}

impl XrContext {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { xr::Entry::load()? };
        let available = entry.enumerate_extensions()?;
        if !available.khr_vulkan_enable2 {
            return Err("runtime lacks XR_KHR_vulkan_enable2".into());
        }
        let mut extensions = xr::ExtensionSet::default();
        extensions.khr_vulkan_enable2 = true;
        let instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "aurora",
                engine_name: "aurora",
                ..Default::default()
            },
            &extensions,
            &[],
        )?;
        let props = instance.properties()?;
        info!(
            "OpenXR runtime: {} {}",
            props.runtime_name, props.runtime_version
        );
        let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        // Required by the spec before any Vulkan object goes through the runtime.
        let reqs = instance.graphics_requirements::<xr::Vulkan>(system)?;
        info!(
            "OpenXR Vulkan API range: {} - {}",
            reqs.min_api_version_supported, reqs.max_api_version_supported
        );
        Ok(Self { instance, system })
    }

    /// `xrCreateVulkanInstanceKHR`: the runtime adds the instance extensions it needs on top
    /// of `info` and performs the create.
    pub unsafe fn create_vk_instance(
        &self,
        entry: &ash::Entry,
        info: &vk::InstanceCreateInfo,
    ) -> ash::Instance {
        unsafe {
            let raw = self
                .instance
                .create_vulkan_instance(
                    self.system,
                    transmute(entry.static_fn().get_instance_proc_addr),
                    info as *const _ as *const c_void,
                )
                .expect("xrCreateVulkanInstanceKHR")
                .map_err(vk::Result::from_raw)
                .expect("vkCreateInstance (via OpenXR)");
            ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(raw as u64))
        }
    }

    /// The VkPhysicalDevice the HMD is driven by (`xrGetVulkanGraphicsDevice2KHR`).
    pub fn vk_physical_device(&self, instance: &ash::Instance) -> vk::PhysicalDevice {
        let raw = unsafe {
            self.instance
                .vulkan_graphics_device(self.system, instance.handle().as_raw() as *const c_void)
                .expect("xrGetVulkanGraphicsDevice2KHR")
        };
        vk::PhysicalDevice::from_raw(raw as u64)
    }

    /// `xrCreateVulkanDeviceKHR`: device create routed through the runtime so it can inject
    /// the device extensions the compositor's interop needs.
    pub unsafe fn create_vk_device(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        info: &vk::DeviceCreateInfo,
    ) -> ash::Device {
        unsafe {
            let raw = self
                .instance
                .create_vulkan_device(
                    self.system,
                    transmute(entry.static_fn().get_instance_proc_addr),
                    physical_device.as_raw() as *const c_void,
                    info as *const _ as *const c_void,
                )
                .expect("xrCreateVulkanDeviceKHR")
                .map_err(vk::Result::from_raw)
                .expect("vkCreateDevice (via OpenXR)");
            ash::Device::load(instance.fp_v1_0(), vk::Device::from_raw(raw as u64))
        }
    }
}

impl XrState {
    /// Creates the session on the renderer's device/queue plus the stereo swapchain and
    /// reference space. Called from `RayRenderPlugin::build` right after the device exists.
    pub fn new(context: &XrContext, device: &RenderDevice) -> Result<Self, xr::sys::Result> {
        let instance = context.instance.clone();
        let (session, frame_waiter, frame_stream) = unsafe {
            instance.create_session::<xr::Vulkan>(
                context.system,
                &xr::vulkan::SessionCreateInfo {
                    instance: device.instance.handle().as_raw() as *const c_void,
                    physical_device: device.physical_device.as_raw() as *const c_void,
                    device: device.device.handle().as_raw() as *const c_void,
                    queue_family_index: device.queue_family_idx,
                    queue_index: 0,
                },
            )?
        };
        // LOCAL (seated origin) rather than STAGE: always available, and the first slice has
        // no locomotion to anchor to the floor anyway.
        let space =
            session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;
        let view_space =
            session.create_reference_space(xr::ReferenceSpaceType::VIEW, xr::Posef::IDENTITY)?;
        let actions = XrActions::new(&instance, &session)?;

        let views = instance.enumerate_view_configuration_views(
            context.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let extent = vk::Extent2D {
            width: views[0].recommended_image_rect_width,
            height: views[0].recommended_image_rect_height,
        };

        // The window path renders gamma-encoded values into a UNORM image; an SRGB XR format
        // would have the blit re-encode them. Prefer the matching UNORM format and let the
        // compositor treat it as sRGB bytes (the proper fix lands with the per-eye render
        // path, which will render straight into this image).
        let formats = session.enumerate_swapchain_formats()?;
        let format = [
            vk::Format::B8G8R8A8_UNORM.as_raw() as u32,
            vk::Format::R8G8B8A8_UNORM.as_raw() as u32,
            vk::Format::B8G8R8A8_SRGB.as_raw() as u32,
            vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
        ]
        .into_iter()
        .find(|f| formats.contains(f))
        .unwrap_or(formats[0]);

        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::TRANSFER_DST,
            format,
            sample_count: 1,
            width: extent.width,
            height: extent.height,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })?;
        let images = swapchain
            .enumerate_images()?
            .into_iter()
            .map(vk::Image::from_raw)
            .collect();

        let blend_mode = instance.enumerate_environment_blend_modes(
            context.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?[0];

        // Matches the post-process pipeline's hard-coded attachment format.
        let eye_targets = [0; 2].map(|_| {
            let info = vk_init::image_info(
                extent.width,
                extent.height,
                vk::Format::B8G8R8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            let image = device.create_render_target(&info);
            let view = unsafe {
                device
                    .create_image_view(
                        &vk_init::image_view_info(image, vk::Format::B8G8R8A8_UNORM),
                        None,
                    )
                    .unwrap()
            };
            EyeTarget { image, view }
        });

        info!(
            "OpenXR session: {}x{} per eye, format {:?}",
            extent.width,
            extent.height,
            vk::Format::from_raw(format as i32)
        );

        Ok(Self {
            instance,
            session,
            frame_waiter,
            frame_stream,
            space,
            swapchain,
            images,
            extent,
            eye_targets,
            blend_mode,
            running: false,
            focused: false,
            predicted_time: None,
            view_space,
            actions,
        })
    }

    /// Syncs the action set and locates head/aim/grip at the last predicted display time.
    /// Returns the controller states plus, per [`XrTracked`], the rig-space pose when the
    /// runtime has one. None of this touches the Vulkan queue; no mutex needed.
    pub fn poll_input(&mut self) -> (XrInput, Vec<(XrTracked, Option<Transform>)>) {
        let mut input = XrInput {
            focused: self.focused,
            ..Default::default()
        };
        if self.running && self.focused {
            match self.session.sync_actions(&[(&self.actions.set).into()]) {
                Ok(()) => {
                    input.left = self.actions.hand_state(&self.session, 0);
                    input.right = self.actions.hand_state(&self.session, 1);
                }
                Err(xr::sys::Result::SESSION_NOT_FOCUSED) => input.focused = false,
                Err(e) => warn!("xr: sync_actions failed: {e:?}"),
            }
        }
        let mut poses = Vec::with_capacity(5);
        if let (true, Some(time)) = (self.running, self.predicted_time) {
            poses.push((XrTracked::Head, locate(&self.view_space, &self.space, time)));
            for hand in XrHand::ALL {
                let i = hand.index();
                poses.push((
                    XrTracked::Aim(hand),
                    locate(&self.actions.aim_spaces[i], &self.space, time),
                ));
                poses.push((
                    XrTracked::Grip(hand),
                    locate(&self.actions.grip_spaces[i], &self.space, time),
                ));
            }
        }
        (input, poses)
    }

    /// The image + view the post-process renders eye `eye` into.
    pub fn eye_target(&self, eye: usize) -> (vk::Image, vk::ImageView) {
        (self.eye_targets[eye].image, self.eye_targets[eye].view)
    }

    /// Queues the Vulkan objects this state owns; the session itself dies on drop. Call
    /// from teardown, before the device goes.
    pub fn destroy(&self, device: &RenderDevice) {
        for target in &self.eye_targets {
            device.destroyer.destroy_image_view(target.view);
            device.destroyer.destroy_image(target.image);
        }
    }

    /// Pumps session events, paces on `xrWaitFrame`, and acquires the swapchain image.
    /// Returns `None` when there is nothing to render into (session not running, or the
    /// compositor asked for a no-render frame — which is still begun and ended here).
    ///
    /// The runtime submits its own work on the shared queue inside `xrBeginFrame` /
    /// `xrEndFrame` / `xrAcquireSwapchainImage` (Monado's Vulkan path submits the
    /// compositor→app layout barrier there, `oxr_swapchain_vk.c`) /
    /// `xrReleaseSwapchainImage`, so those calls take the device's queue mutex — otherwise
    /// they race the asset workers' transfer submits (Xid 32/69, `ERROR_DEVICE_LOST` on the
    /// next fence wait; the `dev` validation layers happen to serialize it).
    pub fn begin_frame(&mut self, device: &RenderDevice) -> Option<XrFrame> {
        let mut buffer = xr::EventDataBuffer::new();
        while let Some(event) = self.instance.poll_event(&mut buffer).unwrap() {
            use xr::Event::*;
            if let SessionStateChanged(changed) = event {
                debug!("OpenXR session state: {:?}", changed.state());
                match changed.state() {
                    xr::SessionState::READY => {
                        self.session
                            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)
                            .unwrap();
                        self.running = true;
                    }
                    xr::SessionState::STOPPING => {
                        self.session.end().unwrap();
                        self.running = false;
                        self.focused = false;
                    }
                    xr::SessionState::FOCUSED => self.focused = true,
                    xr::SessionState::VISIBLE | xr::SessionState::SYNCHRONIZED => {
                        self.focused = false
                    }
                    _ => {}
                }
            }
        }
        if !self.running {
            return None;
        }

        // Pacing wait outside the lock — it can block for most of a frame.
        let frame_state = self.frame_waiter.wait().unwrap();
        self.predicted_time = Some(frame_state.predicted_display_time);
        {
            let _queue = device.queue.lock().unwrap();
            self.frame_stream.begin().unwrap();
            if !frame_state.should_render {
                self.frame_stream
                    .end(frame_state.predicted_display_time, self.blend_mode, &[])
                    .unwrap();
                return None;
            }
        }
        let index = {
            let _queue = device.queue.lock().unwrap();
            let index = self.swapchain.acquire_image().unwrap();
            self.swapchain.wait_image(xr::Duration::INFINITE).unwrap();
            index
        };
        let (_, views) = self
            .session
            .locate_views(
                xr::ViewConfigurationType::PRIMARY_STEREO,
                frame_state.predicted_display_time,
                &self.space,
            )
            .unwrap();
        Some(XrFrame {
            frame_state,
            image: self.images[index as usize],
            views,
        })
    }

    /// Records one eye's handoff: the eye target (just written by its post-process pass,
    /// ATTACHMENT_OPTIMAL) blitted 1:1 into layer `eye` of the XR image, which is left in
    /// COLOR_ATTACHMENT_OPTIMAL — the layout the compositor consumes. The blit also covers
    /// a runtime that picked a different channel order than the render target's.
    pub fn record_eye_to_layer(
        &self,
        device: &RenderDevice,
        cmd: vk::CommandBuffer,
        frame: &XrFrame,
        eye: usize,
    ) {
        let target = &self.eye_targets[eye];
        unsafe {
            layout_barrier(
                device,
                cmd,
                frame.image,
                eye as u32,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            layout_barrier(
                device,
                cmd,
                target.image,
                0,
                1,
                vk::ImageLayout::ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            let offsets = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: self.extent.width as i32,
                    y: self.extent.height as i32,
                    z: 1,
                },
            ];
            let region = vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .src_offsets(offsets)
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(eye as u32)
                        .layer_count(1),
                )
                .dst_offsets(offsets);
            device.cmd_blit_image(
                cmd,
                target.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
                vk::Filter::NEAREST,
            );

            layout_barrier(
                device,
                cmd,
                frame.image,
                eye as u32,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }
    }

    /// Releases the swapchain image and submits the projection layer, each eye declared
    /// with the pose and fov it was actually rendered with. Call after the queue submit
    /// that recorded the [`Self::record_eye_to_layer`] pair.
    pub fn end_frame(&mut self, device: &RenderDevice, frame: XrFrame) {
        let time = frame.frame_state.predicted_display_time;
        let _queue = device.queue.lock().unwrap();
        self.swapchain.release_image().unwrap();
        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: self.extent.width as i32,
                height: self.extent.height as i32,
            },
        };
        let projection_views: Vec<xr::CompositionLayerProjectionView<xr::Vulkan>> = frame
            .views
            .iter()
            .enumerate()
            .map(|(eye, view)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchain)
                            .image_rect(rect)
                            .image_array_index(eye as u32),
                    )
            })
            .collect();
        let layer = xr::CompositionLayerProjection::new()
            .space(&self.space)
            .views(&projection_views);
        if let Err(e) = self.frame_stream.end(time, self.blend_mode, &[&layer]) {
            // The runtime hands back POSE_INVALID while the headset has no tracking yet
            // (waking from dormancy, session not yet focused): drop the frame, not the game.
            log::warn!("xr: frame end failed: {e:?} (frame dropped)");
        }
    }
}

/// Full-subresource layout transition with an ALL_COMMANDS execution dependency — the mirror
/// blit is once per frame, precision buys nothing here.
unsafe fn layout_barrier(
    device: &RenderDevice,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    base_layer: u32,
    layers: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .base_array_layer(base_layer)
                .layer_count(layers),
        );
    let info = vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));
    unsafe { device.cmd_pipeline_barrier2(cmd, &info) };
}
