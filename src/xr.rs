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
use bevy::prelude::*;
use openxr as xr;

use crate::render_device::RenderDevice;
use crate::vk_init;

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
}

struct EyeTarget {
    image: vk::Image,
    view: vk::ImageView,
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
/// `Mat4::perspective_infinite_reverse_rh` (fov angles are signed, left/down negative).
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
            }
        }
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
        })
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
