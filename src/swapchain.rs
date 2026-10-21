use ash::vk;
use ash::vk::TaggedStructure as _;
use bevy::prelude::*;
use bevy::window::RawHandleWrapper;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::color_target::WINDOW_CANVAS_FORMAT;
use crate::post_process_filter::Output;
use crate::ray_render_plugin::RenderWindow;
use crate::render_device::RenderDevice;
use crate::render_texture::{RenderTexture, create_blank_texture};

/// The format of an 8-bit display target: the XR eye targets, UI surfaces and image targets.
/// sRGB, so the shaders write linear light and the hardware applies the transfer function on
/// store and blends in linear space. The window instead draws into a float canvas
/// ([`WINDOW_CANVAS_FORMAT`]) that one pass encodes into the swapchain.
pub const DISPLAY_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

/// The swapchain's format and colour space, best first: HDR10, scRGB, 10-bit sRGB, 8-bit sRGB.
fn pick_surface_format(formats: &[vk::SurfaceFormatKHR]) -> (vk::SurfaceFormatKHR, Output) {
    use vk::{ColorSpaceKHR as C, Format as F};
    let ten = [F::A2B10G10R10_UNORM_PACK32, F::A2R10G10B10_UNORM_PACK32];
    let wanted: [(&[F], C, Output); 4] = [
        (&ten, C::HDR10_ST2084_EXT, Output::Pq10),
        (&[F::R16G16B16A16_SFLOAT], C::EXTENDED_SRGB_LINEAR_EXT, Output::Scrgb),
        (&ten, C::SRGB_NONLINEAR, Output::Srgb10),
        (&[F::B8G8R8A8_UNORM], C::SRGB_NONLINEAR, Output::Srgb8),
    ];
    for (fmts, space, output) in wanted {
        for &format in fmts {
            if let Some(f) = formats
                .iter()
                .find(|f| f.format == format && f.color_space == space)
            {
                return (*f, output);
            }
        }
    }
    panic!("surface offers none of aurora's swapchain formats; it has {formats:?}")
}

#[derive(Resource)]
pub struct Swapchain {
    device: RenderDevice,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_format: vk::Format,
    /// How the canvas is encoded into the swapchain (its format and colour space).
    pub encode: Output,
    /// The window's linear canvas, the swapchain's size: the frame, gizmos and UI draw here,
    /// then the encode pass writes it into the acquired image.
    pub canvas: RenderTexture,
    /// The window size the current swapchain was built for; a different `RenderWindow`
    /// triggers a rebuild in `aquire_next_image`.
    requested: (u32, u32),
    /// Bumped on every (re)creation: Reflex's sleep mode is per swapchain (latency.rs).
    pub generation: u64,
    pub current_image_idx: u32,
    /// Acquire semaphores, one per frame parity: the next frame acquires while the previous
    /// frame's submit may still be waiting on its own.
    pub image_available_semaphores: [vk::Semaphore; 2],
    /// The parity of the image acquired last.
    available: usize,
    /// One per swapchain image, indexed by the acquired image: a present's wait on it is not
    /// covered by the in-flight fence, so a single shared semaphore could be re-signaled while
    /// still pending.
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub resized: bool,
}

unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &RawHandleWrapper,
) -> vk::SurfaceKHR {
    unsafe {
        ash_window::SurfaceFactory::new(
            entry,
            instance,
            window.get_handle().display_handle().unwrap().as_raw(),
        )
        .unwrap()
        .create_surface(window.get_handle().window_handle().unwrap().as_raw(), None)
        .unwrap()
    }
}

impl Swapchain {
    pub unsafe fn from_window(device: RenderDevice, window: &RawHandleWrapper) -> Self {
        unsafe {
            let surface = create_surface(&device.entry, &device.instance, window);
            device
                .ext_surface
                .get_physical_device_surface_support(
                    device.physical_device,
                    device.queue_family_idx,
                    surface,
                )
                .unwrap();
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let image_available_semaphores = std::array::from_fn(|_| {
                device
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            });

            Swapchain {
                device,
                surface,
                swapchain: vk::SwapchainKHR::null(),
                swapchain_images: Vec::new(),
                swapchain_image_views: Vec::new(),
                swapchain_extent: vk::Extent2D::default(),
                swapchain_format: vk::Format::UNDEFINED,
                encode: Output::Srgb8,
                canvas: RenderTexture::default(),
                requested: (0, 0),
                image_available_semaphores,
                available: 0,
                generation: 0,
                render_finished_semaphores: Vec::new(),
                current_image_idx: 0,
                resized: false,
            }
        }
    }

    pub unsafe fn on_resize(&mut self, window: &RenderWindow) {
        unsafe {
            {
                let queue = self.device.queue.lock().unwrap();
                self.device.queue_wait_idle(*queue).unwrap();
            }
            let formats = self
                .device
                .ext_surface
                .get_physical_device_surface_formats(self.device.physical_device, self.surface)
                .unwrap();
            debug!("surface formats: {formats:?}");
            let (surface_format, encode) = pick_surface_format(&formats);
            self.encode = encode;

            let surface_caps = self
                .device
                .ext_surface
                .get_physical_device_surface_capabilities(self.device.physical_device, self.surface)
                .unwrap();

            let mut desired_image_count = surface_caps.min_image_count + 1;
            if surface_caps.max_image_count > 0
                && desired_image_count > surface_caps.max_image_count
            {
                desired_image_count = surface_caps.max_image_count;
            }

            let surface_resolution = match surface_caps.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window
                        .width
                        .min(surface_caps.max_image_extent.width)
                        .max(surface_caps.min_image_extent.width),
                    height: window
                        .height
                        .min(surface_caps.max_image_extent.height)
                        .max(surface_caps.min_image_extent.height),
                },
                _ => surface_caps.current_extent,
            };

            self.swapchain_extent = surface_resolution;
            self.swapchain_format = surface_format.format;
            self.requested = (window.width, window.height);
            if self.canvas.image != vk::Image::null() {
                self.device.destroyer.destroy_image_view(self.canvas.image_view);
                self.device.destroyer.destroy_image(self.canvas.image);
            }
            self.canvas = create_blank_texture(
                &self.device,
                WINDOW_CANVAS_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    // The XR spectator is blitted in; screenshots blit it out.
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                surface_resolution.width,
                surface_resolution.height,
            );

            let pre_transform = if surface_caps
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_caps.current_transform
            };
            let present_modes = self
                .device
                .ext_surface
                .get_physical_device_surface_present_modes(
                    self.device.physical_device,
                    self.surface,
                )
                .unwrap();

            // `AURORA_PRESENT_MODE` = fifo (default) | mailbox | immediate. FIFO is the default
            // because MAILBOX on the NVIDIA Wayland WSI (610.43.02, COSMIC) takes the GPU off
            // the bus (Xid 79, reboot) the moment the compositor resizes the window while the
            // renderer is at full load -- four times on 2026-08-30, none with FIFO. Use
            // mailbox / immediate for uncapped fps measurements only, and don't resize.
            let wanted = match std::env::var("AURORA_PRESENT_MODE").as_deref() {
                Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
                Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
                _ => vk::PresentModeKHR::FIFO,
            };
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == wanted)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let old_swapchain = self.swapchain;
            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                // TRANSFER_SRC so a screenshot request can copy the presented frame out.
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        // The XR spectator is blitted in (ray_render_plugin.rs).
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .old_swapchain(old_swapchain);
            // Reflex (latency.rs): a swapchain takes latency markers and sleeps only when
            // created for them.
            let mut latency = vk::SwapchainLatencyCreateInfoNV::default().latency_mode_enable(true);
            let swapchain_create_info = if self.device.ext_low_latency.is_some() {
                swapchain_create_info.push(&mut latency)
            } else {
                swapchain_create_info
            };

            self.swapchain = self
                .device
                .ext_swapchain
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            self.device.destroyer.destroy_swapchain(old_swapchain);
            self.generation += 1;
            for image_view in self.swapchain_image_views.drain(..) {
                self.device.destroyer.destroy_image_view(image_view);
            }
            // The queue is idle (above), so nothing still waits on these.
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }

            self.swapchain_images = self
                .device
                .ext_swapchain
                .get_swapchain_images(self.swapchain)
                .unwrap();

            self.swapchain_image_views = self
                .swapchain_images
                .iter()
                .map(|image| {
                    let view_info =
                        crate::vk_init::image_view_info(image.clone(), surface_format.format);
                    self.device.create_image_view(&view_info, None).unwrap()
                })
                .collect();

            self.render_finished_semaphores = self
                .swapchain_images
                .iter()
                .map(|_| {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap()
                })
                .collect();

            log::info!(
                "swapchain: {}x{} {:?} {:?} ({encode:?})",
                surface_resolution.width,
                surface_resolution.height,
                surface_format.format,
                surface_format.color_space
            );
        }
    }

    /// Acquires the next image for frame `frame` (its parity picks the acquire semaphore).
    pub unsafe fn aquire_next_image(
        &mut self,
        window: &RenderWindow,
        frame: usize,
    ) -> (vk::Image, vk::ImageView) {
        self.available = frame % 2;
        unsafe {
            if self.swapchain == vk::SwapchainKHR::null()
                || self.requested != (window.width, window.height)
            {
                self.on_resize(window);
                self.resized = true;
            }
            // This parity's semaphore was last waited by the submit two frames back, which has
            // finished (one frame in flight), so it can be signalled again.
            // A swapchain the compositor already invalidated (a fullscreen window settling
            // in) surfaces here; rebuild it and acquire again.
            let _acquire = info_span!("acquire_next_image").entered();
            self.current_image_idx = loop {
                match self.device.ext_swapchain.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphores[self.available],
                    vk::Fence::null(),
                ) {
                    Ok((index, _)) => break index,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        log::debug!("swapchain: out of date at acquire, recreating");
                        self.on_resize(window);
                    }
                    Err(e) => panic!("acquire_next_image failed: {e:?}"),
                }
            };

            return (
                self.swapchain_images[self.current_image_idx as usize],
                self.swapchain_image_views[self.current_image_idx as usize],
            );
        }
    }

    /// The semaphore this frame's submit waits on before writing the acquired image, and the
    /// one it signals for the present.
    pub fn frame_semaphores(&self) -> (vk::Semaphore, vk::Semaphore) {
        (
            self.image_available_semaphores[self.available],
            self.render_finished_semaphores[self.current_image_idx as usize],
        )
    }

    /// Present the acquired image once the submit has signalled it, tagged `present_id` when
    /// Reflex is on (latency.rs). The caller holds the queue; hand the result to
    /// [`Self::after_present`] after letting it go.
    pub unsafe fn present(&self, queue: vk::Queue, present_id: Option<u64>) -> ash::VkResult<bool> {
        unsafe {
            let ids = [present_id.unwrap_or(0)];
            let mut id_info = vk::PresentIdKHR::default().present_ids(&ids);
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(
                    &self.render_finished_semaphores[self.current_image_idx as usize],
                ))
                .swapchains(std::slice::from_ref(&self.swapchain))
                .image_indices(std::slice::from_ref(&self.current_image_idx));
            let present_info = if present_id.is_some() {
                present_info.push(&mut id_info)
            } else {
                present_info
            };
            let _present = info_span!("queue_present").entered();
            let result = self
                .device
                .ext_swapchain
                .queue_present(queue, &present_info);
            static PRESENTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !PRESENTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::debug!("first frame presented");
            }
            // Tracy's frame boundary (bevy_render's mark, which this renderer replaces).
            #[cfg(feature = "trace_tracy")]
            bevy::log::event!(
                bevy::log::Level::INFO,
                message = "finished frame",
                tracy.frame_mark = true
            );
            result
        }
    }

    /// Rebuild after a present the compositor called out of date.
    pub unsafe fn after_present(&mut self, result: ash::VkResult<bool>, window: &RenderWindow) {
        unsafe {
            match result {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                    log::debug!("------ SWAPCHAIN OUT OF DATE ------");
                    self.on_resize(window);
                    self.resized = true;
                }
                Err(e) => {
                    crate::aftermath::note_device_lost(e);
                    panic!("Failed to present swapchain image: {:?}", e)
                }
                Ok(false) => {
                    self.resized = false;
                }
            }
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        log::debug!("Dropping Swapchain");
        unsafe {
            {
                let queue = self.device.queue.lock().unwrap();
                self.device.queue_wait_idle(*queue).unwrap();
            }

            for semaphore in self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }

            for &image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            if self.canvas.image != vk::Image::null() {
                self.device.destroyer.destroy_image_view(self.canvas.image_view);
                self.device.destroyer.destroy_image(self.canvas.image);
            }
            self.device
                .ext_swapchain
                .destroy_swapchain(self.swapchain, None);

            self.device.ext_surface.destroy_surface(self.surface, None);
        }
    }
}
