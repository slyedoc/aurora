use ash::vk;
use bevy::prelude::*;
use bevy::window::RawHandleWrapper;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::ray_render_plugin::RenderWindow;
use crate::render_device::RenderDevice;

/// The format of everything drawn for a display: the window swapchain, the XR eye targets,
/// and every graphics pipeline's colour attachment (post-process, UI, gizmos). sRGB, so the
/// shaders write linear light and the hardware applies the transfer function on store and
/// blends in linear space -- the same convention as bevy's swapchain view.
pub const DISPLAY_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

#[derive(Resource)]
pub struct Swapchain {
    device: RenderDevice,
    pub surface: vk::SurfaceKHR,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_format: vk::Format,
    /// The window size the current swapchain was built for; a different `RenderWindow`
    /// triggers a rebuild in `aquire_next_image`.
    requested: (u32, u32),
    pub current_image_idx: u32,
    pub image_available_semaphore: vk::Semaphore,
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
        ash_window::create_surface(
            &entry,
            &instance,
            window.get_handle().display_handle().unwrap().as_raw(),
            window.get_handle().window_handle().unwrap().as_raw(),
            None,
        )
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
            let image_available_semaphore = device
                .device
                .create_semaphore(&semaphore_info, None)
                .unwrap();

            Swapchain {
                device,
                surface,
                swapchain: vk::SwapchainKHR::null(),
                swapchain_images: Vec::new(),
                swapchain_image_views: Vec::new(),
                swapchain_extent: vk::Extent2D::default(),
                swapchain_format: vk::Format::UNDEFINED,
                requested: (0, 0),
                image_available_semaphore,
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

            // Every graphics pipeline in this crate is created against DISPLAY_FORMAT, so
            // the surface has to offer it (every desktop driver does).
            let surface_format = formats
                .iter()
                .find(|f| f.format == DISPLAY_FORMAT)
                .unwrap_or_else(|| {
                    panic!("surface does not offer {DISPLAY_FORMAT:?}; it has {formats:?}")
                });

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
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1)
                .old_swapchain(old_swapchain);

            self.swapchain = self
                .device
                .ext_swapchain
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            self.device.destroyer.destroy_swapchain(old_swapchain);
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

            log::debug!(
                "swapchain: created {}x{} {:?}",
                surface_resolution.width,
                surface_resolution.height,
                surface_format.format
            );
        }
    }

    pub unsafe fn aquire_next_image(
        &mut self,
        window: &RenderWindow,
    ) -> (vk::Image, vk::ImageView) {
        unsafe {
            if self.swapchain == vk::SwapchainKHR::null()
                || self.requested != (window.width, window.height)
            {
                self.on_resize(window);
                self.resized = true;
            }
            // `FrameSync::wait` ran first: the previous frame's submit, which waits on
            // `image_available_semaphore`, is done, so the semaphore can be signalled again.
            // A swapchain the compositor already invalidated (a fullscreen window settling
            // in) surfaces here; rebuild it and acquire again.
            let _acquire = info_span!("acquire_next_image").entered();
            self.current_image_idx = loop {
                match self.device.ext_swapchain.acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphore,
                    vk::Fence::null(),
                ) {
                    Ok((index, _)) => break index,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        log::info!("swapchain: out of date at acquire, recreating");
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
            self.image_available_semaphore,
            self.render_finished_semaphores[self.current_image_idx as usize],
        )
    }

    /// Present the acquired image once the submit has signalled it. The caller holds the queue;
    /// hand the result to [`Self::after_present`] after letting it go.
    pub unsafe fn present(&self, queue: vk::Queue) -> ash::prelude::VkResult<bool> {
        unsafe {
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(std::slice::from_ref(
                    &self.render_finished_semaphores[self.current_image_idx as usize],
                ))
                .swapchains(std::slice::from_ref(&self.swapchain))
                .image_indices(std::slice::from_ref(&self.current_image_idx));
            let _present = info_span!("queue_present").entered();
            self.device
                .ext_swapchain
                .queue_present(queue, &present_info)
        }
    }

    /// Rebuild after a present the compositor called out of date.
    pub unsafe fn after_present(
        &mut self,
        result: ash::prelude::VkResult<bool>,
        window: &RenderWindow,
    ) {
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
        log::info!("Dropping Swapchain");
        unsafe {
            {
                let queue = self.device.queue.lock().unwrap();
                self.device.queue_wait_idle(*queue).unwrap();
            }

            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }

            for &image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.device
                .ext_swapchain
                .destroy_swapchain(self.swapchain, None);

            self.device.ext_surface.destroy_surface(self.surface, None);
        }
    }
}
