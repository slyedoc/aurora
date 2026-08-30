//! The NGX session and one view's Ray Reconstruction feature: its guide images (written by
//! the raygen through storage-image bindings), the evaluate, and the output image.

use ash::vk;
use bevy::math::Mat4;

use super::{
    AuroraDlss, DlssPlan, GuideViews, feature_info,
    ngx::{self, *},
    suggested_jitter,
};
use crate::{render_device::RenderDevice, vk_init};

const F_COLOR: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const F_NORMAL_ROUGHNESS: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const F_ALBEDO: vk::Format = vk::Format::R8G8B8A8_UNORM;
const F_DEPTH: vk::Format = vk::Format::R32_SFLOAT;
const F_MOTION: vk::Format = vk::Format::R16G16_SFLOAT;

struct DlssImage {
    image: vk::Image,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
    /// `VK_IMAGE_USAGE_STORAGE_BIT` == `NVSDK_NGX_Resource_VK::ReadWrite`; the output needs it.
    storage: bool,
}

impl DlssImage {
    fn new(rd: &RenderDevice, format: vk::Format, extent: vk::Extent2D, storage: bool) -> Self {
        let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        if storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let info = vk_init::image_info(extent.width.max(1), extent.height.max(1), format, usage);
        let image = rd.create_render_target(&info);
        let view = unsafe {
            rd.device
                .create_image_view(&vk_init::image_view_info(image, format), None)
                .unwrap()
        };
        Self {
            image,
            view,
            format,
            extent,
            storage,
        }
    }

    fn resource(&self) -> NgxResourceVk {
        ngx::image_view_resource(
            self.view,
            self.image,
            self.format,
            vk::ImageAspectFlags::COLOR,
            self.extent.width,
            self.extent.height,
            self.storage,
        )
    }

    fn destroy(&self, rd: &RenderDevice) {
        rd.destroyer.destroy_image_view(self.view);
        rd.destroyer.destroy_image(self.image);
    }
}

struct DlssView {
    handle: *mut NgxHandle,
    mode: AuroraDlss,
    render: vk::Extent2D,
    output: vk::Extent2D,
    color: DlssImage,
    normal_roughness: DlssImage,
    diffuse: DlssImage,
    specular: DlssImage,
    depth: DlssImage,
    spec_hit: DlssImage,
    motion: DlssImage,
    /// 1x1 R32F holding 1.0: without it NGX auto-exposes and the image re-normalises over
    /// seconds.
    exposure: DlssImage,
    output_image: DlssImage,
    first_evaluate: bool,
    /// Guides have been laid out once (UNDEFINED -> GENERAL); after that they cycle
    /// GENERAL <-> SHADER_READ_ONLY.
    guides_initialized: bool,
}

impl DlssView {
    /// What the raygen writes, binding order 1..=7.
    fn guides(&self) -> [&DlssImage; 7] {
        [
            &self.normal_roughness,
            &self.diffuse,
            &self.specular,
            &self.depth,
            &self.spec_hit,
            &self.motion,
            &self.color,
        ]
    }
}

pub struct DlssRenderer {
    params: *mut NgxParameter,
    rr_available: bool,
    view: Option<DlssView>,
    frame: u32,
}

// Raw NGX pointers; the session is only ever driven from the render systems.
unsafe impl Send for DlssRenderer {}
unsafe impl Sync for DlssRenderer {}

fn get_i(params: *mut NgxParameter, name: &std::ffi::CStr) -> Option<i32> {
    let mut v = 0i32;
    let r = unsafe { NVSDK_NGX_Parameter_GetI(params, name.as_ptr(), &mut v) };
    (r == RESULT_SUCCESS).then_some(v)
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn image_barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range())
}

impl DlssRenderer {
    pub fn new(rd: &RenderDevice) -> Option<Self> {
        let init = feature_info(FEATURE_RAY_RECONSTRUCTION);
        let r = unsafe {
            NVSDK_NGX_VULKAN_Init_with_ProjectID(
                init.project_id(),
                ENGINE_TYPE_CUSTOM,
                init.engine_version(),
                init.data_path(),
                rd.instance.handle(),
                rd.physical_device,
                rd.device.handle(),
                rd.entry.static_fn().get_instance_proc_addr,
                rd.instance.fp_v1_0().get_device_proc_addr,
                init.common(),
                NVSDK_NGX_VERSION_API,
            )
        };
        if r != RESULT_SUCCESS {
            log::info!(
                "dlss: NVSDK_NGX_VULKAN_Init_with_ProjectID -> {} -- running without DLSS",
                ngx::result_name(r)
            );
            return None;
        }
        let mut params: *mut NgxParameter = std::ptr::null_mut();
        let r = unsafe { NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut params) };
        if r != RESULT_SUCCESS || params.is_null() {
            log::info!(
                "dlss: NVSDK_NGX_VULKAN_GetCapabilityParameters -> {}",
                ngx::result_name(r)
            );
            unsafe { NVSDK_NGX_VULKAN_Shutdown1(rd.device.handle()) };
            return None;
        }
        let rr_available = get_i(params, P_SUPERSAMPLING_DENOISING_AVAILABLE).unwrap_or(0) != 0;
        let init_result = get_i(params, P_SUPERSAMPLING_FEATURE_INIT_RESULT).unwrap_or(0);
        log::info!(
            "dlss: RayReconstruction available = {rr_available}, \
             SuperSampling.FeatureInitResult = {}, min driver {}.{}",
            ngx::result_name(init_result as u32),
            get_i(params, P_SUPERSAMPLING_MIN_DRIVER_MAJOR).unwrap_or(-1),
            get_i(params, P_SUPERSAMPLING_MIN_DRIVER_MINOR).unwrap_or(-1),
        );
        if !rr_available {
            log::warn!(
                "dlss: Ray Reconstruction unavailable. If FeatureInitResult is \
                 FAIL_PlatformError, the device is missing VK_NVX_binary_import / \
                 VK_NVX_image_view_handle"
            );
        }
        Some(Self {
            params,
            rr_available,
            view: None,
            frame: 0,
        })
    }

    /// Settles the frame's plan: (re)creates the feature when the mode or output size
    /// changed. Call before recording -- feature creation submits and waits on its own.
    pub fn prepare(
        &mut self,
        rd: &RenderDevice,
        output: vk::Extent2D,
        mode: AuroraDlss,
    ) -> Option<DlssPlan> {
        if !self.rr_available || !mode.is_on() || output.width == 0 || output.height == 0 {
            if mode.is_on() && self.view.is_some() {
                self.release_view(rd);
            }
            if !mode.is_on() && self.view.is_some() {
                self.release_view(rd);
            }
            return None;
        }
        let quality = mode.perf_quality()?;
        let fresh = self
            .view
            .as_ref()
            .is_none_or(|v| v.mode != mode || v.output != output);
        if fresh {
            let render = if mode == AuroraDlss::Dlaa {
                output
            } else {
                match unsafe {
                    ngx::get_optimal_settings(
                        self.params,
                        true,
                        [output.width, output.height],
                        quality,
                    )
                } {
                    Ok(s) if s.optimal[0] > 0 && s.optimal[1] > 0 => vk::Extent2D {
                        width: s.optimal[0],
                        height: s.optimal[1],
                    },
                    Ok(_) => output,
                    Err(e) => {
                        log::error!("dlss: NGX_DLSSD_GET_OPTIMAL_SETTINGS: {e}");
                        return None;
                    }
                }
            };
            self.rebuild(rd, mode, render, output, quality)?;
        }
        let view = self.view.as_ref()?;
        self.frame = self.frame.wrapping_add(1);
        Some(DlssPlan {
            render: view.render,
            jitter: suggested_jitter(self.frame, view.render.width, view.output.width),
        })
    }

    /// Drains the whole device under the queue lock. NGX's feature lifecycle wants
    /// `vkDeviceWaitIdle` (not just our queue idle: the feature may own work on queues we
    /// never see) before `ReleaseFeature` and before `CreateFeature1`, which records GPU
    /// work and reconfigures the feature's allocations. Running either alongside an in-flight
    /// trace is a hard GPU hang -- Xid 79, the card falls off the bus.
    fn wait_idle(rd: &RenderDevice) {
        let _queue = rd.queue.lock().unwrap();
        let _ = unsafe { rd.device.device_wait_idle() };
    }

    fn rebuild(
        &mut self,
        rd: &RenderDevice,
        mode: AuroraDlss,
        render: vk::Extent2D,
        output: vk::Extent2D,
        quality: i32,
    ) -> Option<()> {
        // DRAIN FIRST, ALWAYS -- including the very first creation, where `release_view` has
        // nothing to release. `prepare` runs before the frame's fence wait, so on a runtime
        // mode switch or a resize the previous frame (a full trace + TLAS build) is still
        // executing when NGX would otherwise record and submit its feature creation alongside
        // it. This only moves on a mode change or a resize, so the cost is nil.
        log::info!(
            "dlss: rebuild -> {mode} {}x{} -> {}x{}: draining the device",
            render.width,
            render.height,
            output.width,
            output.height
        );
        Self::wait_idle(rd);
        self.release_view(rd);
        log::info!("dlss: rebuild: creating images + feature");
        let one = vk::Extent2D {
            width: 1,
            height: 1,
        };
        let color = DlssImage::new(rd, F_COLOR, render, true);
        let normal_roughness = DlssImage::new(rd, F_NORMAL_ROUGHNESS, render, true);
        let diffuse = DlssImage::new(rd, F_ALBEDO, render, true);
        let specular = DlssImage::new(rd, F_ALBEDO, render, true);
        let depth = DlssImage::new(rd, F_DEPTH, render, true);
        let spec_hit = DlssImage::new(rd, F_DEPTH, render, true);
        let motion = DlssImage::new(rd, F_MOTION, render, true);
        let exposure = DlssImage::new(rd, F_DEPTH, one, false);
        let output_image = DlssImage::new(rd, F_COLOR, output, true);

        let feature_create_flags = DLSS_FLAG_IS_HDR | DLSS_FLAG_MV_LOW_RES;
        let create = NgxDlssdCreateParams {
            denoise_mode: DENOISE_MODE_DL_UNIFIED,
            roughness_mode: ROUGHNESS_MODE_PACKED,
            use_hw_depth: DEPTH_TYPE_LINEAR,
            width: render.width,
            height: render.height,
            target_width: output.width,
            target_height: output.height,
            perf_quality_value: quality,
            feature_create_flags,
            enable_output_subrects: false,
        };
        let mut handle: Result<*mut NgxHandle, String> = Err("not created".into());
        rd.run_transfer_commands(|cb| {
            handle =
                unsafe { ngx::create_dlssd_feature(rd.device.handle(), cb, self.params, &create) };
        });
        let handle = match handle {
            Ok(h) => h,
            Err(e) => {
                log::error!("dlss: feature creation failed: {e}");
                for image in [
                    &color,
                    &normal_roughness,
                    &diffuse,
                    &specular,
                    &depth,
                    &spec_hit,
                    &motion,
                    &exposure,
                    &output_image,
                ] {
                    image.destroy(rd);
                }
                return None;
            }
        };
        if let Ok((bytes, opt_level, dev_branch)) = unsafe { ngx::get_stats(self.params, true) } {
            log::info!(
                "dlss: {mode} feature {}x{} -> {}x{}, VRAM {:.1} MiB, snippet opt level \
                 {opt_level}, dev branch {dev_branch}",
                render.width,
                render.height,
                output.width,
                output.height,
                bytes as f64 / (1024.0 * 1024.0),
            );
        }
        self.view = Some(DlssView {
            handle,
            mode,
            render,
            output,
            color,
            normal_roughness,
            diffuse,
            specular,
            depth,
            spec_hit,
            motion,
            exposure,
            output_image,
            first_evaluate: true,
            guides_initialized: false,
        });
        Some(())
    }

    fn release_view(&mut self, rd: &RenderDevice) {
        let Some(view) = self.view.take() else {
            return;
        };
        log::info!(
            "dlss: releasing the {} feature (draining the device)",
            view.mode
        );
        Self::wait_idle(rd);
        unsafe { NVSDK_NGX_VULKAN_ReleaseFeature(view.handle) };
        log::info!("dlss: feature released");
        for image in view
            .guides()
            .into_iter()
            .chain([&view.exposure, &view.output_image])
        {
            image.destroy(rd);
        }
    }

    pub fn guide_views(&self) -> Option<GuideViews> {
        let v = self.view.as_ref()?;
        Some(GuideViews {
            normal_roughness: v.normal_roughness.view,
            diffuse: v.diffuse.view,
            specular: v.specular.view,
            depth: v.depth.view,
            spec_hit: v.spec_hit.view,
            motion: v.motion.view,
            color: v.color.view,
        })
    }

    pub fn output_view(&self) -> Option<vk::ImageView> {
        self.view.as_ref().map(|v| v.output_image.view)
    }

    /// Before the trace: guides writable (GENERAL); the exposure texel written once.
    pub fn record_pre_trace(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        let Some(view) = self.view.as_mut() else {
            return;
        };
        let old = if view.guides_initialized {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let mut barriers: Vec<_> = view
            .guides()
            .iter()
            .map(|image| {
                image_barrier(
                    image.image,
                    old,
                    vk::ImageLayout::GENERAL,
                    (
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::MEMORY_READ,
                    ),
                    (
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    ),
                )
            })
            .collect();
        if !view.guides_initialized {
            barriers.push(image_barrier(
                view.exposure.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::empty(),
                ),
                (
                    vk::PipelineStageFlags2::ALL_TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            ));
        }
        unsafe {
            rd.ext_sync2.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            if !view.guides_initialized {
                rd.device.cmd_clear_color_image(
                    cmd,
                    view.exposure.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {
                        float32: [1.0, 0.0, 0.0, 0.0],
                    },
                    std::slice::from_ref(&color_range()),
                );
                let done = image_barrier(
                    view.exposure.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    (
                        vk::PipelineStageFlags2::ALL_TRANSFER,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                    (
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::SHADER_READ,
                    ),
                );
                rd.ext_sync2.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(std::slice::from_ref(&done)),
                );
            }
        }
        view.guides_initialized = true;
    }

    /// After the trace: guides readable, output writable, NGX evaluate, output readable by
    /// the post-process sampler.
    pub fn record_evaluate(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        view_from_world: Mat4,
        clip_from_view: Mat4,
        jitter: [f32; 2],
        reset: bool,
    ) {
        let Some(view) = self.view.as_mut() else {
            return;
        };
        let barriers: Vec<_> = view
            .guides()
            .iter()
            .map(|image| {
                image_barrier(
                    image.image,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    (
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    ),
                    (
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::AccessFlags2::SHADER_READ,
                    ),
                )
            })
            .chain([image_barrier(
                view.output_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ,
                ),
                (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            )])
            .collect();
        unsafe {
            rd.ext_sync2.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
        }
        let mut r_color = view.color.resource();
        let mut r_output = view.output_image.resource();
        let mut r_depth = view.depth.resource();
        let mut r_motion = view.motion.resource();
        let mut r_diffuse = view.diffuse.resource();
        let mut r_specular = view.specular.resource();
        let mut r_normals = view.normal_roughness.resource();
        let mut r_spec_hit = view.spec_hit.resource();
        let mut r_exposure = view.exposure.resource();
        let mut camera = RrEvalCamera {
            world_to_view: view_from_world.transpose().to_cols_array(),
            view_to_clip: clip_from_view.transpose().to_cols_array(),
        };
        let first = std::mem::replace(&mut view.first_evaluate, false);
        let result = unsafe {
            ngx::evaluate_dlssd(
                cmd,
                view.handle,
                self.params,
                RrEvalResources {
                    color: &mut r_color,
                    output: &mut r_output,
                    depth: &mut r_depth,
                    motion_vectors: &mut r_motion,
                    diffuse_albedo: &mut r_diffuse,
                    specular_albedo: &mut r_specular,
                    normals_roughness: &mut r_normals,
                    specular_hit_distance: &mut r_spec_hit,
                    exposure: Some(&mut r_exposure),
                },
                &mut camera,
                [view.render.width, view.render.height],
                [-jitter[0], -jitter[1]],
                reset || first,
                [-(view.render.width as f32), -(view.render.height as f32)],
            )
        };
        if let Err(e) = result {
            log::error!("dlss: {e}");
        }
        let readable = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ);
        unsafe {
            rd.ext_sync2.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&readable)),
            );
        }
    }

    pub fn destroy(&mut self, rd: &RenderDevice) {
        self.release_view(rd);
        unsafe {
            NVSDK_NGX_VULKAN_DestroyParameters(self.params);
            NVSDK_NGX_VULKAN_Shutdown1(rd.device.handle());
        }
    }
}
