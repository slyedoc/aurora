//! Hand-written NGX (NVIDIA DLSS SDK) FFI — only what aurora needs.
//!
//! Lifted verbatim from `examples/dlss_spike/ngx.rs` (S14's spike, which proved
//! the whole call sequence on raw `ash` with zero validation output; see
//! `docs/dlss_notes.md`). Kept dependency-free — no `bindgen`, no `cc`.
//!
//! Layouts and call sequences transcribed from `$DLSS_SDK/include/`:
//! `nvsdk_ngx_defs.h`, `nvsdk_ngx_defs_vk.h`, `nvsdk_ngx_params.h`,
//! `nvsdk_ngx_vk.h`, `nvsdk_ngx_defs_dlssd.h`, `nvsdk_ngx_params_dlssd.h`.
//!
//! The `NGX_*` helpers at the bottom are Rust re-implementations of the
//! `static inline` C helpers in `nvsdk_ngx_helpers*.h` — those are header-only,
//! so bindgen users need `wrap_static_fns` + a `cc` shim to reach them. Doing
//! them by hand keeps aurora free of `bindgen`/`cc` build-dependencies.
//!
//! Structure/idioms cross-checked against jms55's `dlss_wgpu`
//! (<https://github.com/bevyengine/dlss_wgpu>, MIT/Apache-2.0) — specifically
//! `src/feature_info.rs` (FeatureDiscoveryInfo construction, the logging-sink
//! caveat) and `src/sdk.rs` / `src/ray_reconstruction.rs` (call order).

#![allow(non_snake_case, dead_code)]

use std::ffi::{CStr, CString, OsStr, c_char, c_void};

use ash::vk;

// `wchar_t` is a signed 32-bit int on Linux/glibc.
pub type WChar = i32;

pub type NgxResult = u32;

pub const NVSDK_NGX_VERSION_API: u32 = 0x0000015;

pub const RESULT_SUCCESS: NgxResult = 0x1;
pub const RESULT_FAIL: NgxResult = 0xBAD0_0000;

// NVSDK_NGX_Feature
pub const FEATURE_SUPERSAMPLING: u32 = 1;
pub const FEATURE_RAY_RECONSTRUCTION: u32 = 13;

// NVSDK_NGX_EngineType
pub const ENGINE_TYPE_CUSTOM: u32 = 0;

// NVSDK_NGX_Application_Identifier_Type
pub const APP_IDENTIFIER_TYPE_PROJECT_ID: u32 = 1;

// NVSDK_NGX_Logging_Level
pub const LOGGING_LEVEL_OFF: u32 = 0;
pub const LOGGING_LEVEL_ON: u32 = 1;
pub const LOGGING_LEVEL_VERBOSE: u32 = 2;

// NVSDK_NGX_PerfQuality_Value
pub const PERF_QUALITY_MAX_PERF: i32 = 0;
pub const PERF_QUALITY_BALANCED: i32 = 1;
pub const PERF_QUALITY_MAX_QUALITY: i32 = 2;
pub const PERF_QUALITY_ULTRA_PERFORMANCE: i32 = 3;
pub const PERF_QUALITY_ULTRA_QUALITY: i32 = 4;
pub const PERF_QUALITY_DLAA: i32 = 5;

// NVSDK_NGX_DLSS_Feature_Flags
pub const DLSS_FLAG_IS_HDR: i32 = 1 << 0;
pub const DLSS_FLAG_MV_LOW_RES: i32 = 1 << 1;
pub const DLSS_FLAG_MV_JITTERED: i32 = 1 << 2;
pub const DLSS_FLAG_DEPTH_INVERTED: i32 = 1 << 3;
pub const DLSS_FLAG_AUTO_EXPOSURE: i32 = 1 << 6;
pub const DLSS_FLAG_ALPHA_UPSCALING: i32 = 1 << 7;

// NVSDK_NGX_DLSS_Denoise_Mode / Roughness_Mode / Depth_Type
pub const DENOISE_MODE_OFF: u32 = 0;
pub const DENOISE_MODE_DL_UNIFIED: u32 = 1;
pub const ROUGHNESS_MODE_UNPACKED: u32 = 0;
pub const ROUGHNESS_MODE_PACKED: u32 = 1;
pub const DEPTH_TYPE_LINEAR: u32 = 0;
pub const DEPTH_TYPE_HW: u32 = 1;

// ---- Parameter name strings (nvsdk_ngx_defs.h / nvsdk_ngx_defs_dlssd.h) ----
pub const P_SUPERSAMPLING_AVAILABLE: &CStr = c"SuperSampling.Available";
pub const P_SUPERSAMPLING_DENOISING_AVAILABLE: &CStr = c"SuperSamplingDenoising.Available";
pub const P_SUPERSAMPLING_NEEDS_UPDATED_DRIVER: &CStr = c"SuperSampling.NeedsUpdatedDriver";
pub const P_SUPERSAMPLING_MIN_DRIVER_MAJOR: &CStr = c"SuperSampling.MinDriverVersionMajor";
pub const P_SUPERSAMPLING_MIN_DRIVER_MINOR: &CStr = c"SuperSampling.MinDriverVersionMinor";
pub const P_SUPERSAMPLING_FEATURE_INIT_RESULT: &CStr = c"SuperSampling.FeatureInitResult";
pub const P_WIDTH: &CStr = c"Width";
pub const P_HEIGHT: &CStr = c"Height";
pub const P_OUT_WIDTH: &CStr = c"OutWidth";
pub const P_OUT_HEIGHT: &CStr = c"OutHeight";
pub const P_SHARPNESS: &CStr = c"Sharpness";
pub const P_PERF_QUALITY_VALUE: &CStr = c"PerfQualityValue";
pub const P_RTX_VALUE: &CStr = c"RTXValue";
pub const P_CREATION_NODE_MASK: &CStr = c"CreationNodeMask";
pub const P_VISIBILITY_NODE_MASK: &CStr = c"VisibilityNodeMask";
pub const P_DLSS_FEATURE_CREATE_FLAGS: &CStr = c"DLSS.Feature.Create.Flags";
pub const P_DLSS_ENABLE_OUTPUT_SUBRECTS: &CStr = c"DLSS.Enable.Output.Subrects";
pub const P_DLSS_DENOISE_MODE: &CStr = c"DLSS.Denoise.Mode";
pub const P_DLSS_ROUGHNESS_MODE: &CStr = c"DLSS.Roughness.Mode";
// RR render-preset hints, one per PerfQuality mode (nvsdk_ngx_defs_dlssd.h). 0 = Default
// (currently resolves to D), 4 = D (transformer), 5 = E (latest transformer).
pub const P_RR_PRESET_DLAA: &CStr = c"RayReconstruction.Hint.Render.Preset.DLAA";
pub const P_RR_PRESET_QUALITY: &CStr = c"RayReconstruction.Hint.Render.Preset.Quality";
pub const P_RR_PRESET_BALANCED: &CStr = c"RayReconstruction.Hint.Render.Preset.Balanced";
pub const P_RR_PRESET_PERFORMANCE: &CStr = c"RayReconstruction.Hint.Render.Preset.Performance";
pub const P_RR_PRESET_ULTRA_PERFORMANCE: &CStr =
    c"RayReconstruction.Hint.Render.Preset.UltraPerformance";
pub const P_RR_PRESET_ULTRA_QUALITY: &CStr = c"RayReconstruction.Hint.Render.Preset.UltraQuality";
pub const P_DLSS_USE_HW_DEPTH: &CStr = c"DLSS.Use.HW.Depth";
pub const P_DLSS_GET_DYNAMIC_MAX_RENDER_WIDTH: &CStr = c"DLSS.Get.Dynamic.Max.Render.Width";
pub const P_DLSS_GET_DYNAMIC_MAX_RENDER_HEIGHT: &CStr = c"DLSS.Get.Dynamic.Max.Render.Height";
pub const P_DLSS_GET_DYNAMIC_MIN_RENDER_WIDTH: &CStr = c"DLSS.Get.Dynamic.Min.Render.Width";
pub const P_DLSS_GET_DYNAMIC_MIN_RENDER_HEIGHT: &CStr = c"DLSS.Get.Dynamic.Min.Render.Height";
pub const P_DLSS_OPTIMAL_SETTINGS_CALLBACK: &CStr = c"DLSSOptimalSettingsCallback";
pub const P_DLSSD_OPTIMAL_SETTINGS_CALLBACK: &CStr = c"DLSSDOptimalSettingsCallback";
pub const P_DLSS_GET_STATS_CALLBACK: &CStr = c"DLSSGetStatsCallback";
pub const P_DLSSD_GET_STATS_CALLBACK: &CStr = c"DLSSDGetStatsCallback";
// Evaluate-time inputs. The `NGX_VULKAN_EVALUATE_DLSSD_EXT` helper is nothing
// but ~140 `NVSDK_NGX_Parameter_Set*` calls keyed on these strings followed by
// `NVSDK_NGX_VULKAN_EvaluateFeature_C`, so naming the handful we actually feed
// avoids porting the 140-field `NVSDK_NGX_VK_DLSSD_Eval_Params` struct (and the
// silent-corruption risk that a single wrong field offset would carry).
pub const P_COLOR: &CStr = c"Color";
pub const P_OUTPUT: &CStr = c"Output";
pub const P_DEPTH: &CStr = c"Depth";
pub const P_MOTION_VECTORS: &CStr = c"MotionVectors";
pub const P_RESET: &CStr = c"Reset";
pub const P_FRAME_TIME_DELTA: &CStr = c"FrameTimeDeltaInMsec";
pub const P_JITTER_OFFSET_X: &CStr = c"Jitter.Offset.X";
pub const P_JITTER_OFFSET_Y: &CStr = c"Jitter.Offset.Y";
pub const P_MV_SCALE_X: &CStr = c"MV.Scale.X";
pub const P_MV_SCALE_Y: &CStr = c"MV.Scale.Y";
pub const P_EXPOSURE_TEXTURE: &CStr = c"ExposureTexture";
pub const P_TONEMAPPER_TYPE: &CStr = c"TonemapperType";
pub const P_GBUFFER_NORMALS: &CStr = c"GBuffer.Normals";
pub const P_GBUFFER_ROUGHNESS: &CStr = c"GBuffer.Roughness";
pub const P_GBUFFER_SPECULAR_MVEC: &CStr = c"GBuffer.SpecularMvec";
pub const P_DIFFUSE_ALBEDO: &CStr = c"DLSS.Input.DiffuseAlbedo";
pub const P_SPECULAR_ALBEDO: &CStr = c"DLSS.Input.SpecularAlbedo";
pub const P_DLSSD_SPECULAR_HIT_DISTANCE: &CStr = c"DLSSD.SpecularHitDistance";
pub const P_WORLD_TO_VIEW_MATRIX: &CStr = c"WorldToViewMatrix";
pub const P_VIEW_TO_CLIP_MATRIX: &CStr = c"ViewToClipMatrix";
pub const P_RENDER_SUBRECT_WIDTH: &CStr = c"DLSS.Render.Subrect.Dimensions.Width";
pub const P_RENDER_SUBRECT_HEIGHT: &CStr = c"DLSS.Render.Subrect.Dimensions.Height";
pub const P_PRE_EXPOSURE: &CStr = c"DLSS.Pre.Exposure";
pub const P_EXPOSURE_SCALE: &CStr = c"DLSS.Exposure.Scale";
pub const P_INPUT_COLOR_SUBRECT_X: &CStr = c"DLSS.Input.Color.Subrect.Base.X";
pub const P_INPUT_COLOR_SUBRECT_Y: &CStr = c"DLSS.Input.Color.Subrect.Base.Y";
pub const P_INPUT_DEPTH_SUBRECT_X: &CStr = c"DLSS.Input.Depth.Subrect.Base.X";
pub const P_INPUT_DEPTH_SUBRECT_Y: &CStr = c"DLSS.Input.Depth.Subrect.Base.Y";
pub const P_INPUT_MV_SUBRECT_X: &CStr = c"DLSS.Input.MV.Subrect.Base.X";
pub const P_INPUT_MV_SUBRECT_Y: &CStr = c"DLSS.Input.MV.Subrect.Base.Y";
pub const P_OUTPUT_SUBRECT_X: &CStr = c"DLSS.Output.Subrect.Base.X";
pub const P_OUTPUT_SUBRECT_Y: &CStr = c"DLSS.Output.Subrect.Base.Y";
pub const P_SIZE_IN_BYTES: &CStr = c"SizeInBytes";
pub const P_OPT_LEVEL: &CStr = c"Snippet.OptLevel";
pub const P_IS_DEV_SNIPPET_BRANCH: &CStr = c"Snippet.IsDevBranch";

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct NgxParameter {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct NgxHandle {
    _opaque: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxPathListInfo {
    pub path: *const *const WChar,
    pub length: u32,
}

pub type NgxAppLogCallback =
    Option<unsafe extern "C" fn(message: *const c_char, level: u32, feature: u32)>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxLoggingInfo {
    pub logging_callback: NgxAppLogCallback,
    pub minimum_logging_level: u32,
    pub disable_other_logging_sinks: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxFeatureCommonInfo {
    pub path_list_info: NgxPathListInfo,
    pub internal_data: *mut c_void,
    pub logging_info: NgxLoggingInfo,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxProjectIdDescription {
    pub project_id: *const c_char,
    pub engine_type: u32,
    pub engine_version: *const c_char,
}

#[repr(C)]
pub union NgxAppIdentifierV {
    pub project_desc: NgxProjectIdDescription,
    pub application_id: u64,
}

#[repr(C)]
pub struct NgxApplicationIdentifier {
    pub identifier_type: u32,
    pub v: NgxAppIdentifierV,
}

#[repr(C)]
pub struct NgxFeatureDiscoveryInfo {
    pub sdk_version: u32,
    pub feature_id: u32,
    pub identifier: NgxApplicationIdentifier,
    pub application_data_path: *const WChar,
    pub feature_info: *const NgxFeatureCommonInfo,
}

// NVSDK_NGX_Resource_VK_Type
pub const RESOURCE_VK_TYPE_IMAGEVIEW: u32 = 0;
pub const RESOURCE_VK_TYPE_BUFFER: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxImageViewInfoVk {
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub subresource_range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NgxBufferInfoVk {
    pub buffer: vk::Buffer,
    pub size_in_bytes: u32,
}

#[repr(C)]
pub union NgxResourceVkInner {
    pub image_view_info: NgxImageViewInfoVk,
    pub buffer_info: NgxBufferInfoVk,
}

/// `NVSDK_NGX_Resource_VK` (nvsdk_ngx_defs_vk.h).
#[repr(C)]
pub struct NgxResourceVk {
    pub resource: NgxResourceVkInner,
    pub ty: u32,
    pub read_write: bool,
}

/// Port of `NVSDK_NGX_Create_ImageView_Resource_VK` (nvsdk_ngx_helpers_vk.h).
///
/// `read_write` must be true exactly when the underlying `VkImage` was created
/// with `VK_IMAGE_USAGE_STORAGE_BIT` — i.e. for the DLSS output.
pub fn image_view_resource(
    image_view: vk::ImageView,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    width: u32,
    height: u32,
    read_write: bool,
) -> NgxResourceVk {
    NgxResourceVk {
        resource: NgxResourceVkInner {
            image_view_info: NgxImageViewInfoVk {
                image_view,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: aspect,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                format,
                width,
                height,
            },
        },
        ty: RESOURCE_VK_TYPE_IMAGEVIEW,
        read_write,
    }
}

/// `NVSDK_NGX_DLSSD_Create_Params` (nvsdk_ngx_params_dlssd.h).
#[repr(C)]
pub struct NgxDlssdCreateParams {
    pub denoise_mode: u32,
    pub roughness_mode: u32,
    pub use_hw_depth: u32,
    pub width: u32,
    pub height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub perf_quality_value: i32,
    pub feature_create_flags: i32,
    pub enable_output_subrects: bool,
    /// `NVSDK_NGX_RayReconstruction_Hint_Render_Preset`, applied to every mode.
    pub render_preset: u32,
}

/// `NVSDK_NGX_Feature_Create_Params` + `NVSDK_NGX_DLSS_Create_Params`.
#[repr(C)]
pub struct NgxDlssCreateParams {
    pub width: u32,
    pub height: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub perf_quality_value: i32,
    pub feature_create_flags: i32,
    pub enable_output_subrects: bool,
}

type PfnOptimalSettingsCallback = unsafe extern "C" fn(*mut NgxParameter) -> NgxResult;
type PfnGetStatsCallback = unsafe extern "C" fn(*mut NgxParameter) -> NgxResult;

// ---------------------------------------------------------------------------
// Raw entry points (libnvsdk_ngx.a — all plain C symbols, verified with `nm`)
// ---------------------------------------------------------------------------

// `build.rs` emits the `-L` search path (and `cfg(dlss_ngx)`, which gates this
// whole module). The libraries are named HERE rather than from the build script:
// `#[link]` attributes ride along in the rlib metadata, so every downstream
// binary picks them up, and `examples/dlss_spike` — which does NOT link the
// crate's rlib — keeps working off the same `-L`.
//
// libnvsdk_ngx.a is a C++ archive that `dlopen`s the driver's `_nvngx.so` and
// the per-feature snippets, hence stdc++ and dl.
#[link(name = "stdc++")]
unsafe extern "C" {}

#[link(name = "dl")]
unsafe extern "C" {}

#[link(name = "nvsdk_ngx", kind = "static")]
unsafe extern "C" {
    pub fn NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        feature_discovery_info: *const NgxFeatureDiscoveryInfo,
        out_extension_count: *mut u32,
        out_extensions: *mut *mut vk::ExtensionProperties,
    ) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        feature_discovery_info: *const NgxFeatureDiscoveryInfo,
        out_extension_count: *mut u32,
        out_extensions: *mut *mut vk::ExtensionProperties,
    ) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const c_char,
        engine_type: u32,
        engine_version: *const c_char,
        application_data_path: *const WChar,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: vk::PFN_vkGetInstanceProcAddr,
        gdpa: vk::PFN_vkGetDeviceProcAddr,
        feature_info: *const NgxFeatureCommonInfo,
        sdk_version: u32,
    ) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_GetCapabilityParameters(
        out_parameters: *mut *mut NgxParameter,
    ) -> NgxResult;
    pub fn NVSDK_NGX_VULKAN_AllocateParameters(out_parameters: *mut *mut NgxParameter)
    -> NgxResult;
    pub fn NVSDK_NGX_VULKAN_DestroyParameters(parameters: *mut NgxParameter) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_CreateFeature1(
        device: vk::Device,
        cmd: vk::CommandBuffer,
        feature_id: u32,
        parameters: *mut NgxParameter,
        out_handle: *mut *mut NgxHandle,
    ) -> NgxResult;
    pub fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut NgxHandle) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        cmd: vk::CommandBuffer,
        handle: *const NgxHandle,
        parameters: *const NgxParameter,
        callback: Option<unsafe extern "C" fn(f32, *mut bool)>,
    ) -> NgxResult;

    pub fn NVSDK_NGX_VULKAN_Shutdown1(device: vk::Device) -> NgxResult;

    pub fn NVSDK_NGX_Parameter_SetUI(p: *mut NgxParameter, name: *const c_char, value: u32);
    pub fn NVSDK_NGX_Parameter_SetI(p: *mut NgxParameter, name: *const c_char, value: i32);
    pub fn NVSDK_NGX_Parameter_SetF(p: *mut NgxParameter, name: *const c_char, value: f32);
    pub fn NVSDK_NGX_Parameter_SetULL(p: *mut NgxParameter, name: *const c_char, value: u64);
    pub fn NVSDK_NGX_Parameter_SetVoidPointer(
        p: *mut NgxParameter,
        name: *const c_char,
        value: *mut c_void,
    );
    pub fn NVSDK_NGX_Parameter_GetUI(
        p: *mut NgxParameter,
        name: *const c_char,
        out: *mut u32,
    ) -> NgxResult;
    pub fn NVSDK_NGX_Parameter_GetI(
        p: *mut NgxParameter,
        name: *const c_char,
        out: *mut i32,
    ) -> NgxResult;
    pub fn NVSDK_NGX_Parameter_GetF(
        p: *mut NgxParameter,
        name: *const c_char,
        out: *mut f32,
    ) -> NgxResult;
    pub fn NVSDK_NGX_Parameter_GetULL(
        p: *mut NgxParameter,
        name: *const c_char,
        out: *mut u64,
    ) -> NgxResult;
    pub fn NVSDK_NGX_Parameter_GetVoidPointer(
        p: *mut NgxParameter,
        name: *const c_char,
        out: *mut *mut c_void,
    ) -> NgxResult;
}

// ---------------------------------------------------------------------------
// Ergonomics
// ---------------------------------------------------------------------------

pub fn result_name(r: NgxResult) -> String {
    let s = match r {
        RESULT_SUCCESS => "Success",
        v if v == RESULT_FAIL => "Fail",
        v if v == RESULT_FAIL | 1 => "FAIL_FeatureNotSupported",
        v if v == RESULT_FAIL | 2 => "FAIL_PlatformError",
        v if v == RESULT_FAIL | 3 => "FAIL_FeatureAlreadyExists",
        v if v == RESULT_FAIL | 4 => "FAIL_FeatureNotFound",
        v if v == RESULT_FAIL | 5 => "FAIL_InvalidParameter",
        v if v == RESULT_FAIL | 6 => "FAIL_ScratchBufferTooSmall",
        v if v == RESULT_FAIL | 7 => "FAIL_NotInitialized",
        v if v == RESULT_FAIL | 8 => "FAIL_UnsupportedInputFormat",
        v if v == RESULT_FAIL | 9 => "FAIL_RWFlagMissing",
        v if v == RESULT_FAIL | 10 => "FAIL_MissingInput",
        v if v == RESULT_FAIL | 11 => "FAIL_UnableToInitializeFeature",
        v if v == RESULT_FAIL | 12 => "FAIL_OutOfDate",
        v if v == RESULT_FAIL | 13 => "FAIL_OutOfGPUMemory",
        v if v == RESULT_FAIL | 14 => "FAIL_UnsupportedFormat",
        v if v == RESULT_FAIL | 15 => "FAIL_UnableToWriteToAppDataPath",
        v if v == RESULT_FAIL | 16 => "FAIL_UnsupportedParameter",
        v if v == RESULT_FAIL | 17 => "FAIL_Denied",
        v if v == RESULT_FAIL | 18 => "FAIL_NotImplemented",
        _ => "Unknown",
    };
    format!("{s} ({r:#x})")
}

pub fn check(what: &str, r: NgxResult) -> Result<(), String> {
    if r == RESULT_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} -> {}", result_name(r)))
    }
}

pub fn wide(s: &OsStr) -> Vec<WChar> {
    s.to_string_lossy()
        .chars()
        .map(|c| c as WChar)
        .chain([0])
        .collect()
}

/// Owns every allocation a `NVSDK_NGX_FeatureDiscoveryInfo` points at.
///
/// The pointers are taken from `CString`/`Vec`/`Box` heap allocations, so they
/// stay valid across the move into this struct.
pub struct FeatureInfo {
    _project_id: CString,
    _engine_version: CString,
    _data_path: Vec<WChar>,
    _search_paths: Vec<Vec<WChar>>,
    _search_path_ptrs: Vec<*const WChar>,
    common: Box<NgxFeatureCommonInfo>,
    discovery: NgxFeatureDiscoveryInfo,
}

impl FeatureInfo {
    pub fn new(
        project_id: &str,
        engine_version: &str,
        feature_id: u32,
        data_path: &OsStr,
        snippet_search_paths: &[String],
        log_level: u32,
        log_callback: NgxAppLogCallback,
    ) -> Self {
        let project_id = CString::new(project_id).unwrap();
        let engine_version = CString::new(engine_version).unwrap();
        let data_path = wide(data_path);
        let search_paths: Vec<Vec<WChar>> = snippet_search_paths
            .iter()
            .map(|p| wide(OsStr::new(p)))
            .collect();
        let search_path_ptrs: Vec<*const WChar> = search_paths.iter().map(|p| p.as_ptr()).collect();

        let common = Box::new(NgxFeatureCommonInfo {
            path_list_info: NgxPathListInfo {
                path: search_path_ptrs.as_ptr(),
                length: search_path_ptrs.len() as u32,
            },
            internal_data: std::ptr::null_mut(),
            logging_info: NgxLoggingInfo {
                logging_callback: log_callback,
                minimum_logging_level: log_level,
                // NGX's own Linux file sink aborts (`free(): invalid pointer`)
                // during init once an app callback is installed, so disable it
                // whenever we install one. (Same workaround as dlss_wgpu.)
                disable_other_logging_sinks: log_callback.is_some(),
            },
        });

        let discovery = NgxFeatureDiscoveryInfo {
            sdk_version: NVSDK_NGX_VERSION_API,
            feature_id,
            identifier: NgxApplicationIdentifier {
                identifier_type: APP_IDENTIFIER_TYPE_PROJECT_ID,
                v: NgxAppIdentifierV {
                    project_desc: NgxProjectIdDescription {
                        project_id: project_id.as_ptr(),
                        engine_type: ENGINE_TYPE_CUSTOM,
                        engine_version: engine_version.as_ptr(),
                    },
                },
            },
            application_data_path: data_path.as_ptr(),
            feature_info: &*common as *const NgxFeatureCommonInfo,
        };

        Self {
            _project_id: project_id,
            _engine_version: engine_version,
            _data_path: data_path,
            _search_paths: search_paths,
            _search_path_ptrs: search_path_ptrs,
            common,
            discovery,
        }
    }

    pub fn discovery(&self) -> *const NgxFeatureDiscoveryInfo {
        &self.discovery
    }

    pub fn common(&self) -> *const NgxFeatureCommonInfo {
        &*self.common
    }

    pub fn data_path(&self) -> *const WChar {
        self._data_path.as_ptr()
    }

    pub fn project_id(&self) -> *const c_char {
        self._project_id.as_ptr()
    }

    pub fn engine_version(&self) -> *const c_char {
        self._engine_version.as_ptr()
    }
}

// ---------------------------------------------------------------------------
// Rust ports of the header-only `static inline` helpers
// ---------------------------------------------------------------------------

pub struct OptimalSettings {
    pub optimal: [u32; 2],
    pub min: [u32; 2],
    pub max: [u32; 2],
    pub deprecated_sharpness: f32,
}

/// Port of `NGX_DLSSD_GET_OPTIMAL_SETTINGS` (nvsdk_ngx_helpers_dlssd.h) when
/// `ray_reconstruction` is true, otherwise `NGX_DLSS_GET_OPTIMAL_SETTINGS`
/// (nvsdk_ngx_helpers.h). Both fetch a callback function pointer out of the
/// *capability* parameter block and invoke it — `AllocateParameters` blocks
/// return a block with no callback, which is why the SDK insists on
/// `GetCapabilityParameters` here.
///
/// # Safety
/// `params` must be a live capability parameter block.
pub unsafe fn get_optimal_settings(
    params: *mut NgxParameter,
    ray_reconstruction: bool,
    target: [u32; 2],
    perf_quality: i32,
) -> Result<OptimalSettings, String> {
    let cb_name = if ray_reconstruction {
        P_DLSSD_OPTIMAL_SETTINGS_CALLBACK
    } else {
        P_DLSS_OPTIMAL_SETTINGS_CALLBACK
    };

    let mut callback: *mut c_void = std::ptr::null_mut();
    unsafe { NVSDK_NGX_Parameter_GetVoidPointer(params, cb_name.as_ptr(), &mut callback) };
    if callback.is_null() {
        return Err(format!(
            "{} is null — installed DLSS snippet is out of date, or the parameter block did not \
             come from NVSDK_NGX_VULKAN_GetCapabilityParameters",
            cb_name.to_string_lossy()
        ));
    }

    unsafe {
        NVSDK_NGX_Parameter_SetUI(params, P_WIDTH.as_ptr(), target[0]);
        NVSDK_NGX_Parameter_SetUI(params, P_HEIGHT.as_ptr(), target[1]);
        NVSDK_NGX_Parameter_SetI(params, P_PERF_QUALITY_VALUE.as_ptr(), perf_quality);
        // Older DLSS snippets still read this.
        NVSDK_NGX_Parameter_SetI(params, P_RTX_VALUE.as_ptr(), 0);

        let f: PfnOptimalSettingsCallback = std::mem::transmute(callback);
        check("DLSS(D)OptimalSettingsCallback", f(params))?;

        let mut out = OptimalSettings {
            optimal: [0; 2],
            min: [0; 2],
            max: [0; 2],
            deprecated_sharpness: 0.0,
        };
        NVSDK_NGX_Parameter_GetUI(params, P_OUT_WIDTH.as_ptr(), &mut out.optimal[0]);
        NVSDK_NGX_Parameter_GetUI(params, P_OUT_HEIGHT.as_ptr(), &mut out.optimal[1]);
        out.min = out.optimal;
        out.max = out.optimal;
        NVSDK_NGX_Parameter_GetUI(
            params,
            P_DLSS_GET_DYNAMIC_MAX_RENDER_WIDTH.as_ptr(),
            &mut out.max[0],
        );
        NVSDK_NGX_Parameter_GetUI(
            params,
            P_DLSS_GET_DYNAMIC_MAX_RENDER_HEIGHT.as_ptr(),
            &mut out.max[1],
        );
        NVSDK_NGX_Parameter_GetUI(
            params,
            P_DLSS_GET_DYNAMIC_MIN_RENDER_WIDTH.as_ptr(),
            &mut out.min[0],
        );
        NVSDK_NGX_Parameter_GetUI(
            params,
            P_DLSS_GET_DYNAMIC_MIN_RENDER_HEIGHT.as_ptr(),
            &mut out.min[1],
        );
        NVSDK_NGX_Parameter_GetF(params, P_SHARPNESS.as_ptr(), &mut out.deprecated_sharpness);
        Ok(out)
    }
}

/// Port of `NGX_VULKAN_CREATE_DLSSD_EXT1` (nvsdk_ngx_helpers_dlssd_vk.h).
///
/// # Safety
/// `params` must be live, `cmd` must be a command buffer in the recording state
/// on `device`.
pub unsafe fn create_dlssd_feature(
    device: vk::Device,
    cmd: vk::CommandBuffer,
    params: *mut NgxParameter,
    create: &NgxDlssdCreateParams,
) -> Result<*mut NgxHandle, String> {
    unsafe {
        NVSDK_NGX_Parameter_SetUI(params, P_CREATION_NODE_MASK.as_ptr(), 1);
        NVSDK_NGX_Parameter_SetUI(params, P_VISIBILITY_NODE_MASK.as_ptr(), 1);
        NVSDK_NGX_Parameter_SetUI(params, P_WIDTH.as_ptr(), create.width);
        NVSDK_NGX_Parameter_SetUI(params, P_HEIGHT.as_ptr(), create.height);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_WIDTH.as_ptr(), create.target_width);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_HEIGHT.as_ptr(), create.target_height);
        NVSDK_NGX_Parameter_SetI(
            params,
            P_PERF_QUALITY_VALUE.as_ptr(),
            create.perf_quality_value,
        );
        NVSDK_NGX_Parameter_SetI(
            params,
            P_DLSS_FEATURE_CREATE_FLAGS.as_ptr(),
            create.feature_create_flags,
        );
        NVSDK_NGX_Parameter_SetI(
            params,
            P_DLSS_ENABLE_OUTPUT_SUBRECTS.as_ptr(),
            create.enable_output_subrects as i32,
        );
        NVSDK_NGX_Parameter_SetI(
            params,
            P_DLSS_DENOISE_MODE.as_ptr(),
            create.denoise_mode as i32,
        );
        NVSDK_NGX_Parameter_SetUI(
            params,
            P_DLSS_ROUGHNESS_MODE.as_ptr(),
            create.roughness_mode,
        );
        NVSDK_NGX_Parameter_SetUI(params, P_DLSS_USE_HW_DEPTH.as_ptr(), create.use_hw_depth);
        for name in [
            P_RR_PRESET_DLAA,
            P_RR_PRESET_QUALITY,
            P_RR_PRESET_BALANCED,
            P_RR_PRESET_PERFORMANCE,
            P_RR_PRESET_ULTRA_PERFORMANCE,
            P_RR_PRESET_ULTRA_QUALITY,
        ] {
            NVSDK_NGX_Parameter_SetUI(params, name.as_ptr(), create.render_preset);
        }

        let mut handle: *mut NgxHandle = std::ptr::null_mut();
        check(
            "NVSDK_NGX_VULKAN_CreateFeature1(RayReconstruction)",
            NVSDK_NGX_VULKAN_CreateFeature1(
                device,
                cmd,
                FEATURE_RAY_RECONSTRUCTION,
                params,
                &mut handle,
            ),
        )?;
        Ok(handle)
    }
}

/// Port of `NGX_VULKAN_CREATE_DLSS_EXT1` (nvsdk_ngx_helpers_vk.h) — plain
/// super-resolution, the fallback if Ray Reconstruction is unavailable.
///
/// # Safety
/// Same contract as [`create_dlssd_feature`].
pub unsafe fn create_dlss_feature(
    device: vk::Device,
    cmd: vk::CommandBuffer,
    params: *mut NgxParameter,
    create: &NgxDlssCreateParams,
) -> Result<*mut NgxHandle, String> {
    unsafe {
        NVSDK_NGX_Parameter_SetUI(params, P_CREATION_NODE_MASK.as_ptr(), 1);
        NVSDK_NGX_Parameter_SetUI(params, P_VISIBILITY_NODE_MASK.as_ptr(), 1);
        NVSDK_NGX_Parameter_SetUI(params, P_WIDTH.as_ptr(), create.width);
        NVSDK_NGX_Parameter_SetUI(params, P_HEIGHT.as_ptr(), create.height);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_WIDTH.as_ptr(), create.target_width);
        NVSDK_NGX_Parameter_SetUI(params, P_OUT_HEIGHT.as_ptr(), create.target_height);
        NVSDK_NGX_Parameter_SetI(
            params,
            P_PERF_QUALITY_VALUE.as_ptr(),
            create.perf_quality_value,
        );
        NVSDK_NGX_Parameter_SetI(
            params,
            P_DLSS_FEATURE_CREATE_FLAGS.as_ptr(),
            create.feature_create_flags,
        );
        NVSDK_NGX_Parameter_SetI(
            params,
            P_DLSS_ENABLE_OUTPUT_SUBRECTS.as_ptr(),
            create.enable_output_subrects as i32,
        );

        let mut handle: *mut NgxHandle = std::ptr::null_mut();
        check(
            "NVSDK_NGX_VULKAN_CreateFeature1(SuperSampling)",
            NVSDK_NGX_VULKAN_CreateFeature1(
                device,
                cmd,
                FEATURE_SUPERSAMPLING,
                params,
                &mut handle,
            ),
        )?;
        Ok(handle)
    }
}

/// The resource set a DLSS Ray Reconstruction evaluate consumes.
///
/// Every field here is a *required* input for `NVSDK_NGX_DLSS_Denoise_Mode_DLUnified`
/// with `Roughness_Mode_Packed` + `Depth_Type_Linear`. Optional inputs the
/// integration guide lists (bias/responsivity mask, transparency layer, SSS
/// guide, alpha, disocclusion mask, the before/after-effect colour pairs) are
/// simply parameters we never set.
pub struct RrEvalResources<'a> {
    pub color: &'a mut NgxResourceVk,
    pub output: &'a mut NgxResourceVk,
    pub depth: &'a mut NgxResourceVk,
    pub motion_vectors: &'a mut NgxResourceVk,
    pub diffuse_albedo: &'a mut NgxResourceVk,
    pub specular_albedo: &'a mut NgxResourceVk,
    /// Roughness lives in `.w` (Roughness_Mode_Packed).
    pub normals_roughness: &'a mut NgxResourceVk,
    /// World-space distance from primary vertex to the specular hit point.
    pub specular_hit_distance: &'a mut NgxResourceVk,
    /// 1x1 R32_SFLOAT. Optional to NGX, but leaving it out hands exposure to
    /// the driver's auto-exposure, which visibly re-normalises over seconds.
    pub exposure: Option<&'a mut NgxResourceVk>,
}

/// Row-major 4x4 camera matrices, required whenever `specular_hit_distance`
/// is used as the specular guide.
pub struct RrEvalCamera {
    pub world_to_view: [f32; 16],
    pub view_to_clip: [f32; 16],
}

/// Port of the subset of `NGX_VULKAN_EVALUATE_DLSSD_EXT` that a
/// depth/motion/albedo/normal/hit-distance renderer actually needs, followed by
/// `NVSDK_NGX_VULKAN_EvaluateFeature_C`.
///
/// # Safety
/// `cmd` must be recording on the device NGX was initialised with, `handle`
/// must be a live Ray Reconstruction feature, and every image behind
/// `resources` must outlive the queue submission (NGX reads the
/// `NVSDK_NGX_Resource_VK`s during this call, but the GPU reads the images
/// during the submit).
#[allow(clippy::too_many_arguments)]
pub unsafe fn evaluate_dlssd(
    cmd: vk::CommandBuffer,
    handle: *mut NgxHandle,
    params: *mut NgxParameter,
    resources: RrEvalResources<'_>,
    camera: &mut RrEvalCamera,
    render_resolution: [u32; 2],
    jitter: [f32; 2],
    reset: bool,
    // `MV.Scale.{X,Y}`: maps the motion-vector texels onto PIXELS pointing at
    // the history. 1.0 when they already are.
    mv_scale: [f32; 2],
    // Wall-clock milliseconds since the previous evaluate: RR scales its temporal
    // accumulation by it ("helps in determining the amount to denoise").
    frame_time_ms: f32,
) -> Result<(), String> {
    unsafe {
        let set_res = |name: &CStr, res: *mut NgxResourceVk| {
            NVSDK_NGX_Parameter_SetVoidPointer(params, name.as_ptr(), res.cast())
        };

        set_res(P_COLOR, resources.color);
        set_res(P_OUTPUT, resources.output);
        set_res(P_DEPTH, resources.depth);
        set_res(P_MOTION_VECTORS, resources.motion_vectors);
        set_res(P_DIFFUSE_ALBEDO, resources.diffuse_albedo);
        set_res(P_SPECULAR_ALBEDO, resources.specular_albedo);
        set_res(P_GBUFFER_NORMALS, resources.normals_roughness);
        set_res(
            P_DLSSD_SPECULAR_HIT_DISTANCE,
            resources.specular_hit_distance,
        );
        // Packed roughness: explicitly clear the standalone slot, because the
        // parameter block is shared with feature creation and is sticky.
        set_res(P_GBUFFER_ROUGHNESS, std::ptr::null_mut());
        set_res(P_GBUFFER_SPECULAR_MVEC, std::ptr::null_mut());
        match resources.exposure {
            Some(e) => set_res(P_EXPOSURE_TEXTURE, e),
            None => set_res(P_EXPOSURE_TEXTURE, std::ptr::null_mut()),
        }

        // Hit-distance guiding needs the camera basis to reproject the hit
        // point; without these NGX logs "Matrix inverse failed" per evaluate.
        NVSDK_NGX_Parameter_SetVoidPointer(
            params,
            P_WORLD_TO_VIEW_MATRIX.as_ptr(),
            camera.world_to_view.as_mut_ptr().cast(),
        );
        NVSDK_NGX_Parameter_SetVoidPointer(
            params,
            P_VIEW_TO_CLIP_MATRIX.as_ptr(),
            camera.view_to_clip.as_mut_ptr().cast(),
        );

        NVSDK_NGX_Parameter_SetF(params, P_JITTER_OFFSET_X.as_ptr(), jitter[0]);
        NVSDK_NGX_Parameter_SetF(params, P_JITTER_OFFSET_Y.as_ptr(), jitter[1]);
        NVSDK_NGX_Parameter_SetI(params, P_RESET.as_ptr(), reset as i32);
        NVSDK_NGX_Parameter_SetF(params, P_FRAME_TIME_DELTA.as_ptr(), frame_time_ms);
        NVSDK_NGX_Parameter_SetF(params, P_MV_SCALE_X.as_ptr(), mv_scale[0]);
        NVSDK_NGX_Parameter_SetF(params, P_MV_SCALE_Y.as_ptr(), mv_scale[1]);
        NVSDK_NGX_Parameter_SetUI(params, P_TONEMAPPER_TYPE.as_ptr(), 0);
        NVSDK_NGX_Parameter_SetF(params, P_PRE_EXPOSURE.as_ptr(), 1.0);
        NVSDK_NGX_Parameter_SetF(params, P_EXPOSURE_SCALE.as_ptr(), 1.0);
        NVSDK_NGX_Parameter_SetUI(
            params,
            P_RENDER_SUBRECT_WIDTH.as_ptr(),
            render_resolution[0],
        );
        NVSDK_NGX_Parameter_SetUI(
            params,
            P_RENDER_SUBRECT_HEIGHT.as_ptr(),
            render_resolution[1],
        );
        for name in [
            P_INPUT_COLOR_SUBRECT_X,
            P_INPUT_COLOR_SUBRECT_Y,
            P_INPUT_DEPTH_SUBRECT_X,
            P_INPUT_DEPTH_SUBRECT_Y,
            P_INPUT_MV_SUBRECT_X,
            P_INPUT_MV_SUBRECT_Y,
            P_OUTPUT_SUBRECT_X,
            P_OUTPUT_SUBRECT_Y,
        ] {
            NVSDK_NGX_Parameter_SetUI(params, name.as_ptr(), 0);
        }

        check(
            "NVSDK_NGX_VULKAN_EvaluateFeature_C",
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, handle, params, None),
        )
    }
}

/// Port of `NGX_DLSSD_GET_STATS` / `NGX_DLSS_GET_STATS` — VRAM the feature took.
///
/// # Safety
/// `params` must be a live capability parameter block.
pub unsafe fn get_stats(
    params: *mut NgxParameter,
    ray_reconstruction: bool,
) -> Result<(u64, u32, u32), String> {
    let cb_name = if ray_reconstruction {
        P_DLSSD_GET_STATS_CALLBACK
    } else {
        P_DLSS_GET_STATS_CALLBACK
    };
    let mut callback: *mut c_void = std::ptr::null_mut();
    unsafe { NVSDK_NGX_Parameter_GetVoidPointer(params, cb_name.as_ptr(), &mut callback) };
    if callback.is_null() {
        return Err(format!("{} is null", cb_name.to_string_lossy()));
    }
    unsafe {
        let f: PfnGetStatsCallback = std::mem::transmute(callback);
        check("DLSS(D)GetStatsCallback", f(params))?;
        let (mut bytes, mut opt_level, mut dev_branch) = (0u64, 0u32, 0u32);
        NVSDK_NGX_Parameter_GetULL(params, P_SIZE_IN_BYTES.as_ptr(), &mut bytes);
        NVSDK_NGX_Parameter_GetUI(params, P_OPT_LEVEL.as_ptr(), &mut opt_level);
        NVSDK_NGX_Parameter_GetUI(params, P_IS_DEV_SNIPPET_BRANCH.as_ptr(), &mut dev_branch);
        Ok((bytes, opt_level, dev_branch))
    }
}
