//! Nsight Graphics GPU Trace, started from the app (the Nsight Graphics SDK's
//! `NGFX_GPUTrace_*_Vulkan`, its header-only loader ported): run under
//! `ngfx --activity "GPU Trace Profiler" --start-with-ngfx-sdk --stop-with-ngfx-sdk --auto-export`
//! and [`request_gpu_trace`] traces the next frame, so a scripted capture lands on the frame the
//! caller chose (a bench's held view after warm-up) instead of a hotkey or a delay. Outside
//! Nsight the trace library is not loaded and every call is a no-op.

use std::ffi::{CStr, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

static REQUESTED: AtomicBool = AtomicBool::new(false);

/// Traces the next frame the renderer submits, when running under Nsight's GPU Trace.
pub fn request_gpu_trace() {
    REQUESTED.store(true, Ordering::Relaxed);
}

/// Whether the GPU Trace library is injected (the process was launched by Nsight's GPU Trace).
pub fn gpu_trace_available() -> bool {
    api().is_some()
}

#[repr(C)]
struct Version {
    version: u32,
}

#[repr(C)]
struct ActivateParams {
    version: u32,
    queue: vk::Queue,
}

#[repr(C)]
struct StopParams {
    version: u32,
    flags: u32,
    queue: vk::Queue,
    output_resources: *const c_void,
    num_output_resources: i32,
}

/// `NGFX_MAKE_STRUCT_VERSION(T, 1)`.
const fn struct_version<T>() -> u32 {
    std::mem::size_of::<T>() as u32 | (1 << 16)
}

/// `NGFX_GPUTrace_StopTraceFlag_NextFrameBoundary`.
const STOP_NEXT_FRAME_BOUNDARY: u32 = 1;

type Fn<P> = unsafe extern "C" fn(*mut P) -> i32;

struct Api {
    activate: Fn<ActivateParams>,
    start: Fn<Version>,
    stop: Fn<StopParams>,
    activated: AtomicBool,
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}

#[cfg(unix)]
fn load() -> Option<Api> {
    let open = |name: &CStr| unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD) };
    // The injection library marks the activity as injected; the target library holds the API.
    if open(c"libWarpViz.Injection.so").is_null() {
        return None;
    }
    let lib = open(c"libWarpVizTarget.so");
    if lib.is_null() {
        log::warn!("nsight: GPU Trace injected but libWarpVizTarget.so is not loaded");
        return None;
    }
    let sym = |name: &CStr| {
        let p = unsafe { libc::dlsym(lib, name.as_ptr()) };
        (!p.is_null()).then_some(p)
    };
    let init: Fn<Version> =
        unsafe { std::mem::transmute(sym(c"NGFX_GPUTrace_InitializeTraceActivityVulkan")?) };
    let api = Api {
        activate: unsafe { std::mem::transmute(sym(c"NGFX_GPUTrace_ActivateTraceVulkan")?) },
        start: unsafe { std::mem::transmute(sym(c"NGFX_GPUTrace_StartTraceVulkan")?) },
        stop: unsafe { std::mem::transmute(sym(c"NGFX_GPUTrace_StopTraceVulkan")?) },
        activated: AtomicBool::new(false),
    };
    let mut params = Version {
        version: struct_version::<Version>(),
    };
    let result = unsafe { init(&mut params) };
    if result != 0 {
        log::warn!("nsight: GPU Trace initialisation failed ({result})");
        return None;
    }
    log::info!("nsight: GPU Trace SDK ready (request_gpu_trace traces the next frame)");
    Some(api)
}

#[cfg(not(unix))]
fn load() -> Option<Api> {
    None
}

/// Around the frame's submit on `queue` (held by the caller): when a trace was requested, starts
/// it before the submit; [`end_frame`] then stops it at the frame's present.
pub fn begin_frame(queue: vk::Queue) -> bool {
    if !REQUESTED.load(Ordering::Relaxed) {
        return false;
    }
    let Some(api) = api() else {
        REQUESTED.store(false, Ordering::Relaxed);
        return false;
    };
    REQUESTED.store(false, Ordering::Relaxed);
    if !api.activated.load(Ordering::Relaxed) {
        let mut params = ActivateParams {
            version: struct_version::<ActivateParams>(),
            queue,
        };
        let result = unsafe { (api.activate)(&mut params) };
        if result != 0 {
            log::warn!("nsight: GPU Trace activation failed ({result})");
            return false;
        }
        api.activated.store(true, Ordering::Relaxed);
    }
    let mut params = Version {
        version: struct_version::<Version>(),
    };
    let result = unsafe { (api.start)(&mut params) };
    if result != 0 {
        log::warn!("nsight: GPU Trace start failed ({result})");
        return false;
    }
    log::info!("nsight: GPU Trace started");
    true
}

/// After the submit [`begin_frame`] returned true for: the trace ends at the next frame boundary.
pub fn end_frame(queue: vk::Queue) {
    let Some(api) = api() else {
        return;
    };
    let mut params = StopParams {
        version: struct_version::<StopParams>(),
        flags: STOP_NEXT_FRAME_BOUNDARY,
        queue,
        output_resources: std::ptr::null(),
        num_output_resources: 0,
    };
    let result = unsafe { (api.stop)(&mut params) };
    if result != 0 {
        log::warn!("nsight: GPU Trace stop failed ({result})");
    }
}
