//! NVIDIA Nsight Aftermath GPU crash dumps, armed by the engine itself.
//!
//! Aurora owns the `vkCreateInstance`/`vkCreateDevice` call site, so it owns Aftermath
//! registration too: [`enable`] is the first line of `RenderDevice::from_display`, before the
//! Vulkan loader is touched (the SDK requires registration before device creation). Apps do
//! nothing beyond turning the feature on.
//!
//! Three knobs:
//! - Cargo feature `dev`: compiles the FFI in. Off (the default) or without the SDK at build
//!   time, this module is no-op stubs.
//! - `$AFTERMATH_SDK` at BUILD time: the SDK tree holding `lib/x64/libGFSDK_Aftermath_Lib.x64.so`
//!   (default `~/nvidia/aftermath`). A missing lib makes `build.rs` skip the link with a
//!   warning; the feature being on never breaks a build. The `-l`/`-L` ride the rlib to
//!   downstream binaries, the rpath does not: a binary in another workspace needs its own
//!   `-Wl,-rpath` to the SDK's `lib/x64` (aurora_files' `bsn` and `bevy_city` build.rs do).
//! - `$AURORA_AFTERMATH_DIR` at RUN time: where dumps land. Default `./crash-dumps`, created
//!   lazily at the first dump.
//!
//! On device loss the driver collects a `.nv-gpudump` asynchronously; a panic racing that
//! collection would abort the process before the file lands. [`wait_for_dump`] is the stall:
//! poll `GFSDK_Aftermath_GetCrashDumpStatus` until `Finished` (10 s cap). The sites that
//! surface `VK_ERROR_DEVICE_LOST` call [`note_device_lost`] before panicking, and a panic
//! hook (installed by [`enable`]) covers every other death path, e.g. a teardown `Drop`
//! panic after the loss. Decode a dump with Nsight Graphics (GPU Crash Dump Inspector).

/// If `err` is `VK_ERROR_DEVICE_LOST`, block until the driver's crash-dump
/// collection finishes (or 10 s passes). Call before panicking on any Vulkan
/// error — every other error returns immediately. No-op when Aftermath is
/// compiled out or failed to arm.
pub fn note_device_lost(err: ash::vk::Result) {
    if err == ash::vk::Result::ERROR_DEVICE_LOST {
        wait_for_dump();
    }
}

#[cfg(aftermath_gfsdk)]
mod imp {
    use std::{
        ffi::c_void,
        fs,
        path::PathBuf,
        sync::{
            Once,
            atomic::{AtomicBool, AtomicU32, Ordering},
        },
        time::{Duration, Instant, SystemTime, UNIX_EPOCH},
    };

    // GFSDK_Aftermath_Defines.h / GFSDK_Aftermath_GpuCrashDump.h (SDK 2025.5).
    const VERSION_API: u32 = 0x0000_021A; // GFSDK_Aftermath_Version_API 2.26
    const WATCHED_VULKAN: u32 = 0x2; // GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan
    const FEATURE_DEFAULT: u32 = 0x0; // GFSDK_Aftermath_GpuCrashDumpFeatureFlags_Default
    const RESULT_SUCCESS: i32 = 0x1; // GFSDK_Aftermath_Result_Success

    // GFSDK_Aftermath_CrashDump_Status
    const STATUS_NOT_STARTED: i32 = 0;
    const STATUS_COLLECTING_FAILED: i32 = 2;
    const STATUS_FINISHED: i32 = 4;

    /// Shared shape of the crash-dump and shader-debug-info callbacks:
    /// `(const void* blob, uint32_t size, void* user)`.
    type BlobCb = extern "C" fn(*const c_void, u32, *mut c_void);

    unsafe extern "C" {
        fn GFSDK_Aftermath_EnableGpuCrashDumps(
            api_version: u32,
            watched_apis: u32,
            flags: u32,
            gpu_crash_dump_cb: Option<BlobCb>,
            shader_debug_info_cb: Option<BlobCb>,
            description_cb: *const c_void,
            resolve_marker_cb: *const c_void,
            user_data: *mut c_void,
        ) -> i32;
        fn GFSDK_Aftermath_GetCrashDumpStatus(out_status: *mut i32) -> i32;
    }

    static ARMED: AtomicBool = AtomicBool::new(false);
    static DUMP_DONE: AtomicBool = AtomicBool::new(false);
    static DUMP_COUNT: AtomicU32 = AtomicU32::new(0);

    fn dump_dir() -> PathBuf {
        std::env::var_os("AURORA_AFTERMATH_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_default()
                    .join("crash-dumps")
            })
    }

    fn write_blob(blob: *const c_void, size: u32, name: String) -> Option<PathBuf> {
        // Safety: the driver hands the full blob synchronously; it is valid
        // for the duration of the callback.
        let bytes = unsafe { std::slice::from_raw_parts(blob as *const u8, size as usize) };
        let dir = dump_dir();
        let _ = fs::create_dir_all(&dir); // lazily, at the first dump
        let path = dir.join(name);
        match fs::write(&path, bytes) {
            Ok(()) => Some(path),
            Err(e) => {
                // eprintln, not log: this runs on a driver thread mid-crash.
                eprintln!("aftermath: failed to write {}: {e}", path.display());
                None
            }
        }
    }

    fn stamp() -> (u64, u32) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        (ts, DUMP_COUNT.fetch_add(1, Ordering::Relaxed))
    }

    // Free-threaded (driver thread), delivered synchronously on the crash.
    extern "C" fn on_crash_dump(dump: *const c_void, size: u32, _user: *mut c_void) {
        let (ts, n) = stamp();
        let name = format!("aurora-{}-{ts}-{n}.nv-gpudump", std::process::id());
        if let Some(path) = write_blob(dump, size, name) {
            eprintln!(
                "aftermath: GPU crash dump written: {} ({size} bytes)",
                path.display()
            );
        }
        if size > 0 {
            DUMP_DONE.store(true, Ordering::Release);
        }
    }

    // Shader debug info (source-level mapping for the dump's shader frames).
    // Arrives per-shader at compile time and again on a crash.
    extern "C" fn on_shader_debug_info(blob: *const c_void, size: u32, _user: *mut c_void) {
        let (ts, n) = stamp();
        let name = format!("aurora-{}-{ts}-{n}.nvdbg", std::process::id());
        write_blob(blob, size, name);
    }

    /// Register the crash-dump callbacks with the driver. Idempotent; must run
    /// before instance/device creation (it does -- first line of
    /// `RenderDevice::from_display`). Failure warns and continues: device bring-up
    /// never breaks over diagnostics.
    pub fn enable() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let r = unsafe {
                GFSDK_Aftermath_EnableGpuCrashDumps(
                    VERSION_API,
                    WATCHED_VULKAN,
                    FEATURE_DEFAULT,
                    Some(on_crash_dump),
                    Some(on_shader_debug_info),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            };
            if r != RESULT_SUCCESS {
                log::warn!(
                    "aftermath: GFSDK_Aftermath_EnableGpuCrashDumps failed (0x{:X}) — \
                     GPU crash dumps disabled",
                    r as u32,
                );
                return;
            }
            ARMED.store(true, Ordering::Release);
            log::info!("aftermath: armed (dumps -> {})", dump_dir().display());
            // Safety net for death paths that never see a VkResult — e.g. a
            // device-lost teardown panic in a Drop impl aborts the process
            // before the driver finishes collecting. Stall the first panic
            // until the pending dump lands (fast exit when nothing crashed).
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                super::wait_for_dump();
                previous(info);
            }));
        });
    }

    /// Block until the driver finishes collecting a pending GPU crash dump:
    /// poll `GFSDK_Aftermath_GetCrashDumpStatus` until `Finished`/failure,
    /// capped at 10 s. Returns immediately when Aftermath isn't armed, the
    /// dump already landed, or no crash collection ever starts (~1 s grace —
    /// a plain CPU panic must not stall 10 s).
    pub fn wait_for_dump() {
        if !ARMED.load(Ordering::Acquire) || DUMP_DONE.load(Ordering::Acquire) {
            return;
        }
        let start = Instant::now();
        let mut saw_activity = false;
        let mut logged = false;
        while start.elapsed() < Duration::from_secs(10) {
            if DUMP_DONE.load(Ordering::Acquire) {
                eprintln!("aftermath: crash dump complete");
                return;
            }
            let mut status = STATUS_NOT_STARTED;
            if unsafe { GFSDK_Aftermath_GetCrashDumpStatus(&mut status) } != RESULT_SUCCESS {
                return;
            }
            match status {
                STATUS_FINISHED => return,
                STATUS_COLLECTING_FAILED => {
                    eprintln!("aftermath: driver crash-dump collection failed");
                    return;
                }
                STATUS_NOT_STARTED => {
                    if !saw_activity && start.elapsed() > Duration::from_secs(1) {
                        return; // nothing crashed; don't hold a normal panic hostage
                    }
                }
                _ => {
                    // CollectingData / InvokingCallback — the driver is working.
                    saw_activity = true;
                    if !logged {
                        logged = true;
                        eprintln!("aftermath: waiting up to 10s for the GPU crash dump…");
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        eprintln!("aftermath: timed out waiting for the crash dump");
    }
}

#[cfg(not(aftermath_gfsdk))]
mod imp {
    /// Compiled out (no `dev` feature, or no GFSDK lib at build time).
    pub fn enable() {}
    /// Compiled out — nothing to wait for.
    pub fn wait_for_dump() {}
}

pub use imp::{enable, wait_for_dump};
