//! DLSS Ray Reconstruction — NGX on aurora's Vulkan device.
//!
//! The path tracer writes, at render resolution, the guides DLSS-RR consumes (noisy HDR
//! colour, linear depth, motion vectors, diffuse/specular albedo, normal + roughness,
//! specular hit distance); NGX denoises and upscales them into the output-resolution image
//! the post-process blit reads. Ray Reconstruction is the renderer's ONLY resolve -- there
//! is no unfiltered fallback path, so the engine requires a working NGX (the SDK at build
//! time via `$DLSS_SDK` -> `cfg(dlss_ngx)` from `build.rs`, and an RR-capable GPU/driver at
//! run time; [`DlssPlugin`] panics otherwise). The camera's [`AuroraDlss`] picks the mode.
//!
//! The NGX FFI ([`ngx`]) and the call sequence are the ones proved in the old engine
//! (`bevy_aurora_old/docs/dlss_notes.md`).

use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use bevy::prelude::*;

use crate::{
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_device::RenderDevice,
};

#[cfg(dlss_ngx)]
pub mod ngx;
#[cfg(dlss_ngx)]
mod renderer;
#[cfg(dlss_ngx)]
pub use renderer::DlssRenderer;
#[cfg(not(dlss_ngx))]
mod stub;
#[cfg(not(dlss_ngx))]
pub use stub::DlssRenderer;

/// Fixed UUID identifying aurora to NGX (OTA snippet updates and per-app tuning key off it):
/// never change it, and it must parse as a real UUID.
pub const AURORA_PROJECT_ID: &str = "a17b0d3e-5c42-4f9a-9d31-6b0e2f8c74d5";

/// Set by [`DlssPlugin`] before NGX initialises; [`feature_info`] then points the snippet
/// search path at the SDK's `dev` libraries instead of `rel`.
static DEV_SNIPPET: AtomicBool = AtomicBool::new(false);

/// `NVSDK_NGX_RayReconstruction_Hint_Render_Preset` value; set by [`DlssPlugin`], read at
/// feature creation. Same pattern as [`DEV_SNIPPET`].
static RR_PRESET: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Which Ray Reconstruction model the feature is created with (RR guide 3.13). Changing it
/// at runtime (the dev panel has a row for it) rebuilds the feature.
#[derive(Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Default, Clone, PartialEq)]
pub enum RrPreset {
    /// NVIDIA's pick, may move with OTA updates (currently resolves to D).
    #[default]
    Default,
    /// The default transformer model (`diamond_wallaby`).
    D,
    /// The latest transformer model (`truthful_shrimp`); required for the DoF guide.
    E,
}

impl RrPreset {
    fn to_ngx(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::D => 4,
            Self::E => 5,
        }
    }

    /// The active preset ([`DlssPlugin::preset`], or a later runtime change).
    pub fn current() -> Self {
        match RR_PRESET.load(Ordering::Relaxed) {
            4 => Self::D,
            5 => Self::E,
            _ => Self::Default,
        }
    }

    /// Make `self` the active preset; the renderer rebuilds the feature when it differs
    /// from the one the current view was created with.
    pub fn make_current(self) {
        RR_PRESET.store(self.to_ngx(), Ordering::Relaxed);
    }
}

/// DLSS Ray Reconstruction mode -- a component on the `Camera3d` entity, so the world
/// inspector edits it like anything else. Ray Reconstruction is the renderer's only resolve
/// (there is no unfiltered path), so every mode is an NGX `PerfQuality` value: the render
/// resolution comes from `NGX_DLSSD_GET_OPTIMAL_SETTINGS` at the output resolution. `Dlaa`
/// is 1:1 (denoise + AA, no upscale). A camera spawned without one gets `$AURORA_DLSS`
/// (or `Dlaa`); F3 cycles it.
#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Component, Default, Clone, PartialEq)]
pub enum AuroraDlss {
    #[default]
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl AuroraDlss {
    pub const ALL: &'static [AuroraDlss] = &[
        Self::Dlaa,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Dlaa => "dlaa",
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Performance => "performance",
            Self::UltraPerformance => "ultra-performance",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|m| m.label() == s || (s == "ultra" && *m == Self::UltraPerformance))
    }

    /// `NVSDK_NGX_PerfQuality_Value`.
    pub fn perf_quality(self) -> i32 {
        match self {
            Self::Dlaa => 5,
            Self::Quality => 2,
            Self::Balanced => 1,
            Self::Performance => 0,
            Self::UltraPerformance => 3,
        }
    }

    /// `$AURORA_DLSS` (`dlaa|quality|balanced|performance|ultra-performance`), else `Dlaa`.
    pub fn from_env() -> Self {
        std::env::var("AURORA_DLSS")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or_default()
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
}

impl core::fmt::Display for AuroraDlss {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// What the renderer settled on for this frame: the resolution to trace at and the
/// sub-pixel jitter (pixels, `[-0.5, 0.5)`, Halton(2,3)) the raygen adds to the pixel centre;
/// NGX is told its negation.
#[derive(Clone, Copy, Debug)]
pub struct DlssPlan {
    pub render: vk::Extent2D,
    pub jitter: [f32; 2],
}

/// The render-resolution images the raygen writes for NGX, in descriptor binding order 1..=7.
#[derive(Clone, Copy, Debug)]
pub struct GuideViews {
    pub normal_roughness: vk::ImageView,
    pub diffuse: vk::ImageView,
    pub specular: vk::ImageView,
    pub depth: vk::ImageView,
    pub spec_hit: vk::ImageView,
    pub motion: vk::ImageView,
    pub color: vk::ImageView,
}

pub(crate) fn halton(mut index: u32, base: u32) -> f32 {
    if base < 2 {
        return 0.0;
    }
    let mut f = 1.0f32;
    let mut result = 0.0f32;
    for _ in 0..32 {
        if index == 0 {
            break;
        }
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

/// `phase_count = max(8·ratio², 32)` (dlss_wgpu's `suggested_jitter`).
pub(crate) fn suggested_jitter(frame: u32, render_width: u32, output_width: u32) -> [f32; 2] {
    let ratio = output_width.max(1) as f32 / render_width.max(1) as f32;
    let phase_count = ((8.0 * ratio * ratio) as u32).max(32);
    let i = frame % phase_count;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

/// The one NGX session (`None` when the SDK is compiled out, the GPU/driver lacks DLSS, or
/// NGX failed to initialise -- every mode then renders as `Off`).
#[derive(Resource, Default)]
pub struct DlssState {
    pub renderer: Option<DlssRenderer>,
    /// Set by [`DlssReset`]; the next evaluate drops its history.
    pub reset_requested: bool,
    /// Dev snippet only: frames left to serialize with `vkDeviceWaitIdle` around a debug
    /// hotkey. The snippet reallocates its visualisation resources inside evaluate when a
    /// hotkey lands; overlapped with in-flight work that realloc has taken this 5090 off
    /// the bus (Xid 79), same as the un-drained NGX create/release used to.
    pub dev_drain: u32,
}

/// Drop Ray Reconstruction's temporal history on the next frame: trigger it on the camera
/// entity after a cut (teleport, scene switch), when reprojection has nothing valid to reuse.
///
/// ```ignore
/// commands.trigger(DlssReset { entity: camera });
/// ```
#[derive(EntityEvent, Clone, Copy, Debug)]
pub struct DlssReset {
    pub entity: Entity,
}

fn request_reset(_reset: On<DlssReset>, mut state: ResMut<DlssState>) {
    state.reset_requested = true;
}

/// Every 3D camera carries a mode (so it is always there to inspect and to cycle).
fn default_mode(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<AuroraDlss>)>,
) {
    for camera in &cameras {
        commands.entity(camera).insert(AuroraDlss::from_env());
    }
}

/// How many frames a dev-snippet hotkey serializes. The snippet polls evdev itself, so it
/// can act a frame or two after winit reports the press; drain a small window around it.
const DEV_DRAIN_FRAMES: u32 = 4;

/// Dev snippet only: the snippet's debug hotkeys (shift/ctrl+alt + F-keys) make it rebuild
/// visualisation resources inside the next evaluate. Spot the same combos here and have the
/// render system drain around them -- a few hitched frames instead of Xid 79.
fn dev_snippet_hotkeys(keyboard: Res<ButtonInput<KeyCode>>, mut state: ResMut<DlssState>) {
    use KeyCode::*;
    let shift = keyboard.pressed(ShiftLeft) || keyboard.pressed(ShiftRight);
    let ctrl_alt = (keyboard.pressed(ControlLeft) || keyboard.pressed(ControlRight))
        && (keyboard.pressed(AltLeft) || keyboard.pressed(AltRight));
    if (shift || ctrl_alt)
        && [F4, F5, F6, F8, F9, F11, F12]
            .iter()
            .any(|k| keyboard.just_pressed(*k))
    {
        state.dev_drain = DEV_DRAIN_FRAMES;
    }
    // Diagnostic: hold F10 to serialize every frame. If a dev-overlay artifact (flicker,
    // tearing) stops while held, it's an ordering race in our frame; if it persists under
    // full serialization, it's snippet-internal.
    if keyboard.pressed(F10) {
        state.dev_drain = state.dev_drain.max(2);
    }
}

/// F3 cycles the DLSS mode on every 3D camera (D is the free camera's strafe).
fn cycle_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<&mut AuroraDlss, With<Camera3d>>,
) {
    if keyboard.just_pressed(KeyCode::F3) {
        for mut mode in &mut cameras {
            *mode = mode.next();
            log::info!("dlss: {}", *mode);
        }
    }
}

#[cfg(dlss_ngx)]
mod ext {
    use std::ffi::{CStr, CString, c_char};
    use std::sync::atomic::Ordering;

    use ash::vk;

    use super::ngx::*;

    unsafe extern "C" fn ngx_log(message: *const c_char, level: u32, feature: u32) {
        if message.is_null() {
            return;
        }
        let msg = unsafe { CStr::from_ptr(message) };
        log::debug!(
            "[ngx l{level} f{feature}] {}",
            msg.to_string_lossy().trim_end()
        );
    }

    pub(crate) fn feature_info(feature_id: u32) -> FeatureInfo {
        // NGX looks for the per-feature snippets (libnvidia-ngx-dlss*.so) here first, then
        // falls back to the driver-installed copies.
        let mut search = Vec::new();
        if let Ok(sdk) = std::env::var("DLSS_SDK") {
            let dir = if super::DEV_SNIPPET.load(Ordering::Relaxed) {
                "dev"
            } else {
                "rel"
            };
            search.push(format!("{sdk}/lib/Linux_x86_64/{dir}"));
        }
        let verbose = std::env::var_os("AURORA_NGX_LOG").is_some();
        FeatureInfo::new(
            super::AURORA_PROJECT_ID,
            env!("CARGO_PKG_VERSION"),
            feature_id,
            std::env::temp_dir().as_os_str(),
            &search,
            if verbose {
                LOGGING_LEVEL_VERBOSE
            } else {
                LOGGING_LEVEL_OFF
            },
            if verbose { Some(ngx_log) } else { None },
        )
    }

    unsafe fn names(count: u32, ptr: *mut vk::ExtensionProperties) -> Vec<CString> {
        if ptr.is_null() {
            return Vec::new();
        }
        (0..count as usize)
            .map(|i| unsafe { CStr::from_ptr((*ptr.add(i)).extension_name.as_ptr()).to_owned() })
            .collect()
    }

    /// Instance extensions DLSS-RR needs (queried, not hardcoded).
    pub fn instance_extensions() -> Vec<CString> {
        let info = feature_info(FEATURE_RAY_RECONSTRUCTION);
        let mut count = 0u32;
        let mut ptr: *mut vk::ExtensionProperties = std::ptr::null_mut();
        let r = unsafe {
            NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                info.discovery(),
                &mut count,
                &mut ptr,
            )
        };
        if r != RESULT_SUCCESS {
            log::warn!("dlss: instance extension query -> {}", result_name(r));
            return Vec::new();
        }
        unsafe { names(count, ptr) }
    }

    /// Device extensions DLSS-RR needs on this physical device.
    pub fn device_extensions(instance: vk::Instance, pd: vk::PhysicalDevice) -> Vec<CString> {
        let info = feature_info(FEATURE_RAY_RECONSTRUCTION);
        let mut count = 0u32;
        let mut ptr: *mut vk::ExtensionProperties = std::ptr::null_mut();
        let r = unsafe {
            NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                instance,
                pd,
                info.discovery(),
                &mut count,
                &mut ptr,
            )
        };
        if r != RESULT_SUCCESS {
            log::warn!("dlss: device extension query -> {}", result_name(r));
            return Vec::new();
        }
        unsafe { names(count, ptr) }
    }
}

#[cfg(dlss_ngx)]
pub(crate) use ext::feature_info;
#[cfg(dlss_ngx)]
pub use ext::{device_extensions, instance_extensions};

#[cfg(not(dlss_ngx))]
pub fn instance_extensions() -> Vec<std::ffi::CString> {
    Vec::new()
}
#[cfg(not(dlss_ngx))]
pub fn device_extensions(_: vk::Instance, _: vk::PhysicalDevice) -> Vec<std::ffi::CString> {
    Vec::new()
}

fn teardown(world: &mut World) {
    world.resource_scope(|world, mut state: Mut<DlssState>| {
        if let Some(mut renderer) = state.renderer.take() {
            let rd = world.resource::<RenderDevice>();
            renderer.destroy(rd);
        }
    });
}

/// Adds DLSS Ray Reconstruction. `dev_snippet` swaps the NGX feature libraries for the
/// SDK's development builds (`$DLSS_SDK/lib/Linux_x86_64/dev`) and turns their status
/// HUD on -- integration debugging only, never in anything shipped.
///
/// The dev snippet's debug controls are its OWN keyboard poller, with the app window
/// focused (the NGX parameter-block overlay names exist but are inert in 310.7):
/// - `ctl+alt+shift+f12` cycle the input visualisation (`ctl+alt+f12` is documented
///   too, but Linux VT-switches on it)
/// - `ctl+alt+f11` visualisation size (window / fullscreen)
/// - `shift+f11`/`shift+f12` responsivity scale, `alt+shift+f11/f12` bias
/// - `shift+f4` bias-current mask, `shift+f5` specular-motion mode,
///   `shift+f6` flip depth-inverted, `shift+f8` accumulation mode
///
/// The snippet acts on these inside evaluate, rebuilding its visualisation resources
/// there; [`dev_snippet_hotkeys`] mirrors the combos and drains the device around them
/// (a short hitch) so that rebuild never overlaps in-flight work.
#[derive(Default)]
pub struct DlssPlugin {
    pub dev_snippet: bool,
    /// Which RR model to create the feature with; `Default` unless experimenting.
    pub preset: RrPreset,
}

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        DEV_SNIPPET.store(self.dev_snippet, Ordering::Relaxed);
        self.preset.make_current();
        app.register_type::<RrPreset>();
        if self.dev_snippet {
            // The dev snippet gates its status HUD on this (SR guide 8.2); set before
            // NGX loads it in `DlssRenderer::new` below.
            // SAFETY: main thread, before the renderer/task pools spawn readers.
            unsafe { std::env::set_var("__NGX_SHOW_INDICATOR", "1") };
        }
        app.register_type::<AuroraDlss>();
        let renderer = {
            let rd = app.world().resource::<RenderDevice>();
            DlssRenderer::new(rd)
        };
        // RR is the renderer's only resolve; without it every frame would present raw
        // 1-2 spp noise, so fail loudly here instead.
        assert!(
            renderer.is_some(),
            "DLSS Ray Reconstruction is required: build with $DLSS_SDK set and run on an \
             RR-capable NVIDIA GPU/driver"
        );
        app.insert_resource(DlssState {
            renderer,
            reset_requested: false,
            dev_drain: 0,
        });
        app.add_observer(request_reset);
        app.add_systems(Update, (default_mode, cycle_mode));
        if self.dev_snippet {
            app.add_systems(Update, dev_snippet_hotkeys);
        }
        app.add_systems(TeardownSchedule, teardown.before(on_shutdown));
    }
}
