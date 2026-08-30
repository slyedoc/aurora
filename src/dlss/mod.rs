//! DLSS Ray Reconstruction — NGX on aurora's Vulkan device.
//!
//! The path tracer writes, at render resolution, the guides DLSS-RR consumes (noisy HDR
//! colour, linear depth, motion vectors, diffuse/specular albedo, normal + roughness,
//! specular hit distance); NGX denoises and upscales them into an output-resolution image
//! that the post-process blit reads instead of the accumulation target. Mode lives in
//! [`RenderConfig::dlss`](crate::ray_render_plugin::RenderConfig): `Off` traces at output
//! resolution exactly as before.
//!
//! The NGX FFI ([`ngx`]) and the call sequence are the ones proved in the old engine
//! (`bevy_aurora_old/docs/dlss_notes.md`). Everything is gated on `cfg(dlss_ngx)`, set by
//! `build.rs` when `$DLSS_SDK` holds the SDK; without it [`DlssRenderer`] is a stub and every
//! mode renders as `Off`.

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

/// DLSS Ray Reconstruction mode -- a component on the `Camera3d` entity, so the world
/// inspector edits it like anything else. Every non-`Off` mode is an NGX `PerfQuality` value:
/// the render resolution comes from `NGX_DLSSD_GET_OPTIMAL_SETTINGS` at the output resolution.
/// `Dlaa` is 1:1 (denoise + AA, no upscale). A camera spawned without one gets
/// `$AURORA_DLSS` (or `Off`); F3 cycles it.
#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Component, Default, Clone, PartialEq)]
pub enum AuroraDlss {
    #[default]
    Off,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
}

impl AuroraDlss {
    pub const ALL: &'static [AuroraDlss] = &[
        Self::Off,
        Self::Dlaa,
        Self::Quality,
        Self::Balanced,
        Self::Performance,
        Self::UltraPerformance,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
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

    /// `NVSDK_NGX_PerfQuality_Value` (`None` when off).
    pub fn perf_quality(self) -> Option<i32> {
        Some(match self {
            Self::Off => return None,
            Self::Dlaa => 5,
            Self::Quality => 2,
            Self::Balanced => 1,
            Self::Performance => 0,
            Self::UltraPerformance => 3,
        })
    }

    /// `$AURORA_DLSS` (`dlaa|quality|balanced|performance|ultra-performance`), else `Off`.
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

    #[inline]
    pub fn is_on(self) -> bool {
        self != Self::Off
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
            search.push(format!("{sdk}/lib/Linux_x86_64/rel"));
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

pub struct DlssPlugin;

impl Plugin for DlssPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraDlss>();
        let renderer = {
            let rd = app.world().resource::<RenderDevice>();
            DlssRenderer::new(rd)
        };
        app.insert_resource(DlssState {
            renderer,
            reset_requested: false,
        });
        app.add_observer(request_reset);
        app.add_systems(Update, (default_mode, cycle_mode));
        app.add_systems(TeardownSchedule, teardown.before(on_shutdown));
    }
}
