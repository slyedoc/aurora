//! Keyboard-triggered swapchain captures, plus `AUTO_SCREENSHOT_MS` for headless runs.
//!
//! `App::add_screenshot(KeyCode::F12)` queues a capture; the render loop copies the next
//! presented frame into a host buffer and writes `./target/tmp/screenshot-<ms>.png`.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ash::vk;
use bevy::prelude::*;

/// Pending captures, drained one per frame by the render loop.
#[derive(Resource, Default)]
pub struct ScreenshotRequests(Vec<PathBuf>);

impl ScreenshotRequests {
    /// Captures the next presented frame to `path` (PNG).
    pub fn request(&mut self, path: impl Into<PathBuf>) {
        self.0.push(path.into());
    }

    pub(crate) fn take_next(&mut self) -> Option<PathBuf> {
        (!self.0.is_empty()).then(|| self.0.remove(0))
    }
}

pub trait ScreenshotExt {
    /// `trigger` saves a timestamped PNG under `./target/tmp`. `AUTO_SCREENSHOT_MS=8000`
    /// (or a comma list, `3000,3200`) captures at those milliseconds after startup --
    /// headless/agent runs with no keyboard.
    fn add_screenshot(&mut self, trigger: KeyCode) -> &mut Self;
}

fn take_screenshot(requests: &mut ScreenshotRequests) {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = PathBuf::from(format!("./target/tmp/screenshot-{ms}.png"));
    info!("screenshot: saving to {}", path.display());
    requests.request(path);
}

impl ScreenshotExt for App {
    fn add_screenshot(&mut self, trigger: KeyCode) -> &mut Self {
        self.add_systems(
            Update,
            move |mut requests: ResMut<ScreenshotRequests>, input: Res<ButtonInput<KeyCode>>| {
                // Bare keypress only: modifier combos on the same key belong to others
                // (the NGX dev overlay drives its debug layers with ctl/shift+F-keys).
                use KeyCode::*;
                let modified = [
                    ControlLeft,
                    ControlRight,
                    ShiftLeft,
                    ShiftRight,
                    AltLeft,
                    AltRight,
                ]
                .iter()
                .any(|m| input.pressed(*m));
                if input.just_pressed(trigger) && !modified {
                    take_screenshot(&mut requests);
                }
            },
        )
        .add_systems(
            Update,
            |mut requests: ResMut<ScreenshotRequests>,
             time: Res<Time>,
             mut remaining: Local<Option<Vec<u64>>>| {
                let remaining = remaining.get_or_insert_with(|| {
                    std::env::var("AUTO_SCREENSHOT_MS")
                        .ok()
                        .map(|v| {
                            let mut ms: Vec<u64> =
                                v.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                            ms.sort_unstable();
                            ms
                        })
                        .unwrap_or_default()
                });
                let now = time.elapsed_secs_f64() * 1000.0;
                if remaining.first().is_some_and(|&ms| now >= ms as f64) {
                    remaining.remove(0);
                    take_screenshot(&mut requests);
                }
            },
        )
    }
}

/// Write one frame's tightly packed pixels as PNG, swizzling BGRA surfaces to RGBA.
pub(crate) fn save_png(
    pixels: &mut [u8],
    format: vk::Format,
    extent: vk::Extent2D,
    path: &std::path::Path,
) {
    let bgra = matches!(
        format,
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_SNORM
    );
    for px in pixels.chunks_exact_mut(4) {
        if bgra {
            px.swap(0, 2);
        }
        // Presentation composites as OPAQUE, so surface alpha is meaningless in a file.
        px[3] = 255;
    }

    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            log::error!("screenshot: cannot create {}: {e}", path.display());
            return;
        }
    };
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), extent.width, extent.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    match encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(pixels))
    {
        Ok(()) => log::info!("screenshot written to {}", path.display()),
        Err(e) => log::error!("screenshot: png encode failed: {e}"),
    }
}
