//! Keyboard-triggered swapchain captures, plus `AUTO_SCREENSHOT_MS` for headless runs.
//!
//! `App::add_screenshot(KeyCode::F12)` queues a capture; the render loop copies the next
//! presented frame into a host buffer, picks it up a frame later (its fence passed, so nothing
//! waits on the GPU) and writes `./target/tmp/screenshot-<ms>.png` on a thread of its own.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use ash::vk;
use bevy::prelude::*;

use crate::render_buffer::{Buffer, BufferProvider};
use crate::render_device::RenderDevice;

/// Pending captures, drained one per frame by the render loop, and the ones already on
/// disk. A capture lands several frames after it is asked for, so a caller that has to
/// answer for it -- the editor's remote screenshot method -- polls [`take_written`].
#[derive(Resource, Default)]
pub struct ScreenshotRequests {
    pending: Vec<PathBuf>,
    /// Image-target captures: the next frame drawn into that camera target (a windowless app's
    /// screenshot).
    pending_images: Vec<(AssetId<Image>, PathBuf)>,
    /// Copies recorded into this frame, read back the next.
    in_flight: Vec<InFlight>,
    /// Filled by the PNG threads.
    written: Arc<Mutex<Vec<Capture>>>,
}

struct InFlight {
    /// The frame whose commands copy into `buffer`.
    frame: usize,
    path: PathBuf,
    buffer: Buffer<u8>,
    extent: vk::Extent2D,
    format: vk::Format,
}

/// A capture that reached disk, with the size it was written at.
pub struct Capture {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
}

impl ScreenshotRequests {
    /// Captures the next presented frame to `path` (PNG).
    pub fn request(&mut self, path: impl Into<PathBuf>) {
        self.pending.push(path.into());
    }

    /// Captures the next frame drawn into the display image target `image` to `path` (PNG).
    pub fn request_image(&mut self, image: AssetId<Image>, path: impl Into<PathBuf>) {
        self.pending_images.push((image, path.into()));
    }

    pub(crate) fn take_image(&mut self, image: AssetId<Image>) -> Option<PathBuf> {
        let at = self.pending_images.iter().position(|(i, _)| *i == image)?;
        Some(self.pending_images.remove(at).1)
    }

    /// Every capture written since the last call.
    pub fn take_written(&mut self) -> Vec<Capture> {
        std::mem::take(&mut *self.written.lock().unwrap())
    }

    pub(crate) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(crate) fn take_next(&mut self) -> Option<PathBuf> {
        (!self.pending.is_empty()).then(|| self.pending.remove(0))
    }

    /// This frame's copy of a capture into `buffer`, for [`collect`](Self::collect) next frame.
    pub(crate) fn in_flight(
        &mut self,
        path: PathBuf,
        buffer: Buffer<u8>,
        extent: vk::Extent2D,
        format: vk::Format,
        frame: usize,
    ) {
        self.in_flight.push(InFlight {
            frame,
            path,
            buffer,
            extent,
            format,
        });
    }

    /// After frame `done`'s wait: the copies of frames up to it have landed, and their pixels
    /// go to a PNG thread each.
    pub(crate) fn collect(&mut self, rd: &RenderDevice, done: usize) {
        let (landed, waiting): (Vec<_>, Vec<_>) =
            self.in_flight.drain(..).partition(|c| c.frame <= done);
        self.in_flight = waiting;
        for mut capture in landed {
            let mut pixels = rd.map_buffer(&mut capture.buffer).as_slice_mut().to_vec();
            rd.destroyer.destroy_buffer(capture.buffer.handle);
            let written = self.written.clone();
            let (path, extent, format) = (capture.path, capture.extent, capture.format);
            std::thread::spawn(move || {
                if save_png(&mut pixels, format, extent, &path) {
                    written.lock().unwrap().push(Capture {
                        path,
                        width: extent.width,
                        height: extent.height,
                    });
                }
            });
        }
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
) -> bool {
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
            return false;
        }
    };
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), extent.width, extent.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    match encoder
        .write_header()
        .and_then(|mut w| w.write_image_data(pixels))
    {
        Ok(()) => {
            log::info!("screenshot written to {}", path.display());
            true
        }
        Err(e) => {
            log::error!("screenshot: png encode failed: {e}");
            false
        }
    }
}
