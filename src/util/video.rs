//! Video capture of the presented window (UI included) to H.264 `.mp4`, through `ffmpeg`.
//!
//! [`VideoRecorder::start`] / [`stop`](VideoRecorder::stop) (or the toggle key from
//! [`VideoExt::add_video_recording`]). While recording, the frame due at `fps` is copied out of
//! the swapchain into one of two host buffers; one frame in flight means it has landed by the
//! next frame, which hands the bytes to an encoder thread piping raw frames into `ffmpeg`
//! (NVENC). The game never waits on it: a frame the encoder has no room for is dropped and its
//! time goes to the next one. Real-time playback: a frame that covers several slots of the
//! output rate (the game ran slower than `fps`) is written that many times.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ash::vk;
use bevy::prelude::*;

use gpu_allocator::MemoryLocation;

use crate::render_buffer::{Buffer, BufferProvider};
use crate::render_device::RenderDevice;

/// Most slots one frame may fill (a hitch longer than this is cut short, not frozen in).
const MAX_REPEAT: u32 = 8;

/// The recording state, its capture buffers and the encoder.
#[derive(Resource)]
pub struct VideoRecorder {
    /// Output frame rate.
    pub fps: u32,
    /// NVENC constant quality (lower = better, bigger).
    pub cq: u32,
    request: Option<Request>,
    session: Option<Session>,
}

enum Request {
    Start(PathBuf),
    Stop,
}

struct Session {
    path: PathBuf,
    extent: vk::Extent2D,
    buffers: [Buffer<u8>; 2],
    /// Copies recorded and not yet sent: (frame, buffer, output slots it covers), oldest first.
    pending: Vec<(usize, usize, u32)>,
    next: usize,
    started: Instant,
    slots_written: u64,
    /// Slots of frames the encoder had no room for, given to the next frame sent.
    carry: u32,
    frames: SyncSender<(Vec<u8>, u32)>,
    encoder: JoinHandle<()>,
}

impl Default for VideoRecorder {
    fn default() -> Self {
        Self {
            fps: 60,
            cq: 20,
            request: None,
            session: None,
        }
    }
}

impl VideoRecorder {
    /// Starts recording to `path` (`.mp4`) from the next frame.
    pub fn start(&mut self, path: impl Into<PathBuf>) {
        self.request = Some(Request::Start(path.into()));
    }

    /// Ends the recording; the file is finished once the encoder drains (a moment later).
    pub fn stop(&mut self) {
        self.request = Some(Request::Stop);
    }

    pub fn is_recording(&self) -> bool {
        matches!(self.request, Some(Request::Start(_))) || self.session.is_some()
    }

    /// After frame `done`'s wait: its captures (and older) go to the encoder.
    pub(crate) fn latch(&mut self, rd: &RenderDevice, done: usize) {
        if let Some(session) = self.session.as_mut() {
            session.flush(rd, done);
        }
    }

    /// Render side, once per presented frame `frame`: applies a start/stop, and copies the
    /// frame out of the swapchain image (in TRANSFER_SRC_OPTIMAL) when one is due; [`Self::latch`]
    /// hands it to the encoder once the frame has finished. Returns whether it copied.
    pub(crate) fn record(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
        frame: usize,
    ) -> bool {
        match self.request.take() {
            Some(Request::Start(path)) => {
                self.finish(rd);
                self.session = Session::open(rd, path, extent, format, self.fps, self.cq);
            }
            Some(Request::Stop) => self.finish(rd),
            None => {}
        }
        let Some(session) = self.session.as_mut() else {
            return false;
        };
        if session.extent != extent {
            warn!("video: the window resized; recording stopped");
            self.finish(rd);
            return false;
        }
        let due = (session.started.elapsed().as_secs_f64() * self.fps as f64) as u64 + 1;
        if due <= session.slots_written {
            return false;
        }
        let repeat = (due - session.slots_written).min(MAX_REPEAT as u64) as u32;
        session.slots_written = due;
        let i = session.next;
        session.next ^= 1;
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        unsafe {
            rd.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                session.buffers[i].handle,
                std::slice::from_ref(&region),
            );
        }
        session.pending.push((frame, i, repeat));
        true
    }

    /// Whether this frame needs the swapchain image copied (a session or a start is live).
    pub(crate) fn wants_frame(&self) -> bool {
        self.session.is_some() || matches!(self.request, Some(Request::Start(_)))
    }

    fn finish(&mut self, rd: &RenderDevice) {
        if let Some(mut session) = self.session.take() {
            // The last copy may still be in flight: drain it before reading.
            if !session.pending.is_empty() {
                let _queue = rd.queue.lock().unwrap();
                let _ = unsafe { rd.device.device_wait_idle() };
                session.flush(rd, usize::MAX);
            }
            let Session {
                path,
                buffers,
                frames,
                encoder,
                ..
            } = session;
            drop(frames);
            for buffer in buffers {
                rd.destroyer.destroy_buffer(buffer.handle);
            }
            // The encoder finishes the file on its own thread.
            std::thread::spawn(move || {
                let _ = encoder.join();
                info!("video written to {}", path.display());
            });
        }
    }
}

impl Session {
    fn open(
        rd: &RenderDevice,
        path: PathBuf,
        extent: vk::Extent2D,
        format: vk::Format,
        fps: u32,
        cq: u32,
    ) -> Option<Self> {
        let pix_fmt = match format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_SNORM => {
                "bgra"
            }
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB | vk::Format::R8G8B8A8_SNORM => {
                "rgba"
            }
            vk::Format::A2B10G10R10_UNORM_PACK32 => "x2bgr10le",
            vk::Format::A2R10G10B10_UNORM_PACK32 => "x2rgb10le",
            other => {
                error!("video: cannot record a {other:?} swapchain");
                return None;
            }
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // x264 wants even dimensions.
        let (w, h) = (extent.width & !1, extent.height & !1);
        let child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"])
            .args(["-pix_fmt", pix_fmt])
            .args(["-s", &format!("{}x{}", extent.width, extent.height)])
            .args(["-r", &fps.to_string(), "-i", "-"])
            .args(["-vf", &format!("crop={w}:{h}:0:0")])
            .args([
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-rc",
                "vbr",
                "-cq",
                &cq.to_string(),
            ])
            .args(["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(&path)
            .stdin(Stdio::piped())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(e) => {
                error!("video: cannot start ffmpeg: {e}");
                return None;
            }
        };
        let stdin = child.stdin.take()?;
        // A few frames of slack; past that frames are dropped (their time carried), never
        // buffered without bound or waited for.
        let (frames, rx) = sync_channel::<(Vec<u8>, u32)>(4);
        let encoder = std::thread::spawn(move || encode(child, stdin, rx));
        let size = extent.width as u64 * extent.height as u64 * 4;
        // Read back every frame: host-cached memory (CpuToGpu can sit in VRAM behind the BAR,
        // where reading 8 MB took a frame's worth of time).
        let buffers = [0, 1].map(|_| {
            rd.create_buffer(
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )
        });
        info!(
            "video: recording {}x{} at {fps} fps to {}",
            extent.width,
            extent.height,
            path.display()
        );
        Some(Self {
            path,
            extent,
            buffers,
            pending: Vec::new(),
            next: 0,
            started: Instant::now(),
            slots_written: 0,
            carry: 0,
            frames,
            encoder,
        })
    }

    /// Sends the captures of frames up to `done` (landed: those frames finished) to the
    /// encoder, in order.
    fn flush(&mut self, rd: &RenderDevice, done: usize) {
        while let Some(&(frame, i, repeat)) = self.pending.first() {
            if frame > done {
                break;
            }
            self.pending.remove(0);
            let repeat = (repeat + self.carry).min(MAX_REPEAT);
            let bytes = rd.map_buffer(&mut self.buffers[i]).as_slice_mut().to_vec();
            self.carry = match self.frames.try_send((bytes, repeat)) {
                Ok(()) => 0,
                Err(TrySendError::Full(_)) => repeat,
                Err(TrySendError::Disconnected(_)) => 0,
            };
        }
    }
}

fn encode(mut child: Child, mut stdin: ChildStdin, frames: Receiver<(Vec<u8>, u32)>) {
    'frames: while let Ok((frame, repeat)) = frames.recv() {
        for _ in 0..repeat.max(1) {
            if stdin.write_all(&frame).is_err() {
                error!("video: ffmpeg stopped taking frames");
                break 'frames;
            }
        }
    }
    drop(stdin);
    let _ = child.wait();
}

/// A timestamped path under `./target/tmp`.
pub fn video_path() -> PathBuf {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    PathBuf::from(format!("./target/tmp/video-{ms}.mp4"))
}

pub trait VideoExt {
    /// `toggle` starts and stops a recording to a timestamped `./target/tmp/video-<ms>.mp4`.
    fn add_video_recording(&mut self, toggle: KeyCode) -> &mut Self;
}

impl VideoExt for App {
    fn add_video_recording(&mut self, toggle: KeyCode) -> &mut Self {
        self.add_systems(
            Update,
            move |mut video: ResMut<VideoRecorder>, input: Res<ButtonInput<KeyCode>>| {
                if !input.just_pressed(toggle) {
                    return;
                }
                if video.is_recording() {
                    video.stop();
                } else {
                    video.start(video_path());
                }
            },
        )
    }
}
