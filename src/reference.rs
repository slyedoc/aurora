//! The reference: the traced image accumulated to convergence with Ray Reconstruction out of the
//! loop (docs/plans/reference-mode.html).
//!
//! Per view, at trace resolution, an RGBA32F buffer: rgb the summed radiance in nits (each
//! frame's noisy colour with the exposure divided back out), a the frame count. The raygen's
//! `accumulate` entry adds a frame after the trace and gi_shade; a reset clears the buffer first.
//! It resets whenever anything that changes the image changes (camera, settings, scene, size, a
//! `DlssReset`). On while a view shows `AuroraDebugView::Reference` / `ReferenceDiff` or a capture
//! runs ([`crate::util::capture`]).
//!
//! The buffers are host-visible: a capture pauses accumulation and reads them a few frames later,
//! with no copy. GPU writes to host memory are slow, which a debug mode can afford.

use ash::vk;
use bevy::prelude::*;
use gpu_allocator::MemoryLocation;

use crate::{
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// What makes a view's image: when it changes, the accumulation starts over.
#[derive(Clone, PartialEq, Default)]
pub struct ReferenceKey {
    pub view_proj: [f32; 16],
    pub trace: (u32, u32),
    /// `RenderSettings` and the environments' skies, as debug strings.
    pub settings: String,
    pub skies: String,
    pub light_epoch: u32,
}

#[derive(Default)]
struct ReferenceView {
    accum: Buffer<[f32; 4]>,
    /// RR's output in nits, one per output pixel, while a capture asks for it.
    output: Buffer<[f32; 4]>,
    trace: vk::Extent2D,
    out: vk::Extent2D,
    key: Option<ReferenceKey>,
    /// Frames summed since the last reset (the CPU's count; the buffer's alpha is the GPU's).
    frames: u32,
    reset: bool,
}

impl ReferenceView {
    fn release(&mut self, rd: &RenderDevice) {
        for handle in [self.accum.handle, self.output.handle] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        *self = ReferenceView::default();
    }
}

/// The views' accumulations, and the controls a capture drives.
#[derive(Resource, Default)]
pub struct ReferenceState {
    views: [ReferenceView; crate::MAX_VIEWS],
    /// A capture runs: view 0 accumulates whatever it shows.
    pub capture: bool,
    /// Skip the accumulation (a capture is reading the buffers).
    pub paused: bool,
    /// Start over next frame (a capture's single-frame and final passes).
    pub reset_next: bool,
    /// The composite writes RR's output in nits this frame.
    pub write_output: bool,
    /// Ray Reconstruction takes the accumulation's mean instead of each frame's noisy colour.
    pub feed: bool,
}

/// The addresses a view's passes take this frame (0 = off).
#[derive(Clone, Copy, Default)]
pub struct ReferenceFrame {
    pub accum: (vk::Buffer, u64),
    /// Clear the accumulation before this frame's add.
    pub reset: bool,
    /// Add this frame (false while paused).
    pub add: bool,
    pub output: u64,
}

impl ReferenceState {
    /// View `slot`'s accumulation this frame: made, resized or freed as `wanted` says, and reset
    /// when its key changed or `changed` (the scene moved, a DlssReset).
    pub fn prepare(
        &mut self,
        rd: &RenderDevice,
        slot: usize,
        wanted: bool,
        trace: vk::Extent2D,
        out: vk::Extent2D,
        key: ReferenceKey,
        changed: bool,
    ) -> ReferenceFrame {
        let wanted = wanted || (self.capture && slot == 0);
        let paused = self.paused;
        let reset_next = slot == 0 && std::mem::take(&mut self.reset_next);
        let write_output = self.write_output && slot == 0;
        let view = &mut self.views[slot];
        if !wanted {
            if view.accum.handle != vk::Buffer::null() {
                view.release(rd);
            }
            return ReferenceFrame::default();
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let mut reset = reset_next || changed || view.key.as_ref() != Some(&key);
        if view.trace != trace || view.accum.handle == vk::Buffer::null() {
            if view.accum.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(view.accum.handle);
            }
            view.accum = rd.create_buffer(
                trace.width as u64 * trace.height as u64,
                usage,
                MemoryLocation::GpuToCpu,
            );
            view.trace = trace;
            reset = true;
        }
        if write_output && (view.out != out || view.output.handle == vk::Buffer::null()) {
            if view.output.handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(view.output.handle);
            }
            view.output = rd.create_buffer(
                out.width as u64 * out.height as u64,
                usage,
                MemoryLocation::GpuToCpu,
            );
            view.out = out;
        }
        if reset {
            let why = match &view.key {
                _ if reset_next => "requested",
                _ if changed => "the scene changed",
                None => "new",
                Some(old) if old.view_proj != key.view_proj => "the camera moved",
                Some(old) if old.trace != key.trace => "the size changed",
                Some(old) if old.settings != key.settings => "the settings changed",
                Some(old) if old.skies != key.skies => "the sky changed",
                Some(old) if old.light_epoch != key.light_epoch => "the lights changed",
                _ => "the buffer was made",
            };
            log::debug!(
                "reference: view {slot} starts over ({why}) after {} frames",
                view.frames
            );
        }
        view.key = Some(key);
        if paused {
            // A capture reads what is there: hold the reset for after it.
            view.reset |= reset;
            return ReferenceFrame {
                accum: (view.accum.handle, view.accum.address),
                reset: false,
                add: false,
                output: 0,
            };
        }
        let reset = reset || std::mem::take(&mut view.reset);
        view.frames = if reset { 1 } else { view.frames + 1 };
        if view.frames.is_power_of_two() && view.frames >= 64 {
            log::debug!("reference: view {slot} at {} frames", view.frames);
        }
        ReferenceFrame {
            accum: (view.accum.handle, view.accum.address),
            reset,
            add: true,
            output: if write_output { view.output.address } else { 0 },
        }
    }

    /// Lowers view 0's CPU count so `frames` frames more run before it reaches its target again
    /// (the capture's top-up when the GPU summed fewer than the CPU counted).
    pub fn top_up(&mut self, frames: u32) {
        let view = &mut self.views[0];
        view.frames = view.frames.saturating_sub(frames);
    }

    /// Frames summed in view `slot`'s accumulation.
    pub fn frames(&self, slot: usize) -> u32 {
        self.views[slot].frames
    }

    /// View `slot`'s accumulation as radiance per traced pixel (rows top first), its size, and
    /// the frames the GPU summed (the largest count; 0 = nothing traced yet). Only meaningful
    /// while paused, after the frames that wrote it have finished.
    pub fn read_reference(
        &mut self,
        rd: &RenderDevice,
        slot: usize,
    ) -> Option<(Vec<[f32; 3]>, u32, u32, f32)> {
        let view = &mut self.views[slot];
        if view.accum.handle == vk::Buffer::null() {
            return None;
        }
        let mut data = rd.map_buffer(&mut view.accum);
        let slice = data.as_slice_mut();
        let count = slice.iter().fold(0.0f32, |m, p| m.max(p[3]));
        let pixels = slice
            .iter()
            .map(|p| {
                let n = p[3].max(1.0);
                [p[0] / n, p[1] / n, p[2] / n]
            })
            .collect();
        Some((pixels, view.trace.width, view.trace.height, count))
    }

    /// View `slot`'s captured RR output (nits per output pixel, rows top first), and its size.
    pub fn read_output(
        &mut self,
        rd: &RenderDevice,
        slot: usize,
    ) -> Option<(Vec<[f32; 3]>, u32, u32)> {
        let view = &mut self.views[slot];
        if view.output.handle == vk::Buffer::null() {
            return None;
        }
        let mut data = rd.map_buffer(&mut view.output);
        let pixels = data
            .as_slice_mut()
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Some((pixels, view.out.width, view.out.height))
    }
}

fn cleanup_reference(mut state: ResMut<ReferenceState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        view.release(&render_device);
    }
}

pub struct ReferencePlugin;

impl Plugin for ReferencePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReferenceState>();
        app.add_systems(TeardownSchedule, cleanup_reference.before(on_shutdown));
    }
}
