//! Reference captures (docs/plans/reference-mode.html): run an app windowless into a fixed-size
//! image target and write, as float PFMs, the reference (the traced image accumulated over N
//! frames, Ray Reconstruction out of the loop), a few single noisy frames and RR's output for the
//! same frames, all in nits, plus `capture.json`. Nothing outside the program can change the size
//! or move the camera.
//!
//!   app ... --capture DIR [--size WxH] [--frames N] [--noisy K] [--settle F] [--feed-reference]
//!
//! An example builds its app without `WinitPlugin` when [`CaptureArgs::from_args`] is some and
//! hands it to [`run`] instead of `App::run`.

use std::{
    io::Write,
    path::{Path, PathBuf},
};

use bevy::{camera::RenderTarget, prelude::*};

use crate::{
    camera_target::camera_target_placeholder, reference::ReferenceState,
    render_device::RenderDevice,
};

/// `--capture DIR [--size WxH] [--frames N] [--noisy K] [--settle F]`.
#[derive(Clone, Debug)]
pub struct CaptureArgs {
    pub dir: PathBuf,
    pub size: UVec2,
    /// Frames accumulated into the reference.
    pub frames: u32,
    /// Single frames captured (noisy colour and RR's output) before it.
    pub noisy: u32,
    /// Frames run first (assets, pipelines, BLASes, RR's history).
    pub settle: u32,
    /// `--feed-reference`: Ray Reconstruction takes the accumulation's mean instead of the noisy
    /// colour; after the N frames, `noisy` frames of its output are captured while feeding goes
    /// on (no noisy frames: a single-frame accumulation would feed it noise).
    pub feed: bool,
}

impl CaptureArgs {
    /// From the process arguments; none without `--capture`.
    pub fn from_args() -> Option<Self> {
        let args: Vec<String> = std::env::args().collect();
        let value = |name: &str| {
            let at = args.iter().position(|a| a == name)?;
            args.get(at + 1).cloned()
        };
        let dir = value("--capture")?;
        let size = value("--size")
            .and_then(|s| {
                let (w, h) = s.split_once('x')?;
                Some(UVec2::new(w.parse().ok()?, h.parse().ok()?))
            })
            .unwrap_or(UVec2::new(1920, 1080));
        Some(Self {
            dir: dir.into(),
            size,
            frames: value("--frames")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024),
            noisy: value("--noisy").and_then(|v| v.parse().ok()).unwrap_or(4),
            settle: value("--settle")
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            feed: args.iter().any(|a| a == "--feed-reference"),
        })
    }
}

/// Frames to run after pausing before the buffers are read: the frames that wrote them have
/// finished (two in flight).
const DRAIN: u32 = 4;

/// Runs `app` (built without `WinitPlugin`) to a capture in `args.dir`, then tears the renderer
/// down. The first `Camera3d` renders into a `args.size` image target.
pub fn run(app: &mut App, args: &CaptureArgs) {
    std::fs::create_dir_all(&args.dir).expect("capture directory");
    app.finish();
    app.cleanup();
    let target = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(args.size));
    // Startup spawns the scene and its camera.
    app.update();
    let camera = app
        .world_mut()
        .query_filtered::<Entity, With<Camera3d>>()
        .iter(app.world())
        .next()
        .expect("a Camera3d to capture");
    // DLAA: Ray Reconstruction at the trace's own resolution, so the reference, the noisy input
    // and RR's output are the same pixels (no upscale in the comparison).
    app.world_mut().entity_mut(camera).insert((
        RenderTarget::Image(target.into()),
        crate::dlss::AuroraDlss::Dlaa,
    ));
    app.world_mut().resource_mut::<ReferenceState>().capture = true;
    for _ in 0..args.settle {
        app.update();
    }
    // The CPU counts frames before the trace can run (pipelines still compiling): wait for the
    // GPU's own count to move.
    while gpu_frames(app) < 1.0 {
        for _ in 0..60 {
            app.update();
        }
    }

    if args.feed {
        feed(app, args);
        return;
    }
    for k in 0..args.noisy {
        // One frame: reset, add it and write RR's output; then hold still and read.
        loop {
            {
                let mut reference = app.world_mut().resource_mut::<ReferenceState>();
                reference.reset_next = true;
                reference.write_output = true;
            }
            app.update();
            pause(app, true);
            let Some((noisy, w, h, count)) = read_reference(app) else {
                pause(app, false);
                continue;
            };
            if count < 1.0 {
                // The trace did not run that frame yet.
                pause(app, false);
                app.update();
                continue;
            }
            write_pfm(&args.dir.join(format!("noisy-{k}.pfm")), &noisy, w, h);
            if let Some((out, ow, oh)) = read_output(app) {
                write_pfm(&args.dir.join(format!("output-{k}.pfm")), &out, ow, oh);
            }
            pause(app, false);
            break;
        }
    }

    app.world_mut().resource_mut::<ReferenceState>().reset_next = true;
    app.update();
    loop {
        while app.world().resource::<ReferenceState>().frames(0) < args.frames {
            app.update();
        }
        // The GPU's count is the truth; top up until it has them all.
        let have = gpu_frames(app) as u32;
        if have >= args.frames {
            break;
        }
        let mut reference = app.world_mut().resource_mut::<ReferenceState>();
        reference.top_up(args.frames - have);
    }
    pause(app, true);
    let (reference, w, h, count) = read_reference(app).expect("the accumulation");
    write_pfm(&args.dir.join("reference.pfm"), &reference, w, h);
    let settings = app
        .world()
        .get_resource::<crate::render_settings::RenderSettings>()
        .map(|s| format!("{s:?}"))
        .unwrap_or_default();
    let json = format!(
        "{{\n  \"width\": {w},\n  \"height\": {h},\n  \"output_width\": {},\n  \"output_height\": {},\n  \"frames\": {count},\n  \"noisy\": {},\n  \"settings\": \"{}\"\n}}\n",
        args.size.x,
        args.size.y,
        args.noisy,
        settings.replace('\\', "\\\\").replace('"', "\\\"")
    );
    std::fs::write(args.dir.join("capture.json"), json).expect("capture.json");
    info!(
        "capture: {} ({w}x{h}, {count} frames, {} noisy)",
        args.dir.display(),
        args.noisy
    );
    crate::ray_render_plugin::shutdown_renderer(app.world_mut());
}

/// `--feed-reference`: accumulate N frames with RR fed the running mean, then capture its output
/// for `noisy` frames (feeding on), then the reference.
fn feed(app: &mut App, args: &CaptureArgs) {
    app.world_mut().resource_mut::<ReferenceState>().feed = true;
    app.world_mut().resource_mut::<ReferenceState>().reset_next = true;
    app.update();
    while app.world().resource::<ReferenceState>().frames(0) < args.frames {
        app.update();
    }
    for k in 0..args.noisy {
        app.world_mut()
            .resource_mut::<ReferenceState>()
            .write_output = true;
        app.update();
        app.world_mut()
            .resource_mut::<ReferenceState>()
            .write_output = false;
        for _ in 0..DRAIN {
            app.update();
        }
        if let Some((out, ow, oh)) = read_output(app) {
            write_pfm(&args.dir.join(format!("output-{k}.pfm")), &out, ow, oh);
        }
    }
    pause(app, true);
    let (reference, w, h, count) = read_reference(app).expect("the accumulation");
    write_pfm(&args.dir.join("reference.pfm"), &reference, w, h);
    std::fs::write(
        args.dir.join("capture.json"),
        format!(
            "{{\n  \"width\": {w},\n  \"height\": {h},\n  \"output_width\": {},\n  \"output_height\": {},\n  \"frames\": {count},\n  \"noisy\": 0,\n  \"feed\": true\n}}\n",
            args.size.x, args.size.y
        ),
    )
    .expect("capture.json");
    info!(
        "capture (fed): {} ({w}x{h}, {count} frames)",
        args.dir.display()
    );
    crate::ray_render_plugin::shutdown_renderer(app.world_mut());
}

/// The frames the GPU has summed into view 0's accumulation (pauses to read, then resumes).
fn gpu_frames(app: &mut App) -> f32 {
    pause(app, true);
    let count = read_reference(app).map_or(0.0, |r| r.3);
    pause(app, false);
    count
}

/// Stops (and resumes) adding frames; stopping also lets the frames in flight finish.
fn pause(app: &mut App, paused: bool) {
    {
        let mut reference = app.world_mut().resource_mut::<ReferenceState>();
        reference.paused = paused;
        reference.write_output = false;
    }
    if paused {
        for _ in 0..DRAIN {
            app.update();
        }
    }
}

fn read_reference(app: &mut App) -> Option<(Vec<[f32; 3]>, u32, u32, f32)> {
    app.world_mut()
        .resource_scope(|world, mut reference: Mut<ReferenceState>| {
            reference.read_reference(world.resource::<RenderDevice>(), 0)
        })
}

fn read_output(app: &mut App) -> Option<(Vec<[f32; 3]>, u32, u32)> {
    app.world_mut()
        .resource_scope(|world, mut reference: Mut<ReferenceState>| {
            reference.read_output(world.resource::<RenderDevice>(), 0)
        })
}

/// A portable float map: `PF`, the size, -1 (little-endian), then rows bottom first.
fn write_pfm(path: &Path, pixels: &[[f32; 3]], width: u32, height: u32) {
    let mut out = Vec::with_capacity(pixels.len() * 12 + 32);
    write!(out, "PF\n{width} {height}\n-1.0\n").unwrap();
    for y in (0..height as usize).rev() {
        for p in &pixels[y * width as usize..(y + 1) * width as usize] {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    std::fs::write(path, out).expect("pfm");
}
