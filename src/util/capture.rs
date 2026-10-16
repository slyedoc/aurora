//! Reference captures (docs/plans/reference-mode.html): run an app windowless into a fixed-size
//! image target and write, as float PFMs, the reference (the traced image accumulated over N
//! frames, Ray Reconstruction out of the loop), a few single noisy frames and RR's output for the
//! same frames, all in nits, plus `capture.json`. Nothing outside the program can change the size
//! or move the camera.
//!
//!   app ... --capture DIR [--size WxH] [--frames N] [--noisy K] [--settle F] [--feed-reference]
//!           [--motion SPEED] [--sun-motion DEG] [--sun-jump DEG [--after N]]
//!
//! `--motion SPEED`: each noisy frame ends a second of camera strafe (60 steps at SPEED m/s along
//! the camera's right) at the captured pose, so RR's output and the reuse there come from a moving
//! history; the reference is accumulated at that pose, held still. `--sun-motion DEG`: likewise the
//! first directional light sweeps about the vertical at DEG per second into its pose.
//! `--sun-jump DEG`: the light sits DEG off its pose for a second, then jumps to it N frames
//! (`--after`, default 1) before the captured frame.
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
    /// Traced frames run first (RR's history, the exposure's adaptation).
    pub settle: u32,
    /// `--feed-reference`: Ray Reconstruction takes the accumulation's mean instead of the noisy
    /// colour; after the N frames, `noisy` frames of its output are captured while feeding goes
    /// on (no noisy frames: a single-frame accumulation would feed it noise).
    pub feed: bool,
    /// `--motion SPEED`: strafe speed (m/s at 60 frames a second) into each noisy frame.
    pub motion: Option<f32>,
    /// `--sun-motion DEG`: the sun's sweep (degrees a second at 60 frames a second).
    pub sun_motion: Option<f32>,
    /// `--sun-jump DEG` and `--after N`: the sun jumps DEG to its pose N frames before.
    pub sun_jump: Option<(f32, u32)>,
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
            motion: value("--motion").and_then(|v| v.parse().ok()),
            sun_motion: value("--sun-motion").and_then(|v| v.parse().ok()),
            sun_jump: value("--sun-jump").and_then(|v| v.parse().ok()).map(|deg| {
                (
                    deg,
                    value("--after")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1u32)
                        .max(1),
                )
            }),
        })
    }
}

/// Frames without a reset before a capture starts.
const QUIET: u32 = 240;

/// Frames to wait for a `Camera3d` to appear.
const CAMERA_WAIT: u32 = 100_000;

/// Frames of camera motion into each `--motion` noisy frame.
const MOTION_STEPS: u32 = 60;

/// Frames to run after pausing before the buffers are read: the frames that wrote them have
/// finished (two in flight).
const DRAIN: u32 = 4;

/// The image the captured camera renders into, and that camera.
#[derive(Resource)]
struct CaptureTarget {
    image: Handle<Image>,
    camera: Option<Entity>,
}

/// Keeps the image target on the first active `Camera3d`: a game may replace its camera once it
/// has loaded, and a camera without a target (no window) renders nothing. DLAA: Ray
/// Reconstruction at the trace's own resolution, so the reference, the noisy input and RR's
/// output are the same pixels (no upscale in the comparison).
fn retarget(
    mut commands: Commands,
    mut capture: ResMut<CaptureTarget>,
    cameras: Query<(Entity, &Camera, Option<&RenderTarget>), With<Camera3d>>,
) {
    let mut active: Vec<_> = cameras.iter().filter(|(_, c, _)| c.is_active).collect();
    active.sort_by_key(|(e, _, _)| *e);
    let Some((camera, _, target)) = active.first().copied() else {
        capture.camera = None;
        return;
    };
    let ours = matches!(target, Some(RenderTarget::Image(t)) if t.handle == capture.image);
    if !ours {
        commands.entity(camera).insert((
            RenderTarget::Image(capture.image.clone().into()),
            crate::dlss::AuroraDlss::Dlaa,
        ));
    }
    capture.camera = Some(camera);
}

/// Runs `app` (built without `WinitPlugin`) to a capture in `args.dir`, then tears the renderer
/// down. The first active `Camera3d` renders into a `args.size` image target.
pub fn run(app: &mut App, args: &CaptureArgs) {
    std::fs::create_dir_all(&args.dir).expect("capture directory");
    app.finish();
    app.cleanup();
    let image = app
        .world_mut()
        .resource_mut::<Assets<Image>>()
        .add(camera_target_placeholder(args.size));
    app.insert_resource(CaptureTarget {
        image,
        camera: None,
    });
    app.add_systems(PostUpdate, retarget);
    // Startup spawns the scene and its camera (a game may spawn it later, once it has loaded).
    let mut camera = None;
    for _ in 0..CAMERA_WAIT {
        app.update();
        camera = app.world().resource::<CaptureTarget>().camera;
        if camera.is_some() {
            break;
        }
    }
    camera.expect("a Camera3d to capture");
    app.world_mut().resource_mut::<ReferenceState>().capture = true;
    // The CPU counts frames before the trace can run (pipelines still compiling): wait for the
    // GPU's own count to move, then settle on traced frames (the auto-exposure's RR input eases
    // in from its reference over ~a second; captured earlier, RR's history lagged it dark).
    while gpu_frames(app) < 1.0 {
        for _ in 0..60 {
            app.update();
        }
    }
    for _ in 0..args.settle {
        app.update();
    }
    // A reference needs a world that holds still: animation, physics and the clock stop (the
    // capture moves the camera and the sun itself).
    app.world_mut().resource_mut::<Time<Virtual>>().pause();
    // Then until the scene is quiet: QUIET frames with no reset (a game streams textures and
    // levels of detail in for a while; a capture taken mid-stream matched neither another run
    // nor its own reference).
    while app.world().resource::<ReferenceState>().frames(0) < QUIET {
        app.update();
    }
    let camera = app
        .world()
        .resource::<CaptureTarget>()
        .camera
        .expect("a Camera3d to capture");

    if args.feed {
        feed(app, args);
        return;
    }
    let pose = *app
        .world()
        .get::<Transform>(camera)
        .expect("the camera's transform");
    let sun = app
        .world_mut()
        .query_filtered::<Entity, With<DirectionalLight>>()
        .iter(app.world())
        .next();
    let sun_pose = sun.and_then(|e| app.world().get::<Transform>(e).copied());
    let set_sun = |app: &mut App, yaw_deg: f32| {
        if let (Some(e), Some(p)) = (sun, sun_pose) {
            *app.world_mut().get_mut::<Transform>(e).unwrap() =
                p.with_rotation(Quat::from_rotation_y(yaw_deg.to_radians()) * p.rotation);
        }
    };
    for k in 0..args.noisy {
        // A second of motion that ends on the pose (the last step is the captured frame).
        if args.motion.is_some() || args.sun_motion.is_some() {
            for i in (1..MOTION_STEPS).rev() {
                let t = i as f32 / MOTION_STEPS as f32;
                if let Some(speed) = args.motion {
                    app.world_mut()
                        .get_mut::<Transform>(camera)
                        .unwrap()
                        .translation = pose.translation - pose.right() * (speed * t);
                }
                if let Some(deg) = args.sun_motion {
                    set_sun(app, -deg * t);
                }
                app.update();
            }
            *app.world_mut().get_mut::<Transform>(camera).unwrap() = pose;
            set_sun(app, 0.0);
        }
        if let Some((deg, after)) = args.sun_jump {
            set_sun(app, deg);
            for _ in 0..MOTION_STEPS {
                app.update();
            }
            set_sun(app, 0.0);
            for _ in 1..after {
                app.update();
            }
        }
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
    let resets = app.world().resource::<ReferenceState>().unrequested_resets;
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
    let changed = app.world().resource::<ReferenceState>().unrequested_resets - resets;
    if changed > 0 {
        warn!(
            "capture: the scene changed {changed} times while capturing; the frames may not match"
        );
    }
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
