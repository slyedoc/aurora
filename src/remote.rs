//! The running app over the Bevy Remote Protocol (`--features remote`): BRP's own methods (query
//! entities, read and change any reflected component or resource: `RenderSettings`, a
//! `WaterMaterial`, a `Transform`) on <http://127.0.0.1:15702>, plus aurora's:
//!
//! - `aurora.camera` `{position, look_at}` or `{position, yaw, pitch}` (degrees) holds the main
//!   camera there every frame, `{release: true}` lets it go; no params: its pose now.
//! - `aurora.screenshot` `{path?, settle?, view?, ...pose}`: the presented frame (Ray
//!   Reconstruction's, tone mapped) to a PNG once `settle` frames have run at the pose; `view`
//!   shows a debug view for this shot only. A job.
//! - `aurora.view` `{view?}`: the main camera's debug view (F4's: roughness, normals, motion,
//!   luminance, cost, reference, reference_diff, ...), or set it; `none` is the normal frame.
//! - `aurora.capture` `{dir?, frames?, ...pose}`: the reference, `frames` traced frames
//!   accumulated with the world held still (no denoiser, nits), to `dir/reference.pfm` and a PNG
//!   preview. A job.
//! - `aurora.timing` `{frames?}`: each GPU profiler scope's mean over `frames` frames. A job.
//! - `aurora.memory` `{top?}`: GPU memory by who allocated it (`memory.rs`): the allocator's
//!   capacity and live bytes, the `top` (20) source files and `file:line` sites holding the
//!   most, the device's heaps (this process's usage and budget, the driver's own included), and
//!   the blocks by how full they are, with the emptiest blocks' remaining tenants.
//! - `aurora.job` `{id}`: a job's state, and its result once it is done.
//! - `aurora.ready`: whether frames are traced yet (and since when), the TLAS's instances, the
//!   frames the profiler has read.
//! - `aurora.exit`: quits through `AppExit` (the renderer's ordered teardown, unlike a signal).
//!
//! A windowless app idles at [`IDLE_FPS`] while no job is running.
//!
//! A job runs over frames; its method answers `{job: id}` at once and `aurora.job` is polled. A
//! job fails when nothing has been traced for [`EMPTY_SECS`] with no instance in the TLAS (an
//! empty scene draws nothing, not even its sky). A game adds methods of its own with
//! [`RemoteAppExt::add_remote_method`].

use std::path::{Path, PathBuf};

use bevy::{
    platform::collections::HashMap,
    prelude::*,
    remote::{
        BrpError, BrpResult, RemoteMethodSystemId, RemoteMethods, RemotePlugin,
        http::RemoteHttpPlugin,
    },
};
use serde_json::{Value, json};

use crate::{
    debug_view::AuroraDebugView,
    gpu_profiler::GpuProfiler,
    reference::ReferenceState,
    render_device::RenderDevice,
    tlas_builder::TLAS,
    util::{HeadlessTarget, ScreenshotRequests},
};

/// Frames to run after pausing the accumulation before its buffer is read.
const DRAIN: u32 = 4;
/// Frames a job may wait before it fails.
const JOB_TIMEOUT: u32 = 60 * 600;
/// Seconds after the first traced frame before a screenshot or capture is taken: the exposure
/// adapts over about this long (a fresh start; later jobs are past it).
const WARMUP_SECS: f64 = 2.0;
/// Frames a second a windowless app runs at while no job wants frames (it would otherwise
/// trace flat out, hundreds of frames a second, for nobody).
const IDLE_FPS: f64 = 10.0;
/// Seconds a job waits on an untraced, instance-free scene before it fails.
const EMPTY_SECS: f64 = 10.0;

pub struct AuroraRemotePlugin {
    pub port: u16,
}

impl Default for AuroraRemotePlugin {
    fn default() -> Self {
        Self {
            port: bevy::remote::http::DEFAULT_PORT,
        }
    }
}

impl Plugin for AuroraRemotePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            RemotePlugin::default(),
            RemoteHttpPlugin::default().with_port(self.port),
        ))
        .init_resource::<RemotePose>()
        .init_resource::<RemoteJobs>()
        .init_resource::<FirstTrace>()
        .init_resource::<KeepAwake>()
        .add_systems(PostUpdate, hold_pose)
        .add_systems(Last, (note_first_trace, drive_jobs, idle).chain())
        .add_remote_method("aurora.camera", camera_method)
        .add_remote_method("aurora.screenshot", screenshot_method)
        .add_remote_method("aurora.capture", capture_method)
        .add_remote_method("aurora.timing", timing_method)
        .add_remote_method("aurora.job", job_method)
        .add_remote_method("aurora.ready", ready_method)
        .add_remote_method("aurora.exit", exit_method)
        .add_remote_method("aurora.view", view_method)
        .add_remote_method("aurora.memory", memory_method);
    }
}

pub trait RemoteAppExt {
    /// Adds a BRP method (after [`AuroraRemotePlugin`]).
    fn add_remote_method<M>(
        &mut self,
        name: &str,
        handler: impl IntoSystem<In<Option<Value>>, BrpResult, M> + 'static,
    ) -> &mut Self;
}

impl RemoteAppExt for App {
    fn add_remote_method<M>(
        &mut self,
        name: &str,
        handler: impl IntoSystem<In<Option<Value>>, BrpResult, M> + 'static,
    ) -> &mut Self {
        let id = self.world_mut().register_system(handler);
        self.world_mut()
            .resource_mut::<RemoteMethods>()
            .insert(name, RemoteMethodSystemId::Instant(id));
        self
    }
}

/// The pose `aurora.camera` holds the main camera at (a game whose camera rig would move it
/// reads this too).
#[derive(Resource, Default)]
pub struct RemotePose(pub Option<Transform>);

/// When (real seconds since startup) the profiler first read a frame with a trace in it.
#[derive(Resource, Default)]
struct FirstTrace(Option<f64>);

fn note_first_trace(
    mut first: ResMut<FirstTrace>,
    profiler: Res<GpuProfiler>,
    time: Res<Time<Real>>,
) {
    if first.0.is_none() && profiler.last_frame.iter().any(|(name, _)| *name == "trace") {
        first.0 = Some(time.elapsed_secs_f64());
    }
}

/// Past the warm-up after the first traced frame.
fn warmed_up(world: &World) -> bool {
    let now = world.resource::<Time<Real>>().elapsed_secs_f64();
    world
        .resource::<FirstTrace>()
        .0
        .is_some_and(|t| now - t >= WARMUP_SECS)
}

/// The first active `Camera3d`.
fn main_camera(world: &mut World) -> Option<Entity> {
    let mut cameras = world.query_filtered::<(Entity, &Camera), With<Camera3d>>();
    let mut active: Vec<_> = cameras
        .iter(world)
        .filter(|(_, c)| c.is_active)
        .map(|(e, _)| e)
        .collect();
    active.sort();
    active.first().copied()
}

fn hold_pose(pose: Res<RemotePose>, mut cameras: Query<(&Camera, &mut Transform), With<Camera3d>>) {
    let Some(pose) = pose.0 else {
        return;
    };
    if let Some((_, mut transform)) = cameras.iter_mut().find(|(c, _)| c.is_active) {
        if *transform != pose {
            *transform = pose;
        }
    }
}

fn vec3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    (a.len() == 3).then(|| {
        Vec3::new(
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
        )
    })
}

fn invalid(message: impl Into<String>) -> BrpError {
    BrpError {
        code: bevy::remote::error_codes::INVALID_PARAMS,
        message: message.into(),
        data: None,
    }
}

/// A pose from `{position, look_at}` or `{position, yaw, pitch}` (degrees); none when the
/// params name no position.
fn pose_from(params: &Value) -> Result<Option<Transform>, BrpError> {
    let Some(position) = params.get("position") else {
        return Ok(None);
    };
    let position = vec3(position).ok_or_else(|| invalid("position: [x, y, z]"))?;
    if let Some(target) = params.get("look_at") {
        let target = vec3(target).ok_or_else(|| invalid("look_at: [x, y, z]"))?;
        return Ok(Some(
            Transform::from_translation(position).looking_at(target, Vec3::Y),
        ));
    }
    let yaw = params.get("yaw").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let pitch = params.get("pitch").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    Ok(Some(Transform::from_translation(position).with_rotation(
        Quat::from_euler(EulerRot::YXZ, yaw.to_radians(), pitch.to_radians(), 0.0),
    )))
}

fn pose_json(t: &Transform) -> Value {
    let (yaw, pitch, _) = t.rotation.to_euler(EulerRot::YXZ);
    let forward = t.forward();
    json!({
        "position": t.translation.to_array(),
        "yaw": yaw.to_degrees(),
        "pitch": pitch.to_degrees(),
        "forward": forward.to_array(),
    })
}

fn camera_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    if params.get("release").and_then(Value::as_bool) == Some(true) {
        world.resource_mut::<RemotePose>().0 = None;
    } else if let Some(pose) = pose_from(&params)? {
        world.resource_mut::<RemotePose>().0 = Some(pose);
        if let Some(camera) = main_camera(world) {
            *world.get_mut::<Transform>(camera).unwrap() = pose;
        }
    }
    let camera = main_camera(world).ok_or_else(|| BrpError::internal("no active Camera3d"))?;
    let transform = *world.get::<Transform>(camera).unwrap();
    let mut out = pose_json(&transform);
    out["held"] = json!(world.resource::<RemotePose>().0.is_some());
    Ok(out)
}

enum JobKind {
    Screenshot {
        path: PathBuf,
        /// Traced frames to run first (Ray Reconstruction's history after a jump).
        settle: u32,
        /// The profiler's frame count last seen, and the traced frames counted since.
        seen: u64,
        traced: u32,
        requested: bool,
        /// The view to put back once the shot is taken (a `view` param).
        restore: Option<AuroraDebugView>,
    },
    Capture {
        dir: PathBuf,
        frames: u32,
        stage: CaptureStage,
        was_paused: bool,
        resets: u32,
    },
    Timing {
        frames: u32,
        seen: u64,
        count: u32,
        sums: Vec<(&'static str, f64)>,
    },
}

#[derive(Clone, Copy, PartialEq)]
enum CaptureStage {
    Start,
    Accumulating,
    Draining(u32),
}

enum JobState {
    Running(JobKind),
    Done(Value),
    Failed(String),
}

struct Job {
    state: JobState,
    age: u32,
    /// Since when (real seconds) the scene has had nothing traced and no instance.
    empty_since: Option<f64>,
}

#[derive(Resource, Default)]
struct RemoteJobs {
    next: u64,
    jobs: HashMap<u64, Job>,
}

fn start_job(world: &mut World, kind: JobKind) -> BrpResult {
    let mut jobs = world.resource_mut::<RemoteJobs>();
    jobs.next += 1;
    let id = jobs.next;
    jobs.jobs.insert(
        id,
        Job {
            state: JobState::Running(kind),
            age: 0,
            empty_since: None,
        },
    );
    Ok(json!({ "job": id }))
}

/// Where a job writes when the caller names no path: `target/tmp/remote/` under the working
/// directory, by milliseconds since the epoch.
fn default_path(stem: &str) -> PathBuf {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    std::env::current_dir()
        .unwrap_or_default()
        .join("target/tmp/remote")
        .join(format!("{stem}-{ms}"))
}

fn screenshot_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    if let Some(pose) = pose_from(&params)? {
        world.resource_mut::<RemotePose>().0 = Some(pose);
    }
    let path = params
        .get("path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_path("shot").with_extension("png"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(BrpError::internal)?;
    }
    let settle = params.get("settle").and_then(Value::as_u64).unwrap_or(30) as u32;
    let restore = match params.get("view").and_then(Value::as_str) {
        Some(name) => {
            let view = parse_view(name)?;
            Some(set_view(world, view)?)
        }
        None => None,
    };
    let seen = world.resource::<GpuProfiler>().frames_read;
    start_job(
        world,
        JobKind::Screenshot {
            path,
            settle,
            seen,
            traced: 0,
            requested: false,
            restore,
        },
    )
}

fn capture_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let params = params.unwrap_or(Value::Null);
    if let Some(pose) = pose_from(&params)? {
        world.resource_mut::<RemotePose>().0 = Some(pose);
    }
    let dir = params
        .get("dir")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_path("capture"));
    std::fs::create_dir_all(&dir).map_err(BrpError::internal)?;
    let frames = params.get("frames").and_then(Value::as_u64).unwrap_or(256) as u32;
    start_job(
        world,
        JobKind::Capture {
            dir,
            frames: frames.max(1),
            stage: CaptureStage::Start,
            was_paused: false,
            resets: 0,
        },
    )
}

fn timing_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let frames = params
        .as_ref()
        .and_then(|p| p.get("frames"))
        .and_then(Value::as_u64)
        .unwrap_or(120) as u32;
    let seen = world.resource::<GpuProfiler>().frames_read;
    start_job(
        world,
        JobKind::Timing {
            frames: frames.max(1),
            seen,
            count: 0,
            sums: Vec::new(),
        },
    )
}

/// A debug view by name: its variant's, any case, `_` / `-` / spaces ignored (`spec_hit`).
fn parse_view(name: &str) -> Result<AuroraDebugView, BrpError> {
    let squash = |s: &str| {
        s.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let want = squash(name);
    AuroraDebugView::ALL
        .iter()
        .copied()
        .find(|v| squash(&format!("{v:?}")) == want)
        .ok_or_else(|| invalid(format!("view: one of {}", view_names().join(", "))))
}

fn view_names() -> Vec<String> {
    AuroraDebugView::ALL
        .iter()
        .map(|v| {
            // SpecHit -> spec_hit
            let mut out = String::new();
            for (i, c) in format!("{v:?}").chars().enumerate() {
                if c.is_uppercase() && i > 0 {
                    out.push('_');
                }
                out.extend(c.to_lowercase());
            }
            out
        })
        .collect()
}

/// Shows `view` on the main camera; the view it had.
fn set_view(world: &mut World, view: AuroraDebugView) -> Result<AuroraDebugView, BrpError> {
    let camera = main_camera(world).ok_or_else(|| BrpError::internal("no active Camera3d"))?;
    let previous = world
        .get::<AuroraDebugView>(camera)
        .copied()
        .unwrap_or_default();
    world.entity_mut(camera).insert(view);
    Ok(previous)
}

fn view_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    if let Some(name) = params
        .as_ref()
        .and_then(|p| p.get("view"))
        .and_then(Value::as_str)
    {
        let view = parse_view(name)?;
        set_view(world, view)?;
    }
    let camera = main_camera(world).ok_or_else(|| BrpError::internal("no active Camera3d"))?;
    let current = world
        .get::<AuroraDebugView>(camera)
        .copied()
        .unwrap_or_default();
    let names = view_names();
    let index = AuroraDebugView::ALL
        .iter()
        .position(|v| *v == current)
        .unwrap_or(0);
    Ok(json!({ "view": names[index], "views": names }))
}

fn memory_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let top = params
        .as_ref()
        .and_then(|p| p.get("top"))
        .and_then(Value::as_u64)
        .unwrap_or(20) as usize;
    let rd = world
        .get_resource::<RenderDevice>()
        .ok_or_else(|| BrpError::internal("no render device"))?;
    let report = crate::memory::report(rd);
    let mib = |bytes: u64| (bytes as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0;
    let uses = |list: &[crate::memory::MemoryUse]| {
        list.iter()
            .take(top)
            .map(|u| json!({ "name": u.name, "mib": mib(u.bytes), "count": u.count }))
            .collect::<Vec<_>>()
    };
    let heaps: Vec<Value> = report
        .heaps
        .iter()
        .map(|h| {
            json!({
                "device_local": h.device_local,
                "size_mib": mib(h.size),
                "usage_mib": h.usage.map(mib),
                "budget_mib": h.budget.map(mib),
            })
        })
        .collect();
    Ok(json!({
        "capacity_mib": mib(report.capacity),
        "allocated_mib": mib(report.allocated),
        "blocks": report.blocks,
        "allocations": report.allocations,
        "files": uses(&report.files),
        "sites": uses(&report.sites),
        "heaps": heaps,
        "occupancy": report.occupancy.iter().map(|o| json!({
            "under": o.under,
            "blocks": o.blocks,
            "capacity_mib": mib(o.capacity),
            "used_mib": mib(o.used),
        })).collect::<Vec<_>>(),
        "sparse_blocks": report.sparse_blocks.iter().map(|b| json!({
            "size_mib": mib(b.size),
            "used_mib": mib(b.used),
            "allocations": b.allocations,
            "sites": uses(&b.sites),
        })).collect::<Vec<_>>(),
    }))
}

fn exit_method(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    world.write_message(AppExit::Success);
    Ok(Value::Null)
}

fn job_method(In(params): In<Option<Value>>, world: &mut World) -> BrpResult {
    let id = params
        .as_ref()
        .and_then(|p| p.get("id"))
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("id: the job's number"))?;
    let jobs = world.resource::<RemoteJobs>();
    let job = jobs
        .jobs
        .get(&id)
        .ok_or_else(|| invalid(format!("no job {id}")))?;
    Ok(match &job.state {
        JobState::Running(_) => {
            let mut out = ready(world);
            out["state"] = json!("running");
            out["job_frames"] = json!(job.age);
            out
        }
        JobState::Done(result) => json!({ "state": "done", "result": result }),
        JobState::Failed(error) => json!({ "state": "failed", "error": error }),
    })
}

/// A frame with a trace in it, the profiler's latest.
fn tracing(world: &World) -> bool {
    world
        .resource::<GpuProfiler>()
        .last_frame
        .iter()
        .any(|(name, _)| *name == "trace")
}

fn ready(world: &World) -> Value {
    let now = world.resource::<Time<Real>>().elapsed_secs_f64();
    let first = world.resource::<FirstTrace>().0;
    json!({
        "tracing": tracing(world),
        "first_trace_secs_ago": first.map(|t| now - t),
        "warmed_up": warmed_up(world),
        "instances": world.get_resource::<TLAS>().map_or(0, |t| t.instance_count()),
        "frames": world.resource::<GpuProfiler>().frames_read,
    })
}

fn ready_method(In(_): In<Option<Value>>, world: &mut World) -> BrpResult {
    Ok(ready(world))
}

/// Nothing traced and no instance to trace: an empty scene.
fn empty(world: &World) -> bool {
    !tracing(world)
        && world
            .get_resource::<TLAS>()
            .is_none_or(|t| t.instance_count() == 0)
}

/// Set by the app while something needs full-rate frames with no job running (a web page
/// loading or playing): a windowless app then skips its [`IDLE_FPS`] sleep. Cleared every frame.
#[derive(Resource, Default)]
pub struct KeepAwake(pub bool);

/// A windowless app with no job running sleeps out the rest of an [`IDLE_FPS`] frame.
fn idle(
    jobs: Res<RemoteJobs>,
    headless: Option<Res<HeadlessTarget>>,
    mut awake: ResMut<KeepAwake>,
    mut last: Local<Option<std::time::Instant>>,
) {
    let busy = std::mem::take(&mut awake.0)
        || jobs
            .jobs
            .values()
            .any(|job| matches!(job.state, JobState::Running(_)));
    if headless.is_some()
        && !busy
        && let Some(previous) = *last
    {
        let frame = std::time::Duration::from_secs_f64(1.0 / IDLE_FPS);
        if let Some(rest) = frame.checked_sub(previous.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    *last = Some(std::time::Instant::now());
}

/// Advances every running job by a frame.
fn drive_jobs(world: &mut World) {
    let ids: Vec<u64> = world
        .resource::<RemoteJobs>()
        .jobs
        .iter()
        .filter(|(_, j)| matches!(j.state, JobState::Running(_)))
        .map(|(id, _)| *id)
        .collect();
    for id in ids {
        let Some(mut job) = world.resource_mut::<RemoteJobs>().jobs.remove(&id) else {
            continue;
        };
        job.age += 1;
        let now = world.resource::<Time<Real>>().elapsed_secs_f64();
        job.empty_since = empty(world).then(|| job.empty_since.unwrap_or(now));
        if let JobState::Running(kind) = job.state {
            job.state = if job.age > JOB_TIMEOUT {
                JobState::Failed("timed out".into())
            } else if job.empty_since.is_some_and(|t| now - t > EMPTY_SECS) {
                JobState::Failed(format!(
                    "nothing to trace: no instance in the scene for {EMPTY_SECS} s (an empty \
                     scene draws nothing, not even its sky)"
                ))
            } else {
                step(world, kind)
            };
        }
        world.resource_mut::<RemoteJobs>().jobs.insert(id, job);
    }
}

fn step(world: &mut World, kind: JobKind) -> JobState {
    match kind {
        JobKind::Screenshot {
            path,
            settle,
            mut seen,
            mut traced,
            requested,
            restore,
        } => {
            if !requested {
                // Settle on traced frames: a frame the profiler read with a trace scope in it
                // (pipelines compiling or a scene still loading draw nothing).
                let profiler = world.resource::<GpuProfiler>();
                if profiler.frames_read != seen {
                    seen = profiler.frames_read;
                    if profiler.last_frame.iter().any(|(name, _)| *name == "trace") {
                        traced += 1;
                    }
                }
                if traced < settle.max(1) || !warmed_up(world) {
                    return JobState::Running(JobKind::Screenshot {
                        path,
                        settle,
                        seen,
                        traced,
                        requested,
                        restore,
                    });
                }
                // A windowless app's camera target, else the window.
                let target = world.get_resource::<HeadlessTarget>().map(|t| t.image.id());
                let mut requests = world.resource_mut::<ScreenshotRequests>();
                match target {
                    Some(image) => requests.request_image(image, &path),
                    None => requests.request(&path),
                }
                return JobState::Running(JobKind::Screenshot {
                    path,
                    settle,
                    seen,
                    traced,
                    requested: true,
                    restore,
                });
            }
            let written = world.resource_mut::<ScreenshotRequests>().take_written();
            match written.iter().find(|c| c.path == path) {
                Some(c) => {
                    if let Some(view) = restore {
                        let _ = set_view(world, view);
                    }
                    JobState::Done(json!({
                        "path": c.path,
                        "width": c.width,
                        "height": c.height,
                    }))
                }
                None => JobState::Running(JobKind::Screenshot {
                    path,
                    settle,
                    seen,
                    traced,
                    requested,
                    restore,
                }),
            }
        }
        JobKind::Capture {
            dir,
            frames,
            stage,
            mut was_paused,
            mut resets,
        } => {
            let next = match stage {
                CaptureStage::Start if !warmed_up(world) => CaptureStage::Start,
                CaptureStage::Start => {
                    // The world holds still for the reference: the clock stops (animation,
                    // physics) and the shader clock pins.
                    let mut time = world.resource_mut::<Time<Virtual>>();
                    was_paused = time.is_paused();
                    time.pause();
                    let mut reference = world.resource_mut::<ReferenceState>();
                    reference.capture = true;
                    reference.paused = false;
                    reference.reset_next = true;
                    resets = reference.unrequested_resets;
                    CaptureStage::Accumulating
                }
                CaptureStage::Accumulating => {
                    if world.resource::<ReferenceState>().frames(0) < frames {
                        CaptureStage::Accumulating
                    } else {
                        world.resource_mut::<ReferenceState>().paused = true;
                        CaptureStage::Draining(0)
                    }
                }
                CaptureStage::Draining(n) if n < DRAIN => CaptureStage::Draining(n + 1),
                CaptureStage::Draining(_) => {
                    let read = world.resource_scope(|world, mut r: Mut<ReferenceState>| {
                        r.read_reference(world.resource::<RenderDevice>(), 0)
                    });
                    let Some((pixels, w, h, count)) = read else {
                        return finish_capture(world, was_paused, Err("no accumulation".into()));
                    };
                    if (count as u32) < frames {
                        // The GPU summed fewer than the CPU counted: run the rest.
                        let mut reference = world.resource_mut::<ReferenceState>();
                        reference.top_up(frames - count as u32);
                        reference.paused = false;
                        return JobState::Running(JobKind::Capture {
                            dir,
                            frames,
                            stage: CaptureStage::Accumulating,
                            was_paused,
                            resets,
                        });
                    }
                    let changed = world.resource::<ReferenceState>().unrequested_resets - resets;
                    let result = write_capture(&dir, &pixels, w, h, count, changed);
                    return finish_capture(world, was_paused, result);
                }
            };
            JobState::Running(JobKind::Capture {
                dir,
                frames,
                stage: next,
                was_paused,
                resets,
            })
        }
        JobKind::Timing {
            frames,
            mut seen,
            mut count,
            mut sums,
        } => {
            // Traced frames only (a fresh start's first frames time nothing but the setup).
            let warm = warmed_up(world);
            let profiler = world.resource::<GpuProfiler>();
            if profiler.frames_read != seen && warm {
                seen = profiler.frames_read;
                count += 1;
                let busy: f64 = profiler
                    .last_frame
                    .iter()
                    .filter(|(name, _)| !matches!(*name, "frame" | "span" | "idle"))
                    .map(|(_, ms)| *ms)
                    .sum();
                for (name, ms) in profiler.last_frame.iter().copied().chain([("busy", busy)]) {
                    match sums.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, total)) => *total += ms,
                        None => sums.push((name, ms)),
                    }
                }
            }
            if count < frames {
                return JobState::Running(JobKind::Timing {
                    frames,
                    seen,
                    count,
                    sums,
                });
            }
            let means: serde_json::Map<String, Value> = sums
                .into_iter()
                .map(|(name, total)| (name.to_string(), json!(total / count as f64)))
                .collect();
            JobState::Done(json!({ "frames": count, "ms": means }))
        }
    }
}

fn finish_capture(world: &mut World, was_paused: bool, result: Result<Value, String>) -> JobState {
    let mut reference = world.resource_mut::<ReferenceState>();
    reference.capture = false;
    reference.paused = false;
    reference.reset_next = true;
    if !was_paused {
        world.resource_mut::<Time<Virtual>>().unpause();
    }
    match result {
        Ok(v) => JobState::Done(v),
        Err(e) => JobState::Failed(e),
    }
}

/// `reference.pfm` (nits), a PNG preview exposed to its mean and tone mapped, `capture.json`.
fn write_capture(
    dir: &Path,
    pixels: &[[f32; 3]],
    w: u32,
    h: u32,
    count: f32,
    resets: u32,
) -> Result<Value, String> {
    crate::util::capture::write_pfm(&dir.join("reference.pfm"), pixels, w, h);
    let luma = |p: &[f32; 3]| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
    let mean = pixels.iter().map(luma).sum::<f32>() / pixels.len().max(1) as f32;
    // Mid-grey at the log mean, Reinhard, sRGB.
    let log_mean = (pixels.iter().map(|p| (luma(p) + 1.0e-4).ln()).sum::<f32>()
        / pixels.len().max(1) as f32)
        .exp();
    let scale = 0.18 / log_mean.max(1.0e-6);
    let encode = |c: f32| {
        let x = c * scale;
        let t = x / (1.0 + x);
        let s = if t <= 0.0031308 {
            12.92 * t
        } else {
            1.055 * t.powf(1.0 / 2.4) - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let rgb: Vec<u8> = pixels
        .iter()
        .flat_map(|p| [encode(p[0]), encode(p[1]), encode(p[2])])
        .collect();
    let png = dir.join("reference.png");
    let file = std::fs::File::create(&png).map_err(|e| e.to_string())?;
    let mut encoder = ::png::Encoder::new(std::io::BufWriter::new(file), w, h);
    encoder.set_color(::png::ColorType::Rgb);
    encoder.set_depth(::png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(&rgb))
        .map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("capture.json"),
        format!("{{\n  \"width\": {w},\n  \"height\": {h},\n  \"frames\": {count},\n  \"noisy\": 0\n}}\n"),
    )
    .map_err(|e| e.to_string())?;
    Ok(json!({
        "dir": dir,
        "pfm": dir.join("reference.pfm"),
        "png": png,
        "width": w,
        "height": h,
        "frames": count,
        "mean_nits": mean,
        "scene_changed": resets,
    }))
}
