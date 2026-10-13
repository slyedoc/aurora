//! The interaction field: where pushers (characters, creatures) disturbed the ground lately,
//! for scattered plants to lean away from (`landscape/interaction.slang`).
//!
//! A [`FIELD_N`]² grid of [`FIELD_CELL`]-metre texels around the viewer, addressed
//! toroidally (texel = world cell mod N), so it scrolls with the viewer without a copy; each
//! texel remembers the world cell it holds and resets when a new one scrolls in. Every frame
//! a kernel decays every texel towards rest and splats every [`ScatterPusher`] into it (a push
//! away from the pusher, strongest at its centre), keeping the stronger of the two: what was
//! walked through stays bent a moment, then springs back. The scatter kernel shears each
//! plant's instance away by the push at its spot.

use ash::vk;
use bevy::prelude::*;
use bevy_aurora::{
    compute::{ComputeModules, FrameCommands, memory_barrier, record_dispatch},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

use crate::{LandscapeKernels, LandscapeViewer, tiles::viewer_position};

/// Texels per side of the field.
pub const FIELD_N: u32 = 256;
/// Metres per texel (the field spans `FIELD_N * FIELD_CELL` around the viewer).
pub const FIELD_CELL: f32 = 0.25;
/// Most pushers the field takes per frame; the nearest win.
pub const PUSHERS_MAX: usize = 64;

/// Something that parts the plants it walks through: a push of `strength` (the tangent of
/// the lean it gives, at its centre) fading out to `radius` metres.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component, Default)]
pub struct ScatterPusher {
    #[reflect(@0.0..=10.0_f32)]
    pub radius: f32,
    #[reflect(@0.0..=2.0_f32)]
    pub strength: f32,
}

impl Default for ScatterPusher {
    fn default() -> Self {
        Self {
            radius: 0.7,
            strength: 1.0,
        }
    }
}

/// The field's tuning.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource, Default)]
pub struct InteractionSettings {
    /// Seconds for a bent plant to come most of the way back (1/e of its push).
    #[reflect(@0.05..=10.0_f32)]
    pub recover: f32,
}

impl Default for InteractionSettings {
    fn default() -> Self {
        Self { recover: 1.2 }
    }
}

/// `FieldTexel` in interaction.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct FieldTexel {
    push: [f32; 2],
    owner: u32,
    pad: u32,
}

/// `FieldPush` in interaction.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FieldPush {
    field: u64,
    pushers: u64,
    count: u32,
    decay: f32,
    centre_x: i32,
    centre_z: i32,
}

/// The field on the GPU, and the frame's pushers.
#[derive(Resource, Default)]
pub struct InteractionField {
    texels: Buffer<FieldTexel>,
    pushers: Buffer<[f32; 4]>,
    /// Elapsed seconds until which the field may still be disturbed (pushers seen lately).
    live_until: f64,
}

impl InteractionField {
    /// The texels' address for the scatter kernel, 0 while nothing has disturbed the field.
    pub fn address(&self, now: f64) -> u64 {
        if now <= self.live_until {
            self.texels.address
        } else {
            0
        }
    }

    fn destroy(&mut self, rd: &RenderDevice) {
        for handle in [self.texels.handle, self.pushers.handle] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
        *self = Self::default();
    }
}

/// Decay the field and splat this frame's pushers into it.
pub fn update_field(
    rd: Res<RenderDevice>,
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    time: Res<Time>,
    settings: Res<InteractionSettings>,
    mut field: ResMut<InteractionField>,
    mut frame: ResMut<FrameCommands>,
    viewers: Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    pushers: Query<(&GlobalTransform, &ScatterPusher)>,
) {
    if modules.get(&kernels.interaction).is_none() {
        return;
    }
    let Some(viewer) = viewer_position(&viewers, &cameras) else {
        return;
    };
    let now = time.elapsed_secs_f64();
    let reach = FIELD_N as f32 * FIELD_CELL * 0.5;
    let mut near: Vec<(f32, [f32; 4])> = pushers
        .iter()
        .filter(|(_, p)| p.radius > 0.0 && p.strength > 0.0)
        .map(|(t, p)| {
            let at = t.translation();
            let d = (at - viewer).xz().length();
            (d, [at.x, at.z, p.radius, p.strength])
        })
        .filter(|(d, _)| *d < reach)
        .collect();
    if !near.is_empty() {
        // Long enough for the last push to decay away.
        field.live_until = now + settings.recover as f64 * 6.0;
    }
    if now > field.live_until {
        return;
    }
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    near.truncate(PUSHERS_MAX);
    if field.texels.handle == vk::Buffer::null() {
        let n = (FIELD_N * FIELD_N) as u64;
        field.texels = rd.create_device_buffer(
            n,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
        rd.run_transfer_commands_after_frames(|cmd| unsafe {
            rd.device
                .cmd_fill_buffer(cmd, field.texels.handle, 0, vk::WHOLE_SIZE, 0);
        });
        field.pushers = rd.create_device_buffer(
            PUSHERS_MAX as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        );
    }
    let mut rows = [[0.0f32; 4]; PUSHERS_MAX];
    for (row, (_, p)) in rows.iter_mut().zip(&near) {
        *row = *p;
    }
    let centre = (viewer.xz() / FIELD_CELL).floor().as_ivec2();
    let push = FieldPush {
        field: field.texels.address,
        pushers: field.pushers.address,
        count: near.len() as u32,
        decay: (-time.delta_secs() / settings.recover.max(0.01)).exp(),
        centre_x: centre.x,
        centre_z: centre.y,
    };
    // Recorded into the frame's command buffer: a blocking submit here stalled the CPU on
    // the GPU every frame.
    let pushers = field.pushers.handle;
    let kernel = kernels.interaction.clone();
    frame.push(move |rd, modules, cmd| {
        let Some(module) = modules.get(&kernel) else {
            return;
        };
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, pushers, 0, bytemuck::cast_slice(&rows));
        }
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        record_dispatch(
            rd,
            cmd,
            module,
            "field_update",
            &push,
            FIELD_N * FIELD_N,
            None,
        );
    });
}

pub fn release_field(mut field: ResMut<InteractionField>, rd: Res<RenderDevice>) {
    field.destroy(&rd);
}
