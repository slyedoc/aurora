//! Neural radiance cache (Müller, Rousselle, Novák, Keller 2021): a small MLP that learns the
//! scattered outgoing radiance at a surface point from the frame's own paths, so a path that has
//! spread far can end in the network's answer. Plan and gates: docs/plans/nrc.html.
//!
//! [`NrcNetwork`] is the network and its training state (assets/shaders/nrc_kernels.slang):
//! six 64 x 64 layers whose f16 copies feed the tensor cores (cooperative vectors), an f32
//! master copy with Adam's moments, and a ring of training records.

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};
use half::f16;
use rand::{Rng, SeedableRng};

use crate::{
    assets::aurora_asset,
    compute::{
        CompiledComputeModule, ComputeModule, ComputeModules, memory_barrier, record_dispatch,
    },
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_settings::RenderSettings,
};

pub const NRC_WIDTH: u64 = 64;
pub const NRC_LAYERS: u64 = 6;
pub const NRC_MATRIX: u64 = NRC_WIDTH * NRC_WIDTH;
/// Weights, then biases (the master copy's layout).
pub const NRC_PARAMS: u64 = NRC_LAYERS * (NRC_MATRIX + NRC_WIDTH);
/// Batch rows per gradient workgroup (nrc_kernels.slang NRC_CHUNK).
pub const NRC_CHUNK: u32 = 256;
/// Must match NrcRecord in nrc.slang.
pub const NRC_RECORD_BYTES: u64 = 48;

/// Must match NrcNet in nrc_kernels.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NrcNetGpu {
    pub weights: u64,
    pub weights_t: u64,
    pub biases: u64,
    pub params: u64,
    pub adam_m: u64,
    pub adam_v: u64,
    pub partials: u64,
    pub acts: u64,
    pub dz: u64,
    pub loss: u64,
    pub records: u64,
    pub counter: u64,
    pub ema: u64,
    pub weights_ema: u64,
    pub biases_ema: u64,
    pub ring: u32,
    pub batch: u32,
    pub learning_rate: f32,
    pub radiance_scale: f32,
    pub loss_scale: f32,
    pub ema_decay: f32,
}

/// Must match NrcPush in nrc_kernels.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NrcPush {
    net: u64,
    out: u64,
    base: u32,
    offset: u32,
    step: u32,
    pad: u32,
}

/// Training hyperparameters.
#[derive(Clone, Copy, Debug)]
pub struct NrcTraining {
    pub learning_rate: f32,
    pub radiance_scale: f32,
    pub loss_scale: f32,
    /// Per Adam step (four a frame): 0.99 averages the last ~100 steps.
    pub ema_decay: f32,
}

impl Default for NrcTraining {
    fn default() -> Self {
        Self {
            learning_rate: 2.0e-3,
            radiance_scale: 1.0,
            loss_scale: 1.0 / 16.0,
            ema_decay: 0.99,
        }
    }
}

pub struct NrcNetwork {
    pub weights: Buffer<u8>,
    pub weights_t: Buffer<u8>,
    pub biases: Buffer<u8>,
    pub params: Buffer<u8>,
    pub adam_m: Buffer<u8>,
    pub adam_v: Buffer<u8>,
    pub partials: Buffer<u8>,
    pub acts: Buffer<u8>,
    pub dz: Buffer<u8>,
    pub loss: Buffer<u8>,
    pub records: Buffer<u8>,
    pub counter: Buffer<u8>,
    /// The parameters' moving average and its f16 copies: what inference reads.
    pub ema: Buffer<u8>,
    pub weights_ema: Buffer<u8>,
    pub biases_ema: Buffer<u8>,
    pub net: Buffer<NrcNetGpu>,
    pub batch: u32,
    pub ring: u32,
    /// Adam steps taken.
    pub step: u32,
    training: NrcTraining,
    dirty: bool,
}

/// He-uniform initial weights ([layer][out][in]) and zero biases, from `seed`.
pub fn initial_params(seed: u64) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let bound = (6.0 / NRC_WIDTH as f32).sqrt();
    let mut params = vec![0.0f32; NRC_PARAMS as usize];
    for w in &mut params[..(NRC_LAYERS * NRC_MATRIX) as usize] {
        *w = rng.gen_range(-bound..bound);
    }
    params
}

/// The f16 forward copy ([layer][out][in]) and its transpose ([layer][in][out]) of `params`'s
/// weights, and its biases.
pub fn half_copies(params: &[f32]) -> (Vec<f16>, Vec<f16>, Vec<f16>) {
    let n = NRC_WIDTH as usize;
    let weights: Vec<f16> = params[..(NRC_LAYERS * NRC_MATRIX) as usize]
        .iter()
        .map(|&w| f16::from_f32(w))
        .collect();
    let mut weights_t = vec![f16::ZERO; weights.len()];
    for l in 0..NRC_LAYERS as usize {
        for o in 0..n {
            for i in 0..n {
                weights_t[l * n * n + i * n + o] = weights[l * n * n + o * n + i];
            }
        }
    }
    let biases = params[(NRC_LAYERS * NRC_MATRIX) as usize..]
        .iter()
        .map(|&b| f16::from_f32(b))
        .collect();
    (weights, weights_t, biases)
}

impl NrcNetwork {
    /// The network with fresh weights (uploaded and waited on) and `ring` training records,
    /// trained `batch` records per step.
    pub fn new(rd: &RenderDevice, batch: u32, ring: u32, seed: u64) -> Self {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let chunks = batch.div_ceil(NRC_CHUNK) as u64;
        let rows = batch as u64 * NRC_LAYERS * NRC_WIDTH;
        let mut this = Self {
            weights: rd.create_device_buffer(NRC_LAYERS * NRC_MATRIX * 2, usage),
            weights_t: rd.create_device_buffer(NRC_LAYERS * NRC_MATRIX * 2, usage),
            biases: rd.create_device_buffer(NRC_LAYERS * NRC_WIDTH * 2, usage),
            params: rd.create_device_buffer(NRC_PARAMS * 4, usage),
            adam_m: rd.create_device_buffer(NRC_PARAMS * 4, usage),
            adam_v: rd.create_device_buffer(NRC_PARAMS * 4, usage),
            partials: rd.create_device_buffer(chunks * NRC_PARAMS * 4, usage),
            acts: rd.create_device_buffer(rows * 2, usage),
            dz: rd.create_device_buffer(rows * 2, usage),
            loss: rd.create_device_buffer(batch as u64 * 4, usage),
            records: rd.create_device_buffer(ring as u64 * NRC_RECORD_BYTES, usage),
            counter: rd.create_device_buffer(4, usage),
            ema: rd.create_device_buffer(NRC_PARAMS * 4, usage),
            weights_ema: rd.create_device_buffer(NRC_LAYERS * NRC_MATRIX * 2, usage),
            biases_ema: rd.create_device_buffer(NRC_LAYERS * NRC_WIDTH * 2, usage),
            net: rd.create_device_buffer(1, usage),
            batch,
            ring,
            step: 0,
            training: NrcTraining::default(),
            dirty: true,
        };
        this.reset(rd, seed);
        this
    }

    /// Fresh weights from `seed`, Adam's moments cleared, records zeroed (waits for the upload).
    pub fn reset(&mut self, rd: &RenderDevice, seed: u64) {
        let params = initial_params(seed);
        let (weights, weights_t, biases) = half_copies(&params);
        let uploads: [(&Buffer<u8>, &[u8]); 7] = [
            (&self.params, bytemuck::cast_slice(&params)),
            (&self.weights, bytemuck::cast_slice(&weights)),
            (&self.weights_t, bytemuck::cast_slice(&weights_t)),
            (&self.biases, bytemuck::cast_slice(&biases)),
            (&self.ema, bytemuck::cast_slice(&params)),
            (&self.weights_ema, bytemuck::cast_slice(&weights)),
            (&self.biases_ema, bytemuck::cast_slice(&biases)),
        ];
        let total: usize = uploads.iter().map(|(_, b)| b.len()).sum();
        let mut staging: Buffer<u8> =
            rd.create_host_buffer(total as u64, vk::BufferUsageFlags::TRANSFER_SRC);
        let mut view = rd.map_buffer(&mut staging);
        let bytes = view.as_slice_mut();
        let mut at = 0;
        for (_, data) in &uploads {
            bytes[at..at + data.len()].copy_from_slice(data);
            at += data.len();
        }
        rd.run_transfer_commands(|cmd| unsafe {
            let mut at = 0u64;
            for (buffer, data) in &uploads {
                rd.device.cmd_copy_buffer(
                    cmd,
                    staging.handle,
                    buffer.handle,
                    &[vk::BufferCopy::default()
                        .src_offset(at)
                        .size(data.len() as u64)],
                );
                at += data.len() as u64;
            }
            for buffer in [&self.adam_m, &self.adam_v, &self.records, &self.counter] {
                rd.device
                    .cmd_fill_buffer(cmd, buffer.handle, 0, vk::WHOLE_SIZE, 0);
            }
        });
        rd.destroyer.destroy_buffer(staging.handle);
        self.step = 0;
    }

    pub fn set_training(&mut self, training: NrcTraining) {
        let old = self.training;
        if (
            old.learning_rate,
            old.radiance_scale,
            old.loss_scale,
            old.ema_decay,
        ) != (
            training.learning_rate,
            training.radiance_scale,
            training.loss_scale,
            training.ema_decay,
        )
        {
            self.training = training;
            self.dirty = true;
        }
    }

    pub fn gpu(&self) -> NrcNetGpu {
        NrcNetGpu {
            weights: self.weights.address,
            weights_t: self.weights_t.address,
            biases: self.biases.address,
            params: self.params.address,
            adam_m: self.adam_m.address,
            adam_v: self.adam_v.address,
            partials: self.partials.address,
            acts: self.acts.address,
            dz: self.dz.address,
            loss: self.loss.address,
            records: self.records.address,
            counter: self.counter.address,
            ema: self.ema.address,
            weights_ema: self.weights_ema.address,
            biases_ema: self.biases_ema.address,
            ring: self.ring,
            batch: self.batch,
            learning_rate: self.training.learning_rate,
            radiance_scale: self.training.radiance_scale,
            loss_scale: self.training.loss_scale,
            ema_decay: self.training.ema_decay,
        }
    }

    /// Records the parameter block's update when the training settings changed.
    pub fn record_block(&mut self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let gpu = self.gpu();
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, self.net.handle, 0, bytemuck::bytes_of(&gpu));
        }
        barrier(rd, cmd, vk::PipelineStageFlags2::TRANSFER);
    }

    fn push(&self, out: u64, offset: u32, step: u32) -> NrcPush {
        NrcPush {
            net: self.net.address,
            out,
            base: 0,
            offset,
            step,
            pad: 0,
        }
    }

    /// One training step (Adam's step `step`) on `batch` records drawn from the ring.
    pub fn record_step_at(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
        offset: u32,
        step: u32,
    ) {
        let push = self.push(0, offset, step);
        let base = Some(std::mem::offset_of!(NrcPush, base));
        record_dispatch(
            rd,
            cmd,
            module,
            "nrc_train_forward",
            &push,
            self.batch,
            base,
        );
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let groups = self.batch.div_ceil(NRC_CHUNK) * NRC_LAYERS as u32;
        record_dispatch(rd, cmd, module, "nrc_train_grads", &push, groups * 64, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
        record_dispatch(rd, cmd, module, "nrc_adam", &push, NRC_PARAMS as u32, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    /// One training step on the `batch` records from ring position `offset`: forward and
    /// backward, the weight gradients, Adam.
    pub fn record_step(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
        offset: u32,
    ) {
        self.record_gradients(rd, cmd, module, offset);
        self.record_adam(rd, cmd, module);
    }

    /// Forward, backward and the per-chunk weight gradients of the `batch` records from ring
    /// position `offset` (no update).
    pub fn record_gradients(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
        offset: u32,
    ) {
        self.record_block(rd, cmd);
        let push = self.push(0, offset, self.step);
        let base = Some(std::mem::offset_of!(NrcPush, base));
        record_dispatch(
            rd,
            cmd,
            module,
            "nrc_train_forward",
            &push,
            self.batch,
            base,
        );
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
        let groups = self.batch.div_ceil(NRC_CHUNK) * NRC_LAYERS as u32;
        record_dispatch(rd, cmd, module, "nrc_train_grads", &push, groups * 64, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    /// Adam on the gradients [`record_gradients`](Self::record_gradients) left.
    pub fn record_adam(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
    ) {
        self.step += 1;
        let push = self.push(0, 0, self.step);
        let base = Some(std::mem::offset_of!(NrcPush, base));
        record_dispatch(rd, cmd, module, "nrc_adam", &push, NRC_PARAMS as u32, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    /// The network's raw output for the `batch` records from ring position `offset`, into
    /// `out` (3 f32 each).
    pub fn record_eval(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
        offset: u32,
        out: u64,
    ) {
        self.record_block(rd, cmd);
        let push = self.push(out, offset, self.step);
        let base = Some(std::mem::offset_of!(NrcPush, base));
        record_dispatch(rd, cmd, module, "nrc_eval", &push, self.batch, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    /// The gym's synthetic records (`batch` of them at ring position 0), seeded by `seed`.
    pub fn record_gym_records(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        module: &CompiledComputeModule,
        seed: u32,
    ) {
        self.record_block(rd, cmd);
        let push = self.push(0, 0, seed);
        let base = Some(std::mem::offset_of!(NrcPush, base));
        record_dispatch(rd, cmd, module, "nrc_gym_records", &push, self.batch, base);
        barrier(rd, cmd, vk::PipelineStageFlags2::COMPUTE_SHADER);
    }

    pub fn destroy(&mut self, rd: &RenderDevice) {
        for handle in [
            self.weights.handle,
            self.weights_t.handle,
            self.biases.handle,
            self.params.handle,
            self.adam_m.handle,
            self.adam_v.handle,
            self.partials.handle,
            self.acts.handle,
            self.dz.handle,
            self.loss.handle,
            self.records.handle,
            self.counter.handle,
            self.ema.handle,
            self.weights_ema.handle,
            self.biases_ema.handle,
            self.net.handle,
        ] {
            if handle != vk::Buffer::null() {
                rd.destroyer.destroy_buffer(handle);
            }
        }
    }
}

/// Must match NrcFrame in rt_types.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NrcFrameGpu {
    pub weights: u64,
    pub biases: u64,
    pub records: u64,
    pub counter: u64,
    pub box_: [f32; 4],
    pub ring: u32,
    pub spread: f32,
    pub train_fraction: f32,
    pub flags: u32,
}

/// Training records in the ring.
const RING: u32 = 1 << 16;
/// Records per training step, and steps per frame (the paper's 4 x 16k).
const BATCH: u32 = 1 << 14;
const STEPS: u32 = 4;
/// The cache's anchor (D5): positions within RADIUS of it are encoded at full resolution, those
/// beyond contracted toward the box's walls (nrc.slang `nrcBoxPosition`). It snaps to a GRID near
/// the camera and moves once the camera is RECENTRE from it.
const RADIUS: f32 = 16.0;
const GRID: f32 = 16.0;
const RECENTRE: f32 = 12.0;
/// Vertices a training path records, about (training_fraction's cap).
const VERTICES_PER_PATH: f32 = 3.0;

#[derive(Resource)]
pub struct NrcState {
    module: Handle<ComputeModule>,
    network: Option<NrcNetwork>,
    frame: Buffer<NrcFrameGpu>,
    centre: Option<Vec3>,
    /// The Adam step this frame's training starts after (0 = no training this frame).
    first_step: u32,
}

impl NrcState {
    /// Records this frame's cache work before the trace (the parameter blocks, the record
    /// counter cleared) and returns the raygen's block (0 when the cache is off or the device has
    /// no cooperative vectors).
    pub fn prepare(
        &mut self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        settings: &RenderSettings,
        camera: Vec3,
        pixels: u32,
    ) -> u64 {
        self.first_step = 0;
        if !settings.nrc || !crate::render_device::cooperative_vector() {
            return 0;
        }
        if modules.get(&self.module).is_none() {
            return 0;
        }
        let network = self
            .network
            .get_or_insert_with(|| NrcNetwork::new(rd, BATCH, RING, 1));
        network.set_training(NrcTraining {
            learning_rate: settings.nrc_learning_rate,
            ..default()
        });
        network.record_block(rd, cmd);
        if self.frame.handle == vk::Buffer::null() {
            self.frame = rd.create_device_buffer(
                1,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            );
        }
        let snap = |p: Vec3| (p / GRID).round() * GRID;
        let centre = match self.centre {
            Some(c) if (camera - c).abs().max_element() < RECENTRE => c,
            _ => {
                let c = snap(camera);
                log::debug!("nrc: cache anchored at {c}");
                c
            }
        };
        self.centre = Some(centre);
        let train_fraction = settings
            .nrc_train_fraction
            .min(RING as f32 / (pixels.max(1) as f32 * VERTICES_PER_PATH));
        let gpu = NrcFrameGpu {
            weights: network.weights_ema.address,
            biases: network.biases_ema.address,
            records: network.records.address,
            counter: network.counter.address,
            box_: [centre.x, centre.y, centre.z, 1.0 / RADIUS],
            ring: RING,
            spread: settings.nrc_spread,
            train_fraction,
            flags: NRC_FLAG_INLINE,
        };
        unsafe {
            rd.device
                .cmd_update_buffer(cmd, self.frame.handle, 0, bytemuck::bytes_of(&gpu));
            rd.device
                .cmd_fill_buffer(cmd, network.counter.handle, 0, vk::WHOLE_SIZE, 0);
        }
        barrier(rd, cmd, vk::PipelineStageFlags2::TRANSFER);
        self.first_step = network.step + 1;
        network.step += STEPS;
        self.frame.address
    }

    /// Records this frame's training on the records the trace wrote (after it, on the same
    /// command buffer). False when there was none (the caller's pipeline is still bound).
    pub fn record_training(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
    ) -> bool {
        let (Some(network), Some(module)) = (&self.network, modules.get(&self.module)) else {
            return false;
        };
        if self.first_step == 0 {
            return false;
        }
        barrier(rd, cmd, vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR);
        for s in 0..STEPS {
            network.record_step_at(rd, cmd, module, s * BATCH, self.first_step + s);
        }
        true
    }
}

/// Must match NRC_FLAG_INLINE in rt_types.slang.
const NRC_FLAG_INLINE: u32 = 1;

fn cleanup_nrc(mut nrc: ResMut<NrcState>, rd: Res<RenderDevice>) {
    if let Some(mut network) = nrc.network.take() {
        network.destroy(&rd);
    }
    if nrc.frame.handle != vk::Buffer::null() {
        rd.destroyer.destroy_buffer(nrc.frame.handle);
    }
    nrc.frame = Buffer::default();
}

pub struct NrcPlugin;

impl Plugin for NrcPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/nrc_kernels.slang"));
        let module = asset_server.add(ComputeModule::new(
            shader,
            &[
                "nrc_train_forward",
                "nrc_train_grads",
                "nrc_adam",
                "nrc_eval",
                "nrc_gym_records",
            ],
        ));
        app.insert_resource(NrcState {
            module,
            network: None,
            frame: Buffer::default(),
            centre: None,
            first_step: 0,
        });
        app.add_systems(TeardownSchedule, cleanup_nrc.before(on_shutdown));
    }
}

fn barrier(rd: &RenderDevice, cmd: vk::CommandBuffer, src: vk::PipelineStageFlags2) {
    memory_barrier(
        rd,
        cmd,
        src,
        vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR
            | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::TRANSFER_READ,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_shaders() {
        assert_eq!(std::mem::size_of::<NrcNetGpu>(), 144);
        assert_eq!(std::mem::offset_of!(NrcNetGpu, ring), 120);
        assert_eq!(std::mem::size_of::<NrcPush>(), 32);
        assert_eq!(std::mem::size_of::<NrcFrameGpu>(), 64);
        assert_eq!(std::mem::offset_of!(NrcFrameGpu, ring), 48);
    }

    #[test]
    fn transposed_copy_is_the_transpose() {
        let params = initial_params(3);
        let (w, wt, b) = half_copies(&params);
        let n = NRC_WIDTH as usize;
        assert_eq!(w[2 * n * n + 5 * n + 9], wt[2 * n * n + 9 * n + 5]);
        assert_eq!(b.len(), (NRC_LAYERS * NRC_WIDTH) as usize);
        assert!(b.iter().all(|x| *x == f16::ZERO));
    }
}
