//! The composite's prepass (post_prep.slang): after Ray Reconstruction and the output's pyramid,
//! per view, what quad.slang would otherwise gather per output pixel is worked out once at a
//! coarser grid into buffers the composite reads with one lookup. `bloom_prep`: bloom's six
//! pyramid levels of 9-tap tents, at the pyramid's level 1 (a quarter of the output across).
//! `key_prep`: local tone mapping's three smoothed surroundings, at the pyramid's level 0.
//! `range_prep`: the 3x3 range of the fog veil and the sky mask that clamps the composite's
//! histories, at render resolution.

use ash::vk;
use bevy::prelude::*;

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    descriptor_heap::Mapping,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};

/// Must match `PrepPush` in post_prep.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepPush {
    bloom: u64,
    bloom_width: u32,
    bloom_height: u32,
    base: u32,
    inputs: u32,
    keys: u64,
    key_width: u32,
    key_height: u32,
    fog_range: u64,
    sky_range: u64,
    range_width: u32,
    range_height: u32,
    auto_exposure: u64,
    display_exposure: f32,
    pad: u32,
}

/// A view's prepass outputs for the composite: buffer addresses and their grids (0 = none).
/// Must match the tail of `QuadPush` in quad.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrepFrame {
    pub bloom: u64,
    pub bloom_width: u32,
    pub bloom_height: u32,
    pub keys: u64,
    pub key_width: u32,
    pub key_height: u32,
    pub fog_range: u64,
    pub sky_range: u64,
    pub range_width: u32,
    pub range_height: u32,
}

/// What the composite will apply this frame (RenderSettings), so the prepass skips the rest.
pub struct PrepInputs {
    pub pyramid: vk::ImageView,
    pub fog: vk::ImageView,
    pub depth: vk::ImageView,
    pub auto_exposure: u64,
    pub display_exposure: f32,
    pub bloom: bool,
    pub local_tonemap: bool,
}

#[derive(Default)]
struct PrepView {
    /// Half4 per texel (packed in two u32).
    bloom: Buffer<[u32; 2]>,
    /// The three keys as half floats (packed in two u32).
    keys: Buffer<[u32; 2]>,
    /// The fog veil's min and max, half4 each.
    fog_range: Buffer<[u32; 4]>,
    /// The sky mask's min and max as two halves.
    sky_range: Buffer<u32>,
}

fn sized<T>(rd: &RenderDevice, buffer: &mut Buffer<T>, len: u64) -> u64 {
    if buffer.nr_elements != len {
        rd.destroyer.destroy_buffer(buffer.handle);
        *buffer = rd.create_device_buffer(len, vk::BufferUsageFlags::STORAGE_BUFFER);
    }
    buffer.address
}

#[derive(Resource)]
pub struct PostPrepState {
    module: Handle<ComputeModule>,
    views: [PrepView; crate::MAX_VIEWS],
}

impl PostPrepState {
    /// The buffers of view `slot` at its output and trace sizes.
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        slot: usize,
        output: vk::Extent2D,
        trace: vk::Extent2D,
    ) -> PrepFrame {
        // dlss/renderer.rs `Pyramid`: level 0 is half the output.
        let half = |n: u32| (n / 2).max(1);
        let (kw, kh) = (half(output.width), half(output.height));
        let (bw, bh) = (half(kw), half(kh));
        let (rw, rh) = (trace.width.max(1), trace.height.max(1));
        let view = &mut self.views[slot];
        PrepFrame {
            bloom: sized(rd, &mut view.bloom, bw as u64 * bh as u64),
            bloom_width: bw,
            bloom_height: bh,
            keys: sized(rd, &mut view.keys, kw as u64 * kh as u64),
            key_width: kw,
            key_height: kh,
            fog_range: sized(rd, &mut view.fog_range, rw as u64 * rh as u64),
            sky_range: sized(rd, &mut view.sky_range, rw as u64 * rh as u64),
            range_width: rw,
            range_height: rh,
        }
    }

    /// After the pyramid of view `slot` is built: its prepass, ready for the composite's fragment
    /// reads. False (nothing recorded) until the kernels are compiled.
    pub fn record(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        frame: &PrepFrame,
        block: u32,
        inputs: &PrepInputs,
    ) -> bool {
        let Some(module) = modules.get(&self.module) else {
            return false;
        };
        if frame.bloom == 0 {
            return false;
        }
        let heaps = rd.heaps();
        for (i, view) in [inputs.pyramid, inputs.fog, inputs.depth].into_iter().enumerate() {
            heaps.write_prep(block, i as u32, view);
        }
        // The pyramid's blits before the reads; last frame's composite reads before the writes.
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        let push = PrepPush {
            bloom: frame.bloom,
            bloom_width: frame.bloom_width,
            bloom_height: frame.bloom_height,
            base: 0,
            inputs: block,
            keys: frame.keys,
            key_width: frame.key_width,
            key_height: frame.key_height,
            fog_range: frame.fog_range,
            sky_range: frame.sky_range,
            range_width: frame.range_width,
            range_height: frame.range_height,
            auto_exposure: inputs.auto_exposure,
            display_exposure: inputs.display_exposure,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(PrepPush, base));
        if inputs.bloom {
            let texels = frame.bloom_width * frame.bloom_height;
            record_dispatch(rd, cmd, module, "bloom_prep", &push, texels, base);
        }
        if inputs.local_tonemap && inputs.auto_exposure != 0 {
            let texels = frame.key_width * frame.key_height;
            record_dispatch(rd, cmd, module, "key_prep", &push, texels, base);
        }
        let texels = frame.range_width * frame.range_height;
        record_dispatch(rd, cmd, module, "range_prep", &push, texels, base);
        memory_barrier(
            rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        true
    }
}

fn cleanup_post_prep(mut state: ResMut<PostPrepState>, render_device: Res<RenderDevice>) {
    for view in &mut state.views {
        render_device.destroyer.destroy_buffer(view.bloom.handle);
        render_device.destroyer.destroy_buffer(view.keys.handle);
        render_device.destroyer.destroy_buffer(view.fog_range.handle);
        render_device.destroyer.destroy_buffer(view.sky_range.handle);
        *view = PrepView::default();
    }
}

pub struct PostPrepPlugin;

impl Plugin for PostPrepPlugin {
    fn build(&self, app: &mut App) {
        let asset_server = app.world().resource::<AssetServer>();
        let shader = asset_server.load(aurora_asset("shaders/post_prep.slang"));
        let module = asset_server.add(ComputeModule::with_heap(
            shader,
            &["bloom_prep", "key_prep", "range_prep"],
            &[Mapping::Prep {
                set: 0,
                binding: 0,
                push_offset: std::mem::offset_of!(PrepPush, inputs) as u32,
            }],
        ));
        app.insert_resource(PostPrepState {
            module,
            views: std::array::from_fn(|_| PrepView::default()),
        });
        app.add_systems(TeardownSchedule, cleanup_post_prep.before(on_shutdown));
    }
}
