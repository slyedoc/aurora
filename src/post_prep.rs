//! The composite's prepass (post_prep.slang): after Ray Reconstruction and the output's pyramid,
//! per view, what quad.slang would otherwise gather per output pixel is worked out once at a
//! coarser grid that the composite reads with one lookup. `bloom_prep`: bloom's six pyramid
//! levels of 9-tap tents, at the pyramid's level 1 (a quarter of the output across). `key_prep`:
//! local tone mapping's three smoothed surroundings, at the pyramid's level 0. Both are RGBA16F
//! textures the composite filters in one fetch (post inputs 10, 11). `range_prep`: the 3x3 range
//! of the fog veil and the sky mask that clamps the composite's histories, at render resolution
//! (buffers, read point-wise).

use ash::vk;
use bevy::prelude::*;

use crate::{
    assets::aurora_asset,
    compute::{ComputeModule, ComputeModules, memory_barrier, record_dispatch},
    descriptor_heap::Mapping,
    ray_render_plugin::{TeardownSchedule, on_shutdown},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    render_texture::{RenderTexture, create_blank_texture},
};

/// The keys and bloom textures' format.
const TEXTURE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Must match `PrepPush` in post_prep.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepPush {
    bloom_width: u32,
    bloom_height: u32,
    base: u32,
    inputs: u32,
    key_width: u32,
    key_height: u32,
    range_width: u32,
    range_height: u32,
    fog_range: u64,
    sky_range: u64,
    auto_exposure: u64,
    display_exposure: f32,
    pad: u32,
}

/// A view's prepass outputs for the composite: the range buffers and their grid (0 = none), and
/// whether its keys and bloom textures exist. Must match the tail of `QuadPush` in quad.slang.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrepFrame {
    pub fog_range: u64,
    pub sky_range: u64,
    pub range_width: u32,
    pub range_height: u32,
    /// 1: post inputs 10 and 11 are this view's keys and bloom textures.
    pub textures: u32,
    pub pad: u32,
}

/// A view's keys and bloom textures (post inputs 10, 11), when made.
#[derive(Clone, Copy)]
pub struct PrepTextures {
    pub keys: vk::ImageView,
    pub bloom: vk::ImageView,
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
    /// Bloom at the pyramid's level 1 and the three keys (rgb) at its level 0; GENERAL.
    bloom: Option<RenderTexture>,
    keys: Option<RenderTexture>,
    /// The fog veil's min and max, half4 each.
    fog_range: Buffer<[u32; 4]>,
    /// The sky mask's min and max as two halves.
    sky_range: Buffer<u32>,
}

/// `texture` at `width` x `height` (made, cleared, left GENERAL), or remade at a new size.
fn sized_texture(rd: &RenderDevice, texture: &mut Option<RenderTexture>, width: u32, height: u32) {
    if texture.as_ref().is_some_and(|t| t.width == width && t.height == height) {
        return;
    }
    if let Some(old) = texture.take() {
        destroy_texture(rd, &old);
    }
    let made = create_blank_texture(
        rd,
        TEXTURE_FORMAT,
        vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::ImageLayout::GENERAL,
        width,
        height,
    );
    // Read before the first prepass writes it (the kernels still compiling): black, not garbage.
    let image = made.image;
    rd.run_transfer_commands(|cmd| unsafe {
        rd.device.cmd_clear_color_image(
            cmd,
            image,
            vk::ImageLayout::GENERAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            &[vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1)],
        );
    });
    *texture = Some(made);
}

fn destroy_texture(rd: &RenderDevice, texture: &RenderTexture) {
    rd.destroyer.destroy_image_view(texture.image_view);
    rd.destroyer.destroy_image(texture.image);
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
    /// The textures and buffers of view `slot` at its output and trace sizes.
    pub fn ensure(
        &mut self,
        rd: &RenderDevice,
        slot: usize,
        output: vk::Extent2D,
        trace: vk::Extent2D,
    ) -> (PrepFrame, PrepTextures) {
        let (kw, kh, bw, bh) = texture_sizes(output);
        let (rw, rh) = (trace.width.max(1), trace.height.max(1));
        let view = &mut self.views[slot];
        sized_texture(rd, &mut view.keys, kw, kh);
        sized_texture(rd, &mut view.bloom, bw, bh);
        let textures = PrepTextures {
            keys: view.keys.as_ref().unwrap().image_view,
            bloom: view.bloom.as_ref().unwrap().image_view,
        };
        let frame = PrepFrame {
            fog_range: sized(rd, &mut view.fog_range, rw as u64 * rh as u64),
            sky_range: sized(rd, &mut view.sky_range, rw as u64 * rh as u64),
            range_width: rw,
            range_height: rh,
            textures: 1,
            pad: 0,
        };
        (frame, textures)
    }

    /// After the pyramid of view `slot` is built: its prepass, ready for the composite's fragment
    /// reads. False (nothing recorded) until the kernels are compiled.
    pub fn record(
        &self,
        rd: &RenderDevice,
        cmd: vk::CommandBuffer,
        modules: &ComputeModules,
        frame: &PrepFrame,
        textures: &PrepTextures,
        output: vk::Extent2D,
        block: u32,
        inputs: &PrepInputs,
    ) -> bool {
        let Some(module) = modules.get(&self.module) else {
            return false;
        };
        if frame.fog_range == 0 {
            return false;
        }
        let heaps = rd.heaps();
        for (i, view) in [inputs.pyramid, inputs.fog, inputs.depth]
            .into_iter()
            .enumerate()
        {
            heaps.write_prep(block, i as u32, view);
        }
        heaps.write_prep_output(block, 0, textures.keys);
        heaps.write_prep_output(block, 1, textures.bloom);
        let (kw, kh, bw, bh) = texture_sizes(output);
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
            bloom_width: bw,
            bloom_height: bh,
            base: 0,
            inputs: block,
            key_width: kw,
            key_height: kh,
            range_width: frame.range_width,
            range_height: frame.range_height,
            fog_range: frame.fog_range,
            sky_range: frame.sky_range,
            auto_exposure: inputs.auto_exposure,
            display_exposure: inputs.display_exposure,
            pad: 0,
        };
        let base = Some(std::mem::offset_of!(PrepPush, base));
        if inputs.bloom {
            record_dispatch(rd, cmd, module, "bloom_prep", &push, bw * bh, base);
        }
        if inputs.local_tonemap && inputs.auto_exposure != 0 {
            record_dispatch(rd, cmd, module, "key_prep", &push, kw * kh, base);
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
        for texture in [view.bloom.take(), view.keys.take()].into_iter().flatten() {
            destroy_texture(&render_device, &texture);
        }
        render_device
            .destroyer
            .destroy_buffer(view.fog_range.handle);
        render_device
            .destroyer
            .destroy_buffer(view.sky_range.handle);
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
            &[
                Mapping::Prep {
                    set: 0,
                    binding: 0,
                    push_offset: std::mem::offset_of!(PrepPush, inputs) as u32,
                },
                Mapping::PrepOutputs {
                    set: 0,
                    binding: 1,
                    push_offset: std::mem::offset_of!(PrepPush, inputs) as u32,
                },
            ],
        ));
        app.insert_resource(PostPrepState {
            module,
            views: std::array::from_fn(|_| PrepView::default()),
        });
        app.add_systems(TeardownSchedule, cleanup_post_prep.before(on_shutdown));
    }
}

/// The keys texture (the pyramid's level 0: half the output) and the bloom texture (level 1).
fn texture_sizes(output: vk::Extent2D) -> (u32, u32, u32, u32) {
    // dlss/renderer.rs `Pyramid`: level 0 is half the output.
    let half = |n: u32| (n / 2).max(1);
    let (kw, kh) = (half(output.width), half(output.height));
    (kw, kh, half(kw), half(kh))
}
