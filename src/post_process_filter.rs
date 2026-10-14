use ash::vk;
use ash::vk::TaggedStructure as _;
use bevy::{ecs::system::lifetimeless::SRes, prelude::*};

use crate::color_target::{DISPLAY_FORMAT, FormatPipelines, SCENE_RADIANCE_FORMAT};
use crate::descriptor_heap::{Mapping, POST_INPUTS, stage_mapping};
use crate::vulkan_asset::{VulkanAsset, VulkanAssetExt};

/// Must match `QuadPush` in quad.slang.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessPushConstants {
    pub uniforms: u64,
    pub auto_exposure: u64,
    /// 0 = follow the metering; else a fixed linear exposure re-applied over it.
    pub display_exposure: f32,
    /// [`AuroraDebugView`](crate::debug_view::AuroraDebugView) as its shader index.
    pub debug_view: u32,
    /// 1: write physical radiance (no exposure or tonemap) for a scene-radiance target.
    pub scene_radiance: u32,
    /// 1: composite the camera ray's fog (slot 8) over the resolved image.
    pub fog: u32,
    /// `RenderSettings::local_tonemap`.
    pub local_tonemap: f32,
    /// `RenderSettings::bloom`.
    pub bloom: f32,
    /// The view's auto-exposure meter buffer (render resolution, 0 = none): the pass meters what
    /// Ray Reconstruction resolved into it, over the raygen's own noisy meter.
    pub lum: u64,
    pub lum_width: u32,
    pub lum_height: u32,
    /// The Cost debug view's buffer at the meter's size (0 = none; `RaytracingPushConstants::cost`).
    pub cost: u64,
}

/// The post pass's push data: the constants, then the target's block of inputs in the resource
/// heap (`DescriptorHeaps::post_block`: one per frame parity and target, targets being one per
/// view slot plus `MAX_VIEWS` for the XR spectator window).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessPushData {
    pub constants: PostProcessPushConstants,
    pub inputs: u32,
    pub pad: u32,
}

#[derive(Asset, TypePath, Debug, Clone)]
pub struct PostProcessFilter {
    #[dependency]
    pub vertex_shader: Handle<crate::shader::Shader>,
    #[dependency]
    pub fragment_shader: Handle<crate::shader::Shader>,
}

pub struct CompiledPostProcessFilter {
    pub pipelines: FormatPipelines,
}

/// Composite targets per frame parity: one per view slot, plus the XR spectator window.
pub const POST_TARGETS: usize = crate::MAX_VIEWS + 1;

impl VulkanAsset for PostProcessFilter {
    type ExtractedAsset = (crate::shader::Shader, crate::shader::Shader);
    type ExtractParam = SRes<Assets<crate::shader::Shader>>;
    type PreparedAsset = CompiledPostProcessFilter;

    fn extract_asset(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let Some(vertex_shader) = param.get(&self.vertex_shader) else {
            log::debug!("Vertex shader not ready yet");
            return None;
        };

        let Some(fragment_shader) = param.get(&self.fragment_shader) else {
            log::debug!("Fragment shader not ready yet");
            return None;
        };

        Some((vertex_shader.clone(), fragment_shader.clone()))
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        let (vertex_shader, fragment_shader) = asset;

        // No descriptor sets: quad.slang's `test[10]` (set 0, binding 0) reads the target's block
        // of the resource heap, its index in push data after the constants.
        let heaps = render_device.heaps();
        let mappings = heaps.mappings(&[Mapping::Indexed {
            set: 0,
            binding: 0,
            count: 1,
            base: heaps.post_base(),
            block: POST_INPUTS,
            push_offset: std::mem::offset_of!(PostProcessPushData, inputs) as u32,
            combined: true,
        }]);
        let mapping_info = stage_mapping(&mappings);

        let shader_stages = [
            render_device.load_shader_entry(
                &vertex_shader.spirv.unwrap(),
                vk::ShaderStageFlags::VERTEX,
                c"vertex",
            ),
            render_device.load_shader_entry(
                &fragment_shader.spirv.unwrap(),
                vk::ShaderStageFlags::FRAGMENT,
                c"fragment",
            ),
        ];
        let shader_stages = shader_stages.map(|stage| {
            let mut stage = stage;
            stage.p_next = (&raw const mapping_info).cast();
            stage
        });

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let pipelines =
            FormatPipelines::build(&[DISPLAY_FORMAT, SCENE_RADIANCE_FORMAT], |format| {
                let formats = [format];
                let mut pipeline_rendering_info =
                    vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
                let mut flags2 = vk::PipelineCreateFlags2CreateInfo::default()
                    .flags(vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT);
                let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&shader_stages)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&dynamic_state)
                    .push(&mut pipeline_rendering_info)
                    .push(&mut flags2);
                unsafe {
                    render_device.create_graphics_pipelines(
                        render_device.pipeline_cache,
                        &[pipeline_info],
                        None,
                    )
                }
                .unwrap()[0]
            });

        unsafe {
            render_device.destroy_shader_module(shader_stages[0].module, None);
            render_device.destroy_shader_module(shader_stages[1].module, None);
        }

        CompiledPostProcessFilter { pipelines }
    }
    fn destroy_asset(
        render_device: &crate::render_device::RenderDevice,
        prepared_asset: &Self::PreparedAsset,
    ) {
        prepared_asset.pipelines.destroy(render_device);
    }
}

pub struct PostProcessFilterPlugin;

fn propagate_modified(
    filters: Res<Assets<PostProcessFilter>>,
    mut shader_events: MessageReader<AssetEvent<crate::shader::Shader>>,
    mut parent_events: MessageWriter<AssetEvent<PostProcessFilter>>,
) {
    for event in shader_events.read() {
        match event {
            AssetEvent::Modified { id } => {
                for (parent_id, filter) in filters.iter() {
                    if filter.vertex_shader.id() == *id || filter.fragment_shader.id() == *id {
                        parent_events.write(AssetEvent::Modified {
                            id: parent_id.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

impl Plugin for PostProcessFilterPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<PostProcessFilter>();
        app.init_vulkan_asset::<PostProcessFilter>();
        app.add_systems(Update, propagate_modified);
    }
}
