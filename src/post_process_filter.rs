use ash::vk;
use bevy::{ecs::system::lifetimeless::SRes, prelude::*};

use crate::vulkan_asset::{VulkanAsset, VulkanAssetExt};

/// Must match `Registers` in quad.frag.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessPushConstants {
    pub uniforms: u64,
    pub auto_exposure: u64,
    /// 0 = follow the metering; else a fixed linear exposure re-applied over it.
    pub display_exposure: f32,
    /// [`AuroraDebugView`](crate::debug_view::AuroraDebugView) as its shader index.
    pub debug_view: u32,
}

#[derive(Asset, TypePath, Debug, Clone)]
pub struct PostProcessFilter {
    #[dependency]
    pub vertex_shader: Handle<crate::shader::Shader>,
    #[dependency]
    pub fragment_shader: Handle<crate::shader::Shader>,
}

pub struct CompiledPostProcessFilter {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: [vk::DescriptorSet; 2],
}

impl VulkanAsset for PostProcessFilter {
    type ExtractedAsset = (crate::shader::Shader, crate::shader::Shader);
    type ExtractParam = SRes<Assets<crate::shader::Shader>>;
    type PreparedAsset = CompiledPostProcessFilter;

    fn extract_asset(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let Some(vertex_shader) = param.get(&self.vertex_shader) else {
            log::warn!("Vertex shader not ready yet");
            return None;
        };

        let Some(fragment_shader) = param.get(&self.fragment_shader) else {
            log::warn!("Fragment shader not ready yet");
            return None;
        };

        Some((vertex_shader.clone(), fragment_shader.clone()))
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        let (vertex_shader, fragment_shader) = asset;

        // 8 sampled images: the DLSS output plus the guide buffers for the debug views
        // (quad.frag's `test[8]`).
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .binding(0)
            .descriptor_count(8)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

        let descriptor_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            render_device
                .create_descriptor_set_layout(&descriptor_layout_info, None)
                .unwrap()
        };

        let push_constant_info = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(std::mem::size_of::<PostProcessPushConstants>() as u32);

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_info));
        let pipeline_layout = unsafe {
            render_device
                .create_pipeline_layout(&layout_info, None)
                .unwrap()
        };

        let descriptor_sets = {
            let descriptor_pool = render_device.descriptor_pool.lock().unwrap();
            let layouts = [descriptor_set_layout; 2];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(*descriptor_pool)
                .set_layouts(&layouts);
            unsafe {
                render_device
                    .allocate_descriptor_sets(&alloc_info)
                    .unwrap()
                    .try_into()
                    .unwrap()
            }
        };

        let shader_stages = [
            render_device.load_shader(&vertex_shader.spirv.unwrap(), vk::ShaderStageFlags::VERTEX),
            render_device.load_shader(
                &fragment_shader.spirv.unwrap(),
                vk::ShaderStageFlags::FRAGMENT,
            ),
        ];

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

        let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&[vk::Format::B8G8R8A8_UNORM]);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut pipeline_rendering_info);

        let pipeline = unsafe {
            render_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        }
        .unwrap()[0];

        unsafe {
            render_device.destroy_shader_module(shader_stages[0].module, None);
            render_device.destroy_shader_module(shader_stages[1].module, None);
        }

        CompiledPostProcessFilter {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_sets,
        }
    }
    fn destroy_asset(
        render_device: &crate::render_device::RenderDevice,
        prepared_asset: &Self::PreparedAsset,
    ) {
        render_device
            .destroyer
            .destroy_descriptor_set_layout(prepared_asset.descriptor_set_layout);
        render_device
            .destroyer
            .destroy_pipeline_layout(prepared_asset.pipeline_layout);
        render_device
            .destroyer
            .destroy_pipeline(prepared_asset.pipeline);
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
