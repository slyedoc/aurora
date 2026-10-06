use std::time::Instant;

use ash::vk;
use bevy::{
    app::{Plugin, Update},
    asset::{Asset, AssetApp, AssetEvent, Assets, Handle},
    ecs::{
        message::{MessageReader, MessageWriter},
        system::{Res, lifetimeless::SRes},
    },
    reflect::TypePath,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    shader::Shader,
    surface_group::{SurfaceClass, SurfaceGroupRegistry},
    vk_utils,
    vulkan_asset::{VulkanAsset, VulkanAssetExt},
};

/// The fixed shaders plus every registered surface group's, resolved to their loaded
/// [`Shader`]s. A named struct rather than a tuple because the tuple was already six long
/// before the groups.
pub struct ExtractedRaytracingPipeline {
    raygen: Shader,
    miss: Shader,
    hit: Shader,
    any_hit: Shader,
    sphere_intersection: Shader,
    sphere_hit: Shader,
    /// `(label, closest_hit, any_hit)` per registered class, in class order.
    surface_groups: Vec<(String, Shader, Option<Shader>)>,
}

#[derive(Asset, TypePath, Debug, Clone)]
pub struct RaytracingPipeline {
    #[dependency]
    pub raygen_shader: Handle<Shader>,
    #[dependency]
    pub miss_shader: Handle<Shader>,
    #[dependency]
    pub hit_shader: Handle<Shader>,
    #[dependency]
    pub any_hit_shader: Handle<Shader>,
    #[dependency]
    pub sphere_intersection_shader: Handle<Shader>,
    #[dependency]
    pub sphere_hit_shader: Handle<Shader>,
}

pub type RTGroupHandle = [u8; 32];

pub struct CompiledRaytracingPipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// One set per frame parity and view slot: `[frame_parity * MAX_VIEWS + slot]`. Each
    /// trace dispatch in a frame needs its own set — a set already recorded into the
    /// command buffer must not be rewritten.
    pub descriptor_sets: [vk::DescriptorSet; 2 * crate::MAX_VIEWS],
    pub raygen_handle: RTGroupHandle,
    pub miss_handle: RTGroupHandle,
    /// The built-in opaque triangle group: [`SurfaceClass::OPAQUE`], and
    /// `surface_handles[0]`.
    pub hit_handle: RTGroupHandle,
    pub sphere_hit_handle: RTGroupHandle,
    /// Triangle hit groups by [`SurfaceClass`]: index 0 is the built-in opaque group, the
    /// rest are the registry's in registration order. A record asking beyond the end is
    /// clamped by [`Self::surface_handle`] rather than reading past the table.
    pub surface_handles: Vec<RTGroupHandle>,
}

impl CompiledRaytracingPipeline {
    /// The hit-group handle for `class`, falling back to the opaque group. A class can
    /// outlive the pipeline that knew it if the registry grew after this pipeline was
    /// built -- shading with the wrong-but-valid group for one rebuild beats writing a
    /// handle the pipeline has never heard of into the SBT.
    pub fn surface_handle(&self, class: SurfaceClass) -> RTGroupHandle {
        self.surface_handles
            .get(class.0 as usize)
            .copied()
            .unwrap_or(self.hit_handle)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RaytracingPushConstants {
    pub uniform_buffer: u64,
    pub material_buffer: u64,
    pub bluenoise_buffer2: u64,
    pub focus_buffer: u64,
    pub sky_texture: u32,
    pub padding: [u32; 1],
    /// Last frame's instance transforms (`tlas_builder`), for motion vectors.
    pub prev_instances: u64,
    /// The emissive-triangle light table header (`lights`; 0 until built).
    pub lights: u64,
    /// This frame's instance rows, for light-sample transforms.
    pub instances: u64,
    /// ReSTIR DI reservoirs: last frame's and this frame's (`restir`).
    pub reservoirs_prev: u64,
    pub reservoirs_cur: u64,
    /// Radiance-cache entries (`sharc`; 0 until first enabled).
    pub sharc: u64,
    /// Auto-exposure: per-pixel luminance out, smoothed exposure in (`auto_exposure`).
    pub lum_buffer: u64,
    pub auto_exposure: u64,
    /// Atmosphere + cloud parameter block (`atmosphere`; 0 until that sky first renders).
    pub atmo: u64,
}

impl VulkanAsset for RaytracingPipeline {
    type ExtractedAsset = ExtractedRaytracingPipeline;
    type ExtractParam = (
        SRes<Assets<crate::shader::Shader>>,
        SRes<SurfaceGroupRegistry>,
    );
    type PreparedAsset = CompiledRaytracingPipeline;

    fn extract_asset(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        let (shaders, registry) = param;
        let shaders: &Assets<crate::shader::Shader> = &**shaders;

        let Some(raygen_shader) = shaders.get(&self.raygen_shader) else {
            log::warn!("Raygen shader not ready yet");
            return None;
        };

        let Some(miss_shader) = shaders.get(&self.miss_shader) else {
            log::warn!("Miss shader not ready yet");
            return None;
        };

        let Some(hit_shader) = shaders.get(&self.hit_shader) else {
            log::warn!("Hit shader not ready yet");
            return None;
        };

        let Some(any_hit_shader) = shaders.get(&self.any_hit_shader) else {
            log::warn!("Any-hit shader not ready yet");
            return None;
        };

        let Some(sphere_intersection_shader) = shaders.get(&self.sphere_intersection_shader) else {
            log::warn!("Sphere intersection shader not ready yet");
            return None;
        };

        let Some(sphere_hit_shader) = shaders.get(&self.sphere_hit_shader) else {
            log::warn!("Sphere hit shader not ready yet");
            return None;
        };

        // Every registered group has to be loaded before the pipeline can be built: the
        // classes are positional, so compiling a pipeline with a group missing would shift
        // every later class and silently re-point live records.
        let mut surface_groups = Vec::with_capacity(registry.groups().len());
        for group in registry.groups() {
            let Some(closest_hit) = shaders.get(&group.closest_hit) else {
                log::warn!("surface group {:?}: closest-hit not ready yet", group.label);
                return None;
            };
            let any_hit = match &group.any_hit {
                Some(handle) => match shaders.get(handle) {
                    Some(shader) => Some(shader.clone()),
                    None => {
                        log::warn!("surface group {:?}: any-hit not ready yet", group.label);
                        return None;
                    }
                },
                None => None,
            };
            surface_groups.push((group.label.clone(), closest_hit.clone(), any_hit));
        }

        Some(ExtractedRaytracingPipeline {
            raygen: raygen_shader.clone(),
            miss: miss_shader.clone(),
            hit: hit_shader.clone(),
            any_hit: any_hit_shader.clone(),
            sphere_intersection: sphere_intersection_shader.clone(),
            sphere_hit: sphere_hit_shader.clone(),
            surface_groups,
        })
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        render_device: &crate::render_device::RenderDevice,
    ) -> Self::PreparedAsset {
        let start = Instant::now();
        let ExtractedRaytracingPipeline {
            raygen: raygen_shader,
            miss: miss_shader,
            hit: hit_shader,
            any_hit: any_hit_shader,
            sphere_intersection: sphere_intersection_shader,
            sphere_hit: sphere_hit_shader,
            surface_groups,
        } = asset;

        // 0..=6: the DLSS guide images (normal+roughness, diffuse, specular, depth,
        // specular hit distance, motion, colour); 100: the TLAS.
        let storage = |binding: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
        };
        let bindings = [
            storage(0),
            storage(1),
            storage(2),
            storage(3),
            storage(4),
            storage(5),
            storage(6),
            vk::DescriptorSetLayoutBinding::default()
                .binding(100)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        ];

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout = unsafe {
            render_device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .unwrap()
        };

        let push_constant_info = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(std::mem::size_of::<RaytracingPushConstants>() as u32);

        let set_layouts = [
            descriptor_set_layout,
            render_device.bindless_descriptor_set_layout,
        ];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_info));

        let pipeline_layout = unsafe {
            render_device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let descriptor_sets = {
            let descriptor_pool = render_device.descriptor_pool.lock().unwrap();
            let layouts = [descriptor_set_layout; 2 * crate::MAX_VIEWS];
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

        // Stage indices 0..=5 are fixed and the group table below names them literally;
        // each registered surface group appends its own stages after them.
        let mut shader_stages = vec![
            render_device.load_shader(
                &raygen_shader.spirv.unwrap(),
                vk::ShaderStageFlags::RAYGEN_KHR,
            ),
            render_device.load_shader(&miss_shader.spirv.unwrap(), vk::ShaderStageFlags::MISS_KHR),
            render_device.load_shader(
                &hit_shader.spirv.unwrap(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ),
            render_device.load_shader(
                &sphere_intersection_shader.spirv.unwrap(),
                vk::ShaderStageFlags::INTERSECTION_KHR,
            ),
            render_device.load_shader(
                &sphere_hit_shader.spirv.unwrap(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ),
            render_device.load_shader(
                &any_hit_shader.spirv.unwrap(),
                vk::ShaderStageFlags::ANY_HIT_KHR,
            ),
        ];

        let shader_group = [
            // Raygen shader
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Miss shader
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Triangle hit shader (any-hit does the alpha-mask cutout)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(5)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Sphere shader
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(4)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(3),
        ];
        // Group indices: 0 raygen, 1 miss, 2 opaque triangles, 3 spheres, then one per
        // registered surface class. `SurfaceClass(n)` is group `OPAQUE_GROUP + n`, which is
        // why the handle table below starts at the opaque group and runs to the end.
        const OPAQUE_GROUP: usize = 2;
        let mut shader_group = shader_group.to_vec();
        for (index, (label, closest_hit, any_hit)) in surface_groups.iter().enumerate() {
            let closest_hit_index = shader_stages.len() as u32;
            shader_stages.push(render_device.load_shader(
                closest_hit.spirv.as_ref().unwrap(),
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ));
            let any_hit_index = match any_hit {
                Some(any_hit) => {
                    let index = shader_stages.len() as u32;
                    shader_stages.push(render_device.load_shader(
                        any_hit.spirv.as_ref().unwrap(),
                        vk::ShaderStageFlags::ANY_HIT_KHR,
                    ));
                    index
                }
                None => vk::SHADER_UNUSED_KHR,
            };
            shader_group.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(closest_hit_index)
                    .any_hit_shader(any_hit_index)
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
            );
            // Class from the registry's own order, not from the group index: the groups
            // array has raygen, miss, opaque and the sphere group ahead of these, so
            // deriving it from the index is an off-by-one waiting to happen.
            log::debug!("surface group {label:?} -> class {}", index + 1);
        }

        // Pipelines that trace structures referencing opacity micromaps must say so.
        let flags = if render_device.ext_micromap.is_some() {
            vk::PipelineCreateFlags::RAY_TRACING_OPACITY_MICROMAP_EXT
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(flags)
            .stages(&shader_stages)
            .groups(&shader_group)
            .max_pipeline_ray_recursion_depth(1)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            render_device
                .ext_rtx_pipeline
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            for shader in shader_stages {
                render_device.destroy_shader_module(shader.module, None);
            }
        }

        let rtprops = vk_utils::get_raytracing_properties(&render_device);
        let handle_size = rtprops.shader_group_handle_size;
        assert!(
            handle_size as usize == std::mem::size_of::<RTGroupHandle>(),
            "at the time we only support 128-bit handles (at time of writing all devices have this)"
        );

        let handle_count = shader_group.len() as u32;
        let handle_data_size = handle_count * handle_size;
        let handles: Vec<RTGroupHandle> = unsafe {
            render_device
                .ext_rtx_pipeline
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    handle_count,
                    handle_data_size as usize,
                )
                .unwrap()
                .chunks(handle_size as usize)
                .map(|chunk| {
                    let mut handle = RTGroupHandle::default();
                    handle.copy_from_slice(chunk);
                    handle
                })
                .collect()
        };

        let raygen_handle = handles[0];
        let miss_handle = handles[1];
        let hit_handle = handles[OPAQUE_GROUP];
        let sphere_hit_handle = handles[3];
        // Class order: the opaque group, then the registered ones. `surface_handles[0]`
        // and `hit_handle` are deliberately the same handle.
        let surface_handles: Vec<RTGroupHandle> = std::iter::once(hit_handle)
            .chain(handles[OPAQUE_GROUP + 2..].iter().copied())
            .collect();

        log::debug!("Raytracing pipeline compiled in {:?}", start.elapsed());

        CompiledRaytracingPipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_sets,
            raygen_handle,
            miss_handle,
            hit_handle,
            sphere_hit_handle,
            surface_handles,
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

fn propagate_modified(
    filters: Res<Assets<RaytracingPipeline>>,
    mut shader_events: MessageReader<AssetEvent<Shader>>,
    mut parent_events: MessageWriter<AssetEvent<RaytracingPipeline>>,
    registry: Res<SurfaceGroupRegistry>,
    mut seen_generation: bevy::ecs::system::Local<Option<u64>>,
) {
    // A group registered after the pipeline was built would otherwise never reach it: the
    // pipeline is an asset that only rebuilds on an event, and registration touches no
    // asset. Rebuild once per generation instead of silently rendering without the group.
    if *seen_generation != Some(registry.generation()) {
        if seen_generation.is_some() {
            for (parent_id, _) in filters.iter() {
                parent_events.write(AssetEvent::Modified {
                    id: parent_id.clone(),
                });
            }
        }
        *seen_generation = Some(registry.generation());
    }
    for event in shader_events.read() {
        match event {
            AssetEvent::Modified { id } => {
                for (parent_id, filter) in filters.iter() {
                    let in_surface_group = registry.groups().iter().any(|group| {
                        group.closest_hit.id() == *id
                            || group.any_hit.as_ref().is_some_and(|h| h.id() == *id)
                    });
                    if in_surface_group
                        || filter.raygen_shader.id() == *id
                        || filter.miss_shader.id() == *id
                        || filter.hit_shader.id() == *id
                        || filter.any_hit_shader.id() == *id
                        || filter.sphere_intersection_shader.id() == *id
                        || filter.sphere_hit_shader.id() == *id
                    {
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

pub struct RaytracingPipelinePlugin;

impl Plugin for RaytracingPipelinePlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.init_asset::<RaytracingPipeline>();
        app.init_vulkan_asset::<RaytracingPipeline>();
        app.add_systems(Update, propagate_modified);
    }
}
