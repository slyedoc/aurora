//! `VK_EXT_descriptor_heap`: one resource heap and one sampler heap, host-visible buffers the
//! descriptors are written straight into -- no descriptor sets, layouts or pools. Shaders keep
//! their `[[vk::binding]]` declarations; each pipeline maps them onto heap offsets when it is
//! created ([`Mapping`]), and per-draw indices and the TLAS ride in push data.
//!
//! Resource heap layout, in descriptor slots of `stride` bytes after the driver's reserved
//! range: the bindless textures, then per frame parity the views' trace guides, the post
//! targets' inputs and the gizmo depth. Sampler heap: the reserved range, then the one linear
//! sampler every combined image sampler uses.

use std::collections::HashMap;
use std::sync::Mutex;

use ash::vk;
use ash::vk::TaggedStructure as _;

use crate::post_process_filter::POST_TARGETS;
use crate::render_buffer::{Buffer, BufferProvider};
use crate::render_device::RenderDevice;

/// Bindless texture slots (binding 200).
pub const BINDLESS_SLOTS: u32 = 16536;
/// Trace guides per view: normal+roughness, diffuse, specular, depth, spec hit, motion,
/// colour, fog (raygen bindings 0..=7).
pub const GUIDES: u32 = 8;
/// Post-process inputs per target (quad.slang's `test[10]`).
pub const POST_INPUTS: u32 = 10;

/// What a heap-using pipeline reads: [`DescriptorHeaps::mappings`] turns these into the
/// create-time mapping of the shader's `(set, binding)` declarations.
#[derive(Clone, Copy)]
pub enum Mapping {
    /// The bindless texture array at `(set, binding)`: a constant heap range.
    Bindless { set: u32, binding: u32 },
    /// `count` consecutive bindings from `binding`, at the heap block whose index is the u32 at
    /// `push_offset` in push data (blocks of `block` slots from `base`).
    Indexed {
        set: u32,
        binding: u32,
        count: u32,
        base: u32,
        block: u32,
        push_offset: u32,
        combined: bool,
    },
    /// An acceleration structure whose device address is the u64 at `push_offset`.
    Address {
        set: u32,
        binding: u32,
        push_offset: u32,
    },
}

/// The pieces of an image view a descriptor is written from (the heap takes the view's create
/// info, not the view).
#[derive(Clone, Copy)]
pub struct ViewDesc {
    pub image: vk::Image,
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub components: vk::ComponentMapping,
    pub range: vk::ImageSubresourceRange,
}

impl ViewDesc {
    pub fn of(info: &vk::ImageViewCreateInfo) -> Self {
        Self {
            image: info.image,
            view_type: info.view_type,
            format: info.format,
            components: info.components,
            range: info.subresource_range,
        }
    }

    fn info(&self) -> vk::ImageViewCreateInfo<'static> {
        vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(self.view_type)
            .format(self.format)
            .components(self.components)
            .subresource_range(self.range)
    }
}

struct Mapped(*mut u8);
// SAFETY: the pointer is a persistent mapping of a heap buffer that lives as long as the
// device; writers touch disjoint descriptor slots.
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

pub struct DescriptorHeaps {
    pub ext: ash::ext::descriptor_heap::Device,
    resource: Buffer<u8>,
    sampler: Buffer<u8>,
    resource_ptr: Mapped,
    sampler_ptr: Mapped,
    resource_reserved: vk::DeviceSize,
    sampler_reserved: vk::DeviceSize,
    /// Bytes per resource descriptor slot.
    pub stride: u32,
    /// Byte offset of the linear sampler in the sampler heap.
    linear_sampler: u32,
    /// A descriptor set [`Self::leave`] binds (layout, pipeline layout, pool, set).
    legacy: (
        vk::DescriptorSetLayout,
        vk::PipelineLayout,
        vk::DescriptorPool,
        vk::DescriptorSet,
    ),
    bindless: u32,
    guides: u32,
    post: u32,
    gizmo: u32,
    /// Every image view the device made, by handle: descriptors are written from these.
    views: Mutex<HashMap<vk::ImageView, ViewDesc>>,
}

fn align(value: u64, to: u64) -> u64 {
    value.div_ceil(to.max(1)) * to.max(1)
}

impl DescriptorHeaps {
    pub fn new(rd: &RenderDevice) -> Self {
        let mut props = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push(&mut props);
        unsafe {
            rd.instance
                .get_physical_device_properties2(rd.physical_device, &mut properties)
        };
        let stride = align(
            props.image_descriptor_size.max(props.buffer_descriptor_size),
            props
                .image_descriptor_alignment
                .max(props.buffer_descriptor_alignment),
        );
        let resource_reserved = align(props.min_resource_heap_reserved_range, stride);
        let slot = |n: u64| resource_reserved + n * stride;
        let bindless = slot(0);
        let guides = slot(BINDLESS_SLOTS as u64);
        let post = guides + 2 * crate::MAX_VIEWS as u64 * GUIDES as u64 * stride;
        let gizmo = post + 2 * POST_TARGETS as u64 * POST_INPUTS as u64 * stride;
        let resource_size = align(
            gizmo + 2 * crate::MAX_VIEWS as u64 * stride,
            props.resource_heap_alignment,
        );
        let sampler_stride = align(props.sampler_descriptor_size, props.sampler_descriptor_alignment);
        let sampler_reserved =
            align(props.min_sampler_heap_reserved_range, props.sampler_descriptor_alignment);
        let sampler_size = align(sampler_reserved + sampler_stride, props.sampler_heap_alignment);
        log::info!(
            "vulkan: descriptor heaps: resource {} KB ({stride} B slots, {} B reserved), sampler {} B, push data {} B",
            resource_size / 1024,
            resource_reserved,
            sampler_size,
            props.max_push_data_size,
        );

        let usage = vk::BufferUsageFlags::DESCRIPTOR_HEAP_EXT;
        let mut resource: Buffer<u8> = rd.create_host_buffer(resource_size, usage);
        let mut sampler: Buffer<u8> = rd.create_host_buffer(sampler_size, usage);
        let resource_ptr = Mapped(rd.map_buffer(&mut resource).as_ptr_mut());
        let sampler_ptr = Mapped(rd.map_buffer(&mut sampler).as_ptr_mut());
        let ext = ash::ext::descriptor_heap::Device::load(&rd.instance, &rd.device);
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let set_layout = unsafe {
            rd.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )
        }
        .unwrap();
        let legacy_layout = unsafe {
            rd.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&set_layout)),
                None,
            )
        }
        .unwrap();
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool = unsafe {
            rd.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(std::slice::from_ref(&pool_size)),
                None,
            )
        }
        .unwrap();
        let legacy_set = unsafe {
            rd.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(std::slice::from_ref(&set_layout)),
            )
        }
        .unwrap()[0];

        let heaps = Self {
            ext,
            resource,
            sampler,
            resource_ptr,
            sampler_ptr,
            resource_reserved,
            sampler_reserved,
            stride: stride as u32,
            linear_sampler: sampler_reserved as u32,
            legacy: (set_layout, legacy_layout, pool, legacy_set),
            bindless: bindless as u32,
            guides: guides as u32,
            post: post as u32,
            gizmo: gizmo as u32,
            views: Mutex::new(HashMap::new()),
        };
        let linear = linear_sampler_info();
        unsafe {
            heaps
                .ext
                .write_sampler_descriptors(
                    std::slice::from_ref(&linear),
                    &[vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(
                        heaps.sampler_ptr.0.add(heaps.linear_sampler as usize),
                        sampler_stride as usize,
                    ))],
                )
                .unwrap();
        }
        heaps
    }

    /// The heap buffers, freed with the device.
    pub fn buffers(&self) -> [vk::Buffer; 2] {
        [self.resource.handle, self.sampler.handle]
    }

    /// Notes how `view` was made, for the descriptors written from it.
    pub fn track_view(&self, view: vk::ImageView, info: &vk::ImageViewCreateInfo) {
        self.views.lock().unwrap().insert(view, ViewDesc::of(info));
    }

    /// Binds both heaps on `cmd`. Heap state is cleared by any descriptor-set or push-constant
    /// command (a compute dispatch, NGX), so heap users bind before they draw or trace.
    pub fn bind(&self, cmd: vk::CommandBuffer) {
        let resource = vk::BindHeapInfoEXT::default()
            .heap_range(
                vk::DeviceAddressRangeEXT::default()
                    .address(self.resource.address)
                    .size(self.resource.nr_elements),
            )
            .reserved_range_offset(0)
            .reserved_range_size(self.resource_reserved);
        let sampler = vk::BindHeapInfoEXT::default()
            .heap_range(
                vk::DeviceAddressRangeEXT::default()
                    .address(self.sampler.address)
                    .size(self.sampler.nr_elements),
            )
            .reserved_range_offset(0)
            .reserved_range_size(self.sampler_reserved);
        unsafe {
            self.ext.cmd_bind_resource_heap(cmd, &resource);
            self.ext.cmd_bind_sampler_heap(cmd, &sampler);
        }
    }

    /// Before NGX: its NVX image-view handles only resolve once a descriptor set is bound again
    /// (with our heap current, RR resolved a blur; push constants alone are not enough).
    pub fn leave(&self, rd: &RenderDevice, cmd: vk::CommandBuffer) {
        let (_, layout, _, set) = self.legacy;
        unsafe {
            rd.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&set),
                &[],
            );
        }
    }

    /// The Vulkan objects behind [`Self::leave`], destroyed with the device.
    pub unsafe fn destroy_objects(&self, device: &ash::Device) {
        let (set_layout, layout, pool, _) = self.legacy;
        unsafe {
            device.destroy_descriptor_pool(pool, None);
            device.destroy_pipeline_layout(layout, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        }
    }

    /// Pushes `data` at offset 0 of push data (where the shaders' push-constant block sits).
    pub fn push<T: bytemuck::NoUninit>(&self, cmd: vk::CommandBuffer, data: &T) {
        let bytes = bytemuck::bytes_of(data);
        let info = vk::PushDataInfoEXT::default()
            .offset(0)
            .data(vk::HostAddressRangeConstEXT::default().address(bytes));
        unsafe { self.ext.cmd_push_data(cmd, &info) };
    }

    /// Writes an image descriptor for `view` at `offset` bytes into the resource heap. A view
    /// the device did not make (or a null one) leaves the slot as it was.
    pub fn write_image(
        &self,
        offset: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
        ty: vk::DescriptorType,
    ) {
        let Some(desc) = self.views.lock().unwrap().get(&view).copied() else {
            log::warn!("descriptor heap: {view:?} was not made by the device; slot {offset} left as is");
            return;
        };
        let view_info = desc.info();
        let image = vk::ImageDescriptorInfoEXT::default()
            .view(&view_info)
            .layout(layout);
        let resource = vk::ResourceDescriptorInfoEXT::default()
            .ty(ty)
            .data(vk::ResourceDescriptorDataEXT { p_image: &image });
        unsafe {
            self.ext
                .write_resource_descriptors(
                    std::slice::from_ref(&resource),
                    &[vk::HostAddressRangeEXT::default().address(std::slice::from_raw_parts_mut(
                        self.resource_ptr.0.add(offset as usize),
                        self.stride as usize,
                    ))],
                )
                .unwrap();
        }
    }

    /// Bindless slot `index` samples `view`.
    pub fn write_bindless(&self, index: u32, view: vk::ImageView) {
        self.write_image(
            self.bindless + index * self.stride,
            view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::SAMPLED_IMAGE,
        );
    }

    /// The block of guide slots for `(parity, view slot)`: its index in push data.
    pub fn guide_block(parity: usize, slot: usize) -> u32 {
        (parity * crate::MAX_VIEWS + slot) as u32
    }

    /// Writes the trace guides (storage images, GENERAL) of `block`.
    pub fn write_guides(&self, block: u32, views: &[vk::ImageView; GUIDES as usize]) {
        let base = self.guides + block * GUIDES * self.stride;
        for (i, view) in views.iter().enumerate() {
            self.write_image(
                base + i as u32 * self.stride,
                *view,
                vk::ImageLayout::GENERAL,
                vk::DescriptorType::STORAGE_IMAGE,
            );
        }
    }

    /// The block of post inputs for `(parity, target)`.
    pub fn post_block(parity: usize, target: usize) -> u32 {
        (parity * POST_TARGETS + target) as u32
    }

    /// Writes post input `i` of `block`.
    pub fn write_post(&self, block: u32, i: u32, view: vk::ImageView, layout: vk::ImageLayout) {
        self.write_image(
            self.post + (block * POST_INPUTS + i) * self.stride,
            view,
            layout,
            vk::DescriptorType::SAMPLED_IMAGE,
        );
    }

    /// The gizmo depth slot for `(parity, view slot)`.
    pub fn gizmo_slot(parity: usize, slot: usize) -> u32 {
        (parity * crate::MAX_VIEWS + slot) as u32
    }

    pub fn write_gizmo_depth(&self, index: u32, view: vk::ImageView) {
        self.write_image(
            self.gizmo + index * self.stride,
            view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::SAMPLED_IMAGE,
        );
    }

    /// The heap offsets [`Mapping::Indexed`] blocks start from.
    pub fn guides_base(&self) -> u32 {
        self.guides
    }
    pub fn post_base(&self) -> u32 {
        self.post
    }
    pub fn gizmo_base(&self) -> u32 {
        self.gizmo
    }

    /// The create-time mapping for a pipeline reading `mappings`.
    pub fn mappings(&self, mappings: &[Mapping]) -> Vec<vk::DescriptorSetAndBindingMappingEXT<'static>> {
        mappings
            .iter()
            .map(|m| match *m {
                Mapping::Bindless { set, binding } => vk::DescriptorSetAndBindingMappingEXT::default()
                    .descriptor_set(set)
                    .first_binding(binding)
                    .binding_count(1)
                    .resource_mask(vk::SpirvResourceTypeFlagsEXT::ALL)
                    .source(vk::DescriptorMappingSourceEXT::HEAP_WITH_CONSTANT_OFFSET)
                    .source_data(vk::DescriptorMappingSourceDataEXT {
                        constant_offset: vk::DescriptorMappingSourceConstantOffsetEXT::default()
                            .heap_offset(self.bindless)
                            .heap_array_stride(self.stride)
                            .sampler_heap_offset(self.linear_sampler)
                            .sampler_heap_array_stride(0),
                    }),
                Mapping::Indexed {
                    set,
                    binding,
                    count,
                    base,
                    block,
                    push_offset,
                    combined,
                } => {
                    let mut push = vk::DescriptorMappingSourcePushIndexEXT::default()
                        .heap_offset(base)
                        .push_offset(push_offset)
                        .heap_index_stride(block * self.stride)
                        .heap_array_stride(self.stride);
                    if combined {
                        push = push
                            .sampler_heap_offset(self.linear_sampler)
                            .sampler_push_offset(push_offset)
                            .sampler_heap_index_stride(0)
                            .sampler_heap_array_stride(0);
                    }
                    vk::DescriptorSetAndBindingMappingEXT::default()
                        .descriptor_set(set)
                        .first_binding(binding)
                        .binding_count(count)
                        .resource_mask(vk::SpirvResourceTypeFlagsEXT::ALL)
                        .source(vk::DescriptorMappingSourceEXT::HEAP_WITH_PUSH_INDEX)
                        .source_data(vk::DescriptorMappingSourceDataEXT { push_index: push })
                }
                Mapping::Address {
                    set,
                    binding,
                    push_offset,
                } => vk::DescriptorSetAndBindingMappingEXT::default()
                    .descriptor_set(set)
                    .first_binding(binding)
                    .binding_count(1)
                    .resource_mask(vk::SpirvResourceTypeFlagsEXT::ALL)
                    .source(vk::DescriptorMappingSourceEXT::PUSH_ADDRESS)
                    .source_data(vk::DescriptorMappingSourceDataEXT {
                        push_address_offset: push_offset,
                    }),
            })
            .collect()
    }
}

fn linear_sampler_info() -> vk::SamplerCreateInfo<'static> {
    vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        // The default max_lod of 0 would pin every read to level 0.
        .max_lod(vk::LOD_CLAMP_NONE)
}

/// Chains the shader-stage mapping and the heap flag for a pipeline: keep `mapping` and `flags`
/// alive until the pipeline is created.
pub fn stage_mapping<'a>(
    mappings: &'a [vk::DescriptorSetAndBindingMappingEXT<'a>],
) -> vk::ShaderDescriptorSetAndBindingMappingInfoEXT<'a> {
    vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(mappings)
}
