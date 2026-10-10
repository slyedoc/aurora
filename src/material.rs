//! The ray tracer's material: the data is [`aurora_material`], re-exported here; this module
//! is the GPU side.
//!
//! An entity without an [`AuroraMaterial3d`] traces as [`AuroraMaterial::default`]. On the GPU
//! the asset becomes an [`RTXMaterial`] record plus the image ids its texture slots come from; the
//! slots are resolved to bindless indices when instances are prepared, so textures apply
//! whenever they finish loading.

pub use aurora_material::{
    AlphaMode, AuroraMaterial, AuroraMaterial3d, AuroraMaterialTypesPlugin, Face,
    ParallaxMappingMethod,
};
use bevy::prelude::*;

use crate::{
    blas::{RTXMaterial, absorption_from_attenuation},
    render_device::RenderDevice,
    render_env::{DEFAULT_NORMAL_TEXTURE_IDX, WHITE_TEXTURE_IDX},
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

// ---- GPU side -------------------------------------------------------------------------------

impl RTXMaterial {
    pub fn from_material(material: &AuroraMaterial) -> Self {
        RTXMaterial {
            base_color_factor: {
                // Linear, like the glTF loader's factors; the hit shaders use it as albedo.
                let c = material.base_color.to_linear();
                [c.red, c.green, c.blue, c.alpha]
            },
            base_emissive_factor: {
                let c = material.emissive;
                [c.red, c.green, c.blue, c.alpha]
            },
            base_color_texture: WHITE_TEXTURE_IDX,
            base_emissive_texture: WHITE_TEXTURE_IDX,
            normal_texture: DEFAULT_NORMAL_TEXTURE_IDX,
            specular_transmission_texture: WHITE_TEXTURE_IDX,
            metallic_roughness_texture: WHITE_TEXTURE_IDX,
            specular_transmission_factor: material.specular_transmission,
            roughness_factor: material.perceptual_roughness,
            metallic_factor: material.metallic,
            refract_index: material.ior,
            absorption: absorption_from_attenuation(
                material.attenuation_color.to_linear(),
                material.attenuation_distance,
            ),
            alpha_cutoff: match material.alpha_mode {
                AlphaMode::Opaque => 0.0,
                AlphaMode::Mask(cutoff) => cutoff.max(1.0 / 255.0),
                // Blend traces as a 0.5 cutout for shadow rays; the raygen's stochastic
                // alpha still handles the camera-visible transparency. The raster blend
                // modes have no tracing analogue and take the same path.
                AlphaMode::Blend
                | AlphaMode::Premultiplied
                | AlphaMode::AlphaToCoverage
                | AlphaMode::Add
                | AlphaMode::Multiply => 0.5,
            },
            surface_param_index: 0,
            diffuse_transmission: material.diffuse_transmission.clamp(0.0, 1.0),
            clearcoat: material.clearcoat.clamp(0.0, 1.0),
            clearcoat_roughness: material.clearcoat_perceptual_roughness.clamp(0.0, 1.0),
            pad: 0,
        }
    }
}

/// An [`AuroraMaterial`] as the tracer sees it: the record plus the images its texture slots
/// come from.
#[derive(Clone, Default)]
pub struct ExtractedMaterial {
    pub material: RTXMaterial,
    pub base_color_texture: Option<AssetId<Image>>,
    pub emissive_texture: Option<AssetId<Image>>,
    pub metallic_roughness_texture: Option<AssetId<Image>>,
    pub normal_map_texture: Option<AssetId<Image>>,
}

impl ExtractedMaterial {
    /// The record with every texture slot filled from `textures` (or its fallback).
    pub fn resolve(
        &self,
        render_device: &RenderDevice,
        textures: &VulkanAssets<Image>,
    ) -> RTXMaterial {
        self.resolve_checked(render_device, textures).0
    }

    /// Like [`resolve`](Self::resolve), plus whether every referenced texture was found (a
    /// `false` means a fallback stands in and the record is worth resolving again later).
    pub fn resolve_checked(
        &self,
        render_device: &RenderDevice,
        textures: &VulkanAssets<Image>,
    ) -> (RTXMaterial, bool) {
        let mut complete = true;
        let mut slot = |id: Option<AssetId<Image>>, fallback: u32| {
            let Some(id) = id else { return fallback };
            match textures.get_by_id(id) {
                Some(texture) => render_device.register_bindless_texture(texture),
                None => {
                    complete = false;
                    fallback
                }
            }
        };
        let material = RTXMaterial {
            base_color_texture: slot(self.base_color_texture, WHITE_TEXTURE_IDX),
            base_emissive_texture: slot(self.emissive_texture, WHITE_TEXTURE_IDX),
            metallic_roughness_texture: slot(self.metallic_roughness_texture, WHITE_TEXTURE_IDX),
            normal_texture: slot(self.normal_map_texture, DEFAULT_NORMAL_TEXTURE_IDX),
            ..self.material
        };
        (material, complete)
    }
}

impl VulkanAsset for AuroraMaterial {
    type ExtractedAsset = ExtractedMaterial;
    type ExtractParam = ();
    type PreparedAsset = ExtractedMaterial;

    fn extract_asset(
        &self,
        _param: &mut bevy::ecs::system::SystemParamItem<Self::ExtractParam>,
    ) -> Option<Self::ExtractedAsset> {
        Some(ExtractedMaterial {
            material: RTXMaterial::from_material(self),
            base_color_texture: self.base_color_texture.as_ref().map(Handle::id),
            emissive_texture: self.emissive_texture.as_ref().map(Handle::id),
            metallic_roughness_texture: self.metallic_roughness_texture.as_ref().map(Handle::id),
            normal_map_texture: self.normal_map_texture.as_ref().map(Handle::id),
        })
    }

    fn prepare_asset(
        asset: Self::ExtractedAsset,
        _render_device: &RenderDevice,
    ) -> Self::PreparedAsset {
        asset
    }

    fn destroy_asset(_render_device: &RenderDevice, _prepared_asset: &Self::PreparedAsset) {}
}

pub struct MaterialPlugin;

impl Plugin for MaterialPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<AuroraMaterialTypesPlugin>() {
            app.add_plugins(AuroraMaterialTypesPlugin);
        }
        app.init_vulkan_asset::<AuroraMaterial>();
    }
}
