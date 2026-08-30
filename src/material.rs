//! The ray tracer's material.
//!
//! [`AuroraMaterial`] is the asset a `.bsn` scene or an example authors; it carries the same
//! field names as bevy's `StandardMaterial` for the subset the path tracer consumes, so baked
//! scenes migrate by a type-path swap. [`AuroraMaterial3d`] is the component that binds one to a
//! ray-traced entity; entities without one get [`DefaultAuroraMaterial`].
//!
//! ```text
//! bevy_aurora::material::AuroraMaterial3d(bevy_aurora::material::AuroraMaterial {
//!     perceptual_roughness: 0.55, base_color_texture: "bistro/textures/x.png",
//!     alpha_mode: bevy_aurora::material::AlphaMode::Mask(0.5),
//! })
//! ```
//!
//! On the GPU side the asset becomes an [`RTXMaterial`] record plus the image ids its texture
//! slots come from; the slots are resolved to bindless indices when instances are prepared, so
//! textures apply whenever they finish loading.

use bevy::{asset::AssetApp, ecs::template::FromTemplate, prelude::*};

use crate::{
    blas::{RTXMaterial, absorption_from_attenuation},
    render_device::RenderDevice,
    render_env::{DEFAULT_NORMAL_TEXTURE_IDX, WHITE_TEXTURE_IDX},
    vulkan_asset::{VulkanAsset, VulkanAssetExt, VulkanAssets},
};

/// How a surface's alpha is treated. Only `Mask` changes tracing (any-hit cutout, once the
/// hit shaders support it); `Blend` is carried for authoring and currently traces as opaque.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Default, Clone, PartialEq)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask(f32),
    Blend,
}

#[derive(Asset, Reflect, Clone, Debug, PartialEq)]
#[reflect(Default, Clone)]
pub struct AuroraMaterial {
    pub base_color: Color,
    pub base_color_texture: Option<Handle<Image>>,
    /// Linear radiance.
    pub emissive: LinearRgba,
    pub emissive_texture: Option<Handle<Image>>,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    /// Roughness in G, metallic in B (glTF layout), scaling the factors.
    pub metallic_roughness_texture: Option<Handle<Image>>,
    pub normal_map_texture: Option<Handle<Image>>,
    /// Displacement / height map, carried for tessellation; unused by the tracer today.
    pub depth_map: Option<Handle<Image>>,
    pub specular_transmission: f32,
    pub ior: f32,
    /// Beer-Lambert volume: light travelling `attenuation_distance` through the surface is
    /// tinted to `attenuation_color`. An infinite distance is a clear medium.
    pub attenuation_color: Color,
    pub attenuation_distance: f32,
    pub alpha_mode: AlphaMode,
}

impl Default for AuroraMaterial {
    fn default() -> Self {
        Self {
            base_color: Color::WHITE,
            base_color_texture: None,
            emissive: LinearRgba::BLACK,
            emissive_texture: None,
            perceptual_roughness: 0.5,
            metallic: 0.0,
            metallic_roughness_texture: None,
            normal_map_texture: None,
            depth_map: None,
            specular_transmission: 0.0,
            ior: 1.5,
            attenuation_color: Color::WHITE,
            attenuation_distance: f32::INFINITY,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

impl From<Color> for AuroraMaterial {
    fn from(base_color: Color) -> Self {
        Self {
            base_color,
            ..default()
        }
    }
}

/// The material of a ray-traced entity (`Mesh3d`, `Sphere`, `GltfModelHandle`).
#[derive(Component, FromTemplate, Clone, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
pub struct AuroraMaterial3d(pub Handle<AuroraMaterial>);

/// Given to `Mesh3d` entities spawned without an [`AuroraMaterial3d`].
#[derive(Resource)]
pub struct DefaultAuroraMaterial(pub Handle<AuroraMaterial>);

impl FromWorld for DefaultAuroraMaterial {
    fn from_world(world: &mut World) -> Self {
        let mut materials = world.resource_mut::<Assets<AuroraMaterial>>();
        Self(materials.add(AuroraMaterial::default()))
    }
}

fn default_material(
    mut commands: Commands,
    default: Res<DefaultAuroraMaterial>,
    bare: Query<Entity, (With<Mesh3d>, Without<AuroraMaterial3d>)>,
) {
    for entity in &bare {
        commands
            .entity(entity)
            .insert(AuroraMaterial3d(default.0.clone()));
    }
}

// ---- GPU side -------------------------------------------------------------------------------

impl RTXMaterial {
    pub fn from_material(material: &AuroraMaterial) -> Self {
        RTXMaterial {
            base_color_factor: {
                let c = material.base_color.to_srgba();
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
        app.init_asset::<AuroraMaterial>();
        app.register_type::<AuroraMaterial>();
        app.register_type::<AlphaMode>();
        app.register_type::<AuroraMaterial3d>();
        app.register_asset_reflect::<AuroraMaterial>();
        app.init_vulkan_asset::<AuroraMaterial>();
        app.init_resource::<DefaultAuroraMaterial>();
        app.add_systems(PostUpdate, default_material);
    }
}
