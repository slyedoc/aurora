//! The ray tracer's material, as data.
//!
//! [`AuroraMaterial`] is the asset a `.bsn` scene or an example authors; it carries the same
//! field names as bevy's `StandardMaterial` for the subset the path tracer consumes, so baked
//! scenes migrate by a type-path swap. [`AuroraMaterial3d`] is the component that binds one to a
//! ray-traced entity.
//!
//! ```text
//! bevy_aurora::material::AuroraMaterial3d(bevy_aurora::material::AuroraMaterial {
//!     perceptual_roughness: 0.55, base_color_texture: "bistro/textures/x.png",
//!     alpha_mode: bevy_aurora::material::AlphaMode::Mask(0.5),
//! })
//! ```
//!
//! Every reflected type here keeps the `bevy_aurora::material` type path it had before it moved
//! out of the renderer: that string is in every scene on disk.
//!
//! The GPU side -- the record the hit shaders read, bindless texture resolution -- is
//! `bevy_aurora::material`, which re-exports this crate.

use bevy::{asset::AssetApp, ecs::template::FromTemplate, prelude::*};

/// Which side of a surface is culled.
///
/// Aurora's own rather than bevy's: bevy's `Face` lives in `bevy_render::render_resource`,
/// a wgpu type this crate cannot reach. Same variants, so an editor's material inspector
/// matches on it unchanged.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[reflect(Clone, PartialEq)]
#[type_path = "bevy_aurora::material"]
pub enum Face {
    Front,
    Back,
}

/// How a height map is stepped through when parallax mapping.
///
/// Aurora's own for the same reason as [`Face`] -- bevy's lives behind `bevy_pbr`. Carried
/// for authoring round-trip; the tracer does no parallax mapping, it traces the geometry.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Default, Clone, PartialEq)]
#[type_path = "bevy_aurora::material"]
pub enum ParallaxMappingMethod {
    #[default]
    Occlusion,
    Relief {
        max_steps: u32,
    },
}

/// How a surface's alpha is treated. Only `Mask` changes tracing (any-hit cutout, once the
/// hit shaders support it); `Blend` is carried for authoring and currently traces as opaque.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Default, Clone, PartialEq)]
#[type_path = "bevy_aurora::material"]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask(f32),
    Blend,
    // The raster blend modes, carried so an authored material round-trips. The tracer
    // treats each as `Blend` -- it resolves transparency by tracing through the surface,
    // not by a framebuffer blend equation, so the distinctions have no analogue.
    Premultiplied,
    AlphaToCoverage,
    Add,
    Multiply,
}

#[derive(Asset, Reflect, Clone, Debug, PartialEq)]
#[reflect(Default, Clone)]
#[type_path = "bevy_aurora::material"]
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

    // --- Authoring surface -------------------------------------------------------------
    // Carried so a `StandardMaterial` round-trips through this type without losing what an
    // editor lets you set. NONE of them reach the tracer yet: they are saved, reflected and
    // re-loaded, but do not change the image. Each notes what wiring it would take, so the
    // gap is a task rather than a surprise.
    /// Dielectric specular at normal incidence, 0.5 mapping to 4% like bevy. The hit
    /// shaders hardcode 4%; wiring it is a field on `RTXMaterial`.
    pub reflectance: f32,
    /// Ambient occlusion map. A path tracer computes occlusion by tracing it, so a baked AO
    /// map would double-darken -- this is carried for round-trip and for whatever a surface
    /// group wants to do with it, not as something the opaque shader should start reading.
    pub occlusion_texture: Option<Handle<Image>>,
    /// Shade back faces instead of culling them. Wiring it means clearing
    /// `VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT` per instance in
    /// `tlas_builder`, not a material-record change.
    pub double_sided: bool,
    /// Which face is culled when `double_sided` is false.
    pub cull_mode: Option<Face>,
    /// Green channel of the normal map is inverted (DirectX-style maps). Wiring it means a
    /// flag in `RTXMaterial` and a sign flip in the hit shaders' normal decode -- a
    /// coordinated change to the GPU record and `types.glsl`.
    pub flip_normal_map_y: bool,
    /// Emit `base_color` directly with no lighting.
    pub unlit: bool,
    /// Whether distance fog applies. Aurora has no fog pass -- atmosphere and participating
    /// media are the physical equivalents -- so this is round-trip only.
    pub fog_enabled: bool,
    /// Depth-sort nudge for coplanar raster geometry. A ray tracer has no depth buffer to
    /// bias -- it intersects the geometry -- so this is round-trip only.
    pub depth_bias: f32,
    /// Parallax mapping, all three round-trip only: parallax fakes displacement in a
    /// fragment shader, and the tracer traces the real surface. `depth_map` is the height
    /// map these steer.
    pub parallax_depth_scale: f32,
    pub parallax_mapping_method: ParallaxMappingMethod,
    pub max_parallax_layer_count: f32,
    /// Affine transform applied to every UV before sampling. Round-trip only: the hit
    /// shaders sample the mesh's UVs directly.
    pub uv_transform: bevy::math::Affine2,
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
            reflectance: 0.5,
            occlusion_texture: None,
            double_sided: false,
            cull_mode: Some(Face::Back),
            flip_normal_map_y: false,
            unlit: false,
            fog_enabled: true,
            depth_bias: 0.0,
            parallax_depth_scale: 0.1,
            parallax_mapping_method: ParallaxMappingMethod::Occlusion,
            max_parallax_layer_count: 16.0,
            uv_transform: bevy::math::Affine2::IDENTITY,
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

/// The material of a ray-traced entity. Its default is the default handle, which
/// [`AuroraMaterialTypesPlugin`] fills with [`AuroraMaterial::default`], so a mesh spawned
/// without a material traces as that.
#[derive(Component, FromTemplate, Clone, Debug, Default, Reflect, PartialEq, Eq)]
#[reflect(Component, Default, Clone, PartialEq)]
#[type_path = "bevy_aurora::material"]
pub struct AuroraMaterial3d(pub Handle<AuroraMaterial>);

/// The asset and its reflection, so a scene naming a material loads and round-trips. Nothing
/// here draws; `bevy_aurora::material::MaterialPlugin` adds this and the GPU side.
pub struct AuroraMaterialTypesPlugin;

impl Plugin for AuroraMaterialTypesPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<AuroraMaterial>()
            .register_type::<AuroraMaterial>()
            .register_type::<AlphaMode>()
            .register_type::<Face>()
            .register_type::<ParallaxMappingMethod>()
            .register_type::<AuroraMaterial3d>()
            .register_asset_reflect::<AuroraMaterial>();
        app.world_mut()
            .resource_mut::<Assets<AuroraMaterial>>()
            .insert(
                &Handle::<AuroraMaterial>::default(),
                AuroraMaterial::default(),
            )
            .expect("the default handle's slot is free");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_paths_are_the_ones_scenes_on_disk_carry() {
        assert_eq!(
            AuroraMaterial::type_path(),
            "bevy_aurora::material::AuroraMaterial"
        );
        assert_eq!(
            AuroraMaterial3d::type_path(),
            "bevy_aurora::material::AuroraMaterial3d"
        );
        assert_eq!(AlphaMode::type_path(), "bevy_aurora::material::AlphaMode");
        assert_eq!(Face::type_path(), "bevy_aurora::material::Face");
        assert_eq!(
            ParallaxMappingMethod::type_path(),
            "bevy_aurora::material::ParallaxMappingMethod"
        );
    }
}
