//! Offscreen render targets for `Camera3d`.
//!
//! A camera with `RenderTarget::Image` traces into its own Vulkan image instead of the
//! window, and that image is parked in [`VulkanAssets<Image>`] under the asset's id -- so
//! anything referencing the `Handle<Image>` gets the render target transparently: a UI node,
//! a material, a hit shader. Shaped exactly like [`crate::ui_render`]'s surface targets,
//! which do the same for offscreen `bevy_ui` trees; the split between them is `Camera3d`.
//!
//! Cost: a view is per-view state (uniform buffer, reservoirs, DLSS feature) plus its own
//! raygen dispatch, but the world-scale work -- TLAS, light table, GPU transforms, the SHARC
//! cache -- is built once per frame no matter how many views read it. So N targets cost
//! roughly their total pixels, not N frames. [`crate::MAX_VIEWS`] caps how many render in
//! one frame; a host wanting more (a wall of asset thumbnails) should render them over
//! successive frames rather than raising it.

use bevy::{
    camera::{Camera, Camera3d, Projection, RenderTarget, RenderTargetInfo},
    image::Image,
    platform::collections::HashMap,
    prelude::*,
};

use ash::vk;

use crate::{
    render_device::RenderDevice,
    render_texture::{RenderTexture, create_blank_texture},
    vulkan_asset::{VulkanAsset, VulkanAssetLoadingState, VulkanAssets},
};

/// The image-targeted `Camera3d`s this frame, and the Vulkan images backing them.
#[derive(Resource, Default)]
pub struct CameraTargets {
    /// Camera entity -> (target asset, size in pixels). Rebuilt every frame.
    pub by_camera: HashMap<Entity, (AssetId<Image>, UVec2)>,
    /// The created images, keyed by asset. App-lifetime: a despawned camera's texture stays
    /// registered so re-entering a state reuses the same asset id and bindless slot.
    pub map: HashMap<AssetId<Image>, (RenderTexture, vk::Extent2D)>,
}

/// A size-carrying placeholder [`Image`] for a camera render target: `data` is `None`, so
/// the texture upload path skips it and [`prepare_camera_targets`] builds the real target in
/// its place.
pub fn camera_target_placeholder(size: UVec2) -> Image {
    let mut image = Image::new_target_texture(
        size.x.max(1),
        size.y.max(1),
        wgpu_types::TextureFormat::Bgra8Unorm,
        None,
    );
    image.data = None;
    image
}

/// `PostUpdate`: record which `Camera3d`s target an image, and publish the texture's size as
/// the camera's target geometry so anything sizing off the camera agrees with the texture.
///
/// `Without<Camera2d>` is not the filter -- `With<Camera3d>` is, and `ui_render`'s surface
/// scan takes the complement. A UI panel camera is a plain `Camera`; a scene camera is a
/// `Camera3d`. Without that split both would claim the same asset and one would clobber the
/// other's texture in `VulkanAssets`.
pub fn sync_camera_targets(
    mut targets: ResMut<CameraTargets>,
    mut cameras: Query<(Entity, &mut Camera, &RenderTarget), With<Camera3d>>,
    images: Res<Assets<Image>>,
) {
    targets.by_camera.clear();
    for (entity, mut camera, render_target) in &mut cameras {
        let RenderTarget::Image(target) = render_target else {
            continue;
        };
        let asset = target.handle.id();
        let Some(image) = images.get(asset) else {
            continue;
        };
        let size = image.size();
        if size.x == 0 || size.y == 0 {
            continue;
        }
        let info = RenderTargetInfo {
            physical_size: size,
            scale_factor: target.scale_factor,
        };
        // Guarded write: a `Camera` flagged changed every frame is a change-detection lie.
        let stale = camera.computed.target_info.as_ref().is_none_or(|current| {
            current.physical_size != info.physical_size || current.scale_factor != info.scale_factor
        });
        if stale {
            camera.computed.target_info = Some(info);
        }
        targets.by_camera.insert(entity, (asset, size));
    }
}

/// `RenderSet::Prepare`: create the image for every new target.
///
/// `STORAGE` is the difference from a UI surface: the composite writes here as a colour
/// attachment, but the image is also sampled, and a future direct-write path needs storage.
pub fn prepare_camera_targets(
    render_device: Res<RenderDevice>,
    mut targets: ResMut<CameraTargets>,
    mut textures: ResMut<VulkanAssets<Image>>,
) {
    // Create on first sight, and RECREATE when the size changed: a viewport in a dock
    // resizes whenever the splitter moves, and `bevy_ui`'s
    // `update_viewport_render_target_size` resizes the Image asset to match. Without this
    // the asset says one size and the Vulkan image stays at the size it was first built
    // at, so the view traces at a resolution nothing samples back.
    let pending: Vec<(AssetId<Image>, UVec2)> = targets
        .by_camera
        .values()
        .filter(|(asset, size)| {
            targets.map.get(asset).is_none_or(|(_, extent)| {
                extent.width != size.x || extent.height != size.y
            })
        })
        .copied()
        .collect();
    for (asset, size) in pending {
        let texture = create_blank_texture(
            &render_device,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            size.x,
            size.y,
        );
        render_device.register_bindless_texture(&texture);
        // `VulkanAssets::insert` hands back whatever was under this id -- a previous target
        // on a resize, or an ordinary uploaded texture if the asset carried bytes before a
        // camera claimed it. Either way it may still be referenced by an in-flight frame,
        // so it goes through the deferred destroyer rather than being freed here. This is
        // the ONLY place the old texture is destroyed; `targets.map` holds a copy of the
        // same handles and must not free them again.
        if let Some(VulkanAssetLoadingState::Loaded(old)) =
            textures.insert(asset, VulkanAssetLoadingState::Loaded(texture))
        {
            <Image as VulkanAsset>::destroy_asset(&render_device, &old);
        }
        targets.map.insert(
            asset,
            (
                texture,
                vk::Extent2D {
                    width: size.x,
                    height: size.y,
                },
            ),
        );
        log::info!("camera: render target {:?} ({}x{})", asset, size.x, size.y);
    }
}

/// `PostUpdate`, after every `target_info` writer: fills `Camera::computed.clip_from_view`
/// from the `Projection`.
///
/// Only `bevy_render`'s `camera_system` writes that field, and aurora replaces
/// `bevy_render`. Left at its default, `Camera::viewport_to_world` returns the same ray for
/// every cursor position -- an editor's click-to-place lands everything on one spot.
///
/// The probe copy keeps an unchanged viewport from flagging `Camera` and `Projection`
/// changed every frame.
pub fn sync_camera_projections(mut cameras: Query<(&mut Camera, &mut Projection)>) {
    for (mut camera, mut projection) in &mut cameras {
        let Some(size) = camera.logical_viewport_size() else {
            continue;
        };
        if size.x == 0.0 || size.y == 0.0 {
            continue;
        }
        let mut probe = projection.clone();
        probe.update(size.x, size.y);
        let clip_from_view = match &camera.sub_camera_view {
            Some(sub_view) => probe.get_clip_from_view_for_sub(sub_view),
            None => probe.get_clip_from_view(),
        };
        if camera.computed.clip_from_view != clip_from_view {
            *projection = probe;
            camera.computed.clip_from_view = clip_from_view;
        }
    }
}

/// Registers the offscreen camera-target scan and allocation. Mirrors
/// [`crate::ui_render`]'s surface half, in the same schedule slots.
pub struct CameraTargetPlugin;

impl Plugin for CameraTargetPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraTargets>()
            .add_systems(PostUpdate, sync_camera_targets)
            .add_systems(
                PostUpdate,
                sync_camera_projections
                    .after(sync_camera_targets)
                    .after(crate::ui_render::sync_ui_surfaces)
                    .after(crate::ui_render::ui_camera_target_system),
            )
            .add_systems(
                Last,
                prepare_camera_targets.in_set(crate::ray_render_plugin::RenderSet::Prepare),
            );
    }
}
