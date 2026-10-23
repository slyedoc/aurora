//! World UI panels as components.
//!
//! [`UiPanel3d`] on an entity turns it into a glowing quad whose emissive texture is a live
//! `bevy_ui` tree: the plugin mints the offscreen target, the routing camera, the UI root and
//! the mesh + material, then hands the root back as [`UiPanel3dRoot`] so the app spawns its
//! widgets under it. Every [`crate::ui_render::UiPointerSource`] (window mouse, VR
//! controller aim) operates the panel through [`crate::ui_render::UiSurfacePanel`].
//!
//! [`InspectorPanel3d`] goes one step further: a `bevy_feathers_inspector` section for a
//! reflected resource / component / entity built onto the panel — the one-line way to expose
//! a tool's state in the world (a wrist panel in VR, a floating card on desktop). The
//! inspector needs no changes for this: it only ever spawns under the host it is given.

use std::any::TypeId;

use bevy::{
    camera::{Camera, ImageRenderTarget, RenderTarget},
    ecs::{lifecycle::HookContext, world::DeferredWorld},
    feathers::{theme::ThemeBackgroundColor, tokens},
    feathers_inspector::{BuildComponentInspector, BuildEntityInspector, BuildResourceInspector},
    prelude::*,
    ui::UiTargetCamera,
};

use crate::{
    environment::InEnvironment,
    material::{AlphaMode, AuroraMaterial, AuroraMaterial3d},
    sky::EnvironmentSkies,
    mesh::{AuroraMesh, AuroraMesh3d},
    ui_render::{UiSurfacePanel, ui_target_placeholder},
};

/// A world-space UI panel: `size` meters on the entity's local XY plane, facing +Z, rendered
/// from a `px` texture. `scale` is the UI scale factor (2.0 doubles every widget — VR wants
/// big hit targets); `nits` the panel's emission. Once built, the entity also carries
/// [`UiPanel3dRoot`], [`UiSurfacePanel`], an `AuroraMesh3d` and an [`AuroraMaterial3d`].
///
/// The pointer bridge reads the panel's `GlobalTransform`; for a panel parented to something
/// (a wrist panel under a controller grip) that is the GPU read-back pose, one frame old.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct UiPanel3d {
    /// World size in meters.
    pub size: Vec2,
    /// Texture resolution. Keep its aspect equal to `size`'s so UI pixels stay square.
    pub px: UVec2,
    /// UI scale factor (logical px = physical px / scale).
    pub scale: f32,
    /// Emission of lit pixels, in nits; with `relative`, in multiples of the white of the
    /// environment the panel is in.
    pub nits: f32,
    /// `nits` follows the environment's exposure (1.0 reads as paper white anywhere, by day or
    /// night): for signs and labels that must read wherever they stand.
    pub relative: bool,
    /// Give the root the theme's window background (rounded). Off = transparent root; style
    /// it yourself.
    pub opaque: bool,
}

impl Default for UiPanel3d {
    fn default() -> Self {
        Self {
            size: Vec2::new(0.4, 0.25),
            px: UVec2::new(1024, 640),
            scale: 1.0,
            nits: 300.0,
            relative: false,
            opaque: true,
        }
    }
}

impl UiPanel3d {
    /// A panel `size` meters wide at `px_per_meter` texels, height from `aspect` (w/h).
    pub fn sized(width: f32, aspect: f32, px_per_meter: f32) -> Self {
        let size = Vec2::new(width, width / aspect);
        Self {
            size,
            px: (size * px_per_meter).round().as_uvec2().max(UVec2::ONE),
            ..default()
        }
    }
}

/// The built panel's pieces; spawn UI under `root`. Inserted by [`build_panels`]; removing
/// it (or despawning the panel) despawns the camera and the root tree.
#[derive(Component, Clone, Debug)]
#[component(on_remove = on_root_removed)]
pub struct UiPanel3dRoot {
    /// The `UiTargetCamera` root node.
    pub root: Entity,
    /// The routing camera (`RenderTarget::Image`).
    pub camera: Entity,
    /// The surface texture (also the material's emissive texture).
    pub target: Handle<Image>,
}

fn on_root_removed(mut world: DeferredWorld, ctx: HookContext) {
    let Some(built) = world.get::<UiPanel3dRoot>(ctx.entity).cloned() else {
        return;
    };
    let mut commands = world.commands();
    commands.entity(built.root).try_despawn();
    commands.entity(built.camera).try_despawn();
}

/// A feathers inspector on a [`UiPanel3d`]: the section is (re)built onto the panel's root
/// whenever this component changes. Requires the target type to be `Reflect` + registered.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
#[require(UiPanel3d)]
pub enum InspectorPanel3d {
    Resource(TypeId),
    Component { entity: Entity, type_id: TypeId },
    Entity(Entity),
}

impl InspectorPanel3d {
    pub fn resource<T: Resource>() -> Self {
        Self::Resource(TypeId::of::<T>())
    }

    pub fn component<T: Component>(entity: Entity) -> Self {
        Self::Component {
            entity,
            type_id: TypeId::of::<T>(),
        }
    }
}

/// The node an [`InspectorPanel3d`]'s section lives under (a child of the panel root, so the
/// app can add its own children beside it).
#[derive(Component, Clone, Copy, Debug)]
pub struct InspectorPanel3dHost(pub Entity);

pub struct UiPanelPlugin;

impl Plugin for UiPanelPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                resize_panels,
                build_panels,
                build_inspector_panels,
                follow_exposure,
            )
                .chain(),
        );
    }
}

/// Relative panels emit `nits` times their environment's white, updated as it changes.
fn follow_exposure(
    skies: Option<Res<EnvironmentSkies>>,
    panels: Query<(&UiPanel3d, &AuroraMaterial3d, Option<&InEnvironment>), With<UiPanel3dRoot>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    let Some(skies) = skies else {
        return;
    };
    for (panel, material, world) in &panels {
        if !panel.relative {
            continue;
        }
        let Some(world) = world else {
            continue;
        };
        let want = panel.nits * skies.white_nits(world.0);
        let Some(current) = materials.get(&material.0).map(|m| m.emissive.red) else {
            continue;
        };
        if (current - want).abs() > want * 0.01
            && let Some(mut material) = materials.get_mut(&material.0)
        {
            material.emissive = LinearRgba::WHITE * want;
        }
    }
}

/// Builds every new [`UiPanel3d`]: target image, routing camera, UI root, quad + material.
fn build_panels(
    mut commands: Commands,
    panels: Query<(Entity, &UiPanel3d), Without<UiPanel3dRoot>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    for (entity, panel) in &panels {
        let target = images.add(ui_target_placeholder(panel.px));
        let scale = panel.scale.max(0.1);
        let camera = commands
            .spawn((
                Name::new("ui panel surface"),
                Camera::default(),
                RenderTarget::Image(ImageRenderTarget {
                    handle: target.clone(),
                    scale_factor: scale,
                }),
            ))
            .id();
        let mut root = commands.spawn((
            Name::new("ui panel root"),
            // The target's size, whatever the content wants (a long line would widen it).
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                min_width: Val::Px(0.0),
                min_height: Val::Px(0.0),
                max_width: Val::Percent(100.0),
                max_height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(8.0)),
                row_gap: Val::Px(6.0),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                overflow: Overflow::clip(),
                ..default()
            },
            UiTargetCamera(camera),
        ));
        if panel.opaque {
            root.insert(ThemeBackgroundColor(tokens::WINDOW_BG));
        }
        let root = root.id();

        // One face, facing +Z; `Rectangle` already puts v = 0 at the top, the UI target's
        // image convention.
        let mesh = Mesh::from(Rectangle::new(panel.size.x, panel.size.y));
        let built = (
            AuroraMesh3d(meshes.add(AuroraMesh::from_mesh(&mesh).expect("cuboid has positions"))),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::WHITE * panel.nits,
                emissive_texture: Some(target.clone()),
                // A screen glows; it is not a lamp (its texture is mostly dark widgets).
                glow_only: true,
                // Nor a mirror: its dark parts show dark, not the sun's gloss.
                unlit: true,
                // Without a background the quad is cut out wherever the UI drew nothing: rays
                // pass through to the world behind, and only the widgets float.
                base_color_texture: (!panel.opaque).then(|| target.clone()),
                alpha_mode: match panel.opaque {
                    true => AlphaMode::Opaque,
                    false => AlphaMode::Mask(0.05),
                },
                ..default()
            })),
            UiSurfacePanel {
                target: target.clone(),
                size: panel.size,
                scale,
            },
            UiPanel3dRoot {
                root,
                camera,
                target,
            },
        );
        // The panel can be despawned before this lands (closed the frame it opened): then its
        // camera and UI root go with it instead of rendering for nobody.
        commands.queue(move |world: &mut World| match world.get_entity_mut(entity) {
            Ok(mut panel) => {
                panel.insert(built);
            }
            Err(_) => {
                world.despawn(camera);
                world.despawn(root);
            }
        });
    }
}

/// A built panel whose `size`, `px` or `scale` changed gets a new quad and a new target at
/// the new resolution (surface targets are never resized in place); its UI tree is kept.
fn resize_panels(
    mut panels: Query<
        (
            &UiPanel3d,
            &mut UiPanel3dRoot,
            &mut UiSurfacePanel,
            &AuroraMesh3d,
            &AuroraMaterial3d,
        ),
        Changed<UiPanel3d>,
    >,
    mut cameras: Query<&mut RenderTarget>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<AuroraMesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    for (panel, mut built, mut surface, mesh, material) in &mut panels {
        let scale = panel.scale.max(0.1);
        let px = images
            .get(built.target.id())
            .map(|image| UVec2::new(image.width(), image.height()));
        if surface.size == panel.size && surface.scale == scale && px == Some(panel.px) {
            continue;
        }
        if px != Some(panel.px) {
            let target = images.add(ui_target_placeholder(panel.px));
            if let Ok(mut render_target) = cameras.get_mut(built.camera) {
                *render_target = RenderTarget::Image(ImageRenderTarget {
                    handle: target.clone(),
                    scale_factor: scale,
                });
            }
            if let Some(mut material) = materials.get_mut(&material.0) {
                material.emissive_texture = Some(target.clone());
                if material.base_color_texture.is_some() {
                    material.base_color_texture = Some(target.clone());
                }
            }
            built.target = target.clone();
            surface.target = target;
        } else if surface.scale != scale
            && let Ok(mut render_target) = cameras.get_mut(built.camera)
        {
            *render_target = RenderTarget::Image(ImageRenderTarget {
                handle: built.target.clone(),
                scale_factor: scale,
            });
        }
        if surface.size != panel.size {
            if let Some(mut mesh) = meshes.get_mut(&mesh.0) {
                *mesh = AuroraMesh::from_shape(Rectangle::new(panel.size.x, panel.size.y));
            }
        }
        surface.size = panel.size;
        surface.scale = scale;
    }
}

/// (Re)builds the inspector section of every new or changed [`InspectorPanel3d`] whose panel
/// is built.
fn build_inspector_panels(
    mut commands: Commands,
    panels: Query<
        (
            Entity,
            &InspectorPanel3d,
            &UiPanel3dRoot,
            Option<&InspectorPanel3dHost>,
        ),
        Or<(Changed<InspectorPanel3d>, Added<UiPanel3dRoot>)>,
    >,
) {
    for (entity, inspector, built, host) in &panels {
        let host = match host {
            Some(host) => host.0,
            None => {
                let host = commands
                    .spawn((
                        Name::new("inspector host"),
                        Node {
                            flex_direction: FlexDirection::Column,
                            align_self: AlignSelf::Stretch,
                            ..default()
                        },
                        ChildOf(built.root),
                    ))
                    .id();
                commands.entity(entity).insert(InspectorPanel3dHost(host));
                host
            }
        };
        match *inspector {
            InspectorPanel3d::Resource(type_id) => {
                commands.queue(BuildResourceInspector {
                    type_id,
                    panel: host,
                });
            }
            InspectorPanel3d::Component { entity, type_id } => {
                commands.queue(BuildComponentInspector {
                    target: entity,
                    type_id,
                    panel: host,
                });
            }
            InspectorPanel3d::Entity(target) => {
                commands.queue(BuildEntityInspector {
                    target,
                    panel: host,
                });
            }
        }
    }
}
