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
    mesh::VertexAttributeValues,
    prelude::*,
    ui::UiTargetCamera,
};

use crate::{
    material::{AuroraMaterial, AuroraMaterial3d},
    ui_render::{UiSurfacePanel, ui_target_placeholder},
};

/// A world-space UI panel: `size` meters on the entity's local XY plane, facing +Z, rendered
/// from a `px` texture. `scale` is the UI scale factor (2.0 doubles every widget — VR wants
/// big hit targets); `nits` the panel's emission. Once built, the entity also carries
/// [`UiPanel3dRoot`], [`UiSurfacePanel`], a `Mesh3d` and an [`AuroraMaterial3d`].
///
/// The pointer bridge reads the panel's `GlobalTransform`, so a panel parented to something
/// (a wrist panel under a controller grip) needs `TransformPlugin { propagate_on_cpu: true }`;
/// aurora's default syncs roots only and a child panel would never move.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct UiPanel3d {
    /// World size in meters.
    pub size: Vec2,
    /// Texture resolution. Keep its aspect equal to `size`'s so UI pixels stay square.
    pub px: UVec2,
    /// UI scale factor (logical px = physical px / scale).
    pub scale: f32,
    /// Emission of lit pixels, in nits.
    pub nits: f32,
    /// Quad thickness in meters (a slab, so it has edges to see from the side).
    pub thickness: f32,
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
            thickness: 0.01,
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
        app.add_systems(Update, (build_panels, build_inspector_panels).chain());
    }
}

/// Builds every new [`UiPanel3d`]: target image, routing camera, UI root, quad + material.
fn build_panels(
    mut commands: Commands,
    panels: Query<(Entity, &UiPanel3d), Without<UiPanel3dRoot>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<AuroraMaterial>>,
) {
    for (entity, panel) in &panels {
        let target = images.add(ui_target_placeholder(panel.px));
        let camera = commands
            .spawn((
                Name::new("ui panel surface"),
                Camera::default(),
                RenderTarget::Image(ImageRenderTarget {
                    handle: target.clone(),
                    scale_factor: panel.scale.max(0.1),
                }),
            ))
            .id();
        let mut root = commands.spawn((
            Name::new("ui panel root"),
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
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

        // The UI target follows image convention (v = 0 at the TOP); bevy's `Cuboid` maps
        // its +Z face with v = 0 at the bottom — flip V or the panel reads upside down.
        let mut mesh = Mesh::from(Cuboid::new(panel.size.x, panel.size.y, panel.thickness));
        if let Some(VertexAttributeValues::Float32x2(uvs)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0)
        {
            for uv in uvs.iter_mut() {
                uv[1] = 1.0 - uv[1];
            }
        }
        commands.entity(entity).insert((
            Mesh3d(meshes.add(mesh)),
            AuroraMaterial3d(materials.add(AuroraMaterial {
                base_color: Color::BLACK,
                emissive: LinearRgba::WHITE * panel.nits,
                emissive_texture: Some(target.clone()),
                ..default()
            })),
            UiSurfacePanel {
                target: target.clone(),
                size: panel.size,
            },
            UiPanel3dRoot {
                root,
                camera,
                target,
            },
        ));
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
