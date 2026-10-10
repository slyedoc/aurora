//! Layered terrain for aurora.
//!
//! A [`Landscape`] entity owns a rectangle of pages (256² height texels each). Its children
//! carrying a [`HeightLayer`] are the height stack, evaluated bottom to top (child order):
//! [`HeightmapLayer`]s (imported tiles and stamps) and [`SculptLayer`]s (sparse brush deltas).
//! Layers are never baked into each other: moving, fading, hiding or re-ordering one only
//! re-evaluates the pages under it (`landscape/eval.slang`).
//!
//! Children carrying a [`MaterialLayer`] are the material stack, evaluated after the heights
//! into each texel's top four palette materials: [`SplatmapLayer`]s (imported splats),
//! [`MaterialRuleLayer`]s (slope / height rules) and [`PaintLayer`]s. With a
//! [`LandscapePalette`], tiles shade from those pages (`landscape/hit.slang`).
//!
//! The evaluated pages feed LOD tiles (one quadtree per page, `ProceduralMesh` tiles filled by
//! `landscape/tile.slang`, stitched across levels), heightfield colliders per page, and CPU
//! height queries ([`LandscapePages::height_at`], [`LandscapePages::raycast`]).

pub mod bake;
pub mod collider;
pub mod erosion;
pub mod generator;
pub mod layer;
pub mod page;
pub mod quadtree;
pub mod scatter;
pub mod sculpt;
pub mod shading;
pub mod tiles;
pub mod topology;

use bevy::prelude::*;
use bevy_aurora::{
    assets::aurora_asset,
    compute::ComputeModule,
    ray_render_plugin::{TeardownSchedule, on_shutdown, render_device_exists},
    surface_group::SurfaceGroupRegistry,
};

pub use bake::{BakedPages, LandscapeBake, LandscapeLive};
pub use erosion::ErosionLayer;
pub use generator::GeneratorLayer;
pub use layer::{
    HeightBlend, HeightLayer, HeightmapLayer, MaterialLayer, MaterialRuleLayer, PaintLayer,
    SculptLayer, SplatmapLayer,
};
pub use page::{LandscapePages, MaterialsEvaluated, PagesEvaluated};
pub use scatter::ScatterSpecies;
pub use sculpt::{BrushKind, LayerPages, PaintPages, SculptDab, SculptDabs, SculptPages};
pub use shading::{LandscapeMaterial, LandscapePalette};

pub use tiles::LandscapeTile;

pub mod prelude {
    pub use crate::{
        BrushKind, ErosionLayer, GeneratorLayer, HeightBlend, HeightLayer, HeightmapLayer, Landscape, LandscapeMaterial,
        LandscapePages, LandscapePalette, LandscapePlugin, LandscapeViewer, MaterialLayer, SphereFace,
        MaterialRuleLayer, PaintLayer, PaintPages, ScatterSpecies, SculptDab, SculptDabs,
        SculptLayer, SculptPages, SplatmapLayer,
    };
}

/// Height texels on a page's side.
pub const PAGE_TEXELS: u32 = 256;

/// A layered terrain over pages `pages_min..pages_max`. The entity's translation places it
/// (no rotation or scale); its `AuroraMaterial3d`, if any, shades every tile.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(Transform, Visibility)]
pub struct Landscape {
    /// First page (inclusive).
    pub pages_min: IVec2,
    /// Last page (exclusive).
    pub pages_max: IVec2,
    /// Metres between height texels.
    pub texel_size: f32,
    /// Levels below a page: leaf tiles are `page / 2^max_lod` across.
    pub max_lod: u8,
    /// Vertices per tile edge (2..=255).
    pub patch_resolution: u32,
    /// Split when the viewer is within `tile_size * split_factor`.
    pub split_factor: f32,
    /// Heightfield colliders per page.
    pub colliders: bool,
    /// Streaming: when > 0, the pages are a window this many pages around the viewer that
    /// follows it (`pages_min` / `pages_max` are kept up to date), for infinite worlds.
    pub stream_radius: u32,
    /// One face of a planet: the pages lie on a cube face around the entity's translation
    /// (the planet's centre) and are pushed out onto the sphere. Flat when `None`.
    pub sphere: Option<SphereFace>,
}

/// A cube-sphere face (`landscape/sphere.slang`): face-local (x, z) spans the cube face of
/// half-side `radius`, normalised outward onto the sphere; heights rise along the normal.
#[derive(Reflect, Clone, Copy, Debug, PartialEq)]
pub struct SphereFace {
    /// 0..6: +X, -X, +Y, -Y, +Z, -Z.
    pub face: u8,
    pub radius: f32,
}

impl SphereFace {
    /// (axis, u, v) of the face (`face_basis` in sphere.slang).
    pub fn basis(self) -> (Vec3, Vec3, Vec3) {
        match self.face {
            0 => (Vec3::X, Vec3::Z, Vec3::Y),
            1 => (Vec3::NEG_X, Vec3::NEG_Z, Vec3::Y),
            2 => (Vec3::Y, Vec3::X, Vec3::Z),
            3 => (Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
            4 => (Vec3::Z, Vec3::Y, Vec3::X),
            _ => (Vec3::NEG_Z, Vec3::X, Vec3::Y),
        }
    }

    /// The direction out of the planet through face-local `xz`.
    pub fn direction(self, xz: Vec2) -> Vec3 {
        let (axis, u, v) = self.basis();
        (axis * self.radius + u * xz.x + v * xz.y).normalize()
    }

    /// The planet-centred point `height` above face-local `xz`.
    pub fn point(self, xz: Vec2, height: f32) -> Vec3 {
        self.direction(xz) * (self.radius + height)
    }

    /// Face-local `xz` of a planet-centred position (projected onto this face's plane).
    pub fn coords(self, p: Vec3) -> Vec2 {
        let (axis, u, v) = self.basis();
        let cube = p / p.dot(axis).max(1.0e-6) * self.radius;
        Vec2::new(cube.dot(u), cube.dot(v))
    }
}

impl Default for Landscape {
    fn default() -> Self {
        Self {
            pages_min: IVec2::ZERO,
            pages_max: IVec2::splat(4),
            texel_size: 0.5,
            // 128 m pages, 16 m leaves, 0.5 m between vertices at 33.
            max_lod: 3,
            patch_resolution: 33,
            split_factor: 3.0,
            colliders: true,
            stream_radius: 0,
            sphere: None,
        }
    }
}

impl Landscape {
    /// The pages covering `min..max` (landscape-local xz).
    pub fn covering(min: Vec2, max: Vec2, texel_size: f32) -> Self {
        let page = PAGE_TEXELS as f32 * texel_size;
        Self {
            pages_min: (min / page).floor().as_ivec2(),
            pages_max: (max / page).ceil().as_ivec2(),
            texel_size,
            ..default()
        }
    }

    /// One face of a planet of `radius`: pages over the whole cube face. Pick the radius a
    /// whole number of half pages (`page_size / 2` multiples) so faces meet on page edges.
    pub fn planet_face(face: u8, radius: f32, texel_size: f32) -> Self {
        Self {
            sphere: Some(SphereFace { face, radius }),
            colliders: false,
            ..Self::covering(Vec2::splat(-radius), Vec2::splat(radius), texel_size)
        }
    }

    pub fn page_size(&self) -> f32 {
        PAGE_TEXELS as f32 * self.texel_size
    }

    /// Landscape-local xz rectangle of a page.
    pub fn page_rect(&self, page: IVec2) -> Rect {
        let s = self.page_size();
        Rect::from_corners(page.as_vec2() * s, (page + 1).as_vec2() * s)
    }

    /// The pages a landscape-local rectangle touches, clamped to the bounds.
    pub fn pages_in(&self, rect: Rect) -> impl Iterator<Item = IVec2> + use<> {
        let s = self.page_size();
        let lo = (rect.min / s).floor().as_ivec2().max(self.pages_min);
        let hi = (rect.max / s).floor().as_ivec2().min(self.pages_max - 1);
        (lo.y..=hi.y).flat_map(move |z| (lo.x..=hi.x).map(move |x| IVec2::new(x, z)))
    }

    pub fn page_count(&self) -> IVec2 {
        (self.pages_max - self.pages_min).max(IVec2::ZERO)
    }
}

/// Streaming landscapes recentre their window on the viewer's page.
fn follow_viewer(
    viewers: Query<&GlobalTransform, With<LandscapeViewer>>,
    cameras: Query<(&GlobalTransform, &Camera), With<Camera3d>>,
    mut landscapes: Query<(&mut Landscape, &GlobalTransform)>,
) {
    let Some(viewer) = tiles::viewer_position(&viewers, &cameras) else {
        return;
    };
    for (mut landscape, transform) in &mut landscapes {
        if landscape.stream_radius == 0 || landscape.sphere.is_some() {
            continue;
        }
        let r = landscape.stream_radius as i32;
        let centre = ((viewer - transform.translation()).xz() / landscape.page_size())
            .floor()
            .as_ivec2();
        let min = centre - IVec2::splat(r);
        if landscape.pages_min != min || landscape.pages_max != min + IVec2::splat(2 * r + 1) {
            landscape.pages_min = min;
            landscape.pages_max = min + IVec2::splat(2 * r + 1);
        }
    }
}

/// The camera tiles split toward. Without one, the first active `Camera3d`.
#[derive(Component, Reflect, Default, Clone, Copy)]
#[reflect(Component, Default)]
pub struct LandscapeViewer;

/// The crate's kernels (`aurora://shaders/landscape/`).
#[derive(Resource)]
pub struct LandscapeKernels {
    pub eval: Handle<ComputeModule>,
    pub generator: Handle<ComputeModule>,
    pub material: Handle<ComputeModule>,
    pub brush: Handle<ComputeModule>,
    pub erosion: Handle<ComputeModule>,
    pub scatter: Handle<ComputeModule>,
    pub tile: Handle<ComputeModule>,
}

pub struct LandscapePlugin;

impl Plugin for LandscapePlugin {
    fn build(&self, app: &mut App) {
        let server = app.world().resource::<AssetServer>();
        let eval = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/eval.slang")),
            &["clear_page", "apply_heightmap", "apply_sculpt"],
        ));
        let generator = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/generator.slang")),
            &["apply_generator"],
        ));
        let material = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/material_eval.slang")),
            &["clear_materials", "apply_splatmap", "apply_rule", "apply_paint"],
        ));
        let brush = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/brush.slang")),
            &["sculpt_brush"],
        ));
        let erosion = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/erosion.slang")),
            &[
                "init",
                "rain",
                "flux",
                "water",
                "erode",
                "advect",
                "evaporate",
                "slip_out",
                "slip_in",
                "write_delta",
            ],
        ));
        let scatter = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/scatter.slang")),
            &["scatter"],
        ));
        let tile = server.add(ComputeModule::new(
            server.load(aurora_asset("shaders/landscape/tile.slang")),
            &["tile_fill"],
        ));
        app.insert_resource(LandscapeKernels {
            eval,
            generator,
            material,
            brush,
            erosion,
            scatter,
            tile,
        });

        app.init_asset::<bake::BakedPages>()
            .register_asset_reflect::<bake::BakedPages>()
            .init_asset_loader::<bake::BakedPagesLoader>()
            .init_resource::<LandscapeLive>()
            .register_type::<LandscapeBake>()
            .init_asset::<sculpt::LayerPagesFile>()
            // Reflected so a layer's file handle saves as its path.
            .register_asset_reflect::<sculpt::LayerPagesFile>()
            .init_asset_loader::<sculpt::LayerPagesLoader>()
            .init_resource::<topology::PatchTopologies>()
            .init_resource::<SculptDabs>()
            .add_message::<PagesEvaluated>()
            .add_message::<MaterialsEvaluated>()
            .register_type::<Landscape>()
            .register_type::<LandscapeViewer>()
            .register_type::<HeightLayer>()
            .register_type::<HeightmapLayer>()
            .register_type::<SculptLayer>()
            .register_type::<MaterialLayer>()
            .register_type::<SplatmapLayer>()
            .register_type::<MaterialRuleLayer>()
            .register_type::<PaintLayer>()
            .register_type::<LandscapePalette>()
            .register_type::<ScatterSpecies>()
            .register_type::<ErosionLayer>()
            .register_type::<GeneratorLayer>()
            .add_observer(generator::on_genome_removed)
            .add_observer(scatter::on_state_removed)
            .add_observer(page::on_pages_removed)
            .add_observer(layer::on_source_removed)
            .add_observer(layer::on_splatmap_removed)
            .add_observer(sculpt::on_layer_pages_removed::<f32>)
            .add_observer(sculpt::on_layer_pages_removed::<[u32; 2]>)
            .add_observer(shading::on_shading_removed)
            .add_observer(tiles::on_root_removed)
            .add_observer(tiles::on_landscape_removed)
            .add_systems(
                PostUpdate,
                (
                    // The stacks: sources, edits, evaluation.
                    (
                        follow_viewer,
                        page::allocate_pages,
                        layer::upload_heightmaps,
                        generator::upload_genomes,
                        layer::upload_splatmaps,
                        layer::track_layers,
                        layer::track_materials,
                        sculpt::load_page_files,
                        sculpt::apply_dabs,
                        bake::apply_bakes,
                        erosion::start_bakes,
                        page::evaluate_pages,
                        erosion::run_bakes,
                    )
                        .chain(),
                    // Geometry and physics.
                    (
                        tiles::spawn_roots,
                        tiles::mark_refills,
                        tiles::update_quadtrees,
                        tiles::spawn_tiles,
                        tiles::refill_tiles,
                        tiles::swap_refilled,
                        tiles::despawn_covered_tiles,
                        collider::update_colliders,
                        collider::finish_colliders,
                    )
                        .chain(),
                    // Scatter and shading.
                    (
                        scatter::size_blocks,
                        scatter::scatter_species,
                        (shading::sync_shading, shading::wear_class)
                            .chain()
                            .run_if(resource_exists::<shading::LandscapeClass>),
                    )
                        .chain(),
                )
                    .chain()
                    .before(TransformSystems::Propagate)
                    .run_if(render_device_exists),
            )
            .add_systems(
                Startup,
                shading::register_class
                    .run_if(resource_exists::<SurfaceGroupRegistry>)
                    .run_if(render_device_exists),
            )
            .add_systems(
                TeardownSchedule,
                (
                    page::release_pages,
                    layer::release_sources,
                    layer::release_splatmaps,
                    sculpt::release_layer_pages::<f32>,
                    sculpt::release_layer_pages::<[u32; 2]>,
                    shading::release_shading,
                    generator::release_genomes,
                    scatter::release_states,
                )
                    .before(on_shutdown),
            );
    }
}
