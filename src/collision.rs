//! Baked collision geometry for `.bsn` scenes — the mesh, not the physics.
//!
//! An importer bakes a model's collision once into a `.collider` file and every placement of
//! it names that file. The render mesh is the wrong shape to collide with (a tree would
//! collide with its leaf cards) and inlining the geometry per entity would multiply it by the
//! placement count, so the two are separate assets sharing one file:
//!
//! ```text
//! bevy_aurora::collision::CollisionMesh("wow/meshes/elwynn_tree01.collider")
//! ```
//!
//! **This crate deliberately stops at the triangles.** It has no physics engine and does not
//! want one — a game turns [`CollisionShape`] into whatever its solver uses (zero inserts an
//! avian `Collider` and a static `RigidBody`). Keeping the component here is what lets a
//! `.bsn` carrying collision load in a viewer that has no physics at all: the type resolves,
//! the shape loads, and nothing happens with it. Emitting the game's own physics components
//! into the scene instead makes every consumer of that scene depend on that game
//! (`unknown type: avian3d::dynamics::rigid_body::RigidBody`, and the whole scene is refused).
//!
//! `.collider`: `ACOL`, version, vertex count, triangle count (u32 LE), then the positions
//! (f32 LE x 3) and the triangles (u32 LE x 3). Written by `aurora_files`' importers.

use bevy::{
    asset::{io::Reader, AssetLoader, LoadContext},
    prelude::*,
};

const MAGIC: &[u8; 4] = b"ACOL";
const VERSION: u32 = 1;

/// One baked collision mesh, in the naming entity's local space.
///
/// Indices are triangles rather than a flat list because every consumer wants them that way
/// and the file already stores them so.
#[derive(Asset, Reflect, Default, Debug)]
pub struct CollisionShape {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl CollisionShape {
    /// Triangle count — the cost figure worth logging before building anything from this.
    pub fn len(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// This entity's collision geometry. Shared: every placement of a model names the same file,
/// and the asset server hands them all one [`CollisionShape`].
#[derive(Component, Reflect, Default, Clone)]
#[reflect(Component, Default, Clone)]
pub struct CollisionMesh(pub Handle<CollisionShape>);

#[derive(TypePath)]
struct CollisionShapeLoader;

impl AssetLoader for CollisionShapeLoader {
    type Asset = CollisionShape;
    type Settings = ();
    type Error = Box<dyn std::error::Error + Send + Sync>;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _load_context: &mut LoadContext<'_>,
    ) -> Result<CollisionShape, Self::Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        let header: &[u8; 16] = bytes
            .get(..16)
            .and_then(|h| h.try_into().ok())
            .ok_or("truncated .collider header")?;
        let word = |i: usize| u32::from_le_bytes(header[i * 4..i * 4 + 4].try_into().unwrap());
        if &header[..4] != MAGIC || word(1) != VERSION {
            return Err("not a version 1 .collider".into());
        }
        let (vertex_count, triangle_count) = (word(2) as usize, word(3) as usize);
        let body = &bytes[16..];
        if body.len() != (vertex_count + triangle_count) * 12 {
            return Err("truncated .collider body".into());
        }
        let (positions, triangles) = body.split_at(vertex_count * 12);
        let float = |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap());
        let index = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        Ok(CollisionShape {
            positions: positions
                .chunks_exact(12)
                .map(|v| Vec3::new(float(&v[..4]), float(&v[4..8]), float(&v[8..])))
                .collect(),
            triangles: triangles
                .chunks_exact(12)
                .map(|t| [index(&t[..4]), index(&t[4..8]), index(&t[8..])])
                .collect(),
        })
    }

    fn extensions(&self) -> &[&str] {
        &["collider"]
    }
}

pub struct CollisionPlugin;

impl Plugin for CollisionPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<CollisionShape>()
            .register_asset_reflect::<CollisionShape>()
            .register_asset_loader(CollisionShapeLoader)
            .register_type::<CollisionMesh>();
    }
}
