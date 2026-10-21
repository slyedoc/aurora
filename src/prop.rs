//! Placeable props: a `.bsn` whose root carries a [`Prop`] is one, and the bakers write it there
//! (aurora_files' importers know a model's bounds when they emit it):
//!
//! ```text
//! #COMP_PROP_cart_city_01
//! bevy_aurora::prop::Prop { min: glam::Vec3 { .. }, max: glam::Vec3 { .. } }
//! ```
//!
//! [`catalog`] lists every prop under an asset root from the files' text alone, without loading
//! a mesh: an `AssetServer` cannot enumerate, and placing a prop wants its size before it spawns.

use std::path::{Path, PathBuf};

use bevy::{
    bsn::{BsnDocument, BsnNodeKind, BsnValue, BsnValueId},
    prelude::*,
};

/// A placeable prop: its bounds in its own space, about the origin it was baked at.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq)]
#[reflect(Component, Default)]
pub struct Prop {
    pub min: Vec3,
    pub max: Vec3,
}

impl Prop {
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

const PROP_PATH: &str = "bevy_aurora::prop::Prop";

/// One prop in a [`catalog`].
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogProp {
    /// The `.bsn`, relative to the asset root, `/`-separated (what `AssetServer::load` takes).
    pub path: String,
    /// The root's `#Name`, else the file stem.
    pub name: String,
    pub prop: Prop,
}

/// Every prop under `root`, sorted by path.
pub fn catalog(root: &Path) -> Vec<CatalogProp> {
    let mut files = Vec::new();
    walk(root, &mut files, 0);
    let mut props: Vec<CatalogProp> = files
        .iter()
        .filter_map(|file| {
            let text = std::fs::read_to_string(file).ok()?;
            let rel = file.strip_prefix(root).ok()?;
            let path = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let stem = rel.file_stem()?.to_string_lossy().into_owned();
            read_prop(&text).map(|(name, prop)| CatalogProp {
                path,
                name: name.unwrap_or(stem),
                prop,
            })
        })
        .collect();
    props.sort_by(|a, b| a.path.cmp(&b.path));
    props
}

/// The root's name and [`Prop`], if the document's root has one.
pub fn read_prop(text: &str) -> Option<(Option<String>, Prop)> {
    // Most scenes are not props: skip the parse.
    if !text.contains(PROP_PATH) {
        return None;
    }
    let doc = BsnDocument::parse(text).ok()?;
    let root = doc.node(*doc.roots.first()?)?;
    let BsnNodeKind::Entity { name, patches, .. } = &root.kind else {
        return None;
    };
    patches.iter().find_map(|&id| {
        let BsnNodeKind::Patch { symbol, value, .. } = &doc.node(id)?.kind else {
            return None;
        };
        if symbol.to_type_path() != PROP_PATH {
            return None;
        }
        let BsnValue::Struct(_, fields) = &doc.value(*value)?.value else {
            return Some((name.clone(), Prop::default()));
        };
        let mut prop = Prop::default();
        for (field, id) in fields {
            let v = vec3(&doc, *id)?;
            match field.as_str() {
                "min" => prop.min = v,
                "max" => prop.max = v,
                _ => {}
            }
        }
        Some((name.clone(), prop))
    })
}

fn vec3(doc: &BsnDocument, id: BsnValueId) -> Option<Vec3> {
    let BsnValue::Struct(_, fields) = &doc.value(id)?.value else {
        return None;
    };
    let mut v = Vec3::ZERO;
    for (field, id) in fields {
        let x = match doc.value(*id)?.value {
            BsnValue::Float(x) => x as f32,
            BsnValue::Int(x) => x as f32,
            _ => return None,
        };
        match field.as_str() {
            "x" => v.x = x,
            "y" => v.y = x,
            "z" => v.z = x,
            _ => {}
        }
    }
    Some(v)
}

/// `.bsn` files under `dir`, skipping hidden directories and symlinks.
fn walk(dir: &Path, out: &mut Vec<PathBuf>, depth: u32) {
    if depth > 16 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            if !entry.file_name().to_string_lossy().starts_with('.') {
                walk(&path, out, depth + 1);
            }
        } else if kind.is_file() && path.extension().is_some_and(|e| e == "bsn") {
            out.push(path);
        }
    }
}

pub struct PropPlugin;

impl Plugin for PropPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Prop>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_root_prop() {
        let text = "#Cart\nbevy_aurora::prop::Prop { min: glam::Vec3 { x: -1.0, y: 0.0, z: -2.0 }, max: glam::Vec3 { x: 1.0, y: 2.5, z: 2 } }\n";
        let (name, prop) = read_prop(text).unwrap();
        assert_eq!(name.as_deref(), Some("Cart"));
        assert_eq!(prop.min, Vec3::new(-1.0, 0.0, -2.0));
        assert_eq!(prop.max, Vec3::new(1.0, 2.5, 2.0));
    }

    #[test]
    fn a_scene_without_one_is_not_a_prop() {
        assert!(read_prop("#Tile\nbevy_camera::visibility::Visibility::Inherited\n").is_none());
    }
}
