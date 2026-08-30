//! The engine's own assets (GLSL shaders, blue-noise textures) served through an `aurora://`
//! asset source rooted at this crate's `assets/` directory, so apps that use the engine from
//! another repo — with their own asset root — still find them. The source is watched, so shader
//! hot-reload works wherever the engine is checked out.

use std::{path::PathBuf, time::Duration};

use bevy::{
    asset::{
        AssetPath,
        io::{
            AssetSourceBuilder, AssetWatcher,
            file::{FileAssetReader, FileWatcher},
        },
    },
    prelude::*,
};

/// Asset source id for the engine's own assets: `aurora://shaders/raygen.rgen`.
pub const AURORA_ASSET_SOURCE: &str = "aurora";

/// Absolute path of this crate's `assets/` directory at build time.
pub const AURORA_ASSET_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets");

/// `path` relative to the engine's `assets/` dir as an `aurora://` asset path.
pub fn aurora_asset(path: &str) -> AssetPath<'static> {
    AssetPath::parse(&format!("{AURORA_ASSET_SOURCE}://{path}")).into_owned()
}

/// Registers the `aurora://` source. Must run before bevy's `AssetPlugin` builds (it is the first
/// plugin in [`crate::ray_default_plugins::RayDefaultPlugins`]).
pub struct AuroraAssetSourcePlugin;

impl Plugin for AuroraAssetSourcePlugin {
    fn build(&self, app: &mut App) {
        app.register_asset_source(
            AURORA_ASSET_SOURCE,
            AssetSourceBuilder::new(|| Box::new(FileAssetReader::new(AURORA_ASSET_DIR)))
                .with_watcher(|sender| {
                    FileWatcher::new(
                        PathBuf::from(AURORA_ASSET_DIR),
                        sender,
                        Duration::from_millis(300),
                    )
                    .ok()
                    .map(|watcher| Box::new(watcher) as Box<dyn AssetWatcher>)
                }),
        );
    }
}
