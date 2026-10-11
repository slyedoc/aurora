//! Shaders are Slang modules, compiled to SPIR-V by `slangc` when they load.

use std::{borrow::Cow, fs::read_to_string};
use thiserror::Error;

use bevy::{asset::AssetLoader, prelude::*, tasks::ConditionalSendFuture};

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("slangc failed for {path}:\n{stderr}")]
    Slang { path: String, stderr: String },
}

#[derive(TypePath, Default)]
pub struct ShaderLoader;

#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    pub path: String,
    pub spirv: Option<Cow<'static, [u8]>>,
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ();
    type Error = ShaderLoaderError;

    fn extensions(&self) -> &[&str] {
        &["slang"]
    }

    fn load(
        &self,
        _reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            // slangc wants a real file (module `import`s resolve beside it): the engine's own
            // dir for `aurora://`, the app's assets dir otherwise.
            let source_path = match load_context.path().source() {
                bevy::asset::io::AssetSourceId::Name(name)
                    if name.as_ref() == crate::assets::AURORA_ASSET_SOURCE =>
                {
                    format!(
                        "{}/{}",
                        crate::assets::AURORA_ASSET_DIR,
                        load_context.path().path().display()
                    )
                }
                _ => bevy::asset::io::file::FileAssetReader::get_base_path()
                    .join("assets")
                    .join(load_context.path().path())
                    .display()
                    .to_string(),
            };
            let spirv = compile_slang(&source_path)?;
            // Every module it imports, read through the loader: bevy then reloads this shader
            // when any of them changes.
            for import in slang_imports(std::path::Path::new(&source_path)) {
                if let Some(asset) = slang_asset_path(&import) {
                    let _ = load_context.read_asset_bytes(asset).await;
                }
            }
            let path = load_context.path().path().to_string_lossy().replace('\\', "/");
            log::debug!("Loaded shader: {path}");
            Ok(Shader {
                path,
                spirv: Some(spirv.into()),
            })
        })
    }
}

/// Where slangc looks for an `import`: beside the importer, then the engine's and the app's
/// shader roots (the `-I` dirs [`compile_slang`] passes).
fn slang_search_dirs(importer: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if let Some(dir) = importer.parent() {
        dirs.push(dir.to_path_buf());
    }
    dirs.push(std::path::PathBuf::from(format!("{}/shaders", crate::assets::AURORA_ASSET_DIR)));
    dirs.push(bevy::asset::io::file::FileAssetReader::get_base_path().join("assets/shaders"));
    dirs
}

/// The files `source` imports, transitively (`import a.b;` is `a/b.slang`).
fn slang_imports(source: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut found: Vec<std::path::PathBuf> = Vec::new();
    let mut todo = vec![source.to_path_buf()];
    while let Some(file) = todo.pop() {
        let Ok(text) = read_to_string(&file) else {
            continue;
        };
        for line in text.lines() {
            let line = line.trim_start();
            let line = line.strip_prefix("__exported ").unwrap_or(line);
            let Some(module) = line.strip_prefix("import ").and_then(|m| m.split(';').next())
            else {
                continue;
            };
            let relative = format!("{}.slang", module.trim().replace('.', "/"));
            let Some(path) = slang_search_dirs(&file)
                .into_iter()
                .map(|dir| dir.join(&relative))
                .find(|path| path.exists())
            else {
                continue;
            };
            if path != source && !found.contains(&path) {
                found.push(path.clone());
                todo.push(path);
            }
        }
    }
    found
}

/// A shader file on disk as an asset path: `aurora://` under the engine's assets, the default
/// source under the app's.
fn slang_asset_path(file: &std::path::Path) -> Option<bevy::asset::AssetPath<'static>> {
    let engine = std::path::Path::new(crate::assets::AURORA_ASSET_DIR);
    if let Ok(rel) = file.strip_prefix(engine) {
        return Some(crate::assets::aurora_asset(&rel.to_string_lossy()));
    }
    let app = bevy::asset::io::file::FileAssetReader::get_base_path().join("assets");
    let rel = file.strip_prefix(&app).ok()?;
    Some(bevy::asset::AssetPath::from(rel.to_path_buf()).into_owned())
}

/// Compiles a `.slang` module to SPIR-V with the Vulkan SDK's `slangc` (`$VULKAN_SDK/bin`, else
/// `PATH`). Every `[shader("...")]` entry point in the file lands in the one module under its
/// own name, so a file with several entries is one asset and one compile. Modules reference
/// buffers through raw pointers (`SPV_KHR_physical_storage_buffer`).
pub fn compile_slang(source_path: &str) -> Result<Vec<u8>, ShaderLoaderError> {
    let slangc = std::env::var_os("VULKAN_SDK")
        .map(|sdk| std::path::PathBuf::from(sdk).join("bin").join("slangc"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("slangc"));
    // slangc has no stdout mode; round-trip through a per-process temp file.
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let out_dir = std::env::temp_dir().join("aurora-slang");
    std::fs::create_dir_all(&out_dir)?;
    let out_path = out_dir.join(format!(
        "{}-{}.spv",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let output = std::process::Command::new(&slangc)
        .arg(source_path)
        .args(["-target", "spirv", "-profile", "spirv_1_6", "-O2"])
        .arg("-fvk-use-entrypoint-name")
        // Engine modules (`import procedural;`) resolve from any asset source, and the app's
        // own shader root so its modules import across directories (`import planet.noise;`).
        .arg("-I")
        .arg(format!("{}/shaders", crate::assets::AURORA_ASSET_DIR))
        .arg("-I")
        .arg(
            bevy::asset::io::file::FileAssetReader::get_base_path()
                .join("assets/shaders")
                .display()
                .to_string(),
        )
        .arg("-o")
        .arg(&out_path)
        .output()?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&out_path);
        return Err(ShaderLoaderError::Slang {
            path: source_path.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let spirv = std::fs::read(&out_path)?;
    let _ = std::fs::remove_file(&out_path);
    Ok(spirv)
}

pub struct ShaderPlugin;

impl Plugin for ShaderPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<crate::shader::Shader>();
        app.init_asset_loader::<crate::shader::ShaderLoader>();
    }
}
