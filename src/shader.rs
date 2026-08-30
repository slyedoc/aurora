use ash::vk;
use std::{borrow::Cow, cell::RefCell, fs::read_to_string, rc::Rc};
use thiserror::Error;

use bevy::{asset::AssetLoader, prelude::*, tasks::ConditionalSendFuture};

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse shader: {0}")]
    Parse(#[from] std::string::FromUtf8Error),
    #[error("Could not compile shader: {0}")]
    Compile(#[from] shaderc::Error),
    #[error("slangc failed for {path}:\n{stderr}")]
    Slang { path: String, stderr: String },
}

#[derive(TypePath)]
pub struct ShaderLoader {
    compiler: shaderc::Compiler,
}

impl Default for ShaderLoader {
    fn default() -> Self {
        Self {
            compiler: shaderc::Compiler::new().unwrap(),
        }
    }
}

#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    pub path: String,
    pub spirv: Option<Cow<'static, [u8]>>,
    #[dependency]
    pub dependencies: Vec<Handle<Shader>>,
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ();
    type Error = ShaderLoaderError;

    fn extensions(&self) -> &[&str] {
        &[
            "vert", "frag", "comp", "rgen", "rint", "rchit", "rmiss", "slang",
        ]
    }

    fn load(
        &self,
        reader: &mut dyn bevy::asset::io::Reader,
        _settings: &Self::Settings,
        load_context: &mut bevy::asset::LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>> {
        Box::pin(async move {
            let ext = load_context.path().get_extension().unwrap().to_string();
            let ext = ext.as_str();
            let path = load_context.path().to_string();
            // On windows, the path will inconsistently use \ or /.
            // TODO: remove this once AssetPath forces cross-platform "slash" consistency. See #10511
            let path = path.replace(std::path::MAIN_SEPARATOR, "/");
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes).await?;

            if ext == "glsl" {
                return Ok(Shader {
                    path: load_context.path().path().to_str().unwrap().to_string(),
                    spirv: None,
                    dependencies: Vec::new(),
                });
            }

            if ext == "slang" {
                let source_path = format!(
                    "{}/{}",
                    crate::assets::AURORA_ASSET_DIR,
                    load_context.path().path().display()
                );
                let spirv = compile_slang(&source_path)?;
                log::info!("Loaded shader: {path} (slang)");
                return Ok(Shader {
                    path: load_context.path().path().to_str().unwrap().to_string(),
                    spirv: Some(spirv.into()),
                    dependencies: Vec::new(),
                });
            }

            let kind = match ext {
                "vert" => shaderc::ShaderKind::Vertex,
                "frag" => shaderc::ShaderKind::Fragment,
                "comp" => shaderc::ShaderKind::Compute,
                "rgen" => shaderc::ShaderKind::RayGeneration,
                "rint" => shaderc::ShaderKind::Intersection,
                "rchit" => shaderc::ShaderKind::ClosestHit,
                "rmiss" => shaderc::ShaderKind::Miss,
                _ => panic!("Unsupported shader extension: {}", ext),
            };

            let mut options = shaderc::CompileOptions::new().unwrap();
            options.set_target_env(shaderc::TargetEnv::Vulkan, vk::make_api_version(0, 1, 3, 0));
            options.set_target_spirv(shaderc::SpirvVersion::V1_6);
            options.set_generate_debug_info();
            options.set_optimization_level(shaderc::OptimizationLevel::Performance);

            let load_context = Rc::new(RefCell::new(load_context));
            let load_context_copy = load_context.clone();
            let dependencies = Rc::new(RefCell::new(Vec::new()));
            let dependencies_copy = dependencies.clone();

            options.set_include_callback(move |fname, _type, _, _depth| {
                // Includes live next to the engine's shaders, wherever the crate is checked out.
                let full_path = format!("{}/shaders/{}", crate::assets::AURORA_ASSET_DIR, fname);
                let Ok(contents) = read_to_string(&full_path) else {
                    return Err(format!("Failed to read shader include: {full_path}"));
                };

                dependencies_copy.borrow_mut().push(
                    load_context_copy
                        .borrow_mut()
                        .load::<Shader>(crate::assets::aurora_asset(&format!("shaders/{fname}"))),
                );

                Ok(shaderc::ResolvedInclude {
                    resolved_name: fname.to_string(),
                    content: contents,
                })
            });

            let binary_result = self.compiler.compile_into_spirv(
                std::str::from_utf8(&bytes).unwrap(),
                kind,
                path.as_str(),
                "main",
                Some(&options),
            );

            let Ok(binary) = binary_result else {
                let e = binary_result.err().unwrap();
                return Err(ShaderLoaderError::Compile(e));
            };

            let dependencies = dependencies.borrow().clone();

            let shader = Shader {
                path: load_context
                    .borrow()
                    .path()
                    .path()
                    .to_str()
                    .unwrap()
                    .to_string(),
                spirv: Some(Vec::from(binary.as_binary_u8()).into()),
                dependencies,
            };

            log::info!("Loaded shader: {:?}", shader.path);
            Ok(shader)
        })
    }
}

/// Compiles a `.slang` module to SPIR-V with the Vulkan SDK's `slangc` (`$VULKAN_SDK/bin`, else
/// `PATH`). Every `[shader("...")]` entry point in the file lands in the one module under its
/// own name, so a kernel file with several entries is one asset and one compile.
///
/// Slang is the engine's shader language going forward; the GLSL stages above are the legacy
/// path. Modules reference buffers through raw pointers in push constants
/// (`SPV_KHR_physical_storage_buffer`), so they need no descriptor sets.
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

        app.add_systems(Update, reload_modified);
    }
}

fn reload_modified(
    shaders: Res<Assets<Shader>>,
    asset_server: Res<AssetServer>,
    mut shader_events: MessageReader<AssetEvent<Shader>>,
) {
    for event in shader_events.read() {
        match event {
            AssetEvent::Modified { id } => {
                for (parent_id, shader) in shaders.iter() {
                    if shader.dependencies.iter().any(|dep| dep.id() == *id) {
                        asset_server.reload(asset_server.get_path(parent_id).unwrap());
                    }
                }
            }
            _ => {}
        }
    }
}
