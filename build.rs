fn main() {
    // The shaderc crate links against libshaderc_shared.so.1 from the Vulkan SDK,
    // but the SDK's setup-env.sh only adds $VULKAN_SDK/lib/VulkanLoader/lib to
    // LD_LIBRARY_PATH -- not $VULKAN_SDK/lib, where shaderc lives. Bake the path
    // in as an rpath so binaries and examples run without extra env setup.
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sdk}/lib");
    }
    dlss();
    aftermath();
}

/// Nsight Aftermath: link the SDK's shared lib when the `dev` feature is on AND the lib
/// exists -- `cfg(aftermath_gfsdk)` gates the real FFI in `src/aftermath.rs`, so a machine
/// without the SDK compiles the no-op stubs instead of failing at link. The `-l`/`-L` ride
/// the rlib to downstream binaries; the rpath link-arg does not propagate, so binaries in
/// other workspaces carry their own (aurora_files' bsn / bevy_city build.rs).
fn aftermath() {
    println!("cargo:rerun-if-env-changed=AFTERMATH_SDK");
    println!("cargo::rustc-check-cfg=cfg(aftermath_gfsdk)");
    if std::env::var_os("CARGO_FEATURE_DEV").is_none() {
        return;
    }
    let sdk = std::env::var("AFTERMATH_SDK").unwrap_or_else(|_| {
        format!(
            "{}/nvidia/aftermath",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let lib = std::path::Path::new(&sdk).join("lib/x64");
    if !lib.join("libGFSDK_Aftermath_Lib.x64.so").exists() {
        println!(
            "cargo:warning=dev feature on but {}/libGFSDK_Aftermath_Lib.x64.so not found -- \
             GPU crash dumps compile out (set AFTERMATH_SDK)",
            lib.display(),
        );
        return;
    }
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=GFSDK_Aftermath_Lib.x64");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rustc-cfg=aftermath_gfsdk");
}

// DLSS (NGX) link wiring, additive and guarded: `DLSS_SDK` must name a tree holding
// `lib/Linux_x86_64/libnvsdk_ngx.a` (the only form of the NGX API on Linux -- a static
// archive that dlopens the driver's `_nvngx.so` and the per-feature snippets). Without it
// `cfg(dlss_ngx)` stays off and `src/dlss` compiles to a stub. The archive is named by
// `#[link]` in `src/dlss/ngx.rs`, so this only supplies the search path.
fn dlss() {
    println!("cargo:rerun-if-env-changed=DLSS_SDK");
    println!("cargo::rustc-check-cfg=cfg(dlss_ngx)");
    if !cfg!(target_os = "linux") {
        return;
    }
    let Ok(sdk) = std::env::var("DLSS_SDK") else {
        println!("cargo:warning=DLSS_SDK unset -- DLSS compiles out");
        return;
    };
    let lib_dir = format!("{sdk}/lib/Linux_x86_64");
    if !std::path::Path::new(&format!("{lib_dir}/libnvsdk_ngx.a")).exists() {
        println!("cargo:warning={lib_dir}/libnvsdk_ngx.a not found -- DLSS compiles out");
        return;
    }
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-cfg=dlss_ngx");
}
