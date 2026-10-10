fn main() {
    // shaderc's libshaderc_shared.so.1 lives in $VULKAN_SDK/lib; rpath it so the examples run
    // without LD_LIBRARY_PATH (bevy_aurora's own rpath does not reach this crate's binaries).
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sdk}/lib");
    }
}
