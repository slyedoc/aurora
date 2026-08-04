fn main() {
    // The shaderc crate links against libshaderc_shared.so.1 from the Vulkan SDK,
    // but the SDK's setup-env.sh only adds $VULKAN_SDK/lib/VulkanLoader/lib to
    // LD_LIBRARY_PATH -- not $VULKAN_SDK/lib, where shaderc lives. Bake the path
    // in as an rpath so binaries and examples run without extra env setup.
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{sdk}/lib");
    }
}
