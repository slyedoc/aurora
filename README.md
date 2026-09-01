## Aurora — hardware ray tracing for Bevy on raw Vulkan (WIP 🔨)

`bevy_aurora` is a custom rendering backend for Bevy that leverages hardware raytracing using Vulkan
(ash, no wgpu). It started as a fork of [HugoPeters1024/bevy_vulkan](https://github.com/HugoPeters1024/bevy_vulkan).
You will need a GPU that supports `VK_KHR_ray_tracing`. A non exhaustive list of supported device
can be found on [gpuinfo.org](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_KHR_ray_tracing&platform=all)

The models required to run some of the examples are not here because they are too big for git. You can download them
from https://hugopeters.me/public/models.zip. Extract them and put them in `./assets/models`.

#### Important

Targets the `aurora` branch of [slyedoc/bevy](https://github.com/slyedoc/bevy/tree/aurora) (upstream main + render-free
`bevy_feathers`, the `.bsn` loader and `bevy_feathers_inspector`), Rust edition 2024, stable toolchain.

## Required packages

Besides the rust toolchain, you will need to:

1. Follow the Bevy [installation guide](https://bevyengine.org/learn/quick-start/getting-started/setup/#installing-os-dependencies)
2. Install the [vulkan-sdk](https://www.lunarg.com/vulkan-sdk/)
3. An NVIDIA RTX GPU with a recent driver — DLSS Ray Reconstruction is the renderer's only
   denoiser, so it is required, not optional.
4. The DLSS SDK: clone [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) and export `DLSS_SDK`
   pointing at it (e.g. `export DLSS_SDK=$HOME/DLSS`) before building. Without it DLSS
   compiles out and every example refuses to start.

## Screenshots

<table id="example-table">
  <tbody>
    <tr>
      <td>
        <img src='./screenshots/sponz_glass_cuboid.png'/>
      </td>
      <td>
        <img src='./screenshots/sponz_glass_sphere.png'/>
      </td>
    </tr>
    <tr>
      <td>
        <img src='./screenshots/san_miquel.png'/>
      </td>
      <td>
        <img src='./screenshots/bisto_exterior.png'/>
      </td>
    </tr>
    <tr>
      <td>
        <img src='./screenshots/bistro_exterior_night.png'/>
      </td>
      <td>
        <img src='./screenshots/spheres.png'/>
      </td>
    </tr>
  </tbody>
</table>


## VR — a Quest over WiVRn

Any aurora app renders into a headset when built with the `xr` feature (per-eye stereo,
head-tracked; the window becomes a spectator view). The baked scenes live in
[aurora_files](https://github.com/slyedoc/aurora_files), whose `bsn` viewer forwards the
feature — `cargo run --release -p bsn --features xr -- bistro/bistro.bsn` walks the
path-traced bistro in a Quest 2/3, streamed from the PC over Wi-Fi with
[WiVRn](https://github.com/WiVRn/WiVRn) (OpenXR, no SteamVR involved). One-time setup:

1. **PC — install the WiVRn server** (Flathub; the client and server must be the same WiVRn
   version, and the flatpak tracks the store client's releases):

   ```sh
   flatpak install flathub io.github.wivrn.wivrn
   ```

2. **PC — firewall**, if one is on: WiVRn streams on 9757 (TCP+UDP) and discovers over
   mDNS (5353/UDP):

   ```sh
   sudo ufw allow 9757 && sudo ufw allow 5353/udp
   ```

3. **Headset — install the WiVRn client**: search "WiVRn" in the Meta Horizon Store (free,
   official). Sideloading an APK is not needed.

4. **Pair**: launch "WiVRn server" on the PC (it has a first-run wizard), open WiVRn in the
   headset — the PC appears in its list via auto-discovery (same network; headset on 5 GHz
   Wi-Fi, PC ideally wired) — hit Connect and enter the PIN the dashboard shows.

5. **Run it** — with the headset showing "Connection ready", from the `aurora_files` repo:

   ```sh
   cargo run --release -p bsn --features xr -- bistro/bistro.bsn
   ```

   Fly with WASD, and your head does the rest. Any example in this repo works the same way
   with `cargo run --release --features xr --example <name>`.

Troubleshooting:

- *Headset says connection refused*: the server app isn't running on the PC.
- *The app starts flat / can't find OpenXR*: WiVRn registers itself as the active
  OpenXR runtime when the headset connects; connect first. If it still isn't found, point
  the loader at the flatpak's manifest explicitly:

  ```sh
  export XR_RUNTIME_JSON=~/.local/share/flatpak/app/io.github.wivrn.wivrn/current/active/files/share/openxr/1/openxr_wivrn.json
  # system-wide flatpak installs: same path under /var/lib/flatpak/app/...
  ```

- *Double vision / eye strain*: the frame rate is below the headset's refresh rate. Use
  `--release`, and cycle the DLSS mode towards Performance with `F3`.

## Examples

run `cargo run --example` to get a list of available examples.

`cargo run --example feathers_ui` shows `bevy_feathers` widgets (buttons, checkbox, toggle,
slider, text) drawn by this crate's own Vulkan UI pass (`src/ui_render.rs`): the app uses the
`bevy_feathers_core` feature, so bevy lays out and picks the UI and no `bevy_ui_render`/wgpu
code runs.

Every example has the dev panel (`DevUIPlugin`): a `bevy_feathers_inspector` card over the
renderer's tunables (gamma, exposure, aperture, fog, sky) plus fps. `F2` toggles it, `F1` opens
the world inspector, `Space` toggles accumulation.

This rendering backend integrates seamlessly with Bevy, as a result, the code needed to run a simple scene is extremely simple:

```rust
use bevy::prelude::*;
use bevy_aurora::{
    debug_camera::{DebugCamera, DebugCameraPlugin},
    dev_shaders::DevShaderPlugin,
    dev_ui::DevUIPlugin,
    gltf_mesh::{GltfModel, GltfModelHandle},
    ray_default_plugins::RayDefaultPlugins,
    ray_render_plugin::RenderConfig,
    sphere::Sphere,
};

fn main() {
    let mut app = App::new();
    app.add_plugins(RayDefaultPlugins);
    app.add_plugins(DevShaderPlugin);
    app.add_plugins(DevUIPlugin);
    app.add_plugins(DebugCameraPlugin);
    app.add_systems(Startup, setup);
    app.run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(4.0, 1.8, 0.0).looking_at(Vec3::new(4.0, 1.8, 0.0), Vec3::Y),
        DebugCamera::default(),
    ));

    commands.spawn((
        GltfModelHandle(asset_server.load::<GltfModel>("models/sponza.glb")),
        Transform::from_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 * 0.0))
            .with_scale(Vec3::splat(0.012)),
    ));

    // glowing sphere
    commands.spawn((
        Transform::from_translation(Vec3::new(0.0, 1.5, 0.0)),
        Sphere,
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            emissive: LinearRgba::new(10.0, 7.0, 5.0, 1.0),
            ..default()
        })),
    ));
}
```
