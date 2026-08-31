# VR / OpenXR Plan

Goal: walk around Bistro in VR on a Quest 2, path-traced, streamed from the Linux desktop.

State of the ecosystem as of 2026-08-31.

## Decision summary

- **Runtime: WiVRn** (Monado-based OpenXR runtime that streams to the Quest over wifi). Our app speaks OpenXR directly to WiVRn — no SteamVR, no ALVR.
- **API: the `openxr` crate ([Ralith/openxrs](https://github.com/Ralith/openxrs)) directly**, with `XR_KHR_vulkan_enable2`. Not `bevy_mod_openxr`.
- **Standalone on-device is a non-starter**: Quest 2 is Adreno — no RT hardware, no NGX. This is PCVR streaming only.

## Why not the Bevy community XR stack

- Bevy core still has no XR. [bevy#115](https://github.com/bevyengine/bevy/issues/115) (open since 2020, `O-XR`) is the tracking issue; the stance remains "external crate". Old [PR #2319](https://github.com/bevyengine/bevy/pull/2319) is long dead. Cited blockers: fallible plugins ([#2337](https://github.com/bevyengine/bevy/issues/2337)), app lifecycle ([#2432](https://github.com/bevyengine/bevy/issues/2432)).
- The living community stack is [awtterpip/bevy_oxr](https://github.com/awtterpip/bevy_oxr): `bevy_mod_xr` (abstraction — XR schedules, tracked views/spaces) + `bevy_mod_openxr` (backend). Actively maintained: 0.6.0 in June 2026 against Bevy 0.19, Vulkan default, Android/Quest builds, `fb_passthrough`.
- **But `bevy_mod_openxr` is welded to `bevy_render`/wgpu** — most of its complexity is wgpu-hal contortions to share a Vulkan device with OpenXR. Aurora has no wgpu, so it doesn't plug in. That's fine: OpenXR's native graphics binding *is* Vulkan, so raw Vulkan is the easy case. The wgpu indirection is the very thing that makes XR in stock Bevy painful.
- `bevy_mod_xr`'s schedule/component design is still worth cribbing from if we want familiar shapes (XrTrackedView, session state schedules), but the render integration would be ours either way.

## Linux + Quest 2 streaming

- **WiVRn** is the current SOTA: v26.2 (Feb 2026) improved head/controller tracking, added SlimeVR trackers. Users report steady 90 fps on Quest 2. ([GamingOnLinux](https://www.gamingonlinux.com/2026/02/wireless-vr-streaming-levels-up-on-linux-with-the-latest-wivrn-release/))
- **ALVR is effectively deprecated on Linux** — broken with SteamVR ≥ 2.16.7; the [Linux VR Adventures wiki](https://wiki.vronlinux.org/docs/hardware/) explicitly says use WiVRn or Steam Link instead.
- WiVRn ships a client APK for the Quest (sideload via their instructions or the Meta store version), and the server registers itself as the active OpenXR runtime (`~/.config/openxr/1/active_runtime.json`).
- Monado's simulated HMD lets us test the whole session/swapchain/frame loop with **no headset attached** — do bring-up against that first.

## Integration sketch (aurora side)

The `openxr` crate is generated from the Khronos registry and has a complete
[Vulkan example](https://github.com/Ralith/openxrs/blob/master/openxr/examples/vulkan.rs) showing the whole loop.

1. **Instance/device creation order changes.** With `XR_KHR_vulkan_enable2`, the XR instance is created *first*, then `create_vulkan_instance` / `create_vulkan_device` wrap our existing creation (the runtime injects the extensions it needs, e.g. external memory for the compositor). This touches our `RenderDevice` init path.
2. **Swapchain**: OpenXR hands us `VkImage`s (typically one swapchain, 2 array layers for stereo, sRGB format the runtime lists). Our final pass (post-process / quad blit) renders into those instead of — or alongside — the window swapchain. Window can stay up as a spectator view.
3. **Frame loop**: `wait_frame` → `begin_frame` → locate views → render → `end_frame` with layer submission. Our deliberate 1-frame-in-flight design is exactly what XR latency wants; `wait_frame` becomes the pacing source instead of the swapchain acquire.
4. **Cameras**: per-eye `XrView` gives pose + *asymmetric* FOV each frame → feeds raygen directly (we build ray dirs from an inverse-projection; must handle asymmetric frusta). Two views = either 2 raygen dispatches or one dispatch writing both array layers.
5. **DLSS-RR per eye**: NGX supports multiple feature handles — one RR feature per eye at per-eye resolution. Doubles the RR cost; budget accordingly. Fallback: run RR at a modest per-eye render res (Quest 2 panel is 1832×1920/eye; render ~60-70% and let RR upscale).
6. **Input**: skip initially (head tracking only, fly with keyboard while seated). OpenXR action sets for controllers later.

Feature-gate all of it (`feature = "xr"`) so the desktop path is untouched.

## Reality check

Two eyes at 72–90 Hz with reprojection punishing every dropped frame is a very different perf envelope from a 1440p window. Bistro on the 5090 through WiVRn at a modest per-eye res with RR upscaling is plausibly in reach, but expect to drop to 72 Hz mode and lean on the ReSTIR/SHaRC work. Motion-to-photon also cares about the encode leg — WiVRn supports h264/h265/AV1; the 5090's encoder is not the bottleneck.

## First slice

1. Install WiVRn server, verify `hello_xr` (or the openxrs vulkan example) runs against the simulated HMD, then against the Quest.
2. `xr` feature: session + swapchain + frame loop, **mono** — mirror the existing camera to the HMD.
3. Stereo raygen from `XrView` poses with asymmetric projection.
4. Per-eye DLSS-RR, perf pass, 72 Hz target.
5. Controllers/input, comfort options — later.
