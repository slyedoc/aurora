# Reference mode: implementation notes

Plan: `reference-mode.html`. Newest last.

## Log

- 2026-10-15: implemented. `src/reference.rs` (per-view accumulation, reset key, capture
  controls), raygen `accumulate` entry (dispatched after gi_shade, its own SBT region and stack),
  `AuroraDebugView::Reference` / `ReferenceDiff` (quad.slang 13 / 14, the buffer through the Cost
  view's slot), the composite's `capture` pointer (RR's output in nits per output pixel),
  `src/util/capture.rs` (headless capture runner, PFM writer), `scripts/refcmp.py`, and
  `--capture DIR --size WxH --frames N --noisy K --settle F` in `sun_room` and `many_lights`.
  Resets are logged at debug level with the reason (`bevy_aurora::reference`).
- First capture, sun room 960x540 (trace 557x313), 256 frames, in 3.3 s: reference relative
  standard error ~2.5 %; noisy input 95 % black pixels, no clumping at 1 spp (own candidates); RR
  output ratio 0.967, block error 0.087 against the reference. The room's light source is the
  sunlit patch, which a bounce finds ~3 % of the time: nearly every sample is 0 and a few huge.

- 2026-10-15: convergence and SHaRC, measured with references (sun room, 960x540):
  - The tracer converges at 1/sqrt(N) (error against a 4096-frame reference 0.194 / 0.100 / 0.042
    at 64 / 256 / 1024 frames); the 64-slice blue noise does not stall it (white noise:
    0.218 / 0.106 / 0.047). The slow convergence is per-sample variance: the room's light comes
    off the sunlit patch, which a bounce rarely finds (95 % of 1-spp pixels black; many pixels
    need three bounces, so 16 spp is still 79 % black).
  - SHaRC is not biased: its reference against the no-cache one, ratio 0.998, block error 0.022
    (GI off against GI on: 1.001 / 0.023, the noise floor). It cuts black input pixels 95 -> 61 %.
    The "29 % dark" of the earlier sweep compared RR's output with RR's output of a 16-spp run.
  - Ray Reconstruction is what darkens: its output against the true reference reads 0.85 at 1 spp,
    0.95 at 4 spp and 0.71 at 16 spp (also without GI). Non-monotonic in spp, which points at what
    we feed it rather than at the input's sparsity alone.
  - Blue noise: the first bounce took one pair for both the diffuse direction and the specular half
    vector; now an independent second pair (32 slices on). First-bounce stratification is not the
    issue (16 spp identical without it).
  - SHaRC restored (off by default) for this.

## Deviations

- The accumulation is a raygen entry, not a compute kernel: the raygen already has the noisy colour
  image bound, so no new descriptor-heap mapping. A reset is a buffer fill from Rust; the shader
  only adds.
- The buffers are host-visible (`GpuToCpu`) and read by mapping, not copied through a staging
  buffer: a capture pauses accumulation for a few frames and reads. Writes to host memory are
  slow, which only the debug mode pays.
- Headless the way aurora's own tests run (tests/windowless.rs): `AuroraDefaultPlugins` without
  `WinitPlugin`, the first camera pointed at `camera_target_placeholder(size)`, and the capture
  driving `app.update()` itself, so the capture sequence is straight-line code, not a per-frame
  state machine.
- The reference's convergence is estimated from the noisy frames' per-pixel spread over sqrt(N)
  rather than by comparing halves of the accumulation (one buffer, no second).
- `--path` (moving captures) not done yet.

## Untested

- Northshire through zero (it needs the same few lines in its examples).
- `ReferenceDiff` view by eye.

- 2026-10-15: blue noise removed from the path tracer (BlueNoisePlugin, the STBN textures, the
  push-constant field); every dimension is the PCG stream. Two bugs it caused: each pixel repeated
  the same 64 first-bounce directions (grain stuck), and its bytes put every pixel on one shared
  256x256 direction lattice, which projected the window onto the sun room's west wall (a CPU
  render of the flat-sky, one-bounce room has no such patch). The DLSS-RR Integration Guide (3.5)
  asks for white noise from a good hash and no pattern shared across the screen. Sun room
  1920x1080: reference unchanged (ratio 1.000, block error 0.007); RR output ratio 0.58 -> 0.67,
  block error 0.22 -> 0.15, frame-to-frame change 0.058 -> 0.038.
