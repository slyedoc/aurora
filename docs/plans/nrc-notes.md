# Neural Radiance Cache: implementation notes

Plan: `nrc.html`. Newest last.

## Log

- 2026-10-16: plan written against 2c3017f. Driver 615.71.09 (July's fault was on 580).
- Device: `VK_NV_cooperative_vector` + `VK_EXT_shader_replicated_composites` (Slang lowers a
  `CoopVec(x)` splat to a replicated composite) when present (`render_device::cooperative_vector()`),
  and `vulkanMemoryModel` + device scope always: a module using CoopVec declares the Vulkan memory
  model (July's lesson: without device scope, Xid 32). Slang's pointer overloads:
  `coopVecMatMulAdd<half, 64>(v, Float16, (void*)w, Float16, (void*)b, Float16, RowMajor, false,
  stride_bytes)`; `coopVecLoad<64, half>(half* p, byte_offset)`; `v.store(p, byte_offset)`.
- Phase 1 (gym, `tests/nrc_gym.rs`, `cargo test --release --test nrc_gym`): forward and backward
  per record with cooperative vectors (backward through a transposed f16 mirror, row-major; no
  optimal layouts), weight gradients by a shared-memory reduction over 256-row chunks (f32, no
  float atomics), Adam per parameter writing the f32 master and both f16 copies. Against a CPU
  reference from the GPU's own encodings: forward 3.9e-3, activations 3.8e-3 (relative), loss
  gradient 4.6e-4, backward 6.2e-4 (of the terms' size), weight gradients 9.6e-8 of the largest.
  Fit of a synthetic field (4 coloured blobs x directional lobes x albedo, fresh records every
  step, batch 4096, lr 2e-3): relative L2 4.83 -> 0.0056 in 5000 steps (July: 0.037), 13,100
  steps/s including record generation (July: 234). Certified.
- f16 overflow: loss scale 128 with a 3e4 output-gradient clamp overflowed backward through Wᵀ
  (65504). Now loss scale 1 and the output gradient clipped at +-1024.
- Phase 0: the raygen asks the network at every indirect vertex (`nrc_inline`, answer x 0 so it
  stays live; `ab` 0 stubs the query). Sun room 1080p, 2048 frames, live: no fault. Importing nrc
  switches the whole raygen module to the Vulkan memory model.

- Position encoding: July's 6 sin/cos octaves over a 512 m box resolve ~32 m. tiny-cuda-nn's NRC
  encoding instead: a triangle wave at 12 frequencies (still 62 inputs). Its lowest period must be
  2, not 1: every triangle is symmetric about its peaks, and with period 1 x and 1 - x encoded the
  same (gym: loss stuck at 3.7; sin/cos survive it through the antisymmetric cosine). Gym: 0.0063.
- Phase 2/3 (inline): training paths (`nrc_train_fraction`, capped near 64k records) record every
  vertex (packed NrcRecord into a ring, atomic slot) and at the path's end its target, the radiance
  gathered after the vertex over its mask, pre-exposed (x input_exposure; the answer is divided by
  it). Training after the trace: 4 steps of 16k rows drawn at random (with replacement) from the
  frame's records. Spread rule as the paper; termination before the vertex's next-event estimate;
  never at or before x_k, never while PT searches.
- Sun room (960x540, --ev100 9, truth = no PT, 4 bounces, 4096 frames):
  | run | bias vs truth (ratio, block) | black | RR error (own ref) |
  |---|---|---|---|
  | spatial PT (today) | 1.000 / 0.010 | 90 % | 0.037 |
  | NRC, 512 m box | 1.015 / 0.089 (room +7-20 %, sun patch dark: blurred) | 27 % | 0.025 |
  | NRC, 32 m box | 0.996 / 0.028 (floor -7 %) | 29 % | 0.037 |
  | NRC, 32 m box, self-training | 1.002 / 0.017 | 26 % | 0.039 |
  | NRC + temporal (pt_history 8) | 1.001 / 0.015 | 3 % | 0.018 |
  | spatial + temporal PT, no NRC | 1.000 / 0.009 | 4 % | 0.019 |
  RR fed the converged image: 0.004 (its floor), so the rest is input noise: light that arrives by
  chance (a bounce off the shaded floor landing on the sun patch), which a cache does not remove.
  A 16-bounce truth is 1.5 % brighter than the 4-bounce one; NRC with self-training reads 1.3 %
  under it (block 0.064).
- Phase 0 soak #1 was void: the running app hot-reloaded raygen edits made meanwhile (a new
  NrcFrame layout against the old binary). Never edit shaders during a soak.

- 2026-10-16 (later): Slang 2026.19 (standalone, ~/.local/share/slang; shader.rs picks the newest
  slangc among the SDK's and PATH's). 2026.12 made CoopVec differentiable; matrix products are
  still not (layout-dependent): each layer is `coopVecMatMulAdd` with a `[BackwardDerivative]`
  rule (Slang's examples/mlp-training-coopvec), the loss and ReLUs are `bwd_diff`. The backward
  reads the same weights `ColumnMajor` (the transposed f16 copy is gone). Gym unchanged (loss
  gradient 3.8e-4, backward 4.0e-4, weight gradients 8.0e-8). Timing, lane 3840x1080 (busy /
  trace, ms; two runs each): 2026.8 off 10.7, 10.4 / 5.35, 5.20, on 21.3, 21.2 / 15.5, 15.4, training
  0.41; 2026.19 same shaders off 10.3, 10.1 / 5.14, 5.05, on 21.1, 21.1 / 15.4, 15.3, training 0.41;
  autodiff port off 10.4, 10.2 / 5.23, 5.11, on 21.3, 21.3 / 15.5, 15.5, training 0.40. Gym 13.3-13.5k
  steps/s throughout. Neither the compiler nor the port moves the cost; inline inference is +10 ms.
- The output-gradient handling trades scenes: clipped at +-1024 (loss scale 1) the sun room reads
  1.002 / 0.016 and the abbey 0.955 (rare bright targets dropped); scaled by 1/16 unclipped it reads
  1.014-1.018 / 0.08-0.10 and the abbey 1.009 (they kick the network up). The relative L2 gradient
  2 (p - t) / (p^2 + 0.01) explodes on a dim prediction; epsilon is in pre-exposed units. Open.

## Deviations

- D5: aurora traces in world space (not camera-relative, the plan assumed solari's convention), so
  nothing rebases. The fixed 512 m box was too coarse (sun room above); the box became a
  contraction around an anchor snapped near the camera (Mip-NeRF 360, L-infinity): full resolution
  within 16 m, coarser beyond, all of space inside. The hash grid remains the upgrade.
- D2: self-training (paper sec. 5) added: every path's last vertex past x_k ends in the cache,
  so targets carry every bounce. Without it records near max_bounces taught "direct light only"
  and indirect-lit surfaces read 7 % dark.
- D3: position by triangle wave x 12 (tiny-cuda-nn's NRC encoding), not sin/cos x 6.
- Gates are read against the plan's truth (4 bounces, as shipped); the 16-bounce truth is reported
  beside it.
