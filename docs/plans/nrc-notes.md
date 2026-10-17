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

- Sun room against Cycles (Blender MCP, the matching scene: flat sky 6000, no sun, no ground;
  960x540; Cycles 16k samples; room only, window masked), ratio / block error:
  | aurora | vs Cycles 2 bounces | vs Cycles 32 bounces |
  |---|---|---|
  | truth (no PT), 4 bounces | 1.039 / 0.062 | 0.955 / 0.093 |
  | truth, 16 bounces | 1.143 / 0.214 | 1.052 / 0.073 |
  | default (PT) | 1.039 / 0.063 | 0.955 / 0.093 |
  | NRC | 1.144 / 0.216 | 1.052 / 0.076 |
  Cycles 14 bounces = 32. NRC against aurora's 16-bounce truth: 1.000 / 0.012 (default 0.987 /
  0.080: its 4 bounces), black input 96 -> 27 %, RR 0.048 -> 0.044. Self-training makes NRC the
  all-bounce answer; aurora's converged image is 5 % above Cycles' (the materials, known). So the
  "sun room reads 1.014-1.018" above was the 4-bounce truth's missing light: judge NRC against a
  16-bounce truth or Cycles, and the scaled (unclipped) gradient stands.

- Cost (sun room, 3840x1080 DLAA, `--capture DIR --frames 0 --time 600`: util/capture.rs times every
  profiler scope with the accumulation off). Trace: NRC off 17.3 ms, on 55.4. With the network's
  answer stubbed (same paths and code, no matrix products) 16.4-17.5: running inference was the
  38 ms (the NRC-off figure still carries the compiled-in network, below). Weights read RowMajor; converted each frame
  (vkCmdConvertCooperativeVectorMatrixNV, nrc.rs `record_convert`) into InferencingOptimal: trace
  27.3 ms (inference 10 ms), image unchanged (reference 1.001 / 0.011 against row-major). Asking
  after the loop, the warp converged, made it worse both times (row-major 116 ms, optimal 65): the
  pending inputs live across every TraceRay. gi spatial doubles with NRC (2.5 -> 5.3 ms) with the
  answer stubbed too: more live samples to reuse (black input 90 -> 30 %), not the network.
- The inline network's code costs the trace even with NRC off (sun room, 3840x1080, `--no-nrc`,
  trace ms, two interleaved runs each): the raygen as is (Vulkan memory model, the NRC block, row-major)
  29.9 / 28.1, optimal layout 25.4 / 24.5; the NRC block removed, the Vulkan model kept by a dead
  CoopVec op 7.11 / 7.11; no NRC import (GLSL450) 7.04 / 7.03. The memory model is 1 %; the block
  compiled in (its matrix products: registers or stack for the whole raygen, likely) is 3.5x. The
  Northshire 2.63 -> 5.0 ms NRC-off trace is this. Inference belongs in a compute pass (4M queries
  0.41 ms there, optimal layout, measured by a throwaway test of nrc_eval) with no CoopVec left in
  the raygen.
- Under motion (sun room, `--motion 1.5 / 4`): no haze or brightening; RR 1.005-1.013 / block
  0.027-0.037 with NRC, 1.003-1.010 / 0.037-0.038 without, input unbiased. The reported 7-18 % is
  not in this scene (Northshire: anchor snaps, fog, distance).

- Inference out of the raygen (NVIDIA's NRC library's shape, RTXGI NrcGuide: query records,
  QueryAndTrain, Resolve). The raygen imports nrc_record.slang (no CoopVec: back to the GLSL450
  memory model, no network code). A path ending in the cache writes its input to its pixel and
  sample's slot of `NrcFrame::queries`; after the sample, the route (pixel, or its ReSTIR PT
  sample's light past x_k) and the f16 weight (mask / SAMPLES, or mask over the weight at x_k). A
  training path's tail takes an extra slot (counter[1]); its records keep what the path gathered
  and the tail's weight over theirs. `nrc_infer` (compute, optimal layout, every slot so the warp
  stays whole) writes the answers; `nrc_resolve` (raygen, per pixel) adds them to dlss_color
  (through the atmosphere) or the candidate's L before gi candidates; `nrc_targets` finishes the
  training targets before training. Trace ms, 3840x1080 (two runs each):
  | | NRC on before | on now | off before | off now |
  |---|---|---|---|---|
  | sun room trace | 27.3 (row-major 55) | 9.3, 8.8 | 17-25 | 8.8, 8.3 |
  | Northshire lane trace | 34.9, 35.0 | 10.9, 10.3 | 13.6, 13.7 | 10.3, 9.6 |
  | Northshire lane busy | 49.8, 49.9 | 28.0, 26.5 | 27.2, 27.3 | 25.9, 23.9 |
  Inference + resolve 0.9 ms, training 0.45. Unchanged images: sun room reference 0.999 / 0.009
  against inline, `--samples 2` 1.000 / 0.008; Northshire lane 1.000 / 0.005 (RR 1.012 / 0.018 both);
  GI off against a 16-bounce truth 1.013 / 0.054 and 1.009 / 0.044 (inline 1.014 / 0.055: GI off,
  the cache's regional bias wanders 5-15 % run to run, before and after).
- Training paths inside the main dispatch cost nothing measurable (train_fraction 0: trace 8.6-9.0
  vs 8.8), so no separate low-resolution update pass as NVIDIA's. Training recorded before ReSTIR
  GI's ray tracing passes with compute-only barriers did not overlap them (timeline: gi spatial
  starts when training ends); overlap needs the compute queue.

- Position by a multiresolution hash grid (Müller et al. 2022, Instant-NGP, whose NRC task used
  it: sharper than the triangle wave for +0.7 ms at 1080p), world space, no anchor: Instant-NGP's
  NRC configuration, 16 levels x 2 features, 2^15 entries per level, cells doubling from
  `nrc_grid_extent` / 2^19 to / 16 (default 2048 m: 4 mm to 128 m), and its network, 2 hidden
  layers (the grid holds the detail). Training scatters the input gradient into the tables
  (float atomic adds, VK_EXT_shader_atomic_float; wave-summed where a wave shares an entry: made no
  difference, it is volume) and a sparse Adam updates the touched entries. Inference skips the
  lookups for empty slots (they still join the warp's products: 2.1 -> 0.8 ms). Gym: loss 5.2 ->
  0.005 at 2^17 and 5 hidden layers, 95 -> 0.032 at 2 (its field leans on direction).
  | sun room against the 16-bounce truth (GI on) | ratio / block | RR |
  |---|---|---|
  | triangle wave, anchored box | 1.004 / 0.021 | 1.007 / 0.033 |
  | grid 2^17, 5 hidden layers | 1.001 / 0.011 | 1.005 / 0.037 |
  | grid 2^15, 2 hidden layers (default) | 1.002 / 0.013 | 1.004 / 0.031 |
  Northshire lane static: unchanged (1.000 / 0.004 against the triangle wave's reference; RR 1.011 /
  0.017). Cost, 3840x1080: Northshire inference 0.68 + resolve 0.22 ms (triangle wave 0.88
  together), training 1.14 (0.45; the scatter is ~0.7 of it). At 2^17 / 5 hidden, before the empty
  skip: inference 3.1, training 1.39. Motion reads better by eye (no anchor to snap).
- Grid gradient scatter: where a wave shares an entry (coarse levels) one f32 add of the wave's
  sum; elsewhere one f16x2 add of both features (VK_NV_shader_atomic_float16_vector, Slang
  `InterlockedAddF16x2`; an entry there sums a few rows, which f16 holds); nrc_grid_adam adds the
  two. Training 1.14 -> 0.67 ms (Northshire lane, 3840x1080), sun room 1.2 -> 0.71; sun room
  against the truth 1.002 / 0.012 (f32: 0.013), gym loss 0.0326 (0.032). Validation clean.

## Deviations

- D5: aurora traces in world space (not camera-relative, the plan assumed solari's convention), so
  nothing rebases. The fixed 512 m box was too coarse (sun room above); the box became a
  contraction around an anchor snapped near the camera (Mip-NeRF 360, L-infinity), then the hash
  grid replaced both (world space, no anchor).
- D2: self-training (paper sec. 5) added: every path's last vertex past x_k ends in the cache,
  so targets carry every bounce. Without it records near max_bounces taught "direct light only"
  and indirect-lit surfaces read 7 % dark.
- D3: position by triangle wave x 12 (tiny-cuda-nn's NRC encoding), not sin/cos x 6; since
  replaced by Instant-NGP's hash grid (log above).
- Gates: the plan's truth (4 bounces) under-reads an all-bounce cache; NRC is judged against a
  16-bounce truth and Cycles (sun room).
