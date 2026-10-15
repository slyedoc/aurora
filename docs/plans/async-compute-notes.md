# Async compute: implementation notes

Plan: `async-compute.html`. Notes kept while implementing; newest last.

## Log

- 2026-10-15: plan written. GPU Trace (aurora a62fe19, Northshire 3840x1080 Balanced): frame
  5.473 ms, 26.3 % SM, 20.6 warps/SM. Scene lane: frame commands 0.225 (7 % SM), skins 0.236
  (0.4 %, 0.9 warps/SM: skinned BLAS refits), tlas 0.135 (5 %). Primary 0.491 and trace 1.472 at
  ~18 warps/SM; DLSS 1.817. Compute-only queue family 2 (8 queues, 64-bit timestamps).

- 2026-10-15: phase 1 + C1. Compute queue (family 2), its pool and timeline; buffers and
  images CONCURRENT over both families; `submit_with` (waits on another timeline at a stage).
  Skins (+ skinned BLAS refits) and the SHaRC resolve record into the compute command buffer.
  First A/B: no change -- Northshire's footstep picks keep a pick active every frame, and the
  pick path submitted frame commands + skins + TLAS in one early graphics submit, so the
  compute path never ran. Fixed by splitting the pick's early submit (head: frame commands;
  compute; tail: TLAS + pick, waiting on the compute).
- The profiler logs a per-scope timeline at debug (`gpu timeline`: start-end ms per scope).
  It showed the overlap working (skins 0.000-0.251 beside frame commands 0.007-0.418) and a
  new 0.27 ms gap before the main submit: the CPU submitted it only after waiting for the
  previous frame. A/B then: span -0.056 ms (-1.0 %, +-0.5).
- The frame is now submitted before the CPU waits for the previous one (a full barrier at its
  head keeps the GPU serial); the wait targets the previous frame's point. What depended on the
  old order: the exposure latch / readback / probe (now double-buffered: the exposure is one
  frame later), and bindless slot reuse (retired at `last submitted + 2`, which assumed a submit
  count per frame; now at the next frame submit's point). Gap 0.27 -> 0.03 ms. Validation clean.
  A/B async on vs off (both with the early submit): span 5.651 vs 5.687 (-0.6 %, +-0.8).
- 2026-10-15: exposure on the compute queue (its barriers take their readers' stages: no
  fragment stage there). ReSTIR's prepare + light grid for every view in a second compute submit
  (own command pool: the view lane records it while the scene lane records the first), waiting
  on the submit with the lights; the view lane splits after the first view's primary pass and its
  back half waits on the grids. A view whose ReSTIR buffers were just (re)made skips ReSTIR that
  frame (their clears would race the primary pass). Timeline: restir grid 0.699-0.744 beside fog
  grid 0.701-0.808. Validation clean (threading included). A/B all async vs one queue: span
  5.687 vs 5.727 (-0.7 %, +-1.1), frame 5.673 vs 5.753 (-1.4 %, +-1.0).

## Deviations

- D3's frame split became: (pick) head [uploads, frame commands] / compute [skins, SHaRC] /
  tail [TLAS, pick] / main [UI, lights, views, present]; without a pick: G1 [UI, uploads,
  frame commands] / compute / G2 [TLAS, lights, views, present]. Exposure and atmosphere stay on
  graphics (0.04 ms; the atmosphere may read sky-generation work queued in frame commands).
- Submitting before the CPU wait was not in the plan; without it the overlap only moved the
  idle time to before the main submit.

## Measurements

| build | span | busy | trace |
|---|---|---|---|
| baseline (a62fe19) | 5.74 / 5.71 | 5.66 / 5.78 | 1.61 |
| async C1, pick path split | 5.636 (off 5.692) | 5.86 (double counts) | - |
| + submit before wait | 5.651 (off 5.687) | - | 1.60 |
| + exposure, ReSTIR grids on compute | 5.687 (off 5.727) | - | 1.60 |
