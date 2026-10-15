# Primary pass: implementation notes

Plan: `primary-pass.html`. Notes kept while implementing; newest last.

## Log

- 2026-10-14: plan written. Baseline (aurora 0533cc0, Northshire bench 3840x1080 Balanced,
  GPU Trace): trace 2.05 ms (L2 77.7 %, L1 hit 54 %, no_instruction 8 %), restir temporal
  0.40, spatial 0.37, shade 0.32 ms; raygen stack 544 B. slangc 2026.8 lowers
  `HitObject::MakeHit` to `OpHitObjectRecordHitNV` and `Invoke` to
  `OpHitObjectExecuteShaderNV` (spike in the session scratchpad); aurora's shader build already
  passes `-capability spvShaderInvocationReorderNV`.

- 2026-10-15: phase 0 done. `primary` raygen (pinhole only) traces sample 0's camera ray
  through SER and writes a PrimaryHit; `main` (ab 1, `pc.primary_hits` non-null) rebuilds the
  hit with MakeHit + Invoke when the record's ray is bit-identical to the one it would trace.
  Shots ab 1 vs ab 0: ratio 1.000, error 0.002, boil 0.049 vs 0.048. Diagnostic tint: 0 % of
  pixels fell back to tracing; the inverse tint marked 67.8 % (all geometry) as resumed.
  Bench gained `--ab-set <n>` (hold `ab` for a whole run, shots included).

- 2026-10-15: phase 1 done: `primary` walks the whole camera chain (pass-through, portals,
  guides via `writeGuides`, camera fog via `cameraFog`, meter, focus pull, glass guide chain);
  `main` resumes sample 0 (HIT), adds ENDED radiance, or traces from the camera (RETRACE), and
  never writes guides or camera fog while records exist. The record is read where needed,
  never held across traces (holding it cost trace time). In-process A/B (busy, ab 0 vs new):
  first cut +0.33 ms (primary 0.53, trace -0.03); without SER reorder in `primary`
  0.54 -> 0.36 ms; without reorder at the resume trace -8.6 %. New frame now at break-even
  (busy 6.885 vs 6.890) before `restir_shade` goes.

- 2026-10-15: phase 2 done: `primary` writes the ReSTIR surface at the recorded hit; the view
  lane (records present) runs temporal + spatial between `primary` and the trace and skips
  `restir_shade`; `main` shades the winner at the resumed vertex with its own shadow ray
  (`restirWinner`, D6a). Shots vs the old frame: ratio 1.001, error 0.002, boil equal.
  A/B (ab 0 vs 1): busy +0.13 ms (primary 0.47, trace -0.00, restir shade -0.32).
  GPU Trace (ab 1): primary L2 70.5 % (writes ~170 B/px: guides, fog RMW, record, surface;
  lg_throttle), trace no_instruction 12 % (was 8: `main` holds old + new paths). Phase 3
  (delete the old path) is the next lever.

- 2026-10-15: phase 3 done: the old frame is deleted. `main` keeps no guide, camera-fog,
  focus or surface-writing code and no PATH_IN_VIEW / GUIDE_LIVE / GUIDES_WRITTEN; reads the
  meter's divisor and `primary_t` from the record at the end. `restir_shade` (raygen, pipeline
  group, SBT region, dispatch) is gone; `Frame::primary_hits` is always allocated.
  Bench vs the old frame (p0_old): busy -1.7 %, trace -8.6 %, shots ratio 1.000, error 0.002,
  boil equal. Raygen stack 544 -> 352 B. GPU Trace (p3 vs sdk1 at the start of the day, which
  also predates the ReSTIR target/pack work): frame 6.539 -> 6.292, trace 2.05 -> 1.83,
  primary 0.52 (L2 71.6 %, lg_throttle), restir shade gone; trace no_instruction still 11 %.
- Not yet tested: XR (two views), samples > 1, `Fog::steps == 0` (RETRACE), sphere hits
  (RETRACE), portals.

- 2026-10-15: XR tested by the user: works. Phase 4 started:
  - Ray stack as dynamic state, set per dispatch from the queried group stacks (callee
    stacks are 0): main 352, fog grid 96, primary 384 -> the driver's default had sized every
    dispatch for the largest raygen (384). A/B busy -1.1 %.
  - `primary`'s stack 384 -> 224 B: record fields written in place (no struct held), the
    payload's last use moved before the traces (guides' spec probe, camera fog), and
    `fogMarch`'s HitPayload scoped to its one sky ray (it was held across every sun ray; 192 of
    the 320 B). `fog_grid` / `fogSunLight` lost the unused payload parameter too. Primary
    469 -> 456 us only: its stack was not what it waits on.
  - Tried and rejected: recording the hit's payload (64 B, f16 / snorm16 packed) instead of
    its identity, so `main` skips the second closest hit. Primary +3 %, trace +1 %: the
    resumed closest hit is cheap (screen-coherent hits, textures in cache). D3 stands.

  - D6b measured (ab, noon): trace -15 %, busy -4.9 %; noon shots identical (boil 0.050 vs
    0.051). Night at the bench's view is too dark to judge. Landed as
    `RenderSettings::restir_shared_ray` (off: D6a stays the default until the abbey test;
    editable in F3).
  - Found: `restir_candidates` was still 8 in aurora 0533cc0 (the commit says 4; the
    rejected 8-vs-4 loop had partly run). Set to 4: temporal 0.40 -> 0.25 ms, image unchanged.
  - The camera ray's fog veil accumulates in registers in `primary` and is written once
    (was: init + per-segment + alpha read-modify-writes): primary 460 -> 438 us; its stack
    224 -> 256 B (the veil lives across the chain's traces).
  - Not done: `primary` in its own pipeline (needs a second hit table in the SBT; whether
    NVIDIA allocates registers per pipeline or per shader is unverified).

## Deviations

- D3 said the resume reorders on the rebuilt hit like `traceReordered`. Measured: neither
  the primary pass nor the resume benefits from SER (both are screen-coherent); reordering
  cost the primary pass a third and the trace 8.6 %. Both dropped.

## Measurements

| frame (GPU Trace, Northshire 3840x1080) | frame | trace | primary | restir shade |
|---|---|---|---|---|
| sdk1, start of day (old frame) | 6.539 | 2.051 | - | 0.301 |
| p2, both frames in the binary (ab 1) | 6.544 | 2.029 | 0.524 | - |
| p3, old frame deleted | 6.292 | 1.832 | 0.523 | - |

Next levers (phase 4 candidates): the primary pass's L2 traffic (64 B record -> 32 B; keep the
fog veil in registers and write it once instead of three read-modify-writes), its own pipeline
(occupancy), D6b shared shadow ray (judge in the abbey).
