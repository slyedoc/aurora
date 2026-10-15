# ReSTIR GI: implementation notes

Plan: `restir-gi.html`. Notes kept while implementing; newest last.

## Log

- 2026-10-15: plan written against aurora 9ec9ce1. Code not started: `raygen.slang`,
  `restir.slang`, `rt_types.slang`, `fog.slang`, `fog_grid.slang` carry an uncommitted
  aligned-load pass (`restirSurfaceLoad` / `restirSurfaceStore`, `mat4MulVecAt`) from another
  session; phase 0 starts on top of it once it is committed. `atmoComposite` checked: affine in
  the surface colour (`Tc * (La + Ta * s) + Th * Lc + (1 - Tc) * Lh`), so D6's multiplier
  approach holds.

- 2026-10-15: D8 added: GI may retire SHaRC (the cache was the 1 spp darkness fix, 30be5eb);
  phase 3 runs the four-way GI/SHaRC comparison, phase 4 deletes the cache if GI alone holds.
  `atmoComposite` (aerial perspective + clouds) is unrelated to darkness and stays.

- 2026-10-15: review from the performance session folded in. Base is aurora `5529445` (the
  aligned-load pass was this repo's own, committed in `9caa950`). Added a cost budget (~0.9-1.1 ms
  flat, ~2 ms XR) with gates per phase; `gi_shade` becomes a compute kernel with an inline ray
  query (not a raygen: the deleted `restir_shade` shape, the RT pipeline's register allocation),
  with the alpha-cutout catch (no any-hit in a ray query) decided in phase 0, raygen as fallback;
  memory shrunk up front (RGB9E5, `x_v` from depth: 240 -> 176 B/px, 490 MB in XR); the raygen
  stack check after phase 0; phase 2 judged by the error number; the GI passes sit in the view
  lane's back half (async compute, `ec94ef0`); the meter is null for a camera's second view
  (`c7a5861`).

- 2026-10-15: implemented (phases 0-3). Candidates in `main` (PATH_GI_VERTEX / SEGMENT / LIVE,
  snapshots `gi_mask` / `gi_acc`, position and pdf to memory at the secondary hit, radiance at the
  path's end), `restir_gi.slang` (`gi_temporal`, `gi_spatial`), `gi_shade` raygen, `restir_gi.rs`,
  GI_ON = constant 8, `UniformData::gi_roughness` (was pad_spec_hit), push constants
  `gi_candidates` / `gi_final`, SBT region `gi_shade_region`. `restirReproject` / `restirSimilar`
  moved to rt_types (DI uses them too); `atmoCompositeT` returns the surface multiplier.

  Measurements (zero `--view` presets added for this: canopy, lane, path, abbey, overlook; shots
  vs a 4 spp reference with no cache and no reuse, `compare.py`):

  | abbey (EV 2, noon)              | ratio | error | boil |
  |---------------------------------|-------|-------|------|
  | SHaRC only (today's default)    | 1.020 | 0.088 | 0.79 |
  | neither                         | 0.934 | 0.071 | 1.06 |
  | GI phase 0 (own candidate) + SHaRC | 1.041 | 0.097 | 0.37 |
  | GI spatial 3 + SHaRC            | 1.023 | 0.090 | 0.78 |
  | GI spatial 3, no SHaRC          | 0.990 | 0.055 | 1.09 |
  | GI temporal 8 + spatial 3 + SHaRC | 0.98 | 0.103 | 0.93 |

  Canopy (outdoors): every variant within ~0.01 error of off; temporal reuse boils (0.46 -> 2.2
  at history 8, 1.3 at history 1, 1.7 with the permutation).
  Phase 0 at 4 spp: GI vs no GI ratio 1.004 (the estimator matches; at 1 spp RR reads the lower
  variance input ~2 % brighter).
  Cost, 3840x1080, A/B in one run (GI spatial-only vs SHaRC): gi temporal 0.14, gi spatial 0.28,
  gi shade 0.17 ms, trace +0.19 ms (no cache termination); span +0.91 ms (+16.6 %), inside the
  1.0 ms gate. Raygen stack: 256 B off, 288 B GI + SHaRC (+32, the gate), 192 B GI alone.

  2 spp, abbey, no SHaRC: GI off ratio 1.034 error 0.082, GI on 1.030 / 0.068 (GI's share is
  sample 0's, as designed).

  Decision (phase 3): `restir_gi_history` defaults to 0 (spatial reuse only). On the numbers
  above the user chose D8's answer: GI on by default, SHaRC deleted.

- 2026-10-15: phase 4 done. Removed `sharc.rs`, `sharc.slang`, the resolve on the async compute
  queue, `SharcEntry` and the push-constant pointer, the raygen's cache code (voxel hashing,
  query, deposits, PATH_SHARC_TRAIN / FROM_ROUGH / CACHE_BLIND, three snapshot slots), the
  `sharc` / `sharc_voxel` settings; the uniforms keep their offsets (`pad_cache`), spec constant
  0 is free. `restir_gi` defaults to true.
  Defaults now, vs the 4 spp reference: abbey ratio 0.97-0.98, error 0.049-0.053 (SHaRC's was
  1.020 / 0.088), boil 1.09 (0.79); canopy error 0.014, boil 0.26. Raygen stack 256 -> 192 B.
  Bench 3840x1080, bracketed off/on/off/on: span 5.40 / 6.09 / 5.37 / 6.03 ms (GI +0.67 ms,
  +12.5 %; ~+0.6 ms over the old SHaRC default), trace 1.53 -> 1.63 ms.

- 2026-10-15: tested in XR by the user (defaults: GI spatial-only, SHaRC deleted): works.

- 2026-10-15: memory and cost. Without temporal reuse (the default) only the candidates and one
  final reservoir exist: gi_temporal writes each candidate's W and target into it in place, and
  gi_spatial reads candidates as its samples (`GiFrame::in_place`); the atmosphere multiplier
  moved to the surface record's spare word (`RestirSurface::gi_view_t`). 224 -> 96 B/px (1107x578:
  136 -> 58 MB; ~180 MB less per view at 3840x1080 Balanced). Temporal reuse allocates its two
  reservoirs again when on. gi_spatial hashes neighbours with PCG instead of two 16-round TEAs and
  keeps the usable ones as a bit mask (a register; the second walk skips the test's loads).
  3840x1080: gi temporal 0.141 -> 0.104, gi spatial 0.279 -> 0.219, gi shade 0.169 -> 0.166 ms
  (0.589 -> 0.489 ms); abbey ratio 0.97-0.98, error 0.049-0.053 as before.

- 2026-10-15: eligibility split from sharing (D1 revised). Every primary vertex takes part but
  near-mirrors (perceptual roughness < 0.1, `GI_MIN_ROUGHNESS`: below it the lobe is narrower than
  the half-float normal and view the passes rebuild it with) and glass; every secondary hit but
  glass and emitters gives a candidate. `restir_gi_roughness` now only decides whether a sample is
  shared: a glossy hit's radiance depends on where it is seen from, so its sample (n_s_oct low
  bit clear, `giShareable`) serves its own pixel; the temporal pass skips such history, the
  spatial pass never shifts such a sample to or from a neighbour. Abbey 4 spp GI vs reference
  1.004 / 0.013 (no bias); abbey and canopy at 1 spp unchanged (mostly rough). New test scene
  `examples/many_lights.rs` (bevy's solari many_lights ported; `--no-gi`, `--no-restir`,
  `--samples N`): vs 4 spp reference GI 0.999 / 0.012, no GI 0.998 / 0.012; boil 0.20 -> 0.12.

- 2026-10-15: GI debug views (`AuroraDebugView::Gi`: gi_shade's light alone; `GiReuse`: per
  pixel not taking part / nothing to add / own private / own / neighbour / occluded), written by
  gi_shade into the Cost view's buffer (two words per pixel) through `PushConstants::gi_debug`.
  New `examples/sun_room.rs`: a closed room lit only through one window (flags `--no-gi`,
  `--no-restir`, `--samples`, `--ev100`, `--gi-history`, `--gi-spatial`, `--boiling`).
  Verified there (EV 8 pinned, vs 16 spp without GI or reuse):

  | sun room                | ratio | error | boil |
  |-------------------------|-------|-------|------|
  | reference 16 spp        | 1.000 | 0.004 | 0.40 |
  | no GI                   | 1.027 | 0.059 | 0.68 |
  | GI spatial 3 (default)  | 0.998 | 0.061 | 0.78 |
  | GI own candidate only   | 1.003 | 0.063 | 0.65 |
  | GI temporal 8           | 0.959 | 0.078 | 0.94 |
  | no GI 4 spp             | 1.024 | 0.038 | 0.56 |
  | GI 4 spp                | 1.003 | 0.029 | 0.51 |

  GI fixes the brightness (RR reads the plain path 2.5 % bright at any spp) and at 4 spp the
  error; at 1 spp the local error is unchanged and spatial reuse adds boil: the GiReuse view is
  mostly "nothing to add" -- most bounces land on unlit wall, whose radiance is one more random
  path that rarely finds the sunlit patch, and reuse spreads the rare bright ones. SHaRC is no
  answer here: the pre-GI build in this room reads 0.712 / 0.478 against its own reference (no
  cache 1.024 / 0.057). The boiling filter took 21 % of GI's light here. What would help is a
  better radiance at the secondary hit (path reuse past the first bounce, ReSTIR PT).
  Procedural spheres are never resumed from the primary pass, so neither ReSTIR DI nor GI reaches
  them (GiReuse grey). The earlier many_lights numbers were auto-exposed (ratios normalised).

- 2026-10-15: distance rule for sharing: a secondary hit closer to its vertex than 3 % of the
  vertex's camera distance is not shared (`GI_RECONNECT_MIN`, ReSTIR PT's reconnection criterion);
  temporal reuse's bright and coloured flecks in corners and at contacts went (sun room, history
  8: error 0.078 -> 0.041).
- 2026-10-15: frozen as the baseline ReSTIR PT will replace. Spatial reuse off by default: in the
  noisy input it bunches bright samples into clumps that move each frame, which Ray
  Reconstruction shows as blotches (sun room: never lower error, always more boil; radius 8 far
  worse), while each pixel's own candidate gives the plain path's noise with the right brightness.
  Temporal reuse deleted (it boiled in every scene); with it went last frame's reservoirs, the
  stored vertex and the in-place/temporal layouts: the final reservoir is 32 B, GI 64 B/px.
  Still a setting: `restir_gi_spatial_samples`. Raw view (`RenderSettings::tonemap`, examples'
  `--raw`) and fog off by default (an environment needs `Fog`) added while judging it. A still
  camera still drifts through Ray Reconstruction (sparse 1 spp input in a bounce-lit room; 4 spp
  cuts it ~37 %): the next step is an RR-off accumulation reference to tell the tracer's faults
  from RR's.

## Deviations

- D5: `gi_shade` is a raygen (traceVisible), not a compute kernel with a ray query: the
  visibility ray then has the shadow rays' any-hit alpha test, clear-glass pass-through and the
  analytic spheres for free; a compute kernel needed a manual alpha test (per-instance triangle
  data is only reachable through the SBT record). Its own stack: 32 B.
- D3: `GiReservoir` is 64 bytes and stores the primary vertex (`x_v`). Rebuilding it from depth
  (the plan's 48-byte layout) made the temporal Jacobian wrong enough that reused weights grew
  each frame (10-21x brightness); with the stored vertex it is stable. 208 B/px in all
  (candidate 32 + 3 reservoirs 64 + the surface buffer when DI is off).
- D4: temporal reuse reads last frame's reservoir from a random pixel of the 3x3 around the
  reprojected one (temporal permutation); still boils under RR, hence history 0 by default.
- `gi_spatial` holds no neighbour array: neighbours come from a hash of (pixel, frame, i) and are
  walked twice (counted, then merged). With the 64-byte reservoirs, the `picked[16]` array
  version faulted on the GPU (MMU write fault at a tiny address, Xid 109/31, Aftermath: the
  spatial kernel) every run; the array-free version runs clean.
- `restir_gi_history` 0 now means no temporal reuse (the plan's range started at 1).
- GI eligibility at the primary vertex needs transmission exactly 0 (not < 0.01): the glass
  branch would otherwise take the vertex's light with probability `transmission` outside the
  layered BSDF GI evaluates.
- RGB9E5 clamps radiance at 65408 nits (emitters never become candidates).

## Untested

- Moving camera / object motion with temporal reuse on.
