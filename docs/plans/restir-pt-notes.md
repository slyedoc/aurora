# ReSTIR PT: implementation notes

Plan: `restir-pt.html`. Newest last.

## Log

- 2026-10-15: plan written against 7c35182. Baseline (sun room, DLAA 1920x1080, SHaRC on, GI own candidates):
  RR output 0.58-0.77 of the reference, ~17 %/frame drift; noisy input 61 % black, the top 1 % of a wall's
  samples carrying 97 % of its light; mirror sphere 0.77, glossy 0.87 of the reference through RR.

- 2026-10-15: phase 0, exact BSDF pdf. `sampleDisneyBRDF` picks its lobe by Fresnel at n.v
  (`specularProbability`, the vertex only) and returns the whole base BRDF over the mixture's density
  (one-sample MIS); `evalDisneyBRDF` / `pdfDisneyBRDF` are that BRDF and that density exactly. Fixed on the
  way: the sampled diffuse was scaled by (1 - F) at a random half vector (NEE used the real one) and lacked
  (1 - metallic); NoH was clamped to 0.99 (the GGX peak capped ~180x at roughness 0.2: sun NEE and MIS on
  glossy surfaces). A delta lobe (alpha < 1e-3) is out of eval/pdf; a mirror bounce keeps PATH_WANT_SUN so
  the sun disc reaches it.
- Cycles as a second reference (Blender MCP, scratchpad blender_sun_room.py: the sun room, flat sky 6000,
  no sun, no ground). Whole image 1.00. Pure Lambert, direct sky only (aurora --bounces 2, Cycles 0
  bounces): box top and cabinet match (1.01), but aurora's west wall reads 0.87-0.90 of Cycles and of a
  CPU tracer (both 1.00): an aurora bug independent of the BSDF.
- Found: shading normals. Materials without a normal map read the default texel (128, 128, 255), which
  decodes 0.4 % off axis, and surface.slang took normals to world space with the object-to-world matrix,
  not its inverse transpose: under the sun room's non-uniform scales (floor 8.4 x 0.2 x 6.4) the 0.3 degree
  tilt became ~9, toward the window on the floor (+30-65 %), away on the west wall (-12 %). Now: no normal
  map = the geometric normal exactly; `objectToWorldNormal` (inverse transpose) for triangle, sphere and
  terrain-splat normals; the normal-map frame built in world space. Pure Lambert direct: aurora/CPU median
  1.000 (p10-p90 0.945-1.045, the CPU's noise), block error against Cycles 0.029 -> 0.006. Full materials,
  8 bounces, against Cycles' default (multiscatter GGX): block error 0.057 -> 0.030; walls, cabinet, box
  fronts within a few %. Left: the box top (0.73), the grazing specular model (Disney/Schlick against
  Principled v2: pure Lambert matches there).
- Debugging lesson: the sealed-window test rebuilt the example and later runs used that binary until a
  rebuild (three black captures). Rebuild after reverting an example.

- 2026-10-15: phases 1-2. PT grew out of GI in place (restir_gi.rs, restir_gi.slang, raygen.slang; names
  kept until phase 4). Design checked against RTXDI's ReSTIR PT (/mnt/code/f/RTXDI, read only: its
  library is under the NVIDIA RTX SDKs license, so nothing is copied): replay in primary sample space
  (the replayed prefix's own f cos / pdf is the target; only the reconnection has a Jacobian).
  - Sample (80 B): x_k at the first k >= 1 where x_{k-1} and x_k are rough (>= restir_gi_roughness), not
    glass, no thin layers, and farther apart than 3 % of the primary vertex's camera distance; k <= 3. Light
    before x_k stays in the pixel's path. Stored: x_k, its material and shading normal, the source view,
    what x_k sent toward x_{k-1} split by direction (its next-event sample, its BSDF continuation), the
    density of x_{k-1} -> x_k, the seed, k, the path weight T, the geometry term.
  - BSDF numbers while searching come from a stateless hash of (seed, vertex, dimension); no Russian
    roulette and no radiance-cache answers until the search ends (replay must match the source prefix).
  - gi_spatial is a raygen (replay traces): pairwise MIS over random neighbours, every target the same
    function (the canonical replayed too), visibility of the reconnection segment in the target.
  - x_k's BSDF is evaluated again for the shifted incoming direction: each stored part scaled by
    f(wo')/f(wo_src) (the pdfs and MIS weights inside stay the path's, fixed functions of the directions).
    Storing radiance over pdf instead overflowed RGB9E5 (the sun's next-event term).
- Bugs on the way: the final reservoir's segment normals were stored as zero (gi_shade's visibility ray
  left along (0, 0, 1) and hit its own surface on the west wall and floor: -15 %); a splice cut 99 lines of
  the primary pass (restored from HEAD).
- Measured (sun room 960x540, references against no PT and no SHaRC, the truth):
  | config | bias block error (floor ~0.017) | input black | RR block error | RR change/frame |
  | no PT, SHaRC | 0.018 | 61 % | 0.089 | 0.070 |
  | PT own only | 0.018 | 60 % | - | - |
  | PT 3 neighbours | 0.027 | 60 % | 0.105 | 0.034 |
  | PT 8 | 0.021 | 39 % | 0.106 | 0.031 |
  | PT 16 | 0.017 | 25 % | 0.071 | 0.023 |
  Before x_k's BSDF was re-evaluated, reuse ran bright (+1.2 %, 0.045: RIS favoured samples whose radiance
  was high toward their own pixel). RR's output still reads ~0.72 of the reference (its energy loss on
  sparse input is not solved by spatial reuse; temporal reuse is phase 3).

- 2026-10-16: the residual bias, the cost. Reuse read 1-2 % bright without SHaRC (16 neighbours 0.049, even
  one neighbour, even radius 2, even the pixel as its own neighbour: +2.2 %). Instrumented (debug values
  through the final reservoir's float fields into the noisy colour; RGB9E5 and half floats garbled the first
  dumps): the same shift of a pixel's own sample passed its facing test before the neighbour loop and failed
  it on 45 % of pixels inside it. ptShift inlined into the loop came out wrong; `[noinline]` fixed it
  (self-test 1.001 / 0.015; 16 neighbours 1.001 / 0.015 without SHaRC, 0.999 / 0.017 with). Two
  lessons from the hunt: zsh does not word-split `$F` (whole flag sets were silently ignored: scratchpad
  cap.sh passes arrays now); RGB9E5 debug values with mixed magnitudes read back as garbage.
- Russian roulette runs during the search again: the prefix's survival q goes into the density
  (W = 1 / (pdf q)), not into T (replay rolls no dice). Trace at 1080p 3.3 -> 2.0 ms (no PT: 1.0).
- The radiance cache may answer at x_k: its value is stored as a direction-free part of x_k's light
  (PT_NO_DIRECTION, the cache's own assumption); a PT path teaches the cache from x_k on. The earlier
  attempt read 1.3 % dark because the cache then learned nothing from PT paths (prefix and x_k both
  excluded). Own and 4 neighbours unbiased (0.018 / 0.021).
- gi_spatial cost: the shift's visibility as TraceRay was ~85 % of it (8.2 ms at 4 neighbours, 1080p).
  Now a ray query (opaque triangles and the analytic sphere inline; alpha-tested, glass or other procedural
  candidates fall back to traceVisible) and a sample's target at its own pixel computed once in
  gi_candidates (ptOwnTarget: visibility 1; k >= 2 takes its path's T L pdf) instead of by every neighbour.
  4 neighbours 8.2 -> 1.5 ms, 8 -> 2.8 ms. PT overhead at 4 neighbours ~3 ms (trace +1.0, candidates 0.26,
  spatial 1.5, shade 0.3) on a 7.4 ms frame.

- 2026-10-16: phase 3, temporal reuse, and the Ray Reconstruction gate.
  - Ray Reconstruction is faithful once the exposure is pinned: no PT, RR's output against the reference
    1.008 (block error 0.035, change/frame 0.004) at --ev100 9, against 0.67-0.75 under auto-exposure.
    The darkening every earlier measurement blamed on RR (the plan's "0.58-0.85") comes with
    auto-exposure: in the brightest bands (window, sunlit patch) RR kept ~0.6 though the averaged input
    matched the reference. Either the metered exposure varies between frames (RR's history is in
    pre-exposed units) or the capture's conversion back to nits uses a different exposure: open, outside PT.
    RR gates below are at a pinned exposure.
  - Resolved (2026-10-15): neither. The capture's 600 settle frames were CPU frames; the trace starts
    only once the pipelines compile, so the noisy frames came ~12-20 traced frames in, while RR's input
    exposure was still easing from the -13 reference to the metered -9.9 (~0.03 EV per frame) and
    RR's history, built darker, lagged it. Settle now counts traced frames: under auto-exposure RR's
    output against the reference 0.822 -> 1.003 (block error 0.109 -> 0.034, change 0.125 -> 0.003).
    RR is faithful at any exposure (pinned 6, 9, 12 and auto all ~1.00).
  - gi_temporal (raygen): this pixel's sample against last frame's final reservoir at the reprojected pixel
    (confidence capped by restir_pt_history), pairwise MIS with confidence weights, both shifts (last
    frame's sample here, ours to last frame's vertex, from a compact copy of it in the history). The result
    replaces the candidate in place (W, M, target, geometry term from here) for gi_spatial, which weighs
    neighbours by confidence and writes the history (128 B/px). gi_shade no longer tests visibility: every
    target includes it.
  - Bug on the way: the temporal winner kept last frame's `frame` stamp in the candidate slot, so gi_spatial
    dropped the pixel (53 % black, 0.954).
  | config (sun room 960x540, DLAA, --ev100 9) | bias vs truth | input black | RR block error | RR change |
  | no PT | - | 61 % | 0.035 | 0.004 |
  | PT own | 0.019 | 61 % | 0.035 | 0.004 |
  | PT 4 neighbours | 0.018 | 21 % | 0.027 | 0.003 |
  | PT temporal 8 | 0.020 | 4 % | 0.114 | 0.013 |
  | PT temporal 8 + 4 neighbours | 0.018 | 3 % | 0.036 | 0.003 |
  Temporal reuse is unbiased and densest but RR keeps its correlation (boiling): D4's gate fails, so it
  ships off (restir_pt_history 0). Spatial 4 is the default (restir_gi_spatial_samples 4).
  - many_lights: PT unbiased (0.003 against no PT); its light is mostly direct (ReSTIR DI), the input barely
    changes, RR's block error 0.099 -> 0.074.
- Phase 4: ReSTIR GI is gone (PT replaced it in place). Its names stay where they are public
  (RenderSettings restir_gi*, restir_gi.rs, gi_* passes, the Gi / GiReuse debug views); the GI-only helpers
  (giContribution, giShareable) are removed.

## Deviations

- Phase 0, procedural spheres in the primary record: deferred. Recording them needs MakeHit with the
  sphere's attribute type and a marker other than hit_kind != 0; the sun room's spheres are metal (a
  mirror and roughness 0.2), outside GI/PT eligibility anyway. Sphere pixels keep re-tracing from the
  camera (no ReSTIR surface; their direct light comes from the path's NEE).
- Layered primary vertices (thin translucency, clear coat) take no part: replay draws the base lobe only.
  GI took them for k = 1.
- SHaRC: a PT path teaches the cache only from x_k on. Its search vertices (no cached answers) made the
  voxels read brighter than the truth (own-only 0.035-0.038 against the truth, also with a stronger path
  hash; 0.018 with the restriction). The mechanism is not identified.
- A k >= 2 sample's own target is its path's T L pdf, not a replay from the stored primary vertex (they
  differ by that vertex's stored precision): the pairwise weights' targets are one function only up to that.
- restir_gi_spatial_samples stays 0 by default until the cost is measured (phase 3).
- Names: PT kept ReSTIR GI's (settings, module, passes, debug views) rather than renaming public API
  (Zero sets restir_gi); phase 4's "remove GI" is the replacement, not a rename.
- Untested: Northshire (Zero has no capture yet; its default now runs PT with 4 neighbours), XR.

