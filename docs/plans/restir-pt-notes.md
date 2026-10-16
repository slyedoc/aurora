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

## Deviations

- Phase 0, procedural spheres in the primary record: deferred. Recording them needs MakeHit with the
  sphere's attribute type and a marker other than hit_kind != 0; the sun room's spheres are metal (a
  mirror and roughness 0.2), outside GI/PT eligibility anyway. Sphere pixels keep re-tracing from the
  camera (no ReSTIR surface; their direct light comes from the path's NEE).
- Layered primary vertices (thin translucency, clear coat) take no part: replay draws the base lobe only.
  GI took them for k = 1.
- SHaRC: a PT path teaches the cache only from past x_k. Its search vertices (no cached answers, no
  roulette) made the voxels read brighter than the truth (own-only PT 0.035 against the truth; 0.018 with
  the restriction). The mechanism is not identified. Letting the cache answer at x_k (as a
  direction-free part) densified the input (30 % black) but read 1.3 % dark (0.064): not taken.
- restir_gi_spatial_samples stays 0 by default until the cost is measured (phase 3).

