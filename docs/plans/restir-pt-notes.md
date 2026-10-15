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

## Deviations

- Phase 0, procedural spheres in the primary record: deferred. Recording them needs MakeHit with the
  sphere's attribute type and a marker other than hit_kind != 0; the sun room's spheres are metal (a
  mirror and roughness 0.2), outside GI/PT eligibility anyway. Sphere pixels keep re-tracing from the
  camera (no ReSTIR surface; their direct light comes from the path's NEE).
