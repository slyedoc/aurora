# Water

Target: WoW's Wetlands water (rain, overcast): a sky-and-shore reflection by Fresnel, a
small wind chop that scrolls, rain rings, and murk: green-brown water you see a swimmer's body
in for half a metre and then nothing, opaque green rather than black. Shallows show the mud,
the shore fades in by depth. Under the surface: Snell's window, total internal reflection,
green fog.

Test scene: `examples/water.rs` (a pond: water plane at y = 0, a slope into 3 m, white tiles at
0.25/0.5/1/1.5/2/3 m depth, a striped pole through the surface, a ball floating half under).

    cargo run --release --example water -- [--murk D | --clear] [--under] [--overcast]
    cargo run --release --example water -- --capture DIR --size 960x540 --frames 1024 --noisy 4
    python3 -I scripts/refcmp.py --png DIR

## What aurora has

- `AuroraMaterial`: `specular_transmission`, `ior`, `attenuation_color/_distance` (Beer-Lambert,
  `absorption_from_attenuation`). The raygen applies it only on a segment that ends on the
  *same transmissive surface's back face* (`inside && transmission > 0`, raygen.slang ~2265):
  correct for a closed glass sphere, nothing else.
- Shadow rays ignore every surface with `specular_transmission > 0.5` (any_hit.slang): no
  Fresnel, no absorption.
- Fog volumes (src/fog_volume.rs): boxes/ellipsoids in a table, camera ray marched
  (`fogMarch`), bounces delta-tracked, shadow rays carry the transmittance, a fog light grid
  for single scattering. Grey: one scalar density, the environment's fog colour.
- Surface groups (src/surface_group.rs): per-material closest-hit via the SBT; the module doc
  already names "animated water" as a class. `uniforms.time` exists (fog noise).
- Procedural meshes (compute-filled vertex streams) for displaced geometry if ever needed.

## Baseline (2026-10-19, 960x540, 256 frames, reference)

- From the bank, `--murk 1` and `--clear` are the same image: the camera ray refracts in, hits
  the mud, and no absorption is applied. The floor is sunlit at full strength through 3 m of
  "murky" water. The water reads as a slightly blue mirror over a lit floor.
- `--under`: absorbed correctly on the camera-to-surface segment (the plane's own back face),
  Snell's window and TIR are right; the floor below is not absorbed (its segment ends on mud).

## Rungs

**W1 Water is a medium volume, not a material.** `WaterVolume { half_extents, absorption: Vec3,
scattering: Vec3, g }` on an entity whose box top is the surface (a plane mesh is only the
interface: Fresnel + refraction, no attenuation). The medium is analytic: every segment's
length inside the box is a slab intersection, so transmittance `exp(-sigma_t * len)` goes on
every segment (camera, bounce, NEE/shadow rays, ReSTIR replay) without tracking what medium a
path is in, and the camera under water needs nothing special. Table + uniform like fog
volumes (same upload pattern), RGB coefficients. Shadow rays through the surface keep passing
the interface but now pick up `(1 - F)` and the volume's transmittance.
Check: tile luminance against `exp(-sigma_t (d / cos theta_view' + d / cos theta_sun'))` with
refracted angles, from the captured reference (a script beside refcmp). `--murk` must finally
change the image; main vs reference ~1.00.

*Done 2026-10-19.* src/water_volume.rs (`WaterVolume`: half extents, attenuation colour and
distance as on `AuroraMaterial`, IOR; requires `Visibility`, the upload skips entities without
`InheritedVisibility`), assets/shaders/water.slang (`waterTransmittance` for traced segments,
`waterLightTransmittance` for shadow rays: refracted length + Fresnel when the ray crosses the top).
The table rides the uniform's two free slots (the SHaRC pad -> address, `pad_one_ray` -> count).
Applied: every main-loop segment (camera included, after the primary pass's resume), the primary
pass's ReSTIR surface throughput (camera segment), `ShadowPick::resolve` (DI winner, table
lights, sky, suns), the fog's sun/table shadow rays, `ptReplay` per segment, `ptShift`'s
reconnection, and the k = 1 own target (`ptOwnTargetWater`; k > 1 carries it in T).
Measured (960x540 references): `--no-sky`, the surface reflection subtracted (`--murk 0.001`
capture), murk over clear at the 0.25 / 0.5 / 1 m tiles' top faces against the formula:
R 0.130 / 0.137, 0.0100 / 0.0106, 1e-4 / 1e-4; G 0.242 / 0.255, 0.041 / 0.044, 0.0012 / 0.0012;
B 0.148 / 0.153, 0.0131 / 0.0135 (within 5 %, slightly low). ReSTIR GI/PT unbiased with water:
`--no-gi` vs default, from the bank 1.000 / 0.002, under water 1.007 / 0.001.
Looks: the floor fades into dark blue-grey with depth (the sky's reflection over a black body):
no green until W2. Limits: volumes upright (top = local +y); the surface mesh must sit on the
box's top; sample 0's alpha-cutout segments before the primary record miss the water (the
resume restarts at the recorded segment); shadow rays go straight (caustics are W6).

**W2 Murk scatters.** Absorption alone goes black with depth; the Wetlands look is in-scatter:
the sun and sky light the water itself.

*Done 2026-10-19.* `WaterVolume::scattering` (per metre, linear RGB) and `anisotropy` (HG g).
Every main-loop segment draws a free-flight collision against each scattering volume's mean
extinction (one density for the three channels, the difference in the weight: T / P(no
collision) through, sigma_s T / p(t) at a collision; water.slang `waterSegment`). At a
collision: the suns and the sky into it through the phase function (one shadow ray), then a
phase-sampled continuation whose sky hit shares MIS credit with that estimate. Surviving a
segment while ReSTIR PT searches goes into `pt_q` like the roulette, so replay's transmittance
(now extinction) stays the sample's weight; the primary pass's ReSTIR surface takes the camera
segment's survival weight, so GI is unbiased where the camera ray scatters before x_0 (no
candidate that frame). Transmittance everywhere else is extinction (absorption + scattering).
Measured: GI off vs on, under water 1.000 / 0.002; from the bank 1.000 / 0.001 with the sun
glints left out (8 % of pixels; with them 0.985 / 0.2, heavy-tailed noise, see W8).
Limits: a collision samples no table lights (their hits then take full weight); a path's
collisions spend the loop's iterations (MAX_BOUNCES), so very bright murk loses its high orders;
the camera ray's in-scatter is stochastic in the noisy colour (fog's camera rays march into a veil
instead); RR handles it so far.

**W3 The water surface class.** *Done 2026-10-19.* `WaterSurface` on the surface mesh routes it to
water_surface.slang (registered in `WaterVolumePlugin::finish`, before anything spawns). The hit
finds its volume (the box whose top face it lies on) through the uniform's table address and
reads the waves from that row: `wave_direction`, `wavelength` (the longest), `steepness` (peak
slope k A), `wave_speed`. Twelve sines, each ~0.73x shorter, fanned about the wind, deep-water
dispersion, by `uniforms.time` (a capture pins it to 0). A wave shorter than 4x the ray cone's
width fades out of the slope into slope variance, added to the roughness as GGX alpha^2 += 2
sigma^2. A normal tilted so far that the mirror direction falls under the face bends back until it
clears it (else the BSDF's leak test ended those paths: black patches). Shared types in
water_types.slang (`__exported` by rt_types). Looks: murky green-brown with chop from the bank
(RR output), Snell's window broken into moving patches from below.
Open: the short waves still beat into a faint cross-hatch at mid distance (tuning, or a few
noise octaves instead of sines).

*FFT waves, 2026-10-20* (src/water_waves.rs, water_fft.slang; Tessendorf 2001). Per volume two
256x256 cascades of the surface slope: the first tile 8 peak wavelengths across, the second 6.7x
smaller (no common repeat), each its own band of one Phillips spectrum (first: up to half its
Nyquist; second: above), wind along `wave_direction`, upwind waves at 7 %. h0 is generated on the
CPU when the waves change, scaled so the mean squared slope is steepness^2 / 2 (unit tests: the
slope, the disjoint bands). Each frame, in the frame's commands: h(k, t) by deep-water dispersion,
the slope spectrum (dh/dx real, dh/dz imaginary: one complex IFFT carries both), a radix-2
inverse FFT per row then per column (one 64-thread workgroup per line, groupshared), texels
(s_x, s_z, s_x^2, s_z^2) as RGBA16F into level 0, the chain blitted. The hit shader reads both
cascades with SampleGrad over the ray cone's footprint (`hit.axis1/axis2`): slopes add, E[s^2] -
E[s]^2 is the variance a footprint hides, into GGX roughness (LEAN). `wave_tex0 == 0` keeps the
analytic sines. Cost: 0.08 ms a frame for both cascades (the profiler's `frame commands`). Looks:
random, multi-scale chop with no cross-hatch and no visible repeat in the reference; RR's output is
softer while the waves move (shading changes with no geometric motion to reproject).

Engine fixes found on the way (they matter beyond water):
- Surface classes cast opaque shadows: every class's shadow twin was an empty hit group, so a
  transmissive class surface blocked every shadow ray (the floor under the water got no sun).
  All classes now take the shadow any-hit (alpha test, transmissive pass-through); instances of
  opaque materials are force-opaque and never run it.
- Light behind smooth glass was counted twice under an importance-sampled sky: next-event
  estimation saw the sky straight through the glass (shadow rays pass it, MIS-weighted) and the
  refracted continuation reached it at full weight (the delta lobe reset the MIS partner). Now
  `mis_pdf` / `mis_origin` / `mis_dir` keep the last rough vertex through glass that shadow rays
  pass, and a light or sky hit weighs against its pdfs along the straight direction. Floor under
  shallow clear water, overcast, reflection subtracted, against the floor dry: a flat-colour sky
  (no NEE) 0.45-0.55 (1/n^2 times Fresnel: right); a uniform HDR sky of the same radiance 0.63-0.79
  before, 0.43-0.62 after. Sun room GI off vs on still 1.000 / 0.022.

**W4 Rain.** Ripple rings in the water class: hashed cells, each spawning an expanding damped
ring on a timer, `rain` intensity a parameter (0 = off). Drops and splashes are particles
(game side).

**W5 Shore.** Optional: foam/darkening where the water is shallow, by a short ray down from
the surface hit in the water class (depth to the floor). Much of the shore fade already falls
out of W1/W2.

**W6 Caustics.** Not for now. NEE cannot see the sun through a refracting surface; the W1
pass-through is the average. Later: a caustic map traced from the sun through the wave normal
into the volume floor, sampled by the floor's shading.

**W8 Glints.** The sun on smooth water is reached only by mirror rays that happen to hit its
disc (the glass lobe has no pdf, so no next-event estimate): single pixels at 2e7 nits, the
noisy input's rms 357. Next-event estimation of the suns on the dielectric's reflection lobe
through the wave roughness (as for a rough metal), MIS'd against the disc hit.

**W7 Displaced waves.** Not for now (WoW water is flat). A procedural plane filled per frame
needs previous positions for motion vectors and a per-frame BLAS refit.

Integration checks along the way: ReSTIR GI/PT replay recomputes the analytic transmittance
deterministically (shifts stay unbiased); NRC never queries at a transmission vertex today,
which is fine at the interface but needs a decision for vertices inside the water; reference
mode integrates the same medium.

Order: W1, W3, W2 done; next W8 (glints), then W4.

**Guides at the water, 2026-10-20.** Smooth glass hands Ray Reconstruction the guides of whatever
its dominant Fresnel lobe reaches (the floor under it, the sky it reflects); a wave flips that lobe
pixel by pixel, so the guides flickered between two backgrounds exactly where the water blurred
(the roughness and normals debug views showed a speckle band). The water surface now sets payload
flag bit 2 (`hitPayloadSetGuidesHere`): the guides stop at it, with its wave normal and roughness,
its depth and motion, diffuse albedo 0 and specular albedo 1 (all of its light goes on specularly).

**Materials, 2026-10-20.** What a body of water is like is a `WaterMaterial` asset (medium + waves;
`MURKY_POND` / `CLEAR_LAKE` / `SWAMP` consts, the default handle a pond); a volume carries its
`WaterMaterial3d`, and `WaterVolume` is only the box. Wave textures are per material: every volume
of one material shares one FFT and one pair of textures (water_waves.rs keys by asset id). The
example's panel is `BuildAssetInspector` over the material, so editing it changes every volume
that uses it.

## Game side (zero / aurora_files)

- wow_import defers liquids: classic ADT `MCLQ` (Northshire's stream, Wetlands) → a water plane
  + `WaterVolume` per liquid chunk, the liquid type choosing coefficients (river, lake, ocean,
  swamp).
- Swimming: the KCC in a `WaterVolume` (buoyancy, swim controls).
