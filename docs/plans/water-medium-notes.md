# Water medium: implementation notes

Plan: docs/plans/water_medium.html.

## Built

- src/water.rs (was water_volume.rs): `WaterSurface` requires `WaterMaterial3d`; `WaterKind`
  {Lake, River, Ocean, Pond, Swamp} inserts its `WaterPresets` handle; `RIVER` / `OCEAN` presets;
  `WaterMedia` keeps a row per `WaterMaterial` in use (freed on the asset's Unused/Removed) and
  uploads `GpuWaterMedium` rows (the old volume row less its transform). `WaterVolume` is gone.
- Each surface instance gets `SurfaceRow { row, flags: WATER }` (surface_group.rs); the TLAS
  writes it into the instance's material record (`surface_param_index`, `RTXMaterial::flags`,
  was `pad`). A water-flagged instance is never force-opaque, so shadow rays run its any-hit.
- water_surface.slang reads its row from the record, sets `payload.absorption` to 0 (the medium
  attenuates, not the glass) and hands the raygen its water id (row + 1) in the lobes word's top
  byte.
- Shadows: `ShadowPayload { visible, water, water_t }`; `shadow_any_hit` keeps the nearest water
  crossing; `shadowInline` (was visibleInline) reads water candidates' records itself
  (`MATERIAL_*_WORD`, pinned by blas.rs's test). `traceLight` = visibility x `waterAlong`.
- Path: `water` (a water id) in `main`; a segment ending on a water surface takes that surface's
  water when it met it from below, none from above (the leak guard), else the path's; after a
  scatter at a water surface the path is in its water when the new direction is below the face.
  `waterSegment` is one medium over [0, t].
- ReSTIR: `PtVertex.water`, `PtReplay.water`; the GI surface keeps its water in the restir
  material word (`restirSetWater`); reconnections use the visibility ray's crossing
  (`ptReconnectWater`); the k = 1 own target traces an inline ray for its crossing.
- WoW importer (aurora_files): one surface per liquid kind per tile (`<map>_X_Y_<kind>`) with its
  `WaterKind`; magma a lava surface (emissive, `LAVA_NITS`). Boxes and the terrain floor scan gone.

## Deviations

- **Rows per instance, not per AuroraMaterial.** The plan had `SurfaceGroupData::set_row` per surface
  material, but rows there are keyed by `AuroraMaterial`, which water surfaces share (every WoW
  water surface had the same transmissive material) while their waters differ. A per-instance
  `SurfaceRow` component overrides the material's row instead.
- **Rising/falling from the geometry, not a facing bit.** A shadow crossing's direction is its
  ray's `d.y` sign (surfaces are near level); path segments use the hit's geometric normal turned
  up. No facing bit in the payload.
- **At most 255 water materials** (the water id is a byte in the lobes word and the restir word);
  past that a surface stays glass, with a warning.
- **The GI surface's water lives in the restir material word's coat-roughness byte**, which is
  free without a clear coat; a clear-coated surface under water loses its water for replay.
- **Steps built together**, not M1..M4 one at a time: the box path could not coexist with the
  surface-row table without double attenuation; verified at the end against the old build.
- ReSTIR DI's own shadow rays still ignore water (they did before too).

## After the plan

- Lava read as dark mud by day (150 nits under daylight exposure): `LAVA_NITS` 10000, the Burning
  Steppes lava texture as colour and emission, sliding by `AuroraMaterial::uv_scroll` (new: UV per
  second on the shader clock, in `RTXMaterial`, now 28 words). Lava cells keep a quad each so
  their half-float UVs stay small.
- WMO liquids (MLIQ, in wow.export's WMO json `groups[].liquid`): Blackrock Mountain's lava,
  Stormwind's canals, the Deadmines; a surface per group and kind in the WMO's frame, placed like
  its interior doodads. Tile flags: bit 3 none, low two bits water / ocean / magma / slime.
- Particle-emitter doodads (`is_effect`: particleemitters/, fx/, lavasteam, splashes, steam
  vents; 180 placements) are dropped: the export has only their stand-in geometry. Unlit lava
  doodads (statue lavafalls, bubble holes) take `LAVA_NITS`.

## Results

- Pond (examples/water.rs), references at the example's own camera, 1024 frames over BRP
  (`aurora.capture`; the example takes no `--capture`): new vs the box build (HEAD f2e39ea)
  ratio 1.010, block error 0.050 (refcmp). The box build was within 5 % of Beer-Lambert (W1).
- WoW full map rebaked: 134 lake, 40 river, 416 ocean, 4 magma surfaces; no boxes.
- Wetlands lake (camera at -533, 250, 8400), far past the old 64-box radius: deep blue with the bed
  showing in the shallows, where it was clear glass.
- GPU on the full map: busy 4.6 ms a frame (trace 1.0, gi spatial 0.43).
- Stranglethorn shore (tile 34,46): ocean shallows cyan to deep blue.
- Not yet checked: a lava pool on screen (the 4 magma tiles are 33,44 / 33,46 / 34,46 / 35,47:
  small pools, none found from the air), ReSTIR GI on vs off under water, a before/after timing
  on the same view.
