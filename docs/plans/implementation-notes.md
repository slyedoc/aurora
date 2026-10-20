# Implementation notes: physics on GPU transforms (the simple steps)

Plan: `physics_transforms.html`. Scope agreed 2026-10-20: not the full rewrite, only the steps that
remove the measured pain (avian walking every collider every fixed step, six steps a frame):

1. zero: cap the fixed-step catch-up.
2. aurora: `extract_transforms` publishes the entities whose local or parent changed (seeds).
3. avian fork: a `StaticBody` marker kept by `RigidBody`'s hooks, and a `hierarchy_moves` mode in
   `PhysicsTransformConfig`: the two per-step walks (`transform_to_position` over every body and
   collider, `propagate_collider_transforms` over every collider tree) replaced by a per-step
   check of non-static bodies only, plus a pass over the seeds' collider subtrees.

PR #1039 (`PhysicsTransform` replacing `Position`/`Rotation`, a draft) was reviewed first: the new
interfaces are entity lists so a rebase over it stays mechanical.

## Decisions while building

- **`StaticBody` marker** (avian, `rigid_body/mod.rs`): `RigidBody` is immutable, so its `on_insert`
  hook sees every change and adds or removes the marker; `on_remove` clears it. The per-step check
  is `Query<.., (With<RigidBody>, Without<StaticBody>)>`, an archetype filter: a few hundred rows,
  never the map.
- **One shared sync body** (`TransformSync`, a `SystemParam`): `transform_to_position` (Cpu mode,
  unchanged behaviour), `transform_to_position_moving` and `apply_hierarchy_moves` all call
  `TransformSync::apply`, so the three compose identically (f64, environment-relative,
  tolerances, a user's `Position` write wins).
- **What the seed pass recomposes**: `Position` for statics and bodiless colliders (filtered by
  `iter_many_mut`; colliders on a body follow it in `update_child_collider_position`), and
  `ColliderTransform` for every collider reached. Non-static bodies are left to the per-step check,
  which composes them every step anyway (so a moved parent carries them).
- **The descent** follows `AncestorMarker<ColliderMarker>`, `ColliderMarker` and `RigidBody`, and
  stops at a `PhysicsEnvironment` (moving one moves nothing inside its frame).
- **`ColliderTransform` from the chain** (`collider_transform_of`) reproduces
  `propagate_collider_transforms` exactly: translation and rotation relative to the nearest body
  (the entity itself if it is one), scale accumulated from the root, and a bodiless root's chain
  starting from the root's scale only, as propagation's root step does.
- **Adds**: an `On<Add<(RigidBody, ColliderTransform)>>` observer queues the entity.
  `init_physics_transform` places a new body in world space, not its environment's; and a collider
  attached frames after its entity spawned (aurora's `attach_colliders`) has no transform change of
  its own to list.
- **aurora's seeds** are exactly what `extract_transforms` uploads: a first sight, a reparent, a
  local that really changed (animation rewriting the same value is no seed), an orphaning. Past
  2^18 entries the list is deduped in place, for frames where physics doesn't step.
- **The catch-up cap** is zero's (`MAX_CATCH_UP_STEPS = 3`, as `Time<Virtual>` max delta), not
  aurora's: it's a game policy.

## Deviations

- **No new avian unit tests.** avian can't build on its own here (it pulls unpatched bevy main and
  fails in `Ease`/`Curve`, an existing break, not from this change), so its test suite can't run.
  The conservative option was to validate in the real app instead (Results) and leave the tests for
  when the standalone build is fixed. Cpu mode is still avian's default, so upstream behaviour is
  unchanged.
- **`--timeout` now counts game time**, a side effect of the catch-up cap: when frames are slower
  than three fixed steps, virtual time runs slower than wall time. A 75 s timeout ran past 3 min
  on the full map. Left as is; the MCP server stops games itself.

## Results

Full Eastern Kingdoms (`examples/eastern_kingdoms`, headless, Tracy over 110 s):

| | Before | After |
|---|---|---|
| Frame (CPU, mean) | 139 ms | 72 ms |
| Physics (`FixedMain`) per frame | 113 ms (6.3 steps × 17.8 ms) | 5 ms (2.1 steps × 2.3 ms) |
| `transform_to_position` | 4.7 ms per step | 0.002 ms per step (non-static only) |
| `propagate_collider_transforms` | 10.6 ms per step | off (`apply_hierarchy_moves`: 0.03 ms per step) |

Northshire (`zero -s wow`, headless, over BRP):
- The player and Roy (both kinematic, KCC-driven) land and stand: the player at y = 1.15 on its
  spawn point.
- All 3,334 static bodies' `Position` equal their GPU `GlobalTransform` (worst difference 0).
- Moving tile 32,48's wrapper +10 yd in x carried exactly its 606 statics +10 x; nothing else moved.

Now leading the full map's frame (not physics):
- **`prepare_instances` (~30 ms a frame, steady):** about 200k of the 3.85M TLAS slots stay
  pending, all waiting on textures. Each frame re-resolves every pending slot, and images upload
  on one build thread, so the full map's 5,262 textures trickle in over minutes.
- **The frame fence (~24 ms):** waiting on the GPU.
- **avian `update_collider_aabbs` (~4 ms a step):** still visits every collider to check its pose,
  the remaining per-step scan.
