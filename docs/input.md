# Input design: desktop + VR, one action layer, UI in the world

Status: phases 0-3 built (untested on the Quest), 2026-09-03. Companion to `vr.md` (head tracking + stereo, done) — this doc
covers everything the headset still can't do: controllers/hands, pointing at UI, and running
the terrain tools from inside VR.

## Where we are

Facts from the code as of aurora `a5ed65b` / zero `41bb019`:

- **XR is head-only.** `xr.rs` has no action sets, no controllers, no hand tracking, no
  `xrSyncActions`. The head pose never reaches the ECS — `render_frame` consumes
  `locate_views` inline. The tracking-space origin is the single `Camera3d` entity's
  transform (`anchor * head`), and zero strips pitch from that transform in two ad-hoc places
  (`avatar.rs` wow XR branch, `ahoy.rs::xr_flatten_camera`).
- **UI-in-3D already works end to end, for the mouse.** `UiSurfacePanel` + a bare `Camera`
  with `RenderTarget::Image` renders a feathers tree to a texture that any material samples;
  `drive_surface_pointers` casts the *window cursor* through the `Camera3d`, hits panel
  planes, and feeds bevy_picking a `PointerId::Custom` located on the image target. Hover,
  press, drag, sliders all work (sim console). It is hard-wired to one pointer = the mouse.
- **Screen-space UI never reaches the headset.** `draw_ui` and the gizmo pass draw only into
  the window swapchain (spectator). The dev panel, dialogue, menus, the terrain brush ring:
  all invisible in VR.
- **Feathers inspector is retargetable for free.** `build_resource_inspector(world, TypeId,
  host)` spawns under any entity you hand it; put the host under a `UiTargetCamera` root and
  it renders to a panel texture unchanged. Enum fields become a row of buttons, numbers become
  sliders, bools checkboxes. `TerrainEditor` is already `Reflect` + registered — it's just not
  mounted anywhere.
- **BEI (0.26, fork `bevy-main`) has the extension point we need**:
  `Binding::Custom(CustomInput)` + `CustomInputs::register_input()`; write `ActionValue`s
  into the `CustomInputs` map in `PreUpdate` (after `InputSystems`, before
  `EnhancedInputSystems::Update`) and modifiers/conditions/contexts apply as for any key.
- **Zero's input is three things pretending to be one**: two BEI contexts (`Player`,
  `AhoyInput`), ~22 raw `ButtonInput<KeyCode>` systems, and one app-ext key (`F12`). Known
  clash: `F4` is both aurora's debug-view cycle and the terrain editor toggle.

## Goals

1. Controllers (Quest Touch via WiVRn first) and, later, hands drive the game and tools.
2. Every piece of UI we build is a **component** you spawn in the world; desktop mouse and
   VR controller rays both operate it; the feathers inspector is the default way to expose a
   reflected struct as a panel.
3. Terrain height/paint tools are usable seated in VR: tool panel on the wrist, brush on the
   controller ray.
4. One action vocabulary (BEI) with keyboard/mouse, gamepad and XR bindings side by side, so
   a system never asks "which device".

## Architecture: four layers

```
   OpenXR runtime          window / gamepad
        │                        │
 [L0] XrRig children + XrInput   bevy::input                     (aurora xr.rs)
        │                        │
 [L1] xr_to_bei → CustomInputs ──┴─→ BEI contexts: Shell, Player, Fly, TerrainEdit   (zero src/input/)
        │
 [L2] UiPointerSource entities (mouse ray | aim pose) → PointerInput on panel textures (aurora ui_render.rs)
        │
 [L3] UiPanel3d / InspectorPanel3d components → feathers trees on quads          (aurora ui_panel.rs)
```

### L0 — tracking and raw XR input into the ECS (aurora `xr.rs`)

Keep the existing convention: **the `Camera3d` entity is the rig**. `XrPlugin` spawns one
tracked entity per pose when the session comes up:

```rust
#[derive(Component)] pub enum XrTracked { Head, Grip(XrHand), Aim(XrHand) }
#[derive(Component)] pub struct XrPose { pub local: Transform, pub valid: bool }   // pose in rig space
#[derive(Resource, Default)] pub struct XrInput { pub left: XrHandState, pub right: XrHandState, pub focused: bool }
pub struct XrHandState { pub active: bool, pub trigger: f32, pub squeeze: f32, pub thumbstick: Vec2,
                         pub thumbstick_click: bool, pub primary: bool, pub secondary: bool, pub menu: bool }
```

- Tracked entities are **roots whose `Transform` is written as `rig * pose`** each frame
  (once after the poll, once again before propagation), not children of the camera: with
  aurora's default `TransformPlugin` (no CPU propagation) a camera that gained children would
  stop getting its `GlobalTransform` synced. Same effect: free cam, orbit cam, and any future
  teleport move the controllers for free, which closes the open ask "F toggle must include the
  XR rig". Apps parent their own visuals (laser, wrist panel) under these; they persist across
  game states.
- One action set `aurora` with actions: `aim_pose`, `grip_pose` (Posef, subaction paths
  `/user/hand/left|right`), `trigger`, `grip` (f32), `thumbstick` (Vec2), `thumbstick_click`,
  `primary` (A/X), `secondary` (B/Y), `menu` (bool). Suggested bindings for
  `oculus/touch_controller` (Quest) and `khr/simple_controller` (fallback). `attach_action_sets`
  once after session creation.
- New system set `XrSystems::Poll` in `PreUpdate`, before `bevy::input::InputSystems`:
  `sync_actions` (skip unless session `FOCUSED`), read states, `locate` each pose space at
  the last `predicted_display_time` (the one `begin_frame` got in `Last`), write child
  `Transform`s and `XrInput`. None of these touch the `VkQueue`, so they stay outside the
  queue mutex; the mutex rule in `xr.rs:351` is unchanged for the frame/swapchain calls.
- Move the pitch-strip into aurora: `xr_level_rig` in `PostUpdate` before transform
  propagation strips pitch/roll from the `Camera3d` transform whenever `XrState` exists.
  Delete both zero copies. `FreeCamera` keeps its own pitch in `FreeCameraState` and
  re-writes the transform every frame, so leveling after it is safe.
- Hands (`XR_EXT_hand_tracking`) are phase 4: `XrHandJoints([Isometry3d; 26])` on the grip
  entity, pinch strength synthesized into `trigger` so everything above just works.

Debug: a `dev`-feature system draws 5 cm cubes at grip/aim (as meshes, since gizmos don't reach
the headset) and logs `XrInput` on change.

### L1 — actions (zero `src/input/`)

BEI stays an app decision; aurora does not depend on it. Zero gets a `src/input/` module that
owns all bindings, and the raw-key systems get deleted as their actions land.

- `xr.rs`: `XrBindings` resource created at startup — one `CustomInput` id per
  `(hand, channel)` — and a `PreUpdate` system that copies `XrInput` into `CustomInputs`
  each frame. Helper so bindings read like keys:
  `Binding::xr(Hand::Right, XrChannel::Trigger)`.
- Contexts (each a `Component` + `add_input_context`):

  | Context | Active when | Actions | kb/mouse | gamepad | XR |
  |---|---|---|---|---|---|
  | `Shell` | always | `Back`, `ToggleFly`, `CycleSky`, `ToggleDevPanel`, `Screenshot` | Esc, F, K, F2, F12 | Start, — | left menu, left stick click, — |
  | `Player` | walking | existing `Move/Orbit/Zoom/…` | as today | left stick / right stick | left stick move, right stick x = snap turn |
  | `Fly` | free cam | `Fly: Vec3`, `Turn: f32`, `Fast: bool`, `Speed: f32` | WASD+EQ, RMB look, Shift, wheel | sticks, trigger | left stick xz, right stick y = up/down, right stick x = turn, grip = fast |
  | `TerrainEdit` | editor on | `Apply`, `Sample`, `Radius: f32`, `Strength: f32`, `NextTool`, `PrevTool`, `Save` | LMB, RMB, `[ ]`, `- =`, digits, Ctrl+S | — | right trigger, right grip, right stick y, left stick y, A/B, panel button |

- `Fly` replaces bevy's `FreeCamera` with a small BEI-driven flyer (`FlyRig` on the camera
  entity). Reason: `FreeCamera` reads the keyboard directly and can't take a thumbstick;
  one controller for both devices beats two that drift apart. Same feel on desktop.
- Activity is a tiny state machine in `input/mod.rs`: `ToggleFly` swaps `Player`↔`Fly`;
  the editor toggle swaps `TerrainEdit` on with `consume_input` so LMB/trigger stop reaching
  `Player`.
- Fix the `F4` clash now: terrain editor toggle moves to the panel + `F7`; F1–F4 stay
  engine-owned.
- Rebinding from a file (RON/bsn `InputSettings`) is the "app ext" you liked; it's a phase-5
  layer over the same tables, not something to build first.

### L2 — pointers as entities (aurora `ui_render.rs`)

Generalize `drive_surface_pointers` from "the mouse" to "every `UiPointerSource`":

```rust
#[derive(Component)]
pub struct UiPointerSource {
    pub id: PointerId,            // Custom(uuid), assigned on add
    pub buttons: [bool; 3],       // written by the app each frame (mouse buttons / trigger)
    pub hit: Option<UiPointerHit>, // written by aurora: panel entity, world point, distance
}
```

- The entity's `GlobalTransform` is the ray (origin, `-Z`, matching the OpenXR aim pose).
  Under XR the ray world pose is `rig_global * aim_local`, computed explicitly in `PreUpdate`
  so it doesn't wait for propagation.
- Aurora spawns one `WindowCursorPointer` source on desktop and updates its transform from
  the cursor each frame; that's the current code path, relocated. Zero adds
  `UiPointerSource` to both aim entities with `buttons[0] = trigger > 0.5`.
- Per-source `PointerLocation` and press/release bookkeeping — two hands can hover two
  panels. Nearest panel wins per source; no hit parks the pointer off-screen as today.
- `hit` is the arbitration signal for world tools: the terrain brush skips a frame whose
  source is over a panel, so a slider drag never paints the ground behind it.
- Laser: a thin emissive cylinder mesh from aim to `hit` (or 2 m), spawned by zero under the
  aim entity. Not a gizmo — gizmos are skipped in the eye passes.

### L3 — panels as components (aurora, new `ui_panel.rs`)

```rust
#[derive(Component)]
pub struct UiPanel3d { pub size: Vec2 /* m */, pub px: UVec2, pub scale: f32, pub nits: f32 }
#[derive(Component)] pub struct UiPanel3dRoot(pub Entity);   // the UiTargetCamera root; add children here
#[derive(Component)]
pub enum InspectorPanel3d { Resource(TypeId), Component { entity: Entity, ty: TypeId }, Entity(Entity) }
```

- `on_add` for `UiPanel3d` spawns everything the sim console builds by hand today: the
  placeholder image, the bare `Camera` + `RenderTarget::Image`, the `UiTargetCamera` root
  `Node`, the quad `Mesh3d` + `AuroraMaterial3d { emissive_texture }` + `UiSurfacePanel`.
  `scale` (2.0 for VR) goes into the target's `scale_factor` so taffy lays out at 2× — big
  hit targets without touching the widget code.
- `InspectorPanel3d` requires `UiPanel3d` and calls `build_resource_inspector` &c. with a
  host node under the root. Rebuild on `TypeId`/entity change. This is the "component-based
  feathers inspector in 3D" ask in one line at the call site:
  ```rust
  commands.spawn((InspectorPanel3d::resource::<TerrainEditor>(),
                  UiPanel3d { size: Vec2::new(0.28, 0.18), px: UVec2::new(896, 576), scale: 2.0, nits: 300.0 },
                  ChildOf(left_grip), Transform::from_xyz(0.0, 0.03, -0.12).with_rotation(WRIST_TILT)));
  ```
- Port the sim console and `examples/ui_panel.rs` to `UiPanel3d`; that removes the last
  hand-rolled surface.
- Dev panel: on desktop unchanged; when `XrState` exists, also spawn
  `InspectorPanel3d::resource::<DevUIState>()` at a fixed rig-relative pose, toggled by
  `ToggleDevPanel`. Same resource, two views.
- Screen-space UI in VR (menus, dialogue, boot): phase 5. Convention will be a head-locked
  `UiPanel3d` at 1.5 m that the screen roots re-target to (`UiTargetCamera` swap) when XR is
  up. Not needed to play or edit terrain.

### Terrain tools in VR (zero `wow_terrain.rs`)

- `brush` takes its ray from a `BrushPointer` resource (entity) instead of `viewport_to_world`:
  desktop = the window cursor pointer, XR = right aim. Skip when that source's `hit` is a
  panel. `Apply`/`Sample` come from the `TerrainEdit` actions.
- The brush ring becomes a **shader ring**, not a gizmo: `TerrainCursor { center, radius }`
  in the frame uniform; the terrain closest-hit adds a thin emissive ring where
  `|dist − radius| < ε`. Visible on desktop and in both eyes, no mesh, no extra pass.
- Wrist panel = `InspectorPanel3d::<TerrainEditor>` on the left grip: tool enum (button row
  = tool picker), texture index, radius, strength, flatten target, plus a `Save` button
  added under the same root. Texture thumbnails later.
- Thumbsticks tune radius/strength continuously so the panel is for picking, not fiddling.
- `TerrainEditor.active` gates the `TerrainEdit` context; the wrist panel is visible only
  while active.

## Phases

| # | Where | Deliverable | Status (2026-09-03) |
|---|---|---|---|
| 0 | aurora | `XrTracked` entities + `XrPose`, `XrInput`, action set + Touch/Index/simple bindings, `XrSystems::Poll`, `level_rig`; zero's ahoy pitch strip deleted | built, compiles with `--features xr`; **untested on the Quest** |
| 1 | aurora | `UiPointerSource` (+ `UiPointerSystems::{Aim,Drive}`), `UiPanel3d`, `InspectorPanel3d`; `ui_panel` example ported (sim console keeps `UiSurfacePanel` on its glb quad) | desktop verified: example panel renders + glows through `UiPanel3d`; inspector panel builds. Mouse press on a panel not re-verified after the refactor |
| 2 | zero | `src/input/`: `XrBindings` bridge, `Shell/Fly/TerrainEdit` contexts, `FlyRig` (replaces `FreeCamera`), wow + terrain raw-key systems deleted, F4 clash fixed (editor = F7) | built; Wow state boots clean, keys not exercised in the auto-run |
| 3 | zero | brush from the active pointer (cursor / right aim), shader ring (`TerrainCursor` → uniform → closest-hit), wrist `TerrainEditor` panel, stick radius/strength, panel `save` tick, XR dev panel summon, lasers + grip markers | built; ring shader compiles; **needs a desktop pass with the editor on, then the Quest** |
| 4 | aurora | hand tracking → `XrHandJoints`, pinch → trigger | not started |
| 5 | both | head-locked HUD panel for menus/dialogue in XR; `InputSettings` rebinding file; comfort (vignette on fly, snap-turn angle in dev panel) | not started |

Known rough edges after the build:
- The eleven gait-style keys in `avatar::controls` are gone: the moving gait is the
  player's reflected `Gait { style }` component, picked from a button row the avatar hangs
  under the dev panel (`gait_panel`). Gameplay sets it directly; only semantic actions
  (sneak/sprint) would ever get keys.
- The inspector's collapsed state is global per root: `DevUIState` starts collapsed on the XR
  dev panel because the screen panel collapsed it. One click on the header.
- Wrist panel placement (`wow_terrain::wrist_panel`) and the XR dev panel distance are guesses
  to tune in the headset.

## Decisions and why

- **Rig = the `Camera3d` entity; tracked poses are root entities that follow it.** Zero
  churn, every camera controller keeps working, the "F toggle includes the rig" ask
  disappears, and the camera's own `GlobalTransform` sync is untouched.
- **BEI bridge lives in zero, not aurora.** Aurora exports `XrInput`; the engine stays free of
  an input framework choice. Cost: one 40-line copy system.
- **Replace `FreeCamera` with a BEI `FlyRig`.** Two fly controllers (keyboard one, stick one)
  would diverge immediately.
- **Panels are textures in the traced scene, not an overlay layer.** They get lighting,
  occlusion, DLSS, and both eyes for free; the price is emissive nits tuning, which the sim
  console already solved.
- **Inspector-generated panels for tools, hand-built feathers only for player-facing UI.**
  The inspector's enum-as-button-row and range sliders are exactly a VR tool palette. If a
  field needs a bespoke widget, add a `#[reflect(@...)]` attribute and a leaf widget to
  `bevy_feathers_inspector` rather than hand-building the panel.

## Open questions (decide during phase 0/1, not blocking)

- Monado's simulated driver has no controllers; phases 0/3 verify on the Quest only. The
  pointer + panel layer verifies on desktop with the mouse.
- Snap vs smooth turn, and whether `Player` yaw in XR should come from the stick or the
  head. Default: stick snap-turn 30°, head is view only.
- `PointerId::Custom` per hand is fine for bevy_picking, but `InputFocus` is single: last
  pressed source wins keyboard focus. Acceptable.
