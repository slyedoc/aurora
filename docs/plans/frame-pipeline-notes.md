# Frame pipeline: implementation notes

Plan: `frame-pipeline.html`. Notes kept while implementing; newest last.

## Log

- 2026-10-13: inventory done (per-frame host writes, readbacks, fences, command buffers,
  descriptor sets, reallocations, destroyer). Baseline from Tracy: frame 7.41 ms, GPU-idle
  lower bound 1.74 ms median (wait end -> submit end).
- Phase 0 done: profiler adds synthetic `span` (first..last timestamp of the frame across all
  submits) and `idle` (gap since the previous frame's last timestamp); `cpu_record` = CPU
  ms from the fence wait to the frame submit; F3 line `gpu / idle / rec`; bench scopes
  `idle`/`span` (excluded from `busy`), default scopes now `busy,idle,tlas,trace,dlss`.
  Flat 1908x996 static spawn: span ~6.1-6.3 ms, idle ~1.35 ms. Bench `baseline-pipeline`
  (3840x1080 fullscreen, vsync): busy 13.0, idle 0.75, trace 6.3, dlss 2.7, tlas 0.62.
- Phase 1 sequencing (refined from the plan): 1a timeline replaces frame/readback/pick
  fences; 1b per-frame host writers -> parity/update-buffer; 1c bindless
  UPDATE_UNUSED_WHILE_PENDING + deferred slot reuse; 1d latch (exposure readback, screenshot/
  video collect, profiler readback after the wait); 1e rare writers wait for the previous
  frame on the GPU; 1f reorder (acquire per parity, record before the wait). Early submits
  (transform readback, pick/scene) start with a full barrier so they order after the
  previous frame's work on the queue instead of relying on the CPU wait.
- 1a done: `timeline.rs` (`GpuTimeline`, `SubmitPoint`, submit2 with timeline signal) on
  `RenderDevice`; `FrameSync` keeps its API but waits the last frame's point; picker fence +
  binary semaphore -> its submit point (the frame submit waits it on the GPU); transform
  readback fence -> point. Flat Northshire with validation: clean.
- 1b-1f done: uniforms + environment table + ReSTIR frame via update-buffer at the head of
  the scene cmd; SBT per parity; TLAS staging, picker rays, skin palettes in `Flip<T>`
  (render_buffer.rs); profiler pools per parity (`begin_recording` / `read_previous`);
  bindless UPDATE_UNUSED_WHILE_PENDING + slot retirement on the timeline; exposure latch
  (host `latch` buffer copied into state); screenshots/video collected by frame index after
  the wait; acquire semaphores per parity; `run_transfer_commands_after_frames` (GPU waits
  the last submitted point) for lights, sky_importance, landscape page/sculpt/bake/erosion/
  interaction; `wait_previous_frame` removed; render_frame records, then waits, ticks the
  destroyer, reads the profiler, latches, submits. Flat Northshire with validation: clean,
  idle 1.35 -> ~0.43 ms, span ~5.9 ms.
- XR (WiVRn) verified by the user after 1f.
- Bench `pipeline-1` vs `baseline-pipeline` (3840x1080 fullscreen, vsync, clocks unlocked):
  idle 0.75 -> 0.26 ms, busy 13.0 -> 11.7, trace 6.3 -> 5.6, dlss 2.7 -> 2.5, tlas 0.62 ->
  0.61. Busy/trace moved more than the bubble explains: likely partly clock drift.

- Phase 1.5 done (Reflex): `latency.rs` (`Latency`, `LowLatency` Off/On/Boost on
  `RenderSettings::low_latency`, default On); device enables `VK_NV_low_latency2` +
  `VK_KHR_present_id` (windowed, non-XR); swapchain created with latency mode, re-applies the
  sleep mode per swapchain generation; `First` sleeps (`vkLatencySleepNV` on its own
  timeline semaphore, bounded 100 ms wait) and marks SIMULATION_START; SIMULATION_END before
  the render sets; RENDERSUBMIT_START/END and PRESENT_START/END in render_frame; submits and
  present carry the present id. Driver timings read every 30 frames -> F3 `lat`. Measured:
  sim start -> present end ~5.3-5.5 ms; GPU idle with Reflex On 0.44 vs Off 0.46 ms (no
  throughput cost).
- Phase 2a done: queue created INTERNALLY_SYNCHRONIZED when supported (feature queried,
  `vkGetDeviceQueue2`); the mutex is kept. Validation clean.

- Phase 2 done (parallel recording): render_frame records a serial head (UI surfaces, transforms
  submit, SHaRC first allocation, exposure + atmosphere into the view lane's buffer), then three
  lanes on the compute task pool, each on its own command pool: scene (uniforms, frame commands,
  skins, TLAS, pick, lights, SHaRC), views (trace per view), present (XR eyes, image targets,
  window, gizmos, UI, copies). One submit, fixed order. A frame that grows the instance table or
  the TLAS (`TLAS::stable`) runs the scene lane first. Profiler scopes are `&self` (mutex).
  Tracy, Northshire flat, validation on: record 1.86 -> 1.64 ms (every section 1.4-2x slower in
  parallel). Validation off: whole record 0.25 ms (lanes 0.13 / 0.12 / 0.02). Recording cost is
  the validation layer; the frame stays GPU-bound (1.6 ms waiting on the previous frame).
- Phase 3 done (descriptor heaps): descriptor_heap.rs (resource + sampler heap, host-visible,
  written with vkWrite*DescriptorsEXT; image views recorded by `RenderDevice::create_view`),
  shaders unchanged via VkShaderDescriptorSetAndBindingMappingInfoEXT (bindless = constant
  offset; guides / post inputs / gizmo depth = push index; TLAS = push address). No descriptor
  set, layout or pool left except compute's push-constant layouts and NGX. Validation clean.
- The bug on the way: Ray Reconstruction resolved a stable blur (Northshire: black surfaces under
  a grey haze) although every guide matched HEAD to rounding and the feature alone (HEAD + the
  extension enabled) rendered fine. NGX's kernels address images by vkGetImageViewHandleNVX
  handles into the driver's own descriptor table; with our heap current they read our heap. A
  separate command buffer, a separate submit, binding our heap, a bigger reserved range, no
  sampler heap, and pushing constants before evaluate all failed; binding any descriptor set
  before evaluate (`DescriptorHeaps::leave`) fixes it. Driver 615.71.09 (610.43.02 earlier,
  same result).

## Deviations

- `run_transfer_commands*` keep their own transient fences (they block on them immediately;
  converting buys nothing and touches every asset worker). Revisit in phase 2 (queues).
- Transform staging (`gpu_transform.rs`) is not doubled: `submit` and `apply_readback` both
  wait the previous transform submit's point before the staging is rewritten, so it is
  never in use when written.
- Fog-volume and portal tables: a fresh buffer per change (old one through the destroyer)
  instead of `Flip`: they change rarely, and a fresh buffer needs no second copy kept in
  sync.
- Present timing: `VK_KHR_present_wait2` not added -- Reflex's `vkGetLatencyTimingsNV`
  already gives the per-frame present timings for F3. `VK_EXT_swapchain_maintenance1` present
  fences deferred: they would only replace the `queue_wait_idle` on resize (rare, works).
- Queue mutex kept with the internally synchronized queue (belt and braces; removing it is
  churn across every submit site for no measured gain).
- Shaders keep `[[vk::binding]]` and are mapped at pipeline creation (D8 proposed Slang's
  `spvDescriptorHeapEXT`): no shader changes, and Slang's heap path is untested.
- NGX evaluates stay in the view lane, after `DescriptorHeaps::leave` (a dummy descriptor set).
- `create_buffer` only queries a device address for SHADER_DEVICE_ADDRESS buffers (the screenshot
  readback buffer tripped VUID-VkBufferDeviceAddressInfo-buffer-02601).
