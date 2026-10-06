# Image formats

**The rule: the format carries the encoding.** An sRGB image is created with an `_SRGB`
Vulkan format. The hardware decodes it when sampling (before filtering) and encodes it when
it is a colour attachment. Shaders only ever read and write linear values. Nothing in a
shader converts sRGB by hand, and a texture's colour space is never guessed from its
material slot at runtime. The baked file states it.

Code: `src/color_target.rs` (targets and pipelines), `src/render_texture.rs` (uploads),
`aurora_files/crates/aurora_bsn/src/ktx.rs` (the baker).

## Render targets

What a target holds decides its format. A camera's target kind comes from the format its
`Image` declares (`TargetKind::of`).

| Target | Holds | Vulkan format | Image declares | Written by | Sampled as |
|---|---|---|---|---|---|
| Window swapchain | display colour | `B8G8R8A8_SRGB` (`DISPLAY_FORMAT`) | — | composite, gizmos, UI | — |
| XR eye targets | display colour | `DISPLAY_FORMAT` | — | composite | blitted to the XR swapchain |
| Camera target, display | display colour, exposed and tonemapped | `DISPLAY_FORMAT` | any 8-bit format (`camera_target_placeholder`: `Bgra8UnormSrgb`) | composite, gizmos | linear (UI viewport, thumbnail) |
| Camera target, radiance | physical radiance, no exposure or tonemap | `R16G16B16A16_SFLOAT` (`SCENE_RADIANCE_FORMAT`) | `Rgba16Float` / `Rgba32Float` | composite (`scene_radiance = 1`), no gizmos | linear radiance (an emissive in-world screen) |
| UI surface | display colour | `DISPLAY_FORMAT` | `ui_target_placeholder`: `Bgra8UnormSrgb` | UI | linear (panel material) |

A radiance target is lit scenery: author its material with `emissive` 1.0 and the texture
emits exactly what the camera saw. It is then exposed and tonemapped once, with the rest of
the frame. A display target is for screens: it is already tonemapped, so in the world it
should be treated as a monitor.

## Graphics pipelines

With dynamic rendering a pipeline only fixes its attachment formats, so each graphics
pipeline is built once per format it may draw into (`FormatPipelines`). A pass asks for the
pipeline matching its target.

| Pipeline | Built for | Notes |
|---|---|---|
| Composite (`quad.frag`) | `DISPLAY_FORMAT`, `SCENE_RADIANCE_FORMAT` | `scene_radiance` push constant picks the output |
| Gizmos (`gizmo.frag`) | `DISPLAY_FORMAT` | overlays draw on display targets only |
| UI (`ui.frag`) | `DISPLAY_FORMAT` | window and UI surfaces |

## Textures

The upload path takes these formats as they are (`vk_format_for`). Anything else (RGB, grey,
palette) is converted on the main thread to `Rgba8UnormSrgb` or `Rgba8Unorm`, keeping
whether the source was sRGB.

| bevy `TextureFormat` | Vulkan format | Sampled as | Mips | Typical source |
|---|---|---|---|---|
| `Rgba8UnormSrgb` | `R8G8B8A8_SRGB` | linear colour | generated on upload, filtered in linear light | PNG colour, procedural colour, glyph atlases |
| `Rgba8Unorm` | `R8G8B8A8_UNORM` | raw | generated on upload | procedural data |
| `Rgba16Unorm` | `R16G16B16A16_UNORM` | raw | none | 16-bit PNG data |
| `Rgba16Float` / `Rgba32Float` | `R16G16B16A16_SFLOAT` / `R32G32B32A32_SFLOAT` | raw | none | HDR sky |
| `Bc7RgbaUnormSrgb` | `BC7_SRGB_BLOCK` | linear colour | from the file | baked colour |
| `Bc7RgbaUnorm` | `BC7_UNORM_BLOCK` | raw | from the file | baked packed data |
| `Bc5RgUnorm` | `BC5_UNORM_BLOCK` | `(x, y, 0, 1)` | from the file | baked normal map |
| `Bc4RUnorm` | `BC4_UNORM_BLOCK` | `(r, 0, 0, 1)` | from the file | baked single channel |
| `Bc6hRgbUfloat` | `BC6H_UFLOAT_BLOCK` | raw HDR | from the file | baked HDR |
| `Bc1RgbaUnorm[Srgb]` / `Bc3RgbaUnorm[Srgb]` | `BC1_RGBA_*` / `BC3_*` | as declared | from the file | third-party KTX2 |

Hit shaders read normal maps through `normalTexel`, which rebuilds Z from XY, so RGB and
two-channel maps both work. KTX2 files are loaded by bevy's loader with
`CompressedImageFormats::BC`. The fork's loader takes sRGB or linear from the file's
transfer function, never from `ImageLoaderSettings::is_srgb`.

## Baked material textures

Every importer bake ends in `aurora_bsn::finish_textures`. It bakes each texture a `.bsn`
material field names, choosing the format from the field (never from the file), and rewrites
the field to the `.ktx2`. Files are KTX2 with zstd-supercompressed levels and a full mip
chain made offline.

| Material field (suffix) | Role | Vulkan format | Bits/texel | Mips built as |
|---|---|---|---|---|
| `base_color_texture`, `emissive_texture` | colour | `BC7_SRGB_BLOCK` | 8 | 2x2 box in linear light, alpha linear |
| `normal_map_texture` | normal | `BC5_UNORM_BLOCK` | 8 | decoded, averaged, renormalised |
| `metallic_roughness_texture`, `occlusion_texture` | packed data | `BC7_UNORM_BLOCK` | 8 | raw average |
| `depth_map` | single channel | `BC4_UNORM_BLOCK` | 4 | raw average |

Suffix matching covers layered materials too (`layer_normal_map_texture`,
`detail_base_color_texture`). A texture named by two roles bakes as the first in the order
above and is reported.

## Internal buffers

| Buffer | Format |
|---|---|
| DLSS colour, normal + roughness, output | `R16G16B16A16_SFLOAT` |
| DLSS diffuse / specular albedo guides | `R8G8B8A8_UNORM` |
| DLSS depth, specular hit distance, exposure | `R32_SFLOAT` |
| DLSS motion | `R16G16_SFLOAT` |
| White / default-normal fallbacks | `R8G8B8A8_UNORM` |

## Later

- **HDR displays:** a window in `A2B10G10R10_UNORM_PACK32` with a PQ-encoding composite (or
  `R16G16B16A16_SFLOAT` scRGB). That is one more `TargetKind` and one more entry in the
  composite's format list.
- **Neural Texture Compression (RTXNTC):** a smaller baked file that transcodes to the same
  BCn formats on load. Sampling it in a hit shader needs cooperative vectors in the
  ray-tracing pipeline, which faults on this hardware today.
