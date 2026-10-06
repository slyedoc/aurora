//! What a colour attachment holds, and the graphics pipelines built for each format.
//!
//! The format carries the encoding: display targets are sRGB (the attachment encodes on
//! store, sampling decodes), scene-radiance targets are half float. Shaders always write
//! and read linear values. A target's kind comes from the format its `Image` declares, so
//! a camera rendering for an in-world screen asks for `Rgba16Float` and gets radiance.

use ash::vk;
use wgpu_types::TextureFormat;

use crate::render_device::RenderDevice;

pub use crate::swapchain::DISPLAY_FORMAT;

/// Pre-tonemap scene radiance, for targets a world material shows as emission.
pub const SCENE_RADIANCE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// What a render target holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// Tonemapped, exposed colour for a screen: the window, a UI viewport, a thumbnail.
    Display,
    /// Physical radiance before exposure and tonemapping.
    SceneRadiance,
}

impl TargetKind {
    /// The kind an image target declares by its format: float formats hold radiance.
    pub fn of(format: TextureFormat) -> Self {
        match format {
            TextureFormat::Rgba16Float | TextureFormat::Rgba32Float => Self::SceneRadiance,
            _ => Self::Display,
        }
    }

    pub fn format(self) -> vk::Format {
        match self {
            Self::Display => DISPLAY_FORMAT,
            Self::SceneRadiance => SCENE_RADIANCE_FORMAT,
        }
    }
}

/// One graphics pipeline per colour-attachment format it may draw into.
#[derive(Debug)]
pub struct FormatPipelines {
    pipelines: Vec<(vk::Format, vk::Pipeline)>,
}

impl FormatPipelines {
    pub fn build(formats: &[vk::Format], mut build: impl FnMut(vk::Format) -> vk::Pipeline) -> Self {
        Self {
            pipelines: formats.iter().map(|&format| (format, build(format))).collect(),
        }
    }

    /// The pipeline for `format`, if one was built for it.
    pub fn get(&self, format: vk::Format) -> Option<vk::Pipeline> {
        self.pipelines
            .iter()
            .find(|(f, _)| *f == format)
            .map(|(_, pipeline)| *pipeline)
    }

    pub fn destroy(&self, render_device: &RenderDevice) {
        for (_, pipeline) in &self.pipelines {
            render_device.destroyer.destroy_pipeline(*pipeline);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_images_hold_radiance_and_the_rest_display() {
        assert_eq!(TargetKind::of(TextureFormat::Rgba16Float), TargetKind::SceneRadiance);
        assert_eq!(TargetKind::of(TextureFormat::Bgra8UnormSrgb), TargetKind::Display);
        assert_eq!(TargetKind::of(TextureFormat::Rgba8Unorm), TargetKind::Display);
        assert_eq!(TargetKind::Display.format(), DISPLAY_FORMAT);
    }
}
