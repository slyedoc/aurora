//! `DlssRenderer` with the SDK compiled out: nothing is ever active.

use ash::vk;
use bevy::math::Mat4;

use super::{AuroraDlss, DlssPlan, GuideViews};
use crate::render_device::RenderDevice;

pub struct DlssRenderer;

impl DlssRenderer {
    pub fn new(_rd: &RenderDevice) -> Option<Self> {
        log::info!("dlss: compiled out (set DLSS_SDK at build time)");
        None
    }
    pub fn prepare(
        &mut self,
        _rd: &RenderDevice,
        _o: vk::Extent2D,
        _m: AuroraDlss,
    ) -> Option<DlssPlan> {
        None
    }
    pub fn guide_views(&self) -> Option<GuideViews> {
        None
    }
    pub fn output_view(&self) -> Option<vk::ImageView> {
        None
    }
    pub fn record_pre_trace(&mut self, _rd: &RenderDevice, _cmd: vk::CommandBuffer) {}
    pub fn record_evaluate(
        &mut self,
        _rd: &RenderDevice,
        _cmd: vk::CommandBuffer,
        _v: Mat4,
        _p: Mat4,
        _j: [f32; 2],
        _r: bool,
    ) {
    }
    pub fn destroy(&mut self, _rd: &RenderDevice) {}
}
