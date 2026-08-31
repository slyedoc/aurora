//! Aurora's own guide-buffer visualizer -- stable, unlike the NGX dev snippet's.
//!
//! The composite (quad.frag) can sample any of the DLSS guide images instead of the
//! reconstructed output. Everything renders through aurora's exposure/tonemap control, so a
//! static scene shows a static visualization; the snippet's overlay redraws everything it
//! shows through a per-frame auto-exposure estimate over the noisy input, which flickers by
//! construction (measured: scene stable to 0.2% while the snippet HUD swung 37%).

use bevy::prelude::*;

/// Which buffer the composite displays -- on every `Camera3d`; `F4` cycles it.
#[derive(Component, Reflect, Clone, Copy, PartialEq, Eq, Debug, Default)]
#[reflect(Component, Default, Clone, PartialEq)]
pub enum AuroraDebugView {
    /// The reconstructed frame (normal rendering).
    #[default]
    None,
    /// RR's noisy colour input (pre-exposed linear HDR, tonemapped like the output).
    Color,
    /// World-space normals, `n * 0.5 + 0.5`.
    Normals,
    /// Perceptual roughness (the normals guide's alpha).
    Roughness,
    /// Diffuse albedo guide.
    Diffuse,
    /// Specular albedo guide (EnvBRDF).
    Specular,
    /// Linear depth, `exp2(-d / 32)`.
    Depth,
    /// Specular hit distance, same encoding as depth.
    SpecHit,
    /// Motion vectors (pixels), `mv * 0.1 + 0.5`.
    Motion,
}

impl AuroraDebugView {
    pub const ALL: &'static [AuroraDebugView] = &[
        Self::None,
        Self::Color,
        Self::Normals,
        Self::Roughness,
        Self::Diffuse,
        Self::Specular,
        Self::Depth,
        Self::SpecHit,
        Self::Motion,
    ];

    /// The value quad.frag switches on.
    pub fn shader_index(self) -> u32 {
        self as u32
    }

    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
}

/// Every 3D camera carries a view (so it is always there to inspect).
fn default_view(
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<AuroraDebugView>)>,
) {
    for camera in &cameras {
        commands.entity(camera).insert(AuroraDebugView::default());
    }
}

/// F4 cycles the view on every 3D camera.
fn cycle_view(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut cameras: Query<&mut AuroraDebugView, With<Camera3d>>,
) {
    if keyboard.just_pressed(KeyCode::F4) {
        for mut view in &mut cameras {
            *view = view.next();
            log::info!("debug view: {:?}", *view);
        }
    }
}

pub struct DebugViewPlugin;

impl Plugin for DebugViewPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraDebugView>();
        app.add_systems(Update, (default_view, cycle_view));
    }
}
