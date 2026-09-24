//! `.animclip` — a serialized `AnimationClip` of TRS keyframes, bound by node name-path.
//!
//! `AnimationClip` is a bag of type-erased curves and does not round-trip through reflection,
//! so it cannot live in a `.bsn`. An offline baker (aurora_files `animlib_import`, `zeroday`)
//! writes the clip beside the scene; this loader rebuilds it. Targets are keyed by
//! `AnimationTargetId::from_names(path)` over the node's `Name` chain, the same convention
//! `bevy_gltf` uses, so any hierarchy whose bones carry `(AnimationTargetId, AnimatedBy)`
//! from that chain plays it directly. Skinning reads the resulting bone transforms
//! (`skinning.rs`), so a skeletal clip is just a rigid clip on bone nodes.
//!
//! Format (little-endian) — writers must match byte-for-byte:
//! ```text
//!   magic  "ANIMCLP\x01"                     (8 bytes)
//!   u32    target_count
//!   per target:
//!     u16  path_len;  per component: u16 len + len UTF-8 bytes (Name chain, top-level..node)
//!     u8   channel_mask                       (bit0 = translation, bit1 = rotation, bit2 = scale)
//!     per present channel, in T,R,S order:
//!       u32  key_count
//!       key_count × f32        times (seconds)
//!       key_count × dim × f32  values         (dim: T=3, R=4 [x,y,z,w], S=3)
//! ```
//! Every sampler is LINEAR.

use bevy::{
    animation::{
        animated_field, animation_curves::AnimatableCurve, AnimationTargetId, VariableCurve,
    },
    asset::{io::Reader, AssetLoader, LoadContext},
    math::curve::{ConstantCurve, Interval, UnevenSampleAutoCurve},
    prelude::*,
    reflect::TypePath,
};

pub struct AnimClipPlugin;

impl Plugin for AnimClipPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<AnimationClip>()
            .register_asset_loader(AnimationClipLoader);
    }
}

/// Loads a `.animclip` into an [`AnimationClip`].
#[derive(Default, TypePath)]
pub struct AnimationClipLoader;

const MAGIC: &[u8; 8] = b"ANIMCLP\x01";
const CH_T: u8 = 1;
const CH_R: u8 = 2;
const CH_S: u8 = 4;

#[derive(Debug, thiserror::Error)]
pub enum AnimationClipLoaderError {
    #[error("could not read .animclip: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad .animclip magic")]
    BadMagic,
    #[error("truncated .animclip")]
    Truncated,
    #[error("invalid UTF-8 in .animclip name path")]
    BadUtf8,
}

impl AssetLoader for AnimationClipLoader {
    type Asset = AnimationClip;
    type Settings = ();
    type Error = AnimationClipLoaderError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &(),
        _ctx: &mut LoadContext<'_>,
    ) -> Result<AnimationClip, AnimationClipLoaderError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        parse_animclip(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["animclip"]
    }
}

/// A tiny forward byte cursor for the fixed layout.
struct Cur<'a>(&'a [u8]);

impl Cur<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], AnimationClipLoaderError> {
        if self.0.len() < n {
            return Err(AnimationClipLoaderError::Truncated);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, AnimationClipLoaderError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<usize, AnimationClipLoaderError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as usize)
    }
    fn u32(&mut self) -> Result<usize, AnimationClipLoaderError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, AnimationClipLoaderError> {
        let raw = self.take(n * 4)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// Parse `.animclip` bytes into an [`AnimationClip`]. Exposed for offline validation.
pub fn parse_animclip(bytes: &[u8]) -> Result<AnimationClip, AnimationClipLoaderError> {
    let mut c = Cur(bytes);
    if c.take(8)? != MAGIC {
        return Err(AnimationClipLoaderError::BadMagic);
    }
    let mut clip = AnimationClip::default();
    let target_count = c.u32()?;
    for _ in 0..target_count {
        let parts = c.u16()?;
        let mut path: Vec<Name> = Vec::with_capacity(parts);
        for _ in 0..parts {
            let len = c.u16()?;
            let s = core::str::from_utf8(c.take(len)?)
                .map_err(|_| AnimationClipLoaderError::BadUtf8)?;
            path.push(Name::new(s.to_string()));
        }
        let target = AnimationTargetId::from_names(path.iter());

        let mask = c.u8()?;
        if mask & CH_T != 0 {
            let (times, vals) = read_channel(&mut c, 3)?;
            let pts: Vec<Vec3> = vals
                .chunks_exact(3)
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect();
            if let Some(vc) = translation_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
        if mask & CH_R != 0 {
            let (times, vals) = read_channel(&mut c, 4)?;
            let pts: Vec<Quat> = vals
                .chunks_exact(4)
                .map(|v| Quat::from_array([v[0], v[1], v[2], v[3]]))
                .collect();
            if let Some(vc) = rotation_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
        if mask & CH_S != 0 {
            let (times, vals) = read_channel(&mut c, 3)?;
            let pts: Vec<Vec3> = vals
                .chunks_exact(3)
                .map(|v| Vec3::new(v[0], v[1], v[2]))
                .collect();
            if let Some(vc) = scale_curve(&times, pts) {
                clip.add_variable_curve_to_target(target, vc);
            }
        }
    }
    Ok(clip)
}

fn read_channel(c: &mut Cur, dim: usize) -> Result<(Vec<f32>, Vec<f32>), AnimationClipLoaderError> {
    let keys = c.u32()?;
    let times = c.f32s(keys)?;
    let values = c.f32s(keys * dim)?;
    Ok((times, values))
}

// One builder per field: a single keyframe collapses to a `ConstantCurve` (as bevy_gltf does);
// LINEAR uses `UnevenSampleAutoCurve`, which slerps the quaternion field.
macro_rules! trs_curve {
    ($fn:ident, $field:ident, $ty:ty) => {
        fn $fn(times: &[f32], pts: Vec<$ty>) -> Option<VariableCurve> {
            if pts.is_empty() {
                return None;
            }
            if pts.len() == 1 {
                return Some(VariableCurve::new(AnimatableCurve::new(
                    animated_field!(Transform::$field),
                    ConstantCurve::new(Interval::EVERYWHERE, pts[0]),
                )));
            }
            UnevenSampleAutoCurve::new(times.iter().copied().zip(pts))
                .ok()
                .map(|curve| {
                    VariableCurve::new(AnimatableCurve::new(
                        animated_field!(Transform::$field),
                        curve,
                    ))
                })
        }
    };
}

trs_curve!(translation_curve, translation, Vec3);
trs_curve!(rotation_curve, rotation, Quat);
trs_curve!(scale_curve, scale, Vec3);
