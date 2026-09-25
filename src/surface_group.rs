//! Surface groups: per-material closest-hit shaders.
//!
//! A ray tracer shades at the hit, so a material that needs its own shading model needs its
//! own closest-hit shader -- not more fields on [`crate::material::AuroraMaterial`]. Layered
//! terrain blending, animated water and foliage translucency are shading models, and each
//! becomes a *class* here.
//!
//! The mechanism is the shader binding table. Every hit record already carries its own group
//! handle (`SBTRegionHitTriangle::handle`), so selecting a shader per material is a matter of
//! writing a different handle into the record -- no pipeline switch, no second trace.
//!
//! Class 0 is the built-in opaque triangle group and is always present. Registered classes
//! count up from 1, in registration order, and a class that never registered falls back to 0
//! rather than corrupting the table.
//!
//! ```ignore
//! let class = app.world_mut().resource_mut::<SurfaceGroupRegistry>().register(SurfaceGroup {
//!     label: "water",
//!     closest_hit: asset_server.load("shaders/water.rchit"),
//!     any_hit: None,
//! });
//! ```
//!
//! Registration has to happen before the pipeline is prepared; the registry's
//! [`generation`](SurfaceGroupRegistry::generation) bumps on every `register`, and the
//! pipeline rebuilds when it changes, so a late registration costs a rebuild rather than
//! being silently ignored.

use bevy::prelude::*;

use crate::shader::Shader;

/// Which hit group a material's SBT record routes to: an index into
/// [`SurfaceGroupRegistry`], where 0 is the built-in opaque triangle group.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug, Default, Hash, Reflect)]
#[reflect(Component, Default)]
pub struct SurfaceClass(pub u32);

impl SurfaceClass {
    /// The built-in opaque triangle group -- `closest_hit.rchit` plus the alpha-mask
    /// any-hit. Every material that does not name a class shades with this.
    pub const OPAQUE: Self = Self(0);
}

/// One registered class: the shaders its hit group is built from.
#[derive(Clone, Debug)]
pub struct SurfaceGroup {
    /// Shown in logs when the group is compiled into the pipeline.
    pub label: String,
    pub closest_hit: Handle<Shader>,
    /// Alpha cutout and similar early-outs. `None` reuses no any-hit at all, which is
    /// faster -- only ask for one if the surface actually rejects hits.
    pub any_hit: Option<Handle<Shader>>,
}

/// The registered surface groups, in class order.
#[derive(Resource, Default)]
pub struct SurfaceGroupRegistry {
    groups: Vec<SurfaceGroup>,
    generation: u64,
}

impl SurfaceGroupRegistry {
    /// Appends a group and returns its class. Classes are stable for the run: the registry
    /// only ever grows, so a class handed out stays valid even if another registers later.
    pub fn register(&mut self, group: SurfaceGroup) -> SurfaceClass {
        self.groups.push(group);
        self.generation += 1;
        // +1: class 0 is the built-in group, which is not in `groups`.
        SurfaceClass(self.groups.len() as u32)
    }

    pub fn groups(&self) -> &[SurfaceGroup] {
        &self.groups
    }

    /// The highest valid class this frame. A record asking for more than this clamps to
    /// [`SurfaceClass::OPAQUE`].
    pub fn max_class(&self) -> u32 {
        self.groups.len() as u32
    }

    /// Bumped by every [`register`](Self::register); the pipeline rebuilds when it changes.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Device addresses of each class's own per-material parameter buffer.
///
/// The built-in [`Material`](crate::material::AuroraMaterial) array reaches a closest-hit
/// shader through push constants, indexed by `gl_InstanceCustomIndexEXT +
/// gl_GeometryIndexEXT`. A surface group's parameters are a DIFFERENT shape per class --
/// layered terrain carries two full PBR sets, water carries wave and foam terms -- so they
/// cannot share that array. Each class publishes its own buffer here and the SBT copies the
/// address into every record of that class, where the shader reads it as a buffer reference
/// and indexes it the same way.
///
/// A class with no address gets 0, which its shader must treat as "no parameters" rather
/// than dereference.
#[derive(Resource, Default)]
pub struct SurfaceGroupData {
    addresses: Vec<u64>,
}

impl SurfaceGroupData {
    /// Publishes `class`'s parameter buffer. Call whenever the buffer is (re)allocated: a
    /// stale address outlives its buffer and the shader reads freed memory.
    pub fn set(&mut self, class: SurfaceClass, address: u64) {
        let index = class.0 as usize;
        if self.addresses.len() <= index {
            self.addresses.resize(index + 1, 0);
        }
        self.addresses[index] = address;
    }

    /// `class`'s parameter buffer, or 0 when it has published none.
    pub fn get(&self, class: SurfaceClass) -> u64 {
        self.addresses.get(class.0 as usize).copied().unwrap_or(0)
    }
}

pub struct SurfaceGroupPlugin;

impl Plugin for SurfaceGroupPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SurfaceGroupRegistry>()
            .init_resource::<SurfaceGroupData>()
            .register_type::<SurfaceClass>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(label: &str) -> SurfaceGroup {
        SurfaceGroup {
            label: label.to_string(),
            closest_hit: Handle::default(),
            any_hit: None,
        }
    }

    #[test]
    fn classes_start_at_one_because_zero_is_the_builtin_group() {
        let mut registry = SurfaceGroupRegistry::default();
        assert_eq!(registry.max_class(), 0);
        assert_eq!(registry.register(group("water")), SurfaceClass(1));
        assert_eq!(registry.register(group("foliage")), SurfaceClass(2));
        assert_eq!(registry.max_class(), 2);
    }

    #[test]
    fn a_class_keeps_its_index_when_another_registers_after_it() {
        // Classes are baked into SBT records, so a later registration must not renumber
        // one already handed out.
        let mut registry = SurfaceGroupRegistry::default();
        let water = registry.register(group("water"));
        registry.register(group("foliage"));
        assert_eq!(water, SurfaceClass(1));
        assert_eq!(registry.groups()[water.0 as usize - 1].label, "water");
    }

    #[test]
    fn unpublished_classes_read_zero_rather_than_panicking() {
        let mut data = SurfaceGroupData::default();
        assert_eq!(data.get(SurfaceClass(3)), 0);
        data.set(SurfaceClass(3), 0xdead_beef);
        assert_eq!(data.get(SurfaceClass(3)), 0xdead_beef);
        // The classes it skipped over stay 0, not garbage.
        assert_eq!(data.get(SurfaceClass(1)), 0);
        assert_eq!(data.get(SurfaceClass(9)), 0);
    }

    #[test]
    fn registering_bumps_the_generation_so_the_pipeline_rebuilds() {
        let mut registry = SurfaceGroupRegistry::default();
        let before = registry.generation();
        registry.register(group("water"));
        assert_ne!(registry.generation(), before);
    }
}
