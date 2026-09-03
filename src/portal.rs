//! Ray portals -- surfaces that teleport rays (the solari-era feature, ported).
//!
//! Add [`AuroraPortal`] to a ray-traced surface (a quad, an arch, anything) and every ray
//! that hits it continues from the paired portal instead: into portal-local space, a
//! half-turn about local Y, out through the target's frame -- so looking INTO this surface
//! shows the view OUT of the target's front (+Z). Pair two portals by pointing their
//! `target`s at each other; a one-way portal is just an unpaired one.
//!
//! GPU side: a small host table of (instance slot, target slot) pairs, address + count in
//! the frame uniform; the raygen checks each hit's TLAS slot against it and rewrites the
//! ray (`portalRedirect` in raygen.rgen), reading both transforms from the live
//! `cur_instances` rows -- portals on moving parents stay exact with no CPU reads. Because
//! the redirect happens in the continued ray, recursion is free: portals seen through
//! portals, portals in reflections, portals through glass, bounded by `max_bounces`.
//! Light is NOT transported -- portals carry the view, not next-event estimation, so each
//! side is lit by its own surroundings (and a portal surface still occludes shadow rays).

use ash::vk;
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

use crate::{
    ray_render_plugin::RenderSet,
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    tlas_builder::GpuInstance,
};

/// Marks a ray-traced surface as a portal showing the view out of `target`'s front face
/// (+Z). The surface never shades -- rays redirect on hit.
#[derive(Component, Reflect, Clone, Copy)]
#[reflect(Component)]
pub struct AuroraPortal {
    /// The portal entity this surface looks out of.
    pub target: Entity,
}

/// One table entry; must match the `uvec4` unpack in raygen.rgen's `portalRedirect`.
/// `valid = 0` covers unresolved endpoints (instances still streaming in) and the
/// zero-filled tail of the buffer -- slot 0 is real, so zeros must read as inert.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
struct GpuPortal {
    instance_slot: u32,
    target_slot: u32,
    valid: u32,
    pad: u32,
}

impl GpuPortal {
    const INVALID: Self = Self { instance_slot: 0, target_slot: 0, valid: 0, pad: 0 };
}

/// The uploaded pair table; the frame uniform carries its address + count.
#[derive(Resource, Default)]
pub struct PortalTable {
    buffer: Option<Buffer<GpuPortal>>,
    capacity: u64,
    count: u32,
    /// Last-uploaded entries: the upload is skipped while nothing changed.
    last: Vec<GpuPortal>,
}

impl PortalTable {
    pub fn address(&self) -> u64 {
        if self.count == 0 { 0 } else { self.buffer.as_ref().map_or(0, |b| b.address) }
    }
    pub fn count(&self) -> u32 {
        self.count
    }
}

/// `RenderSet::Prepare`: rebuild + upload the pair table when the portal set, a pairing,
/// or an endpoint's instance slot changed (slots bind late while meshes stream in, so this
/// re-resolves every frame -- the memo makes the idle cost a Vec compare).
fn upload_portals(
    render_device: Res<RenderDevice>,
    portals: Query<(Entity, &AuroraPortal)>,
    slots: Query<&GpuInstance>,
    mut table: ResMut<PortalTable>,
) {
    let mut entries: Vec<GpuPortal> = Vec::new();
    for (entity, portal) in &portals {
        entries.push(match (slots.get(entity), slots.get(portal.target)) {
            (Ok(a), Ok(b)) => GpuPortal {
                instance_slot: a.0,
                target_slot: b.0,
                valid: 1,
                pad: 0,
            },
            _ => GpuPortal::INVALID,
        });
    }
    if entries == table.last && (table.buffer.is_some() || entries.is_empty()) {
        return;
    }

    let needed = entries.len().max(1) as u64;
    if table.buffer.is_none() || table.capacity < needed {
        if let Some(old) = table.buffer.take() {
            // The destroyer defers the free past the frames in flight.
            render_device.destroyer.destroy_buffer(old.handle);
        }
        let capacity = needed.next_power_of_two().max(16);
        table.buffer =
            Some(render_device.create_host_buffer(capacity, vk::BufferUsageFlags::STORAGE_BUFFER));
        table.capacity = capacity;
    }
    let capacity = table.capacity;
    let buffer = table.buffer.as_mut().unwrap();
    let mut mapped = render_device.map_buffer(buffer);
    for i in 0..capacity as usize {
        mapped[i] = *entries.get(i).unwrap_or(&GpuPortal::INVALID);
    }
    table.count = entries.len() as u32;
    table.last = entries;
}

pub struct PortalPlugin;

impl Plugin for PortalPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<AuroraPortal>();
        app.init_resource::<PortalTable>();
        // The Prepare set already gates on the render device existing.
        app.add_systems(Last, upload_portals.in_set(RenderSet::Prepare));
    }
}
