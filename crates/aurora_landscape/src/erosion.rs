//! Erosion as a layer: a region of the stack below re-baked by hydraulic + thermal erosion on
//! the GPU (`landscape/erosion.slang`). The bake's result is a height delta held in the layer's
//! pages, so it evaluates like a sculpt layer: move or hide the stack below and the layer keeps
//! its old result until baked again. The delta saves to `file` (an `.erosion` page file) and
//! loads back from it; nothing re-simulates on load.
//!
//! A bake clears the layer's own delta, waits for the region to re-evaluate without it, then
//! simulates `iterations` steps in one submission and writes the change back.

use ash::vk;
use bevy::prelude::*;
use bevy_aurora::{
    compute::{ComputeModules, compute_to_compute_barrier, record_dispatch},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    HeightLayer, Landscape, LandscapeKernels, PAGE_TEXELS,
    page::{GpuPagePool, LandscapePages},
    sculpt::SculptPages,
};

/// Cells on a side a region may span.
const REGION_MAX: u32 = 2048;

/// A region (centred on the `Transform`, `size` metres) of everything below this layer,
/// eroded. Set `bake` to (re)bake; it clears itself.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default)]
#[require(HeightLayer, SculptPages)]
pub struct ErosionLayer {
    pub size: Vec2,
    /// Metres between simulation cells. Coarser than the texels: channels form at landform
    /// scale (and a texel-scale grid shows its four pipe directions as combed grooves).
    pub resolution: f32,
    pub iterations: u32,
    /// Seconds per step.
    pub dt: f32,
    /// Metres of water per second.
    pub rain: f32,
    /// Fraction of water lost per second.
    pub evaporation: f32,
    /// Sediment a unit of fast water on a steep slope holds.
    pub capacity: f32,
    pub erode: f32,
    pub deposit: f32,
    /// Degrees past which loose material slips.
    pub talus: f32,
    /// Fraction of the excess slipping per step.
    pub thermal: f32,
    /// Metres over which the result fades out at the region's border.
    pub edge: f32,
    /// Set to (re)bake; it clears itself.
    pub bake: bool,
    /// Where the delta is saved (an `.erosion` asset); its contents load into the pages.
    pub file: Handle<crate::sculpt::LayerPagesFile>,
}

impl Default for ErosionLayer {
    fn default() -> Self {
        Self {
            size: Vec2::splat(256.0),
            resolution: 2.0,
            iterations: 200,
            dt: 0.05,
            rain: 0.02,
            evaporation: 0.05,
            capacity: 1.0,
            erode: 0.15,
            deposit: 0.2,
            talus: 40.0,
            thermal: 0.08,
            edge: 24.0,
            bake: false,
            file: Handle::default(),
        }
    }
}

/// A bake waiting for its region to re-evaluate without the old result.
#[derive(Component)]
pub struct ErosionBake {
    region: Rect,
}

/// `Erosion` in erosion.slang.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ErosionGpu {
    b: u64,
    b0: u64,
    d: u64,
    s: u64,
    s2: u64,
    flux: u64,
    vel: u64,
    slip: u64,
    slip2: u64,
    pool: GpuPagePool,
    x0: f32,
    z0: f32,
    nx: u32,
    nz: u32,
    dt: f32,
    rain: f32,
    evaporation: f32,
    capacity: f32,
    erode: f32,
    deposit: f32,
    talus: f32,
    thermal: f32,
    edge: f32,
    cell: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ErosionPush {
    e: u64,
    page: u64,
    page_x: i32,
    page_z: i32,
    iteration: u32,
    pad: u32,
}

/// Requested bakes clear their layer's delta and dirty what it covered.
pub fn start_bakes(
    mut commands: Commands,
    mut layers: Query<
        (Entity, &mut ErosionLayer, &ChildOf, &Transform, &mut SculptPages),
        Without<ErosionBake>,
    >,
    mut landscapes: Query<(&Landscape, &mut LandscapePages)>,
) {
    for (entity, mut layer, child_of, transform, mut delta) in &mut layers {
        if !layer.bake {
            continue;
        }
        layer.bypass_change_detection().bake = false;
        let Ok((landscape, mut pages)) = landscapes.get_mut(child_of.parent()) else {
            continue;
        };
        let old: Vec<IVec2> = delta.pages().collect();
        delta.zero();
        let region = Rect::from_center_size(transform.translation.xz(), layer.size);
        pages.mark_dirty(old);
        pages.mark_dirty(landscape.pages_in(region));
        commands.entity(entity).insert(ErosionBake { region });
    }
}

/// Bakes whose region has re-evaluated: simulate and write the delta.
pub fn run_bakes(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    kernels: Res<LandscapeKernels>,
    modules: Res<ComputeModules>,
    mut layers: Query<(Entity, &ErosionLayer, &ErosionBake, &ChildOf, &mut SculptPages)>,
    mut landscapes: Query<(&Landscape, &mut LandscapePages)>,
) {
    let Some(module) = modules.get(&kernels.erosion) else {
        return;
    };
    for (entity, layer, bake, child_of, mut delta) in &mut layers {
        let Ok((landscape, mut pages)) = landscapes.get_mut(child_of.parent()) else {
            continue;
        };
        let region_pages: Vec<IVec2> = landscape.pages_in(bake.region).collect();
        if region_pages.iter().any(|p| pages.dirty.contains(p)) {
            continue;
        }
        commands.entity(entity).remove::<ErosionBake>();
        let cell = layer.resolution.max(landscape.texel_size);
        let bounds = Rect::from_corners(
            landscape.pages_min.as_vec2() * landscape.page_size(),
            landscape.pages_max.as_vec2() * landscape.page_size(),
        );
        let region = bake.region.intersect(bounds);
        let n = ((region.size() / cell).floor().as_uvec2() + 1).min(UVec2::splat(REGION_MAX));
        if region.is_empty() || n.x < 3 || n.y < 3 {
            continue;
        }
        let cells = (n.x * n.y) as u64;
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let scalars: Vec<Buffer<f32>> = (0..5)
            .map(|_| rd.create_device_buffer(cells, storage))
            .collect();
        let flux: Buffer<[f32; 4]> = rd.create_device_buffer(cells, storage);
        let vel: Buffer<[f32; 2]> = rd.create_device_buffer(cells, storage);
        let slip: Buffer<[f32; 4]> = rd.create_device_buffer(cells, storage);
        let slip2: Buffer<[f32; 4]> = rd.create_device_buffer(cells, storage);
        let mut params: Buffer<ErosionGpu> = rd.create_host_buffer(1, storage);
        rd.map_buffer(&mut params)[0] = ErosionGpu {
            b: scalars[0].address,
            b0: scalars[1].address,
            d: scalars[2].address,
            s: scalars[3].address,
            s2: scalars[4].address,
            flux: flux.address,
            vel: vel.address,
            slip: slip.address,
            slip2: slip2.address,
            pool: pages.pool(),
            x0: region.min.x,
            z0: region.min.y,
            nx: n.x,
            nz: n.y,
            dt: layer.dt,
            rain: layer.rain,
            evaporation: layer.evaporation,
            capacity: layer.capacity,
            erode: layer.erode,
            deposit: layer.deposit,
            talus: layer.talus.to_radians().tan(),
            thermal: layer.thermal,
            edge: layer.edge,
            cell,
        };
        let targets: Vec<(IVec2, u64)> = region_pages
            .iter()
            .map(|&p| (p, delta.ensure(&rd, p)))
            .collect();
        let grid = ErosionPush {
            e: params.address,
            page: 0,
            page_x: 0,
            page_z: 0,
            iteration: 0,
            pad: 0,
        };
        let count = n.x * n.y;
        let started = std::time::Instant::now();
        rd.run_transfer_commands(|cmd| {
            let step = |entry: &str, iteration: u32| {
                let push = ErosionPush { iteration, ..grid };
                record_dispatch(&rd, cmd, module, entry, &push, count, None);
                compute_to_compute_barrier(&rd, cmd);
            };
            step("init", 0);
            for iteration in 0..layer.iterations {
                for entry in [
                    "rain", "flux", "water", "erode", "advect", "evaporate", "slip_out", "slip_in",
                ] {
                    step(entry, iteration);
                }
            }
            for &(page, address) in &targets {
                let push = ErosionPush {
                    page: address,
                    page_x: page.x,
                    page_z: page.y,
                    ..grid
                };
                let len = PAGE_TEXELS * PAGE_TEXELS;
                record_dispatch(&rd, cmd, module, "write_delta", &push, len, None);
            }
        });
        log::info!(
            "landscape: eroded {}x{} cells x {} steps in {:.0} ms",
            n.x,
            n.y,
            layer.iterations,
            started.elapsed().as_secs_f64() * 1000.0
        );
        if log::log_enabled!(log::Level::Debug) {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for &(page, _) in &targets {
                for &v in delta.page(page).unwrap_or(&[]) {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            log::debug!("landscape: erosion delta {lo:.2}..{hi:.2} m");
        }
        for handle in scalars
            .iter()
            .map(|b| b.handle)
            .chain([flux.handle, vel.handle, slip.handle, slip2.handle, params.handle])
        {
            rd.destroyer.destroy_buffer(handle);
        }
        pages.mark_dirty(region_pages);
    }
}
