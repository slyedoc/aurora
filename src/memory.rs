//! GPU memory by who asked for it. Every allocation is named after the source line that asked
//! for it (`#[track_caller]` on the buffer and texture constructors), and [`report`] sums the
//! live ones by file and by line, beside the allocator's blocks and the device's own heap usage
//! (VK_EXT_memory_budget: what the driver holds for this process, the allocator's and the
//! driver's own alike).

use std::{collections::HashMap, ffi::CStr, panic::Location};

use ash::vk::{self, TaggedStructure as _};

use crate::render_device::RenderDevice;

/// An allocation's name: its caller's `file:line`.
#[track_caller]
pub(crate) fn caller_name() -> String {
    let at = Location::caller();
    format!("{}:{}", at.file(), at.line())
}

/// Live allocations under one name.
#[derive(Clone, Debug)]
pub struct MemoryUse {
    pub name: String,
    pub bytes: u64,
    pub count: usize,
}

/// One device memory heap, as the driver sees it.
#[derive(Clone, Debug)]
pub struct HeapUse {
    pub device_local: bool,
    pub size: u64,
    /// This process's usage (VK_EXT_memory_budget), when the device reports it.
    pub usage: Option<u64>,
    /// What this process may use before the driver starts paging, when reported.
    pub budget: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct MemoryReport {
    /// Bytes in the allocator's memory blocks (sub-allocated and dedicated).
    pub capacity: u64,
    /// Bytes in live allocations.
    pub allocated: u64,
    pub blocks: usize,
    pub allocations: usize,
    /// Per source file, largest first.
    pub files: Vec<MemoryUse>,
    /// Per allocation site (`file:line`), largest first.
    pub sites: Vec<MemoryUse>,
    pub heaps: Vec<HeapUse>,
    /// The allocator's blocks by how full they are: empty space in partly used blocks is held
    /// from the device but serves no allocation.
    pub occupancy: Vec<BlockOccupancy>,
    /// For each of the blocks with the most empty space: the sites holding it open.
    pub sparse_blocks: Vec<SparseBlock>,
}

/// Blocks whose live share falls in `under - 0.25 .. under`.
#[derive(Clone, Debug)]
pub struct BlockOccupancy {
    pub under: f32,
    pub blocks: usize,
    pub capacity: u64,
    pub used: u64,
}

/// A block with much empty space, and what still lives in it.
#[derive(Clone, Debug)]
pub struct SparseBlock {
    pub size: u64,
    pub used: u64,
    pub allocations: usize,
    /// The sites with live allocations in it, largest first (up to four).
    pub sites: Vec<MemoryUse>,
}

/// The allocator's live allocations by file and by site, and the device's heaps.
pub fn report(rd: &RenderDevice) -> MemoryReport {
    let allocator = rd.allocator_state.lock().unwrap().generate_report();
    let mut sites: HashMap<&str, MemoryUse> = HashMap::new();
    let mut files: HashMap<&str, MemoryUse> = HashMap::new();
    for allocation in &allocator.allocations {
        let name = allocation.name.as_str();
        let file = name.rsplit_once(':').map_or(name, |(file, _)| file);
        for (map, key) in [(&mut sites, name), (&mut files, file)] {
            let entry = map.entry(key).or_insert_with(|| MemoryUse {
                name: key.to_string(),
                bytes: 0,
                count: 0,
            });
            entry.bytes += allocation.size;
            entry.count += 1;
        }
    }
    let largest_first = |map: HashMap<&str, MemoryUse>| {
        let mut list: Vec<MemoryUse> = map.into_values().collect();
        list.sort_by(|a, b| b.bytes.cmp(&a.bytes));
        list
    };
    let mut occupancy: Vec<BlockOccupancy> = [0.25, 0.5, 0.75, 1.0]
        .map(|under| BlockOccupancy {
            under,
            blocks: 0,
            capacity: 0,
            used: 0,
        })
        .to_vec();
    let mut sparse: Vec<SparseBlock> = Vec::new();
    for block in &allocator.blocks {
        let live = &allocator.allocations[block.allocations.clone()];
        let used: u64 = live.iter().map(|a| a.size).sum();
        let share = used as f32 / block.size.max(1) as f32;
        let index = occupancy
            .iter()
            .position(|o| share < o.under)
            .unwrap_or(occupancy.len() - 1);
        let bucket = &mut occupancy[index];
        bucket.blocks += 1;
        bucket.capacity += block.size;
        bucket.used += used;
        let mut by_site: HashMap<&str, MemoryUse> = HashMap::new();
        for a in live {
            let entry = by_site.entry(a.name.as_str()).or_insert_with(|| MemoryUse {
                name: a.name.clone(),
                bytes: 0,
                count: 0,
            });
            entry.bytes += a.size;
            entry.count += 1;
        }
        let mut sites = largest_first(by_site);
        sites.truncate(4);
        sparse.push(SparseBlock {
            size: block.size,
            used,
            allocations: live.len(),
            sites,
        });
    }
    sparse.sort_by_key(|b| std::cmp::Reverse(b.size - b.used));
    sparse.truncate(8);
    MemoryReport {
        capacity: allocator.total_capacity_bytes,
        allocated: allocator.total_allocated_bytes,
        blocks: allocator.blocks.len(),
        allocations: allocator.allocations.len(),
        files: largest_first(files),
        sites: largest_first(sites),
        heaps: heaps(rd),
        occupancy,
        sparse_blocks: sparse,
    }
}

/// The device's heaps, with usage and budget where VK_EXT_memory_budget is supported (a
/// physical-device query: supported is enough, nothing to enable).
fn heaps(rd: &RenderDevice) -> Vec<HeapUse> {
    let budget_supported = unsafe {
        rd.instance
            .enumerate_device_extension_properties(rd.physical_device)
            .unwrap_or_default()
    }
    .iter()
    .any(|p| p.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME as &CStr));
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let memory = {
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default();
        if budget_supported {
            properties = properties.push(&mut budget);
        }
        unsafe {
            rd.instance
                .get_physical_device_memory_properties2(rd.physical_device, &mut properties)
        };
        properties.memory_properties
    };
    (0..memory.memory_heap_count as usize)
        .map(|i| HeapUse {
            device_local: memory.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
            size: memory.memory_heaps[i].size,
            usage: budget_supported.then(|| budget.heap_usage[i]),
            budget: budget_supported.then(|| budget.heap_budget[i]),
        })
        .collect()
}
