//! Pipelines cached on disk as driver binaries (`VK_KHR_pipeline_binary`), one file per
//! pipeline under `~/.cache/aurora/pipelines/`, named by the driver's key for its create info.
//! A launch that finds the file builds the pipeline from it instead of compiling; one that does
//! not compiles with `CAPTURE_DATA` and writes it. Files unused for [`KEEP_DAYS`] are pruned.
//!
//! The driver keeps descriptor-heap pipelines out of a `VkPipelineCache`, so this is the only
//! cache the ray-tracing pipeline has. Without the extension pipelines go through the device's
//! `VkPipelineCache` as before.

use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use ash::vk;

type VkResult<T> = Result<T, vk::Result>;

use crate::render_device::RenderDeviceData;

/// Pipeline files untouched this long are deleted at startup.
pub const KEEP_DAYS: u64 = 30;

const MAGIC: &[u8; 4] = b"APB1";

pub(crate) fn dir() -> Option<PathBuf> {
    crate::shader::cache_dir().map(|dir| dir.join("pipelines"))
}

/// Builds the pipeline `info` describes: from its binaries on disk when they are there, else by
/// `create` (with `CAPTURE_DATA` set on the chain's `PipelineCreateFlags2CreateInfo`), saving
/// them. `create` gets the info to create from and the cache to pass. `label` names the pipeline
/// beside the driver's key (which can be the same for different kernels of one module).
pub(crate) fn create_pipeline<T: Copy>(
    rd: &RenderDeviceData,
    label: &str,
    info: &T,
    create: impl Fn(&T, vk::PipelineCache) -> VkResult<vk::Pipeline>,
) -> VkResult<vk::Pipeline> {
    let Some(ext) = &rd.ext_pipeline_binary else {
        return create(info, rd.pipeline_cache);
    };
    let fp = ext.fp();
    let device = rd.device.handle();
    let key = unsafe {
        let mut key = vk::PipelineBinaryKeyKHR::default();
        let mut keyed = vk::PipelineCreateInfoKHR::default();
        keyed.p_next = (info as *const T).cast_mut().cast();
        (fp.get_pipeline_key_khr)(device, &keyed, &mut key)
            .result()
            .ok()
            .map(|()| key.key[..key.key_size as usize].to_vec())
    };
    let path = key.as_deref().zip(dir()).map(|(key, dir)| {
        let mut hasher = blake3::Hasher::new();
        hasher.update(key);
        hasher.update(label.as_bytes());
        dir.join(format!("{}.bin", hasher.finalize().to_hex()))
    });

    let mut refused = false;
    if let Some(path) = &path {
        match from_disk(rd, label, path, info, &create) {
            Disk::Built(pipeline) => return Ok(pipeline),
            Disk::Refused => refused = true,
            Disk::Missing => {}
        }
    }
    if refused {
        return create(info, rd.pipeline_cache);
    }

    let mut capture = *info;
    let captured = unsafe { set_capture(&mut capture) };
    let pipeline = create(&capture, vk::PipelineCache::null())?;
    if captured && let Some(path) = &path {
        log::debug!("pipeline binary: capturing {label}");
        if let Err(err) = save(rd, pipeline, path) {
            log::warn!("pipeline binary: save {} failed: {err}", path.display());
        }
        unsafe {
            let release = vk::ReleaseCapturedPipelineDataInfoKHR::default().pipeline(pipeline);
            let _ = (fp.release_captured_pipeline_data_khr)(device, &release, std::ptr::null());
        }
    }
    Ok(pipeline)
}

/// ORs `CAPTURE_DATA` into the chain's flags; false when the chain has none to carry it.
unsafe fn set_capture<T>(info: &mut T) -> bool {
    let mut next = unsafe { (*(info as *mut T).cast::<vk::BaseOutStructure>()).p_next };
    while !next.is_null() {
        unsafe {
            if (*next).s_type == vk::StructureType::PIPELINE_CREATE_FLAGS_2_CREATE_INFO {
                let flags = next.cast::<vk::PipelineCreateFlags2CreateInfo>();
                (*flags).flags |= vk::PipelineCreateFlags2::CAPTURE_DATA_KHR;
                return true;
            }
            next = (*next).p_next;
        }
    }
    false
}

/// What the file for a pipeline gave.
enum Disk {
    Built(vk::Pipeline),
    /// The driver will not build this pipeline from binaries at all (a [`REFUSED`] marker).
    Refused,
    /// No usable file: compile and capture.
    Missing,
}

/// Written in place of a pipeline's binaries when the driver refuses to build it from them:
/// later launches compile it directly instead of capturing it again.
const REFUSED: &[u8; 4] = b"APBX";

fn from_disk<T: Copy>(
    rd: &RenderDeviceData,
    label: &str,
    path: &Path,
    info: &T,
    create: &impl Fn(&T, vk::PipelineCache) -> VkResult<vk::Pipeline>,
) -> Disk {
    let Ok(bytes) = std::fs::read(path) else {
        return Disk::Missing;
    };
    if bytes.as_slice() == REFUSED {
        touch(path);
        return Disk::Refused;
    }
    let Some(entries) = parse(&bytes) else {
        return Disk::Missing;
    };
    let Some(fp) = rd.ext_pipeline_binary.as_ref().map(|e| e.fp()) else {
        return Disk::Missing;
    };
    let device = rd.device.handle();
    let keys: Vec<vk::PipelineBinaryKeyKHR> = entries
        .iter()
        .map(|(key, _)| {
            let mut k = vk::PipelineBinaryKeyKHR::default();
            k.key_size = key.len() as u32;
            k.key[..key.len()].copy_from_slice(key);
            k
        })
        .collect();
    let data: Vec<vk::PipelineBinaryDataKHR> = entries
        .iter()
        .map(|(_, data)| vk::PipelineBinaryDataKHR {
            data_size: data.len(),
            p_data: data.as_ptr() as *mut c_void,
            ..Default::default()
        })
        .collect();
    let keys_and_data = vk::PipelineBinaryKeysAndDataKHR::default()
        .pipeline_binary_keys(&keys)
        .pipeline_binary_data(&data);
    let create_info = vk::PipelineBinaryCreateInfoKHR::default().keys_and_data_info(&keys_and_data);
    let mut binaries = vec![vk::PipelineBinaryKHR::null(); entries.len()];
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default().pipeline_binaries(&mut binaries);
    let made = unsafe {
        (fp.create_pipeline_binaries_khr)(device, &create_info, std::ptr::null(), &mut handles)
    };
    if made != vk::Result::SUCCESS {
        log::debug!("pipeline binary: {}: create binaries {made:?}", path.display());
    }
    let pipeline = if made == vk::Result::SUCCESS {
        let mut with = *info;
        let mut binary_info = vk::PipelineBinaryInfoKHR::default().pipeline_binaries(&binaries);
        unsafe {
            let base = (&raw mut with).cast::<vk::BaseOutStructure>();
            binary_info.p_next = (*base).p_next.cast();
            (*base).p_next = (&raw mut binary_info).cast();
        }
        create(&with, vk::PipelineCache::null())
    } else {
        Err(vk::Result::PIPELINE_COMPILE_REQUIRED)
    };
    for binary in binaries {
        if binary != vk::PipelineBinaryKHR::null() {
            unsafe { (fp.destroy_pipeline_binary_khr)(device, binary, std::ptr::null()) };
        }
    }
    match pipeline {
        Ok(pipeline) => {
            touch(path);
            log::debug!("pipeline binary: hit {}", path.display());
            Disk::Built(pipeline)
        }
        // A changed shader or driver changes the key, so a file under this one that does not
        // build is one the driver will never build from binaries (some pipelines all share one
        // generic key).
        Err(err) => {
            log::info!(
                "pipeline binary: the driver will not build {label} from binaries ({err:?}); it compiles each launch"
            );
            let _ = std::fs::write(path, REFUSED);
            Disk::Refused
        }
    }
}

/// Marks a cache file used, so pruning keeps it.
pub(crate) fn touch(path: &Path) {
    if let Ok(file) = std::fs::File::options().write(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

fn save(rd: &RenderDeviceData, pipeline: vk::Pipeline, path: &Path) -> Result<(), String> {
    let fp = rd.ext_pipeline_binary.as_ref().ok_or("no extension")?.fp();
    let device = rd.device.handle();
    let create_info = vk::PipelineBinaryCreateInfoKHR::default().pipeline(pipeline);
    let mut handles = vk::PipelineBinaryHandlesInfoKHR::default();
    unsafe {
        (fp.create_pipeline_binaries_khr)(device, &create_info, std::ptr::null(), &mut handles)
            .result()
            .map_err(|e| format!("{e:?}"))?;
    }
    let mut binaries = vec![vk::PipelineBinaryKHR::null(); handles.pipeline_binary_count as usize];
    handles = handles.pipeline_binaries(&mut binaries);
    let made = unsafe {
        (fp.create_pipeline_binaries_khr)(device, &create_info, std::ptr::null(), &mut handles)
            .result()
            .map_err(|e| format!("{e:?}"))
    };
    let mut out = Vec::from(&MAGIC[..]);
    let read = made.and_then(|()| {
        out.extend((binaries.len() as u32).to_le_bytes());
        for &binary in &binaries {
            let info = vk::PipelineBinaryDataInfoKHR::default().pipeline_binary(binary);
            let mut key = vk::PipelineBinaryKeyKHR::default();
            let mut size = 0usize;
            unsafe {
                (fp.get_pipeline_binary_data_khr)(
                    device,
                    &info,
                    &mut key,
                    &mut size,
                    std::ptr::null_mut(),
                )
                .result()
                .map_err(|e| format!("{e:?}"))?;
            }
            let mut data = vec![0u8; size];
            unsafe {
                (fp.get_pipeline_binary_data_khr)(
                    device,
                    &info,
                    &mut key,
                    &mut size,
                    data.as_mut_ptr().cast(),
                )
                .result()
                .map_err(|e| format!("{e:?}"))?;
            }
            out.extend(key.key_size.to_le_bytes());
            out.extend(&key.key[..key.key_size as usize]);
            out.extend((size as u64).to_le_bytes());
            out.extend(&data[..size]);
        }
        Ok(())
    });
    for binary in binaries {
        if binary != vk::PipelineBinaryKHR::null() {
            unsafe { (fp.destroy_pipeline_binary_khr)(device, binary, std::ptr::null()) };
        }
    }
    read?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let partial = path.with_extension(format!("{}.part", std::process::id()));
    std::fs::write(&partial, &out).map_err(|e| e.to_string())?;
    std::fs::rename(&partial, path).map_err(|e| e.to_string())?;
    log::debug!("pipeline binary: saved {} ({} KiB)", path.display(), out.len() / 1024);
    Ok(())
}

/// `(key, data)` per binary.
fn parse(bytes: &[u8]) -> Option<Vec<(&[u8], &[u8])>> {
    let mut at = bytes.strip_prefix(&MAGIC[..])?;
    let mut take = |n: usize| -> Option<&[u8]> {
        let (head, rest) = at.split_at_checked(n)?;
        at = rest;
        Some(head)
    };
    let count = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let mut entries = Vec::new();
    for _ in 0..count {
        let key_size = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        if key_size > vk::MAX_PIPELINE_BINARY_KEY_SIZE_KHR {
            return None;
        }
        let key = take(key_size)?;
        let size = u64::from_le_bytes(take(8)?.try_into().ok()?) as usize;
        entries.push((key, take(size)?));
    }
    Some(entries)
}

/// Startup housekeeping in `dir`: files unused for [`KEEP_DAYS`] go, and so do partial writes
/// a killed process left (older than an hour).
pub(crate) fn prune(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    let (mut removed, mut bytes) = (0, 0u64);
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let age = meta
            .modified()
            .ok()
            .and_then(|m| now.duration_since(m).ok())
            .unwrap_or_default();
        let partial = path.to_string_lossy().ends_with(".part");
        let stale = if partial {
            age > Duration::from_secs(3600)
        } else {
            age > Duration::from_secs(KEEP_DAYS * 86_400)
        };
        if meta.is_file() && stale && std::fs::remove_file(&path).is_ok() {
            removed += 1;
            bytes += meta.len();
        }
    }
    if removed > 0 {
        log::info!(
            "cache: pruned {removed} files ({} MiB) from {}",
            bytes >> 20,
            dir.display()
        );
    }
}
