//! The surface BSDF (assets/shaders/bsdf.slang) checked against itself on the GPU: the sampler's
//! weights integrate to the same albedo as its evaluation (uniform sphere sampling), its pdf
//! integrates to the share of samples kept, each sample's weight is f / pdf, eval's derivatives (Slang's bwd_diff) match finite differences, and a
//! PathSampler replay repeats its sample. Windowless.

use ash::vk;
use bevy_aurora::{
    compute::{ComputeModule, record_dispatch},
    render_buffer::{Buffer, BufferProvider},
    render_device::RenderDevice,
    shader::{Shader, compile_slang},
    vulkan_asset::VulkanAsset,
};

const CONFIGS: usize = 64;
const THREADS: usize = 64;
const SAMPLES: u32 = 1 << 16;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Push {
    out: u64,
    base: u32,
    samples: u32,
}

#[test]
fn bsdf_check() {
    let rd = unsafe { RenderDevice::from_display(None, None) };
    let path = format!("{}/tests/bsdf_check.slang", env!("CARGO_MANIFEST_DIR"));
    let spirv = compile_slang(&path).unwrap_or_else(|e| panic!("{e}"));
    let shader = Shader {
        path,
        spirv: Some(spirv.into()),
    };
    let module = <ComputeModule as VulkanAsset>::prepare_asset(
        (shader, vec!["bsdf_check".to_string()], Vec::new()),
        &rd,
    );
    let floats = CONFIGS * THREADS * 5 + CONFIGS * 8;
    let out: Buffer<u8> = rd.create_device_buffer(
        floats as u64 * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    );
    let mut host: Buffer<u8> =
        rd.create_host_buffer(floats as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST);
    let push = Push {
        out: out.address,
        base: 0,
        samples: SAMPLES,
    };
    rd.run_transfer_commands(|cmd| {
        record_dispatch(
            &rd,
            cmd,
            &module,
            "bsdf_check",
            &push,
            (CONFIGS * THREADS) as u32,
            Some(std::mem::offset_of!(Push, base)),
        );
        bevy_aurora::compute::memory_barrier(
            &rd,
            cmd,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        unsafe {
            rd.device.cmd_copy_buffer(
                cmd,
                out.handle,
                host.handle,
                &[vk::BufferCopy::default().size(floats as u64 * 4)],
            );
        }
    });
    let data: Vec<f32> =
        bytemuck::cast_slice::<u8, f32>(&rd.map_buffer(&mut host).as_slice_mut()[..floats * 4])
            .to_vec();
    let mut worst_albedo = 0.0f64;
    let mut worst_pdf = 0.0f64;
    let mut worst_grad = 0.0f64;
    let mut worst_sample = 0.0f64;
    let mut failures = Vec::new();
    for c in 0..CONFIGS {
        let mean = |k: usize| {
            (0..THREADS)
                .map(|t| data[(c * THREADS + t) * 5 + k] as f64)
                .sum::<f64>()
                / THREADS as f64
        };
        let (sampled, uniform, pdf, success) = (mean(0), mean(1), mean(2), mean(3));
        let per_sample = (0..THREADS)
            .map(|t| data[(c * THREADS + t) * 5 + 4] as f64)
            .fold(0.0f64, f64::max);
        let tail = &data[CONFIGS * THREADS * 5 + c * 8..][..8];
        let (rough, metal, layer) = (tail[5], tail[6], tail[7] as u32);
        let label = format!(
            "roughness {rough} metallic {metal} {} view {}",
            ["", "translucent", "coat", "glass"][layer as usize],
            if c / 32 == 0 { "0.9" } else { "0.35" }
        );
        let albedo_err = (sampled - uniform).abs() / uniform.max(0.05);
        // Uniform sphere sampling of a sharp refraction lobe is heavy-tailed: two seeds of
        // roughness 0.3 glass differ by 3 %; the per-sample check is the exact one there.
        let tolerance = if layer == 3 && rough < 0.4 {
            0.06
        } else {
            0.02
        };
        worst_sample = worst_sample.max(per_sample);
        worst_albedo = worst_albedo.max(albedo_err);
        worst_pdf = worst_pdf.max((pdf - success).abs());
        let grad = |a: f32, b: f32| ((a - b).abs() / (b.abs() + 0.05)) as f64;
        let g = grad(tail[0], tail[1]).max(grad(tail[2], tail[3]));
        worst_grad = worst_grad.max(g);
        eprintln!(
            "{label}: albedo sampled {sampled:.4} uniform {uniform:.4}, pdf integral {pdf:.4} (samples kept {success:.4}), \
             d/droughness {:.4} (fd {:.4}), d/dalbedo.r {:.4} (fd {:.4}), replay {}",
            tail[0], tail[1], tail[2], tail[3], tail[4]
        );
        if per_sample > 1.0e-3 {
            failures.push(format!(
                "{label}: a sample's weight is not f / pdf ({per_sample:.2e})"
            ));
        }
        if albedo_err > tolerance {
            failures.push(format!(
                "{label}: sampling weights {sampled:.4} vs eval {uniform:.4}"
            ));
        }
        if (pdf - success).abs() > tolerance {
            failures.push(format!(
                "{label}: pdf integrates to {pdf:.4}, samples kept {success:.4}"
            ));
        }
        if g > 0.01 {
            failures.push(format!("{label}: derivatives off"));
        }
        if tail[4] != 1.0 {
            failures.push(format!("{label}: replay differs"));
        }
    }
    eprintln!(
        "bsdf check: worst albedo mismatch {worst_albedo:.4}, pdf {worst_pdf:.4}, gradient {worst_grad:.4}, \
         sample weight vs f / pdf {worst_sample:.2e}"
    );
    assert!(failures.is_empty(), "{failures:#?}");
}
