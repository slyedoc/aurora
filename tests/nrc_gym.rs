//! The neural radiance cache's gym (docs/plans/nrc.html, phase 1): the training kernels against a
//! CPU reference, then a fit of a synthetic radiance field. Windowless; skipped on a device
//! without cooperative vectors.

use ash::vk;
use bevy_aurora::{
    compute::{CompiledComputeModule, ComputeModule},
    nrc::{NRC_CHUNK, NRC_LAYERS, NRC_PARAMS, NRC_RECORD_BYTES, NRC_WIDTH, NrcNetwork},
    render_buffer::{Buffer, BufferProvider},
    render_device::{RenderDevice, cooperative_vector},
    shader::{Shader, compile_slang},
    vulkan_asset::VulkanAsset,
};
use half::f16;

const N: usize = NRC_WIDTH as usize;
const L: usize = NRC_LAYERS as usize;

fn module(rd: &RenderDevice) -> CompiledComputeModule {
    let path = format!(
        "{}/shaders/nrc_kernels.slang",
        bevy_aurora::assets::AURORA_ASSET_DIR
    );
    let spirv = compile_slang(&path).unwrap_or_else(|e| panic!("{e}"));
    let shader = Shader {
        path,
        spirv: Some(spirv.into()),
    };
    let entries = [
        "nrc_train_forward",
        "nrc_train_grads",
        "nrc_adam",
        "nrc_eval",
        "nrc_gym_records",
    ]
    .map(String::from)
    .to_vec();
    <ComputeModule as VulkanAsset>::prepare_asset((shader, entries, Vec::new()), rd)
}

fn read(rd: &RenderDevice, buffer: &Buffer<u8>, bytes: u64) -> Vec<u8> {
    let mut host: Buffer<u8> = rd.create_host_buffer(bytes, vk::BufferUsageFlags::TRANSFER_DST);
    rd.run_transfer_commands(|cmd| unsafe {
        rd.device.cmd_copy_buffer(
            cmd,
            buffer.handle,
            host.handle,
            &[vk::BufferCopy::default().size(bytes)],
        );
    });
    let out = rd.map_buffer(&mut host).as_slice_mut()[..bytes as usize].to_vec();
    rd.destroyer.destroy_buffer(host.handle);
    out
}

fn halves(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, u16>(bytes)
        .iter()
        .map(|&h| f16::from_bits(h).to_f32())
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
}

fn rgb9e5(p: u32) -> [f32; 3] {
    let e = 2f32.powi((p >> 27) as i32 - 24);
    [
        (p & 511) as f32 * e,
        ((p >> 9) & 511) as f32 * e,
        ((p >> 18) & 511) as f32 * e,
    ]
}

/// The record row `row` of a step trains on (nrc_kernels.slang `nrc_train_forward`).
fn pick(row: u32, step: u32, count: u32) -> usize {
    let mut h = row ^ step.wrapping_mul(0x9e3779b9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    (h % count) as usize
}

fn luma(c: [f64; 3]) -> f64 {
    c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114
}

/// What a record's target asks of the network (nrc_kernels.slang `nrc_train_forward`): the
/// radiance over the albedo factor, scaled.
fn target(record: &[u8], radiance_scale: f32) -> [f64; 3] {
    let word = |i: usize| u32::from_le_bytes(record[i..i + 4].try_into().unwrap());
    let rd = word(20);
    let spec = word(24);
    let t = rgb9e5(word(28));
    std::array::from_fn(|c| {
        let diffuse = ((rd >> (8 * (c + 1))) & 255) as f32 / 255.0;
        let specular = ((spec >> (8 * c)) & 255) as f32 / 255.0;
        (t[c] * 256.0 * radiance_scale / (diffuse + specular).max(0.05)) as f64
    })
}

#[test]
fn nrc_gym() {
    let rd = unsafe { RenderDevice::from_display(None, None) };
    if !cooperative_vector() {
        eprintln!("no VK_NV_cooperative_vector: skipped");
        return;
    }
    let module = module(&rd);
    let batch = 4096u32;
    let mut net = NrcNetwork::new(&rd, batch, batch, 1);
    let out: Buffer<u8> = rd.create_device_buffer(
        batch as u64 * 12,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
    );

    // 1. The forward pass and the gradients of one batch, against the CPU.
    rd.run_transfer_commands(|cmd| {
        net.record_gym_records(&rd, cmd, &module, 1);
        net.record_gradients(&rd, cmd, &module, 0);
        net.record_eval(&rd, cmd, &module, 0, out.address);
    });
    let rows = batch as usize;
    let acts = halves(&read(&rd, &net.acts, (L * rows * N * 2) as u64));
    let dz = halves(&read(&rd, &net.dz, (L * rows * N * 2) as u64));
    let weights = halves(&read(&rd, &net.weights, (L * N * N * 2) as u64));
    let biases = halves(&read(&rd, &net.biases, (L * N * 2) as u64));
    let chunks = batch.div_ceil(NRC_CHUNK) as usize;
    let partials = floats(&read(&rd, &net.partials, chunks as u64 * NRC_PARAMS * 4));
    let gpu_out = floats(&read(&rd, &out, rows as u64 * 12));
    let records = read(&rd, &net.records, rows as u64 * NRC_RECORD_BYTES);
    let gpu = net.gpu();

    let at = |l: usize, r: usize| (l * rows + r) * N;
    let mut forward_err = 0.0f64;
    let mut act_err = 0.0f64;
    let mut grad = vec![0.0f64; NRC_PARAMS as usize];
    let mut dz_err = 0.0f64;
    let mut dz_scale = 0.0f64;
    let mut loss_grad_err = 0.0f64;
    for r in 0..rows {
        // Forward from the GPU's encoding, rounding to f16 between layers as the tensor cores do.
        let mut x: Vec<f64> = acts[at(0, r)..at(0, r) + N]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mut z = vec![0.0f64; N];
        for l in 0..L {
            if l > 0 {
                for i in 0..N {
                    act_err =
                        act_err.max((x[i] - acts[at(l, r) + i] as f64).abs() / (x[i].abs() + 1.0));
                }
            }
            for o in 0..N {
                let mut s = biases[l * N + o] as f64;
                for i in 0..N {
                    s += weights[l * N * N + o * N + i] as f64 * x[i];
                }
                z[o] = f16::from_f64(s).to_f64();
            }
            x = z.iter().map(|&v| v.max(0.0)).collect();
        }
        let rec_index = pick(r as u32, 0, batch);
        for c in 0..3 {
            forward_err = forward_err.max((z[c] - gpu_out[rec_index * 3 + c] as f64).abs());
        }
        // The loss gradient (relative L2, the prediction's luminance held constant).
        let rec = &records[rec_index * NRC_RECORD_BYTES as usize..];
        let t = target(rec, gpu.radiance_scale);
        // From the GPU's own prediction: 2 (p - t) / (luma(p)^2 + 0.01) magnifies f16 noise.
        let p: [f64; 3] = std::array::from_fn(|c| gpu_out[rec_index * 3 + c] as f64);
        let lp = luma(p.map(|v| v.max(0.0)));
        let den = lp * lp + 0.01;
        let mut d = vec![0.0f64; N];
        for c in 0..3 {
            d[c] = (2.0 * (p[c] - t[c]) / den * gpu.loss_scale as f64).clamp(-6.0e4, 6.0e4);
        }
        // Backward from the GPU's output gradient (its f16 rounding is the GPU's own).
        for c in 0..3 {
            let g = dz[at(L - 1, r) + c] as f64;
            loss_grad_err = loss_grad_err.max((g - d[c]).abs() / (d[c].abs() + 1.0));
        }
        let mut d: Vec<f64> = dz[at(L - 1, r)..at(L - 1, r) + N]
            .iter()
            .map(|&v| v as f64)
            .collect();
        for l in (0..L).rev() {
            for o in 0..N {
                grad[L * N * N + l * N + o] += d[o];
                for i in 0..N {
                    grad[l * N * N + o * N + i] += d[o] * acts[at(l, r) + i] as f64;
                }
            }
            if l == 0 {
                break;
            }
            // Each sum's error against the size of its terms (f16 products and sums).
            let mut prev = vec![0.0f64; N];
            let mut size = vec![1.0f64; N];
            for i in 0..N {
                if acts[at(l, r) + i] > 0.0 {
                    let mut s = 0.0;
                    for o in 0..N {
                        let term = weights[l * N * N + o * N + i] as f64 * d[o];
                        s += term;
                        size[i] += term.abs();
                    }
                    prev[i] = s;
                }
            }
            for i in 0..N {
                let g = dz[at(l - 1, r) + i] as f64;
                dz_err = dz_err.max((g - prev[i]).abs() / size[i]);
                dz_scale = dz_scale.max(prev[i].abs());
            }
            d = dz[at(l - 1, r)..at(l - 1, r) + N]
                .iter()
                .map(|&v| v as f64)
                .collect();
        }
    }
    let mut grad_err = 0.0f64;
    let grad_scale = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    for p in 0..NRC_PARAMS as usize {
        let gpu_grad: f64 = (0..chunks)
            .map(|c| partials[c * NRC_PARAMS as usize + p] as f64)
            .sum();
        grad_err = grad_err.max((gpu_grad - grad[p]).abs() / grad_scale);
    }
    eprintln!(
        "nrc gym: forward max error {forward_err:.2e}, activations {act_err:.2e}, \
         loss gradient {loss_grad_err:.2e}, \
         backward {dz_err:.2e} (largest {dz_scale:.1}), weight gradients {grad_err:.2e} of the \
         largest ({grad_scale:.1})"
    );
    assert!(forward_err < 5.0e-3, "forward pass");
    assert!(act_err < 5.0e-3, "activations");
    assert!(loss_grad_err < 1.0e-2, "loss gradient");
    assert!(dz_err < 1.0e-2, "backward pass");
    assert!(grad_err < 1.0e-4, "weight gradients");

    // 2. Training on fresh records each step: the relative L2 loss falls.
    let mean_loss = |net: &NrcNetwork| {
        let loss = floats(&read(&rd, &net.loss, batch as u64 * 4));
        loss.iter().map(|&l| l as f64).sum::<f64>() / loss.len() as f64
    };
    let first = mean_loss(&net);
    let steps = 5000u32;
    let start = std::time::Instant::now();
    let mut losses = Vec::new();
    for block in 0..steps / 100 {
        rd.run_transfer_commands(|cmd| {
            for s in 0..100 {
                net.record_gym_records(&rd, cmd, &module, 2 + block * 100 + s);
                net.record_step(&rd, cmd, &module, 0);
            }
        });
        if block % 10 == 9 {
            losses.push(mean_loss(&net));
        }
    }
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "nrc gym: loss {first:.3} -> {:?} over {steps} steps of {batch} ({:.0} steps/s)",
        losses.iter().map(|l| format!("{l:.4}")).collect::<Vec<_>>(),
        steps as f64 / secs
    );
    let last = *losses.last().unwrap();
    assert!(
        last.is_finite() && last < 0.05 * first,
        "training converges"
    );
    assert!(last < 0.1, "training fits the field");

    net.destroy(&rd);
    rd.destroyer.destroy_buffer(out.handle);
}
