//! Microbenchmark: Q2_K gate/up + Q3_K down row dots at DSV41 expert geometry
//! (n_embd=5120, n_ff=2304), single thread, synthetic weights.
//!
//! Purpose: separate KERNEL cost from IO cost. The engine's cpu_miss path measured
//! ~16 ms per expert; a ggml-class AVX2 kernel should need ~1 ms for these dims.
//! Run: cargo run --release -p infr-cpu --example kq_speed

use std::time::Instant;

fn fill(n: usize, seed: &mut u32) -> Vec<u8> {
    let mut v = vec![0u8; n];
    for b in v.iter_mut() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 17;
        *seed ^= *seed << 5;
        *b = *seed as u8;
    }
    v
}

fn main() {
    let n_embd = 5120usize;
    let n_ff = 2304usize;
    let gate_stride = n_embd / 256 * 84; // Q2_K: 84 bytes / 256 values
    let down_stride = n_ff / 256 * 110; // Q3_K: 110 bytes / 256 values

    let mut seed = 0x1234_5678u32;
    let mut gate = fill(n_ff * gate_stride, &mut seed);
    let mut up = fill(n_ff * gate_stride, &mut seed);
    let mut down = fill(n_embd * down_stride, &mut seed);
    // Keep the fp16 scales finite: Q2_K d/dmin at 80..84, Q3_K d at 108..110.
    for blk in gate.chunks_mut(84) {
        blk[80..84].copy_from_slice(&[0x00, 0x24, 0x00, 0x24]);
    }
    for blk in up.chunks_mut(84) {
        blk[80..84].copy_from_slice(&[0x00, 0x24, 0x00, 0x24]);
    }
    for blk in down.chunks_mut(110) {
        blk[108..110].copy_from_slice(&[0x00, 0x24]);
    }

    let x: Vec<f32> = (0..n_embd)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.001)
        .collect();
    let q8 = infr_cpu::quantize_q8(&x);
    let intermediate: Vec<f32> = (0..n_ff)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.001)
        .collect();
    let q8i = infr_cpu::quantize_q8(&intermediate);

    let gmac = (n_embd as f64 * n_ff as f64 * 3.0) / 1e9;

    let mut acc = 0.0f32;
    let t = Instant::now();
    for r in 0..n_ff {
        acc += infr_cpu::q2k_row_dot(&gate[r * gate_stride..], &q8, n_embd);
        acc += infr_cpu::q2k_row_dot(&up[r * gate_stride..], &q8, n_embd);
    }
    let gu = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for c in 0..n_embd {
        acc += infr_cpu::q3k_row_dot(&down[c * down_stride..], &q8i, n_ff);
    }
    let dn = t.elapsed().as_secs_f64();

    println!("sink={acc:.6}");
    println!(
        "gate+up  {:7.2} ms  ({:.2} GMAC/s 1T)",
        gu * 1e3,
        (n_embd as f64 * n_ff as f64 * 2.0 / 1e9) / gu
    );
    println!(
        "down     {:7.2} ms  ({:.2} GMAC/s 1T)",
        dn * 1e3,
        (n_embd as f64 * n_ff as f64 / 1e9) / dn
    );
    println!(
        "TOTAL    {:7.2} ms  ({:.2} GMAC/s 1T)  = {:.2} GMAC total",
        (gu + dn) * 1e3,
        gmac / (gu + dn),
        gmac
    );
    // Parallel section: exactly the chunking compute_kquant uses (CHUNK=32 rows,
    // one CpuTaskPool across gate/up then down), so this is the kernel's parallel cost.
    use infr_cpu::task_pool::CpuTaskPool;
    let pool = CpuTaskPool::new(8);
    let mut inter = vec![0.0f32; n_ff];
    let mut out = vec![0.0f32; n_embd];
    let t = Instant::now();
    pool.for_chunks_mut(&mut inter, 32, &|chunk, slot| {
        let start = chunk * 32;
        for (i, o) in slot.iter_mut().enumerate() {
            let r = start + i;
            let g = infr_cpu::q2k_row_dot(&gate[r * gate_stride..], &q8, n_embd);
            let u = infr_cpu::q2k_row_dot(&up[r * gate_stride..], &q8, n_embd);
            *o = g / (1.0 + (-g).exp()) * u;
        }
    });
    let par_gu = t.elapsed().as_secs_f64();
    let t = Instant::now();
    pool.for_chunks_mut(&mut out, 32, &|chunk, slot| {
        let start = chunk * 32;
        for (i, o) in slot.iter_mut().enumerate() {
            let c = start + i;
            *o = infr_cpu::q3k_row_dot(&down[c * down_stride..], &q8i, n_ff);
        }
    });
    let par_dn = t.elapsed().as_secs_f64();
    println!(
        "PARALLEL(8T) gate+up {:6.2} ms, down {:6.2} ms, TOTAL {:6.2} ms",
        par_gu * 1e3,
        par_dn * 1e3,
        (par_gu + par_dn) * 1e3
    );
    println!(
        "weights  gate/up {:.2} MiB each, down {:.2} MiB, total {:.2} MiB",
        (n_ff * gate_stride) as f64 / 1048576.0,
        (n_embd * down_stride) as f64 / 1048576.0,
        (2 * n_ff * gate_stride + n_embd * down_stride) as f64 / 1048576.0
    );
}
