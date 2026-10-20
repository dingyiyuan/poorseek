//! Does packing N expert slices and uploading them in ONE transfer beat N per-expert transfers?
//! Also prices the CPU-side packing copy, which is the "expert packaging" cost.
//!
//! Run with:
//!   cargo run --release -p infr-vulkan --example expert_pack
//!
//! Environment:
//!   INFR_PACK_EXPERT_BYTES  bytes per expert (default 12_761_088 ≈ 12.2 MiB, DSV41 Q2_K)
//!   INFR_PACK_EXPERTS       expert count (default 240 = 40 layers × top-6)
//!   INFR_PACK_SMALL_ROUNDS  repeats of the small-upload phase (default 1)

use std::time::Instant;

use infr_core::backend::BufferUsage;
use infr_core::Backend;
use infr_vulkan::VulkanBackend;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn main() -> infr_core::Result<()> {
    let expert_bytes = env_usize("INFR_PACK_EXPERT_BYTES", 12_761_088);
    let n = env_usize("INFR_PACK_EXPERTS", 240);
    let rounds = env_usize("INFR_PACK_SMALL_ROUNDS", 1);
    let total = expert_bytes * n;
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;

    let backend = VulkanBackend::new()?;
    let caps = backend.capabilities();
    println!(
        "=== {} ===\n{n} experts x {:.2} MiB = {:.2} GiB",
        caps.name,
        expert_bytes as f64 / (1u64 << 20) as f64,
        gib(total)
    );

    // Stand-in for the expert payload (the host store's bytes).
    let src = vec![0x5au8; total];

    // ── 1) CPU packaging: gather N slices into one contiguous staging buffer ──────────────
    let mut packed = vec![0u8; total];
    let t = Instant::now();
    for i in 0..n {
        let o = i * expert_bytes;
        packed[o..o + expert_bytes].copy_from_slice(&src[o..o + expert_bytes]);
    }
    let pack_s = t.elapsed().as_secs_f64();
    println!(
        "CPU pack {n} slices: {:.1} ms ({:.2} GB/s)",
        pack_s * 1e3,
        total as f64 / 1e9 / pack_s
    );

    // ── 2) ONE transfer of everything ────────────────────────────────────────────────────
    let big = backend.alloc_uninit(total, BufferUsage::Weights)?;
    let t = Instant::now();
    backend.upload(big.as_ref(), &packed)?;
    let big_s = t.elapsed().as_secs_f64();
    println!(
        "one upload {:.2} GiB: {:.1} ms ({:.2} GB/s)",
        gib(total),
        big_s * 1e3,
        total as f64 / 1e9 / big_s
    );

    // ── 2b) ONE transfer into a HOST-VISIBLE buffer (ReBAR: lands in VRAM, plain memcpy) ──
    // `BufferUsage::Staging` is host-visible + mapped, and on a ReBAR device it lands in
    // device-local VRAM, so this measures "write the payload straight into the GPU's memory"
    // instead of staging it through a bounce buffer + copy engine.
    let rebar = backend.alloc_uninit(total, BufferUsage::Staging)?;
    let t = Instant::now();
    backend.upload(rebar.as_ref(), &packed)?;
    let rebar_s = t.elapsed().as_secs_f64();
    println!(
        "one upload into host-visible (Staging) {:.2} GiB: {:.1} ms ({:.2} GB/s)",
        gib(total),
        rebar_s * 1e3,
        total as f64 / 1e9 / rebar_s
    );

    // ── 3) N per-expert transfers (what a miss-per-expert path pays) ─────────────────────
    let mut bufs = Vec::with_capacity(n);
    for _ in 0..n {
        bufs.push(backend.alloc_uninit(expert_bytes, BufferUsage::Weights)?);
    }
    let mut small_best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        for (i, buf) in bufs.iter().enumerate() {
            let o = i * expert_bytes;
            backend.upload(buf.as_ref(), &src[o..o + expert_bytes])?;
        }
        let s = t.elapsed().as_secs_f64();
        if s < small_best {
            small_best = s;
        }
    }
    println!(
        "{n} small uploads: {:.1} ms ({:.3} ms each, {:.2} GB/s)",
        small_best * 1e3,
        small_best * 1e3 / n as f64,
        total as f64 / 1e9 / small_best
    );

    // ── 4) Chunk-size sweep: same total bytes, different transfer granularity ───────────
    // Tells whether the copy engine is chunk-bound (a bigger chunk should approach the link
    // speed) or already saturated at the 12.2 MiB expert granularity.
    for chunk_mib in [12usize, 32, 64, 128, 256] {
        let chunk = chunk_mib << 20;
        if chunk > total {
            continue;
        }
        let count = total / chunk;
        let mut bufs = Vec::with_capacity(count);
        for _ in 0..count {
            bufs.push(backend.alloc_uninit(chunk, BufferUsage::Weights)?);
        }
        let t = Instant::now();
        for (i, b) in bufs.iter().enumerate() {
            backend.upload(b.as_ref(), &packed[i * chunk..(i + 1) * chunk])?;
        }
        let secs = t.elapsed().as_secs_f64();
        let moved = count * chunk;
        println!(
            "{chunk_mib:>4} MiB x {count:>4} = {:.2} GiB: {:.1} ms ({:.2} GB/s)",
            gib(moved),
            secs * 1e3,
            moved as f64 / 1e9 / secs
        );
    }

    println!(
        "\npack+one-transfer = {:.1} ms | pack+host-visible = {:.1} ms | {n} small = {:.1} ms | small/big={:.1}x small/host-visible={:.1}x",
        (pack_s + big_s) * 1e3,
        (pack_s + rebar_s) * 1e3,
        small_best * 1e3,
        small_best / (pack_s + big_s),
        small_best / (pack_s + rebar_s)
    );
    Ok(())
}
