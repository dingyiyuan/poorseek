//! PCIe *read* bandwidth probe: how fast the GPU can pull bytes that live in **system RAM**.
//!
//! This is the number the "direct expert read" design depends on: Strata keeps one pinned host
//! copy of every non-VRAM expert and hands the GPU a device alias into it, so a cold expert is a
//! PCIe read, not an upload. Whether that is worth building here turns on one measurement:
//! can the B70 read a `HostWeights` (GTT / system RAM) buffer at a useful rate, and how does it
//! compare with (a) uploading through a ReBAR staging buffer and (b) a plain VRAM->VRAM copy?
//!
//! `bandwidth_probe`'s `pcie` column is NOT this number: its staging buffer is device-local
//! host-visible (ReBAR), so that copy is VRAM->VRAM and reports 35-96 GiB/s. This probe forces the
//! source into system RAM (`BufferUsage::HostWeights`, which the backend guarantees is GTT) and
//! times `vkCmdCopyBuffer` from it, i.e. the GPU pulling host memory over the link.
//!
//! Run: cargo test -p infr-vulkan --release --test pcie_direct_read -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;
use std::time::Instant;

const MIB: usize = 1024 * 1024;

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    (bytes as f64 / secs) / (1u64 << 30) as f64
}

fn bench(iters: usize, bytes: usize, mut f: impl FnMut()) -> f64 {
    f(); // warmup
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    gib_per_sec(bytes * iters, t0.elapsed().as_secs_f64())
}

#[test]
#[ignore = "requires a Vulkan GPU; perf probe"]
fn pcie_direct_read() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // One expert = gate 3.69 + up 3.69 + down 4.83 MiB at DSV41 geometry.
    const EXPERT: usize = 12_800_000;
    const BLOCK: usize = 256 * MIB;

    let max = BLOCK.max(EXPERT);
    let host = be
        .alloc_uninit(max, BufferUsage::HostWeights)
        .expect("alloc GTT host weights");
    let staging = be
        .alloc_uninit(max, BufferUsage::Staging)
        .expect("alloc ReBAR staging");
    let dev = be
        .alloc_uninit(max, BufferUsage::Weights)
        .expect("alloc device weights");

    let data = vec![0x5au8; max];
    be.upload(host.as_ref(), &data).expect("fill host weights");
    be.upload(staging.as_ref(), &data).expect("fill staging");
    be.upload(dev.as_ref(), &data).expect("fill device");

    println!(
        "\n{:<28} {:>8} {:>10} {:>14}",
        "path", "MiB", "iters", "GiB/s"
    );

    for (label, block, iters) in [
        ("GTT host -> VRAM (PCIE READ)", 4 * MIB, 64),
        ("GTT host -> VRAM (PCIE READ)", 16 * MIB, 32),
        ("GTT host -> VRAM (PCIE READ)", EXPERT, 40),
        ("GTT host -> VRAM (PCIE READ)", BLOCK, 20),
        ("GTT host -> VRAM (PCIE READ)", 128 * MIB, 8),
        ("GTT host -> VRAM (PCIE READ)", 256 * MIB, 4),
        ("ReBAR staging -> VRAM", EXPERT, 40),
        ("ReBAR staging -> VRAM", BLOCK, 20),
        ("VRAM -> VRAM", BLOCK, 20),
    ] {
        let (src, dst): (&dyn infr_core::backend::Buffer, &dyn infr_core::backend::Buffer) =
            match label {
                l if l.starts_with("GTT") => (host.as_ref(), dev.as_ref()),
                l if l.starts_with("ReBAR") => (staging.as_ref(), dev.as_ref()),
                _ => (dev.as_ref(), staging.as_ref()),
            };
        let g = bench(iters, block, || {
            be.copy_buffer(src, dst, block).expect("copy");
        });
        println!(
            "{:<28} {:>8} {:>10} {:>14.2}",
            label,
            block / MIB,
            iters,
            g
        );
    }

    // The copy path a real CPU->GPU upload pays on top: host memcpy into the mapped staging.
    let g = bench(20, BLOCK, || {
        be.upload(staging.as_ref(), &data).expect("memcpy into staging");
    });
    println!("{:<28} {:>8} {:>10} {:>14.2}", "host memcpy -> ReBAR", BLOCK / MIB, 20, g);
}
