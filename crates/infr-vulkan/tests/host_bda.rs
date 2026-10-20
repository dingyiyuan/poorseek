//! Verifies the GPU-addressable system-RAM pool primitive (`alloc_host_bda`): the buffer must
//! live in NON-device-local memory, carry a real 64-bit device address, and be readable by the
//! GPU. This is the foundation of the "one host copy, GPU reads it over PCIe" expert tier.
//!
//! Run: cargo test -p infr-vulkan --release --test host_bda -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

const MIB: usize = 1024 * 1024;

#[test]
#[ignore = "requires a Vulkan GPU; mechanism verification"]
fn host_bda_pool_readable() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    // One DSV41 expert: gate 3.69 + up 3.69 + down 4.83 MiB.
    let bytes = 12_800_000usize;
    let (pool, addr) = match be.alloc_host_bda(bytes, "expert-pool") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("skip: {e}");
            return;
        }
    };
    assert_ne!(addr, 0, "a GPU-addressable pool must carry a non-zero device address");

    // Fill with a pattern the GPU can be checked against.
    let pattern: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
    be.upload(pool.as_ref(), &pattern).expect("upload into host pool");

    // The mirror buffer lives in VRAM; a copy out of the host pool is the GPU pulling the bytes
    // over PCIe through the buffer's device address / DMA path.
    let dev = be
        .alloc_uninit(bytes, BufferUsage::Weights)
        .expect("alloc vram mirror");
    be.copy_buffer(pool.as_ref(), dev.as_ref(), bytes)
        .expect("GPU read from host pool");

    let mut back = vec![0u8; bytes];
    be.download(dev.as_ref(), &mut back)
        .expect("readback mirror");
    assert_eq!(back, pattern, "GPU-visible host pool bytes must round-trip unchanged");

    // Bandwidth sanity: repeated GPU reads of the host pool.
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        be.copy_buffer(pool.as_ref(), dev.as_ref(), bytes).unwrap();
    }
    let secs = t0.elapsed().as_secs_f64();
    let gib = (bytes as f64 * 20.0) / (1u64 << 30) as f64;
    println!(
        "host pool device_addr={addr:#x} bytes={:.2} MiB GPU-read {:.2} GiB/s",
        bytes as f64 / MIB as f64,
        gib / secs
    );
}
