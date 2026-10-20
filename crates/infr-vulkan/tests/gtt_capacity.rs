//! How large a GPU-addressable SYSTEM-RAM (GTT / `HostWeights`) allocation can this machine make?
//!
//! The "single host copy, GPU reads it directly" design needs one host-side pool holding every
//! expert that is not in VRAM (~159 GiB for DSV41). Whether that is even allocatable on this
//! driver is the gate for the whole plan: the xe driver's GTT window has been observed near
//! ~94 GiB, and a failed allocation must be known before any code depends on it.
//!
//! Run: cargo test -p infr-vulkan --release --test gtt_capacity -- --ignored --nocapture
use infr_core::backend::{Backend, BufferUsage};
use infr_vulkan::VulkanBackend;

const GIB: usize = 1 << 30;

#[test]
#[ignore = "requires a Vulkan GPU + lots of free RAM; capacity probe"]
fn gtt_capacity() {
    let Ok(be) = VulkanBackend::new() else {
        eprintln!("skip: no Vulkan device");
        return;
    };
    println!("\n{:<12} {:>10} {:>10} {:>10}", "request", "result", "elapsed_s", "device_addr");
    for gib in [4usize, 8, 16, 32, 48, 64, 80, 96, 112, 128] {
        let bytes = gib * GIB;
        let t0 = std::time::Instant::now();
        match be.alloc(bytes, BufferUsage::HostWeights) {
            Ok(buf) => {
                let addr = buf.device_addr().is_some();
                println!(
                    "{:<12} {:>10} {:>10.2} {:>10}",
                    format!("{gib} GiB"),
                    "ok",
                    t0.elapsed().as_secs_f64(),
                    addr
                );
                // Touch one byte per 4 MiB so the pages are really committed, then report.
                let mut touched = 0usize;
                for off in (0..bytes).step_by(4 << 20) {
                    if be.upload_range(buf.as_ref(), off, &[0u8]).is_ok() {
                        touched += 1;
                    }
                }
                println!("  committed pages touched: {touched}");
                drop(buf);
            }
            Err(e) => {
                println!(
                    "{:<12} {:>10} {:>10.2} {:>10}  {}",
                    format!("{gib} GiB"),
                    "FAIL",
                    t0.elapsed().as_secs_f64(),
                    "-",
                    e
                );
            }
        }
    }
}
