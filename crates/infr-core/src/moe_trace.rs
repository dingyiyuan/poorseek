//! Per-layer routed-expert histogram, sampled at the PAGER (every resolve), not at the router
//! readback — the all-resident fast path never reads router ids back to the host, so a
//! readback-based trace misses most decode steps.
//!
//! Enabled by `INFR_MOE_ROUTING_TRACE=1`. Block ids are recorded raw; the dump decodes them with
//! the geometry from the environment:
//!
//! * `INFR_MOE_TRACE_ROLE_STRIDE` (default 3200 = 40 layers × 80 experts for the N=80 view),
//! * `INFR_MOE_TRACE_N_EXPERT`   (default 80),
//! * `INFR_MOE_TRACE_LAYERS`     (default 40).
//!
//! Block id layout (see `infr_vulkan::pager`'s `register`):
//! `id = role * role_stride + layer * n_expert + local`.
//!
//! Output: one CSV row per layer with
//! `access, distinct, hhi (Herfindahl), eff_n (1/HHI), top1..top64` — the numbers a pinning
//! policy needs to decide how many experts per layer are worth pinning.

use std::sync::{Mutex, OnceLock};

const MAX_BLOCKS: usize = 1 << 16;

fn enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var_os("INFR_MOE_ROUTING_TRACE").is_some())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn table() -> &'static Mutex<Vec<u64>> {
    static T: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(vec![0u64; MAX_BLOCKS]))
}

/// Count one block resolution (hit or miss) at `id`.
pub fn record_block(id: u32) {
    if !enabled() || id as usize >= MAX_BLOCKS {
        return;
    }
    let Ok(mut guard) = table().lock() else { return };
    guard[id as usize] += 1;
}

/// Print the per-layer histogram once. Idempotent.
pub fn dump() {
    if !enabled() {
        return;
    }
    static DUMPED: OnceLock<()> = OnceLock::new();
    if DUMPED.set(()).is_err() {
        return;
    }
    let role_stride = env_usize("INFR_MOE_TRACE_ROLE_STRIDE", 3200);
    let n_expert = env_usize("INFR_MOE_TRACE_N_EXPERT", 80);
    let n_layers = env_usize("INFR_MOE_TRACE_LAYERS", 40);
    let roles = env_usize("INFR_MOE_TRACE_ROLES", 3);
    let Ok(guard) = table().lock() else { return };
    let mut hist = vec![vec![0u64; n_expert]; n_layers];
    for (id, &count) in guard.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let within = id % role_stride;
        let layer = within / n_expert;
        let expert = within % n_expert;
        if layer < n_layers && expert < n_expert {
            hist[layer][expert] += count;
        }
    }
    println!(
        "[moe routing] per-layer routed-expert concentration (pager-sampled; role_stride={role_stride} n_expert={n_expert} roles={roles})"
    );
    println!(
        "layer,access,distinct,hhi,eff_n,top1,top2,top4,top6,top8,top12,top16,top24,top32,top48,top64"
    );
    for (li, row) in hist.iter().enumerate() {
        let total: u64 = row.iter().sum();
        if total == 0 {
            continue;
        }
        let distinct = row.iter().filter(|&&c| c > 0).count();
        let hhi: f64 = row
            .iter()
            .map(|&c| {
                let p = c as f64 / total as f64;
                p * p
            })
            .sum();
        let eff = if hhi > 0.0 { 1.0 / hhi } else { 0.0 };
        let mut sorted: Vec<u64> = row.iter().copied().filter(|&c| c > 0).collect();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let cum = |k: usize| -> f64 {
            let s: u64 = sorted.iter().take(k).sum();
            s as f64 / total as f64
        };
        println!(
            "{li},{total},{distinct},{hhi:.6},{eff:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            cum(1),
            cum(2),
            cum(4),
            cum(6),
            cum(8),
            cum(12),
            cum(16),
            cum(24),
            cum(32),
            cum(48),
            cum(64)
        );
    }
}
