//! T5 verification: confirm `infr` now RECOGNIZES the DSV4.1 GGUF (arch string
//! "deepseek41") and parses it as the deepseek4 family — WITHOUT loading weights.
//!
//! `Gguf::open` mmaps the file lazily (header only); `Config::from_gguf` reads
//! only metadata. Zero weight pages are touched, so this is safe to run on the
//! full 246 GiB shard (no freeze risk).
//!
//! Run: cargo run --release -p infr-llama --example dsv41_arch -- <shard1.gguf>
use std::path::PathBuf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dsv41_arch <path-to-dsv41-shard1.gguf>");
    let p = PathBuf::from(&path);
    println!("file: {}", p.display());

    // 1. cheap arch-string peek (same fn `infr run` uses to pick sampling defaults)
    println!("arch_of: {:?}", infr_llama::arch::arch_of(&p));

    // 2. full config parse (metadata only — no SeamModel::load, no weight paging)
    match infr_gguf::Gguf::open(&p) {
        Err(e) => {
            println!("Gguf::open ERR: {e:#}");
            return;
        }
        Ok(g) => {
            // metadata/config gate
            match infr_llama::Config::from_gguf(&g) {
                Ok(cfg) => {
                    println!(
                        "Config OK ✓ — deepseek4={}, deepseek41={}, n_layer={}, q_lora_rank={}",
                        cfg.deepseek4, cfg.deepseek41, cfg.n_layer, cfg.q_lora_rank
                    );
                    // T4 config side: the CSA2 source map derived from tensor presence inside
                    // Config::from_gguf (expect kv self-refs at L2/8/14/20, topk at 8 idx layers).
                    println!("  dsv41_kv_source:        {:?}", cfg.dsv41_kv_source);
                    println!("  dsv41_index_key_source: {:?}", cfg.dsv41_index_key_source);
                    println!("  dsv41_topk_source:      {:?}", cfg.dsv41_topk_source);
                    // Parsed-value dump: the graph reads these directly, so a silently wrong
                    // parse is invisible until the output is word salad. Compared by hand
                    // against the GGUF metadata (`parse_gguf.py`).
                    println!(
                        "  hc_mult={} hc_eps={:e} sinkhorn={} rms_eps={:e}",
                        cfg.hc_mult, cfg.hc_eps, cfg.hc_sinkhorn_iters, cfg.rms_eps
                    );
                    println!(
                        "  n_head={} head_dim={} o_groups={} o_lora={} compress_theta={}",
                        cfg.n_head,
                        cfg.head_dim,
                        cfg.o_group_count,
                        cfg.o_lora_rank,
                        cfg.compress_rope_theta
                    );
                    println!(
                        "  indexer: heads={} head_size={} top_k={}",
                        cfg.indexer_n_head, cfg.indexer_head_size, cfg.indexer_top_k
                    );
                    println!(
                        "  swiglu_clamp_exp[0..4]={:?} len={}",
                        &cfg.swiglu_clamp_exp[..cfg.swiglu_clamp_exp.len().min(4)],
                        cfg.swiglu_clamp_exp.len()
                    );
                }
                Err(e) => println!("Config::from_gguf ERR (next metadata gate): {e:#}"),
            }
            // tensor-PRESENCE gates — probed by name via tensor_file_range (returns the byte
            // range, reads NO weight data → zero freeze risk on the 246 GiB file).
            // These are NOT hit by Config::from_gguf (metadata-only); they fire at load/graph
            // time, so the probe mimics them here to complete the V4.1 diff map.
            println!("--- tensor-presence probes (T3 + CSA2 source map) ---");
            let probe = |name: &str| g.tensor_file_range(name).is_ok();
            // T3: V4 ships output_hc_* (learned HC head); V4.1 does NOT (uses last FFN mix).
            for t in ["output_hc_down", "output_hc_norm", "output_hc_up", "output_hc_inject"] {
                println!("  {t}.weight present: {}  (V4=yes / V4.1=no)", probe(&format!("{t}.weight")));
            }
            // CSA2 source derivation (deepseek41.cpp:212-253): a layer is a KV source iff it
            // carries attn_compressor_kv; an index source iff indexer.attn_q_b; ratio-2 sources
            // also carry attn_compressor_gate (the pool gate).
            let mut srcs: Vec<(usize, bool, bool, bool)> = Vec::new();
            for i in 0..40usize {
                let (kv, gate, idx) = (
                    probe(&format!("blk.{i}.attn_compressor_kv.weight")),
                    probe(&format!("blk.{i}.attn_compressor_gate.weight")),
                    probe(&format!("blk.{i}.indexer.attn_q_b.weight")),
                );
                if kv || idx {
                    srcs.push((i, kv, gate, idx));
                }
            }
            println!("  CSA2 source layers (i, kv_src, pool_gate, idx_src): {srcs:?}");
        }
    }
}
