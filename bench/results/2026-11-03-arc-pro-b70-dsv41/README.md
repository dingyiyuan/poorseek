# 2026-11-03: Intel Arc Pro B70, DeepSeek-V4.1-Flash IQ2_XXS, 62K prompt

Two-phase expert GEMM (dequantize each expert's weights **once** to an f16 scratch, then oneMKL
f16 GEMM) against the fused dequantize-inside-the-MMA kernel.

## Hardware and software

| | |
|---|---|
| GPU | Intel Arc Pro B70 32 GB (Battlemage G31), xe driver |
| CPU / RAM | Core i9-13900K, 192 GB |
| Engine | Strata SYCL port (this tree), oneAPI 2026.2, JIT |
| Model | DeepSeek-V4.1-Flash, IQ2_XXS experts (2.06 bit/weight), 6 experts/word |
| Settings | `--prefill-chunk 40960 --vram-experts 1.0 --mmap-experts auto DS4_ATTN_SLAB=2048` |

## Prefill, 62,290-token prompt (service, `/v1/chat/completions`, one run each, 0 KV reuse)

| expert kernel | prefill |
|---|---:|
| fused (dequantize per 64-row block) | 375.8 tok/s |
| **two-phase (this change)** | **944.1 tok/s** |

Decode is untouched by design: the two-phase path only engages at `rows >= 96`, so a single-token
step stays on the fused kernel. Matched A/B (48-token prompt, 160 output tokens, engine restarted
between runs): two-phase on and off are the same within noise, and the steady-state rate is
**20.8-20.9 tok/s**; the first ~80 tokens read 17.4-18.6 tok/s, which is the usual cold-start dip
(experts being paged in), not a regression.

Expert kernel alone, 900 rows/expert, 6 experts (`--bench-tiles 16 6 900`):

| | gate/up | down |
|---|---:|---:|
| fused | 58,334 us | 16,968 us |
| two-phase | 4,011 us | 2,054 us |
| speedup | **14.5x** | **8.3x** |

## A note on end-to-end text equality

The sampler on this build is not deterministic run to run: the same configuration gives 114-117 tokens on a
repeated greedy request. So "identical output" is not a check I can make at the text level. The correctness
evidence is the kernel-level parity above (all seven quant types, worst 3.61e-04 against a 1e-02 tolerance).

## Why

`set_gate_up`/`set_down` launch a grid of `(ceil(max_rows/64), n_ff/64, n_exp)`: every 64-row block
dequantizes the **same** expert weights. At 900 rows/expert that is 15 redundant dequantize passes.
The two-phase path dequantizes once (83 Gval/s measured) and hands the GEMM to oneMKL's f16 path.
At 96 rows/expert the redundancy is only 2x and the speedup drops to 3.7x, which is the same story.

## Attribution (measured, not estimated)

Repeating the dequantize N times inside the two-phase path (`DS4_EXP_2PHASE_DUP=N`):

| N | gate/up, 900 rows |
|---:|---:|
| 1 | 4,011 us |
| 2 | 5,715 us |
| 15 | 27,870 us |
| fused | 58,334 us |

+1,704 us per extra pass, perfectly linear. So of the 14.5x: **6.95x is the redundant dequantize
being removed**, and **2.09x is the inner product moving from F32 SIMT to oneMKL/XMX**.

## Reproduce

```bash
STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 ./build-sycl-tc2/ds4_iq_test_tc --bench-tiles 16 6 900
STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 DS4_EXP_2PHASE_DUP=15 ./build-sycl-tc2/ds4_iq_test_tc --bench-tiles 16 6 900
STRATA_DS4_NO_TC=1 ./build-sycl-tc2/ds4_iq_test_tc --bench-tiles 16 6 900
```

Numerics: `STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 DS4_EXP_2PHASE_MINROWS=1 ./build-sycl-tc2/ds4_iq_test_tc --type 16`
gives `set gate_up` 3.34e-04 (tol 1e-02) and `set down` 1.42e-04 (tol 3e-03), ALL OK - and **better** than the
fused path's 7.02e-03 / 4.19e-04, because f16 weights with an f32 accumulator round less than per-value f32.
