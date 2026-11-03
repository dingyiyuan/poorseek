# PR draft for Niko1221/Strata: two-phase expert GEMM for the prompt path

## Problem

`src/dsv41/ds4_iq.cu:1595` and `:1614` build the prompt set's grid as

```cpp
const dim3 grid((unsigned) ((a.max_rows + BM - 1) / BM),   // 64-row blocks
                (unsigned) (a.n_ff / BN), (unsigned) a.n_exp);
```

The work unit is (row-block, column-block, expert), and each row-block **dequantizes the same
expert weights again** inside the fused kernel. At 900 rows/expert that is
`ceil(900/64) = 15` redundant dequantize passes over the same bytes.

Measured on one Intel Arc Pro B70, DeepSeek-V4.1-Flash IQ2_XXS (2.06 bit/weight), 6 experts/word,
`--bench-tiles` equivalent, 900 rows/expert, gate/up: **58,334 us** for the fused path.
Repeating just the dequantize that many times inside a two-phase path reproduces 27,870 us of it,
and a single pass costs 1,704 us — the accounting closes.

## Proposal (opt-in, no behaviour change by default)

For row-heavy sets only, dequantize each expert's gate/up/down **once** into an f16 scratch and run
one GEMM per matrix through the library BLAS the port already links, instead of the fused kernel:

```cpp
if (two_phase_on() && a.max_rows >= kTwoPhaseMinRows) {   // kTwoPhaseMinRows = 96
    if (set_gate_up_2ph(a, s)) return true;               // falls through on any failure
}
```

Design notes that matter:

- **Row threshold is not optional.** At `rows <= 64` the redundancy is 1-2x and the scratch traffic
  makes two-phase *slower*; decode must stay on the fused kernel. 96 is the measured crossover.
- **`SetArgs::bounds` is device memory.** The two-phase path is host-side control flow and must
  `memcpy` bounds back to the host first (reading `a.bounds[e]` on the host is a segfault).
- **`stream == 0` means the default queue** in this codebase (`strata::q_of`); a two-phase path must
  normalize it before using the queue directly.
- The scratch is ~45 MB at `n_embd 5120 / n_ff 2304`, reused across experts, and must be taken into
  account by `--vram-experts` budgeting.

## Measurements (Intel Arc Pro B70 32 GB, xe driver, oneAPI 2026.2, SYCL port)

Service-level, 62,290-token prompt, one run each, zero KV reuse:

| expert kernel | prefill | decode |
|---|---:|---:|
| fused | 375.8 tok/s | 9.33 tok/s |
| two-phase | **944.1 tok/s** | 9.65 tok/s |

Kernel alone, 900 rows/expert, 6 experts: gate/up **14.5x**, down **8.3x**.
Attribution by repeating the dequantize N times: **6.95x from removing the redundancy**,
**2.09x from the inner product moving to the BLAS/XMX f16 path**.

Numerics: two-phase `set gate_up` worst 3.34e-04 (fused 7.02e-03), `set down` 1.42e-04 (fused
4.19e-04) against the same double-precision reference - i.e. **more** accurate, since f16 weights
with an f32 accumulator round less than per-value f32.
