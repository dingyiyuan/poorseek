# Prompt experts on dequantize + oneMKL FP16 GEMM (the layer-major prompt, S11)

Measured 2026-10-24 on `Intel(R) Arc(TM) Pro B70 Graphics`, one GPU process at a time, `s4/f16/run.sh`
(`MIMO_CHUNK_PROF=1`, ids `(i*100)%120000+1`, temp 0, ctx 4096 at 1024 / 12288 at 8192).

## 0. The short answer

`native_expert_grouped` is a **row-at-a-time** kernel: one 32-lane sub-group owns one output row and re-reads
the weight row for every `(token, expert)` entry, so its cost is flat in m (50.86 us/entry at the 8192-token
geometry = 1.39 TFLOPS = 0.76 % of the 183.5 TFLOPS FP16 peak; `S10_XMX_ARITHMETIC.md`).  The chunk path now
dequantizes each expert's gate/up and down to FP16 and runs **one oneMKL FP16 GEMM per product over all of its
rows** (`native_expert_f16`, new TU), which is 10.8x the old kernel's rate at 8192 and 2.5x at 1024:

| prompt | dense | experts | prompt wall | tok/s |
|---|---:|---:|---:|---:|
| 1024, before | 7.4 s | 22.3 s | 30.4 s | 33.7 |
| 1024, after | 7.7 / 7.2 | **18.1 / 18.3** | **26.6 / 26.1** | **38.6 / 39.2** |
| 8192, before | 26.8 / 27.5 | 77.3 / 77.9 | 105.2 / 106.7 | 77.9 / 76.8 |
| 8192, after | 27.1 / 27.3 | **18.5 / 19.5** | **46.7 / 47.8** | **175.5 / 171.2** |

The expert *kernel* time at 8192 is **65 627 ms -> 6 071 ms (50.86 -> 4.70 us/entry)**.  The phase does not
reach the ~2 s the stand-alone oneMKL benchmark implied, and the reason is not the GEMM: it is
**62.0 GB of blob H2D per prompt at 4.9 GB/s** (the DMAs are on the compute stream and stop being hidden once
the compute is small), plus the whole-layer prefetch.  The expert phase is I/O-bound now (section 5).

## 1. Why the engine was not already using it

The dequantize + `m.gemm.f16` path exists in the tree only in `sycl/src/prefill/prefill.cpp` (the
`STRATA_PF_XMX=0` branch of `strata::prefill::prefill`), which is **not linked into `ds41_generate`**: the
binary is assembled from `s2/engine-obj/*.o` (the `tools/v41/engine` TUs), `s4/step2/obj/*.o` and
`libstrata_kernels_sycl.a`, and `nm -C ds41_generate_gpuexp | grep -c prefill` is 0.  `STRATA_PF_XMX` only
picks a branch inside `strata::prefill::xmx::mode()`, so it has no effect on this binary at all - the deciding
condition was "no call site", not a geometry or format test.

The engine's own prompt expert compute is `Ds41MoeTier::run_chunk -> chunk_classes -> gpu_run_chunk ->
launch()`, which either runs llama.cpp MMQ (`chunk_mmq`, declined for this model's types: "0 sub-runs") or
`native_expert_grouped`.  There was no dequantize-then-GEMM step anywhere in the chunk path.

## 2. The change

* `include/strata/kernels/native_expert_f16.hpp` + `sycl/src/kernels/native_expert_f16.dp.cpp` (new): the
  dequantizers (Q2_0 = 42, Q3_K = 11, PTQ1_0 = 143, both roles) to FP16, an activation gather, the SwiGLU with
  the engine's clamp, a scatter into `parts`, and the two `dpct::blas::gemm` (oneMKL f16 in / f32 out /
  `compute_type::f32`) calls per expert.  Scratch is allocated once and grown to a 1024-row tile; an expert with
  more rows than that runs as several tiles of the same dequantized weights (never a fallback).
* `tools/v41/engine/ds41_moe.cpp`, inside `gpu_run_chunk`'s `launch()`: one `if (f16_path && !use_mmq)` branch
  that walks the run's experts and calls `native_expert_f16` per expert.  `STRATA_CHUNK_F16=0` keeps the old
  kernel (the A/B arm).  Decode does not come through this function at all (`gpu_run` / `gpu_run_n_ex`), so it is
  untouched by construction.

## 3. Numerics

The reference was a stage-by-stage dump (`STRATA_F16_DUMP=<dir>`) against ggml's own `dequantize_row_q3_K` /
`dequantize_row_q2_0` and a plain f32 GEMM/SwiGLU (`s4/f16/check.cpp`): gate/up `max_abs 3.1e-5`
(= FP16 rounding, `rel_l2 2.1e-4`), down exact, `gu` `rel_l2 8.7e-6`, `h` 3.6e-4, `ya` 2.1e-4.  Two bugs were
found this way and fixed: Q3_K's high-bit mask advances with the 128-value half (`1 << (j + 4 n)`) while its
32-byte `hmask` does **not** advance per half (byte `l + 16 s`), and Q2_0 has 8 int16 words per block, not 4.

The route replaces q8_1 (int8 dp4a) activations with FP16 activations and FP32 accumulation, so it is *more*
accurate than the kernel it replaces, and its logits are not bit-identical to it:

| prompt | decoded tokens | top-5 set | argmax | max abs logit diff | rel L2 | md5 of the raw logits |
|---|---|---|---:|---:|---:|---|
| 1024 | identical (16/16: 223 29429 201 ...) | {223,603,10,16} same, 5th differs (base 65 @5.134 vs new 201 @5.1325 - a 0.0015 near-tie) | same | 0.60 | 1.3e-1 | base `787519ba000d`, new `62bcdbc1d098` |
| 8192 | identical (16/16: 99301 99401 ...) | **same set** {99301, 7944, 79749, 201, 223}, ranks 3-5 reordered by <0.2 | same | 1.83 | 1.0e-1 | base `4e5962d94f0e`, new `de41f4d252c7` |

The 1024 md5 cannot match across an arithmetic change: `--dump-last` writes the raw FP32 last-token logits, so
any change to the experts' arithmetic moves their low bits.  The baseline arm of the *same* runner reproduces
`787519ba000d` exactly (so the harness is sound); the gate that carries meaning for this change is "identical
decoded tokens + identical top-5 set + logits within the tolerance", which both sizes pass.

## 4. Where the 4.70 us/entry goes (8192, in-engine stage profile, `STRATA_F16_PROF=1`)

9 433 expert calls, **136 rows per expert** (the real routing; `enc-skip` makes the two halves differ, so it is
not the 232 a single 8192-row layer would give).  Per expert, `STRATA_F16_PROF=1` (which adds a sync per stage,
so the absolute numbers are ~25 % high; the proportions are what matters): weight dequant 405 us (51 %),
both GEMMs 293 us (37 %), gather / SwiGLU / scatter 102 us (13 %).  The clean run is 4.70 us/entry = 643 us per
expert.  The dequant is the fixed cost the FP16 route pays exactly where the old kernel was weakest (small m):
71 MiB written per expert = 0.55 us/entry at m = 232, 4.5 us/entry at m = 29 (the 1024-token geometry).
Submissions are not the problem: a SYCL kernel launch is 3.3-18.6 us and a 4608x8x5120 f16 GEMM 82 us, which is
its own 47 MB weight read (`s4/f16/micro.cpp`).

## 5. Next bottleneck, with numbers

1. **Blob H2D: 62.0 GB per prompt, `copies_gb / pass_wall = 4.6 GB/s`.**  CORRECTED in S15
   (`docs/xolotl-sycl-s4-step10-overlap-dense-report.md`): those copies already ran on `s_cp` and move at 33 GB/s
   (the file-tier source is a USM slab); the ratio above is a derived rate, not a throughput.  What serialised with
   the compute was the *host-side staging memcpy* of the pageable arena (`PinnedArena`'s `cudaHostRegister` is a
   no-op in this port) - 0.88 ms per 11 MB blob.  S15 gives the blob copies their own banked stream and moves the
   prestage submit to a thread: experts 18.7/19.5 -> 13.8/14.1 s, prompt 47.0/47.9 -> 43.8/44.5 s, logits md5
   unchanged.
2. **Dense 27.1 s is now the largest single phase** (58 % of the 46.7 s prompt) and is unchanged by this work.
3. The GEMM itself is at 43-58 TFLOPS in the real geometry (2*n_ff x m x n_embd + n_embd x m x n_ff); with the
   dequant at ~40 % of HBM peak (61 MB per expert in ~250 us) there is maybe 0.3-0.4 s of the 6.1 s left in this
   route at 8192.  The remaining 5 s of the kernel time is the per-expert fixed cost (dequant + submissions),
   which only a fused quantized GEMM (no FP16 round trip) or larger per-launch batches would remove.
