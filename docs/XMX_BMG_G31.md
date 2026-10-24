# Grouped XMX on Battlemage (Xe2 / BMG-G31): the AOT variant, and whether the port pays

Measured 2026-10-24 on `Intel(R) Arc(TM) Pro B70 Graphics` (`sycl-ls` `level_zero:0`, oneAPI 2026.2,
ocloc 26.35.39758.10).  Everything below is standalone: no engine change, no rebuild of `strata` or of
`s4/step2/ds41_generate_gpuexp`, one GPU process at a time.

**Verdict up front:** the AOT XMX variant builds and runs, and it is 6.7-18x faster than the row-at-a-time
expert kernel that takes 65.6 s today - but it is **2.0-2.7x slower than the per-expert oneMKL FP16 GEMM the
tree already has**, at every measured m from 8 to 512 rows/expert.  The expert phase does not need the port;
it needs the GEMM to amortise over m, and oneMKL (which is itself an XMX kernel, at 60-70 TFLOPS here vs the
hand-written kernel's 16-26) already does that.
Details, numbers and the recipe: below.

---

## 1. The AOT Battlemage variant

### What the tree already had

`sycl/src/prefill/xmx_moe.dp.cpp` writes the grouped FP16 expert GEMM for both device generations, chosen at
compile time by `STRATA_XMX_SG16`:

| | sub-group 8 | sub-group 16 (`STRATA_XMX_SG16`) |
|---|---|---|
| device | DG2 / Arc A-series (Alchemist) | Xe2 / Battlemage (`intel_gpu_bmg_g21`, `intel_gpu_bmg_g31`) |
| joint_matrix | `8x8x16` | `8x16x16` (A `8x16` row-major, B `16x16` col-major, accumulator `8x16`) |
| `kSgW` | 8 | 16 |
| tiles | `Cfg<2,8,4,2,32>` (64x128) / `Cfg<4,4,4,4,32>` (128x128) | `Cfg<2,4,4,2,32>` (64x128) / `Cfg<4,2,4,4,32>` (128x128) |
| work-group | 128 threads = 16 sub-groups of 8 | 128 threads = 8 sub-groups of 16 |

`sycl/CMakeLists.txt` (lines 31-49) is the switch:

```
-DSTRATA_SYCL_AOT=bmg-g31
  -> -DSTRATA_XMX_SG16=1
  -> -fsycl-targets=spir64_gen
     -Xsycl-target-backend=spir64_gen "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt"
```

`-DSTRATA_SYCL_AOT=pvc` gives `-DSTRATA_NO_DG2_XMX` (no grouped XMX at all); an empty `STRATA_SYCL_AOT` gives
the JIT/SPIR-V build with the sub-group 8 kernels.  `bmg-g31` is a valid ocloc device (`ocloc compile --help`
lists `bmg-g31`, `bmg-g31-a0`, `bmg-g21`, ...); the runtime side is
`sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31` in `strata::intel_gpu_gen`
(`sycl/include/strata/sycl_queue.hpp:139`).

So the *source* variant existed; what the tree did not have was a **built** one - `strata/build-sycl-aot` was
configured `-DSTRATA_SYCL_AOT=bmg-g31` on 2026-10-19, before `xmx_moe.dp.cpp` entered the build, and the
shipped engine binary is assembled from individual TUs (`s4/step2/build_all.sh`), not from CMake.

### The recipe (`sycl/tools/xmx_bmg_aot.sh`)

```sh
R=<repo>; AOT=bmg-g31
icpx -fsycl -O2 -std=c++20 -fp-model=precise -fsycl-default-sub-group-size=32 \
     -fsycl-device-code-split=per_kernel -DSTRATA_XMX_SG16=1 \
     -I$R/sycl/include -I$R/include -I$R/sycl/third_party/ggml \
     -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \
     "-device bmg-g31 -options -cl-fp32-correctly-rounded-divide-sqrt" \
     -c $R/sycl/src/prefill/xmx_moe.dp.cpp -o xmx_moe_sg16.o
icpx ... -c $R/sycl/src/kernels/xmx_bmg_bench.cpp -o xmx_bmg_bench.o
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "<same device opts>" -qmkl=sequential \
     xmx_moe_sg16.o xmx_bmg_bench.o -o xmx_bmg_bench
```

Out of tree (`./xmx-bmg/build`), ~11 s, `ocloc` reports `Build succeeded` three times (one module per kernel,
`per_kernel` split).  `-fsycl-default-sub-group-size=32` must stay: only the XMX kernels pin their own width
(`[[sycl::reqd_sub_group_size(kSgW)]]`).

### Runtime selection, verified both ways

`xmx::mode()` (`xmx_moe.dp.cpp:151`) reads `STRATA_PF_XMX` and, in a non-SG16 build, refuses on Battlemage.
With `STRATA_PF_XMX=1` and the same 15-line caller linked against each kernel object:

| build | output on the B70 |
|---|---|
| JIT (sub-group 8) | `STRATA_PF_XMX ignored: this build has the sub-group 8 / 8x8x16 kernels (Arc A-series); Battlemage needs the AOT bmg build` -> `mode() = 0` |
| AOT `bmg-g31`, `STRATA_XMX_SG16` | silent -> `mode() = 1` |

and the kernel itself produces correct results (section 4), which is the real proof that `bmg-g31` is the
right arch string and that `8x16x16`/sub-group 16 is the right shape for this card.

The call sites are `sycl/src/prefill/prefill.cpp`: mode 1 (`xmx_on`, line ~4082) dequantizes a group of up to
16 experts to FP16 and calls `grouped_f16` - gate/up only while the group's mean rows stay under
`STRATA_PF_XMX_GU_MAX` (250), otherwise per-expert `m.gemm.f16`; down always through `grouped_f16`.  Mode 2
(`xmx2`, line ~3948) is the fused IQ-decode variant (`iq_xmx_grouped`), not measured here.

---

## 2. The harness

`sycl/src/kernels/xmx_bmg_bench.cpp`, built by the script above, run as
`./xmx-bmg/build/xmx_bmg_bench --m 48,84,145,232,320 --reps 7 --equal`.

* Real routed geometry (V4.1 `text_config`: hidden 5120, moe_intermediate 2304): gate/up
  `X[m x 5120] . W_e[4608 x 5120]^T` (the engine's fused gate+up, N = 2*n_ff), down
  `X[m x 2304] . W_e[5120 x 2304]^T`.
* 16 experts per launch = `kMaxGroup` = the engine's dq ring.
* `--equal` gives exactly m rows per expert (controlled); without it the counts are gamma(1.5) like the real
  router (same conclusions, ~10 % noisier).
* **Timing rule:** every section is read only after `q.wait_and_throw()`.  The 7.5 us/entry figure this port
  was sized on came from reading elapsed time off event pairs that were never waited for (`cudaErrorNotReady`,
  counted as 0 ms); nothing here can reproduce that.
* The fallback column is one `dpct::blas::gemm` per expert (FP16 in, FP32 out, `compute_type::f32`) - exactly
  `Gemm::f16` in `sycl/src/prefill/gemm.dp.cpp` and the arithmetic of the `STRATA_PF_XMX=0` branch.
* Reference: FP32 CPU matmul over every row of the group and the first/last 24 columns of each expert, plus a
  whole-tensor comparison against the oneMKL output.
* GPU clocks ramp: the first measured point of a cold run reads ~20 % high.  Run it twice and report the
  second pass (that is what `xmx-bmg/bench_bmg_g31.log` is).

---

## 3. Results (B70, 16 experts/launch, eager per-product launches)

`us/ent` = microseconds per (row, expert) row of one product; `total` = gate/up + down per row, which is what
the 50.86 us/entry of `native_expert_grouped` measures.

| m rows/expert | rows | gate/up us/ent | gate/up TFLOPS | down us/ent | down TFLOPS | **XMX total us/row** | oneMKL total us/row | XMX/oneMKL |
|---|---|---|---|---|---|---|---|---|
| 48 | 768 | 6.18 | 7.6 | 1.36 | 17.3 | **7.55** | 3.18 | 0.42 |
| 84 | 1344 | 3.86 | 12.2 | 1.48 | 15.9 | **5.35** | 2.12 | 0.40 |
| 145 | 2320 | 2.85 | 16.5 | 1.34 | 17.6 | **4.20** | 1.22 | 0.29 |
| 232 | 3712 | 1.90 | 24.8 | 0.94 | 25.1 | **2.84** | 1.07 | 0.38 |
| 320 | 5120 | 1.79 | 26.4 | 0.96 | 24.6 | **2.75** | 0.80 | 0.29 |

The task's literal shape (gate and up as two N = 2304 launches instead of one fused N = 4608) measures
7.82 us/ent at m = 48 and 2.16 at m = 232 for gate/up - 6-20 % *worse* than the fused call the engine actually
makes, and 9.18 / 2.96 us/row in total.  Nothing in the conclusion below depends on which of the two is used;
the fused one is what `prefill.cpp` calls.

At the 8192-token geometry (mean m = 232; 1.29M (row, expert) rows - `91.3 TFLOP / 70.8 MFLOP per row`, and
`65.63 s / 50.86 us`):

| path | us/row | expert phase | vs today |
|---|---|---|---|
| today: `native_expert_grouped` (row-at-a-time, flat 47-51 us) | 50.86 | 65.63 s | 1.0x |
| **this work: grouped XMX, AOT bmg-g31, sub-group 16** | 2.84 | **3.7 s** | 17.9x |
| the tree's per-expert oneMKL FP16 GEMM (`m.gemm.f16`) | 1.07 | **1.4 s** | 47.5x |
| dequant floor (1.13 GB of FP16 written per 16-expert group, 0.55 us/row at 551 GB/s) | +0.55 | +0.7 s | both FP16 routes pay it |

So the honest end-to-end expert phase is ~4.4 s with the XMX port and ~2.1 s without it (GEMM + dequant
floor; the dequant kernel also reads ~146 MB of quantized weights and unpacks, so treat 0.55 us/row as a
lower bound).

TFLOPS language: the card's FP16 peak is 183.5.  Today's kernel runs at 1.4 TFLOPS (0.76 %).  The hand-written
grouped XMX kernel reaches 16.5-26.4 TFLOPS (9-14 %).  oneMKL reaches 60-70 TFLOPS (33-38 %).  The gap between
the two XMX paths is the kernel, not the hardware: the tree's kernel does 16 `joint_matrix_load`s from SLM per
32-k step, re-reads the X tile once per 128-wide N block, and re-reads each expert's weights once per M tile -
at m = 232 with BM = 128 that is 2x the weight traffic of a single 232-row GEMM.

## 4. Numerical error

Against an FP32 CPU reference (same FP16 inputs, so what is left is accumulation error):

| product | max abs | max abs / max\\|Y\\| | max rel (denominator max(\\|ref\\|, 1e-2 max\\|Y\\|)) | rel L2 |
|---|---|---|---|---|
| gate/up | 1.30e-5 | 3.1e-6 | 7.7e-5 | 1.32e-6 |
| down | 5.60e-6 | 2.1e-6 | 5.2e-5 | 8.78e-7 |

Against the tree's oneMKL FP16 GEMM on the same inputs: **bit-identical** (`max_abs = 0` over the whole
tensor) for every m >= 145, i.e. at the geometry that matters; at m = 48/84 the difference is 2-4e-6 absolute
(`rel L2` 3e-7 / 2e-7) because oneMKL picks a different kernel for small T there.  Both are DPAS kernels
accumulating K = 5120 in FP32 in the same order, so exact equality is not an accident.  The FP16 format's own
expectation (input rounding, ~1e-3 relative) is met with three orders of magnitude to spare - the kernel is
correct.

## 5. Does XMX pay, and where does it stop?

* **Against today's kernel:** yes, everywhere measured - 6.7x at m = 48, 9.5x at 84, 12.2x at 145, 17.9x at
  232, 18.5x at 320.  No inversion down to 8 rows/expert (36.5 us/row there, still ~7x better than a kernel
  that is flat at ~50).
* **Against the tree's oneMKL per-expert GEMM:** no, nowhere.  XMX/oneMKL is 0.29-0.42 across m = 8 ... 512
  (at m = 8: 36.5 vs 18.1 us/row; at 512: 2.4 vs 0.6).  The A770 result where grouped XMX beats oneMKL below
  ~250 rows does not merely shift on this card - it is absent: oneMKL's Xe2 kernels are 2.4-3.4x more
  efficient per FLOP than the hand-written tile at every size tested.
* **The port is unnecessary for the 1.7-5 s target.**  The dequantize-then-oneMKL route is already in the
  tree (`Gemm::f16` + `iq_dequant_expert_f16`, the `STRATA_PF_XMX=0` / `gu_max_mean` fallback branch of
  `prefill.cpp`) and lands the phase at ~1.4 s of GEMM, ~2.1 s with the dequant floor.  What the 65.6 s
  really measures is zero m-amortisation, not the absence of XMX: any batched GEMM fixes it, and the tuned
  one does it better than the hand-written one.

## 6. Next step

1. Run the existing `STRATA_PF_XMX=0` (dequantize + per-expert `m.gemm.f16`) prompt path on the real 8192-token
   prompt and read the new `kPfGemmGU`/`kPfGemmD` marks.  If the expert phase lands near ~2 s, the XMX port is
   closed.
2. If more is wanted, the remaining lever is the dequant traffic (1.13 GB per 16-expert group).  A *fused*
   quantized grouped kernel that reads the quantized weights once per M tile and DPASes from SLM would remove
   the FP16 round trip entirely (144 MB instead of 1.13 GB per group) - but that kernel has to be written to
   oneMKL's efficiency, not the current tile's: the measured ceiling of the current tile (26 TFLOPS, 14 % of
   peak) is below what a fused version needs to beat 2.1 s.
3. Keep the AOT `STRATA_XMX_SG16` build as the correctness baseline it now is: it is the only configuration
   where `STRATA_PF_XMX` is accepted on this card, and it reproduces oneMKL's FP16 result bit for bit at
   m >= 145.

Reproduce: `sycl/tools/xmx_bmg_aot.sh && ./xmx-bmg/build/xmx_bmg_bench --m 48,84,145,232,320 --reps 7 --equal`
(numbers in `docs/results/2026-10-24-bmg-g31-xmx.csv`, raw console in `xmx-bmg/bench_bmg_g31.log`).
