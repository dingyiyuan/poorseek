# S3 — first real run: the SIGILL, the get_rows abort, and a numeric mismatch (report)

2026-10-24. Stage S3 of [xolotl-sycl-port-plan.md](../../llm-for-glm-flash/docs/xolotl-sycl-port-plan.md).
S1 = `6f02176` (CUDA-runtime shim), S2 = `dfc4b53` (SYCL ggml + CPU-expert link), this stage = `cee1f30`.
The engine now loads the GSQ model on the Arc and emits tokens — but its logits are **not** the CPU backend's.

## 1. The SIGILL: a baseline TU compiled for AVX-512

`SIGILL` inside ~1 s of start, after the ggml-sycl banner and device enumeration. Not a guess — gdb:

```
Thread 1 received signal SIGILL, Illegal instruction.
0x6065e2 in strata::kernels::cpu::detail::get_thread_affinity(...)
#1 strata::kernels::cpu::detect_cpu_topology(bool, PoolAffinity)
#2 strata::kernels::cpu::ExpertPool::ExpertPool(int, bool, bool, PoolAffinity, int)
#3 strata::v41::Ds41MoeTier::init(...)   #4 strata::v41::ds41_build_pipeline(...)   #5 main
=> 0x6065e2: vmovups %zmm0,(%rbx)        <-- 64-byte AVX-512 store
```

Host is an **i9-13900K**: `/proc/cpuinfo` has `avx2 avx_vnni fma f16c` and **no `avx512*` flags**
(13th-gen fuses AVX-512 off). The instruction is an ordinary `memset`-style zeroing that the compiler widened
to zmm because the TU was built with `-mavx512f`.

### The brief's diagnosis was wrong, and the reason matters

`native_expert.cpp` **does** do runtime CPUID dispatch (`cpu_avx512_ok()`), so the AVX-512 expert bodies were
never called. The crash is in the *pool*, not the kernels. Per-object census of the S2 objects
(`objdump -d | grep %zmm`) shows S2 applied the AVX-512 flag set to **four TUs that must stay baseline**:

| object | zmm insns | should be | S2 built it as |
|---|---|---|---|
| `iq_avx512.o` | 5795 | AVX-512 | AVX-512 (correct) |
| `nvfp4_avx512.o` | 1728 | AVX-512 | AVX-512 (correct) |
| `expert.o` | 4512 | AVX-512 | AVX-512 (correct — gated kernel bodies) |
| **`pool.o`** | **135** | **baseline** | **AVX-512 — this is the crash** |
| **`expert_layout.o`** | **133** | **baseline** | **AVX-512 — the probe itself** |
| **`native_expert.o`** | **45** | **baseline** | **AVX-512** |
| `iq_avx2/kq_avx2/nvfp4_avx2/q2_avx2.o` | 0 | AVX2 | ok |
| `iq_avx1/kq_avx1/q2_avx1/ptq1_0.o` | 0 | baseline/AVX | ok |

`expert_layout.cpp` is the worst of the three: it is where `cpu_features()` / `cpu_avx512_ok()` live, and its own
comment (line 300) says so — *"The code that decides whether this CPU has AVX-512 … therefore lives here, in a file
compiled for the x86-64 baseline."* Compiled for AVX-512, the probe would fault **before** it could report the CPU
lacks AVX-512 — which is exactly the trap.

Xolotl's own `CMakeLists.txt` (1173-1182) states the rule that was violated:

> A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target so
> that the flag applies only to the TU that needs it. … **a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY
> of its code, so the scalar fallback and the test drivers must NOT be in it** — otherwise the fallback is not a
> fallback, it is the same bet with worse opt-out.

Its per-file map (line 1390) gives AVX-512 to `iq_avx512.cpp` + `nvfp4_avx512.cpp` only; `pool.cpp`,
`expert_layout.cpp`, `native_expert.cpp` get no arch flag.

`-DSTRATA_PORTABLE=ON` was not needed and not used: Strata's own build of the same sources already gets this right,
and `icpx -march=native` on this machine emits **0** zmm (measured), so the over-application was S2's hand-built
compile line, not a CMake default.

## 2. Flag/build change per file

Rebuilt 3 TUs with the compiler's baseline (no arch flag), everything else reused from S2:

| file | flag change |
|---|---|
| `src/kernels/cpu/pool.cpp` | `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512vbmi` → **(none)** |
| `src/kernels/cpu/expert_layout.cpp` | same → **(none)** |
| `src/kernels/cpu/native_expert.cpp` | same → **(none)** |
| `expert.cpp`, `iq_avx512.cpp`, `nvfp4_avx512.cpp` | unchanged (AVX-512, gated at run time) |

```
icpx -O2 -DNDEBUG -std=c++17 -DSTRATA_NATIVE_EXPERTS=1 -DSTRATA_Q6K_EXPERTS=1 \
  -I<xolotl>/include -I<xolotl>/src/kernels/cpu -I<s2>/llama.cpp/ggml/include \
  -c <xolotl>/src/kernels/cpu/{pool,expert_layout,native_expert}.cpp
```

Relink (the CPU-expert flavour: `ds41_moe_cpu.o`, **not** `ds41_moe.o`, and not `ds41_helper.o` — both define `main`):

```
icpx -fsycl -qmkl -O2 -o ds41_generate_cpuexp \
  <engine-obj>/*.o -x ds41_moe.o -x ds41_helper.o  <s3>/cpu-kernels/*.o \
  -Wl,--start-group libggml-sycl.a libggml-cpu.a libggml.a libggml-base.a -Wl,--end-group -lpthread -ldl -lm
```

Result: 93 480 128 B, 0 undefined, and `get_thread_affinity` is now `xorps %xmm0` + `movups` (SSE).
The broken binary is kept as `engine-obj/ds41_generate_cpuexp.s2-avx512-broken` for comparison.

## 3. Second wall: ggml-sycl's get_rows dst type (fixed, `cee1f30`)

With the SIGILL gone the dense graph reached the first attention K/V gather and **aborted**:

```
getrows.cpp:328: GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32) failed
#1 ggml_print_backtrace  #2 ggml_abort  #3 ggml_sycl_op_get_rows  #4 ggml_sycl_get_rows
#5 ggml_backend_sycl_graph_compute_impl  #7 ggml_backend_graph_compute
#8 Ds41Dense::Impl::compute  #10 Ds41Dense::attn_router_span
```

`rows_as()` (`ds41_dense.cpp:676`) retypes the (unallocated) get_rows result to F16 and relies on the **CUDA** kernel
writing F32/F16/BF16 directly — its own comment says *"CUDA only"*. ggml-sycl accepts only F32/I32.

Fix, behind `DS41_SYCL` so CUDA is untouched: gather in F32 and round once with `ggml_cast` (`GGML_OP_CPY`, kernel
`cpy_1_f32_f16`) — the same dequantize-in-float then float→f16 conversion, so the bits match and only a transient F32
tensor is extra. **Verified exactly neutral**: a SYCL run with the F16 gather enabled and one with
`STRATA_ATTN_F16_GATHER=0` produce bit-identical logits (max|Δ| = 0 across all 129 280 logits).

## 4. Third wall: `--experts cpu` needs the host routed path

`ds41_generate: prefill: tier run refused` — `ds41_generate.cpp:1359` picks `rdev = routed_host ? nullptr :
cur_dense->routed_dev()`, so with a GPU dense backend it calls `ExpertPool::run_chunk_dev(...)`, which the CPU expert
tier cannot service. `STRATA_ROUTED_HOST=1` (read at `ds41_generate.cpp:1107`) selects the documented host path
(`run_chunk` + `upload_routed`). **S3 with `--experts cpu` on the SYCL backend therefore needs
`STRATA_ROUTED_HOST=1`.**

## 5. S3 run result

Command (as specified), plus `STRATA_ROUTED_HOST=1`:

```
cd /home/dyy/deepseek-work/llm-for-glm-flash/models/dsv41-gsq
source /opt/intel/oneapi/setvars.sh --force ; export ONEAPI_DEVICE_SELECTOR=level_zero:0
STRATA_ROUTED_HOST=1 ds41_generate_cpuexp -m V41-GSQ-1017.gguf --backend cuda --experts cpu \
  --engram engram-layers.1:engram-layers.14 --rank rank.bin \
  --ids 100,200,300,400,500,600,700,800 -n 2 --ctx 2048 --temp 0
```

**It completes** (exit 0). 256 CU's, 34242M enumerated, 40 layers, 0.2-1.7 s load.

| run | backend | prefill | dense / experts (file wait) | decode | token ids | wall |
|---|---|---|---|---|---|---|
| cold | `cpu` | 8 tok / 13.81 s = 0.6 tok/s | 0.7 / 12.3 (11.6) s | 2 tok / 1.11 s = 0.90 tok/s | **223, 200** | 20 s |
| cold | `cuda` (SYCL) | 8 tok / 12.90 s = 0.6 tok/s | 4.3 / 7.6 (6.9) s | 2 tok / 0.98 s = 1.02 tok/s | **200, 200** | 20 s |
| warm | `cpu` | 3.57 s = 2.2 tok/s | 0.4 / 3.1 (2.6) s | 0.63 s = 1.59 tok/s | 223, 200 | 10 s |
| warm | `cuda` | 8.06 s = 1.0 tok/s | 3.9 / 3.2 (2.7) s | 0.42 s = 2.36 tok/s | 200, 200 | 15 s |

Peak memory never approached the limit (MemAvailable ≥ 186 GiB throughout; watchdog in kB, same unit as
`/proc/meminfo`).

## 6. SYCL vs CPU: the port is numerically wrong

First differing generated position = **token 0**: CPU `223`, SYCL `200`. Compared at the logits (`--dump-last`,
129 280 floats each, the row that chooses token 0):

| | CPU | SYCL |
|---|---|---|
| top-1 | 223 (14.073) | 200 (12.774) |
| top-1 margin | 0.657 | 1.580 |
| token 223 | rank 1 | **rank 8**, 9.756 (Δ −4.32) |
| token 200 | rank 6, 12.301 | rank 1 |

`max|Δ| = 12.856`, `mean|Δ| = 1.423`, `p99|Δ| = 4.81` — **19.6× the CPU top-1 margin**. This is not FP
accumulation-order drift or a near-tie flip; the dense half computes different numbers. It reproduces identically
with and without the F16 gather path (bit-identical dumps), so it is not an artefact of the §3 fix.

Not yet localised to an op. `STRATA_LAYER_DUMP=<dir>` (`ds41_generate.cpp:1211`) dumps per-layer tensors and is the
cheap next step: diff the first layer's `fn`/`routed` between the two backends.

## 7. What stands between S3 and a usable tok/s number

1. **Correctness.** The SYCL dense half does not reproduce the CPU logits (§6). No tok/s figure means anything until
   the first divergent layer/op is found with `STRATA_LAYER_DUMP` and fixed.
2. **The dense half is slower on the GPU than on the CPU at these shapes.** 8-token prefill: SYCL 3.8-4.3 s vs CPU
   0.4-0.7 s (≈6-9×). At B=8 the graph is launch-overhead-bound; the port currently has a performance deficit as well
   as a correctness one, so "dense on SYCL, experts on CPU" has no time advantage yet.
3. **The expert tier is disk-bound with 0 VRAM hits.** 1920 expert reads for an 8-token prefill, file wait 6.9-11.6 s
   of a ~13 s prefill; CPU-expert mode also forces the `STRATA_ROUTED_HOST` per-layer host→device copy of the routed
   sums. Real throughput needs the routed experts on the GPU, i.e. the 10 SYCL kernels from S2 §5a — S4's job.

## 8. Latent bug of the same class as §1 (not hit by these runs)

`pool.cpp`'s legacy single-token path calls `s2_expert_vnni_q` / `s2_expert_gu_rows` / `act_quant_q8_1` directly
(pool.cpp:675/679/771), and those are **AVX-512** bodies in `expert.cpp`. That path is taken when
`n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()` (pool.cpp:763). On an AVX-512-less CPU it would SIGILL
again. The S3 runs took the native path instead (`run_split_multi_native` → `native_gu_rows`, which *is* gated).
`cpu_require_expert_support()` would refuse the CPU at startup, but only Strata's own
`src/program/generate.cpp:2904` calls it — **the Xolotl engine never does**. Worth wiring in before anyone runs with
`--pool-workers 1`.
