# S2 — SYCL-enabled ggml + the DeepSeek-V4.1 engine's link (report)

2026-10-24. Stage S2 of [xolotl-sycl-port-plan.md](../../llm-for-glm-flash/docs/xolotl-sycl-port-plan.md) (the plan
lives in the llm-for-glm-flash worktree). S1 = `tools/v41/engine/sycl_compat/` (commit `6f02176`). Nothing here runs a
model: the weights were still downloading, and the machine hard-freezes if a heavy link runs beside a loaded model.

## 1. Which ggml tree, and did the Xolotl patch apply?

| | |
|---|---|
| pinned commit (`tools/xolotl/fetch-llama.sh`) | `3cf03257f219afbe7334045ff7c6a06ac68c627d`, 2026-09-20 |
| tree actually used | a copy of `llm-for-glm-flash/llama-cpp-ref` at `a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`, 2026-10-14 |
| why not the pin | the pin's objects are not in the local tree (`git cat-file -e 3cf0325…` fails) and shell-side GitHub is dead on this box (`git ls-remote` times out); reusing the fetched tree was the cheap path the task allowed |
| build location | `/home/dyy/deepseek-work/s2/llama.cpp` (copy; `llama-cpp-ref` was only read) |
| patch | `tools/xolotl/ggml/xolotl-ggml.patch` — **applies with 15 rejected hunks in 4 CUDA files** (48 hunks total, 33 applied) |

Rejected hunks (`git apply --reject`, then the 4 files were restored to pristine so no file is half-patched):

| file | rejected / total | why (drift between 2026-09-20 and 2026-10-14) |
|---|---|---|
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 1 / 1 | `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` changed signature (`ctx` → `cc, dst, ncols1`) and the sparse path was reworked (#28770) |
| `ggml/src/ggml-cuda/getrows.cu` | 1 / 1 | the `get_rows` launch block the hunk anchors on moved |
| `ggml/src/ggml-cuda/mmq.cu` | 7 / 7 | `mmq.cu`'s function preamble changed (`GGML_TENSOR_BINARY_OP_LOCALS`/stream/cc block no longer matches) |
| `ggml/src/ggml-cuda/mmq.cuh` | 6 / 7 | `ggml_cuda_mmq_config`'s macro/parameters changed (the config struct gained `stream_k`, `fallback`) |

All of these are **CUDA-backend files**. The SYCL build compiles none of them (`GGML_CUDA=OFF`), and every hunk in the
files a SYCL build *does* compile applied cleanly: `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/{ggml-cpu.c,ops.cpp,ops.h}`,
`ggml/src/ggml-backend-meta.cpp`. So the SYCL tree carries the Xolotl ops (`GGML_OP_DSV41_FAKE_QUANT`,
`GGML_OP_DSV41_HC_MIXES`, the fp4 KV packers, the no-graph batches) with a **CPU implementation**; the CUDA tree from
this copy would need the 15 hunks rebased before it is buildable — it is not used for S2. If the exact pin is ever
needed, `fetch-llama.sh` remains the way to get it (it needs working network).

## 2. The SYCL build

`icpx` 2026.2.0, CMake 4.2.3, Ninja, oneAPI at `/opt/intel/oneapi`:

```
source /opt/intel/oneapi/setvars.sh --force
cmake -S <tree> -B <out-of-tree>/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
      -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DNN=OFF -DGGML_SYCL_F16=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_CURL=OFF
cmake --build build --target ggml-sycl -j 8      # builds ggml-base too; then `--target ggml ggml-cpu`
```

* `GGML_SYCL_DNN=OFF` because this box has **no oneDNN** (`/opt/intel/oneapi/dnnl` absent) — the backend's default
  `ON` is a hard CMake error without it. Re-enable once `intel-oneapi-dnnl-devel` is installed (20–40% prefill on
  long prompts, per ggml-sycl's own comment).
* `BUILD_SHARED_LIBS=OFF` / `GGML_STATIC=ON` is not cosmetic: the engine calls `ggml_graph_next_uid()`
  (`tools/v41/engine/ds41_dense.cpp:101`), an internal symbol that a shared `libggml.so` does not export. Xolotl's own
  CMake forces the same (`set(GGML_STATIC ON)` + `BUILD_SHARED_LIBS OFF`) for the native-expert build.
* `GGML_SYCL_GRAPH=ON`, `GGML_SYCL_HOST_MEM_FALLBACK=ON`, `GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON` (defaults; the Level
  Zero loader/headers are present).
* Result: `libggml-sycl.a`, `libggml-cpu.a`, `libggml.a`, `libggml-base.a`, all static, ~166 MB of build tree. Source
  180 MB + build 166 MB + engine objects/executable 95 MB + CPU-kernel objects 2 MB = **~443 MB** for everything S2
  produced (inside the 20 GB budget; the 356 GiB model download, not this build, is what the disk constraint is about).

## 3. Source changes

Engine (`tools/v41/engine/`), every change behind `DS41_SYCL` so the CUDA build is byte-for-byte unaffected:

| file | change | guard |
|---|---|---|
| `ds41_pipeline.cpp` | `#include "ggml-sycl.h"` instead of `ggml-cuda.h`; `make_cuda_backend()` calls `ggml_backend_sycl_init(device)` then `ds41_sycl_adopt_ggml_queue(device)` | `#ifdef DS41_SYCL` inside the existing `STRATA_ENABLE_CUDA` block |
| `ds41_generate.cpp` | same include switch; backend `"cuda"` branch calls `ggml_backend_sycl_init(0)` + adopt | `#ifdef DS41_SYCL` |
| `ds41_dense.cpp` | the 6 `ggml_backend_cuda_register_host_buffer` call sites (lines ~1902, ~1914, ~1915, ~1917, ~1918, ~3125) compiled out | `#if defined(DS41_HAVE_CUDA) && !defined(DS41_SYCL)` / `#if defined(STRATA_ENABLE_CUDA) && !defined(DS41_SYCL)` |

No stub was added for `ggml_backend_sycl_register_host_buffer` (it does not exist; `ggml-sycl.h:51-53` keeps it
commented out). Those buffers stay pageable and the shim stages their copies — the same consequence as S1's no-op
`cudaHostRegister`, and honest, unlike a fake success.

Shim (`tools/v41/engine/sycl_compat/`):

| file | change |
|---|---|
| `ds41_sycl_compat.hpp` | declares `cudaError_t ds41_sycl_adopt_ggml_queue(int device)` (S2, not CUDA API) |
| `ds41_sycl_compat.cpp` | weak declaration of `ggml_backend_sycl_get_default_queue`; `ds41_sycl_adopt_ggml_queue()` makes the shim's default stream a handle copy of ggml's in-order queue and — while nothing is allocated yet — adopts ggml's `sycl::context`/`sycl::device` too |

The weak declaration keeps the S1 self-test linkable **without ggml**: rebuilt here, `test_ds41_sycl_compat` still
links standalone (140 kB) with the new shim.

ggml tree (a patch file, committed, not applied to any tracked source):

| file | change |
|---|---|
| `ggml/include/ggml-sycl.h` | declares `void * ggml_backend_sycl_get_default_queue(int device)` |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | implements it as `&dpct::dev_mgr::instance().get_device(device).default_queue()` |

## 4. The ggml-sycl queue accessor and how the shim adopts it

Patch: `tools/xolotl/ggml/xolotl-sycl-queue.patch` (52 lines incl. header; `git apply` verified clean against the
pristine `llama-cpp-ref` tree). Core:

```c
// ggml/include/ggml-sycl.h
GGML_BACKEND_API void * ggml_backend_sycl_get_default_queue(int device);
```
```c++
// ggml/src/ggml-sycl/ggml-sycl.cpp
void * ggml_backend_sycl_get_default_queue(int device) {
    if (device < 0 || device >= ggml_backend_sycl_get_device_count()) return nullptr;
    return (void *) &dpct::dev_mgr::instance().get_device((unsigned int) device).default_queue();
}
```

`default_queue()` is `in_order_queue()` (`dpct/helper.hpp:731`) — the same queue ggml-sycl's buffer types and
`graph_compute` submit to. Shim side (`ds41_sycl_compat.cpp`):

1. `ggml_backend_sycl_get_default_queue` is declared **weak**; when the shim is linked without ggml (S1 self-test) the
   address is null and adoption returns `cudaErrorNotSupported`.
2. `ds41_sycl_adopt_ggml_queue(device)` stores a **copy of the `sycl::queue` handle** (not a second queue: `sycl::queue`
   is a handle type, so submissions through the copy are ordered with ggml's, and ggml keeps ownership), replacing the
   shim's own lazily-created default queue. If one already existed it is drained (`wait()`) first.
3. While `R.allocs` is empty and no other shim stream is live, it also sets `R.ctx = gq->get_context()` and
   `R.dev = gq->get_device()`. The engine calls adopt immediately after `ggml_backend_sycl_init`, before any engine
   allocation, so **USM the engine allocates is created in ggml's context** — the spec-guaranteed form of what S1
   could only verify empirically (cross-context USM). This also removes the device-picking risk: the shim now runs on
   whatever device ggml's `default_queue()` belongs to, instead of independently re-resolving `gpu_selector_v`.
4. `cudaStreamSynchronize(nullptr)` still waits every shim queue, and now that includes ggml's (the default stream).
   Events recorded on the adopted queue lose SYCL profiling info (ggml's queue is not created with
   `enable_profiling`), so `cudaEventElapsedTime` falls back to host timestamps for default-stream events — already a
   documented S1 fallback, now used more often.

## 5. Link result for `ds41_generate`

All 8 S1 engine files + `ds41_conversation.cpp` + `ds41_dense.cpp`'s companion `ds41_engram.cpp` compile with
`-DDS41_SYCL=1 -DSTRATA_ENABLE_CUDA=1 -DDS4_MOE_CUDA=1 -DDS41_HAVE_CUDA=1 -DDS41_POOL_CUDA=1 -DDS41_HAVE_DSPARK=1`
and the shim. Two link flavours, because the real target has two (`v41_engine.cmake:108-121`):

### 5a. GPU-expert flavour (`ds41_moe.cpp` **with** `DS4_MOE_CUDA`) — **BLOCKED**, but not on ggml

With the static ggml-sycl + all Strata CPU-kernel objects available, the link leaves **28 unique undefined symbols**:

| # | symbols | defined in | ported? |
|---|---|---|---|
| 10 | `strata::kernels::{native_expert_grouped, native_expert_layout, native_expert_layout3, native_expert_scratch_bytes, native_expert_supported3, native_quantize_q8_1, native_q8_1_bytes, weighted_rows_sum, decode_rows_sum, scatter_rows_f32}` | `src/kernels/cuda/iq_kernels.cu`, `fork/fork_layout.cuh`, `fork/fork_weighted_sum.cuh`, `elementwise.cu` | **no** — CUDA kernels, S1 shimmed only the CUDA *runtime*, not these |
| 17 | `strata::core::{ExpertCache::*, PinnedArena::*, arena_pin_cap_gib}` | `src/core/` | C++, buildable; not built in S2 |
| 1 | `ds41_add_f32` | `tools/v41/engine/ds41_transcode.cu` | no — CUDA `.cu`, separate from ggml |

**None of the 28 is a ggml symbol.** In particular the two S1 leftovers are gone: `ggml_backend_cuda_init` is now
`ggml_backend_sycl_init` at both sites and `ggml_backend_cuda_register_host_buffer` is compiled out; the weak accessor
resolves against `libggml-sycl.a`; `ggml_graph_next_uid` resolves from the static ggml. The blocker is the engine's
**own CUDA kernel layer**: the routed-expert tier never goes through ggml — `ds41_moe.cpp:826/2966/3705/4070/5003`
call `strata::kernels::native_quantize_q8_1` and `strata::kernels::native_expert_grouped` directly.

### 5b. CPU-expert flavour (`ds41_moe.cpp` **without** `DS4_MOE_CUDA`, i.e. the `ds41_moe_cpu` target) — **LINKS**

`tools/v41/engine/ds41_moe.cpp` without `DS4_MOE_CUDA` + `strata_kernels_cpu`'s sources (13 TUs compiled with
`-DSTRATA_NATIVE_EXPERTS=1` and the CMake's per-file AVX-512/AVX2/AVX flags) + static ggml-sycl + the shim:

```
LINK_RC=0, 0 undefined references
-rwxr-xr-x 93653696  ds41_generate_cpuexp
```

So the S2 goal "the engine links against a SYCL ggml" is met for the configuration S3 actually needs
(`--experts cpu`), and the dense half runs on the SYCL backend. (Link only — nothing was executed.)

## 6. What this means for S3 (CPU fallback for GSQ)

**Does ggml's scheduler fall back to CPU for an unsupported op?** It can, but **this engine does not use it**, and the
backend it does use aborts:

1. The scheduler *would* fall back if it holds the CPU backend as the lowest-priority backend:
   `ggml/src/ggml-backend.cpp:1360-1387` (pass 3) assigns every still-unassigned — i.e. unsupported — node to the
   highest-priority backend that supports it (CPU supports it), then pass 5 splits the graph. The `GGML_ASSERT` at
   `ggml-backend.cpp:1442` (comment at 1470: "this can happen if there is no CPU fallback") fires only when **no**
   backend supports the node.
2. The engine builds **no scheduler**: `grep -rn ggml_backend_sched_new tools/v41/engine` → nothing. The dense half
   calls `ggml_backend_graph_compute(backend, gf)` directly (`ds41_dense.cpp:662`), which dispatches straight to
   `backend->iface.graph_compute` (`ggml-backend.cpp:595-604`).
3. On ggml-sycl that runs `ggml_sycl_compute_forward` per node; an op it does not implement hits `default: return
   false` (`ggml-sycl.cpp:6175-6176`) and the caller does `GGML_ASSERT(ok)` (`ggml-sycl.cpp:6529-6533`) — a **hard
   abort with "op not supported" logged**, not a CPU fallback.

Consequences:

* **S3 should be built and run as the CPU-expert flavour** (`--experts cpu`, `ds41_moe.cpp` without `DS4_MOE_CUDA`,
  `strata_kernels_cpu`): it links today (§5b). The GSQ expert arithmetic is then Strata's CPU kernels, so no GSQ type
  work in ggml is needed for S3 at all — which also means the plan's S4 line "port GSQ GEMV/MMQ into ggml-sycl" targets
  the wrong layer: the expert tier is `src/kernels/cuda/iq_kernels.cu` (`native_expert_grouped`, see
  `include/strata/kernels/iq_kernels.hpp:115` and `native_expert_grouped` at `src/kernels/cuda/iq_kernels.cu:3720`),
  not a ggml type. Porting it is a Strata-kernel port (S4), not a ggml-sycl quant port.
* **The GPU-expert flavour stays blocked** until those 10 kernels + `ds41_add_f32` have SYCL implementations. No amount
  of ggml work unblocks it.
* **Dense-half safety net for S3**: because an unsupported ggml op aborts the process, S3 should either (a) preflight
  the dense graph with `ggml_backend_supports_op(sycl_backend, node)` before `graph_compute` and report the offending
  nodes, or (b) switch the engine's dense path to `ggml_backend_sched` with `{sycl, cpu}` so the fallback in (1)
  applies. Option (a) is a few lines in `Ds41Dense::Impl::compute` and keeps S2's single-backend design.
* **Device selection for S3**: export `ONEAPI_DEVICE_SELECTOR=level_zero:0` in the run environment. The shim's own
  selector (`gpu_selector_v`, S1) and ggml's `dpct` device both then resolve to the Level Zero Arc; with the S2
  adoption the shim's context *is* ggml's, so even a mismatched enumeration can no longer put engine copies on a
  different device/context than the kernels (the iGPU is never selected because it is device 1+ and ggml's device 0
  is the Level Zero Arc).
