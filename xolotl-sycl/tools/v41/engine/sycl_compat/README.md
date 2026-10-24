# `sycl_compat/` — the CUDA-runtime shim for the DeepSeek-V4.1 engine (S1)

Stage S1 of [docs/xolotl-sycl-port-plan.md](../../../../docs/xolotl-sycl-port-plan.md): make the 8 real engine files
that call the CUDA runtime compile under `icpx -fsycl`, by presenting the CUDA names they use over
SYCL (USM + `sycl::queue` + `sycl::event`). The engine sources are **not modified**: the build puts this directory
first on the include path and defines `DS41_SYCL`, so their `#include <cuda_runtime.h>` resolves here. A CUDA build
does neither and keeps using NVIDIA's headers (the forwarding headers `#error` if they ever leak into one). This is
the same mechanism `include/strata/hip_compat/` uses for the HIP backend.

| | |
|---|---|
| header | `ds41_sycl_compat.hpp` — types, enums, `cudaDeviceProp`, declarations (SYCL-free) |
| implementation | `ds41_sycl_compat.cpp` — the only TU that includes `<sycl/sycl.hpp>` |
| forwarding headers | `cuda_runtime.h`, `cuda_runtime_api.h`, `cuda_profiler_api.h` |
| self-test | `test_ds41_sycl_compat.cpp` — 55 runtime checks on the card, no model |
| compile evidence | `compile_check.sh` — one `icpx -fsycl -c` per engine file |
| build wiring | `v41_engine_sycl.cmake` — the `DS41_SYCL` macro + include order, for S2 |

## Model

* one GPU (`sycl::gpu_selector_v`, i.e. **the same selector ggml-sycl uses**) and **one** `sycl::context` shared by
  every queue, so USM allocations are valid everywhere the engine uses them;
* each `cudaStream_t` is its own **in-order** `sycl::queue` (per-stream ordering preserved, inter-stream independence
  preserved); `cudaStream_t == nullptr` is CUDA's legacy default stream → its own queue, but *synchronising* it
  synchronises every shim queue;
* each `cudaEvent_t` is a `sycl::event` + a host timestamp; queues are created with `enable_profiling`, so
  `cudaEventElapsedTime` uses device timestamps where the backend provides them;
* USM allocations are tracked by base address **and range**, because the engine also passes pageable memory
  (`std::vector::data()`, its `mmap`ed arena) to the copy API. A pointer the shim did not allocate is resolved by
  `sycl::get_pointer_type` (the authority on USM) and the answer is cached back into the same map, because the engine
  also receives device pointers from paths that never call the shim - the VRAM expert cache arena
  (`ExpertCache::open_sized` -> `sycl::malloc_device`), ggml's own USM, the dpct-migrated TUs' buffers. Pageable host
  memory is not USM and still takes the staged path (see `cudaMemcpyAsync` below).

## Diagnostics (both env-gated; inert when unset)

| variable | effect |
|---|---|
| `DS41_SHIM_DBG=1` | one line per pointer-classification miss ("not in the shim map; SYCL says usm_device -> device, cached") and per staged copy (addresses, byte count) |
| `DS41_SHIM_CLASSIFY=map` | **diagnostic only**: restore the pre-S4-step-4 map-only classification, i.e. reproduce the expert-PCIe `cudaErrorInvalidValue` abort on this build. Used as the A/B in the step-4 report |

A refused copy always prints its own one-line diagnosis (both addresses, byte count, the kind each classified as,
and the raw `sycl::get_pointer_type` answer), whether or not `DS41_SHIM_DBG` is set: the engine turns the returned
`cudaErrorInvalidValue` into `ds4_moe: pcie dma: invalid argument` + `abort()`, so the shim is the last place that can
still name the pointer.

## API table (every CUDA symbol the 8 engine files reference — 46 APIs)

`U` = used, `R` = refused on purpose. "difference: none" means same observable behaviour for this engine.

| CUDA API | SYCL implementation | semantic difference |
|---|---|---|
| `cudaMalloc` | `sycl::malloc_device(n, dev, ctx)` | none |
| `cudaFree` | `sycl::free(p, ctx)`; double free → `cudaErrorInvalidValue` | none |
| `cudaMallocHost` | `sycl::malloc_host(n, ctx)` | none (USM host is what CUDA calls pinned+portable) |
| `cudaHostAlloc` (+`Portable`/`Mapped`) | same as `cudaMallocHost`; flags ignored | none (every USM host allocation is already portable and device-mapped) |
| `cudaFreeHost` | `sycl::free(p, ctx)` | none |
| `cudaHostRegister` / `cudaHostUnregister` | **no-op** that records the range | **DIFFERENCE**: no way to pin an existing range in SYCL. The engine's registered pages (72 GiB `mmap` arena, remote-helper shm) stay **not device-reachable**; copies to/from them take the staged path below |
| `cudaHostGetDevicePointer` | returns the host address | DIFFERENCE only for a `cudaHostRegister`ed range (there is no device address); for USM host memory it is exact |
| `cudaMemcpy` (sync) | enqueue on the default queue + `queue::wait()` | none observable (CUDA's is ordered with the default stream too) |
| `cudaMemcpyAsync` H2D ≤D2H, D2D | `queue::memcpy` on that stream's queue | none when both sides are USM (the map lookup, then `sycl::get_pointer_type` for anything the shim did not allocate) |
| `cudaMemcpyAsync` where one side is **pageable/registered** host memory | staged through a per-stream USM host buffer: `memcpy` → `queue::memcpy`, then wait | DIFFERENCE: synchronous, and one extra copy (exactly what NVIDIA's driver does for pageable memory, but the registered-arena case loses direct DMA) |
| `cudaMemcpyKind` (`HostToDevice`/`DeviceToHost`/`DeviceToDevice`/`HostToHost`) | honoured directly (`cudaMemcpyDefault` infers from the tracked ranges) | none |
| `cudaMemset` / `cudaMemsetAsync` | `queue::memset` | none (requires USM; a non-USM target is `cudaErrorInvalidValue`) |
| `cudaStreamCreate` / `cudaStreamCreateWithFlags` | new in-order `sycl::queue` from the shared context; flags ignored | `cudaStreamNonBlocking`: all shim queues are non-blocking w.r.t. each other anyway |
| `cudaStreamDestroy` | `queue::wait()` then delete | none |
| `cudaStreamSynchronize(s)` | `queue::wait()` | none |
| `cudaStreamSynchronize(nullptr)` | waits **all** shim queues | DIFFERENCE: a superset of CUDA's legacy-stream rule (conservative: nothing is missed, some overlap is lost) |
| `cudaStreamQuery` | status of the stream's newest submission | none |
| `cudaStreamWaitEvent` | `queue::ext_oneapi_submit_barrier({ev})` (fallback: empty dependent kernel) | none |
| `cudaEventCreate` / `cudaEventCreateWithFlags` | host object; `cudaEventDisableTiming` accepted and ignored | DIFFERENCE: timing stays available even when CUDA would forbid it (harmless) |
| `cudaEventRecord` | `ext_oneapi_submit_barrier()` on the stream + host timestamp | none (an empty ordered command marks the stream position) |
| `cudaEventSynchronize` / `cudaEventQuery` | `sycl::event::wait()` / `command_execution_status` | none |
| `cudaEventElapsedTime` | device timestamps (`event_profiling::command_start/end`) of the two barrier events; **host** timestamps as fallback; `cudaErrorNotReady` while an event is incomplete | DIFFERENCE in the fallback: host *enqueue* time, i.e. it under-reports elapsed GPU work. With profiling (this backend) it is GPU time |
| `cudaEventDestroy` | delete | none (in-flight work keeps running) |
| `cudaDeviceSynchronize` | `wait()` on every shim queue | none |
| `cudaGetDeviceCount` | 1 | single GPU (as on this machine) |
| `cudaGetDevice` / `cudaSetDevice` | always 0; any other ordinal → `cudaErrorInvalidDevice` | DIFFERENCE: ordinal `n>0` is a hard error instead of running on the wrong device |
| `cudaGetDeviceProperties` | `name`, `major/minor`=**10/0**, `totalGlobalMem`, `multiProcessorCount`, `maxThreadsPerBlock`, `sharedMemPerBlock`, `l2CacheSize`, `warpSize`=32, `managedMemory`, `canMapHostMemory`, `concurrentKernels`, `asyncEngineCount`; rest zero | DIFFERENCE: Intel has no CUDA compute capability. `10.0` keeps `strata::cuda_cc_below_mmq_tensor_cores(100)` false, so the CC policy does not disable chunk-MMQ on a non-NVIDIA card. Only `name`/`major`/`minor` are read by the engine |
| `cudaMemGetInfo` | `info::device::global_mem_size` + Intel `ext::intel::info::device::free_memory`; fallback = total − live shim device bytes | DIFFERENCE: the fallback ignores other processes (the real free-memory query is used on this Level Zero stack, verified: 30.0 of 31.9 GiB) |
| `cudaDeviceCanAccessPeer` | `*can = 0`, success | **R** single GPU: the engine's peer probe fails and it takes its pinned-host handoff |
| `cudaDeviceEnablePeerAccess` | `cudaErrorPeerAccessUnsupported` | **R** multi-GPU/layer-split stays off (deliberate, not faked) |
| `cudaDeviceDisablePeerAccess` | success, no-op | none (nothing was enabled) |
| `cudaMemcpyPeer` / `cudaMemcpyPeerAsync` | `cudaErrorPeerAccessUnsupported` | **R** unreachable on one GPU (only the peer path calls it); a real error rather than a silent local copy |
| `cudaStreamBeginCapture` / `cudaStreamEndCapture` | `cudaErrorStreamCaptureUnsupported` | **R** CUDA Graph, see below |
| `cudaGraphInstantiate` / `cudaGraphLaunch` | `cudaErrorStreamCaptureUnsupported` | **R** unreachable while capture is refused |
| `cudaGraphDestroy` / `cudaGraphExecDestroy` | delete the (never created) object | none |
| `cudaProfilerStart` / `cudaProfilerStop` | success, no-op | DIFFERENCE: `nsys --capture-range=cudaProfilerApi` has no meaning; `STRATA_PROFILE_DECODE=1` still works as a switch |
| `cudaGetLastError` | thread-local last error, cleared on read | none |
| `cudaGetErrorString` | string table for the codes above | different wording |

Also declared so the engine compiles, though these 8 files do not call them: nothing else is provided. If a future
file needs a CUDA API outside this list it will fail to compile — deliberately, rather than getting a plausible
wrong implementation.

## CUDA Graph: refused, with the caller able to see it

`ds41_moe.cpp:2746 tier_graph_capture()` begins a capture on a scratch stream and then calls
`strata::kernels::native_expert_grouped(..., cs, 0)`. Those kernels are **not** submitted through this shim (the
stream is an opaque `void *` at the kernel layer, and in S2 the ggml work goes to ggml's queue), so a "record and
replay" shim would record an **empty** stream and replay nothing: the graph would silently compute nothing and the
model would emit wrong tokens with no error. Therefore capture fails visibly:

* `cudaStreamBeginCapture` → `cudaErrorStreamCaptureUnsupported`;
* `tier_graph_capture()` returns false and the engine prints
  `STRATA_TIER_GRAPH: CUDA refused the capture; keeping the per-class launches` and keeps the per-class path — the
  same path the CUDA build takes on any capture failure. `STRATA_TIER_GRAPH` is opt-in, so a default run never asks.

**Risk**: the per-class path costs the six host calls per class launch that the graph folded into one (the CUDA
build's measured win). Recovering it needs graph nodes built inside the SYCL kernel layer
(`sycl_ext_oneapi_graph`, or a submission-recording mechanism there), not in this shim.

## What the engine will do if it reaches a refused call

| path | behaviour under the shim |
|---|---|
| multi-GPU layer split (`--stage-devices`, `--head-device`, `STRATA_DS41_REMOTE`) | `cudaDeviceCanAccessPeer`=0 / `cudaDeviceEnablePeerAccess` fails → `ds41_handoff_mode()` = pinned-host; a second ordinal is `cudaErrorInvalidDevice`, so split/remote setups fail loudly instead of running on device 0 twice |
| `STRATA_TIER_GRAPH=1` | capture refused → per-class launches (logged) |
| `STRATA_PROFILE_DECODE=1` | no-op profiler; the run is unaffected |
| peer copy of the shim's registered host ranges | staged, synchronous, correct |

## S1 status (compiles, no model, no ggml link)

`compile_check.sh` (icpx 2026.2.0, C++20, `-DDS41_SYCL=1 -DSTRATA_ENABLE_CUDA=1 -DDS4_MOE_CUDA=1 -DDS41_HAVE_CUDA=1
-DDS41_POOL_CUDA=1 -DDS41_HAVE_DSPARK=1`): **8/8 engine files + the shim compile**. Only two undefined symbols remain
across all 8 objects that this shim does not provide, and both are ggml-CUDA *backend* functions, not CUDA runtime:
`ggml_backend_cuda_init` (×2) and `ggml_backend_cuda_register_host_buffer` (×1). See the S2 notes in the port plan.

`test_ds41_sycl_compat.cpp` passes 55/55 on the Arc Pro B70 (Level Zero, 31.9 GiB): allocations, staged and direct
copies bit-exact, memset, two streams with a cross-stream event, device-timed events, the peer and graph refusals,
error-state semantics, and full teardown.

## S2 status (2026-10-24): the ggml link

Full report: `docs/xolotl-sycl-s2-report.md`. What changed here in S2:

* `ds41_sycl_adopt_ggml_queue(device)` (declared here, implemented in the `.cpp`): the engine calls it right after
  `ggml_backend_sycl_init`, so the shim's default stream (`cudaStream_t` nullptr) becomes a handle copy of
  **ggml-sycl's own in-order queue**, and — before any allocation — the shim's USM context/device become ggml's. That
  is the ordered-with-ggml fix S1 flagged as a semantic gap, and it also removes the "which of the two Arc views"
  question from the shim: it runs on whatever device ggml's queue belongs to.
* The accessor it needs is a two-function patch to the ggml tree, committed as
  `tools/xolotl/ggml/xolotl-sycl-queue.patch` (`ggml_backend_sycl_get_default_queue`); the shim declares it **weak**, so
  this self-test still links without ggml (`cudaErrorNotSupported` if adoption is attempted there).
* Engine-side S2 edits are all inside `DS41_SYCL`: `ggml_backend_cuda_init` → `ggml_backend_sycl_init` (2 sites), the 6
  `ggml_backend_cuda_register_host_buffer` calls compiled out (no SYCL counterpart), `ggml-cuda.h` → `ggml-sycl.h`.

Link status: the `ds41_moe_cpu` flavour (`--experts cpu`, no `DS4_MOE_CUDA`) **links** against static ggml-sycl with
zero undefined symbols. The GPU-expert flavour is blocked by the routed-expert tier's own CUDA kernels
(`strata::kernels::native_expert_grouped` & co. in `src/kernels/cuda/iq_kernels.cu`), not by ggml — those calls are in
`ds41_moe.cpp`, outside this shim. S3 therefore runs with CPU experts; see the report §6 for the ggml-scheduler answer
(the engine does not use the scheduler, and ggml-sycl aborts on an op it does not implement).

## Notes for S2 (linking ggml with `GGML_SYCL=ON`)

1. `ggml_backend_cuda_init` → `ggml_backend_sycl_init(int device)` (2 call sites: `ds41_pipeline.cpp:34`,
   `ds41_generate.cpp:440`). Signature matches; it is a rename, not a port.
2. `ggml_backend_cuda_register_host_buffer` (5 call sites in `ds41_dense.cpp`) has **no SYCL counterpart** —
   `ggml_backend_sycl_register_host_buffer` is commented out in `ggml/include/ggml-sycl.h:50`. Those calls must be
   compiled out; the same consequence as the no-op `cudaHostRegister` (host staging, not DMA).
3. Device + context must match ggml's: ggml-sycl submits to `dpct::get_device(device).default_queue()`
   (`dpct/helper.hpp:731`, an **in-order** queue) in a context of its own. Pointing the shim's *default stream* at
   that queue is what restores "engine copies and ggml work are ordered"; today the shim has its own context, which
   the Level Zero stack happens to accept for USM (verified) but SYCL does not guarantee. The clean fix is a tiny
   ggml-sycl accessor (`ggml_backend_sycl_get_queue(ggml_backend_t)`).
4. `sycl::gpu_selector_v` is required, not optional: this box enumerates the Arc **twice** (OpenCL and Level Zero)
   plus the UHD 770 iGPU, and `sycl::device::get_devices(gpu)[0]` is the OpenCL view — a different device *and*
   context from the one ggml-sycl uses. (`ONEAPI_DEVICE_SELECTOR` is how the working Qwen stack pins this.)
