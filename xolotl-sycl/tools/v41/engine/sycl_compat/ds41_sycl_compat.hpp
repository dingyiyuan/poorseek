// tools/v41/engine/sycl_compat/ds41_sycl_compat.hpp - the CUDA-runtime surface the DeepSeek-V4.1 engine uses,
// declared for an icpx/-fsycl build and implemented over SYCL (USM + in-order queues + events) in
// ds41_sycl_compat.cpp.  S1 of docs/xolotl-sycl-port-plan.md.
//
// HOW IT GETS INCLUDED
//   The engine sources say `#include <cuda_runtime.h>` (8 real engine files, 46 distinct APIs: memory, streams,
//   events, peer access, CUDA Graph, profiler).  In a SYCL build the build system puts this directory FIRST on the
//   include path and defines DS41_SYCL, so the forwarding headers next to this one (cuda_runtime.h,
//   cuda_runtime_api.h, cuda_profiler_api.h) resolve to the shim.  A CUDA build never adds this directory and never
//   defines DS41_SYCL: it keeps compiling against the NVIDIA headers and NOTHING in this directory is used.
//   Same mechanism (and the same "directory added to the include path") as include/strata/hip_compat/ for HIP.
//
// WHAT IS DELIBERATELY NOT EMULATED
//   Multi-GPU (peer access, cudaMemcpyPeer*): this shim answers "no peer" and "not supported".  A single-GPU box
//   therefore takes the engine's pinned-host handoff path and the layer-split/remote-device paths stay off.
//   CUDA Graph: no exact SYCL equivalent - capture is REFUSED with a visible error (see cudaStreamBeginCapture).
//   Everything else is real: device/host USM, per-stream in-order queues, events, copies, memset, device info.
//
// SEMANTIC DIFFERENCES (also in sycl_compat/README.md, one line per API)
//   * cudaHostRegister/Unregister: NO-OP that only records the range.  USM gives no way to pin an arbitrary range,
//     so host pages the engine believes are DMA-able (the SysV shm of the remote helper, a conversation pool that
//     was registered after the fact) are NOT reachable by the device.  Copies from/to them are staged through a USM
//     host buffer and are therefore synchronous + an extra copy (CUDA would DMA them directly).
//     The expert arena no longer takes that path: it is ALLOCATED as USM host through ds41_sycl_host_alloc() below
//     (strata::core::PinnedArena), so a copy from it is one direct async DMA.  The bound is the device's
//     max_mem_alloc_size, hence the pieces: ds41_sycl_host_alloc_max().
//   * cudaEventElapsedTime: device timestamps from SYCL profiling info when the backend provides them, otherwise
//     host wall-clock at cudaEventRecord time (precision loss; see the .cpp).
//   * cudaStreamSynchronize(nullptr) (CUDA's legacy default stream) waits for EVERY shim stream, not just one.
//   * cudaDeviceProp.major/minor: Intel has no CUDA compute capability.  They are reported as 10/0 so the
//     compute-capability policy (ds41_cuda_policy.cpp) does not disable chunk-MMQ on a non-NVIDIA card.
//
// The header itself is SYCL-free on purpose (opaque handles + POD only): the 8 engine TUs keep compiling as plain
// C++ TUs and only ds41_sycl_compat.cpp sees <sycl/sycl.hpp>.
#pragma once

#if !defined(DS41_SYCL)
#error "sycl_compat headers are for the SYCL build only: define DS41_SYCL and put tools/v41/engine/sycl_compat first on the include path.  CUDA builds must use NVIDIA's <cuda_runtime.h>."
#endif

#include <cstddef>
#include <cstdint>
#include <string>

// ---------------------------------------------------------------- opaque handles
// CUDA's cudaStream_t/cudaEvent_t are pointers to incomplete types, and the engine relies on that: it casts
// `(cudaStream_t) 0` to mean "the default stream", tests handles with `if (s)` / `if (e)`, and stores them in
// void* fields (ds41_pipeline.hpp: `void * cuda_stream`).  Pointers keep all of that source-compatible.
struct Ds41SyclStream;
struct Ds41SyclEvent;
struct Ds41SyclGraph;
struct Ds41SyclGraphExec;
typedef Ds41SyclStream * cudaStream_t;
typedef Ds41SyclEvent * cudaEvent_t;
typedef Ds41SyclGraph * cudaGraph_t;
typedef Ds41SyclGraphExec * cudaGraphExec_t;

/// The sycl::queue behind a stream handle, as void* (null for the null handle): what a Strata kernel launcher's
/// `q_of(stream)` needs, because the handle is NOT a sycl::queue* (it is a Ds41SyclStream with the queue inside).
/// S4 step 2: the engine's GPU-expert tier passes its cudaStream_t straight into strata::kernels::* launchers.
void * ds41_sycl_queue_of(void * stream);
/// The DEFAULT stream's queue (ggml's, after adoption), for a caller that must enqueue on it without a stream handle.
void * ds41_sycl_default_queue();

// ---------------------------------------------------------------- device-reachable host memory (S16)
// The allocator `strata::core::PinnedArena` uses when this shim is in the link (it declares these weak): host memory
// that is USM in the context the copies run on, so `cudaMemcpyAsync` from it is ONE direct async DMA instead of a
// synchronous host memcpy into a stage slot plus a DMA.  `max` is the largest single request that will be granted
// (the device's max_mem_alloc_size less a margin); 0 = no allocator in this link.  `alloc` sets *why on failure.
// The memory must not be freed or reused while a copy from it is outstanding.
extern "C" size_t ds41_sycl_host_alloc_max();
extern "C" void * ds41_sycl_host_alloc(size_t bytes, size_t align, const char ** why);
extern "C" void ds41_sycl_host_free(void * p);

// ---------------------------------------------------------------- error codes
// Same shape as NVIDIA's: an unscoped enum (so `cudaSuccess` etc. are usable unqualified and comparable) with
// cudaError_t as its alias.  Numeric values are CUDA's documented ones where the engine could ever see them; the
// engine only ever compares symbolically (ck(e, ...) / `e != cudaSuccess`).
enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorInvalidDevice = 10,
    cudaErrorNotReady = 34,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorNotSupported = 801,
};
typedef enum cudaError cudaError_t;

// ---------------------------------------------------------------- enums and flags
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};

// Flag bit values are CUDA's.  cudaStreamNonBlocking / cudaEventDisableTiming / cudaHostAllocPortable are accepted
// and ignored where SYCL has nothing to say about them (see README.md).
constexpr unsigned int cudaStreamDefault = 0x00u;
constexpr unsigned int cudaStreamNonBlocking = 0x01u;
constexpr unsigned int cudaEventDefault = 0x00u;
constexpr unsigned int cudaEventBlockingSync = 0x01u;
constexpr unsigned int cudaEventDisableTiming = 0x02u;
constexpr unsigned int cudaHostAllocDefault = 0x00u;
constexpr unsigned int cudaHostAllocPortable = 0x01u;
constexpr unsigned int cudaHostAllocMapped = 0x02u;
constexpr unsigned int cudaHostRegisterDefault = 0x00u;
constexpr unsigned int cudaHostRegisterPortable = 0x01u;
constexpr unsigned int cudaHostRegisterMapped = 0x02u;

// ---------------------------------------------------------------- cudaDeviceProp
// The engine reads exactly three fields (grep: `prop.` in the 8 engine files):
//   prop.name  - ds41_helper.cpp:526  prints "requesting expert tier on %s (device %d)"
//   prop.major - ds41_cuda_policy.cpp:31  strata::cuda_cc_encode(prop.major, prop.minor)
//   prop.minor - ds41_cuda_policy.cpp:31  (same call)
// Everything else is filled from SYCL device info / sane constants so a future reader of other fields gets
// something defensible instead of zeroes; the .cpp comments each field with where its value comes from.
struct cudaDeviceProp {
    char name[256];                 ///< sycl::info::device::name (truncated, NUL-terminated)
    int major;                      ///< NOT a CUDA capability: Intel has none.  10 (see .cpp)
    int minor;                      ///< 0
    int multiProcessorCount;        ///< sycl::info::device::max_compute_units
    int warpSize;                   ///< 32 (Intel Xe subgroup-friendly default; informational only)
    int maxThreadsPerBlock;         ///< sycl::info::device::max_work_group_size
    int maxThreadsPerMultiProcessor;///< 0 = unknown
    int clockRate;                  ///< 0 = unknown (kHz in CUDA)
    int memoryClockRate;            ///< 0 = unknown
    int memoryBusWidth;             ///< 0 = unknown
    int regsPerBlock;               ///< 0 = unknown
    int regsPerMultiprocessor;      ///< 0 = unknown
    int sharedMemPerBlock;          ///< sycl::info::device::local_mem_size
    int sharedMemPerMultiprocessor; ///< sycl::info::device::global_mem_size is wrong for this; 0 = unknown
    int l2CacheSize;                ///< sycl::info::device::global_mem_cache_size (may be 0)
    size_t totalGlobalMem;          ///< sycl::info::device::global_mem_size
    size_t sharedMemPerBlockOptin;  ///< sycl::info::device::local_mem_size
    int integrated;                 ///< sycl::aspect::host_unified_memory
    int unifiedAddressing;          ///< 0: USM host+device is enough, host/device addresses are NOT unified
    int managedMemory;              ///< sycl::aspect::usm_shared_allocations (USM shared ~ cudaMallocManaged)
    int concurrentKernels;          ///< 1 (only a hint; the shim maps one stream to one in-order queue)
    int asyncEngineCount;           ///< 1
    int canMapHostMemory;           ///< 1: USM host allocations are device-accessible
    int computeMode;                ///< 0 = cudaComputeModeDefault
    // Padding so a future CUDA header swap cannot change sizeof() under the engine's feet.  Not read by anything.
    int reserved[8];
};

// ---------------------------------------------------------------- API, as the engine calls it
// Allocation.  Templates so both `void **` (what the engine writes: `cudaMalloc((void**) &d[k], n)`) and a typed
// pointer (`cudaHostAlloc((void**) &r.h_meta, n, ...)`) compile exactly as they do against NVIDIA's headers; the
// CUDA versions are (void**) macros/templates in the same spirit.
namespace ds41_sycl_compat {
cudaError_t malloc_device(void ** p, size_t bytes);
cudaError_t malloc_host(void ** p, size_t bytes);
cudaError_t free_device(void * p);
cudaError_t free_host(void * p);
cudaError_t host_register(const void * p, size_t bytes, unsigned int flags);
cudaError_t host_unregister(const void * p);
cudaError_t host_get_device_pointer(void ** dev_ptr, void * host_ptr, unsigned int flags);
}  // namespace ds41_sycl_compat

template <class T>
inline cudaError_t cudaMalloc(T ** p, size_t bytes) {
    return ds41_sycl_compat::malloc_device(reinterpret_cast<void **>(p), bytes);
}
template <class T>
inline cudaError_t cudaMallocHost(T ** p, size_t bytes) {
    return ds41_sycl_compat::malloc_host(reinterpret_cast<void **>(p), bytes);
}
template <class T>
inline cudaError_t cudaHostAlloc(T ** p, size_t bytes, unsigned int flags) {
    (void) flags;   // cudaHostAllocPortable/Mapped: every USM host allocation is already portable+mapped here
    return ds41_sycl_compat::malloc_host(reinterpret_cast<void **>(p), bytes);
}
inline cudaError_t cudaFree(void * p) { return ds41_sycl_compat::free_device(p); }
inline cudaError_t cudaFreeHost(void * p) { return ds41_sycl_compat::free_host(p); }
inline cudaError_t cudaHostRegister(void * p, size_t bytes, unsigned int flags) {
    return ds41_sycl_compat::host_register(p, bytes, flags);
}
inline cudaError_t cudaHostUnregister(void * p) { return ds41_sycl_compat::host_unregister(p); }
inline cudaError_t cudaHostGetDevicePointer(void ** dev_ptr, void * host_ptr, unsigned int flags) {
    return ds41_sycl_compat::host_get_device_pointer(dev_ptr, host_ptr, flags);
}

// Copies / memset
cudaError_t cudaMemcpy(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind,
                            cudaStream_t stream = nullptr);
cudaError_t cudaMemset(void * dst, int value, size_t bytes);
cudaError_t cudaMemsetAsync(void * dst, int value, size_t bytes, cudaStream_t stream = nullptr);

// Streams
cudaError_t cudaStreamCreate(cudaStream_t * stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

// Events
cudaError_t cudaEventCreate(cudaEvent_t * event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end);

// Device
cudaError_t cudaGetDeviceCount(int * count);
cudaError_t cudaGetDevice(int * device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaMemGetInfo(size_t * free_bytes, size_t * total_bytes);

// Multi-GPU: this shim refuses peer access instead of pretending (single GPU here, see README.md)
cudaError_t cudaDeviceCanAccessPeer(int * can_access, int device, int peer_device);
cudaError_t cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peer_device);
cudaError_t cudaMemcpyPeer(void * dst, int dst_device, const void * src, int src_device, size_t bytes);
cudaError_t cudaMemcpyPeerAsync(void * dst, int dst_device, const void * src, int src_device, size_t bytes,
                                cudaStream_t stream = nullptr);

// CUDA Graph: capture is refused, see the .cpp decision note
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t * exec, cudaGraph_t graph, void * error_node, char * log,
                                 size_t log_size);
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);

// Errors / profiler
cudaError_t cudaGetLastError(void);
const char * cudaGetErrorString(cudaError_t error);
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);

// ---------------------------------------------------------------- SYCL-only introspection (not CUDA API)
// S1 evidence + S2 wiring aid: what the shim picked and how much it has handed out.  Not called by the engine.
// ---------------------------------------------------------------- S2: ggml-sycl queue adoption (not CUDA API)
// Call right after ggml_backend_sycl_init(device).  The shim's default stream (`cudaStream_t` nullptr) then becomes a
// handle to ggml-sycl's own in-order default queue, so copies the engine submits on CUDA's "null stream" are ordered
// with ggml's kernels (without this they merely *happen* to work: cross-context USM, which SYCL does not promise).
// While nothing has been allocated yet, the shim's USM context/device also become ggml's.  Returns
// cudaErrorNotSupported when the shim was linked without ggml (the S1 self-test), cudaErrorInvalidDevice when
// ggml has no such device, cudaSuccess on adoption.  Idempotent.
cudaError_t ds41_sycl_adopt_ggml_queue(int device);

namespace ds41_sycl_compat {
struct info {
    std::string device_name;
    std::string backend;
    size_t device_bytes_live;   ///< sum of live cudaMalloc'd bytes
    size_t host_bytes_live;     ///< sum of live cudaMallocHost/cudaHostAlloc bytes
    size_t registered_bytes;    ///< bytes cudaHostRegister'ed (no-op'd, recorded)
    int streams_live;
};
info get_info();
}  // namespace ds41_sycl_compat
