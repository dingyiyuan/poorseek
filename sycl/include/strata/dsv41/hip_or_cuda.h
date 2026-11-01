// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// include/strata/dsv41/hip_or_cuda.h - the runtime the dsv41 *tests* need, whichever backend the build has.
//
// The engine's own sources take their runtime from the backend's headers (ds4_*.cu compile as HIP or as CUDA).  The tests
// are plain .cpp files that launch kernels and copy buffers, and they used to include <hip/hip_runtime.h> unconditionally,
// which no CUDA toolkit ships: every CUDA box had to pass a shim on -I to build them (found on l620 and z440, 2026-10-23).
// This header is that shim, in the tree: HIP names on a HIP build, mapped onto the CUDA runtime otherwise.
//
// [Intel port] A third branch: HIP's names on the SYCL runtime, so the engine's own parity tests build and run against the
// ported kernels without touching the test sources.  Two things differ from CUDA, and both are spelled out where they are
// used: an allocation is USM on the in-order queue (no kind), and an event is timed on the host after its queue drained -
// `dpct::sync_barrier` + `wait` - because the profiling information needs a queue opened with profiling enabled.  A timed
// region therefore includes its barriers: the numbers a --bench run prints here are wall clock, not pure kernel time.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <chrono>
#include <map>
#include <mutex>

using hipError_t = int;   // SYCL reports through exceptions: there is no error code to carry, 0 is success
using hipEvent_t = dpct::event_ptr;
using hipStream_t = dpct::queue_ptr;
constexpr hipError_t hipSuccess = 0;
constexpr int hipMemcpyHostToDevice = 0;
constexpr int hipMemcpyDeviceToHost = 1;

inline hipError_t hipMalloc(void** p, size_t n) { *p = sycl::malloc_device(std::max<size_t>(1, n), dpct::get_in_order_queue()); return 0; }
template <typename T>
inline hipError_t hipMalloc(T** p, size_t n) { return hipMalloc(reinterpret_cast<void**>(p), n); }
inline hipError_t hipFree(void* p) { sycl::free(p, dpct::get_in_order_queue()); return 0; }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t n, int /*kind: USM needs none*/) {
    dpct::get_in_order_queue().memcpy(dst, src, std::max<size_t>(1, n)).wait();
    return 0;
}
template <typename T>
inline hipError_t hipMemcpy(T* dst, const T* src, size_t n, int kind) {
    // n is BYTES here, as in upstream's CUDA shim (the templates only take typed pointers): multiplying by sizeof(T)
    // over-copied by 2-4x, which read past a device buffer and took the engine down on the first launch.
    return hipMemcpy(static_cast<void*>(dst), static_cast<const void*>(src), n, kind);
}
inline hipError_t hipMemset(void* p, int v, size_t n) { dpct::get_in_order_queue().memset(p, v, std::max<size_t>(1, n)).wait(); return 0; }
inline hipError_t hipDeviceSynchronize() { dpct::get_in_order_queue().wait(); return 0; }
inline hipError_t hipSetDevice(int) { return 0; }   // ONEAPI_DEVICE_SELECTOR picks the card; the tests run on what it names
inline const char* hipGetErrorString(hipError_t) { return "sycl"; }

namespace ds4_port_events {
inline std::mutex& mu() { static std::mutex m; return m; }
inline std::map<const sycl::event*, std::chrono::steady_clock::time_point>& table() {
    static std::map<const sycl::event*, std::chrono::steady_clock::time_point> t;
    return t;
}
}  // namespace ds4_port_events

inline hipError_t hipEventCreate(hipEvent_t* e) { *e = new sycl::event(); return 0; }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t s = nullptr) {
    dpct::queue_ptr q = s ? s : &dpct::get_in_order_queue();
    dpct::sync_barrier(e, q);
    e->wait();                     // (the host stamp below must come after the queue's work)
    std::lock_guard<std::mutex> lk(ds4_port_events::mu());
    ds4_port_events::table()[e] = std::chrono::steady_clock::now();
    return 0;
}
inline hipError_t hipEventSynchronize(hipEvent_t e) { e->wait(); return 0; }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t a, hipEvent_t b) {
    std::lock_guard<std::mutex> lk(ds4_port_events::mu());
    *ms = std::chrono::duration<float, std::milli>(ds4_port_events::table()[b] - ds4_port_events::table()[a]).count();
    return 0;
}
#else
#include <cuda_runtime.h>

using hipError_t = cudaError_t;
using hipEvent_t = cudaEvent_t;
using hipStream_t = cudaStream_t;
constexpr hipError_t hipSuccess = cudaSuccess;
constexpr int hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr int hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;

inline hipError_t hipMalloc(void** p, size_t n) { return cudaMalloc(p, n); }
template <typename T>
inline hipError_t hipMalloc(T** p, size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
inline hipError_t hipFree(void* p) { return cudaFree(p); }
inline hipError_t hipMemcpy(void* dst, const void* src, size_t n, int kind) {
    return cudaMemcpy(dst, src, n, static_cast<cudaMemcpyKind>(kind));
}
template <typename T>
inline hipError_t hipMemcpy(T* dst, const T* src, size_t n, int kind) {
    return cudaMemcpy(static_cast<void*>(dst), static_cast<const void*>(src), n, static_cast<cudaMemcpyKind>(kind));
}
inline hipError_t hipMemset(void* p, int v, size_t n) { return cudaMemset(p, v, n); }
inline hipError_t hipDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline hipError_t hipSetDevice(int d) { return cudaSetDevice(d); }
inline const char* hipGetErrorString(hipError_t e) { return cudaGetErrorString(e); }
inline hipError_t hipEventCreate(hipEvent_t* e) { return cudaEventCreate(e); }
inline hipError_t hipEventRecord(hipEvent_t e, hipStream_t s = nullptr) { return cudaEventRecord(e, s); }
inline hipError_t hipEventSynchronize(hipEvent_t e) { return cudaEventSynchronize(e); }
inline hipError_t hipEventElapsedTime(float* ms, hipEvent_t a, hipEvent_t b) { return cudaEventElapsedTime(ms, a, b); }
#endif
