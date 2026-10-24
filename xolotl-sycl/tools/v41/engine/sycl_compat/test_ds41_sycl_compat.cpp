// tools/v41/engine/sycl_compat/test_ds41_sycl_compat.cpp - runtime self-test for the CUDA->SYCL shim.
//
// S1 was "make the 8 engine files compile"; this proves the shim also BEHAVES like the CUDA runtime it replaces, on
// the real card, WITHOUT a model: allocations, staged and direct copies, memset, per-stream queues, events with
// device timing, the refusal of peer access and of CUDA Graph, and error-state semantics.
//
// build+run (no ggml, no model, ~2 s):
//   source /opt/intel/oneapi/setvars.sh --force
//   cd tools/v41/engine/sycl_compat
//   icpx -fsycl -std=c++20 -O2 -DDS41_SYCL=1 -I. test_ds41_sycl_compat.cpp ds41_sycl_compat.cpp -o /tmp/t && /tmp/t
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, what)                                                             \
    do {                                                                              \
        if (cond) {                                                                   \
            std::printf("PASS  %s\n", what);                                          \
        } else {                                                                      \
            std::printf("FAIL  %s\n", what);                                          \
            ++g_fail;                                                                 \
        }                                                                             \
    } while (0)

static bool all_bytes(const void * p, size_t n, unsigned char v) {
    const unsigned char * b = (const unsigned char *) p;
    for (size_t i = 0; i < n; ++i)
        if (b[i] != v) return false;
    return true;
}

int main() {
    // ---------------------------------------------------------------- device
    int ndev = 0;
    CHECK(cudaGetDeviceCount(&ndev) == cudaSuccess && ndev == 1, "cudaGetDeviceCount == 1");
    CHECK(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0)");
    CHECK(cudaSetDevice(1) == cudaErrorInvalidDevice, "cudaSetDevice(1) -> invalid device");
    CHECK(cudaGetLastError() == cudaErrorInvalidDevice, "cudaGetLastError returns the stored error");
    CHECK(cudaGetLastError() == cudaSuccess, "cudaGetLastError clears it");

    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "cudaGetDeviceProperties");
    std::printf("      device: %s, reported cc %d.%d, %.1f GiB\n", prop.name, prop.major, prop.minor,
                (double) prop.totalGlobalMem / 1073741824.0);
    CHECK(prop.name[0] != '\0' && prop.totalGlobalMem > 0, "prop.name / prop.totalGlobalMem filled");
    CHECK(prop.major == 10 && prop.minor == 0, "prop.major/minor = 10.0 (MMQ policy stays enabled)");

    size_t fr = 0, tot = 0;
    CHECK(cudaMemGetInfo(&fr, &tot) == cudaSuccess && tot > 0 && fr > 0 && fr <= tot, "cudaMemGetInfo sane");
    std::printf("      vram: %.1f / %.1f GiB free\n", (double) fr / 1073741824.0, (double) tot / 1073741824.0);

    // ---------------------------------------------------------------- allocations
    const size_t kDev = 64u << 20;
    const size_t kHost = 4u << 20;
    void * d = nullptr;
    CHECK(cudaMalloc(&d, kDev) == cudaSuccess && d != nullptr, "cudaMalloc 64 MiB");
    void * h = nullptr;
    CHECK(cudaMallocHost(&h, kHost) == cudaSuccess && h != nullptr, "cudaMallocHost 4 MiB");
    void * h2 = nullptr;
    CHECK(cudaHostAlloc(&h2, 4096, cudaHostAllocPortable) == cudaSuccess && h2 != nullptr,
          "cudaHostAlloc(portable) 4 KiB");

    // pageable host memory, the engine's std::vector case -> staged path
    std::vector<unsigned char> pageable(1u << 20), back(1u << 20);
    for (size_t i = 0; i < pageable.size(); ++i) pageable[i] = (unsigned char) (i * 31u + 7u);

    CHECK(cudaMemcpy(d, pageable.data(), pageable.size(), cudaMemcpyHostToDevice) == cudaSuccess,
          "cudaMemcpy H2D from PAGEABLE memory (staged)");
    CHECK(cudaMemcpy(back.data(), d, pageable.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
          "cudaMemcpy D2H to PAGEABLE memory (staged)");
    CHECK(std::memcmp(back.data(), pageable.data(), pageable.size()) == 0, "pageable round trip is bit-exact");

    // USM host allocation -> the direct (no staging) path
    std::memcpy(h, pageable.data(), pageable.size());
    CHECK(cudaMemcpy(d, h, pageable.size(), cudaMemcpyHostToDevice) == cudaSuccess, "cudaMemcpy H2D from USM host");
    std::memset(h, 0, pageable.size());
    CHECK(cudaMemcpy(h, d, pageable.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "cudaMemcpy D2H to USM host");
    CHECK(std::memcmp(h, pageable.data(), pageable.size()) == 0, "USM host round trip is bit-exact");

    // memset + async + device-to-device
    void * d2 = nullptr;
    CHECK(cudaMalloc(&d2, kDev) == cudaSuccess, "cudaMalloc 64 MiB (second)");
    CHECK(cudaMemsetAsync(d, 0, kDev, nullptr) == cudaSuccess, "cudaMemsetAsync 64 MiB");
    CHECK(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize");
    CHECK(cudaMemcpy(h, d, 4096, cudaMemcpyDeviceToHost) == cudaSuccess && all_bytes(h, 4096, 0),
          "cudaMemsetAsync really zeroed the device buffer");
    CHECK(cudaMemcpyAsync(d2, d, kDev, cudaMemcpyDeviceToDevice, nullptr) == cudaSuccess, "cudaMemcpyAsync D2D");
    CHECK(cudaStreamSynchronize(nullptr) == cudaSuccess, "cudaStreamSynchronize(nullptr) = sync everything");

    // ---------------------------------------------------------------- streams + events
    cudaStream_t s1 = nullptr, s2 = nullptr;
    CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) == cudaSuccess &&
              cudaStreamCreate(&s2) == cudaSuccess,
          "cudaStreamCreate / cudaStreamCreateWithFlags");
    CHECK(cudaStreamQuery(s1) == cudaSuccess, "cudaStreamQuery on an idle stream");

    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
    CHECK(cudaEventCreate(&e0) == cudaSuccess &&
              cudaEventCreateWithFlags(&e1, cudaEventDisableTiming) == cudaSuccess &&
              cudaEventCreate(&e2) == cudaSuccess,
          "cudaEventCreate / cudaEventCreateWithFlags");

    CHECK(cudaEventRecord(e0, s1) == cudaSuccess, "cudaEventRecord(e0, s1)");
    CHECK(cudaMemsetAsync(d2, 0xAB, kDev, s1) == cudaSuccess, "32/64 MiB memset on s1");
    CHECK(cudaEventRecord(e1, s1) == cudaSuccess, "cudaEventRecord(e1, s1)");
    CHECK(cudaEventSynchronize(e1) == cudaSuccess, "cudaEventSynchronize(e1)");
    float ms = -1.0f;
    CHECK(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess && ms > 0.0f, "cudaEventElapsedTime gives a GPU interval");
    std::printf("      memset 64 MiB measured: %.3f ms\n", ms);

    // cross-stream ordering: s2 must not run its memset before s1's work
    CHECK(cudaStreamWaitEvent(s2, e1, 0) == cudaSuccess, "cudaStreamWaitEvent(s2 waits s1)");
    CHECK(cudaMemsetAsync(d2, 0xCD, 4096, s2) == cudaSuccess, "memset on s2 after the wait");
    CHECK(cudaStreamSynchronize(s2) == cudaSuccess, "cudaStreamSynchronize(s2)");
    CHECK(cudaMemcpy(h, d2, 4096, cudaMemcpyDeviceToHost) == cudaSuccess && all_bytes(h, 4096, 0xCD),
          "the s2 write landed after the s1 era (cross-stream event works)");

    // ---------------------------------------------------------------- host register (documented no-op)
    std::vector<int> raw(1024, 5);
    void * devp = nullptr;
    CHECK(cudaHostRegister(raw.data(), raw.size() * sizeof(int), cudaHostRegisterDefault) == cudaSuccess,
          "cudaHostRegister is a recorded no-op");
    CHECK(cudaHostGetDevicePointer(&devp, raw.data(), 0) == cudaSuccess && devp == (void *) raw.data(),
          "cudaHostGetDevicePointer returns the host address (USM)");
    CHECK(cudaHostUnregister(raw.data()) == cudaSuccess, "cudaHostUnregister");

    // ---------------------------------------------------------------- multi-GPU: refused, not faked
    int can = 1;
    CHECK(cudaDeviceCanAccessPeer(&can, 0, 0) == cudaSuccess && can == 0, "cudaDeviceCanAccessPeer -> 0 on one GPU");
    CHECK(cudaDeviceEnablePeerAccess(0, 0) == cudaErrorPeerAccessUnsupported,
          "cudaDeviceEnablePeerAccess -> not supported");
    CHECK(cudaMemcpyPeer(d2, 0, d, 0, 4096) == cudaErrorPeerAccessUnsupported, "cudaMemcpyPeer -> not supported");
    CHECK(cudaGetLastError() == cudaErrorPeerAccessUnsupported, "peer error is visible to cudaGetLastError");
    CHECK(cudaDeviceDisablePeerAccess(0) == cudaSuccess, "cudaDeviceDisablePeerAccess is a harmless success");

    // ---------------------------------------------------------------- CUDA Graph: refused
    CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal) == cudaErrorStreamCaptureUnsupported,
          "cudaStreamBeginCapture -> stream capture unsupported");
    cudaGraph_t g = (cudaGraph_t) 0x1;
    CHECK(cudaStreamEndCapture(s1, &g) == cudaErrorStreamCaptureUnsupported && g == nullptr,
          "cudaStreamEndCapture -> unsupported, graph nulled");
    CHECK(cudaGetLastError() == cudaErrorStreamCaptureUnsupported, "capture error is visible to cudaGetLastError");

    // ---------------------------------------------------------------- profiler / errors
    CHECK(cudaProfilerStart() == cudaSuccess && cudaProfilerStop() == cudaSuccess, "cudaProfilerStart/Stop no-ops");
    CHECK(std::strcmp(cudaGetErrorString(cudaErrorPeerAccessUnsupported), "") != 0, "cudaGetErrorString");

    // ---------------------------------------------------------------- teardown
    CHECK(cudaEventDestroy(e0) == cudaSuccess && cudaEventDestroy(e1) == cudaSuccess &&
              cudaEventDestroy(e2) == cudaSuccess,
          "cudaEventDestroy x3");
    CHECK(cudaStreamDestroy(s1) == cudaSuccess && cudaStreamDestroy(s2) == cudaSuccess, "cudaStreamDestroy x2");
    CHECK(cudaFree(d) == cudaSuccess && cudaFree(d2) == cudaSuccess, "cudaFree x2");
    CHECK(cudaFreeHost(h) == cudaSuccess && cudaFreeHost(h2) == cudaSuccess, "cudaFreeHost x2");
    CHECK(cudaFree(d) == cudaErrorInvalidValue, "double cudaFree is an error");

    const ds41_sycl_compat::info i = ds41_sycl_compat::get_info();
    std::printf("      shim: %s / %s, device bytes live %zu, host bytes live %zu, registered %zu, streams %d\n",
                i.device_name.c_str(), i.backend.c_str(), i.device_bytes_live, i.host_bytes_live, i.registered_bytes,
                i.streams_live);
    CHECK(i.device_bytes_live == 0 && i.host_bytes_live == 0, "all tracked allocations were released");

    std::printf("\n%s (%d failure%s)\n", g_fail == 0 ? "ALL PASS" : "FAILURES", g_fail, g_fail == 1 ? "" : "s");
    return g_fail == 0 ? 0 : 1;
}
