// strata_ds4_launch(kernel, grid, block, shared, stream, args...): the engine's one spelling of a kernel launch.  On CUDA it is
// kernel<<<grid, block, shared, stream>>>(args...); on Metal metal/include/metal_compat/cuda_launch.h maps the same call onto
// a dispatch.  (Every file that launches kernels includes this header.)
//
// [Intel port] The SYCL branch is the port's own: dpct turned every kernel into a plain function (and every launch site into
// a call to a generated wrapper that submits its own parallel_for), so a launch here goes through dpct::kernel_launcher,
// which hands the wrapper the queue and the nd_range.
#pragma once

#if defined(STRATA_METAL)
// the Mac port provides strata_ds4_launch / ds4_metal_flush / host_view (metal/include/metal_compat/cuda_launch.h)
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <cstdlib>
#include <cstring>

// Launch diagnostics for a card that resets: a hung kernel cannot be cancelled (the process turns into a zombie and the
// driver wedges), so the line printed BEFORE the launch is what names the kernel that hung.  Run with DS4_LAUNCH_LOG=1
// and read the last "ds4 launch" line after a reset; DS4_SKIP_LAUNCH="<text>[,...]" turns a launch whose "file:line"
// contains that text into a no-op, which lets a bisect continue without another reboot (the numbers are then wrong: for
// finding a hang, not for answers).
namespace ds4_port {
// [65600] ds4_model.dp.cpp turns this on for a prompt whose row count needs three chunks or more: see the long comment
// by forward_chunks' PromptSyncGuard.  Everything else keeps the async queue it has always had.
inline std::atomic<bool> &sync_prompt_launches() {
    static std::atomic<bool> v{false};
    return v;
}
inline bool launch_log() {
    static const bool v = std::getenv("DS4_LAUNCH_LOG") != nullptr;
    return v;
}
inline const char *skip_list() {
    static const char *v = std::getenv("DS4_SKIP_LAUNCH");
    return v;
}
inline bool launch_skipped(const char *file, int line) {
    const char *want = skip_list();
    if (want == nullptr || *want == static_cast<char>(0)) return false;
    char here[512];
    std::snprintf(here, sizeof(here), "%s:%d", file, line);
    for (const char *p = want; *p != static_cast<char>(0);) {
        const char *comma = std::strchr(p, ',');
        const size_t n = comma ? (size_t)(comma - p) : std::strlen(p);
        if (n > 0) {
            if (std::strncmp(p, here, n) == 0) return true;        // a prefix of "file:line"
            for (const char *h = here; *h != static_cast<char>(0); ++h)   // or that text anywhere in it
                if (std::strncmp(h, p, n) == 0) return true;
        }
        p += n + (comma ? 1 : 0);
    }
    return false;
}
// [port] A queued copy with neither end in device memory runs on the Arc's copy engine (engine_class=bcs). On this
// driver that engine has hung on such copies: the kernel log shows "Engine reset: engine_class=bcs" and every later
// transfer crawls (measured on the B70, 2026-10-29: two resets inside ten minutes, and the same 8K prompt then took
// 300+ s where it takes 30).  Upstream hit it too and fixed it the same way (Niko1221/Strata#3248,
// conversation_snapshot.cpp: "host-only copies run on the CPU").  DS4_HOST_COPY=0 keeps the queue's path for the A/B.
inline bool host_copy_on() {
    static const bool v = [] {
        const char *e = std::getenv("DS4_HOST_COPY");
        return e == nullptr || e[0] != '0';
    }();
    return v;
}
// USM host, USM shared and unknown (a pageable mmap: the model's files) are all read and written by the CPU.
inline bool cpu_side(const void *p) {
    const sycl::usm::alloc t = sycl::get_pointer_type(p, dpct::get_in_order_queue().get_context());
    return t == sycl::usm::alloc::host || t == sycl::usm::alloc::shared || t == sycl::usm::alloc::unknown;
}
inline bool cpu_copy(void *dst, const void *src, size_t n) {
    if (!host_copy_on() || !cpu_side(dst) || !cpu_side(src)) return false;
    std::memcpy(dst, src, n);
    return true;
}
// (dpct::async_dpct_memcpy's pitched form: h rows of w bytes)
inline bool cpu_copy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t w, size_t h) {
    if (!host_copy_on() || !cpu_side(dst) || !cpu_side(src)) return false;
    for (size_t i = 0; i < h; ++i) std::memcpy((char *) dst + i * dpitch, (const char *) src + i * spitch, w);
    return true;
}
}  // namespace ds4_port

template <class... KA, class... CA>
inline void strata_ds4_launch_at(const char *file, int line,
                                 void (*kernel)(KA...), dpct::dim3 grid,
                                 dpct::dim3 block, size_t shared,
                                 dpct::queue_ptr stream, CA &&...args) {
    // the generated wrapper reads kernel_launcher's thread_local queue and nd_range and submits the parallel_for itself;
    // the shared-memory size it asks for is always 0 (dpct moved the block tables to group_local_memory).
    auto &q = stream ? *stream : dpct::get_in_order_queue();
    if (ds4_port::launch_log()) {
        std::fprintf(stderr, "ds4 launch %s:%d grid=%u,%u,%u block=%u,%u,%u shared=%zu", file, line, grid.x, grid.y,
                     grid.z, block.x, block.y, block.z, shared);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
    if (ds4_port::launch_skipped(file, line)) {
        std::fprintf(stderr, "ds4 launch SKIPPED at %s:%d (DS4_SKIP_LAUNCH)", file, line);
        std::fputc('\n', stderr);
        return;
    }
    // [port] Niko1221/Strata#2911 (#2743): on CUDA a grid's y is at most 65535 and the prompt path launches one grid
    // row per token of the chunk, so upstream caps a named --prefill at 65280.  This port has no such limit - the grid
    // goes onto nd_range<3>, whose ranges are size_t (dpct/kernel.hpp:490) - and its chunk is already at most
    // kMaxChunk = 32768 (ds4_model.dp.cpp:2606), so a launch above 65535 here means some path bypassed that ceiling.
    // Naming the site is the difference between "a hang on the prompt path" and "a grid-limit overflow": print it once
    // per site rather than per launch.
    if (grid.y > 65535u || grid.z > 65535u) {
        static std::atomic<unsigned> said{0};
        if (said.fetch_add(1, std::memory_order_relaxed) < 16) {
            std::fprintf(stderr, "ds4 launch %s:%d grid=%u,%u,%u is past CUDA's 65535 grid rows (#2911); SYCL does not "
                                 "limit it, and this port's chunk cap (32768 rows) says it should not happen\n",
                         file, line, grid.x, grid.y, grid.z);
            std::fflush(stderr);
        }
    }
    dpct::kernel_launcher::launch(kernel, grid, block, (unsigned int)shared, stream,
                                  static_cast<KA>(args)...);
    if (std::getenv("DS4_LAUNCH_CHECK") != nullptr) std::fflush(stderr);   // a failing launch raises in the queue anyway
    // DS4_SYNC_LAUNCH=1 (or a prompt run that turned sync_prompt_launches() on): wait on every launch, and name the
    // one that failed
    if (std::getenv("DS4_SYNC_LAUNCH") != nullptr ||
        ds4_port::sync_prompt_launches().load(std::memory_order_relaxed)) {
        // [65600] A sync wait on a card that has stopped retiring one kernel blocks for ever - the very zombie the
        // bounded layer wait exists to prevent, and the only way out of it is a kill, which resets the engine.  So the
        // sync wait is bounded too: DS4_SYNC_LAUNCH_MS (default 60000, 0 the old unbounded wait).  On the deadline the
        // site printed is the kernel whose completion the device never reported; the process leaves by itself.
        static const long long sync_ms = [] {
            const char *v = std::getenv("DS4_SYNC_LAUNCH_MS");
            return v != nullptr ? (long long) std::atoll(v) : 60000LL;
        }();
        sycl::queue *const qp = &q;
        auto wait_one = [qp] { qp->wait_and_throw(); };
        if (sync_ms <= 0) {
            try {
                wait_one();
            } catch (const sycl::exception &e) {
                std::fprintf(stderr, "ds4 launch failed at %s:%d: %s", file, line, e.what());
                std::fputc('\n', stderr);
                std::exit(3);
            }
        } else {
            // (detached, not std::async: a std::future's destructor blocks until its task returns, so the std::async
            // shape could never actually give up on the deadline - see wait_bounded in ds4_model.dp.cpp)
            auto state = std::make_shared<std::atomic<int>>(0);   // 0: waiting, 1: returned, 2: threw
            std::thread([qp, state] {
                try {
                    qp->wait_and_throw();
                    state->store(1, std::memory_order_release);
                } catch (...) {
                    state->store(2, std::memory_order_release);
                }
            }).detach();
            const auto t0 = std::chrono::steady_clock::now();
            int st = 0;
            while ((st = state->load(std::memory_order_acquire)) == 0) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
                        .count() > sync_ms) {
                    std::fprintf(stderr,
                                 "ds4 launch %s:%d did not retire in %lld ms - this is the kernel the device stopped "
                                 "completing (DS4_SYNC_LAUNCH; DS4_SYNC_LAUNCH_MS=0 waits for ever)\n",
                                 file, line, sync_ms);
                    std::fflush(stderr);
                    std::_Exit(4);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (st == 2) {
                std::fprintf(stderr, "ds4 launch failed at %s:%d (the queue's wait threw)\n", file, line);
                std::fputc('\n', stderr);
                std::exit(3);
            }
        }
    }
}
#define strata_ds4_launch(...) strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// [port] CUDA's __nanosleep has no SYCL counterpart: a relaxed device-scope fence keeps the spin loops that call it (they
// re-read a USM flag right after) honest without stalling the sub-group.
inline void ds4_nanosleep(unsigned) {
    sycl::atomic_fence(sycl::memory_order::relaxed, sycl::memory_scope::device);
}

// (the Metal debug dumps in ds4_model.cu name these; on CUDA there is no deferred queue and a device pointer is not a host view)
inline void ds4_metal_flush() {}
namespace strata::gpu {
inline void* host_view(const void* p) { return const_cast<void*>(p); }
}

// [port] ggml-common.h's codebooks are `static const` HOST arrays.  Their address is not a device pointer: dpct wrapped
// them in dpct::global_memory and initialised them on the queue; passing &array[0] instead read host memory from the
// kernel (wrong sums on the iGPU, a hang and an engine reset on the B70).  One device copy per (type, size), filled once.
template <typename T, size_t N>
inline T *ds4_dev_table(const T (&host)[N]) {
    static T *dev = [&host] {
        auto *buf = new dpct::global_memory<T, 1>(sycl::range<1>(N));
        auto &q = dpct::get_in_order_queue();
        buf->init(q);   // (allocates on the device)
        q.memcpy(buf->get_ptr(), host, sizeof(T) * N).wait();
        return buf->get_ptr();
    }();
    return dev;
}
#else
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
template <class... KA, class... CA>
inline void strata_ds4_launch_at(const char* file, int line, void (*kernel)(KA...), dim3 grid, dim3 block, size_t shared,
                                 cudaStream_t stream, CA&&... args) {
    kernel<<<grid, block, shared, stream>>>(static_cast<KA>(args)...);
    static const bool check = std::getenv("DS4_LAUNCH_CHECK") != nullptr;   // report the launch that failed (not a later call)
    if (check) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s (grid %u,%u,%u block %u,%u,%u shared %zu)\n", file, line,
                         cudaGetErrorString(e), grid.x, grid.y, grid.z, block.x, block.y, block.z, shared);
    }
}
#define strata_ds4_launch(...) strata_ds4_launch_at(__FILE__, __LINE__, __VA_ARGS__)

// (the Metal debug dumps in ds4_model.cu name these; on CUDA there is no deferred queue and a device pointer is not a host view)
inline void ds4_metal_flush() {}
namespace strata::gpu {
inline void* host_view(const void* p) { return const_cast<void*>(p); }
}
#endif

// ds4_dbg("NAME"): the Mac engine's opt-in diagnostics (DS4_DBG_*, DS4_CPU_LOG, DS4_TLOG, ...).  On Metal it is an environment
// lookup; everywhere else it is the constant false, so the CUDA and HIP engines carry none of those checks on their hot paths.
#if defined(STRATA_METAL)
#include <cstdlib>
inline bool ds4_dbg(const char* name) { return std::getenv(name) != nullptr; }
#else
constexpr bool ds4_dbg(const char*) { return false; }
#endif
