// tools/v41/engine/sycl_compat/ds41_sycl_compat.cpp - the SYCL implementation of the CUDA-runtime surface declared
// in ds41_sycl_compat.hpp.  S1 of docs/xolotl-sycl-port-plan.md.
//
// MODEL
//   * one sycl::context on one GPU, shared by every queue, so USM allocations are usable from all of them;
//   * every cudaStream_t is its own IN-ORDER sycl::queue (in-order preserves CUDA's per-stream ordering; separate
//     queues preserve CUDA's independence BETWEEN streams, which the engine relies on for prefetch/demand-copy
//     overlap).  cudaStream_t == nullptr is CUDA's legacy default stream: its own queue, but synchronising it
//     synchronises every shim stream (a superset of CUDA's "legacy stream also serialises with the others");
//   * every cudaEvent_t is a sycl::event plus a host timestamp, and - because the queues are created with
//     enable_profiling - the device timestamps feed cudaEventElapsedTime where the backend provides them;
//   * USM allocations (device + host) are tracked by base address and range, because the engine also passes
//     pageable memory (std::vector::data(), mmap'd arenas) to the copy API; that memory is not device-reachable in
//     SYCL, so it is staged (see copy_in/copy_out below).
//   * that map is NOT the authority on "is this pointer device-reachable" - it only knows what went through THIS
//     shim.  The engine also receives device pointers from paths that never call us (the VRAM expert cache arena,
//     ggml's own USM, dpct-allocated buffers), so a map miss asks SYCL (sycl::get_pointer_type) and caches the
//     answer.  See "pointer classification" below; the reason it matters is a measured SIGABRT in the expert PCIe
//     path (docs/xolotl-sycl-s4-step4-pcie-dma-fix-report.md).
#include "ds41_sycl_compat.hpp"

#include <sycl/sycl.hpp>
#if defined(SYCL_EXT_INTEL_DEVICE_INFO)
#include <sycl/ext/intel/info/device.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------- S2: ggml-sycl's queue accessor
// Defined by ggml-sycl when this process links it (see ggml/include/ggml-sycl.h, added by
// tools/xolotl/ggml/xolotl-sycl-queue.patch).  Weak on purpose: the S1 self-test links this shim WITHOUT ggml, and
// there adoption must be a refused call, not a link error.
extern "C" void * ggml_backend_sycl_get_default_queue(int device) __attribute__((weak));

// The opaque handles of the header.  Defined here (and only here) so the engine TUs stay SYCL-free.
struct Ds41SyclEvent {
    sycl::event ev;
    bool has = false;                                              ///< recorded at least once
    bool timing_disabled = false;                                  ///< cudaEventDisableTiming was passed (we time anyway)
    std::chrono::steady_clock::time_point host_at;                 ///< host time of cudaEventRecord (fallback timing)
};

struct Ds41SyclStream {
    std::unique_ptr<sycl::queue> q;
    sycl::event last;          ///< newest submission on this queue (cudaStreamQuery)
    bool has_last = false;
    bool is_default = false;

    // Staging buffers for host memory the device cannot reach, in a RING.  A pageable H2D is host -> stage, then a
    // device memcpy of the stage; the slot's previous device copy must be home before the host overwrites it, and
    // that is the ONLY wait left - it is kStageSlots copies behind, so a run of pageable copies (the RAM arena's
    // expert blobs, the layer prestage, a wide activation upload) pipelines instead of draining the queue after
    // each one.  (S6 measured the drain for the small control copies; the same drain was still on every arena DMA.)
    static constexpr int kStageSlots = 8;
    struct StageSlot {
        void * p = nullptr;
        size_t cap = 0;
        sycl::event ev;
        bool live = false;     ///< `ev` holds an unfinished copy out of `p`
    };
    StageSlot stage[kStageSlots];
    int stage_at = 0;
    std::mutex stage_mu;
};

struct Ds41SyclGraph {
    int unused = 0;
};
struct Ds41SyclGraphExec {
    int unused = 0;
};

namespace {

// ---------------------------------------------------------------- runtime state
// `SharedUsm` is a SYCL USM *shared* allocation: the same address is valid on the host and on the device, so it is
// as device-reachable as device memory (it takes the direct copy path, not the staging path).
enum class PtrKind { None, Device, HostUsm, SharedUsm, RegisteredHost };

struct Alloc {
    size_t bytes = 0;          ///< 0: an exact-address entry discovered through SYCL (SYCL reports the kind, not the extent)
    PtrKind kind = PtrKind::None;
    bool external = false;     ///< found by the SYCL fallback: allocated outside this shim (expert cache, ggml, dpct)
};

// How much host memory the pooled staging buffers may hold per stream (they are reused, never per-copy).
constexpr size_t kStageCapBytes = size_t(512) << 20;

sycl::device pick_device() {
    // MUST be the same selector ggml-sycl uses (sycl::gpu_selector_v), because in S2 this shim's USM allocations and
    // the ggml kernels have to live on the same GPU.  It matters here: DPC++ on this box enumerates BOTH the OpenCL
    // and the Level Zero view of the card AND the UHD 770 iGPU, and get_devices(gpu)[0] is the OpenCL Arc - a
    // different device (and context) from the one ggml-sycl would pick.  gpu_selector_v resolves to the Level Zero
    // Arc (verified: backend=level_zero, 31.9 GiB).
    try {
        return sycl::device(sycl::gpu_selector_v);
    } catch (...) {
    }
    try {
        std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (!gpus.empty()) return gpus.front();
    } catch (...) {
    }
    return sycl::device(sycl::default_selector_v);
}

struct Runtime {
    sycl::device dev;
    sycl::context ctx;
    std::mutex mu;
    std::map<unsigned long long, Alloc> allocs;   ///< live USM allocations by base address
    std::map<unsigned long long, size_t> regs;    ///< cudaHostRegister'ed ranges (recorded only, never used)
    std::vector<Ds41SyclStream *> streams;
    Ds41SyclStream * def = nullptr;

    Runtime() : dev(pick_device()), ctx(dev) {}
};

// Leaked on purpose: a static Runtime would be destroyed at exit while the engine's static handles may still be
// torn down, and a sycl::context/queue destroyed under a late cudaFreeHost is a crash on exit for no benefit.
Runtime & rt() {
    static Runtime * r = new Runtime();
    return *r;
}

thread_local cudaError_t g_last_error = cudaSuccess;

cudaError_t fail(cudaError_t e) {
    if (e != cudaSuccess) g_last_error = e;
    return e;
}

sycl::queue & q_of(Ds41SyclStream * s) { return *s->q; }

Ds41SyclStream * resolve(cudaStream_t s) {
    if (s != nullptr) return s;
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    if (R.def == nullptr) {   // the default stream is created on first use
        auto * d = new Ds41SyclStream();
        d->q = std::make_unique<sycl::queue>(
            R.ctx, R.dev, sycl::property_list{sycl::property::queue::in_order(),
                                              sycl::property::queue::enable_profiling()});
        d->is_default = true;
        R.def = d;
        R.streams.push_back(d);
    }
    return R.def;
}

bool ev_complete(const sycl::event & ev) {
    try {
        return ev.get_info<sycl::info::event::command_execution_status>() ==
               sycl::info::event_command_status::complete;
    } catch (...) {
        return false;   // no state / backend without the query: treat as incomplete
    }
}

// ---------------------------------------------------------------- pointer classification
// The engine allocates through us (so we know the range) and also hands us pageable memory it allocated itself.
// A copy's pointer is only usable as a SYCL USM argument when it lies inside one of OUR allocations - or when SYCL
// itself says it is USM.
//
// WHY THE MAP ALONE WAS WRONG (S4 step 4).  `Runtime::allocs` only knows what passed through THIS shim's
// cudaMalloc/cudaMallocHost.  The engine also receives device pointers from paths that never call us:
//   * the VRAM expert cache arena - ExpertCache::open_sized -> strata::malloc_device_guarded -> sycl::malloc_device
//     (sycl/src/core/expert_cache.cpp), i.e. the `gp.cache->device_slot(sl)` destinations the expert PCIe path DMAs
//     straight into (ds41_moe.cpp:3222 gpu_run's LRU victim slot, :4155 the chunked twin);
//   * ggml-sycl's own backend buffers, and the dpct-migrated TUs' device buffers.
// Such a pointer classified as PtrKind::None, so do_copy() refused a perfectly valid DMA with cudaErrorInvalidValue -
// which the engine reports as `ds4_moe: pcie dma: invalid argument` and then aborts on.  A map MISS therefore now
// asks SYCL, which is the authority on USM, and caches the answer back into the same map (as an exact-address entry,
// because sycl::get_pointer_type reports the kind but not the extent).  The hot path stays one O(log n) lookup, and
// the interesting pointers stay visible in the map.
//
// WHAT THIS DOES NOT DO: it does not "assume device and try anyway".  Pageable host memory (the engine's 60 GiB
// mmap'd expert arena, std::vector data) still resolves to None through sycl::get_pointer_type - it is not USM -
// and still takes the staged path in copy_in/copy_out.  Treating host memory as device memory is how a wedge
// replaces an error.
struct Classified {
    PtrKind kind = PtrKind::None;
    bool was_in_map = false;   ///< hit in Runtime::allocs BEFORE the SYCL fallback (i.e. the shim allocated it)
    bool external = false;     ///< resolved through SYCL, so it is a USM allocation the shim did not make
};

const char * kind_name(PtrKind k) {
    switch (k) {
        case PtrKind::Device: return "device";
        case PtrKind::HostUsm: return "host-usm";
        case PtrKind::SharedUsm: return "shared-usm";
        case PtrKind::RegisteredHost: return "registered-host";
        default: return "none";
    }
}

const char * sycl_kind_name(sycl::usm::alloc a) {
    switch (a) {
        case sycl::usm::alloc::host: return "usm_host";
        case sycl::usm::alloc::device: return "usm_device";
        case sycl::usm::alloc::shared: return "usm_shared";
        default: return "unknown";
    }
}

/// Opt-in trace of the classification decisions (one line per map miss / staged copy).  Inert when unset.
bool shim_dbg() {
    static const bool on = std::getenv("DS41_SHIM_DBG") != nullptr;
    return on;
}

/// DS41_SHIM_CLASSIFY=map: reproduce the old map-only behaviour (the A/B that shows what the fallback fixed).
/// Diagnostic only; the default is to ask SYCL.
bool classify_map_only() {
    static const bool on = [] {
        const char * s = std::getenv("DS41_SHIM_CLASSIFY");
        return s != nullptr && std::strcmp(s, "map") == 0;
    }();
    return on;
}

Classified lookup_map(const void * p) {
    Classified c;
    if (p == nullptr) return c;
    Runtime & R = rt();
    const unsigned long long a = (unsigned long long) (uintptr_t) p;
    std::lock_guard<std::mutex> lock(R.mu);
    auto it = R.allocs.upper_bound(a);
    if (it == R.allocs.begin()) return c;
    --it;
    const Alloc & al = it->second;
    // bytes == 0 marks an exact-address entry (extent unknown); everything else is a real [base, base+bytes) range,
    // which is what makes an interior pointer (d_stage + slot * blob) a hit.
    const bool hit = al.bytes == 0 ? (a == it->first) : (a < it->first + (unsigned long long) al.bytes);
    if (!hit) return c;
    c.kind = al.kind;
    c.was_in_map = true;
    c.external = al.external;
    return c;
}

/// Ask SYCL what this pointer is.  Authoritative, and context-agnostic on this Level Zero stack (verified: an
/// allocation made in one sycl::context resolves correctly when asked from another, and for interior pointers).
PtrKind resolve_with_sycl(const void * p, sycl::usm::alloc * raw) {
    if (classify_map_only()) return PtrKind::None;
    try {
        const sycl::usm::alloc a = sycl::get_pointer_type(p, rt().ctx);
        if (raw) *raw = a;
        switch (a) {
            case sycl::usm::alloc::device: return PtrKind::Device;
            case sycl::usm::alloc::host: return PtrKind::HostUsm;
            case sycl::usm::alloc::shared: return PtrKind::SharedUsm;
            default: return PtrKind::None;
        }
    } catch (...) {
        return PtrKind::None;   // a backend that cannot answer is treated as "not USM": stage, never assume
    }
}

void remember_external(unsigned long long base, PtrKind kind) {
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    R.allocs[base] = Alloc{0, kind, true};
}

Classified classify_full(const void * p) {
    Classified c = lookup_map(p);
    if (c.kind != PtrKind::None || p == nullptr) return c;
    sycl::usm::alloc raw = sycl::usm::alloc::unknown;
    const PtrKind k = resolve_with_sycl(p, &raw);
    if (k != PtrKind::None) {
        remember_external((unsigned long long) (uintptr_t) p, k);
        c.kind = k;
        c.external = true;
        if (shim_dbg())
            std::fprintf(stderr,
                         "ds41 shim: classify(%p): not in the shim map; SYCL says %s -> %s (allocated outside the "
                         "shim: expert cache / ggml / dpct); cached\n",
                         p, sycl_kind_name(raw), kind_name(k));
    } else if (shim_dbg()) {
        std::fprintf(stderr,
                     "ds41 shim: classify(%p): not in the shim map; SYCL says %s%s -> not device-reachable, staged\n",
                     p, classify_map_only() ? "(not asked: DS41_SHIM_CLASSIFY=map) " : "", sycl_kind_name(raw));
    }
    return c;
}

PtrKind classify(const void * p) { return classify_full(p).kind; }

bool usm_reachable(PtrKind k) { return k == PtrKind::Device || k == PtrKind::HostUsm || k == PtrKind::SharedUsm; }

void remember(unsigned long long base, size_t bytes, PtrKind kind) {
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    R.allocs[base] = Alloc{bytes, kind};
}

bool forget(const void * p, PtrKind kind_wanted, size_t * bytes_out) {
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    auto it = R.allocs.find((unsigned long long) (uintptr_t) p);
    if (it == R.allocs.end() || it->second.kind != kind_wanted) return false;
    if (bytes_out) *bytes_out = it->second.bytes;
    R.allocs.erase(it);
    return true;
}

size_t live_bytes(PtrKind kind) {
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    size_t n = 0;
    for (const auto & kv : R.allocs)
        if (kv.second.kind == kind) n += kv.second.bytes;
    return n;
}

// ---------------------------------------------------------------- refusals that explain themselves
// A refused copy used to be a bare cudaErrorInvalidValue, and in the engine that is an abort 4 stack frames later
// with the numbers gone (`ds4_moe: pcie dma: invalid argument`), which cost a reboot-sized diagnosis twice.  The
// error path therefore ALWAYS prints one line naming both pointers, the byte count, what each classified as, and
// whether the shim knew the pointer before asking SYCL.  Nothing on the success path pays for it.
const char * map_text(const Classified & c) { return c.was_in_map ? "yes" : "no"; }

/// The raw SYCL answer for an error line.  Deliberately NOT gated by DS41_SHIM_CLASSIFY=map: the point of the line
/// is to say what the pointer IS even when the classification that refused it was the map-only one.
const char * raw_sycl_kind_name(const void * p) {
    if (p == nullptr) return "null";
    try {
        return sycl_kind_name(sycl::get_pointer_type(p, rt().ctx));
    } catch (...) {
        return "query-failed";
    }
}

cudaError_t refuse_copy(const char * op, void * dst, const void * src, size_t bytes, const Classified & cd,
                        const Classified & cs, const char * bad_side) {
    std::fprintf(stderr,
                 "ds41 shim: %s refused (invalid argument): dst=%p kind=%s shim-map=%s sycl-resolved=%s "
                 "sycl-kind=%s | src=%p kind=%s shim-map=%s sycl-resolved=%s sycl-kind=%s | bytes=%zu\n",
                 op, dst, kind_name(cd.kind), map_text(cd), cd.external ? "yes" : "no", raw_sycl_kind_name(dst), src,
                 kind_name(cs.kind), map_text(cs), cs.external ? "yes" : "no", raw_sycl_kind_name(src), bytes);
    std::fprintf(stderr,
                 "ds41 shim:   the %s side is not USM-reachable: pageable/registered HOST memory is staged through the "
                 "stream's USM host buffer, and a DEVICE pointer the shim did not allocate must be resolved by "
                 "sycl::get_pointer_type (this build does that).  DS41_SHIM_DBG=1 traces every classification miss and "
                 "staged copy.\n",
                 bad_side);
    return fail(cudaErrorInvalidValue);
}

cudaError_t refuse_memset(const char * op, void * dst, size_t bytes, int value) {
    const Classified c = classify_full(dst);
    std::fprintf(stderr,
                 "ds41 shim: %s refused (invalid argument): dst=%p kind=%s shim-map=%s sycl-resolved=%s sycl-kind=%s "
                 "bytes=%zu value=%d\n",
                 op, dst, kind_name(c.kind), map_text(c), c.external ? "yes" : "no", raw_sycl_kind_name(dst), bytes,
                 value);
    std::fprintf(stderr, "ds41 shim:   memset needs a non-null USM destination (device, shared or host-USM).\n");
    return fail(cudaErrorInvalidValue);
}

// ---------------------------------------------------------------- staged copies
// Host memory the device cannot reach (everything not allocated through cudaMalloc/cudaMallocHost - in this port
// that includes the whole RAM expert arena, because cudaHostRegister cannot pin an existing range) is copied
// through a ring of USM host staging buffers.  It is ONE extra copy (host -> stage -> device, exactly what NVIDIA's
// driver does for pageable memory).  The host -> stage copy is synchronous (the source must be free when
// cudaMemcpyAsync returns, as with the driver), but the stage -> device copy is NOT waited on: the wait moved to
// the reuse of that slot, kStageSlots copies later.  That is what keeps the queue from draining (S6's finding,
// finished here for the big blobs: one arena blob per pass x 2x13 MiB of host memcpy + a full queue drain each).
Ds41SyclStream::StageSlot * stage_take(Ds41SyclStream * s, size_t bytes) {
    Ds41SyclStream::StageSlot * sl = &s->stage[s->stage_at++ % Ds41SyclStream::kStageSlots];
    if (sl->live) {          // the copy out of this slot must be home before the host writes it again
        sl->ev.wait();
        sl->live = false;
    }
    if (sl->cap < bytes) {
        if (sl->p) {
            sycl::free(sl->p, rt().ctx);
            sl->p = nullptr;
            sl->cap = 0;
        }
        const size_t want = std::max(bytes, size_t(1) << 20);   // grow in >=1 MiB steps
        void * p = sycl::malloc_host(want, rt().ctx);
        if (p == nullptr) return nullptr;
        sl->p = p;
        sl->cap = want;
    }
    return sl;
}

cudaError_t copy_in(Ds41SyclStream * s, void * dst, const void * src, size_t bytes) {
    // dst is USM-reachable, src is pageable/registered host memory.
    if (shim_dbg()) {
        const Classified cs = classify_full(src);
        std::fprintf(stderr, "ds41 shim: staged H2D dst=%p src=%p (%zu bytes: src kind=%s map=%s -> not USM)\n", dst,
                     src, bytes, kind_name(cs.kind), map_text(cs));
    }
    std::lock_guard<std::mutex> lock(s->stage_mu);
    Ds41SyclStream::StageSlot * sl = stage_take(s, bytes);
    if (sl == nullptr) {
        std::fprintf(stderr, "ds41 shim: staging buffer for %zu bytes refused: sycl::malloc_host failed\n", bytes);
        return fail(cudaErrorMemoryAllocation);
    }
    std::memcpy(sl->p, src, bytes);
    sl->ev = q_of(s).memcpy(dst, sl->p, bytes);
    sl->live = true;
    s->last = sl->ev;
    s->has_last = true;
    // NOT waiting here is the DEFAULT (S9 ships it).  S7 kept the drain because "not draining changed the prompt's
    // last-row logits at 2048 (6c6884b0... -> a5787d7e..., -n 8 emitting token 0 forever)" - an observation that could
    // never adjudicate: 2048 is not a reproducible point in this engine.  The UNCHANGED build gives two different last
    // rows there (6c6884b0 / 5189d36b; S9: 4 runs = 2 + 2), the S8 control-ring fix does not change that, and adding
    // host I/O to one of those same runs produced S7's a5787d7e as a THIRD outcome - the wide-prompt nondeterminism is
    // upstream of this shim (see chunk_pf_min_rows' note and the S9 report), not this switch.  Where the point IS
    // reproducible this lever is bit-exact, on the pre-fix binary too: 1024 -> 787519ba (gate off and on), 256 ->
    // fce1788f, async off and on.  It pays +2.4 % prefill at 1024 (33.6 -> 34.4 tok/s: S8, idle box, -n 16, 2 runs)
    // and it reads no extra disk bytes: the staging ring is the same single host -> stage -> device copy, only the
    // per-copy queue drain is gone (the wait moved to the reuse of that slot, kStageSlots copies later).
    // DS41_SHIM_ASYNC_STAGE=0 puts the drain back for an A/B; any other value (or unset) = on.
    static const bool async_stage = !(std::getenv("DS41_SHIM_ASYNC_STAGE") && std::getenv("DS41_SHIM_ASYNC_STAGE")[0] == '0');
    if (!async_stage) {
        sl->ev.wait();
        sl->live = false;
    }
    return cudaSuccess;
}

cudaError_t copy_out(Ds41SyclStream * s, void * dst, const void * src, size_t bytes) {
    // src is USM-reachable, dst is pageable/registered host memory.  D2H has to wait: the bytes go to the host now.
    if (shim_dbg())
        std::fprintf(stderr, "ds41 shim: staged D2H dst=%p src=%p (%zu bytes: dst is not USM, one extra copy)\n", dst,
                     src, bytes);
    std::lock_guard<std::mutex> lock(s->stage_mu);
    Ds41SyclStream::StageSlot * sl = stage_take(s, bytes);
    if (sl == nullptr) {
        std::fprintf(stderr, "ds41 shim: staging buffer for %zu bytes refused: sycl::malloc_host failed\n", bytes);
        return fail(cudaErrorMemoryAllocation);
    }
    sl->ev = q_of(s).memcpy(sl->p, src, bytes);
    s->last = sl->ev;
    s->has_last = true;
    sl->ev.wait();
    std::memcpy(dst, sl->p, bytes);
    return cudaSuccess;
}

const char * copy_op_name(cudaMemcpyKind k) {
    switch (k) {
        case cudaMemcpyHostToDevice: return "cudaMemcpyAsync HostToDevice";
        case cudaMemcpyDeviceToHost: return "cudaMemcpyAsync DeviceToHost";
        case cudaMemcpyDeviceToDevice: return "cudaMemcpyAsync DeviceToDevice";
        case cudaMemcpyHostToHost: return "cudaMemcpyAsync HostToHost";
        default: return "cudaMemcpyAsync";
    }
}

cudaError_t do_copy(Ds41SyclStream * s, void * dst, const void * src, size_t bytes, cudaMemcpyKind kind) {
    if (bytes == 0) return cudaSuccess;

    if (kind == cudaMemcpyDefault) {
        const PtrKind kd = classify(dst), ks = classify(src);
        if (kd == PtrKind::Device && usm_reachable(ks)) kind = cudaMemcpyDeviceToDevice;
        else if (kd == PtrKind::Device) kind = cudaMemcpyHostToDevice;
        else if (usm_reachable(ks)) kind = cudaMemcpyDeviceToHost;
        else kind = cudaMemcpyHostToHost;
    }

    switch (kind) {
        case cudaMemcpyHostToHost:
            std::memcpy(dst, src, bytes);
            return cudaSuccess;

        case cudaMemcpyHostToDevice: {
            const Classified cs = classify_full(src);
            if (usm_reachable(cs.kind)) {
                s->last = q_of(s).memcpy(dst, src, bytes);
                s->has_last = true;
                return cudaSuccess;
            }
            const Classified cd = classify_full(dst);
            if (!usm_reachable(cd.kind))
                return refuse_copy(copy_op_name(kind), dst, src, bytes, cd, cs, "destination");
            return copy_in(s, dst, src, bytes);
        }

        case cudaMemcpyDeviceToHost: {
            const Classified cd = classify_full(dst);
            if (usm_reachable(cd.kind)) {
                s->last = q_of(s).memcpy(dst, src, bytes);
                s->has_last = true;
                return cudaSuccess;
            }
            const Classified cs = classify_full(src);
            if (!usm_reachable(cs.kind))
                return refuse_copy(copy_op_name(kind), dst, src, bytes, cd, cs, "source");
            return copy_out(s, dst, src, bytes);
        }

        case cudaMemcpyDeviceToDevice: {
            const Classified cd = classify_full(dst);
            const Classified cs = classify_full(src);
            if (!usm_reachable(cd.kind) || !usm_reachable(cs.kind))
                return refuse_copy(copy_op_name(kind), dst, src, bytes, cd, cs,
                                   usm_reachable(cd.kind) ? "source" : "destination");
            s->last = q_of(s).memcpy(dst, src, bytes);
            s->has_last = true;
            return cudaSuccess;
        }

        default:
            return fail(cudaErrorInvalidValue);
    }
}

void erase_stream(Ds41SyclStream * s) {
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    R.streams.erase(std::remove(R.streams.begin(), R.streams.end(), s), R.streams.end());
    if (R.def == s) R.def = nullptr;
}

}  // namespace

void * ds41_sycl_queue_of(void * stream) {
    return stream ? (void *) ((Ds41SyclStream *) stream)->q.get() : nullptr;
}

// The queue of the DEFAULT stream (the one the engine adopts from ggml at startup), as void*: a caller that has no
// cudaStream_t but must enqueue on the same in-order queue - the dense GEMM route submits between two
// ggml_backend_graph_compute calls - asks for it here.  Null only if the queue cannot be created.
void * ds41_sycl_default_queue() {
    Ds41SyclStream * s = resolve(nullptr);
    return s != nullptr && s->q ? (void *) s->q.get() : nullptr;
}

// ================================================================ ggml-sycl queue adoption (S2)
cudaError_t ds41_sycl_adopt_ggml_queue(int device) {
    if (ggml_backend_sycl_get_default_queue == nullptr) return fail(cudaErrorNotSupported);
    void * pq = ggml_backend_sycl_get_default_queue(device);
    if (pq == nullptr) return fail(cudaErrorInvalidDevice);
    sycl::queue * gq = static_cast<sycl::queue *>(pq);
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    if (R.def == nullptr) {
        auto * d = new Ds41SyclStream();
        // A COPY of the handle, not a second queue: sycl::queue is a handle, and the copy refers to the same
        // underlying in-order queue, so submissions through it are ordered with ggml's.  ggml keeps ownership; when
        // this handle goes away the queue itself lives on.
        d->q = std::make_unique<sycl::queue>(*gq);
        d->is_default = true;
        R.def = d;
        R.streams.push_back(d);
    } else {
        // Normally unreachable: the engine adopts immediately after ggml_backend_sycl_init, before any shim call.
        // If the default stream already exists, drain it first - an ordered wait preserves every earlier copy - and
        // swap in ggml's queue.
        R.def->q->wait();
        R.def->q = std::make_unique<sycl::queue>(*gq);
    }
    // Adopt ggml's context/device too, while that is still safe: USM the engine allocates from here on must be
    // creatable in the context of the queues that will use it (S1 verified cross-context USM on this Level Zero
    // stack, but the SYCL spec does not promise it).  With another shim stream already live, leave the context
    // alone: that stream is bound to the old one.
    if (R.allocs.empty() && R.streams.size() == 1) {
        R.ctx = gq->get_context();
        R.dev = gq->get_device();
    }
    return cudaSuccess;
}

// ================================================================ allocation
namespace ds41_sycl_compat {

cudaError_t malloc_device(void ** p, size_t bytes) {
    if (p == nullptr) return fail(cudaErrorInvalidValue);
    *p = nullptr;
    if (bytes == 0) return cudaSuccess;
    Runtime & R = rt();
    void * q = sycl::malloc_device(bytes, R.dev, R.ctx);
    if (q == nullptr) return fail(cudaErrorMemoryAllocation);
    *p = q;
    remember((unsigned long long) (uintptr_t) q, bytes, PtrKind::Device);
    return cudaSuccess;
}

cudaError_t malloc_host(void ** p, size_t bytes) {
    if (p == nullptr) return fail(cudaErrorInvalidValue);
    *p = nullptr;
    if (bytes == 0) return cudaSuccess;
    Runtime & R = rt();
    void * q = sycl::malloc_host(bytes, R.ctx);
    if (q == nullptr) return fail(cudaErrorMemoryAllocation);
    *p = q;
    remember((unsigned long long) (uintptr_t) q, bytes, PtrKind::HostUsm);
    return cudaSuccess;
}

// ================================================================ the arena's device-reachable host memory (S16)
// `strata::core::PinnedArena` used to be an mmap whose cudaHostRegister this port cannot honour, so every arena
// blob paid one synchronous host memcpy into a USM stage slot before its DMA (6.2 GB/s instead of 33 GB/s, ~4.5 s
// per 8192-token prompt).  It cannot register an existing range, but it CAN be handed memory that is USM from the
// start - and a caller that is about to allocate can ask.  These three functions are that ask; `pinned.dp.cpp`
// declares them weak, so a link without this shim keeps the mmap path and the staged copies.
//
// The allocation is `sycl::aligned_alloc_host` in the context the copies run on (ggml's, adopted at startup), so
// `sycl::get_pointer_type` answers "host" and do_copy() takes the direct async path: one DMA per blob at the
// measured 33 GB/s.  It must not be freed or handed out again while a copy from it is outstanding - the caller
// (the arena) lives for the whole process, and freeing it is the caller's last act.
//
// A single request is bounded by the device's max_mem_alloc_size (30.3 GiB on this card: the driver refused 31 GiB
// host USM in a direct zeMemAllocHost probe, and sycl::malloc_host returns null above it without saying why), so
// the caller pieces a larger arena at its own blob boundaries.  `ds41_sycl_host_alloc_max` is that bound less a
// margin; 0 means "no allocator in this link".
namespace { thread_local std::string g_arena_why; }

extern "C" size_t ds41_sycl_host_alloc_max() {
    Runtime & R = rt();
    try {
        const size_t mx = R.dev.get_info<sycl::info::device::max_mem_alloc_size>();
        // 512 MiB under the reported bound: the bound is the driver's per-allocation limit and the arena's pieces are
        // rounded down to a blob boundary anyway, so the margin costs nothing but one more piece.
        const size_t margin = (size_t) 512 << 20;
        return mx > margin ? mx - margin : 0;
    } catch (...) {
        return 0;
    }
}

extern "C" void * ds41_sycl_host_alloc(size_t bytes, size_t align, const char ** why) {
    if (why) *why = "";
    if (bytes == 0) return nullptr;
    Runtime & R = rt();
    if (align < 64) align = 64;
    const size_t mx = ds41_sycl_host_alloc_max();
    if (mx != 0 && bytes > mx) {
        g_arena_why = "the request is over the device's max single allocation";
        if (why) *why = g_arena_why.c_str();
        return nullptr;
    }
    try {
        void * p = sycl::aligned_alloc_host(align, bytes, R.ctx);
        if (p == nullptr) {
            g_arena_why = "sycl::aligned_alloc_host returned null (out of host memory or over the driver's limit)";
            if (why) *why = g_arena_why.c_str();
            return nullptr;
        }
        // register it here so the FIRST copy already classifies as host-usm instead of asking SYCL per blob
        remember((unsigned long long) (uintptr_t) p, bytes, PtrKind::HostUsm);
        return p;
    } catch (const std::exception & e) {
        g_arena_why = e.what();
        if (why) *why = g_arena_why.c_str();
        return nullptr;
    } catch (...) {
        g_arena_why = "an unknown SYCL exception";
        if (why) *why = g_arena_why.c_str();
        return nullptr;
    }
}

extern "C" void ds41_sycl_host_free(void * p) {
    if (p == nullptr) return;
    size_t bytes = 0;
    (void) forget(p, PtrKind::HostUsm, &bytes);
    try {
        sycl::free(p, rt().ctx);
    } catch (...) {
        // a free that throws is reported by nothing downstream; the arena is going away either way
    }
}

cudaError_t free_device(void * p) {
    if (p == nullptr) return cudaSuccess;
    size_t bytes = 0;
    if (!forget(p, PtrKind::Device, &bytes)) return fail(cudaErrorInvalidValue);
    sycl::free(p, rt().ctx);
    return cudaSuccess;
}

cudaError_t free_host(void * p) {
    if (p == nullptr) return cudaSuccess;
    size_t bytes = 0;
    if (!forget(p, PtrKind::HostUsm, &bytes)) return fail(cudaErrorInvalidValue);
    sycl::free(p, rt().ctx);
    return cudaSuccess;
}

// NO-OP BY DESIGN (semantic difference, reported): SYCL/USM has no "pin this existing range" operation.  We only
// record the range so the shim can report it.  The pages are NOT device-reachable: copies from/to them take the
// staged path in do_copy() and are synchronous.  The engine's own comment (ds41_moe.cpp:2132) already treats a
// failed cudaHostRegister as "stage the PCIe DMAs through the driver", which is exactly what happens here.
cudaError_t host_register(const void * p, size_t bytes, unsigned int flags) {
    (void) flags;   // cudaHostRegisterDefault / Portable / Mapped all mean nothing here
    if (p == nullptr) return fail(cudaErrorInvalidValue);
    if (bytes == 0) return cudaSuccess;
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    R.regs[(unsigned long long) (uintptr_t) p] = bytes;
    return cudaSuccess;
}

cudaError_t host_unregister(const void * p) {
    if (p == nullptr) return cudaSuccess;
    Runtime & R = rt();
    std::lock_guard<std::mutex> lock(R.mu);
    auto it = R.regs.find((unsigned long long) (uintptr_t) p);
    if (it != R.regs.end()) R.regs.erase(it);
    return cudaSuccess;
}

// USM host allocations are device-accessible at the SAME address, so the "device pointer" of a host allocation is
// the host pointer.  For a cudaHostRegister'ed range (no USM) we also return the same address: the engine stores it
// and later copies through it, and do_copy() classifies it as unreachable and stages - correct, just slower.
cudaError_t host_get_device_pointer(void ** dev_ptr, void * host_ptr, unsigned int flags) {
    (void) flags;
    if (dev_ptr == nullptr || host_ptr == nullptr) return fail(cudaErrorInvalidValue);
    *dev_ptr = host_ptr;
    return cudaSuccess;
}

info get_info() {
    Runtime & R = rt();
    info i{};
    i.device_name = "<unknown>";
    i.backend = "<unknown>";
    try {
        i.device_name = R.dev.get_info<sycl::info::device::name>();
        i.backend = R.dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero"
                     : R.dev.get_backend() == sycl::backend::opencl          ? "opencl"
                                                                             : "other";
    } catch (...) {
    }
    std::lock_guard<std::mutex> lock(R.mu);
    i.streams_live = (int) R.streams.size();
    for (const auto & kv : R.allocs) {
        if (kv.second.kind == PtrKind::Device) i.device_bytes_live += kv.second.bytes;
        if (kv.second.kind == PtrKind::HostUsm) i.host_bytes_live += kv.second.bytes;
    }
    for (const auto & kv : R.regs) i.registered_bytes += kv.second;
    return i;
}

}  // namespace ds41_sycl_compat

// ================================================================ copies
cudaError_t cudaMemcpy(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind) {
    if (bytes == 0) return cudaSuccess;
    Ds41SyclStream * s = resolve(nullptr);
    const cudaError_t e = do_copy(s, dst, src, bytes, kind);
    if (e != cudaSuccess) return e;
    // Synchronous cudaMemcpy: the copy must be done when this returns.  (CuDNN/CUDA semantics: it is ordered with
    // the default stream only; ours also waits for that stream, which is the same thing here.)
    q_of(s).wait();
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    if (bytes == 0) return cudaSuccess;
    return do_copy(resolve(stream), dst, src, bytes, kind);
}

cudaError_t cudaMemset(void * dst, int value, size_t bytes) {
    if (bytes == 0) return cudaSuccess;
    if (dst == nullptr || !usm_reachable(classify(dst))) return refuse_memset("cudaMemset", dst, bytes, value);
    Ds41SyclStream * s = resolve(nullptr);
    q_of(s).memset(dst, value, bytes);
    q_of(s).wait();
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void * dst, int value, size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return cudaSuccess;
    if (dst == nullptr || !usm_reachable(classify(dst))) return refuse_memset("cudaMemsetAsync", dst, bytes, value);
    Ds41SyclStream * s = resolve(stream);
    s->last = q_of(s).memset(dst, value, bytes);
    s->has_last = true;
    return cudaSuccess;
}

// ================================================================ streams
cudaError_t cudaStreamCreateWithFlags(cudaStream_t * stream, unsigned int flags) {
    (void) flags;   // cudaStreamNonBlocking: every shim queue is already non-blocking w.r.t. the others
    if (stream == nullptr) return fail(cudaErrorInvalidValue);
    Runtime & R = rt();
    auto * s = new Ds41SyclStream();
    s->q = std::make_unique<sycl::queue>(R.ctx, R.dev,
                                         sycl::property_list{sycl::property::queue::in_order(),
                                                             sycl::property::queue::enable_profiling()});
    {
        std::lock_guard<std::mutex> lock(R.mu);
        R.streams.push_back(s);
    }
    *stream = s;
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t * stream) { return cudaStreamCreateWithFlags(stream, cudaStreamDefault); }

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream == nullptr) return cudaSuccess;   // destroying the legacy default stream is a no-op here
    if (stream->q) stream->q->wait();
    for (Ds41SyclStream::StageSlot & sl : stream->stage) {
        if (sl.p) {
            sycl::free(sl.p, rt().ctx);
            sl.p = nullptr;
            sl.cap = 0;
            sl.live = false;
        }
    }
    erase_stream(stream);
    delete stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    if (stream == nullptr) {
        // CUDA's legacy default stream serialises with the other (blocking) streams; the conservative equivalent is
        // to wait for every shim queue.  Reported as a semantic difference: nothing is missed, overlap is lost.
        Runtime & R = rt();
        std::vector<sycl::queue *> qs;
        {
            std::lock_guard<std::mutex> lock(R.mu);
            for (Ds41SyclStream * s : R.streams) qs.push_back(s->q.get());
        }
        for (sycl::queue * q : qs) q->wait();
        return cudaSuccess;
    }
    q_of(stream).wait();
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    Ds41SyclStream * s = resolve(stream);
    if (!s->has_last) return cudaSuccess;
    return ev_complete(s->last) ? cudaSuccess : fail(cudaErrorNotReady);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void) flags;
    Ds41SyclStream * s = resolve(stream);
    if (event == nullptr || !event->has) return cudaSuccess;   // CUDA: waiting on an unrecorded event is a no-op
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
    s->last = q_of(s).ext_oneapi_submit_barrier({event->ev});
    s->has_last = true;
#else
    // Portable fallback: an empty kernel that depends on the event.  Same ordering, one launch more.
    s->last = q_of(s).single_task(event->ev, [] {});
    s->has_last = true;
#endif
    return cudaSuccess;
}

// ================================================================ events
namespace {
cudaError_t make_event(Ds41SyclEvent ** out, unsigned int flags) {
    if (out == nullptr) return fail(cudaErrorInvalidValue);
    auto * e = new Ds41SyclEvent();
    e->timing_disabled = (flags & cudaEventDisableTiming) != 0;
    e->host_at = std::chrono::steady_clock::now();
    *out = e;
    return cudaSuccess;
}
}  // namespace

cudaError_t cudaEventCreate(cudaEvent_t * event) { return make_event(event, cudaEventDefault); }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t * event, unsigned int flags) { return make_event(event, flags); }

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (event == nullptr) return fail(cudaErrorInvalidResourceHandle);
    Ds41SyclStream * s = resolve(stream);
    // A barrier is an (empty) command that is ordered after everything already submitted to this in-order queue, so
    // its profiling timestamps mark this position in the stream - the closest SYCL equivalent of cudaEventRecord.
#if defined(SYCL_EXT_ONEAPI_ENQUEUE_BARRIER)
    event->ev = q_of(s).ext_oneapi_submit_barrier();
#else
    event->ev = q_of(s).single_task([] {});
#endif
    event->has = true;
    event->host_at = std::chrono::steady_clock::now();
    s->last = event->ev;
    s->has_last = true;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (event == nullptr || !event->has) return cudaSuccess;
    event->ev.wait();
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    if (event == nullptr || !event->has) return cudaSuccess;
    return ev_complete(event->ev) ? cudaSuccess : fail(cudaErrorNotReady);
}

// Elapsed time between two records.
//   * Preferred: the device timestamps of each barrier event (queues are built with enable_profiling), so the number
//     is GPU time between the two stream positions, like CUDA's.
//   * Fallback (backend without profiling, or a zeroed timestamp): host wall-clock taken at cudaEventRecord, i.e.
//     HOST ENQUEUE time, not completion time.  This is the precision loss stated in the header: the fallback
//     under-reports elapsed GPU work (it measures how long the host took to submit, not how long the GPU ran).
// Like CUDA, an event that has not completed yields cudaErrorNotReady (the engine's event profiler depends on it).
cudaError_t cudaEventElapsedTime(float * ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) return fail(cudaErrorInvalidResourceHandle);
    if (!end->has || !ev_complete(end->ev)) return fail(cudaErrorNotReady);
    if (start->has && !ev_complete(start->ev)) return fail(cudaErrorNotReady);

    unsigned long long s0 = 0, s1 = 0, e0 = 0, e1 = 0;
    bool dev_time = false;
    try {
        s0 = start->ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        e0 = start->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        s1 = end->ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        e1 = end->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        // A backend that does not really profile returns zeros; only trust a sane, increasing interval.
        dev_time = (e1 > s0) && (e1 != 0 || e0 != 0);
    } catch (...) {
        dev_time = false;
    }
    if (dev_time) {
        *ms = (float) ((double) (e1 - s0) / 1.0e6);
        return cudaSuccess;
    }
    const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(end->host_at - start->host_at).count();
    *ms = (float) ((double) d / 1.0e6);
    return cudaSuccess;
}

// ================================================================ device
cudaError_t cudaGetDeviceCount(int * count) {
    if (count == nullptr) return fail(cudaErrorInvalidValue);
    *count = 1;   // single GPU: the peer/multi-device paths are off by construction (see the peer API below)
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int * device) {
    if (device == nullptr) return fail(cudaErrorInvalidValue);
    *device = 0;
    return cudaSuccess;
}

// CUDA ordinals are 0..n-1.  This shim exposes exactly ONE ordinal, so any other index is a real error (the engine
// turns it into "cudaSetDevice(N) failed" instead of running on the wrong GPU).
cudaError_t cudaSetDevice(int device) {
    if (device != 0) return fail(cudaErrorInvalidDevice);
    (void) rt();   // make sure the runtime is up before the first allocation
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device) {
    if (prop == nullptr) return fail(cudaErrorInvalidValue);
    if (device != 0) return fail(cudaErrorInvalidDevice);
    std::memset(prop, 0, sizeof(*prop));
    Runtime & R = rt();
    auto name = R.dev.get_info<sycl::info::device::name>();
    std::snprintf(prop->name, sizeof(prop->name), "%s", name.c_str());
    // NOT a compute capability: Intel GPUs have none.  10.0 is chosen so strata::cuda_cc_encode() = 100 and
    // strata::cuda_cc_below_mmq_tensor_cores(100) == false - i.e. the compute-capability policy in
    // ds41_cuda_policy.cpp does not switch chunk-MMQ off for a card that is not NVIDIA.  This is a lie by design
    // and the only field here that is not derived from the SYCL device.
    prop->major = 10;
    prop->minor = 0;
    try {
        prop->totalGlobalMem = R.dev.get_info<sycl::info::device::global_mem_size>();
        prop->multiProcessorCount = (int) R.dev.get_info<sycl::info::device::max_compute_units>();
        prop->maxThreadsPerBlock = (int) R.dev.get_info<sycl::info::device::max_work_group_size>();
        prop->sharedMemPerBlock = (int) R.dev.get_info<sycl::info::device::local_mem_size>();
        prop->sharedMemPerBlockOptin = prop->sharedMemPerBlock;
        prop->l2CacheSize = (int) R.dev.get_info<sycl::info::device::global_mem_cache_size>();
    } catch (...) {
    }
    prop->warpSize = 32;
    // integrated: the only SYCL source is info::device::host_unified_memory, deprecated in SYCL 2020 and not a
    // discrete-card question.  Reported as 0 (discrete Arc); nothing in the engine reads it.
    prop->integrated = 0;
    prop->managedMemory = R.dev.has(sycl::aspect::usm_shared_allocations) ? 1 : 0;
    prop->unifiedAddressing = 0;
    prop->concurrentKernels = 1;
    prop->asyncEngineCount = 1;
    prop->canMapHostMemory = 1;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    // Every shim stream, plus us: nothing the engine queued anywhere is left in flight.
    Runtime & R = rt();
    std::vector<sycl::queue *> qs;
    {
        std::lock_guard<std::mutex> lock(R.mu);
        for (Ds41SyclStream * s : R.streams) qs.push_back(s->q.get());
    }
    for (sycl::queue * q : qs) q->wait();
    return cudaSuccess;
}

// Free VRAM, as the engine's allocation decisions expect.  Intel's Level Zero extension gives the real number; on a
// backend without it the shim reports total minus what IT has handed out (which ignores other processes - the
// engine treats this only as an upper bound for its own budget, and its own comment at ds41_moe.cpp:2158 already
// notes that cudaMemGetInfo's number was optimistic on Windows).
cudaError_t cudaMemGetInfo(size_t * free_bytes, size_t * total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return fail(cudaErrorInvalidValue);
    Runtime & R = rt();
    size_t total = 0;
    try {
        total = R.dev.get_info<sycl::info::device::global_mem_size>();
    } catch (...) {
        return fail(cudaErrorNotSupported);
    }
    size_t fr = 0;
    bool have = false;
#if defined(SYCL_EXT_INTEL_DEVICE_INFO)
    try {
        fr = R.dev.get_info<sycl::ext::intel::info::device::free_memory>();
        have = true;
    } catch (...) {
        have = false;
    }
#endif
    if (!have) {
        const size_t used = live_bytes(PtrKind::Device);
        fr = total > used ? total - used : 0;
    }
    if (fr > total) fr = total;
    *free_bytes = fr;
    *total_bytes = total;
    return cudaSuccess;
}

// ================================================================ multi-GPU: refused, not faked
// One GPU on this machine.  These return "no peer" / "unsupported" so the engine deterministically takes its
// pinned-host handoff path (ds41_dense.cpp: ds41_handoff_mode() -> kHandoffPinnedHost) and keeps the layer-split /
// remote-device features off instead of moving bytes through a path that cannot work.
cudaError_t cudaDeviceCanAccessPeer(int * can_access, int device, int peer_device) {
    (void) peer_device;
    if (can_access == nullptr) return fail(cudaErrorInvalidValue);
    if (device != 0 || peer_device != 0) { *can_access = 0; return cudaSuccess; }
    *can_access = 0;
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int peer_device, unsigned int flags) {
    (void) peer_device;
    (void) flags;
    return fail(cudaErrorPeerAccessUnsupported);
}

cudaError_t cudaDeviceDisablePeerAccess(int peer_device) {
    (void) peer_device;
    return cudaSuccess;   // nothing was ever enabled
}

cudaError_t cudaMemcpyPeer(void * dst, int dst_device, const void * src, int src_device, size_t bytes) {
    (void) dst;
    (void) src;
    (void) bytes;
    (void) dst_device;
    (void) src_device;
    return fail(cudaErrorPeerAccessUnsupported);
}

cudaError_t cudaMemcpyPeerAsync(void * dst, int dst_device, const void * src, int src_device, size_t bytes,
                                cudaStream_t stream) {
    (void) stream;
    return cudaMemcpyPeer(dst, dst_device, src, src_device, bytes);
}

// ================================================================ CUDA Graph: refused with a visible error
// DECISION (S1): capture is refused rather than faked.
//
// Why not "record and replay the submission sequence": the calls the engine captures are NOT made through this
// shim.  tier_graph_capture() (ds41_moe.cpp:2746) starts a capture and then calls
// strata::kernels::native_expert_grouped(..., cs, 0), passing the stream as an OPAQUE `void *` to the kernel layer.
// In a SYCL build those kernels submit to their own queue (and, in S2, ggml-sycl submits to ggml's queue), so this
// shim would record an EMPTY stream and replay nothing: the graph would silently compute nothing and the model
// would produce wrong tokens with no error anywhere.  That is the one outcome worse than no graph.
//
// What the engine does when capture is refused: tier_graph_capture() returns false and prints
// "STRATA_TIER_GRAPH: CUDA refused the capture; keeping the per-class launches" - the per-class launch path stays,
// which is the same path the CUDA build takes on any capture failure.  STRATA_TIER_GRAPH is opt-in (getenv), so the
// default build never even asks.  Cost: the six host calls per class launch are not folded into one, i.e. what the
// CUDA build loses too.  Graph nodes would have to be built through the SYCL kernel layer
// (sycl_ext_oneapi_graph, or a submission-record mechanism inside the kernel layer) for this to be more.
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    (void) stream;
    (void) mode;
    return fail(cudaErrorStreamCaptureUnsupported);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t * graph) {
    (void) stream;
    if (graph != nullptr) *graph = nullptr;
    return fail(cudaErrorStreamCaptureUnsupported);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t * exec, cudaGraph_t graph, void * error_node, char * log,
                                 size_t log_size) {
    (void) graph;
    (void) error_node;
    if (log != nullptr && log_size > 0) log[0] = '\0';
    if (exec != nullptr) *exec = nullptr;
    return fail(cudaErrorStreamCaptureUnsupported);
}

// Unreachable while capture is refused.  It returns an error instead of doing nothing quietly: if a future edit ever
// makes capture succeed without real graph support, the `ck()` at ds41_moe.cpp:3080 prints it instead of the launch
// silently disappearing (ck() only logs).
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    (void) exec;
    (void) stream;
    return fail(cudaErrorStreamCaptureUnsupported);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    delete exec;
    return cudaSuccess;
}

// ================================================================ errors / profiler
cudaError_t cudaGetLastError(void) {
    const cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;
    return e;
}

const char * cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInitializationError: return "initialization error";
        case cudaErrorInvalidDevice: return "invalid device ordinal";
        case cudaErrorNotReady: return "device not ready";
        case cudaErrorPeerAccessUnsupported: return "peer access is not supported between these devices (SYCL shim: single GPU)";
        case cudaErrorInvalidResourceHandle: return "invalid resource handle";
        case cudaErrorPeerAccessAlreadyEnabled: return "peer access was already enabled";
        case cudaErrorPeerAccessNotEnabled: return "peer access was not enabled";
        case cudaErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing (SYCL shim: CUDA Graph is not implemented)";
        case cudaErrorStreamCaptureInvalidated: return "operation aborted due to an earlier error during capture";
        case cudaErrorNotSupported: return "operation not supported by the SYCL backend";
    }
    return "unknown error";
}

// nsys --capture-range=cudaProfilerApi is meaningless without Nsight; the engine's STRATA_PROFILE_DECODE=1 switch
// must therefore work as a plain "on" and print nothing extra.  Success, no-op (reported).
cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
cudaError_t cudaProfilerStop(void) { return cudaSuccess; }
