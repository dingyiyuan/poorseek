#!/usr/bin/env python3
"""port_dsv41_fixes.py - the Intel port's edits on top of the dpct output for the DeepSeek-V4.1 engine (src/dsv41).

`sycl/tools/migrate.sh` turns src/dsv41/*.cu into sycl/src/dsv41/*.dp.cpp, and it also re-emits the headers it walks
(ds4_launch.h, the dpct helper headers under sycl/include/dpct/).  Everything the port has to say about that output lives
here, so the migration can be re-run at any upstream commit and this script re-applied on top of it:

    sycl/tools/migrate.sh <repo> <repo>/sycl        # once per upstream update
    python3 sycl/tools/port_dsv41_fixes.py          # then this

Every step is idempotent (a second run changes nothing).  What it does, in order:

  1. ds4_launch.h   - a SYCL branch for the engine's one launch spelling, through dpct::kernel_launcher.
  2. dpct/util.hpp  - the masked sub-group shifts without get_tangle_group/get_fixed_size_group (not in this compiler).
  3. ds4_numerics.dp.hpp - the BF16/E4M3 conversions in plain arithmetic (the source's HIP spelling), no CUDA intrinsics.
  4. sycl/include/strata/dsv41/{exl3,moe_win}.hpp - cudaStream_t -> dpct::queue_ptr (port copies, they win the include order).
  5. sycl/src/dsv41/vendor/exl3_cpu.cpp - __builtin_cpu_init only in the host pass of a -fsycl translation unit.
  6. sycl/src/dsv41/ds4_packinfo.cpp - the migrated ds4_iq header instead of the one that includes cuda_runtime.h.
  7. The codebook grids of ggml-common.h are static const arrays: drop dpct's .init()/.get_ptr() on them.
  8. __nanosleep -> ds4_nanosleep (ds4_launch.h), dpct_placeholder -> 0, unnamed parallel_for kernels (a templated
     wrapper's lambda is already a distinct type), the vision teardown's queue cast, cudaHostRegister's flags.
  9. ggml-common.h's SYCL branches (GGML_COMMON_DECL_SYCL) where dpct wrote the CUDA ones.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYCL = ROOT / "sycl"
DS4 = SYCL / "src" / "dsv41"

LAUNCH_H = '''// strata_ds4_launch(kernel, grid, block, shared, stream, args...): the engine's one spelling of a kernel launch.  On CUDA it is
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
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Launch diagnostics for a card that resets: a hung kernel cannot be cancelled (the process turns into a zombie and the
// driver wedges), so the line printed BEFORE the launch is what names the kernel that hung.  Run with DS4_LAUNCH_LOG=1
// and read the last "ds4 launch" line after a reset; DS4_SKIP_LAUNCH="<text>[,...]" turns a launch whose "file:line"
// contains that text into a no-op, which lets a bisect continue without another reboot (the numbers are then wrong: for
// finding a hang, not for answers).
namespace ds4_port {
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
        std::fputc('\\n', stderr);
        std::fflush(stderr);
    }
    if (ds4_port::launch_skipped(file, line)) {
        std::fprintf(stderr, "ds4 launch SKIPPED at %s:%d (DS4_SKIP_LAUNCH)", file, line);
        std::fputc('\\n', stderr);
        return;
    }
    dpct::kernel_launcher::launch(kernel, grid, block, (unsigned int)shared, stream,
                                  static_cast<KA>(args)...);
    if (std::getenv("DS4_LAUNCH_CHECK") != nullptr) std::fflush(stderr);   // a failing launch raises in the queue anyway
    // DS4_SYNC_LAUNCH=1: wait on every launch and name the one that failed
    if (std::getenv("DS4_SYNC_LAUNCH") != nullptr) {
        try {
            q.wait_and_throw();
        } catch (const sycl::exception &e) {
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s", file, line, e.what());
            std::fputc('\\n', stderr);
            std::exit(3);
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
            std::fprintf(stderr, "ds4 launch failed at %s:%d: %s (grid %u,%u,%u block %u,%u,%u shared %zu)\\n", file, line,
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
'''

NUMERICS_H = '''// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_numerics.cuh - the BF16 and FP8 (E4M3) conversions of maya-ds4's kernels.
//
// [Intel port] The source takes CUDA's intrinsics (cuda_bf16.h / cuda_fp8.h) and has a plain-arithmetic spelling of the same
// bits for HIP; the port uses that spelling for both, so this copy carries no CUDA header at all: BF16 rounds to nearest
// even with a NaN passed through (strata/kernels/bf16_bits.hpp, the same rule), E4M3 (OCP "fn": no infinities, max 448) is
// decoded exactly and rounded to nearest even, saturated.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

#include "strata/kernels/bf16_bits.hpp"

namespace strata::ds4 {

/// F32 -> BF16 bits, round to nearest even (a NaN stays a NaN).
__dpct_inline__ uint16_t bf16_rn(float f) { return strata::kernels::bf16_from_f32(f); }

namespace num_detail {
__dpct_inline__ uint32_t as_u32(float f) { return sycl::bit_cast<uint32_t>(f); }
__dpct_inline__ float as_f32(uint32_t u) { return sycl::bit_cast<float>(u); }
}  // namespace num_detail

/// An E4M3 code's value (exact in F32).
__dpct_inline__ float e4m3_value(uint8_t q) {
    const uint32_t e = (q >> 3) & 15u, m = q & 7u;
    float v;
    if (e == 15u && m == 7u) v = num_detail::as_f32(0x7fc00000u);            // NaN
    else if (e == 0u) v = (float) m * 0.001953125f;                          // m 2^-9
    else v = num_detail::as_f32(((e + 120u) << 23) | (m << 20));              // (1 + m / 8) 2^(e - 7)
    return (q & 0x80u) ? -v : v;
}

/// v rounded to the nearest E4M3 value (ties to even), saturated at +-448.
__dpct_inline__ float e4m3_round(float v) {
    const float a = sycl::fabs(v);
    if (!(a == a)) return v;
    float r;
    if (a >= 448.0f) r = 448.0f;
    else {
        const int e = (int) ((num_detail::as_u32(a) >> 23) & 0xffu) - 127;   // a's binade (a > 0)
        const float step = e < -6 ? 0.001953125f : num_detail::as_f32((uint32_t) (e - 3 + 127) << 23);
        r = a == 0.0f ? 0.0f : sycl::rint(a / step) * step;                   // (a power of two: exact)
        r = sycl::fmin(r, 448.0f);
    }
    return v < 0.0f ? -r : r;
}

}  // namespace strata::ds4
'''

TANGLE = '''  if ((1U << sg.get_local_linear_id()) & member_mask) {
    auto inner = sycl::ext::oneapi::experimental::get_tangle_group(sg);
    sycl::group_barrier(inner);
  }
  auto partition =
      sycl::ext::oneapi::experimental::get_fixed_size_group<LogicSubGroupSize>(
          sg);'''
TANGLE_NEW = '''  // [port] every kernel is built with a sub-group of 32 (-fsycl-default-sub-group-size=32), so the logical sub-group
  // IS the sub_group: this compiler has no get_tangle_group / get_fixed_size_group.
  auto partition = sg;'''

# the ggml-common.h codebooks: static const arrays, device-visible as they are (dpct wrapped them in global_memory)
GRIDS = ("iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid", "iq3s_grid", "kvalues_iq4nl", "iq1s_grid_gpu",
         "tq_lut_f16")


def dp_files():
    return sorted(list(DS4.glob("*.dp.cpp")) + list((DS4 / "vendor").glob("*.dp.cpp")) +
                  list(DS4.glob("*.dp.hpp")) + list((DS4 / "vendor").glob("*.dp.hpp")))


def write_if_changed(path: pathlib.Path, text: str) -> bool:
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


def step_launch():
    return write_if_changed(DS4 / "ds4_launch.h", LAUNCH_H)


def step_dpct_util():
    p = SYCL / "include" / "dpct" / "util.hpp"
    s = p.read_text()
    if TANGLE not in s:
        return False
    p.write_text(s.replace(TANGLE, TANGLE_NEW))
    return True


def step_numerics():
    return write_if_changed(DS4 / "ds4_numerics.dp.hpp", NUMERICS_H)


def step_port_headers():
    done = []
    for rel, ns in (("include/strata/dsv41/exl3.hpp", "namespace strata::dsv41k::exl3 {"),
                    ("include/strata/dsv41/moe_win.hpp", None)):
        src = (ROOT / rel).read_text()
        if "using cudaStream_t = dpct::queue_ptr;" not in src:
            src = src.replace("#include <cuda_runtime.h>", "#include <dpct/dpct.hpp>")
            if ns:
                src = src.replace(ns, ns + "\nusing cudaStream_t = dpct::queue_ptr;   // [port] a queue, as upstream's cudaStream_t", 1)
            else:
                m = re.search(r"^namespace [\w:]+ \{$", src, re.M)
                src = src[:m.end()] + "\nusing cudaStream_t = dpct::queue_ptr;   // [port] a queue, as upstream's cudaStream_t" + src[m.end():]
        if write_if_changed(SYCL / rel, src):
            done.append(rel)
    return done


def step_exl3_cpu():
    src = (ROOT / "src/dsv41/vendor/exl3_cpu.cpp").read_text()
    old = "#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))\n    __builtin_cpu_init();"
    new = ("// [port] the device pass of a -fsycl translation unit cannot see __builtin_cpu_init: the host keeps it, the\n"
           "// device compilation of this host-only file takes the fallback (no kernel of it ever runs on the card).\n"
           "#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(__SYCL_DEVICE_ONLY__)\n"
           "    __builtin_cpu_init();")
    if old in src:
        src = src.replace(old, new)
    return write_if_changed(DS4 / "vendor" / "exl3_cpu.cpp", src)


def step_packinfo():
    src = (ROOT / "src/dsv41/ds4_packinfo.cpp").read_text()
    src = src.replace('#include "ds4_iq.cuh"', '#include "ds4_iq.dp.hpp"   // [port] the migrated header (no cuda_runtime.h)')
    return write_if_changed(DS4 / "ds4_packinfo.cpp", src)


def step_kernels():
    changed = []
    for p in dp_files():
        s = p.read_text()
        o = s
        # the ggml codebook grids are static const arrays, not dpct memory objects
        for name in GRIDS:
            s = re.sub(r"^\s*%s\.init\(queue\);\n" % name, "", s, flags=re.M)
            s = re.sub(r"auto %s_ptr_ct1 = %s\.get_ptr\(\);" % (name, name),
                       "auto %s_ptr_ct1 = const_cast<std::remove_const_t<std::remove_reference_t<decltype(%s[0])>> *>(&%s[0]);"
                       % (name, name, name), s)
        # dpct's placeholder for an expression it could not migrate: these kernels take no launch-time local memory
        s = s.replace("0 + dpct_placeholder /* total share local memory size */",
                      "0 /* [port] the kernels take no launch-time shared memory (group_local_memory) */")
        # __nanosleep: the port's helper in ds4_launch.h
        s = re.sub(r"__nanosleep\(([^)]*)\)", r"ds4_nanosleep(\1)", s)
        # a templated wrapper's lambda is already its own type: no explicit kernel name (dpct reuses one, and it collides)
        s = re.sub(r"parallel_for<dpct_kernel_name<class [\w:]+_(?:<[^>]*>)?>>\(", "parallel_for(", s)
        # ggml-common.h is patched for SYCL (third_party/ggml): take its SYCL branches
        s = s.replace("GGML_COMMON_DECL_CUDA", "GGML_COMMON_DECL_SYCL").replace("GGML_COMMON_IMPL_CUDA", "GGML_COMMON_IMPL_SYCL")
        if s != o:
            if "#include <type_traits>" not in s and "remove_const_t" in s:
                s = s.replace("#include <dpct/dpct.hpp>", "#include <dpct/dpct.hpp>\n#include <type_traits>", 1)
            p.write_text(s)
            changed.append(p.name)
    return changed


def step_model_vision():
    changed = []
    p = DS4 / "ds4_model.dp.cpp"
    s = p.read_text()
    o = s
    s = s.replace("unsigned reg_flags = cudaHostRegisterMapped | cudaHostRegisterPortable;",
                  "unsigned reg_flags = 0;   // [port] SYCL has no host registration: the zero-copy store (DS4_STORE_ZC) must stay off\n                (void)reg_flags;")
    if s != o:
        p.write_text(s)
        changed.append(p.name)
    p = DS4 / "ds4_vision.dp.cpp"
    s = p.read_text()
    o = s
    s = s.replace("dpct::get_current_device().destroy_queue((cudaStream_t)st_);",
                  "dpct::queue_ptr q_st = (dpct::queue_ptr)st_;\n        dpct::get_current_device().destroy_queue(q_st);")
    s = s.replace("dpct::get_current_device().destroy_queue((cudaStream_t)cs_);",
                  "dpct::queue_ptr q_cs = (dpct::queue_ptr)cs_;\n        dpct::get_current_device().destroy_queue(q_cs);")
    # (the two statements above end up under one if without braces; give each its own block)
    s = s.replace("    if (st_) dpct::queue_ptr q_st = (dpct::queue_ptr)st_;\n        dpct::get_current_device().destroy_queue(q_st);\n"
                  "    if (cs_) dpct::queue_ptr q_cs = (dpct::queue_ptr)cs_;\n        dpct::get_current_device().destroy_queue(q_cs);",
                  "    if (st_) {\n        dpct::queue_ptr q = (dpct::queue_ptr)st_;\n        dpct::get_current_device().destroy_queue(q);\n    }\n"
                  "    if (cs_) {\n        dpct::queue_ptr q = (dpct::queue_ptr)cs_;\n        dpct::get_current_device().destroy_queue(q);\n    }")
    if s != o:
        p.write_text(s)
        changed.append(p.name)
    return changed


def step_ring_shared():
    """The expert ring is written by the CPU lane's IoQueue (the direct flow) and read by the kernels: shared, not device."""
    p = DS4 / "ds4_model.dp.cpp"
    s = p.read_text()
    if "dalloc_shared" in s:
        return False
    s = s.replace("                if (!dalloc(d, c.ring, (size_t) c.ring_n * stride, err)) return false;",
                  "                if (!dalloc_shared(d, c.ring, (size_t) c.ring_n * stride, err)) return false;   // [port] the CPU writes it", 1)
    anchor = "    template <class T> bool halloc(T *&p, size_t n, std::string &err) try {"
    helper = """    // [port] The expert ring: the CPU lane's IoQueue reads expert rows STRAIGHT into it (the "direct" flow) and the
    // kernels then read them.  On CUDA/HIP that memory is host-writable (the Windows/WDDM builds this engine comes from
    // rely on it); a SYCL device allocation is not, so this one buffer is USM shared and both sides see it.
    template <class T> bool dalloc_shared(Dev &d, T *&p, size_t n, std::string &err) try {
        void* v = nullptr;
        dpct::select_device(d.id);
        const size_t b = align256(n * sizeof(T));
        if (DPCT_CHECK_ERROR(v = (void *)sycl::malloc_shared(b, dpct::get_in_order_queue())) != 0) {
            err = "ring allocation (" + std::to_string(b >> 20) + " MiB, USM shared) failed";
            return false;
        }
        if (DPCT_CHECK_ERROR(d.st->memset(v, 0, b)) != 0 || DPCT_CHECK_ERROR(d.st->wait()) != 0) {
            sycl::free(v, dpct::get_in_order_queue());
            err = "zeroing the ring failed";
            return false;
        }
        d.allocs.push_back(v);
        d.scratch += b;
        p = (T*) v;
        return true;
    } catch (const sycl::exception &e) {
        err = std::string("ring allocation: ") + e.what();
        return false;
    }

""" + anchor
    s = s.replace(anchor, helper, 1)
    p.write_text(s)
    return True


def step_publish_fence():
    """The device's request flag has to leave the GPU: a system-scope release fence after the store (Xe LLC)."""
    p = DS4 / "ds4_kernels.dp.cpp"
    s = p.read_text()
    if "system-scope\n    // release fence follows it" in s or "DS4 port: the publish" in s:
        return False
    old = "    *(volatile unsigned*) &req->seq = seq;\n}"
    if old not in s:
        return False
    new = ("    *(volatile unsigned*) &req->seq = seq;\n"
           "    // [port] the CPU polls this flag in host memory: on Xe the store can settle in the LLC, so a system-scope\n"
           "    // release fence follows it.  (The CUDA build needs none: a host-mapped write is coherent there.  Without this\n"
           "    // the service thread never sees the request and the device's wait kernel spins for ever.)\n"
           "    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);\n}")
    p.write_text(s.replace(old, new, 1))
    return True


def step_host_free():
    """USM host allocations must be freed by the runtime: glibc free() aborts the engine at shutdown."""
    p = DS4 / "ds4_model.dp.cpp"
    s = p.read_text()
    old = "        for (void *p : host_allocs) DPCT_CHECK_ERROR(free(p));"
    if old not in s:
        return False
    new = ("        // [port] these are USM host allocations (sycl::malloc_host): glibc free() on one aborts with\n"
           "        // \"free(): invalid pointer\" at shutdown - the runtime owns them.\n"
           "        for (void *p : host_allocs) DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));")
    p.write_text(s.replace(old, new, 1))
    return True


def step_min_term():
    """dpct dropped the parentheses around the K-quant min term: v - (m0*sl + m1*sh), not v - m0*sl + m1*sh."""
    p = DS4 / "ds4_iq.dp.cpp"
    s = p.read_text()
    old = "        v = v - s.m0 * (float)sl + s.m1 * (float)sh;"
    if old not in s:
        return False
    p.write_text(s.replace(old, "        v = v - (s.m0 * (float)sl + s.m1 * (float)sh);", 1))
    return True


def step_flow_host_copy():
    """The prompt flow's staging -> ring copies have neither end in device memory: on this driver the Arc's copy
    engine hangs on such copies ("Engine reset: engine_class=bcs"; upstream Niko1221/Strata#3248 fixes the same
    hang).  Copy those on the CPU, after the slot's last use; the direct flow (reads landing in the slot) keeps the
    copy stream's barrier, because waiting for it on the CPU there stalls the pump thread (8K prompt 34.2 -> 30.2 s).
    DS4_HOST_COPY=0 restores the queue's path."""
    p = DS4 / "ds4_model.dp.cpp"
    s = p.read_text()
    if "ds4_port::host_copy_on()" in s:
        return False
    anchor = "            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;\n            if (it.direct) {"
    if anchor not in s:
        return False
    note = (
        "            const x3::ExpertLayout& lay = l.nm > 1 ? l.hlay : l.lay;\n"
        "            // [port] a copy with neither end in device memory (staging buffers -> a USM-shared ring) runs on\n"
        "            // the Arc's copy engine, which hung on it and reset the card (ds4_launch.h carries the numbers).\n"
        "            // The direct flow (the reads land in the slot) keeps the copy stream's barrier, which is already\n"
        "            // queued above: waiting for it on the CPU here stalls the pump thread.  A host copy cannot rely on\n"
        "            // that barrier, so it waits here instead.  DS4_HOST_COPY=0: the queue's path, for the A/B.\n"
        "            const bool hc = !it.direct && ds4_port::host_copy_on() && ds4_port::cpu_side(dst) &&\n"
        "                            ds4_port::cpu_side(it.src[0]) && ds4_port::cpu_side(it.src[2]);\n"
        "            if (hc && c.n_dma >= R)\n"
        "                CK(DPCT_CHECK_ERROR(c.ring_free[r]->wait_and_throw()));   // (not ordered by the barrier)\n"
        "            if (it.direct) {")
    s = s.replace(anchor, note, 1)
    # the two row parts of the expert
    old_loop = "            for (int p = 0; p < 2; ++p)\n"
    if old_loop not in s:
        return False
    s = s.replace(old_loop,
                  "            for (int p = 0; p < 2; ++p) {   // [port] a host-only copy runs on the CPU\n"
                  "                if (hc) {\n"
                  "                    std::memcpy(dst + lay.trellis[p], it.src[p], mx_read_len(l, p));\n"
                  "                    continue;\n"
                  "                }\n", 1)
    old_rows = ("                CK(DPCT_CHECK_ERROR(c.cs->memcpy(\n"
                "                    dst + lay.trellis[p], it.src[p], mx_read_len(l, p))));\n")
    if old_rows not in s:
        return False
    s = s.replace(old_rows, old_rows + "            }\n", 1)
    # the down matrix: the pitched one (a tensor-parallel member), then the plain one
    old_2d = ("                CK(DPCT_CHECK_ERROR(dpct::async_dpct_memcpy(\n"
              "                    dst + lay.trellis[2], dh, it.src[2] + (size_t)it.mi * dh,\n"
              "                    dr, dh, (size_t)g.dim, dpct::host_to_device, *(c.cs))));")
    if old_2d not in s:
        return False
    s = s.replace(old_2d,
                  "                if (hc) ds4_port::cpu_copy2d(dst + lay.trellis[2], dh, it.src[2] + (size_t) it.mi * dh,\n"
                  "                                            dr, dh, (size_t) g.dim);\n"
                  "                else " + old_2d.lstrip(), 1)
    old_p2 = ("                CK(DPCT_CHECK_ERROR(c.cs->memcpy(\n"
              "                    dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2))));")
    if old_p2 not in s:
        return False
    s = s.replace(old_p2,
                  "                if (hc) std::memcpy(dst + lay.trellis[2], it.src[2], mx_part_bytes(l, 2));\n"
                  "                else " + old_p2.lstrip(), 1)
    p.write_text(s)
    return True


def step_expert_half2():
    """DS4_EXPERT_HALF2: the expert kernel's inner product as two half2 FMAs folded into F32 once per 128 inputs,
    instead of 4 half->float conversions and 8 F32 FMAs (the B70's expert kernels are ALU-bound: 3.9-6.6 TFLOPS)."""
    p = DS4 / "vendor/exl3.dp.cpp"
    s = p.read_text()
    if "DS4_EXPERT_HALF2" in s:
        return False
    # the include the intrinsic lives in
    inc_anchor = '#include "../ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal'
    if inc_anchor in s and "ext/intel/math" not in s:
        s = s.replace(inc_anchor, inc_anchor + '\n#if defined(DS4_EXPERT_HALF2)\n'
                                 '#include <sycl/ext/intel/math.hpp>   // [port] hfma2, the expert kernel\'s half2 path\n#endif', 1)
    # the accumulators
    acc_anchor = """    float acc[kTok][2];
#pragma unroll
    for (int cc = 0; cc < kTok; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;
#endif"""
    if acc_anchor not in s:
        return False
    s = s.replace(acc_anchor, acc_anchor.replace("#endif",
        "#if defined(DS4_EXPERT_HALF2)\n"
        "    // [port] per-128-input half2 partials: the products of four weights and four activations as two half2 FMAs\n"
        "    sycl::half2 hacc[kTok][2];\n"
        "#endif\n"
        "#endif"), 1)
    # zero them each chunk
    q_anchor = """    for (int q = 0; q < nq; ++q) {
        sycl::half *xq = sm.xb + (q & 1) * (kTok * kCols);"""
    if q_anchor not in s:
        return False
    s = s.replace(q_anchor, q_anchor + """
#if defined(DS4_EXPERT_HALF2)
#pragma unroll
        for (int cc = 0; cc < kTok; ++cc) hacc[cc][0] = hacc[cc][1] = sycl::half2(0.0f);
#endif""", 1)
    # the inner product
    dot_anchor = """            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));"""
    if dot_anchor not in s:
        return False
    s = s.replace(dot_anchor, """#if defined(DS4_EXPERT_HALF2)
            // (row g: f[0], f[1]; row g + 8: f[2], f[3]; the activations 2t, 2t+1 and 2t+8, 2t+9 as two half2 loads)
            const sycl::half2 w01 = u2h(f[0]), w89 = u2h(f[1]), w01b = u2h(f[2]), w89b = u2h(f[3]);
#pragma unroll
            for (int cc = 0; cc < kTok; ++cc)
                if (cc < nc) {
                    const uint32_t *xp = reinterpret_cast<const uint32_t *>(xq + cc * kCols + s * 16 + 2 * t);
                    const sycl::half2 xa = u2h(xp[0]), xb2 = u2h(xp[4]);
                    hacc[cc][0] = sycl::ext::intel::math::hfma2(w89, xb2,
                                       sycl::ext::intel::math::hfma2(w01, xa, hacc[cc][0]));
                    hacc[cc][1] = sycl::ext::intel::math::hfma2(w89b, xb2,
                                       sycl::ext::intel::math::hfma2(w01b, xa, hacc[cc][1]));
                }
#else
""" + dot_anchor, 1)
    tail_anchor = """                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif"""
    if tail_anchor not in s:
        return False
    s = s.replace(tail_anchor, """                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif
#endif""", 1)
    # fold once per chunk
    s_anchor = """#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }"""
    if s_anchor not in s:
        return False
    s = s.replace(s_anchor, """#if defined(DS4_EXPERT_HALF2)
#pragma unroll
        for (int cc = 0; cc < kTok; ++cc) {   // the two lanes of each row's partials, summed into F32
            if (cc >= nc) continue;
            const float2 d0 = __half22float2(hacc[cc][0]), d1 = __half22float2(hacc[cc][1]);
            acc[cc][0] = __fadd_rn(acc[cc][0], __fadd_rn(d0.x, d0.y));
            acc[cc][1] = __fadd_rn(acc[cc][1], __fadd_rn(d1.x, d1.y));
        }
#endif
""" + s_anchor, 1)
    # the file's own TC condition always holds (dpct.hpp defines DPCT_COMPATIBILITY_TEMP 900): make it switchable
    tc_old = ('#if !defined(STRATA_USE_HIP) && defined(DPCT_COMPATIBILITY_TEMP) &&            \\\n'
              '    DPCT_COMPATIBILITY_TEMP >= 800\n#define EXL3_TC 1\n#endif')
    if tc_old in s:
        s = s.replace(tc_old, '#if !defined(STRATA_USE_HIP) && defined(DPCT_COMPATIBILITY_TEMP) &&            \\\n'
                               '    DPCT_COMPATIBILITY_TEMP >= 800 && !defined(DS4_EXPERT_NO_TC)\n'
                               '#define EXL3_TC 1\n#endif', 1)
    p.write_text(s)
    return True


def main() -> int:
    print("1. ds4_launch.h          :", "written" if step_launch() else "already there")
    print("2. dpct/util.hpp         :", "patched" if step_dpct_util() else "already patched")
    print("3. ds4_numerics.dp.hpp   :", "written" if step_numerics() else "already there")
    print("4. port headers          :", step_port_headers() or "already there")
    print("5. vendor/exl3_cpu.cpp   :", "written" if step_exl3_cpu() else "already there")
    print("6. ds4_packinfo.cpp      :", "written" if step_packinfo() else "already there")
    print("7. kernels               :", step_kernels() or "nothing to do")
    print("8. model/vision          :", step_model_vision() or "nothing to do")
    print("9. the expert ring       :", "shared" if step_ring_shared() else "already shared")
    print("10. the request publish  :", "fenced" if step_publish_fence() else "already fenced")
    print("11. host allocation free :", "fixed" if step_host_free() else "already fixed")
    print("12. the K-quant min term:", "fixed" if step_min_term() else "already fixed")
    print("13. the flow's host-only copies:", "cpu" if step_flow_host_copy() else "already on the cpu")
    print("14. expert kernel half2   :", "on" if step_expert_half2() else "already there")
    return 0


if __name__ == "__main__":
    sys.exit(main())
