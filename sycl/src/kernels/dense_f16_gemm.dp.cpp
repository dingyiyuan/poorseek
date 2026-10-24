// sycl/src/kernels/dense_f16_gemm.dp.cpp - see include/strata/kernels/dense_f16_gemm.hpp.
//
// The dense half's GEMM-shaped parts, as the expert path already runs its own: dequantize the weight into an FP16
// device buffer, cast the activation rows into an FP16 tile, one oneMKL FP16 GEMM with FP32 accumulation, write the
// FP32 result into the ggml tensor's buffer.  Per call:
//
//   1. w16[g][n][k] = dequant(w[g][n][k])          (Q8_0 -> FP16; F32/F16 weights are used as they are)
//   2. x16[g][m][k] = (half) x[g][m][k]            (the activations are F32 in every dense matmul here)
//   3. y[g][n][m]  = x16[g] . w16[g]^T             (oneMKL, half in / float out, f32 compute)
//
// LAYOUT.  ggml stores a weight [K, N] with K contiguous, i.e. (k, n) at n*K + k - exactly the column-major
// K x N matrix oneMKL wants with lda = K, so the GEMM is (transA=trans, transB=nontrans, m=N, n=M, k=K): A is the
// weight (N x K after the transpose), B the activation tile (K x M), C (N x M column-major) is the ggml dst [N, M]
// with ldc = N.  This is the call shape native_expert_f16.dp.cpp uses and measures 60-70 TFLOPS with.
//
// The dequantizer, the cast and every GEMM run on the caller's IN-ORDER queue, so the sequence needs no dependency
// token: the cast of x completes before the GEMM that reads it, and the GEMM before whatever the caller submits
// next.  The scratch is allocated once and grown, never reallocated per call; a failed allocation is `false`, so the
// caller falls back to its own op rather than running on garbage.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "strata/kernels/dense_f16_gemm.hpp"

namespace strata::kernels {
namespace {

using half_t = sycl::half;

// ---------------------------------------------------------------- block layouts (ggml-common.h, MIT)
struct block_q8_0 {                 // 32 weights: fp16 scale, 32 signed bytes
    half_t d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "wrong q8_0 block size");

class k_dq_q8_0;
void dq_q8_0(sycl::queue& q, const uint8_t* src, int64_t elems, half_t* dst, int64_t dst_off) {
    const int64_t blocks = elems / 32;
    q.parallel_for<k_dq_q8_0>(sycl::range<1>((size_t) blocks), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const block_q8_0* b = ((const block_q8_0*) src) + i;
        const float d = (float) b->d;
        half_t* o = dst + dst_off + i * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) o[j] = half_t(d * (float) b->qs[j]);
    });
}

class k_cast_f32_f16;
/// x16[(g*M + m)*K + k] = (half) x[g*x_group + m*x_row + k]
void cast_f32_f16(sycl::queue& q, const float* src, int64_t K, int64_t rows_total, int64_t x_row, int64_t x_group,
                  int64_t M, half_t* dst, int64_t dst_off) {
    q.parallel_for<k_cast_f32_f16>(sycl::range<2>((size_t) K, (size_t) rows_total), [=](sycl::id<2> id) {
        const int64_t k = (int64_t) id[0], r = (int64_t) id[1];
        const int64_t g = r / M, m = r % M;
        dst[dst_off + r * K + k] = half_t(src[g * x_group + m * x_row + k]);
    });
}

/// A weight that is already dequantized: the dense half is LAYER-MAJOR (one layer's graph is submitted once per
/// sub-batch of columns, ~5x at a 8192 prompt), so the SAME weight tensor is asked for ~5 times in a row.  Without
/// this cache every one of those calls re-reads the q8_0 blocks and re-writes the f16 matrix, which measured
/// 2.1 s/prompt - more than the GEMMs it feeds (0.8 s).  The key is the weight tensor's identity and shape; the
/// buffer is released when another weight needs the room.
struct WEntry {
    const void* key = nullptr;   ///< the ggml tensor the data came from (stable for the model's lifetime)
    const void* data = nullptr;  ///< its data pointer, also part of the key
    int64_t K = 0, N = 0, groups = 0;
    size_t elems = 0;
    half_t* buf = nullptr;
    uint64_t used = 0;
};

/// The oneMKL buffers this route needs, kept across calls.  One dense layer at a time uses it (the graph compute is
/// single-threaded), but the mutex is there so a second caller cannot grow the buffers under the first.
struct Scratch {
    half_t* x = nullptr;
    size_t cap_x = 0;
    std::vector<WEntry> wc;   ///< dequantized weights, LRU
    size_t wc_bytes = 0;
    uint64_t tick = 0;
    sycl::queue* q = nullptr;
    dpct::blas::descriptor* h = nullptr;
    std::mutex mu;
};

Scratch& scratch() {
    static Scratch s;
    return s;
}

/// How much VRAM the dequantized-weight cache may hold.  384 MiB covers one layer's dense weights in FP16 (the
/// largest single matrix, q_b, is 84 MiB: 32768 x 1280 x 2), which is what a layer-major pass re-asks for.
static size_t wcache_budget() {
    static const size_t mb = [] {
        const char * e = std::getenv("STRATA_DENSE_GEMM_WCACHE_MB");
        return (size_t) (e != nullptr ? std::atoll(e) : 384);
    }();
    return mb << 20;
}

int64_t scratch_bytes() {
    std::lock_guard<std::mutex> g(scratch().mu);
    Scratch& s = scratch();
    return (int64_t) s.cap_x * (int64_t) sizeof(half_t) + (int64_t) s.wc_bytes;
}

bool ensure(sycl::queue& q, Scratch& s, size_t need_x) {
    bool ok = true;
    if (need_x > s.cap_x) {
        if (s.x) sycl::free(s.x, q);
        s.x = sycl::malloc_device<half_t>(need_x, q);
        s.cap_x = s.x ? need_x : 0;
        ok = ok && s.x != nullptr;
    }
    if (!ok) {
        std::fprintf(stderr, "strata: dense GEMM: %.1f MiB of FP16 activation tile did not fit VRAM; the ggml op "
                             "stays in use\n", (double) need_x * 2.0 / 1048576.0);
        return false;
    }
    if (s.h == nullptr) {
        s.h = new dpct::blas::descriptor();
        s.h->set_math_mode(dpct::blas::math_mode::mm_default);
    }
    if (s.q != &q) {
        s.h->set_queue(&q);
        s.q = &q;
    }
    return true;
}

/// STRATA_DENSE_GEMM_PROF=1 (diagnostic, off by default): per-stage wall time, host-serialized at each boundary.
/// Inert (and free) when unset; it is here because "the GEMM is slow" and "the dequantizer is slow" look identical
/// from the prompt's wall clock.
struct StageProf {
    bool on = std::getenv("STRATA_DENSE_GEMM_PROF") != nullptr;
    double dq = 0, cast = 0, gemm = 0;
    int64_t calls = 0, flops = 0, dq_bytes = 0, cast_elems = 0;
    ~StageProf() {
        if (!on || calls == 0) return;
        std::fprintf(stderr,
                     "strata: dense gemm prof: %lld calls, %.2f TFLOP: dequant %.0f ms (%.1f GB), cast %.0f ms "
                     "(%.2f G elem), gemm %.0f ms = %.1f TFLOPS on the gemm alone (total %.0f ms incl. both copies)\n",
                     (long long) calls, (double) flops / 1e12, dq, (double) dq_bytes / 1e9, cast,
                     (double) cast_elems / 1e9, gemm, gemm > 0 ? (double) flops / 1e12 / (gemm / 1000.0) : 0.0,
                     dq + cast + gemm);
    }
};
StageProf& sprof() {
    static StageProf p;
    return p;
}

/// The dequantized form of `a`'s weight, dequantizing it on a miss.  Null = it did not fit (the caller refuses).
half_t* weight_slot(sycl::queue& q, Scratch& s, const DenseGemmArgs& a, size_t elems, bool* fresh) {
    const size_t need = elems * sizeof(half_t);
    const uint64_t now = ++s.tick;
    *fresh = false;
    for (WEntry& e : s.wc)
        if (e.key == a.key && e.data == a.w && e.K == a.K && e.N == a.N && e.groups == a.groups) {
            e.used = now;
            return e.buf;
        }
    const size_t budget = wcache_budget();
    if (budget == 0 || need > budget) return nullptr;
    while (s.wc_bytes + need > budget && !s.wc.empty()) {   // evict the least recently used
        size_t v = 0;
        for (size_t i = 1; i < s.wc.size(); ++i)
            if (s.wc[i].used < s.wc[v].used) v = i;
        sycl::free(s.wc[v].buf, q);
        s.wc_bytes -= s.wc[v].elems * sizeof(half_t);
        s.wc.erase(s.wc.begin() + (long) v);
    }
    half_t* buf = sycl::malloc_device<half_t>(elems, q);
    if (buf == nullptr) {
        std::fprintf(stderr, "strata: dense GEMM: a %.1f MiB dequantized weight did not fit VRAM; the ggml op stays in "
                             "use for it\n", (double) need / 1048576.0);
        return nullptr;
    }
    s.wc.push_back(WEntry{a.key, a.w, a.K, a.N, a.groups, elems, buf, now});
    s.wc_bytes += need;
    *fresh = true;
    return buf;
}

}  // namespace

int64_t dense_gemm_f16_scratch_bytes() { return scratch_bytes(); }

bool dense_gemm_f16(void* qvp, const DenseGemmArgs& a) {
    if (qvp == nullptr || a.w == nullptr || a.x == nullptr || a.y == nullptr) return false;
    if (a.K <= 0 || a.N <= 0 || a.M <= 0 || a.groups <= 0) return false;
    // The shapes this route is for.  Anything else is the caller's op, not a slower version of this one.
    if (a.wtype != DENSE_W_Q8_0 && a.wtype != DENSE_W_F16 && a.wtype != DENSE_W_F32) return false;
    if (a.xtype != DENSE_W_F32) return false;                     // activations are F32 in every dense matmul here
    if (a.wtype == DENSE_W_Q8_0 && (a.K % 32) != 0) return false;  // the q8_0 block is 32 values
    if (a.w_group == 0 && a.groups > 1) return false;
    sycl::queue& q = *static_cast<sycl::queue*>(qvp);
    const bool half_in = a.wtype != DENSE_W_F32;   // an F32 weight keeps an all-FP32 GEMM (no cast, no rounding)

    const size_t w_elems = (size_t) (a.K * a.N * a.groups);
    const size_t x_elems = (size_t) (a.K * a.M * a.groups);
    if (a.wtype == DENSE_W_Q8_0 && w_elems * sizeof(half_t) > wcache_budget()) return false;
    Scratch& s = scratch();
    if (half_in) {
        std::lock_guard<std::mutex> g(s.mu);
        if (!ensure(q, s, x_elems)) return false;
    }
    const float alpha = 1.0f, beta = 0.0f;
    StageProf& SP = sprof();
    auto now_ms = [] {
        return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch()).count() / 1e6;
    };
    double t0 = SP.on ? now_ms() : 0.0;
    if (SP.on) SP.calls++;

    // 1. the weights: Q8_0 is dequantized into FP16 once and kept (the same tensor is asked for again by every
    //    sub-batch of the layer); an FP16 weight is already the GEMM's type.
    half_t* wbuf = nullptr;
    if (a.wtype == DENSE_W_Q8_0) {
        std::lock_guard<std::mutex> g(s.mu);
        bool fresh = false;
        wbuf = weight_slot(q, s, a, w_elems, &fresh);
        if (wbuf == nullptr) return false;
        if (fresh) {
            for (int64_t grp = 0; grp < a.groups; ++grp) {
                // w_group is counted in ELEMENTS; a q8_0 block is 34 bytes for 32 of them
                const uint8_t* src = (const uint8_t*) a.w + (size_t) (a.w_group * (uint64_t) grp / 32) * 34;
                dq_q8_0(q, src, a.K * a.N, wbuf, grp * a.K * a.N);
            }
            if (SP.on) {
                q.wait_and_throw();
                const double t = now_ms();
                SP.dq += t - t0;
                SP.dq_bytes += (int64_t) ((size_t) w_elems / 32 * 34);
                t0 = t;
            }
        }
    }

    // 2. the activations, cast into the tile the GEMM reads (contiguous K rows, M tokens per group)
    if (half_in) cast_f32_f16(q, (const float*) a.x, a.K, a.M * a.groups, a.x_row, a.x_group, a.M, s.x, 0);
    if (SP.on && half_in) {
        q.wait_and_throw();
        const double t = now_ms();
        SP.cast += t - t0;
        SP.cast_elems += (int64_t) x_elems;
        t0 = t;
    }

    // 3. one GEMM per group: C(N x M) = op(A)(N x K) . B(K x M), column-major views of the row-major ggml layouts
    for (int64_t grp = 0; grp < a.groups; ++grp) {
        void* yg = (void*) ((float*) a.y + (size_t) (a.y_group * (uint64_t) grp));
        if (!half_in) {
            const float* wg = (const float*) a.w + (size_t) (a.w_group * (uint64_t) grp);
            const float* xg = (const float*) a.x + (size_t) (a.x_group * (uint64_t) grp);
            dpct::blas::gemm(s.h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) a.N,
                             (int) a.M, (int) a.K, &alpha, wg, dpct::library_data_t::real_float, (int) a.K, xg,
                             dpct::library_data_t::real_float, (int) a.x_row, &beta, yg,
                             dpct::library_data_t::real_float, (int) a.y_row, dpct::compute_type::f32);
            continue;
        }
        const void* wg = a.wtype == DENSE_W_Q8_0
                             ? (const void*) (wbuf + (size_t) grp * (size_t) (a.K * a.N))
                             : (const void*) ((const half_t*) a.w + (size_t) (a.w_group * (uint64_t) grp));
        dpct::blas::gemm(s.h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) a.N, (int) a.M,
                         (int) a.K, &alpha, wg, dpct::library_data_t::real_half, (int) a.K, s.x + grp * a.M * a.K,
                         dpct::library_data_t::real_half, (int) a.K, &beta, yg, dpct::library_data_t::real_float,
                         (int) a.y_row, dpct::compute_type::f32);
    }
    if (SP.on) {
        q.wait_and_throw();
        const double t = now_ms();
        SP.gemm += t - t0;
        SP.flops += 2 * (int64_t) a.M * a.N * a.K * a.groups;
    }
    return true;
}

}  // namespace strata::kernels
