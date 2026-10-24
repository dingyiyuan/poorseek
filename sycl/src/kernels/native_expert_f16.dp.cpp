// sycl/src/kernels/native_expert_f16.dp.cpp - see include/strata/kernels/native_expert_f16.hpp.
//
// The chunk (prompt) path's routed experts, as dequantize + oneMKL FP16 GEMM instead of the row-at-a-time grouped
// kernel.  Per expert, in one call:
//   1. gate/up and down are unpacked from their Q2_0 / Q3_K / PTQ1_0 blocks into FP16 device buffers
//      ([gate rows 0..n_ff) | up rows n_ff..2n_ff)] and [n_embd x n_ff]);
//   2. the expert's tokens' activation rows are gathered into an FP16 tile [rows x n_embd];
//   3. one oneMKL f16 GEMM (FP32 accumulate): GU[rows x 2 n_ff] = X . W_gu^T;
//   4. SwiGLU (the engine's clamp included) into an FP16 H[rows x n_ff];
//   5. one oneMKL f16 GEMM: Y[rows x n_embd] = H . W_d^T;
//   6. the rows are scattered into `parts` at the entries' own rows (`dst`).
//
// Everything runs on the caller's in-order queue, so the dequantizer, both GEMMs and the scatter sequence with the
// chunk path's copies and with each other without a dependency token.  The scratch is allocated once (grown, never
// reallocated per call) and a failure to allocate is reported as `false`, so the caller keeps the grouped kernel.
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cstring>

#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_expert_f16.hpp"

namespace strata::kernels {
namespace {

using half_t = sycl::half;

// ---------------------------------------------------------------- block layouts (ggml-common.h, MIT)
// Transcribed rather than included: ggml-common.h needs GGML_COMMON_DECL_* and dpct_type_* and this TU takes nothing
// else from it.  The static_asserts are the header's own (and iq_kernels_native_expert.dp.cpp's).
struct block_q2_0 {                 // 64 weights: fp16 scale, 2 bits each
    half_t d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size");

struct block_q3_K {                 // 256 weights: 3 bits, 6-bit scales
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half_t d;
};
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size");

struct block_ptq1_0 {               // 128 ternary weights in base 3, fp16 scale (28 bytes)
    uint8_t qs[24];
    uint8_t qh[2];
    half_t d;
};
static_assert(sizeof(block_ptq1_0) == 28, "wrong ptq1_0 block size");

int64_t block_elems(const int ty) { return ty == 42 ? 64 : ty == 11 ? 256 : ty == 143 ? 128 : 0; }
int64_t block_bytes(const int ty) { return ty == 42 ? 18 : ty == 11 ? 110 : ty == 143 ? 28 : 0; }

// ---------------------------------------------------------------- dequantizers (one per format)
// Q2_0: 64 values per block, value = d * ((2-bit field) - 1); one work item per 8 values = one int16 word, the same
// grouping `dot_q2_0` reads (fields 0..7 of the word are the 8 values in order).
class k_dq_q2_0;
void dq_q2_0(sycl::queue& q, const uint8_t* src, int64_t rows, int64_t K, half_t* dst) {
    const int64_t words = K / 8, wpb = 8;   // int16 words per row, words per block (64 values = 8 words)
    q.parallel_for<k_dq_q2_0>(sycl::range<1>((size_t) (rows * words)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t w = i % words, r = i / words;
        const block_q2_0* b = (const block_q2_0*) src + r * (words / wpb) + w / wpb;
        const int t = (int) ((const int16_t*) b->qs)[w % wpb];
        const float d = (float) b->d;
        half_t* o = dst + r * K + w * 8;
#pragma unroll
        for (int f = 0; f < 8; ++f) o[f] = half_t(d * (float) (((t >> (2 * f)) & 3) - 1));
    });
}

// Q3_K: 256 values per block; value = d * (6-bit scale - 32) * (2-bit field - (high-bit mask clear ? 4 : 0)), the
// arithmetic of `dot_q3_K`.  One work item per 16 values: gg = g % 16 -> half n = gg / 8, shift group j = (gg % 8) / 2,
// sub s = gg % 2; the 16 values of a group are qs[32 n + 16 s + l] >> 2 j, l = 0..15.
class k_dq_q3_K;
void dq_q3_K(sycl::queue& q, const uint8_t* src, int64_t rows, int64_t K, half_t* dst) {
    const int64_t groups = K / 16;
    q.parallel_for<k_dq_q3_K>(sycl::range<1>((size_t) (rows * groups)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t g = i % groups, r = i / groups;
        const block_q3_K* b = (const block_q3_K*) src + r * (groups / 16) + g / 16;
        const int gg = (int) (g % 16), n = gg >> 3, j = (gg & 7) >> 1, s = gg & 1;
        // the 16 6-bit scales, ggml's unpack of scales[12] (dequantize_row_q3_K's kmask1 / kmask2 shuffle)
        uint32_t aux[4] = {0, 0, 0, 0};
        std::memcpy(aux, b->scales, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0f0f0f0fu) | (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0f0f0f0fu) | (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0f0f0f0fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0f0f0f0fu) | (((tmp >> 2) & 0x03030303u) << 4);
        const int8_t* sc = (const int8_t*) aux;
        const float dl = (float) b->d * (float) ((int) sc[n * 8 + j * 2 + s] - 32);
        // qs advances by 32 bytes per 128-value half; the HIGH-BIT MASK does not (its 32 bytes are 256 bits, one per
        // value: byte l + 16 s, bit j of the first half / j + 4 of the second - dequantize_row_q3_K's `m`, which keeps
        // shifting across the two halves while `hm` stays put).
        const uint8_t* qs = b->qs + 32 * n + 16 * s;
        const uint8_t* hm = b->hmask + 16 * s;
        const uint8_t m = (uint8_t) (1u << (j + 4 * n));
        half_t* o = dst + r * K + g * 16;
#pragma unroll
        for (int l = 0; l < 16; ++l)
            o[l] = half_t(dl * (float) ((int) ((qs[l] >> (2 * j)) & 3) - ((hm[l] & m) ? 0 : 4)));
    });
}

// PTQ1_0: 128 ternary values per block, value = d * (trit - 1), the trits packed base 3 (llama-tq's layout):
//   element e < 80: byte 4 w + jj, trit t      (e = 16 t + 4 w + jj, w = (e % 16) / 4, jj = e % 4)
//   element e < 120: byte 16 + 4 w + jj        (e = 80 + 8 t + 4 w + jj, w = ((e - 80) % 8) / 4, jj = (e - 80) % 4)
//   element e >= 120: byte 24 + h              (e = 120 + 2 t + h, h = (e - 120) % 2, t = (e - 120) / 2)
// and trit = (((byte * 3^t) mod 256) * 3) >> 8 - `ptq_trit`, byte for byte.
constexpr uint32_t kP3[5] = {1u, 3u, 9u, 27u, 81u};
inline int ptq_at(const uint8_t* blk, const int e) {
    int off, t;
    if (e < 80) {
        t = e >> 4;
        off = 4 * ((e & 15) >> 2) + (e & 3);
    } else if (e < 120) {
        const int e2 = e - 80;
        t = e2 >> 3;
        off = 16 + 4 * ((e2 & 7) >> 2) + (e2 & 3);
    } else {
        const int e3 = e - 120;
        t = e3 >> 1;
        off = 24 + (e3 & 1);
    }
    const uint32_t v = ((uint32_t) blk[off] * kP3[t]) & 0xFFu;
    return (int) ((v * 3u) >> 8);
}
class k_dq_ptq1_0;
void dq_ptq1_0(sycl::queue& q, const uint8_t* src, int64_t rows, int64_t K, half_t* dst) {
    const int64_t grp = K / 8;   // 8 values per work item; 8 divides 128, so a group never spans two blocks
    q.parallel_for<k_dq_ptq1_0>(sycl::range<1>((size_t) (rows * grp)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t g = i % grp, r = i / grp;
        const int e0 = (int) (g * 8);
        const block_ptq1_0* b = (const block_ptq1_0*) src + r * (K / 128) + e0 / 128;
        const float d = (float) b->d;
        const uint8_t* blk = (const uint8_t*) b;
        const int lo = e0 & 127;
        half_t* o = dst + r * K + e0;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = half_t(d * (float) (ptq_at(blk, lo + j) - 1));
    });
}

// ---------------------------------------------------------------- the FP16 path's other kernels
// The expert's rows' activation rows gathered out of the token-major activation buffer: xa[r][j] = x[tok[r]][j].
class k_f16_gather;
void gather_x(sycl::queue& q, const float* x, const int32_t* tok, int64_t rows, int64_t K, half_t* dst) {
    const int64_t grp = K / 8;
    q.parallel_for<k_f16_gather>(sycl::range<1>((size_t) (rows * grp)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t c = (i % grp) * 8, r = i / grp;
        const float* s = x + (size_t) tok[r] * K + c;
        half_t* o = dst + r * K + c;
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = half_t(s[j]);
    });
}

// SwiGLU with the engine's clamps (native_gu_kernel / swiglu_entries_kernel): h = silu(min(g, lim)) * clamp(u, -lim, lim).
// gu is row-major [rows][2 n_ff] with the gate in [0, n_ff) and the up projection in [n_ff, 2 n_ff).
class k_f16_swiglu;
void swiglu_f16(sycl::queue& q, const float* gu, int64_t rows, int64_t n_ff, float lim, half_t* h) {
    const int64_t grp = n_ff / 8;
    q.parallel_for<k_f16_swiglu>(sycl::range<1>((size_t) (rows * grp)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t c = (i % grp) * 8, r = i / grp;
        const float* g = gu + r * 2 * n_ff + c;
        const float* u = g + n_ff;
        half_t* o = h + r * n_ff + c;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float gg = sycl::fmin(g[j], lim);
            const float uu = sycl::fmin(sycl::fmax(u[j], -lim), lim);
            o[j] = half_t((gg / (1.0f + sycl::exp(-gg))) * uu);
        }
    });
}

// The product's rows into `parts` at the entries' own rows (the chunk path's row index, `dst`).
class k_f16_scatter;
void scatter_rows(sycl::queue& q, const float* y, const int32_t* dst, int64_t rows, int64_t H, float* parts) {
    const int64_t grp = H / 8;
    q.parallel_for<k_f16_scatter>(sycl::range<1>((size_t) (rows * grp)), [=](sycl::id<1> id) {
        const int64_t i = (int64_t) id;
        const int64_t c = (i % grp) * 8, r = i / grp;
        *reinterpret_cast<sycl::vec<float, 8>*>(parts + (size_t) dst[r] * H + c) =
            *reinterpret_cast<const sycl::vec<float, 8>*>(y + r * H + c);
    });
}

// ---------------------------------------------------------------- scratch and the oneMKL handle
struct Scratch {
    int64_t rows = 0, n_embd = 0, n_ff = 0;
    half_t* dq_gu = nullptr;   // [2 n_ff][n_embd] halves: gate rows then up rows
    half_t* dq_d = nullptr;    // [n_embd][n_ff] halves
    half_t* xa = nullptr;      // [rows][n_embd] halves
    half_t* h = nullptr;       // [rows][n_ff] halves
    float* gu = nullptr;       // [rows][2 n_ff] floats
    float* ya = nullptr;       // [rows][n_embd] floats
    dpct::blas::descriptor_ptr handle = nullptr;
    sycl::queue* q = nullptr;
    bool dead = false;         // an allocation failed: stay on the grouped kernel for the rest of the run
    int dumped = 0;            // STRATA_F16_DUMP: the stage dump happens once
};
Scratch& scratch() {
    static Scratch s;
    return s;
}

// 2048 rows covers every routed expert at the widest prompt this engine runs (mean 232 at 8192 tokens; the router's
// tail is gamma-shaped).  Beyond it the grouped kernel takes the expert back rather than growing into the dense half.
constexpr int64_t kMaxRows = 2048;

void free_scratch(sycl::queue& q, Scratch& s) {
    if (s.dq_gu) sycl::free(s.dq_gu, q);
    if (s.dq_d) sycl::free(s.dq_d, q);
    if (s.xa) sycl::free(s.xa, q);
    if (s.h) sycl::free(s.h, q);
    if (s.gu) sycl::free(s.gu, q);
    if (s.ya) sycl::free(s.ya, q);
    s.dq_gu = nullptr;
    s.dq_d = nullptr;
    s.xa = nullptr;
    s.h = nullptr;
    s.gu = nullptr;
    s.ya = nullptr;
    s.rows = 0;
}

bool ensure_scratch(sycl::queue& q, Scratch& s, int64_t rows, int64_t n_embd, int64_t n_ff) {
    if (s.dead || rows > kMaxRows) return false;
    if (s.rows >= rows && s.n_embd == n_embd && s.n_ff == n_ff) return true;
    const int64_t want = std::min<int64_t>(kMaxRows, std::max<int64_t>(rows, 256));
    free_scratch(q, s);
    s.rows = want;
    s.n_embd = n_embd;
    s.n_ff = n_ff;
    s.dq_gu = sycl::malloc_device<half_t>((size_t) (2 * n_ff * n_embd), q);
    s.dq_d = sycl::malloc_device<half_t>((size_t) (n_embd * n_ff), q);
    s.xa = sycl::malloc_device<half_t>((size_t) (want * n_embd), q);
    s.h = sycl::malloc_device<half_t>((size_t) (want * n_ff), q);
    s.gu = sycl::malloc_device<float>((size_t) (want * 2 * n_ff), q);
    s.ya = sycl::malloc_device<float>((size_t) (want * n_embd), q);
    if (!s.dq_gu || !s.dq_d || !s.xa || !s.h || !s.gu || !s.ya) {
        std::fprintf(stderr, "strata: prompt experts: %lld rows of FP16 expert scratch did not fit VRAM; staying on the "
                             "grouped kernel\n", (long long) want);
        free_scratch(q, s);
        s.dead = true;
        return false;
    }
    return true;
}

}  // namespace

bool native_expert_f16_supported(const NativeExpertLayout& L) noexcept {
    const int gt = L.gu_type, dt = L.d_type;
    if (gt != 42 && gt != 11 && gt != 143) return false;
    if (dt != 42 && dt != 11 && dt != 143) return false;
    if (L.up_type >= 0 && L.up_type != L.gu_type) return false;   // MiMo GSQ-RCO's mixed gate/up
    if (L.tail_off != 0) return false;                            // NVFP4's per-expert scales
    if (L.n_embd <= 0 || L.n_ff <= 0) return false;
    // the GEMM tiles and the dequantizers' work-item groups: 8 values, block sizes 64 / 256 / 128
    const int64_t gu_be = block_elems(gt), d_be = block_elems(dt);
    if (L.n_embd % 256 != 0 || L.n_ff % 256 != 0) return false;   // 256 = the coarsest block (Q3_K)
    if (L.n_embd % 8 != 0 || L.n_ff % 8 != 0) return false;
    if (L.n_embd < 256 || L.n_ff < 256) return false;
    if (L.gu_row != (size_t) (L.n_embd / gu_be * block_bytes(gt))) return false;
    if (L.d_row != (size_t) (L.n_ff / d_be * block_bytes(dt))) return false;
    return true;
}

// STRATA_F16_PROF: per-stage wall time (host-serialized at each boundary; a diagnostic, never on by default)
namespace {
struct StageProf {
    bool on = std::getenv("STRATA_F16_PROF") != nullptr;
    double dq = 0, gu_dq = 0, xg = 0, g1 = 0, sw = 0, d_dq = 0, g2 = 0, sc = 0;
    double host = 0;   // host wall inside native_expert_f16 (submission cost), independent of stage sums
    double h_gu = 0, h_xa = 0, h_g1 = 0, h_sw = 0, h_wd = 0, h_g2 = 0, h_sc = 0;   // host-side cost per stage
    int64_t calls = 0, rows = 0;
    ~StageProf() {
        if (!on || calls == 0) return;
        std::fprintf(stderr,
                     "strata: f16 stage us/expert over %lld calls (%lld rows/expert): wgu %.1f  xa %.1f  gemm_gu %.1f  "
                     "swiglu %.1f  wd %.1f  gemm_d %.1f  scatter %.1f  | total %.1f us/expert, %.2f us/entry; host "
                     "submit %.0f ms (%.0f us/expert); host split us/expert: wgu %.0f xa %.0f g1 %.0f sw %.0f wd %.0f "
                     "g2 %.0f sc %.0f\n",
                     (long long) calls, (long long) (rows / calls), gu_dq / calls, xg / calls, g1 / calls, sw / calls,
                     d_dq / calls, g2 / calls, sc / calls, (gu_dq + xg + g1 + sw + d_dq + g2 + sc) / calls,
                     (gu_dq + xg + g1 + sw + d_dq + g2 + sc) / (double) std::max<int64_t>(1, rows), host / 1000.0,
                     host / calls, h_gu / calls, h_xa / calls, h_g1 / calls, h_sw / calls, h_wd / calls, h_g2 / calls,
                     h_sc / calls);
    }
};
StageProf& sprof() {
    static StageProf p;
    return p;
}
inline double now_us() {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;
}
}  // namespace

bool native_expert_f16(const NativeExpertLayout& L, const void* blob, const float* x_tokens, const int32_t* tok,
                       const int32_t* dst, int64_t rows, float* parts, void* stream) {
    if (!native_expert_f16_supported(L) || rows <= 0 || blob == nullptr || x_tokens == nullptr || tok == nullptr ||
        dst == nullptr || parts == nullptr)
        return false;
    sycl::queue& q = *strata::q_of(stream);
    Scratch& s = scratch();
    const int64_t want_rows = std::min<int64_t>(rows, 1024);   // the row tile: a big expert runs in several of these
    if (!ensure_scratch(q, s, want_rows, L.n_embd, L.n_ff)) return false;
    if (!s.handle) {
        s.handle = new dpct::blas::descriptor();
        s.handle->set_math_mode(dpct::blas::math_mode::mm_default);
    }
    if (s.q != &q) {
        s.handle->set_queue(&q);
        s.q = &q;
    }
    const uint8_t* base = (const uint8_t*) blob;
    const int64_t E = L.n_embd, F = L.n_ff;
    StageProf& SP = sprof();
    double t_stage = 0;
    const double t_enter = SP.on ? now_us() : 0;
    struct HostAdd { StageProf& SP; double t0; ~HostAdd() { SP.host += now_us() - t0; } } host_add{SP, t_enter};
    double th = 0;
    auto hmark = [&](double* into) {   // host cost of the block just submitted (no waits)
        if (!SP.on) return;
        const double t = now_us();
        if (into) *into += t - th;
        th = t;
    };
    auto mark = [&](double* into) {
        if (!SP.on) return;
        q.wait_and_throw();
        const double t = now_us();
        if (into) *into += t - t_stage;
        t_stage = t;
    };
    if (SP.on) { SP.calls++; SP.rows += rows; q.wait_and_throw(); t_stage = now_us(); th = t_stage; }
    // 1. weights: gate rows [0, n_ff) then up rows [n_ff, 2 n_ff) in one buffer, and the down matrix
    {
        const int gt = L.gu_type;
        auto dq = [&](const uint8_t* src, int64_t rws, int64_t K, half_t* out) {
            if (gt == 42) dq_q2_0(q, src, rws, K, out);
            else if (gt == 11) dq_q3_K(q, src, rws, K, out);
            else dq_ptq1_0(q, src, rws, K, out);
        };
        // gate rows then up rows: the blob's two regions are adjacent (up_off = n_ff * gu_row), so ONE launch covers
        // both matrices and the destination is contiguous
        dq(base, 2 * F, E, s.dq_gu);
    }
    mark(&SP.gu_dq);
    hmark(&SP.h_gu);
    {
        const int dt = L.d_type;
        const uint8_t* src = base + L.down_off;
        if (dt == 42) dq_q2_0(q, src, E, F, s.dq_d);
        else if (dt == 11) dq_q3_K(q, src, E, F, s.dq_d);
        else dq_ptq1_0(q, src, E, F, s.dq_d);
    }
    mark(&SP.d_dq);
    hmark(&SP.h_wd);
    // 2..6 per ROW TILE.  The weights above are dequantized once for the whole expert; an expert with more rows than
    // the scratch (a wide prompt's hottest expert) runs here as several tiles of the same weights, never as a fallback.
    for (int64_t r0 = 0; r0 < rows; r0 += want_rows) {
    const int64_t nr = std::min<int64_t>(want_rows, rows - r0);
    const float* xt = x_tokens;
    const int32_t* tk = tok + r0;
    const int32_t* ds = dst + r0;
    gather_x(q, xt, tk, nr, E, s.xa);
    mark(&SP.xg);
    hmark(&SP.h_xa);
    // 3. GU[rows][2 n_ff] = X[rows][n_embd] . W_gu[2 n_ff][n_embd]^T   (oneMKL's column-major view of row-major inputs)
    {
        const float alpha = 1.0f, beta = 0.0f;
        dpct::blas::gemm(s.handle, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) (2 * F),
                         (int) nr, (int) E, &alpha, s.dq_gu, dpct::library_data_t::real_half, (int) E, s.xa,
                         dpct::library_data_t::real_half, (int) E, &beta, s.gu, dpct::library_data_t::real_float,
                         (int) (2 * F), dpct::compute_type::f32);
    }
    // 4. SwiGLU -> FP16 H
    mark(&SP.g1);
    hmark(&SP.h_g1);
    swiglu_f16(q, s.gu, nr, F, L.swiglu_limit, s.h);
    mark(&SP.sw);
    hmark(&SP.h_sw);
    // 5. Y[rows][n_embd] = H[rows][n_ff] . W_d[n_embd][n_ff]^T
    {
        const float alpha = 1.0f, beta = 0.0f;
        dpct::blas::gemm(s.handle, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, (int) E, (int) nr,
                         (int) F, &alpha, s.dq_d, dpct::library_data_t::real_half, (int) F, s.h,
                         dpct::library_data_t::real_half, (int) F, &beta, s.ya, dpct::library_data_t::real_float,
                         (int) E, dpct::compute_type::f32);
    }
    mark(&SP.g2);
    hmark(&SP.h_g2);
    // 6. the entries' own rows
    scatter_rows(q, s.ya, ds, nr, E, parts);
    mark(&SP.sc);
    hmark(&SP.h_sc);
    }  // row tile
    // STRATA_F16_DUMP=<dir>: stage-by-stage bytes of the first call (a parity harness for this path, no cost when unset)
    static const char* dump_dir = std::getenv("STRATA_F16_DUMP");
    if (dump_dir != nullptr && s.dumped == 0) {
        s.dumped = 1;
        q.wait_and_throw();   // the dump is host-side: every stage buffer must be complete
        auto write = [&](const char* nm, const void* dev, size_t bytes) {
            char path[1024];
            std::snprintf(path, sizeof path, "%s/%s.bin", dump_dir, nm);
            std::vector<float> f;   // (padding type only: the bytes are written raw)
            (void) f;
            if (FILE* fp = std::fopen(path, "wb")) {
                std::vector<uint8_t> hb(bytes);
                q.memcpy(hb.data(), dev, bytes).wait_and_throw();
                std::fwrite(hb.data(), 1, bytes, fp);
                std::fclose(fp);
            }
        };
        write("blob", blob, L.bytes);
        write("xa", s.xa, (size_t) (std::min<int64_t>(rows, want_rows) * E * 2));
        write("dq_gu", s.dq_gu, (size_t) (2 * F * E * 2));
        write("dq_d", s.dq_d, (size_t) (E * F * 2));
        write("gu", s.gu, (size_t) (std::min<int64_t>(rows, want_rows) * 2 * F * 4));
        write("h", s.h, (size_t) (std::min<int64_t>(rows, want_rows) * F * 2));
        write("ya", s.ya, (size_t) (std::min<int64_t>(rows, want_rows) * E * 4));
        char path[1024];
        std::snprintf(path, sizeof path, "%s/meta.txt", dump_dir);
        if (FILE* fp = std::fopen(path, "w")) {
            std::fprintf(fp, "gu_type %d\nd_type %d\nn_embd %lld\nn_ff %lld\nrows %lld\nup_off %zu\ndown_off %zu\n"
                             "gu_row %zu\nd_row %zu\nswiglu_limit %.9g\n",
                         L.gu_type, L.d_type, (long long) E, (long long) F, (long long) rows, L.up_off, L.down_off,
                         L.gu_row, L.d_row, (double) L.swiglu_limit);
            int32_t* ht = (int32_t*) std::malloc((size_t) rows * 4);
            int32_t* hd = (int32_t*) std::malloc((size_t) rows * 4);
            if (ht && hd) {
                q.memcpy(ht, tok, (size_t) rows * 4).wait_and_throw();
                q.memcpy(hd, dst, (size_t) rows * 4).wait_and_throw();
                std::fprintf(fp, "tok");
                for (int64_t r = 0; r < rows; ++r) std::fprintf(fp, " %d", ht[r]);
                std::fprintf(fp, "\ndst");
                for (int64_t r = 0; r < rows; ++r) std::fprintf(fp, " %d", hd[r]);
                std::fprintf(fp, "\n");
                std::fprintf(fp, "x_tokens");
                for (int64_t r = 0; r < rows; ++r) {
                    std::vector<float> row((size_t) E);
                    q.memcpy(row.data(), x_tokens + (size_t) ht[r] * E, (size_t) E * 4).wait_and_throw();
                    std::fprintf(fp, " %d", (int) ht[r]);
                    char p2[1100];
                    std::snprintf(p2, sizeof p2, "%s/x_tok_%lld.bin", dump_dir, (long long) r);
                    if (FILE* xp = std::fopen(p2, "wb")) {
                        std::fwrite(row.data(), 4, row.size(), xp);
                        std::fclose(xp);
                    }
                }
                std::fprintf(fp, "\n");
            }
            std::free(ht);
            std::free(hd);
            std::fclose(fp);
        }
        std::fprintf(stderr, "strata: STRATA_F16_DUMP wrote %lld rows of expert type %d/%d to %s\n", (long long) rows,
                     L.gu_type, L.d_type, dump_dir);
    }
    return true;
}

}  // namespace strata::kernels
