// sycl/src/kernels/cuda/iq_kernels_fork_expert.dp.cpp - the fork's routed-expert pieces that the GPU-expert tier
// needs and the SYCL tree did not carry yet (S4 step 2):
//
//   native_expert_layout3 / native_expert_supported3   src/kernels/cuda/fork/fork_layout.cuh
//   weighted_rows_sum / decode_rows_sum                src/kernels/cuda/fork/fork_weighted_sum.cuh
//
// plus the four HOST functions the tier's init calls, which live in src/kernels/cuda/iq_kernels.cu inside the
// 5.8k-line dpct migration `sycl/src/kernels/cuda/iq_kernels.dp.cpp`:
//
//   iq_row_bytes, native_expert_supported, native_expert_layout, native_expert_scratch_bytes
//
// They are transcribed here, verbatim from the CUDA source at iq_kernels.cu:2899/4708/4712/4727 (the fork's copies,
// which carry the fork's extra formats - Q2_K 10, MXFP4 39, PTQ1_0 143 - and NVFP4's 16-byte tail), because that TU
// does not compile with icpx 2026.2: twenty errors from `dpct::global_memory` device globals (the iq2xs_grid tables)
// that this dpct version no longer converts to pointers.  Pulling the whole migration in for four host functions -
// and with it the stale `native_expert_grouped` it also defines, which would collide with the ported one in
// iq_kernels_native_expert.dp.cpp - buys nothing: everything below is host arithmetic (no kernels), so the only thing
// that matters is that it is the same arithmetic.  When the migration is refreshed, delete this file's four host
// functions and link iq_kernels.dp.cpp instead.
//
// The two sums ARE kernels and are ports of the CUDA ones (same launch shape, same accumulation order, same rounding:
// decode_rows_sum accumulates in double and rounds once, exactly as `__dadd_rn`/`__dmul_rn` did - `contract(off)`
// stops the device from fusing the product into the add, which is the only way the double sum could move).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// PTQ1_0 (type 143): the fork's ternary block (src/kernels/cuda/fork/fork_vecdot.cuh), 28 B per 128 weights.  Only
// its SIZE is used here (row bytes); the kernels for it live in iq_kernels_native_expert.dp.cpp.
struct block_ptq1_0_size { uint8_t qs[24], qh[2]; ggml_half d; };
static_assert(sizeof(block_ptq1_0_size) == 28, "PTQ1_0 block is 28 bytes");

constexpr int GU_ROWS = 8;   // rows per block of the grouped gate/up kernel; native_expert_supported3 checks n_ff % 8

}  // namespace

// ---- host: the layout arithmetic (verbatim from iq_kernels.cu) ------------------------------------------------------

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 10: return (size_t) (n / 256) * sizeof(block_q2_K);    // fork: Q2_K
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 39: return (size_t) (n / 32) * sizeof(block_mxfp4);    // fork: MXFP4
        case 143: return (size_t) (n / 128) * sizeof(block_ptq1_0_size);   // fork: PTQ1_0
        case 40: return (size_t) (n / 64) * sizeof(block_nvfp4);    // NVFP4 (layout only; no SYCL kernel)
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

namespace {

bool is_iq(int t) {   // the types the grouped expert kernels have a Fmt<> for (iq_kernels.cu is_iq, fork list included)
    if (t == 10 || t == 39 || t == 143) return true;
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8 || t == 40;
}

// values per block of the types the grouped expert kernels take for each role (0 = none)
int gu_qk(int t) {
    switch (t) {
        case 16: case 17: case 18: case 21: case 22: case 23: case 29: return 256;
        case 42: return 64;
        case 12: case 13: case 14: return 256;
        case 6: case 2: case 3: case 8: return 32;
        case 10: case 11: return 256;                                 // fork
        case 39: return 32;                                           // fork: MXFP4
        case 143: return 128;                                         // fork: PTQ1_0
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
        case 20: case 7: case 6: case 2: case 3: case 8: return 32;
        case 23: case 13: case 14: return 256;
        case 42: return 64;
        case 10: case 11: case 12: case 18: case 22: return 256;       // fork
        case 39: return 32;                                           // fork: MXFP4
        case 143: return 128;                                         // fork: PTQ1_0
        case 40: return 64;
        default: return 0;
    }
}

}  // namespace

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    // NVFP4: {s_gate, s_up, s_down, 0} after down whenever any projection is NVFP4; a mixed expert's other projection
    // has 1.0 there, so the kernels multiply by it unconditionally
    if (gu_type == 40 || d_type == 40) {
        L.tail_off = L.bytes;
        L.bytes += 16;
    }
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) +
           (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

// ---- host: the fork's gate/up of different types (fork_layout.cuh) --------------------------------------------------

NativeExpertLayout native_expert_layout3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L = native_expert_layout(gate_type, d_type, n_embd, n_ff);
    if (up_type == gate_type) return L;
    L.up_type = up_type;
    L.up_row = iq_row_bytes(up_type, n_embd);
    L.down_off = L.up_off + (size_t) n_ff * L.up_row;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

bool native_expert_supported3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    if (gate_type == up_type) return native_expert_supported(gate_type, d_type, n_embd, n_ff);
    const bool pair = (gate_type == 10 && up_type == 11) || (gate_type == 11 && up_type == 10);
    return pair && native_expert_supported(gate_type, d_type, n_embd, n_ff) &&
           native_expert_supported(up_type, d_type, n_embd, n_ff) && n_ff % GU_ROWS == 0;
}

// ---- the prompt chunk's top-k mix (fork_weighted_sum.cuh:8) ---------------------------------------------------------
//
// out[t][i] = sum_k w[t*K + k] * parts[(t*K + k)][i] - summed in k order, like the host loop it replaces, so only n
// rows come back instead of n*K.  Grid: min((n*H + 255)/256, 65535) blocks of 256, one element per thread, strided.

void weighted_rows_sum(const float* parts, const float* w, int K, int64_t H, int64_t n, float* out, void* stream) {
    const int64_t total = n * H;
    if (total <= 0 || K <= 0) return;
    const int64_t blocks = std::min<int64_t>((total + 255) / 256, 65535);
    const int64_t stride = blocks * 256;
    strata::q_of(stream)->parallel_for<dpct_kernel_name<class weighted_rows_sum_kernel_9a1f2c>>(
        sycl::range<1>((size_t) stride), [=](sycl::id<1> idx) {
            for (int64_t q = (int64_t) idx[0]; q < total; q += stride) {
                const int64_t t = q / H, i = q - t * H;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += w[t * K + k] * parts[(t * K + k) * H + i];
                out[q] = acc;
            }
        });
}

// One decode token's mix: out[i] = sum over k of w[k] * parts[k][i], k in routing order, accumulated in DOUBLE and
// rounded once - the host loop it replaces bit for bit (a float x float product is exact in double, so fused or not
// the sum is the same); w[k] == 0 (a dropped miss: its row was never computed) is skipped, as there.  K is 6 here, so
// the double arithmetic is 6 x 5120 adds on a card that has fp64 (Arc B70: `aspect::fp64` yes); contract(off) is what
// makes the plain `+`/`*` the same IEEE operations the CUDA source spelled __dadd_rn/__dmul_rn.
void decode_rows_sum(const float* parts, const float* w, int K, int64_t H, float* out, void* stream) {
    if (H <= 0 || K <= 0) return;
    const int64_t blocks = std::min<int64_t>((H + 255) / 256, 1024);
    const int64_t stride = blocks * 256;
    strata::q_of(stream)->parallel_for<dpct_kernel_name<class decode_rows_sum_kernel_9a1f2d>>(
        sycl::range<1>((size_t) stride), [=](sycl::id<1> idx) {
#pragma clang fp contract(off)
            for (int64_t i = (int64_t) idx[0]; i < H; i += stride) {
                double s = 0.0;
                for (int k = 0; k < K; ++k) {
                    const float wk = w[k];
                    if (wk == 0.0f) continue;
                    s = s + (double) wk * (double) parts[(int64_t) k * H + i];
                }
                out[i] = (float) s;
            }
        });
}

}  // namespace strata::kernels
