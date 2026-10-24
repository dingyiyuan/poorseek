// sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp - the SYCL port of `native_expert_grouped`, the routed-expert
// kernel of the DeepSeek-V4.1 tier (src/kernels/cuda/iq_kernels.cu, upstream + fork/fork_*).
//
// WHY A SECOND FILE BESIDE iq_kernels.dp.cpp.  The sycl tree's naming rule is `src/X.cu` -> `sycl/src/X.dp.cpp`, and
// `iq_kernels.dp.cpp` is that file for iq_kernels.cu - but it was migrated from an older iq_kernels.cu and its copy of
// `native_expert_grouped` predates the fork API (no `lora` argument, no PTQ1_0 (143) / Q2_K (10) / Q3_K (11) formats,
// no `NativeExpertLayout::swiglu_limit`), so the current engine cannot link against it (xolotl-sycl-s3-report.md
// section 7, "10 genuinely unported CUDA").  This TU is the current, validated implementation of that one kernel
// family; it is split out rather than patched into the 5.8k-line migrated file so that (a) the ported code is
// readable and reviewable, (b) it cannot perturb the mmvq/dequant half of the port, and (c) it can be built and
// parity-tested on its own (sycl/src/kernels/native_expert_grouped_parity.cpp).  To fold this into the CMake project,
// delete the stale `native_expert_grouped` / `native_grouped_set_v1` definitions and the kernels only they use from
// iq_kernels.dp.cpp; both TUs define those symbols, so linking both is a duplicate-symbol error by design.
//
// WHAT IS PORTED, AND WHAT REFUSES.  Every format this model's tier uses is ported, on both roles:
//     gate/up: Q2_0 (42), Q3_K (11), PTQ1_0 (143)     down: Q2_0 (42), Q3_K (11), PTQ1_0 (143)
// Any other (gate/up, down) type pair - and any non-null `lora`, and any `up_type` that differs from the gate's
// (MiMo GSQ-RCO) - prints which one is unsupported and exits, exactly as the CUDA dispatch does for an unknown type
// (`native_expert_grouped: gate/up type %d`).  Nothing falls back to a cheaper arithmetic: a call either computes the
// CUDA kernels' values or does not run.  Also not ported, and not needed for correctness (both are A/B experiments
// whose results are documented as bitwise identical to the default path): STRATA_EXPERT_V2 / V2K (AMD-tuned S26/S27
// kernels), the AMD `STRATA_EXP_MODE` layouts, the LDS/fused-XMX variants, the `STRATA_PROM` prompt-batch kernels, and
// the `STRATA_OLD_IQ_MMVQ` / `STRATA_NO_SUB16_GU` / STRATA_TSUM switches.  `STRATA_GROUPED_V1=1` (two-pass SwiGLU +
// q8_1 instead of the fused pass) IS honoured, because it is the cheap A/B the parity harness uses to check the
// fused quantizer against the two-kernel composition.
//
// THE ARITHMETIC IS THE CUDA KERNELS', OPERATION FOR OPERATION.  Every dot product keeps the CUDA source's integer
// accumulation (dp4a chains, the same int/float split of the scale multiplies) and `row_dot`'s lane partition and
// butterfly, so the FP32 additions happen in the same order.  `__byte_perm` / `__vsubss4` are byte shuffles and
// saturating adds on *integers*; they are rewritten here as ordinary integer expressions with the same value, which
// cannot change the result.  The places where the result is not forced by the integer arithmetic are (1) the float
// expressions (`d2 * d8 * sumi`, `sumf += d8[i] * (dp4a * sc)`, `d * sumf`, the `s += dot` accumulation and its
// possible FMA contraction) and (2) the SwiGLU's `__expf`.  (1) is written in the CUDA source's order here; (2) is
// `sycl::exp`.  The parity harness (native_expert_grouped_parity) measures both, and reports which arithmetic the
// device produced (contracted vs not, exp-path) - see s4/REPORT.md for the numbers.
#include <sycl/sycl.hpp>

// NO FMA CONTRACTION: the CUDA source writes `sumf += d8[i] * (float)(dp4a * sc)` and `s += d3 * sumf`, and with
// clang's default -ffp-contract=on the device fuses those into fma, which changes their last bits (measured: the
// Q3_K down output moved by up to 3.5e-4 relative - 4096 ULP - against a reference that evaluates the same source
// order).  Off makes the port compute the expression the source writes, which is what the parity harness checks bit
// for bit.  This is TU-local so no build flag has to remember it; a build that wants the fused form instead can pass
// -ffp-contract=fast, and the harness then reports the fused reference's diff (see its "contraction moves it" line).
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace strata::kernels {
namespace {

// ---------------------------------------------------------------- block layouts (ggml-common.h, MIT)
// The structs are transcribed rather than included: ggml-common.h needs GGML_COMMON_DECL_* and dpct_type_* and this
// TU takes nothing else from it.  The static_asserts are the header's own.
struct block_q8_1 {                 // 32 activations: fp16 scale, fp16 sum, 32 int8
    sycl::half d;
    sycl::half s;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "wrong q8_1 block size");

struct block_q2_0 {                 // 64 weights: fp16 scale, 2 bits each
    sycl::half d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q2_0) == 18, "wrong q2_0 block size");

struct block_q3_K {                 // 256 weights: 3 bits, 6-bit scales
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    sycl::half d;
};
static_assert(sizeof(block_q3_K) == 110, "wrong q3_K block size");

struct block_ptq1_0 {               // 128 ternary weights in base 3, fp16 scale (28 bytes)
    uint8_t qs[24];
    uint8_t qh[2];
    sycl::half d;
};
static_assert(sizeof(block_ptq1_0) == 28, "wrong ptq1_0 block size");

// ---------------------------------------------------------------- integer helpers (vecdotq.cuh)
// CUDA's __dp4a: four signed byte products summed into c.  Integer addition is associative modulo 2^32, so the scalar
// order below cannot change the result; IGC pattern-matches the unrolled form into the DP4A instruction.
inline int dp4a(const int a, const int b, const int c) {
    int s = c;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        s += (int) (int8_t) ((a >> (8 * k)) & 0xFF) * (int) (int8_t) ((b >> (8 * k)) & 0xFF);
    return s;
}

inline int get_int_b2(const void* x, const int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    return (int) ((uint32_t) x16[2 * i32 + 0] | ((uint32_t) x16[2 * i32 + 1] << 16));
}

inline int get_int_b4(const void* x, const int i32) { return ((const int*) x)[i32]; }

// Four int8 values as one int: the CUDA source gets this from __byte_perm, whose result is byte-masked by
// construction.  ORing raw ints would smear a negative value across all four bytes (-1 | x<<8 == -1), so each byte is
// masked here - the one place where a transcription can silently change the arithmetic rather than only its spelling.
inline int pack_i8x4(const int v0, const int v1, const int v2, const int v3) {
    return (v0 & 0xFF) | ((v1 & 0xFF) << 8) | ((v2 & 0xFF) << 16) | ((v3 & 0xFF) << 24);
}

// CUDA's __vsubss4: byte-wise saturating signed subtract.
inline int vsubss4(const int a, const int b) {
    int r = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int v = (int) (int8_t) (uint8_t) (a >> (8 * k)) - (int) (int8_t) (uint8_t) (b >> (8 * k));
        v = v > 127 ? 127 : v < -128 ? -128 : v;
        r |= (int) ((uint32_t) v & 0xFFu) << (8 * k);
    }
    return r;
}

// PTQ1_0 trit t of byte b: ((uint8_t)(b * 3^t) * 3) >> 8 (llama-tq's base-3 packing), reduced mod 256 per step.
inline int ptq_trit(const uint8_t b, const int t) {
    uint32_t v = b;
    for (int i = 0; i < t; ++i) v = (v * 3u) & 0xFFu;
    return (int) (((v * 3u) >> 8) & 0xFFu);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block, step = the iqs stride between calls (iq_kernels.cu's Fmt<T>).
template <int TY> struct Fmt;
template <> struct Fmt<42> {
    static constexpr int qk = 64, ipb = 2, step = 1;
};
template <> struct Fmt<11> {
    static constexpr int qk = 256, ipb = 16, step = 1;
};
template <> struct Fmt<143> {
    static constexpr int qk = 128, ipb = 1, step = 0;
};

// ---------------------------------------------------------------- the dot products, one 32-value call each
// The CUDA source advances the q8_1 pointer by kbx*(qk/32) in row_dot and by iqs inside the dot, so `q8` below is
// that already-advanced pointer.

// vec_dot_q2_0_q8_1 (iq_kernels.cu:76): 8 two-bit fields per int16, value = field - 1, two dp4a per int16.
inline float dot_q2_0(const void* vbq, const block_q8_1* q8, const int kbx, const int iqs) {
    const block_q2_0* b = (const block_q2_0*) vbq + kbx;
    const float d2 = (float) b->d;
    const int16_t* qs = (const int16_t*) b->qs + iqs * 4;
    const block_q8_1* bq8 = q8 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = (int) qs[j];   // 8 two-bit fields of one int16, sign-extended
        const int u = get_int_b4(bq8->qs, j * 2 + 0);
        const int v = get_int_b4(bq8->qs, j * 2 + 1);
        // __byte_perm(0x020100FF, 0x020100FF, q) selects {FF,00,01,02}[field] = field - 1 per byte.
        const int qx = pack_i8x4(((q >> 0) & 3) - 1, ((q >> 2) & 3) - 1, ((q >> 4) & 3) - 1, ((q >> 6) & 3) - 1);
        const int qy = pack_i8x4(((q >> 8) & 3) - 1, ((q >> 10) & 3) - 1, ((q >> 12) & 3) - 1, ((q >> 14) & 3) - 1);
        sumi = dp4a(u, qx, sumi);
        sumi = dp4a(v, qy, sumi);
    }
    const float d8 = (float) bq8->d;   // __low2float(bq8->ds)
    return d2 * d8 * sumi;
}

// vec_dot_q3_K_q8_1 (fork/fork_vecdot.cuh:45): 4 q8_1 blocks per call, the inverted high-bit mask subtracts 4.
inline float dot_q3_K(const void* vbq, const block_q8_1* q8, const int kbx, const int iqs) {
    const block_q3_K* b = (const block_q3_K*) vbq + kbx;
    const int bq8_offset = 4 * (iqs / 8);                       // QR3_K * (iqs / (QI3_K / 2))
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;     // iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2)
    const float d3 = (float) b->d;
    const int vl = get_int_b2(b->qs, iqs);
    const int vh = ~get_int_b2(b->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = get_int_b4(q8[bq8_offset + i].qs, iqs % 8);
        d8[i] = (float) q8[bq8_offset + i].d;
    }
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        sumf += d8[i] * (float) (dp4a(vi, u[i], 0) * sc);   // the int product first, as the CUDA source
    }
    return d3 * sumf;
}

// vec_dot_ptq1_0_q8_1 (fork/fork_vecdot.cuh:120): the whole 128-weight block per call, 4 q8_1 sub-blocks, the -1
// folded in as sum(t*a) - sum(a) per sub-block.
inline float dot_ptq1_0(const void* vbq, const block_q8_1* q8, const int kbx, const int iqs) {
    const block_ptq1_0* b = (const block_ptq1_0*) vbq + kbx;
    int sumi[4] = {0, 0, 0, 0}, sumu[4] = {0, 0, 0, 0};
    // qs[0..15]: element 16 t + m; the CUDA kernel takes the 4 bytes of one uint32 per dp4a, element 16 t + 4 w.
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = 16 * t + 4 * w;
            const int u = get_int_b4(q8[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = ptq_trit(b->qs[4 * w + 0], t) | (ptq_trit(b->qs[4 * w + 1], t) << 8) |
                          (ptq_trit(b->qs[4 * w + 2], t) << 16) | (ptq_trit(b->qs[4 * w + 3], t) << 24);
            sumi[e >> 5] = dp4a(q, u, sumi[e >> 5]);
            sumu[e >> 5] = dp4a(0x01010101, u, sumu[e >> 5]);
        }
    }
    // qs[16..23]: element 80 + 8 t + (4 w + 0..3)
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int e = 80 + 8 * t + 4 * w;
            const int u = get_int_b4(q8[iqs + (e >> 5)].qs, (e & 31) >> 2);
            const int q = ptq_trit(b->qs[16 + 4 * w + 0], t) | (ptq_trit(b->qs[16 + 4 * w + 1], t) << 8) |
                          (ptq_trit(b->qs[16 + 4 * w + 2], t) << 16) | (ptq_trit(b->qs[16 + 4 * w + 3], t) << 24);
            sumi[e >> 5] = dp4a(q, u, sumi[e >> 5]);
            sumu[e >> 5] = dp4a(0x01010101, u, sumu[e >> 5]);
        }
    }
    // qh[0..1]: element 120 + 2 t + h, one weight per step
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int e = 120 + 2 * t + h;
            const int a = (int) q8[iqs + (e >> 5)].qs[e & 31];
            const int q = ptq_trit(b->qh[h], t);
            sumi[e >> 5] += q * a;
            sumu[e >> 5] += a;
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) acc += (float) q8[iqs + k].d * (float) (sumi[k] - sumu[k]);
    return (float) b->d * acc;
}

// warp_sum: the same XOR butterfly, so the FP32 additions land in the same order (16, 8, 4, 2, 1).
inline float warp_sum(const float v0, const sycl::sub_group sg) {
    float v = v0;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, (uint32_t) o);
    return v;
}

// One row of weights against one q8_1 activation row, the whole 32-lane sub-group: call k = (block, part) is
// lane-strided exactly as the CUDA `row_dot<TY>` (iq_kernels.cu:669).
template <int TY>
inline float row_dot(const uint8_t* row, const block_q8_1* x, const int nb, const int lane, const sycl::sub_group sg) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xa = x + kbx * (F::qk / 32);
        if constexpr (TY == 42) s += dot_q2_0(row, xa, kbx, iqs);
        else if constexpr (TY == 11) s += dot_q3_K(row, xa, kbx, iqs);
        else s += dot_ptq1_0(row, xa, kbx, iqs);
    }
    return warp_sum(s, sg);
}

// ---------------------------------------------------------------- q8_1 (quantize.cu, q8_1_finite.hpp)
constexpr float kHalfMax = 65504.0f;

inline float q8_1_finite(const float v) {
    return sycl::fabs(v) > kHalfMax ? sycl::copysign(kHalfMax, v) : v;
}

inline int8_t q8_1_quant(const float xi, const float d, const float amax) {
    if (amax == 0.0f) return 0;
    const float q = sycl::round(xi / d);   // roundf: half away from zero
    return (int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q);
}

// Value i of a q8_1 row set: the sub-group holds block i / 32, lane = i % 32 (iq_kernels.cu:1668).
inline void q8_1_store(const float xi, block_q8_1* y, const long long i, const sycl::sub_group sg) {
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, (uint32_t) o));
        sum += sycl::permute_group_by_xor(sg, sum, (uint32_t) o);
    }
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) {
        y[ib].d = sycl::half(q8_1_finite(d));
        y[ib].s = sycl::half(q8_1_finite(sum));
    }
}

// ---------------------------------------------------------------- the kernels
constexpr int GU_ROWS = 8;   // rows per block, one 32-lane sub-group each (CUDA: one warp each)

// native_gu_kernel: rows 0..2*n_ff of the group's blob; a block row strides over the groups (gridDim.y in CUDA).
template <int TG>
void native_gu_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                      const int32_t* ent_tok, const block_q8_1* xq, const NativeExpertLayout L, float* gate,
                      float* up, const sycl::nd_item<2> it) {
    const auto sg = it.get_sub_group();
    const int warp = (int) it.get_local_id(0) >> 5, lane = (int) sg.get_local_id();
    const int row = (int) it.get_group(0) * GU_ROWS + warp;   // 0 .. 2*n_ff
    if (row >= 2 * (int) L.n_ff) return;                     // one row per sub-group: the exit is sub-group uniform
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int ng = *n_groups;
    for (int g = (int) it.get_group(1); g < ng; g += (int) it.get_group_range(1)) {
        const uint8_t* blob = (const uint8_t*) (uintptr_t) grp_ptr[g];
        const uint8_t* wr = blob + off;
        float scale = 1.0f;                                   // NVFP4 only: the blob's tail {s_gate, s_up, s_down, 0}
        if (L.tail_off) scale = is_up ? ((const float*) (blob + L.tail_off))[1] : ((const float*) (blob + L.tail_off))[0];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane, sg);
            if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s * scale;
        }
    }
}

// swiglu_entries_kernel (the STRATA_GROUPED_V1 two-pass path's first pass).
inline void swiglu_entries_kernel(const float* gate, const float* up, float* h, const long long n, const float lim,
                                  const sycl::nd_item<1> it) {
    const long long i = (long long) it.get_global_id(0);
    if (i >= n) return;
    const float g = sycl::fmin(gate[i], lim);
    h[i] = (g / (1.0f + sycl::exp(-g))) * sycl::fmin(sycl::fmax(up[i], -lim), lim);
}

// quantize_q8_1_kernel (the two-pass path's second pass; one sub-group per 32-value block).
inline void quantize_q8_1_kernel(const float* x, block_q8_1* y, const long long n, const sycl::nd_item<1> it) {
    const long long i = (long long) it.get_global_id(0);
    if (i >= n) return;
    q8_1_store(x[i], y, i, it.get_sub_group());
}

// swiglu_q8_1_entries_kernel (the default path): SwiGLU and the q8_1 quantization in one pass over the call's own
// entries, [grp_start[0], grp_start[*n_groups]) * n_ff - a sub-group is one q8_1 block because n_ff % 32 == 0, and
// the range starts on a block boundary for the same reason.
inline void swiglu_q8_1_entries_kernel(const float* gate, const float* up, const int32_t* grp_start,
                                      const int32_t* n_groups, const int n_ff, const float lim, block_q8_1* hq,
                                      const sycl::nd_item<1> it) {
    const auto sg = it.get_sub_group();
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    const long long step = (long long) it.get_global_range(0);
    for (long long i = lo + (long long) it.get_global_id(0); i < hi; i += step) {
        const float g = sycl::fmin(gate[i], lim), u = sycl::fmin(sycl::fmax(up[i], -lim), lim);
        const float h = (g / (1.0f + sycl::exp(-g))) * u;
        q8_1_store(h, hq, i, sg);
    }
}

// native_down_kernel.
template <int TD>
void native_down_kernel(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                        const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout L, float* out,
                        const sycl::nd_item<2> it) {
    const auto sg = it.get_sub_group();
    const int warp = (int) it.get_local_id(0) >> 5, lane = (int) sg.get_local_id();
    const int r = (int) it.get_group(0) * 8 + warp;
    if (r >= (int) L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int ng = *n_groups;
    for (int g = (int) it.get_group(1); g < ng; g += (int) it.get_group_range(1)) {
        const uint8_t* blob = (const uint8_t*) (uintptr_t) grp_ptr[g];
        const uint8_t* wr = blob + off;
        const float sd = L.tail_off ? ((const float*) (blob + L.tail_off))[2] : 1.0f;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane, sg);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s * sd;
        }
    }
}

// ---------------------------------------------------------------- launchers
template <int TY> class K_gu;
template <int TY> class K_down;

template <int TY>
void launch_gu(sycl::queue& q, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
               const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L, float* gate, float* up,
               const int64_t gy) {
    const unsigned rows = (unsigned) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS);
    q.parallel_for<K_gu<TY>>(sycl::nd_range<2>({(size_t) rows * 256, (size_t) gy}, {256, 1}),
                             [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                                 native_gu_kernel<TY>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, it);
                             });
}

template <int TY>
void launch_down(sycl::queue& q, const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                 const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L, float* out,
                 const int64_t gy) {
    const unsigned rows = (unsigned) ((L.n_embd + 7) / 8);
    q.parallel_for<K_down<TY>>(sycl::nd_range<2>({(size_t) rows * 256, (size_t) gy}, {256, 1}),
                               [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                                   native_down_kernel<TY>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, it);
                               });
}

bool env_v1() {
    static const bool v1 = [] { const char* v = std::getenv("STRATA_GROUPED_V1"); return v && v[0] == '1'; }();
    return v1;
}

}  // namespace

// Kept for the engine's STRATA_GROUPED_V1 A/B switch; the value is read from the environment at the first call, as the
// CUDA source does (the launcher caches it in a function-local static).
void native_grouped_set_v1(bool v1) { (void) v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups, const NativeExpertLora* lora) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (lora != nullptr) {
        std::fprintf(stderr, "native_expert_grouped (SYCL): the routed-expert LoRA is not ported\n");
        std::exit(1);
    }
    if (L.up_type >= 0 && L.up_type != L.gu_type) {
        std::fprintf(stderr, "native_expert_grouped (SYCL): gate %d / up %d mixed types are not ported\n", L.gu_type,
                     L.up_type);
        std::exit(1);
    }
    if (L.n_ff % 32 != 0) {
        std::fprintf(stderr, "native_expert_grouped (SYCL): n_ff %lld\n", (long long) L.n_ff);
        std::exit(1);
    }
    sycl::queue& q = *strata::q_of(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    const bool v1 = env_v1();
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;

    // ---- one call's three launches, over a given pointer table, into a given output ----
    // (a named lambda so the STRATA_NE_DIAG block below can run the SAME arithmetic over the same bytes at an even
    // stride; the default path calls it exactly once, so nothing changes when the diagnostic is off)
    auto run_table = [&](const unsigned long long* tbl, float* out_dev) {
        switch (L.gu_type) {
            case 42: launch_gu<42>(q, tbl, grp_start, n_groups, ent_tok, X, L, gate, up, gy); break;
            case 11: launch_gu<11>(q, tbl, grp_start, n_groups, ent_tok, X, L, gate, up, gy); break;
            case 143: launch_gu<143>(q, tbl, grp_start, n_groups, ent_tok, X, L, gate, up, gy); break;
            default:
                std::fprintf(stderr, "native_expert_grouped (SYCL): gate/up type %d is not ported\n", L.gu_type);
                std::exit(1);
        }

        const long long nh = (long long) cap_entries * L.n_ff;
        if (v1) {
            // the two-pass A/B: swiglu_entries_kernel + quantize_q8_1_kernel over all cap_entries (bitwise the same
            // values for the call's entries; the other entries' hq is not read by this call's down kernel)
            q.parallel_for<class K_swiglu>(sycl::nd_range<1>((size_t) ((nh + 255) / 256) * 256, 256),
                                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                               swiglu_entries_kernel(gate, up, h, nh, L.swiglu_limit, it);
                                           });
            q.parallel_for<class K_quant>(sycl::nd_range<1>((size_t) ((nh + 255) / 256) * 256, 256),
                                          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                              quantize_q8_1_kernel(h, hq, nh, it);
                                          });
        } else {
            q.parallel_for<class K_swiglu_q8_1>(sycl::nd_range<1>((size_t) 4096 * 256, 256),
                                                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                                                    swiglu_q8_1_entries_kernel(gate, up, grp_start, n_groups,
                                                                               (int) L.n_ff, L.swiglu_limit, hq, it);
                                                });
        }

        switch (L.d_type) {
            case 42: launch_down<42>(q, tbl, grp_start, n_groups, ent_dst, hq, L, out_dev, gy); break;
            case 11: launch_down<11>(q, tbl, grp_start, n_groups, ent_dst, hq, L, out_dev, gy); break;
            case 143: launch_down<143>(q, tbl, grp_start, n_groups, ent_dst, hq, L, out_dev, gy); break;
            default:
                std::fprintf(stderr, "native_expert_grouped (SYCL): down type %d is not ported\n", L.d_type);
                std::exit(1);
        }
    };
    run_table(grp_ptr, out);

    // ---------------------------------------------------------------- STRATA_NE_DIAG (off by default)
    // The resident/staged disagreement is a statement about the POINTER TABLE: the staged path hands the kernel
    // `c_stage + i*SL` (an even stride), the resident path `cache->device_slot(slot_of(...))` (scattered).  This block
    // reads back, from inside a real engine run, (a) the table as the DEVICE sees it and whether its deltas are the
    // even stride, (b) a hash of every group's blob bytes, and (c) whether the same bytes re-placed at an even stride
    // give a bit-identical output.  (c) equal means the kernel computes the same function for both conventions and the
    // bytes, not the layout of the table, are what differ; (c) unequal means the launch is reading something else.
    // On: STRATA_NE_DIAG=<n> (first n calls), output on stderr.  Off: one static int compare per call.
    static const int diag_n = std::getenv("STRATA_NE_DIAG") ? std::atoi(std::getenv("STRATA_NE_DIAG")) : 0;
    static int diag_calls = 0;
    if (diag_n > 0 && diag_calls < diag_n) {
        const int dc = diag_calls++;
        int32_t ng = 0;
        q.memcpy(&ng, n_groups, sizeof(int32_t)).wait();
        if (ng > 0 && ng <= 4096) {
            std::vector<unsigned long long> hp((size_t) ng);
            std::vector<int32_t> hs((size_t) ng + 1, 0);
            q.memcpy(hp.data(), grp_ptr, sizeof(unsigned long long) * (size_t) ng).wait();
            q.memcpy(hs.data(), grp_start, sizeof(int32_t) * (size_t) (ng + 1)).wait();
            const int n_ent = hs[(size_t) ng];
            std::vector<int32_t> hdst((size_t) std::max(1, n_ent), 0);
            if (n_ent > 0) q.memcpy(hdst.data(), ent_dst, sizeof(int32_t) * (size_t) n_ent).wait();
            // is the table the staged path's even stride?
            const unsigned long long stride = (unsigned long long) L.bytes;
            bool even = true;
            for (int g = 1; g < ng; ++g) even = even && (hp[(size_t) g] == hp[0] + (unsigned long long) g * stride);
            std::fprintf(stderr,
                         "[ne-diag %d] ng=%d cap_g=%lld cap_e=%lld gu=%d/d=%d bytes=%zu grid=%lld | ptr0=%llx deltas",
                         dc, (int) ng, (long long) cap_groups, (long long) cap_entries, L.gu_type, L.d_type, L.bytes,
                         (long long) grid_groups, (unsigned long long) hp[0]);
            for (int g = 1; g < ng && g <= 8; ++g)
                std::fprintf(stderr, " %+lld", (long long) (hp[(size_t) g] - hp[0]));
            std::fprintf(stderr, " strided=%s | grp_start", even ? "YES(even)" : "no(scattered)");
            for (int g = 0; g <= ng && g <= 8; ++g) std::fprintf(stderr, " %d", hs[(size_t) g]);
            std::fprintf(stderr, " | n_ent=%d\n", n_ent);
            // a hash of each group's blob bytes, as the kernel will read them
            const int nh_ = std::min(ng, 64);
            uint8_t* hb = (uint8_t*) std::malloc((size_t) stride);
            if (hb) {
                std::fprintf(stderr, "           blob fnv1a:");
                for (int g = 0; g < nh_; ++g) {
                    q.memcpy(hb, (const void*) (uintptr_t) hp[(size_t) g], L.bytes).wait();
                    uint64_t h64 = 1469598103934665603ull;
                    for (size_t i = 0; i < L.bytes; ++i) { h64 ^= hb[i]; h64 *= 1099511628211ull; }
                    std::fprintf(stderr, " g%d=%016llx", g, (unsigned long long) h64);
                }
                std::fprintf(stderr, "\n");
                std::free(hb);
            }
            // the entry arrays matter too (a wrong ent_dst/ent_tok is invisible to the A/B, which shares them)
            if (n_ent > 0 && n_ent <= 64) {
                std::vector<int32_t> htok((size_t) n_ent, 0);
                q.memcpy(htok.data(), ent_tok, sizeof(int32_t) * (size_t) n_ent).wait();
                std::fprintf(stderr, "           dst:");
                for (int e = 0; e < n_ent && e < 20; ++e) std::fprintf(stderr, " %d", hdst[(size_t) e]);
                std::fprintf(stderr, "%s | tok:", n_ent > 20 ? " ..." : "");
                for (int e = 0; e < n_ent && e < 20; ++e) std::fprintf(stderr, " %d", htok[(size_t) e]);
                std::fprintf(stderr, "%s\n", n_ent > 20 ? " ..." : "");
            }
            // (c) the same bytes at an even stride -> the staged convention; compare the entry rows bitwise
            const int64_t out_ntok = cap_entries;
            if (ng <= 8 && n_ent > 0 && n_ent <= 256 && out_ntok > 0 && L.n_embd > 0 && L.n_embd <= 16384) {
                uint8_t* stg = sycl::malloc_device<uint8_t>(stride * (size_t) ng, q);
                unsigned long long* tbl = sycl::malloc_device<unsigned long long>((size_t) ng, q);
                std::vector<unsigned long long> htbl((size_t) ng);
                for (int g = 0; g < ng; ++g) {
                    q.memcpy(stg + (size_t) g * stride, (const void*) (uintptr_t) hp[(size_t) g], L.bytes);
                    htbl[(size_t) g] = (unsigned long long) (uintptr_t) (stg + (size_t) g * stride);
                }
                q.memcpy(tbl, htbl.data(), sizeof(unsigned long long) * (size_t) ng).wait();
                const size_t ob = (size_t) cap_entries * (size_t) L.n_embd * sizeof(float);
                float* out2 = sycl::malloc_device<float>((size_t) cap_entries * (size_t) L.n_embd, q);
                q.memset(out2, 0, ob);
                run_table(tbl, out2);
                float* h1 = sycl::malloc_host<float>((size_t) cap_entries * (size_t) L.n_embd, q);
                float* h2 = sycl::malloc_host<float>((size_t) cap_entries * (size_t) L.n_embd, q);
                q.memcpy(h1, out, ob).wait();
                q.memcpy(h2, out2, ob).wait();
                long long ndiff = 0;
                double maxabs = 0, maxrel = 0;
                double sum1 = 0, sum2 = 0;
                // only the entries this launch's groups own: [grp_start[0], grp_start[ng]).  Entries outside it belong
                // to another launch of the same pass (another class) and are unwritten in `out`.
                for (int e = hs[0]; e < n_ent; ++e) {
                    const int64_t row = hdst[(size_t) e];
                    if (row < 0 || row >= out_ntok) continue;
                    for (int64_t i = 0; i < L.n_embd; ++i) {
                        const float a = h1[(size_t) row * (size_t) L.n_embd + (size_t) i];
                        const float b = h2[(size_t) row * (size_t) L.n_embd + (size_t) i];
                        const double d = std::fabs((double) a - (double) b);
                        const double den = std::fmax(1e-30, std::fmax(std::fabs((double) a), std::fabs((double) b)));
                        if (d != 0) ++ndiff;
                        maxabs = std::fmax(maxabs, d);
                        maxrel = std::fmax(maxrel, d / den);
                        sum1 += (double) a;
                        sum2 += (double) b;
                    }
                }
                std::fprintf(stderr,
                             "           staged-vs-resident A/B: %lld/%lld values differ, max_abs %.6g, max_rel %.3g, "
                             "row sum %+.6g vs %+.6g -> %s\n",
                             ndiff, (long long) (n_ent - hs[0]) * (long long) L.n_embd, maxabs, maxrel, sum1, sum2,
                             ndiff == 0 ? "SAME FUNCTION (the table layout is honoured; bytes decide)" : "DIFFERENT");
                sycl::free(stg, q);
                sycl::free(tbl, q);
                sycl::free(out2, q);
                sycl::free(h1, q);
                sycl::free(h2, q);
            }
        }
    }
}

}  // namespace strata::kernels
