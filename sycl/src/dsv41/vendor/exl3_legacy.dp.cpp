// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/kernels/cuda/exl3_legacy.cu - v1.1.8's EXL3 decode kernels, kept for NVIDIA GPUs before sm_80 (V100, T4,
// RTX 20): #180's decode kernels run the tiles on the tensor cores from sm_80 on, and their CUDA-core fallback was
// slower here than these (a V100 with EXL3-3.05bpw: decode 4.9 -> 4.4 tok/s).  exl3.cu's mv / moe_gate_up / moe_down
// / win_experts / prepare / several_ok call these on such a GPU (STRATA_GLM_EXL3_DECODE=legacy|new picks one on any
// GPU); the prompt path is exl3.cu's on every GPU.  The code below is v1.1.8's as it was, the prompt path's kernels
// left out.
//
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/
// codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh); the kernels around them are shaped for this engine's decode:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block) over a slice of at most 4096 inputs;
//     its 8 warps take the slice's 16-row tile rows in turn, each staging the row's 8 tiles in shared memory with
//     coalesced loads and decoding 8 weights per lane (one fragment: 2 columns x 4 rows), F32 accumulation;
//   * the input rotation runs per block on the slice it reads (F32 or q8_1 in, FP16 in shared memory), so the
//     engine's activation buffers are used as they are;
//   * a matrix with more than 4096 inputs, or too few output blocks to fill the device, splits its inputs over
//     blocks: the partial sums go to a per-stream scratch and the LAST block of an output block adds them up in
//     split order (deterministic) and runs the output rotation.
#if !defined(STRATA_USE_HIP)   // (AMD runs exl3.cu's kernels; EXL3 does not run there anyway)
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "strata/dsv41/exl3.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <utility>
#include <cmath>

namespace strata::dsv41k::exl3::legacy {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs per block (the rotated slice lives in shared memory)
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // the engine's q8_1 block (ggml block_q8_1): d, sum, 32 int8
    sycl::half2 ds;
    int8_t qs[32];
};

std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    /*
    DPCT1010:229: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
    /*
    DPCT1000:228: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        /*
        DPCT1009:230: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::fprintf(stderr, "exl3 %s: %s\n", what,
                     dpct::get_error_string_dummy(e));
        /*
        DPCT1001:227: The statement could not be removed.
        */
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- the codebooks
__dpct_inline__ uint32_t bytesum_plus(uint32_t x, uint32_t acc) {
#if defined(STRATA_USE_HIP)
    return acc + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
#else
    return dpct::dp4a(x, 0x01010101u, acc);
#endif
}

// two states -> two values, FP16 arithmetic as exllamav3's decode_3inst_2 (the tensor-core kernels take the half2)
template <int cb>
__dpct_inline__ sycl::half2 decode2h(uint32_t s0, uint32_t s1) {
    if constexpr (cb == 2) {
        const uint32_t a = bytesum_plus(s0 * 0x83DCD12Du, 0x6400u);
        const uint32_t b = bytesum_plus(s1 * 0x83DCD12Du, 0x6400u);
        const sycl::half2 h = sycl::half2(
            sycl::bit_cast<sycl::half, unsigned short>((unsigned short)a),
            sycl::bit_cast<sycl::half, unsigned short>((unsigned short)b));
        return sycl::fma(
            h, sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(0x1eee)),
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(0xc931)));
    } else {
        uint32_t x0 = cb == 1 ? s0 * 0xCBAC1FEDu : s0 * 89226354u + 64248484u;
        uint32_t x1 = cb == 1 ? s1 * 0xCBAC1FEDu : s1 * 89226354u + 64248484u;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const sycl::half2 lo =
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x0 & 0xffffu)),
                        sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x1 & 0xffffu)));
        const sycl::half2 hi =
            sycl::half2(sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x0 >> 16)),
                        sycl::bit_cast<sycl::half, unsigned short>(
                            (unsigned short)(x1 >> 16)));
        return lo + hi;
    }
}
template <int cb>
__dpct_inline__ sycl::float2 decode2(uint32_t s0, uint32_t s1) {
    return decode2h<cb>(s0, s1)
        .template convert<float, sycl::rounding_mode::automatic>();
}

// step p's state: the 16-bit window ending at bit (p + 1) K of the tile's MSB-first bit stream (w: its 8 K words),
// wrapping around the tile (the + 256 K keeps the start non-negative)
template <int K> __dpct_inline__ uint32_t state_at(const uint32_t *w, int p) {
    constexpr int W = 8 * K;
    const int b1 = (p + 1 + 256) * K;
    const int i1 = (b1 - 1) >> 5, i0 = (b1 - 16) >> 5;
    const int sh = ((i1 + 1) << 5) - b1;
    const uint64_t m = ((uint64_t) w[i0 % W] << 32) | (uint64_t) w[i1 % W];
    return (uint32_t) (m >> sh) & 0xffffu;
}

// lane's 8 steps (lane * 8 + j): v[j] sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column
// (lane >> 2) + 8 (j >> 2)
template <int K, int cb>
__dpct_inline__ void decode8(const uint32_t *w, int lane, float *v) {
    const int p0 = lane * 8;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const sycl::float2 d =
            decode2<cb>(state_at<K>(w, p0 + j), state_at<K>(w, p0 + j + 1));
        v[j] = d.x();
        v[j + 1] = d.y();
    }
}

// ---------------------------------------------------------------- the rotation
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__dpct_inline__ void had128(float *v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            /*
            DPCT1108:13: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float o = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}

__dpct_inline__ float h2f(uint16_t b) {
    return sycl::vec<sycl::half, 1>(
               sycl::bit_cast<sycl::half, unsigned short>(b))
        .convert<float, sycl::rounding_mode::automatic>()[0];
}

// xs[i] = H(x * suh)[k0 + i] for i < kr (kr % 128 == 0); x from xf (F32) or xq (q8_1); every warp of the block
inline void rotate_in(const float *__restrict__ xf, const Q81 *__restrict__ xq,
                      const uint16_t *__restrict__ suh, int k0, int kr,
                      sycl::half *xs) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    for (int c = warp; c < kr / 128; c += kWarps) {
        const int base = k0 + c * 128 + lane * 4;
        float v[4];
        if (xf != nullptr) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = xf[base + i];
        } else {
            const Q81* b = xq + base / 32;
            const float d = b->ds[0];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[base + i]);
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) xs[c * 128 + lane * 4 + i] =
            sycl::vec<float, 1>(v[i])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
    }
}

// ---------------------------------------------------------------- the block GEMV
// a lane's 4 weights of one column times the activation's 4 inputs, added to a: the rounding spelled out, so the
// one-activation and the several-activation GEMVs (block_gemv, block_gemv_n) give each activation the same bits
__dpct_inline__ float dot4_acc(float a, const float *v, float x0, float x1,
                               float x2, float x3) {
    /*
    DPCT1013:231: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return a + sycl::fma(v[3], x3,
                         sycl::fma(v[2], x2, sycl::fma(v[1], x1, v[0] * x0)));
}

// red[c] (c < 128) = sum over the 16-row slices [ks0, ks1) of x' . W'[:, tc0 * 16 + c], x' = xs from slice ks0 on.
// tr: the matrix's first tile, ld tiles per slice row.  stage: kWarps * 64 K words, red: kWarps * 128 floats.
// Ends with a __syncthreads (red[0..127] complete).
template <int K, int cb>
/*
DPCT1110:14: The total declared local variable size in device function
block_gemv exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
inline void block_gemv(const uint32_t *__restrict__ tr, int ld, int tc0,
                       int ks0, int ks1, const sycl::half *xs, uint32_t *stage,
                       float *red) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K, TW = 8 * W;
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    uint32_t* st = stage + warp * TW;
    float acc[8][2];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = 0.0f;
    const int kk = (lane & 3) * 2;
    for (int ks = ks0 + warp; ks < ks1; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) ks * ld + tc0) * W;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for (int i = lane; i < TW; i += 32) st[i] = src[i];
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        const sycl::half *xk = xs + (ks - ks0) * 16;
        const float
            x0 = sycl::vec<sycl::half, 1>(xk[kk])
                     .convert<float, sycl::rounding_mode::automatic>()[0],
            x1 = sycl::vec<sycl::half, 1>(xk[kk + 1])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
        const float
            x2 = sycl::vec<sycl::half, 1>(xk[kk + 8])
                     .convert<float, sycl::rounding_mode::automatic>()[0],
            x3 = sycl::vec<sycl::half, 1>(xk[kk + 9])
                     .convert<float, sycl::rounding_mode::automatic>()[0];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K, cb>(st + t * W, lane, v);
            acc[t][0] = dot4_acc(acc[t][0], v, x0, x1, x2, x3);
            acc[t][1] = dot4_acc(acc[t][1], v + 4, x0, x1, x2, x3);
        }
    }
    // the 4 lanes of one lane / 4 hold the same two columns of every tile
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            /*
            DPCT1108:15: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            acc[t][c] += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                acc[t][c], 1);
            /*
            DPCT1108:16: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            acc[t][c] += dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                acc[t][c], 2);
        }
    if ((lane & 3) == 0)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            red[warp * kCols + t * 16 + (lane >> 2)] = acc[t][0];
            red[warp * kCols + t * 16 + 8 + (lane >> 2)] = acc[t][1];
        }
    /*
    DPCT1065:232: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < kCols) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) s +=
            red[w * kCols + item_ct1.get_local_id(2)];
        red[item_ct1.get_local_id(2)] = s;
    }
    /*
    DPCT1065:233: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

// block_gemv for nc activations at once (nc <= kNC; activation c's rotated slice at xs + c * kMaxKR): each lane decodes
// its 8 weights of a tile once and adds them into every activation's sums - as exllamav3's small-m GEMV and fused MoE
// kernels share a tile's decode among the rows of an MMA, but with block_gemv's own F32 arithmetic, so outc[c * 128 +
// i] (i < 128) is bit for bit block_gemv's red[i] for activation c alone.  Ends with a __syncthreads.
constexpr int kNC = 4;
template <int K, int cb>
/*
DPCT1110:17: The total declared local variable size in device function
block_gemv_n exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
inline void block_gemv_n(const uint32_t *__restrict__ tr, int ld, int tc0,
                         int ks0, int ks1, const sycl::half *xs, int nc,
                         uint32_t *stage, float *red, float *outc) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int W = 8 * K, TW = 8 * W;
    const int lane = item_ct1.get_local_id(2) & 31,
              warp = item_ct1.get_local_id(2) >> 5;
    uint32_t* st = stage + warp * TW;
    float acc[kNC][8][2];
#pragma unroll
    for (int c = 0; c < kNC; ++c)
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[c][t][0] = acc[c][t][1] = 0.0f;
    const int kk = (lane & 3) * 2;
    for (int ks = ks0 + warp; ks < ks1; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) ks * ld + tc0) * W;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for (int i = lane; i < TW; i += 32) st[i] = src[i];
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
        float x[kNC][4];
#pragma unroll
        for (int c = 0; c < kNC; ++c) {
            const sycl::half *xk = xs + c * kMaxKR + (ks - ks0) * 16;
            if (c < nc) {
                x[c][0] =
                    sycl::vec<sycl::half, 1>(xk[kk])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                x[c][1] =
                    sycl::vec<sycl::half, 1>(xk[kk + 1])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                x[c][2] =
                    sycl::vec<sycl::half, 1>(xk[kk + 8])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                x[c][3] =
                    sycl::vec<sycl::half, 1>(xk[kk + 9])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
            }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K, cb>(st + t * W, lane, v);
#pragma unroll
            for (int c = 0; c < kNC; ++c)
                if (c < nc) {
                    acc[c][t][0] = dot4_acc(acc[c][t][0], v, x[c][0], x[c][1], x[c][2], x[c][3]);
                    acc[c][t][1] = dot4_acc(acc[c][t][1], v + 4, x[c][0], x[c][1], x[c][2], x[c][3]);
                }
        }
    }
    // per activation, block_gemv's reduction
#pragma unroll
    for (int c = 0; c < kNC; ++c) {
        if (c >= nc) break;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                /*
                DPCT1108:20: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                acc[c][t][h] += dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc[c][t][h], 1);
                /*
                DPCT1108:21: '__shfl_xor_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                acc[c][t][h] += dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(),
                    acc[c][t][h], 2);
            }
        if ((lane & 3) == 0)
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                red[warp * kCols + t * 16 + (lane >> 2)] = acc[c][t][0];
                red[warp * kCols + t * 16 + 8 + (lane >> 2)] = acc[c][t][1];
            }
        /*
        DPCT1118:18: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:234: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) < kCols) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) s +=
                red[w * kCols + item_ct1.get_local_id(2)];
            outc[c * kCols + item_ct1.get_local_id(2)] = s;
        }
        /*
        DPCT1118:19: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:235: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// one warp: y[c] = alpha * (H(red) * svh)[c] + bias[c] for the block's 128 outputs
__dpct_inline__ void epilogue(const float *red, const uint16_t *svh,
                              float alpha, const float *bias, float *y) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane * 4 + i;
        /*
        DPCT1013:236: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        y[c] = sycl::fma(alpha * v[i], h2f(svh[c]),
                         bias != nullptr ? bias[c] : 0.0f);
    }
}

// ---------------------------------------------------------------- mv
struct KJob {
    const uint32_t* tr;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* xf;
    const Q81* xq;
    float* y;
    const float* bias;
    float* part;       // S x n partial sums (S > 1); mv_n_kernel: nc x S x n
    unsigned* cnt;     // n / 128 counters, zero between launches
    float alpha;
    int k, n, ld, S, kr;
    int nc;            // mv_n_kernel: the activations (1..kNC), xf / xq / y advanced by these per activation
    size_t xf_cs, xq_cs, y_cs;   // (floats, bytes, floats)
};
struct KBatch {
    KJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
};

// the dynamic shared memory of the several-activation kernels (mv_n_kernel, win_gu_kernel, win_down_kernel):
// red[kWarps][128] outc[kNC][128] gu[2][kNC][128] (F32), stage[kWarps * 64 K] (words), xs[kNC][kMaxKR] (FP16) - up to
// 58 KB, past the 48 KB a launch gets without asking (set_smem)
constexpr size_t smem_n_bytes(int K) {
    return (size_t) (kWarps * kCols + kNC * kCols + 2 * kNC * kCols) * 4 + (size_t) kWarps * 64 * K * 4 +
           (size_t) kNC * kMaxKR * 2;
}
struct SmemN {
    float *red, *outc, *gu;
    uint32_t* stage;
    sycl::half *xs;
};
template <int K> __dpct_inline__ SmemN smem_n(unsigned char *p) {
    SmemN s;
    s.red = (float*) p;
    s.outc = s.red + kWarps * kCols;
    s.gu = s.outc + kNC * kCols;
    s.stage = (uint32_t*) (s.gu + 2 * kNC * kCols);
    s.xs = (sycl::half *)(s.stage + kWarps * 64 * K);
    return s;
}

template <int K, int cb>
void mv_kernel(const KBatch b) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xs =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[kMaxKR]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &stage = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[kWarps * 64 * K]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kWarps * kCols]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int ji = 0;
    while ((int)item_ct1.get_group(2) >= b.blk_end[ji])++ ji;
    const KJob& J = b.j[ji];
    const int local =
        (int)item_ct1.get_group(2) - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = sycl::min(J.k, k0 + J.kr);
    rotate_in(J.xf, J.xq, J.suh, k0, k1 - k0, xs);
    /*
    DPCT1065:237: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    block_gemv<K, cb>(J.tr, J.ld, g * 8, k0 / 16, k1 / 16, xs, stage, red);
    const int warp = item_ct1.get_local_id(2) >> 5;
    if (J.S == 1) {
        if (warp == 0)
            epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr,
                     J.y + g * kCols);
        return;
    }
    if (item_ct1.get_local_id(2) < kCols)
        J.part[(size_t)sp * J.n + g * kCols + item_ct1.get_local_id(2)] =
        red[item_ct1.get_local_id(2)];
    /*
    DPCT1078:22: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:238: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &J.cnt[g], 1u) == (unsigned)(J.S - 1);
    /*
    DPCT1065:239: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:23: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < kCols) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < J.S; ++i) s +=
            *(volatile const float *)&J
                 .part[(size_t)i * J.n + g * kCols + item_ct1.get_local_id(2)];
        red[item_ct1.get_local_id(2)] = s;
    }
    /*
    DPCT1065:240: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp == 0)
        epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr, J.y + g * kCols);
    if (item_ct1.get_local_id(2) == 0) J.cnt[g] = 0u;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void mv_kernel_wrapper(const KBatch b) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.parallel_for(
        nr,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            mv_kernel<K, cb>(b);
        });
}

// mv_kernel for a job's nc activations at once (a verify window's tokens): the same input split, the weights read and
// decoded once for all of them (block_gemv_n), each activation's partials and sums in mv_kernel's order - every
// activation's y is mv_kernel's for it alone, bit for bit
template <int K, int cb>
void mv_n_kernel(const KBatch b,
                                             uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    const SmemN sm = smem_n<K>(smem_raw);
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    int ji = 0;
    while ((int)item_ct1.get_group(2) >= b.blk_end[ji])++ ji;
    const KJob& J = b.j[ji];
    const int local =
        (int)item_ct1.get_group(2) - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = sycl::min(J.k, k0 + J.kr);
#pragma unroll
    for (int c = 0; c < J.nc; ++c)
        rotate_in(J.xf != nullptr ? J.xf + c * J.xf_cs : nullptr,
                  J.xq != nullptr ? (const Q81*) ((const uint8_t*) J.xq + c * J.xq_cs) : nullptr, J.suh, k0, k1 - k0,
                  sm.xs + c * kMaxKR);
    /*
    DPCT1065:241: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    block_gemv_n<K, cb>(J.tr, J.ld, g * 8, k0 / 16, k1 / 16, sm.xs, J.nc, sm.stage, sm.red, sm.outc);
    const int warp = item_ct1.get_local_id(2) >> 5;
    const float* bias = J.bias != nullptr ? J.bias + g * kCols : nullptr;
    if (J.S == 1) {
        if (warp < J.nc)
            epilogue(sm.outc + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y + warp * J.y_cs + g * kCols);
        return;
    }
    if (item_ct1.get_local_id(2) < kCols)
#pragma unroll
        for (int c = 0; c < J.nc; ++c)
            J.part[((size_t)c * J.S + sp) * J.n + g * kCols +
                   item_ct1.get_local_id(2)] =
                sm.outc[c * kCols + item_ct1.get_local_id(2)];
    /*
    DPCT1078:24: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:242: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &J.cnt[g], 1u) == (unsigned)(J.S - 1);
    /*
    DPCT1065:243: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:25: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < kCols)
        for (int c = 0; c < J.nc; ++c) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < J.S; ++i)
                s += *(volatile const float *)&J
                          .part[((size_t)c * J.S + i) * J.n + g * kCols +
                                item_ct1.get_local_id(2)];
            sm.outc[c * kCols + item_ct1.get_local_id(2)] = s;
        }
    /*
    DPCT1065:244: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (warp < J.nc) epilogue(sm.outc + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y + warp * J.y_cs + g * kCols);
    if (item_ct1.get_local_id(2) == 0) J.cnt[g] = 0u;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void mv_n_kernel_wrapper(const KBatch b) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for(nr,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                mv_n_kernel<K, cb>(
                    b, dpct_local_acc_ct1
                           .get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
}

// ---------------------------------------------------------------- the routed experts
struct KMoe {
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    float* h;            // [k][n_ff]
    const float* sh_out;
    float* out;
    float* part;         // [k][n_embd]
    unsigned* cnt;       // n_embd / 128
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// the pieces the token's experts (moe_*_kernel) and a window's grouped experts (win_*_kernel) share, so each window
// token gets the token path's bits: one warp's output block of 128 (H, then svh) ...
__dpct_inline__ void block_out(const float *red, const uint16_t *svh,
                               float *dst) {
    const int lane =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) &
        31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    /*
    DPCT1013:245: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    for (int i = 0; i < 4; ++i)
        dst[lane * 4 + i] = v[i] * h2f(svh[lane * 4 + i]);
}
// ... the clamped swiglu ...
__dpct_inline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = sycl::fmin(gv, limit);
        uv = sycl::fmin(sycl::fmax(uv, -limit), limit);
    }
    /*
    DPCT1013:246: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return gv / (1.0f + sycl::native::exp(-gv)) * uv;
}
// ... and one output's sum: the experts' rows in plan order (p[i]: expert i's, 0 for one not on the device), the CPU
// lane's part when flagged (cpu_part: null when not), the shared expert's row (sh: 0 without one)
__dpct_inline__ float combine(const float *plan_w, const float *p, int k,
                              const float *cpu_part, float sh) {
    float m = 0.0f;
    /*
DPCT1013:247: The rounding mode could not be specified and the generated code
may have different accuracy than the original code. Verify the correctness. SYCL
math built-in function rounding mode is aligned with OpenCL C 1.2 standard.
*/
#pragma unroll
    for (int i = 0; i < k; ++i) m = sycl::fma(plan_w[i], p[i], m);
    /*
    DPCT1013:248: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    if (cpu_part != nullptr) m = m + *cpu_part;
    /*
    DPCT1013:249: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    return m + sh;
}

// grid (n_ff / 128, k): expert blockIdx.y's gate and up for 128 h values, then the clamped swiglu
template <int K, int cb>
void moe_gate_up_kernel(const KMoe a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xs =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[2][kMaxKR]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &stage = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[kWarps * 64 * K]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kWarps * kCols]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &gu =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[2][kCols]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int ei = item_ct1.get_group(1), g = item_ct1.get_group(2);
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    const uint8_t* blob = (const uint8_t*) ptr;
    const int warp = item_ct1.get_local_id(2) >> 5;
#pragma unroll
    for (int r = 0; r < 2; ++r) rotate_in(
        a.x, nullptr, (const uint16_t *)(blob + a.suh[r]), 0, a.n_embd, xs[r]);
    /*
    DPCT1065:250: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, xs[r], stage, red);
        if (warp == 0) block_out(red, (const uint16_t*) (blob + a.svh[r]) + g * kCols, gu[r]);
        /*
        DPCT1118:26: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:251: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    if (item_ct1.get_local_id(2) < kCols)
        a.h[(size_t)ei * a.n_ff + g * kCols + item_ct1.get_local_id(2)] =
            swiglu(gu[0][item_ct1.get_local_id(2)],
                   gu[1][item_ct1.get_local_id(2)], a.limit);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void moe_gate_up_kernel_wrapper(const KMoe a) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.parallel_for(
        nr,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            moe_gate_up_kernel<K, cb>(a);
        });
}

// grid (n_embd / 128, k): expert blockIdx.y's down rows for 128 outputs; the last block of each output block adds
// the experts up in plan order (as the engine's moe_down), then the CPU lane's part and the shared expert
template <int K, int cb>
void moe_down_kernel(const KMoe a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &xs =
    *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[kMaxKR]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &stage = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        uint32_t[kWarps * 64 * K]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[kWarps * kCols]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &last = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int ei = item_ct1.get_group(1), g = item_ct1.get_group(2);
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = item_ct1.get_local_id(2) >> 5;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (ptr != 0ull) {
        const uint8_t* blob = (const uint8_t*) ptr;
        rotate_in(a.h + (size_t) ei * a.n_ff, nullptr, (const uint16_t*) (blob + a.suh[2]), 0, a.n_ff, xs);
        /*
        DPCT1118:29: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:254: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        block_gemv<K, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff / 16, xs, stage, red);
        if (warp == 0) block_out(red, (const uint16_t*) (blob + a.svh[2]) + g * kCols, part);
    } else if (item_ct1.get_local_id(2) < kCols) {
        part[item_ct1.get_local_id(2)] = 0.0f;
    }
    /*
    DPCT1078:27: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    /*
    DPCT1065:252: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) == 0) last =
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &a.cnt[g], 1u) == (unsigned)(a.k - 1);
    /*
    DPCT1065:253: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (!last) return;
    /*
    DPCT1078:28: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    if (item_ct1.get_local_id(2) < kCols) {
        const int r = g * kCols + item_ct1.get_local_id(2);
        float p[8];
#pragma unroll
        for (int i = 0; i < a.k; ++i)
            p[i] = *(volatile const float *)&a.part[(size_t)i * a.n_embd + r];
        a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                           a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (item_ct1.get_local_id(2) == 0) a.cnt[g] = 0u;
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void moe_down_kernel_wrapper(const KMoe a) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.parallel_for(
        nr,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            moe_down_kernel<K, cb>(a);
        });
}

// ---------------------------------------------------------------- a verify window's routed experts
// gf::moe_win_experts' grouping (group g: one expert, its blob at grp_ptr[g]; its entries e = t * 8 + i at
// grp_ent[grp_start[g] ..)), each group's weights read and decoded once for up to kNC of its tokens at a time (as
// exllamav3's fused MoE runs a run of slots that picked one expert as the rows of one MMA tile), with the token
// path's arithmetic per entry: every token's output is moe_gate_up + moe_down's for it alone, bit for bit
struct KWin {
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    size_t x_stride;
    float* h;            // [T * 8][n_ff]
    float* part;         // [T * 8][n_embd]
    const float* sh_out; // [T][n_embd]
    float* out;
    size_t out_stride;
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, groups at most): group blockIdx.y's gate and up for 128 h values of each entry, the clamped swiglu
template <int K, int cb>
void win_gu_kernel(const KWin a,
                                               uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    const SmemN sm = smem_n<K>(smem_raw);
    const int grp = item_ct1.get_group(1), g = item_ct1.get_group(2);
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = item_ct1.get_local_id(2) >> 5;
    const int m1 = a.grp_start[grp + 1];
    for (int m0 = a.grp_start[grp]; m0 < m1; m0 += kNC) {
        const int nc = sycl::min(kNC, m1 - m0);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int c = 0; c < nc; ++c)
                rotate_in(a.x + (size_t) (a.grp_ent[m0 + c] >> 3) * a.x_stride, nullptr,
                          (const uint16_t*) (blob + a.suh[r]), 0, a.n_embd, sm.xs + c * kMaxKR);
            /*
            DPCT1118:31: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:256: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
            block_gemv_n<K, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, sm.xs, nc,
                                sm.stage, sm.red, sm.outc);
            if (warp < nc)
                block_out(sm.outc + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                          sm.gu + (r * kNC + warp) * kCols);
            /*
            DPCT1118:32: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:257: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        if (item_ct1.get_local_id(2) < kCols)
#pragma unroll
            for (int c = 0; c < nc; ++c)
                a.h[(size_t)a.grp_ent[m0 + c] * a.n_ff + g * kCols +
                    item_ct1.get_local_id(2)] =
                    swiglu(sm.gu[c * kCols + item_ct1.get_local_id(2)],
                           sm.gu[(kNC + c) * kCols + item_ct1.get_local_id(2)],
                           a.limit);
        /*
        DPCT1118:30: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:255: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void win_gu_kernel_wrapper(const KWin a) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for(nr,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                win_gu_kernel<K, cb>(
                    a, dpct_local_acc_ct1
                           .get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
}

// grid (n_embd / 128, groups at most): group blockIdx.y's down rows for 128 outputs of each entry into its part rows
template <int K, int cb>
void win_down_kernel(const KWin a,
                                                 uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto smem_raw = (unsigned char *)dpct_local;
    const SmemN sm = smem_n<K>(smem_raw);
    const int grp = item_ct1.get_group(1), g = item_ct1.get_group(2);
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = item_ct1.get_local_id(2) >> 5;
    const int m1 = a.grp_start[grp + 1];
    for (int m0 = a.grp_start[grp]; m0 < m1; m0 += kNC) {
        const int nc = sycl::min(kNC, m1 - m0);
#pragma unroll
        for (int c = 0; c < nc; ++c)
            rotate_in(a.h + (size_t) a.grp_ent[m0 + c] * a.n_ff, nullptr, (const uint16_t*) (blob + a.suh[2]), 0,
                      a.n_ff, sm.xs + c * kMaxKR);
        /*
        DPCT1118:33: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:258: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        block_gemv_n<K, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff / 16, sm.xs, nc,
                            sm.stage, sm.red, sm.outc);
        if (warp < nc)
            block_out(sm.outc + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                      a.part + (size_t) a.grp_ent[m0 + warp] * a.n_embd + g * kCols);
        /*
        DPCT1118:34: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:259: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
template <int K, int cb> void win_down_kernel_wrapper(const KWin a) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;

    dpct::has_capability_or_fail(queue.get_device(), {sycl::aspect::fp16});

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range(localMemSize), cgh);

        cgh.parallel_for(nr,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                win_down_kernel<K, cb>(
                    a, dpct_local_acc_ct1
                           .get_multi_ptr<sycl::access::decorated::no>()
                           .get());
            });
    });
}

// grid (n_embd / 128, T): each token's outputs, moe_down_kernel's combine (an entry off the device adds 0)
void win_combine_kernel(const KWin a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int t = item_ct1.get_group(1),
              r = item_ct1.get_group(2) * kCols + item_ct1.get_local_id(2);
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
    }
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k, a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.n_embd + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void win_combine_kernel_wrapper(const KWin a) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            win_combine_kernel(a);
        });
}

// ---------------------------------------------------------------- dispatch
// F(K, cb) for K 1..8 and the three codebooks
template <template <int, int> class F, typename... A>
bool dispatch(int K, int cb, A&&... args) {
#define EXL3_CASE(KK)                                                    \
    case KK:                                                             \
        if (cb == 2) F<KK, 2>::run(std::forward<A>(args)...);            \
        else if (cb == 1) F<KK, 1>::run(std::forward<A>(args)...);       \
        else F<KK, 0>::run(std::forward<A>(args)...);                    \
        return true;
    switch (K) {
        EXL3_CASE(1) EXL3_CASE(2) EXL3_CASE(3) EXL3_CASE(4) EXL3_CASE(5) EXL3_CASE(6) EXL3_CASE(7) EXL3_CASE(8)
        default: return false;
    }
#undef EXL3_CASE
}

template <int K, int cb> struct RunMv {
    static void run(const KBatch &b, int blocks, dpct::queue_ptr s) {
        strata_ds4_launch(mv_kernel_wrapper<K, cb>, blocks, kThreads, 0, s, b);
    }
};
template <int K, int cb> struct RunMvN {
    static void run(const KBatch &b, int blocks, dpct::queue_ptr s) {
        strata_ds4_launch(mv_n_kernel_wrapper<K, cb>, blocks, kThreads,
                          smem_n_bytes(K), s, b);
    }
};
template <int K, int cb> struct RunWin {
    static void run(const KWin &a, int groups, dpct::queue_ptr s) {
        strata_ds4_launch(
            win_gu_kernel_wrapper<K, cb>,
            dpct::dim3((unsigned)(a.n_ff / kCols), (unsigned)groups), kThreads,
            smem_n_bytes(K), s, a);
    }
};
template <int K, int cb> struct RunWinDown {
    static void run(const KWin &a, int groups, dpct::queue_ptr s) {
        strata_ds4_launch(
            win_down_kernel_wrapper<K, cb>,
            dpct::dim3((unsigned)(a.n_embd / kCols), (unsigned)groups),
            kThreads, smem_n_bytes(K), s, a);
    }
};
template <int K, int cb> struct AllowSmemN {
    static void run() {
        const int b = (int) smem_n_bytes(K);
        /*
        DPCT1026:260: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
        /*
        DPCT1026:261: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
        /*
        DPCT1026:262: The call to cudaFuncSetAttribute was removed because SYCL
        currently does not support corresponding setting.
        */
    }
};
template <int K, int cb> struct RunGU {
    static void run(const KMoe &a, dpct::queue_ptr s) {
        strata_ds4_launch(moe_gate_up_kernel_wrapper<K, cb>,
                          dpct::dim3((unsigned)(a.n_ff / kCols), (unsigned)a.k),
                          kThreads, 0, s, a);
    }
};
template <int K, int cb> struct RunDown {
    static void run(const KMoe &a, dpct::queue_ptr s) {
        strata_ds4_launch(
            moe_down_kernel_wrapper<K, cb>,
            dpct::dim3((unsigned)(a.n_embd / kCols), (unsigned)a.k), kThreads,
            0, s, a);
    }
};
// ---------------------------------------------------------------- per-stream scratch
constexpr size_t kPartFloats = (size_t) 2 << 20;   // split partials: 8 MB
constexpr size_t kCounters = 16384;
constexpr size_t kHFloats = (size_t) 8 * 8192;     // the experts' h
struct Scratch {
    float* part = nullptr;
    unsigned* cnt = nullptr;
    float* h = nullptr;
};
std::mutex g_mu;
std::map<std::pair<int, dpct::queue_ptr>, Scratch> g_scratch;

Scratch *scratch(dpct::queue_ptr s) try {
    int dev = 0;
    dev = dpct::get_current_device_id();
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find({dev, s});
    if (it != g_scratch.end()) return &it->second;
    sycl::ext::oneapi::experimental::queue_state cs =
        sycl::ext::oneapi::experimental::queue_state::executing;
    if (DPCT_CHECK_ERROR((cs = s->ext_oneapi_get_state())) == 0 &&
        cs != sycl::ext::oneapi::experimental::queue_state::executing) {
        std::fprintf(stderr, "exl3: scratch for a stream first seen inside a graph capture (call exl3::prepare)\n");
        return nullptr;
    }
    Scratch sc;
    if (DPCT_CHECK_ERROR(sc.part = sycl::malloc_device<float>(
                             kPartFloats, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.cnt = sycl::malloc_device<unsigned int>(
                             kCounters, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(sc.h = sycl::malloc_device<float>(
                             kHFloats, dpct::get_in_order_queue())) != 0 ||
        DPCT_CHECK_ERROR(s->memset(sc.cnt, 0, kCounters * sizeof(unsigned))) !=
            0) {
        std::fprintf(stderr, "exl3: scratch did not allocate\n");
        return nullptr;
    }
    s->wait();
    return &(g_scratch[{dev, s}] = sc);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sm_count() {
    int dev = 0, sms = 0;
    dev = dpct::get_current_device_id();
    sms = dpct::get_device(dev).get_max_compute_units();
    return std::max(1, sms);
}

// the several-activation kernels may use smem_n_bytes of shared memory: asked once per device (prepare, before a graph
// capture; else at the first such launch).  A device that cannot give a block that much (48 KB GPUs) runs one
// activation at a time instead
constexpr int kMaxDev = 64;
std::atomic<int> g_smem_n[kMaxDev];   // 0 not asked, 1 allowed, 2 refused
bool allow_smem_n() {
    int dev = 0;
    dev = dpct::get_current_device_id();
    if (dev < 0 || dev >= kMaxDev) return false;
    const int st = g_smem_n[dev].load(std::memory_order_acquire);
    if (st != 0) return st == 1;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_smem_n[dev].load(std::memory_order_relaxed) != 0) return g_smem_n[dev].load() == 1;
    int optin = 0;
    /*
    DPCT1019:263: local_mem_size in SYCL is not a complete equivalent of
    cudaDevAttrMaxSharedMemoryPerBlockOptin in CUDA. You may need to adjust the
    code.
    */
    optin = dpct::get_device(dev).get_local_mem_size();
    bool ok = (size_t) optin >= smem_n_bytes(8);
    if (ok) {
        for (int K = 1; K <= 8; ++K)
            for (int cb = 0; cb <= 2; ++cb) dispatch<AllowSmemN>(K, cb);
        /*
        DPCT1010:264: SYCL uses exceptions to report errors and does not use the
        error codes. The cudaGetLastError function call was replaced with 0. You
        need to rewrite this code.
        */
        ok = 0 == 0;
    }
    if (!ok)
        std::fprintf(stderr, "exl3: CUDA%d gives a block %d bytes of shared memory, the several-activation kernels "
                             "need %zu - a verify window's tokens run one at a time there\n", dev, optin,
                     smem_n_bytes(8));
    g_smem_n[dev].store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}

// mv's input split for m: S blocks of kr inputs per output block, at most kMaxKR inputs a block and about two blocks
// per SM (the same for every activation count: mv_kernel and mv_n_kernel sum each activation alike)
void mv_split(const Mat& m, int sms, int& S, int& kr) {
    const int groups = m.n / kCols;
    S = (m.k + kMaxKR - 1) / kMaxKR;
    while (groups * S < 2 * sms && m.k / (S * 2) >= 256) S *= 2;
    kr = ((m.k + S - 1) / S + 127) / 128 * 128;
    S = (m.k + kr - 1) / kr;
}

}  // namespace

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

bool prepare(dpct::queue_ptr s) {
    if (scratch(s) == nullptr) return false;
    (void) allow_smem_n();
    return true;
}

bool several_ok() { return allow_smem_n(); }

bool mv(const MvJob *jobs, int n, dpct::queue_ptr s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int sms = sm_count();
    // the launch units: a job of one activation is mv_kernel's, a job of several mv_n_kernel's in runs of up to kNC
    // (one mv_kernel unit an activation where the device lacks the shared memory)
    struct Unit {
        int j, c0, nc;
    };
    Unit units[kMaxMvJobs * kMaxMvCols];
    int nu = 0;
    bool several = false;
    for (int j = 0; j < n; ++j) several = several || jobs[j].ncol > 1;
    const int run = several && allow_smem_n() ? kNC : 1;
    for (int j = 0; j < n; ++j) {
        const Mat& m = *jobs[j].m;
        if (!supported(m) || jobs[j].y == nullptr || (jobs[j].xf == nullptr && jobs[j].xq == nullptr) ||
            jobs[j].ncol < 1 || jobs[j].ncol > kMaxMvCols) {
            std::fprintf(stderr, "exl3 mv: job %d unsupported (k %d n %d K %d cb %d, %d activations)\n", j, m.k, m.n,
                         m.K, m.cb, jobs[j].ncol);
            return false;
        }
        for (int c0 = 0; c0 < jobs[j].ncol; c0 += run) units[nu++] = {j, c0, std::min(run, jobs[j].ncol - c0)};
    }
    bool done[sizeof(units) / sizeof(units[0])] = {};
    for (int i = 0; i < nu; ++i) {
        if (done[i]) continue;
        const Mat& m0 = *jobs[units[i].j].m;
        const bool multi = units[i].nc > 1;
        // a launch: the units of one (K, codebook, kernel), as many as KBatch and the split scratch take (the next
        // launch on the stream may use the scratch again: this one's last blocks are done with it by then)
        KBatch b{};
        int blocks = 0, nb = 0;
        size_t part_at = 0, cnt_at = 0;
        for (int u = i; u < nu && nb < kMaxMvJobs; ++u) {
            const MvJob& job = jobs[units[u].j];
            const Mat& m = *job.m;
            if (done[u] || m.K != m0.K || m.cb != m0.cb || (units[u].nc > 1) != multi) continue;
            const int c0 = units[u].c0, nc = units[u].nc, groups = m.n / kCols;
            int S = 1, kr = 0;
            mv_split(m, sms, S, kr);
            const size_t need = S > 1 ? (size_t) S * m.n * nc : 0;
            if (S > 1 && (part_at + need > kPartFloats || cnt_at + (size_t) groups > kCounters)) {
                if (nb > 0) continue;   // the next launch
                std::fprintf(stderr, "exl3 mv: scratch too small for k %d n %d (%d activations)\n", m.k, m.n, nc);
                return false;
            }
            done[u] = true;
            KJob& J = b.j[nb];
            J.tr = m.trellis;
            J.suh = m.suh;
            J.svh = m.svh;
            J.xf = job.xf != nullptr ? job.xf + c0 * job.xf_cs : nullptr;
            J.xq = job.xq != nullptr ? (const Q81*) ((const uint8_t*) job.xq + c0 * job.xq_cs) : nullptr;
            J.y = job.y + c0 * job.y_cs;
            J.bias = job.bias;
            J.alpha = job.alpha;
            J.k = m.k;
            J.n = m.n;
            J.ld = m.ld > 0 ? m.ld : m.n / 16;
            J.S = S;
            J.kr = kr;
            J.part = sc->part + part_at;
            J.cnt = sc->cnt + cnt_at;
            J.nc = nc;
            J.xf_cs = job.xf_cs;
            J.xq_cs = job.xq_cs;
            J.y_cs = job.y_cs;
            part_at += need;
            if (S > 1) cnt_at += (size_t) groups;
            blocks += groups * S;
            b.blk_end[nb++] = blocks;
        }
        b.n = nb;
        if (!(multi ? dispatch<RunMvN>(m0.K, m0.cb, b, blocks, s) : dispatch<RunMv>(m0.K, m0.cb, b, blocks, s)))
            return false;
    }
    launch_check("mv");
    return true;
}

static bool moe_args(const MoeArgs &a, KMoe &m, dpct::queue_ptr s) {
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    if (a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || (size_t) a.k * a.n_ff > kHFloats || (size_t) a.k * a.n_embd > kPartFloats ||
        (size_t) (a.n_embd / kCols) > kCounters) {
        std::fprintf(stderr, "exl3 moe: geometry not covered (k %d n_embd %d n_ff %d)\n", a.k, a.n_embd, a.n_ff);
        return false;
    }
    m = KMoe{};
    m.plan_ptr = a.plan_ptr;
    m.plan_w = a.plan_w;
    m.cpu_part = a.cpu_part;
    m.cpu_flag = a.cpu_flag;
    m.h = sc->h;
    m.part = sc->part;
    m.cnt = sc->cnt;
    for (int r = 0; r < 3; ++r) {
        m.suh[r] = a.lay.suh[r];
        m.svh[r] = a.lay.svh[r];
        m.tr[r] = a.lay.trellis[r];
    }
    m.k = a.k;
    m.n_embd = a.n_embd;
    m.n_ff = a.n_ff;
    m.limit = a.limit;
    return true;
}

void moe_gate_up(const MoeArgs &a, const float *x, float *h,
                 dpct::queue_ptr s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (a.lay.K[0] != a.lay.K[1]) {
        std::fprintf(stderr, "exl3 moe_gate_up: gate K %d != up K %d\n", a.lay.K[0], a.lay.K[1]);
        return;
    }
    m.x = x;
    if (h != nullptr) m.h = h;
    if (!dispatch<RunGU>(a.lay.K[0], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_gate_up: K %d\n", a.lay.K[0]);
    launch_check("moe_gate_up");
}

void moe_down(const MoeArgs &a, const float *h, const float *sh_out, float *out,
              dpct::queue_ptr s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (h != nullptr) m.h = const_cast<float*>(h);
    m.sh_out = sh_out;
    m.out = out;
    if (!dispatch<RunDown>(a.lay.K[2], a.lay.cb, m, s)) std::fprintf(stderr, "exl3 moe_down: K %d\n", a.lay.K[2]);
    launch_check("moe_down");
}

bool win_experts(const WinArgs &a, dpct::queue_ptr s) {
    if (a.T < 1 || a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || a.lay.K[0] != a.lay.K[1] || a.h == nullptr || a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "exl3 win_experts: not covered (T %d k %d n_embd %d n_ff %d, gate K %d up K %d)\n", a.T,
                     a.k, a.n_embd, a.n_ff, a.lay.K[0], a.lay.K[1]);
        return false;
    }
    if (!allow_smem_n()) return false;
    KWin w{};
    w.grp_ptr = a.grp_ptr;
    w.grp_start = a.grp_start;
    w.grp_ent = a.grp_ent;
    w.n_groups = a.n_groups;
    w.plan_ptr = a.plan_ptr;
    w.plan_w = a.plan_w;
    w.cpu_part = a.cpu_part;
    w.cpu_flag = a.cpu_flag;
    w.x = a.x;
    w.x_stride = a.x_stride;
    w.h = a.h;
    w.part = a.part;
    w.sh_out = a.sh_out;
    w.out = a.out;
    w.out_stride = a.out_stride;
    for (int r = 0; r < 3; ++r) {
        w.suh[r] = a.lay.suh[r];
        w.svh[r] = a.lay.svh[r];
        w.tr[r] = a.lay.trellis[r];
    }
    w.k = a.k;
    w.n_embd = a.n_embd;
    w.n_ff = a.n_ff;
    w.limit = a.limit;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    if (!dispatch<RunWin>(a.lay.K[0], a.lay.cb, w, groups, s) ||
        !dispatch<RunWinDown>(a.lay.K[2], a.lay.cb, w, groups, s))
        return false;
    strata_ds4_launch(win_combine_kernel_wrapper,
                      dpct::dim3((unsigned)(a.n_embd / kCols), (unsigned)a.T),
                      kCols, 0, s, w);
    launch_check("win_experts");
    return true;
}

}  // namespace strata::dsv41k::exl3::legacy
#endif
