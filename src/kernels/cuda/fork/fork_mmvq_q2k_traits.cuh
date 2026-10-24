// src/kernels/cuda/fork/fork_mmvq_q2k_traits.cuh - Q2KTraits for the native MMVQ kernels.
// Fork code kept out of upstream's native_mmvq.cu (one #include line there); included at a fixed point of that
// translation unit and uses its helpers (not a standalone header).
#pragma once

struct Q2KTraits {
    using Block = Q2KBlock;
    // QI2_K = 16 quant values, VDR_Q2_K_Q8_1_MMVQ = 1: one vec_dot call per 256-value block, 16 threads' worth.
    static constexpr int DIV = QK, T = 16, KBY = QK / Q8K, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int v; uint8_t scales[4]; half2 dm; int bq8_offset, u8; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.v = reinterpret_cast<const int*>(w->qs)[iqs];
        r.u8 = iqs % 8;
        r.dm = w->dm;
#pragma unroll
        for (int i = 0; i < 4; ++i) r.scales[i] = w->scales[scale_offset + 2 * i];
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        (void) iqs;
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const Q81Block* bq8i = x + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            u[i] = reinterpret_cast<const int*>(bq8i->qs)[r.u8];
        }
        return q2_k_q8_dot_impl(r.v, u, r.scales, r.dm, d8);
    }
};
