// src/kernels/cuda/fork/fork_mmvq_q2k_dot.cuh - Q2_K x q8_1 dot product (llama.cpp's vec_dot_q2_K_q8_1_impl_mmvq).
// Fork code kept out of upstream's native_mmvq.cu (one #include line there); included at a fixed point of that
// translation unit and uses its helpers (not a standalone header).
#pragma once

// Q2_K's 84-byte stride: contiguous v/x values, four 16-value groups per 64-value q4 part (llama.cpp's
// vec_dot_q2_K_q8_1_impl_mmvq).  Two 4-bit scales per sub-block: the low nibble is the quant scale, the high
// nibble a per-group min the q8_1 activation multiplies directly.
__device__ __forceinline__ float q2_k_q8_dot_impl(int v, const int* __restrict__ u,
                                                  const uint8_t* __restrict__ scales, half2 dm,
                                                  const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int sc = scales[i];
        const int vi = (v >> (2 * i)) & 0x03030303;
        sumf_d += d8[i] * (STRATA_DP4A(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m << 8;
        m |= m << 16;
        sumf_m += d8[i] * STRATA_DP4A(m, u[i], 0);
    }
    const float2 dm2f = __half22float2(dm);
    return dm2f.x * sumf_d - dm2f.y * sumf_m;
}
