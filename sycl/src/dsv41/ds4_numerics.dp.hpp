// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
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
