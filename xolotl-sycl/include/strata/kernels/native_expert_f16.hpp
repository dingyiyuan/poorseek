// native_expert_f16.hpp - the routed experts' products as "dequantize, then GEMM": one expert's Q2_0 / Q3_K / PTQ1_0
// matrices are unpacked to FP16 on the device and every product of the expert runs as ONE oneMKL FP16 GEMM (FP32
// accumulate) over all of the rows that routed to it.
//
// WHY.  `native_expert_grouped` is a row-at-a-time kernel: one 32-lane sub-group owns one output row and loops the
// call's entries with `row_dot`, re-reading the weight row for every entry.  There is no M tile and no register/SLM
// reuse across entries, so its cost is flat in m - measured 47.06 us/entry at m = 48.3 and 50.86 us/entry at m = 231.9
// (1.39 TFLOPS = 0.76 % of this card's 183.5 TFLOPS FP16 peak; s4/step9/S10_XMX_ARITHMETIC.md).  The same products
// through dequantize + oneMKL f16 GEMM measure 3.18 / 2.12 / 1.22 / 1.07 us/entry at m = 48 / 84 / 145 / 232 on this
// card (docs/XMX_BMG_G31.md, section 3): 60-70 TFLOPS, 47x at the 8192-token geometry.  The dequantize is the price
// of the route: 71 MiB of FP16 written per expert (gate+up+down), 0.55 us/entry.
//
// SHAPE.  Blob = [gate rows | up rows | down rows] (NativeExpertLayout::up_off / down_off, native_expert_layout3):
//   gate/up : X[m x n_embd] . W_e[2 n_ff x n_embd]^T   (gate rows [0, n_ff), up rows [n_ff, 2 n_ff) after dequant)
//   down    : H[m x n_ff]   . W_e[n_embd x n_ff]^T
// with H = silu(min(gate, L.swiglu_limit)) * clamp(up, -L.swiglu_limit, L.swiglu_limit), the same SwiGLU the grouped
// kernel's fused pass computes (nothing here changes the arithmetic's intent, only its precision: FP16 weights and
// activations with FP32 accumulation in place of q8_1 activations with int8 dp4a accumulation).
//
// The results land in the caller's FP32 `parts` rows (one row per (token, expert) entry, addressed by `dst`), which is
// what the chunk path's weighted_rows_sum consumes.
#pragma once
#include <cstdint>
#include "strata/kernels/iq_kernels.hpp"

namespace strata::kernels {

/// Whether every matrix of this layout has a dequantizer here (Q2_0 = 42, Q3_K = 11, PTQ1_0 = 143 on both roles, up
/// the same type as the gate, no NVFP4 tail) and the dimensions divide the block sizes.
bool native_expert_f16_supported(const NativeExpertLayout& L) noexcept;

/// One expert's gate/up and down for `rows` routed entries, by dequantize + oneMKL FP16 GEMM.
///   x_tokens: [n_tokens][n_embd] FP32 activations (the chunk path's c_x);
///   tok[r], dst[r]: the entry's token index and its row in `parts` ([cap_entries][n_embd] FP32);
///   rows: the expert's entry count (>= 1).
/// Returns false (nothing launched, the caller's grouped kernel is still the fallback) only when a device buffer
/// cannot be allocated; a layout that `native_expert_f16_supported` rejects must not be passed.
bool native_expert_f16(const NativeExpertLayout& L, const void* blob, const float* x_tokens, const int32_t* tok,
                       const int32_t* dst, int64_t rows, float* parts, void* stream);

}  // namespace strata::kernels
