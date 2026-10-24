// include/strata/kernels/dense_f16_gemm.hpp - the dense half's GEMM-shaped parts as dequantize + oneMKL FP16 GEMM.
//
// The expert path has run its two matmuls this way since S11 (native_expert_f16.dp.cpp: dequantize the quantized
// weights into FP16 device buffers, then one oneMKL FP16 GEMM with FP32 accumulation per matmul, measured at 60-70
// TFLOPS on the Battlemage card).  The dense half's q/kv/out projections, its shared expert and its router are the
// same shape of work - a quantized weight times a full-prompt activation block - but they ran as ggml-sycl's
// quantized matmul kernels at 0.8-12 TFLOPS against the same card.  This is the same route for them: no new
// arithmetic, the proven one.
//
// What it is NOT: the attention core (per-query K gather + cast, flash-attention over 128+512 keys of width 512,
// the indexer's block scores/top-k over every visible row).  That is kernel-shaped, not GEMM-shaped, and it stays
// on its own kernels.
//
// The caller owns the tensors and the queue; this header is SYCL-free on purpose (the engine's TUs are), so the
// queue crosses the boundary as void* - the same handle `ds41_sycl_queue_of` returns for a cudaStream_t.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// ggml type ids this entry point accepts, spelled out so the engine does not have to include ggml here.
enum DenseWType : int { DENSE_W_F32 = 0, DENSE_W_F16 = 1, DENSE_W_Q8_0 = 8 };

struct DenseGemmArgs {
    const void* key = nullptr; ///< identity of the weight TENSOR (not just its data): the dequantized-weight cache key
    const void* w = nullptr;   ///< weight: element (k, n) of matmul group g at g*w_group + n*K + k; K contiguous
    int wtype = DENSE_W_Q8_0;  ///< DenseWType of `w`
    const void* x = nullptr;   ///< activations, F32: element (k, m) of group g at g*x_group + m*x_row + k
    int xtype = DENSE_W_F32;
    void* y = nullptr;         ///< output, F32, (n, m) of group g at g*y_group + m*y_row + n (a ggml dst layout)
    int64_t K = 0, N = 0, M = 0;
    int64_t groups = 1;
    int64_t x_row = 0, x_group = 0;   ///< strides in ELEMENTS
    int64_t y_row = 0, y_group = 0;   ///< strides in ELEMENTS
    int64_t w_group = 0;              ///< stride between weight groups in ELEMENTS (groups > 1 only)
};

/// y = w . x for every group, through dequantize + oneMKL FP16 GEMM (FP32 accumulate).
/// Returns false when the shapes/types are not one this route supports, or the scratch did not fit: the caller then
/// runs the op it would have run before.  Never a silent wrong answer: every refusal is a `false` before the first
/// write.  `q` is a sycl::queue*.
bool dense_gemm_f16(void* q, const DenseGemmArgs& a);

/// What the scratch holds, for the startup/report line ("0 bytes" = nothing allocated yet).
int64_t dense_gemm_f16_scratch_bytes();

}  // namespace strata::kernels
