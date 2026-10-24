// sycl/src/program/ds41_transcode.dp.cpp - the port of `tools/v41/engine/ds41_transcode.cu`'s ONE function the
// GPU-expert tier links: ds41_add_f32 (the remote tier's rows added to an out_dev call, ds41_moe.cpp:1087).
//
// The other two entry points of that file (ds41_ptq1_to_q2_0 / _strided) serve run_chunk's MMQ path, which the
// SYCL engine build compiles out (no strata_mmq); they are not referenced by ds41_moe.o's undefined list, so they
// are not stubbed here - a later MMQ port adds them beside this.
#include "ds41_transcode.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"

#include <algorithm>
#include <cstdint>

void ds41_add_f32(float* dst, const float* src, size_t n, void* stream) {
    if (n == 0) return;
    const int64_t threads = 256;
    const int64_t grid = std::min<int64_t>(((int64_t) n + threads - 1) / threads, 4096);
    const int64_t stride = grid * threads;
    strata::q_of(stream)->parallel_for<dpct_kernel_name<class ds41_add_f32_kernel>>(
        sycl::range<1>((size_t) stride), [=](sycl::id<1> idx) {
            for (int64_t i = (int64_t) idx[0]; i < (int64_t) n; i += stride) dst[i] += src[i];
        });
}
