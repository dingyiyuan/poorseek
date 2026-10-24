// sycl/src/kernels/cuda/iq_mmvq_refuse.dp.cpp - the one symbol the SYCL engine link needs from
// `iq_kernels.dp.cpp`'s MMVQ dispatch, which this tree cannot compile (see iq_kernels_fork_expert.dp.cpp's header:
// twenty `dpct::global_memory` errors against icpx 2026.2).
//
// `iq_mmvq(t, ...)` is the FALLBACK of native_mmvq: native_mmvq.dp.cpp:3786/3796 delegates types 7 (Q5_1) and
// 16/17/18/21/22/29 (the i-quants) to it, and the engine's own tier types are all native there - Q3_K (11) and Q2_0
// (42) have their own kernels, PTQ1_0 (143) is not in native_mmvq_supported() so no caller sends it.  So this body is
// unreachable for this model; it refuses loudly rather than pretending, exactly like the CUDA dispatch's default.
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

void iq_mmvq(int t, const void*, const void*, float*, int, int, int, void*) {
    std::fprintf(stderr,
                 "iq_mmvq: type %d needs the kernel TU iq_kernels.dp.cpp, which does not build under this icpx "
                 "(dpct::global_memory); the SYCL port has no MMVQ kernel for it\n", t);
    std::exit(1);
}

}  // namespace strata::kernels
