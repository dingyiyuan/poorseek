// tools/v41/engine/sycl_compat/cuda_profiler_api.h - forwarding header for the SYCL build.  NOT used by a CUDA build: the build
// system puts this directory first on the include path ONLY for the SYCL target (DS41_SYCL), so the engine's
// `#include <cuda_profiler_api.h>` lands here instead of in the NVIDIA toolkit.  Same mechanism as
// include/strata/hip_compat/ for the HIP backend.
#pragma once
#if !defined(DS41_SYCL)
#error "cuda_profiler_api.h resolved to the SYCL compatibility shim but DS41_SYCL is not defined - do not put tools/v41/engine/sycl_compat on a CUDA build's include path."
#endif
#include "ds41_sycl_compat.hpp"
