# tools/ds4/cmake/ggml_cuda.cmake - fork: ggml's own CUDA backend for the DS4 dense half (one include() line in the
# root CMakeLists.txt, right after upstream sets GGML_CUDA OFF; docs/UPSTREAM_SYNC.md).
# ggml's own CUDA backend is off unless asked for: the DS4 dense half (tools/ds4/ds4_dense) runs its graphs on a
# ggml backend and needs ggml-cuda to run them on the GPU (-DSTRATA_GGML_CUDA=ON, build-ds4-gpu).
option(STRATA_GGML_CUDA "Build ggml's CUDA backend too (for the DS4 dense half on the GPU)" OFF)
if(STRATA_GGML_CUDA)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
  # ggml defaults CUDA graphs OFF when it is not llama.cpp's own build; without them the DS4 dense half's ~17k tiny
  # kernels per token run at launch-overhead speed (measured: 97 ms/token of attention, 2.1M cudaLaunchKernel)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()
