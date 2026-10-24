# tools/v41/engine/sycl_compat/v41_engine_sycl.cmake - S2 wiring for the SYCL build of the DeepSeek-V4.1 engine
# (docs/xolotl-sycl-port-plan.md §5).  Not included by the CUDA/CPU builds; include it only from a SYCL configure.
#
# WHAT IT DOES
#   * builds the shim as its own translation unit (ds41_sycl_compat.cpp);
#   * ds41_sycl_engine(<target>) makes an existing engine target compile against the shim instead of NVIDIA's
#     headers: DS41_SYCL + this directory FIRST on the include path, and the flags the engine's GPU paths need.
#
# The five -D below are the engine's "a GPU backend is present" gates, not CUDA calls: STRATA_ENABLE_CUDA,
# DS4_MOE_CUDA, DS41_HAVE_CUDA, DS41_POOL_CUDA, DS41_HAVE_DSPARK.  They keep their CUDA-era names so the engine
# sources stay untouched; in this build they mean "SYCL GPU".  (DS41_HAVE_DSPARK is required: ds41_generate.cpp
# uses sp_pass/sp_acc outside the #if, so it does not compile without it - true for any build, CUDA included.)

set(DS41_SYCL_COMPAT_DIR ${CMAKE_CURRENT_LIST_DIR})

add_library(ds41_sycl_compat STATIC ${DS41_SYCL_COMPAT_DIR}/ds41_sycl_compat.cpp)
target_include_directories(ds41_sycl_compat PUBLIC ${DS41_SYCL_COMPAT_DIR})
target_compile_definitions(ds41_sycl_compat PUBLIC DS41_SYCL=1)
target_compile_features(ds41_sycl_compat PUBLIC cxx_std_20)
target_compile_options(ds41_sycl_compat PUBLIC -fsycl)

# ds41_sycl_engine(<target>): the target calls the CUDA runtime; make it call the shim.
function(ds41_sycl_engine target)
    # BEFORE: <cuda_runtime.h> must resolve to the shim, not to any CUDA toolkit directory on the include path.
    target_include_directories(${target} BEFORE PRIVATE ${DS41_SYCL_COMPAT_DIR})
    target_compile_definitions(${target} PRIVATE
        DS41_SYCL=1
        STRATA_ENABLE_CUDA=1
        DS4_MOE_CUDA=1
        DS41_HAVE_CUDA=1
        DS41_POOL_CUDA=1
        DS41_HAVE_DSPARK=1)
    target_compile_options(${target} PRIVATE -fsycl)
    target_link_options(${target} PRIVATE -fsycl)
    target_link_libraries(${target} PRIVATE ds41_sycl_compat)
endfunction()

# ds41_sycl_self_test(): the shim's runtime self-test (55 checks, no model, ~1 s).  Add it to ctest in S2.
function(ds41_sycl_self_test)
    add_executable(test_ds41_sycl_compat ${DS41_SYCL_COMPAT_DIR}/test_ds41_sycl_compat.cpp)
    target_link_libraries(test_ds41_sycl_compat PRIVATE ds41_sycl_compat)
    target_compile_options(test_ds41_sycl_compat PRIVATE -fsycl)
    target_link_options(test_ds41_sycl_compat PRIVATE -fsycl)
    add_test(NAME test_ds41_sycl_compat COMMAND test_ds41_sycl_compat)
endfunction()
