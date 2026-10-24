#!/usr/bin/env bash
# xmx_bmg_aot.sh - the AOT Battlemage build of the grouped XMX expert GEMM, and the standalone bench around it.
#
# WHY THIS EXISTS
#   The tree's grouped XMX kernels (strata::prefill::xmx::grouped_f16, sycl/src/prefill/xmx_moe.dp.cpp) are written for
#   two device generations, and a build has exactly one of them:
#
#     sub-group 8,  joint_matrix 8x8x16   DG2 / Arc A-series (Alchemist)  - the JIT build (STRATA_SYCL_AOT empty)
#     sub-group 16, joint_matrix 8x16x16  Xe2 / Battlemage (bmg-g21, bmg-g31) - an AOT build for a bmg target
#
#   sycl/CMakeLists.txt picks the variant from STRATA_SYCL_AOT:
#     STRATA_SYCL_AOT matches ^(bmg|lnl)  -> -DSTRATA_XMX_SG16=1   (the sub-group 16 kernels, kSgW = 16)
#     STRATA_SYCL_AOT matches ^pvc        -> -DSTRATA_NO_DG2_XMX=1 (no grouped XMX compiled in)
#     empty (SPIR-V, JIT at run time)     -> the sub-group 8 / DG2 kernels, and xmx::mode() refuses on Battlemage
#                                            ("Battlemage needs the AOT bmg build")
#   and passes the device to ocloc at link time:
#     -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device <AOT> -options -cl-fp32-correctly-rounded-divide-sqrt"
#
#   Runtime selection: xmx::mode() (xmx_moe.dp.cpp) reads STRATA_PF_XMX; in a STRATA_XMX_SG16 build it accepts the
#   device, in the JIT build it refuses on IntelGpuGen::Battlemage.  The prefill host code (sycl/src/prefill/prefill.cpp,
#   the `xmx_on` / `xmx2` branches) then calls grouped_f16 instead of m.gemm.f16 / the fused IQ kernel.
#
#   Device enums involved: sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31 (runtime detection,
#   strata::intel_gpu_gen in sycl/include/strata/sycl_queue.hpp) and ocloc's `bmg-g31` (the AOT compile target).
#   `ocloc compile --help` lists it; `sycl-ls` shows the card as "Intel(R) Arc(TM) Pro B70 Graphics".
#
#   This script does NOT touch the engine build: it compiles two translation units out of tree (the kernel and the
#   bench) into ./xmx-bmg/build.  Nothing else in the tree is rebuilt, no binary is replaced.
#
# USAGE
#   sycl/tools/xmx_bmg_aot.sh [build-dir]          # build ./xmx-bmg/build/xmx_bmg_bench (or the given dir)
#   AOT=bmg-g21 sycl/tools/xmx_bmg_aot.sh          # another Battlemage SKU (bmg-g21) or lnl-a0 (Lunar Lake)
#   then:  ./xmx-bmg/build/xmx_bmg_bench --m 48,84,145,232
#
# RULES THIS BUILD KEEPS (the port's documented ones):
#   -fsycl-default-sub-group-size=32   every dpct'd kernel was written for 32-lane warps; only the XMX kernels pin
#                                      their own width ([[sycl::reqd_sub_group_size(kSgW)]])
#   -fp-model=precise                  the kernels are compared against ggml's value-safe build
#   -cl-fp32-correctly-rounded-divide-sqrt
#                                      must reach ocloc in an AOT build (it is a link-time backend option)
#   STRATA_SYCL_AOT=bmg-g31            == -DSTRATA_XMX_SG16=1, the sub-group 16 / 8x16x16 variant
set -eo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"      # the engine root (this script lives in sycl/tools)
OUT="${1:-$R/xmx-bmg/build}"
AOT="${AOT:-bmg-g31}"
CXX="${CXX:-icpx}"
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
if ! command -v "$CXX" >/dev/null 2>&1; then
    set +u                                    # oneAPI's setvars.sh is not nounset-clean
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
fi
mkdir -p "$OUT"

FLAGS=(-fsycl -O2 -std=c++20 -fp-model=precise -fsycl-default-sub-group-size=32
       -fsycl-device-code-split=per_kernel -DSTRATA_XMX_SG16=1
       -I"$R/sycl/include" -I"$R/include" -I"$R/sycl/third_party/ggml"
       -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations)
# the device to ocloc: g31 = Arc Pro B70 / B60 class, g21 = the smaller Battlemage die (Arc B580 class)
DEV_OPTS="-device ${AOT} -options -cl-fp32-correctly-rounded-divide-sqrt"
LINK=(-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "$DEV_OPTS" -qmkl=sequential)

echo "[xmx-bmg] kernel  xmx_moe.dp.cpp        (STRATA_XMX_SG16, AOT $AOT)"
"$CXX" "${FLAGS[@]}" -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "$DEV_OPTS" \
    -c "$R/sycl/src/prefill/xmx_moe.dp.cpp" -o "$OUT/xmx_moe_sg16.o"
echo "[xmx-bmg] harness xmx_bmg_bench.cpp"
"$CXX" "${FLAGS[@]}" -c "$R/sycl/src/kernels/xmx_bmg_bench.cpp" -o "$OUT/xmx_bmg_bench.o"
echo "[xmx-bmg] link    -> $OUT/xmx_bmg_bench  (oneMKL, AOT $AOT)"
"$CXX" "${LINK[@]}" "$OUT/xmx_moe_sg16.o" "$OUT/xmx_bmg_bench.o" -o "$OUT/xmx_bmg_bench"
echo "[xmx-bmg] ok: $OUT/xmx_bmg_bench"
