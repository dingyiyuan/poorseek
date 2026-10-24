#!/usr/bin/env bash
# tools/v41/engine/sycl_compat/compile_check.sh - S1 evidence (docs/xolotl-sycl-port-plan.md §5):
# compile the 8 real engine files that call the CUDA runtime against the SYCL shim, one icpx -fsycl -c per file.
# No linking, no ggml build, no model: this only proves the CUDA-runtime surface the engine uses is satisfied.
#
# usage: tools/v41/engine/sycl_compat/compile_check.sh [repo-root]
#   OUT=<dir>            where the .o files go (default /tmp/ds41_sycl_s1, never inside the source tree)
#   STRATA_GGML_DIR=<d>  a llama.cpp/ggml tree for the ggml headers the engine includes (default: the
#                        reference SYCL tree /home/dyy/Strata/third_party/llama.cpp/ggml)
set -u
ROOT=${1:-$(cd "$(dirname "$0")/../../../.." && pwd)}
ENGINE="$ROOT/tools/v41/engine"
GGML=${STRATA_GGML_DIR:-/home/dyy/Strata/third_party/llama.cpp/ggml}
OUT=${OUT:-/tmp/ds41_sycl_s1}
mkdir -p "$OUT" || exit 1
# setvars.sh is not `set -u` clean; the script is, from here on.
set +u
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u
CXX=${CXX:-icpx}

# DS41_SYCL: this shim.  STRATA_ENABLE_CUDA / DS4_MOE_CUDA / DS41_HAVE_CUDA / DS41_POOL_CUDA: the gates the engine
# puts around its CUDA paths - they must be ON, otherwise the CUDA calls are compiled out and nothing is tested.
DEFS="-DDS41_SYCL=1 -DSTRATA_ENABLE_CUDA=1 -DDS4_MOE_CUDA=1 -DDS41_HAVE_CUDA=1 -DDS41_POOL_CUDA=1 -DDS41_HAVE_DSPARK=1"
# sycl_compat FIRST: it turns <cuda_runtime.h> into the shim.
INCS="-I$ENGINE/sycl_compat -I$ENGINE -I$ROOT/include -I$GGML/include -I$GGML/src"
FLAGS="-fsycl -std=c++20 -O2 -c"

FILES="ds41_moe.cpp ds41_helper.cpp ds41_dense.cpp ds41_conv_pool.cpp ds41_generate.cpp ds41_pipeline.cpp \
ds41_cuda_policy.cpp ds41_dspark.cpp"

fail=0
printf '%-22s %-9s %s\n' FILE STATUS NOTE
for f in $FILES; do
    log="$OUT/${f%.cpp}.log"
    if $CXX $FLAGS $DEFS $INCS -o "$OUT/${f%.cpp}.o" "$ENGINE/$f" >"$log" 2>&1; then
        printf '%-22s %-9s %s\n' "$f" COMPILES "$(wc -l <"$log") warning lines"
    else
        fail=$((fail + 1))
        printf '%-22s %-9s %s\n' "$f" BLOCKED "$(grep -m1 'error:' "$log" | cut -c1-120)"
    fi
done

# the shim itself (SYCL header needed, no engine include path)
if $CXX $FLAGS $DEFS -I"$ENGINE/sycl_compat" -o "$OUT/ds41_sycl_compat.o" \
        "$ENGINE/sycl_compat/ds41_sycl_compat.cpp" >"$OUT/ds41_sycl_compat.log" 2>&1; then
    printf '%-22s %-9s %s\n' ds41_sycl_compat.cpp COMPILES "$(wc -l <"$OUT/ds41_sycl_compat.log") warning lines"
else
    fail=$((fail + 1))
    printf '%-22s %-9s %s\n' ds41_sycl_compat.cpp BLOCKED "$(grep -m1 'error:' "$OUT/ds41_sycl_compat.log" | cut -c1-120)"
fi

echo "logs + objects: $OUT"
exit $((fail != 0))
