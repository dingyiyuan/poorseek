// tools/v41/tests/fmt_ref.cpp - the C references behind test_v41_formats.py (stdin -> stdout, raw little-endian).
//   fmt_ref deq  TYPE N    : N weights of TYPE (8 Q8_0, 11 Q3_K, 42 Q2_0 via ggml's to_float; 143 via Strata's
//                            ptq1_0_dequant)                                      -> N float32
//   fmt_ref q8_0 N         : N float32 -> ggml quantize_row_q8_0_ref bytes
//   fmt_ref pack143 N      : N int8 codes then N/128 float32 scales -> Strata ptq1_0_pack bytes
// Built by the test (g++ against the build's libggml-base.a and src/kernels/cpu/ptq1_0.cpp).
#include "ggml.h"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

extern "C" void quantize_row_q8_0_ref(const float* x, void* y, int64_t k);
// ptq1_0.cpp's vec_dot asks this; the reference tool never calls the dot, so no AVX2 dispatch is needed
namespace strata::kernels::cpu { bool cpu_avx2_ok() { return false; } }

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const std::string mode = argv[1];
    std::vector<char> in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    std::vector<char> out;
    if (mode == "deq" && argc == 4) {
        const int t = std::atoi(argv[2]);
        const int64_t n = std::atoll(argv[3]);
        std::vector<float> y((size_t) n);
        if (t == 143) strata::kernels::cpu::ptq1_0_dequant((const uint8_t*) in.data(), y.data(), n);
        else {
            if ((int64_t) in.size() != (int64_t) ggml_row_size((ggml_type) t, n)) return 3;
            ggml_get_type_traits((ggml_type) t)->to_float(in.data(), y.data(), n);
        }
        out.assign((const char*) y.data(), (const char*) y.data() + n * 4);
    } else if (mode == "q8_0") {
        const int64_t n = std::atoll(argv[2]);
        out.resize(ggml_row_size(GGML_TYPE_Q8_0, n));
        quantize_row_q8_0_ref((const float*) in.data(), out.data(), n);
    } else if (mode == "pack143") {
        const int64_t n = std::atoll(argv[2]);
        out.resize((size_t) strata::kernels::cpu::ptq1_0_row_bytes(n));
        strata::kernels::cpu::ptq1_0_pack((const int8_t*) in.data(), (const float*) (in.data() + n), (uint8_t*) out.data(), n);
    } else return 2;
    std::fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
