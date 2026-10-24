// test_ds41_fq [--cuda] (built with -DSTRATA_DSV41_FQ_OP=ON on a patched llama.cpp) - the fused
// GGML_OP_DSV41_FAKE_QUANT vs the stock-op composite ds41_fake_quant (ds41_ops.hpp) on the same backend, bit for bit,
// every mode, over ~2M synthetic activations: blocks from 1e-30 to 1e4 in scale, zeros, -0, subnormals, exact powers
// of two, block maxima at the scale boundaries (448 * 2^k, 6 * 2^k) and round-half ties.
#include "ds41_ops.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::v41;

static std::vector<float> synth(int64_t n) {
    std::mt19937_64 rng(20261011);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 11), ex(-40, 14);
    std::vector<float> x((size_t) n);
    for (int64_t b = 0; b < n / 32; ++b) {
        float * v = x.data() + b * 32;
        const int k = kind(rng);
        const float sc = std::ldexp(1.0f, ex(rng));
        for (int i = 0; i < 32; ++i) {
            switch (k) {
                case 0: v[i] = 0.0f; break;
                case 1: v[i] = (i & 1) ? -0.0f : 0.0f; break;
                case 2: v[i] = nd(rng) * 1e-40f; break;                                  // subnormal
                case 3: v[i] = std::ldexp((i & 1) ? -1.0f : 1.0f, ex(rng)); break;       // exact powers of two
                case 4: v[i] = nd(rng) * sc; if (i == 7) v[i] = 448.0f * sc; break;      // amax at the FP8 boundary
                case 5: v[i] = nd(rng) * sc; if (i == 3) v[i] = -6.0f * sc; break;       // amax at the FP4 boundary
                case 6: v[i] = sc * (0.25f * (float) (i % 16) + 0.125f); break;          // round-half ties
                case 7: v[i] = nd(rng) * 1e-30f; break;
                case 8: v[i] = nd(rng) * 1e4f; break;
                default: v[i] = nd(rng) * sc; break;
            }
        }
    }
    return x;
}

int main(int argc, char ** argv) {
    bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { be = ggml_backend_cpu_init(); cuda = false; }
    const int64_t W = 512, R = 4096;
    const std::vector<float> x = synth(W * R);
    int fails = 0;
    for (auto [mode, name] : { std::pair{ Ds41Quant::FP8_POW2_32, "fp8" }, std::pair{ Ds41Quant::FP4_E4M3_16, "fp4e" },
                               std::pair{ Ds41Quant::FP4_POW2_32, "fp4p" } }) {
        ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, W, R);
        ggml_set_input(in);
        ggml_tensor * ref = ds41_fake_quant(c, in, mode);
        ggml_tensor * fus = ggml_dsv41_fake_quant(c, in, (int) mode);
        ggml_set_output(ref);
        ggml_set_output(fus);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, fus);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
        ggml_backend_tensor_set(in, x.data(), 0, x.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> a(x.size()), b(x.size());
        ggml_backend_tensor_get(ref, a.data(), 0, a.size() * 4);
        ggml_backend_tensor_get(fus, b.data(), 0, b.size() * 4);
        int64_t bad = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
            if (++bad <= 5) std::printf("  %s [%zu] x %.9g composite %.9g fused %.9g\n", name, i, x[i], a[i], b[i]);
        }
        std::printf("%s %s: %zu values, %lld differ bitwise\n", cuda ? "cuda" : "cpu", name, a.size(), (long long) bad);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "fused == composite, bit for bit");
    return fails ? 1 : 0;
}
