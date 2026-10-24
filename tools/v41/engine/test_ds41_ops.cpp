// test_ds41_ops DIR [--cuda] - ds41_fake_quant vs the torch reference (DIR/{x,fp8,fp4e,fp4p}.f32 from model.py's
// rounding via tools/v41/torch_kernels): counts of bit-identical values, worst mismatch.
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
#include <fstream>
#include <string>
#include <vector>

using namespace strata::v41;

static std::vector<float> load(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    std::vector<float> v((size_t) f.tellg() / 4);
    f.seekg(0);
    f.read((char *) v.data(), (std::streamsize) (v.size() * 4));
    return v;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    bool cuda = argc > 2 && std::string(argv[2]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { be = ggml_backend_cpu_init(); cuda = false; }
    const std::vector<float> x = load(dir + "/x.f32");
    const int64_t W = 512, R = (int64_t) x.size() / W;
    int fails = 0;
    for (auto [mode, name] : { std::pair{ Ds41Quant::FP8_POW2_32, "fp8" }, std::pair{ Ds41Quant::FP4_E4M3_16, "fp4e" },
                               std::pair{ Ds41Quant::FP4_POW2_32, "fp4p" } }) {
        const std::vector<float> want = load(dir + "/" + name + ".f32");
        ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, W, R);
        ggml_set_input(in);
        ggml_tensor * out = ds41_fake_quant(c, in, mode);
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
        ggml_backend_tensor_set(in, x.data(), 0, x.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> got(x.size());
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * 4);
        int64_t same = 0, bad = 0;
        double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::memcmp(&got[i], &want[i], 4) == 0 || (got[i] == 0 && want[i] == 0)) { ++same; continue; }
            if (std::isnan(want[i])) { ++same; continue; }   // torch: E4M3 scale overflow (amax/6 > 448) -> NaN; we saturate
            ++bad;
            const double rel = std::fabs(got[i] - want[i]) / std::max(1e-30, (double) std::fabs(want[i]));
            if (rel > worst) worst = rel;
            if (bad <= 3) std::printf("  %s [%zu] x %.9g got %.9g want %.9g\n", name, i, x[i], got[i], want[i]);
        }
        std::printf("%s %s: %lld / %zu identical, %lld differ (worst rel %.3g)\n", cuda ? "cuda" : "cpu", name,
                    (long long) same, got.size(), (long long) bad, worst);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "all identical");
    return fails ? 1 : 0;
}
