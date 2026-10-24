// test_ds41_hcmix [--cuda] (built with -DSTRATA_DSV41_HC_MIX=ON on a patched llama.cpp) - the fused
// GGML_OP_DSV41_HC_MIXES vs the stock-op composite ds41_dense.cpp hc_mixes builds (mul, add, sigmoid, ggml_scale_bias
// / ggml_scale) on the same backend, bit for bit, over ~1M mix values per shape: normal mixes from 1e-30 to 1e30 in
// scale, zeros, -0, subnormals, values that saturate the sigmoid at both ends, and scale/base taken from the model's
// own weight ranges plus 0 / +-1 / powers of two and round-off-prone odd values.
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

static const float NEG_INF = -INFINITY;

static std::vector<float> synth_mixes(int64_t n) {
    std::mt19937_64 rng(20261013);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 11), ex(-40, 30);
    std::vector<float> x((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        const float sc = std::ldexp(1.0f, ex(rng));
        switch (kind(rng)) {
            case 0:  x[i] = 0.0f; break;
            case 1:  x[i] = (i & 1) ? -0.0f : 0.0f; break;
            case 2:  x[i] = nd(rng) * 1e-40f; break;                       // subnormal
            case 3:  x[i] = std::ldexp((i & 1) ? -1.0f : 1.0f, ex(rng)); break;
            case 4:  x[i] = nd(rng) * sc; break;                           // sigmoid saturation
            case 5:  x[i] = 88.0f * ((i & 1) ? -1.0f : 1.0f); break;       // expf overflows / underflows
            case 6:  x[i] = (float) (i % 8) - 4.0f; break;                 // exact small integers
            case 7:  x[i] = 0.5f * ((i & 1) ? -1.0f : 1.0f); break;
            case 8:  x[i] = nd(rng) * 1e30f; break;
            case 9:  x[i] = 1e-30f; break;
            default: x[i] = nd(rng); break;
        }
    }
    return x;
}

static std::vector<float> synth_w(int64_t n) {
    std::mt19937_64 rng(20261014);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> ex(-8, 4);
    std::vector<float> x((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        switch (i % 8) {
            case 0:  x[i] = 0.0f; break;
            case 1:  x[i] = 1.0f; break;
            case 2:  x[i] = -1.0f; break;
            case 3:  x[i] = std::ldexp((i & 1) ? -1.0f : 1.0f, ex(rng)); break;
            case 4:  x[i] = 0.1f; break;
            case 5:  x[i] = -0.0f; break;
            case 6:  x[i] = 3.1415927f; break;
            default: x[i] = nd(rng); break;
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

    const int64_t HC = 4;                       // the model's hc (kernel constant DSV4_HC)
    const int64_t NROW = 2 * HC + HC * HC;      // the real mixes->ne[0]: the pre/post rows + the comb rows
    const float eps = 1e-6f;
    int fails = 0;
    for (int64_t nt : { (int64_t) 131072, (int64_t) 1, (int64_t) 3, (int64_t) 257 }) {
        const std::vector<float> mx = synth_mixes(NROW * nt);
        const std::vector<float> sc = synth_w(2), bs = synth_w(2 * HC);

        ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, NROW, nt);
        ggml_tensor * tsc = ggml_new_tensor_1d(c, GGML_TYPE_F32, 2);
        ggml_tensor * tbs = ggml_new_tensor_1d(c, GGML_TYPE_F32, 2 * HC);
        ggml_set_input(in);
        ggml_set_input(tsc);
        ggml_set_input(tbs);

        // the composite, exactly as ds41_dense.cpp hc_mixes builds it
        ggml_tensor * m_pre  = ggml_view_2d(c, in, HC, nt, in->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, in, HC, nt, in->nb[1], (size_t) HC * in->nb[0]);
        ggml_tensor * pre  = ggml_add(c, ggml_mul(c, m_pre, ggml_view_1d(c, tsc, 1, 0)), ggml_view_1d(c, tbs, HC, 0));
        pre = ggml_scale_bias(c, ggml_sigmoid(c, pre), 1.0f, eps);
        ggml_tensor * post = ggml_add(c, ggml_mul(c, m_post, ggml_view_1d(c, tsc, 1, sizeof(float))),
                                      ggml_view_1d(c, tbs, HC, (size_t) HC * sizeof(float)));
        post = ggml_scale(c, ggml_sigmoid(c, post), 2.0f);

        // the fused op + the two views the engine takes of it
        ggml_tensor * mix = ggml_dsv41_hc_mixes(c, in, tsc, tbs, eps);
        ggml_tensor * f_pre  = ggml_view_2d(c, mix, HC, nt, mix->nb[1], 0);
        ggml_tensor * f_post = ggml_view_2d(c, mix, HC, nt, mix->nb[1], (size_t) nt * mix->nb[1]);

        ggml_tensor * outs[4] = { pre, post, f_pre, f_post };
        for (ggml_tensor * t : outs) ggml_set_output(t);
        ggml_cgraph * gf = ggml_new_graph(c);
        for (ggml_tensor * t : outs) ggml_build_forward_expand(gf, t);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al, gf);
        ggml_backend_tensor_set(in, mx.data(), 0, mx.size() * 4);
        ggml_backend_tensor_set(tsc, sc.data(), 0, sc.size() * 4);
        ggml_backend_tensor_set(tbs, bs.data(), 0, bs.size() * 4);
        ggml_backend_graph_compute(be, gf);

        int64_t bad = 0;
        for (int half = 0; half < 2; ++half) {
            const size_t sz = (size_t) HC * nt;
            std::vector<float> a(sz), b(sz);
            ggml_backend_tensor_get(outs[half], a.data(), 0, sz * 4);
            ggml_backend_tensor_get(outs[2 + half], b.data(), 0, sz * 4);
            for (size_t i = 0; i < sz; ++i) {
                if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
                if (++bad <= 4) std::printf("  %s [%zu] mix %.9g sc %.9g bs %.9g composite %.9g fused %.9g\n",
                                            half ? "post" : "pre", i, mx[i], sc[half], bs[i % HC + half * HC], a[i], b[i]);
            }
        }
        std::printf("%s hc=%lld nt=%lld mixes[%lld]: %lld values x2, %lld differ bitwise\n",
                    cuda ? "cuda" : "cpu", (long long) HC, (long long) nt, (long long) NROW,
                    (long long) (2 * HC * nt), (long long) bad);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    (void) NEG_INF;
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "fused == composite, bit for bit");
    return fails ? 1 : 0;
}
