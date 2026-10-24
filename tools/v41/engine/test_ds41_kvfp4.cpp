// test_ds41_kvfp4 [--cuda] - the --kv-fp4 caches (ds41_dense.cpp): rows fake-quantized as the graph does (the fused
// GGML_OP_DSV41_FAKE_QUANT), written with set_rows into NVFP4 (compressed KV, FP4_E4M3_16) / MXFP4 (indexer keys,
// FP4_POW2_32) and read back with get_rows must equal the fake-quant output BIT FOR BIT - as the BF16 caches do.
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

int main(int argc, char ** argv) {
    bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
    ggml_backend_t be = nullptr;
#if defined(GGML_USE_CUDA)
    if (cuda) be = ggml_backend_cuda_init(0);
#endif
    if (!be) { std::printf("needs --cuda (the FP4 set_rows quantizers are CUDA-only)\n"); return 2; }
    int fails = 0;
    struct Case { Ds41Quant mode; ggml_type t; int64_t W; const char * name; };
    for (const Case & cs : { Case{ Ds41Quant::FP4_E4M3_16, GGML_TYPE_NVFP4, 512, "comp nvfp4" },
                             Case{ Ds41Quant::FP4_POW2_32, GGML_TYPE_MXFP4, 128, "idxk mxfp4" } }) {
        const int64_t W = cs.W, R = 8192;
        std::mt19937_64 rng(20261015);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_int_distribution<int> ex(-12, 6);
        std::vector<float> x((size_t) (W * R));
        for (int64_t r = 0; r < R; ++r) {   // rows like the caches': per-row and per-block scale spread, a few zero blocks
            const float rs = std::ldexp(1.0f, ex(rng));
            for (int64_t i = 0; i < W; ++i) x[(size_t) (r * W + i)] = (r % 97 == 5 && i < 32) ? 0.0f : nd(rng) * rs * (1.0f + (float) (i % 7));
        }
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, W, R);
        ggml_tensor * idx = ggml_new_tensor_1d(c, GGML_TYPE_I32, R);
        ggml_tensor * cache = ggml_new_tensor_2d(c, cs.t, W, R);
        ggml_tensor * cbf = ggml_new_tensor_2d(c, GGML_TYPE_BF16, W, R);
        ggml_set_input(in); ggml_set_input(idx);
        ggml_tensor * fq = ggml_dsv41_fake_quant(c, in, (int) cs.mode);
        // read back: NVFP4 via get_rows (as the engine reads the compressed KV); MXFP4 rows of 128 are below the CUDA
        // get_rows kernel's 256-value super-block, and the engine reads them in the lightning indexer - so the raw
        // blocks come to the host and are decoded there (E8M0 x e2m1)
        ggml_tensor * wrote = ggml_set_rows(c, cache, fq, idx);
        ggml_tensor * back = cs.t == GGML_TYPE_NVFP4 ? ggml_get_rows(c, wrote, idx) : ggml_scale(c, fq, 1.0f);
        ggml_tensor * bback = ggml_get_rows(c, ggml_set_rows(c, cbf, fq, idx), idx);
        ggml_set_output(fq); ggml_set_output(back); ggml_set_output(bback);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, fq);
        ggml_build_forward_expand(gf, wrote);
        ggml_build_forward_expand(gf, back);
        ggml_build_forward_expand(gf, bback);
        ggml_backend_buffer_t cb = ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_get_default_buffer_type(be));
        (void) cb;
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(al, gf)) { std::printf("alloc failed\n"); return 1; }
        std::vector<int32_t> ids((size_t) R);
        for (int64_t r = 0; r < R; ++r) ids[(size_t) r] = (int32_t) r;
        ggml_backend_tensor_set(in, x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(idx, ids.data(), 0, ids.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> a(x.size()), b(x.size()), bb(x.size());
        ggml_backend_tensor_get(fq, a.data(), 0, a.size() * 4);
        ggml_backend_tensor_get(back, b.data(), 0, b.size() * 4);
        if (cs.t == GGML_TYPE_MXFP4) {
            std::vector<uint8_t> raw(ggml_nbytes(cache));
            ggml_backend_tensor_get(cache, raw.data(), 0, raw.size());
            static const float e2m1[16] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6 };
            for (size_t blk = 0; blk < b.size() / 32; ++blk) {
                const uint8_t * q = raw.data() + blk * 17;
                const float d = std::ldexp(1.0f, (int) q[0] - 127);
                for (int j = 0; j < 16; ++j) {
                    b[blk * 32 + j] = d * e2m1[q[1 + j] & 15];
                    b[blk * 32 + j + 16] = d * e2m1[q[1 + j] >> 4];
                }
            }
        }
        ggml_backend_tensor_get(bback, bb.data(), 0, bb.size() * 4);
        int64_t bad = 0, bad_bf = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::memcmp(&a[i], &bb[i], 4) != 0 && a[i] != bb[i]) ++bad_bf;
            if (a[i] == b[i]) continue;   // +0 / -0 count as equal (the packers keep no sign on zero codes)
            if (++bad <= 8) {
                const size_t b0 = i / 16 * 16;
                std::printf("  %s [%zu] x %.9g fq %.9g packed %.9g | block:", cs.name, i, x[i], a[i], b[i]);
                for (size_t k = b0; k < b0 + 16; ++k) std::printf(" %.6g", a[k]);
                std::printf("\n");
            }
        }
        std::printf("%s: %zu values, %lld differ (bf16 round trip: %lld differ)\n", cs.name, a.size(), (long long) bad, (long long) bad_bf);
        fails += bad > 0;
        ggml_gallocr_free(al);
        ggml_free(c);
    }
    ggml_backend_free(be);
    std::printf("%s\n", fails ? "DIFFERENCES" : "packed == fake-quant, bit for bit");
    return fails ? 1 : 0;
}
