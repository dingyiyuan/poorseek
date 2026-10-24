// test_ds41_prom <file.gguf> [--rows 1,7,64,512,1024] [--refrows 16] [--layers 0] - the prompt-batch ("PROM") grouped
// expert kernels of the PTQ1_0 (143) and Q2_0 (42) tiers against (a) the existing native grouped kernels and (b) a
// dequant + F32 reference, on REAL expert blobs read out of the GGUF, at group (row) counts 1, 7, 64, 512, 1024.
//
// It drives `native_expert_grouped` directly with one group (one real expert) holding T entries, so exactly the
// tiles the PROM kernels target are exercised: the same expert applied to T token rows, with the kernel choice toggled
// in-process between the two arms (native_expert_prom_set_enabled).  The F32 reference is computed for the first
// `--refrows` tokens only (its cost is T x the full expert); the remaining rows are compared against the native arm.
#include "ds41_moe.hpp"

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::v41;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

void deq(int type, const uint8_t * src, float * y, int64_t n) {
    if (type == 143) kernels::cpu::ptq1_0_dequant(src, y, n);
    else ggml_get_type_traits((ggml_type) type)->to_float(src, y, n);
}

// The dequant + F32 reference of one expert blob ([gate | up | down]) on one token row (double accumulation).
struct Ref {
    int gt, dt;
    int64_t D, FF;
    size_t rg, rd;
    float swiglu_limit;
    void run(const uint8_t * blob, const float * x, float * out) const {
        std::vector<float> row((size_t) std::max(D, FF)), a((size_t) FF);
        for (int64_t i = 0; i < FF; ++i) {
            double sg = 0, su = 0;
            deq(gt, blob + (size_t) i * rg, row.data(), D);
            for (int64_t k = 0; k < D; ++k) sg += (double) row[(size_t) k] * x[k];
            deq(gt, blob + (size_t) FF * rg + (size_t) i * rg, row.data(), D);
            for (int64_t k = 0; k < D; ++k) su += (double) row[(size_t) k] * x[k];
            float gg = (float) sg, uu = (float) su;
            if (std::isfinite(swiglu_limit)) {
                uu = std::min(std::max(uu, -swiglu_limit), swiglu_limit);
                gg = std::min(gg, swiglu_limit);
            }
            a[(size_t) i] = gg / (1.0f + std::exp(-gg)) * uu;
        }
        for (int64_t i = 0; i < D; ++i) {
            deq(dt, blob + 2 * (size_t) FF * rg + (size_t) i * rd, row.data(), FF);
            double s = 0;
            for (int64_t k = 0; k < FF; ++k) s += (double) row[(size_t) k] * a[(size_t) k];
            out[i] = (float) s;
        }
    }
};

double rel(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return 1e9;
        const double d = (double) a[i] - (double) b[i];
        num += d * d; den += (double) b[i] * (double) b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

// One grouped call of T entries on one expert blob; out holds T x n_embd (entry e -> row e).
void launch_group(const kernels::NativeExpertLayout& L, const uint8_t * d_blob, const void * d_xq, void * d_scratch,
                  float * d_out, int64_t T, int64_t cap_groups, void * stream) {
    const unsigned long long ptr = (unsigned long long) (uintptr_t) d_blob;
    unsigned long long d_ptr = 0;
    int32_t d_start[2] = { 0, (int32_t) T }, d_ng = (int32_t) cap_groups;
    std::vector<int32_t> ent((size_t) T);
    for (int64_t t = 0; t < T; ++t) ent[(size_t) t] = (int32_t) t;
    unsigned long long * gp = nullptr;
    int32_t * gs = nullptr, * gn = nullptr, * ge = nullptr;
    ck(cudaMalloc((void**) &gp, sizeof(unsigned long long)), "malloc gp");
    ck(cudaMalloc((void**) &gs, 2 * sizeof(int32_t)), "malloc gs");
    ck(cudaMalloc((void**) &gn, sizeof(int32_t)), "malloc gn");
    ck(cudaMalloc((void**) &ge, (size_t) T * sizeof(int32_t)), "malloc ge");
    ck(cudaMemcpyAsync(gp, &ptr, sizeof(ptr), cudaMemcpyHostToDevice, (cudaStream_t) stream), "cp ptr");
    ck(cudaMemcpyAsync(gs, d_start, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream), "cp start");
    ck(cudaMemcpyAsync(gn, &d_ng, sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream), "cp ng");
    ck(cudaMemcpyAsync(ge, ent.data(), (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, (cudaStream_t) stream), "cp ent");
    kernels::native_expert_grouped(L, gp, gs, gn, ge, ge, cap_groups, T, d_xq, d_scratch, d_out, stream, cap_groups);
    ck(cudaStreamSynchronize((cudaStream_t) stream), "sync");
    ck(cudaFree(gp), "free"); ck(cudaFree(gs), "free"); ck(cudaFree(gn), "free"); ck(cudaFree(ge), "free");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_ds41_prom <file.gguf> [--rows 1,7,64,512,1024] [--refrows 16]\n"); return 2; }
    std::string path = argv[1];
    std::vector<int64_t> rows = { 1, 7, 64, 512, 1024 };
    int refrows = 16;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) {
            rows.clear();
            std::stringstream ss(argv[++i]);
            std::string x;
            while (std::getline(ss, x, ',')) rows.push_back(std::atoll(x.c_str()));
        } else if (a == "--refrows" && i + 1 < argc) refrows = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    Ds41FileBlobs src(path);
    std::string err;
    if (!src.open(err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    const Ds41MoeGeom & g = src.geom();
    std::printf("geom: n_embd %lld n_ff %lld layers %lld experts %lld classes %zu\n", (long long) g.n_embd,
                (long long) g.n_ff, (long long) g.n_layers, (long long) g.n_experts, g.cls_gu.size());

    kernels::native_expert_prom_set_force(true);   // validate the kernels at every format/row count, not the tuned gate
    void * d_blob = nullptr, * d_xq = nullptr, * d_scratch = nullptr, * d_out = nullptr, * d_xf = nullptr;
    std::vector<uint8_t> blob(src.blob_bytes());
    std::vector<float> x, native_out, prom_out, want;
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    double worst_f32 = 0, worst_nat = 0;

    for (size_t c = 0; c < g.cls_gu.size(); ++c) {
        const int gt = g.cls_gu[c], dt = g.cls_d[c];
        if (gt != 143 && gt != 42 && dt != 143 && dt != 42) continue;   // this slice's tiers
        // one real (layer, expert) of this class
        int64_t l = -1, e = -1;
        for (int64_t ll = 0; ll < g.n_layers && l < 0; ++ll)
            for (int64_t ee = 0; ee < g.n_experts; ++ee)
                if (g.cls(ll, ee) == (int) c) { l = ll; e = ee; break; }
        if (l < 0) { std::printf("class %zu (%d/%d): no expert in the file\n", c, gt, dt); continue; }
        const size_t bb = src.expert_bytes(l, e);
        if (bb > blob.size()) { std::fprintf(stderr, "blob buffer too small\n"); return 1; }
        src.read(l, e, blob.data());
        const kernels::NativeExpertLayout L = kernels::native_expert_layout(gt, dt, g.n_embd, g.n_ff);
        const bool supported = kernels::native_expert_supported(gt, dt, g.n_embd, g.n_ff);
        const bool pg = (gt == 143 || gt == 42) && (g.n_embd % 128 == 0) && (8 * (g.n_embd + (g.n_embd / 32) * 4) <= 48 * 1024);
        const bool pd = (dt == 143 || dt == 42) && (g.n_ff % 128 == 0) && (8 * (g.n_ff + (g.n_ff / 32) * 4) <= 48 * 1024);
        std::printf("class %zu: gu %d down %d, L%d e%lld, blob %zu B, %s%s%s\n", c, gt, dt, (int) l, (long long) e, bb,
                    supported ? "" : "UNSUPPORTED ", pg ? "prom-gu " : "gu-native ", pd ? "prom-down" : "down-native");
        if (!supported) { ++fails; continue; }
        Ref ref { gt, dt, g.n_embd, g.n_ff, kernels::iq_row_bytes(gt, g.n_embd), kernels::iq_row_bytes(dt, g.n_ff),
                  g.swiglu_limit };
        ck(cudaMalloc(&d_blob, bb), "malloc blob");
        ck(cudaMemcpy(d_blob, blob.data(), bb, cudaMemcpyHostToDevice), "cp blob");
        const int64_t D = g.n_embd, FF = g.n_ff;
        for (int64_t T : rows) {
            x.assign((size_t) (T * D), 0.0f);
            for (auto & v : x) v = nd(rng) * 0.3f;
            const size_t xq_bytes = (size_t) (T * D / 32) * 36;   // one q8_1 block (32 int8 + fp16 d + fp16 s)
            ck(cudaMalloc(&d_xq, xq_bytes), "malloc xq");
            ck(cudaMalloc(&d_scratch, kernels::native_expert_scratch_bytes(T, FF)), "malloc scratch");
            ck(cudaMalloc(&d_out, (size_t) (T * D) * sizeof(float)), "malloc out");
            ck(cudaMalloc(&d_xf, (size_t) (T * D) * sizeof(float)), "malloc xf");
            ck(cudaMemcpy(d_xf, x.data(), (size_t) (T * D) * sizeof(float), cudaMemcpyHostToDevice), "cp xf");
            kernels::quantize_q8_1_rows((const float*) d_xf, T, D, d_xq, nullptr);   // device pointers
            ck(cudaDeviceSynchronize(), "q8 sync");
            cudaEvent_t ev0, ev1;
            ck(cudaEventCreate(&ev0), "ev"); ck(cudaEventCreate(&ev1), "ev");
            float ms_nat = 1e9f, ms_prom = 1e9f;
            for (int rep = 0; rep < 5; ++rep) {
                kernels::native_expert_prom_set_enabled(false);
                ck(cudaEventRecord(ev0, nullptr), "rec");
                launch_group(L, (const uint8_t*) d_blob, d_xq, d_scratch, (float*) d_out, T, 1, nullptr);
                ck(cudaEventRecord(ev1, nullptr), "rec");
                ck(cudaEventSynchronize(ev1), "ev sync");
                float ms = 0; ck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
                ms_nat = std::min(ms_nat, ms);
                kernels::native_expert_prom_set_enabled(true);
                ck(cudaEventRecord(ev0, nullptr), "rec");
                launch_group(L, (const uint8_t*) d_blob, d_xq, d_scratch, (float*) d_out, T, 1, nullptr);
                ck(cudaEventRecord(ev1, nullptr), "rec");
                ck(cudaEventSynchronize(ev1), "ev sync");
                ck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
                ms_prom = std::min(ms_prom, ms);
            }
            ck(cudaEventDestroy(ev0), "ev"); ck(cudaEventDestroy(ev1), "ev");
            kernels::native_expert_prom_set_enabled(false);
            launch_group(L, (const uint8_t*) d_blob, d_xq, d_scratch, (float*) d_out, T, 1, nullptr);
            native_out.assign((size_t) (T * D), 0.0f);
            ck(cudaMemcpy(native_out.data(), d_out, (size_t) (T * D) * sizeof(float), cudaMemcpyDeviceToHost), "d2h native");
            kernels::native_expert_prom_set_enabled(true);
            launch_group(L, (const uint8_t*) d_blob, d_xq, d_scratch, (float*) d_out, T, 1, nullptr);
            prom_out.assign((size_t) (T * D), 0.0f);
            ck(cudaMemcpy(prom_out.data(), d_out, (size_t) (T * D) * sizeof(float), cudaMemcpyDeviceToHost), "d2h prom");
            kernels::native_expert_prom_set_enabled(false);
            const double r_nat = rel(prom_out, native_out);
            // the reference on the first `refrows` tokens
            const int64_t R = std::min<int64_t>(T, refrows);
            want.assign((size_t) (R * D), 0.0f);
            std::vector<float> one((size_t) D);
            for (int64_t t = 0; t < R; ++t) {
                ref.run(blob.data(), x.data() + (size_t) t * D, one.data());
                std::copy(one.begin(), one.end(), want.begin() + (size_t) t * D);
            }
            const std::vector<float> pv(prom_out.begin(), prom_out.begin() + (size_t) (R * D));
            const std::vector<float> nv(native_out.begin(), native_out.begin() + (size_t) (R * D));
            const double r_prom = rel(pv, want), r_natf = rel(nv, want);
            worst_f32 = std::max(worst_f32, r_prom);
            worst_nat = std::max(worst_nat, r_nat);
            // pass: prom within the tier's tolerance of the reference, no worse than the native arm, and the two
            // kernels agree to within float summation order (~1e-3; exact integer chain, different scale order)
            const bool ok = (r_prom < 1.8e-2) && (r_natf < 1.8e-2) && (r_prom <= r_natf * 1.05 + 1e-5) && (r_nat < 1e-2);
            fails += !ok;
            std::printf("  %s T %5lld  prom-vs-native %.3e | prom-vs-F32 %.3e (native-vs-F32 %.3e, %lld rows) | kernel %.3f ms vs %.3f ms = %.2fx\n",
                        ok ? "ok  " : "FAIL", (long long) T, r_nat, r_prom, r_natf, (long long) R, ms_prom, ms_nat,
                        ms_nat / ms_prom);
            ck(cudaFree(d_xf), "free xf");
            ck(cudaFree(d_xq), "free xq");
            ck(cudaFree(d_scratch), "free scratch");
            ck(cudaFree(d_out), "free out");
        }
        ck(cudaFree(d_blob), "free blob");
    }
    std::printf("worst prom-vs-native %.3e, worst prom-vs-F32 %.3e -> %s\n", worst_nat, worst_f32, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
