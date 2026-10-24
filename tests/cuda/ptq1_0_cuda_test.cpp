// tests/cuda/ptq1_0_cuda_test.cpp - PTQ1_0 (type 143, the V4.1 ternary tier) on the GPU against the CPU kernels.
//   1. iq_dequant_f16(143) == ptq1_0_dequant rounded to fp16, bit for bit (every code and scale).
//   2. iq_mmvq(143) on quantize_q8_1_rows activations vs an exact double reference built from the same integers
//      (the q8_1 d and q of each 32-block are recomputed on the host the way the kernel does, d stored as fp16 in
//      block_q8_1.ds): only the fp32 summation order may differ.  Shapes: the V4.1 expert's gate/up (5120 -> rows) and down (2304 -> rows),
//      1..4 columns.
//   3. native_expert_grouped (the MoE path) for every gate/up x down pair of the V4.1 tiers {Q2_0, Q3_K, PTQ1_0} -
//      the calls a mixed-tier layer makes - 3 experts x 2 tokens, V4.1 shape 5120 / 2304.  Reference: the weights
//      from iq_dequant_f32 (bit-exact for 143 by test 1), the activations and the SwiGLU output quantized to q8_1 the
//      way the kernels do (fp16 d), double sums.  Left over: fp32 summation order and the odd h code that rounds
//      the other way.
// Exit 0 = pass.  Needs a CUDA device.
#include "strata/kernels/cpu/ptq1_0.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace gk = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static int failures = 0;

static void test_dequant(std::mt19937& rng) {
    const int64_t n = 256 * 64;   // 128 blocks
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    std::vector<uint8_t> q(cpu::ptq1_0_row_bytes(n));
    cpu::ptq1_0_quantize_ref(x.data(), q.data(), n);
    std::vector<float> ref(n);
    cpu::ptq1_0_dequant(q.data(), ref.data(), n);
    void* dq;
    uint16_t* dy;
    CK(cudaMalloc(&dq, q.size()));
    CK(cudaMalloc(&dy, n * 2));
    CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
    gk::iq_dequant_f16(cpu::kPtq1Type, dq, n, dy, nullptr);
    std::vector<uint16_t> got(n);
    CK(cudaMemcpy(got.data(), dy, n * 2, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        const __half want = __float2half(ref[i]);
        uint16_t wb;
        std::memcpy(&wb, &want, 2);
        if (wb != got[i] && bad++ < 5) std::printf("  dequant mismatch at %lld: want %g\n", (long long) i, ref[i]);
    }
    std::printf("dequant f16: %lld values, %d mismatches %s\n", (long long) n, bad, bad ? "FAIL" : "ok");
    failures += bad != 0;
    cudaFree(dq);
    cudaFree(dy);
}

// the q8_1 integers the kernel sees for one 32-value block (quantize_q8_1_kernel: d = amax/127, q = round(x/d))
static void q8_block(const float* x, float& d, int8_t* q) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    d = amax / 127.f;
    for (int i = 0; i < 32; ++i) q[i] = amax == 0.f ? 0 : (int8_t) std::lround(x[i] / d);
}

static void test_mmvq(std::mt19937& rng, int n_in, int n_out, int ncols) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> W((size_t) n_out * n_in), X((size_t) ncols * n_in);
    for (auto& v : W) v = nd(rng) * 0.02f;
    for (auto& v : X) v = nd(rng);
    const size_t rb = cpu::ptq1_0_row_bytes(n_in);
    std::vector<uint8_t> Q(rb * n_out);
    for (int r = 0; r < n_out; ++r) cpu::ptq1_0_quantize_ref(W.data() + (size_t) r * n_in, Q.data() + rb * r, n_in);
    // exact reference from the integers: sum_b d_w[b] * sum_k d_a[k] * sum_i t_i * q_i
    std::vector<int8_t> T((size_t) n_out * n_in);
    std::vector<float> D((size_t) n_out * (n_in / 128));
    for (int r = 0; r < n_out; ++r)
        cpu::ptq1_0_unpack(Q.data() + rb * r, T.data() + (size_t) r * n_in, D.data() + (size_t) r * (n_in / 128), n_in);
    std::vector<double> ref((size_t) ncols * n_out);
    for (int c = 0; c < ncols; ++c) {
        std::vector<float> da(n_in / 32);
        std::vector<int8_t> qa(n_in);
        for (int k = 0; k < n_in / 32; ++k) q8_block(X.data() + (size_t) c * n_in + 32 * k, da[k], qa.data() + 32 * k);
        for (int r = 0; r < n_out; ++r) {
            double s = 0;
            for (int b = 0; b < n_in / 128; ++b) {
                double sb = 0;
                for (int k = 0; k < 4; ++k) {
                    long isum = 0;
                    for (int i = 0; i < 32; ++i) {
                        const int e = 128 * b + 32 * k + i;
                        isum += (long) T[(size_t) r * n_in + e] * qa[e];
                    }
                    // the kernel reads d from block_q8_1.ds, a half2: q comes from the fp32 d, the dot uses fp16 d
                    sb += (double) __half2float(__float2half(da[4 * b + k])) * (double) isum;
                }
                s += (double) D[(size_t) r * (n_in / 128) + b] * sb;
            }
            ref[(size_t) c * n_out + r] = s;
        }
    }
    void *dw, *dx, *dxq;
    float* dy;
    CK(cudaMalloc(&dw, Q.size()));
    CK(cudaMalloc(&dx, X.size() * 4));
    CK(cudaMalloc(&dxq, (size_t) ncols * (n_in / 32) * 36));
    CK(cudaMalloc(&dy, (size_t) ncols * n_out * 4));
    CK(cudaMemcpy(dw, Q.data(), Q.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
    gk::quantize_q8_1_rows((const float*) dx, ncols, n_in, dxq, nullptr);
    gk::iq_mmvq(cpu::kPtq1Type, dw, dxq, dy, n_in, n_out, ncols, nullptr);
    std::vector<float> got((size_t) ncols * n_out);
    CK(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    for (double v : ref) scale = std::max(scale, std::fabs(v));
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - ref[i]) / scale);
    // fp32 accumulation of ~n_in/128 block terms: a few ulp of the largest output
    const bool ok = worst < 2e-5;
    std::printf("mmvq %4d -> %3d x %d cols: worst |gpu - exact| / max|exact| = %.2e %s\n", n_in, n_out, ncols, worst,
                ok ? "ok" : "FAIL");
    failures += !ok;
    cudaFree(dw);
    cudaFree(dx);
    cudaFree(dxq);
    cudaFree(dy);
}

// q8_1 round trip of a row as the kernels see it: q from the fp32 d, the dot with the fp16 d
static void q8_roundtrip(const float* x, int64_t n, std::vector<double>& out) {
    out.resize((size_t) n);
    for (int64_t k = 0; k < n / 32; ++k) {
        float d;
        int8_t q[32];
        q8_block(x + 32 * k, d, q);
        const double dh = __half2float(__float2half(d));
        for (int i = 0; i < 32; ++i) out[(size_t) (32 * k + i)] = dh * q[i];
    }
}

// random valid blocks of `type` for rows x n weights: random codes, small fp16 scales
static void random_rows(std::mt19937& rng, int type, int64_t rows, int64_t n, uint8_t* dst) {
    std::normal_distribution<float> nd(0.f, 1.f);
    if (type == cpu::kPtq1Type) {
        std::vector<float> w((size_t) n);
        const size_t rb = cpu::ptq1_0_row_bytes(n);
        for (int64_t r = 0; r < rows; ++r) {
            for (auto& v : w) v = nd(rng) * 0.02f;
            cpu::ptq1_0_quantize_ref(w.data(), dst + rb * r, n);
        }
        return;
    }
    const size_t bytes = gk::iq_row_bytes(type, n) * rows;
    for (size_t i = 0; i < bytes; ++i) dst[i] = (uint8_t) rng();
    const int bs = type == 42 ? 18 : 110, doff = type == 42 ? 0 : 108;   // Q2_0: d first; Q3_K: d last
    std::uniform_real_distribution<float> ud(0.002f, 0.01f);
    for (size_t b = 0; b < bytes / bs; ++b) {
        const __half h = __float2half(ud(rng) * (type == 11 ? 0.1f : 1.f));   // Q3_K: d * (sc - 32), |sc - 32| <= 32
        std::memcpy(dst + b * bs + doff, &h, 2);
    }
}

static void test_grouped(std::mt19937& rng, int gt, int dt) {
    const int64_t H = 5120, FF = 2304;
    const int NE = 3, PER = 2, NTOK = 4, NENT = NE * PER;
    if (!gk::native_expert_supported(gt, dt, H, FF)) {
        std::printf("grouped %3d/%3d: NOT SUPPORTED FAIL\n", gt, dt);
        ++failures;
        return;
    }
    const auto L = gk::native_expert_layout(gt, dt, H, FF);
    std::vector<uint8_t> blobs((size_t) NE * L.bytes);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * L.bytes;
        random_rows(rng, gt, FF, H, b);
        random_rows(rng, gt, FF, H, b + L.up_off);
        random_rows(rng, dt, H, FF, b + L.down_off);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> X((size_t) NTOK * H);
    for (auto& v : X) v = nd(rng);
    const int32_t tok[NENT] = {0, 1, 2, 3, 1, 3}, dst[NENT] = {0, 1, 2, 3, 4, 5};
    void *dblob, *dx, *dxq, *dscr;
    float *dout, *dw;
    unsigned long long* dptr;
    int32_t *dstart, *dn, *ddst, *dtok;
    CK(cudaMalloc(&dblob, blobs.size()));
    CK(cudaMalloc(&dx, X.size() * 4));
    CK(cudaMalloc(&dxq, (size_t) NTOK * H / 32 * 36));
    CK(cudaMalloc(&dscr, gk::native_expert_scratch_bytes(NENT, FF)));
    CK(cudaMalloc((void**) &dout, (size_t) NENT * H * 4));
    CK(cudaMalloc((void**) &dw, (size_t) FF * H * 4));
    CK(cudaMalloc((void**) &dptr, NE * 8));
    CK(cudaMalloc((void**) &dstart, (NE + 1) * 4));
    CK(cudaMalloc((void**) &dn, 4));
    CK(cudaMalloc((void**) &ddst, NENT * 4));
    CK(cudaMalloc((void**) &dtok, NENT * 4));
    CK(cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, X.data(), X.size() * 4, cudaMemcpyHostToDevice));
    unsigned long long ptr[NE];
    int32_t start[NE + 1];
    for (int e = 0; e < NE; ++e) ptr[e] = (unsigned long long) dblob + (unsigned long long) e * L.bytes;
    for (int e = 0; e <= NE; ++e) start[e] = e * PER;
    const int32_t ng = NE;
    CK(cudaMemcpy(dptr, ptr, sizeof ptr, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dstart, start, sizeof start, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn, &ng, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddst, dst, sizeof dst, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dtok, tok, sizeof tok, cudaMemcpyHostToDevice));
    CK(cudaMemset(dout, 0, (size_t) NENT * H * 4));
    gk::quantize_q8_1_rows((const float*) dx, NTOK, H, dxq, nullptr);
    gk::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, NE, NENT, dxq, dscr, dout, nullptr);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t) NENT * H);
    CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
    // reference
    std::vector<std::vector<double>> xa(NTOK);
    for (int t = 0; t < NTOK; ++t) q8_roundtrip(X.data() + (size_t) t * H, H, xa[t]);
    std::vector<float> Wg((size_t) FF * H), Wu((size_t) FF * H), Wd((size_t) H * FF);
    double worst = 0, scale = 0;
    std::vector<double> ref((size_t) NENT * H);
    for (int e = 0; e < NE; ++e) {
        const uint8_t* b = (const uint8_t*) dblob + (size_t) e * L.bytes;
        gk::iq_dequant_f32(gt, b, FF * H, dw, nullptr);
        CK(cudaMemcpy(Wg.data(), dw, Wg.size() * 4, cudaMemcpyDeviceToHost));
        gk::iq_dequant_f32(gt, b + L.up_off, FF * H, dw, nullptr);
        CK(cudaMemcpy(Wu.data(), dw, Wu.size() * 4, cudaMemcpyDeviceToHost));
        gk::iq_dequant_f32(dt, b + L.down_off, H * FF, dw, nullptr);
        CK(cudaMemcpy(Wd.data(), dw, Wd.size() * 4, cudaMemcpyDeviceToHost));
        for (int j = start[e]; j < start[e + 1]; ++j) {
            const auto& x = xa[tok[j]];
            std::vector<float> h((size_t) FF);
            for (int64_t r = 0; r < FF; ++r) {
                double g = 0, u = 0;
                for (int64_t c = 0; c < H; ++c) {
                    g += (double) Wg[(size_t) r * H + c] * x[(size_t) c];
                    u += (double) Wu[(size_t) r * H + c] * x[(size_t) c];
                }
                h[(size_t) r] = (float) (g / (1.0 + std::exp(-g)) * u);
            }
            std::vector<double> hq;
            q8_roundtrip(h.data(), FF, hq);
            for (int64_t r = 0; r < H; ++r) {
                double o = 0;
                for (int64_t c = 0; c < FF; ++c) o += (double) Wd[(size_t) r * FF + c] * hq[(size_t) c];
                ref[(size_t) dst[j] * H + r] = o;
                scale = std::max(scale, std::fabs(o));
            }
        }
    }
    double err2 = 0, ref2 = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::fabs(got[i] - ref[i]));
        err2 += (got[i] - ref[i]) * (got[i] - ref[i]);
        ref2 += ref[i] * ref[i];
    }
    const double rel = std::sqrt(err2 / std::max(ref2, 1e-300));
    // an h code rounding the other way moves one term by d_h * |w|: rel ~1e-4 at most; a layout bug is ~1
    const bool ok = rel < 1e-3 && std::isfinite(rel) && scale > 0;
    std::printf("grouped gu %3d / down %3d: rel L2 err %.2e, worst %.2e of max %.3g %s\n", gt, dt, rel, worst, scale,
                ok ? "ok" : "FAIL");
    failures += !ok;
    for (void* p : {dblob, dx, dxq, dscr, (void*) dout, (void*) dw, (void*) dptr, (void*) dstart, (void*) dn,
                    (void*) ddst, (void*) dtok})
        cudaFree(p);
}

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) { std::printf("no CUDA device: skipped\n"); return 0; }
    std::mt19937 rng(143);
    test_dequant(rng);
    for (int ncols : {1, 2, 4}) {
        test_mmvq(rng, 5120, 96, ncols);   // V4.1 gate/up rows: n_in = hidden 5120
        test_mmvq(rng, 2304, 96, ncols);   // V4.1 down rows: n_in = moe_inter 2304 (18 blocks per row)
    }
    for (int gt : {42, 11, cpu::kPtq1Type})
        for (int dt : {42, 11, cpu::kPtq1Type}) test_grouped(rng, gt, dt);
    std::printf("%s (%d failed)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
