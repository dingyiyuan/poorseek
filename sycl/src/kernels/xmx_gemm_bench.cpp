// xmx_gemm_bench: 专家 GEMM 的三种结构在同一形状下的对照（明早一条命令出表）
//
//   fused     引擎的 set_tc_kernel 路线（解量化融进 MMA 内层）：xmx_gemm_iq
//   two-phase 先解量化到 f16 scratch（每个专家一次），再用 oneMKL 的 f16 GEMM
//   分解      dequant 单独计时（解量化本体多快）、mkl 单独计时（纯 GEMM 的上限）
//
// 为什么要扫 M：融合路线里每个解出的权重只被 BM=64 个激活行复用，两段路线是
// **每个专家解一次、M 行随便复用**。所以 rows/expert（= M）越大两段式越占优；
// 解码（M=1）永远融合式赢。这张表就是"预填充要不要改成两段式"的判据。
//
// 用法: xmx_gemm_bench [ty=16] [mode=gu|down|both] [reps=30] [M 或 sweep] [real|tiny]
//   real: n_embd 2560, n_ff 1280（DSV4-Flash 实际形状）  tiny: n_embd 2560, n_ff 640（旧小模型）
//   16 = IQ2_XXS
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }

int main(int argc, char** argv) {
    const int ty = argc > 1 ? std::atoi(argv[1]) : 16;
    const std::string mode = argc > 2 ? argv[2] : "both";
    const int reps = argc > 3 ? std::atoi(argv[3]) : 30;
    const std::string ms = argc > 4 ? argv[4] : "sweep";
    const bool real_shape = argc > 5 ? std::string(argv[5]) == "real" : true;
    (void) h2f;
    const int64_t n_embd = 2560, n_ff = real_shape ? 1280 : 640;

    sycl::queue* q = &dpct::get_in_order_queue();
    std::mt19937 rng(7);

    std::vector<int64_t> ms_list;
    if (ms == "sweep") ms_list = {1, 8, 32, 96, 256, 512, 900, 1800};
    else ms_list.push_back(std::atoll(ms.c_str()));

    for (const std::string m : {std::string("gu"), std::string("down")}) {
        if (mode != "both" && mode != m) continue;
        const bool gu = m == "gu";
        const int64_t K = gu ? n_embd : n_ff;
        const int64_t n_out = gu ? 2 * n_ff : n_embd;      // gate/up 交错成 2*n_ff 行
        const size_t row_bytes = strata::kernels::iq_row_bytes(ty, K);
        if (row_bytes == 0) { std::printf("type %d not supported\n", ty); return 2; }
        const int64_t rows_per_blob = gu ? n_ff : n_out;
        const size_t blob_bytes = (size_t) rows_per_blob * row_bytes;
        auto blob = [&]() {
            std::vector<uint8_t> b(blob_bytes);
            for (auto& x : b) x = (uint8_t) rng();
            const size_t blk = strata::kernels::iq_row_bytes(ty, 256);   // 一块的字节数
            for (size_t o = 0; o + 1 < b.size(); o += blk) { b[o] = 0x00; b[o + 1] = 0x3C; }   // d = 1.0
            return b;
        };
        std::vector<uint8_t> hg = blob(), hu = gu ? blob() : std::vector<uint8_t>();
        uint8_t* dg = sycl::malloc_device<uint8_t>(hg.size(), *q);
        uint8_t* du = gu ? sycl::malloc_device<uint8_t>(hu.size(), *q) : nullptr;
        uint16_t* dw = sycl::malloc_device<uint16_t>((size_t) n_out * K, *q);   // f16 scratch（两段式）
        q->memcpy(dg, hg.data(), hg.size()).wait();
        if (gu) q->memcpy(du, hu.data(), hu.size()).wait();
        dpct::blas::descriptor_ptr h = new dpct::blas::descriptor();
        h->set_queue(q);
        auto dequant = [&]() {
            if (gu) strata::kernels::iq_dequant_gu_f16(ty, dg, du, n_ff, K, dw, q);
            else strata::kernels::iq_dequant_f16(ty, dg, n_out * K, dw, q);
        };
        const double dq_values = (double) n_out * K;         // 一次解量化产出的权重个数

        std::printf("\n== ty %d  %s  (n_embd %lld, n_ff %lld, K %lld, n_out %lld, %s) ==\n", ty, m.c_str(),
                    (long long) n_embd, (long long) n_ff, (long long) K, (long long) n_out,
                    real_shape ? "real" : "tiny");
        std::printf("%6s %9s %9s %9s %9s %10s %10s %8s\n", "M", "deq ms", "mkl ms", "both ms", "fused ms",
                    "2ph TF/s", "fused TF/s", "2ph/fus");
        for (const int64_t M : ms_list) {
            uint16_t* dx = sycl::malloc_device<uint16_t>((size_t) M * K, *q);
            float* y_ref = sycl::malloc_device<float>((size_t) M * n_out, *q);
            float* y_xmx = sycl::malloc_device<float>((size_t) M * n_out, *q);
            std::vector<uint16_t> hx((size_t) M * K);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& x : hx) x = f2h(nd(rng));
            q->memcpy(dx, hx.data(), hx.size() * 2).wait();
            auto mkl = [&]() {
                const float alpha = 1.0f, beta = 0.0f;
                dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                                 (int) n_out, (int) M, (int) K, &alpha, dw, dpct::library_data_t::real_half,
                                 (int) K, dx, dpct::library_data_t::real_half, (int) K, &beta, y_ref,
                                 dpct::library_data_t::real_float, (int) n_out, dpct::compute_type::f32);
            };
            auto fused = [&]() {
                return strata::kernels::xmx_gemm_iq(ty, dg, du, K, (int) n_out, dx, (int) M, y_xmx, n_out, q);
            };
            dequant(); mkl(); q->wait();
            const bool okf = fused();   // 通用 XMX 路径（作为融合侧的代理；它拒了这个形状就只量两段式）
            q->wait();
            auto time = [&](auto&& fn) {
                for (int i = 0; i < 3; ++i) fn();
                q->wait();
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) fn();
                q->wait();
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
            };
            const double t_dq = time([&] { dequant(); });
            const double t_mkl = time([&] { mkl(); });
            const double t_both = time([&] { dequant(); mkl(); });
            const double t_fus = okf ? time([&] { fused(); }) : 0.0;
            const double flop = 2.0 * (double) M * n_out * K;
            if (okf)
                std::printf("%6lld %9.3f %9.3f %9.3f %9.3f %10.2f %10.2f %8.2f\n", (long long) M, t_dq, t_mkl,
                            t_both, t_fus, flop / (t_both * 1e9), flop / (t_fus * 1e9), t_fus / t_both);
            else   // 融合侧这个形状不可用（xmx_gemm_iq 只吃它自己支持的类型/形状）：只报两段式
                std::printf("%6lld %9.3f %9.3f %9.3f %9s %10.2f %10s %8s\n", (long long) M, t_dq, t_mkl,
                            t_both, "n/a", flop / (t_both * 1e9), "n/a", "n/a");
            if (M == ms_list.front()) {
                std::printf("   [解量化本体] %.3f ms / %.2f M 值 → %.1f Gval/s（每专家一次，与 M 无关）\n", t_dq,
                            dq_values / 1e6, dq_values / (t_dq * 1e6));
                std::vector<float> r((size_t) M * n_out), x((size_t) M * n_out);
                q->memcpy(r.data(), y_ref, r.size() * 4).wait();
                q->memcpy(x.data(), y_xmx, x.size() * 4).wait();
                double maxerr = 0, maxref = 0;
                size_t bad = 0;
                for (size_t i = 0; i < r.size(); ++i) {
                    const double e = std::fabs((double) r[i] - x[i]);
                    maxref = std::max(maxref, (double) std::fabs(r[i]));
                    maxerr = std::max(maxerr, e);
                    if (e > 1e-2 * (1.0 + std::fabs(r[i]))) ++bad;
                }
                std::printf("   [数值] max |ref| %.3g, max err %.3g, %zu of %zu 超 1%%\n", maxref, maxerr, bad,
                            r.size());
            }
            sycl::free(dx, *q); sycl::free(y_ref, *q); sycl::free(y_xmx, *q);
        }
        std::printf("  列义: 2ph/fus = fused 耗时 / 两段式耗时，> 1 表示两段式更快\n");
        sycl::free(dg, *q);
        if (du) sycl::free(du, *q);
        sycl::free(dw, *q);
    }
    return 0;
}
