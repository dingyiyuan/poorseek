// xmx_bmg_bench: the grouped XMX expert GEMM (strata::prefill::xmx::grouped_f16) measured standalone on a Battlemage
// card at the routed prompt geometry, against the per-expert oneMKL f16 GEMM the engine falls back to.
//
// No engine: this file links xmx_moe.dp.cpp and the SYCL runtime only.  It is the AOT bmg build's bench
// (STRATA_SYCL_AOT=bmg-g31 -> -DSTRATA_XMX_SG16=1: joint_matrix 8x16x16, sub-group 16).  Build with
// sycl/tools/xmx_bmg_aot.sh, which is the recipe this bench was measured with.
//
// Geometry defaults are the real V4.1 expert shapes (models/dsv41-gsq/config.json text_config):
//   hidden_size 5120, moe_intermediate_size 2304, 384 routed experts, top-6 per token.
//   gate/up : X[m x 5120] . W_e[2*2304 x 5120]^T   (the engine's fused gate+up, N = 4608 = 2 * n_ff)
//   down    : X[m x 2304] . W_e[5120 x 2304]^T
// m is the mean rows per expert of one launch's group (kMaxGroup = 16 experts, as the engine's dq ring).
//
//   xmx_bmg_bench [--groups 16] [--reps 5] [--m 48,84,145,232] [--split-gu] [--json out.txt]
//
// Timing rule this bench exists to enforce: every timed section is read only after q.wait_and_throw().  Kernel time
// read off an event that has not been waited for is cudaErrorNotReady / a zero delta (that artefact produced the
// 7.5 us/entry figure the port was sized on; the real number was 50.86).
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strata/sycl_queue.hpp"
#include "strata/prefill/xmx_moe.hpp"

namespace xmx = strata::prefill::xmx;
using half_bits = uint16_t;

static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

struct Args {
    int groups = 16;                       // experts per launch = kMaxGroup (the engine's dq ring)
    int reps = 5;
    int K = 5120;                          // gate/up: hidden; down: moe_intermediate
    int n_ff = 2304;                       // moe_intermediate_size
    int hidden = 5120;                     // hidden_size
    bool split_gu = false;
    bool equal = false;          // exactly m rows per expert (the controlled geometry; the router gives gamma)                 // two N=2304 launches instead of one fused N=4608
    std::vector<int> ms = {48, 84, 145, 232};
    std::string json;
};

template <class F>
static double timeit(sycl::queue& q, F&& f, int reps) {
    f();                                     // warmup
    q.wait_and_throw();
    const auto a = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) f();
    q.wait_and_throw();                      // the whole point: the time is read only after the work is done
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    return ms / reps;
}

// CPU fp32 reference over a column subset of every row of the group, one expert at a time.
// cols: first/last `nc` columns of N (both ends catch an off-by-one in the n-tile mapping).
struct Err {
    double max_abs = 0, max_rel = 0, rel_l2 = 0, ref_absmax = 0;
    size_t bad = 0, n = 0;
};

static Err cpu_check2(const std::vector<half_bits>& hx, const std::vector<half_bits>& hw, int64_t row_off, int64_t rows,
                      int N, int K, int nc, const std::vector<float>& y, int64_t y_off, int ldy) {
    // the subset: every row of the expert, the first and last `nc` columns of N (both ends catch an n-tile off-by-one)
    std::vector<std::pair<size_t, float>> got_ref;   // index into y  ->  fp32 reference
    got_ref.reserve((size_t) rows * 2 * nc);
    for (int64_t r = 0; r < rows; ++r) {
        const half_bits* xr = hx.data() + (size_t) (row_off + r) * K;
        for (int c = 0; c < N; ++c) {
            if (!(c < nc || c >= N - nc)) continue;
            const half_bits* wr = hw.data() + (size_t) c * K;
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += h2f(xr[k]) * h2f(wr[k]);
            got_ref.emplace_back((size_t) (y_off + r) * ldy + c, acc);
        }
    }
    Err e;
    double sumsq = 0, refsq = 0;
    for (const auto& [idx, ref] : got_ref) e.ref_absmax = std::max(e.ref_absmax, std::fabs((double) ref));
    for (const auto& [idx, ref] : got_ref) {
        const double got = y[idx];
        ++e.n;
        if (!std::isfinite(got)) { ++e.bad; continue; }
        const double d = std::fabs(got - (double) ref);
        e.max_abs = std::max(e.max_abs, d);
        // relative to the tensor's own scale near zero: a bare |ref| denominator reports 1e0 for a cancelled entry
        e.max_rel = std::max(e.max_rel, d / (std::fabs((double) ref) + 1e-2 * e.ref_absmax));
        sumsq += d * d;
        refsq += (double) ref * ref;
    }
    e.rel_l2 = refsq > 0 ? std::sqrt(sumsq / refsq) : 0.0;
    return e;
}

// one full product ("gu" or "d") of the group, through the tree's kernel.
struct Prod {
    const char* name;
    int N, K;
    std::vector<half_bits*> w;             // one matrix per expert, row-major [N][K]
    std::vector<std::vector<half_bits>> hw; // host copies for the CPU reference
    std::vector<half_bits> hx;
    half_bits* dx = nullptr;
    float* dy = nullptr;                   // grouped XMX output
    float* ry = nullptr;                   // oneMKL output
};

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : (const char*) ""; };
        if (s == "--groups") a.groups = std::atoi(next());
        else if (s == "--reps") a.reps = std::atoi(next());
        else if (s == "--K") a.K = std::atoi(next());
        else if (s == "--n-ff") a.n_ff = std::atoi(next());
        else if (s == "--hidden") a.hidden = std::atoi(next());
        else if (s == "--split-gu") a.split_gu = true;
        else if (s == "--equal") a.equal = true;
        else if (s == "--json") a.json = next();
        else if (s == "--m") {
            a.ms.clear();
            std::string v = next();
            size_t p = 0;
            while (p < v.size()) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                a.ms.push_back(std::atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else {
            std::fprintf(stderr, "unknown arg %s\n", s.c_str());
            return 2;
        }
    }
    if (a.groups > xmx::kMaxGroup) {
        std::fprintf(stderr, "groups %d > kMaxGroup %d\n", a.groups, xmx::kMaxGroup);
        return 2;
    }
    sycl::queue q = dpct::get_in_order_queue();
    const sycl::device& dev = q.get_device();
    const auto gen = strata::intel_gpu_gen(dev);
    const char* gen_s = gen == strata::IntelGpuGen::Battlemage ? "Battlemage"
                        : gen == strata::IntelGpuGen::Alchemist ? "Alchemist" : "Unknown";
#if defined(STRATA_XMX_SG16)
    const char* variant = "STRATA_XMX_SG16 (joint_matrix 8x16x16, sub-group 16)";
#elif defined(STRATA_NO_DG2_XMX)
    const char* variant = "STRATA_NO_DG2_XMX (no grouped XMX compiled in)";
#else
    const char* variant = "sub-group 8 (joint_matrix 8x8x16)";
#endif
    std::printf("# xmx_bmg_bench\n");
    std::printf("device: %s  [%s]\n", dev.get_info<sycl::info::device::name>().c_str(), gen_s);
    std::printf("build : %s\n", variant);
    std::printf("xmx::mode() = %d (STRATA_PF_XMX)  kMaxGroup = %d\n", xmx::mode(), xmx::kMaxGroup);
#if defined(STRATA_NO_DG2_XMX)
    std::printf("SKIP: this build has no grouped XMX kernel\n");
    return 0;
#endif
    if (xmx::mode() == 0) std::printf("note: STRATA_PF_XMX is off; grouped_f16 is still called directly below\n");
    const int Ngu = a.split_gu ? a.n_ff : 2 * a.n_ff;
    const int Nd = a.hidden, Kd = a.n_ff;

    // ---- weights (static: the dequantise-into-slot cost is a separate kernel and is not what this measures) --------
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<Prod> prods(2);
    prods[0] = Prod{"gate/up", Ngu, a.K, {}, {}, {}, nullptr, nullptr, nullptr};
    prods[1] = Prod{"down", Nd, Kd, {}, {}, {}, nullptr, nullptr, nullptr};
    if (a.split_gu) std::printf("gate/up as two N=%d launches (the task's literal shape)\n", Ngu);
    for (auto& p : prods) {
        p.w.resize(a.groups);
        p.hw.resize(a.groups);
        for (int e = 0; e < a.groups; ++e) {
            p.hw[e].resize((size_t) p.N * p.K);
            for (auto& h : p.hw[e]) h = f2h(nd(rng) * 0.05f);
            p.w[e] = sycl::malloc_device<half_bits>((size_t) p.N * p.K, q);
            q.memcpy(p.w[e], p.hw[e].data(), (size_t) p.N * p.K * 2);
        }
        p.dy = sycl::malloc_device<float>((size_t) 16384 * p.N, q);   // 16 * 256 rows max below
        p.ry = sycl::malloc_device<float>((size_t) 16384 * p.N, q);
    }
    q.wait_and_throw();

    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(&q);

    FILE* jf = a.json.empty() ? nullptr : std::fopen(a.json.c_str(), "w");
    if (jf) std::fprintf(jf, "# product,m,rows,us_per_entry,tflops,onemkl_us_per_entry,relerr_max,relerr_l2\n");

    std::printf("\n%-8s %5s %6s %10s %10s %10s %9s %9s\n", "product", "m", "rows", "xmx us/ent", "xmx TFLOPS",
                "mkl us/ent", "xmx/mkl", "max_rel");
    std::printf("%s\n", std::string(78, '-').c_str());

    std::vector<int32_t> cnt(a.groups);
    std::gamma_distribution<double> gd(1.5, 1.0);   // shape 1.5: the routed row counts are skewed, not uniform
    for (size_t mi = 0; mi < a.ms.size(); ++mi) {
        const int m = a.ms[mi];
        const double scale = m / 1.5;
        int64_t rows = 0;
        for (int e = 0; e < a.groups; ++e) {
            cnt[e] = a.equal ? m : std::max(1, (int) (gd(rng) * scale));
            rows += cnt[e];
        }
        // offsets: the group's rows are packed back to back in expert order (the engine's layout)
        std::vector<int64_t> off(a.groups + 1, 0);
        for (int e = 0; e < a.groups; ++e) off[e + 1] = off[e] + cnt[e];
        if (rows > 16384) { std::fprintf(stderr, "group too big (%lld rows)\n", (long long) rows); return 2; }
        for (auto& p : prods) {
            p.hx.assign((size_t) rows * p.K, 0);
            for (auto& h : p.hx) h = f2h(nd(rng) * 0.25f);
            p.dx = sycl::malloc_device<half_bits>((size_t) rows * p.K, q);
            q.memcpy(p.dx, p.hx.data(), (size_t) rows * p.K * 2);
        }
        q.wait_and_throw();
        const int ncpu = 24;   // columns checked at each end by the CPU reference
        double use_prev = 0, use_r_prev = 0;

        for (size_t prod_i = 0; prod_i < prods.size(); ++prod_i) {
            auto& p = prods[prod_i];
            const char* nm = p.name;
            const int N = p.N, K = p.K;
            // --- the grouped XMX kernel (the thing under test) ---
            // --split-gu: the task's literal shape, gate and up as two N=n_ff launches instead of one fused N=2*n_ff
            const int calls = (a.split_gu && p.name[0] == 'g') ? 2 : 1;
            auto run_xmx = [&]() {
                for (int c = 0; c < calls; ++c)
                    xmx::grouped_f16(p.dx, p.w.data(), cnt.data(), a.groups, p.dy, N, K, (void*) &q);
            };
            const double t_x = timeit(q, run_xmx, a.reps);
            // --- the fallback: one oneMKL f16 GEMM per expert (what STRATA_PF_XMX=0 / m.gemm.f16 does) ---
            const float alpha = 1.f, beta = 0.f;
            auto run_mkl = [&]() {
                for (int c = 0; c < calls; ++c)
                for (int e = 0; e < a.groups; ++e) {
                    dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, cnt[e], K,
                                     &alpha, p.w[e], dpct::library_data_t::real_half, K, p.dx + off[e] * K,
                                     dpct::library_data_t::real_half, K, &beta, p.ry + off[e] * N,
                                     dpct::library_data_t::real_float, N, dpct::compute_type::f32);
                }
            };
            const double t_r = timeit(q, run_mkl, std::max(2, a.reps / 2));
            // --- correctness: XMX vs CPU fp32, and XMX vs oneMKL, on the same inputs ---
            run_xmx();
            q.wait_and_throw();
            std::vector<float> yx((size_t) rows * N), yr((size_t) rows * N);
            q.memcpy(yx.data(), p.dy, yx.size() * 4);
            q.memcpy(yr.data(), p.ry, yr.size() * 4).wait_and_throw();
            // CPU reference: every row of the group, the first and last `ncpu` columns of each expert's N.
            Err ce;
            double sumsq = 0, refsq = 0, absmax = 0;
            {
                int64_t o = 0;
                for (int e = 0; e < a.groups; ++e) {
                    const Err t = cpu_check2(p.hx, p.hw[e], off[e], cnt[e], N, K, ncpu, yx, o, N);
                    ce.max_abs = std::max(ce.max_abs, t.max_abs);
                    ce.max_rel = std::max(ce.max_rel, t.max_rel);
                    absmax = std::max(absmax, t.ref_absmax);
                    ce.bad += t.bad;
                    ce.n += t.n;
                    sumsq += t.rel_l2 * t.rel_l2;    // rel_l2 carries sqrt(sum d^2 / sum ref^2) per expert
                    refsq += 1.0;
                    o += cnt[e];
                }
                ce.rel_l2 = refsq > 0 ? std::sqrt(sumsq / refsq) : 0.0;
                ce.ref_absmax = absmax;
            }
            // a second, whole-tensor check against oneMKL: catches an expert/row mapping error anywhere in N
            Err me;
            {
                double s = 0, r = 0;
                for (size_t i = 0; i < yx.size(); ++i) {
                    if (!std::isfinite(yx[i]) || !std::isfinite(yr[i])) { ++me.bad; continue; }
                    const double d = std::fabs((double) yx[i] - (double) yr[i]);
                    const double a2 = std::fabs((double) yr[i]);
                    me.max_abs = std::max(me.max_abs, d);
                    me.max_rel = std::max(me.max_rel, d / (a2 + 1e-6));
                    s += d * d; r += a2 * a2;
                }
                me.rel_l2 = r > 0 ? std::sqrt(s / r) : 0.0;
            }
            const double use = t_x * 1000.0 / (double) rows;
            const double use_r = t_r * 1000.0 / (double) rows;
            const double flops = 2.0 * (double) rows * N * K;
            const double tflops = flops / (t_x * 1e-3) / 1e12;
            std::printf("%-8s %5d %6lld %10.2f %10.2f %10.2f %9.2f %9.1e\n", nm, m, (long long) rows, use, tflops,
                        use_r, use_r / use, ce.max_rel);
            double s_x = 0, s_r = 0;
            for (size_t i = 0; i < yx.size(); ++i) { s_x += yx[i]; s_r += yr[i]; }
            std::printf("         CPU fp32 ref: n=%zu max_abs=%.3e max_rel=%.3e rel_l2=%.3e nonfinite=%zu | vs oneMKL: "
                        "max_abs=%.3e max_rel=%.3e rel_l2=%.3e | sums x=%.6g mkl=%.6g\n", ce.n, ce.max_abs, ce.max_rel,
                        ce.rel_l2, ce.bad, me.max_abs, me.max_rel, me.rel_l2, s_x, s_r);
            std::printf("         (reference scale: max|Y| = %.3f, so max_abs is %.1e of it)\n", ce.ref_absmax,
                        ce.ref_absmax > 0 ? ce.max_abs / ce.ref_absmax : 0.0);
            if (jf)
                std::fprintf(jf, "%s,%d,%lld,%.3f,%.3f,%.3f,%.3e,%.3e\n", nm, m, (long long) rows, use, tflops, use_r,
                             ce.max_rel, me.rel_l2);
            if (prod_i + 1 == (int) prods.size())
                std::printf("         total expert phase: %.2f us/row (xmx) vs %.2f us/row (oneMKL per expert)\n",
                            use + use_prev, use_r + use_r_prev);
            use_prev = use; use_r_prev = use_r;
            if (ce.bad || me.bad) std::printf("  !! non-finite outputs: xmx_vs_cpu %zu, xmx_vs_mkl %zu\n", ce.bad, me.bad);
            p.dx = (sycl::free(p.dx, q), nullptr);
        }
    }
    // ---- dequant floor: the fp16 route writes this many bytes before any GEMM can start ------------------------
    {
        size_t bytes = 0;
        for (auto& p : prods) bytes += (size_t) a.groups * p.N * p.K * 2;
        const double t_fill = timeit(q, [&]() {
            for (auto& p : prods)
                for (int e = 0; e < a.groups; ++e) q.fill(p.w[e], (half_bits) 0, (size_t) p.N * p.K);
        }, 2);
        const double gbs = (double) bytes / (t_fill * 1e-3) / 1e9;
        std::printf("\ndequant floor: writing the %d experts' fp16 gate/up+down (%.2f GB) takes %.2f ms (%.0f GB/s) -\n"
                    "  every fp16 route (grouped XMX and oneMKL alike) pays this before its first GEMM.\n",
                    a.groups, (double) bytes / 1e9, t_fill, gbs);
        if (jf) std::fprintf(jf, "# dequant floor: %.2f GB in %.2f ms (%.0f GB/s)\n", (double) bytes / 1e9, t_fill, gbs);
    }
    if (jf) std::fclose(jf);
    std::printf("\nus/entry = microseconds per (row, expert) row of the product; TFLOPS = 2*rows*N*K/time.\n");
    std::printf("The fallback column is one oneMKL f16 GEMM per expert (fp16 in, fp32 out, f32 compute) - the engine's\n"
                "m.gemm.f16 path and the STRATA_PF_XMX=0 arithmetic, so the ratio is like-for-like on this card.\n");
    delete hd;
    for (auto& p : prods)
        for (auto* w : p.w) sycl::free(w, q);
    for (auto& p : prods) { sycl::free(p.dy, q); sycl::free(p.ry, q); }
    return 0;
}
