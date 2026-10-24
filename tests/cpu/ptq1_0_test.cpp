// tests/cpu/ptq1_0_test.cpp - PTQ1_0 ternary rows: layout bit-compatible with llama-tq / ik_llama.cpp, pack/unpack
// round trip, and the dot product (scalar = AVX2, both = float reference over the dequantized values).
// The oracle below is llama-tq's quantize_row_ptq1_0_ref / dequantize_row_ptq1_0 (github.com/LL4nc33/llama-tq,
// ggml/src/ggml-quants.c, MIT), restated with plain C types - an independent implementation of the same layout.
#include "strata/kernels/cpu/ptq1_0.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#ifdef PTQ1_TEST_STANDALONE   // g++ tests/cpu/ptq1_0_test.cpp src/kernels/cpu/ptq1_0.cpp -DPTQ1_TEST_STANDALONE (no engine)
namespace strata::kernels::cpu { bool cpu_avx2_ok() { return __builtin_cpu_supports("avx2"); } }
#endif
using namespace strata::kernels::cpu;

namespace oracle {   // ---- llama-tq reference (MIT), fp16 via _Float16 ----
static const size_t stages[3] = {32, 16, 8};
struct blk { uint8_t qs[24]; uint8_t qh[2]; _Float16 d; };
static_assert(sizeof(blk) == 28, "");
void quantize(const float* x, blk* y, long k) {
    for (long i = 0; i < k / 128; i++) {
        float amax = 0; for (int j = 0; j < 128; j++) amax = std::fmax(amax, std::fabs(x[j]));
        const float d = amax, id = d ? 1.f / d : 0.f; y[i].d = (_Float16) d;
        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) { const size_t c = stages[s];
            for (; j + c <= sizeof(y->qs); j += c) {
                for (size_t m = 0; m < c; ++m) { uint8_t q = 0;
                    for (size_t n = 0; n < 5; ++n) { int xi = (int) lroundf(x[m + n * c] * id) + 1; q *= 3; q += xi; }
                    q = ((uint16_t) q * 256 + (243 - 1)) / 243; y[i].qs[j + m] = q; }
                x += 5 * c; } }
        for (size_t h = 0; h < sizeof(y->qh); ++h) { uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) { int xi = (int) lroundf(x[h + m * sizeof(y->qh)] * id) + 1; q *= 3; q += xi; }
            q *= 3; q = ((uint16_t) q * 256 + (243 - 1)) / 243; y[i].qh[h] = q; }
        x += 4 * sizeof(y->qh);
    }
}
void dequant(const blk* x, float* y, long k) {
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    for (long i = 0; i < k / 128; ++i) { const float d = (float) x[i].d; size_t j = 0;
        for (size_t s = 0; s < 3; ++s) { const size_t c = stages[s];
            for (; j + c <= sizeof(x->qs); j += c)
                for (size_t n = 0; n < 5; ++n) for (size_t m = 0; m < c; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n]; int16_t xi = ((uint16_t) q * 3) >> 8; *y++ = (float) (xi - 1) * d; } }
        for (size_t n = 0; n < 4; ++n) for (size_t h = 0; h < sizeof(x->qh); ++h) {
            uint8_t q = x[i].qh[h] * pow3[n]; int16_t xi = ((uint16_t) q * 3) >> 8; *y++ = (float) (xi - 1) * d; } }
}
}  // namespace oracle

int main() {
    int fails = 0;
    auto check = [&](const char* name, bool ok) { std::printf("%s %s\n", ok ? "PASS" : "FAIL", name); fails += !ok; };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int N = 5120;   // DS4.1 hidden (40 blocks); also run the 2304 down width
    for (int n : {N, 2304, 128}) {
        std::vector<float> x(n); for (auto& v : x) v = nd(rng) * 0.02f;
        std::vector<uint8_t> ours(ptq1_0_row_bytes(n)); std::vector<oracle::blk> ref(n / 128);
        ptq1_0_quantize_ref(x.data(), ours.data(), n);
        oracle::quantize(x.data(), ref.data(), n);
        char nm[96]; std::snprintf(nm, sizeof nm, "n=%d quantize bytes == llama-tq reference", n);
        check(nm, std::memcmp(ours.data(), ref.data(), ours.size()) == 0);
        std::vector<float> a(n), b(n);
        ptq1_0_dequant(ours.data(), a.data(), n); oracle::dequant(ref.data(), b.data(), n);
        std::snprintf(nm, sizeof nm, "n=%d dequant == llama-tq dequant", n);
        check(nm, std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0);
        // pack/unpack round trip from codes (GSQ export path)
        std::vector<int8_t> t(n), t2(n); std::vector<float> d(n / 128), d2(n / 128);
        std::uniform_int_distribution<int> ud(-1, 1);
        for (auto& v : t) v = (int8_t) ud(rng);
        for (auto& v : d) v = std::fabs(nd(rng)) * 0.01f + 1e-4f;
        std::vector<uint8_t> pk(ptq1_0_row_bytes(n));
        ptq1_0_pack(t.data(), d.data(), pk.data(), n); ptq1_0_unpack(pk.data(), t2.data(), d2.data(), n);
        bool sc = true; for (int i = 0; i < n / 128; ++i) sc &= d2[i] == (float) (_Float16) d[i];
        std::snprintf(nm, sizeof nm, "n=%d pack/unpack codes exact, scales = fp16(d)", n);
        check(nm, std::memcmp(t.data(), t2.data(), n) == 0 && sc);
        // dot vs Q8_0 activation: scalar == AVX2 bitwise; both == float reference within float rounding
        std::vector<float> xa(n); for (auto& v : xa) v = nd(rng);
        struct Q8 { _Float16 d; int8_t qs[32]; };
        std::vector<Q8> q(n / 32);
        for (int k = 0; k < n / 32; ++k) {
            float am = 0; for (int i = 0; i < 32; ++i) am = std::fmax(am, std::fabs(xa[k * 32 + i]));
            const float dd = am / 127.f, id = dd ? 1.f / dd : 0.f; q[k].d = (_Float16) dd;
            for (int i = 0; i < 32; ++i) q[k].qs[i] = (int8_t) std::lround(xa[k * 32 + i] * id);
        }
        float s1 = 0, s2 = 0; double sref = 0;
        ptq1_0_vec_dot_q8_0_scalar(n, &s1, pk.data(), q.data());
        ptq1_0_vec_dot_q8_0(n, &s2, 0, pk.data(), 0, q.data(), 0, 1);
        std::vector<float> w(n); ptq1_0_dequant(pk.data(), w.data(), n);
        for (int i = 0; i < n; ++i) sref += (double) w[i] * (double) ((float) q[i / 32].d * q[i / 32].qs[i % 32]);
        std::snprintf(nm, sizeof nm, "n=%d dot AVX2 == scalar bitwise (%g)", n, s2);
        check(nm, std::memcmp(&s1, &s2, 4) == 0);
        std::snprintf(nm, sizeof nm, "n=%d dot == float reference (rel %.2e)", n, std::fabs(s1 - sref) / (std::fabs(sref) + 1e-12));
        check(nm, std::fabs(s1 - sref) <= 1e-5 * (std::fabs(sref) + 1e-3));
        // ptq1_0_rows_q8_0 (the pool's multi-token rows): 8 tokens x 6 rows of this n, each == the scalar dot within
        // fp32 reassociation (its sub-block sums are the same exact int32s; only the float accumulation order differs)
        {
            const int R = 6, NT = 8;
            std::vector<uint8_t> rows(R * ptq1_0_row_bytes(n));
            for (int r = 0; r < R; ++r) {
                std::vector<float> wr(n); for (auto& v : wr) v = nd(rng) * (r == 3 ? 1e-3f : 1.f);
                ptq1_0_quantize_ref(wr.data(), rows.data() + r * ptq1_0_row_bytes(n), n);
            }
            std::vector<std::vector<Q8>> qa(NT, q);
            for (int t = 1; t < NT; ++t) for (auto& b : qa[t]) for (auto& v : b.qs) v = (int8_t) std::max(-127, std::min(127, (int) v + (int) ud(rng) * t));
            std::vector<std::vector<float>> o(NT, std::vector<float>(R));
            float* op[NT]; const void* ap[NT];
            for (int t = 0; t < NT; ++t) { op[t] = o[t].data(); ap[t] = qa[t].data(); }
            bool ok = true; double worst = 0;
            for (int nt = 1; nt <= NT; ++nt) {
                ptq1_0_rows_q8_0(rows.data(), ptq1_0_row_bytes(n), n, ap, nt, op, 0, R);
                for (int t = 0; t < nt; ++t) for (int r = 0; r < R; ++r) {
                    float ref; ptq1_0_vec_dot_q8_0_scalar(n, &ref, rows.data() + r * ptq1_0_row_bytes(n), qa[t].data());
                    const double e = std::fabs(ref - o[t][r]) / (std::fabs(ref) + 1e-2);
                    worst = std::max(worst, e); ok &= e <= 1e-3;
                }
            }
            std::snprintf(nm, sizeof nm, "n=%d rows kernel (1-8 tokens) == scalar dot (worst rel %.2e)", n, worst);
            check(nm, ok);
        }
    }
    // mutation check: a wrong trit order must be caught by the layout test
    {
        int8_t t[128]; float d = 1.f; for (int i = 0; i < 128; ++i) t[i] = (int8_t) ((i % 3) - 1);
        uint8_t pk[28]; ptq1_0_pack(t, &d, pk, 128); std::swap(pk[0], pk[1]);
        int8_t t2[128]; float d2; ptq1_0_unpack(pk, t2, &d2, 128);
        check("mutation (swapped bytes) detected", std::memcmp(t, t2, 128) != 0);
    }
    std::printf(fails ? "SOME FAILED\n" : "ALL PASS\n");
    return fails ? 1 : 0;
}
