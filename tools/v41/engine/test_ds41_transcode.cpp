// test_ds41_transcode - ds41_ptq1_to_q2_0 is lossless: random ternary rows packed by the reference packer
// (strata::kernels::cpu::ptq1_0_pack), transcoded on the GPU, then dequantized two ways - Strata's PTQ1_0 reference
// dequant of the source and ggml's own Q2_0 to_float of the result - must agree on every value, bit for bit.
// Also: d = +-0, subnormal, inf-free extremes; all-(-1), all-0, all-(+1) blocks; odd block counts.
#include "ds41_transcode.hpp"
#include "strata/kernels/cpu/ptq1_0.hpp"

#include <cuda_runtime.h>

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace sk = strata::kernels::cpu;

static bool check(const char* name, const std::vector<int8_t>& t, const std::vector<float>& d) {
    const int64_t n = (int64_t) t.size();
    std::vector<uint8_t> src(sk::ptq1_0_row_bytes(n));
    sk::ptq1_0_pack(t.data(), d.data(), src.data(), n);
    const size_t qb = ds41_q2_0_bytes_from_ptq1(src.size());
    void *ds = nullptr, *dq = nullptr;
    if (cudaMalloc(&ds, src.size()) != cudaSuccess || cudaMalloc(&dq, qb) != cudaSuccess) { std::printf("cudaMalloc failed\n"); return false; }
    cudaMemcpy(ds, src.data(), src.size(), cudaMemcpyHostToDevice);
    cudaMemset(dq, 0xAB, qb);
    ds41_ptq1_to_q2_0(ds, dq, src.size(), nullptr);
    std::vector<uint8_t> q(qb);
    cudaMemcpy(q.data(), dq, qb, cudaMemcpyDeviceToHost);
    cudaFree(ds);
    cudaFree(dq);
    if (cudaGetLastError() != cudaSuccess) { std::printf("CUDA error\n"); return false; }
    std::vector<float> a((size_t) n), b((size_t) n);
    sk::ptq1_0_dequant(src.data(), a.data(), n);
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(q.data(), b.data(), n);
    int64_t bad = 0;
    for (int64_t i = 0; i < n; ++i)
        if (std::memcmp(&a[(size_t) i], &b[(size_t) i], 4) != 0) {
            if (bad < 3) std::printf("  %s: element %lld: ptq1 %g q2_0 %g\n", name, (long long) i, a[(size_t) i], b[(size_t) i]);
            ++bad;
        }
    // and the values are the intended trits x d
    int64_t bad_t = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float want = (float) t[(size_t) i] * d[(size_t) (i / 128)];
        if (std::memcmp(&want, &b[(size_t) i], 4) != 0 && !(want == 0.0f && b[(size_t) i] == 0.0f)) ++bad_t;
    }
    std::printf("%s %s: %lld values, %lld differ from the PTQ1_0 dequant, %lld from trit*d\n", bad || bad_t ? "FAIL" : "ok  ",
                name, (long long) n, (long long) bad, (long long) bad_t);
    return bad == 0 && bad_t == 0;
}

int main() {
    std::mt19937 rng(1234);
    bool ok = true;
    for (int64_t nblk : {1, 3, 40, 18 * 2304 / 128 * 4, 5120 / 128 * 2304}) {   // incl. a whole gate matrix (2304 x 5120)
        const int64_t n = nblk * 128;
        std::vector<int8_t> t((size_t) n);
        std::vector<float> d((size_t) nblk);
        for (auto& v : t) v = (int8_t) ((int) (rng() % 3) - 1);
        // stored fp16 d: round-trip through ggml's fp16 so the reference's d equals what the block holds
        for (auto& v : d) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16((float) std::ldexp((double) (rng() % 1000 + 1) / 1000.0, (int) (rng() % 20) - 14) * (rng() & 1 ? 1.f : -1.f)));
        char name[64];
        std::snprintf(name, sizeof name, "random %lld blocks", (long long) nblk);
        ok &= check(name, t, d);
    }
    {   // edge blocks: constant trits, zero / tiny / large d
        std::vector<int8_t> t;
        std::vector<float> d;
        const float ds[] = { 0.0f, -0.0f, ggml_fp16_to_fp32(ggml_fp32_to_fp16(6.0e-8f)), 65504.0f, -65504.0f, 1.0f };
        for (float dv : ds)
            for (int c : { -1, 0, 1 }) {
                for (int i = 0; i < 128; ++i) t.push_back((int8_t) c);
                d.push_back(dv);
            }
        ok &= check("edge blocks", t, d);
    }
    std::printf(ok ? "ALL PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
