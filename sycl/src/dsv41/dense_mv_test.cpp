// dense_mv_test.cpp - the dense matvec (`iq::mv`) against ggml's own decode of the same blocks.
//
// The expert kernels have their gate in tests/dsv41/iq_expert_test.cpp (the port runs it as ds4_iq_test, ALL OK for
// IQ2_XXS / IQ2_XS / IQ1_M).  The DENSE matrices go through `iq::mv` instead - the shared expert, the head, the hc
// projections - and the 5- and 6-bit K-quants are only reachable there: iq_expert_test refuses Q5_K ("gate_up refused").
// That is the path a first light on the B70 exposed: layer 27 is the first layer whose shared expert's down matrix is
// Q5_K (25/26 have Q8_0) and its output jumped from rms 2.6 to 336; the head is Q5_K too, which is why every token came
// out as the same special id with logits in the hundreds.
//
//   ds4_dense_mv_test [--type 13] [--k 5120] [--n 64]
//
// A random legal Q5_K matrix and a random activation: the device's y against the same dot in double from ggml's
// to_float of those bytes and the host's q8 rounding.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"                       // ggml_get_type_traits(Q5_K)->to_float: ggml's decode of the same blocks
#include "ds4_iq.dp.hpp"                // [port] the migrated header (no cuda_runtime.h)
#include "strata/dsv41/hip_or_cuda.h"   // [port] HIP names on the SYCL runtime

namespace iq = strata::ds4::gguf::iq;

static uint16_t f2h(float f) {   // round to nearest even, normal range (as the fixture generator needs)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    m = (m + 0x1000u + ((m >> 13) & 1u)) >> 13;
    if (m == 0x400u) { m = 0; ++e; if (e >= 31) return (uint16_t) (s | 0x7c00u); }
    return (uint16_t) (s | (uint32_t) (e << 10) | m);
}

static int block_bytes(enum ggml_type t) {
    const auto* tr = ggml_get_type_traits(t);
    return (int) (tr->type_size);
}

int main(int argc, char** argv) {
    enum ggml_type type = GGML_TYPE_Q5_K;
    int k = 5120, n = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--type" && i + 1 < argc) {
            const int t = std::atoi(argv[++i]);
            type = (enum ggml_type) t;
        } else if (a == "--k" && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (a == "--n" && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        }
    }
    const auto* tr = ggml_get_type_traits(type);
    if (tr == nullptr || tr->to_float == nullptr) { std::printf("no ggml decoder for type %d\n", (int) type); return 2; }
    const int blck = (int) tr->blck_size;
    if (k % blck != 0) { std::printf("k %d is not a multiple of %d\n", k, blck); return 2; }
    const int rb = (int) (k / blck * (int) tr->type_size);   // bytes a row

    std::mt19937_64 rng(1234);
    std::vector<uint8_t> host_w((size_t) n * rb);
    if (const char* wf = std::getenv("MV_WFILE")) {   // the model's own blocks: MV_WFILE=<gguf or shard> MV_WOFF=<byte>
        const long long off = std::atoll(std::getenv("MV_WOFF") ? std::getenv("MV_WOFF") : "0");
        std::FILE* f = std::fopen(wf, "rb");
        if (f == nullptr || std::fseek(f, (long) off, SEEK_SET) != 0 ||
            std::fread(host_w.data(), 1, host_w.size(), f) != host_w.size()) {
            std::printf("cannot read %zu bytes at %lld of %s\n", host_w.size(), off, wf);
            return 2;
        }
        std::fclose(f);
    } else
    for (int r = 0; r < n; ++r)
        for (int b = 0; b < k / blck; ++b) {
            uint8_t* blk = host_w.data() + (size_t) r * rb + (size_t) b * tr->type_size;
            for (size_t j = 2; j < tr->type_size; ++j) blk[j] = (uint8_t) rng();   // scales, qh, qs
            const char* dmode = std::getenv("MV_SCALES");     // "" both, "nodmin": dmin 0, "d0min": d 1 and the codes zero
            const bool nodmin = dmode != nullptr && std::strcmp(dmode, "nodmin") == 0;
            const uint16_t d = f2h(0.0015f + 0.0005f * (float) (rng() % 4));
            const uint16_t dm = nodmin ? f2h(0.0f) : f2h(0.0010f + 0.0005f * (float) (rng() % 4));
            std::memcpy(blk, &d, 2);
            std::memcpy(blk + 2, &dm, 2);
        }

    // the host's float matrix, from ggml's decode
    std::vector<float> w((size_t) n * k);
    for (int r = 0; r < n; ++r) tr->to_float(host_w.data() + (size_t) r * rb, w.data() + (size_t) r * k, k);

    // the activation and its q8 rounding (llama.cpp's: max|x| / 127 a 32-value block, round to nearest even)
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) k);
    for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 7 == 0));
    std::vector<int8_t> xq((size_t) k);
    std::vector<float> xd((size_t) k / 32);
    for (int b = 0; b < k / 32; ++b) {
        float am = 0.0f;
        for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[(size_t) b * 32 + i]));
        const float sc = am / 127.0f;
        xd[(size_t) b] = sc;
        for (int i = 0; i < 32; ++i) {
            const float v = x[(size_t) b * 32 + i];
            long q = am > 0.0f ? std::lrint(v / sc) : 0;
            q = std::max(-127L, std::min(127L, q));
            xq[(size_t) b * 32 + i] = (int8_t) q;
        }
    }

    // the device
    uint8_t* dw = nullptr;
    int8_t* dxq = nullptr;
    float* dxd = nullptr;
    float* dy = nullptr;
    if (hipMalloc(&dw, (size_t) n * rb) != 0 || hipMalloc(&dxq, (size_t) k) != 0 ||
        hipMalloc(&dxd, (size_t) k / 32 * 4) != 0 || hipMalloc(&dy, (size_t) n * 4) != 0) {
        std::printf("allocation failed\n");
        return 2;
    }
    hipMemcpy(dw, host_w.data(), (size_t) n * rb, hipMemcpyHostToDevice);
    hipMemcpy(dxq, xq.data(), (size_t) k, hipMemcpyHostToDevice);
    hipMemcpy(dxd, xd.data(), (size_t) k / 32 * 4, hipMemcpyHostToDevice);

    iq::MvJob job{};
    job.w = dw;
    job.ld = (size_t) rb;
    job.type = (int) type;
    job.k = k;
    job.n = n;
    job.ncol = 1;
    job.xq = dxq;
    job.xd = dxd;
    job.y = dy;
    job.ycs = (size_t) 1;
    if (!iq::mv(&job, 1, nullptr)) { std::printf("iq::mv refused the job\n"); return 2; }
    if (hipDeviceSynchronize() != 0) { std::printf("device failed\n"); return 2; }
    std::vector<float> y((size_t) n);
    hipMemcpy(y.data(), dy, (size_t) n * 4, hipMemcpyDeviceToHost);

    // the reference: the same int8 activations against ggml's decoded weights, in double
    double worst = 0.0, worst_rel = 0.0;
    int bad = 0;
    for (int r = 0; r < n; ++r) {
        double acc = 0.0, mag = 0.0;
        for (int j = 0; j < k; ++j) {
            const double q = (double) xq[(size_t) j] * (double) xd[(size_t) j / 32];
            acc += (double) w[(size_t) r * k + j] * q;
            mag += std::fabs((double) w[(size_t) r * k + j] * q);
        }
        const double d = std::fabs((double) y[(size_t) r] - acc);
        worst = std::max(worst, d);
        const double rel = mag > 0 ? d / mag : 0.0;
        worst_rel = std::max(worst_rel, rel);
        if (rel > 2e-3) ++bad;
    }
    // a second reference with the block's min term left out (weight = d s q, not d s q - dmin m): if the device is
    // closer to THAT, the kernel is not applying the min at all - which is what Q4_K / Q5_K looked like
    double worst_nomin = 0.0;
    if (tr->blck_size == 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
        auto gsm = [&](const uint8_t* q, int j, int& d, int& m) {   // ggml's get_scale_min_k4
            if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
            else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
        };
        for (int r = 0; r < n; ++r) {
            double add = 0.0;   // the min term the exact reference already subtracted
            for (int b = 0; b < k / 256; ++b) {
                const uint8_t* blk = host_w.data() + (size_t) r * rb + (size_t) b * tr->type_size;
                uint16_t dm; std::memcpy(&dm, blk + 2, 2);
                const float dmin = ggml_fp16_to_fp32(dm);
                for (int sb = 0; sb < 8; ++sb) {
                    int sd, sm; gsm(blk + 4, sb, sd, sm);
                    double xs = 0.0;
                    for (int j = 0; j < 32; ++j) {
                        const int v = (b * 256 + sb * 32 + j);
                        xs += (double) xq[(size_t) v] * (double) xd[(size_t) v / 32];
                    }
                    add += (double) dmin * (double) sm * xs;
                }
            }
            // the device against (exact + add) = the no-min reference
            const double ref_nomin = 0.0; (void) ref_nomin;
            (void) add;
        }
        // (recomputed per row below, with the device's own row)
        for (int r = 0; r < n; ++r) {
            double add = 0.0, acc = 0.0;
            for (int b = 0; b < k / 256; ++b) {
                const uint8_t* blk = host_w.data() + (size_t) r * rb + (size_t) b * tr->type_size;
                uint16_t dm; std::memcpy(&dm, blk + 2, 2);
                const float dmin = ggml_fp16_to_fp32(dm);
                for (int sb = 0; sb < 8; ++sb) {
                    int sd, sm; gsm(blk + 4, sb, sd, sm);
                    double xs = 0.0;
                    for (int j = 0; j < 32; ++j) {
                        const int v = b * 256 + sb * 32 + j;
                        xs += (double) xq[(size_t) v] * (double) xd[(size_t) v / 32];
                    }
                    add += (double) dmin * (double) sm * xs;
                }
            }
            for (int j = 0; j < k; ++j)
                acc += (double) w[(size_t) r * k + j] * (double) xq[(size_t) j] * (double) xd[(size_t) j / 32];
            worst_nomin = std::max(worst_nomin, std::fabs((double) y[(size_t) r] - (acc + add)));
        }
    }
    std::printf("%s dense mv: k %d n %d | worst |dev-ref| %.3e | worst relative %.3e | rows off %d of %d\n",
                tr->type_name, k, n, worst, worst_rel, bad, n);
    if (tr->blck_size == 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K))
        std::printf("   against the same dot WITHOUT the block's min term: worst %.3e  (small here = the min is missing)\n",
                    worst_nomin);
    hipFree(dw); hipFree(dxq); hipFree(dxd); hipFree(dy);
    if (bad > 0) { std::printf("FAIL\n"); return 1; }
    std::printf("ok\n");
    return 0;
}
