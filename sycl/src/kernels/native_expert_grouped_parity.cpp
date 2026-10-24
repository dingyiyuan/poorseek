// sycl/src/kernels/native_expert_grouped_parity.cpp - numeric parity + timing for the SYCL `native_expert_grouped`
// (sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp) against a host implementation of the CUDA kernel's own
// arithmetic.
//
// WHAT THE REFERENCE IS.  `ref_*` below is a second, independent transcription of
// src/kernels/cuda/iq_kernels.cu's `native_expert_grouped` + `row_dot` + the three forks' `vec_dot_*_q8_1` +
// `swiglu_q8_1_entries_kernel` + `q8_1_store`.  It is deliberately written in a different shape from the kernel: the
// weight unpacking is scalar and indexed by K element (not by dp4a lane), the 32-lane warp is emulated as an array,
// and the reduction is the explicit XOR butterfly.  The parts that must be identical - which integer value each
// weight takes, which activation it multiplies, how the integer sums are grouped into FP32 scale products, and the
// order of every FP32 addition - are copied from the CUDA source, not approximated.
//
// WHAT "EQUAL" MEANS.  The comparison is bitwise on the float/int8 payloads (no tolerance): gate/up, the q8_1 blocks
// of the SwiGLU output, and the final out rows.  Where bitwise equality does not hold the harness reports the actual
// error (max ULP, max relative) and the number of differing elements, and separately reports whether the CUDA
// compiler's permitted FMA contraction (`-fmad=true`, nvcc's default) explains it: the reference is evaluated both
// contracted and not, and the run says which one the device matched.  The SwiGLU's `__expf` is the one operation the
// CUDA binary's bits cannot be reproduced from source (it is a hardware approximation); the harness measures what
// survives the q8_1 quantization downstream.
//
// usage:  native_expert_grouped_parity [--validate] [--bench] [--both] [--threads N]
// exit:   0 = every case bitwise equal (or contract-consistent), 1 = a mismatch, 2 = infrastructure error
#include <sycl/sycl.hpp>

#include "strata/sycl_queue.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels;

// ---------------------------------------------------------------- fp16 on the host (exact, RNE like __float2half)
static float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0x1f) {
        const uint32_t bits = (s << 31) | 0x7f800000u | (m << 13);
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }
    double v;
    if (e == 0) v = std::ldexp((double) m, -24);
    else v = std::ldexp((double) (m | 0x400), (int) e - 25);
    return (float) (s ? -v : v);
}

static uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 31) & 1, e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 0xFF) {   // inf / NaN (keep it a NaN)
        const uint16_t man = m ? (uint16_t) ((m >> 13) | 1) : 0;
        return (uint16_t) ((s << 15) | 0x7C00 | man);
    }
    const int ex = (int) e - 127 + 15;
    if (e == 0) return (uint16_t) (s << 15);   // float subnormal: far below half's smallest subnormal
    if (ex >= 0x1F) return (uint16_t) ((s << 15) | 0x7C00);
    const uint32_t mant24 = m | 0x800000;
    if (ex <= 0) {   // half subnormal (or the round-up into the smallest normal)
        const int k = 126 - (int) e;
        if (k > 40) return (uint16_t) (s << 15);
        const uint64_t base = (uint64_t) mant24 >> k;
        const uint64_t rem = (uint64_t) mant24 & ((((uint64_t) 1) << k) - 1);
        const uint64_t half = ((uint64_t) 1) << (k - 1);
        uint64_t q = base + ((rem > half || (rem == half && (base & 1))) ? 1 : 0);
        return (uint16_t) ((s << 15) | (uint16_t) q);
    }
    uint16_t b = (uint16_t) ((ex << 10) | (m >> 13));
    const uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (b & 1))) ++b;   // the carry into the exponent is the correct rounding
    return (uint16_t) ((s << 15) | b);
}

// ---------------------------------------------------------------- host block layouts (mirror of the device's)
struct H_Q8 {
    uint16_t d, s;
    int8_t qs[32];
};
static_assert(sizeof(H_Q8) == 36);
struct H_Q2_0 {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(H_Q2_0) == 18);
struct H_Q3_K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};
static_assert(sizeof(H_Q3_K) == 110);
struct H_PTQ {
    uint8_t qs[24];
    uint8_t qh[2];
    uint16_t d;
};
static_assert(sizeof(H_PTQ) == 28);

// ---------------------------------------------------------------- the contract, as read from the source
static size_t ref_row_bytes(int ty, int64_t n) {
    switch (ty) {
        case 42: return (size_t) (n / 64) * 18;
        case 11: return (size_t) (n / 256) * 110;
        case 143: return (size_t) (n / 128) * 28;
        default: return 0;
    }
}
static int ref_qk(int ty) { return ty == 42 ? 64 : ty == 11 ? 256 : ty == 143 ? 128 : 0; }
static bool ref_supported(int gu, int d, int64_t n_embd, int64_t n_ff) {
    const int qg = ref_qk(gu), qd = ref_qk(d);
    return qg > 0 && qd > 0 && n_embd % qg == 0 && n_ff % qd == 0 && n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}
static NativeExpertLayout ref_layout(int gu, int d, int64_t n_embd, int64_t n_ff, float lim) {
    NativeExpertLayout L;
    L.gu_type = gu;
    L.d_type = d;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = ref_row_bytes(gu, n_embd);
    L.d_row = ref_row_bytes(d, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    L.swiglu_limit = lim;
    return L;
}
static size_t ref_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) +
           (((size_t) cap * (size_t) (n_ff / 32) * sizeof(H_Q8) + 255) & ~(size_t) 255);
}

// ---------------------------------------------------------------- the reference arithmetic
static int ref_dp4a_sum(const int8_t* w, const int8_t* a, int n) {   // the integers a dp4a chain sums, any order
    int s = 0;
    for (int i = 0; i < n; ++i) s += (int) w[i] * (int) a[i];
    return s;
}

static int ref_trit(uint8_t b, int t) {
    uint32_t v = b;
    for (int i = 0; i < t; ++i) v = (v * 3u) & 0xFFu;
    return (int) (((v * 3u) >> 8) & 0xFFu);
}

struct Acc {
    float v;
};

// q2_0: 64 values per block, 2 calls per block, 8 two-bit fields of value field-1 per int16.
static void ref_acc_q2_0(const uint8_t* row, const H_Q8* q8, int kbx, int iqs, Acc& acc, bool contract) {
    const H_Q2_0* b = (const H_Q2_0*) row + kbx;
    uint64_t bits = 0;
    std::memcpy(&bits, b->qs + iqs * 8, 8);
    int8_t w[32];
    for (int m = 0; m < 32; ++m) w[m] = (int8_t) (((bits >> (2 * m)) & 3) - 1);
    const int sumi = ref_dp4a_sum(w, q8[iqs].qs, 32);
    const float d2 = h2f(b->d), d8 = h2f(q8[iqs].d);
    const float dd = d2 * d8;
    acc.v = contract ? std::fma(dd, (float) sumi, acc.v) : acc.v + dd * (float) sumi;
}

// q3_K: 256 values per block, 16 calls per block, 16 values per call, the inverted high-bit mask subtracts 4.
static void ref_acc_q3_K(const uint8_t* row, const H_Q8* q8, int kbx, int iqs, Acc& acc, bool contract) {
    const H_Q3_K* b = (const H_Q3_K*) row + kbx;
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    // get_int_b2(x, i) is the 32-bit word at 16-bit index 2*i, i.e. at byte 4*i (iq_kernels.cu:35); the shift below is
    // the CUDA source's `int >> int`, an arithmetic shift.
    uint32_t vl, vh_raw;
    std::memcpy(&vl, b->qs + 4 * iqs, 4);
    std::memcpy(&vh_raw, b->hmask + 4 * (iqs % 8), 4);
    const int vh = (int) (~vh_raw) >> bq8_offset;
    float sumf = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int sc_low = (b->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
        const int sc_high = ((b->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        int8_t w[4];
        for (int m = 0; m < 4; ++m) {   // byte m of this call's 4-byte group
            const int lo = (int) ((vl >> (2 * i + 8 * m)) & 3);
            const int hi = (int) ((vh >> (i + 8 * m)) & 1) << 2;
            int v = lo - hi;   // __vsubss4 saturates int8; this range [-4,3] never does
            v = v > 127 ? 127 : v < -128 ? -128 : v;
            w[m] = (int8_t) v;
        }
        const int8_t* a = q8[bq8_offset + i].qs + 4 * (iqs % 8);
        const int prod = ref_dp4a_sum(w, a, 4) * sc;   // the int product first (the CUDA source multiplies ints)
        const float d8 = h2f(q8[bq8_offset + i].d);
        sumf = contract ? std::fma(d8, (float) prod, sumf) : sumf + d8 * (float) prod;
    }
    const float d3 = h2f(b->d);
    acc.v = contract ? std::fma(d3, sumf, acc.v) : acc.v + d3 * sumf;
}

// PTQ1_0: 128 ternary weights per block, one call per block, the -1 folded in as sum(t*a) - sum(a) per sub-block.
static void ref_acc_ptq1_0(const uint8_t* row, const H_Q8* q8, int kbx, int iqs, Acc& acc, bool contract) {
    const H_PTQ* b = (const H_PTQ*) row + kbx;
    int sumi[4] = {0, 0, 0, 0}, sumu[4] = {0, 0, 0, 0};
    auto add = [&](int e, int trit) {
        const int a = (int) q8[iqs + (e >> 5)].qs[e & 31];
        sumi[e >> 5] += trit * a;
        sumu[e >> 5] += a;
    };
    for (int m = 0; m < 16; ++m)          // qs[0..15]: element m + 16 t
        for (int t = 0; t < 5; ++t) add(m + 16 * t, ref_trit(b->qs[m], t));
    for (int m = 0; m < 8; ++m)           // qs[16..23]: element 80 + m + 8 t
        for (int t = 0; t < 5; ++t) add(80 + m + 8 * t, ref_trit(b->qs[16 + m], t));
    for (int h = 0; h < 2; ++h)           // qh[0..1]: element 120 + h + 2 t
        for (int t = 0; t < 4; ++t) add(120 + h + 2 * t, ref_trit(b->qh[h], t));
    float a4 = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const float d8 = h2f(q8[iqs + k].d);
        a4 = contract ? std::fma(d8, (float) (sumi[k] - sumu[k]), a4) : a4 + d8 * (float) (sumi[k] - sumu[k]);
    }
    const float dw = h2f(b->d);
    acc.v = contract ? std::fma(dw, a4, acc.v) : acc.v + dw * a4;
}

static float ref_row_dot(int ty, const uint8_t* row, const uint8_t* xrow, int nb, bool contract) {
    const int xb = 32;   // q8_1 blocks per 32 values
    (void) xb;
    const int ipb = ty == 42 ? 2 : ty == 11 ? 16 : 1;
    const int step = ty == 42 ? 1 : ty == 11 ? 1 : 0;
    const int qk = ref_qk(ty);
    const H_Q8* q8 = (const H_Q8*) xrow;
    float s[32];
    for (int lane = 0; lane < 32; ++lane) {
        Acc acc{0.0f};
        for (int k = lane; k < nb * ipb; k += 32) {
            const int kbx = k / ipb, iqs = step * (k % ipb);
            const H_Q8* xa = q8 + kbx * (qk / 32);
            if (ty == 42) ref_acc_q2_0(row, xa, kbx, iqs, acc, contract);
            else if (ty == 11) ref_acc_q3_K(row, xa, kbx, iqs, acc, contract);
            else ref_acc_ptq1_0(row, xa, kbx, iqs, acc, contract);
        }
        s[lane] = acc.v;
    }
    for (int o = 16; o > 0; o >>= 1) {   // the XOR butterfly, snapshot per step
        float t[32];
        for (int l = 0; l < 32; ++l) t[l] = s[l] + s[l ^ o];
        std::memcpy(s, t, sizeof s);
    }
    return s[0];
}

static float ref_q8_1_finite(float v) { return std::fabs(v) > 65504.0f ? std::copysign(65504.0f, v) : v; }

static void ref_quantize_block(const float* x, H_Q8& y) {   // 32 values
    float a[32], s[32];
    for (int l = 0; l < 32; ++l) { a[l] = std::fabs(x[l]); s[l] = x[l]; }
    for (int o = 16; o > 0; o >>= 1) {
        float an[32], sn[32];
        for (int l = 0; l < 32; ++l) { an[l] = std::fmax(a[l], a[l ^ o]); sn[l] = s[l] + s[l ^ o]; }
        std::memcpy(a, an, sizeof a);
        std::memcpy(s, sn, sizeof s);
    }
    const float amax = a[0];
    const float d = ref_q8_1_finite(amax / 127.0f);
    for (int l = 0; l < 32; ++l) {
        int8_t q = 0;
        if (amax != 0.0f) {
            const float v = std::round(x[l] / d);
            q = (int8_t) (v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
        }
        y.qs[l] = q;
    }
    y.d = f2h(ref_q8_1_finite(d));
    y.s = f2h(ref_q8_1_finite(s[0]));
}

// The whole call, as the CUDA source composes it.
struct RefArgs {
    const NativeExpertLayout* L = nullptr;
    const uint8_t* const* blob = nullptr;
    const int32_t* grp_start = nullptr;
    int32_t ng = 0;
    const int32_t* ent_dst = nullptr;
    const int32_t* ent_tok = nullptr;
    int64_t cap_entries = 0;
    const uint8_t* xq = nullptr;   // H_Q8 rows
    float* gate = nullptr;
    float* up = nullptr;
    float* h = nullptr;
    H_Q8* hq = nullptr;
    float* out = nullptr;
    bool v1 = false;
    bool contract = false;
    int threads = 1;
};

static void ref_swiglu_q8_1(const RefArgs& A) {
    const auto& L = *A.L;
    const long long lo = A.v1 ? 0 : (long long) A.grp_start[0] * L.n_ff;
    const long long hi = A.v1 ? (long long) A.cap_entries * L.n_ff : (long long) A.grp_start[A.ng] * L.n_ff;
    const float lim = L.swiglu_limit;
    if (A.threads <= 1) {
        for (long long ib = lo / 32; ib < hi / 32; ++ib) {
            float hb[32];
            for (int l = 0; l < 32; ++l) {
                const long long i = ib * 32 + l;
                const float g = std::fmin(A.gate[i], lim);
                const float u = std::fmin(std::fmax(A.up[i], -lim), lim);
                hb[l] = (g / (1.0f + std::exp(-g))) * u;
                if (A.v1) A.h[i] = hb[l];
            }
            ref_quantize_block(hb, A.hq[ib]);
        }
        return;
    }
    const long long nb = hi / 32;
    const int nt = A.threads;
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t) {
        th.emplace_back([&, t] {
            for (long long ib = lo / 32 + t; ib < nb; ib += nt) {
                float hb[32];
                for (int l = 0; l < 32; ++l) {
                    const long long i = ib * 32 + l;
                    const float g = std::fmin(A.gate[i], lim);
                    const float u = std::fmin(std::fmax(A.up[i], -lim), lim);
                    hb[l] = (g / (1.0f + std::exp(-g))) * u;
                    if (A.v1) A.h[i] = hb[l];
                }
                ref_quantize_block(hb, A.hq[ib]);
            }
        });
    }
    for (auto& x : th) x.join();
}

static void ref_grouped(const RefArgs& A) {
    const auto& L = *A.L;
    const int gu = L.gu_type, dt = L.d_type;
    const int nb_gu = (int) (L.n_embd / ref_qk(gu)), xb = (int) (L.n_embd / 32);
    const int nb_d = (int) (L.n_ff / ref_qk(dt)), hb = (int) (L.n_ff / 32);

    // gate/up: rows 0..2*n_ff, row r < n_ff is gate[r], the rest up[r - n_ff]
    auto gu_rows = [&](long long r0, long long r1) {
        for (long long row = r0; row < r1; ++row) {
            const bool is_up = row >= L.n_ff;
            const long long r = is_up ? row - L.n_ff : row;
            const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            for (int32_t g = 0; g < A.ng; ++g) {
                const uint8_t* wr = A.blob[g] + off;
                const int e0 = A.grp_start[g], e1 = A.grp_start[g + 1];
                for (int e = e0; e < e1; ++e) {
                    const float s = ref_row_dot(gu, wr, A.xq + (size_t) A.ent_tok[e] * xb * 36, nb_gu, A.contract);
                    (is_up ? A.up : A.gate)[(size_t) e * L.n_ff + r] = s;
                }
            }
        }
    };
    // down: rows 0..n_embd, out row ent_dst[e]
    auto down_rows = [&](long long r0, long long r1) {
        for (long long r = r0; r < r1; ++r) {
            const size_t off = L.down_off + (size_t) r * L.d_row;
            for (int32_t g = 0; g < A.ng; ++g) {
                const uint8_t* wr = A.blob[g] + off;
                const int e0 = A.grp_start[g], e1 = A.grp_start[g + 1];
                for (int e = e0; e < e1; ++e) {
                    const float s = ref_row_dot(dt, wr, (const uint8_t*) (A.hq + (size_t) e * hb), nb_d, A.contract);
                    A.out[(size_t) A.ent_dst[e] * L.n_embd + r] = s;
                }
            }
        }
    };
    auto rows = [&](auto&& body, long long n) {
        if (A.threads <= 1) { body(0, n); return; }
        std::vector<std::thread> th;
        const int nt = A.threads;
        const long long chunk = (n + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const long long r0 = t * chunk, r1 = std::min(n, r0 + chunk);
            if (r0 < r1) th.emplace_back([&body, r0, r1] { body(r0, r1); });
        }
        for (auto& x : th) x.join();
    };
    rows(gu_rows, 2 * L.n_ff);
    ref_swiglu_q8_1(A);
    rows(down_rows, L.n_embd);
}

// Dequantized ground truth in double: the weights as d_w * q, the activations as d_a * q, all products in double.
// This is neither the kernel's nor the reference's arithmetic - it is what the format means - so it arbitrates.
static double truth_dot(int ty, const uint8_t* wr, const H_Q8* xr, int nb) {
    double s = 0.0;
    for (int b = 0; b < nb; ++b) {
        if (ty == 42) {
            const H_Q2_0* q = (const H_Q2_0*) wr + b;
            uint8_t qb[16];
            std::memcpy(qb, q->qs, 16);
            for (int e = 0; e < 64; ++e) {
                const int f = (int) ((qb[e / 4] >> (2 * (e % 4))) & 3);
                const H_Q8& a = xr[b * 2 + e / 32];
                s += (double) h2f(q->d) * (double) (f - 1) * (double) h2f(a.d) * (double) a.qs[e % 32];
            }
        } else if (ty == 11) {
            const H_Q3_K* q = (const H_Q3_K*) wr + b;
            for (int iqs = 0; iqs < 16; ++iqs) {
                uint32_t vl32, vh_raw;
                std::memcpy(&vl32, q->qs + 4 * iqs, 4);
                std::memcpy(&vh_raw, q->hmask + 4 * (iqs % 8), 4);
                const int vl = (int) vl32, vh = (int) (~vh_raw) >> (4 * (iqs / 8));
                const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
                for (int i = 0; i < 4; ++i) {
                    const int isc = scale_offset + 2 * i;
                    const int sc_low = (q->scales[isc % 8] >> (4 * (isc / 8))) & 0xF;
                    const int sc_high = ((q->scales[8 + isc % 4] >> (2 * (isc / 4))) & 3) << 4;
                    const int sc = (sc_low | sc_high) - 32;
                    const int q8b = b * 8 + 4 * (iqs / 8) + i;   // this call's i-th q8_1 block
                    for (int m = 0; m < 4; ++m) {
                        const int f = (vl >> (2 * i + 8 * m)) & 3;
                        const int hb = (vh >> (i + 8 * m)) & 1;   // the COMPLEMENT's bit: 1 means no -4
                        const int v = f - 4 * hb;
                        const H_Q8& a = xr[q8b];
                        s += (double) h2f(q->d) * (double) sc * (double) v * (double) h2f(a.d) *
                             (double) a.qs[4 * (iqs % 8) + m];
                    }
                }
            }
        } else {
            const H_PTQ* q = (const H_PTQ*) wr + b;
            for (int m = 0; m < 16; ++m)
                for (int t = 0; t < 5; ++t) {
                    const int e = m + 16 * t;
                    const H_Q8& a = xr[b * 4 + e / 32];
                    s += (double) h2f(q->d) * (double) (ref_trit(q->qs[m], t) - 1) * (double) h2f(a.d) * (double) a.qs[e % 32];
                }
            for (int m = 0; m < 8; ++m)
                for (int t = 0; t < 5; ++t) {
                    const int e = 80 + m + 8 * t;
                    const H_Q8& a = xr[b * 4 + e / 32];
                    s += (double) h2f(q->d) * (double) (ref_trit(q->qs[16 + m], t) - 1) * (double) h2f(a.d) * (double) a.qs[e % 32];
                }
            for (int h = 0; h < 2; ++h)
                for (int t = 0; t < 4; ++t) {
                    const int e = 120 + h + 2 * t;
                    const H_Q8& a = xr[b * 4 + e / 32];
                    s += (double) h2f(q->d) * (double) (ref_trit(q->qh[h], t) - 1) * (double) h2f(a.d) * (double) a.qs[e % 32];
                }
        }
    }
    return s;
}

// ---------------------------------------------------------------- test scaffolding
static uint64_t g_rng = 0x243F6A8885A308D3ull;
static uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t) ((g_rng >> 16) & 0xFFFFu);
}
static float rnd_f(float lo, float hi) { return lo + (hi - lo) * (rnd() / 65536.0f); }

// A blob is [gate rows | up rows | down rows] of raw blocks (the contract).  The block bodies come from the RNG of the
// case; the fp16 scale of every block is then patched by `patch_scales` to the case's adversarial distribution.
enum FuzzStyle { kFuzzNormal, kFuzzZero, kFuzzExtremeScales, kFuzzSubnormalScales, kFuzzMaxScales };
static void fill_blob(uint8_t* blob, const NativeExpertLayout& L, FuzzStyle style) {
    for (size_t i = 0; i < L.bytes; ++i) blob[i] = (uint8_t) rnd();
    if (style == kFuzzZero) std::memset(blob, 0, L.bytes);
}

static void patch_scales(uint8_t* blob, const NativeExpertLayout& L, FuzzStyle style) {
    auto pick = [&]() -> uint16_t {
        switch (style) {
            case kFuzzZero: return f2h(0.0f);
            case kFuzzExtremeScales: return (rnd() & 1) ? f2h(65504.0f) : f2h(6.103515625e-05f);   // max / smallest normal
            case kFuzzSubnormalScales: return f2h(1.0e-07f);                                      // fp16 subnormal
            case kFuzzMaxScales: return f2h(65504.0f);
            default: return f2h(rnd_f(0.02f, 2.0f));
        }
    };
    auto patch = [&](uint8_t* base, size_t rows, size_t row_bytes, int ty) {
        const size_t per = (size_t) (ref_qk(ty) == 64 ? 18 : ref_qk(ty) == 256 ? 110 : 28);
        const size_t nblk = row_bytes / per;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t b = 0; b < nblk; ++b) {
                uint8_t* blk = base + r * row_bytes + b * per;
                // the fp16 scale sits first in q2_0 and last in q3_K / PTQ1_0
                const size_t doff = ty == 42 ? 0 : per - 2;
                uint16_t* d = (uint16_t*) (blk + doff);
                *d = pick();
            }
        }
    };
    patch(blob, (size_t) L.n_ff, L.gu_row, L.gu_type);             // gate
    patch(blob + L.up_off, (size_t) L.n_ff, L.gu_row, L.gu_type);  // up
    patch(blob + L.down_off, (size_t) L.n_embd, L.d_row, L.d_type);  // down
}

static std::string ty_name(int ty) {
    switch (ty) {
        case 42: return "Q2_0";
        case 11: return "Q3_K";
        case 143: return "PTQ1";
        default: return "?" + std::to_string(ty);
    }
}

// bitwise comparison of floats, with the error report
struct Cmp {
    long long n = 0, diff = 0;
    double max_rel = 0.0;
    long long max_ulp = 0;
    static int64_t ulp(float a, float b) {
        int32_t ia, ib;
        std::memcpy(&ia, &a, 4);
        std::memcpy(&ib, &b, 4);
        if (ia < 0) ia = (int32_t) (0x80000000u - (uint32_t) ia);
        if (ib < 0) ib = (int32_t) (0x80000000u - (uint32_t) ib);
        return (int64_t) ia - (int64_t) ib >= 0 ? (int64_t) ia - (int64_t) ib : (int64_t) ib - (int64_t) ia;
    }
};
static void cmp_floats(const float* a, const float* b, size_t n, Cmp& c) {
    c.n = (long long) n;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], 4) == 0) continue;
        ++c.diff;
        const int64_t u = Cmp::ulp(a[i], b[i]);
        c.max_ulp = std::max<long long>(c.max_ulp, (long long) u);
        const double d = std::fabs((double) a[i] - (double) b[i]);
        const double rel = d / std::max(1e-30, (double) std::fabs(a[i]));
        c.max_rel = std::max<double>(c.max_rel, rel);
    }
}
static void cmp_q8(const H_Q8* a, const H_Q8* b, size_t n, Cmp& c, long long* diff_qs) {
    c.n = (long long) n;
    for (size_t i = 0; i < n; ++i) {
        bool bad = a[i].d != b[i].d || a[i].s != b[i].s;
        if (std::memcmp(a[i].qs, b[i].qs, 32) != 0) { bad = true; ++*diff_qs; }
        if (!bad) continue;
        ++c.diff;
        const long long u = std::max<long long>((long long) Cmp::ulp(h2f(a[i].d), h2f(b[i].d)),
                                                (long long) Cmp::ulp(h2f(a[i].s), h2f(b[i].s)));
        c.max_ulp = std::max<long long>(c.max_ulp, u);
    }
}

struct Case {
    std::string name;
    int gu = 42, dt = 42;
    int64_t n_embd = 512, n_ff = 192;
    int ng = 4, cap_groups = 6;
    int64_t cap_entries = 12;
    FuzzStyle style = kFuzzNormal;
    float lim = 10.0f;
    bool v1 = false;
    int64_t grid_groups = 0;   // 0 = cap_groups; a smaller value = the verify window's strided launch
    int64_t act_style = 0;     // 0 normal, 1 huge (the #606 finite clamp), 2 zero block, 3 tiny, 4 mixed
};

// Probe: what the device actually gives this kernel for sub-group size and local id mapping.
static void probe(sycl::queue& q) {
    auto* buf = sycl::malloc_shared<int>(8, q);
    for (int i = 0; i < 8; ++i) buf[i] = -1;
    q.parallel_for<class probe_k>(sycl::nd_range<2>({256, 1}, {256, 1}),
                                     [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                                         const auto sg = it.get_sub_group();
                                         if (it.get_local_id(0) == 0) {
                                             buf[0] = (int) sg.get_local_range()[0];
                                             buf[1] = (int) it.get_local_range(0);
                                             buf[2] = (int) sg.get_group_id();
                                         }
                                         if (it.get_local_id(0) == 37) {
                                             buf[3] = (int) sg.get_local_id();
                                             buf[4] = (int) sg.get_group_id();
                                         }
                                     });
    q.wait();
    std::printf("probe: sub_group size=%d local range=%d sg groups=%d ; item37: sg local id=%d sg group=%d\n", buf[0],
                buf[1], buf[2], buf[3], buf[4]);
    sycl::free(buf, q);
}


// ---------------------------------------------------------------- one constructed block, three ways
// weights q2_0: d = 1, every field = 2 (value +1); activations: d8 = 1, every q = 1.  A row of n values therefore
// dots to exactly n, whatever the layout is - which makes any indexing error visible as a wrong multiple.
static void selftest(sycl::queue& q) {
    const int64_t n_embd = 512, n_ff = 64;
    const NativeExpertLayout L = ref_layout(42, 42, n_embd, n_ff, 10.0f);
    std::printf("selftest: n_embd %lld n_ff %lld gu_row %zu up_off %zu down_off %zu bytes %zu\n", (long long) n_embd,
                (long long) n_ff, L.gu_row, L.up_off, L.down_off, L.bytes);
    std::vector<uint8_t> blob(L.bytes, 0);
    for (size_t b = 0; b < L.bytes / 18; ++b) {
        H_Q2_0* qq = (H_Q2_0*) blob.data() + b;
        qq->d = f2h(1.0f);
        std::memset(qq->qs, 0xAA, 16);   // every 2-bit field = 2 -> value = 2 - 1 = +1
    }
    std::vector<H_Q8> xq((size_t) 8 * n_embd / 32);
    for (auto& x : xq) {
        x.d = f2h(1.0f);
        x.s = f2h(0.0f);
        std::memset(x.qs, 1, 32);
    }
    // host: truth and reference for gate row 0 (n_embd values) and down row 0 (n_ff values)
    const double tg = truth_dot(42, blob.data(), xq.data(), (int) (n_embd / 64));
    const double rgl = ref_row_dot(42, blob.data(), (const uint8_t*) xq.data(), (int) (n_embd / 64), false);
    const double rgh = ref_row_dot(42, blob.data(), (const uint8_t*) xq.data(), (int) (n_embd / 64), true);
    std::printf("selftest gate row0: truth(should be %lld) %.10g | ref %.10g / contracted %.10g\n", (long long) n_embd, tg,
                rgl, rgh);

    // device: one group, one entry, through the public entry point
    auto* d_blob = (uint8_t*) sycl::malloc_shared(blob.size(), q);
    auto* d_ptr = (unsigned long long*) sycl::malloc_shared(1 * 8, q);
    auto* d_start = (int32_t*) sycl::malloc_shared(2 * 4, q);
    auto* d_dst = (int32_t*) sycl::malloc_shared(1 * 4, q);
    auto* d_tok = (int32_t*) sycl::malloc_shared(1 * 4, q);
    auto* d_ng = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_xq = (uint8_t*) sycl::malloc_shared(xq.size() * 36, q);
    const size_t sbytes = ref_scratch_bytes(1, n_ff);
    auto* d_scr = (uint8_t*) sycl::malloc_shared(sbytes, q);
    auto* d_out = (float*) sycl::malloc_shared((size_t) n_embd * 4, q);
    std::memcpy(d_blob, blob.data(), blob.size());
    d_ptr[0] = (unsigned long long) (uintptr_t) d_blob;
    d_start[0] = 0;
    d_start[1] = 1;
    d_dst[0] = 0;
    d_tok[0] = 0;
    *d_ng = 1;
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq, d_scr, d_out, &q, 0);
    q.wait();
    const float* gate = (const float*) d_scr;
    const float* up = (const float*) (d_scr + (((size_t) 1 * n_ff * 4 + 255) & ~(size_t) 255));
    std::printf("selftest gate row0 dev: %.10g (row1 %.10g) | up row0 dev: %.10g | down row0 dev: %.10g\n", gate[0],
                gate[1], up[0], d_out[0]);
    sycl::free(d_blob, q);
    sycl::free(d_ptr, q);
    sycl::free(d_start, q);
    sycl::free(d_dst, q);
    sycl::free(d_tok, q);
    sycl::free(d_ng, q);
    sycl::free(d_xq, q);
    sycl::free(d_scr, q);
    sycl::free(d_out, q);
}


// ---------------------------------------------------------------- asymmetric single-block probe
// one 64-value block per row (n_embd = 64), each row r's fields = ((e + r) % 3) - 1, activations q = lane + 1 with
// d8 = 1: the reference, the truth and the device must agree *per row*, so a mis-paired weight/activation or a wrong
// row offset shows up as a value that belongs to another row.
static void selftest2(sycl::queue& q) {
    const int64_t n_embd = 64, n_ff = 64;
    const NativeExpertLayout L = ref_layout(42, 42, n_embd, n_ff, 10.0f);
    std::vector<uint8_t> blob(L.bytes, 0);
    auto build_row = [&](uint8_t* rp, int r) {
        H_Q2_0* b = (H_Q2_0*) rp;
        b->d = f2h(1.0f);
        std::memset(b->qs, 0, 16);
        for (int e = 0; e < 64; ++e) {
            const int f = (e + r) % 3;   // value f - 1
            b->qs[e / 4] |= (uint8_t) (f << (2 * (e % 4)));
        }
    };
    for (int r = 0; r < n_ff; ++r) build_row(blob.data() + (size_t) r * L.gu_row, r);            // gate rows
    for (int r = 0; r < n_ff; ++r) build_row(blob.data() + L.up_off + (size_t) r * L.gu_row, 100 + r);  // up rows
    for (int r = 0; r < n_embd; ++r) build_row(blob.data() + L.down_off + (size_t) r * L.d_row, 200 + r);
    std::vector<H_Q8> xq(8 * n_embd / 32);
    for (size_t i = 0; i < xq.size(); ++i) {
        xq[i].d = f2h(1.0f);
        xq[i].s = f2h(0.0f);
        for (int l = 0; l < 32; ++l) xq[i].qs[l] = (int8_t) (l + 1);
    }
    auto* d_blob = (uint8_t*) sycl::malloc_shared(blob.size(), q);
    auto* d_ptr = (unsigned long long*) sycl::malloc_shared(8, q);
    auto* d_start = (int32_t*) sycl::malloc_shared(8, q);
    auto* d_dst = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_tok = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_ng = sycl::malloc_shared<int32_t>(1, q);
    auto* d_xq = (uint8_t*) sycl::malloc_shared(xq.size() * 36, q);
    const size_t sbytes = ref_scratch_bytes(1, n_ff);
    auto* d_scr = (uint8_t*) sycl::malloc_shared(sbytes, q);
    auto* d_out = (float*) sycl::malloc_shared((size_t) n_embd * 4, q);
    std::memcpy(d_blob, blob.data(), blob.size());
    d_ptr[0] = (unsigned long long) (uintptr_t) d_blob;
    d_start[0] = 0; d_start[1] = 1;
    d_dst[0] = 0; d_tok[0] = 0;
    *d_ng = 1;
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq, d_scr, d_out, &q, 0);
    q.wait();
    const float* gate = (const float*) d_scr;
    std::printf("selftest2 (one 64-value block per row, fields ((e+r)%%3)-1, acts lane+1, d=1):\n");
    for (int r = 0; r < 8; ++r) {
        const double t = truth_dot(42, blob.data() + (size_t) r * L.gu_row, xq.data(), 1);
        const double rf = ref_row_dot(42, blob.data() + (size_t) r * L.gu_row, (const uint8_t*) xq.data(), 1, true);
        std::printf("  row %d: truth %12.6g  ref %12.6g  dev %12.6g\n", r, t, rf, gate[r]);
    }
    sycl::free(d_blob, q); sycl::free(d_ptr, q); sycl::free(d_start, q); sycl::free(d_dst, q);
    sycl::free(d_tok, q); sycl::free(d_ng, q); sycl::free(d_xq, q); sycl::free(d_scr, q); sycl::free(d_out, q);
}


// ---------------------------------------------------------------- which activation does weight e meet?
// every weight 0 except field e0 (+1), activations a[l] = l + 1, d = d8 = 1: the gate row's value is a[pi(e0)], so
// one launch per e0 prints the permutation the kernel actually applies.
static void selftest3(sycl::queue& q) {
    const int64_t n_embd = 64, n_ff = 64;
    const NativeExpertLayout L = ref_layout(42, 42, n_embd, n_ff, 10.0f);
    std::vector<uint8_t> blob(L.bytes, 0);
    for (size_t b = 0; b < L.bytes / 18; ++b) {
        H_Q2_0* qq = (H_Q2_0*) blob.data() + b;
        qq->d = f2h(1.0f);
        std::memset(qq->qs, 0x55, 16);   // all fields = 1 -> value 0
    }
    std::vector<H_Q8> xq(8 * n_embd / 32);
    for (auto& x : xq) {
        x.d = f2h(1.0f);
        x.s = f2h(0.0f);
        for (int l = 0; l < 32; ++l) x.qs[l] = (int8_t) (l + 1);
    }
    auto* d_blob = (uint8_t*) sycl::malloc_shared(blob.size(), q);
    auto* d_ptr = (unsigned long long*) sycl::malloc_shared(8, q);
    auto* d_start = (int32_t*) sycl::malloc_shared(8, q);
    auto* d_dst = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_tok = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_ng = sycl::malloc_shared<int32_t>(1, q);
    auto* d_xq = (uint8_t*) sycl::malloc_shared(xq.size() * 36, q);
    const size_t sbytes = ref_scratch_bytes(1, n_ff);
    auto* d_scr = (uint8_t*) sycl::malloc_shared(sbytes, q);
    auto* d_out = (float*) sycl::malloc_shared((size_t) n_embd * 4, q);
    std::memcpy(d_blob, blob.data(), blob.size());
    d_ptr[0] = (unsigned long long) (uintptr_t) d_blob;
    d_start[0] = 0; d_start[1] = 1;
    d_dst[0] = 0; d_tok[0] = 0;
    *d_ng = 1;
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    std::printf("selftest3: activation lane each weight meets (row 0 = gate row 0):\n");
    for (int e0 = 0; e0 < 64; ++e0) {
        H_Q2_0* b0 = (H_Q2_0*) d_blob;   // gate row 0 = block 0
        std::memset(b0->qs, 0x55, 16);
        b0->qs[e0 / 4] |= (uint8_t) (2 << (2 * (e0 % 4)));   // field e0 = 2 -> value +1
        native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq, d_scr, d_out, &q, 0);
        q.wait();
        const float* gate = (const float*) d_scr;
        std::printf("  e=%2d -> a[%d] = %g (truth %d)\n", e0, (int) (gate[0] + 0.5f) - 1, gate[0], e0 % 32);
    }
    sycl::free(d_blob, q); sycl::free(d_ptr, q); sycl::free(d_start, q); sycl::free(d_dst, q);
    sycl::free(d_tok, q); sycl::free(d_ng, q); sycl::free(d_xq, q); sycl::free(d_scr, q); sycl::free(d_out, q);
}


// ---------------------------------------------------------------- three probes on one 64-value row
static float probe_one(sycl::queue& q, const NativeExpertLayout& L, uint8_t* d_blob, uint8_t* d_xq, int32_t* d_start,
                       int32_t* d_dst, int32_t* d_tok, int32_t* d_ng, uint8_t* d_scr, float* d_out) {
    native_expert_grouped(L, (const unsigned long long*) d_start /*unused*/, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq,
                          d_scr, d_out, &q, 0);
    q.wait();
    return ((const float*) d_scr)[0];
}
static void selftest4(sycl::queue& q) {
    const int64_t n_embd = 64, n_ff = 64;
    const NativeExpertLayout L = ref_layout(42, 42, n_embd, n_ff, 10.0f);
    std::vector<uint8_t> blob(L.bytes, 0);
    for (size_t b = 0; b < L.bytes / 18; ++b) {
        H_Q2_0* qq = (H_Q2_0*) blob.data() + b;
        qq->d = f2h(1.0f);
        std::memset(qq->qs, 0x55, 16);
    }
    std::vector<H_Q8> xq(8 * n_embd / 32);
    for (auto& x : xq) {
        x.d = f2h(1.0f);
        x.s = f2h(0.0f);
        for (int l = 0; l < 32; ++l) x.qs[l] = 1;
    }
    auto* d_blob = (uint8_t*) sycl::malloc_shared(blob.size(), q);
    auto* d_ptr = (unsigned long long*) sycl::malloc_shared(8, q);
    auto* d_start = (int32_t*) sycl::malloc_shared(8, q);
    auto* d_dst = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_tok = (int32_t*) sycl::malloc_shared(4, q);
    auto* d_ng = sycl::malloc_shared<int32_t>(1, q);
    auto* d_xq = (uint8_t*) sycl::malloc_shared(xq.size() * 36, q);
    const size_t sbytes = ref_scratch_bytes(1, n_ff);
    auto* d_scr = (uint8_t*) sycl::malloc_shared(sbytes, q);
    auto* d_out = (float*) sycl::malloc_shared((size_t) n_embd * 4, q);
    std::memcpy(d_blob, blob.data(), blob.size());
    d_ptr[0] = (unsigned long long) (uintptr_t) d_blob;
    d_start[0] = 0; d_start[1] = 1;
    d_dst[0] = 0; d_tok[0] = 0;
    *d_ng = 1;
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    H_Q2_0* g0 = (H_Q2_0*) d_blob;
    auto run = [&]() {
        native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, 1, 1, d_xq, d_scr, d_out, &q, 0);
        q.wait();
        return ((const float*) d_scr)[0];
    };
    (void) probe_one;
    // A: one field = +1, activations all 1 -> expect 1.0
    int badA = 0;
    float a0 = 0, a1 = 0;
    for (int e0 = 0; e0 < 64; ++e0) {
        std::memset(g0->qs, 0x55, 16);
        g0->qs[e0 / 4] &= (uint8_t) ~(3u << (2 * (e0 % 4)));
        g0->qs[e0 / 4] |= (uint8_t) (2u << (2 * (e0 % 4)));   // value +1
        const float v = run();
        if (e0 == 0) a0 = v;
        if (e0 == 32) a1 = v;
        if (v != 1.0f) ++badA;
    }
    // B: every field +1, activations all 1 -> expect 64
    std::memset(g0->qs, 0xAA, 16);   // every field = 2 -> +1
    const float bv = run();
    // C: one field +1, activation lane 0 of both q8 blocks = 1, all others 0 -> expect 1 if the field meets lane 0
    for (auto& x : xq) memset(x.qs, 0, 32);
    for (size_t i = 0; i < xq.size(); i += 2) xq[i].qs[0] = 1;   // q8 block 0 and 2 and ... (block 0 of each pair)
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    int hit0 = 0, hit_other = 0;
    for (int e0 = 0; e0 < 64; ++e0) {
        std::memset(g0->qs, 0x55, 16);
        g0->qs[e0 / 4] &= (uint8_t) ~(3u << (2 * (e0 % 4)));
        g0->qs[e0 / 4] |= (uint8_t) (2u << (2 * (e0 % 4)));
        const float v = run();
        if (v == 1.0f) ++hit0; else ++hit_other;
    }
    std::printf("probe A (one weight +1, all activations 1): e0=0 -> %g, e0=32 -> %g, mismatches %d/64\n", a0, a1, badA);
    std::printf("probe B (all 64 weights +1, acts 1): %g (expect 64)\n", bv);
    std::printf("probe C (one weight +1, only q8 lane 0 of block 0 nonzero): met lane 0 %d/64, elsewhere %d\n", hit0,
                hit_other);
    sycl::free(d_blob, q); sycl::free(d_ptr, q); sycl::free(d_start, q); sycl::free(d_dst, q);
    sycl::free(d_tok, q); sycl::free(d_ng, q); sycl::free(d_xq, q); sycl::free(d_scr, q); sycl::free(d_out, q);
}

// ---------------------------------------------------------------- the runner
struct Env {
    sycl::queue* q = nullptr;
    bool ok = true;
};

static bool run_case(Env& E, const Case& C) {
    const NativeExpertLayout L = ref_layout(C.gu, C.dt, C.n_embd, C.n_ff, C.lim);
    if (!ref_supported(C.gu, C.dt, C.n_embd, C.n_ff)) {
        std::printf("SKIP  %-28s %s/%s %lldx%lld: not a shape native_expert_supported accepts\n", C.name.c_str(),
                    ty_name(C.gu).c_str(), ty_name(C.dt).c_str(), (long long) C.n_embd, (long long) C.n_ff);
        return true;
    }
    // host inputs
    std::vector<uint8_t> blobs((size_t) C.cap_groups * L.bytes, 0);
    for (int g = 0; g < C.cap_groups; ++g) {
        fill_blob(blobs.data() + (size_t) g * L.bytes, L, C.style);
        patch_scales(blobs.data() + (size_t) g * L.bytes, L, C.style);
    }
    std::vector<int32_t> grp_start((size_t) C.cap_groups + 1, 0), ent_dst((size_t) C.cap_entries, 0),
        ent_tok((size_t) C.cap_entries, 0);
    int32_t ng = std::min<int32_t>((int32_t) C.ng, (int32_t) C.cap_groups);
    {
        int32_t e = 0;
        for (int g = 0; g < C.cap_groups; ++g) {
            grp_start[g] = e;
            if (g < ng) {
                const int take = (int) (C.cap_entries / std::max(1, C.ng));
                e += std::min<int32_t>(take, (int32_t) (C.cap_entries - e));
            }
        }
        grp_start[C.cap_groups] = e;
        for (int32_t i = 0; i < e; ++i) {
            ent_dst[i] = (int32_t) (C.cap_entries - 1 - i);   // a permutation: one writer per output row
            ent_tok[i] = (int32_t) (i % 8);
        }
        ng = 0;
        for (int g = 0; g < C.cap_groups; ++g) if (grp_start[g] < grp_start[g + 1]) ++ng;
    }
    // activations: 8 tokens of n_embd floats -> q8_1 rows (host reference quantizer = the contract)
    std::vector<float> x((size_t) 8 * L.n_embd, 0.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        const size_t blk = i / 32;
        switch (C.act_style) {
            case 1: x[i] = rnd_f(-1.0f, 1.0f) * (blk % 3 == 0 ? 1.0e9f : 1.0f); break;   // > fp16 max: the #606 clamp
            case 2: x[i] = 0.0f; break;                                                    // amax == 0 blocks
            case 3: x[i] = rnd_f(-1.0f, 1.0f) * 1.0e-30f; break;
            case 4: x[i] = (blk % 5 == 0) ? 0.0f : rnd_f(-3.0f, 3.0f); break;
            default: x[i] = rnd_f(-3.0f, 3.0f); break;
        }
    }
    std::vector<H_Q8> xq((size_t) 8 * L.n_embd / 32);
    for (size_t ib = 0; ib < xq.size(); ++ib) ref_quantize_block(&x[ib * 32], xq[ib]);

    // device buffers (shared USM: the call is handed the pointers, exactly as the engine hands it device ones)
    auto* d_blobs = (uint8_t*) sycl::malloc_shared((size_t) C.cap_groups * L.bytes, *E.q);
    auto* d_ptr = (unsigned long long*) sycl::malloc_shared((size_t) C.cap_groups * sizeof(unsigned long long), *E.q);
    auto* d_start = (int32_t*) sycl::malloc_shared(((size_t) C.cap_groups + 1) * 4, *E.q);
    auto* d_dst = (int32_t*) sycl::malloc_shared((size_t) C.cap_entries * 4, *E.q);
    auto* d_tok = (int32_t*) sycl::malloc_shared((size_t) C.cap_entries * 4, *E.q);
    auto* d_ng = (int32_t*) sycl::malloc_shared(4, *E.q);
    auto* d_xq = (uint8_t*) sycl::malloc_shared(std::max<size_t>(1, (size_t) 8 * L.n_embd / 32 * 36), *E.q);
    const size_t sbytes = ref_scratch_bytes(C.cap_entries, L.n_ff);
    auto* d_scr = (uint8_t*) sycl::malloc_shared(std::max<size_t>(1, sbytes), *E.q);
    auto* d_out = (float*) sycl::malloc_shared(std::max<size_t>(1, (size_t) C.cap_entries * L.n_embd * 4), *E.q);
    if (!d_blobs || !d_ptr || !d_start || !d_dst || !d_tok || !d_ng || !d_xq || !d_scr || !d_out) {
        std::printf("ERROR: allocation failed\n");
        E.ok = false;
        return false;
    }
    std::memcpy(d_blobs, blobs.data(), (size_t) C.cap_groups * L.bytes);
    for (int g = 0; g < C.cap_groups; ++g) d_ptr[g] = (unsigned long long) (uintptr_t) (d_blobs + (size_t) g * L.bytes);
    std::memcpy(d_start, grp_start.data(), (size_t) (C.cap_groups + 1) * 4);
    std::memcpy(d_dst, ent_dst.data(), (size_t) C.cap_entries * 4);
    std::memcpy(d_tok, ent_tok.data(), (size_t) C.cap_entries * 4);
    *d_ng = ng;
    std::memcpy(d_xq, xq.data(), xq.size() * 36);
    std::memset(d_scr, 0, sbytes);
    std::memset(d_out, 0, (size_t) C.cap_entries * L.n_embd * 4);

    if (std::getenv("NE_DBG")) {
        const H_Q2_0* b0 = (const H_Q2_0*) blobs.data();
        std::printf("  dbg blob0: d bits=%04x (%g) qs[0..3]=%02x %02x %02x %02x\n", b0->d, h2f(b0->d), b0->qs[0],
                    b0->qs[1], b0->qs[2], b0->qs[3]);
        std::printf("  dbg xq0: d=%04x (%g) s=%04x (%g) qs[0..3]=%d %d %d %d; x[0..3]=%g %g %g %g\n", xq[0].d,
                    h2f(xq[0].d), xq[0].s, h2f(xq[0].s), xq[0].qs[0], xq[0].qs[1], xq[0].qs[2], xq[0].qs[3], x[0], x[1],
                    x[2], x[3]);
        std::printf("  dbg grp_start[0..%d] =", C.cap_groups);
        for (int i = 0; i <= C.cap_groups; ++i) std::printf(" %d", grp_start[i]);
        std::printf(" ; ent_tok[0..3] = %d %d %d %d\n", ent_tok[0], ent_tok[1], ent_tok[2], ent_tok[3]);
    }
    // host reference buffers
    const size_t fa = ((size_t) C.cap_entries * L.n_ff * 4 + 255) & ~(size_t) 255;
    std::vector<uint8_t> hscr(sbytes, 0);
    float* r_gate = (float*) hscr.data();
    float* r_up = (float*) (hscr.data() + fa);
    float* r_h = (float*) (hscr.data() + 2 * fa);
    H_Q8* r_hq = (H_Q8*) (hscr.data() + 3 * fa);
    std::vector<float> r_out((size_t) C.cap_entries * L.n_embd, 0.0f);
    std::vector<const uint8_t*> hblob((size_t) C.cap_groups);
    for (int g = 0; g < C.cap_groups; ++g) hblob[g] = blobs.data() + (size_t) g * L.bytes;

    // two reference evaluations: contracted and not
    std::vector<float> r_gate_c, r_up_c, r_out_c;
    std::vector<H_Q8> r_hq_c;
    for (int variant = 0; variant < 2; ++variant) {
        RefArgs A;
        A.L = &L;
        A.blob = hblob.data();
        A.grp_start = grp_start.data();
        A.ng = ng;
        A.ent_dst = ent_dst.data();
        A.ent_tok = ent_tok.data();
        A.cap_entries = C.cap_entries;
        A.xq = (const uint8_t*) xq.data();
        A.gate = (float*) hscr.data();
        A.up = (float*) (hscr.data() + fa);
        A.h = (float*) (hscr.data() + 2 * fa);
        A.hq = (H_Q8*) (hscr.data() + 3 * fa);
        A.out = r_out.data();
        A.v1 = C.v1;
        A.contract = variant == 1;
        A.threads = 1;
        std::fill(hscr.begin(), hscr.end(), 0);
        std::fill(r_out.begin(), r_out.end(), 0.0f);
        ref_grouped(A);
        if (variant == 0) {
            r_gate_c.assign(r_gate, r_gate + (size_t) C.cap_entries * L.n_ff);
            r_up_c.assign(r_up, r_up + (size_t) C.cap_entries * L.n_ff);
            r_hq_c.assign(r_hq, r_hq + (size_t) C.cap_entries * (L.n_ff / 32));
            r_out_c = r_out;
        }
    }
    // variant 1 = contracted (the second run overwrote the buffers)
    const std::vector<float> r_gate_f(r_gate, r_gate + (size_t) C.cap_entries * L.n_ff);
    const std::vector<float> r_up_f(r_up, r_up + (size_t) C.cap_entries * L.n_ff);
    const std::vector<H_Q8> r_hq_f(r_hq, r_hq + (size_t) C.cap_entries * (L.n_ff / 32));
    const std::vector<float> r_out_f = r_out;

    // device
    native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, C.cap_groups, C.cap_entries, d_xq, d_scr, d_out, E.q,
                          C.grid_groups);
    E.q->wait();

    const float* d_gate = (const float*) d_scr;
    const float* d_up = (const float*) (d_scr + fa);
    const H_Q8* d_hq = (const H_Q8*) (d_scr + 3 * fa);
    const size_t nrows = (size_t) C.cap_entries * L.n_ff;

    Cmp cg[2], cu[2], chq[2], co[2];
    long long dqs[2] = {0, 0};
    const std::vector<H_Q8>* ref_hq[2] = {&r_hq_c, &r_hq_f};
    const std::vector<float>* ref_gate[2] = {&r_gate_c, &r_gate_f};
    const std::vector<float>* ref_up[2] = {&r_up_c, &r_up_f};
    const std::vector<float>* ref_out[2] = {&r_out_c, &r_out_f};
    for (int v = 0; v < 2; ++v) {
        cmp_floats(ref_gate[v]->data(), d_gate, nrows, cg[v]);
        cmp_floats(ref_up[v]->data(), d_up, nrows, cu[v]);
        cmp_q8(ref_hq[v]->data(), d_hq, (size_t) C.cap_entries * (L.n_ff / 32), chq[v], &dqs[v]);
        cmp_floats(ref_out[v]->data(), d_out, (size_t) C.cap_entries * L.n_embd, co[v]);
    }
    const int best = (co[0].diff <= co[1].diff) ? 0 : 1;
    const bool exact = cg[best].diff == 0 && cu[best].diff == 0 && co[best].diff == 0;

    if (std::getenv("NE_DBG")) {
        std::printf("  dbg %s: dev gate[0..5] =", C.name.c_str());
        for (int i = 0; i < 6; ++i) std::printf(" %.8g", d_gate[i]);
        std::printf("\n  dbg %s: ref gate[0..5] =", C.name.c_str());
        for (int i = 0; i < 6; ++i) std::printf(" %.8g", (*ref_gate[best])[i]);
        std::printf("\n  dbg %s: dev out[0..5] =", C.name.c_str());
        for (int i = 0; i < 6; ++i) std::printf(" %.8g", d_out[i]);
        std::printf("\n  dbg %s: ref out[0..5] =", C.name.c_str());
        for (int i = 0; i < 6; ++i) std::printf(" %.8g", (*ref_out[best])[i]);
        for (int tok = 0; tok < 3; ++tok) {
            std::printf("  dbg %s: truth gate[0..5] tok=%d =", C.name.c_str(), tok);
            for (int i = 0; i < 6; ++i)
                std::printf(" %.8g", truth_dot(C.gu, blobs.data() + (size_t) i * L.gu_row,
                                               (const H_Q8*) ((const uint8_t*) xq.data() + (size_t) tok * (L.n_embd / 32) * 36),
                                               (int) (L.n_embd / ref_qk(C.gu))));
            std::printf("\n");
        }
        std::printf("  dbg %s: truth gate[0..5] =", C.name.c_str());
        for (int i = 0; i < 6; ++i)
            std::printf(" %.8g", truth_dot(C.gu, blobs.data() + (size_t) i * L.gu_row,
                                           (const H_Q8*) ((const uint8_t*) xq.data() + (size_t) ent_tok[0] * (L.n_embd / 32) * 36),
                                           (int) (L.n_embd / ref_qk(C.gu))));
        std::printf("\n");
    }
    std::printf("%-28s %-5s/%-5s %5lldx%-4lld g=%d/%d e=%lld  gate %s  up %s  hq %lld/%lld  out %s", C.name.c_str(),
                ty_name(C.gu).c_str(), ty_name(C.dt).c_str(), (long long) C.n_embd, (long long) C.n_ff, ng,
                C.cap_groups, (long long) C.cap_entries, cg[best].diff ? "DIFF" : "==", cu[best].diff ? "DIFF" : "==",
                (long long) dqs[best], (long long) chq[best].n, co[best].diff ? "DIFF" : "==");
    if (co[best].diff)
        std::printf(" (max %lld ULP, max rel %.3g, n=%lld)", (long long) co[best].max_ulp, co[best].max_rel,
                    (long long) co[best].diff);
    std::printf("  [%s]", best == 1 ? "matches CONTRACTED reference" : "matches UNCONTRACTED reference");
    {
        double worst = 0.0;
        long long worst_u = 0;
        for (size_t i = 0; i < r_out_c.size() && i < r_out_f.size(); ++i) {
            const double d = std::fabs((double) r_out_c[i] - (double) r_out_f[i]);
            const double rel = d / std::max(1e-30, std::fabs((double) r_out_f[i]));
            worst = std::max<double>(worst, rel);
            worst_u = std::max<long long>(worst_u, (long long) Cmp::ulp(r_out_c[i], r_out_f[i]));
        }
        if (worst > 0.0) std::printf(" {FMA contraction alone moves out by <= %lld ULP / %.2g rel}", worst_u, worst);
    }
    if (cg[0].diff != cg[1].diff || co[0].diff != co[1].diff)
        std::printf(" {variants differ: out diff %lld vs %lld}", (long long) co[0].diff, (long long) co[1].diff);
    std::printf("\n");

    sycl::free(d_blobs, *E.q);
    sycl::free(d_ptr, *E.q);
    sycl::free(d_start, *E.q);
    sycl::free(d_dst, *E.q);
    sycl::free(d_tok, *E.q);
    sycl::free(d_ng, *E.q);
    sycl::free(d_xq, *E.q);
    sycl::free(d_scr, *E.q);
    sycl::free(d_out, *E.q);
    return exact;
}

// ---------------------------------------------------------------- benchmark
struct BenchCase {
    int gu, dt;
    int64_t n_embd, n_ff;
    int groups, entries;
    const char* label;
};

static void run_bench(Env& E, int threads, const std::vector<BenchCase>& cases, double* out_sycl_us,
                      double* out_cpu_us) {
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const BenchCase& B = cases[ci];
        const NativeExpertLayout L = ref_layout(B.gu, B.dt, B.n_embd, B.n_ff, 10.0f);
        if (!ref_supported(B.gu, B.dt, B.n_embd, B.n_ff)) {
            std::printf("bench: %s not supported\n", B.label);
            continue;
        }
        // 8 activation tokens
        auto* d_blobs = (uint8_t*) sycl::malloc_device((size_t) B.groups * L.bytes, *E.q);
        auto* d_ptr = (unsigned long long*) sycl::malloc_shared((size_t) B.groups * 8, *E.q);
        auto* d_start = (int32_t*) sycl::malloc_shared(((size_t) B.groups + 1) * 4, *E.q);
        auto* d_dst = (int32_t*) sycl::malloc_shared((size_t) B.entries * 4, *E.q);
        auto* d_tok = (int32_t*) sycl::malloc_shared((size_t) B.entries * 4, *E.q);
        auto* d_ng = (int32_t*) sycl::malloc_shared(4, *E.q);
        auto* d_xq = (uint8_t*) sycl::malloc_device((size_t) 8 * L.n_embd / 32 * 36, *E.q);
        const size_t sbytes = ref_scratch_bytes(B.entries, L.n_ff);
        auto* d_scr = (uint8_t*) sycl::malloc_device(sbytes, *E.q);
        auto* d_out = (float*) sycl::malloc_shared((size_t) B.entries * L.n_embd * 4, *E.q);
        std::vector<uint8_t> blobs((size_t) B.groups * L.bytes);
        g_rng = 0x9E3779B97F4A7C15ull + (uint64_t) ci;
        for (int g = 0; g < B.groups; ++g) {
            fill_blob(blobs.data() + (size_t) g * L.bytes, L, kFuzzNormal);
            patch_scales(blobs.data() + (size_t) g * L.bytes, L, kFuzzNormal);
            d_ptr[g] = (unsigned long long) (uintptr_t) (d_blobs + (size_t) g * L.bytes);
        }
        E.q->memcpy(d_blobs, blobs.data(), (size_t) B.groups * L.bytes);
        int32_t e = 0;
        for (int g = 0; g < B.groups; ++g) {
            d_start[g] = e;
            e += B.entries / B.groups;
        }
        // the last group takes the remainder
        d_start[B.groups] = B.entries;
        for (int i = 0; i < B.entries; ++i) {
            d_dst[i] = i % std::max(1, B.entries);
            d_tok[i] = i % 8;
        }
        *d_ng = B.groups;
        std::vector<H_Q8> xq((size_t) 8 * L.n_embd / 32);
        std::vector<float> x((size_t) 8 * L.n_embd);
        for (size_t i = 0; i < x.size(); ++i) x[i] = rnd_f(-3.0f, 3.0f);
        for (size_t ib = 0; ib < xq.size(); ++ib) ref_quantize_block(&x[ib * 32], xq[ib]);
        E.q->memcpy(d_xq, xq.data(), xq.size() * 36);
        E.q->wait();

        // weights read by the call = groups x (gate+up+down block bytes)
        const double wbytes = (double) B.groups * (double) (2 * (size_t) L.n_ff * L.gu_row + (size_t) L.n_embd * L.d_row);
        const double macs = (double) B.groups * (2.0 * B.n_ff * B.n_embd + (double) B.n_embd * B.n_ff);

        auto gpu_block = [&]() {
            for (int i = 0; i < 5; ++i)
                native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, B.groups, B.entries, d_xq, d_scr, d_out, E.q, 0);
            E.q->wait();
            std::vector<double> t;
            for (int i = 0; i < 50; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                native_expert_grouped(L, d_ptr, d_start, d_ng, d_dst, d_tok, B.groups, B.entries, d_xq, d_scr, d_out, E.q, 0);
                E.q->wait();
                const auto t1 = std::chrono::steady_clock::now();
                t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        const double first_block = gpu_block();

        // CPU: the reference arithmetic, row-parallel (the fallback the port replaces)
        auto* h_blobs = (uint8_t*) sycl::malloc_host((size_t) B.groups * L.bytes, *E.q);
        std::memcpy(h_blobs, blobs.data(), (size_t) B.groups * L.bytes);
        std::vector<const uint8_t*> hptr((size_t) B.groups);
        for (int g = 0; g < B.groups; ++g) hptr[g] = h_blobs + (size_t) g * L.bytes;
        const size_t fa = ((size_t) B.entries * L.n_ff * 4 + 255) & ~(size_t) 255;
        std::vector<uint8_t> hscr(sbytes);
        std::vector<float> hout((size_t) B.entries * L.n_embd);
        std::vector<int32_t> gs((size_t) B.groups + 1), ed((size_t) B.entries), et((size_t) B.entries);
        int32_t ee = 0;
        for (int g = 0; g < B.groups; ++g) { gs[g] = ee; ee += B.entries / B.groups; }
        gs[B.groups] = B.entries;
        for (int i = 0; i < B.entries; ++i) { ed[i] = i % std::max(1, B.entries); et[i] = i % 8; }
        RefArgs A;
        A.L = &L;
        A.blob = hptr.data();
        A.grp_start = gs.data();
        A.ng = B.groups;
        A.ent_dst = ed.data();
        A.ent_tok = et.data();
        A.cap_entries = B.entries;
        A.xq = (const uint8_t*) xq.data();
        A.gate = (float*) hscr.data();
        A.up = (float*) (hscr.data() + fa);
        A.h = (float*) (hscr.data() + 2 * fa);
        A.hq = (H_Q8*) (hscr.data() + 3 * fa);
        A.out = hout.data();
        A.v1 = false;
        A.contract = true;
        A.threads = threads;
        ref_grouped(A);   // warmup
        double cpu_us = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            const auto c0 = std::chrono::steady_clock::now();
            ref_grouped(A);
            const auto c1 = std::chrono::steady_clock::now();
            cpu_us = std::min(cpu_us, std::chrono::duration<double, std::micro>(c1 - c0).count());
        }
        // second GPU block, after the CPU burst: the one reported (the first is printed as a stability check)
        const double sycl_us = gpu_block();

        std::printf("bench %-26s %-5s/%-5s %5lldx%-4lld g=%-2d e=%-2d | SYCL %8.1f us (%8.1f)  %6.2f GMAC/s %6.1f GB/s | "
                    "CPU(%2dT) %9.1f us  %6.2f GMAC/s | speedup %5.1fx\n",
                    B.label, ty_name(B.gu).c_str(), ty_name(B.dt).c_str(), (long long) B.n_embd, (long long) B.n_ff,
                    B.groups, B.entries, sycl_us, first_block, macs / sycl_us / 1e3, wbytes / sycl_us / 1e3, threads,
                    cpu_us, macs / cpu_us / 1e3, cpu_us / sycl_us);
        out_sycl_us[ci] = sycl_us;
        out_cpu_us[ci] = cpu_us;

        sycl::free(d_blobs, *E.q);
        sycl::free(d_ptr, *E.q);
        sycl::free(d_start, *E.q);
        sycl::free(d_dst, *E.q);
        sycl::free(d_tok, *E.q);
        sycl::free(d_ng, *E.q);
        sycl::free(d_xq, *E.q);
        sycl::free(d_scr, *E.q);
        sycl::free(d_out, *E.q);
        sycl::free(h_blobs, *E.q);
    }
}

int main(int argc, char** argv) {
    bool do_validate = false, do_bench = false;
    int threads = 16;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--validate")) do_validate = true;
        else if (!std::strcmp(argv[i], "--bench")) do_bench = true;
        else if (!std::strcmp(argv[i], "--both")) do_validate = do_bench = true;
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }
    if (!do_validate && !do_bench) do_validate = true;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    Env E;
    E.q = &q;

    // fp16 check: the harness's conversions against the runtime's, so a mismatch here cannot be read as a kernel bug
    {
        long long bad = 0;
        for (uint32_t b = 0; b < 65536; ++b) {
            const float f = h2f((uint16_t) b);
            if (f2h(f) != (uint16_t) b && !(std::isnan(f))) ++bad;
        }
        std::printf("fp16 round-trip mismatches: %lld / 65536\n", bad);
    }

    int failures = 0;
    if (do_validate) {
        std::vector<Case> cases;
        // the model's own formats at small legal shapes, with the adversarial inputs the brief asks for
        for (int gu : {42, 11, 143})
            for (int dt : {42, 11, 143}) {
                Case c;
                c.gu = gu;
                c.dt = dt;
                c.n_embd = gu == 42 || dt == 42 ? 512 : 512;   // 512 = 8 x 64, 2 x 256, 4 x 128
                c.n_ff = 768;                                   // 3 x 256, 6 x 128, 12 x 64
                c.ng = 4;
                c.cap_groups = 6;
                c.cap_entries = 12;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " base";
                cases.push_back(c);
                // a single active expert
                c = Case();
                c.gu = gu;
                c.dt = dt;
                c.n_embd = 512;
                c.n_ff = 768;
                c.ng = 1;
                c.cap_groups = 6;
                c.cap_entries = 4;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " one group";
                cases.push_back(c);
                // every group active, several entries each
                c = Case();
                c.gu = gu;
                c.dt = dt;
                c.n_embd = 512;
                c.n_ff = 768;
                c.ng = 6;
                c.cap_groups = 6;
                c.cap_entries = 24;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " all groups";
                cases.push_back(c);
                // zero weights / zero activation blocks
                c = Case();
                c.gu = gu;
                c.dt = dt;
                c.n_embd = 512;
                c.n_ff = 768;
                c.ng = 3;
                c.cap_groups = 4;
                c.cap_entries = 6;
                c.style = kFuzzZero;
                c.act_style = 2;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " zeros";
                cases.push_back(c);
                // extreme / subnormal weight scales, and activations past fp16's range (#606)
                for (FuzzStyle st : {kFuzzExtremeScales, kFuzzSubnormalScales, kFuzzMaxScales}) {
                    c = Case();
                    c.gu = gu;
                    c.dt = dt;
                    c.n_embd = 512;
                    c.n_ff = 768;
                    c.ng = 3;
                    c.cap_groups = 4;
                    c.cap_entries = 6;
                    c.style = st;
                    c.act_style = st == kFuzzMaxScales ? 1 : 0;
                    c.name = ty_name(gu) + "/" + ty_name(dt) +
                             (st == kFuzzExtremeScales ? " extreme-scales" : st == kFuzzSubnormalScales ? " subnormal-scales"
                                                                                                     : " max-scales+huge-act");
                    cases.push_back(c);
                }
                // a column count that is not a multiple of the super-block (the format's own tail)
                c = Case();
                c.gu = gu;
                c.dt = dt;
                // a K that is not a whole number of 256-value super-blocks where the format allows it
                c.n_embd = (gu == 11 || dt == 11) ? 256 : 512;
                c.n_ff = (gu == 11 || dt == 11) ? 256 : (gu == 143 || dt == 143) ? 384 : 384;
                if (!ref_supported(gu, dt, c.n_embd, c.n_ff)) {
                    c.n_embd = 512;
                    c.n_ff = 768;
                }
                c.ng = 2;
                c.cap_groups = 3;
                c.cap_entries = 5;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " short-K";
                cases.push_back(c);
                // the group-stride launch (grid_groups < cap_groups), and the two-pass v1 path
                c = Case();
                c.gu = gu;
                c.dt = dt;
                c.n_embd = 512;
                c.n_ff = 768;
                c.ng = 5;
                c.cap_groups = 6;
                c.cap_entries = 15;
                c.grid_groups = 2;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " strided";
                cases.push_back(c);
                c = Case();
                c.gu = gu;
                c.dt = dt;
                c.n_embd = 512;
                c.n_ff = 768;
                c.ng = 4;
                c.cap_groups = 5;
                c.cap_entries = 9;
                c.v1 = true;
                c.name = ty_name(gu) + "/" + ty_name(dt) + " v1 two-pass";
                cases.push_back(c);
            }
        // a size the model actually runs (5120 x 2304), one class, to show the arithmetic holds at full width
        for (int dt : {42, 11, 143}) {
            Case c;
            c.gu = 42;
            c.dt = dt;
            c.n_embd = 5120;
            c.n_ff = 2304;
            c.ng = 2;
            c.cap_groups = 3;
            c.cap_entries = 6;
            c.name = "real 5120x2304 " + ty_name(42) + "/" + ty_name(dt);
            cases.push_back(c);
        }
        if (std::getenv("NE_SELFTEST")) { selftest(q); return 0; }
    if (std::getenv("NE_SELFTEST2")) { selftest2(q); return 0; }
    if (std::getenv("NE_SELFTEST3")) { selftest3(q); return 0; }
    if (std::getenv("NE_SELFTEST4")) { selftest4(q); return 0; }
    if (std::getenv("NE_DBG")) probe(q);
    std::printf("\n=== numeric parity (%zu cases): SYCL native_expert_grouped vs the CUDA source's arithmetic ===\n",
                    cases.size());
        for (const Case& c : cases) {
            const bool ok = run_case(E, c);
            if (!ok) ++failures;
        }
    }

    if (do_bench) {
        std::vector<BenchCase> bc = {
            {42, 42, 5120, 2304, 6, 6, "decode 6 experts"},
            {42, 42, 5120, 2304, 42, 48, "prefill 8 tok (48 routes)"},
            {42, 42, 5120, 2304, 6, 48, "verify/8-entries-per-expert"},
            {11, 11, 5120, 2304, 6, 6, "decode 6 experts"},
            {11, 11, 5120, 2304, 42, 48, "prefill 8 tok (48 routes)"},
            {143, 143, 5120, 2304, 6, 6, "decode 6 experts"},
            {143, 143, 5120, 2304, 42, 48, "prefill 8 tok (48 routes)"},
            {42, 11, 5120, 2304, 6, 6, "decode 6 experts"},
            {42, 143, 5120, 2304, 6, 6, "decode 6 experts"},
            {11, 42, 5120, 2304, 6, 6, "decode 6 experts"},
            {143, 42, 5120, 2304, 6, 6, "decode 6 experts"},
        };
        std::printf("\n=== timing, model geometry 5120 x 2304 (gate+up 4608 rows x K 5120, down 5120 rows x K 2304) ===\n");
        std::vector<double> s(bc.size(), 0.0), cc(bc.size(), 0.0);
        run_bench(E, threads, bc, s.data(), cc.data());
        std::printf("\nCPU baseline = the reference arithmetic (this file) with %d threads: a scalar implementation of the\n"
                    "same contract, i.e. the fallback the port replaces, not Strata's AVX2/AVX-VNNI native_expert\n", threads);
    }

    std::printf("\n%s\n", failures == 0 ? "ALL CASES BITWISE EQUAL" : "MISMATCHES PRESENT");
    return failures == 0 ? 0 : 1;
}
