// [port] DS4_EXPERT_MMQ=1: the prompt path's routed experts through llama.cpp's own SYCL MMQ (ggml-sycl/mmq.cpp, MIT),
// one Q2_K expert at a time: gate . x -> up . x -> swiglu -> q8_1 -> down . h.
//
// The kernel is `ggml_mul_mat_q2_K_q8_1_sycl(weights, q8_1_rows, dst, cols, rows, acts, padded_row_bytes, nrows_dst)`:
// dst[act * nrows_dst + row].  Its activations must be the padded q8_1 rows that quantize_row_q8_1_sycl writes, which
// mmq_entry.dp.cpp exposes.
//
// Measured on the B70 (docs section 18): 14.7 TFLOPS for one 2304x5120 Q2_K matrix x 128 activation rows, against the
// dp4a path's 3.8-4.2 (gate+up) and 5.6-6.6 (down).  Only Q2_K is wired here, and only if the layer's experts are Q2_K:
// the pack must be requantized (tools/dsv41/expert_requant.cpp), and everything else keeps ds4_iq's kernels.
#include <dpct/dpct.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "strata/dsv41/exl3.hpp"
#include "ds4_iq.dp.hpp"   // (the engine's ggml type ids: kQ2_K = 10)

namespace strata::ds4::mmq_port {
int pad512(int k);
size_t q8_row_bytes(int k);
void quantize_rows(const float* x, void* xq, int rows, int kx, dpct::queue_ptr q);
bool mat_q2k(const void* weights, const void* xq, float* dst, int ncols_x, int nrows_x, int n, dpct::queue_ptr q);
}  // namespace strata::ds4::mmq_port

namespace x3 = strata::dsv41k::exl3;

namespace strata::ds4::ds4port {

bool expert_mmq_on() {
    static const bool v = [] {
        const char* e = std::getenv("DS4_EXPERT_MMQ");
        return e != nullptr && e[0] != '0';
    }();
    return v;
}

namespace {

constexpr int QK_K_BLOCK = 256;   // Q2_K's block size

bool is_q2k(int t) { return t == gguf::iq::kQ2_K; }   // 10 in ggml's enum (14 is Q6_K: this used to skip the branch)

// grow-only device scratch
struct Buf {
    void* p = nullptr;
    size_t have = 0;
};
void* need(Buf& b, size_t bytes, dpct::queue_ptr q) {
    if (b.p != nullptr && b.have >= bytes) return b.p;
    if (b.p != nullptr) sycl::free(b.p, *q);
    b.p = sycl::malloc_device(bytes, *q);
    b.have = b.p != nullptr ? bytes : 0;
    return b.p;
}
Buf g_xq, g_gu, g_hq, g_h;

// h = silu(gate) * up over `rows` x `ff` values, from the [row][2*ff] interleaved gate/up buffer
void swiglu(const float* gu, float* h, int rows, int ff, dpct::queue_ptr q) {
    const int n = rows * ff;
    q->parallel_for(sycl::range<1>((size_t) ((n + 255) / 256) * 256), [=](sycl::item<1> it) {
        const int i = (int) it.get_id(0);
        if (i >= n) return;
        const int r = i / ff, k = i % ff;
        const float g = gu[(size_t) r * 2 * ff + k];
        const float u = gu[(size_t) r * 2 * ff + ff + k];
        h[i] = (g / (1.0f + sycl::native::exp(-g))) * u;
    });
}

}  // namespace

// are the layer's experts MMQ-able here (Q2_K, both matrices)?
bool mmq_ready(int gu_type, int d_type) { return is_q2k(gu_type) && is_q2k(d_type); }

// One group of experts (<= 16) through the Q2_K MMQ kernels.  `cnt` are the group's rows per expert, `x` points at the
// group's first activation row (F32, ldx a row) and `y` at its first output row (F32, ldy a row).
//
// The group's scratch is bounded by walking the rows in windows: a 8K chunk's group can hold thousands of rows, and its
// gate/up tile alone would be 150 MB (which is what an UR_RESULT_ERROR_OUT_OF_RESOURCES cost us on the first full run).
bool mmq_set(const x3::SetArgs& sa, int n, const int* cnt, const float* x, int ldx, float* y, int ldy,
             dpct::queue_ptr q) {
    if (n <= 0 || n > 16) return false;
    const int K = sa.n_embd, FF = sa.n_ff;
    if (K % QK_K_BLOCK != 0 || FF % QK_K_BLOCK != 0) return false;
    int rows = 0;
    for (int i = 0; i < n; ++i) rows += cnt[i];
    if (rows <= 0) return true;

    constexpr int kPass = 512;   // rows a pass, so the scratch stays ~20 MB
    const size_t xq_b = mmq_port::q8_row_bytes(K) * (size_t) kPass;
    const size_t hq_b = mmq_port::q8_row_bytes(FF) * (size_t) kPass;
    void* xq = need(g_xq, xq_b, q);
    void* hq = need(g_hq, hq_b, q);
    float* gu = (float*) need(g_gu, sizeof(float) * (size_t) kPass * 2 * FF, q);
    float* h = (float*) need(g_h, sizeof(float) * (size_t) kPass * FF, q);
    if (!xq || !hq || !gu || !h) return false;

    const uint8_t* g0 = sa.wbase + sa.lay.trellis[0];
    const size_t up_off = sa.lay.trellis[1] - sa.lay.trellis[0];
    const size_t down_off = sa.lay.trellis[2] - sa.lay.trellis[0];
    const size_t xq_stride = mmq_port::q8_row_bytes(K), hq_stride = mmq_port::q8_row_bytes(FF);

    // the experts' row ranges in the group
    int estart[16] = {}, eend[16] = {};
    for (int e = 0, at = 0; e < n; ++e) {
        estart[e] = at;
        at += cnt[e];
        eend[e] = at;
    }
    for (int w0 = 0; w0 < rows; w0 += kPass) {
        const int w1 = std::min(rows, w0 + kPass);
        const int wrows = w1 - w0;
        mmq_port::quantize_rows(x + (size_t) w0 * ldx, xq, wrows, K, q);
        for (int e = 0; e < n; ++e) {
            const int s = std::max(w0, estart[e]), t = std::min(w1, eend[e]);
            if (t <= s) continue;
            const int r = t - s, roff = s - w0;             // this expert's rows in the pass
            const uint8_t* blob = g0 + (size_t) e * sa.stride;
            void* bxq = (uint8_t*) xq + (size_t) roff * xq_stride;
            float* bgu = gu + (size_t) roff * 2 * FF;
            if (!mmq_port::mat_q2k(blob, bxq, bgu, K, FF, r, q)) return false;                  // gate
            if (!mmq_port::mat_q2k(blob + up_off, bxq, bgu + FF, K, FF, r, q)) return false;     // up
            swiglu(bgu, h + (size_t) roff * FF, r, FF, q);
            void* bhq = (uint8_t*) hq + (size_t) roff * hq_stride;
            mmq_port::quantize_rows(h + (size_t) roff * FF, bhq, r, FF, q);
            if (!mmq_port::mat_q2k(blob + down_off, bhq, y + (size_t) s * ldy, FF, K, r, q)) return false;   // down
        }
    }
    return true;
}

}  // namespace strata::ds4::ds4port
