// [port] DS4_EXPERT_I8=1: the prompt path's routed experts through the int8 DPAS kernels the port already carries
// (sycl/src/prefill/esimd_moe.dp.cpp: Battlemage s8 8x16x32, SGM 8-row tiles), instead of ds4_iq's dp4a.
//
// The kernels want per-group layouts, int8 activation tiles and the experts' own quantized blobs by *ggml type*:
//   quantize_rows(F16 rows) -> xq/dx, gate_up(gu_type, blob[], up_off, ...) -> GU[rows][2*n_ff] interleaved,
//   swiglu_quantize -> hq/dh, down(d_type, blob[], down_off, ...) -> Dm[rows][n_embd].
// Their coverage is IQ3_S / IQ3_XXS / IQ2_S (gate+up) and IQ4_NL / Q2_0 (down) - not this pack's IQ2_XXS / IQ2_XS /
// IQ1_M - so the switch only pays once the experts are requantized into those (see docs/DSV41_SYCL_PORT.md section 16);
// until then gate_up/down return false here and the caller keeps the dp4a path.
#include <dpct/dpct.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/prefill/esimd_moe.hpp"
#include "strata/dsv41/exl3.hpp"

namespace x3 = strata::dsv41k::exl3;

namespace strata::ds4::ds4port {

bool expert_i8_on() {   // (not inline: ds4_model declares and calls it)
    static const bool v = [] {
        const char* e = std::getenv("DS4_EXPERT_I8");
        return e != nullptr && e[0] != '0';
    }();
    return v;
}

// the interleaved gate/up tile: rows x 2*n_ff floats, grow-only on the queue's device
inline float* i8_gu(float*& p, size_t& have, size_t want, dpct::queue_ptr s) {
    if (p != nullptr && have >= want) return p;
    if (p != nullptr) sycl::free(p, *s);
    p = (float*) sycl::malloc_device(want * sizeof(float), *s);
    have = p != nullptr ? want : 0;
    return p;
}

// Are these types runnable here at all (checked once per run_set group, before any sub-group writes anything)?
bool i8_ready(dpct::queue_ptr s, int gu_type, int d_type) {
    namespace i8 = strata::prefill::i8;
    // (the type test lives in gate_up/down themselves; a sub-group they refuse makes the caller redo the whole group on
    // the dp4a path, which is correct but wasted, so a pack in the supported types is what this switch wants)
    (void) gu_type;
    (void) d_type;
    return i8::available(s);
}

// One group of experts (<= i8::kMaxGroup of them, rows relative to the group) through the int8 kernels.  `x16` points at
// the group's first activation row (F16, n_embd a row), `y` at its first output row (F32, ld y floats a row).
// false: not applicable (the caller keeps the dp4a path); it is a silent "no" by design.
bool i8_set(const x3::SetArgs& sa, int gu_type, int d_type, const int* cnt, int n, const uint16_t* x16, int ldx,
            float* y, int ldy, dpct::queue_ptr s) {
    namespace i8 = strata::prefill::i8;
    if (n <= 0 || n > i8::kMaxGroup) return false;
    if (!i8::available(s)) return false;
    static const bool dbg = std::getenv("DS4_EXPERT_I8_DBG") != nullptr;
    // the kernels' own alignment rule, and a sanity bound on the group's rows (a wild count here would size the
    // scratch and the output from garbage)
    const size_t up_off0 = sa.lay.trellis[1] - sa.lay.trellis[0];
    const size_t down_off0 = sa.lay.trellis[2] - sa.lay.trellis[0];
    if (sa.lay.trellis[1] < sa.lay.trellis[0] || sa.lay.trellis[2] < sa.lay.trellis[1] || (up_off0 & 3) != 0 ||
        (down_off0 & 3) != 0)
        return false;
    if (dbg) {
        std::fprintf(stderr, "ds4 i8: n=%d K=%d ff=%d wbase=%p stride=%zu trellis=%zu,%zu,%zu up_off=%zu down_off=%zu\n",
                     n, sa.n_embd, sa.n_ff, (const void*) sa.wbase, sa.stride, (size_t) sa.lay.trellis[0],
                     (size_t) sa.lay.trellis[1], (size_t) sa.lay.trellis[2], up_off0, down_off0);
        std::fflush(stderr);
    }
    const int K = sa.n_embd, FF = sa.n_ff;
    int32_t c32[i8::kMaxGroup] = {};
    int64_t rows = 0;
    for (int i = 0; i < n; ++i) {
        c32[i] = cnt[i];
        rows += cnt[i];
    }
    if (rows <= 0) return true;   // nothing to do (handled)
    if (dbg) {
        std::fprintf(stderr, "ds4 i8: rows=%lld cnt=", (long long) rows);
        for (int i = 0; i < n; ++i) std::fprintf(stderr, "%d%s", cnt[i], i + 1 == n ? "\n" : ",");
        std::fflush(stderr);
    }
    if (rows > 16 * 4096) return false;   // (a group of 16 experts of a 8K chunk at most)
    i8::Layout lo{};
    i8::make_layout(lo, c32, n);
    i8::Workspace ws{};
    if (!i8::workspace(lo, K, FF, ws, s)) return false;

    // the experts' blob bases: expert i's gate at blob[i] + 0, its up at + up_off, its down at + down_off
    const uint8_t* gate0 = sa.wbase + sa.lay.trellis[0];
    const size_t up_off = sa.lay.trellis[1] - sa.lay.trellis[0];
    const size_t down_off = sa.lay.trellis[2] - sa.lay.trellis[0];
    const uint8_t* host_blob[i8::kMaxGroup] = {};
    for (int i = 0; i < n; ++i) host_blob[i] = gate0 + (size_t) i * sa.stride;
    const uint8_t** blob = (const uint8_t**) sycl::malloc_device(sizeof(void*) * (size_t) n, *s);
    if (blob == nullptr) return false;
    s->memcpy(blob, host_blob, sizeof(void*) * (size_t) n);

    i8::quantize_rows(x16, lo, K, ws.xq, ws.dx, s);
    static float* gu = nullptr;
    static size_t gu_have = 0;
    float* GU = i8_gu(gu, gu_have, (size_t) rows * (size_t) (2 * FF), s);
    if (GU == nullptr) {
        sycl::free((void*) blob, *s);
        return false;
    }
    if (!i8::gate_up(gu_type, blob, up_off, FF, lo, K, ws.xq, ws.dx, GU, s)) {
        sycl::free((void*) blob, *s);
        return false;
    }
    i8::swiglu_quantize(GU, lo, FF, ws.hq, ws.dh, s);
    if (!i8::down(d_type, blob, down_off, K, lo, FF, ws.hq, ws.dh, y, s)) {
        sycl::free((void*) blob, *s);
        return false;
    }
    sycl::free((void*) blob, *s);
    (void) ldx;
    (void) ldy;   // (the layouts are dense: n_embd a row in, K a row out)
    return true;
}

}  // namespace strata::ds4::ds4port
