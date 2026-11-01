// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_vision.cu - DeepSeek-V4.1's vision tower (see ds4_vision.cuh).
//
// One encode, N patches on a gh x gw grid:
//   patchify (the reference's (x / 255 - 0.5) / 0.5, BF16) -> x = patch_embed(patches)               FP32 [N][1024]
//   per block, its weights in slot (b % 2) while block b + 1's copy runs:
//     xn = norm1(x); qkv = rope(wqkv(xn) + bias)                       (rows in pieces of rb_: the FP32 GEMM output)
//     per query piece of qb_ rows, per head: S = Q K^T / 8 (FP32), P = softmax(S) (BF16), O[:, head] = P V;
//       x[piece] += wo(O) + bias
//     xn = norm2(x); per row piece: x += w2(silu(gate) * up), gate | up = w1(xn)
//   xn = norm(x); the 3 x 3 unfold; GELU(w1 + b1); w2 + b2 -> BF16 rows (the aligner's weights stream in pieces too)
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ds4_launch.h" // strata_ds4_launch: <<<>>> on CUDA and HIP, a dispatch on Metal
#include "ds4_vision.dp.hpp"
#include <dpct/blas_utils.hpp>

#include "ds4_numerics.dp.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <dpct/lib_common_utils.hpp>

namespace strata::ds4 {
namespace {

/*
DPCT1001:585: The statement could not be removed.
*/
/*
DPCT1000:586: Error handling if-stmt was detected but could not be rewritten.
*/
/*
DPCT1009:588: SYCL reports errors using exceptions and does not use error codes.
Please replace the "get_error_string_dummy(...)" with a real error-handling
function.
*/
#define VK(call)                                                               \
    do {                                                                       \
        dpct::err0 e_ = (call);                                                \
        if (e_ != 0) {                                                         \
            err = std::string("vision: ") + #call + ": " +                     \
                  dpct::get_error_string_dummy(e_);                            \
            return false;                                                      \
        }                                                                      \
    } while (0)
#define VB(call)                                                               \
    do {                                                                       \
        int e_ = (call);                                                       \
        if (e_ != 0) {                                                         \
            err = std::string("vision: ") + #call + ": cuBLAS status " +       \
                  std::to_string((int)e_);                                     \
            return false;                                                      \
        }                                                                      \
    } while (0)

constexpr int kT = 256;                  // threads of the row kernels
constexpr int kSmT = 512, kSmPer = 20;   // softmax: a row of up to 10240 keys in registers
constexpr size_t kWs = 16u << 20;        // cuBLAS workspace

__dpct_inline__ float bf(uint16_t b) {
    return sycl::bit_cast<float>((uint32_t)b << 16);
}
__dpct_inline__ uint16_t bfr(float f) { return bf16_rn(f); }

inline float block_reduce(float v, bool is_max) {
    /*
    DPCT1115:578: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &red = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    /*
    DPCT1115:579: The sycl::ext::oneapi::group_local_memory_for_overwrite is
    used to allocate group-local memory at the none kernel functor scope of a
    work-group data parallel kernel. You may need to adjust the code.
    */
    auto &tot = *sycl::ext::oneapi::group_local_memory_for_overwrite<float>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108:65: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const float u = dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), v,
            o);
        v = is_max ? sycl::fmax(v, u) : v + u;
    }
    const int lane = item_ct1.get_local_id(2) & 31,
              w = item_ct1.get_local_id(2) >> 5,
              nw = (item_ct1.get_local_range(2) + 31) >> 5;
    /*
    DPCT1065:575: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier(); // (an earlier call's readers of red / tot are done)
    if (lane == 0) red[w] = v;
    /*
    DPCT1065:576: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (w == 0) {
        v = lane < nw ? red[lane] : (is_max ? -FLT_MAX : 0.0f);
        for (int o = 16; o > 0; o >>= 1) {
            /*
            DPCT1108:66: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float u = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                v, o);
            v = is_max ? sycl::fmax(v, u) : v + u;
        }
        if (lane == 0) tot = v;
    }
    /*
    DPCT1065:577: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    return tot;
}

// patch i of the gh x gw grid: its 3 x p x p values, channel-major (the reference's reshape / permute)
void patchify_kernel(const uint8_t* __restrict__ rgb, int width, int gw, int p, uint16_t* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = item_ct1.get_group(2), e = item_ct1.get_local_id(2),
              pp = p * p;
    if (e >= 3 * pp) return;
    const int c = e / pp, r = e % pp, y = (i / gw) * p + r / p, x = (i % gw) * p + r % p;
    const float u = (float) rgb[((size_t) y * width + x) * 3 + c];
    /*
    DPCT1013:580: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    out[(size_t)i * 3 * pp + e] = bfr((u / 255.0f - 0.5f) / 0.5f);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void patchify_kernel_wrapper(const uint8_t *__restrict rgb, int width, int gw,
                             int p, uint16_t *__restrict out) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            patchify_kernel(rgb, width, gw, p, out);
        });
}

// x[r] += y[r] (y null: nothing) + bias
void add_rows_kernel(float* __restrict__ x, const float* __restrict__ y, int ldy,
                                const uint16_t* __restrict__ bias, int dim) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2)) {
        float v = x[(size_t) r * dim + e];
        if (y != nullptr) v += y[(size_t) r * ldy + e];
        if (bias != nullptr) v += bf(bias[e]);
        x[(size_t) r * dim + e] = v;
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void add_rows_kernel_wrapper(float *__restrict x, const float *__restrict y,
                             int ldy, const uint16_t *__restrict bias,
                             int dim) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            add_rows_kernel(x, y, ldy, bias, dim);
        });
}

void rmsnorm_kernel(const float* __restrict__ x, const uint16_t* __restrict__ w, int dim, float eps,
                               uint16_t* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2);
    const float* xr = x + (size_t) r * dim;
    float ss = 0.0f;
#pragma unroll
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2)) ss += xr[e] * xr[e];
    ss = block_reduce(ss, false);
    const float rs = sycl::rsqrt(ss / (float)dim + eps);
#pragma unroll
    for (int e = item_ct1.get_local_id(2); e < dim;
         e += item_ct1.get_local_range(2))
        out[(size_t)r * dim + e] = bfr(bf(w[e]) * (xr[e] * rs));
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void rmsnorm_kernel_wrapper(const float *__restrict x,
                            const uint16_t *__restrict w, int dim, float eps,
                            uint16_t *__restrict out) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            rmsnorm_kernel(x, w, dim, eps, out);
        });
}

// a piece's rows of wqkv's output (+ bias, rounded to BF16 as the reference's linear) into qkv, q and k rotated: the
// head's first half pairs with its second; frequency j < hd / 4 turns with the patch's row, the rest with its column
void qkv_rope_kernel(const float* __restrict__ y, const uint16_t* __restrict__ bias, int r0, int gw,
                                const float* __restrict__ inv_freq, int heads, int hd, uint16_t* __restrict__ qkv) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(2), gr = r0 + r, dim = heads * hd,
              half = hd / 2, nf = hd / 4;
    const int row = gr / gw, col = gr % gw;
    const float* yr = y + (size_t) r * 3 * dim;
    uint16_t* o = qkv + (size_t) gr * 3 * dim;
    for (int i = item_ct1.get_local_id(2); i < 2 * heads * half;
         i += item_ct1.get_local_range(2)) {
        const int base = (i / half) * hd, d = i % half;
        const float x1 = bf(bfr(yr[base + d] + bf(bias[base + d])));
        const float x2 = bf(bfr(yr[base + d + half] + bf(bias[base + d + half])));
        float sn, cs;
        /*
        DPCT1017:581: The sycl::sin/cos call is used instead of the sincosf
        call. These two calls do not provide exactly the same functionality.
        Check the potential precision and/or performance issues for the
        generated code.
        */
        sn = sycl::sin((float)(d < nf ? row : col) * inv_freq[d % nf]);
        cs = sycl::cos((float)(d < nf ? row : col) * inv_freq[d % nf]);
        o[base + d] = bfr(x1 * cs - x2 * sn);
        o[base + d + half] = bfr(x2 * cs + x1 * sn);
    }
#pragma unroll
    for (int e = 2 * dim + item_ct1.get_local_id(2); e < 3 * dim;
         e += item_ct1.get_local_range(2)) o[e] = bfr(yr[e] + bf(bias[e]));
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void qkv_rope_kernel_wrapper(const float *__restrict y,
                             const uint16_t *__restrict bias, int r0, int gw,
                             const float *__restrict inv_freq, int heads,
                             int hd, uint16_t *__restrict qkv) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            qkv_rope_kernel(y, bias, r0, gw, inv_freq, heads, hd, qkv);
        });
}

/*
DPCT1110:67: The total declared local variable size in device function
softmax_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void softmax_kernel(const float *__restrict__ s, int n,
                    uint16_t *__restrict__ p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *sr = s + (size_t)item_ct1.get_group(2) * n;
    uint16_t *pr = p + (size_t)item_ct1.get_group(2) * n;
    float v[kSmPer];
    float m = -FLT_MAX;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        v[k] = e < n ? sr[e] : -FLT_MAX;
        m = sycl::fmax(m, v[k]);
    }
    m = block_reduce(m, true);
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        v[k] = e < n ? sycl::native::exp(v[k] - m) : 0.0f;
        sum += v[k];
    }
    const float inv = 1.0f / block_reduce(sum, false);
#pragma unroll
    for (int k = 0; k < kSmPer; ++k) {
        const int e = item_ct1.get_local_id(2) + k * kSmT;
        if (e < n) pr[e] = bfr(v[k] * inv);
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void softmax_kernel_wrapper(const float *__restrict s, int n,
                            uint16_t *__restrict p) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr,
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
            softmax_kernel(s, n, p);
        });
}

// mid = silu(gate) * up, gate | up the halves of w1's row (BF16 steps, as the reference's)
void silu_mul_kernel(const float* __restrict__ y, int inter, uint16_t* __restrict__ mid) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(1),
              e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (e >= inter) return;
    const float gt = bf(bfr(y[(size_t) r * 2 * inter + e])), up = bf(bfr(y[(size_t) r * 2 * inter + inter + e]));
    mid[(size_t)r * inter + e] =
        bfr(bf(bfr(gt / (1.0f + sycl::native::exp(-gt)))) * up);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void silu_mul_kernel_wrapper(const float *__restrict y, int inter,
                             uint16_t *__restrict mid) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            silu_mul_kernel(y, inter, mid);
        });
}

// the aligner's input: block (bi, bj) of the m x m patches as one row, value c * m * m + kh * m + kw (F.unfold's
// order), zero past the grid
void unfold_kernel(const uint16_t* __restrict__ xn, int gh, int gw, int m, int dim, int lw,
                              uint16_t* __restrict__ un) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int l = item_ct1.get_group(2), bi = l / lw, bj = l % lw, mm = m * m;
    for (int f = item_ct1.get_local_id(2); f < dim * mm;
         f += item_ct1.get_local_range(2)) {
        const int c = f / mm, k = f % mm, y = bi * m + k / m, x = bj * m + k % m;
        un[(size_t) l * dim * mm + f] = y < gh && x < gw ? xn[((size_t) y * gw + x) * dim + c] : (uint16_t) 0;
    }
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void unfold_kernel_wrapper(const uint16_t *__restrict xn, int gh, int gw, int m,
                           int dim, int lw, uint16_t *__restrict un) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            unfold_kernel(xn, gh, gw, m, dim, lw, un);
        });
}

// out = act(bf16(y + bias)): GELU (erf) or none
void bias_act_kernel(const float* __restrict__ y, const uint16_t* __restrict__ bias, int n, int gelu,
                                uint16_t* __restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int r = item_ct1.get_group(1),
              e = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);
    if (e >= n) return;
    float v = bf(bfr(y[(size_t) r * n + e] + bf(bias[e])));
    if (gelu) v = 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f));
    out[(size_t) r * n + e] = bfr(v);
}

// Auto generated SYCL kernel wrapper used to migration kernel function pointer.
void bias_act_kernel_wrapper(const float *__restrict y,
                             const uint16_t *__restrict bias, int n, int gelu,
                             uint16_t *__restrict out) {
    sycl::queue queue = *dpct::kernel_launcher::_que;
    unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;


    queue.parallel_for(
        nr, [=](sycl::nd_item<3> item_ct1) {
            bias_act_kernel(y, bias, n, gelu, out);
        });
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

// y[r][o] = alpha * sum_k x[r][k] W[o][k] (row-major W [out][in] with row stride ldw, x [rows][ldx], y [rows][ldy])
bool gemm(dpct::blas::descriptor_ptr h, const uint16_t *W, int ldw,
          const uint16_t *x, int ldx, void *y, dpct::library_data_t yt, int ldy,
          int rows, int in, int out, float alpha, std::string &err) try {
    const float beta = 0.0f;
    VB(DPCT_CHECK_ERROR(dpct::blas::gemm(
        h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, out,
        rows, in, &alpha, W, dpct::library_data_t::real_bfloat16, ldw, x,
        dpct::library_data_t::real_bfloat16, ldx, &beta, y, yt, ldy,
        dpct::compute_type::f32)));
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace

template <class T>
bool VisionTower::dalloc(T *&p, size_t n, std::string &err) try {
    void* v = nullptr;
    const size_t b = (n * sizeof(T) + 255) / 256 * 256;
    if (DPCT_CHECK_ERROR(v = (void *)sycl::malloc_device(
                             b, dpct::get_in_order_queue())) != 0) {
        err = "vision: cudaMalloc " + std::to_string(b >> 20) + " MiB on CUDA" + std::to_string(at_) + " failed";
        return false;
    }
    dallocs_.push_back(v);
    dev_bytes_ += b;
    p = (T*) v;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

VisionTower::~VisionTower() { release(); }

void VisionTower::release() {
    if (at_ < 0) return;
    int cur = 0;
    cur = dpct::get_current_device_id();
    /*
    DPCT1093:582: The "at_" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(at_);
    for (int i = 0; i < 2; ++i) {
        if (copied_[i]) dpct::destroy_event((dpct::event_ptr)copied_[i]);
        if (used_[i]) dpct::destroy_event((dpct::event_ptr)used_[i]);
        copied_[i] = used_[i] = nullptr;
    }
    if (blas_) delete ((dpct::blas::descriptor_ptr)blas_);
    if (st_) {
        dpct::queue_ptr q = (dpct::queue_ptr)st_;
        dpct::get_current_device().destroy_queue(q);
    }
    if (cs_) {
        dpct::queue_ptr q = (dpct::queue_ptr)cs_;
        dpct::get_current_device().destroy_queue(q);
    }
    for (void *p : dallocs_)
        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
    if (host_) sycl::free(host_, dpct::get_in_order_queue());
    /*
    DPCT1093:583: The "cur" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(cur);
    blas_ = st_ = cs_ = nullptr;
    dallocs_.clear();
    host_ = nullptr;
    dev_ = at_ = -1;
    dev_bytes_ = 0;
}

bool VisionTower::load(const VisionGeo& g, int device, const Reader& read, std::string& err) {
    g_ = g;
    const int D = g.dim, I = g.inter, P = g.patch, M = g.merge, hd = g.heads > 0 ? g.dim / g.heads : 0;
    if (g.layers <= 0 || hd * g.heads != D || hd % 4 != 0 || P <= 0 || M <= 0 || g.max_tokens < 5) {
        err = "vision: geometry not covered";
        return false;
    }
    // the largest aligner grid within max_tokens (n_h * (n_w + 1) + 2): the work buffers' size
    int cells = 0;
    for (int h = 1; h <= g.max_tokens - 2; ++h) {
        const int w = (g.max_tokens - 2) / h - 1;
        if (w >= 1) cells = std::max(cells, h * w);
    }
    max_n_ = cells * M * M;
    if (max_n_ > kSmT * kSmPer) {
        err = "vision: " + std::to_string(max_n_) + " patches are more than the attention covers";
        return false;
    }
    /*
    DPCT1093:584: The "device" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(device);
    at_ = device;
    dev_bytes_ = 0;
    // the checkpoint's tensors, checked before anything is read
    struct Want {
        std::string name;
        std::vector<int64_t> shape;
    };
    std::vector<Want> want = {{"vision.patch_embed.proj.weight", {D, 3 * P * P}},
                              {"vision.patch_embed.proj.bias", {D}},
                              {"vision.norm.weight", {D}},
                              {"aligner.w1.weight", {g.out, (int64_t) D * M * M}},
                              {"aligner.w1.bias", {g.out}},
                              {"aligner.w2.weight", {g.out, g.out}},
                              {"aligner.w2.bias", {g.out}}};
    for (int l = 0; l < g.layers; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        want.push_back({B + "norm1.weight", {D}});
        want.push_back({B + "norm2.weight", {D}});
        want.push_back({B + "attn.wqkv.weight", {3 * D, D}});
        want.push_back({B + "attn.wqkv.bias", {3 * D}});
        want.push_back({B + "attn.wo.weight", {D, D}});
        want.push_back({B + "attn.wo.bias", {D}});
        want.push_back({B + "mlp.w1.weight", {2 * I, D}});
        want.push_back({B + "mlp.w2.weight", {D, I}});
    }
    for (const Want& w : want) {
        std::vector<int64_t> s;
        if (!read(w.name, s, nullptr)) {
            err = "vision: " + w.name + " is not in the checkpoint";
            return false;
        }
        if (s != w.shape) {
            err = "vision: " + w.name + " has another shape than the config's";
            return false;
        }
    }
    // the streamed weights, pinned
    block_n_ = (size_t) 3 * D * D + (size_t) D * D + (size_t) 2 * I * D + (size_t) D * I;
    al1_off_ = block_n_ * g.layers;
    al2_off_ = al1_off_ + (size_t) g.out * D * M * M;
    const size_t host_n = al2_off_ + (size_t) g.out * g.out;
    /*
    DPCT1048:12: The original value cudaHostAllocDefault is not meaningful in
    the migrated code and was removed or replaced with 0. You may need to check
    the migrated code.
    */
    if (DPCT_CHECK_ERROR(host_ = (uint16_t *)sycl::malloc_host(
                             host_n * 2, dpct::get_in_order_queue())) != 0) {
        host_ = nullptr;
        err = "vision: pinning " + std::to_string((host_n * 2) >> 20) + " MiB of host memory failed";
        return false;
    }
    std::vector<int64_t> s;
    for (int l = 0; l < g.layers; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        uint16_t* b = host_ + (size_t) l * block_n_;
        if (!read(B + "attn.wqkv.weight", s, b) || !read(B + "attn.wo.weight", s, b + (size_t) 3 * D * D) ||
            !read(B + "mlp.w1.weight", s, b + (size_t) 4 * D * D) ||
            !read(B + "mlp.w2.weight", s, b + (size_t) 4 * D * D + (size_t) 2 * I * D)) {
            err = "vision: block " + std::to_string(l) + ": read failed";
            return false;
        }
    }
    if (!read("aligner.w1.weight", s, host_ + al1_off_) || !read("aligner.w2.weight", s, host_ + al2_off_)) {
        err = "vision: the aligner: read failed";
        return false;
    }
    // the small ones on the device
    auto up = [&](const std::string& name, size_t n, uint16_t*& d) -> bool {
        std::vector<uint16_t> h(n);
        std::vector<int64_t> sh;
        if (!read(name, sh, h.data())) {
            err = "vision: " + name + ": read failed";
            return false;
        }
        if (!dalloc(d, n, err)) return false;
        /*
        DPCT1114:587: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        VK(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(d, h.data(), n * 2)));
        return true;
    };
    bool ok = up("vision.patch_embed.proj.weight", (size_t) D * 3 * P * P, pe_w_) &&
              up("vision.patch_embed.proj.bias", D, pe_b_) && up("vision.norm.weight", D, norm_) &&
              up("aligner.w1.bias", g.out, al1_b_) && up("aligner.w2.bias", g.out, al2_b_);
    n1_.assign(g.layers, nullptr);
    n2_.assign(g.layers, nullptr);
    bqkv_.assign(g.layers, nullptr);
    bo_.assign(g.layers, nullptr);
    for (int l = 0; l < g.layers && ok; ++l) {
        const std::string B = "vision.blocks." + std::to_string(l) + ".";
        ok = up(B + "norm1.weight", D, n1_[l]) && up(B + "norm2.weight", D, n2_[l]) &&
             up(B + "attn.wqkv.bias", 3 * D, bqkv_[l]) && up(B + "attn.wo.bias", D, bo_[l]);
    }
    if (!ok) return false;
    {   // the rotary's frequencies: theta ^ -(2i / (hd / 2)), hd / 4 of them
        const int nf = hd / 4;
        std::vector<float> f(nf);
        for (int i = 0; i < nf; ++i) f[i] = 1.0f / std::pow(g.theta, (float) (2 * i) / (float) (hd / 2));
        if (!dalloc(inv_freq_, nf, err)) return false;
        /*
        DPCT1114:589: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        VK(DPCT_CHECK_ERROR(
            dpct::get_in_order_queue().memcpy(inv_freq_, f.data(), nf * 4)));
    }
    // the work buffers: the patch grid's rows whole (the residual, its norm, q / k / v), the GEMM outputs a piece of
    // rows at a time, the attention's scores a piece of queries at a time (~48 MB)
    const int L = cells;
    rb_ = 2048;
    qb_ = std::max(16, (int) ((48u << 20) / ((size_t) max_n_ * 4)) / 16 * 16);
    slot_n_ = std::max(block_n_, (size_t) 1 << 22);
    const size_t y_n = (size_t) rb_ * std::max({3 * D, 2 * I, D});
    const size_t qkv_n = std::max({(size_t) max_n_ * 3 * D,
                                   (size_t) max_n_ * 3 * P * P + ((size_t) max_n_ * 3 * P * P + 1) / 2,   // patches, pixels
                                   (size_t) L * D * M * M + (size_t) 2 * L * g.out});                      // the aligner's
    if ((size_t) L * g.out > y_n || (size_t) max_n_ * D > y_n) {
        err = "vision: work buffer sizes";
        return false;
    }
    ok = dalloc(x_, (size_t) max_n_ * D, err) && dalloc(xn_, (size_t) max_n_ * D, err) && dalloc(qkv_, qkv_n, err) &&
         dalloc(y_, y_n, err) && dalloc(o_, (size_t) max_n_ * D, err) && dalloc(mid_, (size_t) rb_ * I, err) &&
         dalloc(s_, (size_t) qb_ * max_n_, err) && dalloc(p_, (size_t) qb_ * max_n_, err) &&
         dalloc(slot_[0], slot_n_, err) && dalloc(slot_[1], slot_n_, err) && dalloc(ws_, kWs, err);
    if (!ok) return false;
    patch_ = qkv_;   // (before block 0's qkv)
    rgb_ = (uint8_t*) (qkv_ + (size_t) max_n_ * 3 * P * P);
    un_ = qkv_;      // (after the last block's attention)
    a16_ = un_ + (size_t) L * D * M * M;
    outb_ = a16_ + (size_t) L * g.out;
    dpct::blas::descriptor_ptr h = nullptr;
    VB(DPCT_CHECK_ERROR(h = new dpct::blas::descriptor()));
    blas_ = h;
    dpct::queue_ptr st = &dpct::get_in_order_queue(),
                    cs = &dpct::get_in_order_queue();
    /*
    DPCT1025:590: The SYCL queue is created ignoring the flag and priority
    options.
    */
    VK(DPCT_CHECK_ERROR(st = dpct::get_current_device().create_queue(true)));
    /*
    DPCT1025:591: The SYCL queue is created ignoring the flag and priority
    options.
    */
    VK(DPCT_CHECK_ERROR(cs = dpct::get_current_device().create_queue(true)));
    st_ = st;
    cs_ = cs;
    VB(DPCT_CHECK_ERROR(h->set_queue(st)));
    /*
    DPCT1027:592: The call to cublasSetWorkspace was replaced with 0 because
    this functionality is redundant in SYCL.
    */
    VB(0);
    for (int i = 0; i < 2; ++i) {
        dpct::event_ptr a = nullptr, b = nullptr;
        VK(DPCT_CHECK_ERROR(a = new sycl::event()));
        VK(DPCT_CHECK_ERROR(b = new sycl::event()));
        copied_[i] = a;
        used_[i] = b;
    }
    dev_ = device;
    return true;
}

bool VisionTower::encode(const uint8_t *rgb, int width, int height,
                         std::vector<uint16_t> &rows, int &n_h, int &n_w,
                         std::string &err) try {
    if (dev_ < 0) {
        err = "vision: not loaded";
        return false;
    }
    const VisionGeo& g = g_;
    const int D = g.dim, I = g.inter, P = g.patch, M = g.merge, hd = D / g.heads, H3 = 3 * D;
    if (width <= 0 || height <= 0 || width % P != 0 || height % P != 0) {
        err = "vision: an image of " + std::to_string(width) + " x " + std::to_string(height) +
              " pixels (both sides must be multiples of " + std::to_string(P) + ")";
        return false;
    }
    const int gh = height / P, gw = width / P, N = gh * gw;
    n_h = cdiv(gh, M);
    n_w = cdiv(gw, M);
    const int L = n_h * n_w;
    if (N > max_n_ || n_h * (n_w + 1) + 2 > g.max_tokens) {
        err = "vision: an image of " + std::to_string(width) + " x " + std::to_string(height) + " pixels is more than " +
              std::to_string(g.max_tokens) + " image tokens";
        return false;
    }
    /*
    DPCT1093:593: The "dev_" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(dev_);
    dpct::blas::descriptor_ptr h = (dpct::blas::descriptor_ptr)blas_;
    dpct::queue_ptr s = (dpct::queue_ptr)st_, cs = (dpct::queue_ptr)cs_;
    // the weights in the order the encode takes them: the blocks, then the aligner's in pieces of a slot
    std::vector<Piece> pcs;
    for (int l = 0; l < g.layers; ++l) pcs.push_back({host_ + (size_t) l * block_n_, block_n_});
    const int in1 = D * M * M, r1 = (int) std::min<size_t>(g.out, slot_n_ / in1), r2 = (int) std::min<size_t>(g.out, slot_n_ / g.out);
    for (int o0 = 0; o0 < g.out; o0 += r1)
        pcs.push_back({host_ + al1_off_ + (size_t) o0 * in1, (size_t) std::min(r1, g.out - o0) * in1});
    for (int o0 = 0; o0 < g.out; o0 += r2)
        pcs.push_back({host_ + al2_off_ + (size_t) o0 * g.out, (size_t) std::min(r2, g.out - o0) * g.out});
    auto fetch = [&](size_t i) -> bool {
        const int k = (int) (i % 2);
        VK(DPCT_CHECK_ERROR(
            cs->ext_oneapi_submit_barrier({*(dpct::event_ptr)used_[k]})));
        /*
        DPCT1124:594: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        VK(DPCT_CHECK_ERROR(cs->memcpy(slot_[k], pcs[i].host, pcs[i].n * 2)));
        /*
        DPCT1024:595: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        VK(DPCT_CHECK_ERROR(
            dpct::sync_barrier((dpct::event_ptr)copied_[k], cs)));
        return true;
    };
    size_t pi = 0;   // the piece being used
    auto take = [&]() -> const uint16_t* {   // the next piece's slot, once its copy is in (the one after it started)
        const int k = (int) (pi % 2);
        if (pi + 1 < pcs.size() && !fetch(pi + 1)) return nullptr;
        if (DPCT_CHECK_ERROR(s->ext_oneapi_submit_barrier(
                {*(dpct::event_ptr)copied_[k]})) != 0) {
            err = "vision: waiting for a weight copy failed";
            return nullptr;
        }
        return slot_[k];
    };
    auto done = [&]() {   // the piece's work is queued: its slot is free once that ran
        dpct::sync_barrier((dpct::event_ptr)used_[pi % 2], s);
        ++pi;
    };
    if (!fetch(0)) return false;

    // the patches
    /*
    DPCT1124:596: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    VK(DPCT_CHECK_ERROR(s->memcpy(rgb_, rgb, (size_t)N * 3 * P * P)));
    strata_ds4_launch(patchify_kernel_wrapper, N, 3 * P * P, 0, s, rgb_, width,
                      gw, P, patch_);
    if (!gemm(h, pe_w_, 3 * P * P, patch_, 3 * P * P, x_,
              dpct::library_data_t::real_float, D, N, 3 * P * P, D, 1.0f,
              err)) return false;
    strata_ds4_launch(add_rows_kernel_wrapper, N, kT, 0, s, x_, nullptr, 0,
                      pe_b_, D);

    const float scale = 1.0f / std::sqrt((float) hd);
    const int qb = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, (size_t) qb_ * max_n_ / (size_t) N));
    for (int l = 0; l < g.layers; ++l) {
        const uint16_t* wqkv = take();
        if (wqkv == nullptr) return false;
        const uint16_t *wo = wqkv + (size_t) H3 * D, *w1 = wo + (size_t) D * D, *w2 = w1 + (size_t) 2 * I * D;
        strata_ds4_launch(rmsnorm_kernel_wrapper, N, kT, 0, s, x_, n1_[l], D,
                          g.eps, xn_);
        for (int r0 = 0; r0 < N; r0 += rb_) {
            const int nr = std::min(rb_, N - r0);
            if (!gemm(h, wqkv, D, xn_ + (size_t)r0 * D, D, y_,
                      dpct::library_data_t::real_float, H3, nr, D, H3, 1.0f,
                      err)) return false;
            strata_ds4_launch(qkv_rope_kernel_wrapper, nr, kT, 0, s, y_,
                              bqkv_[l], r0, gw, inv_freq_, g.heads, hd, qkv_);
        }
        for (int q0 = 0; q0 < N; q0 += qb) {
            const int nq = std::min(qb, N - q0);
            const float one = 1.0f, zero = 0.0f;
            for (int hh = 0; hh < g.heads; ++hh) {
                // S^T (keys x queries, column-major) = K Q^T: row-major S [query][key]
                VB(DPCT_CHECK_ERROR(dpct::blas::gemm(
                    h, oneapi::mkl::transpose::trans,
                    oneapi::mkl::transpose::nontrans, N, nq, hd, &scale,
                    qkv_ + D + hh * hd, dpct::library_data_t::real_bfloat16, H3,
                    qkv_ + (size_t)q0 * H3 + hh * hd,
                    dpct::library_data_t::real_bfloat16, H3, &zero, s_,
                    dpct::library_data_t::real_float, N,
                    dpct::compute_type::f32)));
                strata_ds4_launch(softmax_kernel_wrapper, nq, kSmT, 0, s, s_, N,
                                  p_);
                // O^T (hd x queries) = V^T P^T: row-major O [query][head's hd] at column hh * hd
                VB(DPCT_CHECK_ERROR(dpct::blas::gemm(
                    h, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, hd, nq, N, &one,
                    qkv_ + 2 * D + hh * hd, dpct::library_data_t::real_bfloat16,
                    H3, p_, dpct::library_data_t::real_bfloat16, N, &zero,
                    o_ + hh * hd, dpct::library_data_t::real_bfloat16, D,
                    dpct::compute_type::f32)));
            }
            if (!gemm(h, wo, D, o_, D, y_, dpct::library_data_t::real_float, D,
                      nq, D, D, 1.0f, err)) return false;
            strata_ds4_launch(add_rows_kernel_wrapper, nq, kT, 0, s,
                              x_ + (size_t)q0 * D, y_, D, bo_[l], D);
        }
        strata_ds4_launch(rmsnorm_kernel_wrapper, N, kT, 0, s, x_, n2_[l], D,
                          g.eps, xn_);
        for (int r0 = 0; r0 < N; r0 += rb_) {
            const int nr = std::min(rb_, N - r0);
            if (!gemm(h, w1, D, xn_ + (size_t)r0 * D, D, y_,
                      dpct::library_data_t::real_float, 2 * I, nr, D, 2 * I,
                      1.0f, err)) return false;
            strata_ds4_launch(silu_mul_kernel_wrapper,
                              dpct::dim3(cdiv(I, kT), nr), kT, 0, s, y_, I,
                              mid_);
            if (!gemm(h, w2, I, mid_, I, y_, dpct::library_data_t::real_float,
                      D, nr, I, D, 1.0f, err)) return false;
            strata_ds4_launch(add_rows_kernel_wrapper, nr, kT, 0, s,
                              x_ + (size_t)r0 * D, y_, D, nullptr, D);
        }
        done();
    }
    strata_ds4_launch(rmsnorm_kernel_wrapper, N, kT, 0, s, x_, norm_, D, g.eps,
                      xn_);

    // the aligner
    strata_ds4_launch(unfold_kernel_wrapper, L, kT, 0, s, xn_, gh, gw, M, D,
                      n_w, un_);
    for (int o0 = 0; o0 < g.out; o0 += r1) {
        const uint16_t* w = take();
        if (w == nullptr) return false;
        if (!gemm(h, w, in1, un_, in1, y_ + o0,
                  dpct::library_data_t::real_float, g.out, L, in1,
                  std::min(r1, g.out - o0), 1.0f, err))
            return false;
        done();
    }
    strata_ds4_launch(bias_act_kernel_wrapper, dpct::dim3(cdiv(g.out, kT), L),
                      kT, 0, s, y_, al1_b_, g.out, 1, a16_);
    for (int o0 = 0; o0 < g.out; o0 += r2) {
        const uint16_t* w = take();
        if (w == nullptr) return false;
        if (!gemm(h, w, g.out, a16_, g.out, y_ + o0,
                  dpct::library_data_t::real_float, g.out, L, g.out,
                  std::min(r2, g.out - o0), 1.0f, err))
            return false;
        done();
    }
    strata_ds4_launch(bias_act_kernel_wrapper, dpct::dim3(cdiv(g.out, kT), L),
                      kT, 0, s, y_, al2_b_, g.out, 0, outb_);
    rows.resize((size_t) L * g.out);
    /*
    DPCT1124:597: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    VK(DPCT_CHECK_ERROR(s->memcpy(rows.data(), outb_, rows.size() * 2)));
    VK(DPCT_CHECK_ERROR(s->wait()));
    /*
    DPCT1010:598: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    VK(0);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::ds4
