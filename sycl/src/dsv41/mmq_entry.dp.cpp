// [port] llama.cpp's SYCL MMQ (ggml-sycl/mmq.cpp, MIT) behind two plain calls, for the dsv41 prompt's routed experts.
//
// The Q2_K kernel is `static` in that file, so this TU includes it (the pinned llama.cpp checkout, unmodified) and
// exposes:
//   mmq_rows()    float activation rows -> the padded q8_1 layout MMQ reads (quantize_row_q8_1_sycl<quantize_q8_1>)
//   mmq_q2k()     a Q2_K weight matrix (rows x cols) times `n` q8_1 rows -> F32 [n x rows]
//
// The kernel's parameters come from ggml-sycl's own caller (mmq.cpp, ggml_sycl_op_mul_mat_q):
//   ncols_x = ne00                      the weight row length (5120)
//   nrows_x = row_diff                  the weight rows (2304)
//   ncols_y = src1_ncols                the activation rows
//   nrows_y = src1_padded_row_size      ONE padded q8_1 row in BYTES  (= (pad512(ncols_x) / 32) * sizeof(block_q8_1))
//   nrows_dst                           the destination row count
#include "mmq.cpp"

#include "quantize.hpp"


// --- the two backend symbols mmq.cpp reads, without linking ggml-sycl.cpp (the whole backend) -------------------
// The kernels pick their tile shape from the device's compute capability; this port only ever runs on the B70, which is
// Xe2 and takes the largest (VER_GEN13) branch.  ggml_sycl.cpp defines these for the full backend; linking it would pull
// the entire SYCL backend in, so they live here instead.
ggml_sycl_device_info g_ds4_sycl_info{};

const ggml_sycl_device_info & ggml_sycl_info() {
    static const bool once = [] {
        g_ds4_sycl_info.device_count = 1;
        g_ds4_sycl_info.devices[0].cc = VER_GEN13;   // Battlemage
        return true;
    }();
    (void) once;
    return g_ds4_sycl_info;
}

int get_current_device_id() { return 0; }
// ----------------------------------------------------------------------------------------------------------------

namespace strata::ds4::mmq_port {

int pad512(int k) { return ((k + 511) / 512) * 512; }

// bytes of one padded q8_1 row of `k` values
size_t q8_row_bytes(int k) { return (size_t) (pad512(k) / QK8_1) * sizeof(block_q8_1); }

// `rows` float rows of `kx` values -> the padded q8_1 buffer (rows * q8_row_bytes(kx) bytes)
void quantize_rows(const float* x, void* xq, int rows, int kx, dpct::queue_ptr q) {
    quantize_row_q8_1_sycl<quantize_q8_1>(x, xq, kx, rows, pad512(kx), q);
}

// dst[n][ncols_x] = activations[n][ncols_x] . weights^T  (weights: nrows_x rows of ncols_x Q2_K values)
bool mat_q2k(const void* weights, const void* xq, float* dst, int ncols_x, int nrows_x, int n, dpct::queue_ptr q) {
    if (n <= 0 || nrows_x <= 0 || ncols_x % QK_K != 0) return false;
    ggml_mul_mat_q2_K_q8_1_sycl(weights, xq, dst, ncols_x, nrows_x, n, (int) q8_row_bytes(ncols_x), nrows_x, q);
    return true;
}

}  // namespace strata::ds4::mmq_port
