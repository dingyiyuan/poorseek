// src/kernels/cuda/fork/fork_mmvq_q2k_launch.cuh - the Q2_K native MMVQ entry point.
// Fork code kept out of upstream's native_mmvq.cu (one #include line there); included at a fixed point of that
// translation unit and uses its helpers (not a standalone header).
#pragma once

void native_q2_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q2KTraits)
    if (ncols > 1) {
        launch_multi<Q2KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    // One kernel template for every column count, so a column's sum is bitwise the same with or without the
    // others riding along (the native MMVQ multi_exact contract, mmvq_multi_parity).
    const auto s = static_cast<cudaStream_t>(stream);
    launch_multi_n<Q2KTraits, 1>(weights, x_q8_1, y, n_in, n_out, s);
    launch_check();
}

void native_q2_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}
