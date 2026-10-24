// src/kernels/cuda/fork/fork_layout.cuh - native_expert_layout3 / native_expert_supported3 (gate/up of different types).
// Fork code kept out of upstream's iq_kernels.cu (one #include line there) so upstream merges do not conflict
// on it; included at a fixed point of that translation unit and uses its helpers (not a standalone header).
#pragma once

NativeExpertLayout native_expert_layout3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L = native_expert_layout(gate_type, d_type, n_embd, n_ff);
    if (up_type == gate_type) return L;
    L.up_type = up_type;
    L.up_row = iq_row_bytes(up_type, n_embd);
    L.down_off = L.up_off + (size_t) n_ff * L.up_row;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

bool native_expert_supported3(int gate_type, int up_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    if (gate_type == up_type) return native_expert_supported(gate_type, d_type, n_embd, n_ff);
    const bool pair = (gate_type == 10 && up_type == 11) || (gate_type == 11 && up_type == 10);
    return pair && native_expert_supported(gate_type, d_type, n_embd, n_ff) &&
           native_expert_supported(up_type, d_type, n_embd, n_ff) && n_ff % GU_ROWS == 0;
}
