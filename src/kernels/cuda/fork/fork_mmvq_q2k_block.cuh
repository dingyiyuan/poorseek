// src/kernels/cuda/fork/fork_mmvq_q2k_block.cuh - Q2_K block layout (DeepSeek-V4 down projections) for the native MMVQ path.
// Fork code kept out of upstream's native_mmvq.cu (one #include line there); included at a fixed point of that
// translation unit and uses its helpers (not a standalone header).
#pragma once

struct Q2KBlock {
    uint8_t scales[16];
    uint8_t qs[64];
    half2 dm;
};

static_assert(sizeof(Q2KBlock) == 84 && alignof(Q2KBlock) == 4 && offsetof(Q2KBlock, qs) == 16 &&
              offsetof(Q2KBlock, dm) == 80);
