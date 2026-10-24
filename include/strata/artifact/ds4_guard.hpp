// include/strata/artifact/ds4_guard.hpp - Strata-DS4 fork: the cheap `deepseek4` architecture check, kept out of
// upstream's gguf_reader.hpp (docs/UPSTREAM_SYNC.md).
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <string>

namespace strata {

// ---- DeepSeek-V4-Flash (`deepseek4`) guard, beside the Qwen one rather than replacing it: the two model
// families share this reader, and a guard is what keeps a Qwen-only engine from silently mis-running a DSV4
// file.  The geometry itself (every `deepseek4.*` key, the tensor map, the shape contracts) lives in
// `strata/artifact/ds4_geometry.hpp`; this is the cheap presence/value check a loader can call first.
//
// Values are the ones in docs/ds4/DSV4_ARCH_SPEC.md s1.3 and the real artifact (1328 tensors, 43 layers,
// 256 experts top-6, MLA q_lora_rank 1024 / output_lora_rank 1024 / 8 groups, hc 4, 64 q heads / 1 kv head).
struct Ds4Guard {
    uint32_t block_count = 43, hidden = 4096, head_count = 64, head_count_kv = 1,
             experts = 256, experts_used = 6, gating_func = 4;
};

inline std::string check_architecture_ds4(const GgufFile& g, const Ds4Guard& want = {}) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "deepseek4") return "architecture is '" + arch->s + "', this engine requires 'deepseek4'";
    struct Req {
        const char* key;
        uint64_t want;
    };
    const Req reqs[] = {
        {"deepseek4.block_count", want.block_count},
        {"deepseek4.embedding_length", want.hidden},
        {"deepseek4.attention.head_count", want.head_count},
        {"deepseek4.attention.head_count_kv", want.head_count_kv},
        {"deepseek4.expert_count", want.experts},
        {"deepseek4.expert_used_count", want.experts_used},
        {"deepseek4.expert_gating_func", want.gating_func},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (r.want && v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    return {}; // empty == ok
}

} // namespace strata
