// IQ2_XXS 专家权重解量化的微基准：把 set_tc_kernel 的内层拆成可单独计时的四段。
//
// 背景（2026-11-02 实测）：专家内核 gate+up 只有 4.2 TFLOPS、down 6.0 TFLOPS（B70 fp16 峰值约 100），
// 且与 rows/expert 无关（900→930 无变化）→ 批量/瓦片不是瓶颈。
//
// 【本次要证伪的假设链】
//   set_tc_kernel 的每个 workgroup 每步（32 个 K）要 *现解* 64 行的 32 个权重（共 2048 值），
//   然后才做 64×64×32 的 MMA。也就是说：**每个解出来的权重只被 BM=64 行复用**。
//   于是  implied_TFLOPS = Gval/s × 2 × BM / 1000 = Gval/s × 0.128。
//   反过来：4.2 TFLOPS ⇔ 33 Gval/s。若明早测得 GPU 解量化吞吐就在 30–40 Gval/s，
//   则"解量化+瓦片构建"就是墙，且**两段式（每专家只解一次，900 行复用）能省 900/64 ≈ 14 倍**。
//   若测得远高于此（例如 300 Gval/s），则墙在别处（barrier / MMA 形状），本假设不成立。
//
// 【四段对照】（ref/bat 与引擎 :208 同构；deq8/tile 与引擎 :449/:1216 同构）
//   ref   一块一个 work-item，逐 ib × 逐 l 查表
//   bat   一个 work-item 处理 B 块，表搬进 local memory
//   deq8  引擎真正的原语：一个 work-item 解 8 个值 → f32（ds4_iq.dp.cpp:449 deq8）
//   tile  在上面加"f16 转换 + 写 local 瓦片 + barrier"（:1216-1230 的落位），
//         并且扫 BM=32/64/128 × 标量存/16B向量存 × 标量解量化/SIMD 解量化
//
// 不读模型（不加载 164 GB 包），合成权重；块数/步数都是参数。
// 用法: iq_dequant_bench [blocks] [reps] [batch] [ksteps] [groups]
//   blocks  供 ref/bat 用的块数（默认 131072，33.6 M 值）
//   ksteps  tile 变体的 K 步数（默认 80 → K = 2560，与 n_embd 一致）
//   groups  tile 变体的 workgroup 数（默认 128 → 1 M 值/步）

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

namespace {

constexpr int kQK = 256;          // 每块的值数
constexpr int kBlockBytes = 66;   // IQ2_XXS：half d + 32 × uint16 qs
constexpr int kGrid = 512;        // 引擎 tab 是 uint32[512]，IQ2_XXS 用前 256 项（每项 2 个 u32 = 8 个值）
constexpr int kBM = 64;           // set_tc_kernel 的激活行瓦片（决定权重复用倍数）
constexpr int kKS = 32;           // 每次迭代的 K 步长
constexpr int kLdT = 40;          // 引擎瓦片的行距（80 B，16 B 对齐）
constexpr int kThreads = 256;     // 8 warps × 32

// ---- 与 ds4_iq.dp.cpp:110-127 等价 ----
inline float h2f(const uint8_t* p) {
    const unsigned short u = (unsigned short)(p[0] | (p[1] << 8));
    return (float)sycl::bit_cast<sycl::half, unsigned short>(u);
}
inline uint32_t ksigns8(uint32_t v) {
    v &= 127u;
    return v | ((uint32_t)(sycl::popcount(v) & 1u) << 7);
}
inline int sgn4(uint32_t g, uint32_t b) {
    const uint32_t lo = (((b & 15u) * 0x10204080u) & 0x80808080u) >> 7;
    return (int)((g ^ (lo * 255u)) + lo);
}
inline uint32_t load_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// 一块（256 值）→ 256 个 int8 码。逐 ib / 逐 l，与引擎 :208 同构。
inline void dequant_one(const uint8_t* b, const uint32_t* tab, int8_t* out) {
    (void)h2f(b);
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const uint8_t* q = b + 2 + 8 * ib;
        const uint32_t a0 = load_u32(q), a1 = load_u32(q + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = (a0 >> (8 * l)) & 255u;
            const uint32_t gx = tab[(size_t)idx * 2], gy = tab[(size_t)idx * 2 + 1];
            const uint32_t sg = ksigns8(a1 >> (7 * l));
            const int v0 = sgn4(gx, sg), v1 = sgn4(gy, sg >> 4);
            int8_t* o = out + (size_t)ib * 32 + (size_t)l * 8;
#pragma unroll
            for (int k = 0; k < 4; ++k) o[k]     = (int8_t)((v0 >> (8 * k)) & 0xff);
#pragma unroll
            for (int k = 0; k < 4; ++k) o[4 + k] = (int8_t)((v1 >> (8 * k)) & 0xff);
        }
    }
}

// ---- 与 ds4_iq.dp.cpp:449 deq8<kIQ2_XXS> 逐行同构 ----
// values [8p, 8p+8) of sub-block kb，作为 f32
inline void deq8_iq2(const uint8_t* row, int kb, int p, const uint32_t* tab, float* w) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t)(kb >> 3) * kBlockBytes;
    const uint8_t* q = b + 2 + 8 * ib;
    const uint32_t a1 = load_u32(q + 4);
    const uint32_t idx = q[p];
    const uint32_t gx = tab[(size_t)idx * 2], gy = tab[(size_t)idx * 2 + 1];
    const uint32_t v = (a1 >> (7 * p)) & 127u;
    const uint32_t sg = v | ((uint32_t)(sycl::popcount(v) & 1u) << 7);
    const float dl = h2f(b) * (float)(2 * (int)(a1 >> 28) + 1) * 0.125f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const uint32_t gr = j < 4 ? gx : gy;
        const float gv = (float)((gr >> (8 * (j & 3))) & 255u);
        w[j] = (sg >> j) & 1u ? -(dl * gv) : dl * gv;
    }
}

// ---- SIMD 版：与 deq8 逐位等价，但 8 个值一次算完 ----
// 参考实现（:449 deq8）每值做：字节抽取 (shr+and) → 符号取位 (shr+and) → select → mul，8 值 = 约 40 条标量指令。
// 这里改成：uchar8 一次转 float8 → 一次乘 dl → 一次乘 ±1 掩码。
// 逐位等价的原因：每 lane 仍是 (dl*gv) 的一次 IEEE 乘，再乘 ±1（乘以 ±1 是精确的，不会改变尾数）。
inline void deq8_iq2_vec(const uint8_t* row, int kb, int p, const uint32_t* tab, float* w) {
    const int ib = kb & 7;
    const uint8_t* b = row + (size_t)(kb >> 3) * kBlockBytes;
    const uint8_t* q = b + 2 + 8 * ib;
    const uint32_t a1 = load_u32(q + 4);
    const uint32_t idx = q[p];
    const uint32_t gx = tab[(size_t)idx * 2], gy = tab[(size_t)idx * 2 + 1];
    const uint32_t v = (a1 >> (7 * p)) & 127u;
    const uint32_t sg = v | ((uint32_t)(sycl::popcount(v) & 1u) << 7);
    const float dl = h2f(b) * (float)(2 * (int)(a1 >> 28) + 1) * 0.125f;

    // 8 个 grid 码（每字节 0..63）→ float8（一次向量转换）
    const sycl::uchar8 g8 = {(uint8_t)gx, (uint8_t)(gx >> 8), (uint8_t)(gx >> 16), (uint8_t)(gx >> 24),
                             (uint8_t)gy, (uint8_t)(gy >> 8), (uint8_t)(gy >> 16), (uint8_t)(gy >> 24)};
    const sycl::vec<float, 8> gv = g8.convert<float, sycl::rounding_mode::rte>();
    // 符号掩码 ±1（一次 select，没有逐值分支）
    sycl::vec<float, 8> s;
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = ((sg >> j) & 1u) ? -1.0f : 1.0f;
    const sycl::vec<float, 8> r = (gv * sycl::vec<float, 8>(dl)) * s;
#pragma unroll
    for (int j = 0; j < 8; ++j) w[j] = r[j];
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }
};

// 单块 kernel
struct KOne {
    const uint8_t* w; const uint32_t* tab; int8_t* out; int n;
    void operator()(sycl::id<1> i) const {
        const int b = (int)i;
        if (b >= n) return;
        dequant_one(w + (size_t)b * kBlockBytes, tab, out + (size_t)b * kQK);
    }
};

// 批处理 kernel：一个 work-item 处理 B 块；表搬进 local memory 一次
template <int B>
struct KBatch {
    const uint8_t* w; const uint32_t* tab; int8_t* out; int n;
    void operator()(sycl::nd_item<1> it) const {
        auto& ltab = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t[2 * 256]>(
            it.get_group());
        for (int i = (int)it.get_local_id(0); i < 512; i += (int)it.get_local_range(0))
            ltab[i] = tab[i];
        it.barrier(sycl::access::fence_space::local_space);
        const int base = (int)it.get_global_id(0) * B;
        const uint32_t* t = (const uint32_t*)ltab;
#pragma unroll
        for (int k = 0; k < B; ++k) {
            const int b = base + k;
            if (b >= n) return;
            dequant_one(w + (size_t)b * kBlockBytes, t, out + (size_t)b * kQK);
        }
    }
};

// 引擎真正的原语：一个 work-item = 一个 (row, kb, p) 的 8 个值 → f32（引擎里写到寄存器/SLM；
// 这里只写回 1 个 float/work-item，把写出流量压到 1/8，免得计时被 HBM 拖住）
struct KDeq8 {
    const uint8_t* w; const uint32_t* tab; float* out; int n_rows; size_t rb; int ksteps; int n;
    void operator()(sycl::id<1> i) const {
        const int s = (int)i;
        if (s >= n) return;
        const int p = s & 3, kb = (s >> 2) % ksteps, row = (s >> 2) / ksteps;
        float w8[8];
        deq8_iq2(w + (size_t)(row % n_rows) * rb, kb, p, tab, w8);
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += w8[j];
        out[s] = acc;
    }
};

// SIMD 版的同样切片（每 work-item 8 个值），用来和 KDeq8 直接对比吞吐
struct KDeq8Vec {
    const uint8_t* w; const uint32_t* tab; float* out; int n_rows; size_t rb; int ksteps; int n;
    void operator()(sycl::id<1> i) const {
        const int s = (int)i;
        if (s >= n) return;
        const int p = s & 3, kb = (s >> 2) % ksteps, row = (s >> 2) / ksteps;
        float w8[8];
        deq8_iq2_vec(w + (size_t)(row % n_rows) * rb, kb, p, tab, w8);
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += w8[j];
        out[s] = acc;
    }
};

// 瓦片构建：set_tc_kernel 的内层，去掉 MMA（tr = 权重行, p = 该 32 组里的第 p 个 8 值）
//   BM_  行瓦片高度 —— 也就是"一个解出来的权重被多少个激活行复用"，直接乘在隐含 TFLOPS 上
//   VEC  true 时用 16 B 向量存（引擎现在是逐 half 标量存），量化这一处值多少钱
//   SIMD true 时用 deq8_iq2_vec（8 值向量化）而不是标量 deq8
// 线程数 = BM_ × 4（每行 4 个线程，每个线程 8 个 K 值）；一个 workgroup 建一张 BM_×32 的 half 瓦片。
template <int BM_, bool VEC, bool SIMD>
struct KTile {
    const uint8_t* w; const uint32_t* tab; int n_rows; size_t rb; int ksteps;
    sycl::half* vout;      // kb == 0 时把瓦片写出去做校验（计时影响可忽略）
    float* sink;
    void operator()(sycl::nd_item<1> it) const {
        auto& tile = *sycl::ext::oneapi::group_local_memory_for_overwrite<sycl::half[BM_ * kLdT]>(it.get_group());
        const int tid = (int)it.get_local_id(0), tr = tid >> 2, p = tid & 3;
        const int g = (int)it.get_group(0);
        const uint8_t* row = w + (size_t)((g * BM_ + tr) % n_rows) * rb;
        float acc = 0.0f;
        for (int kb = 0; kb < ksteps; ++kb) {
            float w8[8];
            if (SIMD) deq8_iq2_vec(row, kb, p, tab, w8);
            else      deq8_iq2(row, kb, p, tab, w8);
            sycl::half h8[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) h8[j] = sycl::half(w8[j]);
            if (VEC) {
                *reinterpret_cast<sycl::half8*>(&tile[tr * kLdT + p * 8]) =
                    *reinterpret_cast<const sycl::half8*>(h8);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) tile[tr * kLdT + p * 8 + j] = h8[j];   // 引擎当前写法
            }
            it.barrier(sycl::access::fence_space::local_space);
            acc += (float)tile[(tid * 7) & (BM_ * kLdT - 1)];                      // 消费一次，防止被优化掉
            if (kb == 0 && vout != nullptr) {
                for (int j = 0; j < 8; ++j) vout[(size_t)g * BM_ * 32 + tr * 32 + p * 8 + j] = h8[j];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
        if (sink && tid == 0) sink[g] = acc;
    }
};

}  // namespace

int main(int argc, char** argv) {
    int blocks = argc > 1 ? std::atoi(argv[1]) : 131072;   // 131072 块 = 33.6 M 值 ≈ 8.7 MB
    int reps   = argc > 2 ? std::atoi(argv[2]) : 20;
    const int BATCH  = argc > 3 ? std::atoi(argv[3]) : 4;
    const int KSTEPS = argc > 4 ? std::atoi(argv[4]) : 80;   // 80 × 32 = K 2560（n_embd）
    const int GROUPS = argc > 5 ? std::atoi(argv[5]) : 128;

    std::printf("=== iq_dequant_bench（专家内核 ALU 墙）===\n");
    std::printf("blocks %d (%.1f M values, %.1f MB), reps %d, batch %d, ksteps %d, groups %d\n",
                blocks, (double)blocks * kQK / 1e6, (double)blocks * kBlockBytes / 1e6, reps, BATCH,
                KSTEPS, GROUPS);
    std::printf("模型：每个解出的权重被 BM=%d 行复用 → 隐含 TFLOPS = Gval/s × 2 × %d / 1000 = Gval/s × %.3f\n\n",
                kBM, kBM, 2.0 * kBM / 1000.0);

    std::vector<uint32_t> tab((size_t)kGrid, 0u);
    Rng r(12345);
    for (int i = 0; i < 2 * 256; ++i) tab[(size_t)i] = r.next() & 0x3f3f3f3fu;   // 每字节 0..63
    std::vector<uint8_t> w((size_t)blocks * kBlockBytes);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)(r.next() & 0xff);
    // 块头的 d 是 half 尺度：随机字节会造出 NaN half（NaN 的符号位取决于操作数顺序，
    // 会让"逐位一致"的判定变成噪声）。真实权重里 d 是有限值，这里统一给 1.0（0x3C00）。
    for (int b = 0; b < blocks; ++b) { w[(size_t)b * kBlockBytes] = 0x00; w[(size_t)b * kBlockBytes + 1] = 0x3C; }

    // ---- CPU 参考 & 吞吐 ----
    std::vector<int8_t> ref((size_t)blocks * kQK, 0), got((size_t)blocks * kQK, 0);
    std::vector<float> fref((size_t)blocks * kQK, 0.f), fgot((size_t)blocks * kQK, 0.f);
    std::vector<float> fvec((size_t)blocks * kQK, 0.f);
    for (int b = 0; b < blocks; ++b)
        dequant_one(w.data() + (size_t)b * kBlockBytes, tab.data(), ref.data() + (size_t)b * kQK);
    // CPU 侧只算一小片（1.3 Gval/s，全量要好几百秒）；GPU 侧仍跑全量
    const int krows_cpu = std::max(1, std::min(blocks / 10, 512));
    const size_t rb0 = (size_t)(KSTEPS * 32) / 256 * kBlockBytes;   // 一行（K = KSTEPS*32）的字节数
    for (int row = 0; row < krows_cpu; ++row)
        for (int s = 0; s < KSTEPS * 4; ++s)
            deq8_iq2(w.data() + (size_t)row * 10 * kBlockBytes, s >> 2, s & 3, tab.data(),
                     fref.data() + ((size_t)row * KSTEPS * 4 + s) * 8);
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; ++rep)
            for (int b = 0; b < blocks; ++b)
                dequant_one(w.data() + (size_t)b * kBlockBytes, tab.data(), ref.data() + (size_t)b * kQK);
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double vals = (double)reps * blocks * kQK;
        std::printf("CPU ref : %9.1f ms  %7.2f Gval/s  %7.2f GB/s(in)\n", ms,
                    vals / (ms * 1e6), (double)reps * blocks * kBlockBytes / (ms * 1e6));
    }
    {
        const int creps = std::min(reps, 3);
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < creps; ++rep)
            for (int row = 0; row < krows_cpu; ++row)
                for (int s = 0; s < KSTEPS * 4; ++s)
                    deq8_iq2(w.data() + (size_t)row * 10 * kBlockBytes, s >> 2, s & 3, tab.data(),
                             fref.data() + ((size_t)row * KSTEPS * 4 + s) * 8);
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double vals = (double)creps * krows_cpu * KSTEPS * 32;
        std::printf("CPU deq8: %9.1f ms  %7.2f Gval/s\n", ms, vals / (ms * 1e6));
    }
    // ---- 逐位校验：SIMD 版 vs 标量版（目标 ③ 的验收门槛 max err = 0）----
    for (int row = 0; row < krows_cpu; ++row)
        for (int s = 0; s < KSTEPS * 4; ++s)
            deq8_iq2_vec(w.data() + (size_t)row * 10 * kBlockBytes, s >> 2, s & 3, tab.data(),
                         fvec.data() + ((size_t)row * KSTEPS * 4 + s) * 8);
    {
        const size_t n = (size_t)krows_cpu * KSTEPS * 4 * 8;
        size_t ndiff = 0;
        double maxabs = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::memcmp(&fref[i], &fvec[i], sizeof(float)) != 0) {
                ++ndiff;
                maxabs = std::max(maxabs, (double)std::fabs((double)fref[i] - (double)fvec[i]));
            }
        }
        std::printf("SIMD vs 标量（CPU，%zu 值）: %zu 处不同，max |err| = %g %s\n", n, ndiff, maxabs,
                    ndiff == 0 ? "(逐位一致)" : "<-- 不一致");
        if (ndiff && std::getenv("IQ_BENCH_DEBUG")) {   // 只印前 8 处，带 bit pattern
            int shown = 0;
            for (size_t i = 0; i < n && shown < 8; ++i) {
                if (std::memcmp(&fref[i], &fvec[i], sizeof(float)) == 0) continue;
                uint32_t a, b2;
                std::memcpy(&a, &fref[i], 4);
                std::memcpy(&b2, &fvec[i], 4);
                std::printf("    [%zu] 标量 %.9g (0x%08x)  SIMD %.9g (0x%08x)\n", i, (double)fref[i], a,
                            (double)fvec[i], b2);
                ++shown;
            }
        }
        std::printf("  → %s\n", ndiff == 0 ? "max err = 0，SIMD 版可以替换现有实现" : "有差异，必须查清再谈收益");
    }
    // ---- 瓦片落位的逐位校验（CPU 模拟一个 workgroup）：引擎现在的逐 half 标量存 vs 16 B 向量存 ----
    // 目的：让早上"16B 向量存快多少"只可能是性能问题，不可能是正确性问题。
    {
        const int threads = kBM * 4, steps = 8;
        std::vector<sycl::half> t_sc((size_t)kBM * kLdT, sycl::half(0.f)), t_vc((size_t)kBM * kLdT, sycl::half(0.f));
        size_t bad = 0;
        const int nrows = std::max(1, blocks / 10);
        for (int kb = 0; kb < steps; ++kb) {
            for (int tid = 0; tid < threads; ++tid) {
                const int tr = tid >> 2, p = tid & 3;
                float w8[8];
                deq8_iq2(w.data() + (size_t)(tr % nrows) * rb0, kb, p, tab.data(), w8);
                sycl::half h8[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) h8[j] = sycl::half(w8[j]);
#pragma unroll
                for (int j = 0; j < 8; ++j) t_sc[(size_t)tr * kLdT + p * 8 + j] = h8[j];   // 引擎写法
                *reinterpret_cast<sycl::half8*>(&t_vc[(size_t)tr * kLdT + p * 8]) =
                    *reinterpret_cast<const sycl::half8*>(h8);                              // 向量写法
            }
            bad += std::memcmp(t_sc.data(), t_vc.data(), t_sc.size() * sizeof(sycl::half)) != 0 ? 1 : 0;
        }
        std::printf("瓦片落位（CPU 模拟 %d 线程 × %d 步，%zu half）: 标量存 vs 16B向量存 → %s\n", threads, steps,
                    t_sc.size(), bad == 0 ? "逐位一致" : "不一致！");
    }
    {
        const int creps = std::min(reps, 3);
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < creps; ++rep)
            for (int row = 0; row < krows_cpu; ++row)
                for (int s = 0; s < KSTEPS * 4; ++s)
                    deq8_iq2_vec(w.data() + (size_t)row * 10 * kBlockBytes, s >> 2, s & 3, tab.data(),
                                 fvec.data() + ((size_t)row * KSTEPS * 4 + s) * 8);
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double vals = (double)creps * krows_cpu * KSTEPS * 32;
        std::printf("CPU SIMD: %9.1f ms  %7.2f Gval/s（本机 CPU 无自动向量化收益是正常的，看 GPU 那一行）\n", ms,
                    vals / (ms * 1e6));
    }

    // ---- GPU（IQ_BENCH_CPU_ONLY=1 时跳过：夜间只用 CPU 验证逻辑）----
    if (std::getenv("IQ_BENCH_CPU_ONLY")) { std::printf("CPU-only mode: GPU part skipped\n"); return 0; }
    try {
        sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
        std::printf("\ndevice  : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        const size_t rb = (size_t)(KSTEPS * 32) / 256 * kBlockBytes;   // 一行 K=KSTEPS*32 的字节数
        uint32_t* d_tab = sycl::malloc_device<uint32_t>((size_t)kGrid, q);
        uint8_t*  d_w   = sycl::malloc_device<uint8_t>((size_t)blocks * kBlockBytes, q);
        int8_t*   d_out = sycl::malloc_device<int8_t>((size_t)blocks * kQK, q);
        const int n8 = blocks * kQK / 8;
        float*    d_fout = sycl::malloc_device<float>((size_t)n8, q);
        sycl::half* d_vout = sycl::malloc_device<sycl::half>((size_t)GROUPS * kBM * 32, q);
        float*    d_sink = sycl::malloc_device<float>((size_t)GROUPS, q);
        q.memcpy(d_tab, tab.data(), (size_t)kGrid * sizeof(uint32_t));
        q.memcpy(d_w, w.data(), w.size());
        q.wait();

        auto report = [&](const char* name, double gval) {
            std::printf("%-9s %8.2f Gval/s  %7.2f GB/s(in)  隐含 %7.2f TFLOPS\n", name, gval,
                        gval * kBlockBytes / kQK, gval * 2.0 * kBM / 1000.0);
        };
        auto timeit = [&](double vals, auto&& launch) {
            launch(); q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < reps; ++rep) launch();
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return vals / (ms * 1e6);
        };
        const double vals_blk = (double)reps * blocks * kQK;

        const double a = timeit(vals_blk, [&] { q.parallel_for(sycl::range<1>((size_t)blocks), KOne{d_w, d_tab, d_out, blocks}); });
        report("GPU ref", a);

        double bat = 0.0;
        const size_t gb = 128, nb = ((size_t)(blocks / BATCH) + gb - 1) / gb;
        switch (BATCH) {
            case 2: bat = timeit(vals_blk, [&]{ q.parallel_for(sycl::nd_range<1>(nb*gb, gb), KBatch<2>{d_w, d_tab, d_out, blocks}); }); break;
            case 4: bat = timeit(vals_blk, [&]{ q.parallel_for(sycl::nd_range<1>(nb*gb, gb), KBatch<4>{d_w, d_tab, d_out, blocks}); }); break;
            case 8: bat = timeit(vals_blk, [&]{ q.parallel_for(sycl::nd_range<1>(nb*gb, gb), KBatch<8>{d_w, d_tab, d_out, blocks}); }); break;
            default: std::printf("batch must be 2/4/8\n"); return 2;
        }
        char nm[32];
        std::snprintf(nm, sizeof nm, "GPU bat%d", BATCH);
        report(nm, bat);
        std::printf("          batch speedup vs ref: %.2fx\n", bat / a);

        // deq8：引擎 :449 的原语，每 work-item 8 个值
        const double vals8 = (double)reps * n8 * 8;
        const double d8 = timeit(vals8, [&] {
            q.parallel_for(sycl::range<1>((size_t)n8), KDeq8{d_w, d_tab, d_fout, blocks / 10, rb, KSTEPS, n8});
        });
        report("GPU deq8", d8);
        const double d8v = timeit(vals8, [&] {
            q.parallel_for(sycl::range<1>((size_t)n8), KDeq8Vec{d_w, d_tab, d_fout, blocks / 10, rb, KSTEPS, n8});
        });
        report("GPU deq8v", d8v);
        std::printf("          SIMD 版 vs 标量版: %.2fx\n", d8v / d8);

        // 瓦片构建 = deq8 + f16 转换 + local 存 + barrier（去掉 MMA）
        // 三件事分开量：BM（权重复用倍数）、存法（标量 vs 16B 向量）、解量化（标量 vs SIMD）
        {
            auto tile_run = [&](auto tag, int bm, int threads, auto&& launch) {
                const double vals_tile = (double)reps * GROUPS * KSTEPS * bm * kKS;
                const double g = timeit(vals_tile, launch);
                char nm[48];
                std::snprintf(nm, sizeof nm, "tile BM%-3d %-8s", bm, tag);
                std::printf("%-16s %8.2f Gval/s  隐含 %7.2f TFLOPS\n", nm, g, g * 2.0 * bm / 1000.0);
                return g;
            };
            const int g32 = GROUPS, g64 = GROUPS / 2 > 0 ? GROUPS / 2 : 1, g128 = GROUPS / 4 > 0 ? GROUPS / 4 : 1;
            const double s64 = tile_run("标量", 64, 256, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g64 * 256, 256),
                               KTile<64, false, false>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
            const double v64 = tile_run("16B向量", 64, 256, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g64 * 256, 256),
                               KTile<64, true, false>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
            const double vs64 = tile_run("向量+SIMD", 64, 256, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g64 * 256, 256),
                               KTile<64, true, true>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
            std::printf("          16B 向量存 vs 标量存: %.2fx   SIMD 解量化再加: %.2fx\n", v64 / s64, vs64 / v64);
            tile_run("标量", 32, 128, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g32 * 128, 128),
                               KTile<32, false, false>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
            tile_run("向量+SIMD", 32, 128, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g32 * 128, 128),
                               KTile<32, true, true>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
            tile_run("向量+SIMD", 128, 512, [&] {
                q.parallel_for(sycl::nd_range<1>((size_t)g128 * 512, 512),
                               KTile<128, true, true>{d_w, d_tab, blocks / 10, rb, KSTEPS, d_vout, d_sink});
            });
        }

        // ---- 校验 ----
        q.memcpy(fgot.data(), d_fout, (size_t)n8 * sizeof(float)).wait();
        const size_t nchk = (size_t)krows_cpu * KSTEPS * 4;      // 与 CPU 的 fref 一一对应
        double maxre = 0; size_t ndiff = 0;
        for (size_t i = 0; i < nchk; ++i) {
            double want = 0;
            for (int j = 0; j < 8; ++j) want += (double)fref[i * 8 + j];
            const double d = std::fabs(want - (double)fgot[i]);
            const double rel = d / (1e-6 + std::fabs(want));
            if (d != 0) { ++ndiff; if (rel > maxre) maxre = rel; }
        }
        std::printf("deq8 累加 GPU vs CPU: %zu of %zu differ, max rel err %.3g %s\n", ndiff, nchk, maxre,
                    maxre < 1e-6 ? "(OK)" : "<-- CHECK");
        q.memcpy(got.data(), d_out, (size_t)blocks * kQK).wait();
        size_t diff = 0; int maxe = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            const int e = std::abs((int)ref[i] - (int)got[i]);
            if (e) { ++diff; if (e > maxe) maxe = e; }
        }
        std::printf("int 码 GPU vs CPU: %zu of %zu differ, max |err| = %d %s\n", diff, ref.size(), maxe,
                    diff ? "<-- MISMATCH" : "(OK)");
        sycl::free(d_tab, q); sycl::free(d_w, q); sycl::free(d_out, q); sycl::free(d_fout, q);
        sycl::free(d_vout, q); sycl::free(d_sink, q);
        return (diff || maxre >= 1e-6) ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("no GPU run: %s\n", e.what());
        return 0;
    }
}
