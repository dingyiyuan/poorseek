# DeepInfer MoE 在小 m（每专家 ~1 token）下的实现：逐条回答我们的 6 个问题

**读的版本**

- `DI:` = [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) @ `a679bb86`（2026-11-03）
- `DG:` = [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) @ `1fc9c669`，即 DeepInfer 的 `vendor/DeepGEMM` 子模块（`.gitmodules` 指向该仓库）。DeepInfer 自己**没有**一行 MoE GEMM，全部委托给 DeepGEMM 的 `mega` 系列 / `m_grouped_*_contiguous`。

**一个必须先说清的事实**：DeepInfer 的 decode MoE 主路径不是 "grouped GEMM + 单独 dispatch"，而是一个**融合 kernel** `fp8_fp4_mega_moe` / `bf16_mega_moe`：gate 之后的 topk 分发、按专家归并、L1（gate/up）+ SwiGLU + 量化、L2（down）、按 token 加权合并、跨 rank 的 NVLink 收发、以及 shared expert，全在**一次 launch** 里（`DI:deepinfer/model_executor/models/v4_modules/mlp.py:254-258`、`:528-563`）。他们的官方口径是：*"Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE"*（`DI:doc/inference_deployment_qa.md:283-285`）。所以下面 6 个问题的答案，绝大多数要在 `DG:csrc/**/mega_moe*` 和 `DG:deep_gemm/include/deep_gemm/{scheduler,impls,layout}/mega_moe.cuh` 里找。

---

## Q1. 每专家 ~1.2 个 token 时到底怎么算？pad 到固定 block_m 还是按真实 token 数？

**是按真实 token 数切块，不是 pad 到固定块数；但最后一块内部会 pad 到 16 行参与 MMA。**

### 1) 每专家切几块：`ceil(m_e / block_m)`，不是固定块数

```cpp
// DG:deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:272-281
CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
    uint32_t num_blocks = 0;
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
        if (i * 32 + ptx::get_lane_idx() < expert_idx)
            num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
    }
```
每个专家的 pool block 数 = `ceil(num_tokens_e / BLOCK_M)`，全局 pool block 总数 = Σ 各专家（`DG:.../scheduler/mega_moe.cuh:282-284`）。**没有任何 "pad 到固定 block 数" 的动作** —— 1 个 token 的专家就是 1 块，10 个 token 的专家也是 1 块。

### 2) 决定 `block_m` 的那几行（这就是他们文档里 "≤10 → 16" 的原始出处）

```cpp
// DG:csrc/jit_kernels/heuristics/mega_moe.hpp:93-108
// Expected tokens per expert, plus a one-sigma routing margin for the tile choice
const float num_expected_tokens = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);

// Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
// per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
// A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
int block_m = num_expected_tokens <= 10 ? 16 : 32;
if (num_expected_tokens > 24) {
    int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
    if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
        num_blocks = 2;
    for (const int& candidate: {64, 128, 192, 240}) {
        block_m = candidate;
        if (num_blocks * candidate >= num_covered_tokens)
            break;
    }
}
```

要点：

- `num_expected_tokens` 用的是**全局每专家期望 token 数** = `num_tokens(本 rank) × num_ranks × topk / num_experts`（`DG:.../heuristics/mega_moe.hpp:94`）。`num_tokens` 是**本次 forward 的真实行数** `y.size(0)`，不是 `max_num_tokens_per_rank`（`DG:csrc/apis/mega_moe.hpp:175`）；后者只用于 buffer 容量与 ring 大小（`DG:csrc/apis/mega_moe.hpp:79-85`）。
- 代入我们的场景（单卡、1 token、384 专家、top-6、无 EP）：`1×1×6/384 = 0.016 ≤ 10` → **`block_m = 16`**。MTP 窗口 36 个 (token,expert) 对 / ~30 个专家 → ~1.2 ≤ 10 → 仍是 **16**。
- 代入他们生产环境（Wide-EP64/DP64、每 rank ~60 seq、top-6、384 专家）：`60×64×6/384 = 60 > 24` → 走 `{64,128,192,240}` 分支得 **64**。**也就是说他们线上根本不在"每专家 1.2 token"这个区制里，我们要抄的是他们 `block_m=16` 那一档的行为。**
- 候选集合有 8：`{8, 16, 32, 64, 96, 128, 192, 240}`（`DG:deep_gemm/include/deep_gemm/layout/mega_moe.cuh:14`），但 heuristic 最小只到 16，**8 永远不会被选中**。
- `block_m` 是**编译期模板参数**，进 JIT kernel 源码（`DG:csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:178`：`config.block_m` 作为 `sm100_fp8_fp4_mega_moe_impl<...>` 的模板实参）。所以 batch 变了会编译出不同的 kernel 实例（有 cache），不是纯运行时参数。

### 3) 那块只装了 1 行的 block，15 个空行会不会被真算进 GEMM？

**会算，但只体现在 UMMA 的 N 维上 —— 因为他们把 GEMM 的 A/B 互换了：权重进 M 维，token 进 N 维。**

```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:161-162
constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;   // = 2*128 = 256，这是权重（N）方向
constexpr uint32_t UMMA_N = BLOCK_M;           // Swap AB  ← token 方向
```

```cpp
// DG:deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:127-129
CUTLASS_DEVICE
uint32_t get_umma_aligned_valid_m() const {
    return math::align(valid_m, 16u);          // valid_m=1 → 16
}
```

```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:858-860
// Dynamic update of UMMA N based on effective M
auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());
// DG:deep_gemm/include/deep_gemm/mma/sm100.cuh:150-153
//   desc.n_dim_ = umma_n >> 3;   → 16>>3 = 2，即 N=16 的 MMA
```

所以：**`valid_m = 1` 时真的发一条 N=16 的 MMA 指令，那 15 列（我们说"行"）确实被算掉了。** 但因为 N 的最小粒度本来就是 16（`DG:.../impls/sm100_fp8_fp4_mega_moe.cuh:847-850` 的断言：`UMMA_N % 16 == 0 and 16 <= UMMA_N`），**N=1 的 MMA 在这个硬件上根本不存在**，pad 到 16 不额外付钱。

与此同时，**写回严格按 `valid_m` 截断**，pad 行结果丢弃：

```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1332-1334
// Skip padding rows beyond the actual token count for this expert
if (m_idx_in_block >= valid_m)
    break;
```

而且 pad 行**根本没被写数据**，只是被"记账"成已到达：

```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:611-616
issue_and_wait_pull_store(kNumChunks - 1);
const bool is_last_token = (token_idx == expert_end_idx - 1);
ptx::red_add_rel(
    workspace.get_l1_full_count_ptr(pool_block_idx % kNumRingBlocks),
    is_last_token ? BLOCK_M - (token_idx_in_expert % BLOCK_M) : 1u
);
```
（symm buffer 在分配时整块清零，`DG:deep_gemm/mega/__init__.py:58` `self.buffer.zero_()`，所以 pad 行初值是 0；跨 ring 轮次留下的脏值也不会被读回，因为结果按 `valid_m` 丢弃。）

### 4) 他们真正的成本模型：不是"空行"，是"多一个 m block = 权重重读一遍"

```cpp
// DG:csrc/jit_kernels/heuristics/mega_moe.hpp:97
// Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
```
这句话是整个设计的钥匙：**pad 行不花钱，多切一个 m block 才花钱**。所以他们宁可选一个大 tile 装下尽量多 token（1 块最多装 240 行），把每专家的 block 数压到 1。对我们的直接推论见文末。

---

## Q2. 验证窗口内按专家去重吗？一个 expert blob 一轮读一次，还是每个 (token,expert) 对读一次？

**按专家去重，窗口内一个专家的权重只读 `ceil(m_e/block_m)` 次 —— `m_e ≤ block_m` 时就是 1 次。绝不按 (token,expert) 对读。**

三个环节各自给出证据：

1. **分发阶段按专家归并**：kernel 内每个 SM 认领一部分专家的接收桶，`(token,expert)` 对被打到该专家在 ring 里的连续 slot 上：
```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:642-651
} else {
    // Other SMs: clean blocks
    for (uint32_t i = sm_idx - 1; i < kNumExpertsPerRank; i += kNumSMs - 1) {
        const auto num_recv_tokens = static_cast<uint32_t>(*workspace.get_expert_recv_count_sum_ptr(i));
        const auto num_recv_m_blocks = math::ceil_div(num_recv_tokens, BLOCK_M);
        expert_pool_block_offset = scheduler.get_pool_block_offset(i);
```
```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:384-390  每个 (token,expert) 对写一条 src 索引
read_topk_idx([&](const uint32_t& token_topk_idx, const int& expert_idx) {
    const auto dst_rank_idx = expert_idx / kNumExpertsPerRank;
    const auto dst_slot_idx = atomicAdd_block(shared_storage.expert_token_count + expert_idx, 1);
    const auto dst_ptr = workspace.get_src_token_topk_idx_ptr(expert_idx % kNumExpertsPerRank, sym_buffer.rank_idx, dst_slot_idx);
    *sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx;
});
```

2. **GEMM 阶段以 "pool block"（= 某专家的某 m 块）为最小任务**，一次任务把该专家的权重沿 K 流一遍：
```cpp
// DG:deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:326-337
const uint32_t num_tokens = stored_num_tokens_per_expert[i];
const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
...
const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
result.m_block_idx     = ptx::exchange(owner_m_block_idx, owner_lane_idx);
result.valid_m         = ptx::exchange(owner_valid_m, owner_lane_idx);
```
```cpp
// DG:deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:176-179  权重 tile 的 N 方向再切 cluster
// L1: kNumL1Clusters = kNumL1BlockNs / 2 = L1_SHAPE_N / BLOCK_N / 2   （L1_SHAPE_N = 2*moe_inter）
```

3. **权重只从 HBM 走一次的依据**：每个 task 里由一个专用 warp 沿 K 逐块 TMA 权重进 `smem_b`（`DG:.../impls/sm100_fp8_fp4_mega_moe.cuh:760-805`，`for (k_block_idx ...)` 在 `:781`，N 方向只有 `half_n_idx` 一片 128 宽），进 smem 后只服务本 pool block 的那些行；代码里不存在"按 (token,expert) 对重读权重"的路径（对比我们 `gg::win_experts` 每个 (token,expert) 对都把码本和权重走一遍）。

**对照：没开 mega 时（非 NVL 域、或 Ascend 之类）** 用 DeepEP dispatch + `m_grouped_*_nt_contiguous(use_psum_layout=True)`（`DI:.../mlp.py:109-137`、`:477`、`:593-598`）。psum 布局 = 每专家的 token 段在 M 轴上是连续一段，同样是"一专家一段、一次读"。但这条路会把每专家的段按 `mk_alignment` **对齐到 256 行**（`DG:csrc/jit_kernels/heuristics/runtime.hpp:12,18,57-63` + `DI:.../mlp.py:250-252`），小 m 下很浪费 —— 这也解释了他们的官方建议（见 Q6）。

---

## Q3. draft（单 token）和 verify（k+1 token）用同一套 grouped GEMM 吗？

**是同一套 kernel、同一个 module，甚至同一份 JIT 源码；差别只有 `num_tokens`（进而 `block_m`）和 shared expert 是否融合。没有"verify 专用 kernel"。**

1. **draft 层和主模型层是同一个 MLP 类、同一个 forward**：
```python
# DI:deepinfer/model_executor/models/decoder_layer.py:34-41   DFlashDecoderLayer 与 DeepseekDecoderLayer 都建 DeepseekMoE
        mlp = DeepseekMoE(layer.experts, layer_idx, cfg)
# DI:.../decoder_layer.py:169-173   主模型（verify 在这里走）
        hidden_states = self.mlp(hidden_states, hidden_states_bf16, input_metadata)
# DI:.../decoder_layer.py:238       DFlash draft 层（draft 在这里走）
        draft_hiddens = self.mlp(draft_hiddens, hidden_states_bf16, input_metadata)
```
2. **`DeepseekMoE.forward` 里唯一的分叉是 use_mega_moe，不区分 prompt/decode/draft/verify**：
```python
# DI:deepinfer/model_executor/models/v4_modules/mlp.py:565-572
    def forward(self, hidden_states, hidden_states_bf16, input_metadata) -> torch.Tensor:
        if self.use_mega_moe:
            return self.mega_moe_forward(hidden_states, hidden_states_bf16, input_metadata)
```
3. **区分参数就是 `num_tokens`**：它决定 `num_expected_tokens` → `block_m` → 一整套编译期配置（`DG:.../heuristics/mega_moe.hpp:94-108`，`DG:.../impls/sm100_fp8_fp4_mega_moe.hpp:174-189`）。grid 完全不变，永远 `dim3(num_sms,1,1)` + `cluster_dim=(2,1,1)`（`DG:.../impls/sm100_fp8_fp4_mega_moe.hpp:195-197`）。
4. **两边一次 forward 的 token 数都是"好几行"，不是 1 行**：
   - draft：`real_cnt × dflash_block_size`（= 每请求 5 个 draft 位一拍算完），见 `DI:deepinfer/model_executor/input_metadata/dflash.py:36-80`（`dflash_newx_slot_mapping_store` 形状 `[real_cnt * dflash_block_size]`，`:30-31`）与 `DI:.../models/mtp/dflash.py` 的 `_embed_draft_init_tokens`（`[req_cnt, dflash_block_size]` 展平）。
   - verify：变长验证，一次 forward 吃 所有 seq 的 `accepted+1` 行（`DI:doc/inference_deployment_qa.md:307`：DSpark 动态长度验证，每步平均验证 3.2–4.4 个 token）。
5. **如果没开 mega**：两条路都落到同一个 `m_grouped_fp8_fp4_gemm_nt_contiguous(use_psum_layout=True, expected_m_for_psum_layout=None)`（`DI:.../mlp.py:119-128`、`:593-598`）——`expected_m` 在 decode 路径是 `None`（`:597`），也就是说 M 轴配置按实际 `m` 走，不做"按最大窗口预留"。

---

## Q4.（最关键）他们的专家权重是 FP8/E4M3（乘 scale 即解量化）、没有码本查表；那他们**有没有任何**"每组权重只解一次、多个 token 复用"的做法？

**结论：有"复用"，但没有"解量化"这个动作 —— 因为他们的格式根本不需要解。权重以压缩格式进 smem，被 MMA 在 M 维（最多 240 行）上复用；跨 m block 不复用，宁可重读。**

### 1) 不动点：smem 里只有一份**压缩**权重

```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:203-206
alignas(kSharedMemoryAlignment) a_dtype_t      smem_a[kNumStages][LOAD_BLOCK_M * BLOCK_K];
alignas(kSharedMemoryAlignment) weight_dtype_t smem_b[kNumStages][LOAD_BLOCK_N * BLOCK_K];   // ← 权重，1 byte/elem
uint32_t smem_sfa[kNumStages][SF_BLOCK_M * (BLOCK_K / 128)];
uint32_t smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K / 128)];
```
`weight_dtype_t` 对 4-bit 权重就是 `float_e2m1_unpacksmem_t`（`DG:.../impls/sm100_fp8_fp4_mega_moe.cuh:156` 的断言），**没有 float32/bf16 的权重副本**。

### 2) 唯一的"解包"发生在 TMA→smem 的硬件路径上，一个 tile 只做一次

```cpp
// DG:csrc/jit_kernels/impls/runtime_utils.hpp:69-70
case kPackedFP4:            return fp4_unpacked_smem ? "cutlass::detail::float_e2m1_unpacksmem_t"
                                                     : "cutlass::float_e2m1_t";
// DG:csrc/jit_kernels/impls/runtime_utils.hpp:96-97
case kPackedFP4:            return fp4_unpacked_smem ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                                                     : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
```
```cpp
// DG:csrc/jit_kernels/heuristics/mega_moe.hpp:214
// NOTES: FP8 activations, FP8 weights, and FP4 weights unpacked to 8-bit in smem all use 128B swizzle
```
也就是说：**"把 4-bit 权重展开成 1 byte"这一步，是 TMA 在搬进 smem 时硬件做的，每个 (n_block, k_block) 只做一次，然后这些 smem 字节被本 m-block 的全部行（≤240）复用，用完即弃。**

### 3) 复用范围：块内多行复用 + cluster 内两半 multiply 复用；跨块不复用

```cpp
// DG:csrc/jit_kernels/heuristics/mega_moe.hpp:143
const int load_block_m = block_m / 2;      // Always multicast on A
// DG:.../heuristics/mega_moe.hpp:179-184  B tile 每 stage 只有 block_n*block_k 一份
const int smem_b_size_per_stage = block_n * block_k * num_mma_elem_bytes;
```
- 一条 MMA 就把 `UMMA_N = align(valid_m,16)` 行全吃掉（Q1），所以"一次权重量、多 token 用"是在**一条 MMA 指令内部**完成的，代码里看不到 for-m 循环。
- 2-CTA cluster 下 A（token）按 `block_m/2` 双播（`DG:.../heuristics/mega_moe.hpp:143`，`DG:.../impls/sm100_fp8_fp4_mega_moe.cuh:743-751`），权重 tile 则靠 cluster 共享 smem 的 TMA multicast 一份给两个 CTA。
- 跨 m block：**明确不复用**，重读（`DG:.../heuristics/mega_moe.hpp:97`）。

### 4) 对我们的确切含义

他们代码库中**不存在**"先把权重解到 shared/register，再对多个 m 复用"的显式写法 —— 因为对他们来说这一步由 TMA/MMA 硬件免费完成，而且是**每 tile 一次、不是每 token 一次**。我们要在 IQ 码本 + SYCL 上做等价的事，只能自己写：**解一次到 local/register tile，然后对组内所有 entry 做 MAC**。他们能给我们的是"复用单位 = 权重 tile，tile 的行数尽量装多"这个结构，而**不是**可抄的实现。

---

## Q5. 每个专家只有 1 个 token 时的并行度：一个 CTA 算一个专家？多个专家一个 CTA？grid 怎么定？

**都不是。它是一个 persistent kernel：grid = 全部 SM，2-CTA 一簇，任务 = (某专家的一个 m block) × (N 方向的一个 cluster)，所有 CTA 用全局 atomic 抢任务。一个 1-token 的专家照样铺满全 GPU（几十个 CTA pair 一起算它一个）。**

```cpp
// DG:csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146,192-197
const auto num_sms = get_num_sms_for_mega_moe();     // 断言 == 设备 SM 数（`DG:.../heuristics/mega_moe.hpp:67-71`）
jit->launch(kernel, {
    .num_smem_bytes = config.smem_size,
    .grid_dim    = dim3(num_sms, 1, 1),
    .block_dim   = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
    .cluster_dim = dim3(2, 1, 1),
}, ...);
```
```cpp
// DG:deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:97
const uint32_t sm_idx = blockIdx.x;                  // 每 SM 一个常量身份，之后全是任务循环
```
```cpp
// DG:deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:343-363  抢任务：atomic_add 一个计数器
static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
    uint32_t result = 0;
    if (cute::elect_one_sync())
        result = ptx::atomic_add(task_count_ptr + locality_domain_idx, 1u);
    return ptx::exchange(result, 0);
}
template <uint32_t kNumClusters>
static CUTLASS_DEVICE bool get_next_task_idx(..., uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
    constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
    const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
    if (task_idx >= num_m_blocks * kNumLocalClusters) return false;
    m_block_idx   = task_idx / kNumLocalClusters;
    n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
    return true;
}
```
- `num_m_blocks` 已经是**全局** pool block 数（Σ 各专家 `ceil(m_e/block_m)`，`DG:.../scheduler/mega_moe.cuh:301`），所以 `m_block_idx` 就是"第几个专家的第几块"；`create_task` 再反查回 `(local_expert_idx, m_block_idx_within_expert, valid_m)`（`DG:.../scheduler/mega_moe.cuh:309-341`）。
- **一个专家被切多少个任务**：`kNumL1Clusters + kNumL2Clusters`，其中 `kNumL1Clusters = 2*moe_inter/(128*2) = moe_inter/128`，`kNumL2Clusters = hidden/(128*2) = hidden/256`（`DG:.../scheduler/mega_moe.cuh:176-179`；分母即 `BLOCK_N*2`）。以 `moe_inter=2048` 为例，**1 个 token 的专家 → L1 有 16 个 CTA pair（32 个 CTA）、L2 有 hidden/256 个 CTA pair 同时在算它**。这是他们能把 m=1 也跑满 GPU 的真正原因，与我们 `grid = (n_ff/(kWarps*RW), groups)` 的思路同源，但粒度细得多（一次只搬 `block_n=128` 宽的权重切片）。
- 还带两个 locality domain 的 work stealing（`DG:.../scheduler/mega_moe.cuh:363-370`，`kNumMinStealWaves=2`，注释："when there are few tasks, we disable work stealing to avoid cross-die latency"）。
- L1/L2 任务交错 + 用 mask 做 L2 对 L1 的细粒度依赖（`DG:.../scheduler/mega_moe.cuh:141-176`），所以 SwiGLU 一出来 L2 就能起。
- ring buffer 容量按"最坏在飞的 pool block 数"卡死，也是 SM 数驱动：`DG:csrc/apis/mega_moe.hpp:55-63`、`DG:.../scheduler/mega_moe.cuh:48-79`（`get_num_max_live_pool_blocks`）。

**注意可移植性**：这套依赖 tcgen05 UMMA + tmem、TMA multicast、2-CTA cluster DSMEM（`UMMA_M = LAYOUT_AD_M*2` = 跨 CTA 拼 M）、以及 `torch.distributed._symmetric_memory` 的 NVLink 对称缓冲（`DG:deep_gemm/mega/__init__.py:45-58`）。SYCL 单卡上只能借它的**调度思想**，不能借实现。

---

## Q6. 有没有"小 m 干脆不用 grouped GEMM、改用 GEMV"的分支？阈值是多少？

**没有。整条 MoE 路径不存在 GEMV 分支，最小 tile 就是 `block_m = 16`（等价 UMMA_N=16）；而且 DeepGEMM 里明确写着"小 m 别用非 mega 路径，直接上 Mega MoE"。**

1. **非 mega 路径的官方劝退**（这就是"阈值"的实际答案 —— 不是切 GEMV，而是切 mega）：
```cpp
// DG:csrc/jit_kernels/heuristics/runtime.hpp:57-63
static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
    if (jit->device.get_arch_major() != 10)
        return kLegacyMKAlignmentForContiguousLayout;      // = 128（:12）

    // The newly supported UMMA_N=256 allows a fixed alignment of 256, which is friendlier to MoE cast, M-grouped, and K-grouped operators.
    // NOTES: `expected_m` is ignored, so small values may incur performance loss; use Mega MoE directly for such workloads.
    return 256;
}
```
   而 DeepInfer 正是把这两条路都按 256 对齐使唤的：`DI:.../mlp.py:250-252`（`expert_alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()`，然后 `set_mk_alignment_for_contiguous_layout` + `tile_kernels.set_token_alignment`），dispatch 时把 `expert_alignment` 传给 DeepEP（`DI:.../mlp.py:477`）。**每专家 token 段按 256 行对齐**，1 token 也要占 256 行的 buffer/对齐空档。
2. **mega 路径没有 GEMV 也没有小 m 特判**：`block_m` 下限 16（`DG:.../heuristics/mega_moe.hpp:100`），MMA N 下限 16（`DG:.../impls/sm100_fp8_fp4_mega_moe.cuh:847-850`）。`m == 0` 才早退（`DG:csrc/apis/gemm.hpp:226-227`，非 mega 路径）。
3. **唯一的 GEMV 在别处**：DFlash 的 confidence head / markov 投影（一个 rank-1 点积，m=1 时就是逐 token 内积），手写 TileLang GEMV：`DI:deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py:21-66`（`GEMV` 宏 + `T.Kernel(n, threads=128)`，每 block 一个 token）。**和 MoE 无关，不能当先例。**
4. 全局搜不到 MoE GEMV：`deep_gemm` 里 `gemv` 零命中；`cublaslt_*` 只在 ViT 用（`DI:deepinfer/model_executor/models/vision_encoder/vit.py:50`）。

---

## 判断（≤200 字）

**能直接搬的是调度骨架，不是解码。** ①"token 当 MMA 的 N、权重当 M"的 AB 互换 + 最小 16 的 pad：我们窗口路径应改成每专家只切 `ceil(m_e/BLOCK_M)` 个块、块内 `valid_m` 截断、空行不写回，并把 `ne_max` 从 T 改成"每专家 entry 数上界"，避免 `NE=8` 那种空转；②一专家一块（1.2 entry → 1 块）意味着码本每轮只该搬一次、entry 只该扫一次，我们现在按 (token,expert) 重读是纯浪费。**搬不了的是 Q4**：他们权重不解码，靠 TMA 硬件展开 + tcgen05 原生消费，代码里没有"解一次、多 m 复用"的实现；我们的码本表必须自己"解一次到 local、对组内 entry 复用"，这块无源可抄。
