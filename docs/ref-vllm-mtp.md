# vLLM 投机解码（EAGLE / MTP / n-gram）实现拆解 —— 可借鉴清单

阅读对象：`vllm-project/vllm`，`main` 分支（`pushed_at 2026-11-03T16:52:09Z`）。
取数方式：`api.github.com` / `raw.githubusercontent.com`（未跑任何 GPU 负载、未构建）。
下文所有 `file:line` 均指该分支上该文件的**当前**行号。

> **先纠正一处路径**：任务描述里的 `vllm/v1/spec_decode/medusa.py`、`topk_proposer.py` 在当前 main 上**已不存在**（HTTP 404）；
> `eagle.py` 只剩 554 字节的壳（`vllm/v1/spec_decode/eagle.py:10`，`class EagleProposer(SpecDecodeBaseProposer)`），
> 真正的 EAGLE/MTP/DFlash 逻辑全部在 **`vllm/v1/spec_decode/llm_base_proposer.py`（1865 行）**。
> 目录现状：`llm_base_proposer.py`(84 KB)、`dflash.py`、`fly.py`、`gemma4.py`、`step3p5.py`、`ngram_proposer.py`（CPU/numba）、
> `ngram_proposer_gpu.py`（GPU 版）、`suffix_decoding.py`、`metadata.py`、`utils.py`、`dynamic/`。
> MTP 不是单独文件，而是 `method="mtp"` 走同一个 proposer（`llm_base_proposer.py:1004-1014` 里 `DeepSeekMTPModel` / `Glm5NextMTPModel` / `KimiK3MTPModel` 走「返回两个 hidden states」的分支）。

---

## 1. k+1 个 verify query 到底怎么打包

**结论：一条扁平 token 维度的 batch，不是每序列一个 batch。** 全局只有 `total_num_scheduled_tokens` 这一个 token 数，
每请求贡献 `1 + len(draft)` 个 token，用 `query_start_loc` 切分。

- `vllm/v1/worker/gpu_model_runner.py:1947` `req_indices = np.repeat(arange[:num_reqs], num_scheduled_tokens)` —— 先摊平。
- `:1951-1953` `cu_num_tokens = cumsum(...)`；`:2032-2038` 写 `query_start_loc[0]=0, [1:B+1]=cu_num_tokens`，末尾 padding 成**非递减**（FlashAttention 要求）。
- `:2044-2049` 先算 **optimistic seq_lens**（假设草稿全接受），纯度是 CPU 侧上界，供 `max_seq_len` 与 `discard_request_mask` 用。
- `:2147-2151` `compute_slot_mapping(num_reqs, query_start_loc, positions)` —— 一次 kernel 给全部 k+1 个 token 生成 KV 槽位。

### 1.1 采样元数据：`SpecDecodeMetadata`

`vllm/v1/spec_decode/metadata.py:8-25` 只有 7 个字段：`draft_token_ids[num_tokens]`、`num_draft_tokens[B]`（**list[int]，留在 CPU**）、
`cu_num_draft_tokens[B]`、`cu_num_sampled_tokens[B]`、`target_logits_indices`、`bonus_logits_indices`、`logits_indices`。

构造处 `gpu_model_runner.py:2583-2656` 有一段可直接照抄的注释示例：

```
cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
num_draft_tokens:         [  3,   0,   2,   0,   1]
logits_indices:           [0,1,2,3, 103,104,105,106, 206,207,208]
target_logits_indices:    [0,1,2, 5,6, 9]        # 只取"草稿位置"的 logits
bonus_logits_indices:     [3, 4, 7, 8, 10]       # 每请求额外 +1 个 bonus token
```

要点：**logits 只对 `k+1` 个位置算，不对全部 token 算**（prefill chunk 里多余的位置被 index 掉）；
`bonus_logits_indices` 是"全部接受后再采样一个"的那一列，所以一次 verify 最多产出 `k+1` 个 token。
`vllm/v1/sample/rejection_sampler.py:290-330` 的 `parse_output` 用 `PLACEHOLDER_TOKEN_ID` 原地标记被拒位置，再过滤——**不在 GPU 上压缩**。

### 1.2 让 attention 认识"k+1 个 query"

`vllm/v1/attention/backend.py:394-457` 的 `CommonAttentionMetadata` 是唯一契约：

| 字段 | verify 时的含义 |
|---|---|
| `query_start_loc[B+1]` | 每请求 query 起点；decode verify 时**每段长 k+1** |
| `seq_lens[B]` | 含本次 k+1 个 token 的上下文长（先给上界） |
| `max_query_len` | = `1 + num_spec_tokens`（`backend.py:579-596` `max_decode_query_len`） |
| `causal: bool=True` | verify 仍是因果的（草稿 token 只能看自己之前） |
| `block_table_tensor` / `slot_mapping` | 分页 KV 的页表 + 逐 token 槽位 |

构造点 `gpu_model_runner.py:2356-2372`。后端只要求"query 长度一致"就能把它当 decode 处理：
枚举 `AttentionCGSupport.UNIFORM_BATCH`=`"decodes" are 1 + num_speculative_tokens`（`backend.py:569-573`），
`reorder_batch_threshold` 会被抬到 `max_decode_query_len`（`backend.py:665-684`）——即 **k+1 个 query 被归入 decode 桶，而不是掉进 prefill 路径**。这是关键：掉进 prefill 路径就会走 varlen/变长 kernel，慢一个档次。

### 1.3 分页 KV 的 verify 与"回滚"

- 槽位是按 **position** 算的（`vllm/v1/worker/block_table.py:153-179`，一个 triton kernel，`slot = block_table[pos//bs]*bs + pos%bs`），
  与本次接不接受草稿无关。
- 因此被拒 token 写进 KV 的数据**不需要任何截断/拷贝**：下一次 verify 用同 position 覆盖同一槽位，而"读到哪"由 `seq_lens`/`num_computed_tokens` 决定。
  这是 `:2102-2123` 里 `update_num_computed_tokens_for_batch_change` 只改计数器、不碰 KV 的原因。
- 推论（代码未明写但由上述两条决定）：**rollback 成本 = 1 个 int 数组的修正，与 k 无关**。这对我们尤其重要——不要为回滚做 KV memset 或块回收。

### 1.4 对 batch size = 1 的含义

B=1、k=5 时：`query_start_loc = [0, 6]`，`seq_lens=[n+6]`，`block_table` 1 行，`logits_indices` 长度 6。
**没有任何"per-sequence"维度的浪费**：vLLM 的整个 metadata 在 B=1 时退化成 6 个 token 的一维向量 + 1 个标量 seq_len。
真正的问题不是打包，而是**这 6 个 token 必须在一次 kernel 提交里完成**（见第 3、5 节）。
另有 `metrics` 里 `max_num_sampled_tokens = int(num_sampled_tokens.max())`（`:2249`）这类"batch 级标量"，B=1 时恒为 k+1，可编译期常量。

---

## 2. tiny per-expert token count 下 MoE 怎么做

这是**最值得抄的一节**。vLLM 有两条截然不同的路径，并在运行时按"稀疏度"选择。

### 2.1 稀疏度启发式：直接跳过 `moe_align_block_size`

`vllm/model_executor/layers/fused_moe/fused_moe.py:1614-1675` `_prepare_expert_assignment()`：

```python
naive_block_assignment = (
    expert_map is None
    and not weights_host_resident
    and num_tokens * top_k_num * 4 <= global_num_experts      # :1638
    and not (wna16 且 block_shape[1] > 0)
)
if naive_block_assignment:
    return None, topk_ids.view(-1), full((1,), topk_ids.numel()*BLOCK_SIZE_M)   # 不排序、不填充
```

注释（`:1629-1634`）写得很直白：`SPARSITY_FACTOR = 4` 保证 `tokens*top_k` 只激活一小部分专家时，走 Triton kernel 的
`sorted_token_ids is None` 分支，**"每个 (token, expert) 对读一次权重"**，即不做 per-expert 分块、不填充。

> 对我们：6 token × 6 routed = 36 条路由，`36*4 = 144 <= 384` → **vLLM 在我们的 batch 上正好会走这条不填充路径**。

两个例外恰好是我们踩的坑：
1. `weights_host_resident`（UVA offload 的权重）：注释说"重复读会多次过 PCIe，所以还是要按 expert 分组"。我们的 IQ 权重若在 host / 走 PCIe，这条就生效。
2. **块量化权重（wna16 且 `block_shape[1] > 0`）会被排除在 naive 路径之外**（`:1640-1645`）——因为块量化 kernel 需要按 expert 分组读 scale。我们的 1.75-bit IQ 就是块量化，所以**不能简单抄 2.1 的 naive 路径**，只能抄 2.2 的"分组但不浪费"。

### 2.2 若走对齐路径：填充量 = 命中专家数 × BLOCK_SIZE_M

`vllm/model_executor/layers/fused_moe/moe_align_block_size.py:55-104`：把每个 expert 的 token 数**向上取整到 `block_size`**。
`block_size` 直接取 Triton 的 `BLOCK_SIZE_M`（`fused_moe.py:1658-1665` 传入 `config["BLOCK_SIZE_M"]`）。
于是最坏情况（36 条路由落在 36 个不同 expert）填充后行数 = `36 * BLOCK_SIZE_M`。
vLLM 的止损手段是**把 M tile 压到 16**：

`fused_moe.py:1335-1445` `get_default_config(M, E, N, K, topk, dtype, block_shape)`：

- `M <= 32 → block_m = 16`（`:1413-1414`）；int4/int8 wna16 路径 `M <= 20 → BLOCK_SIZE_M=16, GROUP_SIZE_M=1`（`:1402-1407`）
- `block_n = 64 if M <= 64 else 128`（`:1422`），`block_k = 128 if fp8 or M <= 64 else 64`（`:1427`）
- `tokens_per_expert = M // max(E,1)`；`group_m = 16 if tokens_per_expert > 128 else 1`（`:1432-1433`），注释："expert 多时每个 expert 只有几个 token，分组毫无用处"。**我们 384 expert、每 expert ≤6 token，`GROUP_SIZE_M` 必须是 1**——这一条直接省掉一类无效调参。
- `num_warps = 4 if M <= 128 else 8`（`:1437`）
- 另有 `should_moe_wna16_use_cuda(...)`：`num_valid_tokens / num_experts <= 6` 时切到 CUDA wna16 kernel（`:1322-1332`）——即"每 expert 平均 ≤6 token"本身被当成一个 kernel 选择的判据。

### 2.3 expert-parallel dispatch

- `expert_map`（全局 expert → 本地 shard，不属于本 rank 记 -1）+ `ignore_invalid_experts`：`moe_align_block_size.py:66-96` 的 docstring 明确说，EP 下先按全局 expert 对齐，返回前把非本 rank 的 expert 标 -1，好让 matmul 整块跳过。
- 单卡（EP=1，`expert_map is None`）时，上面 `naive_block_assignment` 的第一项成立。**我们是单卡，所以 EP 相关的复杂度可以直接不抄**，只保留"按 expert 分组的唯一收益 = 权重只读一次"。

---

## 3. 怎么做到 draft+verify 全程不回 host 等

vLLM 的原则：**能留在 GPU 的标量一律不回 CPU；必须回 CPU 的走 pinned buffer + event + 独立 copy stream，且延迟到真正需要时才等。**

具体机制（按调用顺序）：

1. **不压缩被拒 token，用 padding 占位。** `llm_base_proposer.py:1101-1157` `prepare_inputs_padded()`，docstring 原话
   "No blocking CPU operations should be introduced in this function"，"all tokens are included as inputs to the speculator, **with the rejected tokens used as padding** and filtered out later by `token_indices_to_sample`"。
   后果：草稿模型的 `query_start_loc` 恒为 `arange[:B+1]`（`:672-677`，`max_query_len=1`），**接受率任意变化都命中同一张图**。
2. **拒绝数在 GPU 上算。** `_eagle_prepare_inputs_padded(...)` 直接产出 `num_rejected_tokens_gpu`（`:1120-1131`，triton，一个 program 一请求）；
   `next_token_ids` 与 `valid_sampled_tokens_count` 同样由 `_eagle_prepare_next_token_padded` 出（`:1051-1099`）。
3. **seq_lens 在 device 上就地修正**：`common_attn_metadata.seq_lens -= num_rejected_tokens_gpu`（`:683-684`），随后 `invalidate_cached_layout()`（`:685`）。
4. **CPU 侧只保留 list 型小标量**：`SpecDecodeMetadata.num_draft_tokens` 是 `list[int]`（`metadata.py:13`），供调度器下一步用，不参与 GPU 计算。
5. **D2H 抄送是懒加载的。** `gpu_model_runner.py:4580-4600` `_copy_draft_token_ids_to_cpu()`：
   ```python
   if self.use_async_scheduling and not (
       scheduler_output.has_structured_output_requests
       or self.input_batch.sampling_metadata.output_token_ids):
       return          # 默认什么都不抄
   ```
   需要抄时也只写进 `pin_memory` 的 `draft_token_ids_cpu`（`:926-940`），并用 `torch.Event` + 独立 `torch.cuda.Stream`（`:920-931`），消费端 `num_accepted_tokens_event.synchronize()`（`:2075`）才等。
6. **`async_tensor_h2d` 传小数组**（`gpu_model_runner.py:2631-2641`），避免 `.item()` 拉回。
7. **刻意留在 host 的东西**：
   - **n-gram 匹配**：`vllm/v1/spec_decode/ngram_proposer.py:11-60` 是 numba `@njit`+`prange` 的 CPU 实现，且 `num_numba_thread_available = min(1, cpu_count//2) // tp_size`（`:47-52`，注释"TODO: 从 1 提到 8"）——即**连 vLLM 都只给 1 个 CPU 线程**，因为服务端 CPU 不是瓶颈。只有在 CPU 成为瓶颈时才切 `ngram_proposer_gpu.py`（该文件 docstring 写明"fully async PyTorch tensor operations / fully vectorized"）。
   - 调度器、请求状态、structured output / penalties / bad_words 这类需要真实 token id 的逻辑。

> 对我们的直接含义：我们 32 ms host enqueue ≈ GPU 时间，**vLLM 的"不回 host"不等于"host 便宜"**——它是靠"提交次数少"把 host 成本摊掉的（1 次 replay vs 我们 ~800 次提交）。抄它的语义（padded batch、GPU 侧计数），但必须先解决提交次数。

---

## 4. CUDA graph：几张图、哪些 batch、draft 与 verify 是否分开

### 4.1 两个独立的 dispatcher，draft 与 verify 分开

- **target/verify**：`gpu_model_runner.py:875` 自己的 `CudagraphDispatcher`，可 FULL。
- **drafter**：`llm_base_proposer.py:181` 另建一个 `CudagraphDispatcher`，但 `initialize_cudagraph_keys`（`:420-435`）里
  **只允许 PIECEWISE**：
  ```python
  if not enforce_eager and cudagraph_mode.mixed_mode() in [PIECEWISE, FULL]:
      eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
  ```
- 相应地，drafter 拿到的 attention metadata 是**未 padding** 的：`gpu_model_runner.py:2522-2530` 注释
  "the drafter still only uses piecewise cudagraphs … therefore does not want to use padded attention metadata"，显式调用 `.unpadded(num_tokens, num_reqs)`（`backend.py:528-558`）。

### 4.2 有多少张图

- 默认 capture size 列表（`vllm/config/compilation.py:708-717`）：
  `[1, 2, 4] + range(8, 256, 8) + range(256, max_cudagraph_capture_size+1, 16)`，默认封顶 **512**（Blackwell 数据中心卡 1024）。
  即默认约 **~70 个 token 尺寸**。
- **spec decode 会把尺寸对齐到 k+1 的倍数**（MRV1 路径）：`compilation.py:1508-1520` →
  `adjust_cudagraph_sizes_for_spec_decode()`（`:1549-1577`）用 `round_up(size, uniform_decode_query_len)` 重建列表，
  并同步收缩 `max_cudagraph_capture_size`。k=5 时尺寸大致变成 6,12,24,48,…（数量级 ~70 → 更少），且**没有一个尺寸不是 6 的倍数**。
- FULL decode 图的 `num_reqs` 由 token 数反推：`vllm/v1/cudagraph_dispatcher.py:141-143`
  ```python
  num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
  assert num_tokens_padded % uniform_decode_query_len == 0
  ```
  并只保留 `>= k+1` 且 `<= max_num_seqs*(k+1)` 的尺寸（`:209-229`）。
- 两层键：FULL 用精确 `num_reqs`（注释：FA3 的 scheduler_metadata 依赖它），PIECEWISE 放宽为 `num_reqs=None, uniform=False`（`:197-201`、`:311-316`）。

### 4.3 batch size 变化怎么办

1. **向上 padding 到最近的 capture size，再 replay。** 映射表是纯 CPU 查表：`cudagraph_dispatcher.py:71-108` `_compute_bs_to_padded_graph_size()`（一次 `list[int]`，O(1) 查询）。
2. **padding 出来的假行必须有合法页表**：`gpu_model_runner.py:2306-2308` 用 `NULL_BLOCK_ID`（block 0 保留）填 `block_table[num_reqs:num_reqs_padded]`，`seq_lens[num_reqs:] = 0`（`:2145`），`num_decode_draft_tokens[num_reqs:] = -1`（`:2233`）。
3. **未命中就直接不抓图**：`dispatch()` 在 `num_tokens > max_size` 或无 key 时返回 `NONE` + 原始 descriptor（`:272-279`）——优雅降级，不抛错。
4. **更新版：varlen decode 图**（比"每尺寸一张图"更强）。`vllm/v1/worker/gpu/cudagraph_utils.py:305-321`
   "Varlen decode graphs take any mix of 1..decode_query_len tokens per request, worst case 1 token per request"，
   `max_query_len=self.decode_query_len`，`num_reqs=min(num_tokens, max_num_reqs)`；
   能力声明在 `backend.py:642-663` `get_varlen_cudagraph_max_query_len()`：
   **一张图能重放"每请求 1..L 个 query、padding 请求 0 个 query"的任意 decode batch**，per-request 长度从 device 侧 `query_start_loc` 读，host 只给 token 总数和上界。
   这正是为我们这种"接受率任意"的场景准备的：**不需要为每个接受数各抓一张图**。
5. **drafter 内部的 metadata 重建也可以进图**：`backend.py:610-617` + `:760-768` `update_draft_decode_metadata()`
   ——"Speculators can record this call inside their draft decode CUDA graphs, so it runs once at capture and never again in Python"，要求实现只用 capture-safe op、写进 persistent tensor。
6. **Dynamic SD（自适应 k）也要预抓**：`cudagraph_utils.py:276-303`，为每个可能的 `num_spec` 值都算一个 `decode_query_len` 并抓图。
7. 抓图顺序与内存：`cudagraph_dispatcher.py:324-347`，PIECEWISE 先、FULL 后，descriptor 按 `(num_tokens, num_active_loras)` **降序**（大图先抓，省内存碎片）。

---

## 5. 我们引擎该按什么顺序试

前提：单卡、B=1、40 层 MoE/384 expert/6 routed、1.75-bit IQ、无图捕获、~20 次 kernel 提交/层、host 32 ms ≈ GPU 32 ms。
MTP-5 接受率 100% 但 304 ms/round（19 tok/s）反而更慢。

| # | 动作 | 为什么（对着我们的数字） | 上游参照 |
|---|---|---|---|
| **1** | **把 40 层 × ~20 提交压成一次提交**：先做"整轮 verify 的 CUDA/SYCL graph 捕获"，退而求其次做 kernel 融合把提交数从 ~800 砍到 <100 | 我们 32 ms host ≈ GPU 时间，6 token 一轮时 host 直接翻倍 → 这是 304 ms/round 里最大的单项。vLLM 用 1 次 replay 干掉整个 host 路径 | `vllm/v1/cudagraph_dispatcher.py:141-143`；`vllm/v1/worker/gpu/cudagraph_utils.py:305-321`；`gpu_model_runner.py:2306-2308`（padding 行用 NULL_BLOCK_ID） |
| **2** | **verify 走"一条扁平 token 流 + 每请求 k+1"的 metadata**，attention 用一次 kernel 处理全部 6 个 query（因果、页表 1 行、`query_start_loc=[0,6]`）；**不要**为每个 draft token 单独发一次 attention | 我们 attention 从 11 ms/token 变成 1.8x/token ≈ 线性放大，说明现在是"6 次单 token 调用"。vLLM 的额外成本只有 6×6 的 mask 逻辑，K/V 流量不变 | `gpu_model_runner.py:1947`、`:2032-2038`、`:2356-2372`；`vllm/v1/attention/backend.py:569-573`（UNIFORM_BATCH）、`:579-596` |
| **3** | **MoE 按 expert 分组、每个命中 expert 的权重只读一次**；块量化权重不能走"gather 型 naive 路径"，但要把 M tile 压到 16、`GROUP_SIZE_M=1` | 6 token × 6 expert ≈ 36 条路由（≤36 个不同 expert），naive 路径会读 36 次 expert 切片；分组后每个 expert 只读一次，且填充上限 = 命中数 × 16。我们"routed experts 2x 变贵"正是没分组/或分组后填充过大的表现 | `fused_moe.py:1614-1675`（naive 判定 + host-resident/块量化例外）、`:1335-1445`（`M<=32→block_m=16`、`:1432-1433` GROUP_SIZE_M）、`moe_align_block_size.py:55-104` |
| **4** | **接受/拒绝判定全留 GPU**：不把被接受的 token 数拉回 host，被拒 token 当 padding 用；k 自适应也不要靠 host 读计数 | 我们每轮都要 host 决定下一步调度；只要有一次 `.item()`，第 1 项的图捕获就白抓。vLLM 用 GPU kernel 出 `num_rejected_tokens`，用 padded drafter batch 让接受率任意变化都命中同一张图 | `llm_base_proposer.py:1101-1131`（"No blocking CPU operations"）、`:672-684`；`gpu_model_runner.py:2102-2123`、`:4580-4600`（D2H 懒加载） |
| **5** | **回滚零成本化**：只推进 `num_computed_tokens`/`seq_lens`，KV 内容不截断、不清零、不拷贝（按 position 覆盖） | 我们 MTP 一轮 304 ms 里若含任何 KV 修正/拷贝，直接白扔；vLLM 的槽位是 position 的纯函数 | `vllm/v1/worker/block_table.py:153-179`；`gpu_model_runner.py:2147-2151` |
| **6** | **给 draft 侧配一个便宜得多的"小图"**：draft 用 PIECEWISE/更小 batch 的图，verify 用大图；不要两边用同一套 capture size | vLLM 明确分开：drafter 只允许 PIECEWISE 且用 unpadded metadata；verify 才是 FULL decode 图。我们 304 ms 里 draft 与 verify 混在一起，无法分别调优 | `llm_base_proposer.py:420-435`；`gpu_model_runner.py:2522-2530`；`backend.py:528-558` |
| **7** | **先试 n-gram / prompt-lookup 当"免费草稿"**：命中就省掉整个 draft 前向；匹配放 host 且只给 1 线程，别过度优化 | 我们 B=1、无并发，CPU 完全空闲；n-gram 命中率在代码/结构化输出上很高，命中时省的是 5×draft 前向 = 我们 304 ms 里的大头 | `ngram_proposer.py:11-60`（numba，`min(1, cpu//2)` 线程）；`ngram_proposer_gpu.py:1-18`（GPU 版仅当 CPU 成瓶颈）；`suffix_decoding.py` |
| **8** | 只有在 1-7 之后仍有余量，才去动 1.75-bit 的 dequant/融合 | vLLM 的小 batch 收益全在"少提交 + 少读权重 + 不回 host"，不在量化 kernel 本身 | — |

### 不建议照抄的三点

1. **`naive_block_assignment` 的"不分组"路径**：它的前提是权重在 HBM 且 `num_tokens*top_k*4 <= num_experts`；我们是块量化 + 可能 host-resident，`fused_moe.py:1633-1645` 两个例外都命中。
2. **`async_scheduling` 那一整套 event + copy stream**：它换来的是"host 不阻塞调度"，而我们的 host 本来就是瓶颈且已经阻塞；先做图捕获，再谈异步。
3. **~70 个 capture size**：那是为服务端 `max_num_seqs` 从 1 到几百的分布准备的。我们 B=1，**只需要 k+1 及其 padding 上界的极少数几张图**（甚至 1 张 varlen 风格的图）。

---

## 附：一句话版本

vLLM 的投机解码不是"更快的小模型"，而是**把 k+1 个 query 塞进 decode 的同一套 kernel/同一张 CUDA graph，并让"拒绝"只体现为 device 上的一个 int 修正**；
MoE 在 tiny batch 下的答案不是"更快的 GEMM"，而是**按 expert 分组让权重只读一次、把 M tile 压到 16、并尽量不填充**。
我们 304 ms/round 的病根大概率是**提交次数 × 6（每 token 一次）× 40 层**，而不是 k 本身。
