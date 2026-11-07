# SGLang 的 MTP / 投机解码 verify 路径拆解（供自研引擎借鉴）

快照：`sgl-project/sglang` @ `main`，commit `7b88f092392a4b673c39b2a6b66a18131cdfb60e`（2026-11-03）。
所有 `path:line` 均指该 commit。取证方式：`api.github.com` + `raw.githubusercontent.com`（github.com 页面不稳定），逐文件下载后本地比对行号。

> **目录已重构，先纠正常见过时认知**
> - `python/sglang/srt/speculative/eagle_worker.py` **已不存在**，v1 worker 被删，只剩 `eagle_worker_v2.py`（内含 `EagleDraftWorker` + `EAGLEWorkerV2` 两个类，draft 和 target 都在这一个文件里）。
> - `python/sglang/srt/model_executor/cuda_graph_runner.py` **已拆分**为：
>   - `model_executor/runner/decode_cuda_graph_runner.py`（DECODE / TARGET_VERIFY / DLLM_EXTEND 三种 phase 共用，见其文件头注释）
>   - `model_executor/runner/base_cuda_graph_runner.py`（抓取/桶/pad 骨架）
>   - `model_executor/runner_backend/*`（full / breakable / tc-piecewise 三种 graph backend）
> - `eagle_info.py` / `spec_info.py` 位置未变，`eagle_info.py` 里 `EagleVerifyInput.generate_attn_arg_prefill()` 是整条 verify 布局的**唯一真相点**。

---

## 1. verify 到底怎么跑：k+1 个位置就是一个 extend batch

### 1.1 布局：`bs*(k+1)` 个 query 平铺成一个 prefill/extend batch

关键在 `python/sglang/srt/speculative/eagle_info.py:91-121`：

```python
qo_indptr = torch.arange(0, (1 + batch_size) * self.draft_token_num,
                         step=self.draft_token_num, ...)   # :107-113
paged_kernel_lens = paged_kernel_lens + self.draft_token_num  # :118
total_tokens = paged_kernel_lens_sum + self.draft_token_num * batch_size  # :121
```

- 每个 request 固定贡献 `draft_token_num`（= k+1，EAGLE/MTP 下等于 `speculative_num_draft_tokens`，见 `spec_utils.py:78-98` 的 `resolve_num_tokens_per_req`）个 query 行，**等宽、无条件 padding**，因此 batch 的 token 数是一个常量 `bs*(k+1)`。
- KV 侧不是"取最近 k+1 个"，而是把整条 context 当作 prefix（`paged_kernel_lens += draft_token_num`），即 `cache_seqlens = 已提交长度 + k+1`。这就是**"verify 用 prefill kernel 而不是 decode kernel"**的根本原因：query 长度 > 1，且每个 query 要看到自己的完整前缀。
- 方法名就叫 `generate_attn_arg_prefill`，`EagleDraftExtendInput` 的同类方法也是 `generate_attn_arg_prefill`（`eagle_info.py:426-476`）。

### 1.2 attention kernel：extend/prefill 树掩码核，且 topk==1 时**完全不需要 mask**

- Triton 后端：`layers/attention/triton_backend.py:1008-1060` 为 `is_target_verify()` 单独建一套 metadata：
  `qo_indptr = arange(step=num_tokens_per_req)`、`custom_mask = spec_info.custom_mask`、`mask_indptr[1:bs+1] = cumsum(num_tokens_per_req*(seq_lens+num_tokens_per_req))` → **每个 request 的 mask 块在扁平 buffer 里的偏移**，等价于一个 CSR 的块状 mask。
- 真正下发的是**一个专门 kernel**，不是通用 extend：`triton_backend.py:1861-1900` → `verify_shared_kv_fwd` / `verify_splitkv_fwd`（选择逻辑 `triton_backend.py:93-101`、`:241-253`）。一个 kernel 一次吃下整个窗口的 `q`(`q.view(-1, heads, dim)`)、整条 KV 流、`qo_indptr`、`custom_mask`、`mask_indptr`、`max_extend_len`。
- **XPU 后端（离我们最近，SYCL）**：`layers/attention/xpu_backend.py:487-511`，`topk <= 1` 时：
  ```python
  num_tokens_per_req = spec_info.num_tokens_per_req                       # :494
  cache_seqlens_int32 = forward_batch.seq_lens + num_tokens_per_req       # :495-497
  cu_seqlens_q = arange(0, bs*num_tokens_per_req+1, step=num_tokens_per_req)  # :501-506
  ```
  → 直接走**普通 flash-attention prefill，纯 causal，不传任何 mask 张量**。
  `topk > 1` 才需要树：`:512-600` 把树掩码离线压成"每 query 的 page table 重排 + 排序后的 key"（无 mask 的等价实现）。
- 树掩码有三种模式：`kernels/ops/speculative/spec_tree.py:46-49`
  `FULL_MASK=0 / QLEN_ONLY=1 / QLEN_ONLY_BITPACKING=2`；
  `layers/attention/verify_mask.py:12-25` 的 `tree_mask_numel()` 直言：**FULL_MASK 在长上下文会到几百 MB，QLEN_ONLY 只有 KB 级**（`bs*N*N` vs `bs*N*(max_context_len+N)`）。
  默认值：`eagle_utils.py:169-172` `default_tree_mask_mode()` → **GPU 上是 FULL_MASK**，只有 CPU（intel_amx）后端用 QLEN_ONLY；非"被读取"用途的 mask 一律强制 QLEN_ONLY（`verify_mask.py:78-95`）。所以 FULL_MASK 在 GPU 上仍是默认，SGLang 只是**保留了省带宽的档位**并让不读整块 mask 的路径切过去。
- 掩码是**设备端原地生成**的：`VerifyMask` docstring（`verify_mask.py:28-40`）说明 `build_tree_kernel_efficient` 在 draft 之后原地写这块 buffer，**这正是 worker 能省掉 `seq_lens_sum` D2H 的原因**。

### 1.3 k+1 个 query 是否塞进同一个 GEMM：是

- 整条 verify 走 `ForwardMode.TARGET_VERIFY`（`speculative/eagle_utils.py:419`），而 `forward_batch_info.py:1796-1801` 的注释写明 `TARGET_VERIFY counts as extend`。因此所有 dense 层就是标准的 `[bs*(k+1), H] @ W`，**k+1 行一次性进同一个 GEMM**，dense 权重只读一遍 → 这就是我们实测"21 ms/token → 5 ms/token"能对齐的机制。
- lm_head 也是同样的 `[bs*(k+1), H] @ [H, V]`；但 draft-extend 阶段用 `select_index` 只对"每 request 最后被接受的那一行"跑 lm_head + LAST hidden（`eagle_info.py:376-380`）。

### 1.4 MoE 在"每 expert 只有几行"时怎么办

三件事，全在 triton MoE 路径上：

1. **先对齐**：`layers/moe/moe_runner/triton_utils/moe_align_block_size.py:37-78` 把 `topk_ids` 展平、按 expert 排序、**每个 expert 的 token 数补到 `block_size` 的整数倍**，产出 `sorted_token_ids / expert_ids / num_tokens_post_padded`。上游 up-GEMM 和 down-GEMM **共用这一次排序**（`fused_moe_triton_config.py:326-339` 明确要求两个 kernel 的 `BLOCK_SIZE_M` 一致，否则不能共用 sort）。
2. **薄 M 配置**：`triton_utils/fused_moe_triton_config.py:246-259`，当 `M <= E`（M = 本轮被路由的 token 行数，E = expert 数）：
   ```python
   BLOCK_SIZE_M = 16; BLOCK_SIZE_N = 32; BLOCK_SIZE_K = 64; GROUP_SIZE_M = 1
   ```
   否则默认 `64/64/32, G=8`。fp8 路径同理 `M <= E` 时 `BLOCK_SIZE_M 128→64`（`:213-227`）。
   注意 **M 是"token×topk"**：我们单 token 解码 M=6，verify k=5 时 M=36，**两者都 ≤ 384，都会落在同一档薄 M 配置**——所以"窗口变慢"不能靠换配置解释，见 §4。
3. **draft 与 verify 用不同后端**：`layers/moe/utils.py:1035-1048` `speculative_moe_backend_context()` 在 draft 期间临时把 `runner_backend` 换成 `get_speculative_moe_runner_backend()`；worker 在 `eagle_worker_v2.py:219 / 280 / 292 / 1456` 三处包住 draft / draft-extend 调用，**verify 不走这个 context**（用 target 的后端）。

---

## 2. batch 很小时如何保持 verify 窗口效率

### 2.1 窗口也吃 CUDA graph，而且是**复用 decode graph runner**

- `model_executor/runner/decode_cuda_graph_runner.py:364-380`：
  ```python
  self.captured_req_width = model_runner.decode_num_tokens_per_req(num_draft_tokens=...)  # :367
  if model_runner.spec_algorithm.is_speculative():
      self.capture_forward_mode = ForwardMode.TARGET_VERIFY                        # :376
  ```
  即：**投机解码开启时，这个 runner 抓的图就是 verify 图**（draft worker 走 `TARGET_VERIFY` 变体，`:371-374`）。没有单独的 "verify graph runner"。
- 抓取形状 = `bs` 桶 × 固定宽度 `(k+1)`，token 数由 `bs*(k+1)` 决定（`:400-404` 的 `get_batch_sizes_to_capture`，实现在 `runner/base_cuda_graph_runner.py:67-121`，会按 attn-TP/CP 对齐约束过滤桶）。
- 运行时**向上取桶**：`runner/base_cuda_graph_runner.py:180-197` `_pad_to_bucket()`，调用点 `decode_cuda_graph_runner.py:1573-1579`：
  ```python
  bs = self._pad_to_bucket(raw_bs, self.capture_bs)
  padded_num_tokens = bs * self.captured_req_width
  ```
  多余的行用 dummy request 填（`seq_len_fill_value`，`:454`），**不改变真实结果，只是多算几行**。
- 能否 replay 的判据 `can_run_graph()`：`decode_cuda_graph_runner.py:883-929`，其中一条是硬约束 —— `spec_info.num_tokens_per_req` 必须等于抓取宽度，否则退回 eager（`:899-904`）。
- verify 的图**和 decode 的图同构**，因此 attention backend 只需要一套 `init_cuda_graph_state`；metadata 在 `load_batch` 里被填进静态 buffer（`triton_backend.py:595-660` 的 `_update_target_verify_buffers`）。
- **整个 target forward 是一次 replay**：40 层 × 每层 ~20 kernel 全部压在 1 次提交里。draft 的 k 步也同理是一次 replay（见 §3.3）。

### 2.2 更激进的：ragged / compact verify（只算真正需要的行数）

- `speculative/ragged_verify.py:10-25` 定义 `RaggedVerifyMode{STATIC, CAP_ACCEPT, COMPACT}`，由 `SGLANG_RAGGED_VERIFY_MODE` 选。
- COMPACT 模式下 graph 以 **token 数**而不是 bs 为 key：`decode_cuda_graph_runner.py:724-728` `_build_ragged_verify_token_buckets()` = `{bs*width}`，`:753-754` `_capture_graph_size` 返回 `num_tokens`。
- 每轮按实际 `verify_lens` 求和后**向上取 token 桶**：`ragged_verify.py:44-53` `round_up_grid()`，`:162-183` `from_verify_lens()`；桶内剩余行用 `padded_to_bucket()`（`:185-215`）补齐。
- 收益：接受长度不满时不再按满宽 `(k+1)` 白算。限制写在 `decode_cuda_graph_runner.py:432-441`：COMPACT 不支持 two-batch-overlap / LoRA / `disable-cuda-graph-padding`。

### 2.3 draft / verify 用不同的 kernel 组合

- attention：draft 用多步 draft 专用 backend（`speculative/draft_worker_common.py:70-95` 的 `resolve_draft_worker_attention_backend()`，例如 `MultiStepDraftBackend`），verify 用 target 的 backend。draft 图抓取见 `eagle_draft_cuda_graph_runner.py`，名字就叫 `EAGLEDraftCudaGraphRunner`，且**在 `eagle_worker_v2.py:492-499` 的 `Device2DraftCudaGraphRunner` 表里显式注册了 `"xpu"`**（Intel 也是一等公民）。
- MoE：draft 走 `speculative_moe_backend_context()`（§1.4-3），verify 走 target 后端。
- 更进一步，draft 的 k 步可以整体变成一个 megakernel：`speculative/eagle_megakernel.py:1-40` + `eagle_worker_v2.py:700-703`（`self.megakernel.begin_draft_round(...)`），**一次 launch 跑完整个 draft loop**；限制在 `eagle_megakernel.py:36-52`：只支持 `topk==1` 的链式 draft、不支持 rejection sampling。

---

## 3. 层循环里如何不出现 host 同步

原则：**一个 round 只允许 0～1 次 D2H，且不在层循环内**。具体落点：

### 3.1 不再需要 `seq_lens_sum`（host int）

- `eagle_utils.py:389-390` 的注释：`end offsets (= start + draft_token_num) are computed inside the kernel, keeping the eager seq_lens + N add off the host critical path (bs=1 MTP inter-phase seam)`。
- 兑现方式：
  - `assign_extend_cache_locs_uniform_func`（`eagle_utils.py:395-402`）在 kernel 内算 KV 写入位置；
  - `mask_indptr` 在 device 上 `cumsum`（`triton_backend.py:1055-1060`）；
  - kv/gather 用 `runner_ids.pack_read_stream(...)` 直接按 plan 打包成 `kv_indices`（`eagle_info.py:123-136`）；
  - 允许 `seq_lens_sum is None` 时**过量分配**：`triton_backend.py:1015-1019` `# gpu_only: seq_lens_sum may be None; over-allocate is safe (ragged write)`。

### 3.2 accept / sampling 全在 device 上：一个 kernel 走完树

- greedy 分支：`eagle_utils.py:700-730`，先对 `[bs*(k+1), V]` 求 argmax，再调**一个** `verify_tree_greedy`（`kernels/ops/speculative/tree.py:37-60`）：
  输入 `candidates / retrieve_index / retrieve_next_token / retrieve_next_sibling / target_predict`，**原地写出** `predicts / accept_index / accept_token_num`。
  → 树怎么走、走到第几层停、accept 几个 token，**host 完全不知道也不需要知道**。
- 连 `predict`/`accept_index`/`num_correct_drafts` 都从**一块 buffer 里切视图**，为的是接受结果只做**一次** TP broadcast：`eagle_utils.py:934-956` `_verify_outputs(packed=...)`（`SGLANG_OPT_SPEC_PACKED_VERIFY_BCAST`）。
- 整个 post-verify 段（sampling + accept + commit 输出 + mamba state commit）可以整体变成一张图：`speculative/eagle_post_verify_graphs.py:1-10` 文件头（`SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH`），key 见 `:93-120`，body/commit 见 `:159-200`。**这意味着 accept 的标量结果从不回到 host**。

### 3.3 draft-extend（变长第二阶段）也不回 host

- 接受长度是变长的，但 SGLang 不做 `num_accepts.tolist()`：
  - `eagle_worker_v2.py:1138-1156`：`prepare_draft_extend_inputs(accept_lens, next_token_ids, speculative_num_draft_tokens)` 是**一个 device kernel**，产出 `num_correct_drafts / select_index / next_token_ids`；
  - draft-extend 的 batch 仍然按**固定满宽** `num_draft_tokens/req` 构造（`eagle_worker_v2.py:1157-1172`），越界行靠 `select_index` 在 lm_head 里剪掉（`eagle_info.py:376-380`）。
- 显式提醒"别用阻塞拷贝"：`eagle_utils.py:998-1000`
  `non_blocking H2D: a blocking .to() syncs the schedule stream ... -> host stalls a full forward`。
- 唯一允许的 host 读取：draft-extend 的 attention 需要 `num_accept_tokens_cpu`（`eagle_info.py:350-352` 注释），且它是**上一轮**的结果，可异步 / 用上界估算。

### 3.4 跨流：plan stream 提前跑，然后只在两处 join

- `eagle_worker_common.py:585-600`：verify 前的 metadata 准备 `eagle_prepare_for_verify` 放在 `plan_stream`（`eagle_utils.py:342-445`），主 compute stream 只在 `:589` 与 `:602-603` 两处 `wait_stream`。
- 注释还点破了代价：`eagle_worker_common.py:616-625` —— plan 阶段用的是 draft 的**旧值**，draft 完成后必须重算 `custom_mask` / `positions`（`update_verify_buffers_to_fill_after_draft`）。

---

## 4. 为什么"朴素实现"会让窗口每 token 更贵，以及 SGLang 的对策

按我们测到的现象（dense 摊薄成功、attention 0.9→1.6 ms/layer/token、MoE 每 token 2×、无 graph、每层 ~20 kernel）：

| # | 朴素实现的动作 | 为什么每 token 变贵 | SGLang 的做法（file:line） |
|---|---|---|---|
| A | 每个 query 各跑一遍 decode attention（k+1 次 kernel / 或 1 个 kernel 里 per-query 循环读 KV） | KV 前缀被读 k+1 遍 → attention 线性放大，而不是"读一遍 KV 乘 k+1 个 q" | 一个 extend 型 kernel 吃整个窗口：`qo_indptr` 分段、KV tile 在 q 块内复用（`triton_backend.py:1861-1900`）；XPU 直接 `cu_seqlens_q` 分段 prefill（`xpu_backend.py:501-506`） |
| B | 为窗口生成完整 `(k+1) × (ctx+k+1)` 布尔 mask | mask 字节数 ≈ `bs*N*(ctx+N)`，长上下文直接几百 MB，纯带宽浪费 | 提供 `QLEN_ONLY` 档（`bs*N*N`，`verify_mask.py:12-25`，GPU 默认仍是 FULL_MASK，见 `eagle_utils.py:169-172`）；**topk==1 时 XPU 后端完全不传 mask**（`xpu_backend.py:487-511`）；树掩码设备端原地写（`verify_mask.py:28-40`） |
| C | 每轮重新算 metadata / 重新 JIT / 重新 plan | host 侧每轮固定开销被 k+1 个 token 摊，但摊不动 | 固定宽度 + bs 桶 + 整图 replay（`decode_cuda_graph_runner.py:364-404`、`base_cuda_graph_runner.py:180-197`） |
| D | 每层 20 次 host 提交（我们现在的样子） | 一次 round 是 `40 × 20 = 800` 次 launch，round 越长 host 越可能变成瓶颈 | 整个 forward 1 次 replay；再进一步 draft loop → megakernel（`eagle_megakernel.py:1-40`） |
| E | 变长第二阶段触发 `.item()` / `.tolist()` | 每 round 一次 stall = 一次完整 forward 的时间 | `prepare_draft_extend_inputs` 纯 device + 满宽 padding + `select_index`（`eagle_worker_v2.py:1138-1172`） |
| F | 满宽 `(k+1)` 白算（接受长度常常 < k+1） | 无效行按真实比例付费 | ragged/compact verify 按 token 数取桶（`ragged_verify.py:44-53`、`decode_cuda_graph_runner.py:724-728`） |
| G | MoE 按 token 逐个 dispatch / 每 expert 只 1 行却用大 BLOCK_M | 排序/对齐/权重流的固定开销被小 M 放大 | 一次 align+sort 两 GEMM 共用（`moe_align_block_size.py:37-78`）+ `M<=E` 薄 M 配置 `16/32/64`（`fused_moe_triton_config.py:246-259`） |

**对我们最要紧的一条推论**（不是 SGLang 明说的，是我们数据 + 它的结构共同指向的）：
verify 窗口能摊薄的只有 **"每 round 读一次"的权重**（dense、shared expert、lm_head）；**routed expert 的权重流量本质上是 per-token 的**——窗口里 6 个 token 最多触达 36 个不同 expert，而单 token 只触达 6 个，**每 round 的 expert 字节数涨 ~6×，正好抵消 6× 的 token 数**。所以 100% 接受率下 MoE 的每 token 成本**理论上就是 1×，不会更好**；我们测到的 2× 说明窗口路径额外多花了钱（padding 行、mask、A 项）。这也意味着 **MTP 在 weight-bound MoE 上的天花板就是"dense 全摊掉 + expert 不摊"**，收益上界可以提前估出来。

---

## 5. 我们引擎该按什么顺序试

按"预期收益 / 改动成本"排序。每条给出理由与上游出处。

1. **把 verify 窗口做成"一个 extend 段"，attention 一次调用吃 k+1 个 query（先只做因果，不做树）**
   理由：这是 A + B 两条浪费的根。我们的 accept 是链式（topk=1），k+1 个位置是**连续位置**，`eagle_info.py:437-440` 明确写了"queries are consecutive positions, so the kernel's causal window masks them exactly"——**根本不需要 mask 张量**，只要 `cu_seqlens_q = arange(step=k+1)`、`cache_seqlens = committed + k+1`。我们现在的 0.9→1.6 ms/layer/token 说明 KV 没有在窗口内复用；改成 one-launch extend 后目标应回到 ≤0.9 ms/layer/token。
   出处：`eagle_info.py:91-121`；`xpu_backend.py:487-511`（最贴近的 SYCL 实现）；`triton_backend.py:1861-1900`。

2. **先给"单 token decode + verify 窗口"两个形状各抓一张 SYCL graph（等价于 CUDA graph），把每层 20 次提交压成 1 次**
   理由：我们 host 32 ms ≈ GPU 39 ms，是等价的流水；round 一旦变长，800 次 launch 的固定成本不会随 token 数下降，会直接吃掉收益。SGLang 就是复用同一个 decode graph runner 抓 `TARGET_VERIFY`（`decode_cuda_graph_runner.py:364-380`），并按 bs 桶 pad（`base_cuda_graph_runner.py:180-197`）。即使 SYCL 侧只能做到 "command-buffer 录制 + 重放"，也要做——这是 D 条。
   出处：`runner/decode_cuda_graph_runner.py:364-404, 883-929, 1573-1579`；`runner/base_cuda_graph_runner.py:180-197`；draft 侧一次 replay 抓 k 步：`eagle_draft_cuda_graph_runner.py:517-546`。

3. **把 accept / sampling 做成 device 端单 kernel，禁止在层循环与 round 循环里出现 `.item()`/`.tolist()`**
   理由：Greedy 下 accept 只是"沿树走一遍 + 比较"，`verify_tree_greedy` 一个 kernel 原地写 `predicts/accept_index/accept_num`（`tree.py:37-60`）；把 `new_seq_lens`、bonus token、下一轮 `select_index` 全部留在 device 上。我们目前 5 ms/token 的 "CPU-lane wait" 里很可能就有这类 stall。
   出处：`eagle_utils.py:700-730`（一个 kernel 走树）、`:934-956`（结果做一块 buffer，一次 broadcast）、`eagle_post_verify_graphs.py:1-10`（整段 can be graphed）、`eagle_worker_v2.py:1138-1172`（变长阶段不 D2H，用满宽 + `select_index`）。

4. **窗口的 KV 写入/layout 用"固定满宽 + 上界"代替 host 计算**
   理由：我们 round 前必然要知道"窗口写到哪、读到哪"，如果这个用 host 算，就是每 round 一次 stall。SGLang 的套路是：kernel 内算 end offset（`eagle_utils.py:389-390`）、`seq_lens_sum` 可以是 None 然后过量分配（`triton_backend.py:1015-1019`）、mask 偏移在 device 上 cumsum（`triton_backend.py:1055-1060`）。
   出处同上；另见 `eagle_info.py:123-136`（device 端 pack read stream）。

5. **MoE 的薄 M 配置 + 一次 align/sort 两 GEMM 共用（先做诊断再做优化）**
   理由：384 experts / topk 6 / 窗口 6 token → M=36，仍然 ≤ E，**配置不换**，所以问题不在配置而在"是否每 token 重复做 dispatch、是否给每 expert 补到 BLOCK_M=16 的白行"。先量三个数：(a) 每 token 的 expert 权重字节数，(b) 每 token 的 padded rows（= `sum(ceil(cnt_e/16)*16)/tokens`），(c) align/sort kernel 的固定成本占比。然后照 `M<=E → BLOCK_M=16,N=32,G=1` 与"up/down 共用一次 sort"来改。
   出处：`triton_utils/fused_moe_triton_config.py:246-259, 326-339`；`triton_utils/moe_align_block_size.py:37-78`。

6. **（做完 1～2 之后再考虑）ragged/compact 窗口：按实际接受长度取 token 桶**
   理由：我们 accept 是 100%，短期无收益；一旦接受率下降，满宽白算就是主要浪费。上游是按 token 数建桶而不是按 bs（`ragged_verify.py:44-53`、`decode_cuda_graph_runner.py:724-728`）。
   出处：`speculative/ragged_verify.py`。

7. **（长期）把 layer 循环里的多 kernel 合并/做成 megakernel**
   理由：graph 录制解决"提交次数"，但解决不了 kernel 之间的启动/同步气泡。SGLang 对 draft loop 已经做到 megakernel（一次 launch 跑完 k 步）。我们每层 ~20 kernel，先按 1 修 attention、2 上 graph，再看是否还值得做。
   出处：`eagle_megakernel.py:1-40`；`eagle_worker_v2.py:700-703`。

**明确不建议照搬的**：
- 树验证（topk>1）整套机器 —— 我们是链式、100% 接受，`retrieve_*` 三件套 + 树掩码都是纯开销。
- 拒绝采样路径（`draft_probs` 的 `[bs, steps, vocab]` 存储，见 `eagle_info.py:34-36`）—— 显存与带宽都不划算。
- `speculative_moe_backend_context` 的双后端切换 —— 单卡单请求场景收益小。

---

## 6. 关键文件地图（便于二次核对）

| 关心的事 | 文件 | 关键行 |
|---|---|---|
| verify 的 batch 布局 / qo_indptr / KV 上界 | `speculative/eagle_info.py` | 91-121, 137-154, 426-476 |
| verify 前准备 + 是否走 graph | `speculative/eagle_utils.py` | 342-445（`:419` 设 TARGET_VERIFY，`:436-445` load_batch） |
| verify 主流程（采样前） | `speculative/eagle_worker_common.py` | 545-676（`:590`、`:658` `is_verify=True`、`:668-676` post-verify graphs） |
| accept / 采样 | `speculative/eagle_utils.py` | 558-760, 934-956；`kernels/ops/speculative/tree.py:37-60` |
| post-verify 整段成图 | `speculative/eagle_post_verify_graphs.py` | 1-10, 47-80, 93-120, 159-200 |
| TARGET_VERIFY 的 graph 抓取/pad | `model_executor/runner/decode_cuda_graph_runner.py` | 364-404, 432-441, 724-728, 883-929, 1573-1579 |
| 桶与 pad 规则 | `model_executor/runner/base_cuda_graph_runner.py` | 67-121, 180-197 |
| verify attention metadata | `layers/attention/triton_backend.py` | 589-660, 1008-1060, 1861-1900 |
| verify attention（SYCL/XPU 近亲） | `layers/attention/xpu_backend.py` | 487-511（topk≤1，无 mask）, 512-600（topk>1） |
| 树掩码模式与大小 | `layers/attention/verify_mask.py`；`kernels/ops/speculative/spec_tree.py` | 12-25, 28-40；46-49（默认档在 `speculative/eagle_utils.py:169-172`） |
| MoE 薄 M + 对齐 | `layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py`；`.../moe_align_block_size.py` | 246-259, 326-339；37-78 |
| draft 与 verify 的 kernel 分离 | `speculative/draft_worker_common.py`；`layers/moe/utils.py` | 70-95；1035-1048 |
| draft 多步一次 replay / megakernel | `speculative/eagle_draft_cuda_graph_runner.py`；`speculative/eagle_megakernel.py` | 517-546；1-40 |
| 本轮宽度定义 | `speculative/spec_utils.py` | 78-98 |

## 7. 复现取证（本机可跑，只读 GitHub）

```bash
S=/tmp/sgl && mkdir -p $S && cd $S
B=https://raw.githubusercontent.com/sgl-project/sglang/7b88f092392a4b673c39b2a6b66a18131cdfb60e
curl -sSL -o eagle_info.py        $B/python/sglang/srt/speculative/eagle_info.py
curl -sSL -o eagle_utils.py       $B/python/sglang/srt/speculative/eagle_utils.py
curl -sSL -o eagle_worker_v2.py   $B/python/sglang/srt/speculative/eagle_worker_v2.py
curl -sSL -o eagle_worker_common.py $B/python/sglang/srt/speculative/eagle_worker_common.py
curl -sSL -o eagle_post_verify_graphs.py $B/python/sglang/srt/speculative/eagle_post_verify_graphs.py
curl -sSL -o dcg.py               $B/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
curl -sSL -o bcg.py               $B/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py
curl -sSL -o xpu_backend.py       $B/python/sglang/srt/layers/attention/xpu_backend.py
curl -sSL -o triton_backend.py    $B/python/sglang/srt/layers/attention/triton_backend.py
curl -sSL -o verify_mask.py       $B/python/sglang/srt/layers/attention/verify_mask.py
grep -n "generate_attn_arg_prefill" -A 30 eagle_info.py | head -40
```
