# DeepSeek 官方 MTP / 推测解码实现考察（面向外部引擎借鉴）

核对时间：2026-11-04。所有引用为仓库内相对路径与行号；三个仓库：
`DeepSeek-V3`（main, `8868499`）、`DeepSeek-V3.2-Exp`（main）、`DeepInfer`（main，2026-10-20 开源）。
DeepGEMM / DeepEP 以 `DeepInfer/vendor/` 锁定的 commit 为准，另参考各自 main 的 README。

---

## 0. 一句话结论

DeepSeek 的 MTP 语义有**两代**，而且**第一方推理代码直到 2026-10 才开源**：

- **V3 / V3.2-Exp 的参考推理 demo 完全没有 MTP**。MTP 权重在权重转换时被**主动丢弃**，前向结构上也不产出多位置 logits。拿这两份 demo 当"MTP 参考实现"会走错路。
- 真正第一方的 MTP serving 实现在 **DeepInfer**，代号 **DFlash / DSpark**：默认 **5 个 draft**、一个**固定 stride 的 draft block（用噪声 token 填充）**、**Markov 修正头**、**置信度头**，以及最关键的——**变长验证窗**（不是固定 5）。

---

## 1. 事实核对：谁有 MTP 代码

| 仓库 | MTP 权重 | 推理代码中的 MTP |
|---|---|---|
| `deepseek-ai/DeepSeek-V3` | 有（14B，layer 61） | **无** |
| `deepseek-ai/DeepSeek-V3.2-Exp` | 有 | **无** |
| `deepseek-ai/DeepInfer` | 有 | **完整实现** |

证据：

- `DeepSeek-V3/inference/convert.py:53-54`：`if "model.layers.61" in name: continue` —— 转换时直接跳过 MTP 层。
- `DeepSeek-V3.2-Exp/inference/convert.py:57`：同一行逻辑，V3.2 也一样丢弃。
- `DeepSeek-V3/inference/generate.py`（185 行）与 `DeepSeek-V3.2-Exp/inference/generate.py`（186 行）中 `mtp|draft|verify|accept` **零命中**：纯单 token 循环。
- `DeepSeek-V3/inference/model.py` 中无 `mtp` / `MTPBlock` / `share_head` 任何符号。

**结构上也无法验证**：V3 `inference/model.py:792` 是 `h = self.norm(h)[:, -1]`，只取最后一个位置的 logits；V3.2-Exp `inference/model.py:908` 是 `logits = self.head(h[:, -1].float())`，同样只取末位。多 token 输入只服务于 prefill（V3 `model.py:789` 的上三角 mask），不会产出 k+1 个位置的 logits。要在这些 demo 上做验证循环，必须改模型代码，不只是改调度。

V3 的 MTP 只有**权重文档**，没有代码：`README_WEIGHTS.md:33-48`。

---

## 2. MTP 到底预测什么（问题 1）

### 2.1 V3 语义（我们当前 drafter 在模仿的那套）

参数构成见 `README_WEIGHTS.md:41-48`：

- `embed_tokens` —— 与主模型 Embedding **共享**
- `enorm` / `hnorm` —— 两个 RMSNorm，分别作用于"下一个真 token 的 embedding"和"上一层 hidden"
- `eh_proj` —— 2·d → d 的降维投影（就是两路 norm 结果 concat 后的融合）
- 一个与主模型同构的 Transformer block（layer 61）
- `shared_head` —— 与主模型 lm_head **共享**

规模：11.5B 独立参数、2.4B 激活（含共享的 0.9B embed + 0.9B head），见 `README_WEIGHTS.md:38-39`。

前向语义（DeepSeek-V3 技术报告 arXiv:2412.19437 的 MTP 一节，代码未开源，只能从参数形状反推）：

```
h'_k = eh_proj( concat[ RMSNorm(Emb(t_{i+k})), RMSNorm(h_i^{k-1}) ] )
h_k  = TransformerBlock_61(h'_k)
p_{i+k+1} = shared_head(h_k)
```

关键点：**输入是"下一个真 token 的 embedding"与"上一层 hidden"的拼接**——这是训练期 teacher-forcing 的并行形态，一层只多预测 **1 个** token。没有 noise token、没有 markov 头、没有置信度头。

### 2.2 V4.1-flash / DFlash 语义（2026 版，真正在服务里跑的）

- **draft 数 = 5**：`DEEPINFER_USE_NEXTN_DECODE` 默认 5（`deepinfer/environ.py:136`），官方文档口径一致："the draft model generates 5 tokens at a time"（`doc/inference_deployment_qa.md:303`）。与我们"5 token 窗"的设定同源。
- **噪声 token 的真正用途**：每请求构造一个长度 `dflash_block_size = max_MTP_value` 的 block，**第 0 列填 golden token，其余全部填 `noise_token_id`**（`deepinfer/model_executor/models/mtp/dflash.py:230-239`；noise id 来自 `cfg.dflash_cfg.dspark_noise_token_id`，`dflash.py:139`）。整个 block **一次前向算完**，之后按 `step_i :: block_size` 跨步切片逐步取样（`dflash.py:309`、`316-317`）。
  → 这不是"训练用的噪声"，而是**推理期的形状固定器**：反正要按固定 stride 走，未定位置就用噪声占位，算废了也不影响正确性。
- **输入特征不是 concat(Emb, h)**：而是把主模型**若干 target 层**的 hidden 拼接后投影——`projection = Linear(len(dspark_target_layer_ids)·d → d)` + `project_norm`（`deepinfer/model_executor/models/v4_modules/heads.py:100-101`，target 层 id 见 `dflash.py:146-153`）。比 V3 的"只吃上一层"信息量大得多。
- **Markov 修正头**（V3 没有的"额外头"）：最后一层 `shared_head` 是 `MarkovGPTHead` = 共享 lm_head + **一阶 markov 项**：`logits += markov_w2(Emb_markov(上一个 draft token))`（`heads.py:26-34`；应用处 `dflash.py:310-312`）。rank 由 `dspark_markov_rank` 配置（`dflash.py:155-156`）。
  → 语义上等于给 drafter 加了一个 **bigram 先验**，补偿"并行出 5 个 token 时彼此看不见"的问题——因为 5 个 draft 是并行算的，markov 项是唯一把它们串起来的信息流。
- **置信度头**：`AcceptRatePredictor(hidden_size + markov_rank)` → 每个 draft token 一个接受概率（`heads.py:105`；实现 `dflash.py:46-79`），**逐步独立温度**（`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 必须恰好 nextn 个值，否则报错，`environ.py:142-148`），用融合 kernel `confidence_head_fused` 算（`dflash.py:73-79`）。

**没有"多个独立 head"**：draft 的 5 个 token 由同一个 block + 同一个共享 head 并行产出，靠 markov 项与位置 stride 区分，不是 5 套 head。

---

## 3. 验证窗怎么批（问题 2）—— 最有借鉴价值的部分

核心事实：**每请求的验证长度不相等**（varlen），不是"每人固定 5 个"。

- **预算定义**：`seqlens_q[i]` = 第 i 个请求本轮 query 数 = `1 + 参与验证的 draft 数`；`total_q = Σ seqlens_q[i]`（`csrc/engine/mtp_varlen.hpp:150-156`、`263-266`）。
- **平铺布局（契约级，直接可抄）**：`accepted_token_ids` 是 `[total_q]` 一维数组，每请求占 `seqlens_q[i]` 个位置，**前 `seqlens_q[i]-1` 个是 draft 位（`-1` 表示未接受），最后一个位置是 main token 位（不使用其接受结果）**（`csrc/engine/scheduler.hpp:873-875`）。后端按此契约抽取通过的前缀（`scheduler.hpp:955-967`：`num_draft_i = seqlens_q[i] - 1`，逐个取、遇到 `-1` 截断）。
- **容量**：`max_num_decoding_tokens = max_num_seqs * (1 + nextn)`（`deepinfer/config.py:224`），按最大窗预留，实际用 `seqlens_q` 收缩。
- **attention 布局**：draft 走 `decode_mla_mqa_sparse`，带 `dflash_kv_index`，与主模型 decode **同一 kernel 家族**（`deepinfer/model_executor/models/v4_modules/attn.py:994-1007`）。所以 k+1 个 query 不是"一次特殊的 batched attention"，而是**一次 `total_q` 行的 varlen MLA**，每行按自己的窗口索引取 KV。**不需要 draft token 之间的 causal mask**，因为每个 draft 位置的 KV 已经在 draft forward 里写好了。
- **draft KV 是 scratch，不是 cache**：注释写明 "Draft keys live for one decode step and are overwritten by the next step"（`attn.py:975`），写入 `dflash_newx_slot_mapping_store`（`attn.py:989`）。draft 的 KV 生命周期只有一步，不需要淘汰/驱逐逻辑。
- **KV 写不需要按接受结果屏蔽**（很值得借的技巧）：主模型 forward 时把**全部** draft 位置的 main-token key 无条件写入（接受结果此时还不存在），因为"将被拒绝的 draft 位置，正是 `forward_drafts` 随后写 draft key 的位置，在任何读取之前就会被覆盖，因此无需按接受结果屏蔽"（`dflash.py:204-209` docstring 原话）。**验证结果与 KV 更新之间没有数据依赖**。
- **MoE 侧**：`total_q` 行一起进一次分组 GEMM（见 §4），不按请求切分。

---

## 4. MoE 在极少 token 下怎么不塌（问题 3）

- **只分组 M 轴，N/K 固定** → 一个 kernel 覆盖所有 expert（DeepGEMM `README.md:78`）。
- **按对齐 padding**：每个 expert 段必须对齐到 GEMM 的 M block（`get_mk_alignment_for_contiguous_layout()`，DeepGEMM `README.md:78`）。DeepInfer 显式设置该对齐值：`set_mk_alignment_for_contiguous_layout(get_theoretical_mk_alignment_for_contiguous_layout())`（`deepinfer/model_executor/models/v4_modules/mlp.py:250-252`），并让 dispatch 用**同一个对齐值**（`mlp.py:477`）——两边一致就不用再搬一次数据。
- **偏移用前缀和**：`m_grouped_fp8_fp4_gemm_nt_contiguous(..., psum_num_recv_tokens_per_expert, use_psum_layout=True, expected_m_for_psum_layout=expected_m)`（`mlp.py:119-128`）。**`expected_m` 就是"小 m 提示"**：把预期每 expert 的 token 数告诉 kernel，让它据此选 block / 安排 occupancy，而不是假设大 M。
- **计数不在 CPU**：decode 用 psum layout，host 侧不需要知道每 expert 收多少 token。只有"CPU 不知道计数 **且** 开 CUDA graph"时才需要 masked 分组 GEMM（`m_grouped_fp8_gemm_nt_masked`，DeepGEMM `README.md:82-86`）；mask 版本按 mask 只算有效部分，masked 路径的典型输入就是 DeepEP low-latency kernel 的输出（`README.md:86`）。
- **SwiGLU 融合 topk weight**：`swiglu_forward_and_per_token_cast(..., psum_num_tokens_per_expert=..., topk_weights=topk_weights, ...)`（`mlp.py:436-442`）——激活、per-token cast、**路由权重相乘**在一次里做完，down_proj 输出即为加权结果，省掉一次 scatter-multiply。
- **shared expert 折进 combine 的 bias**：shared 在受限 SM 上跑（`limit_num_sms(DEEPINFER_USERBUFFER_COMM_SMS)`），其输出作为 combine 的 `bias` 直接进 epilogue（`mlp.py:602-609`；`combine(..., bias=shared_hidden_states, ...)` 见 `mlp.py:505`）——少一个 kernel、少一遍大张量读写。

---

## 5. 用计算换同步（问题 4）

这是 DeepInfer 与"教科书推测解码"差别最大的地方，也是我们最该学的：

1. **提案池 + slot 0 padding**：`MTPInfoPool` 按 scheduler slot 存 draft 提案，`pool_rows = (max_num_seqs + 1) * nextn`，**slot 0 专门保留给 padding**（`deepinfer/model_executor/layers/mtp_info_pool.py:14-52`；`write_slots` 为 0 的行直接跳过，`:54-67`）。GPU 侧 sampler 直接把提案写进池（`dflash.py:339-344`），CPU 下一轮按 slot 读 —— **不为拿 proposal 做 host 同步**。
2. **双轮滞后调度**：`MtpState` 存 `accept_rates`（**已累乘**）与 `prev_accept_rates`（上上轮），配 `schedule_cnt` / `process_cnt` 两个计数器判断"这份 accept_rate 属于第几轮"（`csrc/engine/sequence.hpp:118-131`）。调度**永远用 N-2 轮的率算总预算、N-1 轮的率分每请求长度**（`mtp_varlen.hpp:160-207`，两个队列 `acc_queue_for_bsz` / `acc_queue_for_seqs_q`），因此**不需要等本轮结果**。
3. **CPU 只定总量，细分交给 GPU**（官方原话，`doc/inference_deployment_qa.md:324`）："When `schedule(N+1)` runs, the confidence from step N has not been returned yet, so CPU scheduling only fixes the total budget; the per-request lengths are recomputed on the GPU by a TileLang kernel (cumulative product, global sort for top-k, with a fused remap scatter), and the result is written back to a pinned buffer."
   Kernel 实现见 `deepinfer/model_executor/layers/tilelang_ops/asc/schedule_mtp_varlen.py:16-121`：先对 nextn 个置信度做**累积乘积**（`:60-66`，`acc = acc * prev_confidence_rates[prev, j]`）——这就是"必须验证前缀"的正确边际概率；再对 `(seq, token)` 全局排名（`:98-111`，带 index 的 tie-break）；`rank < total_q - num_seqs` 者胜出，`seqlens_q = 1 + 胜出数`（`:113-119`）。
4. **固定形状换 graph**：整个 DFlash decode —— broadcast、固定 stride 的 metadata、draft 层、sampling（含池写）—— 作为**一个 CUDA graph 单元** capture/replay（`dflash.py:353-357`、`385-386`）。
5. **每轮固定算满 nextn 个 draft**（噪声占位），验证几个由预算后定 → 形状恒定，graph 才能捕。**用多算的 draft 换掉形状可变性**。

顺带：`MTPVarlen` 里还有一个**在线自校准成本模型**：`time = baseline_ms + slope_ms * graph_size + sampler_ms`，其中 slope 是 `(forward_ms - baseline_ms) / graph_size` 的两层中位数在线拟合（5000 样本窗、每 graph size 至少 10 样本），见 `mtp_varlen.hpp:48-131`、`139-148`、`228-249`。

---

## 6. 官方实测：变长验证到底赚多少（可直接对标）

`doc/inference_deployment_qa.md:307-316`，Open-PerfectBlend，EP64/DP64：

| 并发 (seqs/rank) | 固定 5 验证长度 | 变长验证长度 | 固定 5 接受长度 | 变长接受长度 | 变长相对固定 5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 4.44 | 3.32 | 3.24 | **−1.5%** |
| 32 | 5 | 3.65 | 3.25 | 3.17 | +1.6% |
| 48 | 5 | 3.94 | 3.30 | 3.26 | +3.6% |
| 60 | 5 | 3.58 | 3.30 | 3.21 | +5.4% |
| 72 | 5 | 3.24 | 3.29 | 3.12 | +6.7% |
| 96 | 5 | 3.24 | 3.30 | 3.15 | +11.0% |

要点：
- 平均验证长度 5.00 → 3.24–4.44（**省掉约 30% 验证算力**），接受长度几乎不变（3.29 vs 3.20）。
- **低并发（16）时变长也是负收益（−1.5%）**。这与我们 batch=1 的观测同向：低并发下"省验证算力"救不了固定开销。
- 生产点（~60 seqs/rank）+5.4%。

---

## 7. 给我们引擎的清单（按期望收益排序）

> 前提回顾：单卡 Arc Pro B70 / SYCL，batch=1，40 层，384 expert top-6，1.75-bit IQ，dense 6.2 GB/token。
> 当前：无 MTP 39 ms/token（23.9 tok/s）；5 token 窗 304 ms/round → 19 tok/s（dense 摊销到 5 ms/token，但 attention 1.8×、MoE 2×）。

**1) 把"固定 5 窗"换成本轮预算驱动的变长窗，允许本轮验证 0 个 draft。**
参考 `mtp_varlen.hpp:150-267` + `schedule_mtp_varlen.py:16-121`。
理由：他们调度的目标函数就是"期望产出 / 实测耗时"（`mtp_varlen.hpp:228-249`），窗口会自动退到 1（等价纯 decode）。我们的实测 19 < 23.9 tok/s 说明当前窗口**必然**会被这个判据否掉——这不是放弃 MTP，而是让 MTP 只在净正收益时开启。**最坏情况等于无 MTP，风险为零**，所以排第一。

**2) 加一条按"本轮 query 数"标定的成本曲线 + 在线斜率自校准。**
参考 `mtp_varlen.hpp:48-131`（两层中位数）、`139-148`（`update_runtime`）、`228-249`（吞吐比较）。
理由：没有成本曲线就只能拍固定 5。我们无 graph capture，但"submission 数 + 实测 ms"同样能拟合出一条；slope 项正好对应我们 per-layer hand-off 的固定开销，能直接量化"每多 1 个 query 贵多少"。

**3) 用置信度/边际收益决定"验证几个"，而不是"能不能接受"。**
参考 `heads.py:105`、`dflash.py:46-79`（置信度头）、`sequence.hpp:123-126`（累乘 vs 原始）、`schedule_mtp_varlen.py:60-66`（累积乘积）。
理由：我们接受率 100%，所以置信度头对我们**不是筛坏 draft**，而是**证明窗口可以缩小**。没有置信度头时可退化为纯边际收益：多验证 1 个 token 的增量耗时 vs 增量产出。注意必须用**累积乘积**（前缀语义），不能直接用第 j 个 token 的独立概率。

**4) 消灭 per-layer host hand-off：整段 draft 一次提交，验证段一次提交。**
参考 `dflash.py:353-357`（整段作为一个 graph 单元）、`mtp_info_pool.py:54-78`（GPU 写池、CPU 下轮读）、`qa.md:324`（CPU 只定总预算）。
理由：我们"~20 kernel/层 + 每层 host hand-off"是真正的天花板；DeepSeek 的做法是**把决策移出热路径**（CPU 只做粗调度，细分在 GPU kernel 里算，结果写 pinned buffer）。SYCL 上等价物是：先做到"每层一个 submission、参数常驻、无 per-layer host 决策"，再谈 graph capture。

**5) 认清收益只可能来自 dense 权重，别指望 MoE 摊销。**
参考 `mlp.py:119-136`（`total_q` 行一次分组 GEMM）、`attn.py:994-1007`（`total_q` 行一次 varlen MLA）。
我们自己的算术：384 expert、top-6、窗内 6 个 token → 36 个 (token, expert) 槽，期望不同 expert 数 `384·(1-(1-6/384)^6) ≈ 34.6`，**重叠率仅约 4%**。所以 **MoE 权重流量随 token 数近乎线性、不会摊薄**；MTP 的收益只能来自 dense（已观测 21 → 5 ms/token）与 attention/KV。把验证窗的成本记在 attention 与专家权重流量上，据此判断窗口上限。

**6)（小但便宜）draft KV 用 scratch + 覆盖写，不做拒绝屏蔽。**
参考 `attn.py:975`、`dflash.py:204-209`。
理由：去掉"验证结果 → KV 写"的数据依赖，少一次条件分支/一个 kernel。

**7)（小但便宜）shared expert 输出折进 combine 的 bias。**
参考 `mlp.py:602-609`、`505`。
理由：少一个 kernel、少一遍大张量读写——对 6.2 GB/token 的带宽敏感引擎收益直接。

---

## 8. 明确不要照抄 / 存疑

- **别拿 V3 的 `convert.py` 当 MTP 参考**：`convert.py:53-54` 明确丢弃 MTP 权重；V3/V3.2 demo 的 `model.py` 只出末位 logits（`792` / `908`），改调度无法得到验证能力。
- **`expected_m` 不是万能加速开关**：常规 decode 路径传的是 `expected_m=None`（`mlp.py:597`），只在 mega MoE 融合路径才用（`mlp.py:549-559`）。它的价值在"给出提示"而不是"自动变快"。
- **变长验证在低并发是负收益**（`qa.md:311`，16 并发 −1.5%）。必须按我们 batch=1 的曲线判定，不能直接引用 +5.4% 的生产数字。
- **DFlash 的 5 个 draft 是并行出的**，靠 markov 项互相串联；如果我们改成串行自回归 drafter，markov 项的收益就不成立了。
- V3 的 MTP 前向公式是从参数形状 + 技术报告反推的（**DeepSeek 从未开源 V3 MTP 推理代码**），引用时请标注为"报告语义"而非"代码语义"。

---

## 附：关键文件索引

| 主题 | 路径 |
|---|---|
| 验证窗预算调度（CPU） | `DeepInfer/csrc/engine/mtp_varlen.hpp` |
| 调度器 / accept 契约 | `DeepInfer/csrc/engine/scheduler.hpp:873-875`, `:955-967` |
| MTP 状态机 | `DeepInfer/csrc/engine/sequence.hpp:118-131` |
| 验证窗预算（GPU kernel） | `DeepInfer/deepinfer/model_executor/layers/tilelang_ops/{asc,cuda}/schedule_mtp_varlen.py` |
| DFlash drafter 主体 | `DeepInfer/deepinfer/model_executor/models/mtp/dflash.py` |
| head / markov / 置信度头 | `DeepInfer/deepinfer/model_executor/models/v4_modules/heads.py` |
| draft attention | `DeepInfer/deepinfer/model_executor/models/v4_modules/attn.py:884-1008` |
| MoE 分组 GEMM | `DeepInfer/deepinfer/model_executor/models/v4_modules/mlp.py:109-137`, `:406-459` |
| 提案池 | `DeepInfer/deepinfer/model_executor/layers/mtp_info_pool.py` |
| 官方设计问答（含实测表） | `DeepInfer/doc/inference_deployment_qa.md:299-334` |
| V3 MTP 权重结构 | `DeepSeek-V3/README_WEIGHTS.md:33-48` |
| V3 MTP 被丢弃 | `DeepSeek-V3/inference/convert.py:53-54` |
| DeepGEMM 分组 GEMM / masked | `DeepGEMM/README.md:76-86` |
