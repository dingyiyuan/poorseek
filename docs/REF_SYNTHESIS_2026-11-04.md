# 上游 MTP 实现考察与结论（2026-11-04）

三个子智能体分别啃了 vLLM、SGLang、DeepSeek 官方的 MTP/投机解码实现，原始报告：
[ref-vllm-mtp.md](ref-vllm-mtp.md)、[ref-deepseek-mtp.md](ref-deepseek-mtp.md)、`ref-sglang-mtp.md`（进行中）。
本文只写**对本引擎有用的结论**。

## 0. 最重要的两条

1. **DeepSeek-V3 / V3.2-Exp 的参考推理代码里没有 MTP**：`DeepSeek-V3/inference/convert.py:53-54` 转换时直接
   跳过 `model.layers.61`（V3.2 同处 `:57`），`model.py:792` 只取末位 logits（`h[:, -1]`），`generate.py` 里
   `mtp|draft|verify|accept` 零命中。**第一方 MTP serving 代码是 DeepInfer（DFlash/DSpark，2026-10-20 开源）**，
   和本机 pack 的 `block_size 5 / markov_rank 256 / noise_token_id` 正好对上。
2. **DeepSeek 自己的实测：低并发下 MTP 是负收益**。变长验证窗在 60 并发 +5.4%，**16 并发 −1.5%**
   （`DeepInfer/doc/inference_deployment_qa.md:307-316`）。我们是 batch=1，比 16 并发还低一个量级。

## 1. 为什么 MTP 在我们这儿不赚（三方一致的算术）

| 来源 | 结论 |
|---|---|
| DeepInfer `v4_modules/mlp.py:119-136` | MoE 按 `total_q` 行一次分组 GEMM，小 m 用 `expected_m` 提示 |
| 子智能体算术 | 384 专家 top-6、6 token 窗 → 期望不同专家 `384(1-(1-6/384)^6) ≈ 34.6/36`，**重叠仅 4%** |
| 我们实测 | dense 21 → 5 ms/token（**摊薄成功**），但 MoE 每 token 贵 2×、注意力贵 1.8×、车道贵 3× |
| DeepInfer 实测 | 低并发 −1.5% |

**MTP 的收益只能来自 dense 与 attention 的权重摊薄；MoE 的权重流量近乎线性，摊不了。** 我们把这四个数
量出来之后，结论与上游状态一致：batch=1 时 MTP 不赚。

## 2. 真要接着做，唯一零风险的形态：变长验证窗

DeepInfer 不是固定 5 窗：`seqlens_q[i] = 1 + 本轮验证的 draft 数`，窗口大小由
`期望产出 / 实测耗时` 最大化决定（`csrc/engine/mtp_varlen.hpp:228-249`，在线成本模型 `:48-131`）。
最坏情况退化成 1，即**纯 decode，等于我们现在的 23.9 tok/s**。

对我们是零风险但也几乎零收益：按我们的实测（T=4 15.1、T=5 19.1 vs 不开 23.2），这个判据会永远选 1。

## 3. vLLM 那边真正能搬的三条（与 MTP 无关，对单 token 解码也有用）

1. **k+1 个 query 是一条扁平 token 流**，attention 把 "1+k" 归入 decode 桶（`attention/backend.py:569-596`），
   verify 不掉进 prefill/varlen kernel。我们的窗口路径是独立的一套内核，正是效率低 1.8× 的原因之一。
2. **回滚零成本**：KV 槽位是 position 的纯函数（`worker/block_table.py:153-179`），被拒 token 的 KV 不截断不清零。
3. **tiny per-expert MoE**：`num_tokens*top_k*4 <= num_experts` 时跳过 block 对齐、每 (token,expert) 读一次权重
   （`fused_moe.py:1614-1675`）；但块量化权重与 host-resident 权重被排除（`:1633-1645`）——**这两条例外我们都踩**，
   所以只能抄「分组 + M tile=16（`:1413-1414`）+ `GROUP_SIZE_M=1`（`:1432-1433`）」。

## 4. 结论与建议

- **batch=1 的服务上，MTP 不值得继续投入**：上游（DeepSeek 自己）在低并发也是负收益，我们的实测与之一致。
  生产继续 `--dspark off`。
- 真正值得做的是 vLLM 建议的第一条：**压每层的提交次数**。它同时改善单 token 解码（主机 32 ms ≈ GPU 计算 32 ms，
  §12）和窗口路径（每层 7 ms 主机时间，§26–27），是唯一对两条路都有效的杠杆。
- 若将来做多并发（>16 请求），MTP + 变长窗才有意义；那时再按 DeepInfer 的形态实现。

---

## 5. SGLang 补充（commit `7b88f09`，2026-11-03）

1. **verify 窗口 = 一个 extend 批，不是 k+1 次 decode**：每请求固定 k+1 行平铺，`qo_indptr = arange(step=k+1)`，
   KV 侧 `seq_lens += k+1`，方法名就叫 `generate_attn_arg_prefill`（`speculative/eagle_info.py:91-121`）——
   dense 层一次 GEMM 吃 k+1 行。**XPU 后端在 topk==1 时不传任何 mask**：`cu_seqlens_q=arange(step=N)`、
   `cache_seqlens=seq_len+N`，纯因果（`layers/attention/xpu_backend.py:487-511`）。我们是链式 100% 接受，
   完全可以走这条无 mask 路径；现在 0.9 → 1.6 ms/层/token 说明**窗口内 KV 没有复用**。
2. 窗口也吃 CUDA graph：`capture_forward_mode=ForwardMode.TARGET_VERIFY`、宽度固定 `captured_req_width=k+1`
   （`model_executor/runner/decode_cuda_graph_runner.py:364-380`）。对应我们 40 层 × ~20 kernel 的 800 次提交。
3. accept/sampling 全 device：greedy 下整窗 argmax + 一个 `verify_tree_greedy` 原地写 predicts/accept_index
   （`kernels/ops/speculative/tree.py:37-60`），结果一块 buffer 一次 broadcast。
4. MoE 薄 M：`M<=E` 时 BLOCK_M=16/N=32/G=1（`fused_moe_triton_config.py:246-259`）；**单 token M=6 与窗口 M=36
   都 ≤384 → 配置不变**，所以「MoE 每 token 贵 2×」不是配置问题，得先量每 token 的 expert 权重字节 + padded rows。

## 6. 于是可以提前把 MTP 的收益上界算死

SGLang 的算术 + 我们的实测，两边一致：**能摊薄的只有「每轮读一次」的权重**（dense / shared expert / lm_head）；
routed expert 的流量本质上是 per-token（6 token 窗最多触达 36 个 expert，对 6 个单 token 的 6× 正好抵消 6× token）。

按 T=1 实测阶段表（39.45 ms/token）推窗口的理论下限（假设窗口路径与 T=1 一样高效、且 attention 走 extend 批）：

| 项 | T=1 实测 | MTP 窗口的理论下限 |
|---|---:|---:|
| 注意力（KV 与权重都摊薄） | 15.9 ms/token | **~2.7 ms/token** |
| routing + shared expert（权重读一次） | 6.2 | ~1.0 |
| routed experts（摊不了） | 9.3 | **9.3** |
| CPU 车道等待 | 8.0 | ~6 |
| 合计 | 39.4 | **~19 ms/token ≈ 52 tok/s** |

**上界 ≈ 50 tok/s，前提是窗口路径的注意力真的走 extend 批（KV 复用）且主机侧不再串行。**
我们现在是 48 ms/token（19 tok/s），离这个下限还差 2.5×，差的正是「窗口路径每个 token 比 T=1 贵 1.8–2×」。

---

## 7. 一条重要的自我更正：主机时间不是「发射成本」，是 GPU 的节流

前面（§12、§26–27）我一直把「主机 32 ms/token、每层 7 ms」当成「提交次数太多」。这一轮把它证伪了：

| 证据 | 结果 |
|---|---|
| 车道线程数 `--cpu-threads 2` vs 默认 24 | qkv 段 8.13 vs 7.03 ms（**没有变好**，车道还更慢） |
| 车道留核 `STRATA_DS4_LANE_FREE=2/8/12` | 19.27/19.17/18.93 tok/s（无差别） |
| 专家数 `--experts 1`（只缩小 GPU 的 MoE） | qkv 段 **7.37 → 2.81 ms**（主机时间随 GPU 工作量掉） |
| 层边界（`f_pre` 拷贝 + stage-5 标记） | **0.02 ms**（不是拷贝在等） |
| 启动路径 `strata_ds4_launch` / `dpct::kernel_launcher` | 无 `wait()`；`DS4_SYNC_LAUNCH` 未设，队列是异步的 |

**结论：主机并不是在「忙发射」，而是被 in-order 队列反压——Level Zero 的提交在队列变深时把主机节流到 GPU 的速率。**
所以「主机时间 ≈ GPU 时间」是同义反复，不是两个独立瓶颈；**减少提交次数本身不会提速，只有减少 GPU 工作量才会**
（`--experts 1` 那一行就是证明：GPU 工作量一降，主机时间跟着降）。

这条更正同时否掉了我在 §6 里给的建议（「先压提交次数」）：那是 vLLM/SGLang 的建议，前提是它们有 CUDA graph
把提交压到接近 0；**我们没有图捕获，提交是被 GPU 节流的，压提交数不改变 GPU 的工作量。**

## 8. 于是整件事收敛成一句话

**这台机器上解码（无论开不开 MTP）的唯一杠杆是「每个 token 的 GPU 工作量」。**
- 单 token 路径：39 ms/token，其中稠密 21 ms 已贴带宽（§21），剩下是 m=1 内核的延迟/占用率（§17）。
- MTP 窗口：稠密摊薄到 5 ms/token（真收益），但注意力 1.8×、MoE 2× 抵掉，净 −20%。
- MoE 摊不了是算术（4% 专家重叠），不是实现。

要真正拿到 MTP 的 ~50 tok/s 上界，必须把窗口路径的注意力与 MoE 内核的 per-token 效率做到与 T=1 持平——
那是 3–4 个内核的改造工程（每个编译 15 分钟、要逐位对拍），不是调参能到的。

---

## 9. 最终定位：主机自己的工作只有 0.7 ms/层，MTP 的代价 100% 在 GPU

在每层第一个提交之后加了一个标记，把「等上一层 GPU」和「本层真正的主机工作」分开：

| 段（ms/层） | T=1 | T=5（窗口） |
|---|---:|---:|
| **wait（层首提交时被队列反压）** | **0.51** | **6.59** |
| boundary | 0.02 | 0.02 |
| qkv（真实工作） | 0.18 | 0.21 |
| comp+idx | 0.03 | 0.04 |
| sparse attn | 0.04 | 0.04 |
| attn tail | 0.00 | 0.00 |
| routing | 0.02 | 0.02 |
| shared | 0.16 | 0.10 |
| rest | 0.09 | 0.20 |
| **主机自身工作合计** | **~0.54** | **~0.63** |

**主机在两种窗口下干的活几乎一样（0.54 vs 0.63 ms/层）。差的 6 ms 全是「等 GPU」。**
也就是说：**GPU 每层时间 T=1 约 1.0 ms，T=5 约 7.0 ms——5 倍的 token 换 7 倍的 GPU 时间。**

所以 MTP 的全部代价都在 GPU 侧（T>1 的 MoE 与注意力内核效率），主机侧没有任何可省的东西。
这与 §7 的更正一致，也解释了为什么前面所有「减少派发 / 减少同步」的尝试都无效。

### 9.1 剩下的唯一工作

把窗口路径的 GPU 每层时间从 7.0 ms 压到 1.5–2.0 ms（= T=1 的 1.5–2 倍而不是 7 倍），需要：
1. 注意力走 SGLang 的 extend 批形态（KV 在窗口内只读一次，无 mask，topk==1）；
2. MoE 按 expert 分组、每个 expert 的权重只读一次（现在是 36 个 (token,expert) 对各自读）；
3. 验证窗口的 KV 行（`ext_to_ring` / `ring_to_ext`）不重复搬运。

这三件都是内核改造，每件都要编译 15 分钟 + 逐位对拍；按当前节奏一个 session 能做完 1–2 件。

---

## 10. 终于指到那一个内核了：窗口的 MoE 每层 6.23 ms，是整轮的 82%

`DS4_PROF` 的窗口表原来把「每层的标记」除以「token 数」，读出来偏大 40/5.8 ≈ 6.9 倍。按标记数重新摊平后
（原始累加值 ÷ 层数：T=5 是 33 轮 × 40 层 = 1320，T=1 是 192 × 40 = 7680）：

| 阶段（ms/层） | T=1 | T=5 窗口 | 倍数 |
|---|---:|---:|---:|
| between layers | 0.037 | 0.186 | 5.0× |
| attn tail (o_fin+wo) | 0.163 | 0.291 | 1.8× |
| routing+shared | 0.161 | 0.463 | 2.9× |
| experts gate/up | 0.120 | （并进 down+post） | — |
| cpu-lane wait | 0.222 | （并进 down+post） | — |
| **down+post（T>1 时整段 MoE）** | **0.245**（0.125+0.120） | **6.233** | **25×（每 token 4.4×）** |
| attn proj (norm+q_a/kv) | 0.179 | 0.384 | 2.1× |
| compressor+indexer | 0.020 | 0.026 | 1.3× |
| sparse attn | 0.044 | 0.074 | 1.7× |
| **合计** | **1.071** | **7.659** | **7.2×** |

**注意注意力那几格其实摊得很好（1.3–2.1×，5 倍的 token 只涨 2 倍）；唯一爆掉的是 MoE：6.23 ms/层 = 249 ms/轮，
占整轮 304 ms 的 82%，而按「摊不了但也不该变贵」的下限它应该只有 ~1.4 ms/层。**

也就是说：**把窗口的专家内核做到与 token 路径同等效率，一轮就是 304 − (6.23−1.4)×40 ≈ 111 ms → 52 tok/s。**
正好就是目标值，而且只需要动一处。

### 10.1 为什么是它：两套内核

- T=1 走 `moe_gate_up` / `moe_down`（`iq::gate_up` / `iq::down`，为 m=1 手写的 token 内核，实测 0.245 ms/层）；
- T>1 走 `win_experts` → `set_gate_up` / `set_down`（两段式 GEMM，为预填充的大 m 设计；`DS4_EXP_2PHASE_MINROWS`
  默认 96 才接管，而验证窗口每专家只有 ~1.2 个 token）。

m≈1 时跑「为大 m 设计的两段式」→ 4.4 倍开销。**下一步就是让 T≤8 的验证窗口改走 token 内核**
（或给 `win_experts` 加一条 m 很小的快路径），然后用 `--dump-logits` 对拍、再测一轮。

---

## 11. 重要更正：MTP 不是 −20%，512 token 下已经打平甚至略胜

之前所有 MTP 结论都基于 **192 token**（≈33 轮）的跑次，那里前几轮冷专家缓存主导，把 MTP 拖成 −20%。
用 **512 token**（≈88 轮）重测，同一二进制、相邻三次：

| 跑次 | 配置 | 解码 |
|---|---|---:|
| off1 | `--dspark off` | 23.91 tok/s |
| **on1** | `--dspark on` | **24.37 tok/s（+1.9%）** |
| off2 | `--dspark off` | 23.73 tok/s |

同时拿到干净的每层表（标记数已按层摊平）：

| 阶段（ms/层） | T=1 | T=5 窗口 |
|---|---:|---:|
| down+post（T>1 时整段 MoE） | 0.10（+gate/up 0.13） | **4.89** |
| routing+shared | 0.15 | 0.40 |
| attn proj | 0.17 | 0.36 |
| attn tail | 0.16 | 0.27 |
| sparse attn | 0.06 | 0.10 |
| comp+idx | 0.02 | 0.02 |
| between | 0.04 | 0.18 |
| **合计** | **0.99** | **6.23** |

**窗口的 MoE 一层 4.89 ms，占整轮 6.23 ms 的 78%。** 若把它压到 token 路径的水平（每 token 0.23 × 5.8 ≈ 1.4 ms/层），
一轮就是 (6.23 − 4.89 + 1.4) × 40 ≈ **110 ms → 5.8 token / 110 ms ≈ 53 tok/s**，正好是目标值。

### 11.1 下一步（明确、单一）

验证窗口（T≤8）的专家改走 **token 内核** `moe_gate_up` / `moe_down`（m=1 手写、T=1 路径实测 0.23 ms/层），
而不是 `win_experts` 的两段式 `set_gate_up` / `set_down`（为预填充的大 m 设计，每专家 ~1.2 token 时亏 4.4 倍）。
改完先 `--dump-logits` 逐位对拍，再测 512 token 的 on/off。


---

## 12. 更正 §11：窗口的 MoE 不是「两段式内核的锅」

用零代码开关对照（512 token，相邻跑次）：

| 配置 | down+post ms/层 | 总 ms/层 | 解码 |
|---|---:|---:|---:|
| 默认（light 路径可用） | 5.06 | 6.39 | 23.39 tok/s |
| `STRATA_DS4_NO_LIGHT=1`（强制走 GEMM tiles） | 4.98 | 6.30 | 23.66 tok/s |

**完全没差别** → 窗口的专家根本不走 `light_gate_up` 那套（或走了也不构成成本）。`--dspark-block 15` 也被 pack 的
drafter 配置忽略（两轮都是 5.95 token/轮），所以「16 行 padding」的猜测也测不了。

再看引擎自报：窗口里 `ds4 dspark cpu: lane 97–114 ms/window` = **2.4–2.9 ms/层**，而 down+post 总共 4.98 ms/层。
**也就是窗口的 MoE 成本大约一半是 CPU 车道（37% 的专家落车道），一半才是 GPU 专家内核。**

### 12.1 修正后的结论

- 窗口 MoE 4.98 ms/层 = 车道 ~2.5 + GPU 专家 ~2.5。
- 车道占比 37%（token 路径 9%）的根因：一个窗口 36 次挑选摊到更多不同专家，而 VRAM 每层只有 ~35 个槽位。
- 要把它压下去，优先「降低窗口内的车道占比」（按窗口路由造 profile / 提高 VRAM 命中），而不是改专家内核。
  按 §11 的算法，MoE 若降到 token 路径的水平（~1.4 ms/层），一轮 110 ms → 53 tok/s 的目标仍然成立。

---

## 13. 窗口路由的 profile：命中率能到 97.9%，但实测反而慢一倍（否决）

用 MTP 自己跑的 512 token 窗口 trace（3480 条 T=6 记录、126,480 次挑选）算出的每层热度：
**top-40/层覆盖 97.9% 的窗口挑选**，看起来应该把车道占比从 37% 压到 ~2%。实测（512 token，相邻三次）：

| 配置 | 解码 | down+post | 车道 |
|---|---:|---:|---:|
| 无 profile | 23.87 / 23.73 tok/s | 4.92 / 4.95 ms/层 | 73.5 / 74.8 ms/窗口 |
| `--expert-profile`（窗口热度） | **12.53 tok/s** | 4.34 ms/层 | 64.5 ms/窗口 |

**车道确实降了（74 → 64 ms/窗口），但整轮翻倍变慢**：prof 标记数从 2792 涨到 6424，
即每轮接受数从 5.95 掉到约 3.2——**放置一变，drafter 的草稿质量就崩了**（它的专家也被重新排了）。

结论：在 MTP 下动专家放置是负收益，**否决**。

## 14. 到这里 MTP 的所有「不动内核」的路都试完了

| 尝试 | 结果 |
|---|---|
| 车道线程/留核（`--cpu-threads`、`LANE_FREE`） | 中性 |
| 两段式 / light 内核（`DS4_EXP_2PHASE`、`STRATA_DS4_NO_LIGHT`） | 中性 |
| 窗口大小（`--dspark-block`） | pack 的 drafter 配置覆盖，无效 |
| `DS4_WIN_HOSTWAIT`、`h_part` 拷贝、`f_pre` 拷贝 | 中性或已回退 |
| 窗口路由 profile（命中 97.9%） | **−47%**（drafter 质量崩） |
| 稠密权重降精度 | 用户明确不做 |

剩下的只有「把窗口的专家计算内核 + 车道协同重做」这类真正的内核工程，收益上界仍是 §11 的 ~53 tok/s。
在那之前，**生产维持 `--dspark off` 是正确选择**：512 token 下 on/off 已经打平（24.37 vs 23.73–23.91），
但窗口对冷缓存更敏感，长上下文/首轮会吃亏。
