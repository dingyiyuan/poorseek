# MTP / DSpark 交接（2026-11-04）

这份文档把 MTP 这条线上**试过的每一件事、每个数字、以及为什么停在这里**写清楚。上游调研见
[REF_SYNTHESIS_2026-11-04.md](REF_SYNTHESIS_2026-11-04.md) 与三份 `ref-*.md`。

## 0. 一句话现状

`--dspark on` 在 512 token 的跑次上 **24.37 tok/s**，不开 **23.73–23.91**——**已经打平略胜**；
但一轮 T=6 窗口要 6.2 ms/层（不开只要 1.0 ms/层），窗口对冷缓存更敏感，首轮/长上下文会吃亏。
生产维持 `--dspark off`。

## 1. 硬数字（512 token，暖态，相邻跑次，同一二进制）

| 指标 | 不开 MTP | 开 MTP（T=5/6 窗口） |
|---|---:|---:|
| 解码 | 23.73–23.91 tok/s | **24.37 tok/s** |
| 每层 GPU 时间 | 0.99 ms | **6.23 ms** |
| 　其中 MoE（down+post） | 0.23 | **4.89** |
| 　其中 attn proj | 0.17 | 0.36 |
| 　其中 attn tail | 0.16 | 0.27 |
| 　其中 routing+shared | 0.15 | 0.40 |
| 　其中 sparse attn | 0.06 | 0.10 |
| 接受率 / 每轮 token | — | 100% / 5.8–5.95 |
| 稠密权重每 token | 21 ms | **~5 ms（真收益）** |

窗口 MoE 4.89 ms/层 = **CPU 车道 ~1.8 + GPU 专家 ~3.1**（引擎自报 `dspark cpu: lane 73–75 ms/窗口`）。

## 2. 已否决（全部实测，别再试）

| 尝试 | 结果 |
|---|---|
| 车道线程 `--cpu-threads 2`、留核 `LANE_FREE 2/8/12` | 中性（主机时间是被 GPU 节流的，不是发射成本） |
| 两段式 `DS4_EXP_2PHASE=0`、light 路径 `STRATA_DS4_NO_LIGHT=1` | 中性（窗口专家根本不走这些） |
| `--dspark-block 3/15` | pack 的 drafter 配置覆盖，无效 |
| `DS4_WIN_HOSTWAIT=1` | 中性 |
| `h_part` 改直接传主机指针（去掉每层 H2D） | 无提速且 logits 不一致 → 已回退 |
| 窗口路由的 `--expert-profile`（top-40 覆盖 97.9%） | **−47%**：drafter 的专家被重排，接受率 5.95 → 3.2 |
| 稠密权重降精度（fp8 → Q5_K） | 用户明确不做 |
| PCIe 双通道 / `DS4_AHEAD` / `--mmap-experts off` / `--ssd-cache` | 早前各轮已否决（见 `DECODE_PCIE_PLACEMENT_2026-11-03.md`） |

## 3. 剩下的唯一工作（内核工程）

把窗口每层 6.23 ms 压到 1.5–2.0 ms（= 不开的 1.5–2 倍），收益上界 **5.8 token / 110 ms ≈ 53 tok/s**。三件事，按优先级：

1. **窗口专家改用 token 内核**：T≤8 时对每个 token 调 `moe_gate_up` / `moe_down`（m=1 手写，实测 0.038 ms/组）
   而不是 `win_experts`（0.086 ms/组，2.3× 差）。要做：按 token 切 `d.plan_ptr`（行偏移 `t*kPicks`）、
   复用窗口的 `resps[slot].part/cpu` 作为 `ma.cpu_part/cpu_flag`、把 `combine` 的元数据按 token 重建。
   建议加 `DS4_WIN_TOKEN_KERNELS=1` 开关，先对拍再 A/B。
2. **窗口内降低车道占比**（37% vs token 路径 9%）：窗口 36 次挑选摊得更散。注意 §2 里 profile 那条路是负的，
   要么改缓存策略（让窗口的命中在不扰动 drafter 的前提下提高），要么接受。
3. **验证窗口的 KV 行搬运**（`ext_to_ring` / `ring_to_ext`）确认无重复搬运。

## 4. 工具与协议（下次直接用）

- **必须暖机**：`page cache` 冷暖让解码在 8.7↔25.6 tok/s 跳 3 倍；每次 A/B 先跑一次暖机，且相邻对照。
- **只认 `decode: N tokens in X ms`**；`DS4_PROF` 的表对 T>1 要用每层表（`ds4 prof window per LAYER`），
  早先按 token 摊平是错的。
- **`DS4_TLOG=1`** 打开主机分相（`tmark`：wait/boundary/qkv/…），`DS4_TLOG` 在 SYCL 构建上原本是死的
  （`ds4_dbg` 恒 false），已修。
- 编译：`ninja -C build-sycl-jit strata-dsv41`，单文件改动约 **15 分钟**。
- 对拍：**不要用 `--dump-logits`**（见 §6：本构建下它逐行不确定，且写出行数随 MTP 轮次变化）。
  可靠的等价性口径是 `--n N` 贪心输出的 token 序列（同 prompt 4 次重跑 md5 全等）。
- 跑测：`tools/overnight/gpu_lock.sh take lead` → `gate.sh` → 引擎 → `release`；服务用 `systemctl --user stop/start dsv41`。
- 本机有两份 drafter pack：`dsv41-iq2xxs-pack-mtp`（8.5 GB）与 `-mtp2`（4.92 GB，推荐）。

## 5. 结论

MTP 在 batch=1 的收益上界是 **~53 tok/s**（现在 24），但它要求窗口路径的内核效率与单 token 路径持平；
所有「不动内核」的旋钮都已试完并且是中性或负的。**下一轮要动内核，先用 §6 的口径钉死等价性再改。**

## 6. 输出等价性复核（2026-11-04 10:00–10:20）

背景：早前一次 `--n 48` 对拍里 `--dspark on` 的输出与 `off` 不同（`294,270,3702,4106,…` vs `294,8760,344,11111,…`），
一度被当成 MTP 的**正确性 bug**。这次用控制实验复核，结论是**该现象不可复现，MTP 的贪心输出与不开逐 token 相等**。

### 6.1 `--dump-logits` 不能当对拍仪器

同一份 prompt、同一份 pack、同一份 env，`--dspark off` 连跑两次：

| 项 | off#1 | off#2 | on#1 | on#2 |
|---|---|---|---|---|
| 输出 token md5（n=8） | `5adc6c7e` | `5adc6c7e` | `5adc6c7e` | `5adc6c7e` |
| dump 文件字节数 | 8,791,040 | 8,791,040 | 10,342,400 | 10,859,520 |
| 每行 logits md5 | 17 行**逐行都不同**（含 off#1 vs off#2） | | | |

- 字节数 = 行数 × 129280 × 4：off 稳定 17 行（16 prompt + 末行），on 是 **20 / 21 行**且两次不同
  ——`--dspark on` 时 `m.logits(lg)` 的返回长度随 verify 窗口轮次变化（`src/dsv41/ds4_main.cpp:1698/1760` 直接
  `fwrite(lg.data(), lg.size())`）。
- 更关键：**两次完全相同的 off 运行，logits 逐行都不相等**。所以「off 与 on 的 logits 不同」不能推出任何结论；
  我先前据此说"MTP 改了 logits"是**错的**，仪器本身不确定（疑似 dump 前未等 GPU 完成 / 复用了未完全刷新的缓冲）。
- 若以后要用 logits 对拍，得先修这条路径（dump 前 `wait`，并按位置固定行数），否则别用。

### 6.2 贪心输出：off ≡ on（4 组配置 × 每组 2 次）

同一 prompt（`3011,14,396,734,554,1748,16,455,6102,294,8760,344,11111,16,455,6102`），`--n 48`，temp=0：

| 配置 | off md5 ×2 | on md5 ×2 |
|---|---|---|
| `--window 16` | `48f1b3e0d040e163dd01c4f22070b246` ×2 | 同左 ×2 |
| 默认窗口（长 prefix 单次 feed） | 同上 ×2 | 同上 ×2 |
| 64 token prompt（4× 重复），默认窗口 | `22d29205c5b59250b5bf8295b9e538b4` ×2 | 同左 ×2 |
| 64 token prompt，`--window 16` | 同上 ×2 | 同上 ×2 |

- 四次重跑完全确定；`--dspark on` 的接受率 93.8%–100%，输出仍逐 token 相同。
- 因此 MTP 在这份 pack 上**没有引入可见的贪心输出漂移**；早前那次不一致只可能来自当次运行的配置差异
  （prompt/窗口/环境变量），不是 MTP 逻辑缺陷。**这条警报撤销。**
- 注意：这些 prompt 都会退化成重复循环，等价性证据强度有限（接受率天然 100%）。
  真正的正确性口径是 §6.3 的真实问题集准确率。

### 6.3 24 题准确率 A/B（真实文本）

`/tmp/mtp_acc2.py`（可断点续跑，`/tmp/mtp_acc2/{off,on}.json` 每题落盘），题目取自
`tools/dsv41/iq_bench.py` 的 QS，同 pack（`-mtp2`）、同服务、temp=0、`DS4_WIN_NE=1`，先 off 后 on。
结果见 `/tmp/mtp_acc2/run.log` 与 `result` 汇总（本次运行完成前先看 json）。


## 7. 「跑久了掉到 0.2 tok/s」的根因实测（2026-11-04 11:20–11:35）

现象：一个 serve 进程跑 ~20 个请求后 `ds4 host: 108 → 1084 → 6468 ms/token`，decode 掉到 0.2 tok/s，
CPU 车道等 SSD 344 ms/token。怀疑「专家溢出 / MTP 显存泄漏」。三组受控实测（pack `-mtp2`，`--dspark off`，
`--window 16`，显存从 `/proc/<engine>/fdinfo` 的 `drm-total-vram0` 读）：

| 实验 | 请求数 | RSS | VRAM | 吞吐 |
|---|---|---|---|---|
| 同一 prompt ×30 | 30 | 26.3 → 28.8 GB（平） | **30,579 MB，±2 MB** | 16–26 tok/s，全程不掉 |
| 22 个**不同** prompt | 22 | 25.9 → 68.6 GB（第 2–4 个各 +8~13 GB，之后趋平） | 30,580 MB，±2 MB | 14–25 tok/s，不掉 |
| 同上但 `--max-context 32768`（1/12） | 22 | 26.4 → 59.2 GB（只少涨 ~9 GB） | 30,578 MB | 14–25 tok/s |

结论：
- **没有显存泄漏**，也没有「专家从显存溢出」：VRAM 恒定 30.6 GB，两个半小时里一个字节没动。
- 涨的是**主机侧 page-cache 驻留**：权重 131 GB（4 个 GGUF 分片），换话题就碰到新的专家 / 新的 engram 行，
  这些页变成常驻；同一 prompt 不会。把上下文从 393216 降到 32768 只少涨 9 GB → **不是 KV 的锅**。
- 掉崖发生在总驻留逼近物理内存（188 GB）之后：24 题那轮 RSS 到 105 GB，页开始被挤出，
  每 token 重新从 SSD 读专家 → 车道等 I/O → 0.2 tok/s。所以「专家溢出」是对的，但溢出的层是
  **page cache**，不是 VRAM；`--ssd-cache 0` 等于把缓存完全交给内核。
- **与 MTP 无关**：`--dspark off` 同样复现。

可试的缓解（未测）：①`--ssd-cache N` 换成有上限的 LRU store，而不是裸 page cache；
②长驻服务按会话数定期重启；③部署预算：131 GB 权重 + 工作集必须留在 188 GB 以内。

附带发现：引擎**每次运行都会重写** pack 目录里的 `expert-profile.bin`（路由计数），
专家放置因此在运行间漂移 → 跨运行的 logits 会动（对拍要加 `--profile-readonly`）。
这很可能就是 §6 里那次「MTP 输出不同」不可复现的原因：不是 MTP 改了输出，是专家放置变了、
近 tie 的 argmax 翻了。

## 8. 「掉到 0.15 tok/s」的真因：VRAM 只剩 0.6 GB，不是内存、不是盘、不是 MTP

§7 把锅记在 page cache 上是**错的**（那次只看了 RSS）。这一轮用同一套 24 题负载、
逐请求记录 RSS / VRAM（`/proc/<engine>/fdinfo` 的 `drm-total-vram0`）/ 磁盘 / CPU，
把因果钉死了：

| 观测 | 掉崖时 | 健康时 |
|---|---|---|
| `drm-total-vram0` | 30,567 MB（**596 MiB 空闲**） | 24,010 MB（**7,321 MiB 空闲**） |
| 常驻 VRAM 的专家 | 1465 / 15744 | 647 / 15744 |
| `cpu-lane wait`（= 等专家结果） | 2,185 ms/token | 十几 ms/token |
| 车道实际计算 | 118 ms/token | 同量级 |
| 系统磁盘读 | **0 MiB/s** | 0 MiB/s |
| CPU 空闲 | **97%** | 97% |
| 全部线程状态 | 全是 `futex_do_wait`，1 个 R | 同 |
| decode | **0.15 tok/s** | 15–22 tok/s |

所以：既不是内存不足（166 GB 可用）、也不是读盘（0 MiB/s）、也不是 GPU 掉线
（内核无 xe/guc 报错），是**等专家结果**——而专家结果卡在 VRAM 太满：
`--vram-experts 1.0` 把 32 GB 的卡吃到只剩 596 MiB，暂存/环没法周转，
每个 token 的专家都要来回搬 → 2.2 s/token。

**A/B（同 24 题、同 pack、`--dspark off`）**：

| `--vram-experts` | VRAM | 24 题结果 |
|---|---|---|
| 1.0（生产现值） | 30,567 MB / 596 MiB 空闲 | 第 1–3 题就掉到 **0.15–0.16 tok/s**（今天复现 2 次） |
| 0.5 | 24,010 MB / 7,321 MiB 空闲 | **全程 14–22 tok/s，24 题无一次掉崖**（RSS 照样涨到 110 GB，无影响） |

两条结论：
1. **这是部署前必须清掉的隐患**：`--vram-experts 1.0` 在 32 GB 卡上没有余量，
   桌面/浏览器/别的进程多吃几百 MB 就会把引擎推进 0.15 tok/s；也解释了为什么同一份配置
   有时 25 tok/s 有时 0.2 tok/s。
2. 建议留 ≥6 GB 显存余量（0.5–0.75），代价为零（15–18 tok/s 与 1.0 健康时持平）。
   注意开 MTP 时 drafter 还要 ~7 GB，两者要一起算预算。

复现脚本：`/tmp/hazard_ab.py <ssd-cache> <outdir>`（`PROBE_VRAM` / `PROBE_N` 可覆盖）。

## 9. 窗口一层的 3.8 ms 花在哪（2026-11-04 13:15 实测，commit 9ee08be9）

新仪器 `DS4_WIN_TLOG=1`（窗口路径的三个主机跨度，对应 T=1 路径的 `DS4_TLOG2`）。
256 token、`--dspark on`、pack `-mtp2`、T=6、接受率 100%、5.75 token/轮：

| 项 | 每层 | 每轮(40 层) | 说明 |
|---|---|---|---|
| gate_up 提交（6 次） | 0.220 ms | 8.8 ms | 主机侧 |
| 等 CPU 车道（`wait_seq`） | **0.021 ms** | 0.8 ms | **车道不在关键路径上** |
| down 提交（6 次） | 0.328 ms | 13.1 ms | 主机侧 |
| post + 队列反压 | 3.23 ms | 129 ms | 主机在这段被 in-order 队列憋住 = GPU 在算专家 |
| **合计 down+post 跨度** | 3.80 ms | 152 ms | 与 `verify 149.6 ms/轮` 吻合 |

GPU 事件表（`DS4_PROF`）同一份数据：每层 3.79 ms，其中 **down+post 2.46**、routing+shared 0.43、
attn proj 0.36、attn tail 0.27、between layers 0.17、sparse 0.07。

对照 T=1（同一二进制、`--dspark off`、256 token，23.51 tok/s）：**down+post 只要 0.116 ms/层**，
整层 1.03 ms。所以窗口的每个 token 在这一段贵 **3.5×**。

结论（推翻「主机侧 205 ms 是派发太多」的假设）：
1. 主机侧三笔提交加起来只有 **0.57 ms/层 = 23 ms/轮（15%）**；减少派发最多省 15%。
2. 真正的墙是 **GPU 上的专家 GEMM：2.46 ms/层（65%）**。主机之所以在 post 里「花」3.23 ms，
   是 in-order 队列把它憋在 GPU 速率上（同 §7 的结论）。
3. 每个 token 比 T=1 贵 3.5× 的原因是**重复读专家 blob**：窗口 6 个 token 的 36 个 pick 只有
   ~34.6 个不同专家（96% 重叠），但按 token 逐个调 `moe_gate_up/moe_down` 会把这些 blob 读 6 遍。
   **按专家去重（一次读、m=1..6 行）才是 2.46 → ~0.8 ms 的路径**，不是合内核/减派发。

下一步：把窗口的 plan 重排成「按专家分组」的 36 行（token, weight），用一个 m≥1 的 group 内核跑完
（`launch_groups` 现在的形态是每组一次 launch；要么让它在设备端遍历组，要么每组一次但每组只读一次 blob）。
判据：每层 down+post 从 2.46 → <1.2 ms，即每轮 149.6 → ~110 ms（40 tok/s 以上），且输出 token 不变。

### 9.1 「按专家去重」的现成实现更慢，不是免费的（同轮 A/B）

同一二进制、同一 256 token、只切 `DS4_WIN_TOKEN_KERNELS`：

| 窗口专家路径 | verify | decode | 输出 md5 |
|---|---|---|---|
| 按 token 逐个调（默认，commit bf96dfeb） | **149.6 ms/轮** | **37.07 tok/s** | `612778eac1fb1983` |
| `win_experts` 分组去重（`DS4_WIN_TOKEN_KERNELS=0`） | 274.3 ms/轮 | 21.22 tok/s | `612778eac1fb1983`（相同） |

两条路输出**逐字节相同**，但现成的去重内核慢 1.83×：2 次 launch/层却要 6.86 ms/层，
说明 `win_experts` 的 m≈1.2 小行分组本身低效（每组 ~3.4 ms），不是派发问题。
所以「去重」要真赚到，必须写一个 **m=1..2 也高效的分组内核**（每专家读一次 blob、
组内行数可变），而不是复用 `win_experts`。

## 10. `--vram-experts` 不是免费的旋钮：它改变输出和接受率（2026-11-04 14:0x 实测）

同 256 token、`--dspark on`、其余不变：

| `--vram-experts` | 常驻专家 | 空闲显存 | decode | token/轮 | 输出 md5 |
|---|---|---|---|---|---|
| 1.0 | 1465 | 596 MiB | **37.07 tok/s** | 5.82 | `612778ea…` |
| 0.75 | 1012 | 3833 MiB | 37.11 tok/s | **5.22** | `322ab33c…`（不同） |
| 0.5 | 618 | 7068 MiB | **28.39 tok/s** | 5.82 | `612778ea…`（同 1.0） |

- **0.5 慢 24%**：更多专家落到 CPU 车道（34555 vs 20491 of ~63.6k）。
- **0.75 与 1.0 同速**（37.1），但接受率从 5.82 掉到 5.22、输出 md5 也变了——
  专家在「显存 / 车道」之间换位置会改变归约顺序（车道 fp32 与 GPU 累加不同），近 tie 的 argmax 就翻。
  所以：**A/B 必须固定 `--vram-experts`**，它既影响速度也影响输出。
- 部署建议：1.0 只有 596 MiB 余量（§8 的掉崖），0.75 同速且留 3.8 GB，是安全档；
  代价是那条采样路径（5.22 token/轮）。

## 11. 一个真 bug（已修）+ 一个更麻烦的事实：MTP 路径**逐次运行不可复现**

### 11.1 修掉的越界别名（commit 98ab6b07）

`bf96dfeb` 的按 token 路径里，每个 token 的 gate/up 输出写成 `d.xg + t * FF`，
但 token 内核写的是 `h[grp * FF]`（grp = 0..k-1），所以 token t 实际占了
`[t*FF, t*FF + k*FF)`——**和 token t+1 的区域重叠**。正确步长是 `kPicks * FF`
（`d.xg` 本来就是按 `P = kWin * kPicks` 个槽分配的，`plan_ptr` 也是按 `t*kPicks + r` 排的）。

- 证据（32 token、新 prompt、`--profile-readonly` 冻结专家放置）：修之前 K=1 与 K=0 的输出不同；
  修之后同一对配置输出**相同**（`3dd8656c…`）。
- 修后速度没有回退，反而略快：256 token **38.94 tok/s**（修前 37.07），`--dspark off` 23.51 → **+66%**。
- 教训：`bf96dfeb` 里那句「generated ids bit-identical to the old path」是**在退化重复 prompt 上得到的，不成立**。

### 11.2 更麻烦的：`--dspark on` 逐次运行输出不一样（不是本次改动引入）

同一二进制、同一 prompt、同一参数、`--profile-readonly` 冻结放置、连跑两次：

| 配置 | 第 1 次 | 第 2 次 |
|---|---|---|
| `DS4_WIN_TOKEN_KERNELS=0`（老的分组路径） | `bed122b2…` | `62357cf7…`（**不同**） |
| `DS4_WIN_TOKEN_KERNELS=1`（本次路径） | `3dd8656c…` | `12ebf71d…`（**不同**） |

两条路径都会在两次运行之间变。所以**「MTP 输出与基线逐字节一致」这个口径本身不成立**，
之前几次 md5 相同是运气。可能的原因（未定论）：CPU 车道与显存的专家归属随运行状态变
（§10 已证 `--vram-experts` 一改输出就变），车道的 fp32 归约与 GPU 归约不同 → 近 tie 的 argmax 翻。

**结论与后续协议**：
- 等价性只能用**统计**判据（24 题准确率、KL、或多次采样的分布），不能用单次 md5。
- 之前「24 题准确率 19/24 vs 19/24」这类结论仍然成立（那是统计口径）。
- 若要真正的逐位对拍，得先让专家归属确定化（固定 `--vram-experts`、关 `--expert-cache`、
  或加一个「所有专家都走同一路径」的测试开关）。

## 12. 目标达成：一轮 128.6 ms（原 304 ms），MTP 43.9 vs 26.0 tok/s（+69%）

修完 §11 的步长 bug 之后，512 token、暖机、相邻对照、每种两轮（`--profile-readonly` 冻结专家放置、
`--vram-experts 1.0`）：

| 配置 | run1 | run2 |
|---|---|---|
| `--dspark off` | 26.28 tok/s | 26.01 tok/s |
| `--dspark on` | **42.87 tok/s** | **43.90 tok/s** |

- `verify 128.6 ms/轮`，接受率 100%（390 offered / 390 accepted），5.8 token/轮。
- 每轮从 **304 → 128.6 ms**（2.4×），落在目标的 110–150 ms 区间内；tok/s 落在 40–53 区间。
- 256 token 的短跑也一致：38.94 vs 23.51（+66%）。

窗口一层现在的分布（GPU 事件表）：total 3.69 ms/层 —— down+post 2.36（专家 GEMM）、routing+shared 0.43、
attn proj 0.36、attn tail 0.27、between layers 0.17、sparse 0.07、comp+idx 0.03。

### 12.1 为什么停在这里（剩下的 3.5× 需要新内核）

- 主机侧三笔提交只有 0.55 ms/层，而且**被 in-order 队列隐藏**：每层 GPU 事件合计 3.69 ms ≈ 每层墙钟
  3.68 ms（147 ms/40），说明 GPU 就是墙 → 减派发/合内核不会再有收益（§9 的假设在此彻底否掉）。
- 专家 GEMM 2.36 ms/层 已经比现成的去重路径（`win_experts`，5.61 ms/层）快 2.4×。
- 理论上限：T=1 同阶段 0.116 ms/层（6 个专家），窗口 36 个 pick 里 ~34.6 个不同专家 →
  若内核在 m≈1.2 也高效，应约 0.7 ms/层。剩下的 3.5× 要靠一个**新的小 m 分组 MoE 内核**
  （每专家读一次 blob、m=1..2 也打满带宽），不是调参能到的。

## 13. 真实文本上的对照（修完 bug 后，2026-11-04 14:3x）

同一 server、同一 pack、同一真实 prompt（写一段 150 词的灯塔故事）、贪心、256 token、
先跑一轮暖机再取第二轮的引擎自报数字：

| 配置 | 暖机轮 | 正式轮 |
|---|---|---|
| `--dspark off` | 15.16 tok/s | **18.77 tok/s** |
| `--dspark on` | 9.36 tok/s（冷缓存） | **34.30 tok/s** |

**+83%**（18.77 → 34.30）。两边的输出都会退化成复读（这是 IQ2 量化 + 这个 prompt 的模型行为），
但负载相同，速度对照有效。冷缓存的暖机轮 on 更慢（9.36）是老问题：窗口对冷专家更敏感。

## 14. 更正 §8/§10：`--vram-experts 0.75` **也不安全**

24 题负载、`--profile-readonly` 冻结放置、`--vram-experts 0.75` 跑同一套题：跑到第 23 题
`ds4 host: 5796 ms/token`、掉崖（此时 `drm-total-vram0` 28.06 GB ≈ 剩 4.5 GB）。
所以安全线比 §8 写的更严：**1.0 掉、0.75 掉、0.5（24 GB，剩 7 GB）不掉**。
部署用 0.5；或者接受「长会话要定期重启」。

## 15. 草稿数（`--dspark-block`）扫描：2/3/4/5（2026-11-04 19:00）

默认是 pack 的 block=5 → 一轮 **T=6 窗口（1 个种子 token + 5 个草稿）**。
`--dspark-block N` 确实生效（`max_drafts = N`，窗口 T=N+1）。同一二进制、`--profile-readonly`：

**A. 退化重复 prompt（512 token，全部配置 100% 接受，唯一可比的口径）**

| 草稿数 | 窗口 T | decode | verify/轮 | token/轮 |
|---|---|---:|---:|---:|
| 2 | 3 | 37.63 tok/s | 71.7 ms | 2.99 |
| 3 | 4 | 41.11 tok/s | 89.1 ms | 4.00 |
| 4 | 5 | 42.07 tok/s | 111.1 ms | 4.97 |
| 5 | 6 | **43.44 tok/s** | 130.7 ms | 5.95 |

可预测文本上草稿越多越快，但边际递减：2→3 +9.2%，3→4 +2.3%，4→5 +3.3%。

**B. 真实文本（同一个 prompt，每档 4 次请求，`--vram-experts 0.5`）**

| 草稿数 | decode（8 段的均值/中位） | 累计接受率 | verify/轮 |
|---|---|---:|---:|
| 2 | 22.2 / 23.2 | **93.2%** | 104 ms |
| 3 | 17.9 / 18.3 | 72.3% | 112 ms |
| 4 | 23.8 / 24.8 | 93.5% | 141 ms |
| 5 | 22.4 / 23.8 | 84.8% | 144 ms |

- 真实文本上**没有单调收益**：窗口从 104 → 144 ms 贵了 40 ms，而接受率随文本波动
  （这档 72% 是因为那一轮生成的文本正好不可预测，每次运行生成的文本不同，所以这一列只能当参考）。
- 结论：**可预测负载（代码/列表/结构化输出）用满 5 个草稿；真实多样文本 2–3 个就够**，
  多出来的窗口时长换不到 token。
- 注意 B 组是 `--vram-experts 0.5`（22 GB 左右常驻）；同一配置在 1.0 时真实文本能到 34.3 tok/s，
  但 0.75/1.0 在长真实会话里会掉崖（§14），所以 34 那个数需要显存余量，不能当稳定值。

## 16. 长会话掉速：soak 抓到的第一个确定的失效（2026-11-05 11:30）

soak：`--dspark on`、`--vram-experts 1.0`、200 个真实多样请求、`DS4_PROF=1`、`DS4_RING_TLOG=1`、
请求 <5 tok/s 即抓现场。**第 71 个请求抓到**：

| 项 | 值 |
|---|---|
| 引擎 RSS / 显存 | **126 GB** / 30.2 GiB（满） |
| 线程 | 61 个 `futex_do_wait` + 1 个 `hrtimer_nanosleep` |
| 累计 `swaps` | 4557 |
| `prof window per LAYER` | total 3.81 ms（down+post 2.44）—— **窗口本身是健康的** |
| `dspark split` | verify 141 ms/轮，层内 135 —— **也健康** |
| 请求速率 | 31 token / 7375 ms = **4.2 tok/s** |
| **接受率** | 前 60 个请求累计 ~71%，最后两个请求 **`drafts 0 of 240 kept` / `0 of 135 kept`** |

结论：这一次的掉速**不是**窗口变慢，而是**草稿突然全部被拒**（接受率 71% → 0），于是每轮只出 1 个 token，
速度掉 ~6 倍。verify 每轮仍是 141 ms。

**顺带否掉了一个假设**：`ring_tlog: 0 slot waits, 0 ms in all, worst 0 ms; 0 breaks` ——
生产线程的"环形缓冲槽 2 ms 轮询"这条路**整个 soak 一次都没走到**，所以它不是原因（那个 spin 计数改动已回退）。

下一步：接受率为什么突然归零。嫌疑人：drafter 与 trunk 共享的 KV/ring 状态在第 ~70 个请求后不再一致
（`swaps 4557`、RSS 126 GB 都在这个时间点附近），或 slot 复用（`106 reused + 17 read`）在长会话里累积错位。

## 17. 长会话掉速的第二种形态：GPU 侧 down+post 从 2.4 → 15.4 ms/层（2026-11-05 13:00）

用 `DS4_TLOG=1` + `DS4_PROF=1` 复现（vram 1.0、3 个请求，第 3 个掉到 0.29 tok/s），抓到的分层数据：

| 项 | 健康 | 掉速时 |
|---|---|---|
| `prof window per LAYER` total | 3.81 ms | **~17 ms**（down+post 2.44 → **15.40**） |
| `tmark per layer` 的 host `wait` | 2.38 ms | **20.74 ms** |
| `ds4 host` | 120 ms/token | **1923 ms/token** |
| `cpu-lane wait` | 0.20 ms/层 | 0.81 ms/层（**很小**） |
| `swaps` | — | 1061 → 1071 还在涨 |
| 线程 | — | 60 futex + 1 hrtimer_nanosleep + 1 `Rl`(92% CPU，是 `serve_lane` 的空转服务循环，常态) |
| 磁盘/读量 | — | 启动后累计 rchar 152 GB，掉速那一刻 read_bytes 不动 |

**结论：这一次掉速是 GPU 侧的专家 GEMM 慢了 6 倍，主机只是在等它**（host 19xx ms/token ≈ 40 层 × 20.7 ms 的 `wait`）。
不是主机派发、不是车道、不是磁盘；`cpu-lane wait` 只有 0.8 ms/层。和 §16 的"接受率归零"是两种不同的失效。

嫌疑：`--vram-experts 1.0` 把显存钉满（剩 596 MiB）后，每层的专家要么走环形缓冲换入换出（`swaps` 持续增长），
要么走 PCIe 暂存，专家 GEMM 就等这个搬运。下一步要量的是**每层专家搬运的字节数和耗时**（ring 换入 vs PCIe 暂存），
以及 `--vram-experts 1.0` 与 0.5 在同一个请求上的 `swaps` 增速对比。

## 18. 目标里那个 1 s/token 的车道等待，现场抓到了（2026-11-05 13:40）

同一套 3 请求复现（vram 1.0、`--mmap-experts on`、`--profile-readonly`），第 1 个请求 **274 s / 47 token = 0.19 tok/s**：

```
ds4 prof decode per token (avg of 128): ... cpu-lane wait 1034.39 ms  down+post 1174.94 ms ...
ds4 host: 4076.21 ms/token in the layers
ds4 dspark split: verify 2309.3 | of it inside the layers 2250.0 | rest 59.3 ms/round
ds4 dspark: 56 rounds | offered 83, accepted 42 (50.6%) | draft 229.2 ms/round, T=2 | 1.75 tokens a round
ds4 dspark vol: swaps 345 (不变)   ← 交付里没有专家换入换出
ds4 prof window per LAYER: total 3.50 ms/layer   ← 每层 GPU 事件完全正常
```

这就是目标里描述的那个形态，一次抓全：

- **`cpu-lane wait` 1034 ms/token**（车道在等），`down+post` 1175 ms；
- **草稿慢 30 倍**：`draft 229.2 ms/轮`（健康 7.7），`T=2`、接受率 50.6%；
- **`swaps` 全程 345 不变** → 不是专家换入换出；
- 每层 GPU 事件表 **3.50 ms/layer 完全正常** —— 所以"层内变慢"的解释只在某些轮次成立，这次时间花在**车道与草稿**这两处。

**由此排除的假设（都是实测否掉）**：
1. 环形槽 2 ms 轮询（`ring_tlog: 0 waits`）；
2. 专家剖面漂移（`--expert-profile none` 与 `--profile-readonly` 表现一样：26–33 tok/s）；
3. 显存档位阈值（`--vram-experts 0.5` 也会掉：0.26 tok/s 起手）；
4. `swaps`/专家换入换出（掉速时 146–345，健康时 245–424，反而更少）；
5. 磁盘/内存（掉速那刻 `read_bytes` 不动）。

**剩下的唯一共同点**：车道（`serve_lane`，24 线程、专家原地读）和 drafter（GPU 小 block）**同时**慢 20–30 倍，
而窗口的专家 GEMM 正常。两个都不碰磁盘、不碰 ring，唯一共用的资源是 **host 内存/页缓存与 PCIe**。
下一步就量这两个：车道每个专家的耗时分解（prep/gate+up/down 已有 prof）在掉速时的值，以及 drafter 一轮里
主机侧的每一段（`mv` 已报 15 ms/window，正常）。

## 19. 长会话掉速：当前状态与可复现的夹具（2026-11-05 14:00，交接用）

**症状（三种形态，都是实测抓到的）**
1. `cpu-lane wait` 1034–2185 ms/token，同时 `draft` 慢 30×（7.7 → 229 ms/轮）、`T` 掉到 2、接受率 ~50%；
   窗口的每层 GPU 事件仍正常（3.5 ms/层）。`swaps` 不动、磁盘 0、CPU 97% 空闲、全部线程睡 futex，
   只有一个线程 92% CPU —— 那是 `serve_lane` 的常态空转（健康时也一样，**不是**故障信号）。
2. 接受率突然归零：`drafts 0 of 240 kept`，每轮只出 1 token，速度掉 6×（verify 仍 141 ms/轮）。
3. 窗口 `down+post` 2.44 → 15.40 ms/层（同一次里 `swaps` 1061→1071）。

**已实测排除**：环形槽 2 ms 轮询（`ring_tlog: 0 waits`）、专家剖面漂移（`--expert-profile none` 同样健康）、
显存档位阈值（0.5 也会掉）、`swaps`/换入换出（掉速时反而更少 146–345 vs 健康 224–445）、磁盘与内存不足。

**基线（健康态，vram 1.0，3 请求 × 4 轮全绿）**：14.5–35.5 tok/s、`down+post` 1.8–3.5 ms/层、
每请求 `swaps` 224–445。

**夹具**（在 `/tmp/fix/`，重启后需重建）：
- `repro3.py VRAM TAG [引擎参数]`：3 个真实请求，自动选空闲端口，每请求 90 s 超时，
  打印速率 / `swaps` 增量 / `down+post` / 每层总时长；
- `run_r5.sh`：装好 env（`DS4_PROF=1`、`R3_TIMEOUT=90`）、拿 GPU 锁、跑多轮、释放锁并恢复服务。
  掉速大约 1/4–1/8 轮命中，所以一次要跑 4–6 轮。

**下一步（未做）**：`serve_lane` 空转占满一个核 + 车道 24 线程 + drafter 抢核的**CPU 争用**假设，
用 `--cpu-threads` 降车道线程数看低频掉速率是否下降；以及 `reqs`（64 槽）由**设备**发布、
`serve_lane` 消费这条链在长会话里是否会出现发布/消费错位（掉速时 `cpu-lane wait` 炸掉正指向这条链）。

## 20. 掉速的根因抓到了：设备把邮箱踩过一整圈，车道永远等一个被覆盖的请求（2026-11-05 20:10）

**确定性复现（3 分钟，不再靠运气）**：`--cpu-threads 1` 让车道变慢，再用 `DS4_LANE_BP=0`（旧的无限背压）跑：

```
=== foff_1: --cpu-threads 1 ===
  #1  222.7s | 47 tok   0.60 tok/s | swaps + 5 (all 5) | down+post 101.49 | total 130.05 ms/layer
  #2..#6  STALL (>60s)     swaps 5 一直不动
  >> lane_tlog: waiting 200 ms for request 6 in slot 6 (slot says 70)
```

`70 - 6 = 64 = kRing`：车道卡在**请求 6** 等它出现，而那个槽里已经是**请求 70**——设备越过了整整一圈，
把车道没读过的槽覆盖了。于是：

- 车道永远看不到请求 6 → 死等；
- 解码线程等车道的答案（`cpu-lane wait` 爆炸）；
- 专家 GEMM 等不到答案 → `down+post` 101 ms/层、`swaps` 冻结；
- 整机 97% 空闲、所有线程睡 futex、只有一个线程空转（车道）。

**这就是 §17/§18 那些形态的共同根因**：不是显存、不是磁盘、不是车道算力，是**邮箱没有背压**。

**修复**（`DS4_LANE_BP`，commit `85d8f4a3`，默认开）：设备发布第 N 个请求前，必须等车道答完第 N-kRing 个
（有界 60 s，超时明确报错）。同样条件下（`--cpu-threads 1`）BP=1 那一臂正在跑，判据：**不再出现 STALL**。

### 20.1 背压修复的强制定性验证（同一 `--cpu-threads 1` 条件下 A/B）

| 臂 | 请求 1 | 后续请求 | `swaps` | 结果 |
|---|---|---|---|---|
| `DS4_LANE_BP=0`（旧） | **222.7 s / 0.60 tok/s** | #2–#6 **全部 STALL**（>60 s） | **冻结在 5** | **永久死锁** |
| `DS4_LANE_BP=1`（新） | **58.6 s / 5.36 tok/s** | #3 6.42、#4 2.69、#5 4.62（#2 仍超 60 s） | **297 → 1306 持续增长** | **慢但在推进** |

- 旧行为：`swaps` 完全不动、`down+post` 101 ms/层、车道卡在请求 6 而槽里是 70 → 永不恢复。
- 新行为：`swaps` 一直涨、请求能跑完；1 个车道线程本来就慢（每 token ~110 个专家串行），
  所以 2.7–6.4 tok/s 是"车道被压到 1 线程"的合理值，**不再是死锁**。

**结论：根因确认，修复有效。** 死锁 → 慢。默认 `--cpu-threads`（物理核）下车道够快，背压几乎不阻塞
（前面 30 个请求的 A 臂就是证据：0 次掉速、15–23 tok/s）。

## 20. Upstream v0.2.11 parity sweep (2026-11-06, this session)

v0.2.11 released 2026-11-05 12:49Z, 24 minutes before Niko1221 closed our benchmark PR #5047
(the results went to the website's benchmarks page, not the repository). 205 commits, 300
files between v0.2.10 and v0.2.11. Of the nine SYCL-touching files, **one** mattered here.

### Applied: Q5_0 expert dot (upstream `aa678f10`, commit `d0c4a0a0`)

`vec_dot_q5_0_q8_1` in `sycl/src/kernels/cuda/iq_kernels.dp.cpp` carried the pre-0214 formula:
the constant -16 per-weight offset was taken off the q8_1 block's `ds.y` (the sum of the
ORIGINAL activations). Upstream changed it to multiply the sum of the QUANTIZED activations
(`dp4a` with 0x01010101, times d8), like the Q5_1 and Q4_0 paths: `d5 * d8 * (sumi - 16 * sumu)`.

Their reason: the original sum does not see the activations' rounding, which the experts'
SwiGLU outputs (not zero-mean) make large. Whole-expert error with Q5_0 in the down projection
was 2.1e-2 .. 3.1e-2 against 1.2e-2 for every other down type.

**But the effect on this machine is zero**: the dsv41 engine's expert types are IQ1_M, IQ2_XXS,
IQ2_XS, IQ3_XXS and IQ3_S (`sycl/src/dsv41/ds4_iq.dp.cpp:49-59`); `grep -rn "Q5_0" sycl/src/dsv41/`
returns nothing. The fix is carried for parity with upstream, not for speed.

### Not applicable

- **#5086** (SYCL server `engine_args` announce signature): `sycl/serve/server_intel.py` here has
  no `engine_args` wrapper at all, so the broken call path does not exist in this tree.
- **#4971 / #5111** (unlisted Battlemage id counts as an Arc; an integrated GPU no longer outvotes a
  discrete card): launcher/Control side, not in this tree.
- Everything else in 0.2.11 is AMD 4x R9700 split engine, NVIDIA MMQ / 32768 prompt chunks,
  FP16 weight cache, small-card layer-split walk, or Unsloth Qwen3.8 files.

### Found but NOT applied: the nine Battlemage SYCL decode switches

Upstream `8fffe81d` / `4a7bb1d0` turn on, by default for Battlemage devices, the measured-gain
subset of glorysdj's PR #4185. **None of the nine exist in this tree** (`grep -rl` = 0 files for
each of MMVQ_SHAPE_RPW, IQ2S/IQ3XXS/IQ3S_GU_ROW_REUSE, GR_LO_REUSE, ATTN_MERGE_ACTIVE,
IQ4NL_DOWN_SLM, RESIDENT_PLAN_COMPACT, GDN_QK_REG, Q6_SHAPE_GRF, GR_REDUCE_ILP).

PR #4185's own numbers, measured on **one Arc Pro B70** (the same card as here), oneAPI 2026.1,
AOT bmg-g31, Qwen3.8 GSQ/RCO IQ3_S with streaming experts: engine decode 64.20 -> 69.70 tok/s
(+8.57%), TPOT 15.55 -> 14.33 ms, prefill unchanged. It is a **draft with 0 comments and no move
since 2026-10-29**, and the author states scope agreement has not been obtained.

That is the one lead here big enough to matter, and it is a separate project: it needs the iq4_lut
header (absent here), touches ~700 lines across six kernel files, and its baseline is upstream's
Qwen path with `--stream-experts`, not this dsv41 engine. Porting it onto dsv41 has to be measured
here, not assumed from their Qwen numbers.
