# 交接：dsv41/SYCL → Vulkan（2026-11-07）

下一个会话主攻 **Vulkan 实现**。这份文件只写「你需要知道的」和「不要去踩的」。

## 一、现在这条线的状态（已收束）

| 项 | 状态 |
|---|---|
| 生产服务 | `dsv41` **active**，`pack-real`，`--dspark off`，`--vram-experts 0.75` |
| 解码 | 15–16 tok/s（生产）；MTP 开可到 1 063+ 但需 `-mtp2` pack |
| **预填充 100 K** | **1 063 tok/s**（`DS4_PROMPT_SYNC=0`），基线 624 |
| 预填充 260 K | 881.8 tok/s |
| 死锁修复 | `DS4_LANE_BP`（commit `85d8f4a3`），默认开 |
| 未推送 commit | **0**（GitHub `dingyiyuan/poorseek` 已同步） |

## 二、换到 Vulkan 时，这几条直接搬过去

### 1. 邮箱必须有背压（`DS4_LANE_BP` 的教训）

**症状**：长会话解码掉到 0.15 tok/s，`cpu-lane wait` 涨到 1–2 s/token，全部线程睡 futex，
只有一个线程空转，磁盘和显存都不动。

**根因**：设备往环形邮箱 `reqs[N % kRing]` 发布请求时**没有检查车道读到哪了**。
设备越过整整一圈（实测 `70 - 6 = 64 = kRing`）就覆盖了车道还没读的槽，
那个请求永远不会被应答，车道等一个永远不来的东西。

**修法**：发布第 N 个请求前，先确认车道已经答完第 N − kRing 个（有界等待，超时点名报错）。

**Vulkan 版要注意**：Vulkan 的提交顺序由 fence 和 signal semaphore 管，
但这个 bug 的本质是**两个独立推进者共享一个有限环**，Vulkan 一样会踩。
（这条已作为 issue #5426 提给上游，包含 CUDA `serve_lane` 也没有消费端背压的对照。）

### 2. 不要用「每次提交都等一次」来解决问题

`DS4_PROMPT_SYNC=1` 就是这么干的，它让 `nc >= 3` 的预填充从 1 063 掉到 639（**−40%**）。
它是 workaround，不是解。真正的解是让两个队列不成环、让邮箱有背压。

### 3. 测量纪律（这几次踩过的坑，每个都花过一次 15 分钟的模型加载）

- **`--dspark on` 的输出逐次不可复现**：专家放置会漂移，近 tie 的 argmax 会翻。
  正确性只能用**统计口径**（24 题准确率），不能用单次 md5。
- **`--dump-logits` 不能当对拍仪器**：同一次配置连跑两次，logits 文件逐行都不相等。
- **`--vram-experts` 不是免费旋钮**：`1.0`/`0.75` 在长会话里会掉到 0.15 tok/s（显存只剩 595 MiB /
  4.5 GB），`0.5` 才稳（剩 7 GB）。切换它会改变专家归属，输出也跟着变。
- **冷启动很慢**：这个 pack 要 189 GB 预读 + 127 GB 车道拷贝，**每个请求的超时设 ≥1200 s**，
  否则你会以为是崩溃，其实只是还在加载。
- **测预填充必须带唯一 nonce**：否则服务的 KV/slot 缓存直接命中，测出来的数没有意义。

### 4. GPU 硬规则（一天 wedge 四次换来的）

**绝不对 GPU 作业中的引擎发 SIGKILL/SIGTERM**。会留下挂起的 dma_fence，
下一个进程的第一次 GPU 作业就超时 + 四连 reset + `Xe has declared device as wedged`，
只有重启能救。要停就用引擎自己的停止路径，或等它自然结束。

**另外**：`lspci`/sysfs 的 `LnkCap` 读数在本机是假的（显示 x1 2.5GT/s，实测 9.5 GB/s），
带宽必须用 pinned 内存实测。

## 三、别做的事

- **不要移植上游 PR #4185 的 Battlemage 开关到 dsv41**：已查证，那 9 个开关改的是
  Qwen 的共享内核（`native_mmvq`/`fused_gr`/`qsa_decode_attn`/`verify_kernels`），
  dsv41 走自己的 `ds4_iq`/`expert_mmq`，`grep` 命中 0。那 +8.57% 不会跟过来。
- **不要把 `PROMPT_SYNC=0` 写成代码默认**：根因（设备约 2 000 个 outstanding 命令后停止退休）
  仍未定位，`RING_DEP=host` 只覆盖流式 DMA 段，批量路径还带着同样的边。
- **不要在这条线上再提 PR 给上游**：三个候选都不成立（Q5_0 上游已修；候选修复是 CUDA/SYCL 不同后端
  且本机无 N 卡；CONTRIBUTING 第 19 行禁止换皮重提）。详见 `PR_DECISION_2026-11-07.md`。

## 四、仓库现状

| 位置 | 内容 |
|---|---|
| `dingyiyuan/poorseek` | **权威副本**。SYCL 引擎在 `sycl/`，文档在 `docs/`，工具在 `tools/dsv41/` |
| `/data/strata-new` | 开发树，分支 `b70-sycl-dsv41-perf`，工作区已清理干净 |
| `/data/models/dsv41-iq2xxs-pack-real` | 生产 pack（**无 drafter**，所以只能 `--dspark off`） |
| `/data/models/dsv41-iq2xxs-pack-mtp2` | 带 drafter（1 244 个 mtp 张量），MTP 只能跑这个 |
| `/data/measurements/2026-11-07-prefill/` | 今天所有预填充原始日志（`ab_*`/`stab_*`/`big_*`/`prof5.log`） |

## 五、起点建议

Vulkan 的第一件事不该是写内核，而是**把测量夹具先立起来**：

```bash
tools/dsv41/prefill_bench.py <tokens> <arm:sync|nosync> <out>   # 100K/132K 预填充
tools/dsv41/iq_bench.py --api http://127.0.0.1:PORT --tag X --out Y  # 24 题（需 DS4_API_KEY）
tools/dsv41/slow/repro.py                                       # 长会话掉速复现
```

这三个夹具里的每个参数都是前面用失败换来的（见第二节第 3 条）。
先把它们对着 SYCL 引擎跑通、拿到基线，再开始动 Vulkan——这样任何一道改动都有对照。
