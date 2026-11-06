# 要不要给 Strata 提 PR —— 结论与依据

> 2026-11-07 01:20。基于 v0.2.11（2026-11-05 12:49Z）+ CONTRIBUTING.md 全文 + 上游代码实测。

## 结论：**不提。** 但有一件事该做——开 issue，不是 PR。

---

## 一、为什么 Q5_0 那个修复不能提

上游 `aa678f10` **已经修了**，而且 CUDA/HIP/SYCL 三个后端一起改：

```
Q5_0 expert dot: the -16 offset takes the sum of the quantized activations (CUDA/HIP and SYCL), like Q5_1 and Q4_0
```

我们把本地版本补齐到一致（commit `d0c4a0a0`），但那是**追平上游**，不是新贡献。原样提 PR 会被当作「重复已有工作」，CONTRIBUTING 第 111 行明确写了这会被 close。

而且**对你的引擎零影响**：dsv41 用 IQ1_M / IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S，`grep -rn "Q5_0" sycl/src/dsv41/` 为空。

## 二、另外两个 0.2.11 项也不适用

| 项 | 为什么不适用 |
|---|---|
| #5086 SYCL server `announce` | 我们的 `sycl/serve/server_intel.py` **根本没有 `engine_args` 函数**，那条坏路径不存在 |
| #4971 / #5111 显卡识别 | launcher/Control 侧，不在本树 |

## 三、LANE_BP 背压修复：技术上提不了 PR，但值得开 issue

这是这两天**最有价值**的成果（邮箱无背压 → 设备越过车道一整圈 → 请求丢失 → 死锁），
上游**没有任何相关 issue**（`LANE_BP` / `backpressure` / `mailbox` / `kRing` 全 0 命中）。

**但提 PR 卡在两个硬问题上：**

### 1. 代码不在同一个后端

| | 上游 | 我们 |
|---|---|---|
| 文件 | `src/dsv41/ds4_model.cu` | `sycl/src/dsv41/ds4_model.dp.cpp` |
| 语言 | CUDA | SYCL |
| 我们的改动 | — | `DS4_LANE_BP`，约 30 行 |

CONTRIBUTING 第 76 行：「Build every backend a code change touches, including CUDA, HIP, and SYCL where applicable, before requesting review.」
**这台机器没有 NVIDIA 卡，我编不了 CUDA，也就不可能给出合格的验证。** 第 78-80 行还写了：缺环境就该保持 draft 并说明缺什么，而不是声称兼容。

### 2. 这是我们自己树的 bug，还是上游的？

`DS4_LANE_BP` 修的是**我们这个 SYCL 移植**里的邮箱环路。上游 CUDA 版是否有同样问题，我**没有证据**——没读过它的 mailbox 实现，也没有 CUDA 硬件去复现。CONTRIBUTING 第 32-34 行：不能编造影响面和严重性。

### 3. PR #5047 那次和 CONTRIBUTING 第 19 行

> Do not resubmit a declined proposal under a different title or through another agent. If new evidence changes the situation, bring it to the original discussion.

#5047 是被「结果移到官网」关闭的，不是被拒。但这条规则的精神很明确：**同一件事不要换皮重来**。

---

## 四、所以该做什么

### 该做：开一个 issue，标题 `[Engine] decode stall: the lane mailbox lets the device publish past the lane's read position`

内容（全部有实测支撑，符合 CONTRIBUTING 第 36-46 行模板）：

```
Problem: 长会话解码会在十几到二十几个请求后掉到 0.15 tok/s，CPU lane wait 涨到
         1034-2185 ms/token，所有线程睡在 futex 上，只有一个线程空转，磁盘/显存
         都不动。
Environment: SYCL 移植（这台树的 sycl/），Arc Pro B70 32 GB，oneAPI 2026.2，
         DeepSeek-V4.1 IQ2_XXS pack，--vram-experts 1.0/0.75，--dspark off/on。
Reproduce: --cpu-threads 1 可确定性复现（1-2 个请求内）。
Expected: 请求不丢，解码持续。
Actual: lane_tlog: waiting 200 ms for request 6 in slot 6 (slot says 70)
        70 - 6 = 64 = kRing —— 设备越过整整一圈，覆盖了车道还没读的槽。
Evidence: lane_tlog 输出、swaps 冻结、down+post 101 ms/层。
Reproduction status: reproduced locally
```

**并注明这是 SYCL 移植侧，问维护者上游 CUDA 版是否有同样的无界发布。** 这样是**提问 + 给证据**，不是自称修好了别人的框架。

### 可选做：把 #4185 的 Battlemage 开关拿来测

上游 `8fffe81d`/`4a7bb1d0` 已经把 glorysdj 的 PR #4185 里**测出增益的六个**在 Battlemage 上默认打开。
那组数字是**在同一型号 B70 上测的**：engine decode **64.20 → 69.70 tok/s（+8.57%）**。

我们树里**九个开关一个都没有**。但：
- #4185 还是 **draft，0 评论，10-29 之后没动**
- 它的数字是 **Qwen3.8 + `--stream-experts`**，不是 dsv41
- 730 行改动，跨 6 个内核文件，还缺 `sycl/include/strata/kernels/iq4_lut.hpp`

**移植过来必须在 dsv41 上重测，不能假设数字会转移。** 这是下一个独立项目，不是今晚的活。

---

## 五、给用户的一句话

你的两个 PR 都没被拒绝：#5047 是结果进了官网，#4911/#150 是你自己撤的。
现在再提 PR 的收益很低（要么重复上游已有修复，要么缺 CUDA 验证环境）；
**真正的杠杆是把 LANE_BP 的根因作为 issue 反馈上去，以及把 #4185 那 8.57% 在 dsv41 上验证。**

---

## 六、已执行（2026-11-07 01:56）

**Issue 已开：[Niko1221/Strata#5426](https://github.com/Niko1221/Strata/issues/5426)**

标题：`[Engine] decode stall: the CPU lane waits for a request the mailbox has already overwritten (seq - next == kRing)`

关键发现（开 issue 前才查清，修正了本文档第四节的假设）：上游 `src/dsv41/ds4_model.cu` 的
`serve_lane()`（第 9390 行）**同样没有消费端背压**——它也是只等 `r.seq != next`，
不检查设备是否越过 `kRing`。`layer_of[kRing]`（1309）、`reqs`/`resps` 按 `kRing` 分配（3739-3752）
在两个版本里结构一致。

所以这不是「我们移植引入的 bug」那么简单。但因为本机没有 NVIDIA 卡、没跑过 CUDA 版，
issue 里**没有声称 CUDA 路径也会 stall**，而是把问题问成：设备侧发布是否被约束住了。

这比原计划更值得发：如果上游设备侧没有边界，N 卡用户会遇到一模一样的症状。

---

## 七、#4185 移植评估：**对 dsv41 无效**（2026-11-07 06:35）

移植前先查了「那些开关改的内核，dsv41 到底走不走」，答案是**不走**：

```
grep -rn "native_mmvq|fused_gr|qsa_decode_attn|verify_kernels" sycl/src/dsv41/   → 0 hits
grep -rn "IQ3S_GU_ROW_REUSE|GR_LO_REUSE|MMVQ_SHAPE_RPW" sycl/src/               → 0 hits
```

- #4185 的 730 行全在 **Qwen 的共享内核**（`iq_kernels.dp.cpp` / `native_mmvq` / `fused_gr` /
  `qsa_decode_attn` / `verify_kernels`）
- **dsv41 有自己的内核**：`ds4_kernels.dp.cpp` / `ds4_iq.dp.cpp` / `expert_mmq.dp.cpp`，
  专家 GEMM 走 `gg::moe_gate_up` / `moe_down`（`ds4_model.dp.cpp:13016`）
- dsv41 从 `iq_kernels.hpp` 只借了 **`iq_dequant_f16` 一个函数**

所以：**把 #4185 那八个开关搬过来，dsv41 一个都不会命中**，8.57% 不会跟着来。
那个数字是 Qwen3.8 + `--stream-experts` 的，和 dsv41 的龙头路径不是同一段代码。

要拿到类似的收益，得把这些优化**重新实现到 ds4_iq / expert_mmq 上**，
那是重写而不是移植，工作量不在一个量级。

### 顺带的硬事实

- `pack-real` **没有 drafter**（`tensors.txt` 里 mtp/dspark 张量 0 个），
  所以生产用 `--dspark off` 不是保守，是**没得选**
- `pack-mtp2` 有 1244 个 drafter 张量，MTP 只能跑这个 pack
