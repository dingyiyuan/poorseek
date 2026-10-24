# 图捕获补丁（2026-10-24）

`GGML_SYCL_ENABLE_GRAPH=1` 以前直接报错退出：
`Graph nodes cannot depend on events from outside the graph`（字符串来自 libsycl.so，出现在 ggml-sycl.cpp 的 catch 里）。

**病因**：首次遇到未重排的 q8_0 权重时，`opt_for_reorder()` → `reorder_qw()` → `reorder_qw_q8_0()`
在**录制区内**提交 `async_malloc + memcpy + parallel_for + async_free`，节点带上录制外创建的事件。

**修法**：录制期间延迟重排（该 pass 用普通布局，数值相同）；丢弃半成品录制、unccaptured 重跑一次把重排装好；
此后捕获命中。自限：每个新权重最多浪费一次录制。

**实测（B70, V41-GSQ-1017, 256-token prompt, -n 8）**
| | 捕获 ON | 捕获 OFF |
|---|---|---|
| prefill | 14.5 tok/s | 15.4 tok/s |
| dense | 4.1 s | 3.9 s |
| decode | 6.39 tok/s | 6.22 tok/s |

**数值**：`--dump-last` md5 完全相同，129280 个 logits `max|d| = 0.000e+00`。

**结论：捕获可用且逐位精确，但没有加速** —— dense 是设备执行时间受限，不是宿主派发受限。
保留此补丁的价值：`GRAPH=1` 从"完全不可用"变成"可用且精确"，是后续融合/kernel 工作的地基；
但**不要指望它提速**，除非 dense 的瓶颈被证明转移到派发侧。
