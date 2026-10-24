# Xolotl SYCL 移植：2026-10-24 单日优化快照

这是 Xolotl（Strata 的分叉，C++/SYCL 推理引擎）在 Intel Arc Pro B70 上跑
DeepSeek-V4.1-Flash-1017（GSQ Q2_K，2.35 bit，162.4 GB）的**单日优化成果备份**。

上游基线：`Indras-Mirror/Xolotl` @ `08e28a0`。
本目录 = 相对该基线改动的 55 个源文件 + 60 份报告，路径与仓库内一致
（还原：`cp -r xolotl-sycl/{sycl,tools,include} <xolotl>/.`）。

## 成绩

| 8192 token 预填充 | 起点 | 终点 |
|---|---|---|
| 吞吐 | 2.3 tok/s（参数设错）→ 15 → 35 → 46 → 61 → 77 | **213 tok/s** |
| 总耗时 | — | **38.4 s** |
| 128K token 命中率 | — | 98% |

decode：9–10 tok/s（本轮未优化）。

## 8192 token 的 38.4 s 构成

| 阶段 | 时间 | 占比 |
|---|---|---|
| attention core | 16.5 s | 43% |
| 专家阶段（6.0 内核 + ~5 s 文件层） | 11.0 s | 29% |
| dense GEMM | 6.3 s | 16% |
| elementwise 等 | ~4.6 s | 12% |

## 六个关键改动（全部是"消除本来不该存在的步骤"，无一是写更快的内核）

1. **`--prefill-chunk 8` 是我自己的错误参数**，去掉后 prefill 9–20 倍提升。
2. **dequant + oneMKL f16 专家路径的代码一直存在，但 `sycl/src/prefill/prefill.cpp`
   从未被链进 `ds41_generate`**（`nm | grep -c prefill` = 0）。新增
   `sycl/src/kernels/native_expert_f16.dp.cpp` 接上它 → 专家内核
   50.86 → 4.70 µs/条目（10.8 倍），8192 的 77 → 173 tok/s。
   顺带修出两个真反量化 bug：Q3_K 的高位掩码按 128 值半块推进（`1<<(j+4n)`）
   但 32 字节 hmask 不按半块推进（字节 `l+16s`）；Q2_0 每块是 8 个 int16 字，不是 4 个。
3. **`cudaHostRegister` 在这个 SYCL 移植里是空实现** → 60 GiB RAM arena 不是 USM
   可达，每个 blob 都要一次同步 memcpy。改成 USM host 分片分配（设备单次上限
   30.3 GiB，必须切 3 片）→ staged H2D 3544 次降到 0。
4. **class-pass 控制环在一个层内部就绕回**（`kCtlRing=16`，每个格式类 pass 占 3 槽，
   一层 9 个类 ≥27 槽），没有复用依赖 → 异步 H2D 未完成时宿主就覆写了槽位，
   导致层 4–39 路由崩坏。加逐槽位事件修复，四条路径全部逐位相同。
5. **dense GEMM 形状的部件走 oneMKL f16**（实测 82.9 TFLOPS = 峰值 45%）。
   教训：把部件全部耗时除以 FLOPs 会高估 GEMM 占比（据此算 -8.2 s，实际 -3.5 s，
   因为段内约 4.6 s 是 rope/rms_norm/hc 而非矩阵乘）。
6. **attention 的 K 被 materialise 成 F32 再 cast 回 F16**，且 128 行 padding 也在算。
   attention core 16.5 s 的构成：K cast 4,872 ms（29.3%）+ FA 8,134（49.0%）+
   gather 2,027（12.2%）+ concat 1,562（9.4%）+ mask 11 ms；提交开销仅 12 ms。
   gather+cast = 41.5%，int8 一点也够不到 → 布局才是杠杆（在做）。

## 已实测证伪、勿再重试

- **oneDNN（`GGML_SYCL_DNN=ON`）是倒退**：1024 token 33.6 → 16.8 tok/s。
- **f16 XMX tile 不如 oneMKL**（16.5–26.4 vs 60–70 TFLOPS），但这是 tile 质量问题，
  不能据此说 XMX 不行。
- **整层预取 slab 本身有竞态**（已修），其旧的 50.7 tok/s 是错结果下的数字。
- **图捕获可用且逐位精确，但不提速。**

## 测量陷阱（都真的踩过）

- `cudaEventElapsedTime` 对**未完成的 event pair** 返回 `cudaErrorNotReady`，
  当 0 ms 累计会把 50.86 µs 误报成 3.09 µs（6.8 倍误差）。
- 编译在跑时测出的数字是被污染的（同一基线二进制负载 21 时 26.4 vs 空闲 33.6 tok/s）。
- 失败的 `ld` 会**删掉**输出二进制。
- B≥2048 的末行 logits 不可复现（三个 md5，时序敏感），但三者解码出完全相同的 token
  与 top-5 —— 大 B 的正确性门必须用 token + top-5 + 容差，**不能用 md5**。

## 运行硬规则

- **绝不对正在跑的 `ds41_generate` 发 SIGKILL/SIGTERM**：会让 xe 驱动 wedge，
  只有重启能恢复（一天踩了四次）。
- 模型进程、编译、基准测试三者串行，同一时刻只能有一个。
- `/home/dyy/Strata` 只读。
