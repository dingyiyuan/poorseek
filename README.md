# poorseek

MoE4All 的一个 fork：在 MoE4All 的专家分页 / host-tier 基础设施之上，做 **DeepSeek-V4.1 的分层专家调度**，
后端覆盖 **Vulkan（桌面 GPU）、Metal、CPU**。

> **状态：进行中**（工作区版本 1.0.3；CHANGELOG 最近一次发布是 1.0.2 / 2026-10-15）。

## 组件（按 crate）

| crate | 作用 |
|---|---|
| `infr-vulkan` | Vulkan compute 后端（`ash`）：`adapter` / `arena` / `caps` / `ep`，以及 `cpu_miss`——VRAM 未命中的专家走 CPU 路径 |
| `infr-metal` | Metal 后端 |
| `infr-cpu` | CPU 后端 |
| `infr-gguf` | GGUF 加载 |
| `infr-llama` | 与 llama.cpp 的模型/运行面（`seam/`）对接 |
| `infr-core` | 图与核心数据结构 |
| `infr-engine` | 引擎装配 |
| `infr-prof` / `infr-prof-rt` | 性能剖析与运行时支撑 |
| `infr-server` / `infr-cli` / `infr-chat` | 服务、命令行与对话前端 |
| `infr-hub` / `infr-embedding` / `infr-vision` / `infr-testkit` | 模型仓库、向量化、视觉与测试工具 |

## 已知的实测结论（均出自本仓库的提交记录）

- **解码的时间去哪了**（B70，N=80 DSV41）：一个 token 发出 66.5 次 command-buffer 提交、携带 4157 个
  pipeline barrier（平均每提交 62.5 个），192.9 ms 的 forward 里有 **127.1 ms 在等队列空闲**；
  `device.submit_dispatches=0` 不改变计数，所以这部分成本随 **提交次数 × 往返** 增长，而不是算力。
  （提交 `5f73749`）
- **专家传输的形态**：打包 2.85 GiB 约 0.87 s（3.5 GB/s）；整块传输 4.3 GB/s；host-visible 直写 1.8 GB/s；
  240 次逐专家传输 6.3 GB/s —— 打包 + 一次传输约为逐专家吞吐的 0.3×，加大块也没用，传输路径在 6–7 GB/s 附近饱和。
  （提交 `5f6ada4`）
- **分页策略**：per-layer 专家缓存配额 + routed-expert 直方图（提交 `448430a`）；
  N=80 的虚拟 bank slice 与 paged-MoE stride 调整（提交 `2066de6`、`5b0d56a`）。

> 本仓库目前**没有**对外发布的基准或吞吐承诺；上面引用的是仓库自身的测量记录，不构成对其他硬件/其他模型的性能声明。

## 致谢

poorseek 站在这些项目之上：

- **[MoE4All](https://github.com/Headmaster218/MoE4All)** — 直接上游：通用分页 / Slot / LUT / host-tier 底层，DSV4.1 的加载与内核框架。
- **Strata** — 冷热专家分页（`expert_cache` / `expert_source` / host-tier 补集）的原始设计，MoE4All 的分页思路承自它。
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF 格式、量化内核、MoE 调度（`n_expert_used`）与 `deepseek41.cpp` 参考实现。
- **[DeepSeek](https://github.com/deepseek-ai)** — DeepSeek-V4.1-Flash 模型、CED 架构，以及官方推理引擎 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。

上游的完整说明见 [MoE4All README](https://github.com/Headmaster218/MoE4All#readme)。

## License

本项目的修改与整体发行采用 [Apache License 2.0](LICENSE)；继承自上游 infr 的代码保留其原始
[MIT License](LICENSE-MIT) 与版权声明，详细归属见 [NOTICE](NOTICE)。
