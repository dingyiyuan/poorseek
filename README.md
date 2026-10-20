# poorseek

MoE4All 的一个 fork，用来做 **DeepSeek-V4.1 分层专家调度**的实验。

> **状态：进行中。**
> 当前仓库只有上游基础设施 + 少量前置改动（CED 层划分 guard、虚拟专家视图）。
> 分层专家调度本身还没写；README 等真正的实现落地后再补。

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
