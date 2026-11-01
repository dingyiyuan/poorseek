# poorseek

**一个只为一种配置维护的 DeepSeek-V4.1 推理框架：Intel Arc Pro B70（32 GB 显存）+ 192 GB 主机内存，SYCL 后端。**

> **非官方、个人项目。** 这不是上游项目的分支，也不跟随任何上游的发布节奏；它只为我这一台机器的配置更新。
> 不提供通用安装支持，不承诺接口稳定。想参考哪一块直接取用，不必等我合并什么。

## 目标配置

| | |
|---|---|
| GPU | Intel Arc Pro B70，32 GB 显存（Battlemage / BMG-G21） |
| 主机内存 | 192 GB DDR5 |
| 模型 | DeepSeek-V4.1-Flash，MoE（40 层，每层 384 个路由专家，共 15 360 个） |
| 后端 | **SYCL**（oneAPI / Level Zero） |
| 上下文 | 384 K |

32 GB 显存放在哪里都不够放全部专家：显存里只留最热的约 1 500 个专家，其余走主机内存与 CPU 车道。这个项目围绕这一个事实来做取舍。

## 状态

早期、自用。SYCL 线在本仓库的分支上（`xolotl-sycl-*`）；`main` 目前仍是继承来的基础设施。
Vulkan / Metal 后端来自最初的移植，**不是这个项目的方向**（Vulkan 那条由别人适配，我不再投入）。

## 同机参考数据

在同一台机器（Arc Pro B70 32 GB + 192 GB DDR5）上跑 DeepSeek-V4.1-Flash IQ2_XXS（384 K 上下文、40 层、15 360 专家）的实测值：

| 场景 | 实测 |
|---|---|
| 解码（真实多样 prompt：短-中文 128 / 长-英文 512 / 中-中文 256） | **19.5 / 23.2 / 20.5 tok/s** |
| 解码（同一 prompt 重复跑，专家全热） | 26–27 tok/s |
| 预填充 80 983 token | 517.6 s（**156.5 tok/s**） |
| 解码 @81 K 上下文 | 12.6 tok/s |

规模换算：100 K 输入约 9–11 分钟预填充，之后解码 11–13 tok/s。

> 口径说明：**同一 prompt 重复跑会虚高**，因为专家命中率被推到接近 100%，不能拿它做配置对比。
> 上面第一行才是真实负载的数字。

## 已知的墙（测出来的，不是猜的）

- 真实负载下 **40% 的时间在等 CPU 车道**（约 110 个专家/token 只能从主机内存来）；CPU 车道已经跑在内存带宽上（约 38 GB/s，本机 DDR5 纯读实测 49–51.6 GB/s），**没有余量**。
- 稠密投影每 token 要把 6.2 GB 权重重读一遍，卡在显存带宽（约 483 GB/s），**也没有余量**。
- 投机解码（DSpark/MTP）在本机**不划算**：接受率 100%、草稿只要 7 ms/轮，但验证一步 285 ms，因为 6 个 token 的窗口每层要 36 个专家而不是 6 个。唯一能翻正的场景是显存大到让窗口的专家全部驻留。
- 每层解码约有 5–10 个内核串行发射，**GPU 有一半时间在等发射**（验证窗口实测 GPU 忙 50%，单 token 路径接近 100%）。

## 组件（按 crate）

| crate | 作用 |
|---|---|
| `infr-vulkan` | Vulkan compute 后端（`ash`）。继承自最初的移植，不是本项目的方向 |
| `infr-metal` | Metal 后端 |
| `infr-cpu` | CPU 后端 |
| `infr-gguf` | GGUF 加载 |
| `infr-llama` | 与 llama.cpp 的模型/运行面（`seam/`）对接 |
| `infr-core` | 图与核心数据结构 |
| `infr-engine` | 引擎装配 |
| `infr-prof` / `infr-prof-rt` | 性能剖析与运行时支撑 |
| `infr-server` / `infr-cli` / `infr-chat` | 服务、命令行与对话前端 |
| `infr-hub` / `infr-embedding` / `infr-vision` / `infr-testkit` | 模型仓库、向量化、视觉与测试工具 |

## 站在谁上面

- **[MoE4All](https://github.com/Headmaster218/MoE4All)** — 起点：通用分页 / Slot / LUT / host-tier 底层。本项目由它起步，但**不再是它的分支**，也不再跟随它的发布。
- **Strata** — 冷热专家分页（`expert_cache` / `expert_source` / host-tier 补集）的原始设计。
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF 格式、量化内核、`deepseek41.cpp` 参考实现。
- **[DeepSeek](https://github.com/deepseek-ai)** — DeepSeek-V4.1-Flash 模型与 CED 架构。

## License

本仓库的修改与整体发行采用 [Apache License 2.0](LICENSE)；继承自上游的代码保留其原始
[MIT License](LICENSE-MIT) 与版权声明，详细归属见 [NOTICE](NOTICE)。
