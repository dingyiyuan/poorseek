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

早期、自用。**SYCL 引擎就在 `main` 的 `sycl/` 里**（此前 Rust 侧的 `xolotl-sycl-*` 分支是更早的一次尝试，已被它取代）。
Vulkan / Metal / CPU 后端在 `crates/` 里，作为控制面与后续扩展的底座保留。

## 同机参考数据

在同一台机器（Arc Pro B70 32 GB + 192 GB DDR5）上跑 DeepSeek-V4.1-Flash IQ2_XXS（384 K 上下文、40 层、15 360 专家）的实测值：

| 场景 | 实测 |
|---|---|
| 解码（真实多样 prompt：短-中文 128 / 长-英文 512 / 中-中文 256） | **19.5 / 23.2 / 20.5 tok/s** |
| 解码（同一 prompt 重复跑，专家全热） | 26–27 tok/s |
| **预填充 45.6 K token** | **347 tok/s**（345.0 / 348.4 / 348.4，三次） |
| **预填充 122 K token** | **271 tok/s**（优化前 168） |
| **预填充 200 K token** | **267 tok/s** |
| 解码 @81 K 上下文 | 12.6 tok/s |
| 解码 @81 K 上下文 | 12.6 tok/s |

规模换算：100 K 输入约 **6 分钟**预填充（267–271 tok/s），之后解码 11–13 tok/s。

> 口径说明：**同一 prompt 重复跑会虚高**，因为专家命中率被推到接近 100%，不能拿它做配置对比。
> 上面第一行才是真实负载的数字。

## 已知的墙（测出来的，不是猜的）

- 真实负载下 **40% 的时间在等 CPU 车道**（约 110 个专家/token 只能从主机内存来）；CPU 车道已经跑在内存带宽上（约 38 GB/s，本机 DDR5 纯读实测 49–51.6 GB/s），**没有余量**。
- 稠密投影每 token 要把 6.2 GB 权重重读一遍，卡在显存带宽（约 483 GB/s），**也没有余量**。
- 投机解码（DSpark/MTP）在本机**不划算**：接受率 100%、草稿只要 7 ms/轮，但验证一步 285 ms，因为 6 个 token 的窗口每层要 36 个专家而不是 6 个。唯一能翻正的场景是显存大到让窗口的专家全部驻留。
- **长上下文的预填充曾经慢一半**，原因不是同步而是注意力 slab 的批大小按剩余显存自动收缩：
  `DS4_ATTN_SLAB=2048` + `--vram-experts 1.0` 之后 122 K 从 168 → 271 tok/s（`q_b+rope` 一项从 42.9% 降到 8.8%）。
- 每层解码约有 5–10 个内核串行发射，**GPU 有一半时间在等发射**（验证窗口实测 GPU 忙 50%，单 token 路径接近 100%）。


## 仓库里有什么

这是一个**完整的单卡推理框架**，两种后端并存：

| 目录 | 是什么 | 语言 |
|---|---|---|
| `crates/` | 控制面：分页 / Slot / LUT / host-tier、模型加载、服务端、CLI、对话、剖析。**`infr-vulkan` 是 Vulkan 后端（扩展性最强的那个，保留）**，另有 Metal 与 CPU 后端 | Rust |
| `sycl/` | **SYCL 引擎**：DeepSeek-V4.1-Flash 在 Intel Arc 上的实际推理实现（[说明](sycl/README.md)） | C++17 / oneAPI |
| `serve/` | HTTP 服务（OpenAI 兼容接口），驱动 `sycl/` 引擎 | Python |
| `tools/` | GGUF 打包与量化：`dsv41/pack_gguf.py`（含 DSpark drafter）、`trellis/` | Python |
| `docs/` | 实测记录：MTP/投机解码全量数据、DeepInfer 精读报告、故障恢复清单 | Markdown |

**目标配置**：Intel Arc Pro B70 32 GB + 192 GB 主机内存，DeepSeek-V4.1-Flash（40 层 × 384 路由专家，IQ2_XXS），上下文到 384 K。
实测真实负载解码 19.5–23.2 tok/s、预填充 45–200 K 为 347/271/267 tok/s（100 K 约 6 分钟）（详见 [sycl/README.md](sycl/README.md) 与 [docs/MTP_STATUS_2026-11-01.md](docs/MTP_STATUS_2026-11-01.md)）。

> 非官方、个人项目。不跟随任何上游的发布节奏，不提供通用安装支持。

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

## 阶段感知混合精度（预填充 / 解码分开配置）

DeepSeek 官方 checkpoint 里的路由专家是 **FP4(E2M1) + E8M0/32**，与 ggml MXFP4 同构，引擎已有该格式的内核。
所以我们把**前 20 层专家换成官方 FP4、后 20 层保持 IQ2**，打成一个"预填充专用包"：

| | IQ2 包（生产） | 混合包（L0–19 官方 FP4） |
|---|---|---|
| 专家内核耗时（122 K 预填充） | 72.7 s | **56.8 s（−22%）** |
| 122 K 预填充（端到端参考） | 367 tok/s | 416 tok/s |
| **解码** | ~21 tok/s | **6.74 tok/s（不可用）** |

解码掉到 6.74 的原因：MXFP4 每专家 blob 18.80 MB vs IQ2 的 7.74–10.23 MB，
显存按最大 blob 分槽 → 驻留槽位从约 40/层掉到约 25/层，大量专家改为从盘读。

**所以它只当预填充包用**，解码换回 IQ2。切换靠引擎自带的会话存档（含 KV）：

```
① 混合包 → 预填充 → slot_save（含 KV）
② 换 pack，重启引擎（约 10 s + 60 s 安全冷却）
③ IQ2 包 → slot_load → 解码（21 tok/s，原样）
```

已实测 **KV 可以跨 pack 承接**：换包重启后 839 token 的请求有 **822 个复用**，只新读 17 个
（专家格式不同，但 KV 布局与它无关）。工具：`tools/dsv41/two_phase.sh`、`tools/dsv41/mixed_pack.py`。

短输入不划算（切换开销约 70 s，滚雪球到 30 K token 以上才回本），应按提示长度自动选择。

> 精度上不亏：前 20 层从 IQ2 提升到官方 FP4，后 20 层不变；实测 IQ2 解码在这套模型的推理题上
> 没有可测量的损失（12 道硬题 11/12 vs 全精度 12/12，唯一的差异是我的答案提取器把 `9 sheep are left`
> 读成了 `left`）。IQ2 的量化损失要上千道题才测得出来，与文献里 GSM8K 上 1–4 分的量级一致。

工具链（都在 `tools/` 下）：`mixed_pack.py`（safetensors→混合包，逐位校验 max err = 0）、
`two_phase.sh`（两段式编排）、`overnight/gpu_lock.sh`（多 agent 时的 GPU 独占锁）、
`iq_bench.py` / `iq_hard.json`（可自动判分的智商基准）。
