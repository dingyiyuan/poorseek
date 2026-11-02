# 预填充高精度权重 / 解码 IQ2 分离设计

> 2026-11-02。只读分析 + 设计，全程未编译、未启动引擎、未碰 GPU。
> 所有代码结论带 `文件:行号`；数据来自 `tensors.txt` / safetensors 头 / GGUF 头的离线解析与 `serve-engine.log` 的实测 profile。
> 找不到的东西明确写「未找到」。

---

## 一、结论摘要（先看这一节）

| # | 结论 | 依据 |
|---|---|---|
| **1** | **引擎完全没有读 FP8 / safetensors 的路径。** 只有 `ds4.txt + tensors.txt + 平铺文件` 的 pack 格式；FP8 在本仓库只作为 KV cache 布局（`kKvFp8`）、Engram 表格式、数值转换工具存在，**不是权重格式**。 | `ds4_model.dp.cpp:425-523`（Pack::load）；`ds4_kernels.dp.hpp:26-27`；`ds4_numerics.dp.hpp:28-38`；`ds4_packinfo.cpp:43-45`（`dense_type` 声称支持 `MXFP8`，但 `mat()` 无此分支 → 见 §3.2） |
| **2** | **但官方 checkpoint 的专家根本不是 FP8，是 FP4(E2M1)+E8M0/32 —— 与 ggml `MXFP4` 同构**，而引擎**已有** MXFP4 专家的完整内核（matvec + WMMA tile）。所以「预填充高精度专家」不需要新格式，只需要**离线重打包**。 | `/data/models/dsv41-abliterated-fp8/config.json`（`expert_dtype: fp4, weight_block_size [32,32], scale_fmt ue8m0`）；`tools/dsv41/drafter_pack.py:10-12,68,193`（现成的 `fp4_to_mxfp4()`）；`ds4_gguf.dp.hpp:30`（`kFmtMXFP4`）；`ds4_gguf.dp.cpp:1360`（`mx_set_tc_kernel`，F16 WMMA） |
| **3** | **Q4_K_M 可以读（Q4_K 是唯一能当专家的 K-quant），但直接拿 417 GB 那套 GGUF 会静默算错**：① 它是 llama.cpp 命名的 `blk.N.ffn_*_exps.weight` 批量张量，需要 `pack_gguf.py` 拆包；② 其中 **20 层的 `ffn_down_exps` 是 Q6_K，而专家内核 `DS4_IQ_SWITCH` 不含 Q6_K/Q5_K**，`supported()` 却返回 true → 该层专家上 VRAM 后每个 kernel 都 `return false` 被吞掉。 | `pack_gguf.py:141-143,663-677`；`ds4_iq.dp.cpp:1840-1855`（switch 只有 13 个类型）；`ds4_iq.dp.cpp:2438-2454`（`supported()` 却含 kQ5_K/kQ6_K）；`ds4_iq.dp.cpp:2608-2648`（只有 GEMV `mv` 有 Q5_K/Q6_K）；`ds4_model.dp.cpp:4977-4985` + `ds4_gguf.dp.cpp:2341`（失败被吞） |
| **4** | **「两套权重并存」在 32 GB 上不可能双份驻留**（MXFP4 专家 blob 18.80 MB vs 现 IQ2 7.74–10.23 MB，2.43×）。但**根本不需要驻留**：chunk 预填充对每个专家只读一次（无复用），所以高精度专家可以**只走流式路径**，解码期完全不碰。 | blob 计算见 §2.3；`place_experts` 逐层按 `n_slots*unit(l)` 分配（`ds4_model.dp.cpp:5494-5505`）；chunk 流式见 `flow_push`（`:4347`）、ring（`:3858-3904`） |
| **5** | **最大风险不是精度也不是带宽，是速度**：现有专家内核 4.2–6.0 TFLOPS 是 dp4a 路径；MXFP4 tile 走 F16 WMMA。若它在 B70 上实测也只有 6 TFLOPS，整个方案就只剩质量收益。**必须先花半天测这一个数。** | `serve-engine.log:25-31`（实测 4.2–6.0 TFLOPS）；`ds4_gguf.dp.cpp:1352-1366`（MXFP4 tile 用 shared-mem F16 + WMMA） |

**一句话方案**：把官方 FP8/FP4 checkpoint 的 **L0–19 专家离线重打包成 ggml MXFP4**（逐位精确，144.4 GB）+ dense 转 BF16/Q8_0；引擎侧只加「按相位选择权重来源」的一层间接（约 6 处改动），预填充走第二来源的流式路径，解码维持现有 IQ2 + VRAM 常驻槽位。

---

## 二、数据基线（本机实测 + 离线解包）

### 2.1 实测 profile（122k token prompt，当前生产配置）

`/data/strata-new/serve-engine.log`：

| 行 | 内容 |
|---|---|
| `:4` | `the expert ring: 208 slots x 10229760 bytes (2029 MB) in VRAM` |
| `:7` | `weights 6.20 GB, experts 14.23 GB (1616 of 15360 in VRAM), caches 0.36 GB, 595 MiB free` |
| `:17` | `experts (incl waits) 234006.2 ms 52.4%` |
| `:31` | `expert kernels total 222970.9 ms` |
| `:32` | `prompt 121937 tokens ... 447976 ms (272.2 tok/s)` |

专家内核按类型（`:25-30`）：IQ2_XXS gate+up 42 950 ms @4.2 TFLOPS、down 10 052 @7.0；IQ2_XS gate+up 16 019 @4.5；IQ1_M gate+up **106 885 @4.2**、down 46 761 @6.0。

→ 预填充时间的 **49.8%** 是 2-bit 金币解量化的 ALU。gate+up 是重灾区（两个 5120-深的内积），4.2 TFLOPS。

### 2.2 生产包（`/data/models/dsv41-iq2xxs-pack-real`，从 `tensors.txt` 统计）

| 项 | 值 |
|---|---|
| 张量 / 文件数 | 47 248 / 6 |
| 总计 | 341.3 GB |
| 路由专家 | **132.07 GB**（IQ1_M 59.45 + IQ2_XXS 49.05 + IQ2_XS 23.57） |
| Engram 表 | 202.97 GB（`engram.embed.weight` F8_E4M3 196.61 + `.scale` F8_E8M0 6.14） |
| 其余 dense（attn/shared/gate/embed/head） | **5.38 GB**（Q5_K 3.96、Q6_K 1.38、Q8_0 0.73、BF16 0.28、F32 0.16） |

每专家 blob（= `Layer::blob`，`ds4_model.dp.cpp:845`）：L0 7 741 440 B（7.74 MB）、L19 9 400 320 B（9.40 MB）、L20 7.74 MB、**L39 10 229 760 B（10.23 MB）** —— 与 `serve-engine.log:4` 的 ring stride 完全一致。

### 2.3 三种候选权重集的字节账（离线解析，非估算）

**(a) 官方 `dsv41-abliterated-fp8`（48 分片 safetensors，510.3 GB）**

| 角色 | dtype/形状 | 字节 |
|---|---|---|
| 专家 `w1.weight` | **I8** `(2304, 2560)` = 2304×5120 的一半字节 → **FP4 两值/字节** | 5 898 240 |
| 专家 `w1.scale` | F8_E8M0 `(2304, 160)`，160 = 5120/32 → **每 32 值一个 E8M0** | 368 640 |
| 专家 L0–19 | | **144.39 GB** |
| 专家 L20–39 | | 144.39 GB |
| dense(attn+shared) L0–19 | F8_E4M3 + E8M0 | 3.45 GB |
| Engram | F8_E4M3 | 203.07 GB |

→ 专家 = 4 bit + 8/32 = **4.25 bpw**，与 ggml MXFP4（32 值 17 字节）**同位宽、同值集**。重打包为 MXFP4 后 **144.4 GB for L0–19**，逐位不变。
交叉验证：40 层 × 384 专家 × 3 矩阵 × 11796480 值 × 4.25/8 = **288.8 GB** = 144.39×2 ✓。

**(b) `dsv41-abliterated-q4km`（10 分片 GGUF，417 GB，header 实测）**

| 角色 | 类型 | L0–19 | L20–39 |
|---|---|---|---|
| `ffn_gate_exps` + `ffn_up_exps` | Q4_K | 101.92 GB | 101.92 GB |
| `ffn_down_exps` | Q4_K（20 层）/ **Q6_K（20 层）** | 25.48 + 37.16 | 25.48 + 37.16 |
| 合计专家 | | **164.6 GB** | 164.6 GB |

dense 非 engram ≈ 5.0 GB；`engram_embd` 110.6 GB（Q4_K）。

**(c) 重打包成 MXFP4 后每专家 blob**

`3 × (11796480/32 × 17) = 3 × 6 266 880 = 18 800 640 B = 18.80 MB`。

| 层 | IQ2（MB） | MXFP4（MB） | 倍数 |
|---|---|---|---|
| L0 | 7.74 | 18.80 | **2.43×** |
| L39 | 10.23 | 18.80 | 1.84× |

→ 同样 14.23 GB 的专家 VRAM：现在 40 槽/层（1616/40），混合 pack 下按 `place_experts` 的比例分配（`ds4_model.dp.cpp:5427-5440`）会降到约 **25 槽/层（−38%）**。这是「静态混合 pack」对**解码**的代价，量化见 §6.2。

---

## 三、Q1：引擎能直接读 FP8 或 Q4_K_M 吗？

### 3.1 引擎认哪些权重格式（完整清单）

专家与 dense 共用两套内核命名空间：`gg::`（GGUF 列式块量化，`ds4_gguf.dp.cpp/.hpp`）与 `x3::`（EXL3 trellis，`strata/dsv41/exl3.hpp`）。`gg::` 内部再分两路：MXFP4 自己的内核，以及转给 `iq::`（`ds4_iq.dp.cpp`）的低位类型。

**矩阵存储格式（`x3::Mat::fmt`）** — `ds4_gguf.dp.hpp:29-32`：

| fmt | 名称 | 谁用 |
|---|---|---|
| 0 | EXL3（trellis/suh/svh + mul1） | 老 pack |
| 1 | `kFmtBF16` | dense |
| 2 | `kFmtMXFP4` | 专家（也用于专家矩阵的 matvec） |
| 3 | `kFmtQ8_0` | dense |
| 4 | `kFmtGG` | dense，低位 ggml 块类型（`m.K`） |

**专家块类型（`x3::ExpertLayout::K[r]`，r=gate/up/down）** — `ds4_gguf.dp.hpp:44-45` + `ds4_gguf.dp.cpp:1803-1812`：

| 类型 | ggml id | 解码器 | dense 读 | matvec(GEMV) | tile(chunk 预填充) |
|---|---|---|---|---|---|
| MXFP4 | 39 | `ds4_gguf.dp.cpp` 自带 | ❌（`gg::rows:2073` 拒） | ✅ `:1907` | ✅ `mx_set_tc_kernel:2380` |
| IQ2_XXS | 16 | `ds4_iq` | ✅ | ✅ | ✅ |
| IQ2_XS | 17 | 〃 | ✅ | ✅ | ✅ |
| IQ2_S | 22 | 〃 | ✅ | ✅ | ✅ |
| IQ3_XXS | 18 | 〃 | ✅ | ✅ | ✅ |
| IQ3_S | 21 | 〃 | ✅ | ✅ | ✅ |
| IQ4_XS | 23 | 〃 | ✅ | ✅ | ✅ |
| IQ1_S | 19 | 〃 | ✅ | ✅ | ✅ |
| IQ1_M | 29 | 〃 | ✅ | ✅ | ✅ |
| IQ4_NL | 20 | 〃 | ✅ | ✅ | ✅ |
| Q4_K | 12 | 〃 | ✅ | ✅ | ✅ |
| Q2_K | 10 | 〃 | ✅ | ✅ | ✅ |
| Q3_K | 11 | 〃 | ✅ | ✅ | ✅ |
| **Q5_K** | 13 | 〃 | ✅ | ✅（`ds4_iq.dp.cpp:2641`） | **❌**（不在 `DS4_IQ_SWITCH`） |
| **Q6_K** | 14 | 〃 | ✅ | ✅（`:2642`） | **❌**（同上） |
| trellis TQK4/5/6/7 | 145/146/224-227 | `ds4_tq.dp.cpp` | — | ✅ | ✅ |
| **FP8 E4M3 / E5M2（权重）** | — | **未找到** | **未找到** | **未找到** | **未找到** |

依据：`DS4_IQ_SWITCH` 宏 `ds4_iq.dp.cpp:1840-1855`（只列 13 个类型，无 Q5_K/Q6_K）vs `supported()` `:2438-2454`（含 kQ5_K/kQ6_K）—— 两者不一致，`supported()` 会把 Q5_K/Q6_K 的层判为「可上 VRAM」，然后 `set_gate_up`/`gate_up` 走 switch 的 `default: return false`。**这是一个已存在的静默错误陷阱**（见 §8 R5）。

### 3.2 FP8 / safetensors：没有读取路径

- pack = 目录 + `ds4.txt`（几何/元数据）+ `tensors.txt`（`file <i> <name>` 与 `t <name> <dtype> <file> <off> <bytes> <dims>`）+ 被 `model_dir` 引用的平铺文件。`Pack::load` 只做 `open`/`mmap`/`pread`（`ds4_model.dp.cpp:425-523`），**没有 safetensors 头解析**。全仓库对 `safetensors` 的引用只出现在 `tools/`（离线 python）和 `serve/test_setup_nvfp4.py`。
- FP8 在引擎里只出现在三处，**都不是权重矩阵**：
  1. KV cache 布局 `kKvFp8 = E4M3 + UE8M0/32`（`ds4_kernels.dp.hpp:26-27`）；
  2. Engram 行格式：MXFP8（scale+32 codes 交错，33 B/32）、Q6_K、IQ4_NL，以及「老 pack：FP8 codes 一个张量、E8M0 scale 另一个张量」（`ds4_model.dp.cpp:1196-1197,3301-3314`；`ds4_engram_io.hpp:95`）；
  3. E4M3 数值转换工具（`ds4_numerics.dp.hpp:28-48`）。
- `ds4_packinfo.cpp:43-45` 的 `dense_type()` **声称** dense 可读 `MXFP8`，但 `mat()`（`ds4_model.dp.cpp:1996-2000`）只分派 `BF16/Q8_0/Q4_K/Q5_K/Q6_K`，其余落到 EXL3 分支并以 `:2006 "not an EXL3 matrix in the pack"` 失败。**这是 pack-info 与加载器的口径不一致（报告 bug），不是能力。**

### 3.3 关键发现：官方 checkpint 的专家 == MXFP4

`config.json`：`{"quant_method":"fp8","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}`。

张量实形：`layers.0.ffn.experts.0.w1.weight` = `I8 (2304, 2560)` + `w1.scale` = `F8_E8M0 (2304, 160)`。2560 = 5120/2 → 每字节 2 个 4-bit 码；160 = 5120/32 → 每 32 值一个 E8M0 指数。**这正是 ggml MXFP4 的值集与缩放粒度**，只差「两个张量分开存 + nibble 顺序」。

仓库里已经有这个转换器，且已用于 DSpark drafter：

- `tools/dsv41/drafter_pack.py:10-12`（注释写明 FP4 两值一字节、低位在前、E8M0 每 32）
- `tools/dsv41/drafter_pack.py:68` `def fp4_to_mxfp4(w, s, rows, k)` —— **逐位精确**重排成 MXFP4 块
- `tools/dsv41/drafter_pack.py:193-196` `elif e["dtype"] == "I8": ... out, dt = fp4_to_mxfp4(...), "MXFP4"`
- 参考解码器：`tools/dsv41/dspark_ref.py:77-85`

> 待确认（低风险）：`drafter_pack.py:84` 的 E2M1 表是 `[0,1,2,3,4,6,8,12,...]`（两倍），`dspark_ref.py:44` 是 `[0,0.5,1,1.5,...]`；两者差一个 2 的因子，落到 pack 里必须和 `gg::` 的 `kvalues`（`ds4_gguf.dp.hpp:9-11`：`{0,1,2,3,4,6,8,12,...}`，注释明确说是「E2M1 值 ×2」）对齐后再做数值比对。

### 3.4 Q4_K_M：能读，但有两个硬坑

1. **命名/形状**：llama.cpp 约定 `blk.N.ffn_gate_exps.weight` 是 `[n_exp][ff][dim]` 一个张量（实测 `[5120,2304,384]` Q4_K，2.548 GB）。pack 要 `layers.N.ffn.experts.E.w1.weight` 逐专家。→ `pack_gguf.py:141-143,663-677` 已有拆分逻辑，**这一步是成熟的**。
2. **Q6_K 的 down（20 层）没有专家内核**：`DS4_IQ_SWITCH`（`:1840-1855`）无 `kQ5_K/kQ6_K`，`iq::set_gate_up/set_down/gate_up/down` 全部走 `default: return false`。而 `supported()`（`:2438`）返回 true → 层被判为设备可达 → 专家上 VRAM → 每个 kernel 失败被 `fail()` 吞掉（`ds4_gguf.dp.cpp:2341,2405`）+ `run_set` 不看返回值（`ds4_model.dp.cpp:4978-4980`）→ **输出是静默错值，不是崩溃**。
   → 必须先把那 20 层 down 重量化成 Q4_K（或 IQ4_XS/IQ3_S 等 switch 内类型）。`tools/dsv41/expert_requant.cpp` 就是干这个的（`--down IQ4_NL` 之类，见其头注释 `:1-8`），`patch_pack_experts.py` 负责改写 `tensors.txt`（`:43-88`）。

### 3.5 要新增什么（按代价从小到大）

| 选项 | 新增内容 | 判定 |
|---|---|---|
| **A. 官方 FP4 → MXFP4（推荐）** | 一个离线 python 转换器：safetensors 头解析 + `fp4_to_mxfp4()`（已存在）+ 按 `layers.N.ffn.experts.E.w{1,3,2}.weight` 写 pack + `tensors.txt`。dense 走 F8_E4M3→BF16/Q8_0（`drafter_pack.py` 已有 FP8→BF16）。**C++ 零改动**（单格式 pack）。 | ✅ 逐位精确、命中引擎最快的 tile 内核、工具已有 80% |
| **B. Q4_K_M 直接上** | 需要把 20 层 Q6_K down 重量化成 Q4_K（`expert_requant`），再 `pack_gguf.py` 重打包。C++ 零改动。 | ⚠️ 可行但多一道 417 GB 重打包 + 4.5 bpw 比 4.25 bpw 更大 |
| **C. 让引擎原生读 safetensors / FP8** | `Pack` 加 safetensors 后端；`mat()`/`expert_layout_mx()` 加 FP8 分支；新增 `gg::` matvec+tile 的 FP8 内核；`gg::rows` 加 FP8 dense。 | ❌ 改造面 ≫ 方案 A，收益为 0（FP8 dense 直接转 BF16 即可，专家本来就是 FP4） |

**所以：不需要新格式，不需要新内核。** 需要的是打包工具 + 相位选择机制（§6）。

---

## 四、Q2：专家路径的格式分派点与改动面

### 4.1 现有的 5 个分派层

```
pack 张量 dtype ──► gg_type_of()                     ds4_model.dp.cpp:2269       "MXFP4"/"Q4_K"/"IQ2_XXS"… → ggml id
                └─► expert_layout_mx()                ds4_model.dp.cpp:2274       → lay.fmt=kFmtMXFP4, lay.K[r], blob, mx_file/off/stride
                    （否则 expert_layout() :2414 走 EXL3 trellis/mul1）        :2306-2312 层内混用校验
lay.fmt/K ──────► gg::mx_experts(lay)                ds4_gguf.dp.hpp:45          → MXFP4 自己的内核 or 转 iq::
                ├─► gg::moe_gate_up / moe_down        ds4_gguf.dp.cpp:2128 / 2164   （解码 T=1 与 window）
                ├─► gg::set_gate_up / set_down        ds4_gguf.dp.cpp:2325 / 2391   （chunk 预填充 tile）
                └─► gg::win_experts                   ds4_gguf.dp.cpp:2214
x3::Mat::fmt ───► gg::mv                              ds4_gguf.dp.cpp:1907        fmt1 BF16 / fmt2 MXFP4 / fmt3 Q8_0 / fmt4→iq::mv
                └─► gg::rows                          ds4_gguf.dp.cpp:2062        dense 反量化（BF16/Q8_0/GG，**无 MXFP4**）
调用侧 ─────────► mv_any()                            ds4_model.dp.cpp:9062       `fmt != 0 ? gg::mv : x3::mv`
                ├─► run_set() chunk 专家分派          ds4_model.dp.cpp:4977-4985  `lay.fmt == kFmtMXFP4 ? gg::set_* : x3::set_*`
                ├─► 解码 token 专家                   ds4_model.dp.cpp:12786-12796 `mx ? gg::moe_gate_up : x3::…`
                └─► 解码 window 专家                  ds4_model.dp.cpp:12849      `lay.fmt == kFmtMXFP4 ? gg::win_experts : x3::…`
```

### 4.2 新增一种专家格式要动的地方（若是新格式；本次方案 A 不需要）

| # | 位置 | 改什么 |
|---|---|---|
| 1 | `ds4_model.dp.cpp:2269-2273` `gg_type_of` | 把新 dtype 名映射到 id |
| 2 | `ds4_model.dp.cpp:2274-2337` `expert_layout_mx` | 行字节、blob 布局、`lay.K[r]`、`:2306-2312` 混用规则 |
| 3 | `ds4_gguf.dp.cpp:1803-1812` | `type_row_bytes / type_block / type_name / type_on_device / type_on_chunk` |
| 4 | `ds4_iq.dp.cpp:2438-2454, 2463-2482` | `supported() / block_bytes()`（**必须与内核 switch 严格一致**，现在 Q5_K/Q6_K 就不一致） |
| 5 | `ds4_iq.dp.cpp:1840-1855` `DS4_IQ_SWITCH` | 内核实例化：`Ty<T>` 布局 + 反量化 + dp4a |
| 6 | `gg::` 5 个入口 | `moe_gate_up:2128 / moe_down:2164 / win_experts:2214 / set_gate_up:2325 / set_down:2391` 各加一路分支 |
| 7 | `gg::mv:1907` | `f` 分档（现在 3 档：BF16/MXFP4/Q8_0 + GG 单独） |
| 8 | `gg::rows:2062` | dense 反量化（现在无 MXFP4、无 FP8） |
| 9 | `ds4_model.dp.cpp:1996-2000` `mat()` | dense 加载分派 |
| 10 | `ds4_model.dp.cpp:9062` `mv_any` | `fmt != 0` 的判定 |
| 11 | `ds4_packinfo.cpp:30-45` | `device_type()/dense_type()` 报告口径 |
| 12 | `ds4_model.dp.cpp:6282` `init_store` | 现在硬要求 `L[0].lay.fmt == kFmtMXFP4` |

**结论：新增格式约需改 12 处 / 加 2–5 个内核入口。而本方案（FP4→MXFP4 重打包）一处都不用改。**

---

## 五、Q3：两个权重集并存的资源账

### 5.1 引擎当前的单 pack 假设在哪

| 位置 | 内容 |
|---|---|
| `ds4_model.hpp:26` | `Options::pack` 单一路径 |
| `ds4_model.dp.cpp:1025` | `Pack pk;` —— `Impl` 的成员，一个 Model 一个 pack |
| `ds4_model.dp.cpp:2502-2505` | `load()` 第一句 `pk.load(o.pack)` |
| `ds4_model.dp.cpp:845` | `Layer::blob` —— 每层一个固定专家 blob 大小 |
| `ds4_model.dp.cpp:883-884` | `mx_file[3]/mx_off[3]/mx_stride[3]` —— 每层一个文件几何 |
| `ds4_model.dp.cpp:5494-5505` | `sycl::malloc_device(l.n_slots * unit(x))` —— 驻留区按 blob 分配一次 |
| `ds4_model.dp.cpp:1743` | `unit(l) = nm>1 ? hblob : blob` |
| `ds4_model.dp.cpp:6280-6303` | `init_store` 把每层几何拷进 `ExpertStore::LayerGeo`，**init 之后不可变** |
| `ds4_model.dp.cpp:3858-3864` | ring stride = **max over layers** 的 `align256(unit(l))` |

### 5.2 三个方案

**S1 — 两个 `Model` 实例**

- 做法：prefill 用 `Model B`（高精度 pack），decode 用 `Model A`（IQ2 pack），相位间用 `slot_save`/`slot_load` 搬状态（`ds4_model.dp.cpp:5811 / 5975`，已经在 serve 里用于会话存取，`ds4_main.cpp:905,949`）。
- 改动面：**C++ ≈ 0**；`ds4_main.cpp` 里加"两个 model、按相位切换指针 + 状态搬运"，外加 `Options` 允许两个 pack。
- VRAM：**双份 dense（6.20 GB × 2）+ 双份专家驻留区 + 双份 scratch**。32 GB 上不可能（现单实例已只剩 595 MiB）。即使 `--vram-experts` 减半，两个实例的专家驻留都严重不足，解码会退化。
- 状态搬运成本：`slot_save/slot_load` 走完整会话状态（ring + 压缩 KV + index keys + 位置），122k 会话状态量级 ≈ 数百 MB ~ 1 GB 级（`serve-engine.log:33` 显示 121 941 token 会话 = 0.12 GB）→ 可接受，但会**丢掉设备上的 KV 与专家热区**。
- 判定：**❌ 不可行**（VRAM），除非允许"串行两次 ready / 两个进程"（那预热要 2× 加载时间）。

**S2 — 单 Model，逐层双几何 + 相位重映射驻留区**

- 做法：`Layer` 增一个 `alt` 权重来源结构（`{mx_file[3], mx_off[3], mx_stride[3], lay, blob, lay2}`），`phase ∈ {prefill, decode}` 决定 `mx_mat/mx_part_bytes/mx_read_off/unit` 读哪套；相位切换时 `cudaFree + malloc` 每层驻留区（尺寸按相位）。
- 改动面：`Layer:805-885`、`load:2502`、`place_experts:5356-5505`、`read_mx:2378 / read_mx_file:2386 / mx_mat:2338 / mx_part_bytes:2348 / mx_read_off:4333`、`flow_push:4347`、`unit:1743`、`init_store:6280`。**约 8 处，且每处都是"多一个来源参数"。**
- 成本：切换时重填驻留区 ≈ 14–20 GB / ~20 GB/s ≈ **0.7–1 s**（一次性，每请求 2 次）。dense 不用动（解码与预填充共用同一套 BF16/Q8_0 dense，见 §7 Phase 3）。
- 风险：`cudaFree/malloc` 在 32 GB 上碎片化；`ExpertStore` 需重建（arena 是按 `max blob` 预留的，见 `ds4_store.hpp:80`）。

**S3 — 单 Model，预填充不驻留（**推荐**）**

- 观察：chunk 预填充对每个专家**只读一次、无复用**（122k prompt 一次跑完全部 15 360 个专家：`serve-engine.log:7` 只有 1616/15 360 驻留，其余走 CPU lane / 流式）。所以预填充阶段给高精度专家准备 VRAM 驻留区**毫无收益**。
- 做法：解码维持现状（IQ2 + 14.23 GB 驻留 + `ExpertStore`）；**预填充的 chunk 路径改从第二个来源流式读**——即 `flow_push`（`:4347`）/ `read_mx_file`（`:2386`）/ `mx_read_off`（`:4333`）按当前相位选 `mx_file/mx_off/mx_stride/lay`，ring 槽位按 `max(blob_A, blob_B)` 定尺寸。
- 改动面：**4–6 处**（`Layer:805-885`、`load:2502`、`flow_push:4347`+`:4408`+`:4644-4649`、`run_set:4900-4912` 的 `sa.wbase/stride`、ring stride `:3858-3864`）。**不动 `place_experts`、不动驻留区、不动 store、不 malloc/free。**
- 代价：
  - ring stride 10.23 MB → 18.80 MB。要保 208 槽 = 3.9 GB（多 1.9 GB VRAM），或保 2 GB → 108 槽。注意 `:3901` 有一条实测硬地板：`if (ds4_ring_device() && mb < 2048) mb = 2048;`（1014 MB 曾把 8K prompt 的 copy stream 卡死），所以不能简单减半，**需要实测 108 槽的 overlap 是否够**。
  - 预填充专家字节：L0–19 从 ~60 GB 涨到 ~144 GB（每专家读一次，PCIe/页缓存）。按 20–24 GB/s PCIe → +3.5 s/次长 prompt（相对 448 s 可忽略）；但如果那时数据要从 NVMe 冷读（144 GB / 6 GB/s = 24 s）就会痛 → **必须让第二 pack 常驻页缓存**（192 GB 内存，装 144 GB + IQ2 的一部分是可行的）。
- 判定：**✅ 主方案**。

---

## 六、Q4：相位切换点与现有 swap 机制能否复用

### 6.1 切换点在哪里

**服务层（真正发生"预填充→解码"的地方）** — `src/dsv41/ds4_main.cpp`：

| 行 | 内容 |
|---|---|
| `:1092-1116` | `while (ok && done < total)` —— prompt 的 chunk 循环，每次调 `m.feed()`（`:1102/1104`）或 `m.feed_mark()`（`:1106`） |
| `:1112` | `PP %zu %zu` 进度行（预填充结束的那一行） |
| `:1118` | `m.set_cancel(nullptr)` —— **prompt 读完** |
| `:1144` | `const double t1 = now_ms()` —— 预填充计时收尾 |
| `:1206` | `round(m, tok, room, ...)` —— **第一个解码 round**（内部 `Model::step` → `forward`） |

→ **相位切换点 = `ds4_main.cpp:1116` 与 `:1206` 之间**（插在 `:1118` 附近最干净）。

**引擎层**：

- 预填充：`Model::feed()` `ds4_model.dp.cpp:13218-13233` → `pick_path()` → `forward_chunks()` `:11301`
- 解码：`Model::step()` `:13217` → `forward()` `:11901` → `forward_par()` `:12153` / `forward_serial()` `:11997` → window 路径（`T==1` 分支 `:12749`）
- YOCO 判据全在这里：`yoco_layer()` `:11285-11297`（`ratios[il]==1` 且 `kv_src` 的第一个 → **20**），`keep` 计算 `:11330-11332`，`run.yoco/cut_k/cut_r` 赋值 `:11344-11348`，层跳过判据 `runs/kv_only/r0` `:10362-10364`，chunk 流式载荷 `fl_yoco/fl_cut_k/fl_cut_r` `:10643-10645` / `:779` / `:4481-4493`。
- `DS4_YOCO_KEEP=160`（生产 `strata-server.json`）→ `a = n - 160`，`per = n`（单 chunk，因为 `feed_unit = max(chunk, max_context)` 且 `--prefill-chunk 40960`）。

### 6.2 现有的 `preload_store` / expert swap 能复用吗

**不能直接复用，但是最好的模板。**

| 机制 | 位置 | 粒度 | 为什么不能直接用 |
|---|---|---|---|
| `preload_store()` | `:6474-6545` | 一次性把 profile 最热的专家读进 RAM arena | 只往 RAM 里装、只按 `l.blob` 尺寸、方向单一（file→RAM），且 `ExpertStore` 几何在 `init_store:6280` 定死 |
| `Swap` + `cache_loop` | `:1265-1270`、`:8240-8256` | **单个专家**（layer, in, out） | 槽位大小 = `l.blob`（`swap_in_mx:8389` 拷 `l.blob` 字节），格式一换 blob 就变 → 越界/错位 |
| `cache_tick_swaps` / `cache_settle` | `:8572` / `:8509` | window 边界批量提交 | 只处理"同一个 pack 内的专家进出"，不处理"整个权重集换源" |
| `store_sync()` | `:6548` | window 边界把 store 槽位发布成 zero-copy | 同上 |

**能复用的部分（正是 S3 需要的）**：pinned staging、异步 copy stream、`st.Wait`/`settle` 在 window/chunk 边界的对齐、`flow_push` 的 ring 槽位管理、`read_mx_file` 的 `pread` 路径。S3 不新增搬运器，只是**让这些路径的参数（file/off/stride/lay/blob）来自当前相位的来源**。

若要"整层整段换入换出"（S2），需要新写一个 `phase_switch()`：在 `:1118` 处 `d.st->wait()` → 释放/重分配每层驻留区 → 重填 → 重建 `ExpertStore`（或用两个 store 实例）→ 重写 device 侧 `vram_tab`/`zc_tab`。这是**新增 ~200 行**，不是复用。

---

## 七、Q5：分阶段实施计划

### Phase 0（半天，**零 C++**，先做这个）
**测 MXFP4 tile 内核在 B70 上的真实 TFLOPS。** 全案的收益假设压在这一个数上。

- 做法：把 1–2 层的专家离线转成 MXFP4（`drafter_pack.py:68` + `patch_pack_experts.py`），用 `--prefill-chunk` 跑一个 4k prompt，读 `ds4 prof expert kernels type MXFP4 ...` 那一行（打印在 `ds4_model.dp.cpp:11611`，格式见 `serve-engine.log:25-30`）。
- 现成基准 `sycl/src/dsv41/iq_expert_test.cpp` 只覆盖 iq 类型（`--type 16,17,18,21,22,23,29`，`:5`），**没有 MXFP4 用例**；要么补 ~100 行，要么就用端到端。
- 判据：**MXFP4 tile ≥ 12 TFLOPS → 继续；≤ 8 TFLOPS → 停，改做其它方向**（专家时间 223 s 只省 <25%，不值得引入相位切换的复杂度）。

### Phase 1（1–2 天，零 C++）：静态混合 pack
L0–19 = MXFP4，L20–39 = 现 IQ2。**验证精度收益 + 量化解码代价。**

- 需要写的：`tools/dsv41/fp8_to_pack.py`（~200 行：读 safetensors 头 → 专家 `fp4_to_mxfp4`（已存在）→ dense F8_E4M3→BF16/Q8_0 → 写 `tensors.txt` + 一个平铺 bin）。**引擎零改动**：`expert_layout_mx:2274` 本来就按层读类型，`:2306-2312` 的混用校验只在**层内**生效。
- 必须同时量：
  - 预填充 tok/s（期望 +30~60%，取决于 Phase 0）；
  - **解码 tok/s**：专家槽位会从 40/层掉到约 25/层（§2.3 的 2.43× 与 `place_experts` 的比例分配 `:5427-5440`），CPU lane 占比会上升 → 这是这个方案的**唯一硬伤**，也是 Phase 2 存在的理由；
  - ring：stride 抬到 18.80 MB，按 `:3901` 的地板要么 3.9 GB/208 槽，要么 2 GB/108 槽。
- **已知坑**：Q4_K_M 路线要先把 20 层 Q6_K down 重量化（§3.4），否则静默错值。

### Phase 2（3–5 天，小 C++）：相位分离（S3）
- 目标：预填充用 MXFP4 流式，解码用 IQ2 驻留，把 Phase 1 的解码损失归零。
- 改动清单（预计 ~150 行 diff）：
  1. `Layer:805-885` 加 `struct Wsrc { int mx_file[3]; uint64_t mx_off[3], mx_stride[3]; x3::ExpertLayout lay; size_t blob; }` 数组（2 个）+ `int phase;`
  2. `load():2502` 解析第二个 pack 目录（新 `Options::pack_prefill`），只填 `Wsrc`，**不分配 VRAM**
  3. `mx_mat:2338 / mx_part_bytes:2348 / read_mx:2378 / read_mx_file:2386 / mx_read_off:4333` 取当前相位的 `Wsrc`
  4. `flow_push:4347`（`:4368/4375/4408`）的 `l.lay.trellis` / stride → 相位来源
  5. `run_set:4900-4912` 的 `sa.lay` + `stride`（`wbase` 是 ring，不受影响）
  6. ring stride `:3858-3864` 取 `max(unit_A, unit_B)`
  7. 切换 API：`Model::set_phase(prefill|decode)`，在 `ds4_main.cpp:1118` 调一次
- 验收：解码回到 21.2 tok/s（`docs/STATUS_2026-11-02.md` 基线）±2%，预填充保持 Phase 1 的增益，答案逐 token 与 Phase 0 前一致（同一 prompt 的 logits 比对）。

### Phase 3（1–2 天，零 C++）：dense 高精度（先于专家，性价比最高）
- **关键事实**：`gemm():3944-3998` 无论权重是什么格式，都先 `gg::rows()`/`x3::reconstruct_rows()` 反量化到 `c.w16`（F16），再走 `dpct::blas::gemm`（F16 操作数、`compute_type::f32`）。**所以 dense 精度上限 = BF16/F16，且改精度不改 GEMM 时间**，只改反量化带宽（0.87 TFLOPS 的 GEMM 主导）。
- 做法：`tools/dsv41/dense_requant.py`（`:4-13`，支持 Q4_K/Q5_K/Q6_K/Q8_0，`--include/--exclude` 正则）把 dense 换成 **Q8_0 或 BF16**；或直接用官方 F8_E4M3 转 BF16（`drafter_pack.py` 的 FP8 分支）。`mat():1999` 与 `gg::rows:2062` 都已支持 BF16/Q8_0。
- 收益：dense 只占 5.38 GB、预填充里 `q_b+rope 8.9% + compressor/indexer 17.4% + sparse attention 10.6%`（`serve-engine.log`）中相当一部分来自 dense 权重精度 → 质量收益近乎免费。
- **注意**：这是"顺带做"的一步，**不要**为 dense 新增 FP8 内核（收益 0）。

### Phase 4（可选）：Q4_K_M 路线
只有当 Phase 0 显示 MXFP4 不如 Q4_K 的 dp4a、或需要 4.5 bpw 的更高精度时才做。要先补 Q6_K 的专家 kernel（或重量化），风险最高。

### 不建议做
- 让 `Pack` 原生读 safetensors / 原生 FP8 权重（§3.5 C）。
- 双 `Model` 实例（§5.2 S1，VRAM 不允许）。
- 在 Phase 1 之前动 `place_experts` / `ExpertStore`（它们不是瓶颈）。

---

## 八、风险清单（按严重度）

| # | 风险 | 量化 | 缓解 |
|---|---|---|---|
| **R1** | **MXFP4 tile 在这颗 B70 上的吞吐未知。** dp4a 路径 4.2–6.0 TFLOPS 是实测；WMMA 路径没有任何实测数据。若也慢，全案只剩质量收益。 | 专家占预填充 49.8%（`serve-engine.log:31`）；若 6→6 TFLOPS 则收益 0 | **Phase 0 先测**，不达标就停 |
| **R2** | 前 20 层换 4.25 bpw 后 expert blob 2.43×（7.74→18.80 MB），解码驻留槽位 40→~25/层 | 解码可能 −10~20% | Phase 2 相位分离；或 `--vram-experts` 重调 |
| **R3** | **`run_set` 忽略 `gg::set_*` 的返回值**（`:4978-4980`），格式不支持时静默产出错值 | 全 40 层结果污染，无报错 | 任何新格式落地前，先把 `:4977-4985` 改成硬失败 + `errors++` 检查 |
| **R4** | 「后 20 层不必送显存」成立，但**「不必读字节」不成立**：keep=160 行 × 6 ≈ 960 次选择覆盖全部 384 专家 → 20 层仍要各流一次 ≈ 60 GB | +8~10 s/次长 prompt（IQ2） | 后 20 层继续用 IQ2（正是本方案），且让它常驻页缓存 |
| **R5** | `supported()` 与 `DS4_IQ_SWITCH` 不一致：Q5_K/Q6_K 被报为「设备可达」但内核 `return false` | Q4_K_M 直接上必踩 | 修 `supported()`，或打包时重量化 |
| **R6** | ring 有硬地板 `mb ≥ 2048`（`:3901`，1014 MB 曾卡死）；stride 涨到 18.80 MB 后只剩 108 槽 | overlap 不足可能掉 prefill 速度 | Phase 1 就量 108 槽；必要时 ring 放到 3.9 GB |
| **R7** | 144 GB 的预填充 pack 若冷读 NVMe（6 GB/s）要 24 s | 长 prompt +24 s，反而变慢 | 依赖 192 GB 内存做页缓存；`--ssd-cache` 与该 pack 的配合需设计 |
| **R8** | 相位切换时延（S2 需重填驻留区 ≈1 s/次） | 短 prompt 亏损 | S3 不需要；若要 S2，按 prompt 长度设阈值（如 ≥ 4k 才切） |

---

## 九、行号索引（本文件所有引用）

**`sycl/src/dsv41/ds4_model.dp.cpp`**：`Pack:389`、`Tensor:382`、`Pack::load:425-523`、`Geo:555`、`Dev:624`、`Chunk:707`、`Layer:805-885`、`Swap:1265-1270`、`unit:1743`、`mat_gg:1856`、`mat_bf16:1920`、`mat:1996-2000`、`gg_type_of:2269`、`expert_layout_mx:2274-2337`、`mx_mat:2338`、`mx_part_bytes:2348`、`read_mx:2378`、`read_mx_file:2386`、`expert_layout:2414`、`load:2502`、`eg engraw fmt:3301-3314`、`ring stride:3858-3864`、`ring alloc:3893-3904`、`gemm:3950-3999`、`gemm_f16w:4006`、`flow_region:4325`、`mx_read_off:4333`、`flow_push:4347`、`flow_spec:4424`、`flow_rest:4467`、`fl_rows:4491`、`fl_yoco 判据:4481-4493`、`run_set:4879-5018`、`PREFILL_LAYERS 探针:4885-4889`、`格式分派:4977-4985`、`place_experts:5356-5505`、`vram alloc:5494-5505`、`init_store:6280-6303`、`preload_store:6474-6545`、`store_sync:6548`、`cache_loop:8240`、`swap_in_mx:8389`、`cache_settle:8509`、`cache_tick:8555`、`cache_tick_swaps:8572`、`mv_any:9062-9070`、`mv:9071`、`yoco_layer:11285-11297`、`forward_chunks:11301`、`keep:11330-11332`、`run.yoco:11344-11348`、`forward:11901`、`token 专家分派:12786-12796`、`window 专家分派:12849`、`Model::step:13217`、`Model::feed:13218-13233`、`Model::feed_mark:13234-13250`

**`sycl/src/dsv41/ds4_gguf.dp.hpp`**：`kFmt*:29-32`、`expert_type:44`、`mx_experts:45`、`type_*:49-60`、`mv:68`、`rows:73`、`moe_gate_up:77`、`moe_down:79`、`win_experts:81`、`set_gate_up:82`、`set_down:84`

**`sycl/src/dsv41/ds4_gguf.dp.cpp`**：`mx_set_tc_kernel:1360`、`tc_ok:1710`、`type_row_bytes:1803`、`type_block:1808`、`type_name:1809`、`type_on_device:1810`、`type_on_chunk:1811`、`gg::mv:1907`、`gg::rows:2062`、`moe_gate_up:2128`、`moe_down:2164`、`win_experts:2214`、`set_gate_up:2325`、`set_down:2391`

**`sycl/src/dsv41/ds4_iq.dp.cpp`**：`dequant_rows:2400`、`supported:2438-2454`、`block_values:2458`、`block_bytes:2463`、`gate_up:2510`、`down:2525`、`light_gate_up:2540`、`set_gate_up:2568`、`set_down:2588`、`mv:2608-2648`、`DS4_IQ_SWITCH:1840-1855`

**`sycl/src/dsv41/ds4_packinfo.cpp`**：`device_type:30-40`、`dense_type:43-45`

**`sycl/src/dsv41/ds4_store.hpp`**：`LayerGeo:40-47`、`slot_bytes:80`、`kSlack:122`、`find:125`、`ensure:146`、`preload:214`、`put_begin:268`

**`sycl/src/dsv41/ds4_model.hpp`**：`Options:25`、`Options::pack:26`

**`src/dsv41/ds4_main.cpp`**：prompt 循环 `:1092-1116`、`PP:1112`、`set_cancel(nullptr):1118`、`t1:1144`、解码 `round():1206`、`slot save/load:905/949`

**`tools/dsv41/`**：`drafter_pack.py:10-12,68,84,193-196`、`dspark_ref.py:44,77-85`、`pack_gguf.py:141-143,663-677`、`expert_requant.cpp:1-8`、`dense_requant.py:4-13`、`patch_pack_experts.py:43-88`

**实测数据**：`serve-engine.log:4,7,17,25-31,32`；`docs/STATUS_2026-11-02.md`
