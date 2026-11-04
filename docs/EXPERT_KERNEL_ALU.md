# 专家内核的墙：不是 ALU，是张量核从来没被打开

> 2026-11-02 夜间（不碰显卡，只编译）→ 11-03 早上一条命令验证。
> 带 `⏳` 的是明早要跑的命令；带 **✅已证** 的是今晚从源码/构建产物里直接读出来的事实，不需要跑卡。

## 零、一句话结论（今晚最大的发现）

**4.2 TFLOPS 不是"2-bit 解量化的 ALU 墙"，是正在跑的是 F32 标量专家内核，张量核内核一行都没执行。**
两道锁同时在关着它：

| # | 锁 | 位置 | 证据 |
|---|---|---|---|
| 1 | **编译期**：`STRATA_SYCL_DSV41_TC=OFF` → 注入 `-DSTRATA_DSV41_NO_TC=1`，`set_tc_kernel` 整段被 `#if !defined(...NO_TC)` 编掉 | `sycl/CMakeLists.txt:500`、`sycl/src/dsv41/ds4_iq.dp.cpp:1166` | **✅已证** `build-sycl-jit/CMakeCache.txt: STRATA_SYCL_DSV41_TC:BOOL=OFF`；`build.ninja` 里 `STRATA_DSV41_NO_TC=1` 出现 29 次 |
| 2 | **运行期**：`tc_ok()` 用 `device.get_major_version() >= 7`（CUDA 的 sm_70）当门槛，而 dpct 的 `get_major_version()` 解的是 **OpenCL 版本串**（`"OpenCL 3.0"` → 3），Intel 显卡上恒为 3 < 7 | `ds4_iq.dp.cpp:1817`、`ds4_gguf.dp.cpp:1710`；`sycl/include/dpct/device.hpp:39-58` | **✅已证**：`detail::get_version()` 对 `info::device::version` 取第一个数字 |
| 3 | 就算 1、2 都放开了，`set_tc_kernel` 是 DPCT 从 CUDA 的 `nvcuda::wmma` 翻译过来的 `joint_matrix`，在 icpx 上**从未运行过** | `ds4_iq.dp.cpp:1176-1400` | 详见第三节 |

**编译层的铁证**（今晚 21:44 两份二进制对比）：

```
nm -C build-sycl-tc2/strata-dsv41 | grep -c set_tc_kernel   → 296      # TC=ON：张量核内核在
nm -C build-sycl-jit/strata-dsv41 | grep -c set_tc_kernel   →   0      # 生产：一行都没有
strings -a build-sycl-tc2/strata-dsv41 | grep -o 'STRATA_DS4_TC[A-Z_]*'
    → STRATA_DS4_TC   STRATA_DS4_TC_MINROWS                            # 新加的运行期开关也进了二进制
```

两种配置都编译通过（TC=OFF 那份用 `-fsyntax-only` 以生产编译选项验过，0 error）。

**推论**：4.2 TFLOPS 应当被读作"F32 FMA 内核在 32 个 K 步 × 8 warp 的循环里跑出来的数"，它的上限是这张卡的 **fp32 峰值**（粗估 8–15 TFLOPS 量级），不是 100。所以"提升 2 倍预填充"的正确姿势不是省解量化的 ALU，而是**把 XMX 打开**——目标 9–30 TFLOPS。

---

## 一、今晚动的东西

| 文件 | 改动 | 状态 |
|---|---|---|
| `sycl/src/dsv41/ds4_iq.dp.cpp:1824-1837` | `tc_ok()` 加显式 opt-in：`STRATA_DS4_TC=1` 打开张量核，`STRATA_DS4_NO_TC=1` 仍最优先关掉（默认行为不变，仍是 F32） | 编译中 |
| `sycl/src/dsv41/ds4_gguf.dp.cpp:1729-1736` | 同上（MXFP4 专家的 `mx_set_tc_kernel` 走这一份） | 编译中 |
| `build-sycl-tc2/`（新） | 与 `build-sycl-jit` **同源同选项，唯一差别 `STRATA_SYCL_DSV41_TC=ON`**：这就是 A/B 用的二进制 | 编译中 |
| `sycl/src/kernels/iq_dequant_bench.cpp` | 四段式解量化微基准（ref / bat / deq8 / tile±vec16），带"隐含 TFLOPS"换算 | ✅ 编好，CPU 已跑 |
| `sycl/src/kernels/xmx_gemm_bench.cpp` | 改成 **rows/expert 扫描表**：融合 vs 两段式（dequant+oneMKL）在 M=1…1800 上的交叉点 | ✅ 编好 |
| `sycl/CMakeLists.txt:613-616` | `xmx_gemm_bench` 补 `-qmkl=sequential`（`dpct::blas::gemm` 需要 MKL 头） | ✅ |
| `ds4_iq.dp.cpp:1819-1834`、`ds4_gguf.dp.cpp:1712-1727` | 运行期三个开关：`STRATA_DS4_TC=1` 打开张量核、`STRATA_DS4_TC_MINROWS=N` 只在 ≥N 行时用（让解码留 F32）、`STRATA_DS4_TC_ONLY=gu\|down` 只开一半（排错定位用）。`STRATA_DS4_NO_TC=1` 仍然最优先关掉 | 编译中 |
| `sycl/CMakeLists.txt:591-604` | 新增 **`ds4_iq_test_tc`**：上游的专家内核对拍测试（`iq_expert_test.cpp`，拿 ggml 解码 + double 参考对拍 token / window / prompt-set 三条路径），但**不**定义 `STRATA_DSV41_NO_TC`，所以 `set_tc_kernel` 真的在里面。原来的 `ds4_iq_test` 把 NO_TC 写死了，测不到张量核 | ✅ 已编（`build-sycl-tc2/ds4_iq_test_tc`，`nm` 里 264 个 `set_tc_kernel` 符号） |

**踩到的坑（写下来省得再踩）**：`build-sycl-jit` 的 `CMakeFiles/rules.ninja` 被一次失败的 reconfigure 删了，而 reconfigure 失败的原因是 (a) `ninja` 必须在 `source /opt/intel/oneapi/setvars.sh` 之后跑，否则 found 不到 `icx/icpx`；(b) ggml 的 `FetchContent` 会去 GitHub 拉 638 MB（夜里没有网就卡 10 分钟然后失败）。修法：把 `build-xmx-bench/_deps/strata_llamacpp-src`（同一 commit `3cf0325`）**硬链接**到 `build-sycl-jit/_deps/`（不占额外空间），再用 `-DSTRATA_GGML_DIR=...` 指过去，1 秒配完。

---

## 一点五、先对齐既有证据（2026-10-30 已经做过一个直接相关的实验）

`DSV41_SYCL_PORT.md:475-490` 记录了当时在同一段提示上换**内积单元**的三次测量：

| expert 内积 | 2K 提示 |
|---|---:|
| dpct `mma<16,8,16, half, float>`（EXL3_TC 一直开着） | 12.72 s |
| F32 点积（`STRATA_SYCL_DSV41_EXPERT_NO_TC=ON`） | 12.80 s |
| F32 + half2 折叠（`..._EXPERT_HALF2=ON`） | 12.67 s |

**三者差 1%**，当时的结论是："3.9–6.6 TFLOPS 不是乘加，时间在 trellis 瓦片加载与每 128 输入的 staging/barrier 上"。

这条证据**直接约束我今晚的判断**，有两条后果：

1. **"打开张量核"的期望值必须调低**。那条路径换的正是内积单元，而既有实验说换它没用（±1%）。
   张量核仍值得花 15 分钟验（二进制都编好了），但它**不再是我认为的一线希望**——除非
   `joint_matrix` 顺带把 ALU 槽位让给解量化，这个只有在卡上才知道。
2. **两段式的期望值要调高**。同一份文档 `B70_PR_SEARCH_BRIEF.md:27` 记着 oneMKL fp16 GEMM 在引擎自己的
   调用形状上测到 **94.9 TFLOPS**（我原先记的 9 TFLOPS 是 M=96 的小形状）。按 95 TFLOPS 重算：

| 方案（rows=900，每专家） | 解量化 | GEMM | 合计 | 相对现状 |
|---|---:|---:|---:|---:|
| gate+up 融合 BM=64（现状） | 2.81 ms | — | 2.81 ms | 1.0× |
| gate+up 两段式（95 TFLOPS） | 0.19 ms | 0.12 ms | **0.31 ms** | **9.0×** |
| gate+up 两段式（保守按 45 TFLOPS） | 0.19 ms | 0.26 ms | 0.45 ms | 6.3× |
| down 融合 BM=64（现状） | 0.98 ms | — | 0.98 ms | 1.0× |
| down 两段式 | 0.07 ms | 0.06 ms | 0.13 ms | 7.7× |

显存流量这时不能忽略，但算了也只有 0.08 ms（写 13.1 MB f16 + 读回 + 读激活 + 写 fp32 输出 ≈ 40 MB / 450 GB/s），
**不构成限制**。所以：**两段式是 5–9× 量级，张量核最多是几个百分点到 1.2×。优先级反过来。**

## 二、解量化本身的算术（第二位的假设，明早验）

`set_tc_kernel` 的内层（`ds4_iq.dp.cpp:1216-1230`）每个 32-K 步要现解 64 行 × 32 个权重，才做一次 64×64×32 的 MMA。
也就是说：**一个解出来的权重只被 BM=64 个激活行复用**。

```
隐含 TFLOPS = Gval/s × 2 × BM / 1000 = Gval/s × 0.128
```

反过来：**4.2 TFLOPS ⇔ 33 Gval/s**。所以明早只要看 `deq8` 那一行的 Gval/s：

- 落在 30–40 Gval/s → 解量化确实是这条路的墙，而且**两段式（每专家只解一次，900 行复用）能省 900/64 ≈ 14 倍的解量化**；
- 远高于（≥150 Gval/s）→ 墙在 barrier / MMA 形状 / 瓦片构建，不在解量化；`tile` 那一行（deq8 + f16 转换 + local 存 + barrier，没有 MMA）与 `deq8` 的差距就是"瓦片构建"的价钱。

`deq8` 每 8 个值的指令账（逐行照抄 `:449`）：1 次 8 B 查表 + `popcount` 奇偶 + 8 次字节抽取 + 8 次符号选择 + 8 次乘 = **约 5 条 ALU/值**（比原来估的 0.6 高 8 倍，因为原估算漏了 fp32 路径的逐值抽取与选择）。

### SIMD 原型（目标 ③）：已写、已逐位验过

`deq8_iq2_vec()`：8 个值一次算完——`uchar8 → float8`（一次向量转换）→ 乘 `dl`（一次）→ 乘 ±1 掩码（一次）。
逐位等价的原因是每 lane 仍然只有一次 IEEE 乘法 `dl*gv`，再乘 ±1（乘 ±1 精确，不改尾数）。

```
SIMD vs 标量（CPU，1047040 值）: 0 处不同，max |err| = 0 (逐位一致)
  → max err = 0，SIMD 版可以替换现有实现
```

落位也验了（同一份 CPU 侧检查）：**标量逐 half 存 vs 16 B 向量存，逐位一致**——所以早上"向量存快多少"只可能是性能问题。

**每权重 ALU 指令数（目标 ② 要的那个指标）**：设备 ISA 这一层今晚数不出来——这个 icpx 发行版禁用了 IR 输出
（`-emit-llvm` 直接报 "IR output is not supported"），机器上只有 LLVM 17 的 `llvm-dis`，读不了 Intel 2026.2 的
新格式 bitcode，也没有 `spirv-dis`。退一步做了 **x86 代理计数**（把两个版本 `noinline` 编出来数指令）：

| 版本 | x86 指令数 / 8 值 | 每值 |
|---|---|---|
| 标量 `deq8_scalar` | 139 | 17.4 |
| 向量 `deq8_vec` | 138 | 17.3 |

**几乎一模一样**——这正好解释了 CPU 侧两者同速。它也给出一个对 GPU 的警示：Xe 上 `vec<float,8>` 并没有真正的
8 宽 fp32 运算单元，一条向量乘法落地就是 8 条标量乘，所以 float8 这种写法在 GPU 上也很可能打平。
**真要省一半，得走 packed fp16（half2，两个值一个 32 位通道）**，但那会引入"f32 乘再舍入 vs 直接 f16 乘"的二次舍入，
**不再逐位一致**，是个要单独评估的取舍。所以早上 `deq8` 与 `deq8v` 那两行如果打平，结论就是"解量化写法不是墙"，
这与第一节的张量核发现是一致的。

**踩到的坑（值得记）**：第一次跑出来 16300 处"不同但 |err|=0"——全是 **NaN 的符号位**（`0xffe08000` vs `0x7fe08000`）。
原因是合成权重把块头那个 half 尺度 `d` 也填成随机字节，随机字节可以是非数；而 NaN 的符号位取决于乘法操作数顺序。
真实权重的 `d` 是有限值，所以基准里现在统一给 `d = 1.0`（`0x3C00`）——`xmx_gemm_bench` 本来就一直是这么干的。

---

### 权重到底被重复解量化几遍（决定了 "BM 调大" 和 "两段式" 的上限）

`ds4_iq.dp.cpp:2607`（`set_gate_up`）与 `:2627`（`set_down`）都这样建 grid：

```cpp
const dpct::dim3 grid((a.max_rows + BM - 1) / BM,   // 行块数：每个行块独立解量化同一份权重
                      a.n_ff / BN, a.n_exp);        // 列块数、专家数
```

工作单元是 (行块, 列块, 专家)。**同一个专家的同一份权重，被 `ceil(rows/BM)` 个行块各解一遍。**

| rows/expert | BM=64 | BM=128 |
|---|---|---|
| 96 | 2× | 1× |
| 256 | 4× | 2× |
| 900（122k 预填充实测） | **15×** | 8× |
| 1800 | 29× | 15× |

两段式则是 **1×**。用这个反推解量化吞吐（假设设备级吞吐由解量化决定，`T = T_deq`）：
gate+up 实测 2.81 ms/专家、15 遍 → **D ≈ 35 Gval/s**；down 0.98 ms、15 遍 → **D ≈ 50 Gval/s**。
（两个数不一致是正常的，down 的 MMA 更便宜、掩盖得更彻底。**明早 `deq8` 那一行就是在直接验证这两个数**。）

有上百个 workgroup 同时在跑，设备级吞吐由较慢的那个单元决定，所以预测用 `T = max(T_deq, T_MMA)`：

| 方案（rows=900） | t | 等效 TFLOPS | 相对现状 |
|---|---|---|---|
| gate+up 融合 BM=64（现状） | 2.81 ms | 4.20 | 1.00× |
| gate+up 融合 BM=128 | 1.50 ms | 7.88 | **1.88×** |
| gate+up 两段式（解一次 + oneMKL 9T） | 1.31 ms | 9.00 | **2.14×** |
| down 融合 BM=64（现状） | 0.98 ms | 6.00 | 1.00× |
| down 融合 BM=128 | 0.52 ms | 11.25 | **1.88×** |
| down 两段式 | 0.66 ms | 9.00 | 1.50× |

**两个杠杆都要动的东西很便宜**：`BM` 只是一个 `constexpr`（`ds4_iq.dp.cpp:1000`），且 BM=128 的收益已由基准里
`tile BM=128` 那几行直接给出；两段式的收益上限由 `xmx_gemm_bench` 的 M 扫描给出。两者都指向 ~1.9×，
**而打开张量核是另一个数量级的事**（4.2 → 20+ TFLOPS 才有可能），所以明早的优先级仍是第 3 步。

## 三、`set_tc_kernel` 能不能跑起来（明早唯一的真问题）

今晚把 1176–1400 行整段读完了。**结构是对的**：8 个 warp 覆盖 64×64（wm_=warp>>1 给 4 个 16 行块，
wn=(warp&1)*2 + j 给 4 个 16 列块），`kk` 两次 16 步正好吃掉 32 个 K，B 用 `col_major` + `ldm=kLdT` 与
"`Tl.wg[n][k]` 按 n 逐步、k 连续"的内存布局自洽，收尾用 `joint_matrix_store` 到 `O.g/O.u` 再
`if (r >= nr) continue` 跳过越界行——和 F32 内核的语义一致。

按风险排的可疑点（每条都带修法，出问题时直接照着改）：

| # | 风险 | 位置 | 判断依据 | 修法 |
|---|---|---|---|---|
| S1 | **A/B 载入用的是 generic 地址空间** | `:1192`、`:1311`、`:1327`（`address_space_cast<generic_space, decorated::no, ...>`） | Xe 的 `dpas` 要的是带 SLM surface 描述的指针；DPCT 把 CUDA 的 `__shared__` 直译成 generic 指针是经典坑。能编译过≠运行时对 | 改成 `local_space`（必要时 `decorated::yes`），或把指针直接传 `local_accessor` 拿到的地址 |
| S2 | **fp16 溢出**：激活值 `x` 由 f32 转 half 才进 MMA | `:1246-1250`（`Tl.x[tr][tq*8+j] = half(...)`） | F32 路径全程 f32，TC 路径多一次 f16 舍入；激活里有大值时 >65504 变 inf。这是**两条路径真实的数值差异**，不是 bug | 对拍容差暂时放宽到 1e-2 再看；若只有个别 expert 超，就是这个 |
| S3 | `Tiles` 与 `Out` 共用同一块 SLM（union），靠 barrier 排序 | `:1183-1189`、`:1384` | 语义上够（store 在 barrier 后），但 `joint_matrix_store` 是内联汇编边界，编译器可能重排 | 把那个 `barrier()` 显式写成 `barrier(fence_space::local_space)` |
| S4 | `kLdT=40` 的 32 B 对齐 | `:1288`、`:1316`、`:1342` | 算过：偏移 = 1280·(行块) + 2·kk 字节，kk∈{0,16} → 全是 32 B 整数倍，**没问题**，列出来是为了排除 | — |
| S5 | `nr < BM` 的越界行走 MMA 但输出被跳过 | `:1387-1396` | 越界行的 `Tl.x` 被写成 0（`sr<0` 分支），不会产生 inf/nan；输出由 `if (r >= nr) continue` 丢弃 | — |

⇒ **明早对拍如果 FAIL，按 S1 → S2 → S3 的顺序查**，S1 是"直译坑"，S2 是"数值差异"，
两者现象不同：S1 通常是**整片乱**（甚至崩），S2 是**只差一点**（worst 略高于 tol）。

S1 的具体改法（三处同样的形状，`&Tl.x[...]` / `&Tl.wg[...]` / `&Tl.wu[...]` 各一次）：

```cpp
// 现在（DPCT 直译）：
sycl::address_space_cast<sycl::access::address_space::generic_space,
                         sycl::access::decorated::no, sycl::half>(&Tl.x[wm_ * 16][kk])
// 改成（告诉编译器这是 SLM 指针）：
sycl::address_space_cast<sycl::access::address_space::local_space,
                         sycl::access::decorated::yes, sycl::half>(&Tl.x[wm_ * 16][kk])
```

改完只需重编 `build-sycl-tc2`（`ninja -C build-sycl-tc2 ds4_iq_test_tc`，约 3 分钟），
用 `STRATA_DS4_TC=1 ds4_iq_test_tc` 重新对拍即可，不用碰生产。

### 同一个基准里还扫了 BM（权重复用倍数）

融合路线的隐含 TFLOPS 直接正比于 BM：`tile` 那几行按 `BM=32/64/128` × `标量存/16B向量存` × `标量解量化/SIMD解量化`
全部实例化好了（`nm -C build-sycl-jit/iq_dequant_bench | grep -c KTile` = 90）。这给出一个**不用改两段式**的廉价方案：
如果 `BM=128` 那一行的隐含 TFLOPS 明显高于 `BM=64`，那早上第一件事就是把引擎的 `BM` 调大（`ds4_iq.dp.cpp:1000`
的 `constexpr int BM = 64`），而不是重构。

## 四、明早的命令（按顺序，一条一条）

```bash
cd /data/strata-new && set +u && source /opt/intel/oneapi/setvars.sh --force && set -u

# 【最省事】下面三步已经包成一条命令（会用 gpu_lock 独占显卡、输出存到 /tmp/morning-*）：
tools/overnight/morning.sh

# 0) 独占显卡（工具的约定：拿锁之前确认没有引擎在跑）
systemctl --user is-active dsv41.service            # 必须是 inactive
tools/overnight/gpu_lock.sh take

# 1) 解量化算术（Gval/s）+ 瓦片构建 + 隐含 TFLOPS
ninja -C build-sycl-jit iq_dequant_bench
./build-sycl-jit/iq_dequant_bench 131072 20 4 80 128

# 2) 融合 vs 两段式的 rows/expert 交叉点表（DSV4 真实形状 2560/1280）
ninja -C build-sycl-jit xmx_gemm_bench
./build-sycl-jit/xmx_gemm_bench 16 both 30 sweep real

# 3) 张量核专家内核本身：对拍 + 瓦片吞吐（不启引擎，几十秒）
B=build-sycl-tc2/ds4_iq_test_tc
STRATA_DS4_NO_TC=1 $B                        # F32 基线：worst / tol 两列应当全 ok
STRATA_DS4_TC=1    $B                        # 张量核：worst 不许涨过上面那组，出现 FAIL 就是直译错了
STRATA_DS4_NO_TC=1 $B --bench-tiles 16 6 96,256,900    # 预填充瓦片吞吐（IQ2_XXS，6 专家）
STRATA_DS4_TC=1    $B --bench-tiles 16 6 96,256,900

# 4) 端到端 A/B（整机、走服务）—— 见下方脚本（注意：这几步都要独占显卡）

```

`⏳` 第 3 步已经脚本化（`tools/overnight/tc_ab.sh`）：同一个二进制 `build-sycl-tc2/strata-dsv41`，只差环境变量，
跑完自动还原配置、自动出对照表。

```bash
# 3) 张量核 A/B（自动跑三组：F32 基线 / 张量核 / 张量核+解码留 F32，出表）
tools/overnight/tc_ab.sh bench/prompts/p45k.txt 48      # ~50k token 的合成长文本
```

它做的事：临时改 `strata-server.json` 的 `exe`（指向 tc2 二进制）与 `env`（`STRATA_DS4_NO_TC=1` / `STRATA_DS4_TC=1`
/ `+STRATA_DS4_TC_MINROWS=16`），每组 `systemctl start` → 等健康 → 打同一个长提示 → 停服务，**退出时无条件还原配置**。
表里看 `wall s` 与 `prefill tok/s`，并且**必须**把三组的输出开头 90 字符对一遍——数字好看但输出变了就是错的。

对照组的意义：`base` 组跑的是同一个 TC=ON 二进制 + `STRATA_DS4_NO_TC=1`，所以它应当**精确复现**今天 45k→347 tok/s、
122k→270.7 tok/s 的量级；如果对不上，说明不是 TC 的功劳/锅，而是环境变了（先查这个）。

---

## 五、判断标准（提前写死）

| 明早的结果 | 结论 / 下一步 |
|---|---|
| **第 2 步 M=900 的 `2ph/fus` ≥2** | **这是最大的一笔**：预填充改两段式（`prefill.cpp:4364` 已有现成调用），预期 5–9×。接进引擎的具体蓝图：[TWOPHASE_EXPERTS_DESIGN.md](TWOPHASE_EXPERTS_DESIGN.md) |
| 第 1 步 `tile BM=128` ≥1.5× | 改一个常量 `BM`（`ds4_iq.dp.cpp:1000`），预期 1.88×；可与两段式叠加 |
| TC=1 输出正常且预填充 ≥1.5× | **立刻把 `STRATA_SYCL_DSV41_TC=ON` 变成默认**，解量化那条通路直接作废 |
| TC=1 输出正常但 <1.2× | 张量核打开了但瓦片太小/barrier 太多 → 调 `BM/BN/kLdT`，或改用 `prefill` 的 `xmx_gemm_iq`（已测 1.87 TFLOPS，比 F32 更差，作为反例） |
| TC=1 输出胡话 | 是 `joint_matrix` 的直译错误。**先定位是 gate_up 还是 down**：`STRATA_DS4_TC_ONLY=gu` / `=down` 只开一半再跑对拍（`ds4_iq_test_tc` 支持，无需重编），坏的那半关掉、好的那半照样能用。再引擎自带 `DS4_DUMP_*` 中间结果导出（`grep -o 'getenv("DS4_DUMP[^)]*)' sycl/src/dsv41/ds4_model.dp.cpp` 看有哪些），同一段短提示分别导出 F32 与 TC 的隐藏状态，逐层比；再用 `xmx_gemm_bench` 的小 M 对拍单个 GEMM 的 `[数值] max err` |
| TC=1 崩 / 挂死 | 记下 kernel 日志，恢复 `Restart=no` drop-in，先把 TC 冻结在 opt-in；转去两段式（第 2 步的表） |
| `deq8` <100 Gval/s 且 `tile`≫`deq8` | 解量化不是墙、瓦片构建是墙 → 改 16 B 向量存 + 减少 barrier（bench 里 `tile-vec16` 已给出上限） |
| 两段式在 M≈900 时比融合快 ≥1.5× | 预填充改两段式：按专家解一次 f16 到 scratch，再 oneMKL f16 GEMM（`prefill.cpp:4364` 已有现成调用） |

---

### 如果 TC 赢了，怎么落地（三选一，按侵入性排序）

0. **一条命令（推荐）**：`tools/overnight/tc_promote.sh promote [minrows]` —— 只改配置的 `exe` 与 `env`，
   自动备份、`revert` 一键撤回，**任何二进制都不会被覆盖**。`status` 随时看现在用的是哪份。
2. **推荐**：把 `build-sycl-tc2/strata-dsv41` 覆盖到 `build-sycl-jit/strata-dsv41`，`env` 里加 `STRATA_DS4_TC=1`——
   路径不变，`START.sh` / systemd 单元 / 所有既有脚本都不用动；`STRATA_DS4_NO_TC=1` 就是随时回退的开关。
3. **正本清源**：把 `build-sycl-jit` 也按 `-DSTRATA_SYCL_DSV41_TC=ON` 重新配置（只差一个缓存变量，见第一节），
   之后两个目录就不再需要了。

无论哪种，**先跑一次 122k 的完整预填充**确认没有长尾退化（TC 的瓦片在 nr < BM 的边界上行为不同）。

## 五点半、今晚自查抓到的两个坑（都关于"开关类代码"，不是算法）

1. **响的失败**：两段式第一版没挡类型，而 `iq_dequant_f16` 对非 IQ 类型会 `std::exit(1)`
   （`src/kernels/cuda/iq_kernels.dp.cpp:4849`）——用 `DS4_EXP_2PHASE=1` 跑一个含 MXFP4 的包会让引擎直接退出。
   已加 `supported()` 守卫（`ds4_iq.dp.cpp:2693/2715`）。
2. **不响的失败（更阴）**：对拍集合只有 37 行，而两段式行数门槛默认 96 → 它干净回退、数字与融合路径
   **逐位相同**，看起来"通过"其实一次没跑。修法是验证时带 `DS4_EXP_2PHASE_MINROWS=1`，
   并且判据写成"**与不加开关的那次比**"而不是"看有没有 FAIL"。

结论：往热路径加 env 开关，要同时准备"条件不满足就干净回退"和"回退了要让人看得见"两件事。
`morning.sh` 第 4 步现在两者都做了（压门槛 + 打印"若逐位相同先怀疑没接管"）。

## 五点九、11-03 早上的实测（第一批真数据，含两次崩溃）

### 已经测到的（06:44，Arc 正常工作期间，这批可信）

| 变体 | Gval/s | 隐含 TFLOPS（×0.128@BM64） |
|---|---:|---:|
| `ref`（一块一 work-item） | 29.6 | 3.79 |
| `bat4`（块批处理 + 表进 SLM） | 30.1 | 3.86 |
| **`deq8`**（引擎真实原语） | **73.9** | **9.46** |
| `deq8v`（SIMD float8） | 73.5 | 9.41（**0.99×，无收益**） |
| `tile BM64` 标量存 | 90.6 | 11.60 |
| `tile BM64` 16B 向量存 | 106.2 | 13.59（**1.17×**） |
| `tile BM128` 向量+SIMD | **182.5** | **46.72** |
| `tile BM32` 标量 | 48.0 | 3.07 |

四条结论：

1. **SIMD 写法没有收益**（0.99×）——与 x86 代理计数（139 vs 138 条）和 10-30 的"三种内积差 1%"完全一致。解量化的**写法**不是变量。
2. **把表搬进局部内存也没有收益**（ref 29.6 → bat4 30.1，1.02×）。
3. **BM 是唯一立竿见影的杠杆**：BM64 13.6 → BM128 46.7（Gval/s 本身还涨了 1.72×，barrier 摊薄）。这与文档第二节的 `ceil(rows/BM)` 重复解量化模型一致。
4. **真实内核（4.2 TFLOPS）距它自己的瓦片构建天花板差 3–5 倍**（BM64 13.6 / BM128 46.7）。所以融合内核的墙**不在解量化，也不在落位**，而在 **MMA 与它之间的同步**。

### ★ 两段式在真内核上验证通过：900 行时 12.4×（2026-11-03 09:1x，重启后实测）

`ds4_iq_test_tc --bench-tiles 16 6 96,900`（`STRATA_DS4_NO_TC=1` 固定走 F32 融合路径）：

| rows/expert | 融合 gate/up | 两段式 gate/up | 加速 | 融合 down | 两段式 down | 加速 |
|---|---:|---:|---:|---:|---:|---:|
| 96 | 10023.7 µs | 2729.0 µs | **3.67×** | 2567.9 µs | 1165.9 µs | 2.20× |
| 900 | 58257.4 µs | 3992.3 µs | **14.59×** | 16968.3 µs | 2053.5 µs | **8.26×** |

900 行 6 专家合计：**75.2 ms → 6.05 ms = 12.44×**。比文档第二节预测的 5–9× 还好，
而且缩放规律完全对上模型：融合随行数线性涨（10 → 58 ms），两段式几乎不涨（2.7 → 4.0 ms）——
因为融合的重复解量化次数是 `ceil(rows/64)`，两段式恒为 1。

**数值同时通过**（`--type 16`，`DS4_EXP_2PHASE_MINROWS=1` 强制接管，所以不是静默回退）：
`set gate_up` worst 3.34e-04（tol 1e-02）、`set down` 1.42e-04（tol 3e-03），ALL OK。
**而且比融合路径更准**（对照 7.02e-03 / 4.19e-04）——f16 权重 + f32 累加比逐值 f32 更稳。

### ★★ 端到端验证：62k 提示预填充 375.8 → 944.1 tok/s（2.51×）

服务端 A/B（`tools/overnight/tp_ab.sh`，两次都 0 reused、真跑预填充、输出逐字相同）：

| | 预填充耗时 | **预填充吞吐** | 解码（24 tok） |
|---|---:|---:|---:|
| 融合（base） | 165,732 ms | **375.8 tok/s** | 9.33 tok/s |
| 两段式（`DS4_EXP_2PHASE=1`） | 65,975 ms | **944.1 tok/s** | 9.65 tok/s |

wall 184.4 s → 83.8 s（2.20×）。

**解码不受影响**（`DS4_EXP_2PHASE_MINROWS` 默认 96，解码 `rows=1` 走融合路径）。更正：上面表里的
"9.33 / 9.65 tok/s"**不是解码速度**——那是 62k 预填充后只生成 24 个 token 的窗口算出来的，被首 token 与
引擎冷启动吃掉。同条件对照（48 token 提示、160 token 输出、开关前后各重启一次）：

| 配置 | 前 80 token | 之后 63 token |
|---|---|---|
| 两段式 ON | 18.57 | 18.26 |
| 融合 OFF | 17.74 | **20.81** |
| 两段式 ON（复测） | 17.45 | **20.86** |

开关前后一致（噪声内），**稳定段 20.8–20.9 tok/s**；开头 80 个 token 偏慢（17–18）是专家冷启动换入，
不是回退。


**更正（2026-11-03 10:3x）**：之前写的“输出逐字相同”站不住。同一配置连跑两次贪心请求，输出的 token 数就不同
（两段式 ON：117 / 114；融合 OFF：121 / 115），**服务端采样本身就不是逐位确定的**，与本次改动无关。
因此端到端的文本相等**不能**当作判据。真正的正确性证据是内核级对拍：harness 覆盖的 7 种量化类型在两段式接管下
全部通过（最差 3.61e-04，容差 1e-02，即 30 倍余量），且同一输入下两段式比融合路径**更接近**双精度参考。

**已落地**：`strata-server.json` 的 env 加了 `DS4_EXP_2PHASE=1`（备份 `strata-server.json.bak-pre-2phase`），
生产二进制没换（`build-sycl-tc2/strata-dsv41`，与 jit 同源，只差 TC 编译选项）。回滚：把 env 里那一项删掉即可。

**测量陷阱（差点被骗）**：服务端会把 KV 存进 slot 缓存，同一提示第二次跑会 `62279 reused`、根本不预填充，
于是显示"3816 tok/s"或"3701 tok/s"这种假数字。清 `slots/` 也不够（slot 是停止时异步保存的）。
最终做法：**每次给提示加一个唯一 nonce**，token 序列不同就不可能命中。两次 `0 reused` 才是有效测量。

### 归因：14.5× 里 6.95× 来自"少解 14 遍"，2.09× 来自"换到 XMX"（实测，非估算）

在**两段式内部**把解量化重复 N 遍（`DS4_EXP_2PHASE_DUP=N`），900 行 × 6 专家 gate/up：

| N | 时间 | 说明 |
|---:|---:|---|
| 1 | 4,011 µs | 真实两段式 |
| 2 | 5,715 µs | 每多一遍 **+1,704 µs** |
| 15 | 27,870 µs | (27870−4011)/14 = 1,705 µs/遍，线性 |
| 融合原始 | 58,334 µs | — |

过去：4,011 → 27,870 = **6.95×**（把 15 遍压成 1 遍）；剩下 27,870 → 58,334 = **2.09×**
（F32 SIMT MMA + 融合内核结构 → oneMKL/XMX）。合计 14.55×。

**一遍解量化 = 1,704 µs**，对应 6 专家 × 23.6 M 权重 = 142 M 值 → **83 Gval/s**、写 f16 0.28 GB → 166 GB/s。
这个 83 Gval/s 与基准里 `deq8` 的 73.9 Gval/s 吻合，交叉验证通过。

**形状修正**：真实是 `n_embd 5120 / n_ff 2304`（每专家 gate+up = 2×2304×5120 = 23.6 M 权重 = 6.1 MB IQ2）。
本文档早前按 `dequant_bench.cpp` 注释里的 1280/2560 是旧的小模型形状，绝对 TFLOPS 偏小 3.6 倍，**比率不受影响**。

**下一步的靶子**：两段式 900 行那 4,011 µs 里仍有 1,704 µs（42%）是那一遍解量化 + 写 f16 scratch。
解量化本身已到 83 Gval/s（接近纯解量化上限），所以真正的下一步是**别写 f16 scratch**（直接喂 XMX 的 int8/f16 路径、
或用驻留 VRAM 复用），而不是再优化解量化循环。

### RM=128：明确的负结果（数值对，但慢 4–5×）

同一 binary，`DS4_IQ_RM128=1`：96 行 10023.7 → 40228.3 µs、900 行 58257.4 → 290511.9 µs。
数值 worst 与基线**完全一致**（说明 128 行块的边界处理正确），但真内核里 `RT` 从 4 变 8，
每线程 32 个 fp32 累加器 + 解量化临时量 → **寄存器溢出**，收益全被吃掉还倒亏。

**教训：瓦片微基准（无 MMA、无寄存器压力）不能外推到真内核。** 我昨晚据此写的"BM=128 预期 1.88×"
是错的，这里更正。开关保持默认关。

### 张量核：这条路在当前 DPC++ 上不通（有定论了）

`ds4_iq_test_tc` 的 `STRATA_DS4_NO_TC=1` 基线**全类型 ALL OK**；`STRATA_DS4_TC=1` 在 **JIT 阶段**失败：

```
undefined reference to
__builtin_spriv_OpJointMatrixLoadINTEL_PackedA_RowMajor_SG16_16x16_i16_8_generic_v8i8_pi32_i32
error: backend compiler failed build.
```

这正是 `DSV41_SYCL_PORT.md` 早写下的"the packed-i16 joint_matrix that the backend rejects"。**不是我们配错开关，是后端不支持这个 builtin。**
那份文档给的替代路线是"hand-written joint_matrix 或 **oneMKL GEMM per expert**"——后者就是两段式。

### 两次崩溃（都记下来）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | 对拍工具段错误，`timeout` 挂 31 分钟，**Arc 掉出 PCIe 总线** | `s=0`（默认队列）时我的代码解引用 `*s` | 加 `exp2p_queue()` 归一化 |
| 2 | 修完 #1 仍段错误 | **在 host 代码里读了设备内存 `a.bounds[e]`**（引擎里是 `d_bounds`、对拍工具里是 `hipMalloc`）——融合内核在设备端读没事，host 侧读设备 USM 必崩 | 先整份拷回 host（`exp2p_host_bounds`，≤1.5 KB） |

**连带事故**：崩溃发生时 GPU 队列里有在飞的命令，之后 `lspci -vv -s 03:00.0` 读不到配置空间、
`sycl-ls` 里 Arc 消失、内核刷 `pcieport 0000:00:01.0: PCIe Bus Error: Correctable, RxErr` 每分钟 134 条。
这正是记忆里那条硬规则说的场景：**GPU 作业中途进程异常终止 = 把链路/驱动留在坏状态**，只有重启能清。

### BM=128 已经做成开关（等重启后验）

`set_kernel` 的 `RM` 是模板参数（局部内存也按 `RM` 开），所以"BM 调大"不需要改常量：
`ds4_iq.dp.cpp:2003` 起加了一个 `DS4_IQ_RM128=1` 的实例化，`grid.x` 相应地按 `ceil(rows/128)` 算。
`RT` 由 `RM/(kThreads/(BN/TN))` 自动变成 8，静态断言成立。**已编译通过，等卡回来验**
（瓦片微基准给的预期：BM64 隐含 13.6 TFLOPS → BM128 46.7 TFLOPS）。

**流程教训（比代码重要）**：实验开关的代码，第一次上卡**必须**先用最小样本（单类型、几十行、`ulimit -c 0`、
短 `timeout`）跑通再放大。我直接跑了全量对拍，崩了还挂住 31 分钟。`morning.sh` 第 4 步已改成
"先 `--type 16` 60 s，通过再全量 120 s"。

## 六、安全（照旧）

- **不 `systemctl start`，不同时跑两个引擎**；跑之前 `gpu_lock.sh take`，跑完 `release`。
- 一切实验用前台二进制，不动 `~/.config/systemd/user/dsv41.service`。
- OOM / `DEVICE_LOST` 立刻停手；卡死只能重启，而重启可能挂驱动。
- `build-sycl-tc2` 是全新目录，**不会**影响 `build-sycl-jit` 与现有服务。

## 七、已测数字（持续追加）

| 指标 | 数值 | 出处 |
|---|---|---|
| CPU 标量解量化（一块一 work-item） | 1.33 Gval/s | `iq_dequant_bench` CPU ref |
| CPU deq8（8 值/work-item，引擎原语） | 0.91 Gval/s | `iq_dequant_bench` CPU deq8 |
| CPU SIMD 版（float8） | 与标量同速（x86 无收益，正常） | 同上；**逐位一致 max err = 0** |
| 专家内核 gate+up（现状，F32 路径） | 4.2 TFLOPS | 122k 预填充阶段表 |
| 专家内核 down（现状，F32 路径） | 6.0 TFLOPS | 同上 |
| `xmx_gemm_iq`（预填充的 XMX 分组 GEMM，小 M） | 1.87 TFLOPS | 之前的 `xmx_gemm_bench` |
| 两段式（dequant + oneMKL f16） | 9.06 TFLOPS | 同上（M=96, 旧的小形状） |
| GPU 解量化 / 瓦片构建 / 交叉点表 | ⏳ 明早 | 本文第四节 |
| TC=1 的预填充 TFLOPS | ⏳ 明早 | 本文第四节 |

## 八、收尾状态（2026-11-02 22:0x，夜间工作结束）

| 东西 | 位置 | 状态 |
|---|---|---|
| 生产二进制 | `build-sycl-jit/strata-dsv41`（11-02 20:30） | **没动过**，服务仍用这一份；默认行为与今晚之前完全一致 |
| 张量核二进制 | `build-sycl-tc2/strata-dsv41`（11-02 21:51，TC=ON） | 编好、链好，**没跑过**（夜里不碰显卡） |
| 微基准 | `build-sycl-jit/iq_dequant_bench`、`xmx_gemm_bench` | 编好；CPU 部分已跑，GPU 部分等明早 |
| 一键验证 | `tools/overnight/morning.sh`（含 `tc_ab.sh`） | 语法检查过，没跑 |
| 备份 | `/tmp/ds4_{iq,gguf}.dp.cpp.bak-tcok`、`/tmp/{iq_dequant,xmx_gemm}_bench.bak-2026-11-02` | 在 |
| 服务 / 显卡 | `inactive` / 无引擎进程 / GPU 锁空闲 | 干净 |

**生产二进制的事（Round 8 处理过，记清楚）**：今晚把 `build-sycl-jit` 完整重编了一次（8 分钟，通过），
产出的二进制比 20:30 那份**小 4.5 MB**。查清了：差的全是**调试信息**（`.debug_loc` −3.3 MB、`.debug_info` −1.9 MB，
`.text` 只差 0.7 MB），原因是旧静态库里残留着更早一次带 `-g` 编出来的对象；两个 build 目录现在都只有
`ds4_model.dp.cpp` 一个文件带 `-g`。**功能没有区别**（TC=OFF，两边 `set_tc_kernel` 符号都是 0）。

但按夜间约束我不能运行二进制验证，所以**生产继续用已验证的那份**：

| 文件 | 用途 |
|---|---|
| `build-sycl-jit/strata-dsv41`（52.4 MB, 20:30） | **已验证、正在用的那份**（已复原） |
| `build-sycl-jit/strata-dsv41.known-good-2026-11-02` | 同上，备份 |
| `build-sycl-jit/strata-dsv41.fresh-r8-2259`（47.9 MB） | 今晚重编的，未运行过；要用先 `--pack-info` 冒烟 |

早上若要重编生产二进制（`ninja -C build-sycl-jit strata-dsv41` 会自动把 `strata-dsv41` 换成新版），
**先跑 `--pack-info` 冒烟**（纯 host 侧，不建 device、不碰 GPU），再 `START.sh start`：
`LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2026.2/lib:/opt/intel/oneapi/compiler/2026.2/lib:/opt/intel/oneapi/tbb/2022.2/lib/intel64 ./build-sycl-jit/strata-dsv41 --pack-info`。

**一个容易误会的旧产物**：`build-sycl-jit/` 里有 10-29 编的 `strata-dsv41-tc` 和 `strata-dsv41-baseline`。
查过：**两份的 `set_tc_kernel` 符号都是 0**——名字里的 `tc` 不是 tensor core（更可能是 trellis codec），
仓库里也没有任何脚本或文档引用它们。所以明早是**第一次**真正跑张量核专家内核，没有旧结论可借。

**没做的事（故意的）**：没有 commit。理由是今晚所有引擎改动都只是"编译通过"，一条都没在卡上验证过；
仓库当前已有一堆上轮未提交的改动（`git status` 十几个文件），再混进去只会让明早的回滚变难。
明早按第五节判据验完，再决定是一次 commit 还是 revert 这两处 `tc_ok` 改动。

**明早最省事的开头**：

```bash
cd /data/strata-new && tools/overnight/morning.sh
```

只看一页纸的话：[MORNING_2026-11-03.md](MORNING_2026-11-03.md)（做什么、看哪一行、结论对应哪个动作）。

## 七、11-03 午后：上一轮最后两个结论都是错的（都已证实）

上一轮（夜里到午后那一段）把"两段式在引擎里没生效"归因于"某个静默 `return false` 出口退出"，
并把主攻方向定成"IQ1_M 是 trellis 类型、走 vendor exl3 路径、两段式碰不到，占 70% 专家时间"。
这两条**都是错的**。证据如下。

### 7.1 "没接管"的真因：跑的那份二进制里没有这段代码

```
build-sycl-jit/strata-dsv41   52443840 B   11-02 20:30   strings 'DS4_EXP_2PHASE' → 0
git log -1 d9d0ffb2          两段式代码提交于 11-03 09:52
```

生产配置的 `exe` 一直指向这份 **11-02 20:30** 的二进制，而 `DS4_EXP_2PHASE=1` 是 11-03 凌晨才写的代码。
env 里那行开关是**彻底的 no-op**：所有守卫从未被求值，所以"守卫都满足"和"专家内核时间一模一样"
都是必然的，跟 `return false` 没有关系。

> 教训（比对任何"开关没用"的排查都值钱）：**先确认运行的那份二进制里有这段代码**。
> 一个 `strings | grep <env 名>` 三十秒的事，被"猜哪个 return false"吃掉了一整轮。
> 为了让这种沉默不再可能，现在代码**接管了会报、没接管也会报**：
> ```
> two-phase gate/up:ACTIVE: rows=271 n_exp=104 K=5120 N=2304 dup=1
> two-phase gate/up:NO-TAKEOVER: guard (type=...) / geometry ... / scratch ... / bounds ...
> two-phase gate/up:BELOW-MINROWS: max_rows=36 < 96 --- decode-sized set, stays fused
> ```
> 每个 tag 进程内只说一次。**沉默是最贵的失败模式。**

重新编译后（62837760 B，11-03 12:57，`strings` 里 3 处 DS4_EXP_2PHASE），引擎里立刻看到：
两段式 `gate/up` 与 `down` 双双 ACTIVE，`dup=1`，解码侧的 `max_rows=36` 正确走 BELOW-MINROWS。

### 7.2 IQ1_M 不是 trellis 类型，两段式完全够得到它

| 事实 | 证据 |
|---|---|
| trellis 类型只有 145/146/224/225/226/227（TQK6/7/5X/4X/5T/4T） | `ds4_iq.dp.hpp:30-35`，`tq_kbits()` |
| IQ1_M = **29**，`tq_kbits(29) = 0` → `is_tq(IQ1_M) = false` | 同上 |
| `supported(IQ1_M) = true`、`is_iq(29) = true` → `iq_dequant_f16` 直接支持 | `ds4_iq.dp.cpp:2495`、`iq_kernels.dp.cpp:3165` |
| 这个包里 IQ1_M 出现在 **40 层中的 22 层**（gate/up 与 down 都算） | `strata-dsv41 --pack-info` |

所以"IQ1_M 占 62K 专家时间 70%"不是因为它走了 vendor 路径，而是因为**它层数最多**：
22/40 层用 IQ1_M，18/40 层用 IQ2_XXS/IQ2_XS。它走的正是那个 `ceil(rows/64)` 重复解量化的融合内核，
**两段式能直接吃下它**（引擎日志已实测 ACTIVE）。

副产品：真正的 trellis 路径（`tq::set_gate_up`，`ds4_tq.dp.cpp:753`，grid = `(n_ff/(4*16), ceil(rows/16), n_exp)`，
重复倍数是 `ceil(rows/16)`——比 IQ2 的 `/64` 更糟 4 倍）在这个包上根本没被触发，
所以上一轮"主攻 trellis"计划**不需要做**。

### 7.3 当前这个包的形状（`--pack-info`）

```
40 layers x 384 experts, 6 used; dim 5120, ff 2304, vocab 129280
0-6,9,13-16,20,21   IQ1_M / IQ1_M
7,8                IQ1_M / IQ2_XXS
10,18,22,25,28...  IQ2_XXS / IQ2_XXS
11,12,15,17,19     IQ2_XXS|IQ2_XS / IQ1_M
23,26              IQ2_XXS / IQ2_XS
```
每专家 7.7–9.5 MB（1.75–2.06 bit），一个 token 摸 6 个专家 ≈ 46–57 MB。

## 八、11-03 13:1x：两段式端到端 **2.54×** 是真的；上一轮的"缓存预热"撤回**也是错的**

### 8.1 干净的 A/B（同一个二进制，只差一个 env）

新二进制（62837760 B / 11-03 12:57 / 含两段式代码），62,289 token 提示，
每种配置跑两次（第一次预热、第二次判据），每次请求带唯一 nonce 保证 `0 reused`：

| 配置 | 运行 | prefill ms | tok/s | reused | 专家内核 ms |
|---|---|---:|---:|---:|---:|
| 融合 | 预热 | 165,971 | 375.3 | 0 | 111,975.9 |
| 融合 | 热 | 165,433 | 376.5 | 0 | 111,673.7 |
| **两段式** | 预热 | 65,978 | 944.1 | 0 | 12,389.0 |
| **两段式** | 热 | **65,077** | **957.2** | 0 | **12,179.0** |

- **预填充 165.4 → 65.1 s = 2.54×**（375.3 → 957.2 tok/s）
- **专家内核 111,674 → 12,179 ms = 9.17×**
- 同一个配置的两次运行差 <1%（166.0/165.4；66.0/65.1）→ **不是缓存**

### 8.2 单类型 TFLOPS：4.2–7.2 → 33.9–51.8

| 类型 | 阶段 | 融合 | 两段式 | 加速 | TFLOPS 变化 |
|---|---|---:|---:|---:|---|
| IQ1_M | gate+up | 54,540 ms | 5,150 ms | **10.6×** | 4.2 → 44.5 |
| IQ1_M | down | 23,145 ms | 2,730 ms | **8.5×** | 6.1 → 51.8 |
| IQ2_XXS | gate+up | 21,069 ms | 2,619 ms | **8.0×** | 4.2 → 33.9 |
| IQ2_XXS | down | 4,918 ms | 913 ms | **5.4×** | 7.2 → 38.7 |
| IQ2_XS | gate+up | 7,962 ms | 729 ms | **10.9×** | 4.5 → 48.7 |
| IQ2_XS | down | 39.7 ms | 39.1 ms | 1.0× | （rows/expert 4.3，走轻量路径） |

33.9–51.8 TFLOPS 是 **oneMKL f16 GEMM 真的跑在 XMX 上**的直接证据（融合路径 4.2 TFLOPS 是 F32 SIMT）。

### 8.3 上一轮为什么误判成"缓存预热"

`tools/overnight/tp_ab.sh` 的 `set_cfg()` 在切换配置时只清掉 TC 那三个开关：

```python
c['env'].pop('STRATA_DS4_TC',None); c['env'].pop('STRATA_DS4_NO_TC',None); c['env'].pop('STRATA_DS4_TC_MINROWS',None)
if extra: c['env'].update(json.loads(extra))
```

**`DS4_EXP_2PHASE` 从来没被 pop 过。** 所以：

- 正序（base → tp）：base 腿没有该键 → 真融合 376；tp 腿加上 → 944。**这就是对的 2.51×**
- 反序（tp → base）：tp 腿留下 `DS4_EXP_2PHASE=1`，base 腿**继承**了它 → 两条腿都在跑两段式 → 65,338 vs 65,278 ms → 被读成"没有收益"，还把真结果撤回了。

同一条 bug 也解释了 132K 那次"461 vs 457 没有收益"——那一对**也是两条腿都在跑两段式**。

> **规则（写进脚本注释了）**：A/B 脚本里，**每一个**开关都必须在切换函数里被显式清掉；
> 靠"这次没设置就等于关"是不成立的——上一腿留下的键会静默继承。
> 判定一个 A/B 是否可信，先看它是不是"同一个二进制 + 只差一个变量 + 两次顺序都跑过"。

### 8.4 新的阶段表：瓶颈已经换人了（62K，两段式，热）

| 阶段 | 融合 | 两段式 | 占比（两段式） |
|---|---:|---:|---:|
| **sparse attention** | 23,422 ms | **23,512 ms** | **36.1%** |
| **experts（含 waits）** | 119,511 ms | **19,221 ms** | 29.5% |
| compressor + indexer | 6,976 ms | 6,962 ms | 10.7% |
| q_b + rope | 5,764 ms | 5,739 ms | 8.8% |
| ffn mHC + router + sort | 4,372 ms | 4,379 ms | 6.7% |
| engram | 1,576 ms | 1,575 ms | 2.4% |
| attn mHC | 748 ms | 744 ms | 1.1% |
| shared expert | 694 ms | 691 ms | 1.1% |
| **合计** | 165,433 ms | **65,077 ms** | |

**下一步的靶子变了**：专家内核从 73% 的绝对主项降到 29.5%，**sparse attention 23.5 s 现在是第一**。
另外两段式自身还有 7.0 s 的自付成本（`experts incl waits` 19,221 − 内核 12,179）：其中约 6 s 是
**那唯一一遍解量化**（41.5 Gval/s），剩下是 gather/swiglu/copy 三个辅助 kernel。

### 8.5 顺序对照：把"先跑的那个吃亏"这条彻底排掉（62K，反序）

| 顺序 | 两段式 | 融合 |
|---|---:|---:|
| 融合先（8.1 那组） | 65,077 ms / 957.2 tok/s | 165,433 ms / 376.5 tok/s |
| **两段式先（反序）** | **65,072 ms / 957.2 tok/s** | **164,973 ms / 377.6 tok/s** |

四次运行、两种顺序、每次不同 nonce（`0 reused`）：
- 融合：165.4 / 166.2 / 165.0 ms → **165.0–166.2 s**
- 两段式：65.1 / 66.0 / 65.1 ms → **65.1–66.0 s**

组内相差 <1%，组间 2.53×。**顺序无关、缓存无关。** 这一组就是给 8.3 那条 bug 补的控制实验。

### 8.6 132K：272.8 → 487.0 tok/s（1.79×），上一轮的"长上下文无收益"同样是那条 bug

| 配置 | 运行 | prefill ms | tok/s | 专家内核 ms |
|---|---|---:|---:|---:|
| 融合 | 预热 | 485,660 | 271.9 | ~257,000 |
| 融合 | 热 | 484,041 | 272.8 | 257,038 |
| **两段式** | 预热 | 271,950 | 485.6 | ~45,000 |
| **两段式** | 热 | **271,159** | **487.0** | **45,109** |

- **预填充 484.0 → 271.2 s = 1.79×**，专家阶段 257,038 → 45,109 ms = **5.70×**
- 注意融合热跑 272.8 tok/s 与 README 里旧的 122K/271 完全对上 —— 那本来就是融合路径的数
- 上一轮报的"132K 461 vs 457 没有收益"里，**461 那个数已经是两段式的数**（base 腿继承了开关），
  真值是 272.8（融合）vs 487.0（两段式）

**132K 的新阶段表（两段式，热）：**

| 阶段 | 融合 | 两段式 | 两段式占比 |
|---|---:|---:|---:|
| **compressor + indexer** | 82,506 ms | **82,073 ms** | **30.4%** |
| **sparse attention** | 49,973 ms | **50,209 ms** | 18.6% |
| **experts（含 waits）** | 257,038 ms | **45,109 ms** | 16.7% |
| **q_b + rope** | 41,287 ms | **41,127 ms** | 15.3% |
| engram | 32,392 ms | 31,751 ms | 11.8% |
| ffn mHC + router + sort | 10,309 ms | 10,342 ms | 3.8% |
| **合计** | 484,041 ms | **271,159 ms** | |

长上下文这一列把方向改得很清楚：专家内核吃完之后，**compressor + indexer（82 s）是 132K 的第一瓶颈**，
它 + sparse attention + q_b/rope 合计 173 s = 64%。而 62K 的第一瓶颈是 sparse attention（23.5 s）。
下一刀必须切在注意力侧，继续磨专家内核已经没有意义（它只剩 16.7%）。

### 8.7 目标状态

| 目标 | 指标 | 现在 |
|---|---|---|
| 阶段一 稳定 400+ | 62K 预填充 | **957.2 tok/s** ✅ |
| | 132K 预填充 | **487.0 tok/s** ✅ |
| 阶段二 冲刺 1000 上下 | 62K 预填充 | **957.2 tok/s**（距 1000 差 4.5%） |
| | 132K 预填充 | 487.0 tok/s（需要再 2×，靶子是 attention 侧 173 s） |

## 九、132K 的新瓶颈：compressor + indexer（82 s / 30%）诊断（2026-11-03 14:0x）

两段式把专家阶段拿掉之后，132K 的第一瓶颈变成 `compressor + indexer`。下面是**已经确认的事实**与
**还没测的部分**，分开写清楚。

### 9.1 几何（来自模型 gguf，实测读取，不是猜）

| 参数 | 值 |
|---|---|
| `indexer.head_count × key_length` | **32 × 128** |
| `indexer.top_k` | 512 |
| `compress_ratios` | 2 或 1（前 20 层 2，后 20 层 1） |
| `index_source_layer_ids` | **[2, 8, 14, 20, 24, 28, 32, 36]** → **只有 8 层**有 indexer |
| `candidate_block_size` / `candidate_topk_blocks` | 8 / 2048（即候选上限 **16,384 个位置**） |
| 键的存储格式 | `kKvFp4p`：每 32 个值一个 UE8M0 scale，E2M1 码（**每行 68 B**，fp4） |

### 9.2 内核结构（`ds4_kernels.dp.cpp:1216 index_score_kernel`）

- grid = `(ceil(n/256), T)`，256 线程；每个 work-group = **一个 query t**，共享内存里放该 query 的
  `heads*hd + heads = 4128` 个 float（16.5 KB）。
- **每个 work-item = 一个上下文位置**，做 `32 heads × 128 dim = 4096` 次 MAC。
- 内层：`for c(step 8) { 解 8 个 fp4 键值；for h in 0..32 { 从共享内存取 qh[8]；8 次 FMA } }`

→ **每 1 次 FMA 伴随 1 次共享内存读**（`qs` 是广播读，但还是 1 条 load 指令）。这是算术强度上的先天不足：
4096 次 FMA 需要 4096 次 SLM load，FMA 管道填不满。

### 9.3 算一下它到底做了多少 FLOP（132K）

候选裁剪在 `clen > 16,384` 时生效（`cand_blk=8 × cand_topk=2048`），所以每个 query 最多打 16,384 个位置：

```
132,068 queries × 16,384 candidates × 4,096 MAC = 8.9e12 MAC = 17.7 TFLOP   （每个 index layer）
× 8 个 index layer                              = 142 TFLOP（下限，前提是候选裁剪全程生效）
```

实测这个阶段是 **82.1 s** → 隐含 **≈1.7 TFLOPS**。

对比同一张卡上已经测到的数：

| 内核 | 实测 TFLOPS |
|---|---:|
| 融合专家内核（F32 SIMT，被两段式替掉的那个） | 4.2 |
| 两段式 oneMKL f16（XMX） | 33.9–51.8 |
| **indexer（F32 SIMT，1 SLM load / FMA）** | **≈1.7** |

**结论（推断，基于上面的下限算式 + 实测 82.1 s）**：indexer 是全引擎每 FLOP 最慢的内核，
只有那个已经被判定为"慢"的融合专家内核的 40%、oneMKL f16 的 3–4%，而它现在是 132K 预填充的 30%。
如果它能到融合内核的 4.2 TFLOPS，82 s → 34 s，132K 预填充 271 → 223 s（487 → 593 tok/s）。
如果能到 20 TFLOPS，82 s → 7 s，132K → 196 s（677 tok/s）。

### 9.4 为什么不能照搬专家的两段式（三个真实约束）

1. **显存没有余量**：当前 `--vram-experts 1.0` 下只剩 **595 MiB**。score 矩阵是 `T × clen`，
   而 `c.AI`（indexer 的 query slab）就是按 64 MB 预算算出来的：
   `max(32, min(128, (64MB / ((max_context+64)*4)) / 32 * 32)) = 32 行`。
   想吃张量核就得放大 slab（M 太小、调用次数太多），而放大 slab = 放大 score 矩阵 = 显存不够。
2. **oneMKL 没有融合 epilogue**：`S = Σ_h relu(S_h)·w_h`，relu 是逐头的，必须先出 32 个分头结果再合并。
   32 个 `clen × T` 中间矩阵的读写流量（每 slab 约 270 MB）会把收益全吃掉。
3. **调用次数**：slab=32 时，132K 一个 prompt = `ceil(132068/32) × 32 heads × 8 layers ≈ 1.06M` 次小 GEMM。

所以"换 GEMM"这条路在**当前显存预算下不成立**，除非先改 slab/显存策略。真正可行的是**把内核本身写快**：

### 9.5 下一步（三条，按代价排）

| # | 做法 | 预期 | 风险 |
|---|---|---|---|
| 1 | **q 进寄存器 + 一个 work-item 吃 P 个位置**（把"1 SLM load / FMA"降到 1/P） | 1.5–2.5×（→ 82 s 变 35–55 s） | 低：累加顺序不变，**理论上逐位一致** |
| 2 | 键按 head 分块解到 f16 scratch，用 MKL 做 `[clen×128]×[128×T]`，relu 求和放进紧跟的 elementwise（分头不落地？见 9.4 第 2 条） | 3–5× | 中：多一次中间流量 |
| 3 | 把 slab 预算从 64 MB 放大（需要腾显存，例如 `--vram-experts` 降到 0.9） | 未知 | 中 |

### 9.6 还没测的（不要当成结论）

- **阶段内部的拆分**（keys 的 wkv mv / finish / wproj gemv / index_score / block_max / topk 各占多少）。
  9.3 那个 1.7 TFLOPS 是把整段 82 s 都记在 index_score 名下的**上界估计**，真实值只会更高。
  要把这 5 个部分分开，需要给该阶段加子计时或写独立微基准——下一轮做。
- **62K → 132K 增长 11.7× 而提示只长 2.1×** 这件事还没解释。候选裁剪生效后，每个 query 的成本应该
  **封顶**（16,384 个位置），总量应该接近线性增长。多出来的约 2× 目前没有归因，不能排除
  block_max / 候选列表这类按 `clen` 全量扫的配套 kernel。

### 9.7 微基准实测（`index_score_bench`，2026-11-03 14:2x，Arc Pro B70）

独立靶子：不加载模型、不起引擎，直接按引擎真实形状（T=16 queries、32 heads、hd=128、fp4 键 80 B/行、
out 行步长 393,280）跑参考内核，外加一个「一个 work-item 吃 2 个位置」的原型。

```
device: Intel(R) Arc(TM) Pro B70 Graphics
n         ref (1 位置/work-item)   P2 (2 位置/work-item)   加速
          ms      GFLOP/s          ms      GFLOP/s
 4096     0.26     2044             0.32     1668           0.82x
16384     0.98     2198             0.97     2209           1.01x
66034     1.84     4695             1.87     4639           0.99x
逐位校验：三档都是 0 处不同，max|err| = 0
```

三条结论：

1. **内核真实吞吐 ≈ 2.2 TFLOPS**（n=16,384 是候选裁剪后的真实单 query 成本）。
   对照：F32 SIMT 峰值约 19.6 TFLOPS（512 EU × 8 lane × 2 FMA × 2.4 GHz）→ **只用到 11%**；
   融合专家内核 4.2、oneMKL f16 33.9–51.8。
2. **「q 进寄存器 / 一 work-item 吃多位置」这条不成立**：P2 是 0.82–1.01×，等于没动，
   所以内核**不是共享内存带宽受限**（我原来的假设错了，这是负结果）。它同时是逐位相等的，
   所以如果哪天要采纳，正确性上没有风险。
3. **代价模型被整机数据验证了**：按候选上限算，132K 每个 index layer 是
   `132,068 queries × 16,384 × 4,096 MAC = 8.86e12 MAC`；2.2 TFLOPS → **8.0 s/layer × 8 layer = 64 s**。
   实测该阶段 82.1 s → **index_score 约占 64 s（78%）**，其余 18 s 是 wkv mv / finish / wproj / block_max / topk。
   （这条把 9.3 那个 1.7 TFLOPS 的粗略估计换成了实测值 2.2 TFLOPS 和明确的 64 s。）

所以下一步要动的不是「把 FMA 写得更紧」，而是要么**减少每个 query 要打的位置数**
（`candidate_topk_blocks=2048 × block=8 = 16,384`，而 `index_top_k` 只要 512，等于 32 倍过采样），
要么**换计算单元**（张量核，受 9.4 那三个约束）。两者都不是几行代码的事，留到下一轮。

## 十、9.3 的归因被实验证伪：index_score 不是那 82 s 的主项（2026-11-03 14:4x）

### 10.1 微基准给出的强信号

同一份 fp4 键数据，只改每行占的字节数（内容、解码、累加顺序全不变）：

| n=16,384（候选裁剪后的真实单 query 成本） | ms | GFLOP/s |
|---|---:|---:|
| fp4 行跨距 **80 B**（原状，16 字节对齐） | 0.72 | 2,995 |
| fp4 行跨距 **96 B**（32 字节对齐） | **0.16** | **13,491** |
| fp4 行跨距 128 B | 0.17 | 12,985 |
| f16 键（256 B/行） | 0.16 | 13,633 |
| 不读键（纯 FMA 内环） | 0.08 | 25,655 |

80 B 不是 32 的整数倍，每行的 16 次取数跨扇区、无法成宽读。改成 32 B 对齐（`kv_row_bytes(kKvFp4p)`
的 16 → 32）在**孤立环境**里快 4.5×，而且**逐位不变**（1,056,544 个值 0 处不同）。已改并重新编译。

### 10.2 引擎端：一动不动

| 长度 | 阶段 `compressor + indexer` 改前 | 改后 | 变化 |
|---|---:|---:|---:|
| 62K（62,288 tok） | 6,961.6 / 6,976.0 / 6,975.6 ms | **7,019.6 ms** | **≈ 0（噪声内）** |
| 132K（132,067 tok） | 82,073.4 ms | **80,993.8 ms** | **−1.3%（噪声内）** |

端到端：62K 64,972 ms / **958.7 tok/s**；132K 269,325 ms / **490.4 tok/s** —— 与改前同一水平。

### 10.3 所以 9.3 那个"index_score 占 78%"是错的

这条链是我在 9.3 里用算式推的（当时明确标了"推断"），现在被实验推翻：
**如果 index_score 真有 64 s，把它的键读取提速 4.5× 不可能在整机上一点都看不见。**
由空结果可以反推一个上界：62K 那个阶段只有 7.0 s，若 index_score 占 1 s 以上，4.5× 至少该让该阶段掉 10%；
实测掉 0。**所以 index_score 在 62K 的这个阶段里 ≤ 约 0.3 s**，132K 同理远不到 64 s。

也就是说：`compressor + indexer` 那 82 s（132K 的 30%）**花在别处**。剩下的候选是
compressor 的 latent finish/quantize、indexer 的 wkv mv 与 finish、`gemv_f16_rows(wproj)`、
`block_max`、两次 `topk`，以及 `xfer_put`/`replicate` 这类跨设备路径。
它到底是谁，**必须实测，不能再靠算式推**——这是下一轮第一件事。

### 10.4 保留下来的改动与它的边界

- 改动保留：`kv_row_bytes(kKvFp4p)` 16 → 32 字节对齐。**逐位不变**、显存 +20% 于键缓存
  （132K 时 10.6 → 12.7 MB，可忽略），在当前规模上**已实测为中性**。保留的理由是行变成 32 B 对齐，
  对任何将来的宽读/向量化都是前提；不是因为它现在更快。
- 这一轮的净收获不是提速，而是**把一条错的归因钉死了**：9.3 那节已经把"推断"两字写在脸上，
  但仍然影响了"下一步该动哪里"的判断。附带一个可复用的判据：

> **判别一个"部件占大头"的结论真不真的最便宜办法，是把该部件单独提速一次。**
> 提速生效而整机不动 = 归因错了。这比继续在算式上加假设快得多，也可靠得多。

## 十一、132K 那 82 秒的**实测**拆分，以及由此拿到的 +19%（2026-11-03 16:0x）

上一节把 9.3 的算式归因证伪了，但没说清 82 秒到底是什么。这一节用**设备事件子标记**（`prof_mark` 的新槽位
14–19）把它拆开。先验证了标记本身不花钱：`DS4_PROF` 关掉后 62K 是 65,992 / 65,239 ms，
开着是 65,077 / 65,652 ms —— 同水平。

### 11.1 拆分（132K，slab=32 的老配置，预填充 289 s / 457 tok/s）

| 子项 | ms | 占预填充 |
|---|---:|---:|
| compressor（`compress_win`） | 698.7 | 0.3% |
| index keys（wkv + finish） | 110.2 | 0.0% |
| indexer 的 wproj | 24.1 | 0.0% |
| **slab: wq_b GEMM + finish** | **32,891.0** | **12.2%** |
| **slab: index_score + cand + topk** | **48,234.2** | **17.9%** |
| 尾部（kv_src finish/replicate） | 725.9 | 0.3% |
| **合计** | **82,684** | 与旧口径 82,073 对得上 |

两件事：
1. **打分（index_score）确实是主项，48.2 s。** 上一节"≤0.3 s"的反推是错的——那个反推假设了
   "对齐后 4.5× 会体现在这个阶段"，但引擎走的是**候选 gather**路径（`p = cand[t][i/blk]*blk + i%blk`），
   散取访问把按行对齐的收益吃掉了。基准只量了连取的情况。**这是这一节的第二个自我更正。**
2. **还有一个完全没想到的 32.9 s：`wq_b` 的 query 投影 GEMM**，占预填充 12.2%。

### 11.2 那个 32.9 s 的来源：slab 只有 32 行

```cpp
c.AI = max(32, min(128, (64 MB / ((max_context+64) * 4)) / 32 * 32));   // max_context 393216 -> 32
```

`c.AI` 是 indexer 一次处理多少个 query。它是一个**硬编码的 64 MB score 预算**算出来的，
在 393,216 上下文下正好落地成 **32 行**。于是每个 slab 都要跑一次 `wq_b` GEMM，而 M=32 的 GEMM 完全喂不饱：

```
每个 slab：32 行 × q_lora × (32 head × 128) = 201 M MAC
132K 一共 ceil(132068/32) × 8 个 index layer = 33,016 个 slab → 6.6 T MAC = 13.3 TFLOP
实测 32.9 s → 0.4 TFLOPS
```

同一张卡上 oneMKL f16 在专家上能到 33.9–51.8 TFLOPS。**0.4 就是"每 32 行发一次 GEMM"的代价。**

### 11.3 一次改一行：把 slab 预算做成可配（`DS4_IX_SLAB_MB`，默认 64 = 老行为）

132K 用 256 MB（`c.AI` → 128，score 矩阵 128 × 393,280 × 4 = 201 MB，显存里放得下）：

| | slab=32（默认） | **slab=128（256 MB）** | 变化 |
|---|---:|---:|---:|
| wq_b + finish | 32,891 ms | **8,271 ms** | **4.0×** |
| index_score + cand + topk | 48,234 ms | **29,233 ms** | **1.65×** |
| indexer 小计 | 82.7 s | **38.4 s** | **−53%** |
| **132K 预填充** | 271,159 ms / 487.0 tok/s | **228,362 ms / 578.3 tok/s** | **+18.7%** |

`index_score` 也跟着快了 1.65×，因为一次 128 个 query 让每个键行在被读出后摊到更多 query 上。

62K 不变（65,461 ms / **951.5 tok/s**，改前 65,077 / 957.2，噪声内）—— 那个长度上 indexer 只占 7 s，
省下的 3 s 看不出来。

### 11.4 更新后的目标状态

| 目标 | 指标 | 现在 |
|---|---|---|
| 阶段一 稳定 400+ | 62K | **951.5 tok/s** ✅ |
| | 132K | **578.3 tok/s** ✅（本轮从 487 提上来的） |
| 阶段二 冲刺 1000 上下 | 62K | **951.5 tok/s**（距 1000 差 5%） |
| | 132K | 578.3 tok/s（下一步：sparse attention 50.2 s / experts 46.1 s / q_b+rope 40.9 s / indexer 38.4 s） |

### 11.5 这一轮的方法论收获

- 拆分之前必须先确认**改的那段代码真的会被执行**：我第一版子标记打在 `layer()` 里，四个槽全 0，
  才发现那是死代码；真正的分块路径在 10000 行附近，两处代码几乎逐字相同。
- 打印范围也要检查：18/19 号槽没打印，于是"子项加起来只有 1 秒"这个假象把方向带偏了一轮。
- 结论：**`compressor + indexer` 这个标签名本身就是错的**——那 82 秒里 40% 是 query 投影 GEMM，
  32% 是打分，真正属于 compressor 的只有 0.7 秒。

## 十二、`q_b + rope` 那个标签也是错的（2026-11-03 16:5x）

十一节拆分之后，132K 排第二的是 `q_b + rope` 40.9 s（18%），但它在 62K 上只有 5.7 s——2.1 倍的提示
长 7.2 倍，超线性。原因是**标记打在 slab 循环内部**（`prof_mark(d, 8)` 在 `finish` 之后），
所以每个区间都会把上一个 slab 的尾部算进来。加三个子标记（20/21/22）把它劈开：

| 132K | ms | 占预填充 |
|---|---:|---:|
| **上一个 slab 的尾巴**（mark 10 → 本 slab 开始） | **28,063.5** | **12.3%** |
| 注意力 `wq_b` GEMM（M=2048） | 10,732.4 | 4.7% |
| q 的 rope + 量化（`finish`） | 2,244.4 | 1.0% |
| 合计 | 41,040 | 与旧口径 40.9 s 对得上 |

所以"q_b + rope 40.9 s"里 **68% 根本不是 q_b**，而是 mark 10 之后到下一个 slab 之间那段代码
（即将在下一轮再插一个标记把它定位）。同一个毛病在 `sparse attention`（50.2 s）上也可能存在，
因为它的区间也是跨 slab 的——**共享同一套标记的相邻阶段，归属必须逐个验证，不能照标签读**。

到目前为止 132K（228–243 s，578→543 tok/s 之间波动）的**可靠**占比是：

| 项 | 量级 | 状态 |
|---|---|---|
| sparse attention | 50.2 s | 标签单独一段，量级可信 |
| 专家（3 类合计） | ~46 s | 逐类型内核时间，可信 |
| mark 10 → 下一 slab 的尾巴 | 28.1 s | **未定位** |
| indexer 打分 | 29.2 s | 可信（子标记） |
| indexer 的 wq_b + finish | 8.4 s | 可信 |
| 注意力 wq_b GEMM | 10.7 s | 可信 |
| q rope + 量化 | 2.2 s | 可信 |
| engram / ffn mHC / compressor / keys / wproj | 各 ≤1.6 s 或 ~31 s（engram） | — |

62K（65.5 s / 951.5 tok/s）仍然是 sparse attention 23.5 s（37%）+ 专家 19.1 s（30%）最大；
要摸到 1000 tok/s 只需要在这 65.5 秒里省下 3.2 秒——sparse attention 提 14% 就够。

## 十三、132K 的账终于闭合：那 28 秒是**等专家搬运**，不是计算（2026-11-03 17:1x）

十二节留下的 28.1 s（"mark 10 → 下一个 slab"）定位了。mark 10 之后依次是：

1. 共享专家：`gemm(s1)` / `gemm(s3)` / `swiglu` / `rows_f16` / `gemm(s2)` → **`cprof_ms[2]` = 2,409 ms**
2. **`while (s->ext_oneapi_empty()) { flow_pump(c, err); yield(); }`** —— 主机在队列空的时候
   不停喂专家拷贝，直到这一层的专家搬完

而 `cprof_ms[3]`（"experts (incl waits)"）= **45,623 ms** 与逐类型内核时间的合计（≈46 s）几乎相等，
说明**专家等待根本不在那个桶里**——它就在我这轮新加的 28,063 ms 里。

### 13.1 132K（228,362 ms / 578.3 tok/s）的完整分解

| 项 | ms | 占比 | 性质 |
|---|---:|---:|---|
| **sparse attention** | 50,165 | 22.0% | 计算 |
| **专家内核（3 类合计）** | ~45,600 | 20.0% | 计算（oneMKL f16 33.9–51.8 TFLOPS） |
| **专家搬运 pump 等待** | **28,064** | **12.3%** | **等待，不是计算** |
| indexer 打分 + topk | 29,246 | 12.8% | 计算 |
| engram | ~31,400 | 13.7% | 计算 |
| 注意力 wq_b GEMM | 10,732 | 4.7% | 计算 |
| ffn mHC + router + sort | 10,327 | 4.5% | 计算 |
| indexer wq_b + finish | 8,369 | 3.7% | 计算 |
| attn rest / shared expert / compressor / keys / wproj / rope / between layers | ~7,000 | 3.1% | 计算 |

合计 ≈ 228 s，**全部有归属**。

### 13.2 这条发现的价值

**12% 的时间是设备在饿着等权重搬过来**，不是算得慢。而搬运这段正是本会话早先翻过车的地方：

- 上一轮（11-02 晚间）曾把预填充的墙归因为"专家权重搬运"，后来用阶段表**否掉**了——那次的依据是
  "experts (incl waits) 257,038 ms 里等待只有 15,769 ms"。**现在看那个结论对一个 132K 的 485 s
  预填充是对的（3.2%），但它当时是在融合路径上测的，而且 `incl waits` 这个桶里根本没有 pump 等待。**
- 换句话说：**"experts (incl waits)" 这个名字是错的**（第三次遇到同类问题），真正的等待在这 28 秒里。
- 生产配置里 `DS4_AHEAD=0`（预取关）、`DS4_RING_DEP=host`。前者藏的是 SSD→page cache（这台机器上
  专家已在内存里，没用），后者管的是 RAM→VRAM 的 ring。**28 秒的饿等正好落在后者负责的范围**，
  而本会话记忆里那条 DeepInfer 的可搬经验（一层一张票据 + 32 深 DMA 队列 + 设备端完成位图，
  我们自己的 hotpin 实测 4/16/32 队列 = 23.5 / 30.5 / 35.0 GB/s，+49%）正对着它。

### 13.3 下一刀

1. 把 pump 等待从 28 s 压下去：环深度 / 票据粒度（`DS4_RING_DEP`、`DS4_ASYNC_T`，以及每层一张票据）。
   **这一项不涉及任何数值正确性，风险最低，收益按 20% 估就是 −28 s（132K 578 → 640 tok/s）。**
2. sparse attention（50 s / 22%，62K 上是 23.5 s / 37%）——62K 只要省 3.2 s 就破 1000 tok/s。

## 十四、pump 等待：调大异步窗口没用（负结果，2026-11-03 17:3x）

上一节把 132K 的 28.1 s 定位成 `while (s->ext_oneapi_empty()) { flow_pump(...); yield(); }`
这个主机自旋（设备队列空着，等专家拷贝）。最省事的假设是"主机不够早地跑在前面"，于是先试已有的旋钮：

| 132K | prefill | tok/s | pump 等待 |
|---|---:|---:|---:|
| `DS4_ASYNC_T` 默认 8 | 228,362 ms | 578.3 | 28,063 ms |
| **`DS4_ASYNC_T` = 24** | **229,165 ms** | **576.3** | **27,948 ms** |

**在噪声内，一点没动**（`DS4_ASYNC_T` 关的是 window 路径的异步窗口数，分块路径的饥饿不是它管的）。
配置已还原成默认。

所以这个 28 秒不是"提交得太浅"，得往搬运的**粒度/票据**上看：
`DS4_RING_DEP=host`（copy 流不再等 compute 流的 ring_free，已在用）之外，
要么一层一张票据地批量发（DeepInfer 那条经验），要么把每层 6 个专家的拷贝合成更少的更大传输。
这属于结构性改动，需要动 copy 路径，**不能在配置层解决**——下一轮从读 `flow_pump` 的票据逻辑开始。

（这一轮没有提速，唯一的产出是把一个假设干净地否掉了，避免下一轮在错误的旋钮上继续调。）

### 14.1 环本身没问题：208 槽 × 10.2 MB，全在 VRAM

`flow_pump` 的票据逻辑读完了，能提前 208 个专家（≈34 层）拷贝：

```
ds4: the expert ring: 208 slots x 10229760 bytes (2029 MB) in VRAM
```

`DS4_RING_MB` 有 2048 MB 的下限（源码注释里记着 1014 MB 在 B70 上死锁过一次、2029 MB 正常），
`ds4_ring_device()` 默认开。所以**"环太浅 / 环在主机内存"这两条都不是那 28 秒的原因**。
结合 14 节的负结果，配置层与环深度这两个假设都排掉了。

剩下的可能只有两条，都需要动 copy 路径的代码，不是参数：
1. **每层 6 个专家、每个 10.2 MB 地发**（`flow_pump` 里每个 item 组 3 个 `IoQueue::Job`）——
   票据粒度太细，PCIe 上跑不到峰值（记忆里 hotpin 实测 4/16/32 队列 = 23.5/30.5/35.0 GB/s，
   说明队列深度对实际带宽影响 +49%）；
2. `while (s->ext_oneapi_empty())` 这个判据本身——它在**整条计算队列空**时才转起来，
   而设备其实可能只是暂时没有可发的活，这个轮询粒度和 2 ms 的 ring 轮询（`sleep_for(2ms)`）叠加
   会把短暂的饥饿放大成秒级。

下一轮从这里继续：先给 `flow_pump` 加"每次调用发了几张票据 / 实际等了多少"的计数（不改逻辑），
把 28 秒进一步分成"等 ring"、"等依赖"、"纯 PCIe 带宽"三类，再决定改哪一处。

## 十五、同一套标记量 62K：pump 等待只有 3.9 秒，**主项是 sparse attention 37%**（2026-11-03 17:1x）

把 132K 那套子标记原封不动地在 62K 上跑一遍，得到完整对照（62,288 token）：

| 项 | 62K | 132K | 说明 |
|---|---:|---:|---|
| **sparse attention** | **23,470.8 ms（37.4%）** | 50,199.9 ms（22.1%） | 两个长度上都在前列 |
| 专家内核（合计） | ~19,000 ms（30%） | ~45,600 ms（20%） | 逐类型实测 |
| indexer 打分 + topk | 5,587.3 ms（8.9%） | 29,246.3 ms（12.9%） | 超线性（5.2×） |
| ffn mHC + router + sort | 4,376.9 ms（7.0%） | 10,327.2 ms（4.5%） | |
| **pump 等待** | **3,867.7 ms（6.2%）** | 28,063.5 ms（12.3%） | **超线性 7.3×** |
| 注意力 wq_b GEMM | 1,184.6 ms（1.9%） | 10,732.4 ms（4.7%） | |
| q rope + 量化 | 690.6 ms（1.1%） | 2,244.4 ms（2.2%） | |
| compressor / keys / wproj / slab wq_b | 499 ms（0.8%） | 8,546 ms（3.8%） | |

### 15.1 这条对照改变了主攻方向

**62K 上 pump 等待只有 3.9 秒（6.2%），而目标是"1000 tok/s = 62.3 秒"，现在 65.5 秒，只差 3.2 秒。**
就算把 pump 等待压到 0，也不够；而 **sparse attention 23.5 秒只要提 14% 就正好补上这 3.2 秒**。

所以：
- **要破 62K 的 1000**：打 sparse attention（37%），不是 pump。
- **要提 132K**：pump 等待（28 s / 12.3%）和 sparse attention（50 s / 22%）都值，indexer 打分（29 s）次之。

上一轮把"下一步读 flow_pump 票据逻辑"排在首位，是以 132K 的数字为依据的；62K 的对照说明**两个长度的最优动作不同**，
先做 sparse attention 才对阶段二有用。

### 15.2 pump 等待为什么超线性（7.3× 对 2.1× 的 token）

两个长度的 rows/expert 差不多（62K 294–457，132K 328–403），所以不是"每专家搬得更多"。
更可能是**每层被摸到的不同专家数变多**：132K 时每个 chunk 几乎摸遍 384 个专家，而 ring 只有 208 槽
（＝提前 34 层），一旦某一层需要的专家集合与上一轮重叠很少，拷贝就追不上——这条**仍是推断，未验证**。
验证方法是给 `flow_pump` 加"每次调用发了几张票据 / 等在哪一类"的计数（只计数不改逻辑），下一轮做。

## 十六、`DS4_ATTN_SLAB` 4096 vs 2048：中性；顺带修正 62K 基线（2026-11-03 17:4x）

indexer 的 slab 调大赢过一次，于是照同样思路试注意力的 slab（`c.A`，`DS4_ATTN_SLAB`）：

| 62K，slab | 两次运行 | tok/s |
|---|---|---:|
| 2048（原配置） | 64,658 / 64,252 ms | 963.3 / 969.4 |
| 4096 | 64,694 / 64,000 ms | 962.8 / 973.3 |

**一模一样**。`sparse attention` 阶段也没动（23,471 → 23,258 ms）。注意力这条路的 slab 已经够大，
和 indexer 那个 32 行的病不是同一回事。配置已还原成 2048。

### 16.1 顺带要修正的数字（我一直在用偏低值）

八轮里我一直报 62K = **951.5 tok/s**。这次连续四次运行（重启后、热缓存、每次不同 nonce）是：

```
64,658 / 64,252 / 64,694 / 64,000 ms   ->   963.3 / 969.4 / 962.8 / 973.3 tok/s
```

**稳定区间是 963–973 tok/s，不是 951.5。** 差的 2% 来自早先那次测的机器状态（当时的四次是
64,972 / 65,077 / 65,461 / 65,239 ms）。所以：

- 距 1000 tok/s 不是差 5%，而是**差 3%（约 2 秒）**；
- 之前把"需要省 3.2 秒"当目标定得略宽，实际需要 **约 2.0–2.4 秒**。

（这个纠正不改任何结论的方向，但对外口径该用新值。132K 的 578.3 tok/s 没有重测，仍按实测值引用。）

## 十七、解码基线（2026-11-03 18:0x）：**16–18 tok/s，45% 的时间在等 CPU 车道**

预填充收工后转解码。这里先把基线测扎实，规矩沿用预填充那套：≥128 输出 token、每次不同 prompt（不用 nonce，
因为解码本来就没有 KV 复用的收益问题）、连跑多轮取稳定区间，并且**只认引擎自报的 `predicted_ms`**（纯解码段），
不拿墙钟算。

| 跑次 | 提示 | 输出 | 纯解码 ms | **纯解码 tok/s** |
|---|---:|---:|---:|---:|
| 1 | 55 | 200 | 10,652.1 | **17.2** |
| 2 | 55 | 200 | 11,639.2 | **15.7** |
| 3 | 45 | 131 | 6,322.9 | **18.0** |

**稳态解码 ≈ 16–18 tok/s（55–63 ms/token）。**
（注意：上一轮会话报的 20.8–20.9 tok/s 是另一套条件下的数，当前配置下没有复现；这里就是当前基线。）

顺带一个要记住的：55 token 的提示 `prompt_ms` 要 2.0–2.2 s（≈25 tok/s），因为
`ds4: prompt path: the chunked pass from 96 tokens, windows of 16 below`——**96 token 以下是窗口路径，
不是我们刚优化过的分块路径**。所以短提示的"预填充"看起来慢是路径不同，不是回退。

### 17.1 解码的阶段表（`DS4_PROF` 的 decode per token）

```
between layers        0.12 ms        experts gate/up       2.51 ms
attn tail (o_fin+wo)  6.62 ms        cpu-lane wait        27.31 ms   <- 45%
routing+shared        7.62 ms        down+post            12.31 ms
attn proj             7.91 ms        compressor+indexer    1.37 ms
sparse attn           1.94 ms        -----------------------------------
                                     total                67.7 ms/token
```

**第一瓶颈是 `cpu-lane wait`：27–33 ms，占 45%。** 这一项的语义是"GPU 空着，等 CPU 车道把专家算完"。
按引擎日志的 CPU 车道统计（`entries/call 4.8`、`gate+up 0.748 ms`、`down 0.394 ms` ≈ 0.24 ms/专家），
一个 token 要过约 215 个非驻留专家 → 约 51 ms 的 CPU 车道工作量，与 27–33 ms 的等待同量级。

对照：`--vram-experts 1.0` 下显存只放得下 **1616 / 15360** 个专家（10.5%），
其余 89.5% 每 token 都要从主机内存读一遍——这一项本质上是**内存带宽问题**，
而本会话记忆里已有的实测是：CPU 车道已跑到约 **38 GB/s**，本机 DDR5 纯读峰值 **49–51.6 GB/s**，
即最多还有 **1.3×** 的空间。

### 17.2 下一刀（按序，一项一项来）

1. **`cpu-lane wait`（45%）**——先量清楚它到底是"带宽"还是"调度"：给 CPU 车道加"每个 token 读了多少字节 /
   stall 在哪"的计数（只计数不改逻辑）。若是带宽，1.3× 的天花板在；若是调度（线程数/亲和性/分块粒度），
   空间可能更大。
2. `down+post`（12.31 ms，18%）与 `attn proj`（7.91 ms，12%）——这两个是 GPU 侧，可并行调。
3. MTP/草稿留到最后：记忆里那条结论（"投机解码在本机不划算：接受率 100%、草稿 7 ms/轮，但验证一步 285 ms，
   因为 6 个 token 的窗口每层要 36 个专家而不是 6 个"）是在**解题速度 45 ms/token、CPU 车道未优化**的前提下测的，
   等解码主项下来后需要重测，不能直接沿用。

### 17.3 `cpu-lane wait` 到底是什么（2026-11-03 18:2x）

引擎自己有一行更细的账，比阶段表更准：

```
ds4 cpu-lane: 42.00 ms/token of lane compute vs wait 27.79 ms
ds4 cpu-lane: 43.01 ms/token of lane compute vs wait 28.18 ms
ds4 cpu-lane: 42.33 ms/token of lane compute vs wait 28.05 ms
```

所以一个 token 的构成是：

| | ms/token |
|---|---:|
| **CPU 车道计算** | **42–43** |
| **GPU 等车道（`cpu-lane wait`）** | **28** |
| GPU 自己的活（down+post 12.3 + attn proj 7.9 + routing 7.6 + attn tail 6.6 + experts gate/up 2.5 + sparse 1.9 + comp/idx 1.4） | ≈40 |
| **墙钟** | **≈68** |

**关键结论：墙钟 68 ms 而车道计算只有 42 ms、GPU 自己的活约 40 ms。如果这两者完美重叠，一个 token
应该在 42–45 ms（22–24 tok/s），而不是 68 ms（15 tok/s）。那 28 ms 的等待不是"车道太慢"，
是"没有重叠上"——GPU 在每层末尾干等车道交结果，而不是跑到前面去。**

车道 42 ms 的物理来源也对得上：每 token 约 216 个非驻留专家 × 约 8.6 MB ≈ **1.6 GB**，
在实测的 38 GB/s 上就是 43 ms。这条是带宽，天花板 49–51.6 GB/s，**最多 1.3×**。

⚠ 我一分钟前说"CPU 车道只处理 1.6% 的专家所以那 28 ms 不可能是真实工作量"——**这句是错的**，
我把日志里 `CPU lane 1.6% of the experts` 这个字段的语义当成了"专家调用占比"。它显然不是
（42 ms/token ÷ 每 call 约 1.01 ms ≈ 41 call × 4.58 专家 ≈ 190 个专家/token，与 216 吻合）。
那个百分比是什么口径，需要读代码确认，**在弄清之前不要引用它**。

### 17.4 所以解码的第一刀不是"提速车道"，是"让车道和 GPU 重叠"

| 做法 | 预期 |
|---|---|
| **层内填充 / 提前一层发车道任务**（DeepInfer 的 `mlp.py:601-609` 那条：发完路由专家请求就立刻算 shared expert 与 norm/rope/store，回来再 combine） | 把 28 ms 的干等压掉大部分 → 68 → 45–50 ms（**20–22 tok/s**） |
| 车道带宽 38 → 50 GB/s | 42 → 32 ms，天花板 1.3× |
| 减字节（更多专家进显存） | 显存只放下 1616/15360，没有余量 |

下一轮先确认**车道任务与 GPU 的重叠边界在哪**（谁在等谁、每层几个同步点），再决定是改调度还是改线程。

### 17.5 那 28 ms 是**主机侧串行**造成的错峰，不是车道带宽不够（2026-11-03 18:4x）

三块证据连起来看：

1. **每 token 39–40 次设备等待**（`ds4 waits: 39.38 / 37.50 / 40.00 device waits/token`）——
   正好等于层数 40，即**每层一次** `wait_seq(&resps[slot].seq, sq, s)`（`ds4_model.dp.cpp:12865/12887`），
   位置在专家内核启动之前（等车道把这一层的答案发出来）。平均每层等 **0.7 ms**。
2. **`st.cpu_ms` 是主机线程的耗时**（`ds4_model.dp.cpp:13156`：`st.cpu_ms += now_ms() - tc;`，一个 `tc` 到
   `now_ms()` 的包夹），**不是多线程累加**。所以"42 ms/token 车道计算"= **主机线程被占住 42 ms**。
3. 于是：主机占 42 ms 做车道、GPU 自己有约 40 ms 的活，两者本该重叠成 ~42–45 ms，实测 68 ms。
   差出来的 ≈26–28 ms 就是阶段表里的 `cpu-lane wait`。

**结论（推断，但三条实测都指向它）：GPU 空转的原因不是车道算得慢，而是主机线程在车道里阻塞 42 ms，
这段时间里它没法继续往 GPU 派活，GPU 把已排队的活干完就空着了。**
这解释了为什么"车道 42 ms + GPU 40 ms"没有变成 45 ms 而是 68 ms。

### 17.6 下一刀：把车道从主机的提交线程上摘下来

| 做法 | 依据 | 风险 |
|---|---|---|
| **车道的计算放到工作线程，主机的提交循环不再阻塞**（`make_pools()` 已经有线程池，问题在于主机是在等池还是自己算） | 本轮三条实测 | 中：要保证顺序（第 L 层的答案必须在第 L 层内核前就绪），做错会读到上一层的路由 |
| 车道带宽 38 → 50 GB/s | 1.6 GB/token ÷ 38 GB/s = 43 ms | 低，但天花板只有 1.3× |
| 更多专家驻留显存 | 只能放 1616/15360 | 无余量 |

**先做第一项的可行性确认**（不改逻辑）：读 `make_pools()` 到车道调用的那段，确认主机是
`blocking-wait on pool` 还是 `自己算`。如果是前者，把等待改成"上一层的结果在这一层提交后再收"，
就能把这 42 ms 从关键路径上挪走；如果是后者，才需要改线程结构。

### 17.7 可行性确认：主机是**阻塞等池**，不是自己算（实测读代码，2026-11-03 19:0x）

```cpp
bool run_nodes(int n, const std::function<void(int)>* f) {
    for (int X = 0; X < NH; ++X) pools[X]->start(n, f[X]);
    pools[0]->help();                                   // 主机自己也参与算
    for (int X = 0; X < NH; ++X) ok = pools[X]->wait() && ok;   // ← 然后阻塞到所有池算完
    return ok;
}
```

所以 17.5 强的推断成立：**主机线程既参与计算、又在算完后阻塞等所有池**，这一整段就是 `st.cpu_ms` 的 42 ms。
17.6 表格里"若是前者（等池）就把 42 ms 挪出关键路径"——现在确认**是前者**。

### 17.8 但"挪走"没那么容易：单 token 的依赖链是严格串行的

一层的顺序是 `attention(L) → 路由(L) → 专家(L) → attention(L+1)`。
`attention(L+1)` **依赖** `专家(L)` 的输出，所以"车道算 L 的专家时 GPU 去算 L+1 的注意力"是做不到的。
能藏进车道那 42 ms 里的只有**不依赖专家输出**的活（共享专家、部分 norm），而引擎已经在做这件事
（`10200` 附近那句注释：共享专家"while the host waits for the bounds"）。

所以下一刀不是"把车道挪出关键路径"，而是：

1. **先把车道的 42 ms 变短**——它现在是 1.6 GB/token ÷ 38 GB/s 的带宽账，天花板 49–51.6 GB/s（1.3×）。
   做法：提高车道读的并发度（NUMA 池数/每池线程数）、或减少每 token 要读的字节（专家驻留策略）。
2. **再把能藏的活藏干净**：确认共享专家等确实与车道并行，且 28 ms 里没有"本可以并行却串行"的部分。

⚠ 这一节把 17.6 的排序改了：原先设想是"先把车道挪走（预期 20–22 tok/s）"，读代码后发现依赖链
不允许跨层重叠，所以**先做车道带宽（1.3×，稳但小），并行度/驻留策略（未知但可能更大）**。这是本轮
读代码得到的唯一实质修正。

### 17.9 车道不是"线程闲着"——工作项是 `ne × 块数`，池子是用满的（2026-11-03 19:2x）

上一轮猜"每次只有 4.58 个专家，按专家切分会让 30 个线程里大部分闲着"。读代码后**这个猜想不成立**：

```cpp
run_nodes(ne * GB, f);   // 7794 / 8005：工作项 = 专家数 × gate/up 的块数
run_nodes(ne * DB, f);   // 7819 / 8032：同理，down 的块数
```

`ne = 4.58`、`GB` 是块数（n_ff / BN 量级），工作项是 **上百个**，不是 4–5 个；池子按 NUMA 节点建
（`pools.push_back(std::make_unique<Pool>(nw, c, 20000))`，本机单 NUMA，一池覆盖全部核减 `STRATA_DS4_LANE_FREE=2`）。

**所以 38 GB/s 不是"线程没铺满"造成的。** 那把 0.22 ms/专家 换成吞吐：约 35.4 M 个值要解量化 + 约 35 M FLOP，
即 **160 Gval/s** 的解量化 + 160 GFLOP/s —— 这个量级对 30 线程的 IQ1_M（11 位网格索引，每值要做位抽取+查表）
来说是**算力受限**的样子，不是内存受限。

### 17.10 该结论又把下一刀改了（第三次）

| 假设 | 状态 |
|---|---|
| 车道阻塞主机 → 挪出关键路径能到 20–22 tok/s | ❌ 依赖链不允许（17.8） |
| 池子线程没铺满 → 加线程/改切分 | ❌ 工作项已是 ne×块数（17.9） |
| 车道是带宽受限（38 → 50 GB/s，1.3×） | ⚠ **未证实**：160 Gval/s 的量级更像算力受限 |

**要分清"算力"还是"带宽"，最便宜的一步是降线程数看吞吐怎么变**：把 `STRATA_DS4_LANE_FREE` 调大
（少给车道几个核），如果 38 GB/s 随线程数近似线性下降 → 说就算力受限；如果几乎不动 → 才是带宽受限。
这是一次配置级实验（不改代码），下一轮先做这个，再决定是优化解量化循环还是别的。

（本节只做了读代码与量纲推算，**没有实测**，结论按"推断"对待。）

### 17.11 判别实验：给车道少 14 个核，**速度不变**（实测，2026-11-03 19:4x）

用 `STRATA_DS4_LANE_FREE`（给车道留几个核不占）做了一次配置级实验，每档重启后跑 200 token：

| 车道可用核数 | 纯解码 tok/s |
|---|---:|
| FREE=2（约 30 核） | **17.6** |
| FREE=8（约 24 核） | **18.1** |
| FREE=16（约 16 核） | **17.3** |

**三档在噪声内完全一致。** 把车道的核砍掉近一半，解码一点没慢。

### 17.12 这条结果同时否掉了两个假设（第三次修正同一个项）

| 假设 | 判定 |
|---|---|
| 算力受限（解量化 ALU） | ❌ 少一半核也不慢 |
| 带宽受限（38 → 50 GB/s，1.3×） | ❌ 少一半核也没慢，与带宽随并发下降的形态不符 |
| 17.5/17.6 的"主机线程被车道占住 42 ms 是关键路径" | ⚠ **可疑**：如果车道真是关键路径，砍核必慢 |

**所以 `cpu-lane wait` 这 28 ms 大概率又一次是"标签名骗人"**——这是本次会话里第四次遇到
（`compressor + indexer`、`q_b + rope`、`experts (incl waits)`、现在这个）。
它可能测的是"设备在等某个 host 侧序列号"，而那个序列号的延迟与车道用多少核无关（例如固定的
hand-off 粒度、或主线程 help 份额、或内存延迟而非带宽）。

### 17.13 下一轮的判据（改成"先证伪归属"）

不要再假设它是车道。**直接测"把车道彻底停掉解码会怎样"**：
`--experts 6` 不变、但通过 `DS4_*` 让所有专家走 GPU（若显存不够则用最小上下文试探），
或更省事：**把一个 token 的 40 次 `wait_seq` 各自记时**，看这 28 ms 是均匀摊在 40 层（每层 0.7 ms）
还是集中在少数几层。若是前者且与车道核数无关 → 说明它是一个**固定粒度的 hand-off 延迟**，
那 28 ms 的敌人就不是车道，而是同步粒度。

### 17.14 找到了根因：**引擎自报的两个数互相矛盾**，所以前五轮的假设全打不中

| 引擎自报 | 值 | 含义 |
|---|---|---|
| `ds4 cpu-lane: 42.0 ms/token of lane compute` | 42 ms | 主机线程在车道上的耗时 |
| `ds4 prof CPU lane per call: entries/call 4.58, gate+up 0.669 ms, down 0.343 ms` | ≈1.01 ms/call | 每次调用 4.58 个专家 |
| `CPU lane 1.6% of the experts` | 1.6% | `s.cpu_experts / s.experts`（`src/dsv41/ds4_main.cpp:1235`） |

把前两条除一下：**42 ms ÷ 1.01 ms/call ≈ 41 call/token ≈ 190 个专家/token**（在 240 个里占 79%）。
而第三条说车道只占 **1.6%**（约 4 个专家）。

**这两个结论不可能同时成立。** 要么 `cpu_experts` 漏计了约两个数量级，要么 `cpu_ms` 里混进了
不属于车道的耗时（它就在 `dump_sub_host("cpu_part", ...)` 前面，那个块未必只做车道）。

这也解释了一件事：**过去五轮我试的五条路（挪出关键路径、加线程、改切分、算力、带宽）全都打不中，
因为被优化的那个对象本身就没被量准。** 而且 `STRATA_DS4_LANE_FREE` 改核心数完全不影响，
正是应该出现的结果——如果那 42 ms 大部分不是车道算的。

### 17.15 下一轮第一件事：修计量，不是优化

在动任何代码之前，先让三个数自洽：

1. 在 `cpu_experts_iq` 入口计一次调用数，与 `cpu_experts` 的增量比对，确认 41 call/token 这个推算；
2. 把 `st.cpu_ms` 的包夹（`ds4_model.dp.cpp:13156`）换成**只包住 `run_nodes`**，看它还剩多少；
3. 用 `DS4_PROF` 已有的 `wait_seq` 计数（39–40/token）配一次逐层计时，看 28 ms 是均匀还是集中。

只有当"谁花了 42 ms"被三个独立口径确认之后，解码优化才有意义。**这是本次会话在解码上留下的
最重要的一条：不是没找到瓶颈，而是找到了一个测量口径互相打架的地方。**

### 17.16 矛盾解决了：`cpu_experts` 数的是 **slot**，不是专家（实测读代码，2026-11-03 20:0x）

```cpp
if (!ents.empty() && rp.pad[0] == 0) {   // (staged: the device computes them - no CPU arithmetic)
    const double tc = now_ms();
    cpu_experts(l, &r.x[0][0], ents, (size_t) kMaxDim, po);   // ← 车道确实在这儿算
    ... combine 进 rp.part[t] ...
    st.cpu_ms += now_ms() - tc;
    st.cpu_experts += nslots;            // ← 数的是 slot 数，不是 experts.size()
}
```
`ds4_model.dp.cpp:9104–9122`

两条结论：

1. **`cpu_experts += nslots`**：一次调用处理的是 `ents`（日志里 `entries/call 4.58`），却只把 `nslots`
   加进计数器。所以 `CPU lane x% of the experts` 这个百分比**分子分母口径不同**，它既不是专家占比，
   也不是时间占比，**不能当结论引用**。上一轮那个"1.6% vs 79%"的矛盾到此解释完毕——错的是这个计数器。
2. **那句注释 `(staged: the device computes them - no CPU arithmetic)` 是过期的**：代码在它下面照样调
   `cpu_experts(...)` 做全量 CPU 算术。谁照着注释读，就会以为这几层是 GPU 算的。

### 17.17 于是回到最初的判断，但这次三个口径自洽

| 口径 | 值 | 是否自洽 |
|---|---|---|
| `cpu_ms` 42 ms/token，每次 call 1.01 ms | ≈41 call/token | ✅ |
| 日志 `entries/call 4.58` | ≈190 专家/token（240 里 79%） | ✅ |
| 物理：190 × 8.6 MB ÷ 42 ms | **≈39 GB/s** | ✅ 与实测 38 GB/s 吻合 |
| ~~`CPU lane 1.6% of the experts`~~ | ~~4 个专家~~ | ❌ **计数器口径错，作废** |

**所以车道确实做了约 190 个专家、42 ms，且确实在约 38 GB/s 上。** 上一轮"两个数互相矛盾"的结论
要更正：不是数据矛盾，而是其中一个计数器（`cpu_experts`）数错了对象。

### 17.18 但"少一半核不慢"仍然没被解释

这是唯一还悬着的事实：38 GB/s 若是带宽受限，少核不该没关系；若是算力受限，少核该变慢。
两个都不成立。剩下的可能是 `run_nodes` 里 **主机线程 `pools[0]->help()` 参与的那一份**：
少核时池子变小、主机份额变大，总量不变（主机那一核的吞吐被顶上），于是总时间看起来不变。
**这条是推断，未验证**；验证方法是把 `help()` 关掉（若可配置）再重复三档实验。

下一轮做这一条——它同时能判定"38 GB/s 到底是不是带宽上限"。

### 17.19 `help()` 没有开关，最后一悬案要靠一个小补丁（2026-11-03 20:2x）

```cpp
void help() { if (n_ > 0) work(gen_.load(std::memory_order_acquire)); }   // ds4_pool.hpp:105
...
pools[0]->help();                                                          // ds4_model.dp.cpp:7705，无条件
```

整个仓库没有 `LANE_HELP` / `NO_HELP` 之类的开关，所以"关掉主机 help 再测三档核数"**必须先改代码**。
补丁很小（一行 `if (!getenv("DS4_LANE_NOHELP")) pools[0]->help();`），但要重编 + 三档重启测量。

### 17.20 解码这一页的交接状态（写清楚，便于换会话继续）

**已经确定的（实测）**

| 事实 | 数值 |
|---|---|
| 解码稳态基线 | **16–18 tok/s（55–63 ms/token）**，引擎自报 `predicted_ms` |
| 每 token 阶段表 | cpu-lane wait 27–33 / down+post 12.3 / attn proj 7.9 / routing+shared 7.6 / attn tail 6.6 / experts gate-up 2.5 / sparse attn 1.9 / comp+idx 1.4 |
| 每 token 设备等待 | 39–40 次 ≈ 每层 1 次，`wait_seq`（`12865`/`12887`） |
| 车道工作量 | 42 ms/token、≈41 call、≈190 专家（240 的 79%）、≈9 GB、**≈38 GB/s** |
| 核心数敏感性 | FREE=2/8/16 → 17.6/18.1/17.3 tok/s，**无关** |
| 依赖链 | attention(L) → 路由(L) → 专家(L) → attention(L+1)，**跨层不可重叠** |

**已经排除的（都写过理由）**：把车道挪出关键路径、加线程、改工作项切分、当算力受限、当带宽受限。

**两处真实的代码缺陷（已定位，未修）**

1. `st.cpu_experts += nslots;`（`9121`）—— 数 slot 不数专家，导致 `CPU lane x% of the experts` 不可用；
2. `9104` 那句注释 `(staged: the device computes them - no CPU arithmetic)` 与它下面的代码相反。

**下一步（按顺序，都是小改动）**

1. 加 `DS4_LANE_NOHELP` 开关测三档核数 → 判定 38 GB/s 是带宽还是主机份额；
2. 修上面两处（计数器改成 `ents.size()`、删掉过期注释）；
3. 再决定要不要动车道本体，或转去做 MTP（**MTP 可能是更大的杠杆**：一次吃多 token 摊薄每 token 固定的 42 ms 主机开销；注意记忆里"本机不划算"那条结论是在车道未优化、45 ms/token 的前提下测的，**必须重测**）。
