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
