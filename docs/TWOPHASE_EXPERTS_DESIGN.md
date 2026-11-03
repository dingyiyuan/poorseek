# 两段式专家 GEMM：接进 dsv41 的可执行设计

> 2026-11-02 夜间写。背景与收益估算见 [EXPERT_KERNEL_ALU.md](EXPERT_KERNEL_ALU.md) 的"一点五"节：
> 融合路径每份专家权重被重复解量化 `ceil(rows/BM)` 遍（rows=900 时 **15 遍**），而 oneMKL fp16 GEMM 在
> 引擎自己的形状上是 **94.9 TFLOPS**。按这两点重算，预填充的专家阶段有 **5–9×** 的空间。
> 本文只讲**怎么接**，不讲为什么。

## 一、接在哪

`sycl/src/dsv41/ds4_iq.dp.cpp`：

| 函数 | 行 | 现在 |
|---|---|---|
| `iq::set_gate_up(const SetArgs&, queue*)` | `:2595` | 先 `light_gate_up`（≤16 行的小专家），再按 `(ceil(rows/64), n_ff/64, n_exp)` 建 grid 跑融合内核 |
| `iq::set_down(const SetArgs&, queue*)` | `:2615` | 同上 |

在这两个函数体最前面插一个开关（默认关，行为完全不变）：

```cpp
if (exp_two_phase(a)) return set_gate_up_2ph(a, s);   // gate/up
if (exp_two_phase(a)) return set_down_2ph(a, s);      // down
```

`exp_two_phase(a)` 的判据：`getenv("DS4_EXP_2PHASE")` 为真 **且** `a.max_rows >= 阈值`（默认 96，
因为 rows 小时两段式一定更差——解码绝不能走）**且** 类型在支持集里（IQ2_XXS/IQ2_XS/…，即
`is_iq(a.tg)`）。

### 仓库里能照抄的先例是哪个（查过，别抄错）

- ✅ `prefill/prefill.cpp:4364` —— 另一个引擎（`strata_prefill`）的预填充路径，就是 `iq_dequant_gu_f16` +
  oneMKL GEMM 的两段式，**照它写**。
- ❌ `ds4_iq.dp.cpp` 的 `light_set`（`:1510` 附近）—— 看起来像两段式，但它 `light_q8_kernel` 量化的是**激活**
  （q8），专家权重仍然现解现算。**不是权重两段式的先例**，别照它抄。

## 二、数据布局（从 `set_tc_kernel` 的收尾 `:1387-1396` 读出来的，必须一模一样）

| 角色 | 缓冲区 | 类型 | 行索引 | 步长 | 列 |
|---|---|---|---|---|---|
| gate/up 的激活输入 | `a.xf` | **float** | `a.idx[r0 + r]`（按专家 gather） | `a.ldx` | `n_embd` |
| gate/up 的输出 | `a.h` | **half** | `r0 + r`（本地行号，不是激活行号） | `a.ldo` | `n_ff`，值 = `swiglu(g, u, a.limit)` |
| down 的激活输入 | `a.xh` | **half** | `r0 + r` | `a.ldx` | `n_ff` |
| down 的输出 | `a.y` | **float** | `r0 + r` | `a.ldo` | `n_embd` |
| 专家权重 | `a.wbase + e * a.stride` | IQ2 块 | — | 每行 `row_bytes(T, K)` | gate 在 `a.tr[0]`、up 在 `a.tr[1]`、down 在 `a.tr[2]` 偏移处 |

专家 e 的行范围是 `[a.bounds[e], a.bounds[e+1])`（相对 `a.row0`），`a.skip_le` 以内的专家由 light 路径处理、
这里要跳过。**注意 `r0` 是行块偏移**：两段式一次处理整个专家的所有行，所以 `r0` 取 0、行数取
`hi - lo`，输出仍写到 `a.h[(r0 + i) * a.ldo + ...]` 的对应位置。

## 三、每个专家做什么（伪代码）

```cpp
// 缓冲（按 n_embd=2560, n_ff=1280, rows_max=上限 rows）
half*  dwg  = [n_ff  * n_embd]   // 6.55 MB   gate 解量化结果，行主序
half*  dwu  = [n_ff  * n_embd]   // 6.55 MB   up
half*  dwd  = [n_embd * n_ff]    // 6.55 MB   down
half*  xg   = [rows_max * n_embd]// rows=900 时 4.6 MB   gather 来的 gate/up 激活（f32→half）
half*  xh   = [rows_max * n_ff]  // 2.3 MB   down 的激活（a.xh 已经就是 half）
float* yg   = [rows_max * n_ff]  // 4.6 MB
float* yu   = [rows_max * n_ff]  // 4.6 MB
float* yd   = [rows_max * n_embd]// 9.2 MB
// 合计约 45 MB（rows_max=900）；32 GB 的卡上可以忽略，且**每专家复用同一组缓冲**

for (int e = 0; e < a.n_exp; ++e) {
    const int lo = a.bounds[e] - a.row0, hi = a.bounds[e+1] - a.row0;
    if (hi - lo <= a.skip_le) continue;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;

    // ① 解量化（每专家只有这一次——这就是 15x → 1x 的来源）
    iq_dequant_f16(a.tg, blob + a.tr[0], n_ff * n_embd, dwg, s);
    iq_dequant_f16(a.tg, blob + a.tr[1], n_ff * n_embd, dwu, s);
    // ② gather 激活：xg[i] = half(a.xf[a.idx[lo + i] * a.ldx + k])
    gather_x_f16(a.xf, a.idx + lo, hi - lo, n_embd, a.ldx, xg, s);
    // ③ 两个 GEMM：C[rows × n_ff] = A[rows × n_embd] · W[n_ff × n_embd]ᵀ
    gemm_half_f32(xg, dwg, yd_rows, n_ff, n_embd, yg, s);
    gemm_half_f32(xg, dwu, yd_rows, n_ff, n_embd, yu, s);
    // ④ swiglu → a.h（half，本地行号，步长 a.ldo）
    swiglu_to_h(yg, yu, a.limit, a.h + (size_t)(lo) * a.ldo, n_ff, hi - lo, a.ldo, s);
}

// down 同理，但激活直接用 a.xh（按 lo..hi 的本地行），输出 f32 到 a.y
for (int e = 0; e < a.n_exp; ++e) {
    ...
    iq_dequant_f16(a.td, blob + a.tr[2], n_embd * n_ff, dwd, s);
    gemm_half_f32(a.xh + (size_t) lo * a.ldx, dwd, hi - lo, n_embd, n_ff, yd, s);
    copy_f32(yd, a.y + (size_t) lo * a.ldo, n_embd, hi - lo, a.ldo, s);
}
```

`gemm_half_f32` 就是引擎里已经在用的那个 oneMKL 形状（`ds4_model.dp.cpp` 的 dense 路径与
`xmx_gemm_bench` 同一套写法）：

```cpp
dpct::blas::gemm(h, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
                 /*m=*/n_out, /*n=*/rows, /*k=*/K, &alpha, W /*half, lda=K*/,
                 dpct::library_data_t::real_half, K, X /*half, ldb=K*/,
                 dpct::library_data_t::real_half, K, &beta, Y /*float, ldc=n_out*/,
                 dpct::library_data_t::real_float, n_out, dpct::compute_type::f32);
```

（`ds4_model.dp.cpp` 已经 `#include <dpct/blas_utils.hpp>`，target 也已有 `-qmkl=sequential`。）

## 四、必须注意的四件事

1. **输出 dtype 是 f16**：`a.h` 是 half，`a.y` 是 f32。`swiglu` 的 `limit` 语义要照抄
   `ds4_iq.dp.cpp:1387-1396` 的收尾（`swiglu(O.g, O.u, a.limit)` 再 `convert<half, rte>`）。
2. **激活 gather 用的是 `a.idx`**：gate/up 的行来自 `a.idx[r0+r]`，down 的行直接用本地行号。写反了结果会是
   乱码而不是崩溃——这是最容易错的一处。
3. **数值不等于位级**：f16 权重 + f32 累加，与融合路径的 f32 全程相比会有 ~1e-3 相对差。验收用
   `ds4_iq_test` 同款容差（`set gate_up` 的 tol 是 3e-3），**不要指望 max err = 0**。
4. **rows 门槛**：解码（rows=1）时两段式必然更差（每 token 解一次 20 MB 权重），阈值默认 96，
   用 `DS4_EXP_2PHASE_MINROWS` 可调。

## 五、怎么验（明早，按成本从低到高）

1. **先看基准**：`morning.sh` 第 2 步的 `xmx_gemm_bench ... sweep real` 已经给出 M=1…1800 的
   `2ph/fus` 比值，以及第 3 步 `--bench-tiles` 给的**真内核**吞吐。**两个都指向 ≥2 再动手写代码**。
2. **内核级**：`ds4_iq_test_tc`（含 `iq::set_gate_up/set_down` 的对拍）在 `DS4_EXP_2PHASE=1` 下跑一遍，
   看 `set gate_up` / `set down` 的 `worst` 是否仍在 3e-3 内。
3. **端到端**：`tc_ab.sh` 同款做法，但环境变量换成 `DS4_EXP_2PHASE=1`，比较 45k 预填充 tok/s。

## 六、风险

- **一次处理整个专家**意味着要 gather 最多 900 行的激活（4.6 MB），若行数无上限需要设 `rows_max` 上限并在
  超过时回退到融合路径（安全阀）。
- **显存**：45 MB 的 scratch 与专家流式缓存并存；`--vram-experts 1.0` 把显存吃满时要留出这 38 MB，
  否则会触发 OOM——**先减 `--vram-experts`** 再试。
- **融合路径保留**：开关默认关，`DS4_EXP_2PHASE=0` 或行数不足时一行代码都没被替换，随时可回退。

## 七、代码已经写好了（2026-11-02 23:5x，**编译通过，未运行**）

不是设计稿了——`ds4_iq.dp.cpp` 里已经实现：

| 位置 | 内容 |
|---|---|
| `ds4_iq.dp.cpp:2600-2615` | `exp2p_on()` / `exp2p_min_rows()`：`DS4_EXP_2PHASE=1` 打开，`DS4_EXP_2PHASE_MINROWS` 默认 96 |
| `:2617-2730` | `KGatherF16` / `KSwiglu` / `KCopyF32` 三个小 kernel、`TwoPhaseBuf`（每 device+queue 一份 45 MB scratch）、`exp2p_gemm`（oneMKL f16/f32）、`set_gate_up_2ph`、`set_down_2ph` |
| `:2735`、`:2758` | `set_gate_up` / `set_down` 最前面一行开关；不满足条件就原样走融合路径 |

**宏隔离**：整段代码在 `#if defined(DS4_HAVE_MKL_GEMM)` 里，该宏只给两个目标定义——
`strata-dsv41`（本来就有 `-qmkl=sequential`）和 `ds4_iq_test_tc`（为它专门加了 `-qmkl` 并链 `strata_kernels`）。
其它任何编 `ds4_iq.dp.cpp` 的目标（各种 parity/bench）都走 `#else` 分支，**不需要 MKL，行为不变**。

**默认关**：`exp2p_on()` 读不到环境变量就是 false，生产行为与今晚之前完全一致。

### 明早怎么验（两条命令）

```bash
# ① 内核级对拍：数值对不对（几秒钟，不启引擎）
#    **必须带 MINROWS=1**：对拍集合只有 37 行，默认门槛 96 会让两段式静默回退，
#    数字和融合路径逐位相同，看起来"通过"其实一次都没跑。
STRATA_DS4_NO_TC=1 DS4_EXP_2PHASE=1 DS4_EXP_2PHASE_MINROWS=1 build-sycl-tc2/ds4_iq_test_tc
#    看 "set gate_up" / "set down" 的 worst 是否仍在 tol（3e-3）内；
#    与不加 DS4_EXP_2PHASE 的那次比，worst 不应明显变差。

# ② 端到端：值不值（走服务，和 tc_ab.sh 同一套）
DS4_EXP_2PHASE=1 tools/overnight/tc_ab.sh bench/prompts/p45k.txt 48   # 三组里手动在配置 env 加这一项
```

**第一次跑会 malloc 45 MB scratch**；如果显存本来就吃满（`--vram-experts 1.0`），先减到 0.9 再试。

### 自查发现的第一个 bug（Round 13）

`strata::kernels::iq_dequant_f16` 对**非 IQ 类型**会 `std::exit(1)`（见 `src/kernels/cuda/iq_kernels.dp.cpp:4849`）。
我第一版只挡了 `n_embd % 256`、没挡类型——如果哪天用 `DS4_EXP_2PHASE=1` 跑一个含 MXFP4/IQ4 专家的包，
进程会直接退出。已加 `supported(a.tg)` / `supported(a.td)` 守卫（`ds4_iq.dp.cpp:2693/2715`），类型不认识就回退融合路径。

**这条记在这里的原因**：开关类代码最怕的不是慢，是把进程带走。以后往热路径加任何 env 开关，
第一件事都是"不满足条件必须干净地 return false"，而不是靠调用方保证。
