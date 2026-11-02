# CED 预填充移植设计（抄 SGLang / vLLM）

> 2026-11-02。全程未碰 GPU。参考实现已下载到 `/tmp/ds41refs/`。

## 一、SGLang 是怎么做的（带出处）

### 1. 配置层：用三个字段表达 CED
`python/sglang/srt/configs/deepseek_v41.py:44-50`
```python
kv_source_layer_ids = ()        # 哪些层的 KV 取自"源层"而不是自算
index_source_layer_ids = ()     # 索引器同样
candidate_source_layer_id = -1  # indexer 的候选池也来自源层
```
→ **CED 不是特例代码，是配置驱动的**：把 L20–L39 放进 `kv_source_layer_ids`，它们就去取源层的 KV。

### 2. 安装条件（硬约束）
`python/sglang/srt/models/deepseek_v4.py:3866-3905`
```python
if get_exec().features.enable_decoder_swa_bounded_replay:
    last_source = max(config.kv_source_layer_ids)
    self.tail_writes_source_kv = True
    self.late_layer_start = last_source if self.tail_writes_source_kv else last_source + 1
    assert self.layers[last_source].engram is None          # 源层不能带 Engram
    assert last_source in config.index_source_layer_ids      # 源层必须同时是 index 源
    assert config.candidate_source_layer_id in (-1, last_source)
    late_ratios = set(config.compress_ratios[last_source+1:num_hidden_layers])
    assert late_ratios <= {0, 1}                             # 后 20 层不得自己压缩
    assert config.num_hash_layers <= self.late_layer_start    # hash 路由必须在 late 层之前
```

### 3. tail 层的跑法（关键）
`python/sglang/srt/models/deepseek_v4_replay_tail.py:161-176`
```python
return cls(model=model,
           layer_ids=range(model.late_layer_start, model.end_layer),   # 就是从 late_layer_start 到 39
           row_step=SWA_WINDOW,                                        # 每桶 128 行
           max_rows=envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.get())
```
`SWA_WINDOW` 定义在 `layers/attention/deepseek_v4_backend.py`。文档字符串（`:1-10`）说得很直白：
> "DeepSeek-V4.1's **decoder-replay tail layers**, run from row-bucket graphs. … 一个 prefill-graph 形状
> 至少 `SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS` 个 token 时它们从一次 eager break 重放。"

### 4. KV 写法
`deepseek_v4_backend.py` 的 `write_tail_source_kv()`（由 `deepseek_v4.py:931-936` 包装）
→ **最后一个 kv_source 层先为"每一行"写入压缩 KV**（`tail_writes_source_kv = True`），
  tail 层再只对最后 `SWA_WINDOW` 行做完整计算。

### 5. 完整语义（拼起来）
```
预填充（长度 N）:
  L0..L(last_source-1):  全部 N 个 token 正常算（attention+MoE+KV）
  L_last_source:          全部 N 个 token 正常算，并把压缩 KV 写给"所有行"  ← 这是源层的双重职责
  L(late_layer_start..39): 只对最后 SWA_WINDOW(=128) 个 token 算
解码: 照常（每步 1 token，回看最多 SWA_WINDOW 个位置，见 deepseek_v4.py:4100）
```
→ **预填充省掉的是"后 20 层 × (N-128) 个 token"的全部 attention+MoE**，N 越大省得越多。

## 二、移植到 Strata dsv41 的改法

### 需要新增的东西
| 项 | 位置 | 说明 |
|---|---|---|
| `ced_kv_source_layers` 配置 | `ds4_model.hpp` 的 `Opt` | 从 pack 的 `ds4.txt` 读；缺省空=关闭 |
| `late_layer_start` | 同上 | `= max(ced_kv_source_layers)`，与 SGLang 一致 |
| tail 行选择 | `forward()` 的 chunk 路径 | 后 20 层的循环上界从 `T` 改成 `min(T, 128)`，且只对末尾 128 行 |
| 源层"写全行 KV" | 现有 `ext_to_ring` 调用处 | 源层（L19）照常写全部 N 行（现在就是这么做的，无需改） |
| 后 20 层的 KV 来源 | `layer()` 的 attention 段 | 跳过它们的 `attn_kv` 计算？**不**——SGLang 的做法是 tail 层**只对 128 行**算 KV，其余行**没有自己的 KV**（解码时回看窗口内即可）。 |

### 关键判断题（必须先确认）
SGLang 的 `write_tail_source_kv` 表明：**非 tail 行在后 20 层是没有 KV 的**，它们靠
"源层写全行的压缩 KV" + "解码只看 SWA 窗口" 来保证正确性。所以我们**不需要**给后 20 层补投影，
只需要：
1. L0–L19 全行正常；
2. L20–L39 只处理**最后 128 行**；
3. 解码时的 SWA 窗口回看不超过 128（我们本来就是稀疏注意力 + 窗口）。

### 改动清单（按顺序）
1. `ds4_model.dp.cpp` 的 chunk 路径（`run_set` 附近的层循环）：
   加 `const int tail_T = (l.il >= late_layer_start) ? std::min(T, 128) : T;`
   并把该层的 token 范围改成 `[T - tail_T, T)`（只算末尾 128 行）。
2. 该层写 KV / ext_to_ring 的循环同步改成只写末尾 128 行。
3. indexer / candidate 池：源层（L19）已经是全行写，后 20 层复用其候选（`candidate_source_layer_id` 的语义）。
4. 加 `DS4_CED=1` 开关，默认关。
5. **正确性门槛**：用同一 prompt 对比 `DS4_CED=0/1` 的**前 8 个 token 逐 token 一致**（贪心解码）。

### 风险
- **正确性**：如果模型训练时后 20 层确实只需要 128 行窗口，这个改动等价；如果不是，输出会崩。
  验证方式就是上一条的逐 token 对比——**这是唯一可靠的门槛**。
- 我们引擎的稀疏注意力窗口是否也是 128？需先确认 `sparse attn` 的窗口参数（`g.win`?）。
- chunk 边界：128 行要跨 chunk 时，前一 chunk 的尾部 128 行也要算——这是最容易错的地方。

## 三、预期收益（按我们的实测推算）
122k 预填充现在 270.7 tok/s（448 s），其中：
- 专家内核 223 s（50%）
- q_b+rope 39.5 s、压缩器 77.9 s、稀疏注意力 47.2 s

若后 20 层的 token 数从 122k 降到 128（省 99.9%），**理论上这些成本里属于后 20 层的部分几乎全部消失**。
按层均摊粗算：后 20 层 ≈ 一半 → 448 s → **~230 s → ~530 tok/s**。
这正好落在你说的"500 才舒服"。

## 四、交叉验证（vLLM 侧）
`vllm/models/deepseek_v41/sparse_mla.py`（337 行，已下载）——明天补读，确认 vLLM 是否也有
等价的"late layers 只算 tail"实现，以及 SWA 窗口常量的定义。

---

## 五、**阶段矩阵**（预填充/解码必须分开，且解码绝不减层）

用户点名要防的坑：**别把"20 层"套到解码上**。正确语义是"预填充的**填充部分**只走 20 层"，
不是"整个引擎只跑 20 层"。

| | 预填充 · 非尾部行（prompt 的前 N−128 个 token） | 预填充 · 尾部行（最后 128 个 token） | **解码（每步 1 token）** |
|---|---|---|---|
| L0–L19（编码器） | ✅ 全算（attn+MoE+写 KV） | ✅ 全算 | ✅ **全算** |
| L20–L39（解码器） | ❌ **不算**（无 KV，无 attention，无 MoE） | ✅ 全算（只在这 128 行内做 SWA） | ✅ **全算**（写自己的 KV，回看 ≤128） |
| 层数 | 20 | 40 | **40（一个不少）** |
| attn slab（`DS4_ATTN_SLAB`） | 2048（GEMM 批量大） | 2048 | 1 行就够，**应缩到最小** |
| 专家环（`--vram-experts`） | 尽量大 | 尽量大 | 尽量大（slab 缩小后能更大） |

### 必须守住的三条不变量
1. **解码永远跑全部 40 层**。任何"L20–L39 在解码阶段被跳过"的实现都是 bug。
2. **L20–L39 的 KV 只存在最后 128 个位置**。解码对这两组层回看**不得超过 128**——
   超过就会读到不存在的 KV（未定义行为）。我们的稀疏注意力窗口必须 ≤128 才安全。
3. **chunk 边界**：如果 prompt 被切成多个 chunk，只有**最后一个 chunk 的末尾 128 行**是"尾部行"；
   前面所有 chunk 的全部行都算"非尾部行"（只走 20 层）。最容易写错的就是这里。

### 两条分开的配置（`DS4_CED=1` 时自动生效）
| 阶段 | 该用什么 |
|---|---|
| 预填充 | slab=2048、CED=开（L20–39 只算 128 行） |
| 解码 | slab=base（1 行级别）、CED 不生效（40 层全跑） |

**实现方式**：`slab_resize()` 在阶段切换时调用（见本文档 §B 的分段设计）；
`DS4_CED` 只在 chunk 路径的层循环里生效（decode 的 `forward(T=1)` 根本走不到那段代码），
所以"解码误减层"在结构上不可能发生——这也是把 CED 只做在 chunk 路径的原因。

---

## 六、**更正：我们包里已经有 CED 元数据，机制也已在引擎里**

`/data/models/dsv41-iq2xxs-pack-real/ds4.txt` 里本来就有这些键（**不是缺的**）：

```
sliding_window            128                  ← 与 SGLang 的 SWA_WINDOW 完全一致
kv_source_layer_ids       2,8,14,20            ← KV 源层（每 6 层一个源）
index_source_layer_ids    2,8,14,20,24,28,32,36
candidate_source_layer_id 20
candidate_topk_blocks     2048
candidate_block_size      8
compress_ratios           0,0,2,2,...,2, 1,1,...,1, 0,0,0   ← 第 20 层是 2→1 的分界
```

引擎对应实现（`ds4_model.dp.cpp:2760-2780`，`Layer` 结构 805 行）：
```cpp
bool kv_src, ix_src, owns_k, cand_src, uses_cand;
int  ik_src;        // 读哪个源的 index keys
l.kv_src  = il ∈ g.kv_src;      l.owns_k = l.kv_src;
l.ix_src  = il ∈ g.ix_src;      l.cand_src = (il == g.cand_src);
l.uses_cand = (g.cand_src < il) && l.ix_src;
l.ik_src = last_src;            // 最近的前一个源
```
→ **SGLang 的那三个 config 字段，我们已经原样实现了**，而且已经在跑。
（这也解释了为什么之前 `DS4_PREFILL_LAYERS` 探针没效果：KV 共享本来就在，缺的是"行裁剪"。）

### 为什么"后 20 层只算 128 行"在数学上正确
预填充只需要**最后一个 token** 的最终隐藏态来生成第一个输出 token。
前面 N−1 行经过 L20–L39 得到的隐藏态**没有任何地方会读**（它们既不是输出，也不再被后续层用作输入的另一部分）；
这些行在 L20–L39 唯一有价值的产物是 **KV**，而 L20–L39 的注意力只在 `sliding_window=128` 内回看。
→ **所以 N−128 行在 L20–L39 上的 attention+MoE 是纯浪费**，这正是 CED 省掉的那一半。

### 最终改动（比 §二 更精确）
| 改什么 | 位置 | 怎么改 |
|---|---|---|
| late 层起点 | `ds4_model.dp.cpp` | `late = max(g.kv_src)`（本包 = 20） |
| 行裁剪 | chunk 路径的层循环 | `if (il >= late) { 只处理末尾 min(T, 128) 行 }` |
| 源层 | 不变 | L20 本来就把全行 KV 写给共享者（`ext_to_ring` 现有逻辑） |
| 解码 | **一行都不用改** | 解码走 `T=1` 路径，永远 40 层 |
| 开关 | 新增 `DS4_CED=1` | 默认关，A/B 用 |

### 风险（比之前估计小得多）
1. **尾部行要跨 chunk**：只有**最后一个 chunk 的末尾 128 行**算 tail；前面 chunk 的所有行都是 non-tail。实现时用"全局最后 128 个位置"判断，而不是"本 chunk 的最后 128 行"。
2. **本地窗口 vs 稀疏**：late 层如果仍走全局 indexer，就需要 128 窗口外的 KV → SGLang 用 `index_source_layer_ids` 让 late 层复用源层的 index，我们已有同样的 `l.ik_src` 机制 → 理论上已一致，但要实测验证。
3. **正确性门槛**：同一 prompt 下 `DS4_CED=0/1` 的**前 8 个 token 必须逐 token 一致**（贪心）。这是唯一可靠验收。

### 预期收益
122k 预填充现在 270.7 tok/s（448 s）。L20–L39 承担了约一半的层数 → 若这 20 层只算 128 行，
其 attention+MoE 成本几乎归零。粗算 **448 s → ~240 s ≈ 500 tok/s**，正好是"500 才舒服"。

---

## 七、权威算法（vLLM `decoder_replay_layers.py`，逐行印证）

vLLM 的 docstring（`:1-10`）把机制一句话说完了：
> "Once the last KV-source layer has **written every row's KV**, it and the layers past it
> (which **own only SWA KV**) **run on each request's last `window` rows only**."

代码骨架（`vllm/models/deepseek_v41/decoder_replay_layers.py`）：
```python
def __call__(self, hidden_states, *states):
    if self.uses_replay_batch():
        self.write_batch_kv(hidden_states, *states)      # :106  源层：写"每一行"的 KV
        ...
    return self._run(hidden_states, *states)

def _run(self, hidden_states, *states):
    rows = replay_batch.rows                            # :131  本请求最后 window 行
    row_inputs = [t.index_select(0, rows) for t in inputs]   # :137  只取这些行
    row_outputs = self.run_layers(*row_inputs)               # :146  后 20 层只算这些行
    # :147 "The trimmed rows' outputs stay zero; nothing reads them."
    return [out.new_zeros(num_tokens).index_copy_(0, rows, out) ...]
```
**三处关键确认**：
1. 源层必须**先为所有行**写 KV（`write_batch_kv`，在任何裁剪之前）；
2. 后续层只跑 `window` 行（`window` = SWA 窗口 = 128）；
3. **被裁掉的行输出置零、无人读取** —— 这是"跳过"合法性的直接证据。

`vllm/models/deepseek_v41/sparse_mla.py:34` 还给了 ratio 语义：
> "v4.1 per-layer compress ratios: **0 = pure sliding window, 1 = full-length**"
与包里 `compress_ratios` 的 `0,0,2,...,2, 1,1,...,1, 0,0,0` 完全对应（第 20 层是 2→1 的分界）。

## 八、给 Strata 的代码级伪码（可直接照抄）

```cpp
// ds4_model.dp.cpp，chunk 路径（forward() 里 chunk 的层循环）
const int late = *std::max_element(g.kv_src.begin(), g.kv_src.end());   // 本包 = 20
const int W    = g.sliding_window;                                      // 本包 = 128

for (int il = 0; il < g.nl; ++il) {
    Layer& l = L[il];
    // 行范围：late 层只跑"全局最后 W 行"
    int r0 = 0, r1 = T;
    if (ced_enabled && il > late) {
        const int64_t tail_start = std::max<int64_t>(p0, (int64_t) ctx_len - W);
        r0 = (int) std::max<int64_t>(0, tail_start - p0);   // 本 chunk 内属于最后 W 行的起点
        r1 = T;
        if (r0 >= r1) continue;                              // 本 chunk 全在窗口外：整层跳过
    }
    layer(l, r0, r1);     // ← 现有 layer 调用，把行范围传进去（含 attention/MoE/KV 写）
}
```
**必须配套的三处**：
1. `l.kv_src` 层（2/8/14/20）**永远用 r0=0, r1=T**（源层写全行 KV）；
2. KV 写（`ext_to_ring` / compressor / indexer）也用 `[r0, r1)`，否则会写越界或写错位置；
3. **跨 chunk 的 tail 判定用全局位置**（`p0` 与 `ctx_len`），不是"本 chunk 的最后 128 行"。

## 九、验收（明天 GPU 恢复后的第一步）
1. `DS4_CED=0` 跑一次 512 token 的 prompt，记录贪心解码的前 8 个 token；
2. `DS4_CED=1` 同样跑，**前 8 个 token 必须逐字节相同**；不同就是实现有错，立即回滚；
3. 一致后再量 45k / 122k 预填充，对比 347 / 270.7 tok/s。

---

## 十、**最终解释：DeepInfer 的 YOCO 预填充**（/home/dyy/deepseek-work/llm-for-glm-flash/DeepInfer）

DeepSeek 自己的框架里，这套机制的正式名叫 **YOCO（You Only Cache Once）prefill truncation**。

### 配置映射（`deepinfer/hugging_face/config.py`）
```python
def _source_map(source_ids, n_layer):
    """Map each layer to the nearest source layer that is not later than that layer."""
    ids = sorted(source_ids)
    return {layer: max(s for s in ids if s <= layer) for layer in range(ids[0], n_layer)}

shared.layer_kv_map          = _source_map(text["kv_source_layer_ids"], n_layer)
shared.layer_index_module_map= _source_map(text["index_source_layer_ids"], n_layer)
shared.layer_index_k_map     = _source_map(text["index_k_source_layer_ids"] or text["kv_source_layer_ids"], n_layer)
shared.layer_candidate_source_map = {layer: candidate_source for layer in range(candidate_source, n_layer)}
```
→ **与我们引擎的 `l.kv_src / l.ix_src / l.ik_src / l.cand_src` 一一对应**，连"取最近的前一个源"的语义都相同。

### 三个旋钮（`deepinfer/envs` + `worker/cache_engine.py:93,98`）
```python
DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE ∈ {"off", "mask", "compact"}
DEEPINFER_YOCO_PREFILL_KEEP_TOKENS          # 预填充阶段保留的尾部行数
DEEPINFER_MAX_BACKTRACKING_TOKENS           # 投机回退需要多看的行数

# 需要的长度（按模型 window 算）
window_size + MAX_BACKTRACKING_TOKENS
# 硬断言（model_runner.py:110）
assert window_size + MAX_BACKTRACKING_TOKENS <= YOCO_PREFILL_KEEP_TOKENS
```

### 两种实现模式（`model_runner.py:175-178`）
| mode | 做法 | 产出 |
|---|---|---|
| `off` | 不截断 | full 布局 |
| `mask` | 保持 full 布局，用掩码把非尾部行遮掉 | `yoco_mask_swa_index_pivots` |
| `compact` | **真正压紧**成只含尾部行的紧凑批次 | `yoco_consumer_*` |

### **最关键的硬规矩（model_runner.py:240-243 注释原文）**
> "compact 模式按截断后的 output size 计算 prefill sample 位置；mask/off 保持 full 布局。
> **decode 恒走 full 布局，传 off。**"

```python
yoco_prefill_truncation_mode = YOCO_PREFILL_TRUNCATION_MODES[
    self.yoco_prefill_truncation_mode if input_metadata.is_prompt else "off"]     # ← 解码强制 off
```

**这就是"解码绝不减层"的官方做法**：不是靠调用方自觉，而是**在代码里按 `is_prompt` 分支强制 `off`**。
我们的移植必须照抄这一条：`DS4_CED` 只在 chunk（`is_prompt`）路径生效，解码无条件 40 层。

### 我们的对应参数
| DeepInfer | 我们 |
|---|---|
| `window_size` | `sliding_window = 128`（包元数据） |
| `YOCO_PREFILL_KEEP_TOKENS` | 取 `128 + max_backtracking`；我们 MTP 的 block_size=5 → **133**（保守取 160） |
| `MAX_BACKTRACKING_TOKENS` | DSpark 的 `dspark_block_size`（本包 5） |
| `truncation mode` | 建议先做 `compact`（真正省算力）；`mask` 只省采样位置，不省算力 |

### 落地建议（明天按这个顺序）
1. 先实现 **`compact` 模式**：chunk 路径里 `il > late` 时，把行范围压到**全局最后 KEEP 行**（`KEEP = 160`）；
2. `late = max(kv_source_layer_ids) = 20`；
3. 源层（L20）保持全行写 KV；
4. **解码路径完全不动**（`is_prompt=false` 等价物就是我们的 `T==1` 分支）；
5. 验收：`DS4_CED=0/1` 前 8 token 逐字节一致。
