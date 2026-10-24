# v41-dspark-audit — why our offline DSpark harness measured 0.4 when production gets ~5

## Verdict (measured)
**It was not a feeding-convention bug: it was a weight-loading bug.**  `tools/v41/dspark_accept.py` drove the DSpark
stack exactly as SGLang/vLLM/DeepSeek's `model.py` do (re-verified below).  What was wrong: the checkpoint keeps
`attn.wo_a` as FP8 (E4M3, E8M0 **32x32** block scales) while `model.py` builds that module in **bf16** (its comment:
"convert.py dequantizes it to bf16"), so `load()` cast the raw fp8 codes to bf16 and **dropped the block scale** —
`wo_a` came out ~1020x (512..2050x) too large in all 3 blocks, so every draft was garbage.  `v41_calib.py:341-346`
already dequantizes wo_a "like convert.py" and `v41_gguf.py:232` writes it as Q8_0, so the engine and teacher were
always right; the harness was the only broken reader.  Fix = one block in `load()` (default on, `--no-fp8-dequant`
restores the old behaviour).  Probe: those 3 tensors are the **only** fp8-with-scale-on-a-non-fp8-parameter.

## The production convention (a-d), verified against three independent implementations
**(a) main_hidden** = uniform **HC-mean** (`mean over the 4 hc copies`) of the residual stream **ENTERING** each
target layer, concatenated in order 37‖38‖39 (+ engram already added upstream), then `main_norm(main_proj(h))`
(Linear 15360->5120).  Not the layer output, not the `hc_pre` collapse.
(model.py:1264-1266 `if i in target_layer_ids: main_hiddens.append(h.mean(dim=2))` *before* `layer(h,...)`, :1271, :1130 ·
vLLM `~/AI/vllm-v0.30.0/vllm/models/deepseek_v41/nvidia/model.py:716-722,728-729,734-738` + `.../eagle/eagle3_utils.py:113-120`
"v4.1 reads the attention *inputs* of its target layers ... entry stream of layer L ... ids used as-is" + the mhc kernel
doc `model_executor/kernels/mhc/tilelang.py:312-314` "the mean over the post-mapped streams" ·
ds4-ref `~/AI/ds4-ref/ds4.c:18280-18285` `mean_rows[i]=inv_hc` + `:29946` hc_weighted_sum.)
**(b) block input**: N = block_size = 5 slots; slot 0 = the **bonus** token (the trunk's freshly sampled token at
p+1), slots 1..4 = noise id 128799 row; slot 0 sits at position p+1; **every** slot is a prediction with
`sample_pos = query_pos + 1`, so the 5 drafts cover p+2..p+6; the last accepted token is not a slot.
(model.py:1131-1133,1146-1153 · vLLM `dflash/speculator.py:8-17,178-180` + the prepare kernel `:588-592,624-634`
(`query_pos = last_valid_pos+1+off`, `input_id = bonus if off==0 else noise`, `sample_pos = query_pos+1`) ·
`config/speculative.py:1429-1441` `n_predict = dspark_block_size` · ds4-ref ds4.c:34922-34935.)
**(c) window KV**: each draft layer's **own** `kv_norm(wkv(main_x))` + RoPE (+fp8 quant) — not the trunk's KV — one
row per **accepted** position (rejected suffixes never stored), the window being the last 128 positions **ending at
p**, block-anchored; attention is **non-causal**: every query attends the window **plus all 5 block slots**.
(model.py:1040-1042,1064-1067 + `get_dspark_topk_idxs` :1021-1029 · vLLM `dspark.py:156-186` +
`mla/sparse_swa.py:1174-1178,1203-1227` (`start_pos = max(prefix_len - window_size, 0)`); index width
`window + num_spec` = `compressor_utils.py:21-27`.)
**(d) heads/rest**: `markov_w1`=Embedding(vocab,256) of the *previous* token (anchor first), `markov_w2`=256->vocab
**logit bias** added per slot left-to-right; confidence = `sigmoid(proj(cat(pre-norm head hidden, markov_embed)))`,
an acceptance estimate used only to schedule verification.  MoE 128 experts top-3 + 1 shared.  V4.1 = 3 blocks, block
5, targets 37-39, draft head = the trunk's `head.weight`, full vocab (no d2t).
(model.py:1077-1097,1128-1156 · qwen3_dspark.py:48-93,164-189 · dspark.py:132-143,363-385.)

## Diff vs our harness
(a)-(d): **no divergence** — the Python clone of `model.py`'s `__main__` loop is a faithful driver, and the ring
seeding by the 128 prompt-tail steps is equivalent to `model.py`'s `start_pos==0` prefill branch.  The single
difference was in `load()`: it applied no E8M0 block scale to a checkpoint-fp8 tensor whose module parameter is not
fp8.  Only `wo_a` is in that class (wq_a/wq_b/wkv/main_proj/shared-experts carry their own `scale` parameter, so
`fp8_gemm` applied it correctly).  Guarded by an assert (blocks 32/128, like convert.py).

## Numbers (block 5, greedy, mean accepted draft tokens per decode step / tokens per verify pass)
| harness | trunk states | mean acc | tok/pass | first draft == truth | first draft == teacher top-1 | conf Q1..Q4 |
|---|---|---|---|---|---|---|
| old (scale dropped) | FP8 teacher | 0.437 | 1.44 | 34.1% (dist 0:66%) | — | 13/12/8/11 (flat) |
| old (scale dropped) | ours (exact routing) | 0.405 | 1.41 | 32.3% (0:68%) | — | 13/10/9/11 (flat) |
| **fixed** (96 steps) | FP8 teacher | **2.583** | **3.58** | 77.1% | **78.1%** | **12/40/78/99** |

(Full 378-step runs on teacher AND our states: NOT run yet - first job next session, see V41_START_HERE.md §5.)
Fixed distribution (teacher, 96 steps): accepted length 0:22.9% 1:17.7% 2:10.4% 3:8.3% 4:8.3% **5:32.3%**.  Trunk vs
teacher agrees 88.5%, so the low old number was the draft, not the trunk: at the same states the first draft went
from 30.2% (old, matched the teacher only where the trunk did) to 78.1%.  This now brackets the reported production
range (llama.cpp V4 3.4 tok/pass; SGLang ~5 accept length).

## Recommendation for the engine implementation
Proceed with §6 1(c) (DSpark draft + verify) — the acceptance is real, and the confidence head is now usable for
per-request draft depth.  No engine change is needed for the fix: the engine reads the GGUF, whose wo_a is already
Q8_0-dequantized.  Re-run `tools/v41/dspark_accept.py` (now default-correct) before sizing anything.

## Residuals (not done, in scope order)
- The pre-mix / leaving-layer main_hidden variants were **not measured**; they are ruled out from code (vLLM's
  explicit v4.1 entry-stream comment, the mhc kernel's "mean over the post-mapped streams", ds4-ref's `inv_hc`).
  `pre/L*.npy` (the hc_pre coefficients) exists if a future run wants to test the collapse instead of the mean.
- Teacher-vs-ours cosine at layers 37-39 stays 0.876; the fixed harness now separates that drift from the drafter.
- `--max 96` numbers above share the same 128-step window warmup, so they are directly comparable to the full runs.
