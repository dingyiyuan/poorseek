# DeepSeek-V4.1-Flash in Strata-DS4 — engine port plan (2026-10-10)

Goal: run the `deepseek41` GGUF that `tools/v41/v41_gguf.py` writes (mixed-tier routed experts, Q8_0 dense) in the
Strata-DS4 engine, with the V4 engine's speed machinery (VRAM expert LRU, RAM arena, predicted prefetch, soft-prune,
arena-adapt, chunked prefill, MTP/DSpark later).  Plan B of [V41_PLAN.md](V41_PLAN.md), broken into gated phases.

References, in order of authority:
1. **DeepSeek's `inference/model.py`** (`~/AI/gsq-rco/dsv41-inference`, run through `tools/v41/v41_calib.py`'s
   layer streaming + `torch_kernels/kernel.py`): the math.  Golden numbers come from here (§P2).
2. **DwarfStar** (`~/AI/ds4-ref` @fc80bd6, MIT): a working V4.1 engine in C + hand-written CUDA (`ds4.c` `ds41_*`,
   `ds4_engram.c`, `ds4_deepseek41_cuda.cuh`).  Data flow, buffer shapes, the Engram reader, prefill carry layout.
   Not a ggml engine: we take ideas and checks, not code structure.
3. Our V4 engine (`tools/ds4/ds4_dense.*`, `ds4_moe.*`, `ds4_generate.cpp`, docs ENGINE_DENSE.md / ENGINE_MOE.md /
   DS4_ENGINE.md): the structure we extend - ggml graphs on the CUDA backend, fixed-shape graph variants, the
   expert tier and its caches.

Rules: [../UPSTREAM_SYNC.md](../UPSTREAM_SYNC.md) (fork code isolated; upstream files touched by one-line hooks).
Every phase ends in a gate that is a measured number against the reference, recorded in this file.

## What is different from V4 (what the port actually has to build)

| area | V4 engine today | V4.1 |
|---|---|---|
| shape | 43 layers, D 4096, 256 experts, n_ff 2048, q_lora 1024 | 40 layers, D 5120, 384 experts top-6, n_ff 2304, q_lora 1280 |
| q path | per-head q RMS norm | **no q norm** (JigSawPT's bug: corr 0.78 at L39 with it) |
| hyper-connections | pre-mix per sublayer | **single pass**: a sublayer's `hc_mixes` produce the NEXT sublayer's pre-mix (attention uses the previous FFN's) |
| compressors | ratio 4 (CSA) / 128 (HCA), every layer its own | ratio 0 / 2 / 1 per layer; compressor only on **kv source layers [2,8,14,20]**; later layers with the same ratio READ the source's compressed KV |
| indexer | one level, top-512, every CSA layer | indexers on **[2,8,14,20,24,28,32,36]**; **layer 20 picks 2048 candidate blocks of 8**, later indexers rank only inside them; non-source layers reuse the source's top-k |
| compressed KV storage | fp8 | fp4 (groups of 16, E4M3 scales); index keys fp4 (groups of 32, E8M0) - match model.py's act-quant rounding |
| Engram | none | layers 1, 14: n-gram hash -> 24 rows x 256 FP8 from 384M-row tables, wkv -> per-hc key + shared value, gate sigmoid(signed-sqrt dot) added to the residual BEFORE the block |
| router | sqrtsoftplus + bias, hash layers (tid2eid) | same + `bias_vl` (image tokens only: unused for text); no hash layers |
| SwiGLU clamp | one limit | per-layer arrays (exp / shexp); V4.1's config: 10.0 everywhere - check the GGUF's config |
| experts | one type per layer/role, fixed blob stride | **per-expert types** (Q3_K / Q2_0 / PTQ1_0; gate+up share one, down its own): mixed roles are split per type in the GGUF (§Export pipeline in V41_RESUME_PROMPT.md) |
| MTP | 1 nextn layer | DSpark: 3 layers on target layers 37-39 (later; not exported yet) |

## Phases

**P1 - loader + geometry (CPU, no GPU).**  `Ds41Geometry` from the `deepseek41.*` keys (and `deepseek41.config`
JSON for anything the flat keys lack); the full tensor-map check (every name predicted, nothing unknown, the
conditional sets: compressor tensors on kv sources, `attn_compressor_gate` where ratio > 1, indexer tensors on index
sources, Engram tensors on 1/14, per-type expert tensors + `ffn_exps_{gu,down}_type`).  `gguf_reader.hpp` learns
type 143 (block_geometry 128/28, name PTQ1_0) through an isolated fork hook.  Per-expert type table per layer.
*Gate:* loads the writer's files (the 2-layer partial file and a layout-only full file: `--layers` / dry run), and
rejects mutated copies (a missing tensor, a wrong type, an unknown name).

**P2 - goldens (CPU, streaming).**  `tools/v41/v41_golden.py`: DeepSeek's model.py, one layer at a time (the
calibration machinery, `--device cpu`), on a few short prompts (en / zh / code, ~32-64 tokens), dumping per layer:
the block input `h`, the attention output, router top-6 ids/weights, the MoE output, the block output; plus final
logits.  Two weight sources: (a) the FP4/FP8 teacher (quality reference), (b) **the GGUF's own weights** decoded
with `v41_formats` (Q8_0 dense, quantized experts) so the engine can be gated tightly against the same weights.
Engram rows from the HF shards.  Decode positions: full-sequence prefill outputs (the V4 invariant: token-by-token
decode must reproduce the full-sequence forward).

**P3 - dense half (ggml, CPU backend first, then CUDA).**  `tools/ds4/ds41_dense.{hpp,cpp}`, same API shape as
`Ds4Dense` (begin_token / attn_router / finish_layer / logits; *_n multi-token later) so the MoE tier and the
generator plug in: single-pass HC, attention without q norm, compressors on source layers + shared compressed KV,
two-level indexer, fp4 compressed-KV storage (the act-quant rounding of model.py), SwiGLU clamps.  Routed experts
in this phase come from a plain reference computation on the GGUF's experts (as V4's test_ds4_dense did).
*Gate:* per-layer cosine / max error vs P2 (b) on CPU, then the same on CUDA; routing agreement (top-6 sets) per
layer, because the MoE gate amplifies small errors (JigSawPT: 14x).

**P4 - Engram.**  CPU hash (port `ds4_engram.c`'s hash on the GGUF's `deepseek41.engram.*` arrays, cross-checked
against DeepSeek's engram.py on P2's prompts), rows read **straight from HF shards 47/48** (row 256 B + scale 8 B,
2 preads per row; Mal 2026-10-10: no Engram copy), prefetched at token start on a thread pool, gate + add on the
GPU at layers 1 / 14.  *Gate:* Engram output == P2's at layers 1/14; latency per token on Crucible (USB SATA) and
Team CX2 measured; move a table only if it shows in decode time.

**P5 - MoE tier with per-expert types.**  `Ds4MoeGeom` / `Ds4GgufBlobs` today assume one stride per layer/role:
make the expert record per-expert (type, bytes, offsets: from the I32 type maps + the per-type tensors); group
routed experts by (gu_type, d_type) and issue one `native_expert_grouped` per pair present (all 9 pairs pass
`ptq1_0_cuda_test`); VRAM LRU / RAM arena slots sized per expert (or per tier pool); CPU path through
`native_expert.cpp` (PTQ1_0 wrapper exists).  Soft-prune / arena-adapt / predicted prefetch carried over.
*Gate:* layer outputs vs P2 (b) with real experts; then end-to-end decode on the real model: ppl on held-out text
vs the head gate's numbers, tok/s.

**P6 - generator / server, then DSpark.**  `ds41_generate --serve` speaking Strata's engine line protocol (copy
`~/AI/Strata-GLM/tools/glm/glm_generate.cpp`'s serve loop incl. snapshot/restore multi-turn reuse and its logged
reuse decisions), served exactly like GLM (V41_RESUME_PROMPT.md §Serving: shim, launcher, configs, wrapper, the
llama.cpp-style metrics from ~/AI/Strata/serve).  Then export `mtp.*` (DSpark 3 layers) and port the draft.

## Order and what can run without the GPU
P1, P2 (CPU streaming, minutes per layer), P3 on the CPU backend, P4's hash and reader: all CPU, so they proceed
while the calibration / quantize passes own the GPU.  CUDA gates of P3/P5 take short lock slots between layers.

## Gate log
(fill in as phases land: date, commit, numbers)
