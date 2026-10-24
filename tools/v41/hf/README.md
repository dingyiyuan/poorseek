---
license: mit
base_model:
- dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8
- deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
- deepseek
- deepseek-v4.1
- moe
- quantized
- 2-bit
- ternary
- mixed-precision
- gptq
- uncensored
- strata
- consumer-gpu
---

<div align="center">

# DeepSeek-V4.1-Flash · GSQ mixed-tier: ternary / 2-bit / 3-bit experts (152 GiB)

**A 552B-parameter MoE running on one gaming PC — RTX 4090 (24 GB) + 90 GB RAM + NVMe**

**Prefill ~250 tok/s (exact) · DSpark decode 17–30 tok/s at the coding-safe bias · 512K context**

Runs on **[Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)**, an engine built for exactly this model and
this class of machine, on top of [Strata](https://github.com/Niko1221/Strata).

</div>

> **Update 2026-10-15:** the engine's defaults are now the **coding-safe route bias ×0.5** (HumanEval 20/21 and tool calls
> 10/10, vs 16/21 at the old full bias) with DSpark speculative decoding **in the server** (exact speculative sampling),
> a confidence-gated verify (prose 9.5 → 17.3, code 18.8 → 25.5, python 26.6 → 29.1 tok/s), a **bit-exact 4-bit KV cache**
> (`--kv-fp4`, +1.1 GiB VRAM at 512K) and ~2× faster conversational turns (+1000-token turn 40 → 20 s at 512K).
> Details: the repo README "Latest".

> **Upload complete (2026-10-13):** all 16 model parts, `V41-MTP-Vision.gguf` and the small files are on the Hub.
> `SHA256SUMS` lists every part; `reassemble.sh` refuses to join until all of them are present and intact.

> **Experimental research release** from a single developer, measured on one PC.
> This file is **not** a llama.cpp / Ollama / LM Studio GGUF. It uses per-expert quant types those runtimes don't load.
> Use it with Strata-DSV4.1.

## What this is

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) has 40 layers with 384 routed experts each
(top-6 per token), a shared expert, Hyper-Connections, CSA sparse attention, and an **Engram** n-gram memory. Its
15,360 routed experts hold 543.6B weights; even at their native FP4 they're far bigger than any consumer PC's RAM.

This repository holds a **calibrated mixed-tier quantization** of
[dealignai's abliterated checkpoint](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8), in which
**every expert's gate/up and down projections get their own format**:

| Format | Grid | Bits per weight | Share of expert weights |
| --- | --- | ---: | ---: |
| **PTQ1_0** (ternary) | {−1, 0, +1} × fp16 scale per 128 | 1.75 | **42.8%** |
| **Q2_0** | {−1, 0, +1, +2} × fp16 scale per 64 | 2.25 | 39.2% |
| **Q3_K** | 3-bit, 6-bit sub-scales per 16, fp16 per 256 | 3.44 | 18.0% |
| **All experts** | | **2.250** | |

The dense half (attention, shared expert, output head) is **Q8_0**. The embedding, compressor and indexer stay F16,
and norms, router and mHC stay F32.

The total is budget-neutral: exactly the bytes of a uniform 2.25-bit (all-Q2_0) quant, but spent where calibration
says they matter. Nearly half the expert weights are ternary, which pays for 3-bit on the experts that need it. On the
calibration proxy, the mix has **17% less output error than uniform Q2_0 at the same size**.

"GSQ" names the method family this follows: the format grids and the plan come from
[GSQ (Gumbel-Softmax Quantization)](https://github.com/IST-DASLab/GSQ) and
[RCO (budget-exact allocation)](https://arxiv.org/abs/2605.00649) by IST Austria's DAS Lab. This release is the
**GPTQ-initialized stage** of that pipeline with our own "RCO-lite" allocation. GSQ's Gumbel-softmax refinement
hasn't been applied yet; see *Not done yet* below.

## Why

We wanted a frontier-class open model running **locally, on hardware people actually own**, without slicing it down to a
distilled toy. The design question was where the bits matter. A MoE spends most of its parameters on experts that
each token touches only rarely, so we spend the bit budget where the calibration says it counts, and run the rest
as cheaply as the quality numbers allow.

The engine then treats the experts as a three-tier cache:

| Tier | Holds | Fed by |
| --- | --- | --- |
| VRAM cache | the hottest ~1,300–1,600 experts | LRU + routing bias toward resident experts |
| Pinned RAM arena | ~6,700 experts (60 GiB) | DMA straight to the GPU |
| NVMe file tier | the other ~8,700 experts | O_DIRECT reads, prefetched a layer ahead |

## How it was made

The whole pipeline is open source in
[Strata-DSV4.1/tools/v41](https://github.com/Indras-Mirror/Strata-DSV4.1/tree/main/tools/v41). It runs on a single
RTX 4090.

**0. Weight-only metrics can't rank these experts.** V4.1's routed experts were trained in FP4, so all 384 experts of a
layer share the same per-32 weight distribution. Every expert in layer 0 has the same round-to-nearest error to three
decimals (ternary 0.443 / Q2 0.378 / Q3 0.183 relative). Which experts deserve more bits can only come from
**activations**: what the experts actually compute on real text.

**1. Calibration corpus (agentic-first).** Real coding-agent session transcripts (secrets scrubbed value by value, tool
results clipped) plus antirez/ds4's 4,690 imatrix prompts, all re-rendered with DeepSeek's own V4.1 chat encoder (DSML
tool-call format). That's 128 sequences × 2,048 tokens, with about a quarter of each domain **held out**: 98 sequences
fit the quantizer, and 30 only ever measure it.

**2. A layer-streaming teacher forward.** DeepSeek's own reference `model.py` runs **one layer at a time** straight
from the HF shards, keeping the hidden state on disk between layers. That makes the teacher the real model, not an
approximation. Each layer's MoE runs expert-major: every FP4 expert is dequantized once and sees all of its routed
tokens.

**3. Per-expert GPTQ at three exact grids.** For every expert, gate+up are fitted on the Hessian of its input, and down
on the Hessian of its clamped-SwiGLU × routing-weight input. Scales are chosen **before** GPTQ by a diag(H)-weighted
search (11 clip ratios × both signs) and rounded to what each format stores, so every result is exactly representable
in its GGUF type. GPTQ (damp 1.0) then picks the integers. It beat round-to-nearest in every group (error ratios
0.58 ternary / 0.70 Q2 / 0.74 Q3). Each expert's error is measured on the **held-out** sequences only, as output error
with the routing weight included.

**4. Budget-exact allocation ("RCO-lite").** The cost of a choice is held-out output error divided by the layer's
routed output energy, so every layer counts equally; the first two and last three layers count double. Gate+up and
down are allocated separately. The budget is exactly the all-Q2_0 byte count, solved by Lagrangian relaxation
(bisection on λ) plus a greedy pass to spend the remainder. Guard rails: at most 50% ternary per layer, and experts
with fewer than 8 held-out tokens are floored at Q2_0, so an expert can't drift to ternary without evidence.

**5. Sequential quantize pass.** A second layer-by-layer chain quantizes every expert at its chosen format. Each layer
is fitted on the outputs of the **already-quantized** layers below it, so later layers adapt to the errors earlier
ones actually make.

**6. Bit-exact export.** Block encoders write the GGUF bytes directly and are checked bit for bit against ggml's
dequantizers and the PTQ1_0 reference. Ternary packs 5 trits per byte (28 bytes per 128 weights), and Strata runs it
natively on the CPU and in its grouped-expert CUDA kernels.

### Where the bits went

Measured on the held-out calibration tokens; the relative output error is each layer's quantized routed output against
the FP4 original, in the final sequential pass.

| Layers | Expert bpw | Held-out relative output error | |
| --- | ---: | ---: | --- |
| 0–3 | 2.25–2.34 | 2.4–6.8% | |
| 4–19 | 2.21–2.71 | 1.9–12.9% | the richest band: the most 3-bit experts (peaks at layers 12–13) |
| 20–31 | 2.00–2.01 | **0.03–0.44%** | barely sensitive: about half ternary, the rest Q2_0 (the 50% ternary cap binds) |
| 32 | 2.66 | 14.8% | the most sensitive layer in the model; calibration found it and paid for it |
| 33–39 | 2.00–2.03 | 0.13–2.1% | |
| **All 40** | **2.250** | mean 4.3% | |

### Not done yet

- **GSQ refinement:** Gumbel-softmax optimization of the grid assignments at each expert's chosen format. This is
  the step that closes most of the gap to vector quantization in the GSQ paper. It slots in where GPTQ runs and
  writes the same format.
- A Hadamard rotation for the ternary tier.
- The full RCO manifold search: the allocation above is its Lagrangian "lite" version.

## Quality

Perplexity on the **49,146 held-out calibration tokens** (never used for fitting): the quantized model vs the original
checkpoint ("teacher"):

| Domain | GSQ ppl | Teacher ppl |
| --- | ---: | ---: |
| All (head gate) | 5.645 | 4.915 |
| agent | 2.118 | 2.121 |
| reasoning | 6.19 | 5.83 |
| code | 1.98 | 1.74 |
| zh | 7.64 | 6.83 |

Top-1 token agreement: 0.674 (GSQ) vs 0.701 (teacher). In the engine, a 4096-token code prompt gives ppl 2.380 and top-1 0.822.

**Prompt processing is exact:** every prefill speed-up in the engine keeps the last-token logits byte-identical.
**Decode uses two selection-only routing biases by default:** a bonus (+1.0) for VRAM-resident experts and a penalty
(−0.6) for NVMe ones. The mixing weights stay exact and every expert stays reachable, but the bias changes *which*
experts compute each token, and it costs more quality than perplexity suggested (code 2.38 → 2.87). On DwarfStar's
112 official DeepSeek-API fixtures (teacher-forced over the API's own continuations, 6,922 positions; scorer in
[`tools/v41/quality/`](https://github.com/Indras-Mirror/Strata-DSV4.1/tree/main/tools/v41/quality)):

| Routing | API top-1 agreement | NLL | Decode tok/s (docs / python prompt) |
| --- | ---: | ---: | ---: |
| exact (`--route-bias 0 --route-bias-file 0`) | 86.2% | 0.516 | ~2.6 (I/O-bound: most experts on NVMe) |
| bias × 0.25 (`--route-bias 0.25 --route-bias-file 0.15`) | 85.2% | 0.551 | 12.6 / 7.4 |
| bias × 0.5 | 81.7% | 0.676 | 18.4 / 11.2 |
| default bias (1.0 / 0.6) | 72.1% | 1.111 | 29.9 / 20.6 |

(102 short prompts; the 10 long prompts route exactly either way: 94.1%.)  For reference, DwarfStar's own
IQ2_XXS-Q2_K build scores 89.9% / 95.0% on the same short / long groups.  The default bias also causes repetition
loops on some prompts under greedy decoding.  Pick the bias for your use: speed-first = default, quality-first = 0.25.
Smarter biases that only bend near-ties were measured and lose to a flat scale.

**On real tasks** (21 HumanEval problems + 10 tool-call cases, greedy, through the server): exact-ish x0.25 20/21,
**x0.5 20/21 (the default since 2026-10-15)**, x0.75 19/21, full bias 16/21 (the extra failures are logic slips on easy
problems); tool calling 10/10 at every setting.

## Speed (RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe)

| | |
| --- | ---: |
| Prefill, 8K-token prompt | 244–248 tok/s one-shot (tensor-core experts) |
| Prefill, 16K / 20K prompt | 288 / 266 tok/s |
| Decode (256-token prompt) | 25.5–26.3 tok/s · 26.6–28.9 with the opt-in fused ggml op |
| Decode with **DSpark** speculative decoding (one-shot, greedy) | **31.7–39.5 tok/s** (docs 39.5 vs 29.9 plain, python 31.7 vs 20.6, code 35.6 vs 26.7: 1.32–1.53×) |
| Decode in a real agent session (20–32K context) | 22.7–28.2 tok/s |
| Model load | ~50 s |

How each number was reached, step by step:
[V41_PROGRESS.md](https://github.com/Indras-Mirror/Strata-DSV4.1/blob/main/docs/ds4/V41_PROGRESS.md).

## Files

| File | What |
| --- | --- |
| `V41-GSQ.gguf.part-NN-of-MM` | the 163 GB model, split into ~10 GiB byte ranges so a slow or interrupted upload never has to restart more than one part |
| `reassemble.sh` | checks every part, joins them into `V41-GSQ.gguf`, verifies the result |
| `SHA256SUMS` | hashes of every part and of the joined file |
| `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt` | **placeholders - you need these two files, download them separately**: the two Engram shards (101.5 GB each) are not hosted here; these say where to get them (step 2 below) |
| `V41-GSQ.gguf.experts.json` | per-expert offsets and formats |
| `rank.bin` | expert hotness ranking, used to seed the VRAM cache and RAM arena |
| `tokenizer/` | V4.1 tokenizer + chat template |
| `V41-MTP-Vision.gguf` | the **DSpark** speculative-draft stack (`mtp.0-2`: experts MXFP4, a lossless repack of the original FP4; the rest Q8_0) plus the **vision tower** (Q8_0). |
| `V41-MTP-Vision.Q2_K.gguf` | **recommended for DSpark:** the same file with the draft experts requantized to Q2_K (5.66 GB vs 8.42 GB). Frees 2.76 GB of VRAM for the expert cache at the same acceptance (3.23 vs 3.28 accepted drafts/pass offline) - that alone is +10% decode. Use with `STRATA_DSPARK=V41-MTP-Vision.Q2_K.gguf STRATA_DSPARK_DEPTH=4` (one-shot `ds41_generate`; in `--serve` via `STRATA_SERVE_DSPARK`, on by default in the repo's `ds41-engine.sh` - exact speculative sampling at any temperature). Vision engine support is still in progress. |

## Quick start (4 steps)

**You need:** Linux, an NVIDIA GPU with 24 GB, ~90 GB RAM (the RAM arena is 60 GiB by default; `--arena-gib` lowers
it, at some speed cost), the CUDA 13 toolkit, and fast SSD space for **the model (163 GB) + the Engram tables
(203 GB) = ~366 GB**, plus another ~203 GB *temporarily* while the Engram tables are extracted (or download the
Engram shards to a different drive).

**1. Download this repo and join the model** (~163 GB):

```bash
hf download IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored --local-dir V41-GSQ
bash V41-GSQ/reassemble.sh V41-GSQ                 # needs ~163 GB free on top of the parts
# or: bash V41-GSQ/reassemble.sh V41-GSQ --delete-parts   (deletes each part once appended: ~163 GB total)
```

**2. Download the two Engram shards - they are NOT in this repo** (the `…NEED-TO-DOWNLOAD-ENGRAM.txt` files mark
them; 101.5 GB each). V4.1 reads its Engram n-gram memory from disk at run time. The tables are unchanged FP8 rows
from the original checkpoint (the abliteration left them byte-identical), so there is no point hosting a third copy:
get them from [dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8)
or [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (same two files).

```bash
hf download dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8 model-00047-of-00048.safetensors \
   model-00048-of-00048.safetensors model.safetensors.index.json config.json --local-dir FP8-engram
```

**3. Build the engine and extract the Engram tables** (one table per fast drive is ideal: decode reads ~48 random
256-byte rows per token):

```bash
git clone https://github.com/Indras-Mirror/Strata-DSV4.1 && cd Strata-DSV4.1
# CUDA build, sm_89 (RTX 40xx) shown - see the repo README for other GPUs
cmake -S . -B build-ds4-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_GGML_CUDA=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-ds4-gpu ds41_generate
python3 tools/v41/extract_engram.py ../FP8-engram --to layers.1=/fast/a --to layers.14=/fast/b
# when it has finished (it checks sizes and spot-checks 2000 rows per tensor), ../FP8-engram can be deleted
```

**4. Run** (the first load takes ~50 s: it pins the 60 GiB RAM arena):

```bash
STRATA_NO_LARGEPAGES=1 build-ds4-gpu/ds41_generate -m ../V41-GSQ/V41-GSQ.gguf --ids-file prompt.i32 -n 128 \
     --backend cuda --experts gpu --engram "/fast/a:/fast/b" --rank ../V41-GSQ/rank.bin
```

`--ids-file` takes int32 token ids. For chat and agents, run the OpenAI- and Anthropic-compatible server with tool
calls in `tools/v41/serve/` (see "Serving it" in the
[GitHub README](https://github.com/Indras-Mirror/Strata-DSV4.1#serving-it-openai--anthropic-api-tool-calls)); it uses
`V41-GSQ/tokenizer/`.

## Credits

- **DeepSeek** for [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT).
- **[dealignai](https://huggingface.co/dealignai)** for the abliterated FP8 checkpoint this is quantized from (MIT).
  This release inherits that abliteration: the model has **no refusal behaviour**, and you are responsible for how you
  use it.
- **[GSQ](https://github.com/IST-DASLab/GSQ)** (Dadgarnia, Tabesh, Nikdan, Helcig, Kurtić, Kleinegger, Alistarh) and
  **[RCO](https://github.com/IST-DASLab/RCO)** (Helcig, Alistarh), IST Austria DAS Lab, for the quantization grids
  and the budget-exact allocation idea this pipeline follows.
- **[antirez/ds4](https://github.com/antirez/ds4)** for its 4,690 imatrix calibration prompts (re-rendered for V4.1).
- **llama-tq** for the PTQ1_0 ternary block format (type 143) and its reference kernels (MIT).
- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221), the tiered MoE engine this work builds on;
  **ggml / llama.cpp** for the compute backend.
- Quantization pipeline, engine port and measurements: **Indra's Mirror** ([Strata-DSV4.1](https://github.com/Indras-Mirror/Strata-DSV4.1)).
