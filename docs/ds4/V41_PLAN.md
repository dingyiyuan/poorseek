# DeepSeek-V4.1-Flash: our own GSQ-RCO quant + Strata-DS4 engine support (plan, 2026-10-09)

Source checkpoint: `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` (510.3 GB; dense weights abliterated, routed experts,
engram, router, norms byte-identical to the base). Size study: [V41_FEASIBILITY.md](V41_FEASIBILITY.md).

## What V4.1 changes vs V4-0731 (from DeepSeek's `inference/model.py`, mirrored in ~/AI/gsq-rco/dsv41-inference)

| | V4-0731 (our engine today) | V4.1 |
|---|---|---|
| shape | 43 layers, hidden 4096, 256 experts | 40 layers, hidden 5120, 384 experts top-6 + 1 shared, moe_inter 2304 |
| router | sqrtsoftplus + bias | same, plus `bias_vl` for image tokens; route_scale 1.5 |
| compressed attention | ratios 4 (CSA) / 128 (HCA), every layer its own compressor | ratios 0 / 2 / 1; **kv_source_layers [2,8,14,20]**: later layers with the same ratio READ the source's compressed KV; **index_source_layers [2,8,14,20,24,28,32,36]** share one indexer's top-k |
| indexer | one level, top-k 512 | **two levels**: layer 20 picks 2048 candidate blocks of 8, later indexers top-512 only inside those |
| compressed-KV storage | fp8 | fp4 (groups of 16, E4M3 scales); index keys fp4 (groups of 32, E8M0) |
| hyper-connections | hc 4, pre-mix per sublayer | hc 4, **single-pass**: each sublayer's `hc_mixes` produces the NEXT sublayer's pre-mix (attention uses the previous FFN's) |
| engram | none | layers 1 and 14: n-gram hash (orders 2-4, 8 heads = 24 rows) into a 384M x 256 fp8 table, `wkv` -> per-hc-copy key + shared value, sigmoid(signed-sqrt dot) gate added to the residual BEFORE the block |
| swiglu clamp | one limit | per-layer arrays (exp / shexp) |
| MTP | 1 layer | DSpark: 3 layers on target layers 37-39, block 5, its own 128-expert MoE + Markov / confidence heads |
| FP8 block scale | 128x128 | **32x32** (getting this wrong silently rescales every weight) |

## Research: what the video / DwarfStar do that we can use

DwarfStar (antirez/ds4, local ~/AI/ds4-ref updated to fc80bd6) already runs V4.1 (`ds4_engram.c`,
`ds4_deepseek41_cuda.cuh`, `gguf-tools/deepseek41_quantize.py`) - our C reference for the port, next to DeepSeek's
Python model (golden numbers).  Takeaways:
1. **Engram never resident**: rows are pread from the GGUF (posix_fadvise RANDOM + NOREUSE).  The hash depends only on
   token ids, so the 48 row reads/token are **issued before the forward starts** (`ds41_engram_prefetch_start`) and
   joined at layers 1/14 - SSD latency fully hidden.  12 KB/token.  Put the table on any SSD with good random reads.
2. **Their CUDA SSD expert cache is a plain global LRU** (one VRAM slot pool, clock stamps, "protect all hits before
   evicting for a miss", copy from the mmap'd GGUF).  Strata-DS4 already does more (VRAM LRU per layer + 70 GiB RAM
   arena + predicted prefetch + soft-prune + arena-adapt) - nothing to take except the hit-protection rule.
3. **Streamed prefill = wide layer batches**: process a whole prompt chunk per layer and read that layer's experts once,
   overlapping the next layer's reads (Metal) or staging into the device cache (CUDA); "resident prefix layers"
   (`--ssd-streaming-full-layers N`) keep whole early layers resident.  Our chunked prefill (`--prefill-chunk 4096
   --chunk-mmq --chunk-prestage`) is the same idea - extend prestage to read layer L+1 from NVMe while L computes.
4. **Prompt / KV cache to disk** (`--kv-disk-dir`): short appends to a long conversation skip re-prefill (GLM: 30.8 ->
   2.9 s).  Worth adding once V4.1 decodes.
5. Video numbers (5090 + 128 GB + Gen5 SSD, native 4-bit, 5.1 tok/s SSD-bound; 8-ch DDR4 TR Pro resident 3-bit 20 ->
   25.6 tok/s with DSpark): when experts spill, **disk bandwidth is the ceiling, not the GPU**; when resident, RAM
   bandwidth.  Our lever = make the working set fit (GSQ ~1.8-2.25 bpw + soft prune), then DSpark.

## Plan A - GSQ-RCO quant pipeline (~/AI/gsq-rco, tools/v41)

Target formats Strata can already run: **Q2_0** (= GSQ's native symmetric 2-bit grid {-2,-1,0,1} x fp16 scale per 64;
2.25 bpw), Q2_K / Q3_K (GSQ "refine in K-quant format" mode; pfeifferj's V4.1 GSQ-RCO uses Q2_K x26 / Q3_K x14
layers, dense Q8_0).  Sub-2-bit (to fit ~120-130 GB) needs GSQ ternary -> **TQ1_0 (1.69 bpw), no Strata kernel yet**.
Budget examples (543.6B routed weights): all Q2_0 153 GB; 50% TQ1_0 + 50% Q2_0 = 1.97 bpw, 134 GB; RCO-chosen mix at
1.85 bpw = 126 GB.

0. `tools/v41/fetch_v41.sh DEST` - resumable, O_DIRECT sha256-verified download (tested).
1. **v41_ref** (single GPU, layer streaming): DeepSeek's model.py with torch fallbacks for the tilelang kernels
   (fp8/fp4 dequant matmul, sparse_attn gather, hc sinkhorn), weights loaded per layer straight from the HF
   safetensors (no TP convert).  Runs calibration tokens layer by layer, saving each MoE block's input + routing
   (activations on disk; 0.5M tokens x 5120 bf16 = 5 GB per layer-input, hc stream x4).  Doubles as the golden
   oracle for the engine port (like ds4_ref).
2. **Per-expert GSQ**: for each layer, each expert gets its routed tokens; GPTQ init (Hessian from those tokens) ->
   Gumbel-softmax refinement (GSQ's quantizer modules, Lion, 10 epochs) against the FP4 teacher output.  Shared
   expert + dense: Q8_0 (as everyone does).  Engram: stays FP8 rows (lossless, on SSD).
3. **RCO-lite allocation**: per-tensor sensitivity (teacher-KL / output-MSE at each candidate type, plus REAP saliency
   from the routing stats we already collect) -> exact byte budget knapsack.  Full RCO manifold search is optional.
4. **Export** a `deepseek41` GGUF with pfeifferj's / llama.cpp's tensor names (seen in their header), so their patched
   llama.cpp (runtime/ in their repo) is a second oracle.
Compute estimate (4090): GPTQ+GSQ ~15-40 GPU-hours total (per-expert Gumbel steps are memory-bound); calibration
forward a few hours.  Runs in blocks shared with strata-glm.  Calibration: en+zh+code (make_reap_prompts.py) +
reasoning traces, ~256-512 x 2048 tokens.

## Plan B - V4.1 in Strata-DS4

Order (each step gated against v41_ref goldens on a mini fixture first, then the real model):
1. GGUF loader for `deepseek41` names + KV (hidden 5120, 384 experts, per-layer clamps, source-layer lists).
2. Dense half: single-pass HC ordering, ratio 0/2/1 compressors, KV/index source sharing (non-source layers bind the
   source's caches), two-level indexer (candidate blocks at layer 20), fp4 compressed-KV storage.
3. Engram op: CPU hash (port ds4_engram.c logic or DeepSeek engram.py), async pread prefetch at token start, GPU
   gate/add at layers 1/14.
4. MoE tier: 384 experts x 40 layers, expert size from the GSQ GGUF types (Q2_0 / TQ1_0 / Q2_K), soft-prune +
   arena-adapt carried over.
5. DSpark (3-layer draft) after decode works.

## Mixed-tier allocation (Mal, 2026-10-09): Q3 for sensitive experts, ternary for the weakest, Q2 average

Budget-neutral at the all-Q2 size (2.25 bpw, ~153 GB of experts).  Tiers Strata can run (TQ1_0 kernel to write):

| tier | type | bpw | GSQ mode |
|---|---|---:|---|
| sensitive | Q3_K | 3.44 | 3-bit grid, projected onto Q3_K sub-block scales |
| (optional middle) | Q2_K | 2.63 | GSQ in-format K-quant refinement |
| default | Q2_0 | 2.25 | native 2-bit grid {-2,-1,0,1} x scale / 64 |
| weakest | TQ1_0 | 1.69 | native ternary {-1,0,1} x scale |

Neutral mix: every 1% of weights moved to ternary pays for 0.47% moved to Q3_K (e.g. 30% ternary <-> 14% Q3_K).
Signals, all from the calibration pass we already need: REAP saliency (routing frequency x weight x output norm,
same accumulator as s37), and per-expert reconstruction error at each width from GPTQ init (cheap: run GPTQ at
1.58 / 2 / 3 bits, GSQ-refine only at the chosen width).  Allocation = exact-budget knapsack on
error(width) x saliency (RCO-lite); full RCO teacher-KL search optional.
Guard rails: dense, shared expert, router, engram never below Q8_0; ternary only for experts in the bottom saliency
tail AND with low ternary error (per-layer cap, e.g. <= 50% of a layer); first/last layers biased up.  Validation
like s37: per-domain held-out ppl (en/zh/code + never-calibrated languages) vs an all-Q2_0 control and vs the FP4
original on the same tokens.  The ternary tail overlaps the soft-prune set (rarely routed), so speed comes mostly
from the Q3 side being worth it in quality at equal size.

## Research round 2 (2026-10-09): ternary work and V4.1 forks on GitHub

Ternary formats / kernels:
- **PTQ1_0 (id 143) / PQ2_0 (id 142)** = Prism ML's group-128 ternary (Ternary-Bonsai-2-27B), implemented in
  LL4nc33/llama-tq (MIT, CUDA mmvq + MMQ + FWHT) and merged upstream in ikawrakow/ik_llama.cpp #2550 (2026-09-29,
  CPU, Hadamard per role, signs kept at runtime) + #2564 (8-row repack R8, Q8_K128 activations: PTQ1_0 tg 3.4x on
  AVX2).  We use PTQ1_0's block bit-for-bit (tools: src/kernels/cpu/ptq1_0.cpp); our rotation metadata is our own.
  Next CPU speedup = an R8-style interleaved repack at load time.
- Mainline llama.cpp TQ1_0/TQ2_0 are CPU-only (no CUDA).  ik's IQ1_BN/IQ2_BN have no per-block scale (BitNet).
- Ternary PTQ methods to compare against GSQ-ternary: IntelChina-AI/BitTern **CAT-Q** (ICML 2026 oral; has
  Qwen3-30B-A3B / 235B-A22B MoE checkpoints + packed deployment), Kishon-zzx/**TWLA** (ICML'26; Kronecker orthogonal
  rotation to make weights tri-modal), sg-systems-co/**FORGE** (offline-fused randomized Hadamard + GPTQ over a
  ternary codebook, per-256 scale refit).  AngelSlim **STQ1_0** (Tencent Hy4-preview 770B MoE): 3:4-sparse
  symmetric ternary experts at 2.38 bpw, CUDA kernels in their llama.cpp patches.
- Common lesson across all of them: **rotate before ternarizing** (Hadamard / orthogonal), then a second-order
  (GPTQ-style) solve over the ternary grid with the scale refit after compensation.

V4.1 runtimes:
- **sybil-solutions/dsv41-flash-offload** (MIT): 1x RTX 3090 + 503 GiB 8-channel DDR4 + EPYC 24c: EXL3 3.0 bpw,
  decode 19.3/24.0/26.6 tok/s at 1/2/4 streams, prefill 690-790 tok/s up to 261K.  ALL experts pinned in host RAM;
  VRAM expert hit rate only ~0.41, the AVX2 CPU tier carries the misses (needs their ~150 GB/s RAM; ours ~30).
  Worth taking: elastic VRAM expert cache (prefill borrows slots, decode gets them back), staged-DMA prefill with
  FP16 GEMM for busy experts (fidelity: top-1 0.9856 vs 0.988 guard), cross-layer DMA prefetch, exact prefix cache
  (131K-token cached turn: TTFT 200.8 s -> 0.76 s).
- **JigSawPT/deepseek-v41-flash-on-5090** (llama.cpp fork `dsv41-porte`, MIT): 5.12 tok/s new content / 21.27
  resident on a 5090 + 125.7 GiB RAM (269 GiB MXFP4 experts streamed: 18 GiB VRAM + 72 GiB RAM tiers + NVMe).
  **Port defect to avoid: V4.1 has NO per-head RMS norm on q** (V4 has; keeping it gave fluent text but layer-39
  correlation 0.78).  Floors vs the FP8/FP4 reference: fp8 linear 0.99990, window KV 0.99961, fp4 compressed latent
  0.9946, fp4 indexer q/k 0.9877; the MoE gate amplifies small errors 14x (routing flips) - gate goldens on routing
  agreement, not only cosine.  Also has engram prefetch and a DSpark draft GGUF.
- Others: rexlunae/ds4-convert (streaming safetensors -> deepseek41 GGUF), helgard-orlm (8 GB GPU, 1.6 tok/s
  disk-only), 05yuki (dual EPYC + 2x 5070 Ti), vcruz305 (llama.cpp deepseek41 conversion recipe), pfeifferj's
  runtime patch (oracle).

## First data (2026-10-09, shard_check.py on layer 0)
Layer 0 validates clean (2,334 tensors, 0 issues).  Weight-space round-to-nearest error, 256 sampled rows per matrix:
ternary 0.443, Q2 0.378, Q3 0.183 (relative), and **the same for every one of the 384 experts to +-0.001** (p5-p95)
and every projection.  The FP4-native experts all have the same per-32 value distribution, so **weight-only
priors cannot rank experts** - the mixed-tier allocation has to come from calibration: activation-weighted
(GPTQ/GSQ) error per expert x REAP saliency.  Q3 halves the Q2 error; ternary costs +17% over Q2 before
calibration-aware solving.
