# DwarfStar (antirez/ds4 @fc80bd6) and Strata forks: V4.1-specific findings, 2026-10-10

Checked: antirez/ds4 (fc80bd6, 2026-10-07: the same commit our writer copied names from), VecSzn/PolyStrata,
mw00/project-maya, bodhi37/strata (Qwen/GLM only, no DeepSeek code), sectilelabs (GInfer, repo still empty).
**Only DwarfStar has DeepSeek V4.1 code.**  Paths below are in the ds4 repo.

## Use, ranked

1. **Official V4.1 quality fixtures: our ship gate.**  `gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/`
   (38 MB): 112 held-out prompts (100 short + 12 long, up to 23K tokens) with official DeepSeek API continuations
   (thinking off, temp 1, top-20 logprobs, 64 tokens), plus DwarfStar's own scores for its Q2 builds
   (`results/base-*.tsv`, `fix-*.tsv`: per-case teacher-forced NLL, API top-1 agreement, top-20 recall).
   Plan: once ds41_generate can score, run the same manifest on our GSQ file -> per-case NLL / top-1 agreement
   directly comparable with DwarfStar's Q2 (`api_top1_rate`, `avg_nll`).  Scorer: `score_official` +
   `compare_scores.py`.  Caveat: our source is the dealignai UNCENSORED checkpoint, so it drifts from the official
   API a little on its own; compare quants against each other, and ideally score the BF16/FP4 teacher too
   (the head gate's teacher numbers are the in-house version of this).
2. **Mixed-layer evidence (`gguf-tools/mixed/README.md`).**  On V4 Flash, upgrading only the LAST SIX layers'
   routed experts (37-42 of 43) from 2-bit to Q4 made a 91 GB file measurably closer to full Q4 (153 GB) than plain
   2-bit.  Our `v41_alloc.py --edge-weight 2.0` already boosts layers 0-1 and 37-39 (last 3); DwarfStar's
   best result was the last SIX (V4.1: 34-39).  Widening the boost is Mal's call (open question 2026-10-10).
3. **Validator `gguf-tools/deepseek41_validate_gguf.py`.**  Audits a complete V4.1 GGUF's layout and re-encodes
   sampled experts against the source; Engram as 264-B rows in a disk shard.  Our uniform-type tensors and dense
   half use DwarfStar names, so its layout checks are a free second opinion on our writer.  It will NOT accept our
   per-expert mixed tensors (`ffn_<role>_exps.<type>.weight` + type maps): DwarfStar has no per-expert tier
   format, mixing is whole-layer only (splice tool).  So **DwarfStar cannot load our mixed file**; it could load
   an all-uniform control file.
4. **Reasoning effort / think level (serving shim).**  DwarfStar's V4.1 takes `--think-level 1..100`
   (0 = off; `--think` = 75, `--think-max` = 100); server maps OpenAI `reasoning_effort` names and
   `chat_template_kwargs` (`ds4_server.c:1037-1170`, `ds4_deepseek41_reasoning_effort_text`, render at
   `ds4_server.c:3377`).  Copy that mapping into serve_ds41.py so clients' `reasoning_effort` behaves the same.
5. **Tool-text escaping (`ds4_tool_text.h`).**  Tool bodies escape only their own closing delimiter (`&lt;` +
   the rest of the tag, `&amp;` chains).  Our DSML -> Qwen `<tool_call>` rewriter must unescape the same way, or
   tool arguments containing the closing tag break.
6. **Engram (`ds4_engram.[ch]`, 327 lines).**  4-gram, 8 heads x 24 columns, 256-dim rows, 264-B rows
   (row + 8-B scale, = our plan's 2 preads/row stored adjacently in their own shard), read through a separate
   UNCACHED fd (never mmap); hash history snapshot = 3 tokens (copying it snapshots the hash state: relevant to
   our multi-turn snapshot/restore).  Port the hash from here as the CPU reference for P4.
7. **Kernels**: `ds4_deepseek41_cuda.cuh` (633 lines) + `ds4_cuda.cu` - V4.1 text on CUDA (target DGX Spark);
   reference for P3 (mHC, compressor, indexer).  V4.1 + CUDA requires `--power 100`.  DSpark spec decode exists
   for V4 Flash (needs a support GGUF); no V4.1 DSpark found.

## Non-DeepSeek (for strata-glm, FYI)
PolyStrata: cold experts computed on the CPU from the mmap'd GGUF instead of skipped (vs our `--skip-miss`);
drafting from the model's draft block + repeats of the conversation; first-64-token parity vs llama.cpp.

## From the GLM / Qwen forks, applicable to V4.1 (added 2026-10-10 15:15)

Our engine already inherits upstream Strata's 3-tier expert source (VRAM / pinned RAM / NVMe `pread` ring with
O_DIRECT, `src/core/expert_source.cpp`), which V4.1 needs: the ~152 GiB file > 90 GiB RAM + 24 GiB VRAM.

**Quant pipeline (project-maya `docs/QUANT-PLAN.md`, a GLM GSQ/RCO plan close to ours):**
- **KL to the teacher, not only ppl.** They store the reference's top-64 log-probs + logsumexp per eval position
  and score each candidate by KL, top-1 agreement and ppl per domain + long context.  Our head gate
  (`v41_calib.py head`, l.573) records ppl + top-1 only; adding the teacher's top-k to the calibration head
  output and KL to the quant pass's head gate would give a sharper GSQ-vs-teacher number.  Needs to be in place
  before the calibration head step runs (~18:30) - Mal's call; a bug there would stop the unattended chain.
- **Floor: down never below gate/up for the same expert.**  `v41_alloc.py --per-proj` picks gu and dn tiers
  independently (l.40).  Our layer-0 data already says down carries 68% of an expert's Q2 error, so the floor
  likely rarely binds; check the full map's count of dn < gu experts before the export.
- **Loop test** for low-bit files: 50 sampled + 20 greedy long generations; count repeated n-grams, never-ending
  runs, broken think blocks.  Add to our post-export checks (V4.1 thinks by default).
- Pruning: REAP-pruning GLM (60/288 experts) cost KL 0.45; per-expert bits beat pruning.  We do not prune: keep it.

**Engine (PolyStrata `docs/BACKENDS.md` "What was tuned", same Strata base, measured on GLM 5.3 Flash
with OUR GLM file on a 5090/120 GiB):**
- **The draft block is the biggest decode win**: GLM 26.7-27.7 -> 47-51 tok/s with it.  For V4.1: export `mtp.*`
  (DSpark, 3 layers) early rather than last (P6 now says "then DSpark").
- Cached experts launched 8 at a time, one launch per stage (Qwen 132 -> 241-269 tok/s): our
  `native_expert_grouped` is the same idea; keep it for every tier pair.
- Sampling / guess checking without a candidate per vocab entry (159 ms -> 25 ms per 600-token answer).
- Prefill: walk the prompt's calls layer by layer so a layer's experts are copied to the card once
  (GLM 417 -> 516 tok/s), keep the residual in host memory past checkpoints (-> 602-616), leave a round of a
  layer's experts in the staging buffer for the next call (-> 639-643).
- Prefault the experts' pages once after load (first answer 43 -> 47 tok/s).
- Overlap: the card's layers run while the CPU computes the cold experts (52-53 -> 56-59 tok/s).

**bodhi37 `orca-port` (frozen fork: SSD tier on a 32 GiB RAM box) - lessons for our SSD-backed V4.1:**
- Decode wall = NVMe queue-depth-1 latency (~2.8 ms) x serial layers, not bandwidth; issue a layer's misses
  together and compute resident experts while they fly (two-pass `begin_layer` / `wait_layer`).
- Read amplification: page-faulting 4 KiB through the mapping was 6.4x; whole-blob `pread` + O_DIRECT fixed it.
- `cudaHostRegister`-pinned RAM tiers are unswappable: a too-big pinned tier + KV caused swap storms and cuBLAS
  init deaths; register in ~2 GiB slices and shed under memory pressure.  Size V4.1's pinned tier with headroom.
- LRU churn on prefill collapsed decode (12 -> 2.5 tok/s): keep prefill admissions off / profile-seeded static tier.
