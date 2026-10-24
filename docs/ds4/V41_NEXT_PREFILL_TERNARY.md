# V4.1 next: faster prompts (sybil's lever) + meshapplied's ternary experts - PREPARED, NOT RUN

Written 2026-10-15 ~17:00 while Mal ran a real Quetza harness session on the production wrapper (GPU reserved: no
build, no download, no run was done for this).  Everything below is code + plan; each step says what to measure.

## 0. The real-harness numbers this plan is aimed at (strata-ds41-quetza, 512K, serve defaults, 2026-10-15 16:39-16:45)

| turn | new tokens | prompt time | split | decode | DSpark |
| --- | ---: | ---: | --- | ---: | ---: |
| first request (a Quetza side call) | 366 | 20.3 s | dense 3.5 + experts 12.3 + cache open 4.3 | 17.1 tok/s | 2.12 tok/pass |
| the Quetza system prompt (cold) | 20,648 | **70.4 s** (293 tok/s) | 3 batches: dense 25.2 + experts 44.5 | 17.9 | 2.31 |
| tool result | 2,171 (21,018 reused) | **22.9 s** | dense 6.8 + experts 15.3 | 19.6 | 2.33 |
| short tool result | 66 (26,176 reused) | 7.3 s | exact passes (<= 96) | 23.5 | 3.02 |

Prefix reuse works (the 70 s is once per session).  **What Mal feels is the batch-turn floor: every turn of >= 128 new
tokens pays ~12-15 s of "experts" whatever its size** - a batch of a few hundred tokens already routes to nearly every
expert of every layer, so each batch is a full sweep of the model (the arena over PCIe + the file tier off NVMe), and
the log says `experts 0 hit`: the batch path does not use the 12.9 GiB VRAM expert cache at all.

sybil-solutions/dsv41-flash-offload (same model, 3090 + 256 GiB 8-channel RAM, all experts in RAM) prefills at
590-810 tok/s: 16,384-token chunks + copy-engine staging, i.e. half the sweeps per token.  We cannot hold the experts
in RAM (90 GB), so our sweep has an NVMe part they do not have.

## A. Prompt path (sybil's lever, adapted)

**A0. Measure the sweep first (one session, ~30 min of GPU).**  `MIMO_CHUNK_PROF=1` on the 4-turn serve bench + a
fresh 20K prompt + a ~2K turn: per batch, reader-thread time and bytes (file tier), caller waits on reads, PCIe DMA
bytes / time, grouped-kernel time.  The question: is a full sweep NVMe-bound (~80 GB of file tier at ~5.3 GB/s = ~15
s - matches the 12-15 s), PCIe-bound (64 GiB of arena at 21.6 GB/s = ~3 s) or something else.  Session 12's chunk prof
(271 tokens: callers waited only 35 ms on reads) and its mirror result (no gain at +1001) argue against NVMe; the
harness numbers argue for it.  A0 settles it before anything is built.

**A1. `--prompt-batch 16384` (flag only).**  A 20,648-token prompt = 2 sweeps instead of 3 (expected ~70 -> ~55 s); a
16K prompt = 1 instead of 2.  Elastic columns already give the 5 GiB of state back after the prompt, so decode is
untouched.  Risk: VRAM during the batch (16384 x 0.31 MiB columns + the 3 GB chunk budget + the dense graphs' compute
buffers at width 16384; 13.9 GiB free after the dense half).  Measure: fresh 20K and 16K prompts, >= 3 runs a side,
plus the bit-exactness gate (exact routing, `STRATA_PRESTAGE_GIB` pinned, tokens identical).  Then try 24576 / 32768
if it fits (one sweep for the whole Quetza system prompt).

**A2. Keep the VRAM expert cache through small batch turns.**  A 2,171-token batch needs ~0.7 GiB of state columns,
not the whole 12.9 GiB cache that the batch path releases (`experts 0 hit`, then `cache open` 0.6-4.3 s afterwards).
Size the release to the batch (elastic columns already know NC): keep the resident experts and serve their groups
from their VRAM slots in run_chunk (no PCIe for ~1,500 hot experts per sweep), and skip the re-seed.  Saves the
re-open (up to 4.3 s on the first turn) + the hot experts' DMA (~13 GB = ~0.6 s) per small turn.  Code: the deferred
seed / release_cache around the batch in ds41_generate.cpp + run_chunk's resident-slot path (decode's `slot_of`).

**A3. If A0 says NVMe-bound: fewer file-tier bytes per sweep.**  (a) the second-drive mirror was measured on decode
and small turns only - re-test it on a FULL sweep (fresh 20K) before calling it dead for prompts; (b) ternary for
the COLD experts (section B): the 8,241 file-tier experts are by construction the least-routed, so ternary costs
least quality there, and every byte saved there is saved on every sweep AND every decode file miss.

Not applicable from sybil: FP16 GEMMs for busy experts (we already run prompt experts on tensor-core MMQ, PTQ1_0 via
a lossless transcode); cross-layer DMA prefetch (our whole-layer chunk prefetch measured neutral at 5.3 GB/s reads,
s11); their CPU expert tier (needs 8-channel memory; FreeToken rule, s11).

## B. meshapplied ternary experts (`ternary-g128-hadamard-gptq`)

Facts established (no GPU used):
- Pack: huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest, 40 x `layer-LL.safetensors` (~3.6 GB each,
  141 GB total) + manifests; spec github.com/meshapplied/goethe `docs/pack.md`, runtime `ternary-kernels/`.
- **Valid for our model:** built from stock V4.1-Flash; dealignai's card says the routed experts are untouched by the
  abliteration, so their experts drop into our GSQ.
- **Format = our PTQ1_0 exactly:** 128 trits x one fp16 scale per 128 input weights.  Their trits repack LOSSLESSLY
  into PTQ1_0 blocks (`tools/v41/ternary/mesh_pack.py`, `ptq1_rows`, self-test `check`).  The one difference is the
  basis: stored W R^T, R = blockdiag(H_b diag(s))/sqrt(b), b = 1024 for w1/w3 (K 5120), 256 for w2 (K 2304), s =
  `hadamard.signs.<K>` (b values, repeated per block), natural-order Walsh-Hadamard, 1/sqrt(b).
- **Their own eval** (eval.json): ALL experts ternary vs FP4 = +0.20-0.27 nats held-out (code +0.10-0.20, general
  +0.41-0.49).  Our GSQ mix is ~+0.14 vs the FP8 teacher.  So "all mesh" is a quality DROP (Mal: no quality drops);
  the question is per expert.

B1. **Screen (prepared: `tools/v41/ternary/mesh_screen.py`).**  On the FP8 teacher's own layer inputs
(`/media/Crucial1TB/v41-teacher`, h/L00..L40, 639 tokens - any layer, no chain replay), per routed expert: output
error vs the FP4 base of (a) OUR shipped GSQ expert, read from V41-GSQ.gguf at its experts.json offsets, and (b)
mesh's ternary, de-rotated into our basis - whole expert, gate+up alone, down alone.  `summary` splits by OUR type:
  - where ours is PTQ1_0 (same bytes): mesh better -> a free swap (quality up, bytes equal);
  - where ours is Q2_0 / Q3_K: how close is mesh's ternary -> candidates to move down to 1.75 bpw (bytes down).
  Steps: `hf download meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest --include "layer-0[05].safetensors"
  "manifest-layer-0[05].json" --local-dir /media/mal/NVME1TB/mesh-ternary` (7 GB; NVME1TB has 91 GB free - never
  the full 141 GB there), `python3 tools/v41/ternary/mesh_pack.py check /media/mal/NVME1TB/mesh-ternary --layer 5`,
  then per layer `V41_WORK=/media/Crucial1TB/v41-teacher bash tools/ds4/memguard.sh 86 72 -- python3
  tools/v41/ternary/mesh_screen.py layer 5 --pack /media/mal/NVME1TB/mesh-ternary` (GPU, engine stopped), then
  `mesh_screen.py summary`.  Layers to sample: 0 (hash-routed), 5, 14 (Engram), 20, 30, 39.  639 tokens = ~10
  routes per expert, so read the per-type aggregates, not single experts.
  Caveat to check on the first run: mesh_screen's hook passes `expert.w1/w3/w2` of model.py's Expert (the names
  v41_calib.screen_expert uses) and C.K.dequant_fp4_weight - same as screen_expert, so it should hold.

B2. **Engine integration, only if B1 says yes.**  New expert class "PTQ1_0 in the Hadamard basis" (same bytes and
kernels as PTQ1_0; the GGUF keeps type 143 + an I8 flag tensor `blk.L.ffn_exps_rot` [384] for gu / down, and the two
sign vectors once).  Activation side, all paths that compute experts:
  - decode / verify (gpu_run, gpu_run_n_ex): per layer per token one extra q8_1 copy of R x (5 x FWHT-1024, ~10 us);
    rotated groups' gate/up launches read it.  Down: the SwiGLU intermediate of a rotated expert gets R (9 x
    FWHT-256) inside the existing quantize-for-down step.
  - prompt (run_chunk, MMQ via the PTQ1_0 -> Q2_0 transcode, which stays lossless in the rotated basis): same two
    activation rotations per chunk.
  - CPU pool (ptq1_0_rows_q8_0): rotate the q8 activations on the host for rotated experts.
  Gate: a CPU reference (mesh_pack derotated weights x activations) vs each path, then the usual ppl / KL / HumanEval
  20/21 / tools 10/10 before any default changes.
B3. **Re-run the allocation with mesh as a fourth tier** (v41_alloc.py: cost = held-out error, budget = bytes): this
is where the bytes come from - the allocator decides, per expert, PTQ1_0 (ours) vs mesh vs Q2_0 vs Q3_K at a total
size at or BELOW today's, at quality at or ABOVE today's.  Needs the full calibration chain (v41_calib, ~12 min a
layer) with a mesh column, or the teacher-input screen for all 40 layers as a cheaper proxy.

## C. Order for the next GPU session

1. A0 (measure the sweep) and A1 (`--prompt-batch 16384`) - both cheap, A1 is a flag.
2. B1 on layers 0 and 5 (7 GB download) - decides whether B is worth the engine work.
3. Then A2 (keep the cache through small turns) and, if B1 is positive, B2/B3.
