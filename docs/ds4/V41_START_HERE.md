# Strata-DSV4.1 — START HERE

**A new session reads THIS file and nothing else before starting work.**  Everything below is current and measured.
Open the long files only for a specific question (see "Where the details live").  Last updated 2026-10-13 evening (session 6: DSpark D1 in the engine DONE; D2 designed, not coded).

## 1. What this is
Our own engine (`tools/v41/engine/`) running **DeepSeek-V4.1-Flash** (40 layers x 384 routed experts, top-6, Engram,
CSA/indexer attention, mHC) from **our own GSQ mixed-tier quant** (experts 42.8% ternary PTQ1_0 / 39.2% Q2_0 / 18.0%
Q3_K = 2.25 bpw; dense Q8_0) on **one RTX 4090 24 GB + Ryzen 7 5700X + 90 GB DDR4 + NVMe**.  Public: GitHub
`Indras-Mirror/Strata-DSV4.1` (local remote `ds4`, `git push ds4 deepseek4:main`, pushing is OK'd), HF
`IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored` (upload complete; card = `tools/v41/hf/README.md`).

**Mal's goal now: 50+ tok/s decode with DSpark speculative decoding + n-gram drafting + plain-decode work, plus vision.**

## 2. Numbers today
| | |
|---|---|
| Prefill, 20,480-token prompt (adopted default) | **387.6 tok/s one-shot / ~350 in `--serve`** at `--prompt-batch 12288` (52.8 s one-shot; 58.44 s serve at 512K with DSpark).  Was 318.9 / 306 at 8192.  The win is one fewer whole-file-tier sweep: 114.5 GB in 2 vs 157.2 GB in 3 (s13 run17, 2026-10-15, 3 runs a side).  On a shared card the runtime cap narrows the batch: 233.9 tok/s at batch 6428, 251.1 at 9026 |
| Prefill 8K / 16K (SUPERSEDED, batch 8192 era) | 244-248 / 288 tok/s, exact (MMQ default) - re-measure at the new default before quoting |
| Decode today (s13, serve flags, one-shot 256 tok) | DSpark **python 27.96 / prose 16.33 / code 26.68 tok/s**; plain decode 18.61 / 14.67 / 19.53 |
| Decode p256 x 256, forced tokens (identical work) | **26.4-27.5 tok/s = 37.5 ms/token** (28-29.5 on luckier token paths) |
| Quality | 4096-tok code ppl 2.3790 exact routing; decode biases (defaults) cost code 2.38 -> 2.87 |
| Head gate vs FP8 teacher (held-out) | ppl 5.645 vs 4.915, top-1 0.674 vs 0.701 |
| DSpark (offline harness, fixed) | **OUR engine states: 3.27 accepted drafts/pass = 4.27 tokens/pass** (378 steps; 43% full 5-blocks; first draft 88.6%; confidence calibrated 23/61/91/99%); teacher states 3.24 / 4.24 - our 2-bit drift costs nothing. By verify depth 1..5: 1.89 / 2.65 / 3.29 / 3.83 / 4.27 tokens/pass |
| Our hidden states vs teacher at layers 37-39 (exact routing, 639 tokens) | cosine **0.876-0.879** (large drift) |
| DSpark in the engine | **IMPLEMENTED and the `--serve` default** (`STRATA_SERVE_DSPARK`, sidecar Q2_K, depth 4, KEEP 64, conf 0.6): draft+verify passes with exact speculative sampling.  In live traffic 2.1-3.3 tokens/pass (e.g. "DSpark 88 passes, 2.90 tokens/pass"); the 4.27/pass figure above is the offline harness at depth 4 |
| Vision in the engine | **NOT implemented** (weights exported in the sidecar only) |

## 3. Per-token decode budget (37.5 ms; `STRATA_DEC_PROF=1` phase timer + nsys node trace)
- `attn_router` **17.8 ms** = the dense CUDA graph per layer (15.2) + router graph (1.1) + readback.  Inside: dense
  Q8_0 mat-vecs **8.94 ms at ~850 GB/s = 84% of peak** (7.6 GB/token incl. the LM head and the grouped
  `attn_output_a`, nrows 1024 x 8 channels) - at the bandwidth wall; **elementwise "zoo" 5.45 ms** = 3288 kernels/token
  of mean < 5 us (`k_bin_bcast`, `scale_f32`, `quantize_q8_1`, `rms_norm`, `concat`, `cpy`, `dsv4_hc_*`, sigmoid...);
  attention itself ~0.74 ms.
- `tier` **16.6 ms** = expert service: VRAM hits (~96%) + PCIe misses (~5.6/token, ~0.6 ms each) + NVMe file-tier
  misses (~2.6/token, **6.0 ms/token of reads**, on the critical path) + CPU pool (~1.8 ms) + device sum.  Expert
  kernels on the device ~3.7 ms (ternary `native_gu<143>` runs at only ~136 GB/s: ALU-bound trit decode).
- `finish` 1.9, `begin` (Engram) 0.9, `predict` 0.26.
- Structure: per layer `attn_router(l)` -> host reads top-6 ids -> `tier(l)` -> k-ordered device sum -> `attn_router(l+1)`.
  Strictly serial per layer by data dependency (the sum feeds the next layer); prefetch (`--pf-b 1.43`) is the only
  overlap and is tuned.

## 4. Dead ends - measured, do not retry
Slice C (selection on-device: the k-ordered sum needs host-served miss rows every layer) · C' (GPUDirect: no
nvidia-fs, gguf on ntfs3; resident file tier = 83 GB, impossible) · removing ggml-stream syncs (S4: the wait moves) ·
tier metadata packing / tier CUDA graph `STRATA_TIER_GRAPH` / demand-copy overlap `STRATA_DMA_OVERLAP` (all
bit-identical, all neutral: host calls under a busy device) · `--pf-b` other than 1.43 · bigger arena (62/65 GiB:
neutral) · speculative NVMe reads as implemented (`STRATA_SPEC_TOP`: tier -1.4 ms but its extra predict graph +1.4 and
decode-thread issuing +1.2 ate it) · attention-level levers (KV quant, window, MLA absorb, split-K: <= 0.7 ms total) ·
quarter-block ternary dot / `STRATA_EXP_MODE_T*` (~1%) · hard pruning (soft penalties only - Mal) · **striping the
file tier over a SECOND drive** (`STRATA_BLOB_MIRROR`: an 84 GiB mirror of the coldest experts on NVME1TB, the share
of a blob's parts read from it - the pattern does +33% in the microbench and NOTHING in the engine at arena 64 AND
at arena 32; the file tier is not bandwidth-bound. Session 12 block, 2026-10-15). · **a smarter VRAM policy than LRU**
(Belady ceiling 4-8%, every online policy within +-1% of LRU; admission window 4 is already the best) · **in-pass draft
pruning** (oracle -4.8%, a real rule -0.7%) · VMM for the prompt state (elastic columns already did it) · inclusive
VRAM/arena tiers (demotions cost ~0) · plain decode for low-acceptance text (DSpark wins on prose too).  Session 13 block.

## 5. State at shutdown (2026-10-13 ~06:00, session 5) - nothing is running; resume from here
- **DSpark teacher test: DONE.**  Teacher (FP8) states 0.437 accepted/pass vs ours 0.405 (`~/AI/v41-dev/dspark/
  accept_teacher.log`): our 2-bit drift is NOT the cause.  Production DSpark: SGLang accept length ~5 at bs=1 on
  V4-Pro, ~100% at bs=1 on V4-Flash (lmsys blog 2026-07-06; r/LocalLLaMA 1uqpers) => **our harness feeds DSpark
  wrongly**.  model.py's own DSpark loop is an untested smoke test ("nothing calls forward_spec").  Already ruled out
  in our harness: FP8 block size (32, honoured), RoPE/YaRN (compress ratio 0 = plain rope).  Teacher workdir
  `/media/Crucial1TB/v41-teacher`: `h/L00..L40.bin` (4 HC streams ENTERING each layer, 639 tokens = p256 prompt +
  383-token exact-routing generation), `pre/`, `tokens.npy`, `stats/head_argmax_L40.npy` (teacher greedy next token;
  teacher top-1 vs the text 0.857) - any main_hidden variant can be built on the CPU.
- **DSpark audit: SOLVED** (report `docs/ds4/V41_DSPARK_AUDIT.md`).  The harness drove DSpark exactly like
  SGLang / vLLM / model.py (main_hidden = HC-mean of the streams ENTERING 37/38/39; slot 0 = bonus token at p+1 + 4
  noise slots, drafts p+2..p+6; window = each block's own kv of main_x for accepted positions, non-causal block) - the
  bug was `load()`: checkpoint `mtp.N.attn.wo_a` is FP8 + 32x32 E8M0 scales but model.py builds it bf16, so the scale
  was dropped (~1000x too large).  Fixed in `tools/v41/dspark_accept.py` (default on; `--no-fp8-dequant` = old).
  Teacher states, 96 steps: **2.58 accepted (3.58 tokens/pass)**, accepted-length dist 0:23% 1:18% 2:10% 3:8% 4:8%
  5:32%, first draft 77% (78% vs teacher top-1).  The engine/GGUF never had the bug (our export dequantizes wo_a).
  **DONE 2026-10-13 afternoon: ours 3.27 accepted (4.27 tokens/pass) = teacher 3.24 -> no retraining; go to engine
  draft+verify.**  (Logs `~/AI/v41-dev/dspark/fixed_ours.log`, `fixed_teacher.log`; how they were run:)
  `python3 tools/v41/dspark_accept.py ~/AI/v41-dev/dspark/main_exact.bin ~/AI/v41-dev/dspark/gen_exact.txt
  --teacher-top1 /media/Crucial1TB/v41-teacher/stats/head_argmax_L40.npy` (OUR states - the number that sizes the
  engine work; our 37-39 states have cosine 0.876 to the teacher's) and the same with `--teacher
  /media/Crucial1TB/v41-teacher` for the full 378-step teacher number.  Also check the SIDECAR's `mtp.N.attn.wo_a`
  was exported with its block scale (`tools/v41/v41_sidecar.py`; its export check reported rel RMS 6e-3, so likely
  fine - verify before the engine uses it).
- **Vision slice 1 (encoder): DONE, committed** - `tools/v41/vision/ds41_vision.py` + `test_ds41_vision.py`; report
  `~/.quetza-data/conductor/reports/v41-vision-enc.md` (worker's gate PASS: per-row cosine >= 0.999 vs model.py on the
  original weights, 0.0 vs same-weights, protocol smoke as Strata's server drives it; CPU ~11 s/image quiet).
  **Conductor re-run of `python3 tools/v41/vision/test_ds41_vision.py` still pending** (needs a quiet CPU, ~8 min).
  Facts for slice 2: output file = int32 header {'SVE1', n, nx, ny, n_embd} + f32 [n][5120], `n` = the WHOLE span
  incl. delimiters, `nx=n ny=1`; the Strata engine keys image rows on Qwen's `kImagePad 248056` - V4.1 uses
  `image_token_id 129264`; `bias_vl` IS in our GGUF as `blk.N.exp_probs_b_vl.bias`; image tokens get no Engram and no
  attention-mask change; an image span must be prefilled in one chunk.
- **Dense requant (q4_K/q5_K): MEASURED, NOT ADOPTED (Mal: no code/agent quality drop; benchmark first).**  Forced-token
  speed (1 run each): `attn_router` q8 17.11 / q5_K 14.87 / q4_K 13.52 ms -> 26.64 / 27.73 / 29.75 tok/s.  Quality
  (`~/AI/v41-dev/dq/results.txt`, 4096-tok ppl + KL vs the arena-60 q8 logits): a q8 CONTROL at arena 40 already differs
  (code KL 0.0095 / agent KL 0.25, top-1 agreement 0.98 / 0.85 - routing flips on near-ties, accuracy unchanged) - so
  judge candidates against a same-arena control.  q4_K: code ppl +1.7% (2.393 -> 2.434), KL 0.026 (2.8x floor); agent
  ppl 14.87 -> 13.43 (more confident) with KL 0.48 (2x floor), accuracy 0.561 -> 0.559.  q4_K reason + the head-kept-Q8
  variant were cut off by the shutdown (`~/AI/v41-dev/dq/q4qual.sh`, resumable by editing the loop).
- **Folder cleanup done**: 12 old `~/AI/Strata-DS4-*` worker worktrees removed (branches kept in the repo; the wire
  worktree's 2 modified files = `git stash@{0}`), `~/AI/Strata-architectds` deleted (public clone).  Kept: Strata (its
  server is what serve_ds41.py runs), Strata-data, Strata-DS4-private, Strata-MiMo, Strata-GLM* (GLM's; GLM-rebase has
  382 uncommitted changes - Mal to decide), PolyStrata (GLM files in it).
- Repo/GitHub/HF: everything committed + pushed at shutdown (see `git log -5`); HF card has the 4-step quick start and
  the Engram placeholders `model-0004{7,8}-of-00048.safetensors.NEED-TO-DOWNLOAD-ENGRAM.txt`.

## 5c. Session 7 (2026-10-13 evening) - READ FIRST, supersedes 5b where they differ

### Session 11 (2026-10-14 21:00 - 2026-10-15, overnight) - DSpark conf trim, FP4 KV, serve turns 2x faster
**Read this block first.**  Mal's brief: make the NVMe reads and PCIe as efficient as possible, look for novel solutions,
then serve defaults + Quetza.  Logs / scripts: `~/AI/v41-dev/s11/` (run1-6b, gate*, e2e/).  Commits afbc3f3c, d91c69a7,
ee2845ca (+ serve defaults / docs).
- **DSpark gallocr fix VERIFIED** (crash2_reqs at 256K: 3/3 requests; one-shot tokens/pass unchanged 4.267).
- **Confidence-head verify trim** (`STRATA_DSPARK_CONF=0.6`, one-shot + serve; serve default now): the sidecar's
  `mtp.2.confidence_head` was computed and ignored.  It is well calibrated (p >= 0.9: 119/119 accepted; 0.3-0.6: 40-75%)
  and every verified draft costs ~20-27 ms of expert bytes (verify ms by width, docs: 72 / 91 / 118 / 135).  256 tok,
  x0.5 + aa + K=64, depth 4 -> tau 0.6: **prose 9.48 -> 17.25, code 18.80 -> 25.54, docs 29.89 -> 30.30, python
  26.57 -> 29.07** (tau 0.7 same within noise).  Exact routing: deterministic; it leaves depth 4's stream only at a
  0.0009-logit tie - and depth 2 / plain decode leave depth 4's stream EARLIER (#7 / #8 of 64): the greedy stream depends
  on the verify width (rounding), for DSpark itself - "DSpark output = plain greedy output" holds up to ties only.
- **FreeToken q* (CPU share of the misses, `STRATA_TIER_BATCH_PCIE`)**: 0.85 neutral (159.2 -> 158.9 ms/pass), 0.75
  -8%.  Our CPU pool spends ~1.1-1.4 ms per expert (PTQ1_0 ternary base-3 decode) vs ~0.4 ms a DMA - q* only pays
  with a ~3x faster ternary CPU kernel (a project; DRAM-bound ideal ~0.25 ms).  Parked.
- **Compression**: zstd on the expert region = 0.92 ratio (7.5%): not worth a decode stage.  Dropped.
- **NVMe** (PM9A1 Gen4 x4, ntfs3, 1226 extents = not fragmented; MDTS 128 KiB): O_DIRECT microbench 9.4 MB blobs:
  1 pread 5.4 GB/s, 8-way 4.2, 32-way 5.5.  Engine file-tier reads now **32 readers x 32 parts** (were 8 x 8,
  `STRATA_D3_READERS` / `STRATA_D3_PARTS`): -17% ms per blob, decode +4-5% (code 24.72 -> 25.66, prose 15.75 ->
  16.62).  NVME1TB has 16 GB free, so striping the file tier over two drives is impossible without freeing space.
- **`--kv-fp4` (BIT-EXACT, serve default)**: the compressed KV is fake-quantized to e2m1 x E4M3/16 (NVFP4's grid) and
  the indexer keys to e2m1 x 2^k/32 (MXFP4's) before they are stored as BF16.  Now stored as NVFP4 / MXFP4: exact
  packers in the ggml fork (`tools/v41/ggml/kv-fp4.patch`, applied).  `test_ds41_kvfp4 --cuda` 0 / 5.2M differ;
  engine gate (8000 tokens, exact routing, prestage pinned) last-row logits bitwise identical.  512K serve: free VRAM
  after the dense half 10.46 -> 11.59 GiB.  **Gotcha found on the way:** the prompt's prestage region is sized from
  FREE VRAM, so ANY VRAM change (KV size, ctx, desktop) moves prompt numerics (KL ~1e-4, same argmax) - pin
  `STRATA_PRESTAGE_GIB` when A/B-ing bitwise.
- **Serve turns** (`serve: prompt phases` stderr line): the DSpark seed tail (last 128 prompt tokens as 16-token EXACT
  passes, ~2 s each) was half of every turn.  Now a turn that goes to the layer-major batch anyway takes the tail into
  the batch and the seed rows are read off x_state (`Ds41Dense::read_state_cols`; `STRATA_SERVE_BATCH_TAIL=0` = old).
  512K, DSpark + conf 0.6 + kv-fp4: **fresh 8000 51.2 -> 37.9 s, +272 tokens 33.5 -> 18.8 s, +1000 40.2 -> 19.7 s**
  (+61 tokens = step path 8.4 s, unchanged); seed quality the same (tokens/pass 2.2-4.9), decode 22-30 tok/s.
  Then the **chunk-prefetch gate** (`STRATA_CHUNK_PF_MIN`, default 640 rows): the slabs stage a layer's WHOLE file tier
  before its routing is known - a loss for small turns (+272 routes ~39% of it).  Batch s on / off: fresh 8000 33.7 /
  38.8, +1000 19.2 / 22.0, +272 18.3 / 14.5 -> gated: **fresh 36.8 s, +272 14.7 s, +1000 19.4 s** per turn (session
  start: 51.2 / 33.5 / 40.2).
- **E2E on the production stack** (build-ds4-gpu + strata-ds41.json, port 8150): smoke OK, **HumanEval 20/21, tools
  10/10, agentic 3/3**, and a headless Quetza turn (write sum100.py, run it, report 5050) worked: 15.4K-token system
  prompt 50.5 s, tool-result turns of 17 / 27 / 89 tokens 1.9 / 2.3 / 7.2 s, decode 17-27 tok/s.  Re-run after the
  elastic columns + PTQ1_0 kernel + prefetch gate (03:30): again 20/21 / 10/10 / 3/3 + the Quetza turn, suite 858 ->
  792 s, decode 21-30 tok/s.  Third run (04:30, + wave parts, prefetch off, small-turn batches): again 20/21 / 10/10
  / 3/3 + the Quetza turn; suite **599 s** (first run 858 s); agent turns of 133 / 188 new tokens 7.1 / 7.8 s.
- **Decode vs expert-slot VRAM** (one-shot, `--vram-margin` 1.0 / 2.3 / 3.6 = 13.44 / 12.13 / 10.83 GiB of slots):
  verify ms python 131 / 136 / 145, prose 71 / 73 / 78, code 89 / 93 / 107 -> every 2.6 GiB is 10-20% of a pass.
- **Elastic state columns** (serve default, `STRATA_ELASTIC_COLS=0` = off): Ds41Dense's NC-column tensors (~0.31
  MiB a column) live in their own buffer; `set_columns()` resizes them between passes (drops every cached graph and
  both gallocrs - views keep their first addresses; generate re-reads `rdev1` = routed_dev() after a resize, which
  was the first crash).  Serve decodes with S = 512 columns, grows to the batch for a layer-major prompt.  512K:
  **11.59 -> 14.00 GiB free after the dense half, 1227 -> 1509 VRAM experts**, 4-turn bench decode +5-11%.  Gate
  (exact routing, no DSpark, prestage pinned, 8000 + 272 tokens): tokens identical on / off.
- **PTQ1_0 CPU kernel** (`ptq1_0_rows_q8_0`, src/kernels/cpu/ptq1_0.cpp; the pool takes it for ternary gate/up and
  down, `STRATA_NO_PTQ1_ROWS=1` = old): every digit SIMD-decoded once per block for all tokens, sub-block sums exact
  int32, F16C - **3.9x at 1 token, 6-8x at 2-3** (0.74 -> 2.90 GB/s a core).  ptq1_0_test ALL PASS.
- **q* re-test with it** (`STRATA_TIER_BATCH_PCIE`, + recurring-first selection, `STRATA_TIER_QSTAR_RECENCY`):
  verify ms summed over python / prose / code: 1.0 291.3, 0.5 275.9 (old order) / 274.3 (recency), 0.7 285.1 ->
  ~-5%, at the noise floor -> default stays 1.0.  FreeToken's own rule says the same: hybrid only when CPU GEMV
  bandwidth > 2x PCIe; here ~24 GB/s (8 cores x ~3) vs 21.6.
- **Where a small batch turn goes** (`serve: prompt phases` now prints dense / experts / logits): +272 tokens at 8K
  context = dense 3.2 s + experts 11.6 s, and the experts are the NVMe reads of the routed file tier (34.5 GB through
  the chunk's on-demand "wave" reader at 3.2 GB/s; the expert tier's own wall is ~1.1 s).  Wave reader threads 8 ->
  32 (`STRATA_WAVE_THREADS`, default now 32): batch 14.9 -> 13.6 s (3.8 GB/s).  Still below the drive (~5.4 GB/s in
  the microbench): each blob was 3 sequential span preads.  Now split into 8 parts read by different threads
  (`STRATA_WAVE_PARTS`, default 8): 3.60 -> **5.29 GB/s, batch 14.25 -> 10.68 s**, tokens identical (exact routing).
  Production smoke: +310-token turn batch 13.9 -> 11.4 s.  With reads at 5.3 GB/s the whole-layer chunk prefetch
  stopped paying at every size (gate 640 / never: fresh 8000 33.6 / 32.4 s, +1000 19.1 / 16.6 s, +272 10.5 / 10.5) ->
  **default never** (`STRATA_CHUNK_PF_MIN=<rows>` brings it back); its 2 x 2.71 GiB pinned slabs are no longer
  allocated: serve min MemAvailable 15 GiB (arena 64).  Arena 68 then: no visible decode gain, min 11 GiB -> stays 64.
- **Small turns** (`STRATA_SERVE_BATCH_MIN`, default 128 tokens): every turn up to 224 new tokens ran as exact 16-token
  passes (the 128-token DSpark tail + STEP_MAX 96).  512K, prompt s (+ 32-token decode) passes / batch: +30 5.0 (1.7)
  / 6.0 (2.0), +60 6.6 (1.7) / 7.0 (2.0), +90 9.7 (1.0) / 7.7 (1.7), **+150 17.6 (1.4) / 9.8 (1.6)**.  Decode right
  after a batch starts colder (the cache is re-seeded), so the cut sits above the ~80-token crossover.
- **Relay fix** (`~/AI/QuetzaCodetl/wrappers/tools/relay`, untracked file - live via ~/.local/bin/relay; skill text
  regenerated): `ask/reply/broadcast --name X` from a window with another name now gets an UNBOUND daemon instead of
  renaming the window (the session-10 gotcha).  Not committed in QuetzaCodetl (its tree has others' uncommitted work).
- **--prompt-batch trade** (serve, 512K): 8192 / 2048 / 1024 -> free VRAM 10.46 / 12.39 / 12.71 GiB but an 8000-token
  prompt 44.6 / 83.5 / 132.7 s, decode no visible gain -> keep 8192.  The NC-sized prompt state is ~0.31 MiB/column
  (~2.6 GiB): giving it back during decode needs VMM map/unmap (stable addresses: cached graphs keep view pointers) -
  queued, low priority (more slots did not show in decode).
- **Merge question (Mal)**: `~/AI/research/MERGE_REPORT.md` (deepseek-quetza pro, spot-checked): two object stores
  (Strata = main/qwen; Strata-DS4 = deepseek4/glm/mimo + worktrees), dense vs Niko1221 histories joined by grafts;
  glm <-> deepseek4 share base fb58e0db, 26 modified shared files, one real conflict (iq_kernels/native_mmvq split into
  fork/*.cuh).  Code has not diverged much; history has.  Monorepo on one trunk + tools/<model>/ + a shared kernel tier
  ~3-5 days.
- Relay: the overnight chronos wake works (cron 17 * * * * ds4-overnight-nudge.sh -> `relay wait` background task,
  re-armed after every wake; nudge fired from cron/systemd-run, never from the target window).

### Session 12 (2026-10-15 05:00-06:00) - second-drive mirror: MEASURED DEAD END (+ why); warm reseed; wave queue depth
Logs / scripts: `~/AI/v41-dev/s12/` (run1 mirror A/B, run2 serve turns + controls, run3 wave depth, run4 arena 32
mirror control), microbench `rb2.cpp` (the wave reader's pattern, no GPU).
- **Read this first: the 512K turn bench's run-to-run spread is 1.5-4.3%.**  The SAME config re-run back to back
  (`w0` vs `w0b`, identical env): fresh 8000 31.51 -> 31.98 s, +61 8.07 -> 8.42, +273 10.75 -> 11.12, +1001 16.65 ->
  17.06, and the short decodes swing 2.76 -> 3.09 s (+12%).  A single run pair cannot resolve anything smaller; the
  second run of a session is systematically slower.  Use >= 3 alternations, and quote both baselines.
- **Second-drive blob mirror - built, verified, and it does NOT pay** (`--build-mirror F [--mirror-gib G]`,
  `--check-mirror F`; runtime `STRATA_BLOB_MIRROR=<file>` + `STRATA_BLOB_MIRROR_SHARE` per mille of a mirrored blob's
  parts, default 0.28).  NVME1TB had 91 GB free, so 84.0 GiB of the 8817 coldest (layer, expert) - the whole 8241
  file tier + the coldest arena experts - were packed there in 81 s at 1.1 GB/s by the engine's own reader;
  `--check-mirror` re-reads every mirrored blob and compares it byte for byte (8817/8817 identical).  `read_part`
  sends the share's parts (at 8 parts, 0.28 = parts 2 and 5) to it, so both drives serve one blob at once.
  - **The pattern itself is fine** (rb2 = the engine's wave-reader pattern, 6.75 MiB blobs, 8 parts, 32 workers,
    random 4K offsets): model drive alone **6.22 GB/s**, mirror alone 3.26, both with 2/8 parts on the mirror
    **8.29 GB/s (+33%)**; 3/8 -> 7.95, 4/8 -> 6.19 (the mirror's own ceiling is the binding term above 3/8).
  - **In the engine: nothing.**  512K turns (mirror / no mirror / the re-run): fresh 8000 32.15 / 31.51 / 31.98,
    +273 11.23 / 10.75 / 11.12, +1001 17.14 / 16.65 / 17.06 - the mirror sits inside the spread.  One-shot 4096,
    bit-identical stream (code prompt: same 14586/191/46663/4721 expert counts): decode 9.89 vs 9.74 s (+1.5%), with
    the mirror drive demonstrably serving ~19 GB in the run (block counters on nvme1n1).  Positive control at
    `--arena-gib 32` (11647 file experts, the 271-token prompt reads 10234 blobs = 69 GB, 2x the production load):
    prompt 51.64 / 52.76 s, decode 11.54 / 12.03 s - still inside the spread.
  - **Why** (the useful part): the file tier is not bandwidth-bound in this engine.  The +273 turn reads 3161 file
    blobs (21.3 GB) inside a 7.45 s experts phase whose remainder is the arena experts' PCIe DMA + kernels + syncs;
    the wave reader's own rate is ~5.3-6.2 GB/s (rb2).  Splitting a blob across two drives adds a straggler (the
    mirror's per-part service time is ~2x the PM9A1's) and buys throughput the pipeline does not want.  A second
    drive only pays if the file tier must move 2x the bytes - and even arena 32 (2x the file-tier traffic) did not.
  - The 84 GiB mirror file was **deleted 2026-10-15** (NVME1TB 6.5 -> 91 GB free).  If it is ever wanted again:
    `build-ds4-dspark/ds41_generate -m "/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf" --rank "/media/mal/SSD
    NVME/AI/V41-GSQ/rank.bin" --build-mirror "/media/mal/NVME1TB/Models/V41-Mirror/V41-GSQ.experts.mirror"
    --mirror-gib 85` (81 s; `--check-mirror` then compares it blob by blob).
- **warm_reseed** (`release_cache` remembers the (layer, expert) that held VRAM slots, most-recently-used first; the
  next seed fills the first `STRATA_WARM_SHARE` per mille (default 500) of the slots from that list instead of the
  global rank): coded, measured, and **default OFF** (`STRATA_WARM_RESEED=1` to try it).  512K: +273 10.48 vs
  10.75 / 11.12, +1001 16.88 vs 16.65 / 17.06, decode right after a batch 3.12 s / 192 file reads vs 2.76 / 144 -
  inside the spread, and the seed reads 48 MORE file-tier blobs, because the LRU set at release is largely the
  experts that just missed.  Do not build on it without a >= 3-run measurement.
- **Wave-reader queue depth** (`STRATA_WAVE_THREADS` / `STRATA_WAVE_PARTS`): 64 x 16 gave fresh 8000-token batch
  **29.56 s** in one run against 31.51 / 31.98 baselines - and **31.52 s when repeated**, i.e. the -8% was a false
  positive (the wrapper default is back to the code's 32 x 8; the lesson is the bench's single-run spread).  The
  engine's own reader timer (`MIMO_CHUNK_PROF=1`, the `chunk prof` line, same one-shot 271-token prompt, 22.7 GB of
  routed file tier) says why there was nothing to get: 64 x 16 = 22.7 GB in 4281 ms of reader time = **5.29 GB/s**,
  32 x 8 = 4381 ms = **5.17 GB/s** (+2.3%), prompt wall 30.24 vs 30.34 s.  5.2-5.3 GB/s is what this pattern gets in
  the engine (rb2, the same pattern with no GPU, no engram reads and no DSpark sidecar on the drive: 6.22), the
  caller waits on reads for only 35 / 159 ms of a 30 s prompt, and the microbench's queue-depth effect (rb: 8
  outstanding reads 4.72 GB/s, 16 -> 5.22; rb2 32 workers x 8 parts = 256 outstanding 6.22) is already saturated at
  32 x 8.  Do not re-tune this without >= 3 runs a side.
- **The batch turn's "experts" phase is NOT the file tier.**  `chunk prof` on the 271-token prompt: 40 run_chunk
  calls, caller wall **589 ms** (waits on reads 35 + wave/copy waits 0 + final sync 18 + other host 536), reader
  threads 5290 ms, GPU grouped kernels 14 ms - while the prompt's own "experts" timestamp reads 6.9-9.5 s.  The
  reads run on the reader threads and the caller barely waits: what the experts phase actually spends its time on is
  the arena experts' PCIe DMA / prestage / the device-side sum, i.e. the documented PCIe wall (~70 of ~130 ms a
  python verify pass) - which is why VRAM slots (2.6 GiB = 10-20% of a pass) and the elastic columns, not the disk,
  are the levers that have paid.
- **O_DIRECT bounce + memcpy tax: 3%**, not worth a direct-read path (rb `bounce` 0/1: 4.72 / 4.58 GB/s at P=8,
  5.22 / 5.09 at P=16 - `read_direct` reads into a bounce buffer and memcpys because a blob's spans are only
  4K-aligned at the ends).
- **One-shot prompt accounting caveat** (for anyone reading the one-shot bench's prompt numbers): the prompt wall
  includes the DSpark seed tail - the last `min(128, L-1)` prompt tokens run through `pass_v` as 16-token passes
  AFTER the batch, and their tier time is not in `fd_dense` / `fd_tier`.  A 271-token python prompt: 30.32 s wall =
  10.4 s of feed (dense 3.4 + experts 6.9 + logits 0.1) + the ~20 s tail (+ the deferred VRAM seed).  In `--serve`
  a batch turn takes that tail from the batch itself, so the serve numbers are not comparable.
- **Housekeeping 2026-10-15** (Mal's go): the 84 GiB mirror file is deleted (see above) and session 11's hourly
  chronos nudge cron is **removed** (`crontab -l | grep -v ds4-overnight-nudge | crontab -`; the pre-removal copy is
  at `~/.quetza-data/crontab-backup-2026-10-15.txt`, restore with `crontab <that file>`) - re-arm it if an overnight
  run is planned.  Also: a foreign headless Chrome (agent-browser, not ours) holds a 51 MiB CUDA context, so the
  harnesses' `gpu_busy()` ("any compute app") never clears - gate on VRAM used (>= 2 GB = busy) or on
  `pgrep -x ds41_generate`.

### Session 13b (2026-10-15 17:00-) - real-harness findings, PARK/UNPARK, prompt + ternary + brand prep
**Read this block first.**  Mal ran a real Quetza session on the production wrapper; full numbers + the plan in
[V41_NEXT_PREFILL_TERNARY.md](V41_NEXT_PREFILL_TERNARY.md).  Then:
- **Side requests destroyed the conversation cache (FIXED in code, gate pending).**  A 454-token Quetza side call
  between main turns wrote positions 0..453 over the main conversation's position-indexed caches (`comp` compressed KV,
  `idxk` indexer keys) and its snapshot rings, so the next main turn re-read the whole 20,736-token system prompt:
  **69.5 s** each time.  `Ds41Dense::park(n) / unpark() / drop_park()` (host copies of the snapshot rings + rows
  [0, n] and the spare row of every `comp` / `idxk`) + the serve loop: an unrelated request parks a conversation of >=
  `STRATA_SERVE_PARK_MIN` (2048) tokens if prompt + max tokens <= `STRATA_SERVE_PARK_MAX` (65536 positions, ~1 GB
  host); a later request that continues the parked tokens unparks it (resumes through restore(), DSpark window reset);
  a request writing past the saved rows drops the park.  `STRATA_SERVE_PARK=0` = old behaviour.  Logs: `serve: parked
  the N-token conversation`, `serve: unparked ...`.  **Gate + timing in `~/AI/v41-dev/s13/run3.sh`** (exact routing:
  main 8K -> continuation, vs main 8K -> unrelated side request -> the same continuation: tokens must be IDENTICAL;
  and the continuation's prompt time with park on / off).  Production build-ds4-gpu is NOT rebuilt with it yet - do
  that (and restart the wrapper) only after the gate says IDENTICAL.
- **Prompt path:** run3 also sweeps `--prompt-batch 8192 / 16384 / 32768` on a 20,480-token one-shot prompt with
  `MIMO_CHUNK_PROF=1` (2 runs each): 1 sweep instead of 3 is the cheapest step toward sybil's 800 tok/s.  If 16384 /
  32768 fit and win, set it in `tools/v41/serve/strata-ds41.json` (`--prompt-batch`) - note the serve default batch is
  the engine's 8192 today.  Next code step after that: keep the VRAM expert cache through small batch turns (A2 in the
  plan doc - the log says `experts 0 hit` and `cache open` up to 4.3 s on every batch turn).
- **meshapplied ternary:** run3 downloads layers 0 + 5 (7 GB, /media/mal/NVME1TB/mesh-ternary), runs `mesh_pack.py
  check` (lossless PTQ1_0 repack + rotation algebra) and `mesh_screen.py layer 5 / 0` + `summary` (ours vs mesh per
  type).  mesh_screen has never run: if it fails, the likely spots are the hook's use of model.py's Expert (`w1/w3/w2`)
  and C.K.dequant_fp4_weight - same calls as v41_calib.screen_expert.
- **RESULTS (run3, 17:09-17:22):** (a) **the prompt path is NVMe-bound** - 20,480-token one-shot at batch 8192: 62.35 s
  = 328.5 tok/s, experts 40.4 s of which **file wait 27.0 s** (chunk prof: reader wall 35.8 s, GPU grouped kernels
  0.4 s).  So the road to ~800 tok/s is the file tier (bytes per sweep, read pattern, overlap), not PCIe or kernels.
  (b) `--prompt-batch 16384` OOMs (a 1.42 GiB alloc) and 32768 OOMs in `c_stage`: bigger batches need the chunk /
  stage buffers sized down (or the VRAM expert cache kept out), not just the flag.  (c) mesh: `mesh_pack.py check`
  PASSES on layer 5 (381 of 384 experts ternary; rotation verified; PTQ1_0 repack lossless); layer 0 has a manifest
  but no layer file (FP4 upstream); the screen is re-queued as `~/AI/v41-dev/s13/run3c.sh` -> run3c.out (layer 5).
  Park gate rerun: run3b.out.
- **PARK GATE PASSED (run3b, exact routing): continuation tokens IDENTICAL**; the continuation after an unrelated
  side request reused 8,000 tokens in 8.15 s vs 28.99 s with STRATA_SERVE_PARK=0.  **Production build-ds4-gpu rebuilt
  with park (2026-10-15 ~17:50)** - restart strata-ds41-quetza to use it.
- **meshapplied ternary: CLOSED, ours is better.**  Layer 5 screen (teacher inputs, 268 experts): where ours is PTQ1_0
  (same 1.75 bpw) mesh's gate+up error is 1.36x ours (mesh wins 19% of experts), down 0.985x (a tie); vs our Q2_0 /
  Q3_K 1.8x / 11-16x worse.  Our GPTQ ternary already beats their Hadamard GPTQ ternary at equal bits - no swap.
- **Prompt speed outlook:** NVMe-bound (file wait 27 of 40 s experts, 20K prompt, 328 tok/s).  One sweep per prompt
  (batch = prompt, needs the chunk / c_stage buffers sized down so 16K-32K fit VRAM) ~ 500-550 tok/s; ~800 also needs
  the dense half faster (21.9 s of a 20K prompt).
- **PROMPT SPEED, ADOPTED (run4 + run5, 20,480-token prompt):** batch 8192 / prestage 3.0 = 328 tok/s one-shot,
  60.4 s serve; **batch 12288 / prestage 1.5 = 382 tok/s one-shot, 53.4 s serve at 512K** (dense 21.9 -> 15.8 s,
  file wait 26.9 -> 20.1 s: one sweep fewer, and the dense half is faster at width).  Now the serve default
  (strata-ds41.json --prompt-batch 12288, ds41-engine.sh STRATA_PRESTAGE_GIB 1.5).  16384 / 20480 OOM (a 1.42 GiB
  cudaMalloc, even at prestage 0.75) - find that buffer next (the dense graph's compute buffer at width 16384 is the
  likely one) - 16384 would make a 16K prompt one sweep and a 20K one ~420-450 tok/s.  **Second-drive mirror: dead
  for good** (A/B 2x2 at batch 8192: 62.5 / 62.7 vs 63.5 / 62.6 s; it trims file wait ~3 s but the total does not
  move) - the read pipeline, not drive bandwidth, is the limit; deleted again.
- **Production config confirmed (run6, DSpark ON as in ds41-engine.sh, 512K, 20K prompt):** batch 8192 / prestage
  3.0 = 66.85 s (306 tok/s); **12288 / 1.5 = 58.44 s (350 tok/s, +14%)**; 12288 / 0.75 = 59.20 s (no better).  The
  same prompt WITHOUT DSpark = 53.4 s: the draft head's 2.91 GB on the card costs ~5 s of prompt.  NEXT levers for
  400-500: (1) park the DSpark weights in host RAM during a layer-major batch (2.9 GB, ~0.15 s each way) -> DSpark
  serve at the no-DSpark speed; (2) find the 1.42 GiB cudaMalloc that OOMs batch 16384 (it OOMs WITHOUT DSpark too -
  likely the dense graph's compute buffer or chunk c_parts at width 16384; ~89 KiB a column); (3) the read pipeline
  (file wait ~20 s of the 54 s).
- **PRODUCTION HARNESS CHECK (run8, 19:30, the real HTTP stack) - three findings, batch 12288 REVERTED:**
  (1) **VRAM: Quetza's Whisper STT server (`quetza-stt-whisper --port 5101`, 2.5 GB VRAM) shares the card.**  With it
  up, batch 12288 OOMs (the 1.42 GiB graph buffer) -> production is back on 8192 / prestage default; 12288 is opt-in
  (`STRATA_PROMPT_BATCH=12288 STRATA_PRESTAGE_GIB=1.5 strata-ds41-quetza` when the GPU is otherwise free).  Worse:
  even at 8192 a SECOND long prompt OOMed (612 MiB) after DSpark + the VRAM cache were seeded while Whisper sat on the
  card -> the engine must size from free VRAM at prompt time and survive an OOM (release the expert cache / shrink
  the batch and retry) instead of failing the request (the server answered HTTP 400).  This is the top robustness
  item: a desktop always has other GPU users.
  (2) **Park did NOT unpark in this harness:** "parked the 21349-token conversation", then the continuation (23,606
  tokens) had no exact 21,349-token prefix match, so it re-read everything.  Likely cause: the chat template re-renders
  the end of turn 1 differently once the assistant turn exists (generation prefix / thinking tags), so a snapshot at
  the prompt end is not a prefix of the next prompt.  In Mal's live session normal turn-to-turn reuse DID work
  (lcp >= snap), and run8's assistant content was empty (200 max tokens all spent thinking), so it may be the
  harness - check a real Quetza session's log for "unparked".  Robust fix: resume at the longest common prefix even
  when it is BELOW the snapshot (positional caches allow it; re-feed the last window's tokens to rebuild the ring).
  (3) Timing A/Bs from 19:15 on are invalid: load average 18-23 (other jobs + video renders) - run7's prefetch test
  is inconclusive (and its 2 x 2.71 GiB pinned slabs got the engine memguard-killed under RAM pressure).
- **Brand (Xolotl):** `docs/brand/` - XOLOTL_BRAND.md (rationale, taglines, palette, README header, rename advice:
  rename the GitHub repo IN PLACE, stars/forks/issues stay and old URLs redirect; keep "Strata" in description + topics
  for search), `gen_openrouter.py` (OpenRouter google/gemini-3-pro-image; `-n` takes, `--model`), 12 images in
  `docs/brand/out/` (picks: logo.2, wordmark-banner.2, hero.2, tiers.2, axolotl.2, twins.1).  Mal reviews before any
  rename; the brand files are deliberately NOT pushed yet.
- Real-harness decode 16-27 tok/s (2.1-3.3 tokens/pass on thinking text); a 66-token tool result 7.3 s; 1.2-2.2K-token
  results 17-23 s (the per-batch expert sweep floor ~12-15 s); a 10K one 46 s.

### Session 13c (2026-10-15 19:40-22:00) - a SHARED card: VRAM-aware batch width, cache seed, chunk stage, park/unpark
**Read this block first.**  Mal's brief: keep working on prefill and make sure it works in the harness.  This closes
the item the 13b block left on top - a desktop always has other GPU users (Whisper STT), and the engine FAILED the
request instead of degrading.
- **PARK/UNPARK FIXED (the other open 13b item): `STRATA_SERVE_PARK_PARTIAL` (default on).**  A continuation that
  matches the parked conversation one token short of its end used to be REJECTED - the code demanded an exact prefix -
  so an unrelated side call threw the whole cache away (run15 turn 2: 23,531 tokens re-read in **265.9 s**,
  `reused=0`).  New logging says exactly why: `lp=21165` of a 21,166-token park, i.e. the chat template re-renders
  ONE token (the 13b block's hypothesis, now measured).  A partial unpark restores the park and re-feeds from `lp`:
  **run19 turn 2 reused 21,165 tokens (2,326 fed), 0 errors**, on the commonest agent pattern there is (a tool result
  continuing a long conversation).  `STRATA_SERVE_PARK_PARTIAL=0` = old behaviour.
  CAVEAT on the timing: run15's 265.9 s for that same turn was measured at a CAPPED batch 1114 on a loaded box, and
  run19's 24.7 s at batch 5837 on a quiet one - so the wall-clock delta (10.8x) is NOT a clean measure of this fix.
  The proof is the mechanism: `reused=0` before, `reused=21165` after, i.e. a 23,531-token prompt became 2,326 tokens
  of prefill.  A like-for-like timing needs both arms on the same box at the same width.
  The reuse log now prints `park= lp=` too.  Gate: `~/AI/v41-dev/s13/run19*`.
- **Reproduced first (run10a/run11a: the OLD binary + a foreign process holding 4 GiB of VRAM - Whisper's role):
  HTTP 400 on the FIRST turn.**  "allocating 2633.44 MiB (12.5 GiB held) / 2193.81 MiB (4 GiB held) on device 0:
  cudaMalloc failed".  Also reproduced: the seed asking **142.38 GiB** ("ExpertCache: 152881735680 slots x 1 B =
  142.38 GiB, but only 0.53 GiB of VRAM is free") and "ds4_moe: c_stage: out of memory" (-> HTTP 503).
- **Three fixes** (tools/v41/engine/ds41_{generate,moe}.{cpp,hpp}, +99/-8; STRATA_VRAM_TRACE=1 is the trace):
  1. `--serve`: the layer-major batch width is sized from the VRAM actually free (`pb_cap`), and `feed()` retries at
     HALF the width if a column/graph allocation still does not fit.  The cost model is CALIBRATED on four traced
     points: `need(B) ~= 4.3 GiB + 0.95 MiB * B` - a free card (>= 13 GiB free) keeps the full `--prompt-batch`, a
     tight one degrades instead of failing.
  2. The chunk stage (two `MIMO_CHUNK_STAGE_MIB` halves, 1 GiB each: a hard 2 GiB demand before) is clamped from free
     VRAM with a 16 MiB floor; `per_half` then gives more, smaller waves.  c_stage can no longer OOM.
  3. The VRAM cache seed: a slot budget of 0 is the tier's "NO LIMIT", and the AUTO path passed it whenever free <=
     `--vram-margin` (i.e. on any busy card) - so the seed asked for the whole model's experts.  It now floors/skips
     the cache (arena + file tier serve the decode) and retries ONCE with the cache off if the seed still fails:
     never a failed request.  New `Ds41MoeTier::set_cache_enabled()` (`no_cache` had no writer before).
- **GATE (the real HTTP stack: 3 Quetza-like turns + the s11 smoke, a foreign 4 GiB holder resident throughout):
  3/3 turns complete, 0 engine errors** (run15, logs ~/AI/v41-dev/s13/run15*) - against HTTP 400 on turn 1 for the
  same traffic on the old binary.  Turn 1 21,204 tok at the capped B=4281: 105.2 s (201.6 tok/s); turn 2 23,531 tok
  at B=1114: 265.9 s (88.5 tok/s).  Free card (run9, same binary): 3/3 turns, 0 errors, turn 1 86.8 s at the full
  B=8192.  The timings are inflated by a load-17..38 box (foreign nvcc + odin + a render): the run establishes
  correctness, not speed.
- **Why a narrow batch is expensive (the read-path link):** one layer-major sweep streams the WHOLE file tier of every
  layer the batch routes through - ~82 GB per 40-layer sweep - so a 20K prompt at 8192 = **157.3 GB in 2 sweeps**
  (measured, run4's profiler).  A narrower batch = MORE sweeps = more bytes (B=1114 -> 22 sweeps, hence 88 tok/s).
  The lever is therefore VRAM (batch width), not the SSD.
- **Residual (UNCONFIRMED identity):** a FIXED **~2.19 GiB** graph allocation appears at batch layer ~2, byte-identical
  at B=2411 and B=8192 (so it is NOT the piece buffers, which scale with B).  It is why `need()` carries a 4.3 GiB
  fixed term.  Name it (nsys / a gallocr log) and clamp it -> a wider batch on a busy card.
- **Below ~5 GiB free a layer-major batch cannot run at all** (4.3 GiB fixed + the columns): that request fails fast
  at c_stage.  The card's own dense half + KV is ~10.8 GiB, so a 12.5 GiB foreign holder leaves nothing - a sizing
  limit, not a bug to fix.
- **Read path: MEASURED at prompt size, and it is exhausted (Mal's ask: NVMe reads for decode AND prefill).**
  run16 (20:52-21:08, one-shot 20,480-token prompt, build-ds4-dspark, load ~2, sides alternating):
  base 320.9 tok/s / **5.16 GB/s** of reader time; `STRATA_WAVE_PARTS=4` 316.5 / 4.96; `=16` 320.9 / 5.15.
  So parts 4/16 = **0%** - inside the 1.5-4.3% run-to-run noise - and s11's 32-thread x 8-part tuning stands.
  The parts runs ARE token-identical (`10699 2105 6321 11`) once the memguard banner (it embeds a PID) is excluded:
  my first identity gate compared banners, which is why it printed DIFFERS - fix in `~/AI/v41-dev/s13/run16.sh`.
  **`STRATA_READ_SORTED` (issue a wave's parts in FILE-OFFSET order) was TRIED and SEGFAULTED** (exit 139, inside
  the batch, 21:00) - the code is REVERTED, `blob_offset()` with it.  Not re-attempted blind: the source's own note
  puts the whole scattered-vs-dd gap at ~9% (4.65 vs 5.1 GB/s), i.e. near that noise floor, and no GPU window was
  left to debug it.  **Conclusion: at prompt size the read path has no knob left** - the reader does 5.08-5.16 GB/s
  against the microbench's 6.22, and the PREFILL lever is the batch WIDTH (each layer-major sweep re-reads the whole
  file tier: ~82 GB), which is the VRAM work above.  Still untested at prompt size: `STRATA_WAVE_THREADS` 48/64.
- **`STRATA_VRAM_TRACE=1` prints the state tensors' sizes** (run17, run20): the state buffer is **2.57 GiB at
  NC=8192**, **3.86 at 12288**, **5.14 at 16384** - linear in NC at **~0.321 MiB a column**.
- **CORRECTION (2026-10-16, run20): the fixed ~2.19 GiB residual at :383 is NOT this per-column state.**  The earlier
  note here read 2.57 - 0.19 - 0.19 = 2.19 GiB (the non-engram part of the state at NC=8192) as the answer to the
  fixed allocation - but that is the one thing the fixed term cannot be: the state SCALES with the batch, the fixed
  allocation does not (they only coincide in value at NC=8192).  Measured tonight on a clean card, full width
  reached (`first batch B=16384`): the 16384 batch OOM'd asking for exactly **1424.31 MiB** -
  `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1424.31 MiB ... cudaMalloc failed` with 0.36 GiB free - and
  1424 MiB is the value the `ds41_generate.cpp` comment records for **8192**.  So the pass-graph buffer is
  **B-independent** and the cap formula's "graph 0.17 MiB a column" term is a mis-derivation (1424 / 8192).  The same
  four calibrated points fit just as well as `need(B) ~= 5.7 GiB + 0.78 MiB * B` - the aggregate free-VRAM fit cannot
  tell the two decompositions apart, but the OOM's allocation size can.
- **B=16384 fails by ~1.03 GiB** (run20/b16384): the per-column state at 16384 (5.14 GiB, +1.28 over 12288) is what
  eats the reserve the fixed 1424 MiB graph buffer needs.  12288 still passes (line, run20/b12288).
- **`i_emb`/`selm` are bit-exact as F16 and `i_eng` is near-loss-free - and together they are NOT enough.**  `i_emb`
  is written from F16 rows (`ds41_dense.cpp`: `fp16_to_fp32`) and its only consumer cpy-casts back to F32, so F16
  round-trips exactly; `selm` is 0/-inf and is already cast to F16 for the flash-attn mask (`ggml_cast(... F16)`),
  so F16 storage is exactly what it becomes anyway; `i_eng` comes from an e4m3 engram (F16 has more mantissa).  Per
  column: `i_emb` -10,240 B, `i_eng` -24,576, `selm` -8,192 = **0.041 MiB a column = 0.66 GiB at 16384** - under
  the 1.03 GiB deficit, so these alone do not get 16384.  This is the FALLBACK if the 1 GiB speed knob above costs too
  much reader throughput; the cheapest sufficient F16 change is `x_state` + `hap` (0.078 MiB a column = 1.28 GiB at
  16384), which is also 48% of the state.
- **NAMED (run21, `nsys --trace=cuda --cuda-memory-usage=true`, one b8192 leg): the batch's VRAM is plain
  `cudaMalloc`, and the 1424 MiB buffer that OOM'd at 16384 is a GRAPH buffer - transient, allocated at t=61.91 s and
  freed 40 ms later, B-independent because the graph is built per 512-column sub-batch.**  The full >1 GiB picture:
  8332 MiB dense weights + **2633 MiB state at NC=8192, allocated at INIT and held** (so the one-shot path reserves
  the state at NC=NCmax=B before the batch - it is NOT elastic like `--serve`), 65532 MiB = **`cudaHostRegister` of
  the 64 GiB HOST arena, not VRAM**, 1536 MiB prestage, and graph buffers of 1424.3 / 2193.8 / 2194.8 / 2196.8 /
  2276.3 MiB (the 1424.3 and the first two ~2.19 are freed within 20-60 ms; the 2196.8 and 2276.3 are held).  Held
  through the batch: 1417.5 + 1025 + 1025 + 1015.7 + 960.0 + 1536 (prestage) + ~2196 = **~8.0 GiB**, which is the
  drop the VRAM trace shows, and none of it is the engram or the per-column state.
- **So 16384's ~1.05 GiB deficit is a SPEED-knob problem, not a precision problem.**  `STRATA_PRESTAGE_GIB` (1.5,
  allocated at t=59.4, hundreds of ms before the graph at t=61.9) and `MIMO_CHUNK_STAGE_MIB` (1024, the 2 x 1 GiB
  halves at t=57-59) are both pure throughput knobs sized from free VRAM.  Dropping either by ~1 GiB frees it before
  the graph alloc needs it.  The cost is reader speed (prestage 1.5 is what made 12288 a 2-sweep batch), so this is a
  real trade - but it needs NO KL/ppl gate, unlike the F16 state change below, which is the fallback if the speed
  knob costs too much.
- **The state's per-column cost (0.321 MiB, 0.95 with the batch's other buffers):** `x_state` [5120,4] 24.3%,
  `hap` 24.3%, `i_eng` x2 14.6%, `sel`/`selm` x8 idx layers 9.7%, `shexp`/`o_fn`/`routed`/`i_emb` 6.1% each,
  `cand` [2048] 2.4%.  Shrinking the WIDER batch means shrinking those - starting with the two 24.3% residuals.
- **A/B DONE AND ADOPTED (run17, 21:13-21:39, quiet box: load ~2, 3 runs a side, spread < 0.5%):
  `--prompt-batch 12288` beats 8192 by +21.5% on a 20,480-token prompt.**  8192: 64.39 / 63.99 / 64.28 s =
  318.0 / 320.1 / 318.6 tok/s, **157.2 GB** of file-tier reads in 3 sweeps (dense 20.5 s, file wait 26.9 s, 14917
  file).  12288: 52.72 / 52.91 / 52.88 s = **388.5 / 387.1 / 387.3 tok/s**, **114.5 GB** in 2 sweeps (dense 15.3 s,
  file wait 20.4 s, 10942 file).  The reader does the same 4.95-5.15 GB/s either way - the whole win is one fewer
  whole-file-tier sweep.  **12288 is now the serve default** (`strata-ds41.json` `--prompt-batch 12288` +
  `STRATA_PRESTAGE_GIB=1.5` in `ds41-engine.sh`) - safe to default because the runtime cap narrows the batch on a
  busy card instead of OOMing (run15's gate).  **SERVE-STACK SMOKE (run18, 21:41-21:43, the adopted default through
  serve_ds41.py): PASSED** - with another process holding 2.5 GB the engine logged
  `serve: VRAM 11.26 GiB free -> prompt batch 6428 (of 12288)` (the cap working live), ran 21,138 tokens in 90.35 s
  = 233.9 tok/s, the side call in 8.4 s, and reported **0 engine errors, 0 server errors, 0 HTTP 400s**.  That is the
  quantified cost of a 2.5 GB neighbour: 233.9 vs 387 tok/s at full width.
  The A/B itself needed >= 23 GiB free VRAM because the ONE-SHOT path has no runtime cap (that is `--serve` only):
  its dense half takes ~12 GiB and the un-capped batch ~12 GiB more (at 9.5 GiB free that allocation fails, run13).
  What cleared the card was xolotl-video's stray `quetza-stt-whisper` (2.5 GB) - the same consumer that tipped
  run8 over at the start of this session.

### Session 13 (2026-10-15 15:00-17:00) - the three levers CLOSED by trace replay (no engine change pays)
**Read this block first.**  Mal's pick (session 12): in-pass draft pruning, a smarter VRAM policy than LRU, the VMM
refactor.  Method: instead of building each one, record the verify passes' routing once and replay it offline -
`STRATA_ROUTE_TRACE=<file>` (measurement only, inert when unset: `Ds41MoeTier::trace_open` / `trace_mark`) +
`tools/v41/sim/route_sim.py` (`--policy`, `--prune`).  Logs / traces / scripts: `~/AI/v41-dev/s13/` (run1 = the 3
traces, run2 = plain decode, sim1.txt / sim2.txt, swapreg.py).  Serve flags (x0.5 + file 0.3 + arena-adapt + arena 64
+ kv-fp4, DSpark depth 4 + KEEP 64 + conf 0.6), one-shot, 256 tokens: python 27.96 / prose 16.33 / code 26.68 tok/s.
- **Verify-pass cost model** (345 passes, R2 0.977): **verify ms = 25.7 + 5.7 x rows + 0.49 x PCIe misses + 2.93 x file
  misses**.  A python pass (4.5 rows, 125 PCIe, 8 file) = 26 + 26 + 61 + 23 ms: PCIe is ~45%, the file tier ~17%.
- **The replay is calibrated**: the simulator's per-(layer, class) LRU with the 4-call admission window, arena_adapt and
  the D3 prefetch slots (a miss the engine served from a prefetch slot counts as a hit and is LRU-admitted, as in
  gpu_run_n_ex) reproduces the engine: python PCIe 31.89 vs 31.85 observed per token, hit 77.9 vs 78.0%; prose 49.64 /
  49.57; code 25.96 / 26.07.
- **(2) VRAM replacement policy: CLOSED.**  Belady's MIN with bypass (reads the trace ahead = the ceiling of ANY
  policy) cuts PCIe misses/token 31.9 -> 26.5 (python), 49.6 -> 45.4 (prose), 26.0 -> 19.8 (code) = **-2.1 to -3.0 ms
  of a 35-59 ms verify-ms/token, i.e. a 4-8% ceiling**.  Online policies get none of it: admission window 1 / 2 / 8 /
  inf all worse than 4 (the current default is the best LRU); LRU-2 +0.1-0.2 ms; LFU-decay -0.4 / -0.2 / +0.1 ms (decay
  0.9-0.995 identical); TinyLFU admission -0.2 / +0.6 / +0.5 ms (it refuses useful admissions).  DeepSeek's routing is
  too flat for recency or frequency to predict a layer's next use, and a victim's next use is a whole pass away - the
  prefetch predictor (same pass, next layers) cannot rank victims.  **Do not retry without a NEW signal of next-pass
  routing.**
- **(1) In-pass draft pruning: CLOSED.**  Dropping a pass's last draft row costs its unique misses (experts no earlier
  row routes) + 5.7 ms/row; it loses a token only when every verified draft is accepted.  Oracle (knows each row's full
  cost AND its acceptance): **-4.8% verify-ms/token** over all 3 prompts.  A real rule (drop at layer Lc if the row's
  observed unique-miss cost x (40/Lc - 1) > theta x p_cum x ms/token; Lc 2/4/8/12, theta 0.5-3, best picked per prompt
  in hindsight): best -0.7% overall (python -2.0%, prose -0.3%, code -0.4%) - inside the bench's 1.5-4.3% spread.  The
  row's early-layer cost predicts its later cost poorly (corr 0.21 at Lc 2, 0.50 at Lc 8), and the confidence trim
  (conf 0.6) already removed the cheap wins.
- **(3) VMM refactor for the prompt state: ALREADY DONE (by session 11's elastic columns).**  `set_columns()` drops the
  cached graphs and reallocates; serve decodes with S = 512 columns (~160 MiB) - the 11.59 -> 14.00 GiB / 1227 -> 1509
  slots of session 11 IS the 2.6 GiB coming back.  The "needs VMM" note predated elastic columns.  VMM would only save
  the graph rebuild per resize.
- **VRAM swap demotions are free**: adding the replayed LRU swaps per pass (15 a pass) to the cost model gives -0.13
  ms/swap (R2 0.9771 -> 0.9779) - the D2H demotions overlap fully; inclusive tiers (arena copies of VRAM experts) would
  buy nothing.
- **Plain decode is now SLOWER than DSpark on every prompt** (same flags, `STRATA_DEC_PROF=1`): prose 14.67 vs 16.33,
  python 18.61 vs 27.96, code 19.53 vs 26.68 tok/s; a plain step = attn_router ~18.3 + tier 29-45 ms.  So a width-1
  verify pass (~72 ms incl. draft-row work) is no worse than plain steps, and "k = 0 -> plain step + taps" has nothing
  to gain.  Section 2's "37.5 ms/token" was forced tokens at an older config.
- **DSpark's 2.91 GB on the card**: the sidecar holds **128** draft experts per MTP block (not 384): KEEP 64 = 3 x 64
  Q2_K experts x ~11 MiB = 2.1 GiB + attention / main_proj / markov head ~0.6 GiB (the head is the trunk's,
  shared).  Vision is not loaded.  K < 64 costs ~15% acceptance (s10), so the only VRAM left there is lower-precision
  draft experts (untested: ~0.8 GB = ~+6% slots vs some acceptance).
- **What is left for decode** (none measured): lower-precision DSpark draft experts (above); file-tier misses (2.9 ms
  each, ~17% of a python pass - speculative reads were net-negative three times because of the issuing cost); the
  dense half's fixed ~26 ms a pass (elementwise "zoo" kernels, section 3).  The expert tier is capacity-bound and
  policy cannot fix that.

### Session 10b (2026-10-14 19:40-21:10) - DSpark IN --serve (speculative SAMPLING), crash fix UNTESTED; research
**Read this block first.**  Mal's go 2026-10-14: (1) default x0.5 + arena-adapt + draft K=64 + arena 64 GiB, (2) DSpark
in --serve, (3) Quetza wrapper with those settings, (4) GitHub update, (5) research FreeToken / llama.cpp.
- **DSpark in --serve** (`ds41_generate.cpp`, opt-in `STRATA_SERVE_DSPARK=<sidecar>`, + `STRATA_DSPARK_KEEP`,
  `STRATA_DSPARK_DEPTH`; unset = the old serve path bit for bit): a request's last min(128, new) prompt tokens go
  through pass_v (seeds DSpark's window, no extra work); the window carries over when the next request continues
  this one (`dsp_next`), else `dsp->reset()`; a pure regenerate (total == reused) decodes plain.  Decode = the one-shot
  verify loop with `sample_dist` (sample_p's exact distribution) + `spec_accept`: deterministic draft d -> accept with
  p(d), else sample p without d (Leviathan/Chen with q = point mass) -> output distribution IDENTICAL to plain sampling.
  Unit test (scratchpad t.cpp, 400k draws x 12 cases): max |emp - p| 0.0016, never outside the support.
  Line protocol (`~/AI/v41-dev/taskeval/drive_dsp.py` = drive_engine.py with temp/top_p/seed + the dspark build):
  temp 0 fresh 25.5 tok/s (4.57 tok/pass), continuation after a plain turn 13.1 (window reset), **temp 1.0 continuation
  25.9 tok/s (3.84 tok/pass)**.
- **CRASH (fix built, NOT yet tested):** `CUDA error: invalid argument` in ggml_cuda_cpy inside `Ds41DSpark::step`
  (gdb backtrace `~/AI/v41-dev/taskeval/gdb.out`) on a 2nd request whose tail runs the 16-row seed graph after a
  DSpark decode.  Cause: ONE gallocr for every cached DSpark graph and gallocr only places tensors with data == NULL -
  a cached graph kept its FIRST allocation's addresses, freed when a bigger graph grew the buffer.  Fix (in
  `Ds41DSpark::step`): null every tensor of the graph's own context each step + `ggml_gallocr_reserve` on a graph
  change (`last_key`).  This may also have made one-shot drafts read stale memory (verify kept output correct; could
  have cost acceptance) - re-measure one-shot tokens/pass after the fix.  Repro: `crash2_reqs.json` (drive_dsp.py ...
  262144 crash2_reqs.json), `crash2_stdin.txt` + `gdbrun.sh`.  NOT context-specific (crashed at 256K and 512K).
- Serve VRAM after the dense half: ctx 32K 11.83 / 128K 11.56 / 256K 11.17 / 512K 10.36 GiB (one-shot 14.28 - serve
  holds the prompt-batch + snapshot buffers).  HTTP serve DSpark measured 15.9-17.6 tok/s (77-78% hits) vs one-shot
  25-28: fewer expert slots in serve - Mal wants **512K**: shrink KV (Mal: Strata-dense used int8 K / int4 V - check
  what DS4's caches use) and/or the snapshot/prompt buffers, then re-measure.
- **Research** (`~/AI/research/REPORT.md` by deepseek-quetza pro; repos cloned there: freetoken, llama.cpp-master with
  branch pr29887): FreeToken publishes **22-25 tok/s DS-V4-Flash on an RTX 5090** - we are at 25-28 one-shot on a 4090.
  Ideas we lack, ranked: (1) **q* CPU/GPU miss split** (`freetoken/python/freetoken/moe/offload_kernels.py:261-345`:
  fetch F = frac*M misses over PCIe - the most RECENTLY active (recurring) ones - and compute the rest on the CPU
  uncached, frac = the measured bandwidth balance) - our D3 sends 100% of RAM misses over PCIe
  (`STRATA_TIER_BATCH_PCIE`, default 1.0, picks misses in arbitrary order): first test 0.85 / 0.75, then recency
  selection; (2) flag-handshake CPU handoff without cudaLaunchHostFunc; (3) cudaMemcpyBatchAsync coalescing; (4)
  prompt bypass that does not evict decode experts; (5) global (cross-layer) LRU.  Also FreeToken's semantic-anchor
  KV checkpoints (agent context edits resume from the nearest anchor) - a serve-latency win for Quetza.  AcceptMoE
  (arXiv 2608.02989) = our route bias inside verify, validated (-0.27 pt, 2.06x under offload); HOBBIT (low-bit copies
  of missed experts) = a bigger project that could replace part of the route bias.
- Relay gotcha found: `relay ask ... --name X` re-registers the CALLING window's name as X (my test renamed this session
  from strata-dspark to chronos-ds4, so the cron nudge then refused: not-registered).  Cron nudges must run from cron
  (own identity), never be hand-tested from the target window.  `~/.local/bin/chronos/ds4-overnight-nudge.sh` exists;
  its crontab line was REMOVED at handoff.

### Session 10 (2026-10-14 15:10-) - arena-adapt WIRED INTO D3 (DSpark), MARGIN measured, REAL-TASK eval started
**Read this block first.**  Code `976b76fe`.  Mal's aim (2026-10-14): **30-40 tok/s with quality maintained**, and
"have you tested the bias on actual coding / agentic / tool calling?" - answer at handoff: NO, only token agreement
(teacher-forced API top-1) + prose loops.  Real-task screen built and running (below).
- Reboot overnight killed the HF upload; SSD NVME was unmounted (`udisksctl mount -b /dev/nvme0n1p3`).  **HF done:**
  Q2_K sidecar on the Hub (size-verified), card published (commit 89aa0665) - `~/AI/v41-dev/hf/upload_q2k_and_card.sh`.
- `score_official.py score` mapped req files by ROW: scoring short34 against a short (102) run gave garbage (4% top-1).
  Fixed (case id via run.json meta).  short34 tracks the full 102 within ~1 pt - the subset baselines below are free.
- **34-case subset, API top-1 / 128-tok plain decode docs / python tok/s** (`~/AI/v41-dev/s10/`):
  | config | top-1 | plain | plain + `--arena-adapt` | DSpark | DSpark + arena-adapt |
  |---|---|---|---|---|---|
  | exact (no bias) | 85.92% | 5.5 / 3.5 (96 tok) | 6.6 / 5.4 | - | - |
  | MARGIN=0.1 | 85.36% | 8.4 / 5.0 | - | - | - |
  | x0.25 + arena-adapt (quality measured, not assumed) | 85.31% | - | 14.7 / 11.4 | - | **13.7 / 16.1** |
  | flat x0.25 | 84.93% | 11.8 / 6.8 | 14.7 / 11.4 | 8.9 / 8.5 | 13.7 / 16.1 |
  | PROTECT=3 | 84.27% | ~10.6 / 6.6 (256) | - | - | - |
  | MARGIN=0.3 | 83.47% | 13.0 / 7.7 | - | - | - |
  | x0.5 | 81.59% | 16.8 / 10.3 | 18.4 / 15.6 | 16.7 / 14.6 | **22.3 / 22.9** |
  | default 1.0 | 72.98% | 24.4 / 17.3 | ~neutral | 37.7 / 28.2 | **37.7 / 39.1** |
  MARGIN loses to flat x0.25 (+arena-adapt) on both axes - smart bias is done, flat scale + arena-adapt wins.
  arena-adapt with bias on changes residency hence (in principle) routing: x0.25 tokens identical over 128 on docs and
  python, x0.5 diverge at 74 / 30 - so quality is measured per config, never assumed.
- **arena-adapt in D3** (`gpu_run_n_ex`): it was actively HARMFUL there (never bumped tick/last_use -> froze the
  vram_lru clock; disabled FILEG -> file misses fell to serial acquire + pool).  Now FILEG stays on and each file blob
  is read straight into an LRU arena victim slot (pinned; the DMA sources it there), swaps published after the call's
  final sync.  Helpers `arena_victim` / `arena_publish_swap` / `arena_check_swap` shared with gpu_run.  The checker
  now publishes all of a call's swaps before verifying (adjacent victims gave a FALSE "NEIGHBOUR CORRUPT").
  DS41_CHECK_ARENA: 3200+ swaps x 2 runs (x0.5 python, x0.25 docs), 0 bad.
- **Confidence-gate screen** (`~/AI/v41-dev/s10/gate_sim.py`, offline from the q_bias / q_exact logit dumps): "biased
  row unless its top-1 p < t, else exact row".  The bias makes the model CONFIDENTLY wrong (median p at bias errors
  0.57): back to 83.5% needs 40% of positions re-done at bias 1.0 (28% at x0.5 for 85.3%).  Not a free lunch.
- **Real-task screen** `~/AI/v41-dev/taskeval/` (`taskeval.py run|score`, `arm.sh <name> <engine args>`,
  `pipeline.sh`): serve_ds41.py on :8151 with the build-ds4-dspark binary (`engine.sh`, cfg.json), greedy, thinking
  off.  21 HumanEval (every 8th, own check()), 10 tool-choice cases (structural), 3 agentic bug-fix loops (mock FS,
  read_file / write_file / run_tests, hidden tests).  Graders validated (canonical 21/21, buggy files fail, reference
  fixes pass).  Arms: def (bias 1.0) / aa025 (x0.25 + arena-adapt) / exact (+ arena-adapt; routing identical).
  DSpark output = the trunk's greedy output, so plain serve measures DSpark's quality too.
  **RESULTS (stopped at Mal's call 17:45 before the exact arm; aa025's agent set not reached; per-case lines in
  `r_def.out` / `r_aa025.out`, full texts in `r_def/results.json`):**
  | arm | HumanEval (21) | tool calls | agentic bug-fix | tok/s needed path |
  |---|---|---|---|---|
  | bias 1.0 (default, DSpark 37.7 / 39.1) | **16/21** (76%) | 10/10 | 3/3 | |
  | x0.25 + arena-adapt (DSpark 13.7 / 16.1) | **20/21** (95%) | 8/8 run (all pass) | not run | |
  Paired: bias 1.0 failed 0, 32, 80, 120, 128; x0.25 failed only 32 (find_zero - hard).  The 4 extra bias-1.0 failures
  are careless LOGIC slips on easy problems (compares an element with itself; misses one of three pairwise checks;
  descending instead of ascending; sets the sign instead of flipping it) - exactly what a degraded router looks like.
  4-0 discordant pairs: sign test p = 0.125 two-sided - strongly suggestive, not proof at n=21.
  **Follow-up (Mal: keep coding quality, >= 20 tok/s minimum), same 21 HumanEval + 10 tools, arena-adapt on:**
  | bias scale | HumanEval | tools | DSpark + arena-adapt tok/s docs / python (128 tok) |
  |---|---|---|---|
  | x0.25 | 20/21 | 8/8 | 13.7 / 16.1 |
  | **x0.5** | **20/21** (fails only 32, as all) | 10/10 | **22.3 / 22.9** |
  | x0.75 | 19/21 (+160: left-to-right eval, ignores precedence) | 10/10 | 28.3 / 35.2 |
  | 1.0 | 16/21 (+0, 80, 120, 128) | 10/10 | 37.7 / 39.1 |
  | NVMe-only penalty (rb 0, rbf 0.6 / 1.2) | not run | - | 12.7 / 12.9, 13.9 / 15.2 (no faster than x0.25: the VRAM pull IS the speed) |
  Verdict: x0.5 + arena-adapt = coding at the best arm's level at 22-23 tok/s (one-shot DSpark); x0.75 = 28-35 tok/s
  with the first slip showing.  Through --serve today (no DSpark) x0.5 + arena-adapt is plain 18.4 / 15.6 tok/s.
  **Lossless decode work at x0.5 + arena-adapt + DSpark (128 tok, tok/s docs / python; quality per the table above):**
  | step | docs | python | gate |
  |---|---|---|---|
  | D3 without arena-adapt | 16.7 | 14.6 | - |
  | + arena-adapt in D3 (976b76fe) | 22.3 | 22.9 | DS41_CHECK_ARENA 0 bad |
  | + draft pruned `STRATA_DSPARK_KEEP=~/AI/v41-dev/s10/cnt_big.txt:64` (c0123208; 5.14 -> 2.91 GB) | 24.4-24.9 | 26.0-26.4 | exact routing: tokens identical |
  | + `--arena-gib 68` (60 default) | verify 140 -> 138 ms (text shifted) | 27.0-28.4 | min MemAvailable 8 GiB |
  | `--arena-gib 72` | - | 30.3 | min MemAvailable **4 GiB = memguard floor: NOT usable** |
  Draft K below 64 on blocks 1/2 (64,48,48 / 64,40,32 / 64,32,32) cut docs acceptance 3.53 -> 2.98-3.10 despite
  97%+ count coverage - keep 64 everywhere.  Counts: `cnt_big.txt` = 8 API fixture prompts + prose + code (DSpark
  x0.5 runs, `STRATA_DSPARK_COUNT`); test prompts docs/python were held out.
  **Measured, no gain (do not retry):** draft depth 5 (docs flat, python +7% ~noise); `STRATA_D3_PF` 2/4 (flat/worse);
  persistent D3 reader pool (`ReadPool`, kept - neutral, no thread churn); D3 speculative NVMe reads
  (`STRATA_D3_SPEC=n`, default OFF: catches ~30 of ~390 NVMe misses 2 layers ahead, extra predict graph -> -5%;
  gates: exact tokens identical, arena check 0 bad); copy-stream DMA overlap NOT built (nsys: hit kernels before the
  first DMA = 180 ms of a 4.2 s capture, ~4% max).  PCIe link is Gen4 x16 under load (21.6 GB/s pinned H2D; the
  idle "Gen 1" is power saving).  nsys x0.5 (`~/AI/v41-dev/nsys/d3x05.*`): expert DMAs 38% of decode, NVMe ~20-24%,
  GPU idle 38% (mostly the NVMe join).  The lever left = fewer bytes: VRAM / RAM for experts.
  **Not reachable from Quetza yet:** --serve has no DSpark (plain x0.5 + arena-adapt serve = ~16-18 tok/s).
  Reading: the full-speed bias does NOT break tool calling / simple agentic loops, but it DOES cost coding correctness
  (~-19 pt HumanEval).  NEXT: x0.5 + arena-adapt (DSpark 22.3 / 22.9 tok/s, 81.6% agreement) on the same screen, and
  the file-penalty-only sweep, looking for a config at 30+ tok/s that keeps HumanEval near 20/21.
- Ideas queued for the 30-40 tok/s target: (a) file-penalty-only bias (`--route-bias 0 --route-bias-file f`, + arena-
  adapt the file tier = cold experts) - 2D sweep of (rb, rbf); (b) pass-union bias in DSpark verify (bias only toward
  experts another token of the same pass already loads - they are paid for); (c) the task screen decides how much the
  token-agreement loss actually matters.

### Session 9d (2026-10-14 03:00-06:05) - SMART route bias measured; --arena-adapt FIXED (exact routing +26-60%)
**Read this block first.**  Code: `43cf3354` (smart bias), arena fix (commit below this block's date).
- **Smart route bias** (opt-in env, `tools/v41/engine/ds41_dense.cpp` router, `build_attn`): `STRATA_RB_PROTECT=P` keeps
  the model's exact top-P experts (+1e4) so the bias only competes for slots P+1..6; `STRATA_RB_MARGIN=d` applies the
  bias only to experts whose exact score is within d of the 6th/7th cutoff (confident picks kept).  Weights stay
  unbiased.  Unset = the old graph bit for bit.  Correctness: PROTECT=6 with the full bias, forced on bias-off reference
  tokens (`STRATA_DEC_MARGIN=1`) -> ZERO argmax disagreements (free-running runs diverge at token ~13 = plain decode's
  known run-to-run near-tie noise, not a bug).
  | config (default bias 1.0/0.6 otherwise) | short-fixture API top-1 (102 cases) | plain decode tok/s docs / python / prose |
  |---|---|---|
  | exact (no bias) | 86.21% | 5.48 / 3.51 (96 tok) |
  | PROTECT=3 | 84.56% (NLL 0.557) | 10.56 / 6.57 / 4.32 (prose: no loop) |
  | PROTECT=2 | 83.14% (NLL 0.607) | not measured |
  | flat bias x0.25 | 85.18% | 12.63 / 7.39 |
  | flat bias x1.0 (default) | 72.13% | 29.85 / 20.63 / 25.83 (prose loops) |
  Reading: PROTECT=3 keeps quality but the top-3 picks are the heavy I/O, so speed falls to ~bias-x0.25 level -
  **flat x0.25 dominates PROTECT=3 on both axes**.  MARGIN=0.1 / 0.3 NOT yet measured (next: 34-case subset
  `~/AI/v41-dev/quality/short34.jsonl`, ~7 min each, + 128-token speed on docs/python).
- **`--arena-adapt` FIXED** (Mal's idea: the RAM arena tracks what is used - an NVMe-read expert replaces the layer's
  least-recently-used arena expert).  It was BROKEN (garbage output, BOS spam) for two reasons, both in
  `tools/v41/engine/ds41_moe.cpp` gpu_run's file-tier block:
  1. **slot overrun:** arena slots are packed at each seeded expert's own blob size and DS4's 9 format classes differ
     in size; the victim search ignored size -> a larger blob read into a smaller slot overran into the next slot.
     Fix: a victim's slot capacity `off[s+1]-off[s]` must be >= the incoming blob.
  2. **race:** the decode path skips its per-layer stream sync, so a vram_lru demotion (D2H into an arena slot) from
     an earlier call could land AFTER the CPU threads read a file blob into the same slot.  Fix: sync gp.s (+ s_dc)
     before the direct reads.  (1 bad swap in 2200 before this fix.)
  Tools: `DS41_CHECK_ARENA=1` verifies every swap (the new expert AND the next slot's owner) against the file.
  Verified after fix 1: forced-token margin check vs exact routing = ZERO disagreements (routing unchanged by design).
  **After fix 2: `DS41_CHECK_ARENA=1` -> 3600 swaps checked, 0 bad** (python, exact routing, 128 tok; log
  `~/AI/v41-dev/smartbias/fix2_chk.log`).  Fix 2 adds a stream sync only when arena-adapt reads - re-measure speed.
  | exact routing (full quality) | fixed arena | `--arena-adapt` | NVMe reads |
  |---|---|---|---|
  | docs (96 tok) | 5.48 tok/s | **6.90** (+26%) | 4642 -> 2294 |
  | python (96 tok) | 3.51 tok/s | **5.61** (+60%) | 8116 -> 2908 |
  With the default bias it is neutral (docs 28.0, python 22.4 tok/s at 128 tok) - the bias already avoids NVMe.
  NOT yet: arena-adapt in D3 (DSpark verify: FILEG path is used instead), default-on decision, bias-x0.25 + arena-adapt.
- **Published 2026-10-14:** GitHub README (`131a0e3f`: DSpark results, quality baseline, bias cost).  Hugging Face
  `IndrasMirror/DeepSeek-V4.1-Flash-GSQ-Uncensored`: `V41-MTP-Vision.Q2_K.gguf` uploading (pid was 1548863, log
  `/tmp/claude-1000/-home-mal/a3dd5f0a-*/scratchpad/hf_q2k_upload.log`); the updated model card is staged at
  `.../scratchpad/hfcard/README.md` and auto-uploads when the file lands (watcher) - VERIFY on the Hub.
- Mal's standing rule: keep tests small (34-case subset, 128 tokens) unless deciding.

### Session 9c (2026-10-14 ~01:15) - QUALITY BASELINE (item 3a) DONE: the decode route bias costs ~14 pt API top-1
- Tools: `tools/v41/quality/build_cases.py` (DwarfStar's 112 official V4.1 fixtures, `--out ~/AI/v41-dev/quality`) +
  `score_official.py run|score` (one `--serve` process, `STRATA_SERVE_DUMP_TAIL` rows).  Results dirs
  `~/AI/v41-dev/quality/q_exact`, `q_bias`.  Teacher-forced over the API's sampled continuations (6,922 positions).
  | config | NLL | API top-1 | top-20 overlap (lower bound) |
  |---|---|---|---|
  | ours, exact routing (`STRATA_SERVE_PASS_BIAS=0`), all 112 | 0.4899 | 86.94% | 0.8724 |
  | ours, **decode route bias on** the short turns (`PASS_BIAS=1`), all | **1.0303** | **74.14%** | 0.7287 |
  |  short group (102): exact / bias | 0.5161 / 1.1111 | 86.21% / **72.13%** | 0.8654 / 0.7072 |
  |  long group (10, batch path = exact in both: the control) | 0.2322 / 0.2372 | 94.06% / 93.91% | 0.9410 / 0.9402 |
  | DwarfStar IQ2_XXS-Q2_K (their results/base-default.tsv) all / short / long | 0.3968 / 0.4184 / 0.1851 | 90.35% / 89.87% / 95.00% | 0.9076 (all) |
  **The route bias (--route-bias 1.0, file penalty 0.6) halves per-token agreement quality**: top-1 -14 pt, NLL x2.15
  on the tokens it routes.  The earlier "ppl 2.38 -> 2.67" understated it.  Plus the repetition loops (9b).  This is
  the decode speed/quality trade Mal chose on 2026-10-11 - re-decide with these numbers (DSpark makes exact-ish routing
  cheaper: verify passes can run bias-off at some speed cost - not yet measured).
- Ours vs DwarfStar Q2, exact: long -0.9 pt, short -3.7 pt.  NOT the serving path: the short cases through the
  layer-major batch (`q_batch_short`) = 86.15% vs passes 86.21% (paired -0.06 pt, 95% CI [-0.35, +0.23]).  So the gap
  is our weights (GSQ) / numerics vs DwarfStar's IQ2_XXS-Q2_K - a baseline for the dense/expert quant decisions.

- **Route-bias decision curve** (bias scale s: --route-bias s --route-bias-file 0.6s; quality = 102 short fixtures, pass
  path with bias; speed = bench4 docs / python, 256 tokens, tok/s; DSpark = Q2_K sidecar + D3 + 1-slot prefetch).
  DSpark's output = the trunk's greedy output, so quality per s is the same for plain and DSpark:
  | s | API top-1 | NLL | plain docs / python | DSpark docs / python | DSpark verify ms (docs) |
  |---|---|---|---|---|---|
  | 0 | 86.21% | 0.516 | ~2.6 (42% hits) | - | - |
  | 0.25 | 85.18% | 0.551 | 12.63 / 7.39 | 11.67 / 9.60 | 295.6 |
  | 0.5 | 81.66% | 0.676 | 18.37 / 11.22 | 20.54 / 16.19 | 181.5 |
  | 1.0 (default) | 72.13% | 1.111 | 29.85 / 20.63 | **39.51 / 31.66** | 91.2 |
  Reading: s=0.25 keeps quality within 1 pt of exact at ~40% of the s=1 speed; DSpark's edge shrinks as s falls (weaker
  bias -> fewer VRAM hits; a verify pass's expert union makes it PCIe-bound), though acceptance rises (python 4.20 ->
  4.55 tokens/pass at s=0.5).  For Mal: speed-first = s=1 + DSpark (31.7-39.5 tok/s, 72% top-1); quality-first =
  s=0.25-0.5 (82-85% top-1, 10-20 tok/s).  Raising the hit rate at low s (more VRAM for experts, smarter admission)
  is the lever that moves this whole frontier.
- LRU admission window `STRATA_LRU_WIN` (default 4 calls; now env-tunable in gpu_run + D3): plain s=0.5 sweep 1/2/4/8/16 ->
  flat (docs 17.8-19.2, python 11.2-11.6 tok/s, hits unchanged) - not the lever.

### NEXT DESIGN ITEM (not started - needs Mal's go): DSpark in `--serve` (the speedup is one-shot only today)
Why it matters: the Quetza wrappers talk to `ds41_generate --serve` (`tools/v41/serve/ds41-engine.sh`, binary
`build-ds4-gpu`), so tonight's 1.3-1.5x is not reachable from Quetza yet.  Blocker: serve samples at **temperature 1.0,
top_p 0.95** (`strata-ds41.json`) and the verify loop is greedy.  Plan:
1. **Speculative sampling** (Leviathan/Chen): accept draft d_i with prob min(1, p(d_i)/q(d_i)); on reject sample from
   norm(max(0, p - q)); all-accepted -> sample the bonus from p.  p = the trunk's row AFTER temperature/top_p/min_p
   (the exact distribution serve samples today), q = DSpark's draft distribution at the same position - needs
   `Ds41DSpark::step` to expose its draft probabilities (it already computes full-vocab logits through the shared head;
   keep q after the same temperature).  Greedy (temp 0) stays the current exact-match rule.  Output distribution is then
   identical to plain sampling (provably), so no quality question - only speed.
2. **Serve state**: DSpark's window needs main_hidden for the last <= 128 positions; after a request's prompt (batch or
   passes) re-seed with pass_v over min(128, L-1) tail tokens (rewrites those positions identically - every cache is
   position-indexed); prefix reuse / snapshot-restore just re-seed.  DSpark weights stay resident (5.14 GB Q2_K) -
   check the long-prompt layer-major path still fits its 3 GB chunk budget (else release DSpark VRAM around batches).
3. Opt-in `STRATA_SERVE_DSPARK=<sidecar>`; the default serve path untouched; gate = (a) temp 0 serve == one-shot
   tokens, (b) temp 1 token-frequency stats vs plain serve over many seeds (distribution test), (c) tok/s on the
   bench4 prompts through the HTTP server.

### Session 9b (2026-10-13 ~23:30) - DSpark BEATS PLAIN on every clean prompt: 31.7-39.5 tok/s (1.32-1.53x)
- **Recommended DSpark config:** `STRATA_DSPARK=/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.Q2_K.gguf`
  `STRATA_DSPARK_DEPTH=4` - D3 is now the DEFAULT verify tier (`STRATA_TIER_BATCH=0` = old per-token path) and the
  1-slot verify prefetch is the default (`STRATA_D3_PF=n`, 0 = off).
- **4-prompt benchmark** (`~/AI/v41-dev/bench4/`: prompts prose/python/docs/code `.i32`, `bench4.sh`, `runs.py`,
  `summarize.py`), bias ON, -n 256, tok/s:
  | prompt | plain | DSpark D3 MXFP4 | + Q2_K sidecar | + 1-slot prefetch (final) | verify ms | final vs plain |
  |---|---|---|---|---|---|---|
  | docs | 29.85 | 35.73 | 39.00 | **39.51** | 91.2 | 1.32x |
  | python | 20.63 | 27.32 | 31.27 | **31.66** | 128.2 | 1.53x |
  | code | 26.72 | 22.00* | 35.47 | **35.63** | 101.4 | 1.33x |
  | prose | 25.83 | 14.89 | - | 19.13 | 141.1 | 0.74x (BOTH LOOP, see below) |
  *different text (residency-dependent routing).  Q2_K sidecar = +2.76 GB of expert slots -> ~+3 pt VRAM hits ->
  verify -12..-17 ms; its acceptance cost is ~0 here (harness: -0.047).
- **Upstream ggml fixes APPLIED** (`tools/v41/ggml/upstream-fixes.patch`: #30103 flash-attn tile race, #29953 MMQ
  OOB reads).  #29953 changed `mmq_args`' last field (ncols_opt -> J_best): Strata's positional init in
  `src/prefill/moe_mmq.cu` passed max_rows there -> GGML_ABORT "J_best=75"; fixed (J_best computed as ggml's host code)
  + identity id buffers padded 128 ints.  Verified: 512-token prefill logits md5 IDENTICAL pre/post patch and run to
  run (`~/AI/v41-dev/d2/det_pair.sh`), DSpark decode 256 tokens identical.  The prefill was already deterministic;
  plain decode's near-tie flips are not explained by #30103.  Timing noise on identical work: ~+-5% tok/s.
- **Depth sweep** (final config): docs depth 3/4/5 = 38.27/39.51/34.19 tok/s, python 30.42/31.66/30.61 -> keep 4.
- **PCIe is the wall** (nsys, `STRATA_PROFILE_DECODE=1` + `nsys --capture-range=cudaProfilerApi`): ~100 expert DMAs
  (9.6 MB avg, 23 GB/s) per verify pass = 1.4 s of a 4.2 s decode; kernels are cheap.  So: fewer misses (VRAM for
  slots, LRU) beats cleverness; speculative prefetch only pays at 1 slot (precision 40%; 6 slots = -20 ms).
- **QUALITY FLAG for Mal - the decode route bias causes repetition loops:** prose prompt, plain greedy: bias ON
  loops ("in livres de fio, in livres de fio, ..." rep8 0.62), bias OFF clean to EOS (rep8 0.00); code prompt earlier:
  plain bias ON rep8 0.54 vs OFF 0.00.  2/2 prompts.  (Known trade: --route-bias 1.0 = ~4x decode, ppl 2.38 -> 2.67.)
  DSpark inherits it (its draft predicts the unbiased model, so looping text also lowers acceptance).
- **Benchmark discipline:** bias-on text depends on VRAM residency (even nsys's 30 MB shifts it): never judge a change
  by one prompt's tok/s; compare verify ms/pass on the SAME text (tokens/pass identical) and drop rep8 > 0.3 runs.

### Session 9 (2026-10-13 night) - D3 WORKS, correct, and faster than per-token; still opt-in (`STRATA_TIER_BATCH=1`)
- **Crash fixed (6da95756):** `acquire()` grew `im.ftmp` mid-batch (slot index > top_k in a multi-token pass) ->
  every earlier job's blob pointer dangled -> SIGSEGV in the pool.  `reserve_ftmp()` sizes it first (also in `gpu_run_n`).
- **Speed (291ee6b0), bias ON, depth 4, p256 x 256, MXFP4 sidecar** (per-token baseline 19.80 tok/s, verify 167.6 ms):
  | step | tok/s | verify ms/pass |
  |---|---|---|
  | crash fix only | 19.01 | 191.4 |
  | + D3's own PCIe share (`STRATA_TIER_BATCH_PCIE`, default 1) + file tier read in parallel and run on the card | 24.28 | 148.9 |
  | + `Ds41MoeTier::kMaxMulti` = 8 (one call/layer), d_stage waves, packed pinned metadata | 25.82 | 138.1 |
  | + Q2_0 classes batch (they silently fell back to per-token) | 22.52* | 134.5 |
  | + vram_lru ported (D3 never admitted: mixed blob sizes) | see d3c_a | |
  *acceptance 2.20 vs 2.70 that run = trajectory variance (text differs), see correctness.
  Why it was slow: `gpu_run_n_ex` was cloned from the old uniform `gpu_run_n` and had none of `gpu_run`'s work
  (serial file reads on the critical path, 5 pageable copies per launch, CPU pool ~2 ms/expert, no LRU).
  `DS41_PROF_X=1` prints per-phase ms/call every 640 calls (diff consecutive lines for steady state).
- **Correctness CLOSED for the verify core:** `STRATA_DSPARK_DUMP=<file>` (every verify row) +
  `~/AI/v41-dev/d2/vdump_cmp.py <vdump> <arb.logits>` vs an independent prefill of the stream (`arb_run.sh`).
  **Bias OFF** (`nb_gate.sh`): per-token 96.0% argmax agree, D3 95.4%, EVERY disagreement a near-tie (prefill top-1
  gap < 0.6); rows 1..5 no worse than row 0 -> no multi-token structural bug.  **Bias ON comparisons are meaningless**
  (decode route bias depends on VRAM residency -> different routing than the prefill: per-token itself shows 91% /
  max|d| median 6.6).  The old "18 logits worse at 127" was a bias-on artefact.  D3 is ~25% noisier on mean|d|
  (more experts on GPU kernels instead of the CPU's) - legit rounding.
- `DS41_CHECK_LRU=1` (D3 path): slot + demoted arena copy vs the file after every swap - 3200 swaps, 0 bad.
- **Remaining gap:** verify ~135 ms vs the ~100 ms bar (plain decode 27.4 tok/s still beats D3's ~25.8).  Per layer
  call ~2.1-2.8 ms, of it ~1.5 ms GPU wait (DMAs + hit kernels + per-token sums) - nsys capture `d3c_nsys.nsys-rep`
  in ~/AI/v41-dev/d2 to split it.  Ideas: DMAs on a copy stream overlapping the hit kernels; fewer launches per class.
- Relay/GPU sharing: voice-duplex (Quetza) asks per run and waits for an explicit 'go' (Mal: DS4 has priority).

### Session 8 (2026-10-13 late evening) - D3 batched tier CODED but CRASHES; opt-in, default OFF
- **D3 = `gpu_run_n_ex`** (`tools/v41/engine/ds41_moe.cpp`, right after `gpu_run_n`): the batched per-expert-class
  verify tier - the same union sharing as `gpu_run_n` (each distinct expert read ONCE, a group with one entry per
  token that routed to it) but split by the 9 format classes: one `native_expert_grouped` per (class, site) with
  `gp.gll[cls]`, per-expert blob `im.bl[cls]` / fmt `im.fl[cls]`, CPU jobs with `jb.fmt=&im.fl[cls]` and a
  per-(token,class) activation quant via a NEW buffer `im.nact_multi_n` + `nact_slot_n()`, and the per-token card
  sum (`decode_rows_sum` per token - bit-identical to `run_to_dev`).  `Ds41MoeTier::run_multi_to_dev` (new) and
  `run_multi`'s `per_expert()` branch route to it; `run_multi_to_dev`'s fallback is per-token `run_to_dev`
  (NOT `gpu_run_n` - that was a bug).  `pass_v` (`ds41_generate.cpp` ~542) now calls `run_multi_to_dev` once per
  layer, 4 tokens at a time (`kMaxTok`), instead of `tier.run_to_dev` per token.
- **STATUS: BROKEN.** `STRATA_TIER_BATCH=1` -> the D3 arm SIGSEGVs (exit 139) in the CPU pool
  (`native_gu_rows` -> `ggml_vec_dot_q3_K_q8_K`), at layer 0, nt=4, on the prompt tail.  A gdb run also showed
  `ds4_moe: c_stage: out of memory` + SIGABRT (main thread) - likely two symptoms of ONE host OOB.  `[dbg_x] ENTER
  gpu_run_n_ex` prints but the `nj`+per-job dump 2 lines below does NOT -> the fault is between ENTER and the CPU
  job loop; prime suspect a bad `nact` pointer.  **Check first:** `im.nact_multi_n` is sized (init ~line 1197:
  `kMaxTok * nact_multi_act.size() * kNativeActBytes`) and non-empty; `nact_slot_n`'s index
  (`t*nfmt + nact_act_idx`) matches its stride (`kNativeActBytes`); and the PCIe staging `si` bound.
- **OPT-IN now: `STRATA_TIER_BATCH=1`** (default = the per-token path, which WORKS).  Debug with `DS41_DBG_X=1`
  (prints ENTER + `nact_multi_n`/`nfmt` sizes + a per-job blob/fmt/nact dump).  The old `STRATA_NO_TIER_BATCH`
  was renamed/removed.
- **A/B harness** `~/AI/v41-dev/d2/d3_test.sh` (bias OFF, depth 4, -n 64; bias off to match `plain_nb.tok`):
  `d3_a`=batched, `tok_b`=per-token.  It still sets the OLD `STRATA_NO_TIER_BATCH` - update it.  Latest: `d3_a`
  exit 139 (crash), `tok_b` exit 0 (64 clean tokens) -> the rest of the engine + the per-token tier are fine.
  Correctness target: `d3_a.tok == tok_b.tok`.  Perf target: verify ms/pass from ~168 (MXFP4) / 212 (IQ3_XXS)
  to <= ~100 -> 35-40 tok/s.
- **MTP verdict (Mal asked; FINDINGS §13-22):** MTP's draft is cheap (6-9 ms warm; 6.1 ms with `--mtp-keep 2`;
  63-68% depth-1 acceptance) but it LOSES - 18.9-20.6 tok/s vs 24.9 plain - for the SAME reason DSpark loses:
  the multi-token verify pass reads ~2.2x the experts, `run_multi` has no VRAM LRU, and the verify margin costs
  ~170 expert slots.  FINDINGS' own verdict: *"the road to 30 tok/s is a cheaper pass, not a deeper draft."*
  So **D3 (the batched pass) is the lever for BOTH MTP and DSpark** - fix D3 and MTP (§22's `--mtp-keep 2` +
  run_multi LRU) becomes worth re-measuring too.
- **GPU sharing:** another live session `strata-dense` (relay name) builds/benches a Strata dense 27B and wants
  the 4090 for short MTP benches.  My relay name is `strata-dspark`.  Their 11 GiB MTP model + our ~10-13 GB run
  do NOT fit together -> SERIALIZE: ask first via `relay ask strata-dense ...`, yield for their bench, broadcast
  `4090 free` when done.  This session left messages coordinating that.

- **D2 CODED (opt-in, commit below), gate NOT yet passed.** `ds41_generate.cpp` one-shot: `STRATA_DSPARK=<sidecar>`
  `STRATA_DSPARK_DEPTH` (1..5, default 4), greedy.  Pieces: `pass_v` (pass_m + main_hidden via read_state after layers
  36/37/38 + all m rows of logits); DSpark init INSIDE `open_cache` after `release_pass` (cache sizes around it); prompt
  minus last min(128,L-1) via feed, then the tail through pass_v in chunks of 16 **with decode biases OFF** (bug found:
  the tail routed with the decode route bias -> stream diverged at token 0), last chunk drafts; accept loop with the
  odd-start re-feed after a rejection; stats line `DSpark: N verify passes, X accepted drafts/pass ...`.
  `STRATA_DEC_MARGIN=1` + `STRATA_FORCE_IDS` (plain loop) prints argmax vs forced logit gap where they differ.
- **5b's ring analysis was wrong for depth 4 with the re-feed** (a re-fed query at P+1 needs P-126 = the slot a
  rejected P+5 overwrites).  FIX: `ds41_dense.cpp` `im.RAW = im.SWA + 16` (gather by position -> results unchanged by
  construction; NOT yet re-verified with the forced-token 0-diff gate - do that).
- DSpark is compiled INTO ds41_generate (`target_sources`), not linked as the lib: linking ds41_dspark reorders static
  archives -> duplicate `ggml_cuda_error` (ggml-cuda.cu.o vs strata_mmq ggml_cuda_host.cu.o).
- **Shared head**: `Ds41Dense::weight(name)` + `Ds41DSparkConfig::shared_head`; ds41_generate passes the trunk's
  output.weight (-0.7 GB; `STRATA_DSPARK_OWN_HEAD=1` = old).  Built, NOT yet run.
- Measured (p256 x 256, WITH decode biases): plain 27.38 tok/s (97.8% VRAM hits); DSpark depth 4 run 1 (tail-bias bug)
  20.71 tok/s, 4.06 tokens/pass, verify 190 ms/pass, draft 3.5 ms, 87.4% hits; run 2 (fixed) 18.36 tok/s, 3.16
  tokens/pass, first divergence token 9.  Forced margin run: the route bias depends on VRAM residency, which DSpark's
  8.6 GB changes -> different routing -> logit gaps 1-4 = NOT a verify bug.  Also plain decode is not run-to-run
  deterministic (near-tie gap 0.05 flipped at step 4) -> suspect the flash-attn race #30103.
  **Correct gate = both arms with `--route-bias 0 --route-bias-file 0`** (script below; was queued at handoff).
- Speed: v1 verify = ~5x a single step (tier per token) -> D3 (multi-token tier) is THE next speed item.
- Upstream fixes PREPARED (not applied): `tools/v41/ggml/upstream-fixes.patch` (#30103 fattn-mma-f16.cuh 1 line,
  #29953 mmq.cu/.cuh), `tools/v41/ggml/UPSTREAM_FIXES.md`; dry-run clean against the tree (with dsv41-fuse applied).
- Worker `v41-dspark-shrink` (flash) in flight at handoff: requant DSpark's MXFP4 experts (Q3_K/Q2_K/IQ*), acceptance
  per type via test_ds41_dspark; report `~/.quetza-data/conductor/reports/v41-dspark-shrink.md`, results
  `tools/v41/DSPARK_SHRINK.md`, files `/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.<type>.gguf` (40 GB free).
  Its requant tool needs ~12+ GB RAM: told to run RAM-heavy jobs under the GPU lock (it starved an engine run once).
- **Bias-off gate (both arms `--route-bias 0 --route-bias-file 0`)**: plain 4.06 tok/s (57.9% hits - no bias = slow,
  irrelevant for correctness), DSpark 3.49 tokens/pass, first divergence token 8 -> margin check
  (`~/AI/v41-dev/d2/margin_nb.log`, plain forced on the DSpark stream + STRATA_DEC_MARGIN) decides near-tie vs bug.
- **On-load expert requant**: `STRATA_DSPARK_EXP_TYPE=<ggml type name>` (q4_K / q3_K / q2_K / iq4_xs ...; no imatrix
  types) in `ds41_dspark.cpp` load(): MXFP4 -> f32 -> type, all CPU threads, no files.  Works in test_ds41_dspark and
  ds41_generate.  `pass_v split` line (dense / tier / taps / logits ms per pass) printed with the DSpark stats.
- **D2 CORRECTNESS BUG (open, top priority)**: bias-off DSpark stream vs plain: the verify pass is WRONG at some
  tokens - arbitrated by an independent full-path prefill (`~/AI/v41-dev/d2/arb.logits`, prompt + DSpark stream,
  `--dump-logits`): prefill == plain at all 5 disputed steps (8, 13, 127, 195, 198); DSpark's token is ~1 logit worse at
  8/13/195/198 and **18 logits worse at 127** (position 383).  Ruled out: window ring (gathered by position, RAW+16),
  ratio-2 state ring (re-feed covers it - re-check if bisect says "rejections"), candidate blocks (per-pass recompute),
  Engram history (position-indexed, rewritable), tier h_x race (run_to_dev ends with `sum sync`).
  **BISECT DONE (2026-10-13 ~18:45, `~/AI/v41-dev/d2/queue.out` + `bis_d0.tok`/`bis_noacc.tok`)**: both arms diverge
  from `plain_nb.tok` -> **`bis_d0` first divergence token 8** (`STRATA_DSPARK_DEPTH=0`, single-token pass_v passes,
  0.000 accepted/pass) and **`bis_noacc` token 11** (`STRATA_DSPARK_NOACC=1`, every draft rejected).  Both-arms-diverge
  => the bug is in **`pass_v` core with m=1** (read_state / logits rows / begin/attn_router/finish), NOT drafting,
  rollback, or rejected-position residue.  Next: diff `pass_v` (m=1) vs the plain `feed` step for one token - the
  first mismatch is the bug; the token-8 case is a near-tie (plain argmax 588 vs 769, logit gap 0.11), so also confirm
  whether plain's own run-to-run non-determinism (#30103) is in play vs a real pass_v math error.
  Ruled out earlier: window ring, ratio-2 state ring, candidate blocks, Engram history, tier h_x race.
- Profiling arm `prof_iq3` (bias ON, depth 4, `STRATA_DSPARK_EXP_TYPE=iq3_xxs`) DONE but **timings unreliable** - it
  ran while `strata-dense` was doing a CPU-heavy CUDA build: 3.312 tokens/pass, verify 212.5 ms/pass, 93.3% VRAM hits,
  on-load IQ3_XXS requant alone took 735 s.  Re-run on an idle box before trusting the numbers.
- Profiling runs (bias ON, depth 4) DONE (`~/AI/v41-dev/d2/prof_mx.log` MXFP4 / `prof_q3.log`
  `STRATA_DSPARK_EXP_TYPE=q3_K`).  `pass_v split` (async GPU - the total is exact, the dense/tier split approximate):
  | run  | tok/s | tokens/pass | verify ms/pass | VRAM hits | dense / tier ms per pass |
  |------|-------|-------------|----------------|-----------|--------------------------|
  | MXFP4 | 20.50 | 3.556 | 168.6 | 91.1% | 39.6 / 414.9 |
  | q3_K  | 18.74 | 2.865 | 148.5 | 93.0% | 38.3 / 357.0 |
  **Tier ~10x dense -> the expert tier IS the verify cost; D3 batching it is THE lever.**  Per token the tier
  costs ~47 (mx) / ~52 (q3) ms vs plain's 36.5 ms -> v1 is ~30% slower than plain; D3 must drop a depth-4 pass
  under ~100 ms to clear 35-40 tok/s.
  **WATCH (single-run, trajectory-dependent): q3_K's engine acceptance 1.865/pass << MXFP4's 2.556, but
  test_ds41_dspark said Q3_K ~= MXFP4 (3.246 vs 3.280).**  Need repeats to split format effect from trajectory;
  profile with `IQ3_XXS` (the chosen format) next, not q3_K.
- **DSpark shrink DONE (worker v41-dspark-shrink, `tools/v41/DSPARK_SHRINK.md`)**, test_ds41_dspark 378 steps, accepted/pass:
  MXFP4 3.280 (8.61 GB on card) | Q3_K 3.246 (7.23 GB) | **Q2_K 3.233 (5.85 GB, -2.76 GB)** | **IQ3_XXS 3.280 (6.59 GB,
  identical acceptance)**.  Use `STRATA_DSPARK_EXP_TYPE=iq3_xxs` (or q2_K for max VRAM) - on-load env path == file path
  (cross-checked).  Files tool `tools/v41/v41_sidecar_requant.cpp`; `V41-MTP-Vision.{Q3_K,Q2_K}.gguf` on NVME1TB.
- **Profiling (bias ON, depth 4, 2026-10-13 ~18:00)**: MXFP4 20.50 tok/s, 3.56 tokens/pass, verify 168.6 ms/pass, 91.1%
  hits; on-load q3_K 18.74 tok/s, 2.87 tokens/pass, verify 148.5 ms, 93.0% hits (tokens/pass differs because the text
  differs - the open bug).  The `pass_v split` line averages in the 8 seeding passes of 16 prompt tokens, so it is
  NOT per verify pass (dense ~39 ms/pass for ~6-16 tokens; tier dominates).  Estimate: verify = dense ~35 ms + tier
  ~130 ms for ~5 tokens (~25 ms/token, worse than plain's ~16 because pass_v has no prefetch).  D3 projection: a
  batched tier at ~2x one token's (~35 ms) -> verify ~70 ms -> ~45-50 tok/s at 3.5 tokens/pass; ~100 ms -> ~35.
  So DSpark clears Mal's 35-40 bar IF D3 lands.  TODO: split the seed passes out of the pass_v stats.
- **Phase-2 queue RAN (2026-10-13 ~18:33-19:00, `tools/v41/d2_queue.sh`)**: bis_d0, bis_noacc, compare vs
  `plain_nb.tok`, and `prof_iq3` all completed (exit 0); see the BISECT DONE bullet above for verdicts.  Runner
  `tools/v41/d2_queue_whenfree.sh` (idle-trigger) works but DISARM it before starting it by hand to avoid two runners:
  `touch ~/AI/v41-dev/d2/queue.CANCEL` cancels between arms; results in `~/AI/v41-dev/d2/{bis_d0,bis_noacc,prof_iq3}.*`.
- Mal's bar: DSpark is worth it at **35-40 tok/s** (needs verify <= ~100-115 ms/pass at ~4 tokens/pass).
- **Do NOT wrap memguard.sh in flock**: memguard takes `ds4-gpu.lock` itself -> self-deadlock.

## 5b. Session 6 (2026-10-13 afternoon/evening) - READ THIS, it supersedes §5 where they differ
- **DSpark on OUR states (fixed harness, 378 steps): 3.27 accepted/pass = 4.27 tokens/pass** (= FP8 teacher 3.24) ->
  no retraining.  Logs `~/AI/v41-dev/dspark/fixed_ours.log`, `fixed_teacher.log`.
- **D1 DONE + committed `c0fe0e06`: `tools/v41/engine/ds41_dspark.{hpp,cpp}` (class Ds41DSpark) + `test_ds41_dspark.cpp`**.
  The draft stack on CUDA: sidecar weights (its `wo_a` verified correctly scaled, rel err 5.5e-3) + trunk
  `output.weight` / `token_embd` loaded itself (standalone: 8.61 GB VRAM incl. a 0.7 GB duplicate head).  API:
  `step(main_hidden[k*15360], pos[k], k, bonus, ids[5], conf[5])` appends k accepted rows to each block's 128-ring and,
  if bonus >= 0, drafts p+2..p+6 (cached graph per (k, draft); 4.9 ms/draft).  **Gate PASS** on the harness's own
  records (96 steps): engine 2.719 accepted vs harness 2.760; first draft identical 99.0% (slots 99.0/97.9/92.7/91.7/
  89.6%).  Run: `flock ~/.quetza-data/conductor/ds4-gpu.lock build-ds4-dspark/test_ds41_dspark --sidecar
  /media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf -m "$G" --main ~/AI/v41-dev/dspark/main_exact.bin --gen
  ~/AI/v41-dev/dspark/gen_exact.txt --max 96 --dump F` (compare `dspark_accept.py ... --max 96 --dump-drafts F2`).
  Built in a SEPARATE tree `build-ds4-dspark` (same options as build-ds4-gpu minus `STRATA_DSV41_HC_MIX`).
- **Drafter precision cannot hurt output quality** (every draft is verified by the trunk) - only acceptance.  So the
  6.8 GB of MXFP4 DSpark experts can be compressed hard (Q4/Q2/hot-subset) or tiered; measure acceptance with
  test_ds41_dspark only.  Also share the trunk's head instead of loading a second copy (-0.7 GB).
- **D2 DESIGN (not coded yet) - verify/accept/rollback in `ds41_generate.cpp` one-shot path, greedy (temp 0) v1,
  `STRATA_DSPARK=<sidecar>`, `STRATA_DSPARK_DEPTH` (<= 4, see ring):**
  1. *Verify pass* = the existing `pass_m` lambda (ds41_generate.cpp ~513: one dense pass of m tokens, tier per token
     via `run_to_dev` into the routed rows), copied as `pass_v` that also captures main_hidden: after
     `dense.finish(l)` for l in {36,37,38} call `dense.read_state(x, pre)` (token-major [m][HC][D]) and accumulate the
     HC-mean into rows [m][3*D] (same math as the STRATA_DUMP_MAIN block ~490).  Logits: `dense.logits(row0, nrows, out)`.
     Dense `max_pass` = S (512 with the default layer-major prompt batch) so m <= 6 fits.
  2. *Loop*: state P (last fed position), t (trunk token for P+1, not yet fed), drafts d[0..k).  Emit t; pass tokens
     [t, d0..d(k-1)] at P+1..P+1+k; g_i = argmax(row of position P+1+i); a = #leading d_i == g_i; emit d0..d(a-1);
     t' = g_a; DSpark.step(main_hidden rows of P+1..P+1+a, a+1 rows, bonus t') -> new drafts; P += 1+a.
  3. *Rollback is free except two rings*: every cache is position-indexed and simply overwritten when a position is
     fed again (compressed rows, indexer keys, Engram history, sel/cand per column).  (a) the window K ring has
     RAW = SWA+3 = 131 slots (`ds41_dense.cpp` init `im.RAW = im.SWA + 3`, slot = pos % RAW): a rejected position q
     evicts q-131, which a later query at q' needs iff q >= q'+4 -> **safe for depth <= 4**; depth 5 needs RAW >= SWA+1+5
     (one-line change in ds41_dense.cpp, bit-identical by construction - gather by position; do it when the fusion
     slice is done with that file).  (b) the **ratio-2 compressor state ring is 2 slots by parity** (`i_sidx`/`i_sslot`,
     ds41_dense.cpp ~1112): a pass writes only its LAST 2 tokens' states, so after a partial rejection the slot of the
     last accepted position may hold a rejected token's state.  FIX (no engine change): if the pass would start at an
     ODD position and the previous pass rejected anything, start it ONE position earlier, re-feeding the last accepted
     token (its group's states then come from inside the pass, `row(q) = 2 + q - pos0`).
  4. *Seeding the DSpark window*: the layer-major prefill gives no per-position states, so feed the prompt minus its
     last min(128, L-1) tokens normally, open the VRAM cache, then run that tail through `pass_v` in chunks of <= 16
     (set `Ds41DSparkConfig::max_append = 16`), appending the main_hidden rows (bonus -1), the last chunk drafting.
  5. *Gate*: spec output vs plain greedy (`STRATA_DSPARK` unset) - identical up to batch-shape near-ties (report the
     first divergence index + ppl); tokens/pass in-engine vs 4.27 offline; tok/s.
  6. *Known speed ceiling of v1*: `pass_m` runs the tier PER TOKEN (~16 ms each) and `Ds41MoeTier::run_multi` falls back
     token-by-token for per-expert-geometry models (ours), so a 5-token verify costs ~5x tier.  **D3 = a multi-token
     tier path** (union of the pass's experts, each expert's weights read once for all its tokens - `native_expert_grouped`
     already takes `ent_tok`; the prefill `run_chunk_dev` groups by expert but streams whole layers): that is what
     decides 50 tok/s.  Union of 5 tokens' experts also raises misses: keep route bias on (it pulls the union into VRAM).
- **Kernel fusion slice DONE + committed `86724531`** (worker closed): fused HC-mix op `GGML_OP_DSV41_HC_MIXES`
  (`tools/v41/ggml/dsv41-fuse.patch`, APPLIED in the local third_party tree; opt-in `-DSTRATA_DSV41_HC_MIX=ON`, which
  `build-ds4-gpu` is configured with).  Bit-exact (test_ds41_hcmix, forced tokens 0-diff, logits md5 equal); -546
  kernels/token (-14.9% of the dense graph), ~0.5 ms/token - timed A/B within noise (26.42/25.89 vs 25.91).  Next-best
  fusions it listed: router tail softplus/sqrt/+bias (~0.3 ms), the shared expert's clamps block ggml's
  MUL_MAT+MUL_MAT+GLU fusion, F16 window ring to drop the f32->f16 cpy.  Prefill md5 with the op: not re-checked.
- **Dense requant (forced A/B, 1 run each): q8 26.64 / q5_K 27.73 / q4_K 29.75 tok/s** (attn_router 17.11 / 14.87 /
  13.52).  q4_K quality vs a same-arena q8 control: code ppl +1.7% (KL 0.026 vs floor 0.0095), agent KL 0.48 vs floor
  0.25.  NOT adopted - Mal: quality benchmark first (§6 item 3).
- **llama.cpp fixes to pull (queued slice, after fusion)**: #30103 (flash-attn tile race, our D=512 prompt attention;
  maybe the cause of the prefill md5 drift) + #29953 (MMQ OOB reads) - our vendored ggml predates both.
- Strata 0.1.41-0.1.43.1 scanned: its decode-collapse bug does not apply (no pinned thread / host callbacks in our
  engine); useful later: `tools/strata_bench.py` speed suite, pinned ring for pageable H2D, CPU share for short prompts.
- Cleanup done: 12 old worker worktrees removed (branches kept; wire leftovers = `git stash@{0}`), Strata-architectds
  deleted.  Swap is not a lever (8+32 GB swap files on the SATA root SSD ~0.5 GB/s; zram = compressed RAM; the arena is
  pinned).  Mal's mode: **the conductor writes the core code itself; the one flash worker is for research/legwork.**

## 6. Work queue (Mal's order; ONE flash worker at a time)
1. **DSpark on OUR states: DONE - 4.27 tokens/pass, no retraining.**  => item 4 (engine draft+verify) is the top
   priority after the in-flight fusion slice.  50 tok/s needs a verify pass <= ~85 ms at depth 5 (or <= ~53 ms at depth 2).
2. **Kernel fusion of the elementwise zoo** (packet ready: `~/.quetza-data/conductor/packets/v41-zoo-fusion-task.md`):
   ~3,300 tiny kernels/token = 5.45 ms; bit-exact fused ops (forced-token 0-diff is the gate); expect -1.5..-3 ms (+4-8%).
2b. **ggml upstream fixes** (after fusion; it owns the ggml tree): our vendored ggml predates llama.cpp #30103 (race in
   `flash_attn_ext_f16` tile combine: `fattn-mma-f16.cuh:1747` `if (np > 1)` -> `if (np > 1 || nbatch_combine != DKQ/2)`;
   our prompt attention is D=512) and #29953 (MMQ out-of-bounds reads, `mmq.cu:129` fallback).  Ship as
   `tools/v41/ggml/upstream-fixes.patch`; gate = forced decode 0-diff + does the 8K prefill `--dump-last` md5 become
   reproducible across runs with different free VRAM (the FA race may be the real cause of today's md5 drift).
3. **Quality benchmark baseline** (Mal: decide every lossy change against it): (a) DwarfStar's official V4.1 fixtures
   `~/AI/ds4-ref/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/` (112 prompts with official API
   continuations + top-20 logprobs; teacher-forced NLL / API top-1 agreement / top-20 recall; comparable to DwarfStar's
   Q2 = 90.1% top-1) - score exact routing AND the default decode biases (`--bias-prompt` decode-routing proxy);
   (b) Mal's agentic harness bench `~/AI/quetza-workspace/model-gauntlet/harness-bench/` (real QuetzaCodetl runs,
   hidden-graded Scheme/JSON tasks; add a `strata-ds41` backend); (c) speed side: Strata 0.1.43's
   `tools/strata_bench.py` suite (short chat, 4K/16K prompts, five-turn agent session) against our server (port 8150).  Note: the decode route bias (code ppl 2.38 -> 2.87)
   is a bigger quality lever than any dense quant - benchmark the configs Mal would actually choose between.
4. **DSpark engine draft + verify** (after 1): DSpark stack as a ggml graph (resident dense, experts resident/tiered),
   main_hidden in-graph, verify pass of 1+k tokens (dense `pass_m` + tier `run_multi`), KV rollback, depth adaptive by
   the confidence head; port the shape from the V4 engine's `--mtp/--verify` (`tools/ds4`).  Then **n-gram /
   prompt-lookup drafting** on the same verify path.  50 tok/s = 20 ms/token -> needs ~2+ accepted per pass.
5. **Vision engine + serve** (slice 2): `GENI` for V4.1 (facts in §5), `VRAM <mib>` + `--vram-elastic` in `--serve` so
   Strata's server can run the encoder LAZILY (Mal's wish: not resident; start on the first uncached image, the engine
   lends ~1-2 GB of expert slots while it encodes, stop after `idle_s`, cache grows back - the server side exists:
   `start_lazy_vision`, `Engine.vram()` in `~/AI/Strata/serve/server.py`); config `"vision": {..., "lazy": true}`;
   CPU-encode fallback; gate = a synthetic image with a known word + coloured shape answered correctly.  Then make
   `serve_ds41.py` portable (it hard-codes `~/AI/Strata`).
6. Later: NVMe-miss work (reads from the existing top-16 prediction on an I/O thread; frequency admission), dense requant
   decision after the benchmark, faster ternary expert kernel.

## 7. How to measure (mandatory)
- **Correctness bar = forced-token identity**: `STRATA_FORCE_IDS=~/AI/v41-dev/dec7/force256.tok` makes every run
  decode the same 256 tokens; diff the emitted ids vs the file (skip the non-numeric memguard banner line) = 0 lines.
  Numerics gate: `build-ds4-gpu/test_ds41_moe_gpu "$G" --layers 0,20,32,39 --gpu --tokens 8` (worst rel ~1.44e-2).
  Anything that moves bits: 4096-token ppl + KL vs `~/AI/v41-dev/dq/*_q8.logits` (code/agent/reason).
- **Speed A/B**: `~/AI/v41-dev/k143/ab.sh TAG [ENV=VAL ...]` (p256 x 256, `--slots 1500`, phase profile; appends to
  `k143/results.txt`; `BIN=` selects a binary).  Noise floor ~3-5% (worse on a busy box): >= 2 runs per arm, quiet box,
  never beside builds / torch jobs / the HF upload.
- **nsys**: count calls and use `~/AI/v41-dev/nsys/calls.py` / `idle_attrib.py` (they include
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` now - without it the dense half is invisible); `--cuda-graph-trace=node` to see
  inside the dense graph (slows decode ~50%: ratios only).  Tokens in the 4.5 s window = graph launches, not "96".
- Prefill exactness md5 (`--dump-last`) is NOT reproducible right now (MMQ grouping depends on free VRAM); decode is.

## 8. Rules (Mal)
- Kill by PID only.  Never `pkill -f`/`pgrep -f` a pattern that can match your own command line - that includes a
  `case *x*` glob over `/proc/*/cmdline` (matches your own shell; match the EXACT cmdline); never `tmux kill-server`.
- Never run a full-model engine beside a worker's ~16 GB CPU torch job: RAM runs out and memguard kills the engine.  Never edit a running bash script in place (write a copy, `mv`).
- GPU runs: `STRATA_NO_LARGEPAGES=1 bash tools/ds4/memguard.sh 86 72 -- <cmd>` (env vars INSIDE as `env X=1 cmd`
  or before `bash memguard.sh` as ab.sh does); wait for MemAvailable >= 75 GiB; one GPU job at a time; it takes
  `~/.quetza-data/conductor/ds4-gpu.lock`.  Never a CPU-only full-model load (froze the box once).
- Don't rebuild `build-ds4-gpu` while a run or the wrapper uses it.  `third_party/llama.cpp` is untracked: ggml
  changes ship as patch files in `tools/v41/ggml/` (fake-quant + rank-argsort patches are APPLIED in the local tree;
  `/usr/bin/patch`, anaconda's is broken).
- Measure before claiming; report ppl/KL for anything lossy; **Mal decides quality trades**; soft routing only
  (penalties, never masks).  Commits end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Conductor mode: **ONE flash worker at a time** (Mal), no Agent-tool subagents; **the conductor writes the core code
  itself - the worker is for research/legwork** (Mal 2026-10-13); packets in
  `~/.quetza-data/conductor/packets/<slice>-task.md`, dispatch with `qc-dispatch <slice> --model flash --prompt "Read
  the full task packet at ...-task.md and follow it exactly ..."` (`--packet` stalls; `--prompt` overwrites
  `packets/<slice>.md`), arm `qc-await <slice>`.
- NSFW content is headless-only on this PC (never display images); vision tests use synthetic images.
- "SSD NVME" does not auto-mount after a reboot: `udisksctl mount -b /dev/nvme0n1p3` (must be ntfs3).

## 9. Paths
| what | where |
|---|---|
| model | `/media/mal/SSD NVME/AI/V41-GSQ/V41-GSQ.gguf` (+ `rank.bin`, `tokenizer/`, `V41-GSQ.gguf.experts.json`) |
| Engram tables | `--engram "/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram"` (L14 : L1) |
| DSpark + vision sidecar | `/media/mal/NVME1TB/Models/V41-Sidecar/V41-MTP-Vision.gguf` (mtp.0-2 MXFP4 + Q8_0; vision.*, aligner.*, image_*) |
| FP8 source checkpoint | `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/` (shards 1-46; 47/48 = Engram on NVME1TB) |
| DeepSeek reference code | `~/AI/gsq-rco/dsv41-inference/` (model.py incl. DSpark + vision, vision.py, image_processor.py) |
| DwarfStar (antirez ds4, C reference) | `~/AI/ds4-ref` |
| build | `build-ds4-gpu/` (configure line in README "Running the V4.1 engine" + `-DSTRATA_DSV41_FQ_OP=ON`) |
| bench / scratch | `~/AI/v41-dev/` (`k143/` A/B, `dec7/force256.tok`, `r1ab-1237/p256.i32`, `nsys/`, `dspark/`, `dq/`, `eval/*.i32`) |
| serve | `~/.local/bin/strata-ds41-quetza` (port 8150, 512K) -> `tools/v41/serve/{serve_ds41.py,ds41-engine.sh,strata-ds41.json}`; logs `/tmp/strata-ds41.log`, text `/tmp/strata-ds41-text.log` |

Engine files: `ds41_dense.{hpp,cpp}` (dense half: attention, indexer, router, HC, Engram add, snapshot, `pass_m`),
`ds41_moe.{hpp,cpp}` (expert tier: VRAM LRU / arena / file tier / CPU pool / prefetch / grouped launches),
`ds41_generate.cpp` (CLI + `--serve` protocol + decode loop), `ds41_engram.*`, `ds41_ops.hpp` (act-quant rounding),
`ds41_geometry.hpp` (tensor map), `ds41_transcode.cu` (PTQ1_0 -> Q2_0 for MMQ).  Kernels: `src/kernels/cuda/fork/*`,
`src/kernels/cuda/iq_kernels.cu` (`native_expert_grouped`, `decode_rows_sum`).

## 10. Where the details live (open only for a specific question)
- `docs/ds4/V41_RESUME_PROMPT.md` - the full engineering log, newest first (~150 KB: history, every A/B).
- `docs/ds4/V41_DECODE_OVERLAP_PLAN.md` §5-§8 - the decode attribution and why each host-side lever failed.
- `docs/ds4/V41_PROGRESS.md` - the public step-by-step speed history.  `README.md` - public results.
- `~/.quetza-data/conductor/reports/*.md` - worker reports (`attn-node-profile`, `moe-miss-research` = literature on
  miss reduction, `dec-scout`, `gds-scout`, ...).

## 11. Paste-able resume prompt
> Continue the DeepSeek-V4.1-Flash Strata-DSV4.1 work for Mal in `~/AI/Strata-DS4` (branch `deepseek4`).  Read ONLY
> `docs/ds4/V41_START_HERE.md` first - it is complete; open other docs only for a specific question.  Goal: 50+ tok/s
> decode (DSpark retrain + engine draft/verify + n-gram drafting + plain-decode kernel work) and vision.  Check §5
> (in flight) first, then work §6 in order, measuring per §7 and following §8.  Conductor mode, one flash worker.
> Update §2/§5/§6 of START_HERE when something lands (keep it short; history goes in V41_RESUME_PROMPT.md).
