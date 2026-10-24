# DeepSeek-V4.1-Flash-1017 — the quantization plan (written 2026-10-18 08:00, end of session 6)

Everything needed to go from the 1017 FP8 download to a served, validated GSQ quant on this computer (RTX 4090 24 GB,
90 GB RAM). Uncensoring is Mal's part and is not covered here; the steps below run on whichever FP8 weights are going
to be served (the calibration and the quant must see the same weights the engine will run).

## 0. Facts already checked (2026-10-18)

- `deepseek-ai/DeepSeek-V4.1-Flash-1017`: 48 shards, 510.3 GB, 763.2B params; **config.json identical** to V4.1 (every
  key), **tokenizer.json + tokenizer_config.json byte-identical** -> the calibration tokens, the Engram n-gram hashes
  and every architecture constant are reused as they are.
- **No shard is reusable**: all 48 sha256 differ from V4.1 and from the dealignai copy, and sampled bytes of the Engram
  tables (layers.1 / layers.14, 98.3 GB each) and of `embed.weight` differ too. The whole 510 GB is needed.
- Download in progress (session 6): Engram shards 47 + 48 -> `/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017/`,
  resumable (`tools/v41/fetch_v41_loop.sh`; `RESUME-DOWNLOAD.txt` in that folder). Remaining 46 shards ~307 GB.

## 1. Lessons from session 6 that set the recipe

| Decision | Setting | Evidence |
| --- | --- | --- |
| GPTQ ordering | **act-order ON** (`V41_ACT_ORDER=1`) | all-layer sweep: ternary 0.978 / q2 0.988 / q3 0.987 of plain GPTQ, 37/40 layers better, same format |
| Hadamard rotation | **OFF** | < 1 GiB at equal quality, needs the fused down-projection kernel (`verify/s6/hadamard-engine-map.md`) |
| GSQ/AdaRound refinement | **OFF** | my version made errors worse (L13 ternary 24.9 -> 34.1 %) |
| Error-fixing adapters (LQER / dense-side) | **OFF** | 1-5 % error for up to +14 % size |
| Dense part | **Q8_0 in the file, Q6_K at load** (`STRATA_DENSE_REQUANT=q6_K`, the served default) | decode +11-35 %, KL +1 %, edit text identical |
| Tier map | **decide after the ternary-kernel work** (section 3) | v2 (-12 GiB, near-all-ternary band) = equal quality but slower today |
| Allocation proxy | trusted | v1 realized / predicted per layer 0.90-1.20 |
| Budget | v1's 142.38 GiB unless ternary is fixed | -12 GiB cost 0.36 pt on chat fixtures and speed |

## 2. Pipeline, step by step (times measured on this computer)

**Code changes needed first** — item 1 DONE 2026-10-18 (session 7; defaults unchanged, checked):
`V41_SRC="/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017:/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017"`
`V41_SRC_URL=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 V41_SRC_NAME="DeepSeek V4.1 Flash 1017"`.
**Serve/benchmark the 1017 GGUF from ntfs3/ext4, never exFAT** - MEASURED 2026-10-18: expert reads from exFAT NVME1TB
0.96 GB/s vs 5.1-5.3 GB/s from ntfs3 SSD NVME (raw NVME1TB 3.31 GB/s); that, not ternary, made v2 slower
(V41_TERNARY_PROFILE_PLAN.md section 0).  Build on NVME1TB if needed, serve from SSD NVME (it replaces v2 there - v2 is the served quant since 2026-10-18 11:05).
1. Source paths are hard-coded to the dealignai FP8 in `tools/v41/v41_calib.py` (`MAIN`, `DIRS`, lines 36-39),
   `tools/v41/calib_layers.sh` (`ok()`, lines 21-22), `tools/v41/v41_gguf.py` (`MAIN`, `DIRS`, `URL`, lines 55-59) and
   `tools/v41/v41_sidecar.py` (source metadata line 115 + wherever it opens shards). Add one env override, e.g.
   `V41_SRC="dir1:dir2:..."` (first dir = MAIN, all dirs searched for `<shard>.ok`), read by all four, and a
   `V41_SRC_URL` for the GGUF/sidecar metadata. `fetch_v41.sh` already takes the repo as an argument.
2. `v41_quant.py`'s `head` subcommand option bug is fixed (0ecf9111); `quant_layers.sh`'s end gate now works.

**Steps:**

| # | Step | Command (sketch) | Time | Disk |
| --- | --- | --- | --- | --- |
| 1 | Download all 48 shards (+ `.ok` sha256) | `FIT=1 bash tools/v41/fetch_v41_loop.sh <DIR> deepseek-ai/DeepSeek-V4.1-Flash-1017` per drive (`ONLY=`/`SKIP=` to split shards across drives) | ~7 h at 20 MB/s | 475 GiB |
| 2 | Calibration work dir | `mkdir /media/Crucial1TB/v41-calib-1017`; copy `tokens.npy`, `hashes.npy`, `meta.json` from `/media/Crucial1TB/v41-calib` (tokenizer + config identical) | 1 min | ~0.2 GB |
| 3 | Calibration pass: teacher forward + per-expert GPTQ screening at all 3 tiers | `V41_SRC=... V41_WORK=/home/mal/v41-work/calib-1017 bash tools/v41/calib_layers.sh 0 39` (runs `init` + the head gate; edit FIRST if a manifest exists) | ~11 min/layer = ~7.5 h | ~27 GB peak (h, shared, route; `--gc` keeps 2 layers) |
| 4 | Tier map | `python3 tools/v41/v41_alloc.py --work .../v41-calib-1017 --layers 0-39 --per-proj [--ternary-cap-at ... --budget-gib ...] --out .../alloc_1017.json` | seconds | - |
| 5 | GGUF layout (dense Q8_0 from FP8, RTN experts placeholder) | `V41_SRC=... python3 tools/v41/v41_gguf.py --out <NVMe>/V41-GSQ-1017.gguf --rtn-map alloc_1017.json --experts-in-place --no-engram --threads 4` | 12 min | 140-152 GiB on a fast drive |
| 6 | Quantize pass (sequential, act-order) | `V41_ACT_ORDER=1 V41_SRC=... bash tools/v41/quant_layers.sh <QWORK> 0 39 -- --map alloc_1017.json --gguf <file>` | ~3.8 min/layer = ~2.5 h (v2 measured) | ~27 GB QWORK |
| 7 | Engram tables for the engine (2 x ~98 GB) | `python3 tools/v41/extract_engram.py <src dirs> --to layers.1=<DIR_A> --to layers.14=<DIR_B> --verify 64` (NOT on exFAT: NVME1TB is exFAT and serializes random reads; ntfs3/ext4 only) | ~1 h (estimate) | 196 GB |
| 8 | DSpark (MTP) + vision sidecar | `python3 tools/v41/v41_sidecar.py --out <dir>/V41-MTP-Vision-1017.gguf` then the Q2_K requant (`tools/v41/v41_sidecar_requant.cpp`, as for `V41-MTP-Vision.Q2_K.gguf`) | ~30 min (estimate) | ~8.4 + 5.7 GB |
| 9 | Expert rank (VRAM / arena seed order) | `python3 tools/v41/engine/v41_rank.py --stats .../v41-calib-1017/stats --map alloc_1017.json --out <dir>/rank.bin` | 1 min | - |
| 10 | Serve config | copy `tools/v41/serve/strata-ds41.json` -> `strata-ds41-1017.json` with `-m`, `--engram`, `--rank`, `STRATA_DSPARK` sidecar paths | - | - |

**Do not reuse V4.1's rank.bin, Engram, or sidecar for 1017** - all three are derived from the weights or the tier map.
(Session 6 made exactly that mistake for v2: it reused v1's rank.bin, whose per-byte order was computed from v1's map.)

## 3. Decision before step 4: which tier map

**DECIDED BY MEASUREMENT 2026-10-18 10:39: the v2-style map** (v2 on the same drive = +5 % prefill, ~= / +10 % decode,
+12 % edits vs v1, 12 GiB smaller, quality ~equal - V41_TERNARY_PROFILE_PLAN.md s0): `--ternary-cap-at "20-31,33-35=1.0"`,
budget ~130 GiB.  Mal CONFIRMED v2-style (10:45) and then asked for it GENTLER in 20-35 (10:50) - the sweep and pick rule
are in the 'Mal 10:50' block just below.
The old items 0-3 below are history.

**Mal 10:50: protect chat-style quality, accept a little less size saving** (v2 lost 0.36 pt API top-1 / +4 % KL on
the chat-style API fixtures; ~half of it from the near-all-ternary band 20-35; agentic/tools/code stay the priority):
1. **Calibration mix rebalanced** (`v41_calib.py prep --mix`, built 10:50 in `/home/mal/v41-work/calib-1017`):
   agentic_long 8, agent 5, code 6, reason 3, web 2, zh 2, **chat 4**, **multi 2** (V4.1: 10/5/6/4/2/3/1/1); same
   32 x 8192 tokens = same calibration time; held-out now includes a chat sequence.  The V4.1 set is kept in
   `calib-1017/v41-mix/`.  NOTE: prep reads live local text (~/AI sources/docs), so re-running it never reproduces an
   old set - the saved tokens.npy/hashes.npy/meta.json ARE the record.
2. **Allocation**: v2-style but gentler in 20-35 - sweep `--ternary-cap-at "20-31,33-35=X"` for X in 1.0 / 0.85 / 0.75
   against budgets -12 / -10 / -8 GiB on the 1017 stats; pick the smallest file whose proxy is <= v1-recipe's AND whose
   held-out chat domain (head gate per domain) is not worse; then the 100 API fixtures (validation 2-3) decide.


0. Why v2 was slower is open - full analysis + profile order: `docs/ds4/V41_TERNARY_PROFILE_PLAN.md` (stale rank.bin
   = small; exFAT drive = new prime suspect; ternary decode kernel bounded at ~1-2 ms/token; transcode + arena
   fragmentation unmeasured).
1. **First (next session, ~15 min)**: regenerate v2's rank (`v41_rank.py --map /media/Crucial1TB/v41-calib/alloc_v2.json`)
   and re-run the v1/v2 speed test (`verify/s6/v1v2-speed.sh` pattern) - the session-6 v2 slowdown may be partly the
   stale rank.bin.
2. **Ternary kernel work** (priority 2): profile v1 vs v2 (decode per expert type, the PTQ1_0 -> Q2_0 transcode in the
   prompt MMQ path, `ds41_moe.cpp` ~3582), native PTQ1_0 MMQ, a faster trit unpack, arena/VRAM slot sizing per format
   class.
3. Then pick: if v2 >= v1 speed after 1-2 -> a v2-style map (`--ternary-cap-at "20-31,33-35=1.0"`, budget 130.38 GiB or
   re-swept); else v1's recipe (`--per-proj`, cap 0.5, budget 142.38 GiB) with act-order.

## 4. Validation (in order, each against v1 numbers)

1. Head gate (calib held-out, 49K tokens): v1 ppl 5.645 / top-1 0.6745, teacher (V4.1) 4.915 / 0.7011 - the 1017
   teacher number comes from step 3's own head gate.
2. **New FP8 fixture reference for 1017**: `tools/v41/quality/fp8_teacher_score.py prep/score` with `V41_SRC` pointing
   at the 1017 shards (fresh `V41_WORK`, ~17 min) - the old `/media/Crucial1TB/v41-fp8-fixtures` is the V4.1 FP8.
   The API's `deepseek-flash` IS `DeepSeek-V4.1-Flash-1017`, so the 1017 FP8 vs API gap should shrink (V4.1 FP8: 93.65 %).
3. End to end: `tools/v41/v41_e2e.py` with `V41_GSQ_GGUF=<1017 file>` (KL vs the 1017 FP8, API top-1, entropy).
4. Engine: smoke, `verify/s6/v1v2-speed.sh`-style throughput + edit probe (back to back with v1), Gauntlet v2
   (`GAUNTLET_OUT=... bash verify/e2e/gauntlet-local-all.sh ds41` with the 1017 config), the overthinking pairs
   (abc400_c / abc395_f / abc398_f at CB_THINK=98304; traces saved).
5. Only then: HF upload (a new repo/card), README/BENCHMARKS, wrappers.

## 5. Disk plan (Mal, 2026-10-18) - UPDATED 09:50

- **Where the 1017 FP8 lands** (`tools/v41/fetch_1017.sh`, one download at a time after the Engram shards):
  Crucial1TB `/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017` = MAIN (small files + shards 1-39), Crucible
  `/media/mal/Crucible/models/DeepSeek-V4.1-Flash-1017` (shards 40-46 = layers 37-39, head/norm, mtp.0-2), Team CX2
  `/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017` (shards 47-48 = Engram L1 / L14).  Use:
  `V41_SRC="/media/Crucial1TB/models/DeepSeek-V4.1-Flash-1017:/media/mal/Team CX2/models/DeepSeek-V4.1-Flash-1017:/media/mal/Crucible/models/DeepSeek-V4.1-Flash-1017"`
  `V41_SRC_URL=https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 V41_SRC_NAME="DeepSeek V4.1 Flash 1017"`
- Calibration work dir: `/home/mal/v41-work/calib-1017` (system drive, not Crucial).  It can stream behind the
  download: start `calib_layers.sh 0 39` once shards 1-3 are verified (init needs embed = shard 2).

- **Engram shard read speed for the calibration pass** (layers 1/14 read random rows of shards 47/48; measured
  2026-10-18, 4K O_DIRECT reads, 32 threads): NVME1TB exFAT 10.3K/s, Team CX2 5.6K/s (during the download's writes),
  Crucible (fuseblk) 4.4K/s, ntfs3 SSD NVME ~290K/s (s5).  The 1017 shards 47/48 land on Team CX2: if layers 1/14
  of the calibration crawl, time one layer first; the engine never reads the shards (it reads the extracted
  `engram-L1/L14.safetensors`, step 7), so this only affects calibration.
- The dealignai duplicates of shards 47/48 on NVME1TB were deleted 2026-10-18 (copies on Crucible / Team CX2) to make
  room for the v2-on-SSD-NVME speed test (`verify/s7/swap-v2-test.sh`).

- 1017 FP8 to a non-NVMe drive (Team CX2 has the Engram shards; the other 46 need ~307 GB elsewhere).
- (11:05) v2 is now the SERVED quant on SSD NVME (`/media/mal/SSD NVME/AI/V41-GSQ-v2/`); v1 and the NVME1TB v2 copy are
  deleted.  The 1017 GGUF replaces v2 on SSD NVME once validated.
- Keep until 1017 is served: v2 (above), `V41-Engram` (Crucial + SSD NVME), the
  dealignai FP8 (Crucial `models/` + shards 47/48 on NVME1TB), `v41-calib`, `v41-fp8-fixtures`.
- During steps 3/6: ~27 GB work dir each (system drive has ~103 GB free).
