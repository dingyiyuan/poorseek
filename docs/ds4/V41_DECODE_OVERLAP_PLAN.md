# V4.1 decode: how to get the GPU busy (implementation plan)

Companion to `V41_RESUME_PROMPT.md` §UPDATE 2026-10-12 ~20:45.  Written 2026-10-12 ~21:00 after the
host-call attribution (conductor, `nsys/dec3.sqlite`) and the independent overlap scan
(`~/.quetza-data/conductor/reports/decode-overlap-scan.md`).  Every number below is measured, not inferred.

## 1. The problem, measured

Per decode token (256-token prompt, greedy, 4.5 s trace window, 96 tokens):

| | ms/token |
|---|---|
| GPU busy (union of all kernels+memcpy+memset) | 18.27 |
| **GPU idle** | **28.60 (61%)** |
| — host blocked in `cudaStreamSynchronize` | 26.21 total; owns 11.28 of the gaps |
| — **no single call owns the gap** (~884 short host calls) | 14.68 |
| — of that, **host is idle too (= a pure pipeline bubble)** | **12.21** (scan §3.3) |

1680 host CUDA calls/token: `cudaLaunchKernel 417.7` + `cudaMemcpyAsync 387.1` + `cudaStreamSynchronize 257.7`
+ `cudaGraphLaunch 80.7`.  **By stream:** `14 = gp.s (the tier)` 322 kernels/tok, 89.2 syncs = 5.03 ms;
`16 = ggml's dense stream` **121.7 syncs/tok = 17.02 ms** (3/layer, mean 139.9 us).
**Max concurrent kernels = 1** over the whole window - the two streams never overlap.

Two independent attributions agree on all of this (counts within window trim), which is why the numbers are
trusted.

## 1b. BASELINE for slice A - measured on `gen.v7` (packing + S4 in), `nsys/dec7.sqlite`

Reproduce with `python3 ~/AI/v41-dev/nsys/calls.py nsys/dec7.sqlite 96`.

| metric | gen.v7 | dec3 (before packing/S4) |
|---|---|---|
| GPU busy / idle | 18.42 ms/tok / **60.7% idle** | 18.27 / 61% |
| **MAX CONCURRENT KERNELS** | **1** | 1 |
| host calls/token (total) | **1606** | 1680 |
| `cuKernelGetName` | 418.5 | - (not in the top list) |
| `cudaLaunchKernel` | 418.5 | 417.7 |
| `cudaMemcpyAsync` | **350.7** | 387.1 |
| `cudaStreamSynchronize` | **217.5** | 257.7 |
| `cudaEventRecord` | 106.1 | 105.4 |
| `cudaGraphLaunch` | 80.6 | 80.7 |
| `cudaStreamWaitEvent` | 10.0 | 10.1 |
| stream sync ms/tok | s16 17.13, s14 4.73 | s16 17.02, s14 5.03 |
| kernels/token | s14 322.9, s16 56.3 | s14 322.1, s16 56.2 |

**The two changes show up exactly in the counts and nowhere else - which is the point:**
- S4 removal = **-40.2 syncs/token = 1.0/layer**, precisely as designed, and still speed-neutral because the wait
  moves into the sum sync (the tier profile proves it).
- Metadata packing = **-36.4 memcpyAsync/token, NOT the ~280 that "one copy per class" predicted** - most groups
  have only 2-3 distinct format classes among 6 experts, not 9.  That is why it measured neutral, and it is now
  explained rather than just observed.
- `cuKernelGetName` is issued **once per kernel launch** (418.5/token) - a pure diagnostic, ~0.09 ms/token,
  removable for free if anyone wants the last crumbs.

**Slice A's acceptance test:** `MAX CONCURRENT KERNELS > 1` in an `STRATA_DMA_OVERLAP=1` trace, plus forced-token
identity across >=3 runs (it is a race, so one clean run is not evidence).

## 1c. Slice A VERDICT + the volume finding it exposed (2026-10-12 ~21:35)

**Slice A is implemented and CORRECT, but its ceiling is small - and measuring it exposed where the bytes are.**

- Implemented: `STRATA_DMA_OVERLAP=1` (default off) puts the PCIe group's blobs on a new copy stream `gp.s_dc`
  with one event handoff before the miss kernels (see 3b).  It demonstrably took effect: `gp.s` (stream 14)
  memcpy fell **69.87 -> 43.27 MB/token** and the new stream holds the **26.59 MB** the PCIe group moves.
- **Correctness PASS**: 5 forced-token runs (2 off, 3 on) all byte-identical to the forcing file - no corruption,
  which is the bar that matters because this is a race.
- **Speed: no gain.**  off 26.87 / 26.96 tok/s, on 26.52 / 26.55.  GPU idle 60.7% -> 59.3% (only ~0.5 ms/token
  filled).  The ceiling is arithmetic: the PCIe group moves 26.6 MB/token = **~1.7 ms at 15.8 GB/s**, and only
  part of that hides.  Below the noise floor of tok/s, so this is a keep-or-revert-on-taste change, not a win.
- **METRIC BUG FOUND AND FIXED:** `calls.py` swept only `CUPTI_ACTIVITY_KIND_KERNEL`, so a kernel||DMA overlap is
  invisible - it read `MAX CONCURRENT KERNELS = 1` for BOTH traces.  It now sweeps kernels+memcpy+memset and is
  named `MAX CONCURRENT ACTIVITIES`.  On these traces it reads **2 for both** - i.e. slice A did not raise the
  peak concurrency either, consistent with the tiny gain.

**THE FINDING - the H2D volume is one activity, and it is almost all PREFETCH:**

| stream | MB/token | what |
|---|---|---|
| `s_pf` (prefetch) | **274.71** | the speculative expert prefetch - 29.2 DMAs/token of 9.5 MB |
| `gp.s` / demand (pcie+file+metadata) | 69.87 -> 43.27 | the actual misses |
| ggml's stream / s_cg | 0.93 | activations |
| stream 13 | 0.58 | - |

**274.71 MB/token at 15.8 GB/s is 17.4 ms/token of PCIe time - comparable to the entire 37.5 ms token - and the
engine's own counters say only 2499 of 8065 prefetch DMAs per 256 tokens are ever USED (31%; ~10/token vs
~31.5/token issued).**  So roughly two thirds of the expert traffic (and 5x the demand traffic) is speculative
guesses that miss.  Even if it overlaps, it competes for the same PCIe path as the demand copies that ARE on the
critical path - which is the more likely explanation for why the demand-copy overlap (slice A) bought so little.

**Therefore the next slice is not more overlap, it is LESS PREFETCH VOLUME:**
1. skip a prefetch whose expert is already VRAM-resident (is it even checked? - verify in `prefetch()`), and
2. cut the issued set: today `kPredW=16` / `kMaxPf=6` issues ~29 DMAs/token for ~10 used - try issuing only the
   top 4-8, or only when the prediction score is above a percentile (the Fate widening, scout §1), and
3. re-measure with `MAX CONCURRENT ACTIVITIES` plus the `prefetch N issued / D outside / U used` counters, not tok/s.
The prefetch's own break-even was measured NET NEGATIVE once already (resume doc: finish graph +1.5 ms, issuing
+1.3 ms, vs -1.8 ms tier) - this volume number says why, and says the fix is to send far less.

### 1c-bis. THE ABOVE WAS WRONG - the `--pf-b` sweep says the prefetch EARNS its traffic (2026-10-12 ~22:05)

`dec10/seq10.sh` swept `--pf-b` with forced tokens (all runs byte-identical output, so identical work):

| `--pf-b` | tok/s | tier ms | VRAM hits / 256 tok |
|---|---|---|---|
| **1.43 (default)** | **27.45 / 27.49** | **16.x** | 59511 |
| 0 (off) | 25.23 / 25.69 | **21.x** | **56939** |
| 3.0 | 25.90 / 25.74 | 18.x | 59966 |
| 6.0 | 25.64 | 18.x | 60049 |

**The default is the FASTEST of the four, and turning the prefetch OFF is the slowest.**  With `pf-b 0` the VRAM
hits fall 59511 -> 56939 (-2.4 points) and the tier rises 16 -> 21 ms: so the 275 MB/token is buying exactly the
hits it is supposed to, worth ~5 ms of tier, and its DMA overlaps well enough to be net positive.  Going the
other way (3.0, 6.0) buys a few more hits but less than the extra traffic costs.  **So: the prefetch is
well-tuned; "the volume is waste" was a wrong inference from the byte counts alone.**  Do not touch `--pf-b`.

**What the same traces DO establish (the real picture):**
- With the prefetch off, **GPU busy activity falls to 10.01 ms/token** (idle 78.6%) - so the actual COMPUTE is
  only ~10 ms/token; of the 19.47 ms "busy" at the default, **~9.5 ms is prefetch DMA activity**, not compute.
  The remaining ~17 ms of the token is host waits.  That is the budget to attack, and it is not a kernel problem.
- `MAX CONCURRENT ACTIVITIES` is 2 with the prefetch on and **1 with it off** - the prefetch stream is the only
  source of concurrency in the engine.  Slice A added a second copy stream and did not raise the peak, which is
  consistent with its ~0.5 ms of effect.
- NOTE for anyone reusing `~/AI/v41-dev/nsys/calls.py`: it assumes the window is the last 4.5 s == N tokens, so
  the per-token numbers drift when a config changes the decode rate (the window is fixed wall-clock).  Compare
  like-for-like runs, or pass the real token count.

## 2. Why it is serial - the root cause

Per layer the chain is `attn_router(l)` -> **[host reads the top-k ids]** -> `tier.run_to_dev(l)` ->
`finish_predict(l)` -> `attn_router(l+1)`.  The host cannot launch the tier until it has the ids, and the tier's
own launch decisions (which tier each expert is in, the PCIe budget, arena slot -> device pointer) are host loops
over host-side tables (`gp.arena.slot_of`, `gp.cache`, `skip_miss`, the residency bias).  So every layer is a
host round trip, the device queue drains between layers, and the expert H2D copies (328 MB/token, scan §3.1) sit
between the syncs instead of under the compute.

## 3. Targets, in order

### Slice A - overlap the expert H2D with compute (scan's #1; ceiling 11-14 ms/token; output EXACT)
The weights are a *copy*, so reordering/overlapping changes no bits.
- The destination is a fixed per-layer staging region (`gp.d_stage`, `gp.d_pf` + `pf_at` stride, `B` = blob
  stride) and the source is a stable pointer for the token (`gp.arena.ptr(layer, e)`, a pinned file-stage buffer,
  or a prefetch slot).
- Today the copies are issued on `gp.s` (the compute stream) *interleaved* with the launches
  (`ds41_moe.cpp:2513` "pcie dma" in `run()`; the `gpu_run` equivalents at the PCIe/file group builds), so they
  serialise against the kernels on the same stream.
- Change: issue the *demand* copies for layer `l` on a **copy stream** (`gp.s_cp`, already exists,
  `ds41_moe.cpp:2787`) with an event handoff to `gp.s` (`cudaStreamWaitEvent`, the pattern is already used at
  2654/2708/3052/3084/3164/3197), and **double-buffer**: once layer `l`'s experts are known, the copies for the
  next layer's *predicted* experts (already produced by `finish_predict`, `ds41_dense.cpp:1376`) go out
  immediately, so they run under `attn_router(l+1)` + `tier(l+1)`'s kernels.
- Requires per-layer (not one-shot) staging regions: currently `gp.d_stage` is one region per call, and the
  prefetch slots are `pf_at`-strided.  Give it two (ping-pong) regions and an event per region.
- Verify: forced-token identity (`STRATA_FORCE_IDS`), then the nsys per-stream call counts and the **overlap
  metric** (max concurrent kernels > 1 is the whole point).

### Slice B - collapse the host calls (prerequisite for C; the packing part is coded)
1. **`launch_group` metadata packing - CODED** (`ds41_moe.cpp:1900`): every format class of a group is packed
   into `h_meta` contiguously and copied up in ONE `cudaMemcpyAsync` (was one per class; `STRATA_GRP_ONE_COPY=0`
   restores).  ~7 fewer calls/layer (~280/token).
2. **CUDA-graph the tier's class launches.**  `native_expert_grouped` already takes ptr/st/ng/dst/tok as DEVICE
   pointers into the metadata block, so a captured graph's node params are FIXED - only the block's contents
   change per token.  Pack into **fixed per-class slots**, launch all 9 classes every layer (absent -> n=0; the
   kernel must accept `ng == 0`), capture once at init.  Per layer: 1 memcpy + 1 `cudaGraphLaunch` instead of
   ~20 calls.  **884 -> ~80 calls/token.**
3. `finish_predict`'s blocking readback (`ds41_dense.cpp:1391`, `ggml_backend_tensor_get`) is the third ggml
   sync/layer and exists only to feed `tier.prefetch`.  `finish(l)` and `attn_router(l+1)` are on the same
   stream, so no host sync is needed between them: issue the readback async, run `attn_router(l+1)`, then sync
   ONCE for both.  Trade-off to measure: the prefetch then overlaps `tier(l+1)` instead of `attn(l+1)`.
   Control experiment: `pf_b=0` (plain `finish()`, no predict graph, no readback).

### Slice C - move the selection on-device (kills S1 and the 12.21 ms bubble; unlocks real pipelining)
A small kernel that reads the router ids straight from `im.o_ids`, looks up a device-side
`(tier, device-pointer)` table (built once per token from `slot_of`/the cache), and writes the metadata block
for `native_expert_grouped` **in place**.  Then the host never needs the ids: it only ENQUEUES
`attn_router` -> tier -> `finish_predict` for every layer and reads the result at the end, running tens of layers
ahead of the device.  S1 (the last unavoidable host wait) disappears, the bubble disappears, and the copies
overlap by construction.  Prerequisite: slice B's fixed metadata layout.  This is the one that changes the
architecture; A and B are the steps that pay for themselves on the way.

## 3b. Slice A - implementation gotchas found by reading the code (READ BEFORE EDITING)

- **`gp.s_cp` does NOT exist on the decode path.**  It is created only in the prefill/chunk path
  (`ds41_moe.cpp:2834`) and every user guards on it (`2708`, `2733`).  Slice A needs a stream created next to
  `gp.s` in the context init (`ds41_moe.cpp:1194`, where `gp.s`/`gp.s_pf` are made) plus its destroy in the same
  place as theirs, not a reuse of `s_cp`.
- **The arena/pool question is ANSWERED - the ordering is simpler than it first looks.**  The CPU pool's reader
  threads write only arena **victim** slots, and a victim is only ever an arena expert with `last_use < im.tick`,
  i.e. NOT routed by this call (`ds41_moe.cpp:2174-2175`); the PCIe DMAs read THIS call's arena experts, which all
  have `last_use >= im.tick`, so **source and victim sets are disjoint by construction**.  Arena slots are
  indexed per `(layer, expert)` (`NXA`-strided `slot_of`), and a layer is visited once per token, so there is no
  cross-layer aliasing either.  The host also joins the reader threads before any DMA is issued
  (`for (auto& t : rd) t.join();`, `2198`), so "writes before the async copy call" holds in program order.
  Precedent in-tree for the pattern that IS needed: `if (direct && gp.s_pf) cudaStreamSynchronize(gp.s_pf), "pf
  sync"` at `2167` (no prefetch DMA may read a slot the readers are about to overwrite).
- **So the change is:** issue the `pcie dma` copies (`2068/2073/2079`) plus the `lru stage copy` D2D (`2070`) on
  the new copy stream, `cudaEventRecord` on it, then a single `cudaStreamWaitEvent(gp.s, ev)` immediately before
  `launch_group(1, np, ...)` at `2086` - one wait covers the whole group, and the kernels keep running on gp.s.
  `gp.s_cp`-style discipline: nothing else may be issued on the copy stream within the layer without its own
  ordering.
- d_stage regions are disjoint per group (pcie at `[0, np)`, file at `[np, np+nfg)`, stride `B`), so the pcie and
  file copies can share the one copy stream; the file group's copies are issued after its reader threads join
  (`2198`), so they order after the reads for free.
- Ship it behind an env flag (`STRATA_DMA_OVERLAP=1`), default OFF, until the nsys overlap metric
  (max concurrent kernels > 1) and forced-token identity both hold across several runs.  The forced-token check
  is the correctness bar but it is a RACE-check, not a proof: run it several times, and read the overlap number
  as the real evidence that the copies moved off the compute path.
- Expected size: the pcie DMA is ~3.3 ms/token and the file DMA ~1 ms (scout's miss math), so overlapping them
  with the resident-expert compute is worth ~4 ms/token - real but BELOW the 5 ms noise floor of tok/s, which is
  why the metric must be the nsys overlap count, not throughput.  (The scan's 11-14 ms ceiling assumed
  overlapping the copies with the attention too, which needs slice C first.)

## 4. How to measure (mandatory - the noise floor will lie to you)

With IDENTICAL work (same tokens, same 59511 VRAM hits / 255 cpu) the tier phase swung **17.63 - 23.06 ms (30%)**
between runs on a loaded box, against 16.5-17.2 on a quiet one.  A ~2 ms host-time effect is invisible in tok/s.

1. Correctness bar is always **forced-token identity**: `STRATA_FORCE_IDS=<file>`, then `diff` the emitted tokens
   against the forcing file - 0 differing lines.  Any kernel/launch change must pass this before its speed is
   discussed.
2. Speed for host-call work is measured by **nsys call COUNTS** (`nsys profile -t cuda,osrt
   --cuda-graph-trace=graph`), not tok/s: count `cudaLaunchKernel` / `cudaMemcpyAsync` / `cudaStreamSynchronize`
   per token and the **max concurrent kernels** (the overlap metric).  A helper exists at
   `~/AI/v41-dev/nsys/idle_attrib.py`.
3. CPU-heavy work (builds, torch, the local analysers, the HF upload's hashing) must NOT run beside a timed
   decode run - it is what produced the 30% swing.
4. The gate for a numerics-affecting change is still `test_ds41_moe_gpu` (worst rel ~1.44e-2) plus ppl/KL on
   code/agent/reason against the saved `dq/*_q8.logits` if the change can move bits.

## 5. Slice C verdict — CORRECTED (2026-10-12 ~22:30, adversarially verified)

**§3's Slice C does NOT remove S1, and its stated mechanism was mis-attributed.** Two independent readings (worker
`slice-c-ondev-select` + a read-only verifier `verify-slice-c`) agree the negative stands. **Do not re-attempt "move
selection on-device so the host never reads the ids" — the premise is false.**

- **The exact serialising dependency (VERIFIED):** `routed(l)` = `decode_rows_sum` (`ds41_moe.cpp:2378`), which is
  **k-ordered in double** (`iq_kernels.hpp:72-74`), and it is the input of `attn_router(l+1)`
  (`ds41_generate.cpp:451`). It cannot run until **all** k rows exist — including the miss rows produced by
  **host-only work keyed on the host copy of `ids6`**: the NVMe `read_part` threads (`ds41_moe.cpp:2155`/`2218`) and
  the CPU pool dispatch (`2270`). So the host must **both** read the ids **and** finish the miss service *every
  layer*. There is no end-of-token sync; layers cannot run ahead.
- **The id readback is not the bubble.** It is already one batched async sync for all three tensors
  (`ds41_dense.cpp:1322-1326`). Splitting it onto a copy stream removes at most one sync/layer.
- **On-device hits buys ~0.** The hits launches (`2000`) already precede the file join (`2286`) and the pool finish
  (`2283`), so the hits already overlap the miss service.
- **The 12.21 ms bubble ≈ the per-layer host NVMe read wall** (`file 1538 ms` / 256 tok = **6.0 ms/token**, inside
  the 16.58 ms tier) — INFERRED from code order + measured `file_ms` (a direct nsys gap↔reader-window correlation
  would confirm; not run here). Killing S1 does not touch it.
- **Corrections to the §3 text:** the headline "**97.2% hits**" dropped the 672 file routings from the denominator —
  the honest fraction is **96.2%** (and the file tier is the *dominant* time). `launch_group` is at
  `ds41_moe.cpp:1926-1986` (not 1901). "9 classes" is a runtime fact printed at `ds41_generate.cpp:364`.
  The CPU pool (`2270`) is a host *load-balancing* choice, not hardware-forced — unlike the NVMe read.

**Lever ranking, verified (verifier Task C):** **B (#1)** graph-capture the tier's classes (~884→~80 host
calls/token; risk MED; payoff 0–5 ms **UNPROVEN** — S4 sync removal and metadata packing both measured *neutral*)
> **C′ (#2)** device-servable file tier — the only lever that removes the 6.0 ms read wall (risk HIGH; needs
GPUDirect Storage/cuFile, which is still **host-issued** so S1 stays) > **hit-rate (#3)** (pf-b already optimal,
retracted) > **device-hits (#4)** (~0).

Evidence: `~/.quetza-data/conductor/reports/slice-c-ondev-select.md`, `verify-slice-c.md`.

## 6. C′ verdict + the FALSIFIED bubble attribution (2026-10-12 ~22:40, read-only scout `gds-scout`)

**C′ (device-servable file tier) is NOT worth its entry cost — do not start it.**
- The 6.0 ms/token `file_ms` **IS the NVMe transfer itself**: 672 file routings/256 tok = 2.625 blobs × ~9.6 MB =
  **25.2 MB/token** at the measured **4.65 GB/s** = 5.4 ms. It is **not** the thread join and **not** the H2D hop
  (the `file dma` H2D at `ds41_moe.cpp:2403` is separate, ~1 ms, and outside `file_ms`).
- **GPUDirect Storage is not operational here:** `libcufile`/`cufile.h`/`libcufile-13-4` are present, but the
  **`nvidia-fs` kernel module is absent** (not loaded, not on disk, `modinfo` fails, no `/proc/driver/nvidia-fs`)
  — `libcufile` itself prints "nvidia-fs driver is not loaded". Repo has **0** cuFile/GDS refs.
- **Decisive blocker:** the gguf lives on `/dev/nvme0n1p3`, fstype **`ntfs3`**, which is **not** in libcufile's
  supported-fs list (ext4/btrfs/lustre/tmpfs/wekafs) -> C′ needs the **152 GiB blob tier relocated to ext4/xfs** on
  a box whose `/` is 93% full. AD102/4090 GDS is officially best-effort. GDS' ceiling anyway is **~1–1.5 ms/token**
  (removes the bounce-memcpy + H2D hop only; the NVMe transfer and the **host-issued** read — hence S1 — remain).
- **Resident-experts form is capacity-impossible:** file tier ≈ 8652 × 9.6 MB ≈ **83 GB** vs ~13.5 GiB free VRAM /
  60 GiB arena cap / 90 GiB RAM.

> **CORRECTION 2026-10-13 ~05:10 (session 5): the next paragraph's conclusion is a MISREAD.** In the `spec4` run
> the tier phase DID shrink (16.58 -> 15.15 ms/token, -1.43); the gain was eaten by the speculation's own costs
> (`finish` 1.91 -> 3.34 = +1.43 for the extra predict-ahead graph + readback, and a new `spec` phase of 1.23 ms of
> issuing on the decode thread). So the NVMe read wall **is** on the critical path; that implementation cost as much
> as it saved. A version that issues the reads from the top-16 prediction `finish_predict` already computes, on a
> dedicated I/O thread (no extra graph, nothing on the decode thread), would convert.

**THE IMPORTANT PART — the bubble attribution is FALSIFIED:** `STRATA_SPEC_TOP=4` cut `file_ms` 1538 -> 1098 ms
(**−1.7 ms/token = 29% of the wall**) and gained **0.0 tok/s** (26.48 vs 26.35). So the **file read wall is NOT the
critical path** — §5's "the 12.21 ms bubble ≈ the NVMe read wall" was *inferred* and is now **contradicted**.
`spec8` 25.21 / `FILE_PRED` 23.51 (both worse); `--pf-b` already optimal.

**Emerging target:** `attn_router` = **17.83 ms/token (47%** of the token) ≈ the ggml dense stream's **17.13 ms** of
`cudaStreamSynchronize` (121.7 syncs/token, 3/layer) — vs the tier stream `gp.s` at only **4.73 ms** (the stream
slice B targets). Under investigation by `attn-router-decomp`.

Untried cheap partials the scout identified: smaller quant on **file-tier-only** blobs (the wall is linear in
bytes); `arena_adapt=1` (default off — reads straight into the victim arena slot, bits-identical, skips a host
memcpy); more `read_part` threads (drive has ~30% BW headroom). Dead end: `advise_file` (O_DIRECT bypasses the page
cache). Evidence: `~/.quetza-data/conductor/reports/gds-scout.md`.

## 7. CORRECTION — §1/§1b/§2's attribution was a TOOLING ARTIFACT (2026-10-12 ~23:00)

**The engine is NOT host-blocked, and `attn_router` is DEVICE time.** The dense half runs as **CUDA graphs**
(ggml-cuda capture): **80.6 `cudaGraphLaunch`/token (2.01/layer) = 16.28 ms/token of device execution**. Those
activities appear **only** in `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` — a table `calls.py` and `idle_attrib.py` do **not**
sweep — and with the default `--cuda-graph-trace=graph` the graph *nodes* never enter the KERNEL table. Verified
directly in `dec7.sqlite`: `KERNEL` = 36406 rows, **0 with a non-NULL `graphId`**; `GRAPH_TRACE` = 7742 rows in the
same 4.5 s window = 1563.16 ms = **16.28 ms/token**, 84 distinct `graphExecId`. That single missing table is why the
dense stream looked empty to every tool in the tree.

**Overrides:** §1/§1b GPU busy **18.42 -> 27.66 ms/token**; idle **60.7% -> 41.0%**. §2's "117.7 of 121.7
ggml-stream syncs start >2 ms after that stream's last kernel = ~140 us each with nothing to wait for" is
**REFUTED**: with graphs included, **81.9 of 121.7 s16 syncs find device work in flight (17.10 of 17.13 ms)** and the
median sync returns **2.7 us** after the device drains — device-bound, not host-bound. §1c-bis "compute is ~10
ms/token" -> with the prefetch off the dense graph is still **15.07 ms/token**; compute is **~15+ ms**.

**`attn_router` (17.83 ms/token) decomposes as:** **15.18 ms** attention+router graph (39.5 big graphs/token, 384 us
each, one exec per layer) + **1.10 ms** predict graph (27 us, 41.1/token) + ~1.5 ms host/batched readback. §5's
"the 12.21 ms bubble ≈ the NVMe read wall" is a *host* statement inside the tier phase — it is **not** part of
`attn_router`, which has no room for it once the graphs are visible.

**Consequences:**
- **Slice B targets the wrong stream and its premise is false.** It hits `gp.s` = the tier (**4.73 ms** sync) vs the
  dense stream (**17.04 ms**, 3.6x larger). More importantly its rationale ("host blocked on an idle device") is
  false on **both**: s14's syncs are equally device-bound (40.4/49.6 in flight, 1.4 us post-drain lag). B's honest
  ceiling is host CPU (`cudaGraphLaunch` 3.62 ms/token + ~2 ms launch loops), which does not convert to tok/s while
  the device is busy under it. Keep B only as host-CPU / loaded-box hygiene.
- **"CUDA-graph the dense half" is ALREADY DONE** (80.6 launches/token) — plan text proposing it as new work is stale.
- **The #1 and only double-digit lever is shrinking the 384 us/layer attention graph itself** (15.18 ms/token): KV
  quantization (the PTQ1_0/Q3_K work already in the tree behind `STRATA_PTQ1_QUARTER`), a smaller window, MLA
  absorb, split-K/flash-decoding. **Not yet measured** — needs an `nsys --cuda-graph-trace=node` run to see inside
  the graph and get achieved GB/s per node. **That is the next experiment.**
- §5/§6's ranking is superseded: **attention-graph shrink (15.18 ms, HIGH risk) >> host-CPU hygiene (B, B.3,
  `cuKernelGetName` ~0.09 ms) > everything else (~0).**

**Tooling:** `calls.py` / `idle_attrib.py` must add `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` to the busy union and report
graph launches/token, or every future reading repeats this error. Evidence:
`~/.quetza-data/conductor/reports/attn-router-decomp.md`.

## 8. Node-level profile — the real bottleneck, and §7's lever ranking REFUTED (2026-10-12 ~23:05)

`nsys --cuda-graph-trace=node` (nodes EXPANDED into the KERNEL table) on the dense stream, 96-token window, forced
tokens. **This is the first time anything inside the 384 us/layer graph has been visible.**

**Per layer (~393 us of device time):** `mul_mat_vec_q<Q8_0>` **223.4 us (57%)** (8 nodes x 27.85 us) · ~86 us
**elementwise zoo** (49 names, `k_bin_bcast`/`scale_f32`/`quantize_q8_1`/`rms_norm`/`concat`/`dsv4_hc_*`, mean < 5 us)
· 46.4 us other quant matvecs (`mul_mat_vec_f` families) · 23.6 us `rms_norm` x4 · **13.2 us `flash_attn_ext_f16`
(3.4%)**. **No single node is 384 us — the 384 us is the SUM.**

**Per token:** `mul_mat_vec_q<Q8_0>` **8.94 ms** (320.9 nodes, 27.85 us each) · elementwise zoo **5.45 ms (33%)** ·
attention total (flash_attn + softmax + indexer) **~0.74 ms = 4.5%**.

**Bandwidth:** the dense Q8_0 matvecs move **6.17 GB/token at ~690 GB/s blended = 68% of the 4090's 1008 GB/s
peak** -> the dense half is **bandwidth-bound on weight bytes**, NOT on the 2560 KV cache.

**§7's own "#1 lever = shrink the attention graph / KV maths" is REFUTED** — attention is 3.4% of the layer. The
combined ceiling of EVERY attention-level lever (KV quant, smaller window, MLA absorb, split-K) is **~0.7
ms/token**. Do not spend a slice there.

**Levers that have real money (ranked):**
- **A) Requantize the dense projections Q8_0 -> Q4_0/Q4_K: the node profile's ESTIMATE is −4.0 to −4.5 ms/token, but
  THE TREE ALREADY CONTRADICTS IT.** `STRATA_DENSE_REQUANT` was measured (`~/AI/v41-dev/dq/results.txt`): **q6_K
  7.59 -> 5.86 GB and q5_K 7.59 -> 4.91 GB both keep decode at ~29 tok/s** (q5_K 29.17 / 28.85 vs the q8 baseline
  28.3-29.5) — i.e. **22-35% fewer dense bytes bought no decode speed**, and the handoff already concluded "it frees
  VRAM, ~+100 experts, but does not speed decode". The reason is the same one the q6_K row gives: **ggml's K-quant
  `mul_mat_vec` kernels achieve LOWER effective bandwidth per byte than Q8_0's**, so cutting bytes does not cut time.
  **Do NOT sell lever A as a speed lever — its real value is VRAM recovery.** If anyone wants to test it properly the
  honest form is a KERNEL comparison (achieved GB/s of `mul_mat_vec_q<Q8_0>` vs `<Q4_K>` on this 4090), not a byte
  argument. **HIGH risk = a quality trade — MAL'S CALL.**
- **B) Fuse the elementwise zoo (5.45 ms/token, 33%, 3288 ops/token, mean < 5 us): −1 to −3 ms, MEDIUM risk** (must
  stay bit-identical).
- C) split-K flash-decoding −0.2/−0.3 · D) window 2560->512 −0.2/−0.4 (quality) · E) MLA absorb ~0 (`attn_kv` is
  1.85% of the mmvq) · F) KV quant ~−0.05.
- Residual: a `nrows=1024` mmvq bucket (1/layer, 1.60 ms/token, 18% of the mmvq) is unmatched to a tensor.
  **RESOLVED 2026-10-13 (session 5, from the geometry, not a new trace):** it is the grouped output projection
  `attn_output_a` = `ggml_reshape_3d(wo_a, OGD=4096, OL=1024, OG=8)` (`ds41_dense.cpp:617`) - one mmvq with 1024 rows
  x 8 channels = 35.6 MB/layer, i.e. ~890 GB/s at 40 us, efficient. With it counted the dense Q8_0 mat-vecs move
  ~7.6 GB/token (= the 7.59 GB the requant log reports) at ~850 GB/s = **~84% of peak**, so the dense half is at the
  bandwidth wall - kernel tuning there is done. Still unexplained: q5_K cut those bytes 35% with no decode gain
  (unforced runs - re-measure with forced tokens + the phase timer before believing either way).

**Also:** `calls.py`/`idle_attrib.py` should derive tokens-in-window from graph launches rather than assuming N (the
window is fixed wall-clock, so per-token figures drift when the decode rate changes). Evidence:
`~/.quetza-data/conductor/reports/attn-node-profile.md`.
