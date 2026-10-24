# Why is GSQ v2 slower than v1? — the profile plan (written 2026-10-18 ~08:40, session 7, CPU-only prep)

v2 (139.8 GiB, more ternary; then on exFAT NVME1TB, since 2026-10-18 11:05 the served quant on SSD NVME) measured slower than v1 in
session 6 (`verify/s6/v1v2-speed.log`): prefill 170-176 / 277-286 vs 186 / 314-321 tok/s, decode after 6.5K
18.1-19.2 vs 21.7-23.1 tok/s (= ~53.5 vs ~44.5 ms/token, **+9 ms/token**).  Nothing below has been run on the GPU yet.

## 0. RESULT (2026-10-18 09:10) - it is the exFAT drive, not ternary

**RESULT 2026-10-18 10:39 (`verify/s7/swap-v2-test.log`, `verify/s7/v2s-speed.log`) - v2 on SSD NVME (same drive as v1),
2 rounds: prefill 184.5-187.6 / 292-296 (v1 177-178 / 280-281, +5 %), decode after 6K 21.4-23.1 (v1 22.6-24.1, ~=),
after 16.5K 23.4-23.8 (v1 20.3-22.5, +10 %), edits 42.3-45.0 (v1 37.6-39.8, +12 %), small turn 7.7-8.9 s (v1 8.4-10.4).
=> v2 >= v1 on the same drive; the s6 "slower" was exFAT only.  Transcode (MIMO_CHUNK_PROF, build-s6): 52 ms GPU per
6K prompt, 137 ms per 16K (~0.2 % of prompt time, 400-700 GB/s) - NOT worth optimising; STRATA_MMQ_PTQ1=0 (ternary on
MMVQ) is clearly worse (16K 248 tok/s); STRATA_MMQ_TC_GIB=3: 16K 306 tok/s, one run (maybe +3-5 %, confirm x2 before
adopting - it takes VRAM from the prestage region).  v1 restored on SSD NVME (sha256 verified).**


- **Speed re-test with v2's OWN rank** (`verify/s6/v1v2r-speed.log`, 2 rounds, both dense Q6_K): v2 still slower -
  prefill 164 vs 177-178 (6.1K), 241-251 vs 280-281 (16.5K); decode after 6K 19.0-19.5 vs 22.6-24.1, after 16.5K
  17.3-19.9 vs 20.3-22.5 - BUT the **edit probe (short context, experts resident) v2 37.4-41.7 vs v1 37.6-39.8 =
  same or faster**.  v2 loses only where experts come off disk.
- **Disk microbench** (O_DIRECT, 600 random whole experts, the engine's way; scratch `blobread.py`):
  v1 file (ntfs3, SSD NVME) **5.1-5.3 GB/s** at 32 threads; v2 file (exFAT, NVME1TB) **0.96 GB/s** at 32 threads =
  0.74 at 1 thread (no scaling = serialised).  Raw `/dev/nvme1n1p2` (same drive, no filesystem): **3.31 GB/s** at 32
  threads.  Buffered reads on exFAT: 0.58 GB/s (worse).  FIEMAP is not supported on exFAT (filefrag falls back to
  FIBMAP = root), so the Engram raw-device trick does not carry over.
- **Conclusion: the exFAT driver costs ~5x on file-tier reads; ternary compute is not what made v2 slower.**  The
  ternary kernels are still worth tightening (sections 1-3 below: 136 GB/s decode dot, prompt transcode), but they
  are a second-order gain.  A fair v2 speed number needs v2 on ntfs3/ext4 NVMe (no room today: SSD NVME has 7 GB).
- **For the 1017 GGUF: never serve it from NVME1TB while it is exFAT.**  Build/validate it there if needed (writes and
  quality runs don't care), then serve from SSD NVME (it replaces v2 there once validated) - or reformat NVME1TB to
  ext4 (884 GB on it now: Models 753 GB + swap64.img 64 GB) when Mal chooses.

## 1. What the code reading + CPU checks found (this morning)

| Candidate | Status | Evidence |
| --- | --- | --- |
| (a) stale `rank.bin` (v1's order served with v2) | **small - ruled out as the main cause** | routed-token coverage with v2's sizes: VRAM 14.6 GiB 39.3 % (stale) vs 39.8 % (own order); VRAM + 64 GiB arena 86.7 vs 87.2 %. v1: 40.1 / 85.0 %. v2 is NOT worse cached than v1. |
| (b) **different drive + filesystem** | **NEW, unmeasured, likely big** | v1 = `/media/mal/SSD NVME` (**ntfs3**), v2 = `/media/mal/NVME1TB` (**exFAT**, 970 EVO Plus). Expert file-tier misses are O_DIRECT preads through the filesystem (`ds41_moe.cpp` ~217, ~1077) and cost ~6 ms/token on v1's critical path (V41_START_HERE §3). exFAT measured ~13K vs ~290K random reads/s for Engram rows (V41_RESUME_PROMPT ~1228). Prefill sweeps read the whole file tier every batch. |
| (c) ternary decode kernel | **bounded, can't explain +9 ms alone** | v1 nsys (2026-10-12): `native_gu<143>` 2.72 ms + `down<143>` 1.31 ms of a 37.5 ms token; the kernel runs at ~136 GB/s (ALU-bound trit decode). The quarter-block dot (`-DSTRATA_PTQ1_QUARTER`) and `STRATA_EXP_MODE_T143` are already measured dead ends (~1 %, V41_START_HERE §4). v2's extra ternary share adds maybe 1-2 ms, not 9. |
| (d) prompt-path transcode (PTQ1_0 -> Q2_0 before MMQ) | confirmed to exist, cost unmeasured | `ds41_moe.cpp` ~3973-3979, `ds41_transcode.cu` (one thread per output byte, scalar trit loop). Copies capped at `STRATA_MMQ_TC_GIB` = 1 GiB = ~80 experts per MMQ launch: a layer with more transcoded experts than that is split into several smaller launches. |
| (e) arena fragmentation | hypothesis | `arena_victim` (~2099) takes a same-layer victim whose slot is >= the incoming size: a ternary expert can land in a Q3_K-sized slot (wasted bytes), a Q3_K one cannot use a ternary slot. v2's per-layer size mix is wider -> the arena's useful capacity may shrink as it adapts. |

## 2. The profile, in order (GPU free + Mal's go; quiet box: no builds/torch/downloads on the same drives)

0. **Disk first (~10 min, no GPU):** O_DIRECT random reads of real expert blobs from each GGUF at the engine's
   concurrency (blob sizes from `*.gguf.experts.json`; 32 threads, ~2,000 file-tier-class experts), MB/s + reads/s per
   file.  Reading the v1 file disturbs any engine serving it - run when nothing is serving.
1. **Separate the drive from the format** with a **file-tier-free A/B**: both models, same `--arena-gib`, a prompt
   short enough that decode never misses to the file (check file misses = 0 in `STRATA_DEC_PROF`), so the drive
   drops out.  If v2 ~= v1 there, (b) is the cause.  (Moving v2 to ntfs3 needs 150 GB: Crucial1TB is ntfs3 but a
   SATA MX500, slower than both NVMe, and is earmarked for the 1017 shards; NVME1TB has 30 GB free, so v1 can't go
   onto exFAT either.)
2. **The corrected speed test** `verify/s6/v1v2r-speed.sh` (v2 with its own rank: `strata-ds41-v2r.json`, ready),
   ~25 min, same probes as session 6, now both at dense Q6_K.  Add `STRATA_DEC_PROF=1` so each arm logs the decode
   phase split (attn_router / tier / finish; tier = hits, PCIe misses, file misses, CPU pool).
3. **Decode kernel split** (nsys, decode-only window, `--cuda-graph-trace=node` ratios only): per-type expert kernel
   time per token, v1 vs v2.  Expect `native_gu<143>` + `down<143>` up by v2's extra ternary share.
4. **Prompt split** (`MIMO_CHUNK_PROF=1`, fresh 8K + 20K): per batch transcode kernel time, number of MMQ launches per
   layer (the 80-expert tc cap), reader time / bytes, PCIe DMA.  Then try `STRATA_MMQ_TC_GIB=2/3` and
   `STRATA_MMQ_PTQ1=0` (ternary on MMVQ) as one-flag checks.
5. **Arena occupancy:** log bytes in use vs bytes held (sum of slot sizes vs sum of resident experts' sizes) at
   seed and after the 20K prompt + 256 decode, v1 vs v2.

## 3. What each answer leads to

- (b) dominates -> nothing wrong with ternary; serve/benchmark every quant from ntfs3/ext4 (never exFAT), re-measure
  v2, and the 1017 tier-map decision (V41_1017_QUANT_PLAN §3) can lean v2-style again.
- (d) dominates prefill -> native PTQ1_0 MMQ tile loader (expand trits to int8 in shared memory - the Prism-fork
  idea), or at least a coalesced transcode + a bigger tc cap.
- (c) is real but small -> the exact-isum q8_1 trick (drop the `sumu` dp4a, half the ternary dp4a work; safe: our
  Q2_0/Q3_K dots don't read ds.y) is the next kernel lever; the quarter split is done.
- (e) -> size-class arena regions per layer, or seed packing that keeps big slots for big experts.
