# S4 step 6 — expert fetch: what the counters actually count, and where the prefill time really goes

2026-10-24. Follows [step 4](xolotl-sycl-s4-step4-pcie-dma-fix-report.md). Question asked: implement batch-level
expert-fetch dedupe in the PCIe/file path — the premise being that `491520 pcie` for a 2048-token prompt
(= 240 per token = 40 layers × 6 experts) proves the fetch copies per (token, expert) instead of per distinct expert.

**Answer: the premise is false, and the counter says so.** `st.pcie` is an **entry** counter, not a copy counter.
The fetch already streams each distinct expert once per layer (and the kernel already receives the group/entry
structure that expresses it). Measured copies per token at 2048 tokens: **2.86**, and 84 entries per copy. What the
step did find is a genuine fetch-path defect next to it — and it is not a dedupe problem: every small H2D copy in the
chunk path came from **pageable** memory, and the SYCL shim makes a pageable H2D copy **drain the compute stream**, so
the expert pipeline had no overlap at all. Fixing that plus measuring the rest is what follows.

## 1. What `ue` / `u` / `si` index, and why the counter is exactly tokens×6

`gpu_run_n` (`ds41_moe.cpp:3755-3810`) is the multi-token path:

```cpp
for (int j = 0; j < NE; ++j) { ... if (u < 0) { u = nu; ue[nu++] = ids[j]; } ent_u[j] = u; }   // distinct experts
for (int u = 0; u < nu; ++u) { ... }                                                           // classify per expert
for (int u = 0; u < nu; ++u) { if (cls[u] != PCIE) continue;
    ck(cudaMemcpyAsync(gp.d_stage + si*B, gp.arena.ptr(layer, ue[u]), BL, H2D, gp.s), "pcie dma");   // ONE per expert
    dptr[u] = gp.d_stage + si*B; ++si; }
ptr.push_back(dptr[u]); start.push_back(dst.size()); for (j) if (ent_u[j]==u) { dst.push_back(j); ... }
```

* `ue[u]` — the u-th **distinct expert** of this call (first-seen order), `nu <= NE`.
* `u` — an index into that distinct-expert list; `dptr[u]` is **one** device pointer serving all its entries.
* `si` — the `d_stage` slot for the u-th **staged expert**, so `si` advances once per expert, not per entry.
* `ent_u[j]` — the entry→distinct-expert map; `dst/tok` are the entry lists the kernel walks per group.

So the copy loop is already "fetch once, use for many tokens". The kernel is called with `ng = nu` groups and the
entries' token list — exactly the shape the task describes as the target. `gpu_run_chunk` (the batch path,
`ds41_moe.cpp:4743`) does the same thing with `cnt[]/order[]/first[]` per layer: one `stream_e` entry, one H2D
`memcpyAsync` each, then one grouped launch per adjacent run.

**Why 61440 / 491520.** The prompt summary prints `st.pcie`, and in every multi-token path that is the number of
**entries served by a streamed expert**, not the number of transfers:

| site | code | counted unit |
|---|---|---|
| `gpu_run_chunk` last/non-last pass | `st.pcie = ent_str;` (`ent_str += cnt[e]` per streamed expert) | entries |
| `gpu_run_n_ex` | `for (c) if (c == PCIE \|\| c == FILEG) ++st.pcie;` per entry | entries |
| `gpu_run` (single token) | `st.pcie = np + nfg;` per distinct miss | experts (= entries, because nt=1) |

`61440 = 256 × 240 = 256 × 40 × 6` and `491520 = 2048 × 240` are the top-k applications the MoE must perform —
invariant to any fetch improvement. `st.file_tier`, by contrast, is a blob count (`n_file = nF`). New diagnostic
counters (`MIMO_CHUNK_PROF=1`, `chunk prof S6:`) now report the real transfer count:

```
2048 tokens: 248 passes, 491520 entries, 5849 blob copies = 64.2 GB -> 84.0 entries/copy, 2.86 copies/token
 256 tokens: 233 passes,  61440 entries, 3473 blob copies = 38.4 GB -> 17.7 entries/copy, 13.6 copies/token
```

(The remaining per-layer arena experts are DMA'd once per layer by the **lazy prestage** into `p_buf`, not by the wave
loop: 40 layers × ~230 experts ≈ 9200, counted separately. Total ≈ 15000 transfers for ≈15000 distinct
`(layer, expert)` pairs — 1.0 per pair, i.e. already minimal.)

**The staging ceiling.** `d_stage` holds `max(kMaxParts=8, STRATA_D3_STAGE_SLOTS)` blobs of `B = im.blob` bytes — a
decode-call budget; the batch path does not use it. `gpu_run_chunk` streams into two device halves of
`c_half` (default `MIMO_CHUNK_STAGE_MIB=1024`, shrunk to the free VRAM): `per_half = c_half / SL ≈ 95` blobs at the
10.7 MB mean blob, so at most `ceil(384/95) = 5` waves for a layer's whole expert set — and at 2048 tokens a class
pass carries only 23.6 distinct blobs, i.e. one wave. Waves are therefore not a constraint here.

## 2. The real fetch-path defect: pageable H2D copies drain the stream

`cudaMemcpyAsync` whose source is pageable goes through the shim's `copy_in`, which stages through a single host buffer
and **waits on the copy's event**, and that event is ordered *behind every kernel already queued on the same stream*
(`sycl_compat/ds41_sycl_compat.cpp:385-397`). A pass issued ~8 such copies (`c_dst`, `c_tok`, `c_x`, `c_ptr`,
`c_start`, `c_ng`, `c_w`) from `std::vector::data()` / the caller's buffers, so **the expert pipeline had no overlap
at all** — the pass wall was the plain sum of readers + kernels + copies:

```
before, 2048 tokens: pass wall 36181 ms = build 8533 + prestage 4642 + wave 21790 (reader waits 11065, kernels 14908) + out 1035
```

`build` (group/entry construction, 8.5 s = 17 % of the prompt) was almost entirely stream drain, not work.

**Fix** (`Gpu::kCtlRing`, `ds41_moe.cpp`): the chunk path's small H2D sources are copied into **pinned** ring buffers
(`cudaMallocHost`, 16 slots × {group pointers/starts/n_groups, `c_dst`, `c_tokv`, `c_w}`; 2 slots for the ≤64 MiB
activation block) and the async copy comes from pinned memory — a real async USM copy, no `ev.wait()`. Rings because a
slot may still be in flight when a later pass fills its own (the host runs ≤2 passes ahead). Any allocation failure
falls back to the old pageable call: slower, never wrong. Pinned memory: ≤ ~3 MB for the control rings + 117 MB for
two 2048-token activation slots (a wider prompt keeps the pageable upload rather than pinning hundreds of MB).

## 3. Measurements

Same command shape for every row — `-m V41-GSQ-1017.gguf --backend cuda --experts gpu --engram … --rank rank.bin
--ids (i*100)%120000+1, -n 1 --ctx 4096 --temp 0`, no `--prompt-batch` / `--prefill-chunk` / `--slots` (so
PB=min(8192,len), S=512, auto slot budget), one process at a time. `MIMO_CHUNK_PROF=1` on both sides of a pair (timing
only). 2048-token rows are one run each; 256-token rows differ from each other by ~1 s run to run.

| run | prompt | time | prefill | dense | experts | file wait | `pcie` (entries) | file blobs | blob copies | copies/token | entries/copy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before 256 | 256 | 18.73 s | **13.7 tok/s** | 4.2 | 13.8 | 6.0 | 61440 | 3329 | 3473 | 13.6 | 17.7 |
| **after 256** | 256 | **17.68 s** | **14.5 tok/s** | 4.1 | 12.8 | 6.1 | 61440 | 3329 | 3473 | 13.6 | 17.7 |
| before 2048 | 2048 | 49.25 s | **41.6 tok/s** | 11.5 | 36.2 | 11.1 | 491520 | 5671 | 5849 | 2.86 | 84.0 |
| **after 2048** | 2048 | **46.26 s** | **44.3 tok/s** | 11.8 | 33.7 | 10.9 | 491520 | 5671 | 5849 | 2.86 | 84.0 |
| after + whole-layer prefetch (`STRATA_CHUNK_PF_MIN=512`) | 2048 | **40.43 s** | **50.7 tok/s** | 11.3 | 28.3 | 0.0 | 491520 | 979 | 1157 | 0.56 | 425 |

Repeats (no profiler): after 2048 = 45.71 s / 44.8 tok/s; prefetch 2048 = 40.52 s / 50.5 tok/s — so the prefetch
delta (+13-14 %) and the fix's own delta (+7 % at 2048) are both outside run-to-run noise.

The same pair without the profiler: 256 tokens 19.57 s (13.1) → 19.05 s (13.4).

`build` fell from 8.5 s to 0.15 s (256) / 0.37 s (2048) — 23-58× — which is the proof the copies were draining the
stream; the sum moved less than the phase did because the stall reappeared on the copy stream, which still waits on
the compute stream's kernels through `c_used[]` (one pass of pipeline depth, see §5). Copies per token were **2.86
before the change**: the dedupe the step was asked to add was already there.

## 4. Correctness

Not token ids — bitwise logits. The pre-change binary (HEAD) and the changed one are built from the *same* tree and
run with the identical 256-token prompt and `-n 8 --dump-last`:

```
before.logits md5 fce1788f4aa01cc2b411089eda2f046f
after.logits  md5 fce1788f4aa01cc2b411089eda2f046f     n=129280 floats, max|d| = 0.000e+00, 0 of 129280 differ
argmax 26 in both (top-1 margin 1.014); decode counters identical: 1654 hit / 3 cpu / 23 pcie / 15 file
```

That is inside the CPU-vs-CPU perturbation envelope by a wide margin: it *is* the envelope's zero. A decode of 8
tokens stays stable (8.39 vs 8.64 tok/s, same ids, same hit/miss counters), and 3- and 8-token prompts keep their
ids. The change moves transfer mechanics only; nothing about routing, order or arithmetic changed.

## 5. The next bottleneck, with numbers

After the fix, at 2048 tokens (46.3 s = dense 11.8 + experts 33.7):

```
pass wall 33708 ms = build 370 + prestage/slab 4655 + wave 26439 + out/tail 2057
  of the wave: reader waits 10869, GPU kernels 3743 (7.6 us/entry = ~11.6 TFLOP/s effective)
```

1. **File-tier reads: 61.9 GB at 5.1 GB/s = 12.1 s of reader wall, 10.9 s of it exposed as wait.** The biggest single
   item (23 % of the prompt). The whole-layer prefetch slab is disabled by design
   (`STRATA_CHUNK_PF_MIN` default never — it was measured a loss at ~272-row turns where it reads 100 % of a layer's
   file tier for ~39 % of it). At 2048 rows the routing covers essentially all of it, and the row above shows
   **50.7 tok/s (+14.5 %)** with `STRATA_CHUNK_PF_MIN=512`: file wait 10.9 → 0.0 s. Making the gate width-aware (instead
   of one global constant) is a small, contained change.
2. **One pass of pipeline depth** (`final sync 24356 ms` in the prefetch row, `wave 26.4 s` before it): the copy stream
   waits on the compute stream via `c_used[]` for a whole device half, and the host still syncs `s_cp` once per pass
   ("bounce free") although the pinned bounce holds ~8 passes of blobs at 23.6 blobs/pass. Deeper halves or an
   event-disciplined bounce ring is the structural fix; the 24 s "final sync" says the work is now overlapped only at
   the ends.
3. **Dense 11.8 s (26 %)** — unchanged by this step; see the step-3 report (three cold graphs, not per-token cost).
4. **Expert kernels 3.7 s**: 7.6 µs/entry. Not a bottleneck any more; the earlier 14.9 s figure was stream-drain
   attributed to the kernels. MMQ (`chunk_mmq`) is still inactive here (`0 sub-runs`, `mmq::built()` false in this
   SYCL build), so the kernels run the MMVQ/grouped path — worth ~2-3 s if ever wired up on SYCL.

**Dedupe itself has nothing left to give** unless the sweep gets wider: copies/token is `40 × distinct(layer) / B`,
i.e. 2.86 at B=2048 vs the 6-9K-token sweep the README's 178-268 tok/s reference implies (0.9 at B=6144). Beyond
that, the lever is the *tiering* (8977 of 15360 experts are file-tier) — out of scope for this step by instruction.

## 6. Reproduce

```
# before / after binaries (same tree, one file apart)
/home/dyy/deepseek-work/s4/step6/run.sh <tag> <n-prompt-tokens> <n-decode> [extra args...]
#   e.g. MIMO_CHUNK_PROF=1 ./run.sh base 2048 1
#        BIN=/home/dyy/deepseek-work/s4/step6/ds41_before ./run.sh base-before 2048 1
# logits A/B
./run.sh chk 256 8 --dump-last /tmp/after.logits
# build (never while a model runs)
/home/dyy/deepseek-work/s4/step2/build_engine_tu.sh ds41_moe && /home/dyy/deepseek-work/s4/step2/link_gpuexp.sh
```

Logs: `s4/step6/{prof-n256,prof-n2048,pin-n256,pin-n2048,pf-n2048,chk-before,chk-after}.log`.
