# S15: the prompt's expert blob copies overlap, and where the dense 27.7 s goes

Measured 2026-10-24 on `Intel(R) Arc(TM) Pro B70 Graphics`, one GPU process at a time, `s4/f16/run.sh`
(`MIMO_CHUNK_PROF=1`, ids `(i*100)%120000+1`, temp 0, ctx 4096 at 1024 / 12288 at 8192).  Baseline binary
`s4/step2/ds41_generate_gpuexp.1fccc8e7` (the FP16 expert path, no overlap change) vs the new one; the two were run
**interleaved** because the box drifts by ±2 s between runs.

## 0. The short answer

* **The 62 GB of expert-blob H2D was never on the compute stream** and was never bandwidth-bound: it already ran on
  `s_cp`, and the file-tier source is a `cudaMallocHost` slab, i.e. USM host memory that copies at **33.0 GB/s**
  (measured, `s4/t1/bw.cpp`).  What *was* serialised with the compute is the **host-side staging memcpy of the
  pageable arena** (6.2 GB/s through the shim's stage ring) - the engine's `PinnedArena` calls `cudaHostRegister`,
  which this SYCL port turns into a **no-op** (`pinned.dp.cpp`: DPCT replaced it with `0`), so every arena blob costs
  one synchronous `memcpy` into a USM slot plus a DMA.
* Two changes: the copies got a stream of their own (`gp.s_dn`) split into small **banks** with per-bank events, and
  the next layer's whole-arena **prestage submit moved to a worker thread**.  Expert phase **18.7/19.5 s -> 13.8/14.1 s**
  (-27 %), prompt **47.05/47.92 s -> 43.81/44.46 s** (174 -> 186 tok/s).  Bit-exact (same logits md5 before/after).
* The narrow-call arm mattered more than the banking: at 8192 enc-skip cuts the prompt into **5632 + 2560 columns**, so
  layers 20-39 took the *lazy* prestage, which pays its staging memcpy **on the pass's own critical path** (3.2 s).
  With the submit off-thread the eager whole-arena prestage is free to the caller, so the width floor that selected
  the lazy fill is now 0.
* Dense is still the largest phase and grew from 27.2-27.5 s to 28.9 s: the 62 GB of staging memcpy now runs inside
  the dense window instead of the expert window.  It is one host thread doing ~5 s of `memcpy` per prompt.
* **Dense is at 3.46 TFLOPS** (95.8 TFLOP in 27.7 s) against the 60-70 TFLOPS this card measures on oneMKL f16 GEMM.
  **61 % of it is the attention core at 1.5 TFLOPS**; the GEMM-shaped remainder (9.4 s) is at 5.4-12.3 TFLOPS and is
  the low-risk lever, because the f16 oneMKL route is already proven in this tree.

## 1. Why the copies did not overlap (measured, not assumed)

`STRATA_T1_PROF=1` (this commit) bisects the expert phase's wall.  Baseline, 8192:

| bucket | ms |
|---|---:|
| pass wall (248 passes) | 13 358 |
| of it: `build` (groups/entry arrays + control-ring upload) | 1 199 |
| of it: `prestage/slab` handoff (host) | 3 645 |
| of it: `wave` (copy submission + launches) | 1 278 |
| of it: `out/tail` = the layer-end drain | 6 876 |
| outside the pass clock: prefetch prepare/join | 1 117 |
| outside the pass clock: next-layer prestage submit | 4 471 |
| drained expert kernels | 6 021 |
| copy-stream window over the passes | 7 311 |
| prestage copy window | 7 572 |

* `62.0 GB / 13.36 s = 4.6 GB/s` is a **derived** rate (`copies_gb / pass_wall`), not a copy throughput.  The copy
  engine moves those bytes at 33 GB/s; the *window* is long because it contains the host's gaps.
* The pass wall is `host work (build + handoff + wave = 6.1 s) + drain (6.9 s)`: the host is the resource that does
  not overlap.  `t1[0]` isolated the biggest host block: the **lazy prestage fill** (3 108 ms) - 19 of 40 layers ran
  it because `chunk_pstage_min()` compares the *call width* `n` with 4096, and enc-skip makes the last 20 layers
  2560 wide.
* `cudaMemcpyAsync` from the pageable arena is 0.88 ms per 11 MB blob - a **synchronous host memcpy** in the shim
  (`stage_take`), 4.5 s per prompt for the prestage alone.  From the USM slab it is a direct 11 MB `q.memcpy`.
* Micro-benchmark (`s4/t1/bw.cpp`, 3 GiB, 1.088 MB chunks): USM host -> device **32.97 GB/s**, pageable -> device
  **6.16 GB/s**, pageable staged+sync **6.69 GB/s**.  The arena is the slow one.

## 2. The change

1. **`gp.s_dn` for the streamed blob copies** (`STRATA_CHUNK_COPY_STREAM`, default on; the D3 path's copy stream,
   created here if absent).  The prestage keeps `s_cp`, so a layer's blobs never queue behind the next layer's
   prestage, and the two copy flows are independent.
2. **Banks** (`STRATA_CHUNK_COPY_BANK`, default 4 blobs): a pass's 22.5 streamed blobs leave in ~6 banks, bank k's
   copies on `s_dn`, `s` waiting on that bank's event before launching its kernels -> bank k+1's copies are in flight
   while bank k's kernels run.  Off = one bank per device half (the S7 behaviour).
3. **The prestage submit runs on a thread** (`STRATA_PRESTAGE_ASYNC`, default on).  `chunk_pstage_next` spawns it and
   it is joined at the top of `chunk_classes`, before anything else in the chunk path touches `s_cp` / `p_ptr` /
   `p_layer`.  `chunk_pstage_min()` therefore defaults to 0 (eager) when the async submit is on: the eager fill's
   whole-arena volume is hidden under the dense half, and the lazy fill's `memcpy` is not.
4. `STRATA_T1_PROF` (diagnostic, off): the bisect above.

**Why it is ordering-safe** - the same discipline as the D3 two-bank ring:

* A blob's destination is claimed in the device ring (`chunk_ring_start/own`) *before* its copy is issued, and the
  claim waits on `c_pass_free[prev pass]` - the event recorded on the **compute** stream after the kernels that last
  read those bytes were queued.  The copy stream waits for it, so a slot is never rewritten under an outstanding
  reader.
* A bank's kernel launch waits on `c_copied[bank & 1]`, recorded on the copy stream after that bank's last copy, so a
  blob is fully resident before the kernel that reads it launches.
* `c_copied[]` is re-recorded only after `s` has already captured the previous value (the shim's
  `cudaStreamWaitEvent` snapshots the `sycl::event` at call time), so two alternating events are enough.
* The prestage worker owns `s_cp` alone: the main thread never submits to `s_cp` while it is joined, and the join
  happens before `chunk_pf_prepare` and before the reader-wave path's `cudaStreamSynchronize(s_cp)`.

## 3. Before / after (interleaved, 16 decode tokens, same ids)

| 8192 | run | prompt | dense | experts | drained expert kernels |
|---|---|---:|---:|---:|---:|
| before | ab_base_8192_1 | 47.05 s (174.1 tok/s) | 27.3 | 18.7 | 6 071 ms |
| after  | ab_new_8192_1  | 43.81 s (187.0 tok/s) | 28.9 | **13.8** | 6 055 ms |
| before | ab_base_8192_2 | 47.92 s (170.9 tok/s) | 27.5 | 19.5 |  |
| after  | ab_new_8192_2  | 44.46 s (184.3 tok/s) | 28.9 | **14.1** | 6 214 ms |
| before | fin2_base8 | 48.89 s (167.6 tok/s) | 27.4 | 20.4 |  |
| after  | fin2_8192  | 44.75 s (183.1 tok/s) | 29.1 | **14.1** | 6 112 ms |

Median 47.92 -> 44.46 s (-3.5 s, -7.2 %), 172 -> 185 tok/s; the expert phase alone 19.5 -> 14.1 s (-28 %).

| 1024 | prompt | dense | experts |
|---|---:|---:|---:|
| before | 27.50 s (37.2 tok/s) | 7.7 | 19.0 |
| after | 26.37 s (38.8 tok/s) | 7.5 | 18.1 |

At 1024 the change is neutral: the call is narrow, the lazy prestage was already small, and the phase is dominated by
the 4.7 us/entry kernel + per-expert fixed cost, not by copies.  Decode is unchanged (same hit counts, 1.5-1.8 s for 16
tokens before and after).

## 4. Correctness gate

The change moves copies only, so the gate is the strictest form: **bit-identical logits**.

| prompt | before (`1fccc8e7`) | after | gate |
|---|---|---|---|
| 1024 tokens (16/16) | `223 29429 201 223 ...` | identical | pass |
| 1024 logits | md5 `62bcdbc1d098`, top-5 `{223,603,10,16,201}` | **md5 `62bcdbc1d098`**, same top-5, same values | **bit-exact** |
| 8192 tokens (16/16) | `99301 99401 ... 100801` | identical | pass |
| 8192 top-5 set | `{99301, 7944, 79749, 223, 201}` | same set, same order | pass |
| 1024, old kernel (`STRATA_CHUNK_F16=0`) | md5 `787519ba000d` | md5 `787519ba000d` | unchanged |

At 1024 the change is bit-exact: the raw FP32 last-token logits are identical to six decimals (`6.649817 5.603718
5.472691 5.230228 5.132542`), so the md5 `62bcdbc1d098` of the new path and `787519ba000d` of the old kernel both
still hold.  **At 8192 the last row is not run-reproducible** (the S9 note): one interleaved before/after pair matched
md5 `2f46fcf51957` exactly, and a later run of the *same* after-binary gave `f621728a` with the argmax logit at
28.18 instead of 27.51 - while the decoded tokens and the top-5 set stayed identical.  So the 8192 gate is tokens +
top-5 (as specified), and the md5 is only compared within an interleaved pair.

## 5. Dense, part by part (`DS41_PART_PROF=1`)

The dense half is one ggml graph per layer, so `DS41_PART_PROF` submits the *same* graph in consecutive node ranges
(bounded by tensors the builders record) and times each range.  A committed graph's node order is a topological order,
so a cut at tensor T separates T's ancestors from the rest: every node still runs after its inputs and before its
consumers, in the same order, on the same in-order queue.  Cost: one submit+sync per part (a probe, not a fast path)
and the loss of an op fusion that would straddle a cut; the parts sum to 26.95 s against the 27.7 s dense wall.

8192 (geometry D 5120, DH 512, NH 64, RQ 1280, SWA 128, RAW 144, OGD 4096, OG 8, OL 1024, n_ff 2304, NE 384, NU 6,
indexer 32/128/512):

| part | ms | share | MFLOP/token-layer | TFLOP | achieved TFLOPS |
|---|---:|---:|---:|---:|---:|
| attention core (window + compressed + indexer + FA) | **16 498** | **61.2 %** | 117.4 | 25.25 | **1.53** |
| q_a/q_b projection + rope | 3 851 | 14.3 % | 97.0 | 20.86 | 5.42 |
| attn output LoRA + hc_post | 2 854 | 10.6 % | 151.0 | 32.47 | 11.38 |
| kv projection + rope | 1 339 | 5.0 % | 5.2 | 1.13 | 0.84 |
| shared expert | 1 241 | 4.6 % | 70.8 | 15.22 | 12.27 |
| init (embedding + pre_state) | 684 | 2.5 % | - | - | - |
| hc_pre + attn_norm | 528 | 2.0 % | - | - | - |
| finish (routed add + hc_post + state cpy) | 316 | 1.2 % | - | - | - |
| router | 120 | 0.4 % | 3.9 | 0.85 | 7.05 |
| writes (cpys) | 31 | 0.1 % | - | - | - |
| LM head + logits | 18 | 0.1 % | 0.04 | - | - |
| **total** | **26 952** | 100 % | 446.9 | 95.78 | **3.46** |

* **RMSNorm/rope** is 0.53 s as its own bucket and ≈ 0.3 s inside the projection buckets (elementwise over ≤ 32 768
  values/token): ~3 % of dense together.  Not a target.
* **The LM head is not a phase here**: only the last row is computed at prompt time (`logits 0.0 s` in the split), so
  head + logits = 18 ms.  It is 0.1 %, not a candidate.
* **The router is 0.4 %** (120 ms): sqrt-softplus + argsort over 384 experts × 8192 rows.  Not a target.
* Attention's *share* grows 51.9 -> 57.8 -> 60.0 -> 61.2 % from 1024 to 8192 while its per-token-layer cost falls
  103.0 -> 87.8 -> 82.6 -> 76.7 us: the window is fixed (SWA 128 + up to 512 selected compressed rows), the
  indexer/compressor scan grows with the visible length, and everything else amortises its weights over more rows.
  So attention is the slowest-amortising part, not yet superlinear in B at these sizes.

## 6. What to attack, with the arithmetic

Every part is quantized-weight (q8_0) ggml matmul except the attention core:

| part | ms now | TFLOP | TFLOPS | at 60 TFLOPS | saves |
|---|---:|---:|---:|---:|---:|
| q_a/q_b | 3 851 | 20.86 | 5.42 | 348 ms | 3.50 s |
| attn out LoRA + hc_post | 2 854 | 32.47 | 11.38 | 541 ms | 2.31 s |
| kv | 1 339 | 1.13 | 0.84 | 19 ms | 1.32 s |
| shared expert | 1 241 | 15.22 | 12.27 | 254 ms | 0.99 s |
| router | 120 | 0.85 | 7.05 | 14 ms | 0.11 s |
| **GEMM-shaped total** | **9 405** | **70.52** | **7.50** | **1 176 ms** | **8.23 s** |
| attention core | 16 498 | 25.25 | 1.53 | 421 ms* | 16.1 s |

\* the 60 TFLOPS figure is a *reference ceiling*, not what a flash-attention kernel can be expected to do; it is there
to size the gap (the core is 40x off, not 6x).

**Ranking.**  (1) The **attention core** is the biggest part (61 %) and the least efficient (1.53 TFLOPS), but its
content is not oneMKL-shaped: it is the per-query K gather + `cast` to f16, the FA over 128 + 512 keys of width 512,
and the indexer's block scores/top-k over every visible row.  The one lever that does not need a new kernel is the
**indexer/compressed selection** (fewer candidate blocks, a shorter top-k), which trades routing quality for time - a
product decision, and it should be measured on the decoded tokens first.  (2) The **GEMM-shaped remainder (9.4 s, 34 %)**
is where the tree already knows the answer: the expert path's dequantize + oneMKL FP16 GEMM measures **60-70 TFLOPS**
on the same card, so these five matmuls would take **~1.2 s instead of 9.4 s**, saving **8.2 s** of the prompt's 43 s
(186 -> ~239 tok/s) with a *bit-comparable* gate rather than a bit-exact one.  That is the next change worth making.

## 7. Bottleneck order after this work (8192 prompt, 43.8 s)

1. **Dense GEMM efficiency** - 9.4 s of matmul at 7.5 TFLOPS (8.2 s recoverable at oneMKL-class rates).
2. **The attention core** - 16.5 s at 1.53 TFLOPS; the largest single part, needs an FA/indexer answer, not a GEMM one.
3. **The host's staging memcpy of the pageable arena** - ~5 s/prompt on one host thread (62 GB at 6.2-12 GB/s), now
   inside the dense window.  The fix is to make the arena device-reachable (`PinnedArena`'s `cudaHostRegister` is a
   no-op in this port): direct copies at 33 GB/s would remove both the memcpy and 2/3 of the copy-engine time, and
   would also shorten the expert phase's remaining 13.8 - 6.06 = 7.7 s.
4. **The expert kernels themselves** - 6.06 s = 4.69 us/entry (dequant 40-50 %, the two GEMMs, SwiGLU, scatter); only
   a fused quantized GEMM or larger per-launch batches move this.
5. Everything else (prefetch 1.0 s, class list, top-k sum 75 ms).
