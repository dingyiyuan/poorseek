# S16: the arena is device-reachable, and the dense GEMM-shaped parts run through oneMKL

Measured 2026-10-24 on `Intel(R) Arc(TM) Pro B70 Graphics`, one GPU process at a time, `s4/arena/run.sh`
(`MIMO_CHUNK_PROF=1`, ids `(i*100)%120000+1`, temp 0, ctx 4096 at 1024 / 12288 at 8192), before/after arms
**interleaved** because the box drifts by ±2 s between runs.  Two changes, each measured alone: CHANGE 1 = the
arena's backing (`022da386` -> `4c995892`), CHANGE 2 = the dense GEMM route (the same binary with
`STRATA_DENSE_GEMM=0` as its own before-arm).

## 0. The short answer

| 8192 | before | after C1 | after C1+C2 |
|---|---:|---:|---:|
| prompt | 43.18 / 44.32 s (189.7 / 184.8 tok/s) | 41.34 / 41.98 (198.2 / 195.2) | **38.56 / 38.16 (212.4 / 214.7)** |
| dense | 28.7 s | 29.6 s | **26.4 s** |
| experts | 13.0 / 14.0 s | 10.7 / 11.0 s | 11.0 / 11.1 s |

* **CHANGE 1**: the 60 GiB arena is now USM host in three pieces, cut at blob boundaries; every arena copy is one
  direct async DMA.  The shim's staging memcpy is gone - `DS41_SHIM_DBG=1` counts **3544 -> 0** staged H2D copies -
  and the expert phase drops 2.3-3.0 s.  It is **-2.1 s, not the -5 s that was expected**: S15 had already moved
  the staging memcpy onto its own thread, so it was hidden behind the dense half rather than inside it.  What was
  left on the critical path was the copy *window* (62 GB at 6.2 GB/s vs at 33 GB/s), which is the 2.3 s.
* **CHANGE 2**: q_a/q_b, kv, the attention-output LoRA, the shared expert (and the router, available but off by
  default) run as dequantize + oneMKL FP16 GEMM, exactly the expert path's route.  Dense 29.6 -> 26.4 s, prompt
  41.9 -> 38.4 s (**-3.5 s, 196 -> 214 tok/s**), bit-comparable: same decoded tokens, same top-5 set at both arms.
  The oneMKL GEMMs themselves measure **82.9 TFLOPS** (70.5 TFLOP in 0.85 s).  It is **-3.5 s, not the -8.2 s** the
  FLOP model promised: those five parts are 9.6 s in total but only **~5.0 s of them is matmul** (13.8 TFLOPS, not
  7.5) - the rest is rope/rms_norm/hc/submission, which this change does not touch.  3.5 s is that route's ceiling.
* **Decode is unchanged**: 16 tokens 1.32-1.48 s before and after (the route refuses M < 256, so decode never
  enters it; 99.1-99.2 % VRAM hit rate both arms).
* **The attention core is now the bottleneck**: 16.5 s, 62 % of the dense half and 43 % of the whole prompt.

## 1. CHANGE 1: why the copies now go direct

`PinnedArena` called `cudaHostRegister`, which this SYCL port replaced with a no-op, so the 60 GiB arena was
pageable host memory that no device could address.  Every arena blob therefore cost one **synchronous host memcpy
into a USM stage slot** plus the DMA (`copy_in`, 6.2 GB/s) instead of one direct DMA (33 GB/s).

SYCL cannot pin an existing range, but a caller that is *about to allocate* can ask for the memory as USM host.
The arena is the one place in this engine that can (it allocates, then fills):

1. `ds41_sycl_compat.cpp` gains `ds41_sycl_host_alloc_max/alloc/free` - USM host from the context the copies run
   on (`ggml`'s, adopted at startup), so `sycl::get_pointer_type` answers `host` and `do_copy()` takes its
   direct async path.  `pinned.dp.cpp` declares them **weak**: a link without the shim (CUDA/CPU, the parity
   harnesses) keeps the mmap path unchanged.
2. A single USM host request is bounded by the device's `max_mem_alloc_size` - **30.3 GiB**, and the driver
   refuses more: a direct `zeMemAllocHost(60 GiB)` probe returned `OUT_OF_HOST_MEMORY`, `sycl::malloc_host(31 GiB)`
   returns null, 30 GiB succeeds.  So `PinnedArena` allocates **pieces**, cut at the caller's blob boundaries
   (the engine now always hands over every blob start), and resolves an arena offset through `at()`: a blob is
   never allowed to straddle two pieces, because that would be a wrong address rather than a slow copy.
3. The engine's ~15 `arena.base + off[]` sites go through `arena.at(off)` / `at_mut(off)`; `ptr(layer, e)` and
   the whole read path already went through an accessor.
4. **The fallback is explicit**: if any piece is refused, the pieces allocated so far are released, one line says
   what was refused and how big the request was, the arena falls back to the mmap and its note says
   `USM host REFUSED (...): pageable, copies stage`.  A silent fallback on a 60 GiB arena would be invisible.
5. The lifetime rule is unchanged: the arena must outlive every copy from it.  It is freed at shutdown, after the
   streams are drained, and `sycl::free` releases each piece.

**Evidence that the staging is gone** (`DS41_SHIM_DBG=1`, 1024, one token):

| | 022da386 | USM host |
|---|---:|---:|
| `staged H2D` copies in the log | **3 544** | **0** |
| arena line | `cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable; THP requested` | `USM host: 3 device-reachable piece(s) (max single allocation 29.80 GiB, arena 59.99 GiB)` |
| prestage/slab phase of the 8192 pass wall | 629 ms | 557 ms |
| load (60 GiB allocated + first touch) | 5.2-5.8 s | 9.5-19.8 s (allocation + pinning; one-off, before the prompt) |

## 2. CHANGE 1 before/after

Interleaved, `s4/arena/c1_*`:

| 8192 | prompt | dense | experts | drained expert kernels | prestage DMAs |
|---|---:|---:|---:|---:|---:|
| before (022da386) run1 | 43.18 s (189.7) | 28.7 | 13.0 | 6 067 ms | 6 194 blobs / 62.0 GB |
| after  (USM host) run1 | 41.34 s (198.2) | 29.6 | **10.7** | 6 051 ms | 6 383 blobs / 64.4 GB |
| before run2 | 44.32 s (184.8) | 28.7 | 14.0 | 6 076 ms | 6 194 / 62.0 |
| after run2 | 41.98 s (195.2) | 29.9 | **11.0** | 6 040 ms | 6 383 / 64.4 |

| 1024 | prompt | dense | experts | drained kernels |
|---|---:|---:|---:|---:|
| before | 26.45 / 26.02 s (38.7 / 39.4) | 7.5 | 18.2 / 17.6 | 4 770 / 4 759 ms |
| after | 25.08 / 24.24 s (40.8 / 42.2) | 7.9 / 7.5 | **16.3 / 15.9** | 4 852 / 4 863 ms |

Dense grows by ~0.9 s at 8192 and the copy engine now moves 64.4 GB in ~2 s instead of 62.0 GB in ~10 s: the same
DMA that used to run *after* the dense window now runs *inside* it and competes with the dense half for memory
bandwidth.  Net -2.1 s.  At 1024 the prompt is bounded by the 100 GB file-tier read (19.5 s at 5.4 GB/s), so the
win is smaller there (-1.6 s).

## 3. CHANGE 2: the dense GEMM-shaped parts through dequant + oneMKL FP16 GEMM

`strata::kernels::dense_gemm_f16` (new: `sycl/src/kernels/dense_f16_gemm.dp.cpp`) is the expert path's recipe for
one matrix: dequantize the weight into an FP16 device buffer, cast the FP32 activations into an FP16 tile, one
oneMKL FP16 GEMM with FP32 accumulation, write the FP32 result into the ggml tensor's buffer.  ggml's `[K, N]`
weight with K contiguous is exactly oneMKL's column-major `K x N` with `lda = K`, so the call is
`(trans, nontrans, m=N, n=M, k=K)` and the result lands in the ggml `[N, M]` dst with no transposition pass.

**How it gets into the graph without rebuilding it.**  The nine nodes are marked as they are built
(`densegemm::note` right after the `ggml_mul_mat`), and the single submission point (`partprof::run`, extended)
cuts the graph just before each marked node, skips it and runs the oneMKL GEMM into the node's own buffer.  A
committed graph's node order is a topological order, so the cut changes neither the arithmetic nor the buffer
lifetimes: the allocator still plans the node, so its dst and its sources live exactly as long as before.  The
production path submits the ranges **async** on the same in-order queue (one `ggml_backend_synchronize` at the
end), so a cut costs a submission, not a submit + drain; the probe keeps its per-range wait for the timing.

Two things that cost more than they look, and the fix:

* **The dequantized weight is cached per tensor.**  The dense half is *layer-major*: one layer's graph is submitted
  once per sub-batch of columns (~5x at a 8192 prompt), so the same weight was dequantized ~5 times.  Measured
  2.1 s/prompt of dequantization for 0.83 s of GEMM.  An LRU of dequantized weights (384 MiB, keyed by tensor +
  shape) took the dequantized volume from 72.3 GB to 6.9 GB and the time to 0.44 s.
* **Decode never enters**: the route refuses `M < 256` (a GEMM launch plus a per-call dequant is a regression at
  M = 1), so the decode path is bit-for-bit the old one.
* **The router is not in the default set.**  Its weight is F32, so it takes an all-FP32 GEMM, and oneMKL's sgemm at
  N = 384 measured *slower* than ggml's own op (161 vs 131 ms).  `STRATA_DENSE_GEMM_PARTS=...r` enables it.

Per-stage device time at 8192 (`STRATA_DENSE_GEMM_PROF=1`, 3 780 calls over the prompt):

| stage | ms | rate |
|---|---:|---|
| dequantize (360 fresh, the rest cached) | 439 | 6.9 GB read |
| cast activations to FP16 | 532 | 14.0 G elements |
| oneMKL FP16 GEMM | **851** | **82.9 TFLOPS** |
| total | 1 822 | replaced 4 900 ms of ggml matmul |

## 4. CHANGE 2 before/after, part by part (`DS41_PART_PROF=1`, 8192, same binary)

| part | before (ggml ops) | after (oneMKL) | delta | submits |
|---|---:|---:|---:|---|
| q_a/q_b projection + rope | 4 225 ms | **2 887** | -1 338 | 420 -> 1 733 |
| attn out LoRA + hc_post | 2 839 | **1 557** | -1 282 | 420 -> 2 100 |
| kv projection + rope | 1 187 | 1 188 | 0 | 420 -> 1 260 |
| shared expert | 1 238 | **498** | -740 | 420 -> 2 520 |
| router (not routed: F32, see above) | 118 | 183 | +65 (noise) | 420 |
| **GEMM-shaped total** | **9 607** | **6 313** | **-3 294** | |
| attention core (window+compressed+indexer+FA) | 16 515 | 16 557 | 0 | 420 |
| hc_pre + attn_norm | 1 221 | 1 462 | +241 | |
| finish (routed add + hc_post + state cpy) | 1 397 | 1 405 | 0 | |
| init (embedding) | 968 | 679 | -289 (drift) | |
| head, writes | 112 | 111 | 0 | |
| **dense total** | **29 819** | **26 527** | **-3 292** | |

* The kv part does not move because its matmul is only 1.13 TFLOP: 1.19 s of that part is rope/norm/fake-quant and
  the projection's share was already small.  Same story, smaller, for the rest: **of the 9.6 s of GEMM-shaped parts
  only ~5.0 s was matmul** (13.8 TFLOPS), which is why the ceiling of this change is ~3.5 s and not the 8.2 s the
  FLOP table implied (it divided each part's whole time into its FLOPs).
* q_a/q_b keeps 2.9 s after the change for the same reason: 20.9 TFLOP of matmul is now 0.25 s, so the rest is rope
  over `DH*NH*n` and the two rms_norms.

## 5. Correctness

CHANGE 1 moves copies only, so its gate is bit-identity: at 1024 the last-token logits md5 is `62bcdbc1d098` before
and after; at 8192 the last row's md5 (`c63ed1d31ae8`) matched within the interleaved pair as well, and the
top-5 set and values were identical.

CHANGE 2 changes the arithmetic (FP16 products, FP32 accumulate), so the gate is the bit-comparable one.  Reference
dumps: the CHANGE 1 binary (bit-identical to the shipped baseline at 1024).

| | before | after | gate |
|---|---|---|---|
| 1024 decoded tokens (16) | `223 29429 201 ...` | **identical** | pass |
| 1024 top-5, same order | `223 603 10 16 201` (6.649817 5.603718 5.472691 5.230228 5.132542) | **same set, same order** (6.737711 5.785663 5.653891 5.387402 5.199610) | pass |
| 1024 max abs logit delta (all 129 280) | - | 0.676 | |
| 8192 decoded tokens (16) | `99301 99401 ... 100801` | **identical** | pass |
| 8192 top-5, same order | `99301 7944 79749 223 201` (28.072166 13.732338 12.933314 12.514033 12.212054) | **same set, same order** (29.128494 14.532301 13.180292 12.894463 12.336155) | pass |
| 8192 max abs logit delta | - | 2.035 | |

The delta is largest on the argmax logit (+1.06 at 8192, where the reference is 28.07); the ordering of the top-5 -
which is what decides the token - is unchanged at both arms, and the logit scale is large enough that the absolute
delta is a 3.8 % relative shift of one row.  The 8192 last row is *not* run-reproducible (S9's note), so only the
tokens and the top-5 set are a valid gate there.

## 6. Decode

Unchanged, as it must be (both changes are prompt-path): 16 tokens in 1.32/1.45 s before and 1.37/1.48 s after at
8192, 1.45/1.48 and 1.51/1.33 at 1024 - the same 99.1-99.2 % VRAM hit rate, the same 25-28 PCIe experts.  The GEMM
route's `M < 256` refusal is what guarantees it.

## 7. Bottleneck order after both changes (8192 prompt, 38.4 s)

1. **The attention core - 16.5 s (43 % of the prompt, 62 % of the dense half), 25.25 TFLOP at 1.53 TFLOPS.**  Not
   GEMM-shaped: the per-query K gather + cast, the flash attention over 128 + 512 keys of width 512, and the
   indexer's block scores/top-k over every visible row.  This needs a kernel-level effort (FA tile shape / K
   gather / indexer selection), not another GEMM route.  Nothing in this work touched it.
2. **The expert phase - 11.0 s**, of which 6.0 s is the drained expert kernels themselves (4.7 us/entry) and ~5 s
   is host work + the file tier's 100 GB of reads (5.4 GB/s, the 1024 prompt's floor).
3. **The dense GEMM-shaped parts - 6.3 s**, of which the oneMKL route is now only 1.8 s; the remaining ~4.5 s is
   rope/rms_norm/hc/`fake_quant`/submission.  That is the next dense lever, and it is elementwise work.
4. **hc_pre + attn_norm 1.5 s, finish 1.4 s, kv part 1.2 s, init 0.7 s** - all small, all fixed-cost elementwise.
5. The copy window no longer appears: 64 GB of expert blobs now move in ~2 s and are hidden under the dense half.

## 8. Open, with the numbers

* The activation cast (532 ms) still writes 2-byte stores one element at a time; an 8-wide vector store would
  roughly halve it, and `q_a`/`kv` share `xn` and `up`/`gate` share `fn`, so a two-entry cast cache could drop a
  third of the rest.
* The dequantized-weight cache holds one layer's dense weights (154 MiB); at a long context the ~5 sub-batches make
  the single fresh dequant (439 ms) worth hiding, or worth a persistent f16 copy if VRAM allows.
* The dense parts' elementwise residue (~4.5 s) is `rope` over `DH*NH*n` plus rms_norm and hc; those are the same
  shape of problem as the attention core's gather/cast and are where the next second lives.
