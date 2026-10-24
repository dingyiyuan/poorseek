# S3b/3 — fix and validation: ggml-sycl's scan/merge `top_k` emitted `-1` for `-inf` scores

2026-10-24. Follows `s3b/REPORT2.md` (localisation + causal probe) and `docs/xolotl-sycl-s2-report.md` (the SYCL
build). The fix is `tools/xolotl/ggml/xolotl-sycl-topk.patch`; nothing in it is model-specific.

## 1. Root cause, restated with one correction

`top_k_scan_merge_f32` (ggml-sycl.cpp) admitted a candidate only via `if (val > kth)`, with `kth` seeded to
`-FLT_MAX`. A score of `-inf` is a real score (a masked entry), and `-inf > -FLT_MAX` is false, so `-inf` candidates
were never inserted and their slots kept `li = -1`. `-1` is not a row id, so a caller that gathers with the ids reads
one row before the buffer.

**Correction to REPORT2:** the model's `index_topk` metadata is **512**, not 8 (`GGUF
deepseek41.index_topk = 512`). The observed `k = 8` in the 8-token repro comes from the *cap*:
`k = min(gcap, index_topk)`, `gcap = next_pow2(max((pos0+n)/ratio, 8))`. With 8 tokens / chunk 8 and ratio 2,
`gcap = 8`. That does not change the localisation: `k = 8 <= SYCL_TOP_K_SCAN_MERGE_MAX_K`, the scan/merge kernel,
whose row has 8 slots of which only `nvis` are visible, so early queries pass fewer than `k` finite scores.

**Trigger (general, not model-specific):** `ggml_top_k` with `1 <= k <= 8` where a row has **fewer than k scores
strictly greater than `-FLT_MAX`** (i.e. some `-inf`, or the sentinel value itself). For `k > 8` the radix path is used
and it is correct (verified), so ordinary vocabulary top-k (k ≈ 40) was never affected. Any ggml-sycl caller with a
small `k` and masked/`-inf` scores hits it.

## 2. The fix

`tools/xolotl/ggml/xolotl-sycl-topk.patch` (82 diff lines, one function, both stages of it):

* per-lane scan: `int filled = 0`, admit while `filled < k || val > kth`, insert into slot `filled` (so the empty-slot
  tombstone is never compared), `filled++`.
* merge: `int ffilled = 0`, skip a lane at the first `li < 0` (empty slot), admit while `ffilled < k ||
  val > fkth`, same insertion rule, `ffilled++`.

For scores `> -FLT_MAX` this is order-for-order identical to the old code: tombstones are `-FLT_MAX`, so the old
backward search from `k-1` walked through them to exactly the position the new search from `filled` starts at.

Verified to reproduce the built source exactly (`git apply` on the pristine tree + `diff`), and
`git apply --check` is clean against `llama-cpp-ref` (a0e6dc7).

## 3. Kernel-level evidence

Harness: `s3b/topk/test_topk.cpp` (CPU and SYCL in one process; checks every id is in `[0,ncols)`, distinct, and that
the selected **score multiset** equals `std::partial_sort` over all `ncols`, the CPU body's algorithm; set equality is
required when the cut is not tied). Pre-fix library: `s3b/topk/prefix/libggml-sycl-prefix.a`
(sha256 1b4db4f6…c3e5); post-fix: the rebuilt `libggml-sycl.a` (sha256 90fffef6…fbfb). 27 cases × 2 backends.

| case (all k ≤ 8 unless noted) | pre-fix SYCL | post-fix SYCL | CPU |
|---|---|---|---|
| `n8k8_one_inf` (k = ne0 = 8, 3 × -inf) | `[0,4,2,5,7,-1,-1,-1]` | `[0,4,2,5,7,1,3,6]` | same set |
| `n8k8_inf_first` | `[-1,7,-1,-1,-1,-1,-1,-1]` | `[0,7,1,2,3,4,5,6]` | same set |
| `n8k8_all_inf` | all `-1` | `[1,0,2,3,4,5,6,7]` = {0..7} | `[4,7,1,3,5,0,2,6]`, same set |
| `n8k8_fltmax_and_inf` | all `-1` | all 8 valid | same set |
| `n8k8_fltmax_only` (finite boundary) | all `-1` | all 8 valid | same set |
| `n8k3_two_fin` (k < ne0) | `[3,0,-1]` | `[3,0,1]` | `[3,0,1]` |
| `n8k3_all_inf` (k < ne0) | all `-1` | 3 valid, scores all -inf | 3 valid, scores all -inf, other ids (cut tie) |
| `n16k8_four_fin` | 4 valid + 4 `-1` | 8 valid | 8 valid |
| `n8192k8_three_fin` (split + merge) | 3 valid + 5 `-1` | 8 valid | 8 valid, other -inf ids (cut tie) |
| `n8192k8_all_inf` | all `-1` | 8 valid | 8 valid, other ids (cut tie) |
| `n16k16_all_inf` (k = 16 → radix) | **correct** | correct | correct |

* **All-`-inf` rows, defined:** for `k == ne0` the correct output is **every index in `[0,ne0)`**, in any order (score
  multiset = k × `-inf`); CPU and post-fix SYCL both produce {0..7}. For `k < ne0` the correct output is any **k
  distinct** indices (score multiset = k × `-inf`); which tied `-inf` indices are chosen is unspecified on every
  backend — CPU picks one subset, SYCL another (`cut_tied` in the harness). This is the same rule ggml's own
  `tests/test-backend-ops.cpp::test_top_k_inf` encodes (it compares selected *values*, never ids, when ties are
  possible). No duplicates, no `-1`, in either case.
* **No regression:** of 27 SYCL cases, 13 are bit-identical before/after. Every one of them is a row with at least
  k scores `> -FLT_MAX` (all-finite rows, all-equal finite rows, negative rows, the split path with enough finite
  scores, `k = 16`). The 14 that change are exactly the rows that previously got a `-1` (plus `-FLT_MAX`-only). CPU is
  unaffected on all cases (the patch touches only ggml-sycl).
* **Post-fix:** 0 failures on both backends, including all `-inf` rows and both the single-pass and the split/merge
  launch shapes.
* `-FLT_MAX` note: `-FLT_MAX` is the kernel's empty-slot sentinel; a *genuine* score of exactly `-FLT_MAX` used to be
  rejected like `-inf` and is now selectable. No score domain in this tree produces that value; it is the strictly
  more correct generalisation of "emit k valid ids".

## 4. Model-level evidence

Production engine rebuilt from the committed (probe-free) source, only ggml changed
(`s3b/topk/link_prod.sh`, `DENSE_OBJ=obj/ds41_dense_prod.o`; the object is
`git show ea31cc7:tools/v41/engine/ds41_dense.cpp`, sha256 cfffb2fc…0012 — the commit before the diagnostic probes
were committed as ec48fac). Control: the same objects relinked against the pre-fix
library are **bit-identical** to the REPORT2 baseline (`max|d| = 0` on both backends, all 88 per-layer dump files
identical), so the only variable is the kernel. `--prompt-batch 0 --prefill-chunk 8`, `STRATA_ROUTED_HOST=1`,
`--backend cpu` vs `--backend cuda` (= SYCL), 8-token prompt.

| run | token | logits max/mean/p99 | L2 `fn` rel(max) |
|---|---|---|---|
| CPU (any ggml) | 223 | — | — |
| SYCL pre-fix | 200 | 12.856 / 1.423 / 4.810 | 0.5074 |
| SYCL + kernel fix | **175** | **6.018 / 0.636 / 2.104** | **0.05946** |

The kernel-fixed SYCL logits are **bit-identical to the probe-assisted mask repair of REPORT2** (`max|d| = 0` vs
`logits-fix-sycl.f32`), CPU logits are bit-identical to pre-fix, and the two runs' `sel` dumps differ only in the 8
compressed-attention layers, where the fix replaces `-1` by valid ids (the probe left `sel` dirty and repaired only the
mask). So the kernel fix strictly dominates the probe: same final numbers, and the out-of-bounds gather source is gone.

Node level (`DS41_NODEDUMP`, fixed ggml, diagnostic binary; unpinned production numbers above): call 5 / node 180
`TOP_K ne=[8,8]`: `48/64` ids were `-1` → **0/64**, `ids0 = [1,0,2,3,4,5,6,7]`; node 190 (mask `GET_ROWS`): finite
slots **16/64** = the CPU pattern (pre-fix 24/64); node 186 `sel` is a permutation of 0..7.

Per-layer curve vs the REPORT2 §b CPU-vs-CPU band (rel = 1e-7 … 1e-3, recomputed from the same dumps):

| L | pre-fix | fixed | band |
|---|---|---|---|
| 2 | 0.5074 | **0.05946** | 0.01964 – 0.0723 |
| 3 | 0.7675 | 0.1113 | 0.0433 – 0.1222 |
| mean 0–39 | 0.9001 | **0.2991** | 0.194 – 0.367 |
| max 0–39 | 1.967 | **0.7749** | 0.512 – 1.15 |

L2 is inside the band (0.82 × the band's top); mean and max are inside. 35/40 layers are inside; L0, L9, L10 are
slightly **below** the band's floor (better agreement than the least-perturbed CPU realisation); L32 and L38 are
3.4 % and 3.7 % above the top realisation (0.533 vs 0.516, 0.698 vs 0.673) — a 5-sample envelope, not the bug (the
probe-assisted curve of REPORT2 has the same tail). Pre-fix, 38/40 layers are above the band's top, up to **7.9×**
(L5). The fixed per-layer values match REPORT2's A-fixed column to 4 significant digits.

## 5. Falsifier: long context (`nvis >= 8`)

96-token prompt, `--prefill-chunk 8` (12 chunks), both backends, pre-fix and fixed; CPU-vs-CPU band from
`DS41_PERTURB_STATE` 1e-6/1e-5/1e-4.

| logits | max\|d\| | mean\|d\| | p99\|d\| | token |
|---|---|---|---|---|
| CPU vs CPU 1e-6 / 1e-5 / 1e-4 | 0.768 – 1.113 | 0.125 – 0.199 | 0.405 – 0.654 | 223/223/223 |
| CPU vs SYCL **pre-fix** | 0.960 | 0.133 | 0.467 | 223 |
| CPU vs SYCL **fixed** | 0.824 | 0.138 | 0.454 | 223 |

**Pre-fix already agreed at long context** (inside the CPU-vs-CPU envelope on every statistic, same token), and the
fixed run agrees as well. The mechanism is corroborated, with one refinement: a long run's *later* chunks leave the
buggy kernel entirely — `gcap = next_pow2(max((pos0+n)/ratio, 8))` is 64/128, so `k = min(gcap, 512) > 8` and the
**radix** path runs (correct for `-inf`, see `n16k16_all_inf`). The last chunk's `sel` is `-1`-free and identical
pre/post fix. But the long run is **not** bit-identical pre vs post (SYCL `max|d| = 1.090`): its first one or two
prompt chunks still have `gcap = 8`, hence `k = 8`, hence the scan/merge kernel — a node dump of the 96-token pre-fix
run shows the same `48/64` `-1` at call 5 (chunk 0, layer 2), and 0/64 with the fix. So "nvis ≥ k ⇒ no `-1`" is a
per-query statement (confirmed: every kernel case with ≥ k finite scores passed pre-fix, e.g. `n8192k8_half_inf`), but
a model run can never be entirely in that regime: every sequence starts at position 0.

## 6. Paths exercised / not exercised

| path | result |
|---|---|
| sequential chunk path (`--prompt-batch 0`), 8-token | fixed; L2 0.507 → 0.059 |
| layer-major batch path (`--prompt-batch 8`, the `attn_core_slab` `m1` site) | pre-fix CPU 223 / SYCL 200; fixed 223 / 175; fixed batch is **bit-identical** to fixed chunk path on both backends |
| chunk length > 8 (`--prefill-chunk 16`, one 16-token pass) | pre-fix L2 0.546 with `120/128` `sel` slots differing; fixed L2 0.0673, `sel` identical to CPU. Mean/max 0–39 inside the (2-realisation) band; fixed L2 is 1.28× the band's top — at the edge, not claimed as inside |
| more than 8 chunks (96 tokens / 8 = 12) | 96-token run above |
| decode (`-n 4`) | CPU tokens identical pre/fixed (223,200,175,100); SYCL 200,200,400,500 → 175,257,244,248 (first decode token moves 200→175, the same fix signature; later tokens are the model's chaos) |
| **CUDA backend** | **not tested — this box has no CUDA device; `--backend cuda` here is SYCL.** CUDA's top_k uses CUB/radix/bitonic selection, which returns valid indices, but that is unmeasured here |
| NaN scores | not tested (the comparator is not a total order for NaN; `partial_sort` then has unspecified behaviour too) |
| radix path beyond k = 16 (`k = 32`, `k = 512` model sizes), batch B > 8 at long prompt, multi-GPU, `--serve` | not exercised |

## 7. Reproduction

```
# kernel harness (rebuilds in ~20 s each link; no model process may be running)
cd s3b/topk
./build_test.sh test_topk_prefix prefix/libggml-sycl-prefix.a && ./test_topk_prefix both   # pre-fix: 14 FAIL
./build_test.sh test_topk_fixed  /home/dyy/deepseek-work/s2/build/ggml/src/ggml-sycl/libggml-sycl.a \
  && ./test_topk_fixed both                                                               # post-fix: 0 FAIL
# engine binaries (production = committed source, diagnostic = current tree with env-gated probes)
./build_dense_prod.sh && ./link_prod.sh ds41_generate_topkfix <libggml-sycl.a>
DENSE_OBJ=obj/ds41_dense_probe.o ./link_prod.sh ds41_generate_nodedump_fix <libggml-sycl.a>
# model runs
cd s3b && ./runx.sh <tag> {cpu|cuda} <bin>                 # 8-token cheap repro, dumps + logits
RIDS=$(cat long/ids96.txt) ./runx2.sh <tag> {cpu|cuda} <bin>   # 96-token long context
```

Caveats: `DS41_NODEDUMP` pins every node, so its token is not the unpinned token (303 pre-fix in REPORT2; 175 with
the fix, matching the unpinned fixed run because the causal `-1` damage is gone). The long-context chaos band has 3
realisations and the 16-token band 2, so those envelopes are coarse; the 8-token band uses the 5 REPORT2 realisations.
