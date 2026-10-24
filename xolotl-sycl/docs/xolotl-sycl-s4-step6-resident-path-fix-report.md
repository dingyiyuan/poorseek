# S4 step 6 — the "resident expert path" was the route bias: the prompt routed with the decode's +1.0 / −0.6 biases

2026-10-24. Follows [step 5](xolotl-sycl-s4-step5-prefill-cache-report.md) §5, which reported that the VRAM-resident
expert path "computes a different function" from the staged path — layer 0's routed sum 62 % from the CPU reference at
0.48x the magnitude, cosine 0.12, with an identical layer-0 input `fn` and byte-identical blobs — and pointed at the
per-expert pointer table (`gp.c_ptr[] = cache->device_slot(slot_of(layer, e))` vs `c_stage + i*SL`).

**That diagnosis was wrong, and so was the evidence it rested on. The kernel honours `grp_ptr[]`; the resident launch
is bit-identical to the staged one over the same bytes. What actually differs is *which experts get routed*: with the
cache seeded before the prompt, `ds41_generate.cpp` applied the engine's DECODE route bias (+1.0 on every VRAM-resident
expert, −0.6 on the file tier) to the PROMPT's routing as well, because `bias_now` was derived from `!defer_vram` and
`STRATA_PREFILL_CACHE=1` (like `--prefill-chunk 1` and `--prompt-batch 1`) turns `defer_vram` off.** On the 8-token
prompt layer 0 then picks 8 distinct experts instead of the 30 the exact router selects, from a bit-identical input.

TL;DR:

| | |
|---|---|
| kernel boundary (`native_expert_grouped`, 3 classes, 1/6/42 groups) | scattered cache-slot pointers vs the same bytes at an even `i*L.bytes` stride: **every output, gate, up and q1_8 block byte-identical**; a 1-row pointer shift changes the output (so the table *is* dereferenced) |
| engine (L0, `STRATA_NE_DIAG`) | resident table `+4995901440 +5988464640` (scattered), staged `+13455360 +26910720 …` (even); the staged-vs-resident A/B of the same bytes: **0 values differ** |
| routed experts at layer 0 | staged 30, resident 8 — over a **bit-identical** `L00_fn` (md5 `8d216676…` in all three SYCL runs) |
| `--route-bias 0 --route-bias-file 0` | resident layer-0 routing becomes **identical** to the staged run's, logits identical to it (`mean rel 0.3285`, `max|d| 7.774` for both) |
| the fix (`ds41_generate.cpp`) | resident path now **bit-identical to the staged path end to end** (`mean rel 0.0000`, `max|d| 0`, cos 1.0), per-layer routed rel mean **0.2924** (inside the 0.194-0.367 envelope), L0 routed rel **0.0122** — the staged path's own numbers |
| cost | none for decode (unchanged: 9.5-9.9 tok/s, 98.4-98.5 % hits); the **prefill 7.8 → 2.7 tok/s**, because that speedup *was* the biased routing (see §6) |

Files: `tools/v41/engine/ds41_generate.cpp` (the fix, `s4/ne_ab/route-bias-prompt-fix.patch`),
`sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp` (`STRATA_NE_DIAG`, off by default — the evidence instrument),
`s4/ne_ab/ne_ptr_ab.cpp` (the kernel-boundary A/B).

---

## 1. The kernel boundary: `grp_ptr[g]` is what the kernel reads

`s4/ne_ab/ne_ptr_ab.cpp` links the archive's own `ne_kernel.o` (`s4/step2/obj/ne_kernel.o`) and drives
`native_expert_grouped` with no engine, no cache and no model in the loop. For each of the model's three real classes
(Q2_0/Q2_0 9.95 MB, Q3_K/Q3_K 15.2 MB, PTQ1_0/PTQ1_0 7.74 MB) it places the *same blobs* two ways and compares four
things bitwise (out, gate, up, the q1_8 blocks):

| case | what it proves | result |
|---|---|---|
| 1 expert × 1 token, blob at a staging base vs at a cache-like slot base (+15 206 656 B, a Q3_K-shaped slot in front of it) | the literal A/B step 5 asked for | **out BYTE-IDENTICAL**, gate/up/hq identical |
| 6 groups × 1 token, even `g*L.bytes` stride vs slot bases whose gaps cycle over the model's three class sizes | the decode shape with a *variable* stride — where an `i*SL` assumption would read another expert | **0/30720 values differ** |
| 42 groups × 8 tokens, same two placements | the prompt (prefill) shape | **0/245760 values differ** |
| positive control: 1 group, pointer moved by one `gu_row` | that the table is actually dereferenced | out **DIFFERS** (`max_rel` 2) |

```
ALL POINTER CASES PASS (0 failures)
```

So hypothesis 1 ("the kernel assumes the staged even stride and ignores the slot pointer") is **false**, and so is
hypothesis 3 (a slot-offset convention the kernel adds again): the only thing the kernel reconstructs from `grp_ptr[g]`
is `blob`, and `row/up_off/down_off` come from the layout, not from the pointer.

## 2. The engine: the pointer tables, as the device sees them

`STRATA_NE_DIAG=<n>` in the kernel TU reads `n_groups`, `grp_ptr`, `grp_start`, `ent_dst`, `ent_tok` back from the
device on the first `n` calls, prints them, hashes every group's blob (FNV-1a over the bytes the kernel will read), and
then re-runs the same three kernels over *copies of the same blobs at an even stride*, comparing the entry rows bitwise.
Layer 0 of the 8-token prompt, one set of counters (`prompt 8/8`, `experts 1837 hit / 0 cpu / 83 pcie / 21 file` — the
step-5 `c8-cache` numbers exactly):

```
residential slot table:
[ne-diag 0] ng=3 cap_g=3 cap_e=48 gu=11/d=42 bytes=13455360 grid=3 |
    ptr0=ffffd559bf69e800 deltas +4995901440 +5988464640 strided=no(scattered) | grp_start 0 7 15 22 | n_ent=22
    blob fnv1a: g0=3371b4db40ed6384 g1=b7fc6ced1d78c6b4 g2=6cbe6387a103c73b
    dst: 9 16 22 28 33 40 46 1 6 14 20 27 32 38 45 4 11 23 29 35 ... | tok: 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 3 4 5 ...
    staged-vs-resident A/B: 0/112640 values differ -> SAME FUNCTION (the table layout is honoured; bytes decide)

the PCIe-staged launch of the same pass (its bytes are the arena's):
[ne-diag 1] ng=1 cap_g=1 cap_e=48 gu=11/d=42 bytes=13455360 grid=1 |
    ptr0=ffffd55b5ec00000 deltas strided=YES(even) | grp_start 22 23 | n_ent=23
    blob fnv1a: g0=ca3c012df22fae8e
    staged-vs-resident A/B: 0/5120 values differ -> SAME FUNCTION
```

and the same pass in the **deferred** (staged) run (`experts 0 hit / 0 cpu / 1920 pcie / 608 file`, the step-5
`c8-stream` numbers):

```
[ne-diag 0] ng=11 cap_g=11 cap_e=48 gu=11/d=42 bytes=13455360 grid=11 |
    ptr0=ffffd558d6200000 deltas +13455360 +26910720 +40366080 +53821440 +67276800 +80732160 +94187520 +107642880
    strided=YES(even) | grp_start 0 1 2 4 5 6 8 9 10 | n_ent=16
    blob fnv1a: g0=f3aa5cf69da1995a g1=14771dee50bd3234 g2=ca3c012df22fae8e g3=8818aa5f3ec235ec ...
```

Three things follow, and they are the answer to (a):

1. The resident table is **scattered** and the staged one is **even**, exactly as step 5 said — and the kernel's answer
   is the same either way, so the table's shape is not the defect.
2. The slot bytes are the arena bytes: the resident call's `ca3c012df22fae8e` is the *same blob* the staged launch of
   the same class in the same run hashes (`c1`'s `g0`), and `9b89d45dcc42f294` / `495f99f88e427ea3` likewise appear on
   both sides. The fill is not the defect (this corroborates `STRATA_SEED_VERIFY`, which compared 8 slots byte for
   byte).
3. **The grouping is different.** The resident run's class-(11,42) pass has 22+1 applications in 4 groups; the staged
   run's has 16+6+11 in 15 groups. Same 48 `dst` slots, same 8 tokens, different experts.

## 3. The routed ids were never verified — the dump that "proved" them is stale

Step 5 states "the routed ids are identical (the router is a deterministic function of `fn` …)". That was checked
against `STRATA_LAYER_DUMP`'s `L%02d_sel.i32`. **Every one of those files is byte-identical** — all 8 layers in all 8
run directories (`md5 569cb6c4278b0a9a83ce5e2b9612ec06`), because `Ds41Dense::read_sel`'s buffer is not repopulated on
this path (`ds41_dense.cpp:2802`: "per group (read_sel has nothing to show)"). The claim was unfounded.

The `STRATA_ROUTE_LOG=<file>` instrumentation in `gpu_run_chunk` (`layer expert tier count`) is the real record, and it
shows the difference directly. Layer 0 of the same 8-token prompt, distinct routed experts:

| run | layer 0 | layer 1 | layer 2 | … | layer 7 |
|---|---|---|---|---|---|
| deferred (staged, exact) | **30** | 26 | 31 | | 27 |
| `STRATA_PREFILL_CACHE=1` | **8** | 8 | 10 | | 13 |
| `… STRATA_PREFILL_SLOTS=200` | 31 | 26 | 31 | | 7 |

The resident run routes the *same six experts for nearly every token*: `2(8), 35(7), 94(1), 105(8), 138(8), 226(1),
229(8), 242(7)` — seven of the eight tokens share one expert set. Token-independent routing is the signature of a
selection score dominated by something that does not depend on the token — which is exactly what an additive per-expert
term does.

## 4. The defect

`tools/v41/engine/ds41_generate.cpp:104`:

```cpp
float route_bias = 1.0f;      // --route-bias X: selection bonus +X on the experts resident in VRAM at the seed
                              // (DEFAULT 1.0, Mal 2026-10-11: ~4x decode; … --route-bias 0 = exact routing)
float route_bias_file = 0.6f; // --route-bias-file X: selection penalty -X on file-tier experts
```

and `:758` (before the fix):

```cpp
// bias_now: route-bias / file penalty are DECODE levers - before the VRAM seed (the prompt) only the prune part
// applies …
bool bias_now = a.bias_prompt > 0 || !defer_vram;
```

`bias_now` is the switch `apply_bias()` reads. It is turned on when the cache is (re)seeded — `open_cache()` ends with
`bias_now = true; apply_bias();` — and `!defer_vram` was used as the proxy for "the cache is already open, so we must be
decoding". That proxy is right for the deferred path (the seed happens *after* the prompt) and **wrong for every run
that seeds at load**: `STRATA_PREFILL_CACHE=1`, and equally `--prefill-chunk 1` / `--prompt-batch 1`, which leave
`defer_vram` false for the same reason (`defer_vram = (prefill_chunk > 1 || prompt_batch > 1)`). Their prompt routes
with +1.0 on the ~1200 ranked experts and −0.6 on the 637/8977 file-tier experts, i.e. it is steered to the resident
set — the 4x-decode quality trade, taken on the prompt.

Controlled proof, same binary, same prompt, only the two biases off (`--route-bias 0 --route-bias-file 0`), the cache
still seeded before the prompt:

| run | layer-0 experts | logits vs CPU | logits vs deferred |
|---|---|---|---|
| deferred (exact) | 30 | mean rel 0.3285, max 7.774, cos 0.949 | — |
| `PREFILL_CACHE` (biases on) | 8 | mean rel 1.2686, max 31.73, cos 0.476 | — |
| `PREFILL_CACHE --route-bias 0` | 37 | mean rel 0.9085, max 19.44, cos 0.586 | — |
| `PREFILL_CACHE --route-bias 0 --route-bias-file 0` | **30, identical set** | **mean rel 0.3285, max 7.774, cos 0.949** | **bit-identical** |

`--route-bias 0` alone is not enough because the file penalty (default 0.6) is a second, independent bias.

## 5. The fix

`tools/v41/engine/ds41_generate.cpp`, two hunks (also `s4/ne_ab/route-bias-prompt-fix.patch`):

```cpp
// "!defer_vram" alone is NOT "we are decoding": whenever the cache is seeded AT LOAD …, the prompt still lies
// ahead and used to route with the DECODE biases on.  The bias is turned on once the prompt is done: by
// open_cache() for the deferred seed, by the block after the prompt otherwise.
const bool prompt_ahead = prompt.size() > 1;
bool bias_now = a.bias_prompt > 0 || (!defer_vram && !prompt_ahead);
```

and, after the prompt (beside the deferred seed):

```cpp
    if (prompt_ahead && !defer_vram) {
        bias_now = true;
        apply_bias();
    }
```

Why this makes the resident path equal to the staged path: the expert kernel already computes the same thing from
either pointer convention (§1, §2), so the only remaining difference between the two paths was *which* (layer, expert)
pairs the prompt asks for. Both paths now route the prompt exactly, over the same router, the same `fn`, the same blob
bytes and the same kernel; the seeded cache changes only *where* those bytes are read from. The decode keeps the bias
it was designed with, and it is now applied at the right moment (after the prompt), from the real seed's residency.

## 6. Measurements

Command shape: `-m V41-GSQ-1017.gguf --backend cuda --experts gpu --engram engram-layers.1:engram-layers.14 --rank
rank.bin --ids <48 ids> -n 16 --ctx 2048 --temp 0 --slots 1200 --prompt-batch 0 --prefill-chunk 8` (the step-5
`f-base` / `f-cache` shape, no bias flags), `s4/ne_ab/run_ab.sh`, one process at a time, MemAvailable watchdog.

| run | prompt | expert applications | decode | decode hits |
|---|---|---|---|---|
| deferred (step 5's `f-base`) | 2.5 tok/s (18.93 s) | 0 hit / 11520 pcie / 3267 file | 9.39 tok/s | 98.5 % |
| `PREFILL_CACHE=1`, **before the fix** (step 5's `f-cache`) | 7.8 tok/s (6.16 s) | 10944 hit / 576 pcie / 140 file | 9.60 tok/s | 98.4 % |
| `PREFILL_CACHE=1`, **after the fix** | 2.7 tok/s (18.04 s) | 1439 hit / 10081 pcie / 3267 file | 9.54 tok/s | 98.5 % |
| `PREFILL_CACHE=1 --bias-prompt 1200` (after the fix) | 7.7 tok/s (6.22 s) | 10944 hit / 576 pcie / 140 file | 9.92 tok/s | 98.4 % |

**Decode is unchanged** (9.4-9.9 tok/s, 98.4-98.5 % hits before and after): the fix moves the bias's start from load
to the end of the prompt, and the decode still runs with the seed's residency and the same two biases.

**Prefill loses the 3.1x.** That speedup was the biased routing: the prompt's applications were being steered to the
1200 resident experts (95 % hits), which is not the model's routing. With exact routing the same 1200 slots cover
1439/11520 = 12.5 % of a 48-token prompt's applications (a 1200-slot cache holds 30 of 384 experts per layer, and the
ranked set only partially overlaps what the prompt routes), so the DMA/file traffic is back. `--bias-prompt N` is the
engine's existing knob for *deliberately* measuring the biased prompt and it reproduces the old number exactly
(7.7 tok/s, 10944 hits) — the fast behaviour is still available, it is now requested instead of implied, and it can be
compared against a clean reference.

### Correctness, 8-token prompt, resident path with the fix (the criterion of step 5 / s3b)

`STRATA_PREFILL_CACHE=1`, default flags, `STRATA_LAYER_DUMP` + `--dump-logits`, vs `--backend cpu --experts cpu`:

| | layer-0 routed | per-layer routed rel (0-39) | last-row logits |
|---|---|---|---|
| resident, before the fix | rel 1.0036, cos 0.318, 0.60x | mean 1.2406 | mean rel 1.27, max\|d\| 31.7, cos 0.476 |
| resident, **after the fix** | **rel 0.0122, cos 0.9999, 1.00x** | **mean 0.2924** (min 0.0122, max 0.4696) | **mean rel 0.3285, max\|d\| 7.774, cos 0.949** |
| deferred (staged) | rel 0.0122, cos 0.9999, 1.00x | mean 0.2924 | mean rel 0.3285, max\|d\| 7.774, cos 0.949 |

The resident path's numbers are now **the staged path's numbers, digit for digit**, and the fixed resident run against
step 5's own deferred dumps (`/tmp/s5/dA8`, the `c8-stream` run) is **`mean rel 0.0000, max|d| 0, cos 1.00000` on all
40 layers' `routed` and `fn` tensors** — bit-identical, i.e. seeding the cache before the prompt now changes only where
the bytes are read from, not what the model computes. Per-layer routed rel mean
0.2924 sits inside the CPU-vs-CPU chaos envelope (0.194-0.367), and layer 0's routed rel is 0.0122, the same as the
staged path's.

## 7. What else depended on this

1. **[step 5](xolotl-sycl-s4-step5-prefill-cache-report.md) §5 is wrong and must be read as retracted.** The resident
   launch never computed a different function: the kernel and the pointer table are correct (§1), the slot bytes are
   correct (§2), and the 62 % is the prompt's routing (§3, §4). Its "Next experiment" (a one-expert A/B of the two
   pointer conventions) was run here and passes bitwise.
2. **The 3.1x prefill number is a biased-routing number.** It is reproduced exactly by `--bias-prompt`, so it is a real
   measurement of *that* configuration — but it must not be described as free, and `STRATA_PREFILL_CACHE` as a default
   for the one-shot chunked path would trade prompt quality for it (which is why the fix routes exactly by default).
3. **Every "decode is numerically wrong" claim is retracted.** The decode has always run the exact expert arithmetic;
   its routing carries the engine's documented speed/quality bias (default `--route-bias 1.0` since 2026-10-11,
   "~4x decode"), which is a config choice, not a defect. Any engine-vs-CPU comparison that runs with `defer_vram`
   false — `--prefill-chunk 1`, `--prompt-batch 1`, `STRATA_PREFILL_CACHE`, and a 1-token "prompt" — must pass
   `--route-bias 0 --route-bias-file 0` to be a comparison of the *math*.
4. **Step 2 / step 4 are unaffected in what they measured** (they are speed numbers, and the decode bias is part of the
   measured configuration), but their `--prefill-chunk 1` runs were also routing the prompt with the decode bias, so any
   *quality* statement derived from such a run needs re-measuring with the biases off. Nothing in steps 2-4 depends on
   the pointer-table hypothesis.
5. `STRATA_LAYER_DUMP`'s `L%02d_sel.i32` is not a routing record on this path; use `STRATA_ROUTE_LOG` or
   `STRATA_NE_DIAG`. Anyone who compared those files across runs compared one stale buffer with itself.

## 8. Reproduce

```
# the kernel-boundary A/B (no engine, no model): 3 classes x {1 expert/1 token, 6x1, 42x8, pointer-shift control}
cd /home/dyy/deepseek-work/s4/ne_ab && ./build_ab.sh && ONEAPI_DEVICE_SELECTOR=level_zero:0 ./ne_ptr_ab

# the engine-side pointer tables + blob hashes + staged-vs-resident A/B (needs the diag TU linked in)
BIN=/home/dyy/deepseek-work/s4/ne_ab/ds41_ne_diag STRATA_PREFILL_CACHE=1 STRATA_NE_DIAG=9 \
  ./run_ab.sh diag-cache 100,200,300,400,500,600,700,800 1 8
STRATA_NE_DIAG=9 ./run_ab.sh diag-staged 100,200,300,400,500,600,700,800 1 8

# the routing difference (needs no instrumentation change)
STRATA_PREFILL_CACHE=1 STRATA_ROUTE_LOG=/tmp/rc.txt ./run_ab.sh rc 100,200,...,800 1 8
STRATA_ROUTE_LOG=/tmp/rs.txt ./run_ab.sh rs 100,200,300,400,500,600,700,800 1 8
STRATA_PREFILL_CACHE=1 STRATA_ROUTE_LOG=/tmp/rn.txt ./run_ab.sh rn 100,200,300,400,500,600,700,800 1 8 \
  --route-bias 0 --route-bias-file 0        # identical routing to rs.txt

# the fix: build with the other engine objects, private to s4/ne_ab (s2/engine-obj/ds41_generate.o is not touched)
/home/dyy/deepseek-work/s4/ne_ab/build_gen.sh && /home/dyy/deepseek-work/s4/ne_ab/link_fix.sh
BIN=/home/dyy/deepseek-work/s4/ne_ab/ds41_fixed STRATA_PREFILL_CACHE=1 STRATA_LAYER_DUMP=/tmp/dfix \
  ./run_ab.sh fix1 100,200,300,400,500,600,700,800 1 8 --dump-logits /tmp/FIX.f32
python3 cmp_dump.py /tmp/s6/dcpu /tmp/s6/dfix 40      # 0.0122 at L0, mean 0.2924
```

Artifacts: `s4/ne_ab/{ne_ptr_ab.cpp, ne_ptr_ab, build_ab.sh, run_ab.sh, build_gen.sh, link_fix.sh, cmp_dump.py,
route-bias-prompt-fix.patch, *.log}`, dumps in `/tmp/s6/` (volatile). Binaries under test:
`s4/ne_ab/ds41_fixed` (fix + `STRATA_NE_DIAG`), `s4/ne_ab/ds41_ne_diag` (diag only, pre-fix),
`s4/step2/ds41_generate_gpuexp` (the shared build; not modified by this step).
