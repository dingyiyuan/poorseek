# S4 step 9 — the two levers shipped as defaults, and the wide-batch determinism gate that stops the sweep

Status: **both levers shipped and verified bit-exact where the engine is reproducible (256, 1024). The 2048 determinism
gate FAILS on the unchanged build, so the throughput sweep is on hold.** The S8 control-ring fix is *not* the cause:
the pre-fix and fixed binaries produce the same outcome family at 2048, three distinct last rows exist, and with the
shipped defaults the point splits again. The divergence localizes to the *routing* at layer 4 (layers 0-3 route
identically) and then propagates through every later layer; it is present at 4096 and 8192 too. 256 and 1024 are
bit-exact and stayed bit-exact after the flips.

## 1. TL;DR

* **Shipped (defaults, env-reversible):** `DS41_SHIM_ASYNC_STAGE` on (`=0` restores the per-copy queue drain) and the
  whole-layer prefetch **width-gated on at >= 1024 rows, off below** (`STRATA_CHUNK_PF_MIN`, `STRATA_CHUNK_PF_EXACT`/
  `_WIDTH`, `--no-chunk-prefetch` all still override). Comments carry the measured curve.
* **Verification, final binary (`s4/step2/ds41_generate_gpuexp`, md5 `e86e83ba…`), shipped defaults, `--dump-last`:**
  1024 -> `787519ba` (the established baseline, 3 runs today), 256 -> `fce1788f` (3 runs). Both levers are exact; the
  width gate does what it says (no `chunk prefetch:` line at 256, two 3.30 GiB slabs at 1024/2048/4096/8192).
* **Measured with both levers at default (idle box, same prompt, `-n 16`):** 256 `13.5 -> 14.1-14.3 tok/s` (+4-6 %,
  async only, the slab stays off), 1024 `33.6 -> 34.8-35.7` (+3.6…+6.3 %), 2048 `44.4-44.6 -> 45.5-45.8` (+2.4…+3.1 %,
  file wait 10.7 s -> 0.0).
* **2048 determinism (TASK 2): FAILS.** Fixed binary, pre-flip default flags, 4 runs: `6c6884b0`, `5189d36b`,
  `6c6884b0`, `5189d36b` (2 + 2). Pre-fix binary (`s4/step8/ds41_ref`), 2 runs: `5189d36b`, `a5787d7e`. Three distinct
  last rows exist in total, each reproducible to the exact md5; the S8 ring fix did not change the family.
* **First divergence is ROUTING at layer 4.** `STRATA_ROUTE_LOG` on two 2048 runs: layers 0-3 identical (same experts,
  same per-expert token counts), layer 4 differs by 4 experts +-1 token (about two flipped (token, expert) picks), and
  every layer after it differs. So the flip needs no prior routing difference — it is a top-k decision (near-tie or an
  exact tie) whose inputs are nominally identical, i.e. some value upstream moves by ~1e-7, or the selection itself is
  unstable.
* **Not the levers:** the divergence reproduces with the slab off (default arm), with `--no-chunk-mmq` (MMVQ),
  with `GGML_SYCL_ENABLE_DNN=0`, with the async stage off, and with the route log on (that last one produced a *third*
  md5, `a5787d7e` — the very value S7 attributed to the async stage, so S7's async-changed-the-logits observation was
  one member of this family, not a dependency).
* **Wide sizes are not immune:** with the shipped defaults 2048 = 4x`a5787d7e` + 2x`5189d36b` (6 runs), 4096 = 3
  distinct md5s in 4 runs, 8192 = 3 distinct md5s in 3 runs. The engine *does* survive the width itself (8192 prompt:
  78.5 tok/s, correct decode, 11.90 GiB of a 18.96 GiB-free card, clean 6 s refusal when it does not fit).
* **The wall, with the slab shipped:** file wait is 0.0 s at every size >= 1024 (it was 30 % of the prompt at 1024).
  What is left is the expert stream (73-76 % of the prompt) and the dense half (24-26 %). Dense grows ~linearly with
  tokens, experts sublinearly, so per-token prefill cost falls 71.4 -> 28.7 -> 21.9 -> 16.4 -> 12.7 ms/token at
  256/1024/2048/4096/8192 — and the gain per doubling collapses to 1.29-1.33x after 1024.
* **Prefetch economics vs B:** the routed share of a layer's whole file tier (file-tier DMAs / 8977) is 37 % at 256,
  55 % at 1024, **63 % at 2048, then it stops**: 59 % at 4096 and 60 % at 8192, because enc-skip removes the deep
  layers for the cut tokens. So the prediction "the ratio keeps rising with B" holds only up to 2048; past it the slab
  still reads the same ~100 GB to cover ~60 % of the routing, i.e. the win is flat, not growing.
* **New preflight:** the prompt batch's VRAM/RAM is now computed from B before the first allocation and **refused with
  the numbers** (exit 2, 6 s, `--prompt-batch <= N`, `STRATA_PB_CLAMP=1` to halve instead). No size literal in sizing:
  the check is a budget on B, not an assertion on a batch size.

## 2. What shipped (TASK 1)

### 2.1 `DS41_SHIM_ASYNC_STAGE` — default ON

`tools/v41/engine/sycl_compat/ds41_sycl_compat.cpp`, `copy_in`: the staged pageable H2D copy is no longer waited on
per copy; the wait stays on the reuse of its ring slot (`kStageSlots` copies later). `DS41_SHIM_ASYNC_STAGE=0`
restores the drain for an A/B (any other value, or unset, is on).

Why it is safe to default: it is bit-exact wherever this engine is reproducible, on both binaries —
1024 -> `787519ba` (gate off and gate on), 256 -> `fce1788f`, and it moves no extra bytes (the ring is the same single
host -> stage -> device copy). It is worth +2.4 % prefill at 1024 in S8, and at 256 with the slab off it is the whole
of `13.5 -> 14.1-14.3 tok/s`. S7's reason for keeping it opt-in ("async changed the logits at 2048") is void: 2048
changes its own logits (§3).

### 2.2 Whole-layer prefetch — WIDTH GATE ON at >= 1024 rows, OFF below

`tools/v41/engine/ds41_moe.cpp`, `chunk_pf_min_rows()` now returns `kPfWidthRows = 1024` by default (it was
`kPfNever` unless `STRATA_CHUNK_PF_EXACT`). The measured curve is in the comment: the slab is a **22 % loss at 256
rows** (13.5 -> 10.6 tok/s: a 99-102 GB whole-file-tier sweep for a routing share of 37 %) and a **3 % win at 1024**
(33.6 -> 34.6, file wait 9.0 -> 0.0 s), so the break-even sits between them. Overrides: `STRATA_CHUNK_PF_MIN=<rows>`
(`0` = every width, `off` = never), `STRATA_CHUNK_PF_EXACT=1` + `STRATA_CHUNK_PF_WIDTH=<rows>` (the S7 spelling),
`--no-chunk-prefetch`, `STRATA_CHUNK_PF_MODE=2/3` (the S8 bisect arms).

Verified: at 256 the gate keeps the slab off (no `chunk prefetch:` line, file wait 5.9-6.0 s, `fce1788f`); at
1024/2048/4096/8192 it allocates the two 3.30 GiB slabs, file wait 0.0 s, and the md5s match the slab-off baselines at
both reproducible sizes.

### 2.3 Measured effect of the two defaults (same prompt, `-n 16`, idle box)

| rows | shipped defaults | S8 gate-off baseline | delta | file wait |
|---|---|---|---|---|
| 256 | 14.1 / 14.3 | 13.5 | +4…+6 % (async only) | 5.9 s (slab off by design) |
| 1024 | 34.8 / 35.2 / 35.7 | 33.4 / 33.8 (33.6) | +3.6…+6.3 % | 0.0 s (was 9.0) |
| 2048 | 45.5 / 45.7 / 45.8 | 44.4 / 44.5 / 44.6 | +2.4…+3.1 % | 0.0 s (was 10.7-10.9) |

(The 2048 row is a *throughput* observation only; the size does not pass its determinism gate, §3.)

## 3. The 2048 determinism gate (TASK 2) — FAILS, and the ring fix is not the cause

`-n 8 --dump-last`, default flags, md5 of the last prompt row's 129280 logits. Prompts `(i*100)%120000+1`, ctx 4096.

| binary | arm | md5 | file DMAs |
|---|---|---|---|
| fixed (`s4/step2/…gpuexp`) | default (gate off, async off) | `6c6884b0` | 5671 |
| fixed | default | `5189d36b` | 5675 |
| fixed | default | `6c6884b0` | 5671 |
| fixed | default | `5189d36b` | 5675 |
| pre-fix (`s4/step8/ds41_ref`, = S7) | default | `5189d36b` | 5675 |
| pre-fix | default | `a5787d7e` | 5671 |
| fixed | `--no-chunk-mmq` | `5189d36b` | 5675 |
| fixed | `--no-chunk-mmq` | `a5787d7e` | 5671 |
| fixed | `STRATA_ROUTE_LOG` on | `a5787d7e` | 5671 |
| fixed | `STRATA_ROUTE_LOG` on | `5189d36b` | 5675 |
| fixed | `GGML_SYCL_ENABLE_DNN=0` | `5189d36b` | 5675 |
| fixed | `GGML_SYCL_ENABLE_DNN=0` | `5189d36b` | 5675 |
| fixed | **shipped defaults** (slab + async) | `a5787d7e` x4, `5189d36b` x2 | 5671 x4, 5675 x2 |

Three outcomes exist in total — `6c6884b0`, `5189d36b`, `a5787d7e` — and each repeats to the exact md5, across binaries
(S7's `a5787d7e` from its async arm is the third one). So 2048 is a lottery over a small set of discrete forward
passes. It is not a corrupted tensor: all 129280 logits are finite, the top-1 is the same in every arm, the top-5 set is
the same, and the fixed binary decodes the same 16 tokens (`84901 85001 85101 85201 85301`) from all three md5s.

**Where it first diverges: layer 4's routing.** `STRATA_ROUTE_LOG` (per layer, per expert, tier and token count) on
`a5787d7e` vs `5189d36b`: layers 0-3 are byte-identical, layer 4 has 4 experts off by one token (127: 574 vs 575,
152: 427 vs 426, 165: 288 vs 287, 352: 78 vs 79 — about two flipped picks out of 12288 entries), and layers 5-39 all
differ afterwards. That is a *routing* difference with no prior routing difference to explain it: the router input of
layer 4 (`fn`, layer 3's output) must be bit-identical in routing terms, so either a ~1e-7 value difference upstream
flips a near-tie at the top-k boundary, or the selection is unstable on exact ties (the scores go through
`sqrt(softplus(rl))` then `ggml_argsort_top_k`).

**What it is not** (each ruled out by a run pair above): the class-pass control ring (fixed binary == pre-fix binary
family), the whole-layer slab (fails with the slab off), the async stage (fails with it off), the MMQ prompt kernels
(fails on MMVQ too), oneDNN's DNN path (fails with `GGML_SYCL_ENABLE_DNN=0`), and the prefetch/staging levers
(fails on the shipped defaults too). The shift in host lead that the route log itself introduces moves the outcome to
its own md5 — this is a *timing-sensitive* instability, not a fixed data-dependent coin.

## 4. Wide-batch gates: 4096 and 8192 (the sweep stays on hold)

TASK 2 said: if 2048 still differs, stop the sweep. It does, so no throughput table is claimed as admissible. The same
gate was still run at 4096 and 8192 — it is the thing that answers "does the code survive scaling", and it decides
whether those points can ever be measured:

| rows | ctx | runs -> md5 | gate | tok/s (context only) |
|---|---|---|---|---|
| 1024 | 4096 | `787519ba` x3 | PASS (reference) | 34.8 / 35.2 / 35.7 |
| 2048 | 4096 | `a5787d7e` x4, `5189d36b` x2 (6 runs) | **FAIL** | 45.5 / 45.7 / 45.8 / 45.9 |
| 4096 | 8192 | `46666811`, `42e68e90` x2, `a28a487c` | **FAIL** | 60.7 / 60.9 / 61.0 |
| 8192 | 12288 | `c69a4e87`, `324451c9`, `966c3241` | **FAIL** | 78.4 / 78.5 / 78.5 |

Everything the width itself touches works: 4096 and 8192 need `--ctx > prompt`, the preflight clears both, the two
3.30 GiB slabs allocate, `enc-skip` engages (`cut = 1536` at 4096, `5632` at 8192, replay reach 2560), the 8192 prompt
finishes in 104.3-104.5 s and decodes 16 tokens correctly, and the post-prompt VRAM seed finds 18.4-18.4 GiB of slots.
The prompt batch is a *budget*, not a constant: nothing in the path needed a 4096/8192-sized literal.

## 5. Where the wall moves, and what binds

Shipped defaults; `prompt split` is the engine's own accounting (idle box, one process).

| rows | prompt s | tok/s | ms/token | dense | experts | file wait | pcie | file | routed/file-tier |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 17.9-18.1 | 14.1-14.3 | 69.8-70.9 | 3.9 | 13.5 | 5.9-6.0 | 61 440 | 3 329 | 37 % |
| 1024 | 28.7-29.4 | 34.8-35.7 | 28.0-28.7 | 6.9-7.1 | 21.3 | 0.0 | 245 760 | 4 919 | 55 % |
| 2048 | 44.7-44.9 | 45.5-45.8 | 21.8-21.9 | 11.5-11.6 | 32.3-32.8 | 0.0 | 491 520 | 5 671 | 63 % |
| 4096 | 67.2-67.4 | 60.7-61.0 | 16.4 | 17.2-17.4 | 49.1-49.3 | 0.0 | 798 720 | 5 327 | 59 % |
| 8192 | 104.3-104.5 | 78.4-78.5 | 12.7 | 26.7-27.1 | 76.4-76.8 | 0.0 | 1 290 240 | 5 384-5 403 | 60 % |

* **Below 1024 the file wait is the wall** (5.9 s of a 17.9 s prompt at 256 with the slab deliberately off; 9.0 s of
  30.2 s at 1024 in S8), which is why the slab is a loss at 256 and a win at 1024.
* **From 1024 up the slab removes the wall entirely** (file wait 0.0 s at every size) and the remaining split is the
  expert stream at 72-76 % and the dense half at 24-26 %; the dense *share* is flat while its absolute time grows with
  the token count (6.9 -> 27.0 s), so the dense half is what eventually binds, not the file tier.
* **The batch still pays, but with collapsing returns:** tokens/s 14.2 -> 34.9 -> 45.7 -> 60.9 -> 78.5 for
  256 -> 8192 (32x tokens, 5.5x throughput); per-token cost halves from 256 to 1024 (2.4x), then only 1.29-1.33x per
  doubling. The expert half is 1.5x per 2x tokens (the ~100 GB slab sweep amortizes), the dense half is ~1.6x.
* **The routed/file-tier ratio stops rising at ~60-63 % from 2048 on** (63 % at 2048, 59 % at 4096, 60 % at 8192): at
  4096+ `enc-skip` cuts the first 1536/5632 tokens out of the deep layers, so the routed expert set shrinks while the
  slab still sweeps every layer's whole file tier (~100 GB). The prefetch's economics therefore *do not* improve with B
  past 2048 the way the S7 curve suggested; they plateau. That is another reason a pure width sweep would not have paid
  much beyond 2048 even if the determinism gate had passed.

(The `pcie` counter counts (token, expert) applications, the `file` counter counts file-tier blob reads; the routed
share is `file / 8977`, the tier's whole file-expert count. That formula reproduces the S7/S8 route-log numbers —
37 % at 256 and 63 % at 2048 — to the point.)

## 6. The preflight (new), and the size-literal audit

`ds41_generate.cpp`, before the one-shot prompt's first allocation:

```
preflight: prompt batch B=8192 needs 11.90 GiB VRAM (7.60 state+parts, 0.97 span graph for ctx 12288),
           18.96 GiB free (margin 1.00) -> fits up to B=14719
preflight: RAM needs 72.00 GiB (arena 60.0 + logits 0.00 + 12.0), 121.87 GiB available
```

It is the serve path's calibrated budget (s13 run9/run13/run14 + run20), evaluated for the *one-shot* path too:

```
need(B) = max(4.3 GiB + 0.95 MiB * B,  2.5 GiB + 0.78 MiB * B + span_graph(ctx_end, S))
```

4.3 GiB = the tier's two stage halves + the pass-graph buffer; 0.95 MiB a column = state 0.32 + graph 0.17 + part
buffers 0.41; the span graph grows with the *context*, not with B. Both lines are evaluated, the smaller fit wins, and
the check runs against `min` free VRAM over the pipeline's stages. RAM is checked as arena + the batch's own logits
(`--ppl`/`--dump-logits` keep B x vocab floats) + a 12 GiB floor.

If it does not fit, the engine **refuses with the numbers** instead of allocating (exit 2, before the prompt):

```
$ ./run.sh refuse 8192 16 --vram-margin 18
ds41_generate: REFUSED: a prompt batch of 8192 does not fit this box (VRAM 18.98 GiB free with a 18.00 GiB margin
allows B<=512; RAM 121.87 GiB available, 72.00 GiB needed). Re-run with --prompt-batch <= 512, or set
STRATA_PB_CLAMP=1 to let the engine halve the batch itself (exact answer, slower: the layer-major sweep repeats per
batch).
WALL=6s EXIT=2
```

`STRATA_PB_CLAMP=1` keeps s13's "halve it and carry on" (the serve path's behaviour, unchanged); `STRATA_PB_PREFLIGHT=0`
skips the check.

Audit for the extensibility rule: no 1024/2048/4096 appears in sizing anywhere on this path. The prompt batch is
`min(--prompt-batch, prompt length)` and everything downstream is indexed off that B (state columns, graph, part
buffers, the preflight). The two row thresholds that remain are *policies compared against each call's own `n`*, not
size assertions: the prefetch width gate (§2.2, now default-on at >= 1024 rows with the measured curve) and
`STRATA_PRESTAGE_MIN` (default 4096, the lazy next-layer prestage — exercised at 4096/8192, no NaNs, no refusal, no
wedge). The engine's own `B > prompt_batch` / `ctx_max % 256` checks are shape-validity checks, not caps.

## 7. Limits and the next experiments (the sweep is on hold until §3 is fixed)

1. The nondeterminism is **not localized to a kernel yet**. The cheapest next bisect, in order, each ~2 runs at 2048
   (the limit is that a 50/50 flip needs N >= 6-8 runs per arm to separate arms; `a5787d7e` appearing 3 times in a row
   under one arm is not evidence, we saw 3-in-a-row then a different value):
   * `STRATA_WAVE_THREADS=1 STRATA_WAVE_PARTS=1` (the on-demand file wave reader is the only path that both the slab
     and the wide batch change) — if the flip vanishes, the wave reader's bounce slots/DMA overlap is the suspect;
   * `GGML_SYCL_ENABLE_MKL_FA=0` and `GGML_SYCL_FA_ONEDNN=0` (flash attention is the densest shared kernel);
   * `--no-enc-skip` at 4096/8192 (does the cut path add a second flip source?);
   * a `STRATA_LAST_ROW_DBG`-style dump of layer 4's `s0` row (the two boundary scores) would settle near-tie vs
     exact-tie in one run — it does not exist yet.
2. **The 2048-and-up throughput numbers in §5 are diagnostics, not a measurement.** They are reproducible within a
   single arm (the two 8192 runs agreed to 0.1 s) so they can guide work, but the gate that the task requires does not
   pass, and two runs of the same size can differ by a routing flip.
3. The prefetch's plateau (§5) says the next prefill lever is **not** more batch width for the slab: the routed share
   saturates at ~60 %, so the slab already reads ~1.6x the bytes it covers. A slab that staged *only the routed*
   experts (once the routing is known — i.e. the current on-demand reader, made wider) would close that gap; the
   numbers say the current width rule is at its measured optimum.
4. `--prefill-chunk 8` and `--prompt-batch 0` remain forbidden, the DNN build stays off, and no run here was killed —
   the watchdog floor was 24 GiB MemAvailable and the low-water mark across every run was 115.5 GiB.

## 8. Reproduce

```bash
BIN=/home/dyy/deepseek-work/s4/step2/ds41_generate_gpuexp   # shipped defaults, md5 e86e83ba…
cd /home/dyy/deepseek-work/s4/step9
./run.sh f_1024 1024 16 --dump-last f_1024.bin                 # 787519ba  (slab on + async on)
./run.sh f_256   256  8  --dump-last f_256.bin                  # fce1788f  (slab gated off, async on)
./run.sh f_2048  2048 8  --dump-last f_2048.bin                 # the gate that fails: run it 4x
./run.sh g_4096  4096 16 --dump-last g_4096.bin  # CTX=8192
./run.sh g_8192  8192 16 --dump-last g_8192.bin  # CTX=12288
./run.sh f_refuse 8192 16 --vram-margin 18                      # the refusal, exit 2
# the pre-flip arm (what TASK 2 asked for): the S8 binary, gate off + async off
BIN=/home/dyy/deepseek-work/s4/step9/ds41_s8_ref ./run.sh det2048_fix_a 2048 8 --dump-last a.bin
```
