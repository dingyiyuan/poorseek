# S4 step 5 — chunked prefill on the VRAM expert cache: 3.1x, and the resident-expert path is not the staged path

2026-10-24. Follows [step 4](xolotl-sycl-s4-step4-pcie-dma-fix-report.md). Goal: the chunked-prefill bottleneck
(2.3 tok/s vs 10.8 decode) was **structural**: with `--prefill-chunk 8` the engine defers the VRAM expert cache, so
every prompt token's every expert application is a PCIe copy or a file read. This step makes the cache open *before*
the prompt and measures both ways — and, while validating the result the way
[s3b](xolotl-sycl-s3b-topk-report.md) requires, found that the VRAM-resident expert path disagrees with the CPU
reference by 62 % at layer 0 with identical inputs and verified-identical weights (§5). That is a pre-existing defect
of the *decode* path, not of this change, and it is why the switch is opt-in.

TL;DR, same command shape (48 prompt ids, `-n 16`, `--ctx 2048 --slots 1200 --prompt-batch 0 --prefill-chunk 8`):

| | before (deferred) | after (`STRATA_PREFILL_CACHE=1`) |
|---|---|---|
| **prefill** | **2.3-2.6 tok/s** (18.7-20.7 s) | **7.7-8.0 tok/s** (6.0-6.3 s) = **3.1x** |
| prefill split | dense 4.6, experts **14.0-16.1 s** (file wait 4.9-5.6) | dense 4.3-4.5, experts **1.6-1.7 s** (file wait 0.3) |
| prefill counters | **0 hit** / 11520 pcie / 3267 file | 10944 hit / **576 pcie / 140 file** |
| decode (16 tok) | 10.36-10.42 tok/s, 98.5 % hits | 9.61-10.49 tok/s, 98.4 % hits |
| decode (48 tok) | 10.93 tok/s, 98.9 % hits | 10.73 tok/s, 98.4 % hits |
| free VRAM during prefill | 15.8 GiB | 5.7 GiB (1200 slots) / 1.7 GiB (2000 slots) |
| prompt logits vs CPU (8-token prompt) | L0 experts **0.44 %** off, last row max\|d\| 3.69 | **62 %** off, max\|d\| 12.13 (§5) |

Prefill is now **dense-bound** (4.3-4.5 of 6.0 s); the expert half is no longer the bottleneck.

## 1. Why the cache is deferred — the code

`tools/v41/engine/ds41_generate.cpp:685`:

```cpp
// defer the VRAM cache: the prompt's chunk + graph buffers (~4 GiB) use that VRAM first, then the cache opens into
// it (measured) - the old fixed --vram-margin had to cover prefill and starved decode of slots.  --serve does this
// per request: a multi-token prompt closes the cache (release_cache), the decode after it re-seeds (~0.6 s)
const bool defer_vram = !mc.cpu_only && (a.prefill_chunk > 1 || a.prompt_batch > 1);
```

and the seed it selects (`:699`): `defer_vram ? build_arena_ranked(sub, err) : seed_from_ranked(sub, err)` — the
deferred branch builds only the **RAM arena** and never opens the cache (`Ds41MoeTier::build_arena_ranked`,
`ds41_moe.cpp:2298`, does not touch `gp.cache`). The cache then opens after the prompt (`:3124`):

```cpp
if (defer_vram && dm_tail == 0 && !spec) {   // the prompt is done: its buffers go back, the VRAM expert cache opens into the free space
```

So the answer to (a) is **(a), a VRAM budget decision** — not correctness, not a wrong default. The deciding
condition is the two flags (`prefill_chunk > 1 || prompt_batch > 1`); the comment says why: the prompt's chunk +
graph buffers were measured at ~4 GiB and a fixed `--vram-margin` big enough for prefill then "starved decode of
slots". It is a real budget trade, but on *this* card the trade is unnecessary — see §2, where the prompt's transient
is 2.70 GiB + up to 3.0 GiB of prestage, and a 1200-slot cache leaves 5.71 GiB.

There is **no flag** for "seed before the prompt" on the one-shot chunked path. The nearest pre-existing switches:
`--prefill-chunk 1` / `--prompt-batch 1` make `defer_vram` false but change the prefill algorithm too (one token per
`run()` pass); `--serve` does have one — `STRATA_SERVE_SEED_AT_READY` (default on, `:1613`) seeds the cache at
`READY`, before any request. This step adds the one-shot equivalent.

## 2. The VRAM arithmetic (measured, not assumed)

All figures from `STRATA_VRAM_TRACE=1` on this run (ggml/SYCL reports 31.89 GiB usable of the 34242 MB part):

| quantity | GiB | how measured |
|---|---|---|
| usable device memory | **31.89** | `vram trace: … of 31.89 GiB` |
| free after load, cache closed | **21.55** | `after the load seed: free 21.55` (deferred run) — so dense + KV + state + engram = 10.34 |
| dense weights (device-resident) | 8.137 | WRESID probe, step 3 |
| **1200-slot cache arena** | **10.14** | 21.55 − 11.41 (load seed) *and* 20.84 − 10.70 (deferred seed) — both give 10.14 |
| prompt transient (dense pass + chunk buffers + MMQ scratch) | **2.70** | 11.41 − 5.71 with the 3.00 prestage accounted (`chunk done` trace) |
| layer prestage region | 3.00 | `chunk prestage: 3.00 GiB device region` (self-capped: `p_cap = fr > want+keep ? want : fr-keep`, keep 3.0) |

Correction to step 4 (and to the task's starting numbers): the `19.84 GiB` in
`VRAM after the prompt: 20.84 GiB free -> 19.84 GiB of expert slots` is **`seed_gib` = free − `--vram-margin`**, i.e.
the budget the *auto* slot computation would use; it is not the size of the arena. With an explicit `--slots 1200`
`set_slot_gib(0.0)` discards it, and the arena the engine actually allocates is **10.14 GiB** (1200 class-sized slots,
7.7-15.2 MB each, 8.65 MiB on average).

So with the cache open before the prompt:

* 1200 slots: free at feed start 11.41 → prestage 3.00 → transient 2.70 → **5.71 GiB free during the whole prompt**
  (flat over all six chunks).
* 2000 slots: cache 17.17, free 4.38 → after the transient 1.69 GiB, the prestage self-disables (free < keep 3.0).

**Maximum safe `--slots` with the cache open during prefill.** Constraint:
`21.55 − N·0.00838 − 2.70 ≥ reserve`, with the measured 8.38 MB/slot (linear to 2000 — the ranked tail has the same
class mix):

| reserve | max N | note |
|---|---|---|
| `--vram-margin 1.0` | **≈ 2130** | free during prefill ≈ 1.0 GiB |
| keep 3.0 (prestage alive) | ≈ 1890 | above this the prestage turns itself off |
| 0 | ≈ 2250 | not safe, driver headroom |

Tested: **1200** (5.71 GiB spare, the recommended, unchanged decode shape) and **2000** (1.69 GiB spare, works,
prefill 8.0 tok/s, decode 98.2 % hits). 1200 is not near any limit; the budget does **not** forbid this change — no
fp8 KV, no smaller slots needed.

## 3. The change

`ds41_generate.cpp`, one predicate plus a trace mark; `ds41_moe.{hpp,cpp}`, a `set_slots()` setter and the
`STRATA_SEED_VERIFY` diagnostic:

```cpp
// STRATA_PREFILL_CACHE=1 (one-shot, explicit --slots only): seed the VRAM expert cache BEFORE the prompt instead.
// A deferred chunked prefill runs with an EMPTY cache: every expert application of every prompt token is a PCIe
// copy (12.83 MiB) or a file-tier read - measured 432 ms/token, 0 VRAM hits, 11520 pcie + 3267 file for 48 tokens.
const bool prefill_cache = !a.serve && std::getenv("STRATA_PREFILL_CACHE") && ... && a.slots > 0 && !mc.cpu_only;
const bool defer_vram = !mc.cpu_only && !prefill_cache && (a.prefill_chunk > 1 || a.prompt_batch > 1);
```

* `STRATA_PREFILL_CACHE=1` — the switch. It only changes *when* the seed happens (the same `seed_from_ranked`, the
  same `--slots`); the prompt then hits VRAM like decode does.
* `STRATA_PREFILL_SLOTS=N` — cap the number of slots (default: `--slots`) for a card where `--slots` does not leave
  room for the prompt.
* Requires an explicit `--slots`: without one the pipeline sizes the auto budget (`slot_gib`) from the free card
  *before* the prompt, which would take everything; the engine warns and keeps the deferred seed.
* `--serve` is excluded (it already seeds at `READY` and has its own release/re-seed policy per request).
* Diagnostics, both off by default: `STRATA_VRAM_TRACE=1` now also marks the feed start, every chunk end, and the
  seed in/out; `STRATA_SEED_VERIFY=N` reads N seeded slots back and compares them with the arena.

Default is unchanged behaviour (`STRATA_PREFILL_CACHE` off): the deferred seed stays the default because of §5.

## 4. Measurements

Command (unchanged from step 4, `s4/step5/run.sh`): `-m V41-GSQ-1017.gguf --backend cuda --experts gpu --engram
engram-layers.1:engram-layers.14 --rank rank.bin --ids 100,200,…,4800 -n 16 --ctx 2048 --temp 0 --slots 1200
--prompt-batch 0 --prefill-chunk 8`, `STRATA_VRAM_TRACE=1` both ways.

| run | prompt | dense | experts | file wait | hit/pcie/file | decode |
|---|---|---|---|---|---|---|
| `f-base` | 18.72 s = **2.6 tok/s** | 4.7 | 14.0 | 5.6 | 0 / 11520 / 3267 | 10.36 tok/s, 98.5 %, 47/25 |
| `b2`, `f-cache` | 5.97-6.40 s = **7.5-8.0 tok/s** | 4.3-4.5 | 1.6-1.7 | 0.3 | 10944 / 576 / 140 | 9.61-10.49 tok/s, 98.4 %, 55/46 |
| `b5` (2000 slots) | 6.00 s = 8.0 tok/s | 4.3 | 1.7 | 0.2 | 11098 / 422 / 110 | 8.61 tok/s (n=16), 98.2 % |
| `a2`/`b4` (-n 48) | 20.69 → 6.35 s | 4.6 → 4.5 | 16.1 → 1.9 | 4.9 → 0.3 | 0/11520/3267 → 10944/576/140 | 10.93 → 10.73 tok/s |

* **Decode is not affected** beyond run-to-run noise: 10.93 vs 10.73 tok/s over 48 tokens (−1.8 %, inside the
  ±5 % spread of two identical cache runs: 9.61 and 10.49 at n=16). The VRAM hit rate moves 98.9 → 98.4 % and
  file-tier misses 67 → 144 over 48 tokens, because the prompt's own misses now pass through the cache's LRU before
  decode starts (the deferred path seeds a fresh ranked set right before decoding). If decode quality mattered more
  than 1.3 s, the deferred seed's `release_cache()` + re-seed after the prompt would restore it exactly.
* Seed cost moves from after the prompt (1.3 s) to load (4.7 s vs 3.5 s: the arena build + 1200 fills), i.e. out of
  the prompt clock either way.
* 3 runs of each shape, no wedge, `sycl-ls` fine after each. Operational note: back-to-back processes need ~10 s
  between them — one seed failed with "only 4.88 GiB free" while the previous process's VRAM was still being
  reclaimed, and passed on a retry.

## 5. Correctness — the chaos-envelope test fails, and the reason predates this change

The criterion from s3b is the CPU-vs-CPU perturbation envelope, not token ids. Setup: 8-token prompt
(`--ids 100,…,800`), per-layer dumps (`STRATA_LAYER_DUMP`) and all-row logits (`--dump-logits`), three backends:

| run | what | L0 routed-sum vs CPU | per-layer fn mean rel (0-39) | last-row logits vs CPU |
|---|---|---|---|---|
| `c8-cpu` | `--backend cpu --experts cpu` (reference) | — | — | — |
| `c8-stream` | deferred, cache-less chunk path | **4.4e-3** | **0.202** | max 3.69, mean 0.52 |
| `c8-cache` | `STRATA_PREFILL_CACHE=1` | **4.8e-1** | **0.726** | max 12.13, mean 1.54 |
| `c8-cache-nopre` | + `STRATA_PRESTAGE_GIB=0` | 4.8e-1 (identical) | — | max 12.13 (identical) |
| `c8-decpath` | `--prefill-chunk 1` (decode `run()`, cache) | **6.2e-1** | — | max 11.67, mean 1.52 |

Layer 0 is the controlled comparison: its input `fn` is **bit-identical** between the streamed and cached runs
(rel 0.000e+00), the routed ids are identical (the router is a deterministic function of `fn`, no route bias is
set when `route_bias`/`route_bias_file` are 0), and the expert blobs are **verified byte-identical**::

```
seed verify: im.blob 15206400 B, class blobs 7741440..15206400 B, cache 1200 slots, per-slot offsets yes
seed verify: 8 slots checked, 0 differ from the arena
```

Same input, same weights, same kernel symbol (`strata::kernels::native_expert_grouped` for both paths — with
`--no-chunk-mmq` the streamed prompt is *bit-identical*, so MMQ is not built here and both paths run the MMVQ
kernel), and the resident path's expert sum is **62 % away from the CPU's, cosine 0.12, at 0.48x the magnitude**.
That is not chaos: the streamed path's seed-level error is 0.44 %, and a 1e-4 state perturbation (this model's own
chaos probe) gives 0.079 at L0 — the cached path is 6-8x coarser than the chaos the model generates from a 1e-4
seed. Eliminated as causes: the seed fill (verified), the prestage (on/off identical), the LRU/admission
(`run_chunk` does no admission), patchup/prefetch (`prefetch … used` counters unchanged), and the CPU reference
itself (its own per-layer curve is inside the step-3 band: mean 0.202 vs 0.194-0.367).

What remains is the resident launch's per-expert pointer table — `gp.c_ptr[] = cache->device_slot(slot_of(layer,e))`
(one scattered pointer per expert) versus the staged path's evenly strided `c_stage + i·SL` pointers for the same
kernel call. Consequence, independent of this step: **every decode takes the resident path** (`98.9 % VRAM hits`),
and `--prefill-chunk 1` puts the *prompt* on it too, so the engine's decode numerics have been on a path that
disagrees with the CPU reference by 62 % at layer 0 all along. The parent work's decode *speed* numbers stand; its
decode *quality* does not, and token ids could not have shown it.

Therefore: the prefill speedup is real and the budget allows it, but it moves the prompt onto the same path — the
prompt's logits leave the envelope. `STRATA_PREFILL_CACHE` stays **opt-in** until the resident launch is fixed; at
that point it should become the default for the one-shot chunked path (the win costs nothing but the 5.7 GiB that
are already there).

Next experiment for the resident path (cheap, decisive): copy **one** verified slot back into a staging buffer and
run the same kernel on the two pointers for one expert/one token; then dump `gp.c_ptr[]` for the resident launch of
layer 0 against `slot_of()`. That separates "the kernel does not read the pointer it is given" from "the pointer
table is not what `slot_of()` says".

## 6. Reproduce

```
/home/dyy/deepseek-work/s4/step2/build_engine_tu.sh ds41_generate && /home/dyy/deepseek-work/s4/step2/build_engine_tu.sh ds41_moe
/home/dyy/deepseek-work/s4/step2/link_gpuexp.sh
IDS=$(cat /home/dyy/deepseek-work/s4/step4/ids48.txt)
/home/dyy/deepseek-work/s4/step5/run.sh f-base  $IDS 16 8                                   # before
STRATA_PREFILL_CACHE=1 /home/dyy/deepseek-work/s4/step5/run.sh f-cache $IDS 16 8           # after
STRATA_PREFILL_CACHE=1 STRATA_PREFILL_SLOTS=2000 STRATA_VRAM_TRACE=1 .../run.sh ... 16 8   # the slot bound
STRATA_PREFILL_CACHE=1 STRATA_SEED_VERIFY=8 .../run.sh v8-cache 100,200,300,400,500,600,700,800 1 8
/home/dyy/deepseek-work/s4/step5/runcpu.sh c8-cpu 100,200,300,400,500,600,700,800 1 --dump-logits ...
```

Logs: `s4/step5/{f-base,f-cache,f-cache-notrace,b1-prefillcache1080,b2-prefillcache1200,b5-prefillcache2000,
a2-base-n48,b4-prefillcache-n48,a-base-trace,c8-cpu,c8-stream,c8-cache,c8-cache-nopre2,c8-decpath,*.log}`;
dumps in `/tmp/s5/` (volatile). Binary under test: `s4/step2/ds41_generate_gpuexp`, sha256 in
`s4/step5/binary.sha256` — note the tree has a concurrent S6 profiler edit in `ds41_moe.cpp` (not part of this
patch, env-gated `MIMO_CHUNK_PROF`), so relinking after that lands rebuilds it too.
