# S4 step 4 — the expert PCIe abort: a legal VRAM slot the shim had never seen

2026-10-24. Stage S4 of [xolotl-sycl-port-plan.md](../../llm-for-glm-flash/docs/xolotl-sycl-port-plan.md), after
[step 3](xolotl-sycl-s4-step3-dense-dprof-report.md). Goal: fix the `ds4_moe: pcie dma: invalid argument` abort in the
expert PCIe path (reported in step 3 §10), then validate it on the card.

**Root cause**: the CUDA-runtime shim decided "is this pointer device-reachable?" by consulting only the interval map
of allocations it had intercepted itself. The VRAM expert cache arena is allocated *outside* the shim
(`ExpertCache::open_sized` → `strata::malloc_device_guarded` → `sycl::malloc_device`), so every LRU admission that
DMAs an incoming expert **directly into a victim slot** was classified `PtrKind::None` and refused with
`cudaErrorInvalidValue`, which the engine reports as `ds4_moe: pcie dma: invalid argument` and aborts on. The fix makes
the classification authoritative — a map miss asks `sycl::get_pointer_type` and caches the answer — without weakening
the rule for pageable host memory. A refused copy now also prints the pointer, its kind and its provenance instead of
aborting anonymously.

## 0. TL;DR

| | before | after |
|---|---|---|
| 3-token command (`--ids 100,200,300`) | works, id **201**, 708 hit / 3 cpu / **9 pcie** / 1 file | identical: id **201**, 708 / 3 / **9** / 1, prompt 3.75 s |
| 8-token command (`--ids 100..800`) | `pcie dma: invalid argument` → SIGABRT (exit 134) at prompt **4/8** | completes, prompt 4.36 s, **1843 hit / 13 cpu / 64 pcie / 27 file**, token 8993 |
| steady state, decode | never measured (the run aborted) | **9.6-10.8 tok/s = 93-105 ms/token** (48-token decode: 10.76 tok/s), 98.5-98.9 % VRAM hits |
| steady state, prefill | never measured | 2.3 tok/s = 432 ms/token **without a VRAM cache** (see §5: `defer_cache` opens it after the prompt) |
| the ~146 ms/token figure | — | **refuted**: direct measurement is 93-105 ms/token |

## 1. The failure, named

A diagnostic build (`DS41_SHIM_CLASSIFY=map`, which restores the old map-only classification on the *fixed* source)
reproduces the report's abort exactly — same exit code, same message, same position in the prompt:

```
prompt 4/8
ds41 shim: cudaMemcpyAsync HostToDevice refused (invalid argument):
  dst=0xffffd55a2f078000 kind=none shim-map=no sycl-resolved=no sycl-kind=usm_device |
  src=0x7958cf296000 kind=none shim-map=no sycl-resolved=no sycl-kind=unknown |
  bytes=13455360
ds4_moe: pcie dma: invalid argument
EXIT=134
```

* `dst = 0xffffd55a2f078000`, **`sycl-kind=usm_device`**: SYCL says it is a valid USM device allocation. It is not in
  the shim's map, because the shim never allocated it.
* `src = 0x7958cf296000`, `sycl-kind=unknown`: pageable memory inside the engine's 60 GiB `mmap`ed expert arena. This
  is *correctly* not device-reachable — it is what the staged path is for.
* `bytes = 13455360` (12.83 MiB): one expert blob of this class.

**Which pointer it is.** The destination is a **slot of the VRAM expert-cache arena**, written by the LRU admission
path in `gpu_run`:

```cpp
// ds41_moe.cpp:3220-3222
} else {   // the DMA lands in the victim slot: the kernel below reads it there, and it stays
    to = vs;                                                                          // vs = cache->device_slot(...)
    ck(cudaMemcpyAsync(to, p, (size_t) BL, cudaMemcpyHostToDevice, cs), "pcie dma");   // <-- 0xffffd55a2f078000
```

with `vs = gp.cache->device_slot(gp.cache->slot_of(layer, v))` and `device_slot(s) = base_ + s * blob_`. That arena's
`base_` comes from `ExpertCache::open_sized` → `strata::malloc_device_guarded` → `sycl::malloc_device`
(`sycl/src/core/expert_cache.cpp:663`), i.e. a raw SYCL USM allocation that never passes through
`cudaMalloc`, so `Runtime::allocs` never learned about it. The chunked twin of the same copy is
`ds41_moe.cpp:4155` (`gpu_run_chunk`), which `--prefill-chunk 8` exercises too.

**Why the failure needs more slots to appear** (and why 3 tokens pass with 9 PCIe copies): the admission filter
(`admit_ok`, `ds41_moe.cpp:3153`) only lets an expert take a slot if it *also* missed in one of the layer's previous
`kLruWin = 4` calls, so a layer's first call admits nothing; and the direct-into-victim form additionally needs a
victim that still has an arena copy. The 3-token prompt's 9 PCIe copies all went through `d_stage` (which the shim
allocated, so it is in the map); by token 4 of the 8-token prompt there had been 64 admissions, and one of them
targeted a victim slot.

**Two independent attributions.** (a) On the *unfixed* classification, the same 8-token command with
`--no-vram-lru` — which removes every write into `gp.cache->device_slot(...)` from the PCIe path — completes 8/8,
exit 0. (b) On the fixed build with `DS41_SHIM_DBG=1`, the **first** pointer the SYCL fallback ever classifies is
`0xffffd55a2f078000` — the identical address — followed by 5 more in the same arena; the run completes. Six distinct
external device pointers, spanning 4.43 GiB of the 19.84 GiB slot arena.

## 2. Why the map-only classification was wrong, and what must not change

`Runtime::allocs` is `std::map<base, {bytes, kind}>` filled by `remember()` on the allocations the shim itself
intercepts. It is a record of *the shim's own* allocations, not of the process's USM. The engine receives device
pointers from at least three other sources — the expert cache arena (above), ggml-sycl's backend buffers, and the
dpct-migrated TUs (`sycl/src/**`, which link the real SYCL allocator) — and any of those as a copy argument used to
be `cudaErrorInvalidValue`.

The tempting non-fix is "assume unknown pointers are device and try anyway". It is wrong in the direction that costs
the most here: the *sources* of these copies are pageable arena pages, and pageable memory handed to a SYCL command
is a runtime-level failure — a wedge, not an error. So the rule stays: **pageable memory is staged, USM is direct**,
and the only question the shim has to answer better is "which is it".

`sycl::get_pointer_type(p, ctx)` answers it authoritatively. Verified on this stack before relying on it (probe:
`s4/step4/ptrtype_probe.cpp`): an allocation made in one `sycl::context` resolves correctly when asked from a
*second* context on the same device, an interior pointer (`base + 4096`) resolves as well, and a pageable `malloc`
pointer returns `unknown` — which is exactly the distinction the fix needs, given that the cache arena, ggml and the
shim each own their own `sycl::context` here.

## 3. The fix (`tools/v41/engine/sycl_compat/ds41_sycl_compat.cpp`)

1. **`classify()` is authoritative, with the map as a cache.** The map lookup is unchanged and first (one
   `std::map::upper_bound`, O(log n)). On a miss, `resolve_with_sycl()` asks `sycl::get_pointer_type` (`usm_device` →
   `Device`, `usm_host` → `HostUsm`, `usm_shared` → new `SharedUsm`, everything else → `None`), and a non-`None`
   answer is written back with `remember_external()` as an **exact-address** entry (`bytes = 0`, `external = true`) —
   SYCL reports the kind but not the extent, so each distinct pointer is resolved once and then hits the map. The
   hot path is unchanged; the interesting pointers stay visible in `Runtime::allocs`, tagged as external.
   `usm_reachable()` now also accepts `SharedUsm` (same address on host and device).
2. **A copy that is refused explains itself.** `do_copy` (all three directions) and `cudaMemset*` now call
   `refuse_copy()` / `refuse_memset()`, which print one line **unconditionally** (this is the error path; nothing on
   the success path pays for it): both addresses, the byte count, the kind each classified as, whether the shim knew
   it, and the raw `sycl::get_pointer_type` answer. `DS41_SHIM_DBG=1` additionally traces every classification miss
   ("not in the shim map; SYCL says usm_device -> device, cached") and every staged copy. `DS41_SHIM_CLASSIFY=map`
   restores the old behaviour for A/B.
3. **Nothing was weakened.** A pageable pointer still classifies `None` and still goes through `copy_in`/`copy_out`;
   a backend that cannot answer (or throws) is treated as "not USM" and staged. No path assumes device.

Init/teardown semantics are untouched: `free_device`/`free_host` still match exact base *and* kind, and external
entries carry `bytes = 0`, so they contribute nothing to the device/host byte totals `cudaMemGetInfo` falls back on.
The shim TU compiles clean (`-Wno-*` flags as before, 0 errors, 0 warnings in the TU), the link is unchanged
(`LINK_RC=0`, 105 753 912 B).

## 4. Validation

Same command as always, from `models/dsv41-gsq`, `source /opt/intel/oneapi/setvars.sh --force`,
`ONEAPI_DEVICE_SELECTOR=level_zero:0`, never two model processes, never compiling while one runs, and **no
`SYCL_CACHE_*`** (see §6).

| run | command | result |
|---|---|---|
| 3-token | `--ids 100,200,300 -n 1 --prefill-chunk 1` | **201**, 3.75 s, 708 hit / 3 cpu / 9 pcie / 1 file, dense 3.6 s, experts 0.2 s (file 0.0 s) |
| 8-token | `--ids 100,…,800 -n 1 --prefill-chunk 1` | **completes**, 4.36 s, 1843 hit / 13 cpu / 64 pcie / 27 file, dense 3.9 s, experts 0.4 s (file 0.1 s), token 8993 |
| 8-token, old classification (`DS41_SHIM_CLASSIFY=map`) | same | **abort at prompt 4/8**, exit 134, `pcie dma: invalid argument` (§1) |
| 8-token, old classification + `--no-vram-lru` | same | **completes 8/8**, exit 0 |

The 3-token id and all four counters are identical to the pre-fix run (step 3's `dprof4`: 708/3/9/1, id 201), so the
fix changes classification, not arithmetic. The 8-token prompt's 64 PCIe copies (8 per token, against 3 per token for
the 3-token run) are the admissions that used to abort.

## 5. Steady state: the 146 ms/token figure is refuted

Measurement run: 48 prompt ids (`100,200,…,4800`), `-n 16`, `--prefill-chunk 8`, `--slots 1200`, ctx 2048, temp 0 —
i.e. long enough that the per-process cold start (3 first-run dense graphs ≈ 3.6 s, the 1200-slot seed ≈ 1.3 s) is a
small part of the run. Two independent runs plus one with `DS41_DPROF=1` for the split, and a 48-token decode.

| phase | time | rate | split (engine + dprof) |
|---|---|---|---|
| prompt, 48 tokens | 20.4-21.1 s | **2.3 tok/s = 432 ms/token** | dense 4.5-4.7 s, experts 15.7-16.3 s (**file wait 4.9-5.1 s**), logits 0.02 s |
| decode, 16 tokens | 1.44-1.57 s | **9.6-10.4 tok/s = 96-105 ms/token** | dense **823 ms/15 passes = 55 ms/token**; logits 19 ms; expert residual ≈ 45 ms/token; file wait 94-107 ms total (6 ms/token) |
| decode, 48 tokens | 4.37 s | **10.8 tok/s = 93 ms/token** | 98.9 % VRAM hits, 108 pcie / 67 file over 48 tokens |

So the steady-state decode is **93-105 ms/token, not ~146 ms/token** — the earlier figure is ~1.5x too slow and is
**refuted** by direct measurement (twice, on two different decode lengths). The budget that produces 146 is
`~78 ms dense + expert + head`, and the dense half is the number that moved: in the decode segment 40 attention graphs
cost 1.03-1.16 ms each (17.5 ms for layer 0, ~15.4 ms for the other 39 over 15 passes) rather than the 2.0 ms/layer
measured inside the *prompt* in step 3 — 55 ms/token, not 78. That is not a shim effect; the prompt segment pays
per-pass upload/prepare work the decode pass does not.

The prefill rate is the number that looks bad, and it is **not** this bug: with a chunked prefill (`--prefill-chunk 8`)
the engine defers the VRAM cache (`defer_cache`), prints `tier: 0 experts in VRAM`, runs the whole prompt from the PCIe
budget + file tier (11520 pcie / 3267 file reads) and only then opens the 19.84 GiB, 1200-slot cache
(`VRAM after the prompt: … 1200 experts in VRAM (1.3 s)`). The prompt is therefore cache-less by construction, and
5 s of its 16 s expert time is file reads. With `--prefill-chunk 1` the cache is seeded at init instead (the 3/8-token
runs above: `tier: 1200 experts in VRAM … seeded in 4.7 s`). Both are engine policy, unchanged by this fix.

## 6. Wedge risk: not removed, but this failure mode no longer exists

* **What this fix removes**: a legal DMA being turned into an abort. That path is gone — the destination is
  classified correctly, and if something *else* is ever misclassified the shim now prints the pointer, the byte count,
  the classification and the raw SYCL answer before returning the error the engine aborts on.
* **What it does not address**: the two wedges in today's journal are runtime-level, not classification-level — the
  bad expert launch in step 2, and the `SYCL_CACHE_PERSISTENT=1`/`SYCL_CACHE_DIR` probe (312 s, then an abort inside
  `neo/shared/source/helpers/bindless_heaps_helper.cpp`). Neither was attempted here (the constraint was respected:
  no `SYCL_CACHE_*` in any of the 7 runs), and this change neither fixes nor introduces that class of failure.
* **Observed today**: 7 model runs (2 A/B reproductions, 5 completed runs), all with a 900 s timeout and a
  MemAvailable watchdog (minimum seen 124.7 GiB of 188 GiB), zero wedges, `sycl-ls` still lists the Arc Pro B70 after
  the last run. That is "not triggered", not "gone".

## 7. Reproduce

```
# build (never while a model runs)
/home/dyy/deepseek-work/s4/step2/build_all.sh        # or just the shim TU (3 s) + relink
/home/dyy/deepseek-work/s4/step2/link_gpuexp.sh      # ~54 s
# validate
/home/dyy/deepseek-work/s4/step4/run.sh fixed-3  100,200,300 1 1
/home/dyy/deepseek-work/s4/step4/run.sh fixed-8  100,200,300,400,500,600,700,800 1 1
# A/B: the old map-only classification aborts on the same binary
DS41_SHIM_CLASSIFY=map /home/dyy/deepseek-work/s4/step4/run.sh maponly-8 100,200,300,400,500,600,700,800 1 1
DS41_SHIM_CLASSIFY=map /home/dyy/deepseek-work/s4/step4/run.sh maponly-8-nolru 100,200,300,400,500,600,700,800 1 1 --no-vram-lru
# steady state
IDS=$(python3 -c "print(','.join(str(100*i) for i in range(1,49)))")
/home/dyy/deepseek-work/s4/step4/run.sh steady $IDS 16 8
DS41_DPROF=1 /home/dyy/deepseek-work/s4/step4/run.sh steady-dprof $IDS 16 8
```

Logs: `s4/step4/{maponly-8,maponly-8-nolru,fixed-3,fixed-8,final-3,final-8,final-steady16,final-steady16d,final-steady48}.log`.
Diagnostics (env-gated, documented in `tools/v41/engine/sycl_compat/README.md`): `DS41_SHIM_DBG=1` traces
classification misses and staged copies; `DS41_SHIM_CLASSIFY=map` reproduces the old classification for A/B.
