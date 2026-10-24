# S4 step 2 — the GPU-expert tier links and starts running; the card wedged on the first expert work

2026-10-24. Stage S4 step 2 of [xolotl-sycl-port-plan.md](../../llm-for-glm-flash/docs/xolotl-sycl-port-plan.md), after
[step 1](xolotl-sycl-s4-report.md) ported and validated `native_expert_grouped`. Goal of this step: make the GPU-expert
flavour link, run it on the Arc with the VRAM cache on, prove it correct against `--experts cpu`, and report tok/s.

**The link is done and the tier now initialises and enters its device path.** It does not finish a token: at the first
real expert work the GPU hung, the `xe` driver reset the GT four times and then declared the device *wedged*
(`CRITICAL: Xe has declared device 0000:03:00.0 as wedged ... device wedged, needs recovery`, journal 06:18:38). The
device is gone from Level Zero until it is recovered with root (below). **No A/B correctness result and no tok/s: this
report contains none, because none was measured.** What is new and usable is §1-§4; §5 is the exact state to resume
from.

---

## 0. Summary of the boundary

| step | state |
|---|---|
| 1. SYCL `strata_kernels` archive in the link, stale duplicates resolved | **done** — the GPU-expert flavour links, `LINK_RC=0`, 105 657 760 B |
| 2. the four genuinely missing fork pieces | **done** — `native_expert_layout3`/`supported3`, `weighted_rows_sum`, `decode_rows_sum` (new TU), plus 4 more symbols the link needed (§2.3) |
| 3. run `--experts gpu` with `STRATA_ROUTED_HOST` unset | **reached and partially run**: tier init OK, `experts gpu, 9 classes`, enters `gpu_run_chunk` → device kernels; then the GPU hangs (§3) |
| 4. arena re-derived for this model | **not measured**: in the one run that got to init, `--slots 1200` produced **0 experts in VRAM** (§4.2 — a real finding, not a tuning choice) |
| 5. one-layer A/B (chaos envelope) + tok/s | **not done** — needs a working card |

---

## 1. What the link needed, and what was built

### 1.1 The 28 → 12 → 1 undefined symbols

Step 1's §5.1 was right about the shape and wrong about the count in one direction: the failed link's 28 unique
undefined symbols resolve to **12** once the SYCL tree's own TUs are in the link, and to **1** (`iq_mmvq`, below) after
the core TUs are added. Nothing had to be invented except the four fork pieces.

### 1.2 The archive

`libstrata_kernels_sycl.a`, built from these TUs (`icpx -fsycl -O2 -std=c++20 -fp-model=precise
-fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel`):

| TU | what it supplies | why it, and not `iq_kernels.dp.cpp` |
|---|---|---|
| `iq_kernels_native_expert.dp.cpp` (step 1) | `native_expert_grouped` | the bit-exact port; rebuilt here with the CMake project's flags and re-validated (§1.4) |
| `iq_kernels_fork_expert.dp.cpp` **new** | `native_expert_layout3`, `native_expert_supported3`, `weighted_rows_sum`, `decode_rows_sum` **and** `iq_row_bytes`, `native_expert_supported`, `native_expert_layout`, `native_expert_scratch_bytes` | the fork's four are the genuinely missing ones; the other four are host arithmetic that lives only in `iq_kernels.dp.cpp`, which **does not compile** here (§1.3) |
| `native_mmvq.dp.cpp` | `native_quantize_q8_1`, `native_q8_1_bytes` | real TU, unmodified |
| `elementwise.dp.cpp` | `scatter_rows_f32` | real TU, unmodified |
| `expert_cache.cpp` | `ExpertCache` (+ `evict_from`/`admit_at`, §2.2) | real TU, one patch |
| `pinned.dp.cpp` | `PinnedArena`, `arena_pin_cap_gib` | real TU, unmodified |
| `vmm.cpp` (the port's "no VMM" stub) | `VmmRange`, `vmm_*` | real TU |
| `../src/platform/memory.cpp` | `lock_resident`, `unlock_resident` | host-only TU, compiles as-is |
| `ds41_transcode.dp.cpp` **new** | `ds41_add_f32` | the engine's one CUDA file the GPU flavour needs |
| `iq_mmvq_refuse.dp.cpp` **new** | `iq_mmvq` | see below — refuses loudly, unreachable for this model |

### 1.3 `iq_kernels.dp.cpp` is not buildable with this toolchain, so its four host functions were transcribed

`icpx -fsycl` 2026.2 on `sycl/src/kernels/cuda/iq_kernels.dp.cpp` gives **20 errors**, all of the
`dpct::global_memory` / `get_int_from_table_16` family:

```
iq_kernels.dp.cpp:155:14: error: cannot cast from type 'dpct::global_memory<const uint64_t, 1>' … to 'const sycl::uint2 *'
iq_kernels.dp.cpp:226:55: error: invalid operands to binary expression ('dpct::global_memory<const uint64_t, 1>' and 'int')
```

i.e. the migrated copy is older than the `dpct` the tree ships (`sycl/include/dpct/`): its device-side `iq2xs_grid`
tables are `dpct::global_memory` objects that this `dpct` no longer converts to pointers. That is unrelated to this
step's work, and fixing it is a migration refresh, not a wiring change.

Pulling that TU in anyway would also have brought its **stale** `native_expert_grouped` + `native_grouped_set_v1`
(no `lora`, no Q3_K/PTQ1_0, no `swiglu_limit`), which collide with the ported ones — so step 1's "delete the duplicate"
would have had to be done inside a file that does not compile, unverifiable. **Decision: exclude the TU, transcribe its
four host functions** (`iq_row_bytes`, `native_expert_supported`, `native_expert_layout`, `native_expert_scratch_bytes`)
into `iq_kernels_fork_expert.dp.cpp`, verbatim from `src/kernels/cuda/iq_kernels.cu:2899/2150/2163/2181` — the fork's
copies, which carry the fork's formats (Q2_K 10, MXFP4 39, PTQ1_0 143) and NVFP4's 16-byte tail that the migrated copy
lacks. They are host arithmetic (no kernels), so "same source, same numbers" is the whole requirement. The file's header
says exactly this, and the CMake list now carries the same note.

`decode_rows_sum` accumulates in **double** (`fork_weighted_sum.cuh:29`, emulating `__dadd_rn`/`__dmul_rn` with
`#pragma clang fp contract(off)`). The Arc Pro B70 reports `aspect::fp64` = yes and computes IEEE doubles on device
(verified with a 20-line probe before writing the kernel); the **UHD 770 iGPU does not** (`aspect::fp64` = 0, a
`double` USM allocation throws), so this kernel - and therefore the GPU-expert tier - needs the discrete card, not a
fallback device.

`iq_mmvq` is the one symbol left over from that TU: `native_mmvq.dp.cpp:3786/3796` delegates types 7 and 16/17/18/21/22/29
to it. This model's expert types are 11 (Q3_K) and 42 (Q2_0), both native there, plus 143 (PTQ1_0), which
`native_mmvq_supported()` does not claim — so no caller reaches it. It is stubbed to print what is missing and exit.

### 1.4 The port is still bit-exact under the project's flags

`ne_kernel.o` was rebuilt with `-fsycl-default-sub-group-size=32 -fp-model=precise` (the CMake project's flags, not
step 1's minimal line) and the step-1 parity harness re-run against it:

```
93 cases … ALL CASES BITWISE EQUAL     (exit 0)
```

### 1.5 Link result

```
icpx -fsycl -qmkl -O2 -o ds41_generate_gpuexp <all engine objects except ds41_moe_cpu/ds41_helper/ds41_dense/ds41_sycl_compat>
     <s3 cpu-kernels> obj/ds41_dense.o obj/ds41_sycl_compat.o \
     -Wl,--start-group libstrata_kernels_sycl.a libggml-sycl.a libggml-cpu.a libggml.a libggml-base.a -Wl,--end-group \
     -lpthread -ldl -lm
LINK_RC=0  size=105 657 760
```

Script: `/home/dyy/deepseek-work/s4/step2/link_gpuexp.sh`; objects in `s4/step2/obj/`.

---

## 2. Three of the four blockers were not kernel work at all

Worth recording, because each cost a debugging cycle and none of them is in any report yet.

### 2.1 `dpct` ODR: two `dpct::device_ext` layouts in one binary → SIGSEGV inside `ggml_sycl_init`

Symptom: a segfault in `sycl::queue::get_backend()` called from `ggml_sycl_init()`, i.e. in *ggml's* device enumeration,
before any expert work:

```
#0 queue::get_backend()
#1 ggml_sycl_init()
   0x…fea6: add $0x8,%r13          ; "&device + 8"
   0x…fead: call queue::get_backend
   rdi = 0x7c0e868  (heap)  →  *(rdi) = 3  (garbage impl)
```

Cause: ggml-sycl carries **its own vendored dpct** (`s2/llama.cpp/ggml/src/ggml-sycl/dpct/helper.hpp`) and the sycl
tree carries the IBM `dpct` (`sycl/include/dpct/`). Both define `dpct::device_ext`, `dpct::dev_mgr::instance()` and
their inline members with the same mangled names but **different layouts** — the sycl tree's `device_ext` has an extra
`_ctx` member before its queues, ggml's does not. The link keeps one definition; ggml's code then reads the queue at
the other layout's offset. It only bites once any TU from the kernel archive is actually pulled in, which is why the
CPU-expert binary (which pulls none of them) never showed it.

Fix: compile the kernel archive with **`-Ddpct=strata_dpct`**, so the port's dpct world is a different namespace and
cannot merge with ggml's. Documented in `s4/step2/build_all.sh`. The proper fix is one dpct for both, which is a
build-system change for the port, not this step's.

### 2.2 `ExpertCache::evict_from` / `admit_at` do not exist in the SYCL copy

`include/strata/core/expert_cache.hpp:137/154` declares them (`Xolotl s6`, which `ds41_moe.cpp` calls) but
`sycl/src/core/expert_cache.cpp` is an older revision of the file (1059 vs 805 lines, upstream members named
`live_slots_` where this copy has `slots_`). Ported the two functions from `src/core/expert_cache.cpp:636/672` against
this copy's members — pure host bookkeeping, no device call.

### 2.3 `q_of(stream)` was casting the engine's stream to `sycl::queue*`, and it is not one

The engine's `cudaStream_t` is the S1 shim's `Ds41SyclStream*`, which **holds** a `std::unique_ptr<sycl::queue>`; the
sycl tree's `q_of()` does `(sycl::queue*) stream`. The first real call
(`run_chunk → chunk_classes → gpu_run_chunk → native_quantize_q8_1`) therefore submitted through a garbage queue and
segfaulted in `submit_kernel_direct_impl`:

```
#0 queue_impl::submit_kernel_direct_impl()
#2 strata::kernels::native_quantize_q8_1
#3 strata::v41::gpu_run_chunk
#4 strata::v41::chunk_classes
#5 strata::v41::Ds41MoeTier::run_chunk
```

Fix (three files, 15 lines): the shim exposes `void* ds41_sycl_queue_of(void* stream)`
(`sycl_compat/ds41_sycl_compat.{hpp,cpp}`), and `sycl/include/strata/sycl_queue.hpp` gains a
`STRATA_QOF_ENGINE_STREAM` branch in which `q_of` calls it (a null handle still means the default in-order queue).
The kernel TUs are built with that define. **This is the "adopt the engine's stream type" point the rest of the port
needs too** — any Strata kernel launcher called from engine code hits it.

---

## 3. What happened when it ran (the run that wedged the card)

```
$ ./run_gpu.sh smoke4 ds41_generate_gpuexp --slots 1200 --prompt-batch 0 --prefill-chunk 8
```
(`--experts gpu`, `STRATA_ROUTED_HOST` unset, `ONEAPI_DEVICE_SELECTOR=level_zero:0`, mem watchdog armed the whole time;
MemAvailable never below 122 GiB.)

```
ds41: stage 0 layers [0,40) CUDA device 0
arena: pinned (cudaHostRegister PORTABLE ok; MAP_HUGETLB unavailable …; transparent huge pages requested)
tier: 0 experts in VRAM, 0 on the remote card (0.0 GiB), 6383 in the RAM arena (60.0 GiB), 8977 file tier; seeded in 19.5 s
loaded in 21.0 s: 40 layers, ctx 4096, prefill chunk 8, experts gpu, 9 classes
```

So `Cpu_only=false`, the arena built, the 9 format classes were accepted (`native_expert_supported3` +
`native_expert_layout3` ran at tier init and the per-layer blob-size check `L.bytes == im_->bl[l]` passed for all 40
layers — that is a real check that my `iq_row_bytes`/`layout3` transcription is right: the CPU side computes the same
bytes from the CPU kernels' own `NativeFmt`).

Then, ~1 min in, the card:

```
06:18:27 xe 0000:03:00.0: *ERROR* Tile0: GT0: Timed out wait for G2H, fence 24591, action 3003, done no
06:18:27 xe 0000:03:00.0: *ERROR* Tile0: GT0: GuC PC query task state failed: -ETIME
06:18:32 xe 0000:03:00.0: Tile0: GT0: Check job timeout: seqno=369057 … not started
06:18:37 xe 0000:03:00.0: Xe device coredump has been created
06:18:37 … 4 × "Timedout job … trying reset from guc_exec_queue_timedout_job … reset done"
06:18:38 xe 0000:03:00.0: *ERROR* CRITICAL: Xe has declared device 0000:03:00.0 as wedged.
06:18:38 xe 0000:03:00.0: device wedged, needs recovery
```

The engine process went into uninterruptible sleep on a `dma_fence` (`State: D`, `dma_fence_default_wait`), so it is
**unkillable** and still holds `/dev/dri/renderD128`; Level Zero no longer enumerates the Arc at all
(`sycl-ls` → only `[level_zero:0] Intel(R) UHD Graphics 770`; `lspci` still shows `03:00.0 Battlemage G31 [Arc Pro B70]`
on driver `xe`).

**Recovery needs root — this is the first thing the next session has to do:**

```
sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:03:00.0/remove; sleep 2; echo 1 > /sys/bus/pci/rescan'
# or (the stuck D-state process may hold the module): reboot
sycl-ls | grep -i "Arc Pro B70"          # verify: level_zero:0 is the Arc again
```

Also, before that: `/sys/class/drm/card1/device/devcoredump/data` holds the xe coredump of the hang (root-only, 0600).
It is the only artefact that can say *which* job hung the GT — **save it first** (`sudo cp … ~/s4/step2/xe-coredump.bin`,
then `sudo sh -c 'echo 1 > /sys/class/drm/card1/device/devcoredump/…'` to release it, and decode with the xe coredump
tooling).

### 3.1 What most likely hung it (for the next attempt, not a conclusion)

The first GPU work this binary ever did is: `cudaMalloc` of the tier's device buffers, then
`native_quantize_q8_1(…, 5120, ncols)`, then `native_expert_grouped` for the first (layer, class, site). Between the
segfault-free start and the wedge there were ~60 s, which covers model load (21 s), the arena seed (19.5 s) and then
the first chunk. Two candidate causes, both cheap to tell apart once the card is back:

1. **A launch geometry the port has never seen.** `native_expert_grouped` under this engine is called with
   `cap_groups`/`cap_entries` from `kMaxEnt` and a scratch buffer from `native_expert_scratch_bytes(cap, n_ff)`; a
   scratch too small for the call's own entries is an out-of-bounds write by the *kernels*, which on Xe can wedge the
   GT rather than fault the context. Check `STRATA_EXPERT_DEBUG`-style logging of the first call's
   `(cap_groups, cap_entries, n_groups, entries)` against `ds41_moe.cpp`'s allocation.
2. **The Q8_1 quantizer's grid.** `native_quantize_q8_1` launches `(n+255)/256` work-groups for `n = n_in*ncols`; the
   tier's first call passes `n_in = n_embd` and the chunk's column count. Unlike the expert kernel this TU has never
   run inside the engine (only in `iq_multi_parity`).

The cheap discriminator: run the *same binary* with `--experts gpu` but a path that only quantizes / only runs a
zero-entry grouped call, or set `STRATA_EXPERT_DEBUG` if the tier honours it, and watch `xe`'s journal for the first
timeout. A single-token decode (`--ids … -n 1 --prefill-chunk 0` with `--prompt-batch 0`) is a much smaller first step
than an 8-token chunk.

---

## 4. Arena sizing: what was asked for, what the tier did, and why that matters

### 4.1 The numbers used

`--slots 1200 --arena-gib 60` (the default arena). Reasoning for 1200, unchanged from step 1 §5.2: a DS V4.1 expert
blob is 7.7-15.2 MB (mean ~10.7 MB), not the 6.75 MiB the 2150 default was measured on, so 1200 slots × 10.7 MB ≈
12.5 GiB of the 32 GB card, leaving room for the dense half, KV and the engram cache; 1400 would be ~15 GiB and
1600 ~17 GiB, i.e. the range to sweep *after* the first working run.

### 4.2 The tier put **0** experts in VRAM, and that is a bug to chase, not a tuning result

```
tier: 0 experts in VRAM, …, 6383 in the RAM arena (60.0 GiB), 8977 file tier
```

With `--slots 1200` the cache should have opened at tier init (`ds41_moe.cpp:1997`: `cfg.slots > 0 && !mixed_sizes &&
!defer_cache && !(slot_gib > 0)`) and the seed should have admitted ~1200 experts. It admitted none while the arena
took 6383 — so either (a) the cache opened and `admit()` refused everything (per-layer quota: `slots_/n_layers` = 30
per layer against `layer_next_`), or (b) the geometry is `mixed_sizes` (9 classes = 9 blob sizes, so the *mixed* seed
at `ds41_moe.cpp:2334` runs, and its `budget = slots × im_->blob` walk can end with an empty `sizes` list and leave
`gp.cache` null). `--slots auto` (`slots = 0` → `slot_gib` from free VRAM) takes the other branch entirely and is the
cheapest A/B for the next run:

```
STRATA_SEED_LOG=1 … --slots 1200      # prints "seed: X GiB free (margin) -> Y GiB of slots: N experts in VRAM"
STRATA_SEED_LOG=1 … --slots auto
```

Until VRAM is non-zero the tier is *disk-bound by construction*: 8977 of 15360 (layer, expert) pairs are file tier, and
an 8-token chunk touches 1920 experts, ~1/3 of them from NVMe. **The honest statement is: no speed number exists for
this tier yet; the run never got past its first chunk and did not report a timing line.**

### 4.3 What the ceiling would mean if it did run

Unchanged from step 1 §4: ~373 µs of expert kernels per decode layer at the model's class mix → **~15 ms/token, ~67
tok/s, if every blob were resident**. With 1200 of 15360 resident (7.8 %), the per-token traffic would be 2.6 GB from
VRAM + ~2.4 GB from the 60 GiB arena (PCIe) + ~0.5 GB from the file — the ceiling is *not* the number to expect, and
the run's own `prompt split: dense … / experts … (file wait …)` line is what should be quoted once there is one.

---

## 5. The tree as it stands (nothing half-linked)

* **The GPU-expert binary exists and links**: `/home/dyy/deepseek-work/s4/step2/ds41_generate_gpuexp` (105 657 760 B).
  It is not "working" — see §3 — but it is a whole binary, not a partial link.
* **The CPU-expert binary is untouched**: `s2/engine-obj/ds41_generate_cpuexp` sha256
  `8695f2ee03589724f5c5e2517f0196b06123b22ba1e0ed85beed1f7c9dac592a`, identical to
  `s4/backup/ds41_generate_cpuexp.cpu-expert.bin`. Nothing was rebuilt in `s2/engine-obj/`.
* **Not verified this step**: the CPU-expert binary's own smoke run — the Arc is wedged, and `--backend cuda` needs it.
  Nothing in this step's changes can affect it (they are new TUs and one shim accessor, none of which it links).
* **Committed** (one commit): `sycl/src/kernels/cuda/iq_kernels_fork_expert.dp.cpp`,
  `sycl/src/kernels/cuda/iq_mmvq_refuse.dp.cpp`, `sycl/src/program/ds41_transcode.dp.cpp`,
  `sycl/src/core/expert_cache.cpp` (+`evict_from`/`admit_at`), `sycl/include/strata/sycl_queue.hpp`
  (`STRATA_QOF_ENGINE_STREAM`), `tools/v41/engine/sycl_compat/ds41_sycl_compat.{hpp,cpp}` (`ds41_sycl_queue_of`),
  `sycl/CMakeLists.txt` (source list), this report.
* **Build/run scripts** (in `s4/step2/`, not committed — they are this machine's wiring):
  `build_all.sh` (all 11 TUs, with `-Ddpct=strata_dpct -DSTRATA_QOF_ENGINE_STREAM`), `build_dense.sh` (the S3
  `DS41_SYCL` dense TU, needed because `libggml-sycl.a` was rebuilt at 03:52 with the S3b top-k fix and the
  `s2/engine-obj/ds41_dense.o` from 00:17 predates `cee1f30`'s get_rows retype — the current source must be used),
  `link_gpuexp.sh`, `run_gpu.sh` (mem watchdog), `ne_parity2` (the 93-case harness).

---

## 6. Ordered remainder

1. **Recover the card** (§3): `sudo` remove/rescan or reboot; then `sycl-ls | grep "Arc Pro B70"`. Save the xe
   coredump first.
2. **Make the VRAM cache actually fill** (§4.2): `STRATA_SEED_LOG=1` with `--slots 1200` and with `--slots auto`;
   compare `resident()`. If `mixed_sizes` is the cause, the fix is in `seed_from_ranked`'s uniform branch
   (`ds41_moe.cpp:2359-2372`) for a 9-class geometry, not in `--slots`.
3. **Find the job that wedged the GT** (§3.1) before running a full 8-token chunk again: single-token decode first,
   then one chunk; watch `journalctl -k --since` for the first `Timedout job`. If it is `native_expert_grouped`, log the
   first call's `(cap_groups, cap_entries, entries)` against the scratch the tier allocated; if it is
   `native_quantize_q8_1`, its `n = n_in × ncols`.
4. **Then** the one-layer one-token A/B (`--experts gpu` vs `--experts cpu`, same `--ids`, `--prompt-batch 0
   --prefill-chunk 8`, `--dump-last`), compared with S3b's chaos-envelope method (perturbation curve: the two dumps
   must agree to a max|Δ| inside the CPU run's own `--temp 0` noise band; token ids alone prove nothing on this model).
5. **Then** prefill/decode tok/s with VRAM hit rate and file/CPU/PCIe counts, and the `dense vs experts (file wait)`
   split the engine prints. Only at that point sweep `--slots` in 1200-1600.
6. After that, step 1's §5.4 list (multi-entry kernels, PROM variants, Q3_K inner loop).
