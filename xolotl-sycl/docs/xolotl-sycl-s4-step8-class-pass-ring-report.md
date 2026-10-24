# S4 step 8 — the class-pass control ring: why the whole-layer prefetch "changed the routing"

Status: **fixed and verified bit-exact**. The bug that blocked both remaining prefill levers (whole-layer slab
prefetch, async pageable staging) was not in the expert tier's data path and not in the router: it was a pinned
**host control-plane ring that laps inside one layer**, so an earlier format class's launch consumed a later class's
metadata. `tools/v41/engine/ds41_moe.cpp` (S8 commit). The slab path is now bit-exact at 256 and 1024 and is a
+3 % prefill win at 1024 (a 21 % loss at 256, as the width rule already predicted).

## 1. TL;DR

* **First divergent tensor: layer 3's `routed`** (the MoE tier's weighted sum, i.e. `run_chunk`'s output). Gate-off is
  a normal activation (min −0.159, max 0.153); gate-on has 30 of 256 rows fully non-finite (2957/5120 values each, 78
  `inf`), one more row wrong but finite, and finite values up to 2.4e8. Layers 0-2's `fn` and `routed` are
  byte-identical. Routing is still identical at layer 3 (185 distinct experts, 1536 entries) — the collapsed routing
  reported in S7 appears only at layer 4, *because* layer 3's output was NaN.
* The NaN rows belong to a handful of experts of **one format class** of layer 3 (`e = 337, 357, 312, 184, 87, ...`).
  Their `c_parts` rows were never written by the pass that should have written them: a launch executed with another
  class pass's metadata.
* **Root cause**: `Gpu::kCtlRing = 16` pinned metadata slots, taken 3+ per class pass, while a per-expert-geometry
  layer runs **one pass per format class** (9 here) → ≥27 slots per layer. The host memcpy'd the next pass's data into
  a slot whose async H2D copy was still queued, so the earlier pass's `c_ptr / c_start / c_ng / c_dst / c_tokv / c_w`
  were replaced by the later pass's. Nothing recorded a reuse dependency.
* **Why it needed a timing change to appear**: the on-demand reader stalls the host on a file read between class
  passes; the slab prefetch serves those reads from pinned memory, so the host runs ahead and laps the ring. That is
  the entire "extra asynchronous host/device activity perturbs the forward pass" story.
* **Fix**: one `cudaEvent_t` per control slot; record it after the copies that read the slot, `cudaEventSynchronize`
  it when the slot is taken again. Ring stays 16 slots; the dependency no longer depends on host lead.
* S7's bisect ("PF_MODE=2 shows the slab is innocent") was invalid: `chunk_pf_prepare`'s return value is unused and
  it had already set `cs.cur`, so mode 2 consumed the slab too. With a real no-consume mode the slab is *not*
  innocent, and with the event fix it is exact.

## 2. Method

Sequential chunk path so `STRATA_LAYER_DUMP` applies (`--prompt-batch 1 --prefill-chunk 256`; deliberate — the
default path is layer-major and has no dump hook), prompt ids `(i*100)%120000+1`, `-n 8`, both arms:

```bash
# off
STRATA_LAYER_DUMP=…/dump_off STRATA_ROUTE_LOG=…/route_off.log ./run.sh off 256 8 --prompt-batch 1 --prefill-chunk 256
# on
STRATA_CHUNK_PF_MIN=0 STRATA_LAYER_DUMP=…/dump_on STRATA_ROUTE_LOG=…/route_on.log ./run.sh on 256 8 --prompt-batch 1 --prefill-chunk 256
```

`L%02d_fn.f32` (router input, 256×5120) and `L%02d_routed.f32` (tier output) were compared element-wise per layer.
md5s reproduce S7 exactly: off `fce1788f…`, on `5a3024c5…` at 256, and the route log reproduces the S7 collapse
(layers 0-3 match, 4-39 collapse to 6 experts × 253-256 tokens).

Then, because "routing collapsed" only says where the *symptom* starts, the tier was instrumented
(`STRATA_NAN_DBG=1`, off by default):

* `NANDBG PASS layer … class … own_nf … other_nf …` — per class pass, how many of its own / other classes' `c_parts`
  rows are not finite;
* `NANDBG layer … pass_class … e … cls … slab … off … len … | nonfinite … | dev==file / dev==slab / dev!=…` — for each
  bad entry, the expert, its class, its slab offset, and whether the bytes still in the device staging slot equal the
  file's blob (a fresh `read()`) or the slab's copy;
* `STRATA_NAN_SYNC=1` makes the probe sync at the end of every class pass (this perturbs timing and is a probe, not a
  measurement).

That found: the last class pass of layer 3 (class 6, 8 experts) is itself clean; its 33 non-finite rows are entries of
the layer's *other*, earlier class passes, i.e. an earlier class pass had already computed — or rather failed to
compute — those rows.

## 3. The race

`gpu_run_chunk` pushes its per-pass host metadata through a pinned ring (`Gpu::h_lc / h_dst / h_tok / h_w`,
`kCtlRing = 16`, ds41_moe.cpp:485-497) and then copies it with `cudaMemcpyAsync` on the compute stream `gp.s`:

| slot use | code | copies |
|---|---|---|
| `h_dst`/`h_tok` (entry→routing index, entry→token) | `gpu_run_chunk` (`cs = ctl_slot()`) | 2 |
| `h_lc` (blob pointers + `grp_start` + `n_groups`), **one per launch** | `launch()` (`ls = ctl_slot()`) | 3 |
| `h_w` (top-k mix weights) | output section (`ws = ctl_slot()`) | 1 |

The host writes the slot with `std::memcpy` and only *then* queues the copy. The copy reads the slot **when the queue
gets to it**, not when it is issued. So a slot may only be rewritten after its copy has completed. The old comment
claimed this was guaranteed by "the host runs at most ~2 passes ahead of the card (a `cudaStreamSynchronize` per
pass)" — that was true for the single-pass decode path, but:

* `cls(layer, e)` is per-expert here (DS4: 9 format classes), so a layer is `chunk_classes` → one `gpu_run_chunk` per
  class: **9 passes, each taking ≥3 slots**, = ≥27 slot assignments per layer against a 16-slot ring → the ring laps
  inside the layer;
* the only sync in that sequence is the *last* pass's `"chunk sync"` (`if (!last_pass) { … return true; }` returns
  without syncing);
* on the on-demand path the host stalls in `ready[k].wait()` / `cudaStreamSynchronize(gp.s_cp, "wave")` between
  passes, so it never gets far enough ahead to lap the ring. The slab prefetch removes those stalls
  (`pf_slab >= 0` short-circuits the wave reader), the host runs ahead, the ring laps, and an earlier class's launch
  is issued with a later class's `ptr/start/ng/dst/tok/w`. Its kernels then write another class's rows into
  `c_parts` at blob pointers from the wrong batch; the intended rows stay uninitialised and the weighted sum
  (`weighted_rows_sum`, or the host double-sum) turns them into NaN/huge values.

Evidence that it is the *ring*, not the slab, and that it is an ordering bug:

| probe (gate on, 256 tokens) | result |
|---|---|
| default (`kCtlRing = 16`) | wrong (`5a3024c5`), 31 NaN rows in layer 3 |
| `kCtlRing = 64` (no wait added) | **correct** (`fce1788f`) — 27 < 64 slots, no lap inside a layer |
| `cudaStreamSynchronize` at the end of every class pass | **correct**; slab still consumed (3329 file DMAs, file wait 0.0) |
| `PF_MODE=2` as S7 wrote it (`return -1`, no `cs.cur` change) | wrong — consumption was never actually disabled |
| `PF_MODE=2` fixed (`cs.consume = false`) | correct — the on-demand reader's host stalls keep the lead small |
| `PF_MODE=3` (allocate+build, read nothing, do not consume) | correct |
| fixed, slot event + `kCtlRing = 16` | **correct** |

## 4. The fix

```cpp
// Gpu (ds41_moe.cpp:497)
cudaEvent_t ctl_ev[kCtlRing] = {};   // the copies that read that slot, recorded on the stream that issued them

// gpu_run_chunk (:4935)
auto ctl_slot = [&]() -> int {
    const int s = gp.ctl_at++ & (Gpu::kCtlRing - 1);
    if (!gp.ctl_ev[s]) ck(cudaEventCreateWithFlags(&gp.ctl_ev[s], cudaEventDisableTiming), "ctl ev");
    else ck(cudaEventSynchronize(gp.ctl_ev[s]), "ctl slot free");   // the copy that read it is home
    return s;
};
auto ctl_done = [&](int s) { ck(cudaEventRecord(gp.ctl_ev[s], gp.s), "ctl ev record"); };
```

`ctl_done(s)` is called immediately after the copies that read the slot (the `h_dst/h_tok` pair, the three `h_lc`
copies, the `h_w` copy); the events are destroyed in `Gpu::close`. Because the copies are on the in-order compute
stream, one record after the group covers all of them; taking the slot again blocks the *host* (not the stream) until
they are home, which is exactly the lifetime of the pinned buffer. Routing is now independent of DMA timing because
the host can no longer overwrite a staging buffer whose consumer copies are still outstanding — the dependency is per
slot and explicit, not the accident of a file-read stall.

The other resources the host rewrites per pass were audited: `h_cx` (activations) is written once per layer and each
layer's last pass ends in a full `gp.s` sync (safe); the pageable fallbacks are consumed synchronously by the shim's
staging copy (safe); the slab is written only by the joined reader threads (safe); the device staging ring already
had its own per-batch events (S7).

## 5. Proof (default layer-major path, 1024 tokens, `--dump-last`)

`md5` is of the last prompt row's 129280 logits; `787519ba…` is the established baseline.

| binary | arm | md5 | prompt | decode |
|---|---|---|---|---|
| pre-fix (`s4/step8/ds41_ref`, = S7 step2 binary) | gate off | `787519ba…` | 33.4-33.8 tok/s | 8.5 tok/s, 98.3 % VRAM |
| pre-fix | gate on | `e76aac39…` **wrong** | 39.8 tok/s | 1.61 tok/s, 39.5 % VRAM |
| fixed | gate off | `787519ba…` | 33.4-33.8 | 8.5 |
| fixed | gate on | `787519ba…` (2 runs) | 34.4-34.8 | 7.3-8.1, 98.3 % |
| fixed | `DS41_SHIM_ASYNC_STAGE=1` | `787519ba…` | 34.3-34.5 | 98.3 % |
| fixed | async + gate on | `787519ba…` (2 runs) | 35.3-35.9 | 98.3 % |
| fixed | gate on at 256 | `fce1788f…` (= gate-off md5) | 10.6 | — |

Decoded tokens are identical in every arm: `223 29429 201 223 29429 201 223 29429`. The fixed gate-on prompt issues
the same 4919 file-tier DMAs as gate-off (identical routing), where the broken arm issued 977 (collapsed routing).

## 6. Measured speedups (only after §5; idle box, 1024 tokens, `-n 16`, 2 runs each)

| arm | 1024 tok/s | delta | experts / file wait |
|---|---|---|---|
| gate off | 33.4 / 33.8 (33.6) | — | 22.5-22.9 s / 8.9 s |
| gate on | 34.8 / 34.4 (34.6) | **+3.0 %** | 21.5-22.1 s / 0.0 s |
| async only | 34.5 / 34.3 (34.4) | +2.4 % | 21.6-21.9 s / 9.0 s |
| async + gate on | 35.9 / 35.3 (35.6) | **+6.0 %** | 20.9-21.3 s / 0.0 s |

At 256 the slab is a **loss**: 13.5 → 10.6 tok/s (−21 %), because it sweeps the whole file tier (99-102 GB/prompt)
while the routing covers only 37 %. So the S7 width rule (break-even between 256 and 1024, threshold 1024) is
correct; the S7 *numbers* for the "on" column were not (they were measured on collapsed routing).

## 7. Limits and open items

* **2048 is still not a measurement point.** The unchanged baseline gives different last rows run to run there
  (rel 4.7 %), so nothing above is quoted at 2048.
* **Async staging was already exact at 256 and 1024 before this fix** (both binaries, `787519ba`). S7's "async changes
  the logits" observation exists only at 2048, i.e. at a point that cannot adjudicate. It is plausible that the same
  ring lap caused it (async removes the shim's per-copy drain, which also keeps the host back), but that is not
  proven.
* **`chunk_prestage`** (arena experts into the device ring, `n >= 4096` rows) was not exercised at 256/1024 and should
  be re-verified at a wide prompt before trust; it shares the S7 ring logic, not the ctl ring.
* **Default is unchanged**: `chunk_pf_min_rows()` still returns `kPfNever` unless `STRATA_CHUNK_PF_EXACT=1`
  (`STRATA_CHUNK_PF_WIDTH`, or `STRATA_CHUNK_PF_MIN`). Which lever to enable — `+3 %` at 1024, `+6 %` with async — is
  a call for the parent, not for this fix.
* The diagnostics (`STRATA_NAN_DBG`, `STRATA_NAN_SYNC`, `STRATA_CHUNK_PF_MODE=2/3` with a real no-consume arm) stay in
  the tree, env-gated and inert by default; they are what turned this from "the forward pass is wrong" into a
  layer/class/slot.

## 8. Reproduce

```bash
BIN=/home/dyy/deepseek-work/s4/step2/ds41_generate_gpuexp     # fixed build
# correctness gate (default path, 1024)
./run.sh off 1024 8 --dump-last off.bin                       # 787519ba…
STRATA_CHUNK_PF_MIN=0 ./run.sh on 1024 8 --dump-last on.bin   # 787519ba…
DS41_SHIM_ASYNC_STAGE=1 ./run.sh async 1024 8 --dump-last a.bin  # 787519ba…
# the original symptom, pre-fix binary
BIN=/home/dyy/deepseek-work/s4/step8/ds41_ref STRATA_CHUNK_PF_MIN=0 ./run.sh bad 1024 8 --dump-last bad.bin  # e76aac39…
```
