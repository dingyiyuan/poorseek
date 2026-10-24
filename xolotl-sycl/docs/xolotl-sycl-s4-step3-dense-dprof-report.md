# S4 step 3 — per-call-site accounting of the dense half: the 3.5 s is three *cold* graph computes, not 1.15 s/token

2026-10-24. Pure measurement; nothing was optimised, nothing in the hot path changed behaviour.
Instrumentation behind `DS41_DPROF=1` (inert when unset), committed with this report.

## 0. TL;DR

The engine's `prompt split: dense 3.5-3.9 s` for 3 tokens is **not** ~1.2 s/token of steady-state
work. Measured per graph submission, **93 % of the dense wall is paid once, by three first-run graph
computes**:

| contributor | one call | share of the 3833.9 ms dense wall |
|---|---|---|
| **first run of layer 0's attention graph** | **2819.97 ms** | **73.6 %** |
| first run of the `init` graph (embedding → state, 9 nodes) | 690.09 ms | 18.0 % |
| first run of layer 20's attention graph (358 nodes) | 71.93 ms | 1.9 % |
| all 40 layers × 3 tokens of *everything else* (38 warm layers, warm repeats, finish, alloc, build, transfers) | 251.9 ms total | 6.6 % |

The steady state is **~78 ms/token** of dense (40 attention graphs at 2.0 ms each + finish + inputs),
not 1.2 s/token — i.e. the GPU dense half is ~4x faster than the CPU backend's 0.33 s/token, the
same direction the GEMV microbenchmark measured. The "83x" came from dividing a **per-process cold
start** by 3 tokens.

Building is *not* the problem: 58 attention graphs cost **3.8 ms total** to build (0.06-0.19 ms each,
165-389 nodes each). Allocation is 2.9 ms. Transfers are 6 ms. Graph compute is 99.6 % of the dense
wall.

## 1. What was built (`DS41_DPROF=1`)

`tools/v41/engine/ds41_dense.cpp`:

* Every `ggml_backend_graph_compute`, `ggml_backend_synchronize`, `ggml_backend_tensor_get`,
  `ggml_backend_tensor_get_async`, `ggml_backend_tensor_set` in the dense TU (104 call sites) is wrapped
  by a macro that keeps the call's argument list verbatim (`DPROF_SET(...)`, `DPROF_GET(...)`,
  `DPROF_GET_ASYNC(...)`, `DPROF_SYNC(...)`) and records **count, wall time, max and bytes keyed by
  `file:line`**. The diff of those lines is a callee rename, nothing else. There is **no
  `ggml_backend_graph_compute_async` call site in the dense path** (verified): the only graph submission
  is `Ds41Dense::Impl::compute`, one per layer-graph, and in this fork that entry point is
  `async + synchronize` (`src/ggml-backend.cpp:595`), so its interval is submit + every kernel + the
  final stream wait. Encode-only time is therefore not separable from the host side without a
  backend-side probe (see §7).
* **Graph build vs alloc vs compute** per `(kind, layer)`: each `build_*` call site is wrapped with
  `DPROF_BUILD(kind, il, …)`, which records the build wall time, the node count
  (`ggml_graph_n_nodes`) and registers the graph pointer; `compute()` then charges every run of that
  graph to the same row (runs, alloc ms, compute ms, max compute ms, and the device buffer the graph's
  nodes live in).
* **Spans**: `begin`, `upload_inputs`, `engram_rows`, `engram_wait`, `attn_router`, `finish`,
  `finish_predict`, `predict`, `logits`, and the batch path's `span_attn` / `span_kv`. A span is one
  dense API call; leaf calls are charged to the innermost open span, so
  `wall = direct backend calls + child spans + host residue` is exact per span and per segment.
* **Segments**: `dense.dprof_checkpoint("prompt")` is called right after the engine prints
  `prompt split: …`, so the table below is exactly the prompt segment the engine measured. (The rest of
  the run is printed at exit; in this 3-token command there is none — `-n 1` only samples the next
  token from the last prompt row, it does not run a decode pass.)
* With `DS41_DPROF` unset: one branch on a `const bool` per call site and no clock read; the only thing
  still evaluated is the call's own size argument (already an argument of the original call), passed
  through a trivial inline function. No map lookup, no clock, no allocation, and the ops, buffers and
  numbers are untouched.

## 2. The run

The established command, verbatim:

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 DS41_DPROF=1 ds41_generate_gpuexp \
  -m V41-GSQ-1017.gguf --backend cuda --experts gpu \
  --engram engram-layers.1:engram-layers.14 --rank rank.bin \
  --ids 100,200,300 -n 1 --ctx 2048 --temp 0 --slots 1200 --prompt-batch 0 --prefill-chunk 1
```

`--prompt-batch 0 --prefill-chunk 1` → three sequential single-token passes; per token the dense path is
`begin()` + for each of 40 layers `attn_router()` then `finish()`, then the head. Whole 3-token prompt:
**244 graph computes, 120 syncs, 360 async readbacks, 168 sets** — i.e. per token **81.3 computes,
40 syncs, 120 readbacks, 56 sets**. Engine's own line in the published run:
`prompt split: dense 3.9 s, experts 0.2 s (file wait 0.0 s), logits 0.0 s`.

Four runs (the table below is run 4; runs 1-3 differ only in the cold cost):

| run | engine `dense` | instrumented dense top spans | cold cost (3 first-runs) |
|---|---|---|---|
| 1 | 3.9 s | 3930.01 ms | 3587.9 ms (init 700.1 + L0 2887.9) |
| 2 | 4.2 s | 4203.09 ms | 3585.6 ms (init 700.1 + L0 2885.5 + L20 290.9) |
| 3 | 4.1 s | 4116.29 ms | 3777.4 ms (init 824.8 + L0 2952.6) |
| **4 (published)** | **3.9 s** | **3833.86 ms** | **3582.0 ms (init 690.1 + L0 2820.0 + L20 71.9)** |

The spread between runs is almost entirely the cold cost (2.82-2.95 s for layer 0, 0.69-0.82 s for
init) — consistent with first-use compilation/installation inside the runtime, not with work
proportional to the data.

## 3. (a) Per-site table, and it sums to the dense wall

Published run (`dprof4`), prompt segment. `KiB` is the byte total of that site; `max` is the single
worst call (this is what exposes the cold runs).

```
cat        site (ds41_dense.cpp)      count     total_ms    mean_us     max_us        KiB
compute    :1102  graph_compute         244     3836.020    15721.4  2819967.9        0.0   <- 1 call = 2820.0 ms
set        :862   weight upload (load)  923     1133.867     1228.5   187662.6  8532240.4   <- NOT dense (load time)
build      :3071  build_attn (58 graphs) 58        3.795       65.4      109.8        0.0
set        :2338  st_sc reset              3        3.032     1010.8     3017.3       12.0   <- setup
alloc      :1101  gallocr alloc         244        2.851       11.7       81.8        0.0
set        :2537  i_emb (upload_inputs)   3        2.687      895.5     2679.8       60.0
set        :2362  neg_ramp                 1        2.522     2521.8     2521.8       16.0   <- setup
get_async  :3086  o_fn  readback         120        1.212       10.1       36.5     2400.0
get_async  :3087  o_ids readback         120        0.835        7.0       14.3        2.8
get_async  :3088  o_wts readback         120        0.773        6.4       14.9        2.8
set        :3146  routed upload (finish)  120        0.483        4.0        7.4     2400.0
sync       :3093  attn readback sync     120        0.156        1.3        4.5        0.0
set        :2407  router bias              40        0.119        3.0       18.2       60.0
get        :3239  head logits               1        0.099       99.3       99.3      505.0   <- logits, not dense
set        :2619  i_eng rows                6        0.053        8.8       22.7      144.0
set        :2576-2582 i_nvis/i_cslot/i_cpos 18      0.049        2.3        2.8        0.0
build      :2646  build_init                1        0.011       11.1       11.1        0.0
set        :2557/2558/2563/2540/2595/2596 18        0.049        2.5        5.5        3.0
build      :3237  build_head                1        0.008        8.3        8.3        0.0
build      :3157  build_finish              1        0.008        7.5        7.5        0.0
set        :2365  blk_ramp                  1        0.005        4.9        4.9        2.0
```

Span table (same run), and the reconciliation the profiler prints:

```
span                      count         wall         mean direct_calls      residue  children
span:begin                    3       694.60       231.53       690.18         0.01  span:upload_inputs 4.4
span:attn_router            120      3136.36        26.14      3136.02         0.33  span:engram_wait 0.0
span:finish                 120         2.90         0.02         2.79         0.11
span:logits                   1        17.27        17.27        17.26         0.01      (not dense)
---- child spans ----
span:upload_inputs            3         4.42         1.47         2.84         0.05  span:engram_rows 1.5
span:engram_rows              3         1.53         0.51         0.00         1.53
span:engram_wait              9         0.00         0.00         0.00         0.00
[5] reconciled: top-level spans 3851.13 ms = direct backend calls 3846.25 + child spans 4.42 + host residue 0.46
```

So, with `logits` (17.27 ms) excluded, **the dense wall is 3833.86 ms** and it decomposes as:

```
graph compute (dense part)               3818.87 ms   99.61 %   (incl. the three cold 3581.98 ms)
build (58 attn + init + finish graphs)      3.81 ms    0.10 %
gallocr alloc                               2.85 ms    0.07 %
attn readbacks (get_async + sync)           2.98 ms    0.08 %
finish routed upload (set, 40 x 20 KiB)     0.48 ms    0.01 %
upload_inputs span (35 sets + engram read)  4.42 ms    0.12 %
host residue                                0.46 ms    0.01 %
                                         ----------
                                         3833.87 ms  100.00 %
```

i.e. **the instrumented sites explain all 3833.9 ms** with 0.46 ms unaccounted (0.01 %): 99.96 % of
the dense wall is accounted for by measured intervals.

## 4. (b) Build vs compute per layer, with node counts

From the same run. `runs` is over the three prompt tokens; `compute_max_ms` is the worst single run
of that graph — for every layer this is the **first** run. `buf_MiB` is the device buffer the graph's
nodes live in (the gallocr's planned compute buffer).

| kind | layer | nodes | builds | build_ms | runs | alloc_ms | computes | compute_ms | max_compute_ms | us/compute | us/node | buf_MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| init | -1 | 9 | 1 | 0.011 | 3 | 0.038 | 3 | 690.13 | 690.09 | 230043.1 | 25560.34 | 1.0 |
| attn | 0 | 165 | 1 | 0.061 | 3 | 0.144 | 3 | 2821.96 | 2819.97 | 940652.2 | 5700.92 | 8332.3 |
| attn | 1 | 189 | 1 | 0.064 | 3 | 0.145 | 3 | 12.27 | 8.96 | 4089.5 | 21.64 | 8332.3 |
| attn | 2 | 365 | 2 | 0.188 | 3 | 0.189 | 3 | 12.21 | 9.48 | 4070.1 | 11.15 | 8332.3 |
| attn | 3 | 176 | 2 | 0.116 | 3 | 0.096 | 3 | 8.98 | 6.93 | 2992.4 | 17.00 | 8332.3 |
| attn | 4 | 176 | 2 | 0.127 | 3 | 0.044 | 3 | 8.57 | 6.52 | 2855.2 | 16.22 | 8332.3 |
| attn | 5 | 176 | 2 | 0.130 | 3 | 0.043 | 3 | 8.56 | 6.52 | 2853.0 | 16.21 | 8332.3 |
| attn | 6 | 176 | 2 | 0.110 | 3 | 0.042 | 3 | 5.26 | 3.22 | 1754.3 | 9.97 | 8332.3 |
| attn | 7 | 176 | 2 | 0.105 | 3 | 0.041 | 3 | 5.28 | 3.24 | 1759.8 | 10.00 | 8332.3 |
| attn | 8 | 365 | 2 | 0.172 | 3 | 0.191 | 3 | 6.14 | 3.41 | 2047.3 | 5.61 | 8332.3 |
| attn | 9 | 176 | 2 | 0.112 | 3 | 0.092 | 3 | 5.27 | 3.22 | 1755.5 | 9.97 | 8332.3 |
| attn | 10 | 176 | 2 | 0.125 | 3 | 0.040 | 3 | 5.25 | 3.21 | 1748.8 | 9.94 | 8332.3 |
| attn | 11 | 176 | 2 | 0.105 | 3 | 0.038 | 3 | 5.27 | 3.20 | 1757.9 | 9.99 | 8332.3 |
| attn | 12 | 176 | 2 | 0.104 | 3 | 0.038 | 3 | 5.25 | 3.21 | 1750.3 | 9.94 | 8332.3 |
| attn | 13 | 176 | 2 | 0.103 | 3 | 0.042 | 3 | 5.25 | 3.20 | 1748.4 | 9.93 | 8332.3 |
| attn | 14 | 389 | 2 | 0.191 | 3 | 0.148 | 3 | 8.16 | 4.06 | 2719.8 | 6.99 | 8332.3 |
| attn | 15 | 176 | 2 | 0.119 | 3 | 0.066 | 3 | 5.25 | 3.20 | 1748.3 | 9.93 | 8332.3 |
| attn | 16 | 176 | 2 | 0.122 | 3 | 0.027 | 3 | 5.25 | 3.21 | 1749.8 | 9.94 | 8332.3 |
| attn | 17 | 176 | 2 | 0.127 | 3 | 0.025 | 3 | 5.25 | 3.21 | 1751.7 | 9.95 | 8332.3 |
| attn | 18 | 176 | 2 | 0.115 | 3 | 0.024 | 3 | 5.24 | 3.20 | 1746.3 | 9.92 | 8332.3 |
| attn | 19 | 176 | 2 | 0.121 | 3 | 0.024 | 3 | 5.26 | 3.22 | 1752.4 | 9.96 | 8332.3 |
| attn | 20 | 358 | 1 | 0.110 | 3 | 0.146 | 3 | 74.67 | 71.93 | 24890.1 | 69.53 | 8332.3 |
| attn | 21 | 176 | 1 | 0.079 | 3 | 0.072 | 3 | 5.26 | 3.22 | 1754.4 | 9.97 | 8332.3 |
| attn | 22 | 176 | 1 | 0.079 | 3 | 0.031 | 3 | 5.25 | 3.21 | 1749.1 | 9.94 | 8332.3 |
| attn | 23 | 176 | 1 | 0.094 | 3 | 0.039 | 3 | 5.30 | 3.26 | 1766.2 | 10.04 | 8332.3 |
| attn | 24 | 248 | 1 | 0.072 | 3 | 0.109 | 3 | 5.77 | 3.40 | 1924.2 | 7.76 | 8332.3 |
| attn | 25 | 176 | 1 | 0.058 | 3 | 0.071 | 3 | 5.25 | 3.21 | 1751.1 | 9.95 | 8332.3 |
| attn | 26 | 176 | 1 | 0.074 | 3 | 0.034 | 3 | 5.27 | 3.23 | 1757.4 | 9.99 | 8332.3 |
| attn | 27 | 176 | 1 | 0.076 | 3 | 0.032 | 3 | 5.29 | 3.22 | 1763.8 | 10.02 | 8332.3 |
| attn | 28 | 248 | 1 | 0.081 | 3 | 0.104 | 3 | 5.75 | 3.38 | 1917.2 | 7.73 | 8332.3 |
| attn | 29 | 176 | 1 | 0.060 | 3 | 0.061 | 3 | 5.26 | 3.22 | 1752.8 | 9.96 | 8332.3 |
| attn | 30 | 176 | 1 | 0.054 | 3 | 0.033 | 3 | 5.25 | 3.21 | 1748.7 | 9.94 | 8332.3 |
| attn | 31 | 176 | 1 | 0.055 | 3 | 0.032 | 3 | 5.25 | 3.22 | 1750.6 | 9.95 | 8332.3 |
| attn | 32 | 248 | 1 | 0.066 | 3 | 0.093 | 3 | 5.75 | 3.38 | 1916.9 | 7.73 | 8332.3 |
| attn | 33 | 176 | 1 | 0.055 | 3 | 0.065 | 3 | 5.29 | 3.22 | 1762.5 | 10.01 | 8332.3 |
| attn | 34 | 176 | 1 | 0.052 | 3 | 0.030 | 3 | 5.26 | 3.21 | 1753.5 | 9.96 | 8332.3 |
| attn | 35 | 176 | 1 | 0.056 | 3 | 0.030 | 3 | 5.25 | 3.21 | 1750.0 | 9.94 | 8332.3 |
| attn | 36 | 248 | 1 | 0.085 | 3 | 0.102 | 3 | 5.77 | 3.39 | 1922.0 | 7.75 | 8332.3 |
| attn | 37 | 176 | 1 | 0.060 | 3 | 0.064 | 3 | 5.24 | 3.21 | 1747.1 | 9.93 | 8332.3 |
| attn | 38 | 176 | 1 | 0.055 | 3 | 0.032 | 3 | 5.25 | 3.21 | 1749.1 | 9.94 | 8332.3 |
| attn | 39 | 176 | 1 | 0.054 | 3 | 0.028 | 3 | 5.26 | 3.22 | 1751.9 | 9.95 | 8332.3 |
| finish | -1 | 9 | 1 | 0.008 | 120 | 0.079 | 120 | 2.22 | 0.18 | 18.5 | 2.05 | 1.0 |
| head | -1 | 6 | 1 | 0.008 | 1 | 0.027 | 1 | 17.13 | 17.13 | 17127.1 | 2854.52 | 0.5 |

(`max_compute_ms` is the worst single run of that graph; for every cold row it is the **first** run.
`buf_MiB` is the device buffer the graph's nodes live in. Layers 2-19 that show `builds = 2` legitimately
built a second graph mid-prompt because the cache key includes the position-dependent `pass_cap`; that
second graph's first run cost 3-9 ms, i.e. no cold cost.)

* **Building is negligible**: all 58 attention graph builds + init + finish + head = 3.8 ms. The
  alternative hypothesis ("the per-layer graph is rebuilt every token and building dominates") is
  refuted: per graph 0.055-0.19 ms against a 26 ms mean `attn_router`; and 38 of the 40 layers' graphs
  are computed 3x with only 1-2 builds (the key includes the position-dependent `pass_cap`, so a few
  layers legitimately build a second graph as the compressed-row count grows).
* **Warm steady state**: 38 warm attention layers, 229.9 ms of compute over 3 tokens =
  **76.6 ms/token = 2.02 ms/layer = 10.4 µs/node** at 176 nodes. That is the same order as the
  microbenchmark's ~17 µs per submitted-and-synced op: the warm dense half is **dispatch-bound per
  node**, exactly as suspected — but it is 78 ms/token, not 1150.
* **Cold**: the *same* graph objects cost ~0.2-5 ms on their 2nd/3rd run. Layer 0: 2819.97 ms (run 1) +
  ~1 ms (runs 2 and 3). Layers 2-19 even built a *second* graph object mid-prompt and its first run
  cost 3-9 ms, so the cold cost is **per process / per first use of a kernel family**, not per graph
  object and not per layer.

## 5. (c) Transfer byte totals

Inside the dense spans, per token (prompt segment, 3 tokens):

* `set` (host → device): **869.0 KiB/token**, 56 calls. Dominated by the per-layer routed upload
  (40 × 20 KiB = 800 KiB) plus the per-pass input maps (`i_emb` 20 KiB, `i_widx`/`i_wmask`/… a few KiB
  each, `i_eng` 48 KiB).
* `get` + `get_async` (device → host): **801.9 KiB/token**, 120 async calls: 40 × `o_fn` at 20 KiB
  (800 KiB) + `o_ids`/`o_wts` at ~1 KiB each. The `[3]` line prints 970.2 KiB/token because it also
  divides the single 505 KiB head-logits readback by the 3 tokens (that one belongs to `logits`, not to
  the dense half).
* `sync`: 40 per token, 0.156 ms total for 120 → **1.3 µs each**: the readbacks do not wait — the
  graph compute already synchronised and the copy is on an idle stream. Host round trips are *not* a
  cost centre (this also refutes the "one big transfer per layer" reading: there is no big transfer at
  all; 20 KiB/layer is 4 µs).

Outside the dense path, for completeness: the load-time upload of the dense weights is 923 `set` calls,
**8.14 GiB in 1133.9 ms** (~7.2 GB/s, staging-buffer path). It happens once at load, before the prompt,
and proves the dense weights are device-resident before the first token (matches the earlier
`DS41_WRESID` probe).

## 6. (d) The single largest contributor

**One graph submission: the first-ever run of layer 0's attention graph, 2819.97 ms = 73.6 % of the
3833.9 ms dense wall** (74.6 % of the whole 3.8 s `dense` the engine reports). It is not spread over
layers: the same 40 graphs' remaining 118 runs cost 251.9 ms in total.

The surrounding facts that pin its nature (all measured):

* It is inside `ggml_backend_graph_compute` and nothing else: build 0.06 ms, gallocr alloc 0.14 ms,
  and the host-side span residue around it is 0.33 ms.
* It does not repeat for the same graph (next run ~0.5 ms), nor for a new graph object of the same
  layers (3-9 ms), nor for a new token. It is a **first-use** cost.
* It also happens, much smaller, on the `init` graph's first run (690 ms for **9 nodes / 1 MiB**) and on
  layer 20's first run (72-291 ms). The `init` number is the important one: 9 tiny nodes, a 1 MiB
  buffer, 690 ms — no per-node, per-byte or per-buffer explanation can produce that. A
  **per-kernel/per-module first-use cost in the SYCL/IGC runtime** (device code installation) is the
  only mechanism consistent with all three numbers.
* First-touch of the 8.3 GiB gallocr buffer is *not* the explanation: the buffer is the same 8332.3 MiB
  for every attention layer, yet only layer 0 pays ~2.8 s and layer 20 (same buffer) pays 72 ms while
  layers 1-19 and 21-39 pay nothing.

## 7. (e) The next cheapest experiment

**Per-node timing inside the backend for the cold graphs** — one small env-gated patch to
`ggml_backend_sycl_graph_compute_impl` (`src/ggml-sycl/ggml-sycl.cpp:6444`) and a relink of
`libggml-sycl.a`; no change to the engine, no extra model run beyond the same 3-token command:

```cpp
static const bool nprof = std::getenv("GGML_SYCL_NODE_PROF") != nullptr;
for (int i = 0; i < cgraph->n_nodes; i++) {
    const auto t0 = std::chrono::steady_clock::now();
    ... existing fusion / dispatch ...
    if (nprof) { const auto us = ...; if (us > 2000)
        fprintf(stderr, "nodeprof %d %-18s %-24s %8.1f ms\n", i, ggml_op_name(node->op), node->name, us/1000.0); }
}
```

It discriminates the three remaining candidates in one run:

| outcome on the first attention graph | conclusion |
|---|---|
| one node owns ~2.8 s (constant from the first node on) | a single kernel's first-use installation → name it, warm it at load |
| the first N nodes each cost 100-300 ms, then nothing | module-level compilation → the fix is one warm-up pass, not a kernel change |
| the node that first writes the 8.3 GiB buffer owns it | first-touch after all (then the buffer reuse/alloc policy is the target) |

Cheap zero-rebuild sanity checks to run with it (each is one run): `--ids 100` (1 token) must still
show ~3.5 s of cold cost, and repeating the 3-token command must show it again — bounding it as
per-process rather than per-token/per-request. A `SYCL_CACHE_PERSISTENT=1` + `SYCL_CACHE_DIR` pair of
runs would test the "cached device code" variant — **but see §9: that was one of the two runs around
which the card dropped off level_zero, so do not repeat it before a reboot without watching the device.**

## 8. What this does to the "1150 ms/token" premise

* The 3.5 s is 93 % one-time. For a **1-token** run it is the whole number; for a long-lived server it
  is paid once at the first token and never again.
* Steady state: **~78 ms/token dense** (attention 40 × 2.0 ms + finish + inputs) plus the head. The CPU
  backend's 0.33 s/token is therefore ~4x *slower*, the same sign as the GEMV microbenchmark — the
  "SYCL dense half is 10x slower than CPU" reading came from comparing the CPU's steady state against
  the GPU's process start-up.
* What is left to win in the warm path is per-node dispatch: 10.4 µs/node at 176 nodes/layer. That is
  the microbenchmark's ~17 µs fixed submission cost, so a fused/SYCL-captured graph is the lever —
  worth ~70 ms/token, not ~1.1 s/token.
* The head graph is the one warm number that looks wrong on its own: 6 nodes, 703 MB of q8_0 weights,
  17.13 ms = 41 GB/s against the microbenchmark's 594 GB/s for the same shape. It runs once per token
  and was measured once here, so it may still be partly cold; worth one dedicated look.

## 9. Instrumentation traps (both directions)

* **The fork's trap** (documented in the GEMV report) is real and relevant: `ggml_backend_graph_compute`
  is `async + synchronize`, so the interval measured here already contains the wait. Do not read
  "compute" as "encode".
* **My own trap, caught by the reconciliation**: the first version wrapped the compute in `dprof::call`
  *and* charged a manual `leaf_add` for the graph table, double-counting every compute (the totals came
  out at 7840 ms against a 3930 ms wall — exactly 2x). The [`5] reconciled` line is what caught it and
  is kept as a self-check: if `direct backend calls + child spans + residue` does not equal the
  top-level spans, the instrumentation is wrong, not the engine.
* Aggregates hide this: report **max** per graph. The mean `us/compute` for layer 0 (960 ms) looks like a
  plausible per-layer cost; the max (2820 ms) is what says "one cold call".

## 10. Incident: the Arc dropped off level_zero

During this work, after the measurement was complete:

* A variant with **8 prompt ids** (`--ids 100,…,800`, everything else identical) aborted at prompt 4/8
  with `ds4_moe: pcie dma: invalid argument` → SIGABRT. That is the expert/PCIe tier, not the dense
  path (the 3-token command ran clean 4x).
* A subsequent `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=~/.cache/sycl-ds41` probe hung for 312 s and then
  aborted inside the Intel runtime (`neo/shared/source/helpers/bindless_heaps_helper.cpp`).
* Since then `sycl-ls` lists only the UHD 770. The Arc is still on the PCI bus (`03:00.0`, `8086:e223`,
  `xe` bound, `power/runtime_status=active`) but level_zero no longer enumerates it — the same
  driver-level wedge as the earlier bad-expert-launch incident, and it needs a reboot or a root-level
  `xe` rebind.
* No process was left behind (one defunct zombie), MemAvailable stayed ≥ 123 GB throughout, every run
  was `timeout`-guarded and MemAvailable-watched, and no model run was attempted after the device
  disappeared.

## 11. Reproduce

```
# build (never while a model runs)
/home/dyy/deepseek-work/s4/step2/build_dense.sh      # dense TU, 7-8 s
/home/dyy/deepseek-work/s4/step2/link_gpuexp.sh      # relink, ~56 s
# run (watchdog: 300 s timeout, MemAvailable < 4 GiB -> kill)
DS41_DPROF=1 /home/dyy/deepseek-work/s4/step2/run_dprof.sh <tag>
```

Pre-instrumentation `ds41_generate.o` and `ds41_generate_gpuexp` are kept in
`s4/step2/backup/*.pre-dprof`. The instrumented runs' logs: `s4/step2/dprof-dprof{1..7}.log`
(`dprof4` is the published one; `dprof5` is the 8-token abort; `dprof6` the cache probe;
`dprof7` the iGPU fallback).
