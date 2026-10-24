# S4 step 1 — the first routed-expert kernel on SYCL: `native_expert_grouped`, ported and validated

2026-10-24. Stage S4 of [xolotl-sycl-port-plan.md](../../llm-for-glm-flash/docs/xolotl-sycl-port-plan.md), the step the
S3 report calls *"S4: 把路由专家搬上 GPU — 这是唯一能出真实速度的一步"*. Scope: **one** kernel family, as a beachhead.

Files:

| what | where |
|---|---|
| the port | `sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp` (new, 499 lines) |
| its parity + timing harness | `sycl/src/kernels/native_expert_grouped_parity.cpp` (new, 1382 lines) |
| build/run log of the numbers below | `/home/dyy/deepseek-work/s4/s4-both.log` |
| the ported object + binaries | `/home/dyy/deepseek-work/s4/{ne_kernel.o,ne_parity}` |

The CUDA source (`src/kernels/cuda/iq_kernels.cu`, `fork/fork_vecdot.cuh`, `fork/fork_layout.cuh`,
`src/kernels/cuda/elementwise.cu`) was **not modified**; `/home/dyy/Strata` was not touched either (it does not carry
the fork's `native_expert_grouped` *at all* — see §6.1).

---

## 0. What the model actually asks of this kernel (read, not assumed)

From the GGUF header (`V41-GSQ-1017.gguf`, 1350 tensors) and `V41-GSQ-1017.gguf.experts.json`:

| key | value |
|---|---|
| `deepseek41.hidden_size` | **5120** (= n_embd = K of gate/up, N of down) |
| `deepseek41.moe_intermediate_size` | **2304** (= n_ff = N of gate/up, K of down) |
| `deepseek41.n_routed_experts` | 384, `num_experts_per_tok` **6**, `num_hidden_layers` 40 |
| `deepseek41.swiglu_limit` | **10.0** (`NativeExpertLayout::swiglu_limit`) |
| routed-expert types (per expert, 15360 total) | gate=up ∈ {Q3_K 11: 4497, Q2_0 42: 5113, PTQ1_0 143: 5750}; down ∈ {Q3_K: 4231, Q2_0: 6682, PTQ1_0: 4447} |
| the (gate/up, down) class mix | (42,42) 4086 · (143,143) 4081 · (11,11) 2923 · (11,42) 1573 · (143,42) 1023 · (42,11) 662 · (143,11) 646 · (42,143) 365 · (11,143) 1 |

So the tier issues **one `native_expert_grouped` call per (layer, class, site)**, and the nine (gu, d) pairs above are
what has to work. The engine confirms it at load: `loaded in 1.3 s: 40 layers, ctx 4096, prefill chunk 0, experts
cpu-only, **9 classes**` (run of the untouched CPU-expert binary for this report). All nine are `native_expert_supported`-legal at 5120×2304 (`5120 % 64|128|256 == 0`,
`2304 % 64|128|256 == 0`, `5120 % 256 == 0`). Per expert the blob is

| pair | gate/up row | down row | blob |
|---|---|---|---|
| Q2_0/Q2_0 | 1440 B | 648 B | 9.95 MB |
| PTQ1_0/PTQ1_0 | 1120 B | 504 B | 7.74 MB |
| Q3_K/Q3_K | 2200 B | 990 B | 15.2 MB |

→ **~10.7 MB per expert on average, 240 expert applications per decode token (40 layers × 6) = 2.6 GB of weight bytes
per token.** That number is what decides the tier's design (§5), and it is why the kernel has to be measured, not just
ported.

---

## 1. The contract, as read from the source

`native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
const void* x_q8_1, void* scratch, float* out, void* stream, int64_t grid_groups = 0,
const NativeExpertLora* lora = nullptr)` — `include/strata/kernels/iq_kernels.hpp:74`, implemented at
`iq_kernels.cu:3720` (`iq_kernels.cu:3641` in the pre-fork copy that `/home/dyy/Strata` carries). Every pointer below is **device** memory; the counts
are read on the device.

**Layout** (`native_expert_layout`, `iq_kernels.cu:2997`): one expert is one contiguous blob
`[gate rows | up rows | down rows]`, `n_ff` rows of `n_embd` raw GGUF blocks for gate, the same for up, then `n_embd`
rows of `n_ff` blocks for down. Byte offsets: `up_off = n_ff * gu_row`, `down_off = 2 * up_off`,
`bytes = down_off + n_embd * d_row`, `tail_off = bytes (+16)` only when a projection is NVFP4 (type 40; not this
model). `gu_row = (n_embd/qk_gu) * sizeof(block)`, `d_row = (n_ff/qk_d) * sizeof(block)`.
Guards: `gu_qk`/`d_qk` > 0 for both types, `n_embd % qk_gu == 0`, `n_ff % qk_d == 0`, `n_embd % 256 == 0`,
`(n_ff*n_embd) % 256 == 0`.

**Grouping/sorting** — this is the part that makes it a "grouped" kernel:

- `grp_ptr[g]` (`g < cap_groups`) is a 64-bit **device pointer to expert blob g**. Nothing is packed into one arena by
  this kernel; the caller hands it pointers.
- `grp_start[g] .. grp_start[g+1]` are the expert's entry range in the call's entry arrays. Entries of a call are
  contiguous; a group may be **empty** (`grp_start[g] == grp_start[g+1]`), and `n_groups[0] ≤ cap_groups` groups are
  active.
- entry `e` reads token `ent_tok[e]`'s activation row (`n_embd/32` q8_1 blocks, 36 B each) and writes output **row**
  `ent_dst[e]` of `out` (`n_embd` floats). `ent_dst` is a scatter: rows are not the entry index.
- `grid_groups` (1..cap_groups; 0 = cap_groups): the launch may have fewer block-rows than possible groups; each block
  row then strides `g = blockIdx.y; g += gridDim.y`. *The result does not depend on it* (each row's arithmetic is the
  same code on the same values, a group's rows never mix). The scan is 1e^-3 of the model's arithmetic: it exists so a
  verify window's PCIe call, which usually has 0-2 groups, does not launch `cap_groups` empty block rows.

**Phases** (three launches, all over the call's *own* entries only):

1. `native_gu_kernel<T>` — grid `((2*n_ff + GU_ROWS-1)/GU_ROWS, grid_groups)`, 256 threads, `GU_ROWS = 8`, **one
   32-lane warp per row**: for row `r < n_ff` it computes gate row `r` of every entry of every active group, for
   `r ≥ n_ff` the up row `r - n_ff`; lane 0 stores `dst[e * n_ff + r] = s * scale` (the NVFP4 tail scale, 1.0 here).
2. `swiglu_q8_1_entries_kernel` (default) — over the elements `[grp_start[0]*n_ff, grp_start[*n_groups]*n_ff)`:
   `h = __fmul_rn(g / (1 + __expf(-g)), u)` with `g = fminf(gate, lim)`, `u = fminf(fmaxf(up, -lim), lim)`, then the
   q8_1 block quantizer on the spot. `STRATA_GROUPED_V1=1` instead launches `swiglu_entries_kernel` +
   `quantize_q8_1_kernel` over all `cap_entries * n_ff` (documented as bitwise identical; the harness checks it).
3. `native_down_kernel<T>` — grid `((n_embd+7)/8, grid_groups)`, one warp per output row `r`, `s = row_dot` over
   `n_ff` of the SwiGLU's q8_1 blocks, lane 0 stores `out[ent_dst[e] * n_embd + r] = s * sd`.

**The dot** (`row_dot<TY>`, `iq_kernels.cu:669`) is the numeric contract:

```
lane t: s_t = 0; for k = t; k < nb * ipb; k += 32:  s_t += Fmt<TY>::dot(row, x + (k/ipb)*(qk/32), kbx=(k/ipb), iqs=step*(k%ipb))
then the 32 lanes: for o in {16,8,4,2,1}: s += __shfl_xor(s, o)      // one FP32 add per step, in this order
```

with `Fmt<Q2_0> = {qk 64, ipb 2, step 1}`, `Fmt<Q3_K> = {qk 256, ipb 16, step 1}`, `Fmt<PTQ1_0> = {qk 128, ipb 1,
step 0}`. Each `dot` is an **integer** dot (dp4a) times a float scale product:

| format | per call (32 K values) | float tail |
|---|---|---|
| Q2_0 (42) | 8 two-bit fields of one int16, value = field − 1, 2 dp4a per int16 | `d2 * d8 * sumi` |
| Q3_K (11) | 16 values, inverted high-bit mask subtracts 4, `dp4a(vi,u)*sc` (int first) | `sumf += d8[i]*(int)` then `d3 * sumf` |
| PTQ1_0 (143) | the whole 128-weight block (base-3 trits), `sum(t*a) − sum(a)` per q8_1 sub-block | `d_w * Σ_k d8[k]*(int)` |

**q8_1** (`q8_1_store`, `q8_1_finite.hpp`): per 32 values, `amax` and `sum` through the same XOR butterfly,
`d = q8_1_finite(amax/127)` with `q8_1_finite(v) = |v| > 65504 ? copysign(65504, v) : v`, `q_i = amax == 0 ? 0 :
clamp(roundf(x_i/d), ±127)`, `ds = (half(d), half(sum))`.

**Scratch** (`native_expert_scratch_bytes(cap, n_ff) = 3*align256(cap*n_ff*4) + align256(cap*(n_ff/32)*36)`):
`gate = scratch`, `up = scratch + fa`, `h = scratch + 2*fa`, `hq = scratch + 3*fa`, `fa = align256(cap*n_ff*4)`.
Gate/up/h/hq are indexed by the **global entry index** `e`, so the buffers are sized for `cap_entries` even though one
call only touches its groups' entries.

**Not in the contract, refused rather than approximated**: a non-null `lora` (the routed-expert rank-1 adapter),
`up_type != gu_type` (MiMo GSQ-RCO mixed gate/up), and any (gu, d) pair outside the nine this model uses. Each prints
what it refused and exits, the same way the CUDA dispatch does for an unknown type — nothing silently computes a
cheaper arithmetic.

---

## 2. The port

`sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp`. One TU: the four block structs (with their `static_assert`s),
the three dot products, `row_dot`/`warp_sum`, `q8_1_store`, the three kernels and the entry point — the same layering as
the CUDA file, so it can be read side by side with it. Same 15-parameter signature, so `ds41_moe.cpp`'s five call
sites do not change.

**Why a second file beside `iq_kernels.dp.cpp`.** That file *has* a `native_expert_grouped` — a dpct migration of an
older `iq_kernels.cu`: no `lora` argument (so the symbol does not match today's header), no Q3_K (11) / PTQ1_0 (143)
formats, no `swiglu_limit`. It is dead code for this engine (it cannot be called), and patching a 5.8k-line migrated
file in place would have meant re-deriving what the migration had already lost. The port therefore lives in its own TU
and the report lists the exact deletions needed to fold it in (§5.1). `iq_kernels.dp.cpp` itself was left untouched.

**How the SKYL side maps onto this codebase** (the S1 shim's conventions): `void* stream` → `strata::q_of(stream)` →
`sycl::queue&`; USM pointers are the CUDA pointers; `__global__` → `q.parallel_for<name>(nd_range, item)`; the 32-lane
warp → a **sub-group of exactly 32** (`[[sycl::reqd_sub_group_size(32)]]`, verified at run time on the B70: 32, and
`local_id 37 → sub-group local id 5, group 1`); `__shfl_xor_sync` → `sycl::permute_group_by_xor` (same
snapshot-then-add semantics, so the same FP32 addition order); `__dp4a` → a 4-iteration byte-multiply loop (the form
IGC pattern-matches into DP4A); `__byte_perm` → integer byte packing and `__vsubss4` → per-byte saturating subtract
(both are integer, so an equivalent expression cannot change the result).

**Gating** is the port's own: the file exists only in the `sycl/` tree (`src/X.cu` → `sycl/src/X.dp.cpp`), the CUDA
file is untouched, and the sycl CMake project compiles it in place of the CUDA TU. Nothing in `src/` or in the CUDA
build sees this file.

**FMA contraction, explicitly off.** `#pragma clang fp contract(off)` at the top of the TU. With clang's default
`-ffp-contract=on` the device fused `sumf += d8[i] * (float)(dp4a*sc)` into an fma, and the Q3_K paths moved by up to
**4096 ULP / 3.5e-4 relative** against the same source order — an error a tolerance test would have called equal. With
the pragma the TU computes what the source writes, which is what the harness compares bit for bit. (Whether nvcc's
`-fmad=true` fused the same expression in the *real* CUDA binary is the one question this machine cannot answer — no
CUDA toolkit is installed, so the CUDA kernel could not be executed as ground truth; see §3.4.)

---

## 3. Numerical validation

### 3.1 What is compared

`sycl/src/kernels/native_expert_grouped_parity.cpp` builds, for every case, **a host reference that is a second,
independent transcription of the CUDA source's arithmetic**: the weight unpacking is scalar and indexed by K element
(not by dp4a lane), the 32-lane warp is an array with the explicit XOR butterfly, and the three float tails are written
in the source's expression order. It is not a float reimplementation: same integers, same scale grouping, same order of
every FP32 addition. Two reference evaluations are run (contracted and not) and each case reports which one the device
matched.

Comparison is **bitwise** on four things separately — so a failure is attributed to a phase, not to "the kernel":

1. `gate[e*n_ff + r]` (all `cap_entries × n_ff` floats),
2. `up[...]`,
3. `hq` (the SwiGLU's q8_1 blocks: the 32 int8 **and** both fp16 fields, compared as bits),
4. `out[ent_dst[e]*n_embd + r]`.

A third, format-level check runs beside it: `truth_dot()` dequantizes the weights (`d_w × q`) and the activations
(`d_a × q`) and sums the products in **double** — what the format *means* — so a case where reference and device agree
with each other but both misread the layout cannot pass unnoticed (this is what caught the two reference bugs in §3.3).

### 3.2 Result: 93 cases, all four phases bit-identical

```
93 cases, 9 (gate/up, down) pairs x 10 variants + 3 real-geometry cases
gate ==, up ==, hq 0 differing blocks, out ==     for all 93
ALL CASES BITWISE EQUAL                            (exit 0)
```

Adversarial cases, all bit-exact:

| case | what it stresses |
|---|---|
| `zeros` | all weight scales 0 **and** all-zeros activations (`amax == 0` → `q = 0`, `d = 0`) |
| `one group` | a single active expert (the decode shape) |
| `all groups` | every group active, 4 entries each |
| `strided` | `grid_groups = 2` with 6 possible groups (the verify window's shrink) |
| `extreme-scales` | every block scale 65504 or 6.1e-5 (fp16 max / smallest normal) |
| `subnormal-scales` | every block scale 1e-7 (fp16 **subnormal**) |
| `max-scales+huge-act` | scale 65504 **and** activations at 1e9 → the `q8_1_finite` clamp path (`d` clamped to 65504) |
| `short-K` | K not a whole number of 256-value super-blocks (Q2_0 512×384, PTQ1_0 512×384, Q3_K 256×256) |
| `v1 two-pass` | `STRATA_GROUPED_V1` (swiglu + quantize as two kernels over all cap_entries) |
| `real 5120x2304` | the model's own geometry, three classes (Q2_0/Q2_0, Q2_0/Q3_K, Q2_0/PTQ1_0) |

**The SwiGLU's `__expf` did not matter**: `hq` is bit-identical in every case, i.e. the host's `exp` and the device's
`sycl::exp` differ by less than the q8_1 quantizer's resolution on all of these inputs — the 8-bit activation rounding
absorbs it. That is a statement about these inputs, not a proof; a pathological block whose `h/d` sits exactly on a
`.5` boundary could still flip one int8.

### 3.3 What the harness caught (why the comparison is trustworthy)

Four real bugs, three in the harness and **one in the port**, each found by a probe rather than by guessing:

1. **The port**: the Q2_0 dot packed its four int8 selectors with `-1 | (b<<8) | …`. `-1` is `0xFFFFFFFF`, so every
   byte of the packed selector became −1: `gate` was wrong wherever a 2-bit field was 0, while an all-`+1` probe passed
   (a permutation of equal values is invisible). Fixed with an explicit byte-mask (`pack_i8x4`) and explained in the
   source. A 1e-3 tolerance test would never have seen this.
2. The reference read q3_K's `qs`/`hmask` words at byte `2*iqs`; the CUDA `get_int_b2(x,i)` is the 32-bit word at
   16-bit index `2*i`, i.e. byte **4*i** (`iq_kernels.cu:35`). iqs = 0 matched, 15/16 calls did not.
3. The reference used a *logical* shift where the CUDA source's `int >> int` is arithmetic (matters for the inverted
   hmask's top bits).
4. The ground-truth dequantizer copied 16 qs bytes into a `uint64` (stack overwrite) — found because truth, reference
   and device disagreed three ways.

### 3.4 The one honest caveat

Bit-exactness is proven **against the CUDA source's written operation order**, not against a CUDA *binary*: this
machine has no CUDA toolkit (`nvcc` absent; the RTX 2080 is unusable for it), so the CUDA kernel could not be run as
the oracle. The places where a compiler is free to differ are exactly two, and both are now pinned or measured:

- the three float tails' fma contraction → eliminated by `-ffp-contract=off`; the harness still prints what contraction
  would have moved (`≤ 4096 ULP / 3.5e-4 relative`, Q3_K only, none for Q2_0/PTQ1_0);
- `__expf` vs `sycl::exp` → absorbed by q8_1 on every case run, but not proven for every input.

Running `src/kernels/iq_multi_parity.cpp` / `native_grouped_parity.cpp` on a CUDA machine and diffing the same inputs
against this port is the remaining check, and it needs no code change: the harness already generates and saves nothing
platform-specific.

---

## 4. Throughput, at the model's geometry (5120 × 2304)

B70 (`level_zero:0`), icpx 2026.2, `-fsycl -O2`; median of 50 calls after 5 warmups; CPU baseline = the same reference
arithmetic (scalar, no intrinsics) on 16 threads, best of 3. `speedup` = CPU/SYCL. A second SYCL block is printed in
parentheses in the log as a clock/throttle stability check (the CPU baseline runs between the two blocks, which costs
the GPU some clock — the two blocks agree to ~10-20 %).

| shape | (gu/down) | SYCL | GMAC/s | GB/s | CPU 16T | speedup |
|---|---|---|---|---|---|---|
| decode: 6 groups × 1 entry | Q2_0/Q2_0 | **234 µs** | 908 | 255 | 14.4 ms | 61× |
| decode: 6 groups × 1 entry | PTQ1/PTQ1 | **336 µs** | 632 | 138 | 13.5 ms | 40× |
| decode: 6 groups × 1 entry | Q3_K/Q3_K | **488 µs** | 435 | 187 | 48.6 ms | 100× |
| decode: 6 groups × 1 entry | Q2_0/Q3_K | 324 µs | 655 | 217 | 33.2 ms | 102× |
| decode: 6 groups × 1 entry | Q3_K/Q2_0 | 442 µs | 481 | 183 | 36.6 ms | 83× |
| decode: 6 groups × 1 entry | PTQ1/Q2_0 | 293 µs | 725 | 174 | 16.8 ms | 57× |
| prefill 8 tok: 42 groups × 48 routes | Q2_0/Q2_0 | **1758 µs** | 846 | 238 | 94.8 ms | 54× |
| prefill 8 tok: 42 groups × 48 routes | PTQ1/PTQ1 | 2537 µs | 586 | 128 | 93.9 ms | 37× |
| prefill 8 tok: 42 groups × 48 routes | Q3_K/Q3_K | 2803 µs | 530 | 228 | 408 ms | 146× |
| verify window: 6 groups × 8 entries | Q2_0/Q2_0 | 1611 µs | 132 | 37 | 109 ms | 68× |

Reading the numbers:

- **Weighted by the model's class mix, one decode layer costs ≈ 373 µs** of expert kernels for its 6 experts; 40
  layers = **≈ 15 ms/token, i.e. a ~67 tok/s ceiling for the expert tier on this card** — if every expert blob were
  already resident. That is the number that says the port is worth continuing: today's `--experts cpu` decode is
  ~1 tok/s and its expert share is ~1 s/token, so the kernels would stop being the bottleneck by two orders of
  magnitude.
- **It is bandwidth/population bound, not ALU bound**: 200-255 GB/s against a card whose copy peak is ~450-550 GB/s,
  at 0.4-0.9 TMAC/s against an INT8 peak in the tens of TOPS. Two known levers, both absent from this beachhead: the
  CUDA's entry-reuse kernels (`native_gu_multi_kernel`, 4 entries decoded per weight pass) — visible in the table as
  the **6.9× per-MAC drop** of the 8-entries-per-expert row (132 vs 908 GMAC/s) — and the fork's PROM kernels.
- **The Q3_K rows are the slowest per byte** (435 GMAC/s at 187 GB/s vs Q2_0's 908 at 255): its block is 110 B per 256
  values with a 4-way inner loop, and the port reproduces the CUDA's per-call structure rather than batching it.
- The CPU baseline is a *conservative* one: Strata's production CPU expert kernels are AVX2/AVX-VNNI
  (`src/kernels/cpu/native_expert.cpp` + `q2_avx2`/`iq_avx2`/`kq_avx2`/`ptq1_0`), worth an estimated 3-6× over the
  scalar reference at these formats, i.e. the SYCL/CPU ratio for the real fallback is more like **8-35×**, not 40-146×.
  Its absolute number (12-16 GMAC/s on 16 threads) is also *not* what `--experts cpu` delivers end to end: that path
  is disk-bound (S3: 6.9-11.6 s of file wait in a 13 s prefill).

---

## 5. What remains before a GPU-expert tier can run

Ordered by what unblocks the next measurement. The first item is smaller than the S3 report's "10 genuinely unported
CUDA kernels" suggests, and the reason is a finding worth recording:

### 5.1 The blocking link was never given the ported library (evidence in `s2/engine-obj/link_gpuexp2.log`)

The failed GPU-expert link lists exactly these undefined symbols:

```
native_q8_1_bytes (10), native_quantize_q8_1 (9), native_expert_grouped (7), native_expert_scratch_bytes (3),
weighted_rows_sum (2), decode_rows_sum (2), scatter_rows_f32 (1), native_expert_supported3 (1),
native_expert_layout3 (1), native_expert_layout (1)
```

Six of the ten **are already defined in the sycl tree today**: `native_expert_layout` / `native_expert_supported` /
`native_expert_scratch_bytes` in `iq_kernels.dp.cpp`, `native_quantize_q8_1` / `native_q8_1_bytes` in
`native_mmvq.dp.cpp`, `scatter_rows_f32` in `elementwise.dp.cpp`. They were undefined because **no SYCL
`strata_kernels` archive was in that link at all** (the CPU-expert flavour excludes `ds41_moe.o`, so nothing pulled
them). So:

1. Build a SYCL `strata_kernels` (the `sycl/` CMake project, or a hand-picked archive of the TUs the engine needs) and
   put it in the link. Then the genuinely missing pieces are **four fork additions**:
   - `native_expert_layout3` / `native_expert_supported3` (`fork/fork_layout.cuh`, 25 lines of host arithmetic,
     `up_type`/`up_row` handling) — needed by `ds41_moe.cpp:1864-1889` at tier init, even for this model where
     `up_type == gu_type`;
   - `weighted_rows_sum` (the prompt chunk's mix, `fork_weighted_sum.cuh:8`, 10-line kernel);
   - `decode_rows_sum` (the decode mix, `fork_weighted_sum.cuh:29`, 10-line kernel, `double` accumulation — emulated on
     an Arc, but K=6 × 5120 ops, so irrelevant);
   - plus `native_quantize_q8_1` / `native_q8_1_bytes` if the chosen archive's `native_mmvq.dp.cpp` is excluded — the
     quantizer is `q8_1_store` over `n_in × ncols` contiguous elements, which this TU already has.
2. **Resolve the duplicate**: this TU and `iq_kernels.dp.cpp` both define `native_expert_grouped` and
   `native_grouped_set_v1`. Delete from `iq_kernels.dp.cpp` the stale `native_expert_grouped`, `native_grouped_set_v1`,
   and the kernels only they use (`native_gu_kernel`, `native_gu_multi_kernel`, `native_down_kernel`,
   `native_down_multi_kernel`, `swiglu_entries_kernel`, `swiglu_q8_1_entries_kernel`, `quantize_q8_1_kernel`,
   `launch_gu*`/`launch_down*` and the `s26`/`s27` AMD variants). `native_expert_layout`/`supported`/`scratch_bytes`,
   `is_iq`, `iq_row_bytes`, `iq_supported` stay there (other entry points use them).
3. Relink `ds41_moe.o` (the GPU-expert flavour — **not** `ds41_moe_cpu.o`, and not `ds41_helper.o`) with the archive,
   then run with `--experts gpu` (the default; `--experts cpu` is what sets `cpu_only`), `--rank rank.bin`,
   `--slots auto|N`, `--backend cuda`, `STRATA_ROUTED_HOST` **unset**.

### 5.2 Arena / pager: the machinery exists, it is switched off by `cpu_only`

"VRAM hits = 0" is not a missing arena — `Ds41MoeConfig` (`ds41_moe.hpp:280-330`) already carries `slots` (2150),
`arena_gib` (72.6), `pcie_frac` (0.25), `pf_b` (1.43), `arena_adapt`, `defer_cache`, `slot_gib`; `build_arena` +
`seed_from_routes` fill the host arena and the VRAM cache from the routing profile, and `ds41_generate.cpp:584` sets
`cpu_only = (--experts cpu)`, which is what turns all of it off. The same run's tier line is the "before" number:

```
prompt: 8 tokens in 8.88 s = 0.9 tok/s (experts 0 hit / 1920 cpu / 0 pcie / 1920 file)
prompt split: dense 4.1 s, experts 4.0 s (file wait 3.5 s)
decode: ... experts 0 hit / 0 cpu / 0 pcie / 0 file (0.0% VRAM hits)
``` What the GPU tier needs beyond §5.1:

- **The slot budget has to be re-derived for this model.** The 2150 × 6.75 MiB figures are the Qwen/DS4 ones; a
  DS V4.1 expert blob is **7.7-15.2 MB** (§0), so `slots × blob + dense half` will not fit in 32 GB at 2150. Use
  `slot_gib` (per-expert sizing exists for non-uniform geometries) or `--slots ~1200-1600` for ~12-18 GB, and expect
  the cache to hold a minority of the 15360 (layer, expert) pairs — the routing profile (`rank.bin`, which the model
  ships) is what makes that minority the right one.
- **Then the per-token weight traffic decides the speed**: 2.6 GB/token from VRAM at the measured 200-255 GB/s is
  ~10-13 ms/token, i.e. the ceiling above; anything served from the RAM arena or the file appears on top of it.
  Making that traffic smaller is a *quantization* decision (already made by the model), making it faster is §5.4.
- **The device-side host plumbing** (`ds41_moe.cpp`'s 47 CUDA calls, the PCIe share, the pinned arena handshake) is S1
  shim territory and already compiles; it has only ever been exercised on the `cpu_only` side, so the first GPU-tier
  run should be watched with the mem watchdog and `--slots` small.

### 5.3 Correctness of the surrounding tier, not of this kernel

`ds41_moe.cpp`'s GPU path was never run, so its own S3-style bugs (like `get_rows`) are unfound. The cheap first test
is a one-layer, one-token A/B: `--experts gpu` vs `--experts cpu` on the same `--ids`, comparing `--dump-last` logits,
then the same with `STRATA_LAYER_DUMP` if they differ.

### 5.4 Performance (after it runs at all)

In expected-value order: (i) port `native_gu_multi_kernel` + `native_down_multi_kernel` (4 entries per weight pass —
the 6.9× per-MAC gap in §4); (ii) the fork's PROM variants for PTQ1_0/Q2_0 (one weight-tile decode per (row, group)
applied to every entry — the verify-window shape); (iii) the Q3_K dot's inner loop (its call does 4 sequential
`dp4a·sc` products; the CUDA source's shape is kept here); (iv) only then look at the `__byte_perm` emulation, which
measured as *not* the bottleneck (an alternative `spread4` + `sum(a)` integer form was within noise: 234 → 234 µs).

### 5.5 Deliberately not ported (each refuses loudly)

`NativeExpertLora` (the GLM abliteration adapter — `ds41_moe.cpp` passes 14 arguments, so `lora == nullptr` on every
call site of this engine), mixed gate/up types (MiMo GSQ-RCO), NVFP4 (type 40; the `tail_off` scaling *is* implemented),
the AMD `STRATA_EXP_MODE` layouts / LDS / fused-XMX kernels, `STRATA_EXPERT_V2`/`V2K` (S26/S27), `STRATA_PROM`, and
`STRATA_OLD_IQ_MMVQ`/`STRATA_NO_SUB16_GU`/`STRATA_TSUM`. `STRATA_GROUPED_V1` **is** honoured (two-pass SwiGLU + q8_1).

---

## 6. Files, and what was not touched

- **Added** (committed): `sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp`,
  `sycl/src/kernels/native_expert_grouped_parity.cpp`, this report.
- **Untouched**: every CUDA source (`src/kernels/cuda/*`), the whole `src/` tree, `iq_kernels.dp.cpp`, the sycl
  CMake project (the new TU is not yet in its source list — see §5.1 step 1), `/home/dyy/Strata`.
- **The working CPU-expert binary is intact and backed up**: `/home/dyy/deepseek-work/s2/engine-obj/ds41_generate_cpuexp`
  (93 480 128 B, sha256 `8695f2ee…dac592a`), copied to
  `/home/dyy/deepseek-work/s4/backup/ds41_generate_cpuexp.cpu-expert.bin` (same hash). Its object directory
  (`s2/engine-obj/`) was not rebuilt.
- Build line (no AVX-512 anywhere; the host is an i9-13900K):

```
source /opt/intel/oneapi/setvars.sh --force
R=/home/dyy/deepseek-work/.xolotl-ref
icpx -fsycl -O2 -std=c++20 -I$R/sycl/include -I$R/include \
     -c $R/sycl/src/kernels/cuda/iq_kernels_native_expert.dp.cpp -o s4/ne_kernel.o
icpx -fsycl -O2 -std=c++20 -I$R/sycl/include -I$R/include \
     -c $R/sycl/src/kernels/native_expert_grouped_parity.cpp -o s4/ne_parity.o
icpx -fsycl -O2 -o s4/ne_parity s4/ne_parity.o s4/ne_kernel.o
ONEAPI_DEVICE_SELECTOR=level_zero:0 s4/ne_parity --both     # 93 cases + the table in §4
```

`--validate` alone exits 0 iff every case is bitwise equal; `NE_DBG=1` dumps per-case values against the ground truth,
`NE_SELFTEST`/`NE_SELFTEST2`/`NE_SELFTEST3`/`NE_SELFTEST4` are the pairing/value/block probes used in §3.3.
