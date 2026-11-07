# DeepSeek-V4.1 on an Intel Arc: the `strata-dsv41` SYCL port (work log)

State: **the engine builds and its i-quant expert kernels are numerically exact on an Intel GPU; the card we target
(Arc Pro B70) is wedged and needs a reboot before the next run.** Nothing here is upstreamed yet.

Branch/tree: `/data/strata-new`, `origin/main` @ `46c152a1` (Strata 0.1.53), working tree dirty (the port).
Toolchain: oneAPI 2026.2 (`icpx`), dpct (SYCLomatic) 2025.3, CUDA 12.8 headers from the pip wheels.

---

## 1. What is done

| step | result |
| --- | --- |
| Upstream tree | `/data/strata-new` was a sparse checkout of `4b9e05a`; expanded to `46c152a1` with `docs src sycl include third_party tools tests bench cmake packaging data metal` |
| dpct | `intel-oneapi-dpcpp-ct-2025.3` from Intel's apt pool, `dpkg-deb -x` into `~/deepseek-work/dpct` (no root); CUDA headers from `sycl/tools/get-cuda-headers.sh` |
| CUDA -> SYCL migration | all 10 dsv41 translation units: `ds4_{kernels,iq,gguf,model,tq,vision}.cu` + `vendor/{exl3,exl3_legacy,sampler,moe_win}.cu` -> `sycl/src/dsv41/*.dp.cpp` (115 kernels) |
| The Qwen engine (baseline) | builds and runs: `build-sycl/strata` -> `strata 0.1.53` (AOT `bmg-g31`) |
| The DeepSeek engine | builds: `build-sycl/strata-dsv41` -> `strata-dsv41 0.1.53 build sycl-port` |
| its kernel gate | `build-sycl/ds4_iq_test` (upstream's `tests/dsv41/iq_expert_test.cpp`, ported through a SYCL branch of `include/strata/dsv41/hip_or_cuda.h`) |

Build (either engine; `STRATA_SYCL_AOT=bmg-g31` for the card, empty for SPIR-V/JIT):

```sh
source /opt/intel/oneapi/setvars.sh --force
cmake -S sycl -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
      -DSTRATA_SYCL_AOT=bmg-g31 -DSTRATA_SYCL_PARITY=OFF \
      -DSTRATA_GGML_DIR=<a llama.cpp checkout>
cmake --build build-sycl -j 12 --target strata-dsv41 ds4_iq_test
```

## 2. The migration recipe (reproducible)

```sh
# once per upstream update, with dpct on PATH and CUDA_HEADERS=<headers>/include
MIGRATE_ONLY="src/dsv41/ds4_gguf.cu src/dsv41/ds4_iq.cu src/dsv41/ds4_kernels.cu src/dsv41/ds4_model.cu \
              src/dsv41/ds4_tq.cu src/dsv41/ds4_vision.cu src/dsv41/vendor/exl3.cu src/dsv41/vendor/exl3_legacy.cu \
              src/dsv41/vendor/sampler.cu src/dsv41/vendor/moe_win.cu" \
  bash sycl/tools/migrate.sh <repo> <repo>/sycl
python3 sycl/tools/port_dsv41_fixes.py            # every hand edit the port needs, idempotent
```

`port_dsv41_fixes.py` re-applies the port on top of a fresh dpct run (dpct rewrites the headers it walks, so this must
follow every migration). What it does, and why:

1. **`ds4_launch.h`** - the engine's one launch spelling. dpct turns `<<<grid, block, shared, stream>>>` inside this
   header into a broken `kernel_launcher::launch` call with 15 copies of one argument; the port writes a SYCL branch that
   launches through `dpct::kernel_launcher` (which is what the generated wrappers expect: they read its thread_local
   queue and nd_range and submit their own `parallel_for`).
2. **`dpct/util.hpp`** - `get_tangle_group` / `get_fixed_size_group` are not in this compiler; the masked sub-group
   shifts use the sub_group itself (every kernel is built with `-fsycl-default-sub-group-size=32`).
3. **`ds4_numerics.dp.hpp`** - BF16/E4M3 in plain arithmetic (the source's HIP spelling, same bits), no `cuda_bf16.h`.
4. **`sycl/include/strata/dsv41/{exl3,moe_win}.hpp`** - `cudaStream_t` -> `dpct::queue_ptr`.
5. **`sycl/src/dsv41/vendor/exl3_cpu.cpp`** - `__builtin_cpu_init()` only in the host pass of a `-fsycl` TU.
6. **`sycl/src/dsv41/ds4_packinfo.cpp`** - the migrated header instead of the one that includes `cuda_runtime.h`.
7. **ggml's codebooks** (`iq2xxs_grid`, `iq3s_grid`, `kvalues_iq4nl`, `tq_lut_f16`, ...) are `static const` HOST arrays.
   `ds4_launch.h`'s `ds4_dev_table()` makes one device copy of each and hands the kernel that pointer. **This was the bug
   that made every sum wrong** (and hung the card): passing `&array[0]` gave the kernel a host address.
8. **hoisting** - the generated wrappers fetch those pointers inside their command group; a queue operation there is
   `sycl::queue::submit cannot be nested`, so they are hoisted above `queue.submit(...)`.
9. `__nanosleep` -> `ds4_nanosleep` (a relaxed device fence), dpct's `dpct_placeholder` -> 0, no kernel names (a templated
   wrapper's lambda is already its own type; dpct reuses one name and the device stub collides),
   `cudaHostRegister*` -> 0 with the zero-copy store off (`DS4_STORE_ZC`), the vision teardown's queue cast.
10. **the expert ring is USM shared** - the CPU lane's `IoQueue` reads expert rows straight into the ring (the "direct"
    flow, on by default) and the kernels then read it; a SYCL *device* allocation is not host-writable, so that one buffer
    is allocated with `sycl::malloc_shared` (`dalloc_shared`).  Unverified on the card yet - the first load will say.
11. **`STRATA_DSV41_NO_TC=1`** - the tensor-core expert kernels (`nvcuda::wmma` -> dpct joint_matrix) compile to
    `__builtin_spriv_OpJointMatrixLoadINTEL_...` builtins this compiler does not define. They are compiled out and the
    plain F32 kernels serve the prompt (same results, slower prompt). Writing them against
    `sycl::ext::oneapi::experimental::matrix` is the next piece of work.

## 3. Evidence: the expert kernels are exact

`build-sycl/ds4_iq_test --type 16` (IQ2_XXS, the type our model's experts are), on the **UHD 770**:

```
IQ2_XXS token gate_up          worst 5.21e-08 (tol 1e-05) ok
IQ2_XXS token down             worst 3.59e-08 (tol 1e-06) ok
IQ2_XXS set gate_up (f16)      worst 7.02e-03 (tol 1e-02) ok
IQ2_XXS set down               worst 4.19e-04 (tol 3e-03) ok
IQ2_XXS cpu gate_up vs ref     worst 5.59e-08 (tol 1e-05) ok
IQ2_XXS cpu down vs ref        worst 3.35e-08 (tol 1e-05) ok
ALL OK
```

That is upstream's own gate: the card's i-quant decode + dp4a sums against ggml's decode of the same bytes plus a double
reference. Before fix 7 the same run was `3.56e-02 / 4.41e-01` - i.e. wrong - and on the B70 it hung the card.

## 4. The B70 incident (2026-10-28 23:58) - read this before the next run

* Three kernel runs were made on the B70. The first two (before fix 7) hung a kernel; the driver reset the engine,
  and after the third reset it logged
  `CRITICAL: Xe has declared device 0000:03:00.0 as wedged. device wedged, needs recovery`.
* `sycl-ls` then shows only the UHD 770; `/dev/dri` is intact; **no process holds the render nodes**; no sudo here, so
  the PCI remove/rescan recovery cannot be run by the agent.
* **Recovery: reboot.** After the reboot `sycl-ls` should list `level_zero:0 ... Arc(TM) Pro B70` again.
* The two hang incidents each produced ~150 `guc_exec_queue_timedout` lines and recovered on their own; the third did
  not. A hung kernel cannot be cancelled: `timeout(1)` cannot kill a process blocked in the driver (it ends as a
  zombie and the device stays wedged), so **do not start a B70 kernel run you cannot afford to lose**.

### Why the B70 and not the iGPU

The binary is AOT-compiled for `bmg-g31` *and* carries SPIR-V. The iGPU run almost certainly took the SPIR-V path, the
B70 run the AOT image, so the suspicion is on the AOT codegen for these kernels (the `group_local_memory_for_overwrite`
+ `use_root_sync` pattern dpct produces) rather than on the kernels themselves. **Next run: the JIT build** (below).

## 6. The model pack (done, host-side, no GPU)

`--pack-info` reads a pack's whole description without touching a device, so the pack can be validated before any run.

* **Engram stub**: the engine refuses a pack whose Engram layer has no `engram.embed.weight`, and only reads that table as
  MXFP8 / Q6_K / IQ4_NL. Our IQ2_XXS set carries the hash metadata and the small per-layer `engram_{k,q,wkv}` but not the
  rows; the Q4_K_M set's rows are Q4_K, which the reader does not take. `tools/dsv41/make_engram_stub.py` writes a
  **sparse** GGUF with the true shape and Q6_K rows of zeros (150.2 GiB apparent, **8 KiB on disk**): every row reads as
  zeros, so the Engram adds nothing (degraded answers, the whole engine path otherwise real). Swap in the official
  `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` (111 GB) when quality matters.
* **Pack**: `pack_gguf.py --model /data/models/dsv41-uncensored-iq2xxs --engram <stub> --out /data/models/dsv41-iq2xxs-pack`
  -> 47,246 tensors in 5 files. `strata-dsv41 --pack-info` then reports:

```
40 layers x 384 experts, 6 used; dim 5120, ff 2304, vocab 129280
expert tensors by type: IQ1_M 60 IQ2_XS 18 IQ2_XXS 42
an expert 8.60 MB, 6 picks a token: 2.064 GB a token
all experts 132.1 GB at 1.94 bpw (544 G parameters)
dense types: BF16 F32 Q5_K Q6_K Q8_0
engram layers: 1,14
no GPU kernel: none (every tensor is a device type)
files: 5, all present and long enough
```

So: **every expert type this model uses has a device kernel in the port**, and a token touches 2.064 GB of experts. That
last number is the speed budget: on a 32 GB card ~13 GB of experts stay in VRAM, so ~1.8 GB a token comes from RAM
(measured DDR5 bandwidth here ~30-40 GB/s) -> a ceiling around 15-20 tok/s before compute, which is the 10+ the goal asks
for. The pack plus `/data/models/dsv41-iq2xxs-pack/launch-ds41.sh` (`start|bench|status|logs|stop`) are ready.

## 5. Next steps, in order

1. After the reboot: `build-sycl-jit/ds4_iq_test --type 16` on the B70 (`-DSTRATA_SYCL_AOT=""`). If it passes, the AOT
   image is the fault: compare the two, and keep JIT as the default until the AOT issue is isolated.
2. Then the model, which is packed and validated already (section 6): `launch-ds41.sh bench` for the 16-token smoke, then
   `start`. Expect the first token to take minutes (the CPU lane fills its RAM store); watch `run.log` and, above all,
   the kernel log for a reset.
3. The tensor-core expert kernels (`set_tc_kernel`, `mx_set_tc_kernel`) against
   `sycl::ext::oneapi::experimental::matrix`, then `STRATA_DSV41_NO_TC` off (prompt speed).
4. Upstream: the port is `sycl/` + `sycl/tools/port_dsv41_fixes.py`; `docs/INTEL.md` still says "DeepSeek V4.1-Flash
   does not run on Intel", which is what this work changes.

### If a kernel hangs: two switches, no extra reboot per guess

A hung kernel cannot be cancelled, so the ONLY cheap way to find it is before it hangs:

```sh
DS4_LAUNCH_LOG=1 <engine or test>      # prints "ds4 launch <file>:<line> grid=... block=..." and flushes, every launch
                                       # after a reset, the last line in the log is the kernel that hung
DS4_SKIP_LAUNCH="ds4_iq.dp.cpp:1985"   # a comma list of file:line substrings: those launches become no-ops, so a
                                       # bisect can continue in the SAME process (the output is then wrong on purpose)
DS4_SYNC_LAUNCH=1                      # wait after every launch and name the first one that raises
```

For a cheap first light of the model (less RAM to fill, fewer expert reads a token):

```sh
bash /data/models/dsv41-iq2xxs-pack/launch-ds41.sh bench --experts 2 --max-context 2048
```
(`--experts N` clamps the experts used per word; the pack's full setting is 6.  It is a smoke test, not an answer.)

## 9. The speed budget, measured (no GPU needed)

`tools/membw.c` (20 threads, 6 GiB, 64-byte stride) reads this machine's RAM at **40.6 GB/s**.  The pack says a token
touches **2.064 GB** of experts; a 32 GB card keeps the dense weights (~17 GB) and ~13 GB of experts, so ~1.8 GB of the
2.064 comes from RAM: **1.8 / 40.6 = 44 ms -> a ~22 tok/s bandwidth ceiling** before compute, attention and the Engram
rows.  Half of that after the CPU lane's dp4a work is the realistic expectation, i.e. **10-15 tok/s** - the same class as
Strata's own report of 8-10 tok/s on an RTX 3060 (12 GB) with 46 GB of RAM, and the 10+ the goal asks for.

Passing tests in the port today (no device): `ds4_slot_test` "all passed", `ds4_experts_rule_test` "all checks passed",
`ds4_pick_rule_test` "prompt path rule: ok"; `ds4_iq_test` (the device gate) passes on the UHD 770 and is unverified on
the B70 until the reboot.

## 10. First light (2026-10-29): the engine answers, correctly, on the Arc Pro B70

```
$ strata-dsv41 --pack /data/models/dsv41-iq2xxs-pack-real --gpus 0 --ids @ids.txt --n 16 \
               --window 16 --experts 2 --mmap-experts on --ssd-cache 0 --dspark off --vision off
prompt: 7 tokens  ·  decode: 16 tokens in 4022 ms (3.73 tok/s)
ids: 3011,14,396,734,554,1748,16,455,6102,294,8760,344,11111,16,455,6102
text: " No, that's not right. The capital of France is Paris. The capital"
```

The question was "What is the capital of France?" and the answer names **Paris** - upstream's own smoke criterion
(`tools/dsv41/smoke.py`) is met by reading the text, not by the exit code.

### The bug that had it talking nonsense: a pair of parentheses

`ds4_iq.cu` computes a K-quant sub-block's dot as `v - (m0*sl + m1*sh)`; dpct wrote **`v - m0*sl + m1*sh`**, so the high
half's min term was ADDED instead of subtracted. Q4_K and Q5_K were therefore wrong (Q6_K has no min term, which is why it
was the only K-quant that looked right); with `dmin = 0` the error vanished, which is how it was found.

What it cost in the model: layer 27 is the first layer whose shared expert's **down** matrix is Q5_K (25/26 use Q8_0), and
its output jumped to rms 336 (2.6 a layer before); the residual stream then doubled a layer to rms 118k by layer 39, and
because `head.weight` is Q5_K too, every token came out as the same special id with logits in the 74..986 range.

`ds4_dense_mv_test` (new, port-owned) pins it: a random legal matrix and the model's own blocks, the device's `iq::mv`
against ggml's decode plus a double reference - `Q6_K 1.6e-8 ok`, `Q4_K 6.6e-9 ok`, `Q5_K 5.5e-9 ok` after the fix (Q4_K
and Q5_K were 7.5e-3 and 6.1e-3 before, and 12-40 % on the model's real blocks).

### Still open from that run

* `--experts 2` answers correctly; `--experts 3/4/5/6` emit a single special token (the router/MoE sum path at more than
  two picks is the next bug - the expert count changes `ne`, and the engine's `gu_ne`/`dn_ne` NE ladder is not the
  discriminator, since 2..4 all take NE = 4).
* The easy decode rate in that configuration is 3.73 tok/s (experts read from NVMe). The RAM-store configuration
  (`--ssd-cache 100`, 75.6 GB read in 23.9 s) reported `0.0% of the CPU lane's experts in RAM`, so the store and
  `--mmap-experts on` do not combine: `--mmap-experts auto` is the one to measure with.
* Every run since the reboot: **0 engine resets, 0 wedges**.

## 11. The goal, measured (2026-10-29, after the min-term fix)

A **chat-template** prompt (`jinja2` renders the pack's `chat_template.jinja`, then the pack's own tokenizer):
`<｜User｜>What is the capital of France?</think>`, 6 experts a word (the pack's full setting):

```
$ strata-dsv41 --pack /data/models/dsv41-iq2xxs-pack-real --gpus 0 --ids @chat_ids.txt --n 32 \
               --window 16 --experts 6 --threads off --prefill-chunk 0 \
               --mmap-experts auto --ssd-cache 100 --dspark off --vision off
RAM store: 9734 of the profile's hottest experts read in (75.6 GB) in 24.9 s
loaded in 34.5 s
prompt: 10 tokens in 218 ms (45.78 tok/s)
decode: 9 tokens in 372 ms (21.49 tok/s)      ->  "The capital of France is **Paris**."
```

* **21.5 tok/s decode, 45.8 tok/s prompt, 6 experts (full quality), correct answer.**
* A 64-token request answered and stopped at the end token: `9 tokens in 312 ms (25.63 tok/s)`, prompt 43.79 tok/s,
  same answer.  Three independent runs: **21.49 / 20.17 / 25.63 tok/s**, every one of them 6 experts and correct text.
* Consecutive identical runs without the store (`--mmap-experts on --ssd-cache 0`, the head served from the page
  cache): 12.48 then **20.17 tok/s** - the first run pays the head's 455 MB read (213 ms a token), the second does not.
  So the head GEMV is I/O bound on the model file, not compute bound: with the store warm it is not the bottleneck.
* The earlier "--experts 3/4/5/6 only emit one special token" was **not a bug**: without a chat template the prompt is
  plain text, the model just continues it and can end the sequence (token 1) at once. With the template, 2 and 6 experts
  both answer "Paris".

## 12. The big chunk works, and 8K prompts run at 268 tok/s (2026-10-29, after the bounded ring wait)

The 4K-and-up prompt used to stall for good: `flow_pump` waited on an expert ring slot with
`event::wait_and_throw()`, which on this driver never returned (GPU pinned at 2800 MHz, host idle, no
kernel-log message, SIGTERM ignored by the engine, so only a reboot freed the card).  Replacing that
wait with a status poll plus a bound (`DS4_RING_WAIT_MS`, 180 s default) both fixed the stall and
turned any future one into a clean error naming the slot.

| prompt | chunk | time | rate |
|---|---:|---:|---:|
| 3,991 tokens | 8192 | 19.83 s | **201.30 tok/s** |
| 7,981 tokens | 8192 | 29.76 s | **268.15 tok/s** |

Where the 8K prompt's 29.8 s goes (measured by ablation, `--experts 1` vs `6`, both at chunk 8192):
the dense half is ~25.5 s (3.19 ms a token, the same per-token cost the xolotl-sycl port measured for
its dense half: 26.4 s at 8192), the six routed experts ~4.3 s - the chunk amortises the 132 GB expert
read over all its rows, which is the effect that makes a big chunk worth it.

`--prefill-chunk 16384` does NOT work: it stalls in a wait this change does not cover (the same
signature as the old ring stall, from a different call site).  8192 is the largest chunk measured to
finish.

Diagnostics added for this: `DS4_FLOW_TRACE=1` (each item the flow takes, the layer, the expert, the
ring slot) and `DS4_RING_WAIT_MS`.  Both are opt-in and cost nothing when unset.

## 13. Where the 8K prompt's 30 s actually goes (2026-10-29, measured by ablation)

| 8K prompt, chunk 8192, 6 experts | time | note |
|---|---:|---|
| baseline | 29.76 s (268.15 tok/s) | |
| `--vram-experts 1.0` | 29.51 s (270.44) | the planner still leaves 12,850 experts on the CPU lane |
| `--vram-experts 0.3` + 4 GB ring in VRAM (`DS4_RING_DEVICE=1 DS4_RING_MB=4096`) | 29.80 s (267.86) | the ring path runs, but it carries only the VRAM set |
| `--experts 1` | 14.01 s (569.52) | ⇒ 5 extra experts cost 15.75 s |
| `--cpu-threads 1` (the lane keeps its experts, one worker) | >8.4 min, killed | ⇒ the lane is ~95 % of the expert phase |

**The bottleneck is the CPU lane, not the link and not the matrix engines.**  The engine's own docs say it: the
experts that do not fit in graphics memory "are computed on the CPU, read in place from the model files"; the
planner assigns `n_exp - n_vram` to the lane unconditionally (ds4_model.cu, the placement print), and with 32 GB of
VRAM only 620-2,500 of 15,360 experts get a slot (4-16 %), so ~121 GB of expert weights are multiplied by 24 AVX2
threads every chunk (the i9-13900K has no AVX-512).

Two measurements that close the other candidates:

* PCIe is healthy: a 1 GiB pinned copy does **37.77 GB/s host->device and 51.11 GB/s device->host** (60 % / 81 % of
  PCIe 5.0 x16's 63 GB/s a direction).  The endpoint's `2.5 GT/s x1` link registers are wrong on this hardware
  (intel/compute-runtime#980 says so and measured 14.41 GB/s behind the same switch); pageable host memory costs
  another 25 % (28.15 GB/s).
* XMX is already in use: oneMKL's fp16 GEMM does **108.8 TFLOPS** with fp32 readout and **123.3 TFLOPS** with fp16
  out (2048x2048x5120), i.e. matrix-engine rates even with `compute_type::f32`.

So 268-270 tok/s is this engine's ceiling at 8K **on a 32 GB card**: dense ~11 s (1.36 ms a token) plus the lane's
~18.9 s.  Getting past it needs the routed experts on the GPU - the design xolotl-sycl reached in its step 5/11
(arena in USM host + direct async DMA, the lane's share moved to the card) - which the `DS4_RING_DEVICE` scaffold
here does not do yet: its ring carries the VRAM set only.

## 7. Runbook for the first run after the reboot

```sh
source /opt/intel/oneapi/setvars.sh --force

# 0. is the card back?
sycl-ls | grep level_zero                    # expect level_zero:0 ... Arc(TM) Pro B70, level_zero:1 UHD 770
journalctl -k --since "1 hour ago" | grep -cE 'declared device|guc_exec_queue_timedout'   # expect 0

# 1..5 in one go, gated on the kernel log after every step (stops at the first reset):
bash /home/dyy/deepseek-work/llm-for-glm-flash/verify-ds41.sh            # probe, iGPU kernels, B70 kernels, pack
bash /home/dyy/deepseek-work/llm-for-glm-flash/verify-ds41.sh --model    # ... and the first model load (16 tokens)

# 2. OUR kernels, JIT build (the AOT image is the suspect): ONE run, watch it
cd /data/strata-new
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl-jit/ds4_iq_test --type 16
#    expect the six lines of section 3, "ALL OK"
#    if it hangs: the driver resets, the process turns into a zombie and the card wedges -> reboot, then look at the
#    AOT/JIT difference (start with STRATA_SYCL_SPIN_MAX and STRATA_SYCL_I8_ESIMD=OFF, and compare the two images)

# 3. the pack is already validated host-side; this is the first model load
bash /data/models/dsv41-iq2xxs-pack/launch-ds41.sh bench     # 16 tokens, --verbose, log to run.log
bash /data/models/dsv41-iq2xxs-pack/launch-ds41.sh status    # after: GPU + timeout census
```

What to watch in `run.log`: `Engram layer 1/14: ... rows on NVMe`, then the per-layer `experts ... on the CPU engine`, then
`Load weight end` and the decode rate. The first token is slow (the CPU lane fills its RAM store); `--ssd-cache 100` is in
the launcher, raise it to 115 once a run has shown how much RSS is left.

## 8. Files the port touches

```
sycl/CMakeLists.txt                       + strata-dsv41 and ds4_iq_test (option STRATA_SYCL_DSV41)
sycl/src/dsv41/*.dp.cpp, *.dp.hpp         dpct output (regenerated by migrate.sh)
sycl/src/dsv41/ds4_launch.h               the SYCL launch spelling + ds4_dev_table + ds4_nanosleep
sycl/src/dsv41/iq_expert_test.cpp         upstream's gate, on the migrated headers
sycl/src/dsv41/ds4_packinfo.cpp           host copy (the migrated ds4_iq header)
sycl/src/dsv41/vendor/exl3_cpu.cpp        host copy (__builtin_cpu_init)
sycl/include/strata/dsv41/{exl3,moe_win}.hpp   cudaStream_t -> dpct::queue_ptr
sycl/include/strata/dsv41/hip_or_cuda.h   HIP's names on the SYCL runtime (the tests)
sycl/tools/port_dsv41_fixes.py            every hand edit, idempotent, re-run after each migrate.sh
tools/dsv41/make_engram_stub.py           the sparse all-zero Engram table
/data/models/dsv41-iq2xxs-pack/           the pack + launch-ds41.sh
docs/DSV41_SYCL_PORT.md                   this file
```

## 14. 2026-10-29 night: the 8K prompt's budget, the copy-engine hang, and what the experts cost

### The budget (B70, page cache warm, 6 experts, one chunk of 8192)

| measurement | time |
|---|---:|
| 8K prompt, 6 experts | **30.16 s = 264.66 tok/s** |
| 8K prompt, 1 expert, attention skipped | **8.27 s = 964.82 tok/s** |
| 8K prompt, 1 expert | 14.01 s |

so the 30 s is: the base (dense GEMMs, engram, the flow's plumbing, one expert) **8.3 s**, sparse attention **5.7 s**,
the five extra experts **16.2 s**.  The prompt path does **not** use the CPU lane (flow_layer only calls run_set): the
streamed experts are computed on the GPU, so the 16.2 s is the expert kernels' own rate.

### The stage table, from the engine's own profiler (`DS4_PROF=1`, 2K prompt, 12.8 s)

```
experts (incl waits)                  9661.4 ms  76.4%
sparse attention                      1445.8 ms  11.4%
q_b + rope                             788.4 ms   6.2%
ffn mHC + router + sort                281.5 ms   2.2%
attn rest (q_b, wo_a/b, hc_post)        98.2 ms   0.8%   (attn mHC, projections, compressor+indexer: ~0.24 s)
shared expert                           91.0 ms   0.7%
engram                                  47.1 ms   0.4%
```

and inside the experts: the kernels take 7 711 ms, the rest (~1.95 s, 20% of the stage) is the ring/staging waits.

| expert kernel | type | ms | TFLOPS |
|---|---|---:|---:|
| gate+up | IQ2_XXS | 2505.9 | **3.8** |
| down | IQ2_XXS | 346.2 | 6.5 |
| gate+up | IQ2_XS | 952.9 | 4.2 |
| down | IQ2_XS | 170.1 | 6.6 |
| gate+up | IQ1_M | 2319.0 | 3.9 |
| down | IQ1_M | 1417.2 | 5.6 |

`vendor/exl3.cu`'s `block_gemv` takes `kTok = 8` activations per block (the MMA's n), so a block reads a 128-column
tile of the expert's trellis once for 8 rows: 62 FLOP/byte.  At the card's ~450 GB/s that is a 28 TFLOPS ceiling, so the
kernels are **ALU-bound**, not bandwidth-bound (a 16-wide `kTok` would halve the re-reads and buy little).

`EXL3_TC` (the file's own dpct matrix path, `mma<16,8,16, half, float>`) is now a CMake option
(`-DSTRATA_SYCL_DSV41_EXL3_TC=ON`, scoped to that one translation unit).  It **compiles for the B70 and runs** - unlike
the packed-i16 joint_matrix that the backend rejects - but measured 33.3 s against 34.2 s, i.e. ~2.5%: dpct's mma is
not reaching the XMX units.  Real tensor cores need a hand-written `joint_matrix` (PR #2808's 8x16x16 half/float
combination) or a oneMKL GEMM per expert.

### The Arc's copy engine hangs on host-only copies (upstream #3248, hit here)

The flow's expert staging buffers are `sycl::malloc_host` and its ring is a USM **shared** window, so every
staging -> ring memcpy has neither end in device memory.  Queued, those run on the Arc's copy engine, and the kernel log
shows

```
xe 0000:03:00.0: [drm] Tile0: GT0: Engine reset: engine_class=bcs
```

twice inside ten minutes on a healthy card; the same 8K prompt then took 300+ s instead of 30.  The fix is upstream's
(conversation_snapshot.cpp: "host-only copies run on the CPU"): `ds4_launch.h` gains `cpu_side/cpu_copy/cpu_copy2d` and
`flow_pump` copies such a slot itself, waiting for the slot's last use instead of queueing a barrier ahead of the copy.
`DS4_HOST_COPY=0` restores the queue's path.

Measured, 8K, 6 experts:

| configuration | time | notes |
|---|---:|---|
| host ring, queued copies (before) | 30.16 s | hangs: two 200 s timeouts tonight, one bcs reset each |
| host ring, CPU copies (`DS4_HOST_COPY=1`, default) | **34.2 s** | every run completed; the host wait for `ring_free[r]` costs the overlap |
| host ring, CPU copies + `EXL3_TC` | 33.3 s | |
| VRAM ring, 2048 MB (`DS4_RING_DEVICE=1 DS4_RING_MB=2048`) | **29.5 s = 270.24 tok/s** | the best number so far, no reset - but two later runs at a 1014 MB ring deadlocked (ccs + bcs resets) |

So the VRAM ring is the fast shape (and it is the CUDA engine's), but it is **opt-in** until the small-ring deadlock is
diagnosed: `DS4_RING_DEVICE=1` plus `DS4_RING_MB=2048` reproduces the 270 tok/s, and `ds4_expert_ring_mb` now floors a
VRAM ring at 1024 MB and prints the slot count and size at load.

### The wedge, again (2026-10-29 21:58)

Four engine runs back to back (each leaving orphaned, never-started jobs behind: `Timedout job ... in no process [-1]`,
two engine resets) and the following runs all failed at the first allocation with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`,
then

```
CRITICAL: Xe has declared device 0000:03:00.0 as wedged.
```

`sycl-ls` lost the Arc (only the UHD 770 was left); a reboot at 22:24 recovered it.  The rules that came out of it:
**25-30 s between runs**, read `journalctl -k | grep 'Engine reset'` after every run, and stop at the first one rather
than pushing through.

### Cold page cache is 10x, before any of this

The first 8K prompt of a session ran 300+ s because the expert reads had to come from the NVMe: `cat`ing the four pack
shards to /dev/null (138 GB) took 85-145 s first, after which the same prompt is 30 s.  Any benchmark (and any first
prompt in `--serve`) must be warmed first; the engine's own read rate from the page cache is what the flow is designed
around.

### Where the next 2x is

1. **expert kernels, 7.7 s of 12.8 s at 2K** - 3.9-6.6 TFLOPS at a 28 TFLOPS bandwidth ceiling: the win is fewer
   instructions per MAC (a `half2` inner loop: an fp16 accumulator, a numerics change to be measured against the F32
   path) or real XMX hand-written `joint_matrix`.
2. **the expert stage's ~1.95 s of waits** (20% of it): staging slots, ring size, io threads - cheap parameters first.
3. **sparse attention 5.7 s at 8K**: the per-cell score layout of PR #3248, then oneMKL for QK and PV.
4. **`q_b + rope` 0.79 s at 2K (6%)**: far too much for a dense GEMM at 95 TFLOPS - the dequantize is the suspect.

### Resolution (2026-10-30 00:0x): the VRAM ring is the default, floored at 2048 MB

The ring is now in VRAM by default (`DS4_RING_DEVICE=0` keeps the host window), and a VRAM ring is floored at
**2048 MB** (208 slots of 10 229 760 bytes) whatever the sizing rule says, because 1014 MB deadlocked once in the copy
stream.  A ring that does not fit falls back to the host window instead of failing the load.  The copy ordering is
`flow_pump`'s original barrier for everything the reads land in (the direct flow), and a host wait only for the staging
copies the CPU performs - the first version waited for every slot and cost the overlap (34.2 s against 30.16 s).

Four consecutive runs at the new defaults, no engine reset:

| prompt | time | tok/s |
|---|---:|---:|
| 2K (1996 tokens) | 12.72 s | 156.9 |
| 8K (7981 tokens) | 29.44 s | 271.1 |
| 8K again (defaults only, no env) | 29.72 s | 268.5 |
| 2K again (defaults only, no env) | 12.72 s | 156.9 |

so the copy-engine fix plus the VRAM ring puts the 8K prompt back at its best number (271 tok/s) **and** removes the
intermittent hang that turned the same prompt into 300+ s.  Nothing about the kernels changed: the next 2x is still
section 14's list (the expert kernels' 3.9-6.6 TFLOPS, the attention's 5.7 s, the sub-step overhead).

### 2026-10-30 morning: the expert kernel's inner product is not its bottleneck

`dpct.hpp` defines `DPCT_COMPATIBILITY_TEMP (900)`, so `vendor/exl3.cu`'s own condition
(`DPCT_COMPATIBILITY_TEMP >= 800`) has always been true in this port: the experts run **dpct's
`mma<16, 8, 16, half, float>`**, not the F32 dot product below it.  The two CMake options of section 14 never took
effect - `strata_resolve` hands the target the migrated `.dp.cpp` by absolute path, so a property set on the `.cu`
name (or on a relative path) never matched, and both "variants" were the same binary.  That is now fixed
(`set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/dsv41/vendor/exl3.dp.cpp ...)`) and the F32 branch is ported to
SYCL (dpct's `__device__`/`__fmul_rn`/`__half22float2`/`__shfl_xor_sync` spellings that had never been compiled).

Three inner products, measured on the same 2K prompt and the same binary layout (`2K` = one chunk of 1996 tokens):

| expert inner product | 2K | 8K |
|---|---:|---:|
| dpct `mma<16,8,16, half, float>` (the default, `EXL3_TC` was always on) | 12.72 s | 29.44 / 29.72 s |
| F32 dot product (`-DSTRATA_SYCL_DSV41_EXPERT_NO_TC=ON`) | 12.80 s | - |
| F32 + half2 FMAs folded once per 128 inputs (`..._EXPERT_HALF2=ON`) | 12.67 s | 29.54 s |

All three land within 1%.  So the expert kernels' 3.9-6.6 TFLOPS is **not** the multiply-add: the time is in the
trellis tile loads and the per-128-input staging/barriers around them (each block reads a 128-output x K-input tile of
the expert once per 8 activations, `kTok = 8`).  The next experiment is therefore the traffic one - `kTok = 16` - which
needs `kWarps = 16` (512 threads a block) and touches every launch dim in that file; and the alternative is to leave the
expert kernel alone and go at the attention (5.7 s of the 8K prompt, with PR #3248's per-cell scores and oneMKL for
QK/PV as a proven design).

Both switches stay available and off by default: `STRATA_SYCL_DSV41_EXL3_TC` (keep dpct's mma, the default),
`..._EXPERT_NO_TC` (the F32 dot), `..._EXPERT_HALF2` (the half2 fold, needs `..._EXPERT_NO_TC`).

### 2026-10-30, the same morning: the prompt's experts do not go through `block_gemv` at all

Launch census of one 2K prompt with `DS4_LAUNCH_LOG=1` (every launch names its site):

| site | launches |
|---|---:|
| `ds4_iq.dp.cpp:2405` / `:2406` (the `DS4_DQ` dequantize switch) | 4 986 / 1 600 |
| `ds4_kernels.dp.cpp` (attention, embed, mhc, ...) | 2 289 |
| `exl3.dp.cpp:2553` (`rows_f16`, the trellis half's dequantize) | 820 |
| `ds4_gguf.dp.cpp` | 274 |
| **`exl3.dp.cpp:1896` (`mv_kernel`, where the expert tests of the section above live)** | **0** |

So for this pack (IQ2_XXS / IQ2_XS / IQ1_M) the routed experts are computed by the **IQ kernels in
`ds4_iq.dp.cpp`** (`x3::set_gate_up`/`set_down` dispatch to them by ggml type), and the `block_gemv` variants the
sections above measured - dpct mma, F32 dot, half2, kTok 8 vs 16 - are dead code for it.  That is why all of them came
out within 1% of each other: none of them ran.

What the census does show is where the prompt's launches go: the **dequantize** kernels of the dense half
(~5 800 launches at 2K, plus every one of them is a full pass over a weight matrix).  That, not the expert inner loop,
is the next thing to measure: `DS4_DENSE_PROF=1` on the 2K prompt splits the dense half into dequantize and GEMM.

The switches stay (all off by default, no effect on the engine): `..._EXL3_TC`, `..._EXPERT_NO_TC`, `..._EXPERT_HALF2`.

## 15. The next build: the prompt's experts through llama.cpp's MMQ (int8 tensor cores)

The engine already compiles llama.cpp's MMQ (ggml-cuda `mmq.cuh`, MIT) for the Qwen path:
`sycl/src/prefill/moe_mmq.dp.cpp` (320 lines) + `include/strata/prefill/moe_mmq.hpp`.  Its API is already the shape the
dsv41 expert flow has, so this is a wiring job, not a kernel project:

```
mmq::built() / supported(int ggml_type) / fits(int ggml_type, int64_t w_rows)
mmq::matrix_bytes(type, rows, cols)        one expert's gate+up or down weights, in `type`
mmq::q8_bytes(rows, cols)                  its q8_1 activations (row padded to 512)
mmq::quantize(x, ids, xq, type, cols, ld, rows, stream)     activations -> q8_1_mmq
mmq::quantize_scatter(x, slot, src, xq, type, cols, ld, tokens, k_used, stream)   k_used x fewer quantizations
struct Product { const void* w; int type; int64_t w_rows, w_cols; size_t expert_bytes; int n;
                 const void* xq; const int32_t* bounds; const int32_t* ids;
                 int64_t total_rows, max_rows; float* dst; int64_t ld_dst; ... };
Context::run(const Product&, void* stream)      one launch: expert e's rows [bounds[e], bounds[e+1]) x its matrix
mmq::swiglu(gu, h, rows, n_ff, interleaved, stream)
mmq::gather_native / gather_strata_q2            gate|up / down gathered from a blob (one launch per expert)
```

`supported()` covers `Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS`.  **IQ1_M is not covered** (nor by
upstream llama.cpp), and our pack has IQ1_M layers - those keep the dp4a kernels until the experts are requantized
(`tools/dsv41/` already packs and requantizes).

Where it plugs in: `run_set()` in `sycl/src/dsv41/ds4_model.dp.cpp`.  For `sa.lay.fmt != kFmtMXFP4` it calls
`x3::set_gate_up(sa, c.x, D, c.row_tok + a, c.h16, FF, d.st)` and `x3::set_down(sa, c.h16, FF, c.y, D, d.st)`, which
dispatch by ggml type into `ds4_iq`'s dp4a kernels.  An MMQ branch there needs: (a) the q8_1 activation buffer (per
layer, `q8_bytes`) filled by `mmq::quantize` with `ids = c.row_tok + a`; (b) one `Product` per matrix - our blob keeps
gate, up and down as three separate tensor sections (`l.lay.trellis[0..2]`), while the Qwen path feeds one
`[2*n_ff, n_embd]` gate+up matrix, so the first version issues three `Product`s (gate, up, down) instead of one and runs
`swiglu` in F32; (c) `fits(type, w_rows)` before every launch (llama.cpp's tile choice aborts the process when nothing
fits); (d) the output is F32 into `c.y`, so `h16` (the F16 hidden the down product reads) is produced by the swiglu
step from the gate/up outputs instead of by `set_gate_up`.

Why this is the next 2x: the expert stage is 54% of an 8K prompt (16.2 s of 30 s), the dp4a kernels reach 3.9-6.6
TFLOPS, and int8 XMX is a different unit (the same one that makes oneMKL's fp16 GEMM 95 TFLOPS).  Expect ~3x on the
IQ2_XXS/IQ2_XS layers first (about half the expert time), then all of it once IQ1_M is requantized: 8K ~19-24 s
(325-410 tok/s) against today's 29.4 s.

Verify with, in this order: the 8K prompt's first token id against today's (`--ids @bench/8k.ids --n 1`), the chat
prompt ("capital of France" -> Paris), then `DS4_PROF=1` for the per-type TFLOPS (it now prints per ggml type and
should show the MMQ types at a new rate).

### 2026-10-30: MMQ is not in the SYCL build at all - the blocker is llama.cpp's ggml-cuda headers

`strata::prefill::mmq` looks ready (`sycl/src/prefill/moe_mmq.dp.cpp`, `include/strata/prefill/moe_mmq.hpp`, per-type
`mul_mat_q_case<GGML_TYPE_IQ2_XXS>` ...), but `sycl/CMakeLists.txt` **never compiles it**: the SYCL build only carries the
stub `Context::run(const Product&, void*) {}` from `sycl/src/prefill/prefill.cpp:78`, so `mmq::built()` is false for every
SYCL target including the Qwen engine.

Wiring it in is done (`strata_mmq_dsv41`, `sycl/CMakeLists.txt`, option `STRATA_SYCL_DSV41_MMQ`, linked into
`strata-dsv41`; include paths: `include/`, the llama.cpp checkout's `ggml/src/ggml-cuda`, `ggml/src`, `ggml/include`, and
`STRATA_CUDA_HEADERS_DIR` for `cuda_fp16.h`).  With that, the compilation now reaches llama.cpp's own CUDA headers and
stops on their warp intrinsics:

```
ggml/src/ggml-cuda/common.cuh:129: use of undeclared identifier '__syncwarp'
ggml/src/ggml-cuda/common.cuh:418: use of undeclared identifier '__trap'
ggml/src/ggml-cuda/common.cuh:459: use of undeclared identifier '__shfl_xor_sync'
ggml/src/ggml-cuda/common.cuh:502: use of undeclared identifier '__all_sync'
```

`moe_mmq.dp.cpp` is dpct's migration of `moe_mmq.cu`, but it still includes the **unmigrated** `common.cuh`, `mmq.cuh` and
`quantize.cuh` - so this path needs those three headers migrated as well (dpct, then the same fixups.py treatment the
rest of the port got).  That is the real cost of MMQ, and the next session's first job.

The two int8/DPAS expert paths that *are* built do not help this card: `sycl/src/prefill/esimd_moe.dp.cpp` compiles to
its non-ESIMD stub (`available() == false`), and `sycl/src/prefill/xmx_moe.dp.cpp` takes **F16** weights
(`grouped_f16(X, W, cnt, n, Y, N, K)`) and its own comment says Battlemage is 7-11% slower with it (it defaults on only
for Alchemist).

### Round 2: dpct is gone, and the int8 DPAS kernels that ARE built want different quant types

Two findings, one of which changes the route:

1. **dpct no longer exists on this machine** (the extracted `dpcpp-ct` lived under /tmp and the 22:24 reboot took it),
   so "migrate llama.cpp's `common.cuh` / `mmq.cuh` / `quantize.cuh`" starts with re-fetching the
   `intel-oneapi-dpcpp-ct-2025.3` .deb (~a few hundred MB, `dpkg-deb -x`, no root: `sycl/tools/get-cuda-headers.sh`
   shows the pattern).

2. **The int8 DPAS expert kernels that our binary already contains are Battlemage-tuned and complete**
   (`sycl/src/prefill/esimd_moe.dp.cpp`, compiled: `STRATA_I8_NO_ESIMD` is not in build.ninja), and their API is exactly
   the shape of `run_set` - one group, expert blob pointers, int8 activations:

```
i8::make_layout(lo, cnt, n)                                the group's rows per expert
i8::workspace(lo, K, n_ff, ws, stream)                     int8 tiles + scales, library-owned, grow-only
i8::quantize_rows(const uint16_t* X, lo, K, xq, dx, s)     FP16 rows -> int8 tiles  (the engine has c.x16)
i8::gate_up(gu_type, blob[], up_off, n_ff, lo, K, xq, dx, GU, s)    GU FP32 interleaved [row][2*n_ff]
i8::swiglu_quantize(GU, lo, n_ff, hq, dh, s)
i8::down(d_type, blob[], down_off, n_embd, lo, n_ff, hq, dh, Dm, s) Dm FP32 [row][n_embd] = our c.y
```

with `choose_mode()` defaulting to **auto (int8) on a Battlemage device when the kernels are in the binary and run** -
our B70 is bmg-g31 and the device probe asks for the s8 8x16x32 DPAS that Xe2 has.

**But their type coverage is not ours**: `gate_up` handles `IQ3_S, IQ3_XXS, IQ2_S` and `down` handles `IQ4_NL, Q2_0`
(esimd_moe.dp.cpp:981 and :1018), while this pack's experts are `IQ2_XXS / IQ2_XS / IQ1_M`.

So the cheapest route to int8 tensor cores is now a **data-side change**: requantize the pack's experts into the types
those kernels take - gate/up to **IQ2_S** (2.5 bpw against today's 2.06) and down to **Q2_0** (2.56 bpw), `IQ4_NL`
(4.5 bpw) as the alternative for down - and wire the calls above into `run_set`.  `tools/dsv41/pack_gguf.py`'s table
already carries IQ2_S (22), Q2_0 and IQ4_NL (20), and `tools/dsv41/dense_requant.py` is the requantizer.  That skips the
header migration entirely, and the kernels are the ones the port tuned for this card (the file records B70 numbers).

## 16. The int8 DPAS expert path, wired (DS4_EXPERT_I8)

`sycl/src/dsv41/expert_i8.dp.cpp` puts a group of experts through the port's own Battlemage int8 kernels
(`sycl/src/prefill/esimd_moe.dp.cpp`, s8 8x16x32 DPAS, built into a small `strata_i8_dsv41` library that the dsv41 engine
links; option `STRATA_SYCL_DSV41_EXPERT_I8`, on by default, and the run-time switch `DS4_EXPERT_I8` is off by default):

```
run_set -> ds4port::i8_set(sa, gu_type, d_type, cnt, n, c.x16 + row*D, D, c.y + row*D, D, queue)
           i8::make_layout(cnt)                       the group's rows per expert (<= 16 experts a call)
           i8::workspace(lo, K, n_ff, ws)             int8 activation tiles + scales (library-owned)
           i8::quantize_rows(c.x16, ...)              FP16 activations -> int8 tiles
           i8::gate_up(gu_type, blob[], up_off, ...)  -> GU F32 interleaved [rows][2*n_ff]
           i8::swiglu_quantize(GU, ...)               -> hq/dh
           i8::down(d_type, blob[], down_off, ...)    -> c.y directly (F32 [rows][n_embd])
```
`blob[i]` is the expert's own quantized blob in the ring (`wbase + i*stride + trellis[0]`), `up_off`/`down_off` are the
other two tensor offsets relative to it.  It returns false for anything the kernels do not cover, and `run_set` then keeps
the dp4a path - no behaviour change unless the switch is on *and* the pack's types match.

Measured today, 2K prompt, same binary:

| run | time |
|---|---:|
| `DS4_EXPERT_I8=1` (this pack: IQ2_XXS / IQ2_XS / IQ1_M -> fallback) | 12 813 ms (155.8 tok/s) |
| default (no variable) | 12 654 ms (157.7 tok/s) |

i.e. the wiring runs end to end, falls back exactly as designed, and costs nothing when off (0 engine resets either way).
To make it *pay*, the experts must be in the types `gate_up` takes (`IQ3_S / IQ3_XXS / IQ2_S`) and `down` takes
(`IQ4_NL / Q2_0`) - the next step is to requantize a slice of the pack (`tools/dsv41/pack_gguf.py` carries IQ2_S 22,
IQ4_NL 20 and Q2_0; `tools/dsv41/dense_requant.py` is the requantizer), verify the first token against today's, and then
run the 8K number.

### The requantized test pack (tools/dsv41/expert_requant.cpp, patch_pack_experts.py)

To make the int8 DPAS kernels pay, the experts have to be in types they decode.  Two new tools do that from the pack
itself (no original checkpoint, no big copy):

```
# 8 layers of experts: gate/up (w1,w3) -> IQ2_S, down (w2) -> IQ4_NL, through ggml's own decoders and quantizer
build-sycl-jit/expert_requant PACK OUT.bin OUT.json --layers 0-7 --gu IQ2_S --down IQ4_NL --jobs 8
# a pack that points those tensors at the new file and symlinks everything else (shards, engram, tokenizer)
python tools/dsv41/patch_pack_experts.py --pack PACK --out TESTPACK --override OUT.bin --json OUT.json
```
`expert_requant` reads the pack's `tensors.txt` (`file`/`t` lines), resolves the shards through `ds4.txt`'s `model_dir`,
decodes each row with `ggml_get_type_traits(src)->to_float`, encodes it with `ggml_quantize_chunk` into the target type,
and writes a 4096-aligned override file plus `{name: [type, offset, bytes]}`.  `patch_pack_experts.py` copies the two
text files, symlinks every referenced file, appends the override as a new `file` index and rewrites those `t` lines -
a per-layer test bed (8 layers is 20% of the experts, ~43 GB of IQ2_S/IQ4_NL against 8.6 MB an expert at IQ2_XXS).

The types are the intersection of what the int8 kernels take (`IQ3_S / IQ3_XXS / IQ2_S` for gate+up, `IQ4_NL / Q2_0` for
down) and what the engine's own paths already decode (`ds4_iq.cpp`'s `supported()`: IQ2_S yes, IQ4_NL yes, Q2_0 no), so
a layer that the kernels do not take still runs on dp4a - the pack can be mixed.

### First end-to-end run of the requantized test pack (2026-10-30, 11:2x)

The 8-layer requantized pack loads and runs:

```
pack /data/models/dsv41-i8test/pack     9216 expert tensors on IQ2_S (w1,w3) / IQ4_NL (w2), 43.6 GB, 6 files present
2K prompt, dp4a (no switch)             12 916 ms = 154.5 tok/s   (the original pack: 12 654 ms)
2K prompt, DS4_EXPERT_I8=1              SIGSEGV in the first expert group (and a ccs engine reset)
```

So the data side is done and verified (a mixed pack loads and is no slower), and the remaining work is a plain bug in
`expert_i8.dp.cpp`'s first call - the suspects, in order: `sa.lay.trellis[]` for the IQ2_S/IQ4_NL tensors does not give
the gate/up/down section offsets the way the dp4a path reads them (my `up_off = trellis[1] - trellis[0]`, `down_off =
trellis[2] - trellis[0]`, `blob[i] = wbase + i*stride + trellis[0]`), the `cnt`/`Layout` for a group whose rows per expert
are uneven, and the shared `static float* gu` scratch being taken while the queue is still busy with the previous group.
Debug recipe for next time: one layer only (`--layers 0`), `DS4_EXPERT_I8=1` with a single-expert group (a 96-token
prompt), and print `rows`, `n`, `trellis[]`, `stride` before the first launch.

Also learned: the pack's `file` indices must be **contiguous** (a `file 9` after `file 5` makes the loader read a
directory and fail with "cannot map"), a relative `file` name resolves against `ds4.txt`'s `model_dir` (an absolute one
is prefixed too), and the type names in `tensors.txt` are upper case - three things the tools now do.

### The int8 DPAS kernels crash on this toolchain (i8_probe, 2026-10-30 13:1x)

`sycl/tools/bench/i8_probe.cpp` (CMake target `i8_probe`) drives the same call sequence on a synthetic 2-expert layout
(IQ2_S gate/up, IQ4_NL down, ~28 MB) - small enough that a crash leaves the card usable.  It gets through

```
available: 1   battlemage: 1        (the B70's s8 8x16x32 DPAS is detected)
workspace: ok                        xq/dx/hq/dh allocated
step 1: quantize_rows               ok
step 2: gate_up                     SIGSEGV (process killed, ~1 ccs engine reset each time)
```
and it does the same with one expert and 8, 16 or 32 rows, so it is not my geometry: `i8::gate_up` itself dies on this
machine's toolchain.  These kernels have never been exercised here (the port's Qwen engine would have used them, but we
only ever built and ran the dsv41 engine), and `esimd_moe.dp.cpp`'s own header notes that an older IGC died on 77 of them
for bmg-g31 - the newer compiler builds them, but building is not running.

Consequence: the two int8 roads are both blocked for now - the ESIMD kernels crash, and llama.cpp's MMQ is not in the
SYCL build (it needs ggml-cuda's `common.cuh`/`mmq.cuh`/`quantize.cuh` migrated, and `dpct` was lost with the /tmp reboot).
Everything else stays as it was: the engine runs the dp4a path (8K 29.4 s, 271 tok/s) with the VRAM ring, and the
requantized test pack (IQ2_S/IQ4_NL) is ready for whichever int8 path becomes usable.

### The AOT bmg-g31 build of the int8 kernels fails too - the ESIMD route is closed for now

`i8_probe` rebuilt in the AOT `bmg-g31` tree (`build-sycl`, the one that also sets `STRATA_XMX_SG16`) dies even earlier
than the JIT build's: it prints through the synthetic blob's size and never reaches the layout, i.e. it faults around the
device allocations/copies that the JIT build got through.  Two toolchains, one and the same kernels, crashes in different
places, and no backtrace from gdb: this is an ESIMD/codegen problem in kernels that were never exercised on this machine
(`esimd_moe.dp.cpp`'s header already records an IGC that died on 77 of them for bmg-g31).

So of the two roads to int8 tensor cores for the experts:
* the ESIMD ones: **closed** until someone debugs ESIMD on this driver/compiler pair;
* llama.cpp's MMQ: **closed by tooling** - it needs ggml-cuda's `common.cuh`/`mmq.cuh`/`quantize.cuh` migrated, `dpct`
  is gone, and `apt.repos.intel.com/dpcpp-ct/**` answers 403 to every path (the oneAPI 2026 install no longer carries it).

Which leaves the non-kernel work for the next 2x: the sparse attention (5.7 s of an 8K prompt) and the expert stage's
~1.95 s of pipeline waits - both on the engine's existing, working paths.

## 17. The attention, next (5.7 s of an 8K prompt)

The prompt's attention is `ds4_kernels.dp.cpp`'s `sparse_attn_packed_kernel<G>` (dispatched from `sparse_attn()`, the
prompt path takes the `G=4` instantiation at `ds4_kernels.dp.cpp:4814`): grid `(heads/G, T)`, 256 threads, one token and
G heads a block, keys walked in tiles of `kAttTile = 64` staged 16 rows at a time into BF16 local memory
(`kStage = 16`).  Per key row each warp computes one head's 512-value dot: every lane holds 16 q values and 4 words of
the staged key row, so the dot costs 16 multiply-adds and then a sub-group reduction over 32 lanes - **the reduction,
not the arithmetic, is the per-key overhead.**

Arithmetic: an 8K prompt attends ~640 keys (top-512 + a 128 window) for each of T x heads, QK over 576 values plus PV
over 512, which is ~57 TFLOP across 40 layers.  At 5.7 s that is ~10 TFLOPS - a scalar kernel's rate, on a card whose
oneMKL fp16 GEMM does 95 (measured) and whose XMX does the same work with 16x16x16 tiles.

Two designs to try, in order of cost:

1. **Per-cell scoring** (upstream Niko1221/Strata#3248, `qsa_prompt_attn_percell`, 2.0-2.2x on the same card): one
   work-item owns a key row and computes all G heads' dots with it, so no cross-lane reduction per key; the score
   matrix lands in local memory and the softmax/PV pass reads it.  The q values for the block's heads sit in local
   memory once (G x 512 halves = 4 KB), and the reduction moves to one per (token, head) instead of one per key.
2. **oneMKL for QK and PV**: gather the selected key rows for a batch of tokens into the packed buffer the kernel
   already builds, then `gemm` (X) x (K)^T into the score buffer, a softmax kernel, and a second `gemm` for PV.  The
   gather is the cost (640 rows x 576 halves per token), which is why the per-cell rewrite comes first.

Both keep the numerics within FP32 reassociation (upstream measured KL 3.9e-10 against the old kernels at 8K), and the
existing kernel stays behind a switch for the A/B.

## 18. DPCT is not needed: llama.cpp's own SYCL backend ships a hand-written MMQ

Intel deprecated DPCT after 2025.3 (SYCLomatic is the successor), and `apt.repos.intel.com/dpcpp-ct/**` is 403 - but the
migration is no longer the plan.  The pinned llama.cpp checkout already contains an Intel-tuned, hand-written SYCL MMQ:

```
llama-cpp-dsv41/ggml/src/ggml-sycl/mmq.cpp        119 588 bytes, the quantized tensor-core matmul in plain SYCL
llama-cpp-dsv41/ggml/src/ggml-sycl/mmq.hpp:18     void ggml_sycl_op_mul_mat_q(...)   - the one entry
llama-cpp-dsv41/ggml/src/ggml-sycl/mmq.cpp:2987   the type switch (Q4_0 ... Q8_0 and the i-quants)
```

It is written against ggml's tensor metadata (`src0->type/ne/nb/data`) rather than CUDA, and it is **dense only** - no
expert grouping, so the natural use is one call per expert's row range (the numbers per call are what section 14's
profiler already reports for the dp4a path, so an A/B is direct).

Path from here, in order:

1. Confirm the i-quants are in that switch (`grep -c 'case GGML_TYPE_IQ2_XXS' mmq.cpp`) and whether the entry can be fed
   a fabricated `ggml_tensor` (the dsv41 binary already links `libggml-base.a`).
2. Measure it before integrating: a probe like `i8_probe` - one synthetic IQ2_XXS expert (2304 x 5120) x q8_1 activations
   - and read the TFLOPS against the 3.8-4.2 the dp4a path gets.  That number decides whether the integration is worth it.
3. If it wins, wire it into `run_set` behind `DS4_EXPERT_MMQ` (the switch and the 16-expert grouping are already there
   from the int8 attempt), one launch per expert group, and compare the first token against today's.

The requantized test pack (IQ2_S / IQ4_NL) is not needed for this route: llama.cpp's MMQ takes this pack's own
IQ2_XXS / IQ2_XS bytes.

### Correction, same day: the SYCL MMQ is K-quants only - so the pack has to move to Q2_K

`ggml-src/ggml-sycl/mmq.cpp`'s switch covers `Q2_K Q3_K Q4_0 Q4_1 Q4_K Q5_0 Q5_1 Q5_K Q6_K Q8_0` and **no i-quant**
(`grep -c iq2_xxs` = 0).  The i-quants exist only in the CUDA MMQ (`ggml-cuda/mmq.cuh`, the one our
`sycl/src/prefill/moe_mmq.dp.cpp` wraps), which is what needs SYCLomatic.

The way through is therefore the type, not the tool: **Q2_K** is 2.5625 bpw against this pack's IQ2_XXS's 2.06 - about
+24% of expert bytes - and it is
* in the SYCL MMQ's switch (int8 tensor cores, the kernels the Intel backend ships), and
* in the engine's own `ds4_iq.cpp supported()` list (`kQ2_K`), so a Q2_K pack also runs the dp4a path and the CPU lane.

So the next experiment is cheap and does not depend on any deprecated tool: `expert_requant` with `--gu Q2_K --down Q2_K`
on one or two layers (the tool takes the type by name), `patch_pack_experts` as before, and then a port of
`ggml-sycl/mmq.cpp` behind `DS4_EXPERT_MMQ` - plain SYCL, no migration step, entered per expert row range.
Its cost against the dp4a path is what `DS4_PROF`'s per-type TFLOPS line will show.

### Q2_K works end to end, and the kernel to port is small

`expert_requant --layers 0-1 --gu Q2_K --down Q2_K` wrote 8.5 GB for 2304 expert tensors in 2m36s (20 CPU-minutes, 8
threads), `patch_pack_experts` built `/data/models/dsv41-i8test/pack-q2k` (file 6, all six present), and the engine ran it:

| pack | 2K prompt | note |
|---|---:|---|
| original (IQ2_XXS / IQ2_XS / IQ1_M) | 12 673 ms = 157.5 tok/s | |
| Q2_K on two layers | 14 363 ms = 139.0 tok/s | the 8.5 GB override was cold (and Q2_K is +24% bytes) |

Same first token, no error, 0 engine resets - so a Q2_K pack is a drop-in for the engine's own paths, which is what the
MMQ route needs as its fallback.

And the kernel to port is small: in `ggml-src/ggml-sycl/mmq.cpp` the Q2_K path is `allocate_tiles_q2_K` (18 lines),
`load_tiles_q2_K` (~30) and `ggml_mul_mat_q2_K_q8_1_sycl` (115), on top of the shared q8_1 quantizer and the tile
helpers - roughly 300-400 lines to lift into a port file with a plain signature
(weights, rows, cols, float activations, count, out, queue) instead of ggml's tensor metadata.

### llama.cpp's SYCL MMQ compiles standalone in two defines - no dpct, no port

```
icpx -fsycl -std=c++17 -fsyntax-only \
  -DGGML_SYCL_WARP_SIZE=32 -DGGML_SYCL_TARGET=INTEL \
  -I<llama.cpp>/ggml/src/ggml-sycl -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src -I<llama.cpp>/include \
  <llama.cpp>/ggml/src/ggml-sycl/mmq.cpp
=> 0 errors
```

So the kernel is usable as-is: add `ggml-sycl/mmq.cpp` (upstream file, unmodified) plus `quantize.cpp` and `common.cpp`
from the same directory to a port library, and reach the `static` kernels through one small TU that
`#include "mmq.cpp"` and appends a plain entry:

```cpp
// sycl/src/dsv41/mmq_entry.dp.cpp
#include "mmq.cpp"      // (the pinned checkout's file, so its static kernels are in this TU)
void ds4_mmq_q2k(const void* w, const void* q8_1, float* dst, int ncols, int nrows, int n, dpct::queue_ptr q) {
    ggml_mul_mat_q2_K_q8_1_sycl(w, q8_1, dst, ncols, nrows, n, /*nrows_y*/ n, /*nrows_dst*/ nrows, q);
}
```

The tile sizes come from `ggml_sycl_info().devices[id].cc`: Xe2 takes the `VER_GEN13` branch (the largest tiles,
`MMQ_X_Q2_K_RDNA2`), so a probe first prints that branch's numbers on the B70 before anything is wired into `run_set`.
The activations need `quantize_mmq_q8_1` (ggml-sycl/quantize.cpp), not ggml's own q8_1, because MMQ reads the row layout
that pads to 512 values.

#### The three anchors the port needs (measured 2026-10-30, 14:2x)

```
kernel   ggml-sycl/mmq.cpp:  static void ggml_mul_mat_q2_K_q8_1_sycl(const void* vx, const void* vy, float* dst,
                                     int ncols_x, int nrows_x, int ncols_y, int nrows_y, int nrows_dst,
                                     dpct::queue_ptr stream)
caller   ggml-sycl/mmq.cpp:  ggml_sycl_op_mul_mat_q(src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, ...)  - receives the
                                     q8_1 activations (src1_ddq_i) from the generic mul_mat path, it does not build them
build    compile with -DGGML_SYCL_WARP_SIZE=32 -DGGML_SYCL_TARGET=INTEL and the ggml-sycl include dir: 0 errors
```
So the last unknown is the SYCL q8_1 row quantizer the generic path uses (`quantize_row_q8_1_sycl`, in
`ggml-sycl/ggml-sycl.cpp` next to `quantize_q8_1_impl` in `quantize.hpp`) - its exact name and argument order is the
first thing to read next, then `mmq_probe` can time the Q2_K kernel against the dp4a path's 3.8-6.6 TFLOPS.

### The Q2_K MMQ kernel runs on the B70: 8.7 TFLOPS against dp4a's 3.8-4.2

`strata_mmq_sycl` (upstream `ggml-sycl/{mmq.cpp, common.cpp}` + `mmq_entry.dp.cpp`, two defines) builds with 0 errors, and
`mmq_probe` (CMake target) launched the Q2_K kernel on the card:

```
device: Intel(R) Arc(TM) Pro B70 Graphics   acts 128   reps 20
q8_1 activation rows: ok      first launch: ok
0.348 ms a launch -> 8.7 TFLOPS (3.0 GFLOP a launch)
dp4a reference on this pack: 3.8-4.2 TFLOPS (gate+up), 5.6-6.6 (down)
```

So the kernel is real and about **2.2x** the dp4a path's gate+up rate, on one expert of the 8K chunk's shape.  Two caveats
before it can be believed end to end: the synthetic weights the probe fed it decoded to NaN on both sides (a probe-side
data problem - it needs a proper random matrix and a working CPU reference), and the probe's second and third runs could
not enumerate a GPU at all (`get_devices(gpu)` returned empty although `sycl-ls` lists the B70 with the same selector),
so its queue creation needs sorting before the number is repeated.

Next: fix the probe's data and device selection, re-measure at 30 and 128 activation rows, then wire `mat_q2k` into
`run_set` behind `DS4_EXPERT_MMQ` (the switch, the 16-expert grouping and the Q2_K test pack are all in place).

#### Probe repeated: the kernel times at 8.7-14.6 TFLOPS, its numbers are not validated yet

With the device picked by name (the probe no longer goes through dpct's dev_mgr, which came up empty on some runs) and
`quantize_row_q8_1_sycl<quantize_q8_1>(x, xq, kx, rows, pad512(kx), queue)` checked against its signature, one Q2_K expert
(2304 x 5120) x 128 activation rows measures **0.206 ms a launch = 14.6 TFLOPS** (an earlier run of the same binary
shape: 0.348 ms = 8.7).  The weights decode sanely (min -0.107, max 0.111, no non-finites).

What is still wrong is the probe's own check, not the kernel: it compares `dst[0]` against a CPU dot and gets NaN on the
GPU side and 7.4e16 on the CPU side, so either my reference loop or my reading of the kernel's dst layout
(`dst[n][nrows_x]`, row stride `nrows_dst`) is off.  Until that check passes, the 8.7-14.6 TFLOPS is a timing, not a
verified GEMM - so the next step stays "make the probe agree with a CPU reference", then wire `mat_q2k` into `run_set`.

### The Q2_K MMQ kernel is verified: 14.7 TFLOPS, 0.205 ms a launch (2026-10-30 15:0x)

The probe's own data was the problem: `(float)(((i * 17) % 89) - 44)` is `size_t` arithmetic, so every activation below
zero wrapped to ~1.8e19 and both the GPU output and the CPU reference dissolved into NaN/1e17.  With the subtraction done
in `int`:

```
activations: min -0.44 max 0.44   nonfinite 0
dst: nonfinite 0 of 294912, first -0.268037      cpu ref for (act0,row0) -0.26991   (0.7% apart: Q2_K + q8_1 rounding)
0.205 ms a launch -> 14.7 TFLOPS   (2304 x 5120 Q2_K x 128 activation rows)
dp4a on this pack: 3.8-4.2 TFLOPS (gate+up), 5.6-6.6 (down)
```

So llama.cpp's SYCL MMQ runs on the B70 at **~3.7x the dp4a path's gate+up rate**, with a first output agreeing with a
CPU dot to 0.7%.  That is the number the expert stage was waiting for: 16.2 s of an 8K prompt at 3.8 TFLOPS becomes ~4.4 s
at 14.7, i.e. an 8K prompt around 17-18 s (~440 tok/s).

What is left is plumbing, not research: `run_set` has to issue gate (`mat_q2k` over w1), up (w3), the swiglu, a q8_1
quantization of its output and down (w2) per expert group, behind `DS4_EXPERT_MMQ`, using the Q2_K test pack that already
loads (docs section 16).  The shapes are fixed: gate/up are [2304, 5120] Q2_K, down is [5120, 2304] Q2_K, and the
activations are the layer's F32 rows (q8_1 is produced by `quantize_rows`).

### Is V4.1-Flash a CED model (20 encoder + 20 decoder layers, the decoder half only seeing a 128-token tail)?

Tested, not argued: `DS4_PREFILL_LAYERS=20` makes `run_set` return before the expert stage of layers 20-39, and the same
2K prompt then gives a **different first token** (1 against 455), so those layers take part in every prompt token - the
"computation halved" architecture description does not apply to this engine's model.  The knob does work as an ablation:
15 026 ms with all 40 layers against 9 702 ms with the first 20 (measured while a 16-thread requantization was loading the
machine, so both are high), i.e. the expert stage is ~half of a 2K prompt and dropping 20 layers buys ~35% of wall time at
the cost of the output.

### The all-Q2_K pack: 8K in 25.7 s on the dp4a path already, and where the MMQ run broke

`expert_requant --layers 0-39` finished (46 080 tensors, 170 GB, 54 min at ~52 MB/s) and `patch_pack_experts` built
`/data/models/dsv41-i8test/pack-q2k-all` - 6 files present, and only three of them are referenced (shard 3 for the dense
tensors, the engram, and the Q2_K override), so the pack loads with a third of the shards' I/O.

```
8K prompt, page cache warm, dp4a path:   7 981 tokens in 25 704 ms = 310.5 tok/s    (first token 1)
8K prompt, DS4_EXPERT_MMQ=1:             UR_RESULT_ERROR_OUT_OF_RESOURCES at run_set, one ccs engine reset
```

The original IQ2 pack on the same prompt is 29 439 ms (271 tok/s), so **the Q2_K pack alone is 13% faster than the IQ2
one**, before any MMQ.

The OOR is the scratch: a 8K chunk's expert group can hold thousands of rows, and `mmq_set` sized gate/up for all of them
(8 192 rows x 2 x 2 304 x 4 B = 151 MB) on a card that is already holding the 2 GB ring and the dense weights in VRAM.
The fix (built, not yet measured) walks the group in 512-row windows, so the scratch stays ~20 MB; the card took the reset
and is still enumerated, so the measurement is the next thing to do.

### Correction: the CED / long-prompt skip *is* real, and the engine already implements it as "yoco"

My earlier note ("the CED claim is disproved") was wrong, and the reason is worth keeping: `DS4_PREFILL_LAYERS` made
`run_set` skip **every** row of layers 20-39, while the architecture only skips the *early chunks* of a long prompt and
keeps a tail window.  The engine has exactly that mechanism already:

```
ds4_model.dp.cpp:10979  yoco_mode()   (DS4_YOCO / STRATA_DS4_PROMPT_SKIP, on by default)
:1032                   kYocoMin = 1024   the fewest rows worth skipping
:4422-4434              a long prompt's early chunks run the layers from yoco on as kv_only (no experts), and the cut
                        chunk keeps the tail: keep = (nl - yl) * (win - 1) + win = 20 * 127 + 128 = 2668 rows
```

Measured on the all-Q2_K pack at 8K, same prompt, same binary:

| configuration | time | tok/s | first token |
|---|---:|---:|---|
| default (yoco on) | **25 704 ms** | **310.5** | 1 |
| `DS4_YOCO=0` (skip off) | 34 916 ms | 228.6 | 1 |

So the skip is worth **9.2 s of an 8K prompt (26%)** and it does not move the first token - the tail window is what keeps
the output - which is the architecture described in the model's own documentation.  The 25.7 s baseline therefore already
contains the CED optimisation; the 1.2 s left to the 24.5 s target has to come from elsewhere (parameter sweeps first,
the attention kernel after that).

### Two things the chunk sweep taught, and a cache-size warning

1. **`--prefill-chunk 4096` hangs** on this build and pack: the process sat in a `hrtimer_nanosleep` poll loop with the GPU
   at 0 MHz and `rchar` frozen at 186 GB for minutes, i.e. the same family of non-returning waits that 16384 hits
   (docs section 14).  It had to be killed (it ignores SIGTERM) and cost one engine reset.  8192 stays the working size,
   and the sweep should not go below it.
2. **The all-Q2_K pack does not fit in this machine's page cache**: 170 GB of Q2_K experts + 39 GB of dense weights =
   209 GB against 188 GB of RAM, so the 25 704 ms / 310.5 tok/s figure is a *warm* number - the run right after a full
   warm-up.  In steady state the two compete and a single 8K prompt re-reads ~186 GB from NVMe, which is why the same
   binary then takes many minutes.  The IQ2 pack (125 GB total) fits, so its 29 439 ms is the reproducible figure and the
   Q2_K advantage is conditional on a warmed cache.

So the honest state at 8K is: **29.4 s reproducible (IQ2), 25.7 s warm-cache (Q2_K)** - the second one is faster because of
I/O locality, not because of a kernel.  A fair comparison needs both packs measured back to back, which is the next step,
and the 24.5 s target has to be met in the reproducible configuration.

### 8K prefill, final table for this session (same binary, same prompt, one run after another, no new resets)

| configuration | time | tok/s | notes |
|---|---:|---:|---|
| IQ2 pack (125 GB), `--vram-experts 0.3` | 38 882 ms | 205 | cache contested by the Q2_K pack on disk |
| IQ2 pack, second run | 34 697 ms | 230 | |
| **Q2_K pack (209 GB working set), 0.3** | 27 431 ms | 291 | |
| **Q2_K pack, second run** | **27 241 ms** | **293** | steady state |
| Q2_K pack, `--vram-experts 0.5` | 27 375 ms | 292 | no effect (the ring streams anyway) |
| Q2_K pack, `--vram-experts 0.7` | 27 188 ms | 294 | no effect |
| Q2_K pack, warm cache (best seen) | 25 704 ms | 310.5 | right after a full warm-up |
| Q2_K pack, `DS4_YOCO=0` | 34 916 ms | 229 | the engine's long-prompt skip is worth 9.2 s |
| IQ2 pack, clean cache (last night's baseline) | 29 439 ms | 271 | the number to beat |

So the reproducible configuration today is **the Q2_K pack at 27.2 s (293 tok/s)**, 20% better than the IQ2 pack in the
same conditions and 8% better than last night's 29.4 s; its warm-cache best is 25.7 s.  `--vram-experts` does nothing
(the VRAM ring streams experts rather than pinning them), `--prefill-chunk 4096` hangs, and the engine's yoco skip is
already on.

The target is 24.5 s (325 tok/s): the remaining ~2.7 s in steady state is the sparse attention (5.7 s of an 8K prompt at
~10 TFLOPS), and the plan for it is docs section 17 (per-cell scoring first, oneMKL batched GEMM second).

### Usability on the Q2_K pack: the chat answer is right

```
chat.ids ("What is the capital of France?")  ->  ": The capital of France is Paris."     (n=24, 0 resets)
```
Measured on `/data/models/dsv41-i8test/pack-q2k-all` with the same flags as the 8K runs, so the faster pack is not a
quality regression: the requantized Q2_K experts answer correctly.

### The attention score tile is not the lever

`kAttTile` 64 -> 128 (half the loop passes over the key rows, the score tile is 512 B so no shared-memory cost):
8K on the Q2_K pack gives 27 524 ms (290.0 tok/s) against 27 241 ms (293.0) with 64 - no gain, slightly worse, inside the
run-to-run spread.  So the 5.7 s of attention is not loop overhead: it is the per-key dot (16 multiply-adds) plus the
32-lane reduction that upstream #3248 replaces with per-cell scoring, and that rewrite (docs section 17) remains the way to
the last 2.7 s of the 24.5 s target.

### The matrix-core attention path is off on this port and does not help when forced

`fast_attn_default(env, gfx12 = false, v026)` returns false for the SYCL port (it is a gfx12-only default), so
`DS4_FAST_ATTN=1` was tried: 8K gives 27 666 ms (288.5 tok/s) against 27 241 ms (293.0) with the default packed-KV kernel -
no gain, so the chunked prompt's attention here goes through `sparse_attn_packed_kernel` either way, and `DS4_ATTN_SPLIT`
is documented in the source as no-gain on other hardware too.

With `kAttTile` also ruled out, the 5.7 s of attention is exactly the per-key dot plus its 32-lane reduction, and the only
way to the last 2.7 s of the target is the per-cell rewrite (section 17): one work-item a key row, computing every head's
dot against it, so the reduction happens once per (token, head) instead of once per key row.

## 19. The local API service (`dsv41.service`, port 8160)

The engine's `--serve` is a stdio protocol; `serve/server.py` is the HTTP/OpenAI front end.  A user unit now runs it for
this pack, **lazily** (no model until the first request) and with a **5-minute idle unload**:

```
~/.config/systemd/user/dsv41.service
  ExecStart=/bin/bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 &&
    exec python3 -m serve.server --engine strata --config /data/models/dsv41-iq2xxs-pack-real/strata-server.json \
      --host 127.0.0.1 --port 8160 --lazy --idle-unload 300'
  systemctl --user enable --now dsv41.service     # enabled: comes back after a reboot
```

Verified in this session:
```
GET /v1/models            -> {"id": "deepseek-v4.1-flash-abliterated-iq2xxs", "status": {"value": "unloaded"}}   (no load)
POST /v1/chat/completions -> loads on demand, answered in 15 s            (status then {"value": "loaded"})
service log               -> "model unloaded; the first request loads it ..." / "unloads after 300.0 s without requests"
```
Both the OpenAI (`/v1/chat/completions`) and the Anthropic (`/v1/messages`) shapes are served, context 393216, no API key,
127.0.0.1 only.  The client-visible model id is `deepseek-v4.1-flash-abliterated-iq2xxs`.

For DSH, add it as an OpenAI-compatible provider: base URL `http://127.0.0.1:8160/v1`, that model id, no key - the
provider must be registered in DSH's own settings, which the web UI owns (the config files under ~/.dsh are per-profile
and not a place to write blind).

`systemctl --user stop dsv41.service` frees the card; the unit's `KillSignal=SIGINT` is deliberate because the engine
ignores SIGTERM.

## §20 多 chunk 预填充死锁 —— DSH 挂不起来的真因（2026-10-30 夜，实测）

**现象**：8160 服务上，DSH 的每次请求（system prompt + tools = **39,646 token**）都停在
`reading the prompt: 39,646 tokens, N s so far`，数字永不前进；服务端 `engine_silence_s`（955 s）
到点后 **SIGKILL** 引擎（日志 `the engine had stopped (exit code -9)`），每次击杀都带走一次
`Engine reset: engine_class=ccs` + `bcs`（19:14 / 19:50 / 20:20 三次，正好对应三次尝试）。
DSH 侧看到的是 3 × `pi-ai stream idle timeout after 900000ms`。

**根因（用 CLI 复现，与 serve 无关）**：`--prefill-chunk 8192` 下，任何需要 **≥2 个 chunk** 的 prompt 都会死锁：

| prompt | chunk | 结果 |
|---|---:|---|
| 7,981 (8K) | 8192 → 1 chunk | **23.8 s / 334.67 tok/s** ✓ |
| 15,962 (16K) | 8192 → 2 chunks | 200 s 无输出，io 冻结 → kill ✗ |
| 39,905 (40K, Q2_K pack) | 8192 → 5 chunks | io 冻在 186 GB rchar → kill ✗ |
| 39,905 (40K, IQ2 pack) | 8192 → 5 chunks | io 冻在 139 GB rchar → kill ✗ |
| **39,905 (40K)** | **40960 → 1 chunk** | **91.8 s / 434.58 tok/s** ✓ |
| 27,867 (API, 单 chunk) | 65536 | **69 s 读完 = 404 tok/s**，0 reset ✓ |

死锁时：主线程在 `forward_chunks`（ds4_model.dp.cpp:11128）里 50 ms 轮询等 device worker 空闲；
某个 worker 卡在 Level Zero 队列（gdb: `libze_intel_gpu` cond_wait），另一个线程空转 100% CPU；
`/proc/<pid>/io` 的 rchar/read_bytes 完全冻结。**这不是慢，是死锁。**

**教训（最重要的一条）**：上一会话所有基准（2K/4K/8K）**都是单 chunk**，多 chunk 路径从未被跑过。

**已落地的修法**：`/data/models/dsv41-iq2xxs-pack-real/strata-server.json` 的 `--prefill-chunk`
从 8192 改成 **65536**（备份 `strata-server.json.bak-20261030`）→ 只要 prompt ≤ 64K token 就是单 chunk，
实测 39.9K 预填充 91.8 s（434 tok/s）、API 全链路 27.9K 69 s 通、**0 reset**。

**遗留**：单 chunk 的内存/显存上限（chunk 再大要验）；decode 仍 3.3–3.9 tok/s（带宽锁死，见 HANDOFF-attention.md）。

## §21 多 chunk 死锁：待修（优先级最高，2026-10-30 深夜）

**状态**：B70 在 22:28:23 被内核判定 wedged（48K prompt + chunk 65536 + ctx 393216
触发 NEO `bindless_heaps_helper.cpp:69 Abort` → 泄漏作业 → `guc_exec_queue_timedout_job`）。
服务 `dsv41.service` 已 **stop + disable**，`--prefill-chunk` 已回 40960。重启后恢复。

**已确认的失败包络**（全部实测，见 §20 表）：单 chunk ≤ ~40,960 行 ✓；≥2 chunk ✗；>40,960 行 ✗。

**chunk 划分算法**（`forward_chunks`，ds4_model.dp.cpp:11036）：
```cpp
nc  = (n + chunk - 1) / chunk;          // chunk = --prefill-chunk
per = (n + nc - 1) / nc;                // 行数平均分到每个 chunk
// 例：n=7981, chunk=8192 -> nc=1, per=7981   —— 唯一被验证过的形状
//     n=15962             -> nc=2, per=7981   —— 死锁
//     n=39905             -> nc=5, per=7981   —— 死锁
```

**下一轮调试步骤（重启后，不需要猜）**
1. 先用现有 `hang` 探针（`.tmp/hangprobe.sh`，io 冻结后自动 gdb dump 全部线程栈）跑
   16K（2 chunk）→ 拿到 worker 线程卡在哪一行（`flow_pump` / `copy_ahead` / `run_dev_layers`）；
2. 对照 `run_dev_chunks`（10327）与 layer-major 路径（`c.lm`，10142）：多 chunk 时
   `lm_begin/lm_end`、`fl_T`、`chunk_rec/chunk_done` 的生命周期是重点怀疑对象
   （`flow_spec(c, 0, ...)` 只喂 chunk 0 的条目；`c.lm` 走 `run_dev_layers`）；
3. 环境变量二分（不用改代码）：`DS4_YOCO=0`（关跳层）、`DS4_FLOW_PUMP_WAIT=0`、
   `DS4_RING_DEVICE=0`（host ring）、`DS4_HOST_COPY=0`（回到队列拷贝）各跑一次 16K，
   哪一个不卡就锁定那一层逻辑；
4. 修好后必须验证：16K / 32K / 43.5K 三个长度都跑通且首 token 与单 chunk 版一致。

### §21.1 定位到具体路径（2026-10-30 23:20，读代码）
`lm_begin()`（ds4_model.dp.cpp:10887）在 **单卡 + run.T.size() >= 2（≥2 chunk）+ flow 开**
时返回 true → 多 chunk 一律走 **layer-major 路径**（`run_dev_layers`，10142），而单 chunk 走
chunk-major（10373 的 `for k / for il` 循环）。上一会话所有基准都是单 chunk → **layer-major 在
SYCL 端口上从未被跑过**，死锁就在这里。

**已经存在的开关**：`DS4_LAYER_MAJOR=0` → `lm_begin` 直接返回 false → 多 chunk 退回 chunk-major
（每个 chunk 各自把专家再读一遍，2 个 chunk = 2× 文件读，但走的是已验证的单 chunk 代码）。

**下一步试验（重启后第一件事）**：16K prompt + `DS4_LAYER_MAJOR=0`；通了就拿这个当必修的第 1 步
（哪怕慢一点），再去修 layer-major 本身。

**方法论教训（今天三条命换的）**：kill 掉卡在 GPU 队列里的引擎 = 下一次作业超时 → wedge。
所以调试死锁**必须**在引擎内部加超时自毁（把挂死变成干净的 error 返回），而不是外部 kill。

### §21.2 修好了（2026-10-30 23:55，实测）——多 chunk 死锁 = layer-major 路径
**改动**（ds4_model.dp.cpp:10887 `lm_begin`）：layer-major 路径改成**opt-in**（`DS4_LAYER_MAJOR=1` 才用），
默认走 chunk-major（每条 chunk 各自把专家流一遍，正是单 chunk 已经验证过的那套代码）。
```cpp
- if (devs.size() != 1 || run.T.size() < 2 || (env != nullptr && std::atoi(env) == 0)) return false;
+ if (devs.size() != 1 || run.T.size() < 2 || env == nullptr || std::atoi(env) == 0) return false;
```

**实测（新二进制，全部 0 reset）**

| prompt | chunk | chunk 数 | 结果 |
|---|---:|---:|---|
| 7,981 | 8192 | 1 | 24.3 s / **328.6 tok/s**（回归对照，与改前 334.7 一致） |
| 15,962 | 8192 | 2 | 44.9 s / **355.4 tok/s**（改前：死锁） |
| 48,000 | 40960 | 2 | 109.7 s / **437.7 tok/s**（改前：死锁/abort） |
| 45,291（走 8160 API 全链路） | 40960 | 2 | **110 s 读完**，随后 8 token，0 reset |

**服务配置**：`/data/models/dsv41-iq2xxs-pack-real/strata-server.json` → `--prefill-chunk 40960`，
`dsv41.service` 已 enable+start。DSH 的 43.5K token 请求从"15 分钟无响应"变成"约 110 s 出首字"。

**仍然待办**：layer-major 本身没修（它读专家只一遍，比 chunk-major 省一半 I/O，值得修）；
单 chunk 上限 ~40,960 行（再多会撞 NEO 绑定堆）；decode 仍 3.2 tok/s。

### §21.3 前缀缓存 + 上下文上限（2026-10-31 00:30，实测）
**前缀缓存本来就有，而且对 DSH 的"追加式对话"完全有效**（服务端会把 parked 的会话状态接着用）：

| 场景 | 首字 | 总时间 |
|---|---:|---:|
| 43K prompt，**第 1 轮**（冷） | 102.5 s | 112.4 s |
| 第 2 轮（在上一轮基础上追加） | **2.3 s** | 12.1 s |
| 第 3 轮 | **1.9 s** | 11.8 s |
| 45K，同一 prompt 再发一次 | — | **1.6 s** |

也就是说：**每段对话只付一次 ~110 s 的预填充，之后每轮只 prefill 增量**。

**两个配套改动**
1. `--idle-unload` **300 → 3600 s**（原来 5 分钟不请求就把引擎卸掉，缓存的会话状态一起没了，
   下一条消息又付 110 s；现在 1 小时）。
2. **上下文上限 65,536**（引擎 `--max-context`，DSH 侧 `b70local.contextWindow` 同步改成 65536）：
   实测 76,837 token 的请求现在**立刻返回干净的 400**（`request exceeds the context window ... requests are
   never truncated`），不再把引擎卡死。

**为什么要有这条上限**：76,837 token（2 chunk × 38,419 行）在 chunk-major 路径上仍会死锁
（主线程停在 forward_chunks 11132 等 device worker，io 冻结）；48,000 token 正常。
阈值在 48K–77K 之间，未定位。上限挡在 65,536，DSH 侧也会提前压缩。

**当前 DSH 体验（实测）**：第 1 轮 ~110 s 出首字；之后每轮 ~2 s 出首字，decode 3.4 tok/s（仍是唯一的硬瓶颈）。

### §21.4 PCIe 带宽真相（2026-10-31 00:45，独立探针实测）
```
H2D pinned    38.9 GB/s      D2H pinned    54.1 GB/s      (512→256 MiB, best of 5)
H2D pageable   9.9 GB/s      D2H pageable  10.5 GB/s
```
**引擎自己的 `probe_pcie_h2d_gbps()` 用的是 pageable 内存**（源码里 `pin_source=false` 是单卡默认，
注释说 pinned 快 1.7×，我们这里是 **3.9×**）→ 它把这张卡记成 9.5 GB/s，`pcie_frac` 也跟着按慢速算。
探针：`.tmp/pcie/bw2.cpp`（`icpx -fsycl -O2 bw2.cpp -o bw2`）。

**decode 的账**：3.36 tok/s = 297 ms/token，其中 `head wait`（队列收尾，真正在 GPU 上跑完的时间）
**267 ms**，layers 30 ms、磁盘 38.5 MB/token（其余专家字节来自页缓存）。按 ~2.5 GB/token 折算
实际有效带宽 ≈ **8.4–9.9 GB/s = pageable 档**。也就是说 **decode 正跑在慢的那条路上**：
pinned(38.9) → 约 60–70 ms/token ≈ **14–15 tok/s**；63 GB/s 理想值 → 45 ms ≈ 22 tok/s。

**下一步（decode 提速的具体动作）**
1. 引擎启动探针改成 `pin_source=true`（或加环境变量强制），让 `pcie_frac` 按 38.9 GB/s 算；
2. 把 decode 专家流经的 host 侧缓冲换成 **pinned（malloc_host）staging** 再进 ring
   （目前大概率是页缓存/mmio 直接被 kernel 读 → 跑在 9.9 GB/s）；
3. 用同一个 `--n 32` 的 decode 钱夹验证：`head wait` 应从 267 ms 掉到 ~70 ms。

### §21.5 两道墙：磁盘会话（端口没实现）+ 65,536 token 上限（2026-10-31 01:00）

**① KV 落硬盘：引擎有这个能力，但这个 SYCL 端口没实现。**
上游有 session files（`--slot-save-path`，`POST /slots/0?action=save|restore`，跨重启有效，
约 **1.2 GB / 63K token**）；实测本端口回
`ERR unsupported command 'SAVE' (GEN, GENI, STOP and QUIT)` —— 端口只移植了 GEN/GENI/STOP/QUIT。
另外要说清楚：**活动 KV 不可能放硬盘**（每个 token 的注意力都要读它，盘的延迟是 ms 级，
现在 KV 是 host-paged、VRAM 里只放窗口），能落盘的只有"会话快照"。
→ 想跨重启复用长 prompt，需要把上游的 SAVE/RESTORE 移植进来（工作量：一个命令 + slot 序列化）。

**①-b 已实现（2026-10-31 08:45）：`SAVE <path>` / `RESTORE <path>`。**
端口跑的是 `strata-dsv41`，它的 serve 循环是 `src/dsv41/ds4_main.cpp`（host 文件，SYCL 构建经
`strata_resolve` 直接编这一份，CUDA 构建也是它）。DS4 引擎的会话序列化本来就在
`Model::slot_save` / `slot_load` 里（`STRATA_DS4_SLOT_KEEP` 的 slot 目录用的就是它：header
`"DS4SLOT1"` + 引擎 identity + tokens + 每层 window ring / compressor partial group / 每个 KV source 的
compressed KV 与 index keys / checkpoints），所以**没有新格式**：一个会话文件就是 slot 的 `.bin`
（`.meta` 里的 identity 和 tokens 已经在 header 里），单文件、无 sidecar。命令在 `ds4_main.cpp` 的 serve
循环里（`SAVE ` / `RESTORE ` 前缀，路径取到行尾），回复 `SAVED <tokens> <bytes> <ms>` /
`RESTORED <tokens> <bytes> <ms>` / `SERR <invalid|storage|memory|io> <0|1> <reason>`，每 16 MiB 发
`SESSION <done> <total>`、单次阻塞步骤前发 `SWAIT <phase> <s>`。详见
`.tmp/save-restore-notes.md`（在 llm-for-glm-flash 工作区）。

**② 【已证伪，见 §21.7】上下文墙在 ~65,536 token（2^16），不是 VRAM。**
（下面这张表本身没错，但**归因错了**：65,536 不是长度的墙，2^16 也只是巧合。2026-10-31 13:30 起实测
131,072 / 196,608 token 的 prompt 全部跑通、0 reset —— 见 §21.7。）
实测（chunk-major，全部单卡）：
| prompt | chunk | chunk 数 | 结果 |
|---:|---:|---:|---|
| 48,000 | 40960 | 2 | ✓ 109.7 s |
| **64,000** | 32768 | 2 | **✓ 144.7 s / 442.4 tok/s** |
| 72,000 | 40960 | 2 | ✗ 死锁 |
| 72,000 | 16384 | 5 | ✗ 死锁（所以**不是 rows/chunk 的问题**） |
| 76,837 | 40960 | 2 | ✗ 死锁 |

VRAM 账：`--max-context 65536` → caches 0.06 GB、93 MiB 空闲；`--max-context 393216` → caches
**0.36 GB**、0 MiB 空闲。**384K 上下文在显存上是可行的（KV 在 host）**。

~~→ 明天第一件事：在 attention / 分页 KV / indexer 里找按 16 位宽索引 position 的地方（65,536 正好是 2^16）。~~
**这条作废**（§21.7）：131,072 / 196,608 token 都能跑完，没有任何 16 位截断。上表每次死锁的共同点不是
长度，而是**预填充像以前那样异步提交**——跑到 nc ≥ 3 的形状时，一层里同时在飞的命令约 2,089 条，
踩到专家 flow 的双队列环。

### §21.6 早晨那次"还是不行"的真因：冷页缓存 + silence 看门狗（2026-10-31 07:45）
日志：`reading the prompt: 48,480 tokens, 810 s so far`（一个 chunk 都没读完）→
`the engine did not finish the request … within 955 s; the server ended the engine`（exit -9），
连续三次重试同样的死法。**不是死锁，是冷读太慢**：隔夜页缓存被清空，Q2_K 包 209 GB 装不进 188 GB 内存，
cold 每次都要从 NVMe 重读 ~176 GB/chunk，一个 chunk 超过 955 s 静默 → `engine_silence_s` 杀掉。

**修法（已生效）**
1. `engine_silence_s: 3600`（配置项，原来 955）；
2. **pack 换回 IQ2（`dsv41-iq2xxs-pack-real`，164 GB，装得进 188 GB 内存）**——冷读只发生一次，
   之后全在页缓存里；
3. `--idle-unload 3600`、`--max-context 393216`、`--prefill-chunk 40960`。

**实测（IQ2 包）**
| 场景 | 时间 |
|---|---:|
| 重启后第一次请求（引擎加载 237 s + 49,310 token 预填充 152 s） | 391 s |
| 之后**新的** 49,657 token prompt | **147.5 s (≈340 tok/s)** |
| 同一个 prompt 再发（会话缓存命中） | **3.0 s** |

### §21.7 更正：65,536 不是硬墙 —— 是异步并发深度踩到专家 flow 的双队列环（2026-10-31 13:30–14:21，实测）

**结论：长上下文这条路通了。** 同一二进制（`build-sycl-jit/strata-dsv41`，patch `51577eab`）、同一 IQ2 包、
单卡、本 boot 0 次 `Engine reset`，131,072 与 196,608 token 的 prompt 都跑完，服务路径 122,906 token 也跑完。

**真因（不是 2^16，不是长度，不是 chunk 数）**：nc ≥ 3 的预填充里，一层（layer 0）同时在飞的命令约 **2,089 条**
（光 Engram 段就约 2,040 条）；设备在途命令到这个量级就停止退休，而这条路径上唯一的跨队列边是专家 flow 的
那一对，**它可以成环**：

- compute 队列等 copy 队列的 `ring_fill`（`ds4_model.dp.cpp:4784`，run_set 前）
- copy 队列等 compute 队列的 `ring_free`（`ds4_model.dp.cpp:4661`，覆盖 ring 槽之前）

两个队列互相等对方发的事件，谁也不再退休 → 进程静默 forever（SIGTERM 被忽略，只能重启）。所以历史上
"65,536" 只是**当时试到那个规模才踩中**，不是任何 16 位边界。

**修法（commit `51577eab`）**：`forward_chunks` 对 **nc ≥ 3 的 prompt 默认逐条同步**（每个 launch 等完再提交
下一个），`DS4_PROMPT_SYNC=0` 关掉；decode（n = 1）不受影响。代价是那点异步重叠，实测 65,600 是 176.1 tok/s
（nc = 2 的异步对照 184.4）。

**CLI 实测（本 boot，`--max-context 262144 --prefill-chunk 40960`，两次运行独立、间隔 26 s+）**

| prompt（ids 行数） | chunk 数 | 预填充路径 | 时间 | tok/s | reset 增量 |
|---|---:|---|---:|---:|---:|
| 131,072（`.tmp/ladder-131072.ids`，8k.ids ×16.4） | 4 | 逐条同步 | **770.7 s**（770,688 ms） | **170.07** | **0** |
| 196,608（`.tmp/ladder-196608.ids`，8k.ids ×24.6） | 6 | 逐条同步 | **1,168.8 s**（1,168,800 ms） | **168.21** | **0** |

启动日志里那行 `ds4: 4 prompt chunks: every launch is waited on …` / `6 prompt chunks: …` 就是
`DS4_PROMPT_SYNC` 路径触发的凭证 —— 两次都走了。

**服务路径实测（用户真正用的 8160，端到端走 `/v1/chat/completions`，stream）**

配置：`/data/models/dsv41-iq2xxs-pack-real/strata-server.json` 的 `--max-context` **393216 → 262144**
（`--prefill-chunk 40960` 不变），`systemctl --user restart dsv41.service` 后启动行确认
`context 262144 tokens`。DSH provider 同步：`~/.dsh/profiles/web/cordis.patch.yml` 里 `b70local` 的
`contextWindow` **65536 → 262144**。

| 请求 | prompt_tokens | TTFT | 总时间 | 预填充速率 | reset 增量 |
|---|---:|---:|---:|---:|---:|
| warm-up（冷启动，引擎已在跑） | 39 | — | 17.7 s | — | 0 |
| **~120K 文本**（`.tmp/prompt-120k.txt`，510,862 字符） | **122,906** | **733.6 s** | **738.5 s** | **167.5 tok/s** | **0** |

为什么 122,906 而不是 120,000：那份文本是 122,880 个 id 反 detokenize 出来的，chat 模板再加 26 个 token。
一次请求 3 个 chunk（122,906 / 40,960），正好又是 nc = 3 的形状，照样跑通。本次解码只给了 32 token，
模型还在 thinking 里被截断（`finish_reason=length`），计时不受影响。

**已证伪（留着，防止再走一遍）**

| ✗ 曾经的结论 | 出处 | 怎么被推翻的 |
|---|---|---|
| "墙在 2^16：某处按 16 位索引 position" | §21.5 ② | 131,072 与 196,608 都 > 2^16，全通，0 reset |
| "上下文墙在 ~65,536 token" | §21.3 / §21.5 ② | 122,906 走服务、196,608 走 CLI，都跑完 |
| "不是 rows/chunk 的问题 → 是参数/索引" | §21.5 ② 表下 | 同一 chunk 数（nc=2）64,000 通、72,000 死；nc=6 的 196,608 通 —— 轴是并发深度不是长度 |
| "layer-major 路径就是根因" | §21.1 | §21.2 修掉的是另一处（多 chunk 一进 layer-major 就死）；改回 chunk-major 后 72,000 仍然死 |
| "只能靠 kill/重启" | §21.1 方法论 | 从 §21.2 起所有等待都可 bounded，本次全程没对引擎发过任何信号 |

**§21.5 的服务配置也随之更新**：`--max-context 262144`（不再是 65,536）。DSH 侧的 `contextWindow` 必须与
引擎一致，否则引擎会先返回干净的 400（`request exceeds the context window`）。

### §21.7.1 可选 A/B：`DS4_RING_DEP=host` —— 把 copy → compute 这条边从队列上摘掉（2026-10-31 14:31，实测通过）

**改法**（`sycl/src/dsv41/ds4_model.dp.cpp`，flow_pump 的 DMA 段，4661 那一行）：`DS4_RING_DEP=host` 时
copy 队列**不再**对 `ring_free[r]` 提交 barrier，改由 pump 线程自己等这个事件（有界轮询，界取
`DS4_RING_WAIT_MS`，超时把 `"the expert ring slot r was not freed in N ms (DS4_RING_DEP=host)"` 写进 `err`
干净返回）。这样跨队列边只剩 copy → compute（`ring_fill`），**环在结构上不可能成**。默认**不开**，行为不变。

**实测（本 boot，65,600 行，`--prefill-chunk 32768 --max-context 131072`，两次独立运行）**

| 形状 | 预填充 | tok/s | 生成的 token | Engram rows | reset |
|---|---:|---:|---|---:|---:|
| 同步（`DS4_PROMPT_SYNC` 默认，patch 51577eab） | 372,513 ms | 176.10 | — | 3,148,800 | 0 |
| **异步 + `DS4_RING_DEP=host`（第 1 次）** | **184,052 ms** | **356.42** | `54994` | 3,148,800 | **0** |
| **异步 + `DS4_RING_DEP=host`（第 2 次）** | **183,997 ms** | **356.53** | `54994` | 3,148,800 | **0** |

两次都 0 reset、未出现任何 bounded wait 的报错、Engram 行数（65,600 × 48）与同步那次**完全一致**、生成的
下一个 token 也一致 —— 同一个活，做完了，没走捷径。

**为什么是 2 倍而不是"176 → 184"**：§21.7 里那个"异步天花板 184.4 tok/s"是 **`DS4_YOCO=0`** 的那次对照
（每个 token 都要多算一部分层，本身就更重），不是同一种活。同一种活（yoco 默认）的异步本来就在
356–442 tok/s 这一档（8K 单 chunk 355；64,000 nc=2 442）。**同步的代价不是"少一点重叠"，而是把 128 GB
的专家读和 compute 完全串起来**，在这台 I/O 受限的机器上就是 2 倍。

**默认值没有改回 0**（`DS4_PROMPT_SYNC` 仍然随 nc ≥ 3 默认开），理由：本 patch 只覆盖**专家 flow 的 DMA
段**；`51577eab` 里点名的**批量路径**（10167 / 4247）有一模一样的那对边，没有被改成 host 依赖。全局翻转默认值
= 把所有形状都推到没测过的异步路径上，不值。要在这台服务上吃这 2 倍，改成显式 opt-in 即可：
`strata-server.json` 的 `env` 里加 `"DS4_RING_DEP": "host"`、`"DS4_PROMPT_SYNC": "0"`。

**注意二进制版本**：131K / 196K / 122,906 那三组数字是**加这个 flag 之前**的二进制测的；14:29 重新编译后
默认路径（不设 `DS4_RING_DEP`）逐字未变（多出来的代码只在 flag 打开时进入），所以那三组数字依然有效。
