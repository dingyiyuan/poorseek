# sycl/ — the SYCL engine (Intel GPU)

The C++ engine that runs DeepSeek-V4.1-Flash on an Intel Arc GPU through oneAPI / Level Zero.
It started as a CUDA→SYCL port and is now maintained for one configuration:

| | |
|---|---|
| GPU | Intel Arc Pro B70, 32 GB (Battlemage / BMG-G21) |
| Host | 192 GB DDR5 |
| Model | DeepSeek-V4.1-Flash, MoE (40 layers × 384 routed experts = 15 360), IQ2_XXS |
| Context | up to 384 K |

32 GB cannot hold the experts. The engine keeps the hottest ~1 500 of them in VRAM and streams the
rest from host memory through a single-threaded "CPU lane" (a mailbox the device posts requests into).
Everything here is a trade around that fact.

## Build

Two paths, both the repository's own:

```bash
# 1) the guided one (checks the GPU, picks the model, packs it, builds the engine)
./setup.sh --backend sycl

# 2) just the engine (JIT: SPIR-V compiled at the first start)
source /opt/intel/oneapi/setvars.sh
sycl/tools/build.sh            # AOT=bmg-g31 for ahead-of-time device code
```

The build is self-contained: `sycl/CMakeLists.txt` fetches ggml itself, so nothing outside this
directory is needed.

## Run

```bash
./build-sycl-jit/strata-dsv41 --serve \
  --pack /path/to/dsv41-pack --max-context 393216 --gpus 0 --experts 6 \
  --prefill-chunk 40960 --mmap-experts auto --vram-experts 0.9 \
  --ssd-cache 0 --dspark off --vision off --window 16
```

`--pack` is produced by `tools/dsv41/pack_gguf.py` (see that tool's `--help`); a pack is the base
GGUF plus `engram.bin`, `expert-profile.bin`, `tokenizer/` and a `ds4.txt` file table.

The HTTP service in `../serve/` drives it (OpenAI-compatible `/v1/chat/completions` on port 8160 by
default).

## Knobs worth knowing

| Variable | Effect |
|---|---|
| `SYCL_PROGRAM_COMPILE_OPTIONS=-O3` | JIT optimisation; measured +4.8% over the default |
| `DS4_RING_DEP=host` | ring copies that do not wait on the compute queue |
| `DS4_PROF=1` | per-decode-stage table (GPU event timings), every 64 tokens |
| `DS4_PORT_LAYER=1` | per-layer trace with timestamps |
| `DS4_AHEAD=n` | routing lookahead depth (0 disables; **default is better on real prompts**) |
| `DS4_CACHE_INFLIGHT=n` | in-flight expert swaps (default 8) |
| `DS4_WIN_HOSTWAIT=1` | take the CPU lane's answer on the host instead of letting kernels spin |
| `DS4_MAX_CONTEXT` / `--max-context` | context; 393216 verified |

## Measured on that machine

| | |
|---|---|
| decode, three different prompts | **19.5 / 23.2 / 20.5 tok/s** |
| decode, same prompt repeated (experts all hot — inflated, do not compare configurations with it) | 26–27 tok/s |
| prefill, 80 983 tokens | 517.6 s (**156.5 tok/s**) |
| decode at 81 K context | 12.6 tok/s |

100 K input is roughly 9–11 minutes of prefill, then 11–13 tok/s.

## Where it stops, and why

- **The CPU lane is at the memory bandwidth limit.** On a real prompt ~45–49% of the experts per token
  are not resident, and the lane already moves them at ~38 GB/s against a measured 49–51.6 GB/s pure
  read. More threads do not help; more VRAM would.
- **Dense projections are at the VRAM bandwidth limit** (6.2 GB re-read per token, ~483 GB/s).
- **Speculative decoding (DSpark) is a net loss here**: the drafter is cheap (~7 ms a round, 100%
  acceptance) but a 6-token verify window routes to ~36 experts a layer instead of 6, so a round costs
  ~285 ms. The break-even would need a card that can keep a whole window's experts resident.
- **Half of a verify window is GPU idle** (155 ms of GPU work in a 307 ms step) while the single-token
  path is ~100% busy. Dispatch is not the cause (279 dispatches, 10.6 ms of host time); the remaining
  suspects are synchronisation points.

The measurement log behind these numbers is in `../docs/MTP_STATUS_2026-11-01.md`.

## Safety on this hardware (learned the hard way)

- One engine at a time. After a manual run, wait 60–90 s before starting the service: the GPU context
  is not released instantly and the next start fails with `OUT_OF_DEVICE_MEMORY`.
- If a start fails, **stop immediately** and disable automatic restart (`Restart=no`). A service that
  retries every few seconds against a GPU that is already resetting turns a recoverable failure into
  `Xe has declared device … as wedged`, which only a reboot clears.
- Never SIGKILL a running GPU job.
