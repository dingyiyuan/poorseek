# DeepSeek-V4.1-Flash engine - how it got here (measured progression)

The README shows where the engine is now; this page keeps every step, with the number it was measured at
(RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe).  Engineering log with all logs and next steps:
[V41_RESUME_PROMPT.md](V41_RESUME_PROMPT.md).

## How fast is it? (RTX 4090 24 GB · Ryzen 7 5700X · 90 GB DDR4 · NVMe, 2026-10-11)

**Prefill, 8192-token prompt** - all steps keep the output exact (last-token logits byte-identical to the step before):

| Step | tok/s |
| --- | ---: |
| Chunked prefill (2048-token chunks; every chunk re-streams every expert) | ~54 |
| Layer-major prompt batches: each layer's experts stream **once** per 8192-token batch | 102.6 |
| + next-layer file-tier prefetch into pinned slabs (O_DIRECT, 4.6 GB/s) | 105.9 |
| + encoder-only skip: layers above the last KV source only run the last 2560 tokens | 133-140 |
| + one expert pass per layer (no duplicate stream at B=8192) | 142.9 |
| + next layer's arena experts uploaded during its dense half (layer prestage) | 149.1 |
| + routed sums written straight into the dense half's tensor (no host round trip) | 153.9 |
| + Engram rows read in the background during the layers below the first Engram layer | 176.3 |
| + Engram tables moved off exFAT to ntfs3 NVMe / SATA SSD (raw block-device reads for the fragmented file) | **219-227** |
| 16384-token prompt (two 8192 batches) | **288** |

Where the time goes now (8192 tokens, ~37 s): dense half ~11 s, experts ~25 s - PCIe uploads of streamed experts
(~10 GB/s effective on a 25 GB/s link) and the grouped expert kernels. `STRATA_PROM=1` (prompt-batch kernels) gives
~6% more but costs +0.6% ppl (2.3799 -> 2.3946), so it stays off. Target: **300-400 tok/s** (16K prompts: 288).

**Decode** (256-token prompt, 128 generated tokens, greedy, default flags)

| Config | tok/s | ppl code / agentic (4096 tok) |
| --- | ---: | --- |
| exact routing (`--route-bias 0 --route-bias-file 0`) | ~3.5 | 2.38 / 15.11 |
| `--route-bias 1.0` (selection-only bonus for VRAM-resident experts) | 15.1 | 2.67 / 6.88 |
| **default**: + `--route-bias-file 0.6` (selection-only penalty on experts read from NVMe) | **21.1** | 2.87 / 6.39 |
| after an 8192 / 16384-token prompt | 21.0 / 25.4 | |

Both biases change only which experts are *selected* (mixing weights exact, every expert reachable) and only in
decode - the prompt is computed exactly. Their quality effect is domain dependent: code loses (+21%), agentic text
gains (GSQ gave its cheapest formats to rarely-used experts; the bias steers towards the better-quantized hot ones).
`--route-bias 0 --route-bias-file 0` gives exact routing.

Engram: decode reads ~48 random rows per token. On the exFAT volume that held the FP8 checkpoint this cost ~10
ms/token (~13K random reads/s); `tools/v41/extract_engram.py` copies just the two tables (202.8 GB) to fast storage,
and a heavily fragmented copy is read straight from the block device via FIEMAP (1 ms/token).

**Decode work, 2026-10-11 night** (256-token prompt, 256 generated tokens, greedy, default flags; run-to-run ±3%)

| Step | tok/s |
| --- | ---: |
| start of the night (the defaults above) | 23.8 |
| experts read from NVMe with O_DIRECT; one metadata copy per expert launch (was 5); one sync for the dense half's 3 readbacks | 24.5 |
| routed sum computed on the card straight into the dense half's tensor (no parts readback / sum upload per layer; same bits) | 24.7 |
| experts read from NVMe computed on the card (DMA from the read buffer) instead of by the CPU pool | **25.5-26.3** |
| opt-in: activation rounding as one ggml op (`-DSTRATA_DSV41_FQ_OP=ON` + `tools/v41/ggml/dsv41-fake-quant.patch`) | 27.7 |

Every step except the GPU compute of NVMe-read experts keeps the decode tokens bit-identical; that one swaps the CPU
pool's arithmetic for the GPU kernels the VRAM hits use (per-expert rel. error vs the CPU 2.2e-3, hits 1.3e-3).
Where a token goes now (~38 ms): dense half ~19 ms (of which the Q8_0 mat-vecs read ~7.6 GB at ~880 GB/s - the
card's bandwidth), expert tier ~16 ms (GPU expert kernels + PCIe DMA waits ~6 ms, NVMe reads ~5 ms), the rest
launches and syncs. Next: MTP drafting (needs the checkpoint's `mtp.*` tensors exported), fewer graph nodes.

### What changed on 2026-10-11

- **Layer-major prompt batches** (`--prompt-batch B`, default 8192): attention runs in sub-batches in order, the MoE
  runs once per layer over the whole batch, so a layer's ~5 GB of experts crosses PCIe once instead of once per chunk.
- **File-tier prefetch** (default on, `--no-chunk-prefetch`): layer l+1's file-tier experts are read with O_DIRECT
  into one of two pinned slabs while layer l computes (SSD wait 46 s -> 0.4 s on 4096 tokens).
- **Encoder-only skip** (default on, `--serve` included, `--no-enc-skip`): a prompt token older than the last
  `128 × 20 = 2560` positions is never read above layer 20 except through compressed KV that layers <= 20 produce,
  so layers 21-39 only run the tail. Exact (byte-identical last-token logits), -23% wall at 8192 tokens.
- **Determinism**: CUB's GPU top-k returns the indexer's selection in a run-dependent order; the selected ids are
  now sorted before the gather. Prefill is byte-identical run to run (it drifted ~1% ppl before).
- **Engram batch reader**: hashes deduplicated, sorted by file offset and coalesced per 4 KiB page (cold 4-8× faster).
- **Prompt-batch kernels for PTQ1_0 / Q2_0** (`STRATA_PROM=1`, off by default): halve expert compute at large
  groups; enabled once uploads stop being the limit.
- **Layer prestage**: the next layer's RAM-arena experts are copied to a 3 GiB device region while its dense half
  runs (`STRATA_PRESTAGE_GIB`), and the routed sums go straight into the dense half's tensor (`run_chunk_dev`).
- **One expert pass per layer** at B=8192 (a 1 GB buffer cap used to split it in two, uploading every layer twice).
- **Engram**: prompt reads overlap the layers below the first Engram layer; decode reads run in parallel with a
  raw-row cache.
- **Route bias** (`--route-bias X`) and **soft pruning** (`--prune F --prune-penalty X`): penalties only, never masks.

