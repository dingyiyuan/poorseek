# DeepSeek-V4.1-Flash on this box (4090 24 GB, ~90 GB usable DDR4) - desk study 2026-10-09

## Size (read from the safetensors headers / Q2 model cards)
- Total 510 GB native.  **Engram = 202.75 GB (40%)**: two tables, layers 1 and 14, 384M rows x 256 FP8 + E8M0
  scales each.  Per token it reads ~48 rows = ~12 KB -> lives on SSD, needs ~0 RAM (random-read latency matters,
  not bandwidth; dongnhdev re-encoded it to 4 bits, 340 -> 249 GiB, ds4-metal fork only).
- Routed experts: 40 layers x 384 x (3 x 5120 x 2304) = **543.6B weights** (~2x V4 Flash's 277B); top-6 + 1 shared.
  Native FP4: ~4.5 GB read per token.  Q2 (IQ2_XXS gate/up + Q2_K down, 2.25 bpw) = **152.9 GB**; Q2 "main
  weights" with dense = ~163 GB.  Dense/attention ~10 GB.
- Other differences from V4 (0731) that the DS4 engine would need: hidden 5120, 384 experts, sqrtsoftplus router,
  compress ratios 2/1, kv/index source-layer sharing, engram block, 3 MTP layers (DSpark), FP8 32x32 block scales.

## Can we run it
Fast memory for experts ~= 18 GB VRAM slots + ~65 GB arena = ~83 GB vs 153 GB Q2 -> ~54% resident; the rest streams
from NVMe.  Expected (by analogy: 5090 + 128 GB DDR5 + Gen5 SSD, native 4-bit, SSD-bound = 5.1 tok/s):
- stock Q2 + SSD streaming: **~5-8 tok/s** decode (NVMe-bound; 970 EVO Plus ~3.5 GB/s).
- + REAP soft prune 25-40% (rarely-used experts stay on NVMe) + rank-gated arena: working set ~95-115 GB -> maybe
  8-12.
- a ~1.7-1.8 bpw GSQ-RCO expert quant (~115-125 GB) + soft prune: working set fits -> **~12-18 tok/s** (DS4 0731,
  fully resident, does 17-25).  Requires our own quantization pass over the 510 GB source.
- **Disk is the blocker today**: Q2 file 366 GB; free: NVME1TB 27 GB, nvme0n1 (system) ~93 GB, Crucible 113 GB,
  Team CX2 92 GB.  Engram can go on a SATA SSD (MX500/Team, ~100 us reads, ~5 ms/token, page cache helps), experts
  (~163 GB) need the NVMe -> ~170 GB must be cleared on NVME1TB.

## Uncensored options (best first)
1. **antirez/deepseek-v4.1-flash-gguf Q2 (imatrix-calibrated, 90.1% top-1 vs the API) + msuiche GLP control
   vector** (downloaded: ~/AI/Strata-DS4-private/v41/glp.gguf, 803 KB, gated - Mal has access).  39 rank-1
   directions (fp32, 5120), applied at runtime `h <- h - 0.5 (h.d) d` at the post-layer residual (mean over the 4 hc
   copies, pre-engram), layers 1-39.  Base-model evals: refusal32 4 -> 24/32, cyber32 5 -> 31/32, benign 32 -> 31
   (classifier FP).  alpha is sharply non-monotone (2.0 is WORSE and code-switches) - re-run a small ladder on Q2.
   No re-quantization; works on any quant.
2. pyrodog/...-UNCENSORED-DwarfStar-Q2 (dealignai weights, 366 GB, no imatrix = bootstrap quality).
3. audreyt/...-Abliterated-GGUF Q2-imatrix (s-zaizen abliteration, weaker: 97 -> 24/100 refusals).
- dealignai (MPOA) edited dense weights only (experts/engram/router byte-identical, 27 of 48 shards differ), so
  "dense from dealignai + experts from a calibrated quant" also works - but option 1 is far cheaper.
- apetersson MixedQ2 = same recipe as antirez Q2 (2.25 bpw), standard-ish GGUF names, no engram, not uncensored.
- **Bonus for the current DS4 (0731) engine:** msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-42-residual-
  L1-42-a1.5 and ...-GLP-29 exist - same mechanism, can be wired into ds4_dense now.

Video (Local Explained, "If you want to run DeepSeek 4.1 Flash, buy this hardware", transcript in
~/AI/Strata-DS4-private/v41/): consistent with the above - engram on SSD, 4.51 GB experts/token native, Q2 main
163 GB, 5090+128 GB = 5.1 tok/s SSD-bound, RAM bandwidth sets the ceiling when resident (8ch DDR4 TR Pro + A6000 +
3090, 3-bit: 20 -> 25.6 with DSpark), Q2 top-1 agreement 90.1% / Q4 96.7%.  llama.cpp mainline has no V4.1 runtime;
DwarfStar (antirez/ds4) and forks do.
