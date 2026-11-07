---
kind: reference
status: current
scope: drafter-gguf-packing
last_verified: 2026-11-01
---

# Packing a DSpark drafter GGUF (notes from the Lucebox set)

These are the differences that stop a naive packer when the drafter comes from
`Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF` instead of the unsloth `dflash` GGUFs.

| | unsloth `dflash` | Lucebox `deepseek41-dflash-draft` |
|---|---|---|
| `general.architecture` | `dflash` | `deepseek41-dflash-draft` |
| geometry keys | `dflash.block_size`, `dflash.expert_count`, … | `<arch>.dflash.block_size`, … |
| block count | `dflash.block_count` | `<arch>.block_count` |
| head tensors | `fc.weight`, `enc.output_norm.weight`, `conf_proj.weight`, `markov_w1/w2.weight` | `dflash.fc.weight`, `dflash.hidden_norm.weight`, `dflash.dspark.confidence.weight`, `dflash.dspark.markov.w1/w2` |
| noise token | not carried; needs `--dspark-noise-token` | also carried as `<arch>.dflash.mask_token_id` |
| `markov_head.embed` dtype | BF16 | **F16** |

The dtype row is the one that only shows up at load time: an engine that reads
that tensor as BF16 (`bytes / 2` elements) will reject the file with something
like `mtp.N.markov_head: not the BF16 embed and F16 / BF16 head expected`. F16
and BF16 are the same width, so the fix is a value-preserving conversion of the
tensor bytes into a side file at pack time, not a re-quantisation.

Two more facts worth knowing when mapping the tensors:

- `dflash.dspark.confidence.bias` exists in the file but has no counterpart on
  the engine side; skip it explicitly rather than failing the pack.
- the markov head's rank is not a key in this file; it is the first dimension of
  the `markov.*` 2-D tensor.

Verified with `DeepSeek-V4.1-Flash IQ2_XXS` + the Lucebox `Q2K-Q4K` drafter on an
Arc Pro B70 (SYCL engine): the pack loads with `block_size 5`, `markov_rank 256`,
`noise 128799`, 1242 mtp tensors, and drafts at 100% acceptance.
