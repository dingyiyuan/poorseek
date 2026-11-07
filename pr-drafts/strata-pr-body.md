Makes `pack_gguf.py --dspark-gguf` accept Lucebox's DeepSeek-V4.1-Flash-DSpark GGUF, which is a valid DSpark drafter that the tool currently refuses.

**What was wrong.** Lucebox's drafter (`Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF`, architecture `deepseek41-dflash-draft`) differs from unsloth's `dflash` GGUF in five ways, and each one stopped the pack:

| | unsloth `dflash` | Lucebox `deepseek41-dflash-draft` |
|---|---|---|
| architecture check | `dflash` | `deepseek41-dflash-draft` (rejected) |
| geometry keys | `dflash.block_size`, … | `<arch>.dflash.block_size`, … |
| block count | `dflash.block_count` | `<arch>.block_count` |
| head names | `fc.weight`, `conf_proj.weight`, `markov_w1/w2.weight` | `dflash.fc.weight`, `dflash.hidden_norm.weight`, `dflash.dspark.confidence.weight`, `dflash.dspark.markov.w1/w2` |
| noise token | `--dspark-noise-token` only | also carried as `dflash.mask_token_id` |

A sixth issue showed up only after the pack loaded: `markov_head.embed` ships as **F16**, while the engine reads that tensor as **BF16** (`te->bytes / 2`) and refuses it with `mtp.2.markov_head: not the BF16 embed and F16 / BF16 head expected`.

**What this changes.**
- accept both architectures and read the geometry keys with the `<arch>.dflash.<k>` prefix, falling back to `<arch>.<k>`; take `block_count` from `<arch>.block_count`
- take the noise token from `<arch>.dflash.mask_token_id` when `--dspark-noise-token` is not given
- map the `dflash.*` head names onto the checkpoint's `mtp.N.*` names (`fc` → `main_proj`, `hidden_norm` → `main_norm`, `markov.w1/w2` → `markov_head.embed/head`, `confidence.weight` → `confidence_head.proj`); `dflash.dspark.confidence.bias` has no engine-side counterpart and is skipped
- convert `markov_head.embed` from F16 to BF16 into a `<drafter>.mkembed_bf16.bin` side file at pack time and point the tensor at it (same width, no re-quantisation)

**Verification.** Packed `DeepSeek-V4.1-Flash IQ2_XXS` (40 layers, 15360 experts) together with Lucebox's `Q2K-Q4K` drafter:

```
DSpark drafter from DeepSeek-V4.1-Flash-DSpark-draft-Q2K-Q4K.gguf: 3 blocks
dspark_block_size 5, markov_rank 256, noise 128799, n_routed_experts 128
mtp tensors: 1242
```

and it runs on an Arc Pro B70: **acceptance 100% (315/315), 5.84 tokens a round, draft 7 ms/round**.

Not a performance claim — the pack is what this fixes. For what the measurements then say about DSpark on this hardware, the drafter is cheap (7 ms/round) but the *verification window* costs ~285 ms because a 6-token window routes to ~36 experts per layer instead of 6, so speculative decoding is net negative here; that is a separate question from whether the pack is correct.

Happy to adjust naming/placement; the engine-side change this depends on already exists.