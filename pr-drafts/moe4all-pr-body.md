Adds `documentation/reference/drafter-gguf-packing.md`: the differences between the unsloth `dflash` drafter GGUFs and Lucebox's `deepseek41-dflash-draft` set, and the two traps that only appear at pack/load time.

It is written as reference material, not as a claim about this project's code: nothing in the note depends on which engine packs the file. The dtype row (`markov_head.embed` is F16 in the Lucebox set while some engines read it as BF16) is the one that produces a confusing runtime error rather than a clear pack error.

Verified with DeepSeek-V4.1-Flash IQ2_XXS + the Lucebox Q2K-Q4K drafter on an Arc Pro B70: pack loads (block_size 5, markov_rank 256, noise 128799, 1242 mtp tensors), 100% draft acceptance.

Happy to move it elsewhere (`documentation/evidence/` as a change record, or drop it) if this is not the right shelf.