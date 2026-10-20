# poorseek

A fork of MoE4All, used as a workbench for **DeepSeek-V4.1 layered expert scheduling** experiments.

> **Status: work in progress.**
> The repo currently holds the upstream substrate plus a few preparatory changes
> (a CED layer-split guard and a virtual expert view).
> The layered expert scheduler itself is not written yet; this README will be filled in
> once the real implementation lands.

## Acknowledgements

poorseek builds on:

- **[MoE4All](https://github.com/Headmaster218/MoE4All)** — the direct upstream: the generic pager / slot / LUT / host-tier substrate and the DSV4.1 loading and kernel framework.
- **Strata** — the original design of hot/cold expert paging (`expert_cache` / `expert_source` / host-tier complement); MoE4All's paging approach derives from it.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — GGUF format, quantization kernels, MoE dispatch (`n_expert_used`) and the `deepseek41.cpp` reference implementation.
- **[DeepSeek](https://github.com/deepseek-ai)** — the DeepSeek-V4.1-Flash model, the CED architecture, and the official inference engine [DeepInfer](https://github.com/deepseek-ai/DeepInfer).

For the upstream documentation, see the [MoE4All README](https://github.com/Headmaster218/MoE4All#readme).

## License

This project's modifications and overall distribution are under the [Apache License 2.0](LICENSE);
code inherited from upstream infr keeps its original [MIT License](LICENSE-MIT) and copyright
notices. See [NOTICE](NOTICE) for the full attribution.
