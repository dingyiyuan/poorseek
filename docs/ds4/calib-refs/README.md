# V4.1 calibration references (fetched 2026-10-10)

Third-party material gathered while designing the v41 calibration corpus. Data only; nothing here is executed by
the pipeline except `deepseek-official-encoding/encoding/encoding.py` (DeepSeek's own renderer).

| dir | what | use for us |
|---|---|---|
| `deepseek-official-encoding/` | `encoding.py` + tests + `chat_template.jinja` from deepseek-ai/DeepSeek-V4.1-Flash @ `2cba9e42` (pfeifferj's mirror is byte-identical). 48/50 tests pass; the 2 failures need `inference/examples/*` we did not fetch | THE renderer for every chat/agent sample. V4.1 DSML tags are `<｜DSML｜ calls>` / ` invoke` / ` parameter` (leading space); V4 used `tool_calls` / `invoke` |
| `antirez-ds4-imatrix/` | antirez/ds4 @ `fc80bd69` `gguf-tools/imatrix`: 4690 prompts, ~2.9M tokens (agent 1106 with DSML tool schemas/calls/results, source 2074, language 1024, translation 180, eval_reasoning 150, long_context 36, ...), think + nothink | Best public V4-family calibration set. BUT rendered in **V4** DSML tags and short (median ~700 tok, max ~3.7K) -> re-render `messages` with the V4.1 encoder; does not cover long context |
| `hf/pfeifferj/...GSQ-RCO-GGUF` | the other GSQ-RCO V4.1 quant (3.0 bpw backbone). Calib: 64x1024 tokens of fineweb + calibration_mixture + open_thoughts; RCO "kept the initial allocation" (no step improved) | Reference numbers only (heldout.txt PPL 3.4846 at 1K ctx); corpus too small/short to borrow |
| `hf/apetersson/...MixedQ2` | IQ2_XXS gate/up + Q2_K down, 2.25 bpw, **no calibration** | none (uniform map baseline) |
| `hf/pyrodog/...DwarfStar-Q2` | antirez Q2 recipe, weight-energy bootstrap, no imatrix; agentic OpenCode validation projects | validation ideas (tool round trip, long-context cases) |
| `hf/smalinin/...` README | publishes `imatrix-dsv41-262144.gguf` (262K-token llama.cpp imatrix for V4.1, not fetched, ~GBs) | optional cross-check of per-expert activation coverage |
| `hf/INCModel3/...AutoRound` | W4A16, `iters: 0` (RTN, model-free) | none |
| `hf/unsloth/DeepSeek-V4-Flash-0731` README | V4 (not 4.1) GGUF; no calibration details published | none |

Conclusion: no one has published a V4.1 tier map built from activation calibration (pfeifferj's RCO did not move
off its start point), so there is no map to borrow. What we borrow is antirez's prompt set (re-rendered for V4.1)
plus DeepSeek's encoder; long-context agentic coverage has to come from our own transcripts.
