# qwen branch: local additions

- Upstream merge 0.1.43.1 via graft (`git replace --graft 6f32ec07 a1641e9f`); future updates = plain `git merge origin/main`.
- Lazy vision (`"vision": {"lazy": true, "lazy_reserve_mib", "lazy_idle_s"}`) and live thinking/answer text in
  `/metrics` (and `/slots` behind `?verbose=1`: the extra keys are opt-in so llama-server's default shape holds).
- `POST /v1/decide`: one-token voice turn-taking decisions. Engine key `score=ID,...` -> `SCORE id:lp ... top=id:lp`
  from the first generated token's raw logits; server `_decide` (thinking off, max_tokens 1). Needs
  `--conversation-cache-mib/-slots` in the engine args so decisions don't evict the agent's conversation.
- **Voice sidecar** (`serve/voice.py`, tests `serve/test_voice.py`): Strata owns a model-agnostic ASR/TTS child,
  started on the first `/v1/audio/transcriptions` or `/v1/audio/speech` and stopped after `idle_s` (mirrors
  `start_lazy_vision`: shrink the elastic cache before start, grow it back after idle). Config:
  `"voice": {"asr": {"cmd": ..., "args": [..., "{port}"], "port": 5110, "lazy_reserve_mib": 1800}, "lazy_idle_s": 120}`.
- **Audio path (BUILT; engine is arch-locked)**: `tools/audio/strata_audio.cpp` (the `strata-vision` twin; mtmd audio
  -> an `SVA1` embeddings record) + engine `--audio-pad N` (the model's `<|audio_pad|>`), `SVA1` records accepted by
  the `GENI` parser, and **linear 1-D positions** for audio rows (images keep the 2-D grid). Built. **But the engine
  takes `--pack DIR` and only loads the `qwen4exp` architecture**, so it cannot run a Qwen3-ASR decoder — the audio
  path only matters for an audio-capable qwen4exp (Omni-class) chat model. The **sidecar above is the working route.**

Full design, status, build steps and the resume prompt: `~/AI/quetza-workspace/voice-duplex/VOICE_DUPLEX.md`,
`RESUME_PROMPT.md`, and `VOICE_KERNELS.md` (kernel crossover) there.
