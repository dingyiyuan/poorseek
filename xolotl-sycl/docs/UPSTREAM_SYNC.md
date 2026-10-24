# Staying in sync with upstream Strata (DS4 + GLM forks)

Upstream: `upstream` = https://github.com/Niko1221/Strata (MIT).  Our forks: `deepseek4` (Strata-DS4) and `glm`
(Strata-GLM), sharing `src/`, `include/` and the kernels.

## Pulling upstream updates (from 2026-10-10 on)

`deepseek4` contains merge commit `cc55d4c9`, whose second parent is upstream `fb58e0db`, so git knows the shared
history.  An update is a normal merge:

```sh
git fetch upstream
git merge-tree --write-tree --name-only HEAD upstream/main   # dry run: which files conflict
git merge upstream/main
# build (sm_89, STRATA_GGML_DIR=.../third_party/llama.cpp in a worktree), then the gates below
```

Do it in a worktree (`git worktree add -b ds4-upstream ~/AI/Strata-DS4-up deepseek4`), never in the checkout a
running job uses, and fast-forward the branch only after the gates pass.  Merge, never rebase: both forks are
public.

Gates, in order (all must pass before committing the merge):
1. CPU parity: `ptq1_0_test`, `pool_tasks_test`, `q8k_quant_parity`, `iq_avx2_parity`, `ds4_geometry_test`,
   `test_ds4_moe` (all with `CUDA_VISIBLE_DEVICES=`).
2. DS4 dense gate on the mini fixture, bit-identical to the previous build:
   `test_ds4_dense bench/ds4-2026-10-06/dense/mini-ds4-tame.gguf --tokens 63 --work W
    --ref-dir bench/ds4-2026-10-06/dense/cpu-x/ref-allpos-mini-ds4-tame --multi 2,3,1,4`
3. Real model p600 (`bench/upstream-merge-2026-10-10/gate.sh` pauses the v41 calibration between layers, runs it,
   resumes): ppl within the run-to-run band (10.4-10.65 for V4 Q2 + q6_k dense; F32 reference 10.4468).
4. GLM: its own gates in `docs/glm/`.

### How the first merge was done (history only)
Upstream rewrote its history on 2026-10-06, so our `main` (`6f32ec07`) and upstream were unrelated although
`6f32ec07` has exactly upstream `a1641e9f`'s tree.  The one-time fix was
`git replace --graft 6f32ec07 a1641e9f && git merge upstream/main; git replace -d 6f32ec07`.  Replace refs are
repo-global (DS4 and GLM share the object store), so the ref is deleted straight after the merge command.

## Fork rules: keep our edits out of upstream's lines

Every conflict in the first merge was in an upstream file we had edited; our engines (`tools/ds4`, `tools/glm`,
`tools/v41`) are new files and merged with zero conflicts.  So:

1. **New files over in-place edits.**  Fork kernels and helpers go in their own file, hooked in by one line
   (`#include`, CMake `include()`).  Keep a helper next to the code that uses it, so an upstream refactor cannot
   move it out from under us (the stale IQ2_XS block in the first merge was exactly that).
2. **Extend, don't rewrite.**  Keep upstream's lists, tables and macros verbatim and add a fork list beside them.
   Example (`src/kernels/cuda/iq_kernels.cu`): upstream's `STRATA_GU_FMTS` / `STRATA_D_FMTS` / `STRATA_MMVQ_FMTS`
   lines are untouched; Q2_K / Q3_K / MXFP4 live in `STRATA_*_FMTS_FORK`, expanded on the line after each use, and
   `is_iq()` starts with a one-line fork hook.
3. **Defaults keep upstream's behaviour.**  A fork feature that must live inside an upstream kernel takes a
   parameter whose default is a no-op (the SwiGLU clamp `lim` defaults to +inf: bit-identical for every other
   model).
4. **Identical shared files across our forks.**  DS4 and GLM take the same resolution of every shared file, so
   neither fork re-resolves the other's drift.

Footprint (`git diff upstream/main --diff-filter=M --numstat -- src include CMakeLists.txt tools/*.py`): after the
first merge 21 upstream files +727 / -90; after the isolation pass (`20618210`) +327 / -87.  Where fork code lives:
`src/kernels/cuda/fork/*.cuh` (GPU formats/kernels, one `#include` each in `iq_kernels.cu` / `native_mmvq.cu`),
`tools/ds4/cmake/fork.cmake` + `ggml_cuda.cmake` (every fork target; the root CMakeLists.txt has 5 added lines and no
changed upstream line), `include/strata/artifact/ds4_guard.hpp`.  What stays inside upstream files on purpose: the
SwiGLU clamp and per-model row widths in the CPU/GPU expert kernels (behaviour; default no-op), small switch-case
additions for the fork formats, and two generic improvements worth offering upstream as PRs (the I16/I32/I64/MXFP4
GGUF type entries, the tokenizer's multi-pattern pre-split).
