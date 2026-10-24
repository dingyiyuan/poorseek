# The Gauntlet

A small, hard, machine-graded agentic coding benchmark: **10 tasks, 12 agent steps each**, built to measure what a
quantized engine loses against the full-precision API. No LLM judge - every task is graded by hidden pytest suites
or an exact-answer regex.

**Why it exists.** `tools/taskeval` (HumanEval-21 + 10 tool calls + 3 agentic loops) is saturated: DeepSeek-V4.1-Flash
scores 21/21, 10/10, 3/3 on it, so it cannot show what a quant, a route bias or an engine change *costs*. The
Gauntlet is the opposite: the full-precision API scores **3/10** on it, which leaves room to see a weaker engine
drop further - the gap is the signal, not the leaderboard position.

## Measured (2026-10-16, `deepseek-flash` = V4.1-Flash, greedy, no effort override)

| | |
| --- | --- |
| before hardening | **8/10** (failed only 03 and 05) |
| after hardening | **3/10** - PASS **01, 02, 06**, FAIL **03, 04, 05, 07, 08, 09, 10** |
| cost | 246,337 completion tokens, 1,453 s for all ten |

The three passes are **at the ceiling** (12 steps, never submitting `submit`) and the misses are *narrow* rather than
total - 03 fails one scene, 07 and 10 fail two cases, 04 fails two assertions; 05 (122 failures) and 09 (36)
collapse. That is deliberate: the frontier sits on the task set, so a quantized engine falls below it rather than
falling off a cliff.

## v2 (2026-10-17) - turns, not tokens; graded every turn

- **`GAUNTLET_TURNS`** turns per task (default **10**, the same for every model; 0 = the task's own `max_steps`), and
  **no token budget** (`GAUNTLET_TASK_TOKENS` unset): each turn gets the served thinking budget + 4096 answer tokens.
- After every turn that wrote a file or ran code, the hidden tests grade a **copy** of the workspace (the model never
  sees the result; `GAUNTLET_PER_TURN=0` turns it off; `GAUNTLET_TURN_GRADE_SECS` caps that grade, default 120 s - the
  final grade keeps the task's full `grade_timeout`). A run records each task's progress curve and the **turn it was
  first solved** (the task ends there).
- **Progress is normalised** from what the untouched workspace already scores (`start_fraction`: 01 starts at 0.61,
  10 at 0.60, 12 at 0.07), so no run earns free credit. `score` prints solved/N, **SCORE** = the sum of progress (a
  solved task = 1), the mean solve turn, and throughput (decode, prefill over new tokens, new/reused, e2e).
- **The v2 core set: 01, 02, 03, 06, 07, 10, 14** (12, 13, 15, 16 dropped: nobody passed them; even the API at max
  effort fails 15/16). Run: `GAUNTLET_TURNS=10 TASKEVAL_THINK_BUDGET=32768 gauntlet.py run <url> <out> --only
  01,02,03,06,07,10,14` (local), or `bash verify/e2e/gauntlet-local-all.sh <models>` (serves each model in turn).
- **API baseline (`deepseek-flash`, `out/gauntlet-v2-api`): 5/7 solved, SCORE 6.41** - 01@4, 02@3, 07@9, 10@4, 14@4;
  03 stuck at 0.50 (scene B), 06 at 0.91; 106,792 tokens, 546 s. Local models: to be run (session 6).
- `selftest.py` = 7 checks (a-g), all PASS; g = the per-turn grading (solved at turn 1 -> progress 1.0; an unfixed
  workspace that already passes 20/33 scores progress 0.0).

## Measured on every Xolotl model (2026-10-17, core set, 12K tokens per task) (v1, history)

The **core set** is 03, 12, 13, 14, 15, 16 under `GAUNTLET_TASK_TOKENS=12000` (thinking included, identical for every
engine) so one model costs at most 72K completion tokens; local engines think with `TASKEVAL_THINK_BUDGET=32768`.
`bash verify/e2e/gauntlet-local-all.sh [ds41 qflash q27 glm glm-exact]`; the API row is
`GAUNTLET_TASK_TOKENS=12000 ... gauntlet.py run https://api.deepseek.com out/gauntlet-core-api --only 03,12,13,14,15,16`.

| run | pass | partial | truncated | tokens | wall s | decode tok/s | prefill tok/s (new) | e2e tok/s | passed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| API `deepseek-flash` | 1/6 | 1.57 | 2 | 65,277 | 398 | - | - | 164.1 | 14 |
| Qwen3.8-Flash-Next (qflash) | 1/6 | 1.57 | 2 | 62,688 | 464 | 148.6 | 940.5 | 135.1 | 14 |
| Qwen3.8-27B (q27) | 0/6 | 0.57 | 2 | 69,513 | 716 | 104.9 | 512.9 | 97.0 | - |
| DeepSeek-V4.1-Flash (ds41) | 0/6 | 0.57 | 0 | 62,305 | 3,685 | 21.5 | 92.4 | 16.9 | - |
| GLM-5.3-Flash, served config | 0/6 | 0.07 | 6 | 72,000 | 2,505 | 30.6 | 172.9 | 28.7 | - |
| GLM-5.3-Flash, exact routing | 1/6 | 1.07 | 4 | 70,390 | 7,362 | 10.1 | 138.8 | 9.6 | 03 |

Read it with these facts (each checked):
- **12's 0.07 is free**: the untouched starting stub already passes 1 of its 14 hidden tests; no run implemented the
  codec (`NotImplementedError` everywhere). **03's 0.50 is real**: the start scores 0/2, every run but GLM's renders
  scene A and fails scene B (`test_matches_reference[B-scene1]`); only GLM with exact routing cleared both.
- **The core set barely separates the models**: one pass at most per run, and 13/15/16 are zero for everyone. A task
  that nobody passes three runs in a row adds nothing but cost - see DESIGN.md's rule.
- **GLM's served config corrupts tool calls.** Its row ran out of budget on all six tasks and emitted byte-salad call
  names (`<tool_call>function\nlist\n</think>...`, mojibake after `<tool_call>`), 3 malformed of 27 calls. A greedy A/B
  of the failing requests (pre-merge `Strata-GLM` build vs Xolotl, both with and without the approximations) showed
  the two trees give the same tool calls with exact routing (identical on 14), and the malformed calls appear only
  with `--skip-miss/--skip-file/--skip-file-prefill 0.15 --renorm-skip --prune ... --prune-penalty 0.05` on. Without
  them: 0 malformed of 39 calls and the only 03 pass in the table - at a third of the decode speed.

## Run it

```bash
# 1. prove every task is valid (start fails / reference passes / trap fails) - always run this first
python3 tools/gauntlet/validate.py                     # 10x "OK", exit 0
python3 tools/gauntlet/validate.py tools/gauntlet/tasks/04-log-needle      # one task

# 2. prove the runner itself works both ways (answer grading, loud failure, a real task solved)
python3 tools/gauntlet/selftest.py                     # a-e all PASS, exit 0

# 3. grade a model
TASKEVAL_MODEL=deepseek-flash TASKEVAL_KEY="$DEEPSEEK_API_KEY" TASKEVAL_NO_EFFORT=1 \
  python3 tools/gauntlet/gauntlet.py run https://api.deepseek.com out/gauntlet-api
TASKEVAL_THINK_BUDGET=32768 \
  python3 tools/gauntlet/gauntlet.py run http://127.0.0.1:8151 out/gauntlet-ds41   # a local engine
python3 tools/gauntlet/gauntlet.py score out/gauntlet-api out/gauntlet-ds41        # pass/10 + tokens + seconds
```

`--only 03,05` runs a subset. A local engine is served exactly as `verify/e2e/gauntlet-local-ds41.sh` does (that
script is the whole recipe: serve on 8151, wait for `/v1/models`, run, score). **Runtime:** the API does all ten in
~24 min; a local engine that spends a 32K thinking budget takes **hours** - measured 25,446 tokens / 893 s for a
single hard request. Always record which thinking budget a local row used (a 32768 row is not comparable to the
API's native default).

## The ten tasks

| # | id | category | what is hard | trap | measured |
| --- | --- | --- | --- | --- | --- |
| 01 | ledger-rootcause | agentic debug | three modules (`parse`/`report`/`reconcile`); the visible failure is in one, the cause in another | normalise in `report.py` (visible passes, `parse_amount` stays wrong); float + `round()` in reconcile | PASS 12 steps |
| 02 | release-order | spec precision | an *invented* release-stamp ordering plus `merge` and a `select` constraint language that must agree | epoch-first ascending, alphabetical channels | PASS 12 steps |
| 03 | raytracer | graphics / 3D math | pixel-centre rays, y flip, shadows with an epsilon, reflection depth 3, checker floor | `int()` instead of `floor()` for cells crossing negative coordinates | FAIL (1 of 2 scenes) |
| 04 | log-needle | long context | 948-line gateway log; 12 violating ids **plus** a per-tenant count table that must cross-foot; needs a key->tenant join, a one-clause exemption pair and the retired key | the per-key single-rule scan (23 ids, 8 real) | FAIL (2 assertions) |
| 05 | regex-engine | algorithms + performance | semantics deliberately NOT Python's: choices are made once and are the LONGEST (possessive), anchors always line-wise, `.` matches newlines, an invented `X~` | a Python-shaped backtracker (correct for `re`, wrong here) | FAIL (122 of 123) |
| 06 | ttl-cache | concurrency + time | documented purge-before-evict ordering, `stats()` splitting expirations from evictions, one-shot expiry, 8-thread stress | a lock around `get` only; evicting before purging | PASS 12 steps |
| 07 | csv-money | data wrangling | dialect chosen by header **signature**, `rates.yaml` normalisation quantized per row, a duplicate rule, a char-exact report whose per-account section carries a running balance | float/round-the-total; a dropped duplicate that advances the balance | FAIL (2 of 22) |
| 08 | svg-raster | graphics / geometry | `M/L/H/V/Z` **plus `C/Q/A`** flattened by a documented N=32 equal-parameter rule, with the full arc derivation spelled out | the sweep flag read the other way; no radius correction; `int()` flattening | FAIL (28 of 38) |
| 09 | dst-scheduler | time zones | fold/gap/day-length, an exact `agenda()` carrying the offset in force at each instant, `blackouts.txt` precedence over the busy list and a minimum-notice rule, weekly local recurrences | `zoneinfo` silently shifting a nonexistent time; a 24 h day | FAIL (36 of 37) |
| 10 | flaky-suite | agentic debug | three modules under one contract; the visible suite is green in **every** order, so the faults must be found by reading | fixing the in-module faults and stopping (the cross-module ones still fail) | FAIL (2 of 60) |

Each task directory is self-contained:

```
tasks/NN-name/
  task.json     {"id","category","prompt","max_steps","max_test_runs","max_python_runs"[, "answer"][, "trap"]}
  workspace/    everything the model sees and may edit, including the VISIBLE test_*.py
  hidden/       test_hidden_*.py - copied in only at grading time; never visible to the model
  reference/    a correct solution, applied for validation only
  trap/         the tempting wrong fix: must PASS the visible suite and FAIL the hidden one
```

## Adding a task (the rules that make it fair and useful)

1. **Real work, not puzzles.** A bug whose symptom is far from its cause, a spec with sharp edges, a long log with one
   inconsistency.
2. **Machine-graded and deterministic.** Hidden pytest (or an `"answer"` regex over the final message). No judge, no
   wall-clock assertions, no unseeded randomness.
3. **Everything needed is in the workspace.** Every hidden assertion must map to a sentence the model can read - the
   slice's report must contain that mapping (a "derivability table"), and any assertion that cannot be mapped is a
   cheat. Difficulty comes from depth and interaction, never from ambiguity.
4. **Every task has a trap**: a plausible wrong implementation that passes the visible suite and fails hidden.
5. **Bounded**: <= 12 agent steps (14 for the two graphics tasks), <= 6 `run_tests`, workspace <= ~400 lines.
6. **Validated before use**: `python3 tools/gauntlet/validate.py <task>` must print `OK` - start fails hidden,
   reference passes both suites, trap passes visible and fails hidden. A task whose reference cannot pass, or whose
   start already passes, is rejected.

`HARDENING.md` lists the evidence-backed levers for making a task hard (un-puzzled semantics, target-count scaling,
global invariants, near-identical rule pairs, perspective shifts, dynamic correctness) with sources.

## Integrity: the workspace cannot grade itself, and secrets stay out

- `write_file` refuses names that would let the workspace interfere with grading - `conftest.py`, `pytest.ini`,
  `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`, `sitecustomize.py`, `usercustomize.py`, `test_*.py`, and every
  stdlib shadow the tasks import (`decimal`, `re`, `json`, `csv`, `zoneinfo`, `threading`, ...), in `.py` form,
  as a sourceless `.pyc`, as a package directory, or as any path component.
- Anything already present under those names is **stripped from the staging dir before grading**. Measured before the
  fix: a `conftest.py` planted through `run_python` flipped task 01's honest "13 failed, 20 passed" into a green
  "33 passed".
- Every subprocess that can execute model-supplied code (`run_python`, both pytest runs, which import the model's
  modules) gets a **six-variable environment** - `PATH`, `HOME`/`TMPDIR` inside the staging dir, `PYTHONPATH=''`,
  `PYTHONDONTWRITEBYTECODE=1`, `LANG` - and nothing else. Verified by a probe that dumps `os.environ` from inside
  `run_python`: before the fix `DEEPSEEK_API_KEY` was visible, now no credential is.

## How it is verified

- `validate.py` on every task (start/reference/trap) - **10/10 OK** on the frozen set.
- `selftest.py` - five checks (answer grading both ways, loud failure when nothing grades a task, a real task solved
  by a canned model, and the workspace-is-refused case).
- **Repeatability measured**: the whole gate 3x, the thread-stress task 10x, the two float-heavy tasks 10x each,
  01/02/07/10 5x each - ~100 task-validations, **zero flakes**.
- **A fresh checkout validates 10/10** (`git worktree` at HEAD = exactly the tracked file set). This is what caught
  `04-log-needle`'s generated log being swallowed by the repo-wide `*.log` ignore rule, which would have made that
  task unpassable from a clone.
- **An independent fairness audit** (`reports/ga-audit-final.md` in the conductor's data dir) mapped all 246 hidden
  assertions to the sentences that document them and returned *FAIR AS IS*; its four findings (an unstated trailing
  newline, a substring `import re` scan, a tzdb dependency, a wall-clock assertion) are all resolved.

## Known limitations

- **Boundary tasks flip run to run.** 01 passed at 12 steps in the calibration run where an earlier run failed it at
  the same 12 steps; 02 and 06 also passed at the ceiling. Re-measure after any change, and treat a single run as a
  sample, not a verdict.
- **09 needs an IANA tzdb** with 2026 rules; its hidden grading asserts that and records the version rather than
  failing mysteriously.
- **08 compares rasters character-exactly.** A philosophically different but spec-faithful float implementation on a
  different `libm` could in principle flip one boundary pixel; an independent reimplementation reproduced all 28
  rasters byte-for-byte, and the only near-tie decisions are exact zeros.
- **Do not publish `hidden/`** alongside a public leaderboard - rotate or withhold the hidden suites.
