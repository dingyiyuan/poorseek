# The Gauntlet - a small, hard, real-world agentic coding benchmark (DRAFT v0.1)

**Why.** HumanEval-21 / tools-10 / agentic-3 (`tools/taskeval`) is saturated: DeepSeek-V4.1 scores 20/21 quantized and
the full-precision API would too, so it cannot show what a quant, a route bias or an engine change *costs*. The Gauntlet
is the opposite: **10 tasks that most models fail**, so a frontier API might score ~5/10 and a damaged quant 2/10 -
the gap is the signal. It must stay **small**: 10 tasks, 12 agent steps each, no synthetic benchmark.
RUNTIME, measured: the full-precision API does all ten in **~12 min** (700 s, 114K completion tokens). A local
engine is a different matter - the step budget bounds its *actions*, not its tokens: with the 32K thinking
budget the e2e run used, GLM spent 25,446 tokens on one problem (893 s at 28.7 tok/s) and fine-tuning this
benchmark measured ds41 at ~20 min for a single hard request. Budget **~1-3 h for a local model**, not 40 min,
and note which thinking budget a row was measured with (a local row at 32768 is not comparable to the API's
native default).

## Rules every task must follow

1. **Real work, not puzzles.** Each task is something an agent in Claude Code / Codex / Quetza actually meets: a bug
   whose symptom is far from its cause, a spec with sharp edges, a refactor that must not break hidden behaviour, a
   long log with one inconsistency, a tool chain with distractors.
2. **Machine-graded, binary, deterministic.** Pass = the hidden tests pass (pytest) or the exact-answer check
   matches. No LLM judge. Greedy decoding (temperature 0), thinking on, `reasoning_budget_tokens` 32768.
3. **Hard because of depth, not trivia.** Every task has at least one trap a pattern-matching model falls into (the
   obvious fix makes the visible test pass and a hidden test fail). The prompt contains everything needed - no
   outside knowledge beyond a competent engineer's.
4. **Original.** Written for this benchmark (not HumanEval/MBPP/SWE-bench text), so it is not in training data.
   Do not publish the hidden tests in the same repo as a public leaderboard (keep `hidden/` out of the public
   push or rotate tasks).
5. **Bounded.** max 12 agent steps (14 for the two graphics tasks - 03 and 08 - whose engines are bigger; the cap
   is per task in `task.json`), max 6 `run_tests` calls, the model's own token budget; workspace <= ~400 lines;
   prompt + files <= ~12K tokens (04-log-needle is the long-context task: its workspace log is 948 lines / 82 KB).
6. **Validated before use.** `validate.py` proves for every task: the hidden tests FAIL on the starting workspace,
   PASS with `reference/` applied, and (if the task has a trap) FAIL with `trap/` applied. A task that a reference
   cannot pass, or that the starting code already passes, is rejected.

## Task format (`tasks/NN-name/`)

```
task.json        {"id", "category", "prompt", "max_steps", "max_test_runs", "answer"?(exact-answer tasks), "trap"?: "one line"}
workspace/       the files the model sees and edits (incl. VISIBLE tests: test_*.py)
hidden/          test_hidden_*.py - copied in only for grading, never shown
reference/       a correct solution: files that overwrite workspace/ ones (validation only)
trap/            optional: the tempting wrong fix (validation: must FAIL hidden)
```

Tools the model gets (OpenAI function calling, same in every engine): `list_files()`, `read_file(path)`,
`write_file(path, content)`, `run_tests()` (visible tests only - output truncated to 4K chars), `submit(note)`.
Exact-answer tasks get no workspace and are graded on the final message (`answer` = regex the final text must match).

## The 10 tasks - all written and validated

`validate.py` proves for every one: the hidden tests FAIL on the starting workspace, PASS with `reference/`
applied, and FAIL with `trap/` applied (so the trap is tempting AND caught).

| # | id | category | what it needs / the trap | hidden grading |
| --- | --- | --- | --- | --- |
| 01 | ledger-rootcause | agentic debug | three modules (`parse`/`report`/`reconcile`): exact `Decimal`, quantize ONCE at the end with `ROUND_HALF_EVEN`, five rejection classes, and a reconcile pass that compares only after quantizing. Trap: normalise in `report.py` (visible passes, `parse_amount` stays wrong) or float+`round()` in reconcile | 33 hidden cases |
| 02 | release-order | spec precision | an **invented** release-stamp ordering - nothing to recall: non-alphabetical channel rank (`dev<beta<rc<stable<`*no channel*`<hotfix`) and an inverted EPOCH compared as the LAST key. Trap: epoch-first ascending + alphabetical channels | 14 pairs, 23 invalid forms, stable sort |
| 03 | raytracer | graphics / 3D math | implement `render()` to spec: pixel-centre rays, y flip, shadows with an epsilon, reflection depth 3. Trap: `int()` instead of `floor()` for the checker cells (breaks for negative x/z) | pixel-compare vs reference renders, 2 scenes |
| 04 | log-needle | long context | a 948-line gateway log + `limits.yaml` + `exemptions.txt` + `api_keys.txt`: write the **12 violating ids** plus a per-tenant count table that must cross-foot to them. Needs the key->tenant join, the global rule, a **one-clause exemption pair** (`hooli *` whole-tenant vs `hooli k_1e35` single-key, both sides pinned) and the retired key. Trap: the per-key single-rule scan (23 ids, 8 of them real) | the exact id set + the cross-foot |
| 05 | regex-engine | algorithms + performance | an engine whose semantics are deliberately NOT Python's: every choice is made once and is the LONGEST (possessive - `a*a` on `aaa` is False where `re` says True), `^`/`$` are always line-wise, `.` matches newlines, plus an invented `X~` odd-offset assertion. Traps: a Python-shaped backtracker (fails the semantics) and catastrophic backtracking (`(a*)*b` on 40 a's < 2 s) | fuzz vs `re` only on the coincident subset + explicit twist cases (123) + timing |
| 06 | ttl-cache | concurrency + time | LRU+TTL with an injected clock and threads: expire lazily on read AND write, capacity counts live entries only, `get` refreshes recency but NOT expiry. Trap: evicting before purging expired entries | expiry/eviction order + thread stress |
| 07 | csv-money | data wrangling | 3 bank dialects chosen by the **header signature** (a `;`-file carrying dialect A's field names IS dialect A), `rates.yaml` normalisation quantized **per row**, a duplicate rule, and a char-exact report whose per-account section carries a **running balance** (kept rows only, account-ascending, dates ascending with ties in input order - a dropped duplicate never moves a balance). Traps: float/rounding-the-total, a duplicate that advances the balance | exact report text + 22 hidden tests |
| 08 | svg-raster | graphics / geometry | rasterize an SVG path subset - `M/L/H/V/Z` plus `C/c`, `Q/q` and elliptical `A/a` arcs, each flattened by a documented N=32 equal-parameter rule, with the full arc derivation (radius correction, sweep sense, centre rule) spelled out in the workspace. Trap: the sweep flag read the other way, or no radius correction, or `int()` truncation instead of the documented flattening | character-exact rasters, 28 shapes / 38 cases |
| 09 | dst-scheduler | time zones / real-world spec | DST-correct scheduling: a skipped local time must raise, a repeated one is resolved by `fold`, a local day is 23/24/25 h, plus an exact `agenda()` serialization (each line carries the offset in force at that instant and real minutes across a change), a shipped `blackouts.txt` whose precedence over the busy list and the minimum-notice rule is documented, and weekly local-time recurrences. Trap: `zoneinfo` silently shifting a nonexistent time, a 24 h day | 37 hidden cases; grading asserts the tzdb resolves and records its version |
| 10 | flaky-suite | agentic debug | three documented modules (`inventory`, `pricing`, `orders`) with one contract; the visible suite is green in EVERY order (no free symptom - 60/60 shuffled orders on the starting workspace), while the buried faults include a hold that is priced once at reserve time and must not follow the world. Trap: fix the in-module faults (aliased tag lists, stale index) and stop - the cross-module ones still fail | the visible suite in 20 shuffled orders + ~35 contract assertions (60 cases) |

**Calibration (2026-10-16, `deepseek-flash` = V4.1-Flash, greedy, no effort override).** Before hardening the API
scored **8/10** (it failed only 03 and 05) - too easy to show what a quant loses. After the hardening waves the same
model, same settings, scores **3/10**:

| PASS | FAIL | |
| --- | --- | --- |
| **01** ledger, **02** release-order, **06** ttl-cache | **03** raytracer, **04** log-needle, **05** regex-engine, **07** csv-money, **08** svg-raster, **09** dst-scheduler, **10** flaky-suite | 246,337 completion tokens, 1,453 s |

The three passes are all **at the ceiling** (12 steps, never submitted) and the misses are narrow rather than total -
03 fails one scene, 07 and 10 fail 2 cases each, 04 fails 2 assertions, while 05 (122 failures) and 09 (36) collapse -
so the frontier sits right on the task set: a weaker engine (or a quantized one) drops below it, a stronger one
could clear it. **A task every model passes is too easy - replace it; one the references pass and every model fails
3 runs in a row is fine** (keep at most 2 of those). Re-measure after any change to a task: 01 passed here at 12
steps where an earlier run failed it at the same 12 steps, i.e. the boundary tasks are borderline by design.

## Running it

```bash
python3 tools/gauntlet/validate.py                         # every task: start FAILS, reference PASSES, trap FAILS
python3 tools/gauntlet/gauntlet.py run http://127.0.0.1:8150 out/ds41        # a local Xolotl server
TASKEVAL_THINK_BUDGET=32768 python3 tools/gauntlet/gauntlet.py run ... # (same env knobs as taskeval)
python3 tools/gauntlet/gauntlet.py score out/ds41 out/glm                   # table: pass/10, tokens, seconds, tok/s
```

An API model, to measure what a quant loses - the knobs live in `taskeval.py`'s `chat()` now:

```bash
TASKEVAL_MODEL=deepseek-flash TASKEVAL_KEY="$DEEPSEEK_API_KEY" TASKEVAL_NO_EFFORT=1 \
  python3 tools/gauntlet/gauntlet.py run https://api.deepseek.com out/gauntlet-api
```

`TASKEVAL_NO_EFFORT` drops `reasoning_effort: none`, which the API rejects; the local engines keep it. The
gauntlet's `run_python` tool executes the model's snippet with the workspace as cwd, so every task also needs a
`max_python_runs` cap in its `task.json`.

## How it is verified before anyone is graded on it

- **`validate.py`** proves, for every task, the three states: the starting workspace FAILS the hidden tests, the
  `reference/` PASSES both suites, and (where a trap exists) the trap PASSES the visible suite and FAILS hidden.
- **`selftest.py`** drives the real `run_task` end to end against a canned endpoint: (a) an `answer` task whose final
  message matches PASSES, (b) one that does not FAILS, (c) a task with neither hidden tests nor an `answer` is
  refused instead of scored PASS, (d) a real task solved by a canned model PASSES.
- **Determinism, measured repeatedly (2026-10-16):** the whole gate 3x (30/30 OK), the thread-stress task 10x
  (06), the two float-heavy tasks 10x each (03, 08), and 01/02/07/10 5x each - **~100 task-validations, zero
  flakes**.
- **A fresh checkout validates 10/10** (`git worktree` at HEAD = exactly the tracked file set), so every task file
  ships - this is what caught `04-log-needle`'s generated log being swallowed by the repo-wide `*.log` ignore rule.
- **An independent fairness audit** (`~/.quetza-data/conductor/reports/ga-audit-final.md`) mapped all 246 hidden
  assertions to the sentences that document them and returned **FAIR AS IS**; its four findings are resolved or
  recorded: R1 (07's unstated trailing newline) fixed, R2 (05's substring `import re` scan -> an `ast` check) fixed,
  R3 (09's tzdb dependency -> an autouse fixture that fails loudly and records the version) fixed, R4 (03's 60 s
  wall-clock assertion) removed.
- **Grading cannot be subverted from the workspace:** the runner refuses names that would let the workspace
  interfere with grading, strips `conftest.py` / `pytest.ini` / `sitecustomize.py` and stdlib shadows before
  collecting, and runs every subprocess that touches model code with a six-variable environment (no credentials).

## Reporting

Per model: pass count /10 (and which), total tokens, wall time, decode tok/s during the run (server timings, as
`taskeval.py speed`). Run each config **3 times** (greedy is deterministic per engine, but the quantized engines have
run-to-run variation from timing-dependent expert placement) and report min / median.
