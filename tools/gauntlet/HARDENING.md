# Hardening the Gauntlet — what the evidence says makes frontier models fail

Sources read 2026-10-16 (summarised; every claim below is from one of them):

- **Frontier LLMs Still Struggle with Simple Reasoning Tasks** (Malek et al., arXiv:2507.07313) — procedurally
  generated *simple* tasks (counting, first-order logic, proof trees, travel planning) with knobs that **raise the
  computation needed while keeping the difficulty fixed** make even thinking models fail: statistical shortcuts,
  intermediate-step errors, long-context processing. Their **"Unpuzzles"** result is the sharpest one for us: models
  solve the *famous* puzzle but fail a **trivialised variant**, i.e. they recall rather than reason.
- **A Failure-Focused Evaluation of Frontier Models** (llm-stats.com, 2026-02-03) — across seven frontier models:
  failures concentrate in **precise spatial reasoning, multi-step constraint satisfaction, and distinguishing
  superficially similar but logically distinct problems**; HLE is 85.2% wrong on average (46.2% failed by *all*
  models); **retrieval accuracy drops 26 points as the number of targets rises from 2 to 8**; **perspective-shift
  reasoning fails 91.4%**.
- **Topology Trouble: Why Even Frontier LLMs Still Get Lost in a Grid** (cognaptus.com, 2026-03-14) — the precise
  claim: models *can* reason over constraints **once the constraints are available**, but struggle to **extract and
  maintain** them; **single global invariants** (loop closure, symmetry) are nearly unsolved beyond easy tiers.
- **Frontier Models Hit a Wall on Research Thinking** (peremptory.ai) — 3-15% solo accuracy on reconstruction
  research tasks; 42% with a four-model tournament.
- **BlueFin** (benchmarklist.com, 2026-06-28) — financial spreadsheets: strongest models **< 50%**, weakest on
  **dynamic correctness** (staying right *after* something changes).

## The levers, in the order I'd apply them

1. **Scale the computation, keep the mechanics.** Same rules, more items: more violating requests to find (the
   2→8-target result), more accounts/months to reconcile, more shapes to raster. Long context + more targets is the
   cheapest way to move a task from "passes at the step ceiling" to "fails". *Applies to 04, 07, 08.*
2. **Global invariants that a local fix cannot satisfy.** Require a property of the *whole* output that only holds if
   every step was right — a reconciliation that must cross-foot, a counter set whose sum must equal the total number
   of operations under arbitrary interleavings, a boundary that must close. *Applies to 06 (accounting identity under
   threads), 07 (per-account × per-month cross-foot), 01 (statement reconciles to the cent).*
3. **Trivialised-variant / non-standard semantics (Unpuzzles).** Where a model is likely recalling a known spec,
   document a twist that invalidates the memorised answer: invented precedence, an inverted tie-break, a
   half-open-vs-closed boundary that differs from the standard library's, a channel order that is not alphabetical.
   Task 02 does this; **05 now does too** (its old spec was Python's `re` - pure recall - and it now pins possessive/
   longest-once semantics, line-wise anchors, `.` matching newlines and an invented operator).
4. **Superficially similar but logically distinct**: put *pairs* of near-identical rules in one task and pin both
   sides in hidden tests, so a model that pattern-matches one clause fails the other. Cheap to add to any task that
   already has a rule list. *Applies to 04, 07, 09.*
5. **Perspective shift**: make the same data answerable from two views that must agree but are computed differently
   (per-account vs per-month; per-tenant vs global; forward vs backward). *Applies to 07 (running balance vs amended
   history), 04 (per-tenant counts vs the sorted id set).*
6. **Dynamic correctness**: a documented *amendment* pass — after a revision file, the output must equal what it
   would have been had the rows been amended all along (idempotent recomputation). *Applies to 07, 01.*

## What we must NOT do

- **No new mechanics that the workspace cannot document.** Every hidden assertion stays derivable from text the model
  can read (this is what `validate.py` + the derivability table + the independent-reimplementation checks enforce).
- **No tighter step budgets to force failures.** The 12-step cap is a design constant; failures should come from
  depth, not from a leash. (08 runs on 14 for the same reason — its engine is bigger.)
- **No task that nobody can pass.** A task the reference passes and *every* model fails separates nothing; DESIGN
  keeps at most ~2 of those.
- **No wall-clock assertions** and no unseeded randomness — both are flakiness sources, not difficulty.

## Where the set stands (final measurement, 2026-10-16)

`deepseek-flash`, greedy, no effort override, on the frozen hardened set: **3/10 pass** (01, 02, 06 - each at the
12-step ceiling and never submitting), **7/10 fail** (03, 04, 05, 07, 08, 09, 10) for 246K completion tokens and
1,453 s. Before hardening the same model scored 8/10. The narrowness of the misses is the point: 03 fails one scene,
07/10 fail 2 cases, 04 fails 2 assertions - so there is room above and below the frontier, which is what makes the
gap to a quantized engine measurable.

## Correction (2026-10-16 evening): the 3/10 was the harness, not the tasks

`out/gauntlet-api-final` (the "3/10") ran every step with `max_tokens=4096`, and on the DeepSeek thinking API that
cap covers the chain of thought too: **47 of its 118 steps ended on `finish_reason=length`** before a tool call, and
the harness also dropped `reasoning_content` between tool turns (the API's documented protocol requires passing it
back). 05 and 09 never wrote a single file - so their "122 failed" / "36 failed" said nothing about the tasks, and the
earlier reading that 05 "wrote Python's re" was never true. With the step budget at 36,864 (4,096 + 32K thinking,
the same total a local engine gets), `reasoning_content` passed back and `reasoning_effort=max`
(`out/gauntlet-api-max`), the same model passes **9/10** with 0 truncated steps (only 03 fails, 1 of 2 scenes).

What that says about difficulty, measured on new tasks at full potential:
- **A precise quirky spec is not hard** for a 32K-thinking model: 11-kx8-emulator (an invented CPU, 21 programs, whole
  state compared) passed 22/22 first try; 13-crackme (disassemble + invert an 8-round Feistel) passed in 64 s.
- **Discovery with no spec is**: 12-legacy-codec (reimplement a stripped binary bit-for-bit) spent all 12 steps on
  sound differential experiments and never wrote the encoder.
- **Competition algorithms** (14-17, Codeforces 2900-3400 from deepmind/code_contests) - see DESIGN.md for results.
