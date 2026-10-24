#!/usr/bin/env python3
"""Builds every generated artifact of 04-log-needle and cross-checks the design.

  1. make_workspace.generate() writes the workspace files from the fixed seed
  2. checker.py - an independent reader of THOSE files - derives the answer AND the count
     table from them alone
  3. this script asserts the checker's result matches the generator's INTENT (which epoch
     each violation comes from) and that every rule class decides at least one id, so a
     drift between the two descriptions fails loudly instead of shipping a wrong answer
  4. hidden/answer.txt and reference/ANSWER.txt take the checker's two-section body;
     trap/ANSWER.txt takes the single-rule scan - plausible, format-legal, and wrong

  python3 regenerate.py [--table]
"""
import sys
from pathlib import Path

import checker
import make_workspace as M

HERE = Path(__file__).resolve().parent


def single_rule_scan(ws: Path) -> list:
    """The trap: a scan that reads limits.yaml's per-tenant rule only.

    It treats the log's own field (the API key) as the tenant, counts every line it finds
    - ops traffic, /health probes, retired-key 403s, exempt maintenance traffic - and
    never applies the global limit. Every id it reports is a request of a key with too
    many requests in a 10 s window; there is no reason for a model to distrust it.
    """
    limits, _keys, _exempt, ev = checker.load(ws)
    per = {}
    for e in ev:
        per.setdefault(e["key"], []).append((e["t"], e["id"]))
    n, window = limits["per_tenant"], limits["window_s"] * 1_000
    out = set()
    for stream in per.values():
        stream.sort()
        for i in range(n, len(stream)):
            if stream[i][0] - stream[i - n][0] < window:
                out.add(stream[i][1])
    return sorted(out)


def two_sections(ids: list, rows: dict) -> str:
    """The documented ANSWER.txt body: the sorted id set, then the count table."""
    return "\n".join(["# ids", *ids, "# counts", *[f"{k} {n}" for k, n in rows.items()]]) + "\n"


def main(show_table=False):
    meta = M.write(HERE / "workspace")
    limits, keys, exempt, ev = checker.load(HERE / "workspace")

    def flags(**kw):
        return checker.flag(ev, limits, keys, exempt, **kw)

    full = flags()
    per_only = flags(global_rule=False)                 # the per-tenant rule alone
    global_only = flags(per_tenant=False)               # the global rule alone
    no_star = flags(star_rows=False)                    # `*` rows read as key-scoped -> no-ops
    no_keyrow = flags(key_rows=False)                   # named-key rows ignored entirely
    wide_keyrow = flags(key_rows_wide=True)             # named-key rows read as tenant-wide
    no_exempt = flags(star_rows=False, key_rows=False)  # exemptions.txt ignored
    by_key = flags(join=False)                          # no key -> tenant map
    no_retired = flags(retired=False)                   # retired keys counted

    want = {meta["acme-join"][-1],                       # the 6th counted acme request, only
            *meta["globex-per"][-2:],                    # visible once both keys are joined
            *meta["stark-per"][-2:],
            meta["umbrella-per"][-1], meta["wayne-per"][-1],
            *meta["hooli-cutover-other"][-2:],           # hooli's OTHER key, inside the row
            meta["storm-a"][-1], meta["storm-b"][-1], meta["storm-c"][-1]}
    assert full == want, f"answer drifted: {sorted(full)} != {sorted(want)}"
    assert 8 <= len(full) <= 14, f"{len(full)} violating ids is outside the 8-14 target band"
    legal = set(meta["background"] + meta["ops-burst"] + meta["hooli-health"] + meta["wayne-halfopen"]
                + meta["stark-errors"] + meta["hooli-drain"] + meta["stark-failover"]
                + meta["hooli-cutover-named"] + meta["initech-retired"])
    assert not full & legal, "a decoy/exempt/background request became a violation - all are legal"
    assert per_only == full - global_only
    assert global_only == {meta["storm-a"][-1], meta["storm-b"][-1], meta["storm-c"][-1]}

    # the near-identical pair, tested one side at a time: `*` rows vs the named-key row.
    # reading a `*` row as key-scoped makes it match nothing, so the drained tail counts again
    assert no_star - full == set(meta["hooli-drain"][-3:] + meta["stark-failover"][-1:]), \
        "a `*` row read as key-scoped must count exactly the drained/failover tail (4 new ids)"
    # reading the named-key row as tenant-wide exempts hooli's other key too: the 2 real ids go
    assert full - wide_keyrow == set(meta["hooli-cutover-other"][-2:])
    assert not wide_keyrow - full, "the tenant-wide misreading only ever ADDS exemptions"
    # ignoring the key column merges the two streams: the row's OWN key becomes a violation
    assert set(meta["hooli-cutover-named"]) & (no_keyrow - full), \
        "ignoring the key column must invent violations on the very key the row exempts"

    decisions = {                        # each rule class, and the ids it alone decides
        "per-tenant rule decides": full - global_only,
        "global rule decides": full - per_only,
        "`*` exemption rows exclude": no_star - full,
        "named-key exemption rows exclude": no_keyrow - full,
        "join reveals": full - by_key,
        "retired-key rule excludes": no_retired - full,
    }
    assert all(decisions.values()), f"a rule class decides nothing: {decisions}"
    assert set(meta["stark-failover"][-1:]) <= no_exempt - full         # both `*` rows exclude
    assert set(meta["initech-retired"][-1:]) <= no_retired - full       # retired key: last one only
    assert meta["acme-join"][-1] in full - by_key                       # join reveals acme's 6th
    assert not (by_key - full) & set(meta["acme-join"] + meta["globex-per"]), \
        "key-grouping must not invent per-tenant violations of a real tenant"

    # the perspective check must cross-foot: the table partitions the id set
    table = checker.counts(HERE / "workspace")
    tenants = {keys[e["key"]][0] for e in ev}
    assert sum(table.values()) == len(full), "the count table must sum to the id set"
    assert table.get("global") == len(global_only), "the `global` row is the global-only ids"
    assert set(table) <= tenants | {"global"} and all(n > 0 for n in table.values()), \
        "the table's rows are tenants (or `global`), each with a positive count"

    trap = single_rule_scan(HERE / "workspace")
    assert trap != sorted(full) and trap, "the trap must be a different, non-empty set"
    key_of = {e["id"]: e["key"] for e in ev}
    trap_rows = {k: sum(1 for rid in trap if key_of[rid] == k)
                 for k in sorted({key_of[rid] for rid in trap})}
    assert sum(trap_rows.values()) == len(trap)

    answer_body = checker.body(HERE / "workspace")
    (HERE / "hidden" / "answer.txt").write_text(answer_body)
    (HERE / "reference" / "ANSWER.txt").write_text(answer_body)
    (HERE / "trap" / "ANSWER.txt").write_text(two_sections(trap, trap_rows))
    print(f"answer: {len(full)} ids   table: {table}   trap: {len(trap)} ids   log: "
          f"{sum(1 for _ in open(HERE / 'workspace' / 'service.log'))} requests")
    for name, ids in decisions.items():
        print(f"  {name:34s} {len(ids)} id(s): {' '.join(sorted(ids))}")
    if show_table:
        checker.table(HERE / "workspace")


if __name__ == "__main__":
    main("--table" in sys.argv)
