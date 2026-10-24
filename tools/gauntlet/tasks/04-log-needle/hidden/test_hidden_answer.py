import os


def parse(path):
    """ANSWER.txt (or hidden/answer.txt) -> (ids, rows): the two documented sections.

    The format is the one workspace/test_answer_format.py pins: the line '# ids', the ids,
    the line '# counts', the rows.
    """
    lines = open(path).read().splitlines()
    i = lines.index("# counts")
    ids = lines[lines.index("# ids") + 1:i]
    rows = {}
    for ln in lines[i + 1:]:
        name, n = ln.split()
        rows[name] = int(n)
    return ids, rows


def test_the_violating_requests():
    """The '# ids' section must be exactly the violating request ids under every documented rule
    (limits.yaml's COUNTED list and half-open window, api_keys.txt's key -> tenant join and the
    retired key, and BOTH forms of exemptions.txt row; see the derivability table in the report)."""
    want, _ = parse(os.path.join(os.path.dirname(__file__), "answer.txt"))
    got, _ = parse("ANSWER.txt")
    assert got == want, (
        f"the '# ids' section must list exactly the {len(want)} violating request ids, sorted, "
        f"one per line - missing {sorted(set(want) - set(got))}, extra {sorted(set(got) - set(want))}")


def test_the_violation_counts():
    """The '# counts' section must be the per-tenant table OF THOSE SAME ids - the tenant of each
    request the per-tenant limit flags, `global` for an id only the global limit decides - so the
    two representations pin one computation and the counts sum to the number of ids."""
    want_ids, want = parse(os.path.join(os.path.dirname(__file__), "answer.txt"))
    got_ids, got = parse("ANSWER.txt")
    assert got == want, (
        f"the '# counts' section must be {want} (one row per violating tenant, and `global` for "
        f"an id only the global limit decides), not {got}")
    assert sum(got.values()) == len(want_ids), \
        "the counts must sum to the number of violating ids the rules produce - one row per id"
