import re


def parse(text):
    """ANSWER.txt -> (ids, rows): the two documented sections, split on their header lines."""
    lines = text.splitlines()
    assert lines and lines[0] == "# ids", "ANSWER.txt must start with the line '# ids'"
    i = lines.index("# counts")
    ids, rows = lines[1:i], []
    for ln in lines[i + 1:]:
        m = re.fullmatch(r"(\S+) (\d+)", ln)
        assert m, f"every counts row must be '<name> <count>': {ln!r}"
        rows.append((m.group(1), int(m.group(2))))
    return ids, rows


def test_answer_format():
    """ANSWER.txt has two sections: '# ids' (one 8-hex request id per line, sorted ascending,
    no duplicates) and '# counts' ('<name> <count>' rows, sorted by name, every count
    positive) - the counts of the same violations, so they sum to the number of ids."""
    lines = open("ANSWER.txt").read().splitlines()
    assert lines.count("# ids") == 1 and lines.count("# counts") == 1, \
        "exactly one '# ids' line and one '# counts' line, in that order"
    ids, rows = parse("\n".join(lines))
    assert ids, "ANSWER.txt must list at least one request id"
    assert all(re.fullmatch(r"[0-9a-f]{8}", ln) for ln in ids), \
        "every line of the '# ids' section must be exactly one request id (8 hex digits)"
    assert ids == sorted(set(ids)), "one id per line, sorted ascending, no duplicates"
    names = [n for n, _ in rows]
    assert names == sorted(set(names)), "one row per name, sorted ascending, no duplicates"
    assert rows and all(c > 0 for _, c in rows), "a row appears only with a positive count"
    assert sum(c for _, c in rows) == len(ids), \
        "the counts must sum to the number of violating ids (each id sits on one row)"
