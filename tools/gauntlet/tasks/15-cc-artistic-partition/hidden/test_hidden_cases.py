import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = json.load(open(os.path.join(HERE, "cases.json")))
LIMIT = 10.0


def run(inp):
    return subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                          text=True, timeout=LIMIT, cwd=HERE)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_case(i):
    inp, want = CASES[i]
    try:
        r = run(inp)
    except subprocess.TimeoutExpired:
        pytest.fail(f"case {i}: time limit exceeded ({LIMIT} s)")
    assert r.returncode == 0, f"case {i}: exit {r.returncode}: {r.stderr[-500:]}"
    got = r.stdout.split()
    if got != want.split():
        pytest.fail(f"case {i}: wrong answer (input {len(inp)} bytes): got {' '.join(got)[:200]!r}, "
                    f"want {' '.join(want.split())[:200]!r}")
