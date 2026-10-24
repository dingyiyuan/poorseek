import json
import os

import pytest

from kx8 import run
from kx8asm import assemble
from programs import PROGRAMS

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = json.load(open(os.path.join(HERE, "expected.json")))


@pytest.mark.parametrize("name,src,max_steps", PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_program(name, src, max_steps):
    got = run(assemble(src), max_steps)
    want = EXPECTED[name]
    assert isinstance(got, dict), "run() must return a dict"
    diff = {k: (got.get(k), want[k]) for k in want if got.get(k) != want[k]}
    if diff:
        pytest.fail(f"{name}: fields differ (got, want): " + "; ".join(f"{k}={g!r} vs {w!r}" for k, (g, w) in diff.items()))


def test_default_max_steps():
    got = run(assemble("spin: B spin\n"))
    assert got["status"] == "LIMIT" and got["steps"] == 1_000_000 and got["pc"] == 0
