import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [('5 2\n0 0 2 2 0\n', '6546\n'), ('4 1\n0 0 1 1\n', '509\n'), ('4 1\n0 0 0 0\n', '431\n'), ('3 2\n-2 0 4\n', '11\n'), ('4 0\n0 0 0 0\n', '256\n')]


@pytest.mark.parametrize("i", range(len(EXAMPLES)))
def test_example(i):
    inp, want = EXAMPLES[i]
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                       text=True, timeout=10.0)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == want.split()
