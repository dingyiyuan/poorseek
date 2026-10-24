import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [('6 4\n15 2 18 12 13 4\n', '100\n'), ('1 1\n114514\n', '0\n'), ('2 1\n666 888\n', '0\n'), ('11 5\n9 3 7 1 8 12 10 20 15 18 5\n', '176\n')]


@pytest.mark.parametrize("i", range(len(EXAMPLES)))
def test_example(i):
    inp, want = EXAMPLES[i]
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                       text=True, timeout=10.0)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == want.split()
