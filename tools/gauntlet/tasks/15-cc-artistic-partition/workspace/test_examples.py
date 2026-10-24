import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [('4\n6 2\n4 4\n3 1\n10 3\n', '8\n4\n6\n11\n')]


@pytest.mark.parametrize("i", range(len(EXAMPLES)))
def test_example(i):
    inp, want = EXAMPLES[i]
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                       text=True, timeout=10.0)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == want.split()
