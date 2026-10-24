import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = [('1\n1\n', '1\n'), ('4\n3 2 3 1\n', '0\n'), ('8\n2 2 2 2 2 2 1 1\n', '370\n'), ('6\n3 3 3 1 1 1\n', '6\n'), ('7\n4 4 4 4 3 3 3\n', '6\n')]


@pytest.mark.parametrize("i", range(len(EXAMPLES)))
def test_example(i):
    inp, want = EXAMPLES[i]
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                       text=True, timeout=10.0)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == want.split()
