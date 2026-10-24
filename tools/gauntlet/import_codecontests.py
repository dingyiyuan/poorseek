#!/usr/bin/env python3
"""import_codecontests.py - turn a hard Codeforces problem from DeepMind's CodeContests into a Gauntlet task.

  import_codecontests.py <parquet dir> "<problem name>" <task dir> [--max-cases 40]

The task: workspace/PROBLEM.md (the statement, limits, examples), a solution.py stub that reads stdin and writes
stdout, and a visible test over the public examples. hidden/ holds the private + generated tests (capped, the
largest kept), graded by running solution.py as a script with 10x the original (C++-calibrated) time limit and
comparing whitespace-separated tokens. reference/ = an accepted C++ solution compiled at run time by a Python
wrapper; trap/ = an incorrect accepted-language submission from the dataset that passes the public examples and
fails the hidden tests (the importer searches for one and refuses the problem if none exists).
"""
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

CPP = 2
WRAPPER = '''"""{what}: compiles the C++ below with g++ -O2 (once, cached next to this file) and runs it on stdin."""
import hashlib
import os
import subprocess
import sys

SRC = {src!r}

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, ".build-" + hashlib.sha1(SRC.encode()).hexdigest()[:12])
if not os.path.exists(exe):
    with open(exe + ".cpp", "w") as f:
        f.write(SRC)
    subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], check=True)
sys.exit(subprocess.run([exe]).returncode)
'''

HIDDEN_TEST = '''import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = json.load(open(os.path.join(HERE, "cases.json")))
LIMIT = {limit}


def run(inp):
    return subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                          text=True, timeout=LIMIT, cwd=HERE)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_case(i):
    inp, want = CASES[i]
    try:
        r = run(inp)
    except subprocess.TimeoutExpired:
        pytest.fail(f"case {{i}}: time limit exceeded ({{LIMIT}} s)")
    assert r.returncode == 0, f"case {{i}}: exit {{r.returncode}}: {{r.stderr[-500:]}}"
    got = r.stdout.split()
    if got != want.split():
        pytest.fail(f"case {{i}}: wrong answer (input {{len(inp)}} bytes): got {{' '.join(got)[:200]!r}}, "
                    f"want {{' '.join(want.split())[:200]!r}}")
'''

VISIBLE_TEST = '''import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = {examples!r}


@pytest.mark.parametrize("i", range(len(EXAMPLES)))
def test_example(i):
    inp, want = EXAMPLES[i]
    r = subprocess.run([sys.executable, os.path.join(HERE, "solution.py")], input=inp, capture_output=True,
                       text=True, timeout={limit})
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == want.split()
'''


def run_cpp(src, cases, limit):
    """Compile src and run every case; return the list of booleans (passed) or None if it does not compile."""
    d = tempfile.mkdtemp(prefix="cc-")
    exe = os.path.join(d, "a")
    open(exe + ".cpp", "w").write(src)
    if subprocess.run(["g++", "-O2", "-std=gnu++17", "-o", exe, exe + ".cpp"], capture_output=True).returncode:
        return None
    out = []
    for inp, want in cases:
        try:
            r = subprocess.run([exe], input=inp, capture_output=True, text=True, timeout=limit)
            out.append(r.returncode == 0 and r.stdout.split() == want.split())
        except subprocess.TimeoutExpired:
            out.append(False)
    return out


def main():
    src_dir, name, task = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    max_cases = int(sys.argv[sys.argv.index("--max-cases") + 1]) if "--max-cases" in sys.argv else 40
    d = pd.concat([pd.read_parquet(p) for p in sorted(Path(src_dir).glob("*.parquet"))])
    r = d[d.name == name].iloc[0]
    tl = float(r.time_limit["seconds"]) + float(r.time_limit["nanos"]) / 1e9 if r.time_limit is not None else 2.0
    # CodeContests' tests are small (measured: the accepted C++ needs <= 0.5 s on every chosen case of 14-17), so a
    # flat 10 s per test is >= 20x the reference including interpreter start, yet keeps a TLE run's grading bounded
    limit = 10.0
    pub = list(zip(r.public_tests["input"], r.public_tests["output"]))
    hid = list(zip(r.private_tests["input"], r.private_tests["output"])) + \
        list(zip(r.generated_tests["input"], r.generated_tests["output"]))
    hid = [c for c in hid if c not in pub]
    rnd = random.Random(name)
    big = sorted(hid, key=lambda c: -len(c[0]))[: max_cases // 2]
    rest = [c for c in hid if c not in big]
    hid = big + rnd.sample(rest, min(len(rest), max_cases - len(big)))
    # reference: the first accepted C++ solution that passes every chosen case (public + hidden) at the original limit
    ref = None
    for lang, src in zip(r.solutions["language"], r.solutions["solution"]):
        if lang == CPP:
            res = run_cpp(src, pub + hid, 3 * tl + 1)
            if res and all(res):
                ref = src
                break
    if ref is None:
        sys.exit(f"{name}: no accepted C++ solution passes the chosen cases")
    trap = None
    for lang, src in zip(r.incorrect_solutions["language"], r.incorrect_solutions["solution"]):
        if lang == CPP:
            res = run_cpp(src, pub + hid, 3 * tl + 1)
            if res and all(res[: len(pub)]) and not all(res[len(pub):]):
                trap = src
                break
    if trap is None:
        print(f"{name}: no incorrect submission passes the examples and fails hidden - task gets no trap/")
    for sub in ("workspace", "hidden", "reference") + (("trap",) if trap else ()):
        (task / sub).mkdir(parents=True, exist_ok=True)
    rating = int(r.cf_rating)
    statement = r.description.replace("$$$", "$")
    (task / "workspace" / "PROBLEM.md").write_text(
        f"# {name}  (Codeforces, rating {rating})\n\n"
        f"Time limit: {tl:g} s per test on Codeforces (C++). **Here: {limit:g} s per test**, measured on `solution.py` "
        f"run as `python3 solution.py < input`. Memory: {r.memory_limit_bytes // 2**20} MB.\n\n"
        "Write `solution.py`: read the input from stdin, print the answer to stdout. Any approach is allowed: a pure "
        "Python solution of the intended complexity may be too slow on the largest tests, so `solution.py` may also "
        "write a C++ source file and compile it with `g++ -O2` (on PATH) and run it - cache the binary next to "
        "solution.py. Output is compared token by token.\n\n---\n\n" + statement + "\n")
    (task / "workspace" / "solution.py").write_text(
        '"""Solve PROBLEM.md: read stdin, write stdout."""\nimport sys\n\n\ndef main():\n    data = sys.stdin.read()\n'
        '    raise NotImplementedError\n\n\nif __name__ == "__main__":\n    main()\n')
    (task / "workspace" / "test_examples.py").write_text(VISIBLE_TEST.format(examples=pub, limit=limit))
    (task / "hidden" / "test_hidden_cases.py").write_text(HIDDEN_TEST.format(limit=limit))
    json.dump(hid, open(task / "hidden" / "cases.json", "w"))
    (task / "reference" / "solution.py").write_text(WRAPPER.format(what="REFERENCE: an accepted Codeforces "
                                                                   "submission", src=ref))
    if trap:
        (task / "trap" / "solution.py").write_text(WRAPPER.format(what="THE TRAP: a rejected Codeforces "
                                                                  "submission that passes the examples", src=trap))
    tid = task.name
    json.dump({"id": tid, "category": f"competitive programming (Codeforces {rating})",
               "prompt": f"Solve the problem in PROBLEM.md by writing solution.py (stdin -> stdout). It is a real "
                         f"Codeforces problem rated {rating}; the visible tests are its examples, and the hidden tests "
                         f"are {len(hid)} official/generated cases graded token-by-token with a {limit:g} s limit each. "
                         "Use the tools; when you are done, call submit.",
               "max_steps": 12, "max_test_runs": 6, "max_python_runs": 12, "grade_timeout": int(limit * len(hid) + 120),
               "trap": "a submission Codeforces rejected: passes the examples, fails the hidden tests",
               "source": f"deepmind/code_contests: {name}"},
              open(task / "task.json", "w"), indent=1)
    print(f"{tid}: {name} rating {rating}, {len(pub)} examples, {len(hid)} hidden, limit {limit}s")


if __name__ == "__main__":
    main()
