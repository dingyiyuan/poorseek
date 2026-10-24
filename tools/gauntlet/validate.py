#!/usr/bin/env python3
"""validate.py - every Gauntlet task must be valid before any model is graded on it:

  start      hidden tests FAIL on the starting workspace (otherwise there is nothing to do)
  reference  visible AND hidden tests PASS with reference/ applied (otherwise the task is impossible / wrong)
  trap       (if trap/ exists) visible tests PASS and hidden tests FAIL with trap/ applied (the trap is tempting AND caught)

  python3 tools/gauntlet/validate.py [task-dir ...]      (default: every tasks/NN-*)
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pytest_ok(d: Path, which: str, timeout: int = 300) -> bool:
    files = sorted(str(p.relative_to(d)) for p in d.rglob("test_*.py")
                   if (which == "hidden") == p.name.startswith("test_hidden_"))
    if not files:
        return True
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *files], cwd=d,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0


def stage(task: Path, overlay: str | None) -> Path:
    d = Path(tempfile.mkdtemp(prefix="gauntlet-"))
    shutil.copytree(task / "workspace", d, dirs_exist_ok=True)
    if overlay:
        shutil.copytree(task / overlay, d, dirs_exist_ok=True)
    shutil.copytree(task / "hidden", d, dirs_exist_ok=True)
    return d


def ungradable(task: Path) -> str | None:
    """An 'answer' task is graded on the transcript, a hidden-test task on pytest; with neither, nothing grades it."""
    if "answer" in json.loads((task / "task.json").read_text()):
        return None
    if any((task / "hidden").rglob("test_hidden_*.py")):
        return None
    return "no hidden test_hidden_*.py files and no 'answer' key - nothing grades this task"


def validate_answer(task: Path, pattern: str) -> bool:
    """An exact-answer task: reference/answer.txt must match the regex, trap/answer.txt (if any) must not, and a
    final message that hedges with the right answer plus the trap's must not match either."""
    import re
    ref = (task / "reference" / "answer.txt").read_text()
    checks = [("reference answer: matches", re.search(pattern, ref) is not None)]
    if (task / "trap" / "answer.txt").is_file():
        trap = (task / "trap" / "answer.txt").read_text()
        checks.append(("trap answer: rejected", re.search(pattern, trap) is None))
        checks.append(("hedged reference+trap: rejected", re.search(pattern, ref + "\nor maybe " + trap) is None))
    ok = all(c[1] for c in checks)
    print(f"{'OK  ' if ok else 'BAD '} {task.name}: " + ", ".join(f"{n} {'yes' if v else 'NO'}" for n, v in checks))
    return ok


def validate(task: Path) -> bool:
    why = ungradable(task)
    if why:
        print(f"BAD  {task.name}: {why}")
        return False
    meta = json.loads((task / "task.json").read_text())
    if "answer" in meta:
        return validate_answer(task, meta["answer"])
    checks = []
    d = stage(task, None)
    checks.append(("start: hidden FAIL", not pytest_ok(d, "hidden", meta.get("grade_timeout", 300))))
    shutil.rmtree(d)
    d = stage(task, "reference")
    checks.append(("reference: visible PASS", pytest_ok(d, "visible")))
    gt = meta.get("grade_timeout", 300)
    checks.append(("reference: hidden PASS", pytest_ok(d, "hidden", gt)))
    shutil.rmtree(d)
    if (task / "trap").is_dir():
        d = stage(task, "trap")
        checks.append(("trap: visible PASS (tempting)", pytest_ok(d, "visible")))
        checks.append(("trap: hidden FAIL (caught)", not pytest_ok(d, "hidden", meta.get("grade_timeout", 300))))
        shutil.rmtree(d)
    ok = all(c[1] for c in checks)
    print(f"{'OK  ' if ok else 'BAD '} {task.name}: " + ", ".join(f"{n} {'yes' if v else 'NO'}" for n, v in checks))
    return ok


if __name__ == "__main__":
    tasks = [Path(a).resolve() for a in sys.argv[1:]] or sorted(p for p in (HERE / "tasks").iterdir() if p.is_dir())
    sys.exit(0 if all([validate(t) for t in tasks]) else 1)
