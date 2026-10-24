"""Grade codec.encode against frozen legacy_codec outputs. The binary is removed first and codec.py must be pure
computation (see the workspace README's rules), so the grade cannot be obtained by calling the black box."""
import ast
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("legacy_codec",):
    if os.path.exists(os.path.join(HERE, _p)):
        os.remove(os.path.join(HERE, _p))

ALLOWED = {"struct", "itertools", "functools", "math", "collections", "typing", "operator", "string"}
BANNED = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "input",
          "getattr", "setattr", "__builtins__", "memoryview"}


def test_pure_python():
    tree = ast.parse(open(os.path.join(HERE, "codec.py")).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED]
            assert not bad, f"codec.py may not import {bad}"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in ALLOWED, f"codec.py may not import from {node.module}"
        elif isinstance(node, ast.Name):
            assert node.id not in BANNED, f"codec.py may not use {node.id}"
        elif isinstance(node, ast.Attribute):
            assert not node.attr.startswith("__"), f"codec.py may not touch {node.attr}"


CASES = json.load(open(os.path.join(HERE, "expected.json")))


@pytest.mark.parametrize("chunk", range(0, len(CASES), 40))
def test_matches_legacy(chunk):
    from codec import encode
    wrong = [(h, e, encode(bytes.fromhex(h))) for h, e in CASES[chunk:chunk + 40]]
    wrong = [w for w in wrong if w[1] != w[2]]
    if wrong:
        h, e, g = wrong[0]
        pytest.fail(f"{len(wrong)}/40 differ; first: input {h[:80]}{'...' if len(h) > 80 else ''} "
                    f"({len(h) // 2} bytes) -> legacy {e!r}, yours {g!r}")


def test_too_long_rejected():
    from codec import encode
    with pytest.raises(ValueError):
        encode(b"x" * 256)
