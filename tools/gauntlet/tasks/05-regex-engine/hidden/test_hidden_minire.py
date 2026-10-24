"""Hidden grading for 05-regex-engine.

Every assertion is derivable from the workspace docstring (the import ban from the task
prompt).  Sections: the documented examples; malformed patterns; the three twists, each
pinned separately - including the cases where a correct Python-`re`-shaped implementation
returns the other answer; a differential fuzz against Python's `re` restricted to the
subset where this engine's semantics coincide with it; the performance requirement.
"""
import random
import re
import signal

import pytest

from minire import fullmatch


class _Slow(Exception):
    pass


def _alarm(sig, frame):
    raise _Slow()


def timed(fn, seconds=2):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)


def m(pat, text, seconds=2):
    """fullmatch under a time limit, so a hanging engine fails instead of hanging."""
    return timed(lambda: fullmatch(pat, text), seconds)


# --- 1. the documented examples -------------------------------------------------
DOC = [
    # rule 1 - every choice is made once, and it is the longest one
    ("a|ab", "ab", True), ("ab|a", "ab", True), ("a|ab", "a", True),
    ("(a|aa)a", "aa", False), ("(a|aa)a", "aaa", True),
    ("a*a", "aa", False), ("a*ab", "ab", False), ("a*ab", "aaab", False),
    ("(ab|a)(bc|b)c", "abbc", False), ("(a?)(a)", "a", False),
    ("(a|ab)(c|bcd)(d*)", "abcd", True),
    ("(|b)c", "c", True), ("a|", "", True), ("a|", "a", True),
    # rule 2 - `^`, `$` and `.`
    ("a.c", "a\nc", True), ("a\n^b", "a\nb", True), ("a$\nb", "a\nb", True),
    ("ab^c", "abc", False), ("ab$c", "abc", False), ("a$b", "a\nb", False),
    ("^b", "b", True), ("a$", "a", True), ("^\n^", "\n", True), ("$^", "", True),
    ("[$]", "$", True), ("[^$]", "$", False), ("[~]", "~", True),
    # rule 3 - the invented `~`
    ("a~", "a", True), ("a~a", "aa", True), ("a~a~", "aa", False),
    ("a*~", "aaa", True), ("a~*", "aaa", False), ("a~*", "a", True),
    ("(a~)*", "aa", False), ("(a~)*", "a", True), ("(ab)~", "ab", True),
    ("b(a)~", "ba", False), ("ab~", "ab", False), ("b(a~)", "ba", False),
    # the rest of the grammar
    ("", "", True), ("", "a", False), ("abc", "abc", True), ("abc", "abcd", False),
    ("a*b", "aaab", True), ("(ab|cd)+", "abcdab", True), ("[a-c]x?", "b", True),
    ("[^a-c]", "b", False), ("[]a]", "]", True), ("[\\]]", "]", True),
    ("[a\\-z]", "-", True), ("[a-]", "-", True), ("[^]]", "a", True),
    ("\\(a\\)", "(a)", True), ("x*", "", True), ("(a*)+", "aaa", True),
    ("(a?)*b", "b", True), ("(a*)*", "", True), ("(a*)*", "aa", True),
    ("\\.", ".", True), ("\\~", "~", True), ("\\^", "^", True), ("\\$", "$", True),
    ("[a-c]*\n?", "abc\n", True),
]


@pytest.mark.parametrize("pat,text,want", DOC)
def test_documented_examples(pat, text, want):
    assert m(pat, text) == want


# --- 2. malformed patterns ------------------------------------------------------
BAD = ["(", "a)", "*a", "[a", "a|*", "(?", "\\", "+a", "?x", "(a", "[^", "[z-a]",
       "~a", "^*", "$+", "a^*", "a$+", "a~$*"]


@pytest.mark.parametrize("bad", BAD)
def test_malformed_raises_value_error(bad):
    with pytest.raises(ValueError):
        fullmatch(bad, "a")


# --- 3a. twist one: a choice is committed, and it is the longest one -------------
BACKTRACKED = [("(a|aa)a", "aa"), ("a*a", "aaa"), ("a*ab", "ab"), ("a*ab", "aaab"),
               ("(ab|a)(bc|b)c", "abbc"), ("(a?)(a)", "a")]


def test_commitment_rejects_what_a_backtracking_engine_accepts():
    # a correct Python-`re`-shaped engine (leftmost-first, gives choices back) accepts
    # every one of these; this engine commits to the longest candidate and does not
    for pat, text in BACKTRACKED:
        assert re.fullmatch(pat, text, re.S | re.M) is not None, (pat, text)
        assert m(pat, text) is False, (pat, text)


def test_alternation_takes_the_longest_branch():
    assert fullmatch("a|ab", "ab") and fullmatch("ab|a", "ab")
    assert fullmatch("a|ab", "a") and not fullmatch("a|ab", "b")
    assert not fullmatch("(ab|a)(bc|b)c", "abbc")
    assert fullmatch("(ab|a)(b|c)c", "abbc")


# --- 3b. twist two: `.` matches a newline; `^`/`$` are always line-wise ---------
@pytest.mark.parametrize("pat,text", [("a.c", "a\nc"), ("a\n^b", "a\nb"),
                                      ("a$\nb", "a\nb")])
def test_twist_two_disagrees_with_pythons_default_flags(pat, text):
    assert re.fullmatch(pat, text) is None      # Python's defaults reject these
    assert m(pat, text) is True


@pytest.mark.parametrize("pat,text,want", [
    ("^b", "a\nb", False),      # fullmatch consumes the whole text: `^` does not skip it
    ("a\n^b", "a\nb", True),    # `^` holds right after a newline
    ("^\n^", "\n", True),       # ... including both ends of a one-newline text
    ("a$", "a\n", False),       # `$` is zero width: it cannot consume the newline
    ("a$\n", "a\n", True),
    ("a$\nb", "a\nb", True),
    ("\n$", "\n", True),        # a literal newline in the pattern, `$` at the end
    ("[$]$", "$", True),        # a class holds a literal $, the assertion is zero width
    ("[$^~]", "^", True),       # ... and so are a non-leading ^ and a ~ inside a class
    ("[^$]+", "$a", False),
])
def test_twist_two_anchors_are_line_wise_and_zero_width(pat, text, want):
    assert m(pat, text) == want


# --- 3c. twist three: the invented `~` ----------------------------------------
@pytest.mark.parametrize("pat,text,want", [
    ("a~", "a", True), ("a~", "aa", False),
    ("a~a", "aa", True), ("a~a~", "aa", False), ("a~aa", "aaa", True),
    ("a*~", "aaa", True), ("a~*", "aaa", False), ("a~*", "a", True),
    ("(a~)*", "aaa", False), ("(ab)~", "ab", True), ("ab~", "ab", False),
    ("b(a)~", "ba", False), ("b(a~)", "ba", False), ("b(ab)~", "bab", False),
    ("ab(ab)~", "abab", True),
    ("\\~", "~", True), ("[~]", "~", True), ("[~]+", "~~~", True),
])
def test_twist_three_odd_offset_operator(pat, text, want):
    assert m(pat, text) == want


def test_twist_three_binds_to_the_thing_it_follows():
    assert fullmatch("a*~", "aaa") and not fullmatch("a~*", "aaa")
    assert fullmatch("(ab)~", "ab") and not fullmatch("ab~", "ab")
    # the group starts at offset 3 here (odd), so its `~` holds; after one character it
    # would start at offset 2 and the same group must then fail
    assert fullmatch("ab(ab)~", "abab") and not fullmatch("b(ab)~", "bab")


# --- 4. differential fuzz against `re`, on the subset where the semantics agree --
ATOMS = ["a", "b", "c", ".", "[ab]", "[^a]", "[a-c]", "\\.", "\\*", "\\~"]


def possessive(pat):
    """`*` -> `*+` etc: Python's possessive quantifiers are exactly this engine's rule 1,
    so where the syntax has no alternation and no `~` the two engines must agree."""
    out, i = [], 0
    while i < len(pat):
        if pat[i] == "\\":
            out.append(pat[i:i + 2])
            i += 2
            continue
        out.append(pat[i] + "+" if pat[i] in "*+?" else pat[i])
        i += 1
    return "".join(out)


def gen(r, anchors, depth=0):
    """A random pattern with no `|` and no `~`, and `^`/`$` only when `anchors`."""
    items = []
    for _ in range(r.randint(1, 3)):
        if anchors and r.random() < 0.10:
            items.append(r.choice(["^", "$"]))
            continue
        atom = r.choice(ATOMS)
        if depth < 3 and r.random() < 0.25:
            atom = "(" + gen(r, anchors, depth + 1) + ")"
        items.append(atom + r.choice(["", "", "*", "+", "?"]))
    return "".join(items)


@pytest.mark.parametrize("anchors,flags,seed", [(False, re.S, 4242),
                                                (True, re.S | re.M, 1337)])
def test_fuzz_against_re_where_the_semantics_coincide(anchors, flags, seed):
    r = random.Random(seed)
    checked = 0
    for _ in range(300):
        pat = gen(r, anchors)
        for _ in range(6):
            text = "".join(r.choice("abc\n.*~") for _ in range(r.randint(0, 7)))
            want = re.fullmatch(possessive(pat), text, flags) is not None
            assert m(pat, text) == want, (pat, text, want)
            checked += 1
    assert checked == 1800


# --- 5. the performance requirement --------------------------------------------
PERF = [("(a*)*b", "a" * 40), ("(a|aa)*c", "a" * 60), ("(a*)*(b*)*c", "ab" * 25),
        ("((a+)+)+d", "a" * 35), ("(.*)*x", "y" * 50)]


@pytest.mark.parametrize("pat,text", PERF)
def test_no_catastrophic_backtracking(pat, text):
    assert m(pat, text, 2) is False


# --- 6. the import ban ---------------------------------------------------------
def test_does_not_use_re():
    import ast
    import os
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "minire.py")).read()
    modules = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    roots = {m.lstrip(".").split(".")[0] for m in modules}
    assert not (roots & {"re", "regex"}), f"minire.py imports a regex library: {sorted(roots & {'re', 'regex'})}"
