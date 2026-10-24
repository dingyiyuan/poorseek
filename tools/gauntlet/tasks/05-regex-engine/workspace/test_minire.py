from minire import fullmatch


def test_basics():
    assert fullmatch("abc", "abc")
    assert not fullmatch("abc", "abcd")
    assert fullmatch("a*b", "aaab")
    assert fullmatch("(ab|cd)+", "abcdab")
    assert fullmatch("[a-c]x?", "b")
    assert not fullmatch("[^a-c]", "b")


def test_quantifiers_and_escapes():
    assert fullmatch("a?b", "b")
    assert not fullmatch("a+", "")
    assert fullmatch("x*", "")
    assert fullmatch("\\*\\?", "*?")
    assert fullmatch("[a-c]*", "abcabc")


# The three twists in the module docstring are NOT covered here - see the spec for the
# behaviour the hidden suite pins.
