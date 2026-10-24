import itertools
import random

import pytest

from release_order import compare, merge, parse, select, sort_versions

# Ascending precedence. Every entry has a precedence distinct from all the others.
ORDER = [
    "2.0.0~dev",      # EPOCH 2 - equal to the next under every key except EPOCH, which is last and inverted
    "1.0.0~dev",      # EPOCH 1
    "1.1.0~dev",      # MINOR 1 > 0
    "1.1.0~dev.1",    # REVISION 1 > 0
    "1.1.0~beta",
    "1.1.0~rc",
    "1.1.0~stable",
    "1.1.0",          # no channel: above "stable", below "hotfix"
    "1.1.0~hotfix",
    "1.2.0~dev",      # MAJOR 2 > 1
    "1.2.1~dev",      # MINOR 1 > 0
]


def test_total_order():
    for i, j in itertools.combinations(range(len(ORDER)), 2):
        assert compare(ORDER[i], ORDER[j]) == -1, (ORDER[i], ORDER[j])
        assert compare(ORDER[j], ORDER[i]) == 1, (ORDER[j], ORDER[i])


def test_sorted_order():
    xs = ORDER[:]
    random.Random(11).shuffle(xs)
    assert sort_versions(xs) == ORDER


PAIRS = [
    ("1.0.0~dev", "1.0.1~dev", -1),
    ("1.0.0~dev", "1.0.0~dev", 0),
    ("1.0.0~dev", "1.0.0~beta", -1),
    ("1.0.0~beta", "1.0.0~rc", -1),
    ("1.0.0~rc", "1.0.0~stable", -1),
    ("1.0.0~stable", "1.0.0", -1),
    ("1.0.0", "1.0.0~hotfix", -1),
    ("1.0.0~hotfix", "1.0.0~stable", 1),
    ("1.0.0~hotfix", "1.0.0~rc", 1),
    ("1.0.0~beta", "1.0.0~beta.0", 0),
    ("1.0.0~beta.2", "1.0.0~beta.10", -1),
    ("1.0.0~dev.2", "1.0.0~dev.10", -1),
    ("1.0.0~dev@a", "1.0.0~dev@b", 0),
    ("1.0.0~beta.1@z", "1.0.0", -1),
]


@pytest.mark.parametrize("a,b,expected", PAIRS)
def test_pair_table(a, b, expected):
    assert compare(a, b) == expected
    assert compare(b, a) == -expected


def test_epoch_is_the_last_key_not_the_first():
    # MAJOR outranks EPOCH: 4.9.9 stands later than 5.0.1 despite the smaller epoch.
    assert compare("5.0.1~dev", "4.9.9~dev") == -1
    # ... and the reverse only holds once MAJOR and MINOR are equal, where the
    # LARGER epoch is the EARLIER stamp.
    assert compare("9.1.0~dev", "1.1.0~dev") == -1
    assert compare("1.1.0~dev", "9.1.0~dev") == 1


def test_channel_rank_is_not_alphabetical():
    # alphabetical would be beta < dev < hotfix < rc < stable
    assert compare("1.1.0~beta", "1.1.0~dev") == 1
    assert compare("1.1.0~rc", "1.1.0~beta") == 1
    assert compare("1.1.0~hotfix", "1.1.0~rc") == 1
    assert compare("1.1.0~stable", "1.1.0~rc") == 1


def test_missing_channel_beats_stable_but_not_hotfix():
    assert compare("1.1.0~stable", "1.1.0") == -1
    assert compare("1.1.0~rc", "1.1.0") == -1
    assert compare("1.1.0", "1.1.0~hotfix") == -1


def test_bare_channel_equals_revision_zero():
    assert compare("1.1.0~beta", "1.1.0~beta.0") == 0
    assert compare("1.1.0~beta.0", "1.1.0~beta") == 0


def test_stable_sort_keeps_input_order():
    xs = ["1.1.0~dev.0@b", "1.1.0~dev@a", "1.0.0~dev@z", "1.1.0~dev.0@a"]
    assert sort_versions(xs) == ["1.0.0~dev@z", "1.1.0~dev.0@b", "1.1.0~dev@a", "1.1.0~dev.0@a"]


def test_parse_shape():
    assert parse("3.14.15~hotfix.7@us-east-1") == (3, 14, 15, "hotfix", 7, "us-east-1")
    assert parse("0.0.0") == (0, 0, 0, None, 0, None)
    assert parse("1.0.0~stable@eu") == (1, 0, 0, "stable", 0, "eu")
    assert parse("2.3.4@x") == (2, 3, 4, None, 0, "x")


BAD = ["1.2", "1.2.3.4", "01.0.0", "1.02.0", "1.0.03", "v1.0.0", "1.0.0~", "1.0.0~alpha",
       "1.0.0~RC", "1.0.0~beta.", "1.0.0~beta.01", "1.0.0~beta.-1", "1.0.0~stable@",
       "1.0.0@US", "1.0.0@us.", "1.0.0@us~beta", "1.0.0@us@eu", "1.0.0+meta", " 1.0.0",
       "1.0.0 ", "1,0,0", "1.0", "1.0.0~beta.1.2"]


@pytest.mark.parametrize("bad", BAD)
def test_invalid_raises_value_error(bad):
    with pytest.raises(ValueError):
        parse(bad)
    with pytest.raises(ValueError):
        compare(bad, "1.0.0")


def test_compare_rejects_any_invalid_argument():
    with pytest.raises(ValueError):
        compare(None, "1.0.0")
    with pytest.raises(ValueError):
        compare("1.0.0", 1)
    with pytest.raises(ValueError):
        sort_versions(["1.0.0", "nope"])


# --- merge: the union of two catalogs, by identity -----------------------------

def test_merge_is_a_stable_union():
    assert merge(["1.0.0~dev"], ["1.1.0~dev"]) == ["1.0.0~dev", "1.1.0~dev"]
    assert merge(["1.1.0~dev"], ["1.0.0~dev"]) == ["1.0.0~dev", "1.1.0~dev"]
    # MINOR 10 follows MINOR 2, though "1.10.0~dev" sorts first as a string
    assert merge(["1.10.0~dev", "1.2.0~dev"], []) == ["1.2.0~dev", "1.10.0~dev"]
    # no channel sits above "stable" and below "hotfix"
    assert merge([], ["1.1.0~hotfix", "1.1.0", "1.1.0~stable"]) == [
        "1.1.0~stable", "1.1.0", "1.1.0~hotfix"]


def test_merge_identity_ignores_region():
    # the same release under two regions survives once, spelling it as catalog does
    assert merge(["1.2.0~rc@us"], ["1.2.0~rc@eu"]) == ["1.2.0~rc@us"]
    assert merge(["1.2.0~rc@eu"], ["1.2.0~rc@us"]) == ["1.2.0~rc@eu"]
    # a byte-identical release in both lists appears once
    assert merge(["1.1.0~beta"], ["1.1.0~beta"]) == ["1.1.0~beta"]
    # "1.1.0~beta" and "1.1.0~beta.0" name the same release
    assert merge(["1.1.0~beta.0"], ["1.1.0~beta"]) == ["1.1.0~beta.0"]


def test_merge_rejects_a_list_naming_one_release_twice():
    for bad in (["1.0.0~dev", "1.0.0~dev"],
                ["1.2.0~rc@us", "1.2.0~rc@eu"],
                ["1.1.0~beta", "1.1.0~beta.0"]):
        with pytest.raises(ValueError):
            merge(bad, [])
        with pytest.raises(ValueError):
            merge([], bad)


def test_merge_rejects_invalid_stamps():
    with pytest.raises(ValueError):
        merge(["1.0.0"], ["nope"])
    with pytest.raises(ValueError):
        merge(["nope"], ["1.0.0"])


# --- select: the constraint grammar -------------------------------------------

def test_select_empty_and_open_constraints():
    xs = ["1.0.0~rc@us", "1.1.0~dev", "1.2.0~stable"]
    assert select(xs, "") == ["1.0.0~rc@us", "1.1.0~dev", "1.2.0~stable"]
    assert select(xs, "..") == select(xs, "")


def test_select_bare_stamp_matches_identity_not_spelling():
    xs = ["1.1.0~beta", "1.1.0~beta.0", "1.1.0~beta@eu", "1.1.0~beta.1", "1.1.0~rc"]
    assert select(xs, "1.1.0~beta") == ["1.1.0~beta", "1.1.0~beta.0", "1.1.0~beta@eu"]


def test_select_range_is_low_inclusive_high_exclusive():
    xs = ["1.0.0~rc", "1.1.0~rc", "1.2.0~rc"]
    assert select(xs, "1.0.0~rc..1.1.0~rc") == ["1.0.0~rc"]
    assert select(xs, "..1.1.0~rc") == ["1.0.0~rc"]
    assert select(xs, "1.1.0~rc..") == ["1.1.0~rc", "1.2.0~rc"]


def test_select_era_operator_is_epoch_and_major_and_channel_floor():
    xs = ["1.2.0~dev", "1.2.0~stable", "1.2.0", "1.2.0~hotfix", "1.2.5~stable",
          "2.2.0~hotfix", "2.2.0~stable"]
    assert select(xs, "^1.2.0~rc") == ["1.2.0~stable", "1.2.0", "1.2.0~hotfix",
                                       "1.2.5~stable"]
    # the ERA is (EPOCH, MAJOR): a different EPOCH is a different era
    assert select(xs, "^2.2.0~dev") == ["2.2.0~stable", "2.2.0~hotfix"]
    assert select(["1.2.0~stable", "2.2.0~stable"], "^1.2.0~dev") == ["1.2.0~stable"]


def test_select_returns_ascending_order_and_is_stable():
    xs = ["1.1.0~dev@b", "1.0.0~dev@z", "1.1.0~dev.0@a", "1.1.0~dev@a"]
    assert select(xs, "1.1.0~dev") == ["1.1.0~dev@b", "1.1.0~dev.0@a", "1.1.0~dev@a"]


def test_select_rejects_bad_constraints_and_bad_lists():
    for bad in ["^", ">=1.0.0", "1.0.0..2.0.0..3.0.0", "1.0.0..nope", "nope..",
                "^nope", "1.2", "..1.0.0.0", "~1.0.0"]:
        with pytest.raises(ValueError):
            select(["1.0.0"], bad)
    with pytest.raises(ValueError):
        select(["nope"], "")
    with pytest.raises(ValueError):
        select("1.0.0", "")
    with pytest.raises(ValueError):
        select(["1.0.0"], None)


# --- the three surfaces agree --------------------------------------------------

def test_merge_and_select_agree_on_the_union():
    catalog = ["1.0.0~dev", "1.1.0~rc@us", "1.2.0"]
    incoming = ["1.0.0~dev", "1.1.0~rc@eu", "1.1.0~hotfix"]
    rank = {"dev": 0, "beta": 1, "rc": 2, "stable": 3}

    def pid(s):
        epoch, major, minor, channel, revision, _ = parse(s)
        return (major, minor, rank.get(channel, 4), revision, -epoch)

    merged = merge(catalog, incoming)
    for constraint in ["", "..", "1.0.0~dev", "1.1.0~rc..1.2.0", "1.1.0~hotfix..",
                       "^1.0.0~dev", "..1.1.0~rc", "1.1.0~rc@eu"]:
        left = [pid(s) for s in select(merged, constraint)]
        right = sorted(set(pid(s) for s in select(catalog, constraint))
                       | set(pid(s) for s in select(incoming, constraint)))
        assert left == right, constraint
