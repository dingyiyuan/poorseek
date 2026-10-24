"""Release-stamp ordering - THE TRAP: the natural, unread reading of the spec.

Four plausible mistakes, each invisible to the visible suite:
  * EPOCH compared first, left to right, ascending (it looks like the leading key);
  * channels ranked alphabetically (beta < dev < hotfix < rc < stable) with a
    missing channel treated as the highest of all;
  * merge concatenated and sorted as plain strings (no identity, no stability);
  * select closing its upper bound (inclusive both ends), matching a bare stamp
    by its exact spelling, and reading "^" as "same MAJOR" rather than same ERA.
"""
import functools
import re

_CORE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_REVISION = re.compile(r"^(0|[1-9][0-9]*)$")
_REGION = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ALPHABETICAL_RANK = {"beta": 0, "dev": 1, "hotfix": 2, "rc": 3, "stable": 4}
_NO_CHANNEL_RANK = 5


def parse(stamp):
    if not isinstance(stamp, str):
        raise ValueError("a stamp must be a string")
    core, at, region = stamp.partition("@")
    if at and not _REGION.match(region):
        raise ValueError(f"bad region {region!r}")
    core, tilde, tail = core.partition("~")
    channel, revision = None, 0
    if tilde:
        name, dot, rev = tail.partition(".")
        if name not in _ALPHABETICAL_RANK:
            raise ValueError(f"bad channel {name!r}")
        channel = name
        if dot:
            if not _REVISION.match(rev):
                raise ValueError(f"bad revision {rev!r}")
            revision = int(rev)
    m = _CORE.match(core)
    if not m:
        raise ValueError(f"bad core {core!r}")
    return int(m[1]), int(m[2]), int(m[3]), channel, revision, region if at else None


def _rank(channel):
    return _NO_CHANNEL_RANK if channel is None else _ALPHABETICAL_RANK[channel]


def _key(stamp):
    epoch, major, minor, channel, revision, _ = parse(stamp)
    return epoch, major, minor, _rank(channel), revision


def compare(a, b):
    ka, kb = _key(a), _key(b)
    return (ka > kb) - (ka < kb)


def sort_versions(stamps):
    return sorted(stamps, key=functools.cmp_to_key(compare))


def merge(catalog, incoming):
    for stamp in list(catalog) + list(incoming):
        parse(stamp)
    return sorted(set(catalog) | set(incoming))


def select(stamps, constraint):
    for stamp in stamps:
        parse(stamp)
    if not isinstance(constraint, str):
        raise ValueError("a constraint must be a string")
    if constraint in ("", ".."):
        keep = lambda s: True
    elif ".." in constraint:
        low, _, high = constraint.partition("..")
        low_key = _key(low) if low else None
        high_key = _key(high) if high else None
        keep = lambda s: ((low_key is None or _key(s) >= low_key)
                          and (high_key is None or _key(s) <= high_key))
    elif constraint.startswith("^"):
        _, major, _, channel, _, _ = parse(constraint[1:])
        floor = _rank(channel)
        keep = lambda s: parse(s)[1] == major and _rank(parse(s)[3]) >= floor
    else:
        target = parse(constraint)
        keep = lambda s: parse(s) == target
    return sort_versions([s for s in stamps if keep(s)])
