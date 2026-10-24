"""Release-stamp ordering - the artifact-naming spec of this build system.

A release stamp has the shape

    EPOCH "." MAJOR "." MINOR [ "~" CHANNEL [ "." REVISION ] ] [ "@" REGION ]

written entirely in lower-case ASCII. Nothing else is allowed: no "v" prefix, no
whitespace, no character outside the shape, no missing field.

  * EPOCH, MAJOR, MINOR: decimal integers with no leading zero - "0" and "7" are
    fine, "01" and "007" are not.
  * CHANNEL: exactly one of the five names dev, beta, rc, stable, hotfix. No
    other channel name exists, so "alpha", "nightly" and "RC" are invalid.
  * REVISION: a decimal integer with no leading zero. It may appear ONLY
    together with a channel, and a channel written without a REVISION carries
    REVISION 0 - so "1.2.0~beta" and "1.2.0~beta.0" have the same REVISION.
  * REGION: one or more characters from [a-z0-9-], starting with a letter or a
    digit and containing no "." - "us-east-1" is fine, "-x", "us.", "us@eu" and
    "US" are not.

Precedence. compare() applies these keys in order and the first difference
decides; every key outranks all the keys listed below it:

  1. MAJOR, numerically: a larger MAJOR is later.
  2. MINOR, numerically: a larger MINOR is later.
  3. CHANNEL, by the fixed rank table below. This is NOT the alphabetical order
     of the five names, and a stamp with NO channel ranks between "stable" and
     "hotfix" - omitting the channel beats writing "stable":

         dev(0) < beta(1) < rc(2) < stable(3) < (no channel, 4) < hotfix(5)

  4. REVISION, numerically: a larger REVISION is later.
  5. EPOCH, inverted: only when every key above is equal, the stamp with the
     LARGER epoch is the EARLIER one. EPOCH is the last key, never the first,
     and it is compared backwards.

REGION never takes part in precedence: "1.2.0@us" and "1.2.0@eu" compare equal.
sort_versions() is stable, so stamps of equal precedence keep their input order.

ValueError is raised for any string that does not match the shape and the field
rules - for instance "1.2" (no MINOR), "01.0.0" (leading zero), "1.0.0~" (no
channel name), "1.0.0~alpha" (no such channel), "1.0.0~beta." (empty REVISION),
"1.0.0~beta.01" (leading zero), "1.0.0@us~beta" ("@" before "~"), "1.0.0@US"
(upper case) - and compare() raises it for any argument that is not a valid
stamp string.

Two further surfaces must agree with this same ordering.

  IDENTITY. A release's identity is the tuple (EPOCH, MAJOR, MINOR, CHANNEL,
  REVISION) - the fields the ordering compares, taken as written (channel None
  when absent, revision 0 when absent). REGION is NOT part of the identity: it
  never takes part in precedence, so "1.2.0~rc@us" and "1.2.0~rc@eu" are the
  SAME release, and two stamps have equal precedence exactly when their
  identities are equal.

  merge(catalog, incoming) returns a NEW list: the union of the two, sorted
  ascending by precedence. Every stamp in either argument must be valid, else
  ValueError.

    * A single argument may not name the same release twice. Two stamps with
      equal identity inside one list raise ValueError, whether they are written
      identically or differ only in their REGION.
    * A release named in BOTH arguments is a duplicate and appears ONCE, and the
      surviving spelling is the one from catalog - merge never rewrites a
      release that already exists. (This is what stability buys: merge assembles
      catalog then incoming, sorts stably, and keeps the first spelling seen for
      each identity.)
    * Nothing else is deduplicated, and after this no two survivors can compare
      equal: equal precedence means equal identity.

  select(stamps, constraint) returns a new list of the stamps of `stamps` that
  satisfy `constraint`, in ascending precedence order and stable for stamps of
  equal precedence; every stamp in the list must be valid and `constraint` must
  be a string, else ValueError. The grammar, and nothing beyond it:

    ""        every stamp.  ".."  every stamp.
    "A"       the stamps whose IDENTITY equals A's - so "1.2.0~rc" selects
              "1.2.0~rc", "1.2.0~rc.0" and any REGION of it alike.
    "A..B"    A <= s < B by precedence: the LOW bound is INCLUSIVE and the HIGH
              bound is EXCLUSIVE.  "A.."  s >= A.   "..B"  s < B.
    "^A"      the same ERA as A and a channel at or after A's. The ERA is the
              pair (EPOCH, MAJOR) and the test is "same ERA and channel rank
              >= rank(A)"; MINOR and REVISION are unconstrained. "^1.2.0~rc" is
              every release with EPOCH 1 and MAJOR 2 whose channel is rc,
              stable, absent or hotfix, at any MINOR.

  A constraint that is not exactly one of those forms raises ValueError - an
  unknown operator such as ">=1.0.0", a "^" with no anchor, more than one "..",
  or a bound that is not a valid stamp. Because merge is precisely the union and
  select precisely the filter, select(merge(a, b), c) lists exactly the releases
  of select(a, c) and select(b, c) together, each release once.
"""


def parse(stamp):
    """Return (epoch, major, minor, channel, revision, region): the three numbers and
    the revision as ints, channel and region as the strings as written (None when the
    stamp omits them, revision 0 when it omits it). ValueError if the stamp is invalid.
    """
    raise NotImplementedError


def compare(a, b):
    """-1 if a precedes b, 0 if they have equal precedence, 1 if a follows b.
    ValueError if either argument is not a valid stamp string (see the module docstring).
    """
    raise NotImplementedError


def sort_versions(stamps):
    """Return the stamps sorted ascending by precedence, stable for equal precedence
    (stamps that compare equal keep their input order)."""
    raise NotImplementedError


def merge(catalog, incoming):
    """Return the union of the two valid stamp lists, sorted ascending by precedence:
    same-identity stamps inside one list raise ValueError, a release named in both is
    kept once spelling it as catalog does. See the module docstring."""
    raise NotImplementedError


def select(stamps, constraint):
    """Return the valid stamps the constraint grammar admits, ascending by precedence.
    ValueError for an invalid stamp or a constraint that is not one of the documented
    forms. See the module docstring."""
    raise NotImplementedError
