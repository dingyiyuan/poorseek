"""Meeting scheduling across time zones and daylight-saving changes.

Two kinds of clock appear here.

  * UTC instants are ISO-8601 strings ending in "Z" to the second, e.g.
    "2026-04-04T15:30:00Z".
  * Local wall-clock times are naive ISO-8601 strings ("YYYY-MM-DDTHH:MM", the
    ":SS" optional) read in an IANA zone such as "Australia/Melbourne". A naive
    string carries no offset; the zone decides the instant.

Melbourne 2026, for reference: the clocks go BACK on Sunday 2026-04-05 (03:00
AEDT +11 -> 02:00 AEST +10, so 02:00-03:00 happens twice and that local day is
25 h long) and FORWARD on Sunday 2026-10-04 (02:00 AEST +10 -> 03:00 AEDT +11,
so 02:00-03:00 does not exist and that local day is 23 h long).

to_utc(local, tz, fold=0) -> aware UTC datetime
    * a local time that does not exist - skipped by a DST change, e.g. 02:30 on
      the morning the clocks go forward - raises ValueError; it is NEVER shifted;
    * a local time that exists twice (the clocks have gone back) is resolved by
      fold: 0 = the first occurrence (the earlier instant), 1 = the second. fold
      is ignored for an unambiguous time;
    * ValueError for a string that is not a naive local wall-clock time.

has_conflict(existing, start_local, end_local, tz="Australia/Melbourne", fold=0) -> bool
    existing is a list of (start, end) UTC "Z" strings. The requested meeting is
    two naive local wall-clock times; fold applies to both endpoints. Intervals
    are half-open [start, end): back-to-back meetings do not conflict. end must
    be later than start as REAL instants, otherwise ValueError - across a DST
    change the wall-clock duration and the real duration differ.

free_minutes(existing, day_local, tz="Australia/Melbourne") -> int
    the real minutes of a LOCAL calendar day (local midnight to the next local
    midnight - 23, 24 or 25 hours long) that no existing meeting covers.

weekly(start_local, count, tz="Australia/Melbourne", fold=0) -> list[str]
    a recurring meeting that keeps the SAME local wall-clock time once every 7
    local days (its first occurrence is start_local). Returns the next `count`
    occurrences as UTC "Z" strings, earliest first: occurrence k is at the local
    date start_local's date + 7k days, at start_local's local time. Each instant
    is that local date recomputed for its own week, NOT the first instant plus
    7*24 real hours - so the UTC time of day MOVES by the offset change across a
    DST change.
    Policy: an occurrence whose local time does not exist (it falls in the
    spring-forward gap) raises ValueError naming that date - a recurrence is
    never silently shifted either. fold resolves an occurrence that falls in a
    repeated (fall-back) hour, with the same meaning as in to_utc.

agenda(busy, day_local, tz="Australia/Melbourne", blackout_file="blackouts.txt") -> list[str]
    the LOCAL day's schedule: one line per meeting whose start instant falls in
    that local day (local midnight inclusive to the next local midnight
    exclusive), sorted by start instant and then by end instant, entries tying
    on both keeping their order in busy. busy has find_slots's shape and its
    endpoints resolve the same way (a local endpoint with fold 0, a skipped
    local time is ValueError, an end that is not after its start is ValueError).
    Each meeting appears once, whole and unclipped - it may run past the end of
    the local day - and a meeting that began before local midnight does not
    appear. Every entry in busy is resolved and checked, including entries that
    fall on another day - the day only filters, it never excuses. A meeting
    whose real length is not a whole number of minutes is ValueError.
    A line is four space-separated fields in this order:
        <local start> <UTC start> <real minutes> <local end>
    A local field is that instant in tz as "YYYY-MM-DDTHH:MM+11:00" - the offset
    is the one in force AT THE INSTANT, which is what tells the two repetitions
    of a wall-clock time apart: on 2026-04-05 the two 02:30s render as
    2026-04-05T02:30+11:00 and 2026-04-05T02:30+10:00. <UTC start> is that same
    instant as "YYYY-MM-DDTHH:MM:SSZ", and <real minutes> is the true length
    ((end - start) over the INSTANTS), so 01:30 -> 03:30 on 2026-10-04 is 60
    minutes, not 120. The result is a list of lines with no trailing newline and
    no blank lines, so a day with no meeting is [] and never [""];
    "\\n".join(...) is the day's agenda text.
    On a date listed in the blackout file (see find_slots) the desk is closed:
    [] is returned at once, before busy is looked at, so a busy entry that would
    otherwise be rejected does not raise on a closed day.

find_slots(busy, day_local, minutes, count, tz="Australia/Melbourne",
           blackout_file="blackouts.txt", now=None, notice=0) -> list[str]
    Working hours: this desk is staffed without a break, so the searchable
    window is the whole LOCAL calendar day (local midnight to the next local
    midnight - 23, 24 or 25 real hours). busy is a list of (start, end)
    meetings; each endpoint is EITHER a UTC "Z" string OR a naive local
    wall-clock string, and the two forms are freely mixed - a local endpoint is
    turned into an instant exactly as to_utc does (fold 0 for a repeated time,
    ValueError for a skipped one). Intervals are half-open [start, end) and are
    clipped to the window. Return the earliest `count` start instants, as UTC
    "Z" strings, at which a meeting of `minutes` REAL minutes fits entirely
    inside a free stretch of the window without overlapping busy time (the
    meeting may end exactly when a busy meeting starts).
    Ordering and tie-break: packing is greedy and tight - a slot starts at the
    earliest legal instant, so it may begin exactly where the previous meeting
    ended and, after a busy interval, exactly when that interval ends; no earlier
    legal start is ever passed over for a later, better-aligned one. If fewer
    than `count` slots fit, only the ones that fit are returned, so an empty list
    means none fits. ValueError for minutes <= 0 and for a busy entry that ends
    before it starts.
    Blackout file: `blackout_file` names a small UTF-8 text file of LOCAL
    calendar dates, one "YYYY-MM-DD" per line; a "#" starts a comment running to
    the end of the line, blank and comment-only lines are ignored, and any other
    line is ValueError whose message contains that line. A listed date - the same
    date that day_local spells - closes the desk for that whole local day (23, 24
    or 25 h alike) and
    find_slots returns []. The precedence is documented and tight: minutes <= 0
    and notice < 0 are still ValueError first, but on a closed day nothing else
    is examined - not the notice rule, not the busy list - so the answer is []
    even when a busy entry would otherwise raise (a skipped local time, an end
    not after its start) and even when `now` is malformed. A blackout that falls
    on a daylight-saving change day is no different: it closes the whole local
    day and the gap policy never comes into it. blackout_file=None turns
    blackouts off, and a file that is missing is treated as empty.
    Minimum notice: `now` is an instant - a UTC "Z" string or a naive local
    wall-clock string in tz, resolved exactly like a busy endpoint (fold 0) - or
    None (the default), which turns the rule off. `notice` is a whole number of
    REAL minutes. When now is not None a slot is legal only if it STARTS at or
    after now's instant plus notice real minutes; an earlier slot is not
    returned and is NOT shifted later, the first candidate simply becomes that
    cut-off whenever the cut-off is the earliest legal instant, and packing
    stays greedy and tight from there. A cut-off at or after the end of the
    window returns []. notice counts real minutes, so a cut-off taken across a
    daylight-saving change can land anywhere in the local day - inside the
    repeated hour (the slot then starts at that instant, whatever wall clock it
    spells) or inside the hour the clocks skip.
    ValueError for minutes <= 0, for notice < 0 and for a busy entry that ends
    before it starts.
"""

def to_utc(local, tz, fold=0):
    raise NotImplementedError


def has_conflict(existing, start_local, end_local, tz="Australia/Melbourne", fold=0):
    raise NotImplementedError


def free_minutes(existing, day_local, tz="Australia/Melbourne"):
    raise NotImplementedError


def weekly(start_local, count, tz="Australia/Melbourne", fold=0):
    raise NotImplementedError


def agenda(busy, day_local, tz="Australia/Melbourne", blackout_file="blackouts.txt"):
    raise NotImplementedError


def find_slots(busy, day_local, minutes, count, tz="Australia/Melbourne",
               blackout_file="blackouts.txt", now=None, notice=0):
    raise NotImplementedError
