"""Hidden grading for 09-dst-scheduler.

Every case here depends on an installed IANA time-zone database that carries the
2026 rules (the `tzdata` pip package or a system /usr/share/zoneinfo on
`zoneinfo.TZPATH`). The session fixture below asserts that BEFORE any case runs
and fails all of them with one diagnostic that names the requirement and the
database that answered, instead of letting 30 cases fail for an environmental
reason nobody can read; on a healthy run it records the same line in the
pytest warnings summary.
"""
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scheduler import agenda, find_slots, free_minutes, has_conflict, to_utc, weekly

MEL = "Australia/Melbourne"
NY = "America/New_York"
REQUIRED_ZONES = (MEL, NY)
# 2026-04-05: clocks go back 03:00 AEDT -> 02:00 AEST (02:00-03:00 twice; 25 h)
# 2026-10-04: clocks go forward 02:00 AEST -> 03:00 AEDT (02:00-03:00 skipped; 23 h)
# 2026-11-01: New York goes back 02:00 EDT -> 01:00 EST (01:00-02:00 twice; 25 h)
REQUIRED_TRANSITIONS = (
    (MEL, "2026-04-04T15:30:00Z", "2026-04-05T02:30:00+11:00"),
    (MEL, "2026-04-04T16:30:00Z", "2026-04-05T02:30:00+10:00"),
    (MEL, "2026-10-03T15:30:00Z", "2026-10-04T01:30:00+10:00"),
    (MEL, "2026-10-03T16:30:00Z", "2026-10-04T03:30:00+11:00"),
    (NY, "2026-11-01T05:30:00Z", "2026-11-01T01:30:00-04:00"),
    (NY, "2026-11-01T06:30:00Z", "2026-11-01T01:30:00-05:00"),
)
LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d[+-]\d\d:\d\d"
                  r" \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \d+"
                  r" \d{4}-\d\d-\d\dT\d\d:\d\d[+-]\d\d:\d\d$")
SECOND_230 = [("2026-04-04T16:30:00Z", "2026-04-04T16:50:00Z")]     # 02:30-02:50 local, the SECOND time round


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _tzdb_record():
    """which database this run resolves the zones from, plus the probes that verified it"""
    import importlib.metadata as md
    from zoneinfo import TZPATH
    what = []
    try:
        what.append(f"tzdata pip package {md.version('tzdata')}")
    except Exception:
        what.append("no tzdata pip package")
    for base in TZPATH or ():
        f = Path(base) / "Australia" / "Melbourne"
        if f.is_file():
            st = f.stat()
            what.append(f"TZPATH {base}/Australia/Melbourne ({st.st_size} B, mtime {int(st.st_mtime)})")
            break
    else:
        what.append(f"Australia/Melbourne is not on TZPATH {TZPATH}")
    probes, problem = [], None
    from zoneinfo import ZoneInfo
    for zone, instant, expected in REQUIRED_TRANSITIONS:
        try:
            got = _t(instant).astimezone(ZoneInfo(zone)).isoformat()
        except Exception as e:                       # ZoneInfoNotFoundError, OSError, ...
            problem = f"{zone} did not resolve at all: {type(e).__name__}: {e}"
            break
        probes.append(f"{zone} {instant}->{got}")
        if got != expected and problem is None:
            problem = f"{zone} {instant} resolved to {got}, not {expected}"
    return "; ".join(what), probes, problem


@pytest.fixture(scope="session", autouse=True)
def _tzdb_required():
    record, probes, problem = _tzdb_record()
    if problem:
        pytest.fail("this task needs an installed IANA time-zone database carrying the 2026 DST "
                    "rules (the 'tzdata' pip package or a system /usr/share/zoneinfo; the zoneinfo "
                    f"module reads TZPATH) - it is missing or stale here: {problem}. [{record}]")
    warnings.warn(f"tzdb ok: {record}; transitions verified: {'; '.join(probes)}", stacklevel=1)


# ---- to_utc: fold and skipped times ------------------------------------------

def test_ambiguous_fold():
    assert to_utc("2026-04-05T02:30", MEL, 0).isoformat() == "2026-04-04T15:30:00+00:00"
    assert to_utc("2026-04-05T02:30", MEL, 1).isoformat() == "2026-04-04T16:30:00+00:00"
    assert not has_conflict(SECOND_230, "2026-04-05T02:30", "2026-04-05T02:45", MEL, fold=0)
    assert has_conflict(SECOND_230, "2026-04-05T02:30", "2026-04-05T02:45", MEL, fold=1)


@pytest.mark.parametrize("t", ["2026-10-04T02:00", "2026-10-04T02:30", "2026-10-04T02:59"])
def test_nonexistent(t):
    with pytest.raises(ValueError):
        to_utc(t, MEL)
    with pytest.raises(ValueError):
        has_conflict([], t, "2026-10-04T04:00", MEL)


def test_across_gap_real_duration():
    # 01:30 -> 03:30 on the spring-forward day is ONE real hour
    meet = [("2026-10-03T15:00:00Z", "2026-10-03T15:20:00Z")]           # 01:00-01:20 AEST
    assert not has_conflict(meet, "2026-10-04T01:30", "2026-10-04T03:30", MEL)
    assert has_conflict([("2026-10-03T16:15:00Z", "2026-10-03T16:20:00Z")], "2026-10-04T01:30", "2026-10-04T03:30", MEL)


def test_end_before_start_after_fallback():
    # 02:40 (first, AEDT) -> 02:10 (fold applies to both: first occurrence) is backwards -> ValueError
    with pytest.raises(ValueError):
        has_conflict([], "2026-04-05T02:40", "2026-04-05T02:10", MEL, fold=0)


def test_day_lengths():
    assert free_minutes([], "2026-04-05", MEL) == 25 * 60
    assert free_minutes([], "2026-10-04", MEL) == 23 * 60
    assert free_minutes([], "2026-06-10", MEL) == 24 * 60
    busy = [("2026-04-04T14:00:00Z", "2026-04-04T17:00:00Z"), ("2026-04-04T16:00:00Z", "2026-04-04T18:00:00Z")]
    assert free_minutes(busy, "2026-04-05", MEL) == 25 * 60 - 240       # overlapping meetings counted once


def test_other_zone():
    with pytest.raises(ValueError):
        to_utc("2026-03-08T02:30", "America/New_York")
    assert to_utc("2026-11-01T01:30", "America/New_York", 1).isoformat() == "2026-11-01T06:30:00+00:00"


# ---- weekly: same LOCAL wall-clock time, so the UTC instant moves with the offset ----

def test_recurrence_shifts_across_fallback():
    # a Wednesday 09:00 series: the 04-05 fall-back moves the UTC instant from 22:00Z to 23:00Z
    assert weekly("2026-04-01T09:00", 3) == ["2026-03-31T22:00:00Z", "2026-04-07T23:00:00Z", "2026-04-14T23:00:00Z"]
    a, b, c = (_t(s) for s in weekly("2026-04-01T09:00", 3))
    assert (b - a) == timedelta(hours=169)                 # the fall-back week is one real hour longer
    assert (c - b) == timedelta(hours=168)


def test_recurrence_shifts_across_spring_forward():
    assert weekly("2026-09-27T01:30", 3) == ["2026-09-26T15:30:00Z", "2026-10-03T15:30:00Z", "2026-10-10T14:30:00Z"]
    a, b, c = (_t(s) for s in weekly("2026-09-27T01:30", 3))
    assert (b - a) == timedelta(hours=168)
    assert (c - b) == timedelta(hours=167)                 # the 10-04 gap shortens a week


def test_recurrence_over_skipped_hour_raises():
    # the 2026-10-04 occurrence of a Sunday 02:30 series does not exist -> policy: raise, never shift
    with pytest.raises(ValueError):
        weekly("2026-09-27T02:30", 2)
    assert weekly("2026-10-04T03:30", 1) == ["2026-10-03T16:30:00Z"]     # exists, just after the gap


# ---- agenda: the day serialized line by line ---------------------------------

def test_tzdb_environment():
    # R3, stated as its own case: the zones resolve AND the database carries the 2026 rules
    record, probes, problem = _tzdb_record()
    assert problem is None, f"tzdb problem: {problem} [{record}]"
    assert len(probes) == len(REQUIRED_TRANSITIONS)


def test_agenda_plain_day():
    busy = [("2026-06-09T23:30:00Z", "2026-06-10T00:15:00Z"),        # 09:30-10:15 local
            ("2026-06-10T02:00:00Z", "2026-06-10T03:00:00Z")]        # 12:00-13:00 local
    assert agenda(busy, "2026-06-10") == [
        "2026-06-10T09:30+10:00 2026-06-09T23:30:00Z 45 2026-06-10T10:15+10:00",
        "2026-06-10T12:00+10:00 2026-06-10T02:00:00Z 60 2026-06-10T13:00+10:00",
    ]
    assert agenda(busy, "2026-06-10", MEL) == agenda(busy, "2026-06-10")


def test_agenda_repeated_hour_offsets():
    # the SAME wall clock 02:30 twice: the offset in force at each instant tells them apart
    busy = [("2026-04-05T02:30", "2026-04-05T03:00"),                # first 02:30, +11:00
            ("2026-04-04T16:30:00Z", "2026-04-04T16:45:00Z")]        # second 02:30, +10:00
    assert agenda(busy, "2026-04-05") == [
        "2026-04-05T02:30+11:00 2026-04-04T15:30:00Z 90 2026-04-05T03:00+10:00",
        "2026-04-05T02:30+10:00 2026-04-04T16:30:00Z 15 2026-04-05T02:45+10:00",
    ]


def test_agenda_orders_by_instant_not_wall_clock():
    # in the repeated hour the local wall clock runs backwards: 02:50 (first) < 02:30 (second)
    busy = [("2026-04-04T16:30:00Z", "2026-04-04T16:35:00Z"),        # 02:30 local, second time round
            ("2026-04-05T02:50", "2026-04-05T03:00")]                # 02:50 local, first time round
    assert agenda(busy, "2026-04-05") == [
        "2026-04-05T02:50+11:00 2026-04-04T15:50:00Z 70 2026-04-05T03:00+10:00",
        "2026-04-05T02:30+10:00 2026-04-04T16:30:00Z 5 2026-04-05T02:35+10:00",
    ]


def test_agenda_real_minutes_across_spring_forward():
    # 01:30 -> 03:30 on the gap day is 60 real minutes, and the ends carry +10:00 and +11:00
    assert agenda([("2026-10-04T01:30", "2026-10-04T03:30")], "2026-10-04") == [
        "2026-10-04T01:30+10:00 2026-10-03T15:30:00Z 60 2026-10-04T03:30+11:00"]


def test_agenda_real_minutes_across_fall_back():
    # 01:30 -> 03:10 on the fall-back day is 160 real minutes, not the 100 the wall clock shows
    assert agenda([("2026-04-05T01:30", "2026-04-05T03:10")], "2026-04-05") == [
        "2026-04-05T01:30+11:00 2026-04-04T14:30:00Z 160 2026-04-05T03:10+10:00"]


def test_agenda_keeps_only_meetings_starting_on_the_local_day():
    busy = [("2026-06-09T14:00:00Z", "2026-06-09T14:30:00Z"),        # 06-10 00:00 local: inside (inclusive)
            ("2026-06-10T13:59:00Z", "2026-06-10T14:30:00Z"),        # 06-10 23:59 local: runs past the day, kept whole
            ("2026-06-10T14:00:00Z", "2026-06-10T14:30:00Z"),        # 06-11 00:00 local: outside (exclusive)
            ("2026-06-09T13:59:00Z", "2026-06-09T14:30:00Z")]        # 06-09 23:59 local: began before midnight
    assert agenda(busy, "2026-06-10") == [
        "2026-06-10T00:00+10:00 2026-06-09T14:00:00Z 30 2026-06-10T00:30+10:00",
        "2026-06-10T23:59+10:00 2026-06-10T13:59:00Z 31 2026-06-11T00:30+10:00",
    ]


def test_agenda_other_zone_fall_back():
    busy = [("2026-11-01T01:30", "2026-11-01T02:00"),                # 01:30 EDT (-04), the first pass
            ("2026-11-01T06:30:00Z", "2026-11-01T06:45:00Z")]        # 01:30 EST (-05), the second pass
    assert agenda(busy, "2026-11-01", NY) == [
        "2026-11-01T01:30-04:00 2026-11-01T05:30:00Z 90 2026-11-01T02:00-05:00",
        "2026-11-01T01:30-05:00 2026-11-01T06:30:00Z 15 2026-11-01T01:45-05:00",
    ]


def test_agenda_line_shape_and_empty_day():
    assert agenda([], "2026-06-10") == []                          # never [""]
    assert agenda([], "2026-10-04") == []
    lines = agenda([("2026-06-10T02:00:00Z", "2026-06-10T03:00:00Z"),
                    ("2026-06-09T23:30:00Z", "2026-06-10T00:15:00Z")], "2026-06-10")
    assert len(lines) == 2
    assert all(LINE.match(line) for line in lines)
    text = "\n".join(lines)                       # the agenda text: no trailing newline, no blank line
    assert not text.endswith("\n") and "\n\n" not in text
    assert text.splitlines() == lines


def test_agenda_closed_day_is_empty():
    busy = [("2026-06-11T00:30:00Z", "2026-06-11T01:30:00Z")]      # 10:30-11:30 local on 06-11
    assert agenda(busy, "2026-06-11") == []                        # the shipped blackouts.txt lists 06-11
    assert agenda(busy, "2026-06-11", blackout_file=None) == [     # the date is what emptied it
        "2026-06-11T10:30+10:00 2026-06-11T00:30:00Z 60 2026-06-11T11:30+10:00"]


def test_agenda_rejects_bad_input():
    with pytest.raises(ValueError):                                # a local start skipped by the gap
        agenda([("2026-10-04T02:30", "2026-10-04T03:00")], "2026-10-04")
    with pytest.raises(ValueError):                                # 90 seconds is not whole minutes
        agenda([("2026-06-10T02:00:00Z", "2026-06-10T02:01:30Z")], "2026-06-10")
    with pytest.raises(ValueError):                                # end before start
        agenda([("2026-06-10T03:00:00Z", "2026-06-10T02:00:00Z")], "2026-06-10")
    with pytest.raises(ValueError):                                # another day does not excuse it
        agenda([("2026-10-04T02:30", "2026-10-04T03:00")], "2026-06-10")


# ---- the blackout file -------------------------------------------------------

def test_blackout_closes_the_whole_local_day(tmp_path):
    f = tmp_path / "closed.txt"
    f.write_text("2026-06-10\n")
    assert find_slots([], "2026-06-10", 60, 3, blackout_file=f) == []
    assert agenda([("2026-06-10T02:00:00Z", "2026-06-10T03:00:00Z")], "2026-06-10", blackout_file=f) == []
    assert find_slots([], "2026-06-09", 60, 1, blackout_file=f) == ["2026-06-08T14:00:00Z"]   # neighbour open
    assert find_slots([], "2026-06-11", 60, 1, blackout_file=f) == ["2026-06-10T14:00:00Z"]


def test_blackout_precedes_busy_and_notice(tmp_path):
    # a holiday on the day the clocks go forward: the shut desk never reaches the gap policy
    f = tmp_path / "gapday.txt"
    f.write_text("2026-10-04\n")
    skipped = [("2026-10-04T02:30", "2026-10-04T02:45")]           # would be ValueError on an open day
    backwards = [("2026-10-04T03:00:00Z", "2026-10-04T02:00:00Z")]  # ditto
    assert find_slots(skipped, "2026-10-04", 30, 1, blackout_file=f) == []
    assert find_slots(backwards, "2026-10-04", 30, 1, blackout_file=f) == []
    assert find_slots([], "2026-10-04", 30, 1, blackout_file=f, now="not-a-time") == []
    assert agenda(skipped, "2026-10-04", blackout_file=f) == []
    assert find_slots([], "2026-10-03", 60, 1, blackout_file=f) == ["2026-10-02T14:00:00Z"]
    with pytest.raises(ValueError):                                # the argument checks still come first
        find_slots([], "2026-10-04", 0, 1, blackout_file=f)
    with pytest.raises(ValueError):
        find_slots([], "2026-10-04", 30, 1, blackout_file=f, notice=-1)


def test_blackout_file_parsing(tmp_path):
    f = tmp_path / "mixed.txt"
    f.write_text("# desk closures\n\n  2026-06-11  \n2026-06-12 # a comment after the date\n2026-01-01\n")
    assert find_slots([], "2026-06-11", 60, 1, blackout_file=f) == []
    assert find_slots([], "2026-06-12", 60, 1, blackout_file=f) == []
    bad = tmp_path / "bad.txt"
    bad.write_text("2026-6-11\n")
    with pytest.raises(ValueError) as err:
        find_slots([], "2026-06-10", 60, 1, blackout_file=bad)
    assert "2026-6-11" in str(err.value)                           # the offending line is named
    assert find_slots([], "2026-06-11", 60, 1, blackout_file=None) == ["2026-06-10T14:00:00Z"]
    assert find_slots([], "2026-06-11", 60, 1, blackout_file=tmp_path / "absent.txt") == ["2026-06-10T14:00:00Z"]


def test_default_blackout_file():
    # the workspace's own blackouts.txt is the default the desk actually runs on
    assert find_slots([], "2026-06-11", 60, 1) == []
    assert agenda([], "2026-06-11") == []
    assert find_slots([], "2026-01-01", 60, 1) == []
    assert find_slots([], "2026-12-25", 60, 1) == []
    assert find_slots([], "2026-06-12", 60, 1) == ["2026-06-11T14:00:00Z"]   # not listed, desk open


# ---- find_slots: the minimum notice -----------------------------------------

def test_notice_cutoff_moves_the_first_slot():
    # now = local midnight, notice = 90 real minutes -> the first slot starts exactly at the cut-off
    slots = find_slots([], "2026-06-10", 60, 40, now="2026-06-09T14:00:00Z", notice=90)
    assert slots[0] == "2026-06-09T15:30:00Z"                      # 01:30+10:00 local
    assert slots[1] == "2026-06-09T16:30:00Z"                      # tight packing from the cut-off
    assert len(slots) == 22                                        # a 24 h day minus the 1.5 h cut-off
    assert all(s >= "2026-06-09T15:30:00Z" for s in slots)


def test_notice_cutoff_inside_the_skipped_hour():
    # 2026-10-04 is 23 h long: the cut-off is 90 real minutes after local midnight (01:30 +10:00),
    # and the search runs over instants, so it steps straight through the hour the clocks skip
    slots = find_slots([], "2026-10-04", 60, 30, now="2026-10-03T14:00:00Z", notice=90)
    assert slots[0] == "2026-10-03T15:30:00Z"                      # 01:30 AEST
    assert slots[1] == "2026-10-03T16:30:00Z"                      # 03:30 AEDT, an hour later in real time
    assert slots[-1] == "2026-10-04T11:30:00Z"
    assert len(slots) == 21                                        # 23 h minus 1.5 h

def test_notice_cutoff_inside_the_repeated_hour():
    # 2026-04-05 is 25 h long and 02:30 happens twice; now is the FIRST 02:30 (+11:00) and the
    # notice is 60 real minutes, so the earliest legal slot is the SECOND 02:30 (16:30Z)
    slots = find_slots([], "2026-04-05", 60, 30, now="2026-04-05T02:30", notice=60)
    assert slots[0] == "2026-04-04T16:30:00Z"
    assert slots[-1] == "2026-04-05T12:30:00Z"
    assert len(slots) == 21                                        # 25 h minus the cut-off at 16:30Z


def test_notice_boundaries():
    # a cut-off before the window: the window start is still the first legal instant
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-01T00:00:00Z", notice=5) == ["2026-06-09T14:00:00Z"]
    # a meeting may end exactly at the next local midnight, but not later
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-10T13:00:00Z") == ["2026-06-10T13:00:00Z"]
    # a cut-off past the window, and one too late for a whole meeting: nothing is returned
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-11T00:00:00Z") == []
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-10T13:30:00Z") == []
    # now may be a naive local wall clock, and notice 0 is a plain "not before now"
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-10T12:00", notice=0) == ["2026-06-10T02:00:00Z"]
    assert find_slots([], "2026-06-10", 60, 1, now="2026-06-09T14:00:00Z") == ["2026-06-09T14:00:00Z"]
    # notice is ignored while now is None
    assert find_slots([], "2026-06-10", 60, 1, notice=9999) == ["2026-06-09T14:00:00Z"]


def test_notice_cutoff_inside_busy_resumes_at_the_busy_end():
    busy = [("2026-06-10T00:00:00Z", "2026-06-10T01:00:00Z")]      # 10:00-11:00 local
    slots = find_slots(busy, "2026-06-10", 30, 3, now="2026-06-09T14:00:00Z", notice=630)   # cut-off 00:30Z
    assert slots == ["2026-06-10T01:00:00Z", "2026-06-10T01:30:00Z", "2026-06-10T02:00:00Z"]


def test_notice_rejects_bad_arguments():
    with pytest.raises(ValueError):                                # negative notice, even with now=None
        find_slots([], "2026-06-10", 60, 1, notice=-1)
    with pytest.raises(ValueError):                                # now must be Z or a NAIVE local time
        find_slots([], "2026-06-10", 60, 1, now="2026-06-10T12:00+11:00")
    with pytest.raises(ValueError):
        find_slots([], "2026-06-10", 60, 1, now="2026-06-10T25:00")


# ---- find_slots --------------------------------------------------------------

def test_slots_day_length():
    assert len(find_slots([], "2026-10-04", 60, 30)) == 23        # a 23-hour day holds 23 whole hours
    assert len(find_slots([], "2026-04-05", 60, 30)) == 25
    assert len(find_slots([], "2026-06-10", 60, 30)) == 24
    assert find_slots([], "2026-10-04", 1400, 1) == []            # 23 h < 1400 min: nothing fits
    assert find_slots([], "2026-04-05", 1400, 1) == ["2026-04-04T13:00:00Z"]


def test_slots_normalise_mixed_busy_and_the_gap():
    # the two endpoints name the SAME meeting (UTC and local forms); on the gap day it covers
    # 01:30-03:30 local, which is ONE real hour, so the next slot resumes at 16:30Z.
    busy = [("2026-10-03T15:30:00Z", "2026-10-03T16:30:00Z"), ("2026-10-04T01:30", "2026-10-04T03:30")]
    assert find_slots(busy, "2026-10-04", 30, 4) == ["2026-10-03T14:00:00Z", "2026-10-03T14:30:00Z",
                                                      "2026-10-03T15:00:00Z", "2026-10-03T16:30:00Z"]


def test_slots_pack_tightly_and_resume_at_the_busy_end():
    busy = [("2026-06-10T02:00:00Z", "2026-06-10T03:00:00Z")]          # 12:00-13:00 local
    assert find_slots(busy, "2026-06-10", 90, 10) == [
        "2026-06-09T14:00:00Z", "2026-06-09T15:30:00Z", "2026-06-09T17:00:00Z", "2026-06-09T18:30:00Z",
        "2026-06-09T20:00:00Z", "2026-06-09T21:30:00Z", "2026-06-09T23:00:00Z", "2026-06-10T00:30:00Z",
        "2026-06-10T03:00:00Z", "2026-06-10T04:30:00Z"]


def test_slots_ignore_outside_and_clip_busy():
    outside = [("2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z")]       # another day entirely
    assert find_slots(outside, "2026-06-10", 60, 1) == ["2026-06-09T14:00:00Z"]
    # a busy meeting that straddles local midnight is clipped to the window
    midnight = [("2026-06-09T23:00", "2026-06-10T01:00")]
    assert find_slots(midnight, "2026-06-10", 60, 1) == ["2026-06-09T15:00:00Z"]


def test_slots_reject_bad_input():
    with pytest.raises(ValueError):
        find_slots([("2026-10-04T02:30", "2026-10-04T02:45")], "2026-10-04", 30, 1)   # skipped local start
    with pytest.raises(ValueError):
        find_slots([], "2026-06-10", 0, 1)
    with pytest.raises(ValueError):
        find_slots([("2026-06-10T03:00:00Z", "2026-06-10T02:00:00Z")], "2026-06-10", 30, 1)
