"""THE TRAP: the obvious zoneinfo code, extended the obvious way. A skipped local time is
silently shifted instead of raising, a local day is assumed to be 24 h, a weekly occurrence is
the first UTC instant plus 7*24 real hours (so its UTC time of day never moves across a DST
change), busy intervals are not normalised - and the new surfaces are read just as naturally:
the agenda sorts by the wall clock it prints and takes the length as the wall-clock difference,
the blackout file is consulted only after the busy list has been resolved, and the notice
cut-off replaces the window start instead of being clamped to it. Passes the visible suite,
fails the hidden one."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MELBOURNE = "Australia/Melbourne"


def to_utc(local, tz, fold=0):
    return datetime.fromisoformat(local).replace(tzinfo=ZoneInfo(tz), fold=fold).astimezone(timezone.utc)


def _utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve(s, tz):
    return _utc(s) if s.endswith("Z") else to_utc(s, tz)


def has_conflict(existing, start_local, end_local, tz="Australia/Melbourne", fold=0):
    s, e = to_utc(start_local, tz, fold), to_utc(end_local, tz, fold)
    if e <= s:
        raise ValueError("end must be after start")
    return any(_utc(a) < e and s < _utc(b) for a, b in existing)


def free_minutes(existing, day_local, tz="Australia/Melbourne"):
    start = to_utc(day_local[:10] + "T00:00", tz)
    end = start + timedelta(hours=24)
    busy = sum(((min(_utc(b), end) - max(_utc(a), start)).total_seconds() for a, b in existing
                if _utc(a) < end and start < _utc(b)), 0.0)
    return int((end - start).total_seconds() - busy) // 60


def weekly(start_local, count, tz="Australia/Melbourne", fold=0):
    first = to_utc(start_local, tz, fold)                      # collapse to a real instant first
    return [(first + timedelta(weeks=k)).strftime("%Y-%m-%dT%H:%M:%SZ") for k in range(count)]


def agenda(busy, day_local, tz="Australia/Melbourne", blackout_file="blackouts.txt"):
    z = ZoneInfo(tz)
    day = day_local[:10]
    rows = []
    for a, b in busy:
        sa, sb = _resolve(a, tz), _resolve(b, tz)
        la, lb = sa.astimezone(z), sb.astimezone(z)
        if la.date().isoformat() == day:                       # the local day of the start
            rows.append((la, lb, sa))
    rows.sort(key=lambda row: row[0].replace(tzinfo=None))     # by the wall clock the line prints
    out = []
    for la, lb, sa in rows:
        out.append(f"{la.isoformat(timespec='minutes')} {_iso(sa)} "
                   f"{int((lb.replace(tzinfo=None) - la.replace(tzinfo=None)).total_seconds() // 60)} "
                   f"{lb.isoformat(timespec='minutes')}")
    return out


def find_slots(busy, day_local, minutes, count, tz="Australia/Melbourne",
               blackout_file="blackouts.txt", now=None, notice=0):
    start = to_utc(day_local[:10] + "T00:00", tz)
    end = start + timedelta(hours=24)
    spans = sorted((_resolve(a, tz), _resolve(b, tz)) for a, b in busy)
    if blackout_file:
        p = Path(blackout_file)
        if p.is_file() and day_local[:10] in [ln.split("#")[0].strip() for ln in p.read_text().splitlines()]:
            return []
    step = timedelta(minutes=minutes)
    slots, cursor = [], start
    if now is not None:
        cursor = _resolve(now, tz) + timedelta(minutes=notice)
    while len(slots) < count and cursor + step <= end:
        clash = next((b for a, b in spans if a < cursor + step and cursor < b), None)
        if clash is not None:
            cursor = clash
            continue
        slots.append(cursor)
        cursor += step
    return [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in slots]
