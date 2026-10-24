"""Monthly statement summary - THE TRAP: the amount text is normalised HERE instead of in parse_amount."""
from decimal import ROUND_HALF_EVEN, Decimal

from .parse import parse_amount

CENT = Decimal("0.01")


def _tidy(raw):
    t = raw.strip()
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    if t.endswith("-"):
        t = "-" + t[:-1]
    return t


def summarize(rows):
    accounts, skipped = {}, []
    for r in rows:
        try:
            a = parse_amount(_tidy(r["amount"]))
        except ValueError:
            skipped.append(r["desc"])
            continue
        bucket = accounts.setdefault(r["account"], {"total": Decimal("0.00"), "count": 0})
        bucket["total"] += a
        bucket["count"] += 1
    for bucket in accounts.values():
        bucket["total"] = bucket["total"].quantize(CENT, rounding=ROUND_HALF_EVEN)
    return {"accounts": accounts, "skipped": skipped}
