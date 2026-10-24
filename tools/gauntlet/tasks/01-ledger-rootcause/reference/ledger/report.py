"""Monthly statement summary."""
from decimal import ROUND_HALF_EVEN, Decimal

from .parse import parse_amount

CENT = Decimal("0.01")


def summarize(rows):
    """Per-account monthly summary (see the docstring in the workspace)."""
    accounts, skipped = {}, []
    for r in rows:
        try:
            a = parse_amount(r["amount"])
        except ValueError:
            skipped.append(r["desc"])
            continue
        bucket = accounts.setdefault(r["account"], {"total": Decimal("0.00"), "count": 0})
        bucket["total"] += a
        bucket["count"] += 1
    for bucket in accounts.values():
        bucket["total"] = bucket["total"].quantize(CENT, rounding=ROUND_HALF_EVEN)
    return {"accounts": accounts, "skipped": skipped}
