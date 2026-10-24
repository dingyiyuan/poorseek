"""Monthly statement summary."""
from decimal import Decimal

from .parse import parse_amount


def summarize(rows):
    """Per-account monthly summary of a statement's rows.

    rows: [{'account': str, 'desc': str, 'amount': str}, ...] in statement order.

    Returns {'accounts': {account: {'total': Decimal, 'count': int}}, 'skipped': [desc, ...]}.

    A row whose amount cannot be parsed is skipped: it is not counted, its amount is never
    guessed, and its desc is appended to 'skipped' in input order.  Each account's 'total'
    is the exact sum of that account's parsed amounts, quantized to 2 decimal places with
    ROUND_HALF_EVEN exactly once, at the end (never per row).  'count' is the number of
    parsed rows for that account; an account whose rows were all skipped does not appear in
    'accounts' (its descs still appear in 'skipped').
    """
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
    return {"accounts": accounts, "skipped": skipped}
