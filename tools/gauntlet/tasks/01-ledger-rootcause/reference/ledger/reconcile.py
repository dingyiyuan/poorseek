"""Statement reconciliation: each declared total against the sum of its lines."""
from decimal import ROUND_HALF_EVEN, Decimal

from .parse import parse_amount

CENT = Decimal("0.01")


def _quantize(value):
    return value.quantize(CENT, rounding=ROUND_HALF_EVEN)


def reconcile(statements):
    """Reconcile every statement's declared total against the sum of its own lines (see the workspace docstring)."""
    matched, mismatched, skipped = [], [], []
    for s in statements:
        total = Decimal("0.00")
        for line in s["lines"]:
            try:
                total += parse_amount(line["amount"])
            except ValueError:
                skipped.append(line["desc"])
        declared = _quantize(parse_amount(s["declared"]))
        total = _quantize(total)
        if total == declared:
            matched.append(s["account"])
        else:
            mismatched.append({"account": s["account"], "sum": total, "declared": declared,
                               "difference": _quantize(total - declared)})
    return {"matched": matched, "mismatched": mismatched, "skipped": skipped}
