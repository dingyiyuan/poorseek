"""Statement reconciliation: each declared total against the sum of its lines."""
from decimal import Decimal

from .parse import parse_amount


def reconcile(statements):
    """Reconcile every statement's declared total against the sum of its own lines.

    statements: [{'account': str, 'declared': str, 'lines': [{'desc': str, 'amount': str}, ...]}, ...]
    in input order.  'declared' is always a valid amount cell.

    Returns {'matched': [account, ...],
             'mismatched': [{'account', 'sum', 'declared', 'difference'}, ...],
             'skipped': [desc, ...]}.

    A line whose amount cannot be parsed is excluded from its statement's sum and its desc
    is appended to 'skipped' in input order, across all the statements.  A statement's 'sum'
    is the exact sum of its parsed lines quantized to 2 decimal places with ROUND_HALF_EVEN,
    and its declared total is quantized the same way.  The account is 'matched' iff those two
    quantized figures are equal - a declared total that matches only after quantization counts
    as matched - and 'matched' and 'mismatched' keep the input order of the statements.
    A mismatched entry carries the quantized 'sum' and 'declared' and 'difference', the
    quantized sum minus the quantized declared, itself quantized to 2 decimal places.
    """
    matched, mismatched = [], []
    skipped = set()
    for s in statements:
        total = Decimal("0.00")
        for line in s["lines"]:
            try:
                total += parse_amount(line["amount"])
            except ValueError:
                skipped.add(line["desc"])
        declared = parse_amount(s["declared"])
        if total == declared:
            matched.append(s["account"])
        else:
            mismatched.append({"account": s["account"], "sum": total, "declared": declared,
                               "difference": total - declared})
    return {"matched": matched, "mismatched": mismatched, "skipped": sorted(skipped)}
