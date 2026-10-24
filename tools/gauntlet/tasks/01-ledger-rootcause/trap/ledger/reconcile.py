"""Statement reconciliation - THE TRAP: float maths and round() instead of Decimal.quantize."""
from .parse import parse_amount


def reconcile(statements):
    matched, mismatched, skipped = [], [], []
    for s in statements:
        total, declared = 0.0, float(parse_amount(s["declared"]))
        for line in s["lines"]:
            try:
                total += float(parse_amount(line["amount"]))
            except ValueError:
                skipped.append(line["desc"])
        total, declared = round(total, 2), round(declared, 2)
        if total == declared:
            matched.append(s["account"])
        else:
            mismatched.append({"account": s["account"], "sum": total, "declared": declared,
                               "difference": round(total - declared, 2)})
    return {"matched": matched, "mismatched": mismatched, "skipped": skipped}
