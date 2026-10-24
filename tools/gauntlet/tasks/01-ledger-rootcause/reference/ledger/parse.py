"""Amount parsing for bank-statement exports."""
import re
from decimal import Decimal

_NUM = re.compile(r"^(?P<neg>-)?\$?(?P<int>\d{1,3}(?:,\d{3})*|\d+)(?:\.(?P<frac>\d+))?$")
_CODE = re.compile(r"^[A-Z]{3}$")


def parse_amount(text):
    """Parse one amount cell from a bank export into an exact Decimal (see the docstring in the workspace)."""
    t = text.strip()
    neg = False
    if t.startswith("(") or t.endswith(")"):
        if not (t.startswith("(") and t.endswith(")")):
            raise ValueError(f"unbalanced parentheses: {text!r}")
        t, neg = t[1:-1].strip(), True
    parts = t.split()
    if len(parts) == 2 and _CODE.match(parts[1]):
        t = parts[0]
    elif len(parts) != 1:
        raise ValueError(f"not an amount: {text!r}")
    if t.endswith("-"):
        if neg or t.startswith("-"):
            raise ValueError(f"two signs: {text!r}")
        t, neg = t[:-1], True
    m = _NUM.match(t)
    if not m:
        raise ValueError(f"not an amount: {text!r}")
    if m.group("neg"):
        if neg:
            raise ValueError(f"two signs: {text!r}")
        neg = True
    digits = m.group("int").replace(",", "")
    if m.group("frac"):
        digits += "." + m.group("frac")
    d = Decimal(digits)
    return -d if neg else d
