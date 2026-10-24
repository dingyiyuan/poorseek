"""Consolidate bank exports into one money report - THE TRAP.

Everything is right except the rounding rule and the [ledger] balance: the
normalised amount is NOT quantized per row (the raw products are summed and only
each group's / account's / the grand total is quantized at the end), and the
running balance is accumulated as rows are read - from the un-quantized amounts
and advanced by dropped duplicates too.  The visible test (structure only) passes;
the hidden per-row rounding, the exact report and the balance checks fail.
"""
import csv
import os
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation

CENT = Decimal("0.01")
DELIMS = (",", ";", "\t")
DIALECTS = [("A", ("date", "account", "description", "amount")),
            ("B", ("datum", "konto", "beschreibung", "betrag")),
            ("C", ("posted", "account", "memo", "amount"))]


def _rates(path):
    base, rates = None, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key == "base":
                base = val
            else:
                rates[key] = Decimal(val)
    return base, rates


def _detect(first_line):
    line = first_line.lstrip("\ufeff")
    for delim in DELIMS:
        cells = next(csv.reader([line], delimiter=delim))
        names = tuple(c.strip().lower() for c in cells)
        for dialect, fields in DIALECTS:
            if names == fields:
                return dialect, delim
    raise ValueError(f"unknown export header: {first_line.strip()!r}")


def _number(dialect, s):
    s = s.strip()
    if dialect == "A":
        neg, s = s.startswith("-"), s[1:] if s.startswith("-") else s
        s = s.replace(",", "")
    elif dialect == "B":
        neg, s = s.startswith("-"), s[1:] if s.startswith("-") else s
        s = s.replace(".", "").replace(",", ".")
    else:
        neg, s = s.endswith("-"), s[:-1] if s.endswith("-") else s
    try:
        d = Decimal(s)
    except InvalidOperation:
        raise ValueError(f"bad amount {s!r}") from None
    return -d if neg else d


def _cell(dialect, cell):
    num, _, code = cell.strip().rpartition(" ")
    if len(code) != 3 or not code.isalpha() or not code.isupper():
        raise ValueError(f"bad currency code: {cell!r}")
    return _number(dialect, num), code


def _money(value):
    return f"{value.quantize(CENT, rounding=ROUND_HALF_EVEN):.2f}"


def build_report(data_dir, out_path):
    base, rates = _rates(os.path.join(data_dir, "rates.yaml"))
    files = sorted(f for f in os.listdir(data_dir) if f != "rates.yaml"
                   and os.path.isfile(os.path.join(data_dir, f)))
    months, accounts, ledger, dups = {}, {}, {}, []
    grand = Decimal("0.00")
    for name in files:
        with open(os.path.join(data_dir, name), newline="", encoding="utf-8-sig") as f:
            dialect, delim = _detect(f.readline())
            seen = set()
            for row in csv.reader(f, delimiter=delim):
                if not row or all(not c.strip() for c in row):
                    continue
                date, acct = row[0].strip(), row[1].strip()
                if dialect == "C" and acct == "TOTAL":
                    continue
                amount, code = _cell(dialect, row[3])
                if code not in rates:
                    raise ValueError(f"unknown currency {code!r}")
                norm = amount * rates[code]  # TRAP: no per-row quantization
                key = (date, acct, norm)
                # TRAP: every row read goes on the ledger, dropped duplicates included
                ledger.setdefault(acct, []).append((date, norm))
                if key in seen:
                    dups.append((name, date, acct, norm))
                    continue
                seen.add(key)
                month = date[:7]
                m = months.setdefault((acct, month), [Decimal("0.00"), 0])
                m[0], m[1] = m[0] + norm, m[1] + 1
                a = accounts.setdefault(acct, [Decimal("0.00"), 0])
                a[0], a[1] = a[0] + norm, a[1] + 1
                grand += norm
    lines = ["[account-month]", "account,month,net,count"]
    lines += [f"{acct},{month},{_money(months[(acct, month)][0])},{months[(acct, month)][1]}"
              for acct, month in sorted(months)]
    lines += ["[account]", "account,net,count"]
    lines += [f"{acct},{_money(accounts[acct][0])},{accounts[acct][1]}" for acct in sorted(accounts)]
    lines += ["[ledger]", "account,date,net,balance"]
    for acct in sorted(ledger):
        bal = Decimal("0.00")
        for date, norm in sorted(ledger[acct], key=lambda r: r[0]):
            bal += norm
            lines.append(f"{acct},{date},{_money(norm)},{_money(bal)}")
    lines += ["[total]", f"{base},{_money(grand)}", "[duplicates]", "file,date,account,net"]
    lines += [f"{name},{date},{acct},{_money(norm)}" for name, date, acct, norm in dups]
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
