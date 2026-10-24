"""Deterministic generator for task 07-csv-money.

Writes workspace/data/{bank_a.csv,bank_b.csv,bank_c.tsv,rates.yaml}, the hidden
extra exports (hidden/extra_*.{csv,tsv}) and hidden/expected_report.txt.

The expected report is built from the raw rows held in memory (row tuples, no
file parsing, no import of reference/money.py) - an INDEPENDENT code path, so it
actually cross-checks the reference rather than mirroring it.
"""
import csv
import io
import os
import random
from decimal import ROUND_HALF_EVEN, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
R = random.Random(77)  # fixed seed -> deterministic data
CENT = Decimal("0.01")
BASE = "USD"
RATES = [("USD", "1.0000"), ("EUR", "1.0855"), ("GBP", "1.2700"),
         ("CHF", "1.4000"), ("JPY", "0.0067"), ("CAD", "0.7400")]
RATE = {c: Decimal(v) for c, v in RATES}
ACCTS = ["CHK-001", "SAV-002", "CRD-003", "BIZ-004"]
CURR = ["USD", "USD", "USD", "EUR", "EUR", "GBP", "CHF", "JPY", "CAD"]
MEMOS = ["Coffee", "Rent, October", 'Refund "dup"', "Wire\nref 99", "Salary", "Fee; monthly", "ATM", "Transfer"]
MONTHS = ["2026-07", "2026-08", "2026-09", "2026-10"]
HEADERS = {"A": ["date", "account", "description", "amount"],
           "B": ["Datum", "Konto", "Beschreibung", "Betrag"],
           "C": ["posted", "account", "memo", "amount"]}
DELIMS = {"A": ",", "B": ";", "C": "\t"}


def rows(n, accounts=ACCTS):
    out = []
    for _ in range(n):
        date = f"{R.choice(MONTHS)}-{R.randint(1, 28):02d}"
        cents = R.randint(-250000, 900000) if R.random() < 0.85 else R.randint(-5, 5)
        out.append((date, R.choice(accounts), R.choice(MEMOS), cents, R.choice(CURR)))
    return out


def num_a(cents):
    q, r = divmod(abs(cents), 100)
    return f"{'-' if cents < 0 else ''}{q:,}.{r:02d}"


def num_b(cents):
    q, r = divmod(abs(cents), 100)
    return f"{'-' if cents < 0 else ''}{q:,}.{r:02d}".replace(",", "X").replace(".", ",").replace("X", ".")


def num_c(cents):
    q, r = divmod(abs(cents), 100)
    return f"{q}.{r:02d}" + ("-" if cents < 0 else "")


NUM = {"A": num_a, "B": num_b, "C": num_c}


def write_export(path, dialect, data, delim=None):
    delim = delim or DELIMS[dialect]
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\r\n" if dialect == "B" else "\n")
    w.writerow(HEADERS[dialect])
    for date, acct, memo, cents, cur in data:
        w.writerow([date, acct, memo, f"{NUM[dialect](cents)} {cur}"])
    if dialect == "C":  # dialect C ends with a blank line and a TOTAL summary row
        buf.write("\n")
        w.writerow(["", "TOTAL", "", f"{num_c(sum(r[3] for r in data))} USD"])
    enc = "utf-8-sig" if dialect == "B" else "utf-8"
    with open(path, "w", newline="", encoding=enc) as f:
        f.write(buf.getvalue())


def expected_report(files):
    """files: [(name, rows)] -> the report text, computed from the raw rows only."""
    months, accounts, ledger, dups = {}, {}, {}, []
    grand = Decimal("0.00")
    for name, data in files:
        seen = set()
        for date, acct, memo, cents, cur in data:
            norm = (Decimal(cents) / 100 * RATE[cur]).quantize(CENT, rounding=ROUND_HALF_EVEN)
            key = (date, acct, norm)
            if key in seen:
                dups.append((name, date, acct, norm))
                continue
            seen.add(key)
            ledger.setdefault(acct, []).append((date, norm))
            months.setdefault((acct, date[:7]), [Decimal("0.00"), 0])
            months[(acct, date[:7])][0] += norm
            months[(acct, date[:7])][1] += 1
            accounts.setdefault(acct, [Decimal("0.00"), 0])
            accounts[acct][0] += norm
            accounts[acct][1] += 1
            grand += norm
    lines = ["[account-month]", "account,month,net,count"]
    lines += [f"{a},{mo},{m[0]:.2f},{m[1]}" for (a, mo), m in sorted(months.items())]
    lines += ["[account]", "account,net,count"]
    lines += [f"{a},{accounts[a][0]:.2f},{accounts[a][1]}" for a in sorted(accounts)]
    lines += ["[ledger]", "account,date,net,balance"]
    for a in sorted(ledger):
        bal = Decimal("0.00")
        for d, v in sorted(ledger[a], key=lambda r: r[0]):  # stable: equal dates keep input order
            bal += v
            lines.append(f"{a},{d},{v:.2f},{bal:.2f}")
    lines += ["[total]", f"{BASE},{grand:.2f}", "[duplicates]", "file,date,account,net"]
    lines += [f"{n},{d},{a},{v:.2f}" for n, d, a, v in dups]
    return "\n".join(lines) + "\n"


def main():
    bank_a, bank_b, bank_c = rows(30), rows(28), rows(26)
    extra_a, extra_b, extra_c = rows(24), rows(22), rows(20)
    bank_a.append(bank_a[7])      # a deliberate duplicate (bank_a.csv)
    extra_b.append(extra_b[3])    # a deliberate duplicate (extra_b.csv)
    mixed = [("2026-08-03", "SEMI-777", "mixed-sep", 999, "USD"),
             ("2026-08-17", "SEMI-777", "mixed-sep", -250, "EUR"),
             ("2026-09-02", "SEMI-777", "mixed-sep", 123456, "USD"),
             ("2026-09-21", "SEMI-777", "mixed-sep", 5, "CHF")]
    # FX-009: two 0.01 CHF rows (rate 1.4).  Per row: 0.01+0.01 = 0.02.  At the
    # total: quantize(0.014 + 0.014) = quantize(0.028) = 0.03 -> differ by one cent.
    rounding = [("2026-09-05", "FX-009", "fx", 1, "CHF"), ("2026-09-06", "FX-009", "fx", 1, "CHF")]
    raw = Decimal(2) / 100 * RATE["CHF"]
    per_row = (Decimal(1) / 100 * RATE["CHF"]).quantize(CENT, rounding=ROUND_HALF_EVEN) * 2
    print(f"FX-009: per-row={per_row} vs sum-then-quantize={raw.quantize(CENT, ROUND_HALF_EVEN)} "
          f"(diff {raw.quantize(CENT, ROUND_HALF_EVEN) - per_row})")

    data_dir, hidden = os.path.join(HERE, "workspace", "data"), os.path.join(HERE, "hidden")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(hidden, exist_ok=True)
    with open(os.path.join(data_dir, "rates.yaml"), "w") as f:
        f.write("# base currency and units of the base per one unit of each currency\n")
        f.write(f"base: {BASE}\n")
        for c, v in RATES:
            f.write(f"{c}: {v}\n")

    for name, dialect, data in [("bank_a.csv", "A", bank_a), ("bank_b.csv", "B", bank_b), ("bank_c.tsv", "C", bank_c)]:
        write_export(os.path.join(data_dir, name), dialect, data)
    for name, dialect, data in [("extra_a.csv", "A", extra_a), ("extra_b.csv", "B", extra_b), ("extra_c.tsv", "C", extra_c)]:
        write_export(os.path.join(hidden, name), dialect, data)
    write_export(os.path.join(hidden, "extra_mixed.csv"), "A", mixed, delim=";")  # A field names, ';' delimiter
    write_export(os.path.join(hidden, "extra_rounding.csv"), "A", rounding)

    files = [("bank_a.csv", bank_a), ("bank_b.csv", bank_b), ("bank_c.tsv", bank_c), ("extra_a.csv", extra_a),
             ("extra_b.csv", extra_b), ("extra_c.tsv", extra_c), ("extra_mixed.csv", mixed),
             ("extra_rounding.csv", rounding)]
    with open(os.path.join(hidden, "expected_report.txt"), "w", newline="\n") as f:
        f.write(expected_report(sorted(files)))


if __name__ == "__main__":
    main()
