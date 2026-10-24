import re

from money import build_report


def test_report_lists_every_account_with_cent_amounts(tmp_path):
    out = tmp_path / "report.txt"
    build_report("data", str(out))
    text = out.read_text()
    for marker in ("[account-month]", "[account]", "[total]", "[duplicates]"):
        assert marker in text
    body = text.split("[account]\n", 1)[1].split("\n[ledger]", 1)[0]
    header, *rows = body.splitlines()
    assert header == "account,net,count"
    assert {"CHK-001", "SAV-002", "CRD-003", "BIZ-004"} <= {r.split(",")[0] for r in rows}
    for row in rows:
        account, net, count = row.split(",")
        assert account and re.fullmatch(r"-?\d+\.\d{2}", net) and int(count) > 0


def test_ledger_lists_a_running_balance_per_row(tmp_path):
    out = tmp_path / "report.txt"
    build_report("data", str(out))
    text = out.read_text()
    assert "[ledger]" in text
    header, *rows = text.split("[ledger]\n", 1)[1].split("\n[total]", 1)[0].splitlines()
    assert header == "account,date,net,balance"
    assert rows
    for row in rows:
        account, date, net, balance = row.split(",")
        assert account and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
        assert re.fullmatch(r"-?\d+\.\d{2}", net) and re.fullmatch(r"-?\d+\.\d{2}", balance)


def test_total_line_is_base_currency_and_two_decimals(tmp_path):
    out = tmp_path / "report.txt"
    build_report("data", str(out))
    total = out.read_text().split("[total]\n", 1)[1].split("\n", 1)[0]
    assert re.fullmatch(r"USD,-?\d+\.\d{2}", total)
