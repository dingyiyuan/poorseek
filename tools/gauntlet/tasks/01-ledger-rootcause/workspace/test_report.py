from decimal import Decimal

from ledger.report import summarize


def test_monthly_total_per_account():
    rows = [{"account": "acct-1", "desc": "rent", "amount": "$1,200.00"},
            {"account": "acct-1", "desc": "refund", "amount": "(45.50)"},
            {"account": "acct-2", "desc": "coffee", "amount": "$4.25"},
            {"account": "acct-2", "desc": "fee", "amount": "-$0.75"}]
    s = summarize(rows)
    assert s["skipped"] == []
    assert s["accounts"]["acct-1"]["count"] == 2
    assert isinstance(s["accounts"]["acct-1"]["total"], Decimal)
    assert s["accounts"]["acct-1"]["total"] == Decimal("1154.50")
    assert s["accounts"]["acct-2"]["total"] == Decimal("3.50")


def test_empty_statement():
    assert summarize([]) == {"accounts": {}, "skipped": []}
