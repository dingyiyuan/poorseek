"""Hidden grading: the documented contract of ledger.parse, ledger.report and ledger.reconcile."""
from decimal import Decimal

import pytest

from ledger.parse import parse_amount
from ledger.reconcile import reconcile
from ledger.report import summarize


@pytest.mark.parametrize("text,want", [("$1,234.50", "1234.50"), ("(12.50)", "-12.50"), ("($1,000.00)", "-1000.00"),
                                       ("-$3.10", "-3.10"), ("3.10-", "-3.10"), ("1234.50 USD", "1234.50"),
                                       ("1234.50 EUR", "1234.50"), (" 0.10 ", "0.10"), ("$1,000,000.01", "1000000.01"),
                                       ("7", "7")])
def test_parse_accepts_documented_forms(text, want):
    got = parse_amount(text)
    assert isinstance(got, Decimal) and got == Decimal(want)


@pytest.mark.parametrize("bad", ["abc", "", "$", "1.2.3", "1,23.00", "12,3456.00", "(3.10-)", "(-3.10)", "-3.10-",
                                 "(1.00", "()", "--5", "12.00 usd", "12.00 US"])
def test_parse_rejects_malformed_or_ambiguous(bad):
    with pytest.raises(ValueError):
        parse_amount(bad)


@pytest.mark.parametrize("text,want", [("1.005", "1.005"), ("2.500", "2.500"), ("0.10", "0.10")])
def test_parse_never_rounds(text, want):
    got = parse_amount(text)
    assert isinstance(got, Decimal) and str(got) == want


def test_report_quantizes_once_at_the_end_half_even():
    rows = [{"account": "a", "desc": "x", "amount": "0.005"}, {"account": "a", "desc": "y", "amount": "0.005"},
            {"account": "b", "desc": "z", "amount": "10.005"}, {"account": "c", "desc": "w", "amount": "10.015"}]
    accounts = summarize(rows)["accounts"]
    assert all(isinstance(v["total"], Decimal) for v in accounts.values())
    assert str(accounts["a"]["total"]) == "0.01"
    assert str(accounts["b"]["total"]) == "10.00"
    assert str(accounts["c"]["total"]) == "10.02"


def test_report_skips_in_input_order_and_drops_accounts_with_no_parsed_rows():
    rows = [{"account": "a", "desc": "zeta", "amount": "1,23.00"}, {"account": "a", "desc": "ok", "amount": "$5.00"},
            {"account": "b", "desc": "alpha", "amount": "n/a"}, {"account": "b", "desc": "beta", "amount": "$1,00.00"},
            {"account": "a", "desc": "gamma", "amount": ""}]
    s = summarize(rows)
    assert s["skipped"] == ["zeta", "alpha", "beta", "gamma"]
    assert s["accounts"]["a"]["count"] == 1 and s["accounts"]["a"]["total"] == Decimal("5.00")
    assert "b" not in s["accounts"]


def test_report_sums_exactly_and_counts_only_parsed_rows():
    rows = [{"account": "a", "desc": "1", "amount": "0.01"}, {"account": "a", "desc": "2", "amount": "0.02"},
            {"account": "a", "desc": "3", "amount": "(0.30)"}, {"account": "a", "desc": "4", "amount": "$0.10"}]
    s = summarize(rows)
    assert s["skipped"] == []
    assert s["accounts"]["a"]["count"] == 4
    assert isinstance(s["accounts"]["a"]["total"], Decimal)
    assert str(s["accounts"]["a"]["total"]) == "-0.17"


def test_reconcile_matches_only_after_quantization():
    statements = [{"account": "a1", "declared": "10.00", "lines": [{"desc": "x", "amount": "10.005"}]},
                  {"account": "a2", "declared": "2.68", "lines": [{"desc": "y", "amount": "2.675"}]},
                  {"account": "a3", "declared": "0.03",
                   "lines": [{"desc": "p", "amount": "0.01"}, {"desc": "q", "amount": "0.02"}]}]
    r = reconcile(statements)
    assert r["matched"] == ["a1", "a2", "a3"]
    assert r["mismatched"] == [] and r["skipped"] == []


def test_reconcile_mismatches_are_quantized_and_keep_input_order():
    statements = [{"account": "a1", "declared": "10.01", "lines": [{"desc": "x", "amount": "10.00"}]},
                  {"account": "ok", "declared": "1.00", "lines": [{"desc": "y", "amount": "1.00"}]},
                  {"account": "a3", "declared": "$2.005", "lines": [{"desc": "z", "amount": "2.01"}]}]
    r = reconcile(statements)
    assert r["matched"] == ["ok"]
    assert [m["account"] for m in r["mismatched"]] == ["a1", "a3"]
    first, third = r["mismatched"]
    assert isinstance(first["sum"], Decimal) and isinstance(first["difference"], Decimal)
    assert first["sum"] == Decimal("10.00") and first["declared"] == Decimal("10.01")
    assert str(first["difference"]) == "-0.01"
    assert third["declared"] == Decimal("2.00")
    assert str(third["difference"]) == "0.01"


def test_reconcile_lists_skipped_lines_in_input_order():
    statements = [{"account": "a1", "declared": "5.00",
                   "lines": [{"desc": "zeta", "amount": "1,23.00"}, {"desc": "ok", "amount": "$5.00"},
                             {"desc": "alpha", "amount": "n/a"}]},
                  {"account": "a2", "declared": "0.00", "lines": [{"desc": "beta", "amount": ""}]}]
    r = reconcile(statements)
    assert r["matched"] == ["a1", "a2"]
    assert r["mismatched"] == []
    assert r["skipped"] == ["zeta", "alpha", "beta"]
