"""The shop's suite over inventory.py, pricing.py and orders.py; CI runs it in a random order.  A
real signal, but not the whole contract - the docstrings are.
"""
from decimal import Decimal

import pytest

import inventory as inv
import orders as od
import pricing as pr


def test_add_and_total():
    inv.reset()
    inv.add_item("apple", 3)
    inv.add_item("pear", 2, ["fruit"])
    assert inv.total() == 5


def test_tagged_finds_the_tagged_stock():
    inv.reset()
    inv.add_item("kiwi", 1, ["produce"])
    assert inv.tagged("produce") == ["kiwi"]


def test_partial_removal_leaves_the_rest():
    inv.reset()
    inv.add_item("bolt", 4, ["hardware"])
    inv.remove_item("bolt", 1)
    assert (inv.total(), inv.add_item("bolt", 1, ["hardware"])) == (3, ["hardware"])


def test_tags_are_merged_in_first_seen_order():
    inv.reset()
    inv.add_item("nut", 1, ["bulk", "fragile"])
    assert inv.add_item("nut", 1, ["fragile", "bulk"]) == ["bulk", "fragile"]


def test_qty_must_be_positive():
    inv.reset()
    with pytest.raises(ValueError):
        inv.add_item("widget", 0)


def test_price_and_quote():
    inv.reset()
    pr.reset()
    inv.add_item("grommet", 10)
    pr.set_price("grommet", "2.00")
    assert (pr.price_of("grommet"), pr.quote("grommet", 5)) == (Decimal("2.00"), Decimal("10.00"))


def test_clearance_halves_the_unit_price():
    inv.reset()
    pr.reset()
    inv.add_item("screw", 2, ["clearance"])
    pr.set_price("screw", "3.00")
    assert pr.quote("screw", 1) == Decimal("1.50")


def test_reserve_holds_the_stock_until_it_is_released():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("crate", 10)
    pr.set_price("crate", "5.00")
    assert od.reserve("crate", 4, "ord-a")["qty"] == 4
    assert (od.held("crate"), od.total("crate")) == (4, Decimal("20.00"))
    assert od.release("ord-a") == Decimal("20.00")
    assert (od.held("crate"), od.total("crate")) == (0, Decimal("0"))


def test_a_repeated_key_holds_the_stock_once():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("cane", 3)
    pr.set_price("cane", "1.00")
    od.reserve("cane", 2, "ord-b")
    od.reserve("cane", 2, "ord-b")
    assert od.held("cane") == 2
