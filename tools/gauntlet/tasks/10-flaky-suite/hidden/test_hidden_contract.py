"""Hidden grading for 10-flaky-suite.

The visible suite is re-stated here - never imported from the workspace, so a model cannot weaken
grading by editing a test - and run in 20 random orders (the CI order), and then the whole
documented contract of all THREE modules is checked directly: the tag-list identity rules, the tag
index across reset and removal, the accessors and the documented exception TYPES, the price memo
against a price change and against inventory changing underneath it, the price-table ordering, and
orders' holds - the idempotent key, the units already held counting against the stock, the hold
that is a copy, and the two rules that only hold if orders.py is read together with pricing.py and
inventory.py: a live quote must FOLLOW a price/inventory change while a hold made earlier must NOT.
"""
import random
from decimal import Decimal

import pytest

import inventory as inv
import orders as od
import pricing as pr


# --- the visible suite, re-stated, run in 20 random orders ----------------------------------------
def _add_and_total():
    inv.reset()
    inv.add_item("apple", 3)
    inv.add_item("pear", 2, ["fruit"])
    assert inv.total() == 5


def _tagged_finds_the_tagged_stock():
    inv.reset()
    inv.add_item("kiwi", 1, ["produce"])
    assert inv.tagged("produce") == ["kiwi"]


def _partial_removal_leaves_the_rest():
    inv.reset()
    inv.add_item("bolt", 4, ["hardware"])
    inv.remove_item("bolt", 1)
    assert (inv.total(), inv.add_item("bolt", 1, ["hardware"])) == (3, ["hardware"])


def _tags_are_merged_in_first_seen_order():
    inv.reset()
    inv.add_item("nut", 1, ["bulk", "fragile"])
    assert inv.add_item("nut", 1, ["fragile", "bulk"]) == ["bulk", "fragile"]


def _qty_must_be_positive():
    inv.reset()
    with pytest.raises(ValueError):
        inv.add_item("widget", 0)


def _price_and_quote():
    inv.reset()
    pr.reset()
    inv.add_item("grommet", 10)
    pr.set_price("grommet", "2.00")
    assert (pr.price_of("grommet"), pr.quote("grommet", 5)) == (Decimal("2.00"), Decimal("10.00"))


def _clearance_halves_the_unit_price():
    inv.reset()
    pr.reset()
    inv.add_item("screw", 2, ["clearance"])
    pr.set_price("screw", "3.00")
    assert pr.quote("screw", 1) == Decimal("1.50")


def _reserve_holds_the_stock_until_it_is_released():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("crate", 10)
    pr.set_price("crate", "5.00")
    assert od.reserve("crate", 4, "ord-a")["qty"] == 4
    assert (od.held("crate"), od.total("crate")) == (4, Decimal("20.00"))
    assert od.release("ord-a") == Decimal("20.00")
    assert (od.held("crate"), od.total("crate")) == (0, Decimal("0"))


def _a_repeated_key_holds_the_stock_once():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("cane", 3)
    pr.set_price("cane", "1.00")
    od.reserve("cane", 2, "ord-b")
    od.reserve("cane", 2, "ord-b")
    assert od.held("cane") == 2


VISIBLE = [_add_and_total, _tagged_finds_the_tagged_stock, _partial_removal_leaves_the_rest,
           _tags_are_merged_in_first_seen_order, _qty_must_be_positive, _price_and_quote,
           _clearance_halves_the_unit_price, _reserve_holds_the_stock_until_it_is_released,
           _a_repeated_key_holds_the_stock_once]


@pytest.mark.parametrize("seed", range(20))
def test_visible_suite_in_random_order(seed):
    fns = VISIBLE[:]
    random.Random(seed).shuffle(fns)
    for f in fns:
        f()


# --- inventory fault 1: the tags argument and the returned list are copies, no two items share one
def test_caller_tag_list_is_copied():
    inv.reset()
    tags = ["fruit"]
    inv.add_item("apple", 1, tags)
    tags.append("metal")
    assert inv.tagged("metal") == []
    assert inv.tagged("fruit") == ["apple"]


def test_returned_tag_list_is_a_copy():
    inv.reset()
    got = inv.add_item("apple", 1, ["fruit"])
    got.append("metal")
    assert inv.tagged("metal") == []
    assert inv.add_item("apple", 1) == ["fruit"]


def test_no_two_untagged_items_share_one_list():
    inv.reset()
    a = inv.add_item("bolt", 10)
    b = inv.add_item("nut", 5)
    a.append("metal")
    assert b == []
    assert inv.tagged("metal") == []


def test_two_items_with_tags_do_not_share_them():
    inv.reset()
    first, second = ["a"], ["b"]
    inv.add_item("bolt", 1, first)
    inv.add_item("nut", 1, second)
    first.append("z")
    assert inv.tagged("z") == []
    assert inv.tagged("b") == ["nut"]


# --- inventory fault 2: the tag index is maintained on reset and on removal -----------------------
def test_reset_forgets_every_tag():
    inv.reset()
    inv.add_item("kiwi", 1, ["fruit"])
    inv.reset()
    assert inv.tagged("fruit") == []
    assert inv.total() == 0


def test_removing_the_last_unit_drops_the_tag():
    inv.reset()
    inv.add_item("kiwi", 1, ["fruit"])
    inv.remove_item("kiwi", 1)
    assert inv.tagged("fruit") == []


def test_partial_removal_keeps_the_tag():
    inv.reset()
    inv.add_item("kiwi", 3, ["fruit"])
    inv.remove_item("kiwi", 2)
    assert inv.tagged("fruit") == ["kiwi"]


def test_readding_after_removal_is_tagged_again():
    inv.reset()
    inv.add_item("kiwi", 1, ["fruit"])
    inv.remove_item("kiwi", 1)
    inv.add_item("kiwi", 2, ["fruit"])
    assert inv.tagged("fruit") == ["kiwi"]


# --- inventory: the accessors and the documented exception TYPES ----------------------------------
def test_stock_reports_the_units_and_zero_for_an_unknown_name():
    inv.reset()
    inv.add_item("bolt", 4)
    assert (inv.stock("bolt"), inv.stock("ghost")) == (4, 0)


def test_stock_follows_removal_to_zero():
    inv.reset()
    inv.add_item("bolt", 4)
    inv.remove_item("bolt", 4)
    assert (inv.stock("bolt"), inv.total()) == (0, 0)


def test_asking_for_more_than_the_stock_is_an_insufficient_stock():
    inv.reset()
    inv.add_item("bolt", 1)
    with pytest.raises(inv.InsufficientStock):
        inv.remove_item("bolt", 2)
    assert issubclass(inv.InsufficientStock, ValueError)


def test_removing_an_unknown_name_is_a_key_error():
    inv.reset()
    with pytest.raises(KeyError):
        inv.remove_item("ghost", 1)


def test_a_non_positive_qty_is_a_value_error():
    inv.reset()
    with pytest.raises(ValueError):
        inv.add_item("bolt", -1)


# --- pricing fault 3: the price memo never outlives the state the quote is about ------------------
def test_quote_needs_a_price():
    inv.reset()
    pr.reset()
    inv.add_item("cap", 1)
    with pytest.raises(ValueError):
        pr.quote("cap", 1)


def test_quote_follows_a_price_change():
    inv.reset()
    pr.reset()
    pr.set_price("bolt", "2.00")
    assert pr.quote("bolt", 4) == Decimal("8.00")
    pr.set_price("bolt", "2.50")
    assert pr.quote("bolt", 4) == Decimal("10.00")


def test_quote_follows_the_price_back_down():
    inv.reset()
    pr.reset()
    pr.set_price("nut", "1.00")
    assert pr.quote("nut", 2) == Decimal("2.00")
    pr.set_price("nut", "4.00")
    assert pr.quote("nut", 2) == Decimal("8.00")
    pr.set_price("nut", "1.00")
    assert pr.quote("nut", 2) == Decimal("2.00")


def test_quote_follows_inventory_tagging_an_item_for_clearance():
    inv.reset()
    pr.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    assert pr.quote("bolt", 3) == Decimal("6.00")
    inv.add_item("bolt", 1, ["clearance"])
    assert pr.quote("bolt", 3) == Decimal("3.00")


def test_quote_follows_inventory_removing_the_item():
    inv.reset()
    pr.reset()
    inv.add_item("kiwi", 2, ["clearance"])
    pr.set_price("kiwi", "3.00")
    assert pr.quote("kiwi", 1) == Decimal("1.50")
    inv.remove_item("kiwi", 2)
    assert pr.quote("kiwi", 1) == Decimal("3.00")


# --- pricing fault 4: the price table is ordered by value, ties in recording order ----------------
def test_price_table_is_smallest_first():
    pr.reset()
    pr.set_price("washer", "2.00")
    pr.set_price("spanner", "1.50")
    assert pr.price_table() == ["spanner 1.50", "washer 2.00"]


def test_price_table_orders_by_value_not_by_text():
    pr.reset()
    pr.set_price("bolt", "2.00")
    pr.set_price("kiwi", "10.00")
    pr.set_price("nut", "9.00")
    assert pr.price_table() == ["bolt 2.00", "nut 9.00", "kiwi 10.00"]


def test_price_table_keeps_equal_prices_in_recording_order():
    pr.reset()
    pr.set_price("zzz", "2.00")
    pr.set_price("aaa", "2.00")
    pr.set_price("mmm", "1.50")
    assert pr.price_table() == ["mmm 1.50", "zzz 2.00", "aaa 2.00"]


def test_an_unpriced_name_is_a_value_error_not_a_key_error():
    pr.reset()
    with pytest.raises(ValueError):
        pr.price_of("ghost")


def test_a_non_positive_price_is_a_value_error():
    pr.reset()
    with pytest.raises(ValueError):
        pr.set_price("bolt", "0.00")


def test_a_non_positive_quote_quantity_is_a_value_error():
    inv.reset()
    pr.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "1.00")
    with pytest.raises(ValueError):
        pr.quote("bolt", 0)


# --- orders: the hold is a copy, priced from the quote in force, keyed idempotently ---------------
def test_reserve_returns_a_copy_carrying_the_documented_fields():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    hold = od.reserve("bolt", 3, "ord-1")
    assert (hold["sku"], hold["qty"], hold["unit"], hold["total"]) == \
           ("bolt", 3, Decimal("2.00"), Decimal("6.00"))
    hold["qty"] = 99
    hold["total"] = Decimal("0")
    assert (od.held("bolt"), od.total("bolt")) == (3, Decimal("6.00"))


def test_reserve_is_idempotent_for_one_key():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    first = od.reserve("bolt", 3, "ord-1")
    again = od.reserve("bolt", 3, "ord-1")
    assert (first, again) == (again, {"sku": "bolt", "qty": 3, "unit": Decimal("2.00"),
                                      "total": Decimal("6.00")})
    assert od.held("bolt") == 3


def test_a_key_reused_for_another_order_is_a_value_error():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    inv.add_item("nut", 5)
    pr.set_price("bolt", "1.00")
    pr.set_price("nut", "1.00")
    od.reserve("bolt", 1, "ord-1")
    with pytest.raises(ValueError):
        od.reserve("nut", 1, "ord-1")
    with pytest.raises(ValueError):
        od.reserve("bolt", 2, "ord-1")


def test_the_units_already_held_count_against_the_stock():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "1.00")
    od.reserve("bolt", 3, "ord-1")
    with pytest.raises(inv.InsufficientStock):
        od.reserve("bolt", 3, "ord-2")
    assert od.held("bolt") == 3


def test_a_hold_cannot_borrow_another_skus_stock():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 1)
    inv.add_item("nut", 100)
    pr.set_price("bolt", "1.00")
    pr.set_price("nut", "1.00")
    with pytest.raises(inv.InsufficientStock):
        od.reserve("bolt", 5, "ord-1")
    assert od.held("bolt") == 0


def test_reserving_an_unpriced_sku_is_a_value_error():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    with pytest.raises(ValueError):
        od.reserve("bolt", 1, "ord-1")
    assert od.held("bolt") == 0


def test_the_stock_is_read_before_the_price():
    inv.reset()
    pr.reset()
    od.reset()
    with pytest.raises(inv.InsufficientStock):
        od.reserve("ghost", 1, "ord-1")
    assert od.held("ghost") == 0


def test_a_non_positive_reserved_qty_is_a_value_error():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "1.00")
    with pytest.raises(ValueError):
        od.reserve("bolt", 0, "ord-1")


def test_release_frees_the_units_for_a_later_hold():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    od.reserve("bolt", 5, "ord-1")
    assert od.release("ord-1") == Decimal("10.00")
    od.reserve("bolt", 5, "ord-2")
    assert od.held("bolt") == 5


def test_an_unknown_release_key_is_a_key_error():
    inv.reset()
    pr.reset()
    od.reset()
    with pytest.raises(KeyError):
        od.release("ghost")


def test_releasing_twice_is_a_key_error():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    od.reserve("bolt", 2, "ord-1")
    od.release("ord-1")
    with pytest.raises(KeyError):
        od.release("ord-1")


def test_held_and_total_are_zero_for_a_sku_with_no_holds():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 5)
    pr.set_price("bolt", "2.00")
    assert (od.held("bolt"), od.total("bolt")) == (0, Decimal("0"))


# --- the cross-module rules: a LIVE quote follows the world, a HOLD made earlier does not ---------
def test_a_hold_is_priced_from_the_quote_in_force_when_it_was_made():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("screw", 2, ["clearance"])
    pr.set_price("screw", "3.00")
    assert od.reserve("screw", 2, "ord-1")["total"] == Decimal("3.00")
    assert od.total("screw") == Decimal("3.00")


def test_a_hold_survives_a_later_price_change():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 9)
    pr.set_price("bolt", "2.00")
    od.reserve("bolt", 3, "ord-1")
    pr.set_price("bolt", "9.00")
    assert pr.quote("bolt", 3) == Decimal("27.00")
    assert (od.total("bolt"), od.release("ord-1")) == (Decimal("6.00"), Decimal("6.00"))


def test_a_hold_survives_the_sku_being_tagged_for_clearance():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("bolt", 9)
    pr.set_price("bolt", "2.00")
    od.reserve("bolt", 3, "ord-1")
    inv.add_item("bolt", 1, ["clearance"])
    assert pr.quote("bolt", 3) == Decimal("3.00")
    assert od.total("bolt") == Decimal("6.00")


def test_a_hold_survives_the_sku_leaving_the_inventory():
    inv.reset()
    pr.reset()
    od.reset()
    inv.add_item("kiwi", 2, ["clearance"])
    pr.set_price("kiwi", "3.00")
    od.reserve("kiwi", 2, "ord-1")
    inv.remove_item("kiwi", 2)
    assert pr.quote("kiwi", 2) == Decimal("6.00")
    assert od.total("kiwi") == Decimal("3.00")
