"""pricing.py - module B: the unit prices and the quotes; it reads inventory and orders.py holds
its unit prices - one system.

Documented behaviour, for every caller:

  reset()                forget every recorded price AND every memoised quote.
  set_price(name, price) record the unit price of name - a string or a Decimal, > 0, else
                         ValueError.  A price already recorded for the name is replaced.
  price_of(name)         the recorded unit price as a Decimal; ValueError when none.
  quote(name, qty)       the cost of qty units as an exact Decimal: qty must be > 0 and the
                         name priced, else ValueError.  While the item is in stock and tagged
                         "clearance" the unit price is halved, so quote(name, 1) is the unit
                         price IN FORCE - what orders.reserve prices a hold with.  price_of is
                         the recorded price, clearance or not.
  price_table()          every recorded price as "<name> <price>" text, one line each, smallest
                         price first; equal prices keep the order they were recorded.

A memo is not an answer about a world that has moved on: once set_price replaces a price, or
inventory drops an item's "clearance" tag or removes it, every later quote is about the new
state - a quote that outlives the change is a wrong answer, not a fast one.
"""

from decimal import Decimal

import inventory

_prices = {}
_memo = {}


def reset():
    """Forget every recorded price and every memoised quote."""
    _prices.clear()
    _memo.clear()


def set_price(name, price):
    """Record the unit price of name (a string or a Decimal, greater than zero)."""
    p = price if isinstance(price, Decimal) else Decimal(price)
    if p <= 0:
        raise ValueError("price must be positive")
    _prices[name] = p


def price_of(name):
    """The price recorded for name; ValueError when none is recorded."""
    if name not in _prices:
        raise ValueError(f"no price recorded for {name!r}")
    return _prices[name]


def quote(name, qty):
    """The cost of qty units of name; halved per unit while the item is in clearance."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    if name not in _prices:
        raise ValueError(f"no price recorded for {name!r}")
    key = (name, qty)
    if key not in _memo:
        unit = _prices[name]
        if name in inventory.tagged("clearance"):
            unit = unit / 2
        _memo[key] = unit * qty
    return _memo[key]


def price_table():
    """Every recorded price as "<name> <price>", smallest price first."""
    return [f"{n} {p}" for n, p in sorted(_prices.items(), key=lambda kv: str(kv[1]))]
