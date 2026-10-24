"""Unit prices and quotes - module B (reference: the memo is keyed on the unit price in force,
so it cannot survive a price change or a change of the item's clearance status; the table is
ordered by the decimal value)."""

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
    _memo.clear()


def price_of(name):
    """The unit price recorded for name."""
    if name not in _prices:
        raise ValueError(f"no price recorded for {name!r}")
    return _prices[name]


def _unit(name):
    """The price in force for name: the recorded price, halved while the item is in clearance."""
    if name not in _prices:
        raise ValueError(f"no price recorded for {name!r}")
    price = _prices[name]
    if name in inventory.tagged("clearance"):
        price = price / 2
    return price


def quote(name, qty):
    """The cost of qty units of name; halved per unit while the item is in clearance."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    unit = _unit(name)
    key = (name, qty, unit)
    if key not in _memo:
        _memo[key] = unit * qty
    return _memo[key]


def price_table():
    """Every recorded price as "<name> <price>", smallest price first."""
    rows = sorted(_prices.items(), key=lambda kv: kv[1])
    return [f"{name} {p}" for name, p in rows]
