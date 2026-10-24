"""orders.py - module C (reference: a hold is priced ONCE, at reserve time, so it survives any
later price or inventory change; the stock check asks inventory for that sku's own units, net of
what is already held, and refuses with inventory's documented InsufficientStock)."""
from decimal import Decimal

import inventory
import pricing

_holds = {}


def reset():
    """Forget every reservation."""
    _holds.clear()


def reserve(sku, qty, key):
    """Hold qty units of sku against the order named by key, at the quote in force right now."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    if key in _holds:
        hold = _holds[key]
        if hold["sku"] != sku or hold["qty"] != qty:
            raise ValueError(f"key {key!r} already holds a different order")
        return dict(hold)
    if qty > inventory.stock(sku) - held(sku):
        raise inventory.InsufficientStock(f"{sku!r} has no {qty} units left to hold")
    unit = pricing.quote(sku, 1)
    _holds[key] = {"sku": sku, "qty": qty, "unit": unit, "total": unit * qty}
    return dict(_holds[key])


def release(key):
    """Drop the hold named by key, free its units and return the total fixed when it was made."""
    if key not in _holds:
        raise KeyError(key)
    return _holds.pop(key)["total"]


def held(sku):
    """The units held for sku across every order."""
    return sum(h["qty"] for h in _holds.values() if h["sku"] == sku)


def total(sku):
    """The value held for sku, as an exact Decimal."""
    return sum((h["total"] for h in _holds.values() if h["sku"] == sku), Decimal(0))
