"""orders.py - module C: the reservations; a hold is stock that inventory has and a price that
pricing quoted - one system.

Documented behaviour, for every caller:

  reset()                 forget every reservation.
  reserve(sku, qty, key)  hold qty units of sku against the order named by key at the unit price
                          in force right now, and return that hold as a dict of "sku", "qty",
                          "unit" and "total" - a COPY the caller may mutate freely.  qty must be
                          > 0 (ValueError) and fit the units of sku that are not already held,
                          else inventory.InsufficientStock; sku must have a price in force, else
                          pricing's ValueError.  The stock is read before the price, so missing
                          units are refused as InsufficientStock even when sku is also unpriced.
                          The key is idempotent: a repeat of the same key returns the hold
                          already made and holds nothing more; a repeat naming another sku or
                          qty is a different order and raises ValueError.
  release(key)            drop the hold named by key, free its units and return its total.  An
                          unknown key is a KeyError - releasing twice is a KeyError, not a no-op.
  held(sku)               the units held for sku across every order, as an int.
  total(sku)              the value held for sku, as an exact Decimal - 0 when nothing is held.

A hold is a contract at the price in force when it was made: once reserved, a later set_price, the
sku being tagged for clearance, or the sku leaving the inventory changes neither its unit price
nor its total.  Recomputing a held total from pricing is a wrong answer, not a fresh one.
"""
from decimal import Decimal

import inventory
import pricing

_holds = {}


def reset():
    """Forget every reservation."""
    _holds.clear()


def reserve(sku, qty, key):
    """Hold qty units of sku against the order named by key."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    if key in _holds:
        hold = _holds[key]
        if hold["sku"] != sku or hold["qty"] != qty:
            raise ValueError(f"key {key!r} already holds a different order")
        return hold
    if qty > inventory.total():
        raise ValueError("not enough stock")
    unit = pricing.quote(sku, 1)
    _holds[key] = {"sku": sku, "qty": qty, "unit": unit}
    return _holds[key]


def release(key):
    """Drop the hold named by key, free its units and return its total."""
    if key not in _holds:
        raise ValueError(f"no reservation {key!r}")
    hold = _holds.pop(key)
    return pricing.quote(hold["sku"], hold["qty"])


def held(sku):
    """The units held for sku across every order."""
    return sum(h["qty"] for h in _holds.values() if h["sku"] == sku)


def total(sku):
    """The value held for sku, as an exact Decimal."""
    return sum((pricing.quote(h["sku"], h["qty"]) for h in _holds.values() if h["sku"] == sku),
               Decimal(0))
