"""inventory.py - module A: the items and the tags on them; pricing.py quotes them and orders.py
holds them - one system.

Documented behaviour, for every caller:

  reset()                forget every item AND every tag - every index derived from them goes.
  add_item(name, qty, tags=[])
                         qty must be > 0 else ValueError; adds qty units of name, creating the
                         item if it is new.  Its tags are the union of the tags it already had
                         and `tags`, once each, in first-seen order.  Returns its tag list.
  remove_item(name, qty) qty must not exceed the stock, else InsufficientStock; an unknown name
                         is a KeyError.  At zero the item is gone - and so is its tag.
  tagged(tag)            the names of the items currently IN STOCK carrying tag, sorted.
  stock(name)            the units of name in stock - 0 when the name is unknown.
  total()                the stock of every item, summed.

Callers catch these by TYPE, never by "something raised": a qty that is not positive is a
ValueError, a name that is not in stock a KeyError, and asking for more units than there are an
InsufficientStock (a ValueError) - orders.reserve relies on exactly that.

Identity, not values: the `tags` argument is COPIED into the item (the caller may keep the list
and mutate it later), two calls that pass no tags must not share a list, and the list add_item
returns is a copy too.
"""


class InsufficientStock(ValueError):
    """More units were asked for than the stock holds (a ValueError)."""


_items = {}
_by_tag = {}


def reset():
    """Forget every item, and every tag with it."""
    _items.clear()


def add_item(name, qty, tags=[]):
    """Add qty of name (qty > 0); tags is an optional list of strings.  Returns the item's tags."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    item = _items.setdefault(name, {"qty": 0, "tags": tags})
    item["qty"] += qty
    for t in tags:
        if t not in item["tags"]:
            item["tags"].append(t)
        _by_tag.setdefault(t, set()).add(name)
    return item["tags"]


def remove_item(name, qty):
    item = _items[name]
    if qty > item["qty"]:
        raise InsufficientStock("not enough stock")
    item["qty"] -= qty
    if item["qty"] == 0:
        del _items[name]


def tagged(tag):
    """Names of the items in stock that carry tag, sorted."""
    return sorted(_by_tag.get(tag, set()))


def stock(name):
    """The units of name in stock - 0 when the name is unknown."""
    item = _items.get(name)
    return item["qty"] if item else 0


def total():
    return sum(i["qty"] for i in _items.values())
