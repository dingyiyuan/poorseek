"""inventory.py - module A (reference: copied tag lists, a tag index maintained on every removal
and on reset, and the documented exception types)."""


class InsufficientStock(ValueError):
    """More units were asked for than the stock holds (a ValueError)."""


_items = {}
_by_tag = {}


def reset():
    """Forget every item, and every tag with it."""
    _items.clear()
    _by_tag.clear()


def add_item(name, qty, tags=()):
    """Add qty of name (qty > 0); tags is an optional list of strings.  Returns the item's tags."""
    if qty <= 0:
        raise ValueError("qty must be positive")
    item = _items.setdefault(name, {"qty": 0, "tags": []})
    item["qty"] += qty
    for t in tags:
        if t not in item["tags"]:
            item["tags"].append(t)
        _by_tag.setdefault(t, set()).add(name)
    return list(item["tags"])


def remove_item(name, qty):
    item = _items[name]
    if qty > item["qty"]:
        raise InsufficientStock("not enough stock")
    item["qty"] -= qty
    if item["qty"] == 0:
        del _items[name]
        for t in item["tags"]:
            _by_tag.get(t, set()).discard(name)


def tagged(tag):
    """Names of the items in stock that carry tag, sorted."""
    return sorted(_by_tag.get(tag, set()))


def stock(name):
    """The units of name in stock - 0 when the name is unknown."""
    item = _items.get(name)
    return item["qty"] if item else 0


def total():
    return sum(i["qty"] for i in _items.values())
