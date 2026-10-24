from ttlcache import TTLCache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_put_then_get():
    c = TTLCache(4, 10, Clock())
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("a", 9)                  # overwrite
    assert c.get("a") == 9
    assert c.get("zzz") is None
    assert c.get("zzz", "d") == "d"


def test_expires_at_ttl():
    clk = Clock()
    c = TTLCache(4, 10, clk)
    c.put("a", 1)
    clk.t = 9
    assert c.get("a") == 1
    clk.t = 10
    assert c.get("a") is None      # expired exactly at ttl
