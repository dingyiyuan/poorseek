"""Hidden suite for 06-ttl-cache.  Every assertion here traces to a sentence in workspace/ttlcache.py's
docstring (see the derivability table in the report); the visible suite covers only put/get and ttl-at-exact."""
import random
import sys
import threading
import time

from ttlcache import TTLCache


class Clock:
    def __init__(self):
        self.t = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t


def test_put_purges_expired_before_evicting_live():
    clk = Clock()
    c = TTLCache(2, 10, clk)
    assert c.put("x", 1) == []           # under capacity: nothing dropped
    clk.t = 5
    assert c.put("y", 2) == []           # now exactly at capacity
    clk.t = 6
    assert c.get("x") == 1               # x most recently used, y the least
    clk.t = 10.5                         # x has expired, y is live
    dropped = c.put("z", 3)              # over capacity: purge expired x, do NOT evict live y
    assert dropped == ["x"]
    assert c.peek("y") == 2 and c.peek("z") == 3
    assert len(c) == 2
    s = c.stats()
    assert (s["expirations"], s["evictions"]) == (1, 0)


def test_put_purges_every_expired_entry():
    clk = Clock()
    c = TTLCache(2, 10, clk)
    c.put("a", 1)
    c.put("b", 2)
    clk.t = 10                            # both expired, all at once
    dropped = c.put("c", 3)
    assert dropped == ["a", "b"]
    assert len(c) == 1
    s = c.stats()
    assert (s["expirations"], s["evictions"]) == (2, 0)


def test_put_evicts_the_lru_live_entry_when_nothing_is_expired():
    clk = Clock()
    c = TTLCache(2, 100, clk)
    c.put("a", 1)
    c.put("b", 2)
    assert c.put("c", 3) == ["a"]         # nothing expired -> the least recent live key goes
    s = c.stats()
    assert (s["evictions"], s["expirations"]) == (1, 0)
    assert c.peek("a") is None and c.peek("b") == 2 and c.peek("c") == 3


def test_get_expired_counts_one_miss_and_one_expiration():
    clk = Clock()
    c = TTLCache(4, 10, clk)
    c.put("a", 1)
    assert c.get("a") == 1
    clk.t = 10
    assert c.get("a") is None             # expired exactly at ttl: a miss...
    assert c.get("a") is None             # ...and the miss is not charged a second expiration
    assert c.peek("a") is None
    s = c.stats()
    assert (s["hits"], s["misses"], s["expirations"]) == (1, 2, 1)


def test_get_refreshes_recency_but_not_expiry():
    clk = Clock()
    c = TTLCache(2, 10, clk)
    c.put("a", 1)
    clk.t = 5
    c.put("b", 2)
    clk.t = 9
    assert c.get("a") == 1                # a is now the most recent, so b is the LRU
    assert c.put("c", 3) == ["b"]
    assert c.peek("b") is None and c.peek("a") == 1
    clk.t = 10
    assert c.get("a") is None             # ...but a still dies at its own expiry
    assert c.stats()["evictions"] == 1


def test_peek_never_changes_recency_or_counters():
    clk = Clock()
    c = TTLCache(2, 100, clk)
    c.put("a", 1)
    c.put("b", 2)
    assert c.peek("a") == 1
    assert c.put("c", 3) == ["a"]         # the peek did not promote a
    assert c.peek("b") == 2 and c.peek("c") == 3
    s = c.stats()
    assert (s["hits"], s["misses"], s["evictions"]) == (0, 0, 1)


def test_peek_removes_an_expired_entry_and_counts_one_expiration():
    clk = Clock()
    c = TTLCache(4, 10, clk)
    c.put("a", 1)
    clk.t = 10
    assert c.peek("a") is None
    assert c.peek("a") is None
    assert len(c) == 0
    s = c.stats()
    assert (s["expirations"], s["hits"], s["misses"]) == (1, 0, 0)


def test_remove_live_missing_and_expired():
    clk = Clock()
    c = TTLCache(3, 10, clk)
    c.put("a", 1)
    clk.t = 10
    assert c.remove("a") is None
    assert c.stats()["expirations"] == 1
    assert c.remove("a") is None          # already gone: no second expiration
    assert c.stats()["expirations"] == 1
    c.put("b", 2)
    assert c.remove("b") == 2
    assert c.remove("b") is None
    assert c.remove("nope") is None
    assert len(c) == 0
    s = c.stats()
    assert (s["hits"], s["misses"]) == (0, 0)


def test_len_counts_only_live_and_overwrite_resets_expiry():
    clk = Clock()
    c = TTLCache(3, 10, clk)
    c.put("a", 1)
    clk.t = 4
    c.put("b", 2)
    clk.t = 8
    assert c.put("a", 9) == []            # overwrite is not a drop
    assert len(c) == 2
    clk.t = 12                            # the ORIGINAL a-expiry (10) is long past, b (14) is still live
    assert c.peek("a") == 9 and len(c) == 2
    clk.t = 18
    assert len(c) == 0
    assert c.stats()["expirations"] == 2


def test_put_over_an_expired_key_is_not_a_drop():
    clk = Clock()
    c = TTLCache(2, 10, clk)
    c.put("a", 1)
    clk.t = 10
    assert c.put("a", 2) == []
    assert c.stats()["expirations"] == 0  # an overwritten expired entry was never dropped
    assert c.get("a") == 2
    assert c.stats()["expirations"] == 0


def test_thread_stress_safe_and_self_consistent():
    clk = Clock()                         # never advances -> nothing may ever expire here
    cap = 40
    c = TTLCache(cap, 5, clk)
    n_threads, ops = 8, 4000
    errors = []
    gets = [0] * n_threads
    start = threading.Barrier(n_threads)

    def worker(w):
        r = random.Random(9871 + w)
        try:
            start.wait(30)
            for _ in range(ops):
                k = r.randrange(0, 160)
                p = r.random()
                if p < 0.50:
                    c.put(k, (k, w))
                elif p < 0.90:
                    gets[w] += 1
                    v = c.get(k)
                    if v is not None and v[0] != k:
                        errors.append(("wrong value", k, v))
                elif p < 0.95:
                    c.remove(k)
                else:
                    c.peek(k)
        except Exception as e:            # noqa: BLE001 - any exception at all is a failure
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(n_threads)]
    prev = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)           # hand the GIL over constantly: a missing lock cannot hide here
    t0 = time.monotonic()
    try:
        for t in ts:
            t.start()
        for t in ts:
            t.join(30)
        elapsed = time.monotonic() - t0
    finally:
        sys.setswitchinterval(prev)

    assert not any(t.is_alive() for t in ts), "deadlock: a worker did not finish"
    assert not errors, errors[:3]
    assert elapsed < 20, f"the concurrent phase took {elapsed:.1f}s"   # bounded work, generously
    s = c.stats()
    assert s["hits"] + s["misses"] == sum(gets)     # exactly one counter per get outcome
    assert s["expirations"] == 0                    # the clock never advanced
    assert len(c) <= cap
    live = [k for k in range(160) if c.peek(k) is not None]             # independent recount
    assert len(live) == len(c)
    assert all(c.peek(k)[0] == k for k in live)
