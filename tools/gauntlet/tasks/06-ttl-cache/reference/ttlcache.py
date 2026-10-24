"""TTLCache: a thread-safe LRU cache whose entries also expire.

  TTLCache(capacity, ttl, clock=time.monotonic)
    capacity: max number of LIVE entries (>= 1); ttl: seconds an entry lives after its last put (> 0);
    clock: a zero-argument callable returning seconds (tests pass a fake clock).

  An entry is EXPIRED when clock() >= its expiry, i.e. it is live only while clock() < expiry.  The clock
  may be called any number of times and every method may use it.

  put(key, value)      store (or overwrite) value; its expiry becomes clock() + ttl and the key becomes the
                       most recently used.  Returns the list of keys this call dropped from the cache, least
                       recently used first: FIRST every EXPIRED entry (purged), and only then - while the
                       number of live entries still exceeds capacity - the least recently used LIVE entry
                       (evicted), one at a time.  So a put never evicts a live entry while an expired entry
                       could have made room instead.  Overwriting an existing key (even an expired one) is
                       not a drop and counts no expiration.
  get(key, default=None)
                       the value if the key is live - and the key then becomes the most recently used; a get
                       does NOT extend the expiry - else default.  Counts exactly one hit (live key) or one
                       miss (missing or expired key).  A get that finds the key expired also removes the
                       entry and counts one expiration (so it is counted once, not on every later read).
  peek(key, default=None)
                       the value if the key is live, else default; NEVER changes recency and never counts a
                       hit or a miss.  An expired entry it observes is still removed and counted as an
                       expiration, like any other removal.
  remove(key)          remove the key if it is live and return its value; return None if the key is missing,
                       and return None - after removing it and counting one expiration - if it is expired.
                       Counts no hit and no miss.
  __len__()            the number of live entries; every expired entry is purged first, each counting one
                       expiration.
  stats()              a dict {"hits", "misses", "expirations", "evictions"} with the running totals:
                       "hits"/"misses" count get's outcomes, "expirations" counts entries removed *because*
                       they were expired (whichever method noticed), "evictions" counts live entries dropped
                       by put's LRU rule.  An expiration is counted once per entry, the first time that entry
                       is found expired and removed; it is gone after that, so it is never counted again.

All methods must be safe to call from many threads at once: each operation - including put's purge/evict
decision and the counters it updates - happens as one indivisible step, and does a bounded amount of work
(the cache is never walked or rebuilt wholesale, and nothing sleeps), so tens of thousands of mixed
operations from eight threads finish in well under a second.
"""
import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, capacity, ttl, clock=time.monotonic):
        if capacity < 1 or ttl <= 0:
            raise ValueError("capacity >= 1 and ttl > 0")
        self.capacity = capacity
        self.ttl = ttl
        self.clock = clock
        self._d = OrderedDict()               # key -> (value, expiry), LRU first
        self._lock = threading.Lock()         # guards _d and every counter below
        self._hits = self._misses = self._expirations = self._evictions = 0

    def _purge(self, now):
        """Caller holds _lock.  Drop every expired entry; return the keys dropped."""
        gone = [k for k, (_, exp) in self._d.items() if now >= exp]
        for k in gone:
            del self._d[k]
            self._expirations += 1
        return gone

    def put(self, key, value):
        with self._lock:
            now = self.clock()
            self._d[key] = (value, now + self.ttl)
            self._d.move_to_end(key)
            if len(self._d) <= self.capacity:
                return []
            dropped = self._purge(now)             # expired entries go first...
            while len(self._d) > self.capacity:    # ...and only then the LRU live entry
                k, _ = self._d.popitem(last=False)
                self._evictions += 1
                dropped.append(k)
            return dropped

    def get(self, key, default=None):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self._misses += 1
                return default
            if self.clock() >= item[1]:
                del self._d[key]
                self._expirations += 1
                self._misses += 1
                return default
            self._d.move_to_end(key)
            self._hits += 1
            return item[0]

    def peek(self, key, default=None):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return default
            if self.clock() >= item[1]:
                del self._d[key]
                self._expirations += 1
                return default
            return item[0]

    def remove(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            del self._d[key]
            if self.clock() >= item[1]:
                self._expirations += 1
                return None
            return item[0]

    def __len__(self):
        with self._lock:
            self._purge(self.clock())
            return len(self._d)

    def stats(self):
        with self._lock:
            return {"hits": self._hits, "misses": self._misses,
                    "expirations": self._expirations, "evictions": self._evictions}
