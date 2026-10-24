"""Deterministic generator for 04-log-needle's workspace files (fixed seed).

Writes service.log, limits.yaml, api_keys.txt and exemptions.txt and computes NO answer:
checker.py derives the answer from the files this writes, and regenerate.py cross-checks
the generator's intent against the checker's result.

T0 = 2026-10-16T03:00:00Z, 1950 s. Every log line is one request; it records the API key
and never the tenant - api_keys.txt is the only key -> tenant map. All times are integer
milliseconds, so the half-open window edge (the wayne decoy's exactly 10.000 s) is exact
rather than float luck.

Timeline (offsets from T0). Every epoch's tenant has its background stream silenced for
15 s either side of the epoch (_reserved_ranges), so the epoch's counted-request count in
its flag window is exactly the number injected:

  ops-burst          120.0s  12 ops requests in 2.8 s                   decoy: the ops tenant is exempt
  acme-join          150.0s  acme's 6 requests split 3+3 over 2 keys   per-tenant, needs the join
  storm-a            250.0s  18 requests over 9.8 s, 7 tenants         global: no tenant over its own limit
  hooli-health       300.0s  5 /api + 4 /health in 5.6 s               decoy: /health is not counted
  hooli-drain        500.0s  8 requests of BOTH hooli keys             exempt by the `*` row (tenant-wide)
  wayne-halfopen     600.0s  6 requests, first -> last 10.000 s        decoy: half-open window
  initech-retired    650.0s  6 requests on a retired key, status 403   join: retired key not counted
  stark-errors       700.0s  4 x status 500 in 9.3 s                   decoy: errors are not a rate burst
  globex-per         750.0s  7 counted in 9.4 s, one 429               per-tenant x2
  storm-b            800.0s  18 requests over 9.8 s                    global
  stark-failover     900.0s  6 requests inside stark's exempt range    exempt (tenant-wide exit row)
  umbrella-per      1040.0s  6 counted in 9.5 s                        per-tenant
  stark-per         1120.0s  7 counted in 9.4 s                        per-tenant x2
  hooli-cutover-*   1360.0s  7 k_1e35 (exempt) + 7 k_5a77 (counted)   the named-key row: 2 violations
  storm-c           1600.0s  18 requests over 9.8 s                    global
  wayne-per         1760.0s  6 counted in 9.5 s                        per-tenant

hooli-drain and hooli-cutover are the near-identical pair: both are one hooli row in
exemptions.txt of the same shape, and the ONLY difference is the `key` column - `*` (every
key of the tenant) vs `k_1e35` (that key alone). hooli-drain puts requests of both keys
inside the `*` range, so reading that row as key-scoped counts 8 requests and invents 3
violations; hooli-cutover puts 7 requests of the named key and 7 of the tenant's other key
inside the named-key range, so ignoring the key clause swallows the 2 real violations.

A storm is the only way a 10 s window reaches the global limit: its 18 requests are spread
over 9.8 s so that exactly ONE window (the one starting at its first request) holds all 18,
and the background is silent for QUIET around it - so no window that holds 17 storm
requests also holds a background request, and the flag set stays one id per storm. The
background alone can never flag: one request every 1.9-2.5 s is at most 6 in a 10 s window
(limit 17), and the round-robin tenant order repeats a tenant no sooner than 7 x 1.9 s.
"""
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

SEED = 20261016
T0 = datetime(2026, 10, 16, 3, 0, 0, tzinfo=timezone.utc)
DUR_MS = 1_950_000

# key -> tenant (the join), plus the status api_keys.txt carries
KEYS = [("k_7f3a", "acme", "active"), ("k_9d40", "acme", "active"), ("k_2b91", "globex", "active"),
        ("k_4c08", "initech", "active"), ("k_55c2", "initech", "retired"), ("k_8a12", "umbrella", "active"),
        ("k_1e35", "hooli", "active"), ("k_5a77", "hooli", "active"), ("k_6d47", "stark", "active"),
        ("k_3f90", "wayne", "active"), ("k_88aa", "ops", "active")]
BG_KEYS = {"acme": ["k_7f3a", "k_9d40"], "globex": ["k_2b91"], "initech": ["k_4c08"],
           "umbrella": ["k_8a12"], "hooli": ["k_1e35", "k_5a77"], "stark": ["k_6d47"],
           "wayne": ["k_3f90"]}
BG_ORDER = ["acme", "globex", "initech", "umbrella", "hooli", "stark", "wayne"]
PATHS = ["/api/orders", "/api/users", "/api/search", "/api/cart", "/api/pay"]
QUIET = [(236_000, 272_000), (786_000, 822_000), (1_586_000, 1_622_000)]   # see the docstring
SILENCE_MS = 15_000                                                        # per-epoch tenant silence

LIMITS_YAML = """\
# Rate limits (gateway v3). Read every line: the limits are documented only here.
#
# COUNTED. A request is COUNTED unless it is one of
#   * a request of tenant `ops`                      (operations traffic is never limited),
#   * a request to the path `/health`                (health probes are not API traffic),
#   * a request made with an API key whose status is `retired` in api_keys.txt
#                                                    (rejected at the edge, never reaches the limiter),
#   * a request covered by an exemption in exemptions.txt.
# Everything else counts - including requests the gateway rejected with 429, and every
# 4xx/5xx that is not listed above.
#
# A window is HALF-OPEN: [t, t + window_s). A request that is the (limit + 1)-th COUNTED
# request inside some window breaks that limit.
#
# A request that breaks ANY limit below is a VIOLATION.
rate_limit:
  window_s: 10                # one second-resolution window length for every limit
  per_tenant: 5               # counted requests of ONE tenant, per window
  global_per_window: 17       # counted requests of ALL tenants together, per window
"""

API_KEYS_TXT = """\
# API keys (gateway v3). The access log records the key a request was made with (`key=`
# in service.log) and does NOT record the tenant. A request belongs to the tenant of its
# key, and EVERY key of a tenant counts against that tenant.
#
# A key marked `retired` no longer belongs to anyone: the edge rejects it (status 403)
# and it is not counted against any limit.
#
#   key     tenant    status
k_7f3a    acme      active
k_9d40    acme      active
k_2b91    globex    active
k_4c08    initech   active
k_55c2    initech   retired
k_8a12    umbrella  active
k_1e35    hooli     active
k_5a77    hooli     active
k_6d47    stark     active
k_3f90    wayne     active
k_88aa    ops       active
"""

EXEMPTIONS_TXT = """\
# Exemptions (gateway v3). A request covered by a row here is NOT counted against any limit.
# Ranges are left-closed, right-open (from <= t < to) and in UTC.
#
# The `key` column scopes the row, and it is the ONLY difference between the two forms -
# they look alike and they do not behave alike:
#   *        the row covers EVERY key of the tenant,
#   k_xxxx   the row covers ONLY that one key - the tenant's other keys are still counted
#            inside the range.
# Read the column, not just the tenant.
#
#   tenant  key      from                   to                     reason
hooli     *        2026-10-16T03:08:20Z   2026-10-16T03:08:40Z   maintenance: the whole tenant is drained
hooli     k_1e35   2026-10-16T03:22:40Z   2026-10-16T03:22:50Z   staged cutover: only this one key is drained
stark     *        2026-10-16T03:15:00Z   2026-10-16T03:15:20Z   failover drill
"""


def _epochs():
    """The injected request streams: (epoch, t_ms, key, path, status)."""
    ev = []

    def add(epoch, t, key, path, status=200):
        ev.append((epoch, int(t), key, path, status))

    for i in range(12):                                        # decoy: ops is exempt entirely
        add("ops-burst", 120_000 + i * 250, "k_88aa", "/api/admin/reindex")
    for t, key in [(150_000, "k_7f3a"), (151_800, "k_9d40"), (153_600, "k_7f3a"),
                   (155_400, "k_9d40"), (157_200, "k_7f3a"), (159_000, "k_9d40")]:
        add("acme-join", t, key, "/api/cart")                  # 3 + 3 across acme's two keys
    for name, t0 in [("storm-a", 250_000), ("storm-b", 800_000), ("storm-c", 1_600_000)]:
        for i in range(18):                                    # 18 counted in one 9.8 s window,
            tenant = BG_ORDER[i % len(BG_ORDER)]               # 2-3 per tenant: the shared limit
            add(name, t0 + i * 576, BG_KEYS[tenant][0], "/api/search")
    for i in range(9):                                         # decoy: /health is not counted
        add("hooli-health", 300_000 + i * 700, "k_1e35", "/health" if i % 2 else "/api/search")
    for i in range(8):                                         # inside the `*` row: both keys exempt
        add("hooli-drain", 500_000 + i * 1_200, "k_1e35" if i % 2 else "k_5a77", "/api/users")
    for t in [600_000, 602_000, 604_000, 606_000, 608_000, 610_000]:
        add("wayne-halfopen", t, "k_3f90", "/api/cart")        # 1st -> 6th exactly 10.000 s
    for i in range(6):                                         # retired key: edge-rejected
        add("initech-retired", 650_000 + i * 1_200, "k_55c2", "/api/users", 403)
    for i in range(4):                                         # decoy: errors, not a rate burst
        add("stark-errors", 700_000 + i * 3_100, "k_6d47", "/api/pay", 500)
    for i in range(7):                                         # 7 counted in 9.4 s: two flags
        add("globex-per", 750_400 + i * 1_566, "k_2b91", "/api/orders", 429 if i == 3 else 200)
    for i in range(6):                                         # inside stark's `*` row: exempt
        add("stark-failover", 900_000 + i * 1_900, "k_6d47", "/api/pay")
    for i in range(6):                                         # 6 counted in 9.5 s: one flag
        add("umbrella-per", 1_040_000 + i * 1_900, "k_8a12", "/api/cart")
    for i in range(7):                                         # 7 counted in 9.4 s: two flags
        add("stark-per", 1_120_000 + i * 1_566, "k_6d47", "/api/search")
    for i in range(7):                                         # the named key of the row: exempt
        add("hooli-cutover-named", 1_360_000 + i * 1_500, "k_1e35", "/api/users")
    for i in range(7):                                         # hooli's OTHER key: still counted
        add("hooli-cutover-other", 1_360_400 + i * 1_500, "k_5a77", "/api/orders")
    for i in range(6):                                         # 6 counted in 9.5 s: one flag
        add("wayne-per", 1_760_000 + i * 1_900, "k_3f90", "/api/pay")
    return ev


def _reserved_ranges(ev):
    """(tenant, lo, hi) per epoch: that tenant's background is silent for SILENCE_MS either
    side of its epoch, so the epoch's counted count in its flag window is exactly what was
    injected and nothing else."""
    tenant_of = dict((key, tenant) for key, tenant, _ in KEYS)
    span = {}
    for epoch, t, key, _path, _status in ev:
        tenant, seen = tenant_of[key], span.get((epoch, tenant_of[key]))
        span[(epoch, tenant)] = (t, t) if seen is None else (min(seen[0], t), max(seen[1], t))
    return [(tenant, lo - SILENCE_MS, hi + SILENCE_MS)
            for (_epoch, tenant), (lo, hi) in span.items()]


def _quiet(t):
    return any(lo <= t <= hi for lo, hi in QUIET)


def generate():
    """-> (service.log, limits.yaml, api_keys.txt, exemptions.txt, meta).

    meta maps each epoch (and "background") to the request ids it produced, in time order.
    """
    rng = random.Random(SEED)
    ev = _epochs()
    reserved = _reserved_ranges(ev)
    t, i, order = rng.randint(0, 1_500), 0, []
    while t <= DUR_MS:
        if i % len(BG_ORDER) == 0:
            order = BG_ORDER[:]
            rng.shuffle(order)
        tenant = order[i % len(BG_ORDER)]
        if not (any(tenant == x and lo <= t <= hi for x, lo, hi in reserved) or _quiet(t)):
            ev.append(("background", t, rng.choice(BG_KEYS[tenant]), rng.choice(PATHS),
                       rng.choice([200] * 14 + [404, 500, 429])))
        t += rng.randint(1_900, 2_500)
        i += 1

    ev.sort(key=lambda e: (e[1], e[2], e[3]))
    lines, meta, ids = [], {}, []
    for epoch, t, key, path, status in ev:
        rid = "%08x" % rng.getrandbits(32)
        ids.append(rid)
        meta.setdefault(epoch, []).append(rid)
        ts = (T0 + timedelta(milliseconds=t)).strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (t % 1000)
        level = "ERROR" if status >= 500 else ("WARN" if status in (403, 404, 429) else "INFO")
        lines.append(f"{ts} {level:5s} req={rid} key={key} path={path} status={status} ms={rng.randint(3, 900)}")
    assert len(set(ids)) == len(ids), "request ids must be unique"
    return "\n".join(lines) + "\n", LIMITS_YAML, API_KEYS_TXT, EXEMPTIONS_TXT, meta


def write(ws: Path) -> dict:
    log, limits, keys, exempt, meta = generate()
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "service.log").write_text(log)
    (ws / "limits.yaml").write_text(limits)
    (ws / "api_keys.txt").write_text(keys)
    (ws / "exemptions.txt").write_text(exempt)
    return meta


if __name__ == "__main__":
    m = write(Path(__file__).resolve().parent / "workspace")
    print("wrote workspace/;", sum(len(v) for v in m.values()), "requests")
    for name, rids in m.items():
        print(f"  {name:22s} {len(rids):4d} ids")
