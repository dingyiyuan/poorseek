"""The rule checker for 04-log-needle - deliberately independent of make_workspace.py.

It reads only the workspace files and re-derives the violating request ids from the rules
as they are documented there (limits.yaml + api_keys.txt + exemptions.txt). It shares no
constant and no code path with the generator: regenerate.py cross-checks the answer this
computes against the generator's intent, and writes hidden/answer.txt from HERE.

  python3 checker.py [workspace-dir]           the sorted violating ids, one per line
  python3 checker.py --counts [workspace-dir]  the per-tenant / `global` violation count table
  python3 checker.py --answer [workspace-dir]  the two-section ANSWER.txt body, as hidden/answer.txt
  python3 checker.py --table [workspace-dir]   which ids each rule class decides
  python3 checker.py --explain [workspace-dir] each violating id with its log line

Every skip in flag() and counted() is a togglable rule class, so --table can show that no
rule is redundant: turn one off, list the ids whose membership changes. The exemption rows
carry two toggles of their own - the `*` (tenant-wide) rows and the named-key rows - so the
near-identical pair can be tested one side at a time.
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELDS = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z) +\S+ +req=(\S+) +key=(\S+) +path=(\S+) +status=(\d+)")


def ms(stamp: str) -> int:
    """An ISO-8601 UTC stamp -> integer milliseconds since the epoch (exact, no floats)."""
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    return (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days * 86_400_000 \
        + dt.hour * 3_600_000 + dt.minute * 60_000 + dt.second * 1_000 + dt.microsecond // 1_000


def load(ws: Path):
    limits = {}
    for line in (ws / "limits.yaml").read_text().splitlines():
        m = re.match(r"\s+(\w+):\s*(\d+)", line)
        if m:
            limits[m.group(1)] = int(m.group(2))
    keys = {}
    for line in (ws / "api_keys.txt").read_text().splitlines():
        parts = line.split()
        if parts and not line.lstrip().startswith("#"):
            keys[parts[0]] = (parts[1], parts[2])
    exempt = []
    for line in (ws / "exemptions.txt").read_text().splitlines():
        parts = line.split()
        if parts and not line.lstrip().startswith("#"):
            exempt.append((parts[0], parts[1], ms(parts[2]), ms(parts[3])))
    ev = []
    for line in (ws / "service.log").read_text().splitlines():
        m = FIELDS.match(line)
        if not m:
            raise ValueError(f"unparsed log line: {line!r}")
        ts, rid, key, path, status = m.groups()
        ev.append({"t": ms(ts), "id": rid, "key": key, "path": path, "status": int(status)})
    ev.sort(key=lambda e: e["t"])
    return limits, keys, exempt, ev


def exempted(tenant, key, t, exempt, star_rows=True, key_rows=True, key_rows_wide=False):
    """Is the request of `key` (of `tenant`) at `t` inside a row of exemptions.txt?

    The two forms of a row differ only in the `key` column, so each gets its own toggle:
    `*` covers every key of the tenant (star_rows), a key name covers that key alone
    (key_rows) - and key_rows_wide models the misreading that the named-key row covers the
    whole tenant as well.
    """
    for x, spec, lo, hi in exempt:
        if x != tenant or not lo <= t < hi:
            continue
        if spec == "*" and star_rows:
            return True
        if spec == key and key_rows:
            return True
        if spec != "*" and spec != key and key_rows and key_rows_wide:
            return True
    return False


def counted(ev, keys, exempt, join=True, retired=True, star_rows=True, key_rows=True,
            key_rows_wide=False):
    """The COUNTED requests, per limits.yaml's COUNTED list: ops, /health, retired keys and
    requests inside an exemption row are out."""
    out = []
    for e in ev:
        tenant, key_status = keys[e["key"]]
        who = tenant if join else e["key"]
        if who == "ops" or e["path"] == "/health":
            continue
        if retired and key_status == "retired":
            continue
        if exempted(tenant, e["key"], e["t"], exempt, star_rows, key_rows, key_rows_wide):
            continue
        out.append((e["t"], e["id"], who))
    return out


def flag(ev, limits, keys, exempt, join=True, retired=True, star_rows=True, key_rows=True,
         key_rows_wide=False, per_tenant=True, global_rule=True):
    """The set of violating request ids: the (limit + 1)-th COUNTED request in a half-open
    window, checked per tenant and across all tenants."""
    window = limits["window_s"] * 1_000
    rows = counted(ev, keys, exempt, join, retired, star_rows, key_rows, key_rows_wide)
    out = set()
    if per_tenant:
        per = {}
        for t, rid, who in rows:
            per.setdefault(who, []).append((t, rid))
        n = limits["per_tenant"]
        for stream in per.values():
            stream.sort()
            for i in range(n, len(stream)):
                if stream[i][0] - stream[i - n][0] < window:
                    out.add(stream[i][1])
    if global_rule:
        xs = sorted(rows)
        g = limits["global_per_window"]
        for i in range(g, len(xs)):
            if xs[i][0] - xs[i - g][0] < window:
                out.add(xs[i][1])
    return out


def answer(ws: Path):
    limits, keys, exempt, ev = load(ws)
    return sorted(flag(ev, limits, keys, exempt))


def counts(ws: Path):
    """The per-row violation counts of that same answer: each id the PER-TENANT rule flags is
    counted on its own tenant's row, and the ids the GLOBAL rule alone decides go on `global`.
    Every violating id is on exactly one row, so the counts sum to len(answer)."""
    limits, keys, exempt, ev = load(ws)
    full = flag(ev, limits, keys, exempt)
    per_only = flag(ev, limits, keys, exempt, global_rule=False)
    global_only = full - per_only
    tenant_of = {e["id"]: keys[e["key"]][0] for e in ev}
    table = {}
    for rid in per_only:
        table[tenant_of[rid]] = table.get(tenant_of[rid], 0) + 1
    if global_only:
        table["global"] = len(global_only)
    return {k: table[k] for k in sorted(table)}


def body(ws: Path):
    """The ANSWER.txt the workspaces' own rules imply: the sorted id set, then the count table."""
    lines = ["# ids", *answer(ws), "# counts"]
    lines += [f"{name} {n}" for name, n in counts(ws).items()]
    return "\n".join(lines) + "\n"


def _line(ws: Path, rid: str) -> str:
    for line in (ws / "service.log").read_text().splitlines():
        if f"req={rid} " in line:
            return line
    return f"(no line for {rid})"


def table(ws: Path):
    limits, keys, exempt, ev = load(ws)
    full = flag(ev, limits, keys, exempt)
    classes = [("per-tenant rule off", dict(per_tenant=False)),
               ("global rule off", dict(global_rule=False)),
               ("exemption `*` rows off", dict(star_rows=False)),
               ("exemption key rows off", dict(key_rows=False)),
               ("key rows read tenant-wide", dict(key_rows_wide=True)),
               ("exemption file ignored", dict(star_rows=False, key_rows=False)),
               ("join ignored (key = tenant)", dict(join=False)),
               ("retired keys counted (join)", dict(retired=False))]
    print(f"answer ({len(full)} ids): {' '.join(sorted(full))}")
    for name, kw in classes:
        other = flag(ev, limits, keys, exempt, **kw)
        lost, gained = sorted(full - other), sorted(other - full)
        print(f"  {name:28s} removes {' '.join(lost) or '-'}  |  adds {' '.join(gained) or '-'}")
        for rid in lost + gained:
            print(f"      {rid}  {_line(ws, rid)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ws = Path(args[0]) if args else Path(__file__).resolve().parent / "workspace"
    if "--table" in sys.argv:
        table(ws)
    elif "--counts" in sys.argv:
        for name, n in counts(ws).items():
            print(name, n)
    elif "--answer" in sys.argv:
        print(body(ws), end="")
    elif "--explain" in sys.argv:
        for rid in answer(ws):
            print(rid, _line(ws, rid))
    else:
        print("\n".join(answer(ws)))
