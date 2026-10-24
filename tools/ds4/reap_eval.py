#!/usr/bin/env python3
"""Per-domain perplexity of a concatenated eval prompt from DS4_PPL_DUMP files (one per config).
NLL row i scores prompt token i+1, so a segment [a, b) of the prompt is rows [a-1, b-1) (row -1 does not exist).
  reap_eval.py eval.segments base=nll-base.f32 all25=nll-all25.f32 ...
"""
import math
import sys
import numpy as np

segs = [(n, int(a), int(b)) for n, a, b in (l.split() for l in open(sys.argv[1]) if l.strip())]
runs = [(kv.split("=", 1)[0], np.fromfile(kv.split("=", 1)[1], dtype=np.float32).astype(np.float64)) for kv in sys.argv[2:]]
base = runs[0][1]
print("%-8s" % "domain" + "".join("%18s" % n for n, _ in runs))
for n, a, b in segs + [("ALL", segs[0][1], segs[-1][2])]:
    lo, hi = max(a - 1, 0), b - 1
    cells = []
    for name, x in runs:
        p = math.exp(x[lo:hi].mean())
        pb = math.exp(base[lo:hi].mean())
        cells.append("%9.3f" % p + ("" if x is base else " (%+5.1f%%)" % (100 * (p / pb - 1))).ljust(9))
    print("%-8s" % n + "".join("%18s" % c for c in cells))
