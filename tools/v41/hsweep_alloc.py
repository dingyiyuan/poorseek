#!/usr/bin/env python3
"""hsweep_alloc.py - what the Hadamard rotation would buy in the ALLOCATION (bytes at equal predicted quality).

Scales every expert's screened per-tier error (v41-calib stats/LNN.json: e_*/egu_*/edn_*) by its layer's measured
rotated/plain ratio for that tier (v41_hsweep stats/hsweep_LNN.json, one variant), writes the scaled stats to a scratch
work dir, and runs v41_alloc on it at a list of budgets - next to the same allocation on the plain stats.  The answer
is the budget at which the rotated proxy equals the plain v2 proxy (the GiB the rotation saves at equal quality), and
the proxy gain at the v2 budget.  Layers with no measurement use the median ratio of their tier.

  hsweep_alloc.py --hsweep /media/Crucial1TB/v41-hsweep --variant r128 [--budgets 130.38,128.38,...]
"""
import argparse, glob, json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALIB = "/media/Crucial1TB/v41-calib"
TIERS = ("ptq1", "q2_0", "q3_k")
CAPS = "20-31,33-35=1.0"


def proxy(work, budget):
    out = subprocess.run([sys.executable, os.path.join(HERE, "v41_alloc.py"), "--work", work, "--layers", "0-39",
                          "--per-proj", "--ternary-cap-at", CAPS, "--budget-gib", str(budget), "--out", os.devnull],
                         capture_output=True, text=True).stdout
    return float(re.search(r"mixed +([0-9.]+)", out).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsweep", required=True); ap.add_argument("--variant", default="r128")
    ap.add_argument("--budgets", default="130.38,129.38,128.38,127.38,126.38,125.38,124.38")
    ap.add_argument("--scratch", default="/tmp/claude-1000/hsweep-alloc")
    a = ap.parse_args()
    ratio = {}
    for p in glob.glob(os.path.join(a.hsweep, "stats", "hsweep_L*.json")):
        j = json.load(open(p))
        ratio[j["layer"]] = {t: j["ratio_vs_plain"][f"{a.variant}_{t}"] for t in TIERS}
    med = {t: statistics.median(r[t] for r in ratio.values()) for t in TIERS}
    print(f"{len(ratio)} layers measured; median ratio {a.variant}: " + "  ".join(f"{t} {med[t]:.3f}" for t in TIERS))
    work = os.path.join(a.scratch, a.variant)
    os.makedirs(os.path.join(work, "stats"), exist_ok=True)
    for L in range(40):
        s = json.load(open(os.path.join(CALIB, "stats", f"L{L:02d}.json")))
        r = ratio.get(L, med)
        for e in s["experts"].values():
            for k in list(e):
                m = re.match(r"(e|egu|edn)_(ptq1|q2_0|q3_k)$", k)
                if m:
                    e[k] *= r[m.group(2)]
        json.dump(s, open(os.path.join(work, "stats", f"L{L:02d}.json"), "w"))
    base = proxy(CALIB, 130.38)
    print(f"plain stats @130.38 GiB (the v2 map): proxy {base:.4f}")
    rows = []
    for b in map(float, a.budgets.split(",")):
        rows.append((b, proxy(work, b)))
        print(f"{a.variant} @{b:.2f} GiB: proxy {rows[-1][1]:.4f}  ({(rows[-1][1] / base - 1) * 100:+.1f} % vs v2 plain)")
    eq = next((b for b, p in sorted(rows) if p <= base), None)
    print(f"=> {a.variant}: smallest budget at <= the v2 proxy: {eq} GiB "
          f"({'%.1f' % (130.38 - eq) if eq else '?'} GiB saved vs v2; gain at 130.38: {(rows[0][1] / base - 1) * 100:+.1f} %)")


if __name__ == "__main__":
    main()
