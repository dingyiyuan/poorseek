#!/usr/bin/env python3
"""shard_check.py SHARD [...] - per-shard checks of a DeepSeek-V4.1-Flash safetensors checkpoint, runnable as each
shard lands (one backbone layer per shard: model-000{03..42} = layers 0..39).

1. Validation of every tensor against the config shapes: FP8 E4M3 weights with [ceil(N/32), ceil(K/32)] E8M0 scales,
   int8-packed FP4 routed experts with [N, K/32] E8M0 scales, no non-finite codes (E4M3 0x7f/0xff, E8M0 0xff).
2. Data-free sensitivity priors for the mixed-tier allocation (V41_PLAN.md): for each routed expert and projection,
   the relative weight-space error ||W - Q(W)|| / ||W|| of round-to-nearest at the three tiers, best scale per group
   from a small search:
     ternary  {-1,0,1} x s per 128      (PTQ1_0, 1.75 bpw)
     q2       {-2,-1,0,1} x s per 64    (Q2_0 / GSQ 2-bit, 2.25 bpw)
     q3       {-4..3} x s per 16        (stand-in for Q3_K, 3.44 bpw)
   on ROWS sampled rows per matrix (same rows for every expert).  These are priors only - the real allocation uses
   calibration-weighted errors and REAP saliency - but they rank experts by how "ternary-friendly" their weights are.
Output: <out>/layer_NN.json (validation + per-expert errors).  Env: THREADS=4, ROWS=256.
"""
import json, os, re, struct, sys, time
import numpy as np
import torch

THREADS = int(os.environ.get("THREADS", "4")); ROWS = int(os.environ.get("ROWS", "256"))
torch.set_num_threads(THREADS)
FP4_LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
DIM, INTER, NEXP = 5120, 2304, 384


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


class Shard:
    def __init__(self, path):
        self.path = path
        self.h, self.base = header(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        v = self.h[name]; a, b = v["data_offsets"]
        return self.mm[self.base + a: self.base + b], v["dtype"], v["shape"]

    def rows_u8(self, name, rows):
        buf, dt, shape = self.raw(name)
        row_bytes = int(np.prod(shape[1:])) * (1 if dt in ("I8", "U8", "F8_E4M3", "F8_E8M0") else 2 if dt in ("BF16", "F16") else 4)
        m = buf.reshape(shape[0], row_bytes)
        return torch.from_numpy(np.ascontiguousarray(m[rows]))


def e8m0(u8):  # E8M0 -> fp32 (255 = NaN)
    return torch.ldexp(torch.ones(u8.shape), u8.to(torch.int32) - 127)


def fp4_rows(sh, wname, rows):
    codes = sh.rows_u8(wname, rows)                       # [r, K/2]
    sc = e8m0(sh.rows_u8(wname.replace(".weight", ".scale"), rows))   # [r, K/32]
    w = torch.empty(codes.shape[0], codes.shape[1] * 2)
    w[:, 0::2] = FP4_LUT[(codes & 15).long()]; w[:, 1::2] = FP4_LUT[(codes >> 4).long()]
    return (w.unflatten(1, (-1, 32)) * sc.unsqueeze(-1)).flatten(1)


def best_rtn_err(w, group, lo, hi, cands):
    """min over candidate scales (fractions of the group amax / hi) of the squared RTN error, per group; returns total
    squared error.  Grid: integers lo..hi times s."""
    g = w.unflatten(1, (-1, group))                       # [r, G, group]
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best = None
    for c in cands:
        s = amax * c / max(abs(lo), abs(hi))
        q = torch.clamp(torch.round(g / s), lo, hi) * s
        e = (g - q).square().sum(-1)
        best = e if best is None else torch.minimum(best, e)
    return float(best.sum())


TERN_C = [0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0]
Q2_C = [0.55, 0.65, 0.75, 0.85, 1.0]      # {-2,-1,0,1}: hi=1, lo=-2 -> scale over 2
Q3_C = [0.7, 0.8, 0.9, 1.0]


def tier_errors(w):
    tot = float(w.square().sum()) + 1e-30
    return {"ternary": (best_rtn_err(w, 128, -1, 1, TERN_C) / tot) ** 0.5,
            "q2": (best_rtn_err(w, 64, -2, 1, Q2_C) / tot) ** 0.5,
            "q3": (best_rtn_err(w, 16, -4, 3, Q3_C) / tot) ** 0.5,
            "rms": (tot / w.numel()) ** 0.5}


def validate(sh):
    issues, counts = [], {}
    for name, v in sh.h.items():
        dt, shape = v["dtype"], v["shape"]
        counts[dt] = counts.get(dt, 0) + 1
        if dt == "F8_E8M0":
            buf, _, _ = sh.raw(name)
            if (np.asarray(buf) == 255).any(): issues.append(f"{name}: NaN E8M0 scale")
            w = name.replace(".scale", ".weight")
            if w in sh.h:
                ws, wd = sh.h[w]["shape"], sh.h[w]["dtype"]
                exp = [ws[0], ws[1] * 2 // 32] if wd == "I8" else [-(-ws[0] // 32), -(-ws[1] // 32)]
                if shape != exp: issues.append(f"{name}: scale shape {shape}, expected {exp} for {wd} {ws}")
        elif dt == "F8_E4M3":
            buf, _, _ = sh.raw(name)
            if ((np.asarray(buf) & 0x7f) == 0x7f).any(): issues.append(f"{name}: NaN E4M3 code")
        elif dt in ("F32", "BF16"):
            buf, _, _ = sh.raw(name)
            a = np.frombuffer(buf, dtype=np.float32) if dt == "F32" else \
                (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(a).all(): issues.append(f"{name}: non-finite {dt}")
        m = re.search(r"experts\.(\d+)\.w([123])\.weight$", name)
        if m:
            exp_shape = [INTER, DIM // 2] if m.group(2) in "13" else [DIM, INTER // 2]
            if shape != exp_shape or dt != "I8": issues.append(f"{name}: {dt} {shape}, expected I8 {exp_shape}")
    return issues, counts


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), "v41-shard-check")
    os.makedirs(out, exist_ok=True)
    for path in args:
        t0 = time.time(); sh = Shard(path)
        layers = sorted({int(m.group(1)) for k in sh.h for m in [re.match(r"layers\.(\d+)\.", k)] if m})
        issues, counts = validate(sh)
        rep = {"shard": os.path.basename(path), "layers": layers, "tensors": len(sh.h), "dtypes": counts, "issues": issues}
        experts = sorted({int(m.group(1)) for k in sh.h for m in [re.search(r"experts\.(\d+)\.w1\.weight$", k)] if m})
        if experts and len(layers) == 1:
            L = layers[0]; g = torch.Generator().manual_seed(L)
            r_gu = torch.randperm(INTER, generator=g)[:ROWS].sort().values.numpy()
            r_d = torch.randperm(DIM, generator=g)[:ROWS].sort().values.numpy()
            per = {}
            for e in experts:
                p = f"layers.{L}.ffn.experts.{e}"
                per[e] = {wn: tier_errors(fp4_rows(sh, f"{p}.{wn}.weight", r_gu if wn != "w2" else r_d)) for wn in ("w1", "w3", "w2")}
            rep["experts"] = per
            T = np.array([[per[e][w]["ternary"] for w in ("w1", "w3", "w2")] for e in experts])
            Q2 = np.array([[per[e][w]["q2"] for w in ("w1", "w3", "w2")] for e in experts])
            rep["summary"] = {"ternary_rel_err_median": float(np.median(T)), "q2_rel_err_median": float(np.median(Q2)),
                              "ternary_over_q2_p10_p50_p90": [float(x) for x in np.percentile(T / Q2, [10, 50, 90])]}
        rep["seconds"] = round(time.time() - t0, 1)
        with open(os.path.join(out, f"{os.path.basename(path).split('-of-')[0]}.json"), "w") as f:
            json.dump(rep, f)
        print(f"{rep['shard']}: layers {layers}, {rep['tensors']} tensors, {len(issues)} issues"
              + (f", ternary/q2 err median {rep['summary']['ternary_rel_err_median']:.3f}/{rep['summary']['q2_rel_err_median']:.3f}"
                 if "summary" in rep else "") + f" ({rep['seconds']} s)")
        for i in issues[:10]: print("   ", i)


if __name__ == "__main__":
    main()
