"""check_v41_gguf.py - checks of a written V4.1 GGUF that do not go through v41_gguf's plan (its --verify does):

  1. layout: every tensor's bytes (from its type and shape) fit in the gap to the next tensor / the file end, offsets
     aligned to general.alignment;
  2. experts: for sampled experts, the slot is recomputed from the file's own I32 type maps (rank among the experts
     of that type) and the bytes compared with the quantize-pass directory, read at independently computed offsets;
  3. dense: Q8_0 / F16 / F32 tensors decoded and compared with the source's dequantized FP8 / BF16 values;
  4. Engram: sampled rows == the source FP8 row + its 8 E8M0 scales.

    python3 tools/v41/tests/check_v41_gguf.py SHARD1.gguf [--experts QDIR] [--fp4 N] [--layers 0-1]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import v41_formats as F  # noqa: E402
import v41_gguf as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--experts", help="QDIR the experts came from (byte comparison)")
    ap.add_argument("--fp4", type=int, default=0, metavar="N",
                    help="decode N sampled experts per layer and compare with the FP4 source (any expert source)")
    ap.add_argument("--layers", default="0-1")
    a = ap.parse_args()
    bad = []
    files = [a.gguf]
    p2 = a.gguf.replace("-00001-of-00002", "-00002-of-00002")
    if p2 != a.gguf and os.path.exists(p2):
        files.append(p2)
    T = {}
    for path in files:
        kv, ts, start = W.read_gguf(path)
        al = kv["general.alignment"]
        size = os.path.getsize(path)
        ts = sorted(ts, key=lambda x: x[3])
        for i, (name, shape, ty, off) in enumerate(ts):
            n = F.nbytes(ty, shape) if ty in F.BLOCK else None
            nxt = ts[i + 1][3] if i + 1 < len(ts) else size - start
            if n is None or off % al or off + n > nxt:
                bad.append(f"layout {name}: off {off} bytes {n} next {nxt}")
            T[name] = (shape, ty, start + off, path)
    print(f"1. layout: {len(T)} tensors in {len(files)} file(s)")
    src = W.Source()
    rng = np.random.default_rng(5)

    def read(name, off, n):
        shape, ty, base, path = T[name]
        with open(path, "rb") as f:
            f.seek(base + off)
            return f.read(n)

    x, y = map(int, a.layers.split("-"))
    nexp = 0
    for L in (range(x, y + 1) if a.experts else []):
        j = json.load(open(os.path.join(a.experts, f"L{L:02d}.json")))
        gu_f = T.get(f"blk.{L}.ffn_exps_gu_type")
        dn_f = T.get(f"blk.{L}.ffn_exps_down_type")
        gu = np.frombuffer(read(f"blk.{L}.ffn_exps_gu_type", 0, 384 * 4), np.int32) if gu_f else None
        dn = np.frombuffer(read(f"blk.{L}.ffn_exps_down_type", 0, 384 * 4), np.int32) if dn_f else None
        if gu is not None and list(gu) != j["gu_type"] or dn is not None and list(dn) != j["d_type"]:
            bad.append(f"L{L}: type maps differ from the quantize pass")
        sizes = [(F.nbytes(j["gu_type"][e], (5120, 2304)), F.nbytes(j["d_type"][e], (2304, 5120))) for e in range(384)]
        base = np.cumsum([0] + [2 * g + d for g, d in sizes])
        with open(os.path.join(a.experts, f"L{L:02d}.bin"), "rb") as qf:
            for e in sorted({0, 383, *rng.integers(384, size=6).tolist()}):
                for role in ("gate", "up", "down"):
                    tmap = dn if role == "down" else gu
                    t = (j["d_type"] if role == "down" else j["gu_type"])[e]
                    if tmap is None:
                        name, slot = f"blk.{L}.ffn_{role}_exps.weight", e
                    else:
                        name = f"blk.{L}.ffn_{role}_exps.{F.TYPE_NAME[t].lower()}.weight"
                        slot = int(np.sum(tmap[:e] == t))
                    g, d = sizes[e]
                    per = d if role == "down" else g
                    qf.seek(int(base[e]) + {"gate": 0, "up": g, "down": 2 * g}[role])
                    if T[name][1] != t or read(name, slot * per, per) != qf.read(per):
                        bad.append(f"L{L} expert {e} {role}: bytes differ (tensor {name} slot {slot})")
                    nexp += 1
    print(f"2. experts: {nexp} expert matrices compared with the QDIR")
    # 2b. decoded experts vs the FP4 source: weight-space rel error must look like a quantization of THAT expert
    #     (RTN/GPTQ: ~0.15 q3_k, ~0.35 q2_0, ~0.48 ptq1; a hole or a wrong slot decodes to ~1 or more)
    LIM = {F.Q3_K: 0.30, F.Q2_0: 0.50, F.PTQ1_0: 0.65}
    for L in (range(x, y + 1) if a.fp4 else []):
        tmaps = {}
        for key in ("gu", "down"):
            nm = f"blk.{L}.ffn_exps_{key}_type"
            tmaps[key] = np.frombuffer(read(nm, 0, 384 * 4), np.int32) if nm in T else None
        worst = {}
        for e in sorted({0, 383, *rng.integers(384, size=a.fp4).tolist()})[: a.fp4 + 2]:
            for role, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
                tmap = tmaps["down" if role == "down" else "gu"]
                if tmap is None:
                    name, slot = f"blk.{L}.ffn_{role}_exps.weight", e
                    t = T[name][1]
                else:
                    t = int(tmap[e])
                    name = f"blk.{L}.ffn_{role}_exps.{F.TYPE_NAME[t].lower()}.weight"
                    slot = int(np.sum(tmap[:e] == t))
                shape = T[name][0]
                per = F.nbytes(t, shape[:2])
                got = F.decode(t, read(name, slot * per, per), shape[0] * shape[1])
                want = src.f32(f"layers.{L}.ffn.experts.{e}.{w}.weight").reshape(-1)
                rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
                worst[F.TYPE_NAME[t]] = max(worst.get(F.TYPE_NAME[t], 0), rel)
                if not rel < LIM[t]:
                    bad.append(f"L{L} expert {e} {role} ({F.TYPE_NAME[t]}): rel {rel:.3f} vs FP4 source")
        print(f"2b. L{L} experts vs FP4 source, worst rel per type: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(worst.items())))
    dense = [("output_norm.weight", "norm.weight"), ("token_embd.weight", "embed.weight"),
             ("blk.0.attn_q_a.weight", "layers.0.attn.wq_a.weight"), ("blk.0.attn_output_b.weight", "layers.0.attn.wo_b.weight"),
             ("blk.0.ffn_down_shexp.weight", "layers.0.ffn.shared_experts.w2.weight"),
             ("blk.0.hc_attn_fn.weight", "layers.0.hc_attn_fn"), ("blk.0.ffn_gate_inp.weight", "layers.0.ffn.gate.weight"),
             ("blk.1.engram_kv.weight", "layers.1.engram.wkv.weight")]
    for name, sname in dense:
        if name not in T:
            continue
        shape, ty, _, _ = T[name]
        rows = min(shape[1] if len(shape) > 1 else 1, 256)
        n = shape[0] * rows
        got = F.decode(ty, read(name, 0, F.row_bytes(ty, shape[0]) * rows), n)
        want = src.f32(sname, 0, rows).reshape(-1) if len(shape) > 1 else src.f32(sname).reshape(-1)
        rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-30)
        lim = {F.F32: 0, F.F16: 1e-3, F.Q8_0: 1e-2}[ty]
        print(f"3. dense {name:<34} {F.TYPE_NAME[ty]:>5} vs source: rel {rel:.2e}")
        if not rel <= lim:
            bad.append(f"{name}: rel {rel:.3g} > {lim}")
    for name in [k for k in T if k.endswith("engram_embd.weight")]:
        shape, ty, _, _ = T[name]
        L = int(name.split(".")[1])
        sn = f"layers.{L}.engram.embed"
        for r in sorted({0, shape[1] - 1, *rng.integers(shape[1], size=32).tolist()}):
            if read(name, r * 264, 264) != src.read(sn + ".weight", r * 256, 256) + src.read(sn + ".scale", r * 8, 8):
                bad.append(f"{name} row {r} differs")
        print(f"4. engram {name}: {shape[1]} rows, 34 sampled")
    print(f"{'FAIL' if bad else 'PASS'} ({len(bad)} problems)")
    for b in bad[:20]:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
