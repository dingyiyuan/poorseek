#!/usr/bin/env python3
"""v41_hsweep.py - the Hadamard-rotation question, answered for every layer before the v2 requant (Mal 2026-10-17:
"do the proper testing on Hadamard ... don't want to requant more than once").

A GSQ student chain through DeepSeek's model.py (v41_repair's GsqLayer hook: the routed experts are the GSQ v1 file's
own bytes, propagated), and at EVERY layer, for up to --experts experts spread over the layer (enough fit and held-out
tokens), the same expert quantized with this file's method (static diag(H)-weighted scales + GPTQ damp 1.0, v41_quant
.quant) at each tier, plain and on a rotated basis:
  r128    block-diagonal randomized Hadamard, 128-blocks, on the gate/up input (5120) AND the down input (2304)
  r1024   1024-blocks on the gate/up input (5120 = 5 x 1024), 256-blocks on the down input (2304 = 9 x 256)
  gu128   128-blocks on the gate/up input only (down unrotated) - the cheap engine change: one rotated x per layer
          serves every routed expert's gate/up
W' = W R, H' = R^T H R, effective weight Q' R^T (orthonormal R: x W^T = (xR)(WR)^T).  Score: held-out output error
sum ||f(x; W) - f(x; Q)||^2 per tier and variant, and the expert outputs' energy -> per-layer ratios rot / plain.

  v41_hsweep.py layer L --rwork DIR [--experts 16]     (prep + init: v41_repair.py prep / init on the same DIR)
"""
import argparse, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v41_calib as C  # noqa: E402
import v41_quant as VQ  # noqa: E402
import v41_repair as R  # noqa: E402

TIERS = ("ptq1", "q2_0", "q3_k")


def hadamard(n, block, dev, seed):
    Hb = torch.ones(1, 1)
    while Hb.shape[0] < block:
        Hb = torch.cat([torch.cat([Hb, Hb], 1), torch.cat([Hb, -Hb], 1)], 0)
    Hb = Hb / block ** 0.5
    g = torch.Generator().manual_seed(seed)
    sg = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
    assert n % block == 0
    return (torch.block_diag(*[Hb] * (n // block)) * sg[:, None]).to(dev)


class Sweep:
    def __init__(self, a, L):
        self.a, self.L, self.gsq = a, L, R.GsqLayer(L, a.device)
        dev = a.device
        self.V = {"r128": (hadamard(C.D, 128, dev, 0), hadamard(R.FF, 128, dev, 1)),
                  "r1024": (hadamard(C.D, 1024, dev, 2), hadamard(R.FF, 256, dev, 3)),
                  "gu128": (hadamard(C.D, 128, dev, 0), None)}
        self.acc = {"n": 0, "energy": 0.0}
        self.stride = max(1, C.NE // (2 * a.experts))

    def expert(self, L, ex, exp, xe, ge, fit, lim, out_teacher):
        W1q, W3q, W2q = self.gsq.weights(ex)
        out = torch.empty_like(out_teacher)
        for c in range(0, len(xe), C.ROWS):
            out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], W1q, W3q, W2q, ge[c:c + C.ROWS], lim).to(out.dtype)
        hv = ~fit
        if (self.acc["n"] < self.a.experts and ex % self.stride == L % self.stride
                and int(fit.sum()) >= 64 and int(hv.sum()) >= 16):
            self.measure(exp, xe[fit], ge[fit], xe[hv], ge[hv], out_teacher[hv].float(), lim)
        return out, {}

    def measure(self, exp, xf, gf, xv, gv, Yv, lim):
        A = self.acc
        A["n"] += 1
        A["energy"] += float(Yv.square().sum())
        e = lambda Y: float((Yv - Y.float()).square().sum())
        W1, W3, W2 = (C.K.dequant_fp4_weight(m.weight, m.scale) for m in (exp.w1, exp.w3, exp.w2))
        H1, H2 = C.act_hessians(xf, W1, W3, gf, lim)
        for tier in TIERS:
            (a1, _), (a3, _) = VQ.quant([W1, W3], H1, tier, 1.0, True)
            ((a2, _),) = VQ.quant([W2], H2, tier, 1.0, True)
            A[f"plain_{tier}"] = A.get(f"plain_{tier}", 0.0) + e(C.expert_f(xv, a1, a3, a2, gv, lim))
            (o1, _), (o3, _) = VQ.quant([W1, W3], H1, tier, 1.0, True, act_order=True)
            ((o2, _),) = VQ.quant([W2], H2, tier, 1.0, True, act_order=True)
            A[f"actord_{tier}"] = A.get(f"actord_{tier}", 0.0) + e(C.expert_f(xv, o1, o3, o2, gv, lim))
            for name, (R1, R2) in self.V.items():
                (b1, _), (b3, _) = VQ.quant([W1 @ R1, W3 @ R1], R1.T @ H1 @ R1, tier, 1.0, True)
                b1, b3 = b1 @ R1.T, b3 @ R1.T
                if R2 is not None:
                    ((b2, _),) = VQ.quant([W2 @ R2], R2.T @ H2 @ R2, tier, 1.0, True)
                    b2 = b2 @ R2.T
                else:
                    b2 = a2
                A[f"{name}_{tier}"] = A.get(f"{name}_{tier}", 0.0) + e(C.expert_f(xv, b1, b3, b2, gv, lim))

    def finish(self, L, blk, stats):
        A = self.acc
        if not A["n"]:
            C.log(f"hsweep L{L}: no expert measured")
            return
        rel = {k: v / A["energy"] for k, v in A.items() if k not in ("n", "energy")}
        ratio = {f"{v}_{t}": rel[f"{v}_{t}"] / max(rel[f"plain_{t}"], 1e-30) for v in list(self.V) + ["actord"] for t in TIERS}
        json.dump({"layer": L, "experts": A["n"], "rel_err": rel, "ratio_vs_plain": ratio},
                  open(C.wpath("stats", f"hsweep_L{L:02d}.json"), "w"), indent=1)
        C.log(f"hsweep L{L} ({A['n']} experts): " + "  ".join(f"{k} {v:.3f}" for k, v in ratio.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["layer"]); ap.add_argument("layer", type=int)
    ap.add_argument("--rwork", required=True); ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--gc", action="store_true")
    a = ap.parse_args()
    C.WORK = a.rwork
    os.makedirs(C.wpath("stats"), exist_ok=True)
    s = Sweep(a, a.layer)
    a.expert_hook, a.moe_done_hook = s.expert, s.finish
    a.no_gptq, a.screen_only, a.damps, a.expert_stride, a.min_fit, a.gptq_experts = True, None, [1.0], 1, 32, 0
    C.cmd_layer(a)


if __name__ == "__main__":
    main()
