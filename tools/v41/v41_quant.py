"""v41_quant.py - the quantize pass: every routed expert of every layer at its chosen tier, written as the GGUF block
bytes v41_gguf.py --experts reads.  A second layer-by-layer chain through DeepSeek's model.py (v41_calib's
cmd_layer, with its expert hook), in its own work dir.

    v41_quant.py init  --qwork DIR                       links tokens/meta/hashes from the calibration work dir,
                                                         writes h/L00 + pre/L00 (CPU)
    v41_quant.py layer L --qwork DIR (--map alloc.json | --tier q2_0) [--propagate quant|teacher] [--gc]
    v41_quant.py fill  --gguf SHARD1 --qdir DIR/q            QDIR -> an --experts-in-place file
    v41_quant.py head  --qwork DIR [--layer 40]          per-domain ppl / top-1 of the QUANTIZED chain (v41_calib head)

Per expert: GPTQ (v41_gptq, static diag(H)-weighted scales, damp 1.0) at its tier - gate+up stacked on the input
Hessian, down on the Hessian of the clamped SwiGLU x routing weight - fitted on the FIT sequences only, so the held-out
sequences stay held out: the head gate's eval split is then an honest comparison of a mixed map against the all-Q2_0
control (run the same chain with --tier q2_0 in another qwork).  Experts with fewer than --min-fit fit tokens use
an identity Hessian (static scales, no error feedback); experts no token reached are written the same way.

--propagate quant (default): the quantized experts' outputs feed layer L+1 (sequential GPTQ: later layers are fitted
on the inputs the quantized model will really see); teacher: the FP4 outputs, as the screening run.
Dense weights (attention, shared expert) stay at the source's FP8 here; the GGUF stores them as Q8_0 (rel 5e-3).

Output (Sink): --gguf SHARD1 writes each expert in place into the final file (v41_gguf.py --experts-in-place made it,
with the same map); --no-write writes none (the control); default DIR/q/LNN.bin (each expert's gate, up, down rows
in expert order; v41_gguf --experts reads it).  Always DIR/q/LNN.json ({"gu_type": [...], "d_type": [...], "experts": {e: stats}}), written after the layer's MoE - so
a layer is done when its json exists (the chain resumes like the calibration one).  GSQ refinement: TODO (plugs in
where GPTQ is called; same output).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v41_calib as C  # noqa: E402  (sets up model.py / kernel.py)
import v41_formats as F  # noqa: E402
import v41_gptq as G  # noqa: E402
import v41_gptq_graph as GG  # noqa: E402  (bit-identical CUDA-graph gptq, 8x faster)
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

K = C.K
CALIB = C.WORK
WRITE_THREADS = int(os.environ.get("V41_WRITE_THREADS", "3"))
FF = 2304   # moe_intermediate_size (checked against each expert's w2)
REFINE = os.environ.get("V41_REFINE") == "1"
ACT_ORDER = os.environ.get("V41_ACT_ORDER") == "1"
REFINE_STEPS = int(os.environ.get("V41_REFINE_STEPS", "200"))
if REFINE:
    import v41_repair as R  # noqa: E402


def set_work(qwork):
    C.WORK = qwork
    os.makedirs(os.path.join(qwork, "q"), exist_ok=True)


def tiers_for(a, L):
    """(gu tiers [384], down tiers [384]) for layer L from --map (v41_alloc --out) or the uniform --tier."""
    gu, dn = [a.tier] * C.NE, [a.tier] * C.NE
    if a.map:
        m = json.load(open(a.map))["map"].get(str(L))
        if m is None:
            raise SystemExit(f"layer {L} is not in {a.map}")
        for e, pr in m.items():
            for k, t in pr.items():
                if k in ("all", "gu"):
                    gu[int(e)] = t
                if k in ("all", "dn"):
                    dn[int(e)] = t
    return gu, dn


def quant(Ws, H, tier, damp, do_gptq, act_order=None):
    """GPTQ of the stacked Ws (sharing H) at `tier`: [(Q, S)] per W, S in the stored format (v41_formats input).
    act_order (default: V41_ACT_ORDER=1): GPTQ visits the columns by descending diag(H) - the static per-group scales
    were chosen first, so they just travel with their columns; only the integers change (same format)."""
    if act_order is None:
        act_order = ACT_ORDER
    Cc = H.shape[0]
    d = torch.diag(H).clamp_min(0) + 1e-8
    lo, hi, _, _ = G.TIERS[tier]
    S = [G.search_scales(W, d, tier) for W in Ws]
    Wst = torch.cat(Ws)
    Sst = torch.cat([G.expand_scales(s, tier, Cc) for s in S])
    lov = torch.full((Wst.shape[0], 1), float(lo), device=Wst.device)
    hiv = torch.full((Wst.shape[0], 1), float(hi), device=Wst.device)
    if do_gptq and act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        inv = torch.argsort(perm)
        Hinv, dead = G.hinv_chol(H[perm][:, perm], damp)
        Q = GG.gptq(Wst[:, perm].contiguous(), Sst[:, perm].contiguous(), lov, hiv, Hinv, dead)[:, inv].contiguous()
        if REFINE:
            Q = R.refine(Wst, Q, Sst, lo, hi, H, steps=REFINE_STEPS)
    elif do_gptq:
        Hinv, dead = G.hinv_chol(H, damp)
        Q = GG.gptq(Wst, Sst, lov, hiv, Hinv, dead)
        if REFINE:   # V41_REFINE=1: GSQ/AdaRound-style rounding refinement of the GPTQ codes (v41_repair.refine)
            Q = R.refine(Wst, Q, Sst, lo, hi, H, steps=REFINE_STEPS)
    else:
        Q = G.rtn(Wst, Sst, lov, hiv)
    out, r0 = [], 0
    for W, s in zip(Ws, S):
        out.append((Q[r0:r0 + W.shape[0]], s))
        r0 += W.shape[0]
    return out


class Sink:
    """Where a layer's expert bytes go: `qdir` (DIR/q/LNN.bin, expert order - v41_gguf --experts), `gguf` (in place,
    at the offsets of v41_gguf --experts-in-place's <shard>.experts.json; the types there must be this layer's), or
    `none` (--no-write: the all-Q2_0 control only needs its head-gate numbers)."""

    def __init__(self, a, L, gt, dt, size):
        self.size = size
        self.mode = "none" if a.no_write else ("gguf" if a.gguf else "qdir")
        if self.mode == "qdir":
            self.off = np.cumsum([0] + [2 * g + d for g, d in size]).tolist()
            self.fd = os.open(C.wpath("q", f"L{L:02d}.bin"), os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(self.fd, self.off[-1])
        elif self.mode == "gguf":
            lay = json.load(open(a.gguf + ".experts.json"))["layers"].get(str(L))
            if lay is None:
                raise SystemExit(f"{a.gguf}: layer {L} is not in the file")
            if lay["gu_type"] != list(gt) or lay["d_type"] != list(dt):
                raise SystemExit(f"{a.gguf}: layer {L} tiers differ from this run's map")
            self.goff = {int(e): o for e, o in lay["off"].items()}
            self.fd = os.open(a.gguf, os.O_RDWR)

    def put(self, ex, gate, up, down):
        g, d = self.size[ex]
        assert (len(gate), len(up), len(down)) == (g, g, d), (ex, len(gate), len(up), len(down), g, d)
        if self.mode == "qdir":
            os.pwrite(self.fd, gate + up + down, self.off[ex])
        else:
            o = self.goff[ex]
            for role, b in (("gate", gate), ("up", up), ("down", down)):
                os.pwrite(self.fd, b, o[role])

    def close(self):
        if self.mode != "none":
            os.fsync(self.fd)
            os.close(self.fd)


class Layer:
    """One layer's output file and the per-expert work the hooks do."""

    def __init__(self, a, L):
        self.a, self.L = a, L
        self.gu, self.dn = tiers_for(a, L)
        self.gt = [F.TIER_TYPE[t] for t in self.gu]
        self.dt = [F.TIER_TYPE[t] for t in self.dn]
        self.size = [(F.nbytes(self.gt[e], (C.D, FF)), F.nbytes(self.dt[e], (FF, C.D))) for e in range(C.NE)]
        self.sink = Sink(a, L, self.gt, self.dt, self.size)
        self.done, self.info = set(), {}
        # encode + pwrite on worker threads while the GPU does the next expert (numpy, same bytes; distinct offsets)
        self.pool, self.pending = ThreadPoolExecutor(max_workers=max(WRITE_THREADS, 1)), []

    def _encode_put(self, ex, mats):
        b = [F.encode_tier(q, s, t).tobytes() for q, s, t in mats]
        self.sink.put(ex, *b)

    def write(self, ex, q1, q3, q2):
        if self.sink.mode != "none":
            mats = [(q.cpu().numpy(), s.cpu().numpy(), t)
                    for (q, s), t in ((q1, self.gu[ex]), (q3, self.gu[ex]), (q2, self.dn[ex]))]
            if WRITE_THREADS == 0:                 # V41_WRITE_THREADS=0: in-line, as before
                self._encode_put(ex, mats)
                self.done.add(ex)
                return
            while len(self.pending) >= 6:          # bound the host copies in flight (~115 MB each)
                self.pending.pop(0).result()       # re-raises a worker's error here
            self.pending.append(self.pool.submit(self._encode_put, ex, mats))
        self.done.add(ex)

    def drain(self):
        for f in self.pending:
            f.result()
        self.pending = []
        self.pool.shutdown(wait=True)

    def weights(self, exp):
        return (K.dequant_fp4_weight(exp.w1.weight, exp.w1.scale), K.dequant_fp4_weight(exp.w3.weight, exp.w3.scale),
                K.dequant_fp4_weight(exp.w2.weight, exp.w2.scale))

    def expert(self, L, ex, exp, xe, ge, fit, lim, out_teacher):
        a = self.a
        W1, W3, W2 = self.weights(exp)
        assert W2.shape == (C.D, FF), W2.shape
        xf, gf = xe[fit], ge[fit]          # bf16; C.expert_f / C.act_hessians convert per chunk
        n_fit = int(fit.sum())
        do = n_fit >= a.min_fit and ex < a.gptq_experts
        if do:
            H1, H2 = C.act_hessians(xf, W1, W3, gf, lim)
        else:
            H1 = torch.eye(C.D, device=xe.device)
            H2 = torch.eye(W2.shape[1], device=xe.device)
        q1, q3 = quant([W1, W3], H1, self.gu[ex], a.damp, do)
        (q2,) = quant([W2], H2, self.dn[ex], a.damp, do)
        self.write(ex, q1, q3, q2)
        st = {"gu": self.gu[ex], "dn": self.dn[ex], "n_fit": n_fit, "gptq": do}
        xv, gv = xe[~fit], ge[~fit]
        if len(xv):   # held-out output error at the chosen tiers (the fit used none of these tokens)
            Y = C.expert_f(xv, W1, W3, W2, gv, lim)
            st["q_err"] = C.sq_err(xv, q1[0], q3[0], q2[0], gv, lim, Y)
            st["y_energy"] = float(Y.square().sum())
        if a.propagate == "quant":
            # chunk by chunk straight into the output dtype: a hot expert's whole fp32 output (L7: 1.88 GiB) OOM'd
            out = torch.empty((len(xe), C.D), dtype=out_teacher.dtype, device=xe.device)
            for c in range(0, len(xe), C.ROWS):
                out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], q1[0], q3[0], q2[0], ge[c:c + C.ROWS], lim).to(out.dtype)
        else:
            out = out_teacher
        return out, st

    def finish(self, L, blk, stats):
        for ex in range(C.NE):   # experts no token reached: static scales, identity importance
            if ex not in self.done:
                W1, W3, W2 = self.weights(blk.ffn.experts[ex])
                q1, q3 = quant([W1, W3], torch.eye(C.D, device=W1.device), self.gu[ex], self.a.damp, False)
                (q2,) = quant([W2], torch.eye(W2.shape[1], device=W1.device), self.dn[ex], self.a.damp, False)
                self.write(ex, q1, q3, q2)
                stats[ex].update({"gu": self.gu[ex], "dn": self.dn[ex], "gptq": False, "unrouted": True})
        self.drain()                              # every expert's bytes written before fsync + the done json
        self.sink.close()
        j = {"layer": L, "sink": self.sink.mode, "gu_type": self.gt, "d_type": self.dt, "propagate": self.a.propagate, "damp": self.a.damp,
             "map": self.a.map, "tier": self.a.tier, "experts": {str(e): stats[e] for e in range(C.NE)}}
        tmp = C.wpath("q", f"L{L:02d}.json.tmp")
        json.dump(j, open(tmp, "w"))
        os.replace(tmp, C.wpath("q", f"L{L:02d}.json"))
        e_q = sum(s.get("q_err", 0) for s in stats.values())
        e_y = sum(s.get("y_energy", 0) for s in stats.values())
        C.log(f"quant L{L}: {sum(2 * g + d for g, d in self.size) / 2**30:.2f} GiB -> {self.sink.mode}, held-out rel err {e_q / max(e_y, 1e-30):.4f}, "
              f"tiers gu {dict((t, self.gu.count(t)) for t in G.TIERS)} dn {dict((t, self.dn.count(t)) for t in G.TIERS)}")


def cmd_init(a):
    for f in ("tokens.npy", "meta.json", "hashes.npy"):
        dst = C.wpath(f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(a.calib, f), dst)
    C.cmd_init(a)


def cmd_layer(a):
    if os.path.exists(C.wpath("q", f"L{a.layer:02d}.json")) and os.path.exists(C.wpath("stats", f"L{a.layer:02d}.json")):
        C.log(f"quant L{a.layer}: done already")
        return
    if a.fast_scales:   # smoke tests only: one clip ratio
        G.CLIP = [1.0]
    lay = Layer(a, a.layer)
    a.expert_hook, a.moe_done_hook = lay.expert, lay.finish
    a.no_gptq, a.screen_only, a.damps, a.expert_stride = True, None, [a.damp], 1
    C.cmd_layer(a)


def cmd_fill(a):
    """Copy a QDIR's layers into an --experts-in-place GGUF (tests the in-place path; or a QDIR run -> final file)."""
    meta = json.load(open(a.gguf + ".experts.json"))["layers"]
    for Ls in sorted(meta, key=int):
        L = int(Ls)
        j = json.load(open(os.path.join(a.qdir, f"L{L:02d}.json")))
        size = [(F.nbytes(j["gu_type"][e], (C.D, FF)), F.nbytes(j["d_type"][e], (FF, C.D))) for e in range(C.NE)]
        a.no_write = False
        sink = Sink(a, L, j["gu_type"], j["d_type"], size)
        with open(os.path.join(a.qdir, f"L{L:02d}.bin"), "rb") as f:
            for e, (g, d) in enumerate(size):
                b = f.read(2 * g + d)
                sink.put(e, b[:g], b[g:2 * g], b[2 * g:])
        sink.close()
        C.log(f"fill: layer {L} from {a.qdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fill")
    p.add_argument("--gguf", required=True)
    p.add_argument("--qdir", required=True)
    for name in ("init", "layer", "head"):
        p = sub.add_parser(name)
        p.add_argument("--qwork", required=True)
        p.add_argument("--calib", default=CALIB, help="calibration work dir (tokens/meta/hashes)")
        if name == "layer":
            p.add_argument("layer", type=int)
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--map")
            g.add_argument("--tier", choices=list(G.TIERS))
            p.add_argument("--propagate", choices=("quant", "teacher"), default="quant")
            p.add_argument("--damp", type=float, default=1.0)
            p.add_argument("--min-fit", type=int, default=32)
            p.add_argument("--batch", type=int, default=0)
            p.add_argument("--device", default="cuda")
            p.add_argument("--gc", action="store_true")
            w = p.add_mutually_exclusive_group()
            w.add_argument("--gguf", help="write the experts in place into this v41_gguf --experts-in-place shard 1")
            w.add_argument("--no-write", action="store_true", help="no expert bytes (the all-Q2_0 control: head numbers only)")
            p.add_argument("--gptq-experts", type=int, default=C.NE, help="smoke: GPTQ only experts < K")
            p.add_argument("--fast-scales", action="store_true", help="smoke: one clip ratio")
        if name == "head":
            p.add_argument("--layer", type=int, default=40)
            p.add_argument("--device", default="cuda")
            p.add_argument("--save-argmax", action="store_true")   # v41_calib.cmd_head reads it
    a = ap.parse_args()
    if a.cmd == "fill":
        cmd_fill(a)
        return
    if a.cmd == "layer" and a.map is not None:
        a.tier = "q2_0"   # fallback for nothing; tiers_for reads the map
    set_work(a.qwork)
    {"init": cmd_init, "layer": cmd_layer, "head": C.cmd_head}[a.cmd](a)


if __name__ == "__main__":
    main()
