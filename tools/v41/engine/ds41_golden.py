#!/usr/bin/env python3
"""ds41_golden.py - golden activations for the V4.1 engine gate (P2): DeepSeek's own model.py (on our torch kernels),
run ONE LAYER AT A TIME on the CPU with the GGUF's OWN weights (Q8_0 / F16 / F32 dense, the routed experts decoded
from their Q2_0 / Q3_K / PTQ1_0 rows), so the engine is gated against the same numbers it computes with.

    python3 tools/v41/engine/ds41_golden.py --gguf FILE --text "..." --out DIR [--layers 0-2] [--no-actquant]

- Tokens: the HF tokenizer (BOS + text), or --tokens "1,2,3".  Engram hashes: model.py's NgramHashState with the HF
  tokenizer (independent of the GGUF's token map / primes, which the engine uses).
- Engram rows: straight from HF shards 47/48 (as the engine reads them).
- Everything in fp32 (model.py's own bf16 casts stay where model.py makes them: Engram rows, sparse_attn's p).
- --no-actquant: model.py's act_quant / fp4_act_quant (the KV / indexer quantize-dequantize) become identities, for
  gating an engine that does not emulate them; without it they run as model.py has them.
Writes DIR/golden.npz: tokens [S], per layer x{L} [S,HC,D] (block output), pre{L} [S,HC] (the next attention's pre
mix), ids{L} / wts{L} [S,6] (router), and logits [S,V] = head(hc_pre(last x, last pre)) after the last layer given.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V41 = os.path.dirname(HERE)
sys.path.insert(0, V41)
import v41_calib as C  # noqa: E402  (sets up model.py + torch kernels on sys.path)
import v41_formats as F  # noqa: E402
import v41_gguf as W  # noqa: E402

M, K = C.M, C.K
F32 = torch.float32


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class GG:
    """The GGUF's tensors by name, decoded to fp32 on demand (numpy memmap of the file)."""

    def __init__(self, path):
        self.kv, ts, start = W.read_gguf(path)
        self.mm = np.memmap(path, np.uint8, "r")
        self.t = {n: (shape, ty, start + off) for n, shape, ty, off in ts}

    def has(self, n):
        return n in self.t

    def raw(self, n):
        shape, ty, off = self.t[n]
        return self.mm[off:off + F.nbytes(ty, shape)], shape, ty

    def get(self, n):
        """-> fp32 tensor in torch order (GGUF shape reversed)."""
        b, shape, ty = self.raw(n)
        rows = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        n0 = shape[0]
        if ty == F.F32:
            a = np.frombuffer(b, np.float32).copy()
        elif ty == F.F16:
            a = np.frombuffer(b, np.float16).astype(np.float32)
        elif ty == 26:   # I32
            return torch.from_numpy(np.frombuffer(b, np.int32).copy())
        elif ty == F.Q8_0:
            blk = np.frombuffer(b, np.uint8).reshape(-1, 34)
            d = blk[:, :2].copy().view(np.float16).astype(np.float32)
            q = blk[:, 2:].view(np.int8).astype(np.float32)
            a = (q * d).reshape(-1)
        else:
            a = np.concatenate([F.decode(ty, b[r * F.row_bytes(ty, n0):(r + 1) * F.row_bytes(ty, n0)], n0)
                                for r in range(rows)])
        return torch.from_numpy(a.reshape(tuple(reversed(shape))))

    def expert(self, L, role, e, X):
        """fp32 [rows, cols] of expert e's `role` (gate/up/down) in layer L; X = the layer's expert map."""
        k = "down" if role == "down" else "gu"
        ty, slot = X[k][0][e], X[k][1][e]
        name = X["names"][(role, ty)]
        shape, _, off = self.t[name]
        n0, n1 = shape[0], shape[1]
        rb = F.row_bytes(ty, n0)
        base = off + slot * n1 * rb
        b = self.mm[base:base + n1 * rb]
        return torch.from_numpy(F.decode(ty, b, n0 * n1).reshape(n1, n0))


def expert_map(gg, L, NE):
    """{gu: (types, slots), down: (types, slots), names: {(role, type): tensor}} as ds41_geometry builds it."""
    out = {"names": {}}
    for k, roles in (("gu", ("gate", "up")), ("down", ("down",))):
        mn = f"blk.{L}.ffn_exps_{k}_type"
        if gg.has(mn):
            types = gg.get(mn).tolist()
            slots, cnt = [], {}
            for t in types:
                slots.append(cnt.get(t, 0))
                cnt[t] = cnt.get(t, 0) + 1
            for t in cnt:
                for r in roles:
                    out["names"][(r, t)] = f"blk.{L}.ffn_{r}_exps.{F.TYPE_NAME[t].lower()}.weight"
        else:
            t = gg.t[f"blk.{L}.ffn_{roles[0]}_exps.weight"][1]
            types, slots = [t] * NE, list(range(NE))
            for r in roles:
                out["names"][(r, t)] = f"blk.{L}.ffn_{r}_exps.weight"
        out[k] = (types, slots)
    return out


PARAM = {   # GGUF name (after "blk.L.") -> model.py Block parameter
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "attn_sinks.weight": "attn.attn_sink", "attn_q_a.weight": "attn.wq_a.weight", "attn_q_b.weight": "attn.wq_b.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight", "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight", "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight", "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight", "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight", "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight", "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias", "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight", "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight", "engram_q_norm.weight": "engram.q_weight",
    "engram_k_norm.weight": "engram.k_weight", "engram_kv.weight": "engram.wkv.weight",
}
for _s in ("attn", "ffn"):
    for _p in ("fn", "base", "scale"):
        PARAM[f"hc_{_s}_{_p}.weight"] = f"hc_{_s}_{_p}"


class EngramRows(C.MemmapEngramEmbedding):
    """C.MemmapEngramEmbedding (rows from the HF shard, model.py's bf16 rounding kept), handed on as fp32."""

    def forward(self, indices):
        return super().forward(indices).float()


def sparse_attn_exact(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
    """K.sparse_attn without its bf16 rounding of p (--no-actquant: the engine-exact reference)."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h)
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        idx = topk_idxs[:, i0:i1].long()
        valid = idx >= 0
        g = torch.gather(kv.unsqueeze(1).expand(b, i1 - i0, kv.size(1), d), 2,
                         idx.clamp_min(0).unsqueeze(-1).expand(*idx.shape, d))
        g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), dtype=g.dtype))
        sc = torch.einsum("bchd,bckd->bchk", q[:, i0:i1].float(), g.float()) * softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = torch.maximum(sc.amax(-1), torch.full((), -1e30))
        p = torch.exp(sc - mx.unsqueeze(-1))
        den = p.sum(-1) + torch.exp(sink - mx)
        out[:, i0:i1] = (torch.einsum("bchk,bckd->bchd", p, g.float()) / den.unsqueeze(-1)).to(q.dtype)
    return out


class _NoExpert(torch.nn.Module):
    """Stand-in for a routed Expert (MoE.forward is replaced; the 384 fp4 experts are never materialized)."""

    def __init__(self, *a, **k):
        super().__init__()


def build_block(gg, args, layout, L):
    real_expert = M.Expert

    def make(dim, inter, dtype=None, swiglu_limit=0.0):
        return _NoExpert() if dtype is not None else real_expert(dim, inter, dtype=F32, swiglu_limit=swiglu_limit)
    M.Expert = make
    try:
        blk = M.Block(L, args, layout)
    finally:
        M.Expert = real_expert
    blk.float()   # model.py keeps a few bf16 Linears (ratio-1 compressor, indexer wk / proj, wo_a): fp32 here, same values
    params = dict(blk.named_parameters())
    done = set()
    for gname, pname in PARAM.items():
        n = f"blk.{L}.{gname}"
        if not gg.has(n):
            continue
        t = gg.get(n)
        p = params[pname]
        if pname == "attn.wo_a.weight":
            t = t.reshape(p.shape)
        assert tuple(t.shape) == tuple(p.shape), (n, tuple(t.shape), tuple(p.shape))
        p.data.copy_(t.to(p.dtype))
        done.add(pname)
    missing = {k for k in params if not k.startswith("ffn.experts.")} - done
    assert not missing, (L, sorted(missing))
    return blk


def moe_forward(gg, L, X, rec, zero=False):
    def fwd(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim).float()
        weights, indices = self.gate(x, None)
        rec["ids"], rec["wts"] = indices.clone(), weights.clone()
        y = torch.zeros_like(x)
        lim = self.shared_experts.swiglu_limit
        for e in ([] if zero else sorted(set(indices.flatten().tolist()))):
            W1, W3, W2 = (gg.expert(L, r, e, X) for r in ("gate", "up", "down"))
            idx, top = torch.where(indices == e)
            xe = x[idx]
            g, u = xe @ W1.T, xe @ W3.T
            if lim > 0:
                u = u.clamp(-lim, lim)
                g = g.clamp(max=lim)
            y[idx] += (weights[idx, top, None] * (torch.nn.functional.silu(g) * u)) @ W2.T
        y += self.shared_experts(x)
        return y.view(shape)
    return fwd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--text")
    ap.add_argument("--tokens")
    ap.add_argument("--layers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-actquant", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--zero-experts", action="store_true", help="routed experts contribute 0 (a file still being filled)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    gg = GG(a.gguf)
    partial = gg.kv.get("strata.v41.partial_layers")
    layers = W.parse_layers(a.layers, 40) if a.layers else (list(partial) if partial else list(range(40)))
    assert layers == list(range(layers[0], layers[0] + len(layers))) and layers[0] == 0, "layers must be 0..k"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MAIN)
    if a.tokens:
        ids = [int(x) for x in a.tokens.split(",")]
    else:
        ids = [tok.bos_token_id] + tok.encode(a.text, add_special_tokens=False)
    S = len(ids)
    if a.no_actquant:
        M.act_quant = lambda x, *k, **kw: x
        M.fp4_act_quant = lambda x, *k, **kw: x
        M.sparse_attn = sparse_attn_exact
    torch.set_default_dtype(F32)
    M.default_dtype = F32
    args = C.model_args(1, S)
    if os.environ.get("DS41_TEST_TOPK"):
        args.index_topk = int(os.environ["DS41_TEST_TOPK"])
    if os.environ.get("DS41_TEST_CAND_BLOCKS"):
        args.candidate_topk_blocks = int(os.environ["DS41_TEST_CAND_BLOCKS"])
    layout = M.EngramLayout.from_args(args)
    tokens = torch.tensor([ids])
    hashes = M.NgramHashState(args, layout, tok)(tokens, 0)        # [1, S, n_engram_layers, 24]
    emb = gg.get("token_embd.weight")
    h = emb[tokens].unsqueeze(2).repeat(1, 1, args.hc_mult, 1)      # [1, S, HC, D]
    pre = M.make_identity_pre_mix(h, args.hc_mult)
    out = {"tokens": np.array(ids, np.int32), "hashes": hashes[0].numpy()}
    for L in layers:
        t0 = time.time()
        if L in args.engram_layer_ids:
            C.MemmapEngramEmbedding.SRC = (C.shard_file(C.weight_map()[f"layers.{L}.engram.embed.weight"]),
                                           f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale")
            M.ParallelEngramEmbedding = EngramRows
        blk = build_block(gg, args, layout, L)
        X = expert_map(gg, L, args.n_routed_experts)
        rec = {}
        blk.ffn.forward = moe_forward(gg, L, X, rec, a.zero_experts).__get__(blk.ffn)
        with torch.inference_mode():
            if blk.engram is not None:
                h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :], None)
                out[f"engram{L}"] = h[0].float().numpy()
            h, pre = blk(h, 0, pre, None)
            if blk.attn.is_index_source and M.shared_attn.topk_idxs is not None:
                ti = M.shared_attn.topk_idxs[0]                     # [S, K] (+ window offset), -1 = masked
                off = 0 if L == 0 else None
                out[f"sel{L}"] = ti.numpy().astype(np.int32)
        out[f"x{L}"] = h[0].float().numpy()
        out[f"pre{L}"] = pre[0].float().numpy()
        out[f"ids{L}"] = rec["ids"].numpy().astype(np.int32)
        out[f"wts{L}"] = rec["wts"].float().numpy()
        log(f"layer {L}: {time.time() - t0:.1f} s, |x| {h.float().norm():.4g}, routed {len(set(rec['ids'].flatten().tolist()))} experts")
        del blk
    with torch.inference_mode():
        x = M.Block.hc_pre(None, h, pre)
        nw = gg.get("output_norm.weight")
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + args.norm_eps) * nw
        out["logits"] = (xf[0] @ gg.get("output.weight").T).numpy()
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "golden.npz"), **out)
    top = out["logits"].argmax(-1)
    log(f"wrote {a.out}/golden.npz: S={S}, layers {layers[0]}-{layers[-1]}, argmax tail {tok.decode(top[-8:].tolist())!r}")


if __name__ == "__main__":
    main()
