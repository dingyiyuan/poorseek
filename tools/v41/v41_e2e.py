#!/usr/bin/env python3
"""v41_e2e.py - end-to-end variants of DeepSeek-V4.1 on DwarfStar's 100 short API fixtures (the layer-streaming chain
of v41_calib, tokens prepared by quality/fp8_teacher_score.py prep), scored against the API AND the FP8 original:
  --experts fp4      the original FP4 experts      --experts gsq   today's GSQ file (its own bytes, GPU-decoded)
  --dense fp8|q8|q6k the dense half (attention projections + shared expert, the GGUF's Q8_0 set) as the source FP8,
                     or fake-quantized from it to Q8_0 / Q6_K (round-to-nearest, ggml's block layouts)
  setup --ework DIR | init | layer L | score [--ref FP8_WORK]
score: API top-1 / NLL / overlap (score_official's metrics) + vs the FP8 reference: mean KL(FP8 || this), top-1
agreement with FP8, and confidence (mean entropy, mean top-1 probability) next to FP8's - the "unsureness" gap.
"""
import argparse, json, os, shutil, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "quality")]
import v41_calib as C  # noqa: E402
FIX = "/media/Crucial1TB/v41-fp8-fixtures"
DENSE = ("attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
         "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3")


def fake_q8_0(W):
    B = W.float().reshape(-1, 32)
    d = (B.abs().amax(1, keepdim=True) / 127).half().float()
    return (torch.round(B / d.clamp_min(1e-30)).clamp(-127, 127) * d).reshape(W.shape)


def fake_q6_k(W):
    """ggml Q6_K: 256-blocks, 16 sub-blocks of 16 with int8 scales (fp16 d per block), 6-bit q in [-32, 31]."""
    B = W.float().reshape(-1, 16, 16)
    am = B.abs().amax(2)                                            # [nb, 16]
    imax = torch.gather(B, 2, B.abs().argmax(2, keepdim=True)).squeeze(2)
    sub = imax / -32                                                # ggml: scale from the max-|x| value, q = -32 there
    ms = sub.abs().amax(1, keepdim=True)
    iscale = torch.where(ms > 0, -128 / ms.clamp_min(1e-30), torch.zeros_like(ms))
    sc = torch.round(iscale * sub).clamp(-128, 127)
    d = (1 / iscale.where(iscale != 0, torch.ones_like(iscale))).half().float() * (ms > 0)
    step = (d * sc).unsqueeze(2)                                    # [nb, 16, 1]
    q = torch.round(B / step.where(step != 0, torch.ones_like(step))).clamp(-32, 31) * (step != 0)
    return (q * step).reshape(W.shape)


def dense_hook(kind):
    fq = {"q8": fake_q8_0, "q6k": fake_q6_k}[kind]

    def hook(L, blk):
        mods = dict(blk.named_modules())
        for n in DENSE:
            m = mods[n]
            w = m.weight
            W = C.K.dequant_fp8_weight(w, w.scale) if w.dtype == torch.float8_e4m3fn else w.float()
            m.weight = torch.nn.Parameter(fq(W).to(torch.bfloat16), requires_grad=False)
        C.log(f"layer {L}: dense {kind} fake-quant on {len(DENSE)} tensors")
    return hook


def gsq_hook(a):
    import v41_repair as R
    state = {}

    def hook(L, ex, exp, xe, ge, fit, lim, out_teacher):
        if state.get("L") != L:
            # V41_GSQ_HYBRID="20-35=/path/other.gguf": those layers' experts from another GSQ file (which band matters?)
            path = None
            for grp in filter(None, os.environ.get("V41_GSQ_HYBRID", "").split(";")):
                rng, p = grp.split("=", 1)
                lo, _, hi = rng.partition("-")
                if int(lo) <= L <= int(hi or lo):
                    path = p
            state.update(L=L, g=R.GsqLayer(L, a.device, path))
            C.log(f"layer {L}: GSQ experts from {path or R.GGUF}")
        W1, W3, W2 = state["g"].weights(ex)
        out = torch.empty_like(out_teacher)
        for c in range(0, len(xe), C.ROWS):
            out[c:c + C.ROWS] = C.expert_f(xe[c:c + C.ROWS], W1, W3, W2, ge[c:c + C.ROWS], lim).to(out.dtype)
        return out, {}
    return hook


def logprobs(work, dev, head_kind):
    """Per fixture row: log-softmax rows for its continuation positions (as fp8_teacher_score.score)."""
    from safetensors import safe_open
    C.WORK = work
    rows = json.load(open(C.wpath("rows.json")))
    cases = {c["id"]: c for c in map(json.loads, open(C.wpath("cases_sel.jsonl")))}
    tok = np.load(C.wpath("tokens.npy")); N, S = tok.shape
    torch.set_default_dtype(C.BF16)
    args, wm = C.model_args(1, S), C.weight_map()
    with torch.device(dev):
        norm = C.M.RMSNorm(C.D, args.norm_eps)
    with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(C.shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(dev, torch.float32)
    if head_kind != "fp8":
        W = {"q8": fake_q8_0, "q6k": fake_q6_k}[head_kind](W)
    h, pre = C.h_memmap(40, "r", N, S), np.load(C.wpath("pre", "L40.npy"))
    out = {}
    with torch.inference_mode():
        for i, cid in enumerate(rows):
            c = cases[cid]; k, n = len(c["ref"]), len(c["ids"])
            x = torch.from_numpy(np.asarray(h[i:i + 1])).to(dev).view(C.BF16)
            y = norm(torch.sum(torch.from_numpy(pre[i:i + 1]).to(dev).unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]
            out[cid] = torch.log_softmax(y[n - k - 1:n - 1].float() @ W.T, -1)
    return out, cases


def cmd_score(a):
    dev = "cuda"
    me, cases = logprobs(a.ework, dev, a.dense if a.dense != "fp8" else "fp8")
    ref, _ = logprobs(a.ref, dev, "fp8") if a.ref else (None, None)
    t = {"nll": 0.0, "top1_api": 0.0, "ovl": 0.0, "kl": 0.0, "top1_fp8": 0.0, "ent": 0.0, "pmax": 0.0,
         "ent_fp8": 0.0, "pmax_fp8": 0.0}
    n = 0
    for cid, ls in me.items():
        c = cases[cid]; k = len(c["ref"])
        ref_ids = torch.tensor(c["ref"], device=dev); api1 = torch.tensor(c["api_top1"], device=dev)
        t["nll"] += float(-ls.gather(1, ref_ids[:, None]).sum())
        t["top1_api"] += float((ls.argmax(1) == api1).sum())
        p = ls.exp()
        t["ent"] += float(-(p * ls).sum()); t["pmax"] += float(p.max(1).values.sum())
        for j in range(k):
            seen = set()
            for tid, lpa in c["api_top20"][j]:
                if tid >= 0 and tid not in seen:
                    seen.add(tid); t["ovl"] += min(float(p[j, tid]), float(np.exp(lpa)))
        if ref is not None:
            lr = ref[cid]; pr = lr.exp()
            t["kl"] += float((pr * (lr - ls)).sum()); t["top1_fp8"] += float((ls.argmax(1) == lr.argmax(1)).sum())
            t["ent_fp8"] += float(-(pr * lr).sum()); t["pmax_fp8"] += float(pr.max(1).values.sum())
        n += k
    r = {k: v / n for k, v in t.items()}
    r["positions"] = n
    json.dump(r, open(os.path.join(a.ework, "e2e_score.json"), "w"), indent=1)
    print(f"{os.path.basename(a.ework)}: API top-1 {r['top1_api']*100:.2f}%  NLL {r['nll']:.4f}  overlap {r['ovl']:.4f}"
          + (f"  | vs FP8: KL {r['kl']:.4f}  top-1 {r['top1_fp8']*100:.2f}%  entropy {r['ent']:.3f} (FP8 {r['ent_fp8']:.3f})"
             f"  p(top) {r['pmax']:.3f} (FP8 {r['pmax_fp8']:.3f})" if ref is not None else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["setup", "init", "layer", "score"]); ap.add_argument("layer", nargs="?", type=int)
    ap.add_argument("--ework", required=True); ap.add_argument("--experts", default="fp4", choices=["fp4", "gsq"])
    ap.add_argument("--dense", default="fp8", choices=["fp8", "q8", "q6k"]); ap.add_argument("--ref", default=FIX)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--gc", action="store_true")
    a = ap.parse_args()
    if a.cmd == "setup":
        os.makedirs(a.ework, exist_ok=True)
        for f in ("tokens.npy", "hashes.npy", "meta.json", "rows.json", "cases_sel.jsonl"):
            shutil.copy(os.path.join(FIX, f), os.path.join(a.ework, f))
        return
    C.WORK = a.ework
    if a.cmd == "init":
        return C.cmd_init(a)
    if a.cmd == "score":
        return cmd_score(a)
    a.no_gptq, a.screen_only, a.damps, a.expert_stride, a.min_fit, a.gptq_experts = True, None, [1.0], 1, 32, 0
    if a.dense != "fp8":
        a.block_hook = dense_hook(a.dense)
    if a.experts == "gsq":
        a.expert_hook = gsq_hook(a)
    C.cmd_layer(a)


if __name__ == "__main__":
    main()
