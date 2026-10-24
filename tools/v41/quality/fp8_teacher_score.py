#!/usr/bin/env python3
"""The FP8 original (DeepSeek's own model.py, layer-streaming: tools/v41/v41_calib.py) teacher-forced over DwarfStar's
official API fixtures - the same cases, positions and metrics as score_official.py scores our engine on.  Answers:
how close does the full-quality (uncensored) checkpoint itself get to the API?  Then the quant's gap to the API
splits into "quant" (FP8 -> GSQ) and "not quant" (abliteration / API revision / API sampling).

  prep  <cases.jsonl> [--max-len 256]   cases with len(ids) <= max-len (the short group: 100 of 102 fit 256) as one
                                         [N, S] batch, right-padded with the pad token (causal: padding after a
                                         case cannot change its rows); writes V41_WORK/{tokens,hashes}.npy, meta.json,
                                         rows.json (row -> case id) and cases_sel.jsonl (the subset, for score_official)
  (then: V41_WORK=<same> EXTRA=--no-gptq bash tools/v41/calib_layers.sh 0 39)
  score <cases.jsonl>                   final norm + head on h/L40; NLL of the API's token, top-1 vs the API's top
                                         token, top-20 overlap lower bound - per case weighted by its positions
Env: V41_WORK (required - never the calibration work dir).
"""
import json, os, sys

import numpy as np

if not os.environ.get("V41_WORK"):
    sys.exit("set V41_WORK to a fresh directory (the calibration work dir must not be overwritten)")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
import v41_calib as C  # noqa: E402

PAD = 2


def prep(cases_path, max_len):
    cases = [json.loads(l) for l in open(cases_path)]
    sel = [c for c in cases if len(c["ids"]) <= max_len]
    S = max(len(c["ids"]) for c in sel)
    os.makedirs(C.WORK, exist_ok=True)
    assert not os.path.exists(C.wpath("h", "L00.bin")), f"{C.WORK} already has a chain - use a fresh V41_WORK"
    tokens = np.full((len(sel), S), PAD, np.int32)
    for i, c in enumerate(sel):
        tokens[i, :len(c["ids"])] = c["ids"]
    np.save(C.wpath("tokens.npy"), tokens)
    json.dump({"n": len(sel), "seq": S, "domains": [c["id"] for c in sel], "is_eval": [True] * len(sel),
               "corpus": cases_path}, open(C.wpath("meta.json"), "w"))
    json.dump([c["id"] for c in sel], open(C.wpath("rows.json"), "w"))
    with open(C.wpath("cases_sel.jsonl"), "w") as f:
        for c in sel:
            f.write(json.dumps(c) + "\n")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MAIN)
    args = C.model_args(len(sel), S)
    hs = C.M.NgramHashState(args, C.M.EngramLayout.from_args(args), tok)
    with torch.inference_mode():
        hashes = hs(torch.from_numpy(tokens).long(), 0)
    np.save(C.wpath("hashes.npy"), hashes.numpy())
    print(f"prep: {len(sel)} of {len(cases)} cases (len <= {max_len}) -> [{len(sel)}, {S}], "
          f"{sum(len(c['ref']) for c in sel)} scored positions; work {C.WORK}")


def score(cases_path):
    from safetensors import safe_open
    cases = {c["id"]: c for c in map(json.loads, open(cases_path))}
    rows = json.load(open(C.wpath("rows.json")))
    tokens = np.load(C.wpath("tokens.npy"))
    N, S = tokens.shape
    dev = "cuda"
    torch.set_default_dtype(C.BF16)
    args = C.model_args(1, S)
    wm = C.weight_map()
    with torch.device(dev):
        norm = C.M.RMSNorm(C.D, args.norm_eps)
    with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(C.shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(dev, torch.float32)
    h = C.h_memmap(40, "r", N, S)
    pre = np.load(C.wpath("pre", "L40.npy"))
    recs, per = [], {}
    with torch.inference_mode():
        for i, cid in enumerate(rows):
            c = cases[cid]
            k, n = len(c["ref"]), len(c["ids"])
            assert list(tokens[i, :n]) == c["ids"]
            x = torch.from_numpy(np.asarray(h[i:i + 1])).to(dev).view(C.BF16)
            pm = torch.from_numpy(pre[i:i + 1]).to(dev)
            y = norm(torch.sum(pm.unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]
            ls = torch.log_softmax(y[n - k - 1:n - 1].float() @ W.T, -1).double().cpu().numpy()   # row p predicts p+1
            nll = top1 = ovl = 0.0
            for j in range(k):
                nll += -ls[j, c["ref"][j]]
                top1 += int(np.argmax(ls[j]) == c["api_top1"][j])
                seen = set()
                for tid, lpa in c["api_top20"][j]:
                    if tid < 0 or tid in seen:
                        continue
                    seen.add(tid)
                    ovl += min(np.exp(ls[j, tid]), np.exp(lpa))
            recs.append((nll / k, top1 / k, ovl / k, k))
            per[cid] = {"nll": nll / k, "top1": top1 / k, "overlap": ovl / k, "positions": k}
    n = sum(r[3] for r in recs)
    agg = [sum(r[m] * r[3] for r in recs) / n for m in range(3)]
    json.dump({"all": {"nll": agg[0], "top1": agg[1], "overlap": agg[2], "cases": len(recs), "positions": n},
               "cases": per}, open(C.wpath("fp8_score.json"), "w"), indent=1)
    print(f"{'FP8 teacher':40s} {'short':6s} {agg[0]:8.4f} {agg[1] * 100:6.2f}% {agg[2]:8.4f} {len(recs):5d} {n:5d}")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2], int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--max-len" else 256)
    elif sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        sys.exit(__doc__)
