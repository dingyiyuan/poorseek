#!/usr/bin/env python3
"""think_diag.py - does the FP8 original want to stop thinking where the GSQ quant keeps going?

The quant's own thinking trace (codebench traces/<id>.json: the problem messages + its reasoning) is re-rendered with
DeepSeek's encoder exactly as the server did (thinking mode, the request's reasoning effort), tokenized, cut to the
first --max tokens (an exact prefix: teacher forcing needs no truncation in the middle), and fed to BOTH the FP8
original (v41_calib layer-streaming chain) and the GSQ quant (v41_e2e --experts gsq).  Per position: log p(</think>),
the entropy, p(top) - so at every point of the same trace we see how much each model wants to end its thinking, and
how sure it is.

  think_diag.py build TRACE.json OUT_DIR [--effort 50] [--max 8192]      -> OUT_DIR/ids.i32, meta.json
  think_diag.py head WORK OUT.npz                                         per-position stats of a finished chain
  think_diag.py compare FP8.npz GSQ.npz OUT_DIR                          the report
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "serve")]
TOK = "/media/Crucial1TB/models/DeepSeek-V4.1-Flash-UNCENSORED-FP8/tokenizer.json"


def build(trace, out, effort, mx):
    import dsv41_encoding as E
    from tokenizers import Tokenizer
    t = json.load(open(trace))
    msgs = [{"role": m["role"], "content": m["content"]} for m in t["messages"]]
    prompt = E.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=effort)
    assert prompt.endswith(E.thinking_start_token), prompt[-40:]
    tk = Tokenizer.from_file(TOK)
    p_ids = tk.encode(prompt, add_special_tokens=False).ids
    r_ids = tk.encode(t["reasoning"], add_special_tokens=False).ids
    end_id = tk.token_to_id(E.thinking_end_token)
    ids = (p_ids + r_ids)[:mx]
    os.makedirs(out, exist_ok=True)
    np.array(ids, np.int32).tofile(os.path.join(out, "ids.i32"))
    # character offset of each reasoning token, to quote the text around a position
    offs, pos = [], 0
    for i in r_ids[:mx - len(p_ids)]:
        offs.append(pos); pos += len(tk.decode([i]))
    json.dump({"trace": trace, "effort": effort, "n_prompt": len(p_ids), "n": len(ids), "n_reasoning_total": len(r_ids),
               "end_id": end_id, "offsets": offs}, open(os.path.join(out, "meta.json"), "w"))
    print(f"build: prompt {len(p_ids)} + reasoning {len(r_ids)} tokens (kept {len(ids)}), </think> = id {end_id}")


def head(work, out_npz, end_id):
    import torch
    import v41_calib as C
    from safetensors import safe_open
    C.WORK = work
    tok = np.load(C.wpath("tokens.npy")); N, S = tok.shape
    dev = "cuda"
    torch.set_default_dtype(C.BF16)
    args, wm = C.model_args(1, S), C.weight_map()
    with torch.device(dev):
        norm = C.M.RMSNorm(C.D, args.norm_eps)
    with safe_open(C.shard_file(wm["norm.weight"]), "pt") as f:
        norm.weight.data.copy_(f.get_tensor("norm.weight"))
    with safe_open(C.shard_file(wm["head.weight"]), "pt") as f:
        W = f.get_tensor("head.weight").to(dev, torch.float32)
    h, pre = C.h_memmap(40, "r", N, S), np.load(C.wpath("pre", "L40.npy"))
    lp_end, ent, ptop, am = (np.zeros(S, np.float32) for _ in range(3)) + (np.zeros(S, np.int32),)
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(h[0:1])).to(dev).view(C.BF16)
        y = norm(torch.sum(torch.from_numpy(pre[0:1]).to(dev).unsqueeze(-1) * x.float(), dim=2).to(C.BF16))[0]
        for c in range(0, S, 512):
            ls = torch.log_softmax(y[c:c + 512].float() @ W.T, -1)
            p = ls.exp()
            lp_end[c:c + 512] = ls[:, end_id].cpu().numpy()
            ent[c:c + 512] = (-(p * ls).sum(-1)).cpu().numpy()
            ptop[c:c + 512] = p.max(-1).values.cpu().numpy()
            am[c:c + 512] = ls.argmax(-1).cpu().numpy()
    np.savez(out_npz, lp_end=lp_end, ent=ent, ptop=ptop, argmax=am, tokens=tok[0])
    print(f"head: {S} positions -> {out_npz}")


def compare(fp8, gsq, out):
    import dsv41_encoding as E  # noqa: F401
    from tokenizers import Tokenizer
    m = json.load(open(os.path.join(out, "meta.json")))
    t = json.load(open(m["trace"]))["reasoning"]
    A, B = np.load(fp8), np.load(gsq)
    n0 = m["n_prompt"]
    r = slice(n0, m["n"])
    pe_f, pe_g = np.exp(A["lp_end"][r]), np.exp(B["lp_end"][r])
    tk = Tokenizer.from_file(TOK)
    toks = A["tokens"][r]
    res = {"positions": int(len(pe_f)),
           "entropy_mean": {"fp8": float(A["ent"][r].mean()), "gsq": float(B["ent"][r].mean())},
           "ptop_mean": {"fp8": float(A["ptop"][r].mean()), "gsq": float(B["ptop"][r].mean())},
           "argmax_agree": float((A["argmax"][r] == B["argmax"][r]).mean()),
           "p_end_sum": {"fp8": float(pe_f.sum()), "gsq": float(pe_g.sum())},
           "end_is_argmax": {"fp8": int((A["argmax"][r] == m["end_id"]).sum()), "gsq": int((B["argmax"][r] == m["end_id"]).sum())}}
    # where each model most wants to stop
    top = np.argsort(-pe_f)[:12]
    rows = []
    for k in sorted(top):
        off = m["offsets"][k] if k < len(m["offsets"]) else len(t)
        rows.append({"pos": int(k), "p_end_fp8": float(pe_f[k]), "p_end_gsq": float(pe_g[k]),
                     "text_before": t[max(0, off - 160):off].replace("\n", " | ")})
    res["top_fp8_stop_points"] = rows
    # first position each model would stop under greedy (end token = argmax)
    for nm, X in (("fp8", A), ("gsq", B)):
        w = np.where(X["argmax"][r] == m["end_id"])[0]
        res[f"first_greedy_stop_{nm}"] = int(w[0]) if len(w) else None
    json.dump(res, open(os.path.join(out, "think_diag.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "top_fp8_stop_points"}, indent=1))
    for row in rows:
        print(f"  pos {row['pos']:5d}  p_end FP8 {row['p_end_fp8']:.4f}  GSQ {row['p_end_gsq']:.4f}  ...{row['text_before'][-110:]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        kw = dict(zip(sys.argv[4::2], sys.argv[5::2]))
        build(sys.argv[2], sys.argv[3], int(kw.get("--effort", 50)), int(kw.get("--max", 8192)))
    elif cmd == "head":
        head(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4])
