"""Unit checks for v41_gptq.py: every quantized value is exactly representable in its GGUF format, GPTQ beats RTN
on correlated activations, and the stacked multi-tier run equals separate runs.  CPU, a few seconds."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import v41_gptq as G

torch.manual_seed(0)
R, C, N = 96, 512, 4096
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += not cond


# correlated inputs (low-rank + noise) and an FP4-like weight
X = torch.randn(N, 32) @ torch.randn(32, C) * 0.5 + torch.randn(N, C) * 0.1
W = torch.randn(R, C) * (torch.rand(R, C) < 0.9)
H = X.T @ X
d = torch.diag(H)

for t in G.TIERS:
    lo, hi, g, _ = G.TIERS[t]
    s = G.search_scales(W, d, t)
    S = G.expand_scales(s, t, C)
    if t == "q3_k":
        sb = s.view(R, C // 256, 16)
        dsup = sb.abs().amax(-1, keepdim=True) / 31
        dsup16 = dsup.half().float()
        ints = sb / dsup16
        check(torch.allclose(ints, ints.round(), atol=1e-3) and ints.abs().max() <= 32, f"{t}: sub-scales = fp16 d * int6")
    else:
        check(torch.equal(s, s.half().float()), f"{t}: scales are exact fp16")
    Qg, Qr = G.quantize_stack([W], H, [t])
    for name, Q in (("gptq", Qg[t][0]), ("rtn", Qr[t][0])):
        k = Q / S
        check(torch.allclose(k, k.round(), atol=1e-4) and k.round().min() >= lo and k.round().max() <= hi,
              f"{t}/{name}: values on the grid [{lo},{hi}]")
    eg = ((X @ (W - Qg[t][0]).T) ** 2).sum() / ((X @ W.T) ** 2).sum()
    er = ((X @ (W - Qr[t][0]).T) ** 2).sum() / ((X @ W.T) ** 2).sum()
    check(eg < er, f"{t}: GPTQ output err {eg:.4f} < RTN {er:.4f}")

# stacked (two matrices x three tiers) == separate
W2 = torch.randn(64, C)
Qg_s, _ = G.quantize_stack([W, W2], H, list(G.TIERS))
for t in G.TIERS:
    Qg_1, _ = G.quantize_stack([W2], H, [t])
    check(torch.allclose(Qg_s[t][1], Qg_1[t][0], atol=1e-5), f"stacked == separate ({t}, 2nd matrix)")

# error ordering across tiers on the same weights
errs = {t: float(((X @ (W - G.quantize_stack([W], H, [t])[0][t][0]).T) ** 2).sum()) for t in G.TIERS}
check(errs["q3_k"] < errs["q2_0"] < errs["ptq1"], f"tier ordering q3 < q2 < ternary: {errs}")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
