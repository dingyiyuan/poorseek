"""test_gptq_graph.py - v41_gptq_graph.gptq must be bit-identical to v41_gptq.gptq (all tiers, both expert shapes,
realistic Hessians with dead columns); also times both."""
import time, torch, v41_gptq as G, v41_gptq_graph as GG
dev = "cuda"; torch.manual_seed(1)
F32 = torch.float32
torch.set_default_dtype(torch.bfloat16)   # as model.py does in the real run: every buffer must still be fp32
ok = True
for R, Cc in ((4608, 4096), (4096, 2304)):
    X = torch.randn(3000, Cc, device=dev, dtype=F32) * torch.rand(Cc, device=dev, dtype=F32) * 2
    H = X.T @ X; H[:, 5] = 0; H[5, :] = 0                       # one dead column
    Hinv, dead = G.hinv_chol(H, 1.0)
    d = torch.diag(H).clamp_min(0) + 1e-8
    for tier in G.TIERS:
        lo, hi_, _, _ = G.TIERS[tier]
        W = torch.randn(R, Cc, device=dev, dtype=F32) * 0.02
        S = G.expand_scales(G.search_scales(W, d, tier), tier, Cc)
        lov = torch.full((R, 1), float(lo), device=dev, dtype=F32); hiv = torch.full((R, 1), float(hi_), device=dev, dtype=F32)
        torch.cuda.synchronize(); t0 = time.time(); Qa = G.gptq(W, S, lov, hiv, Hinv, dead); torch.cuda.synchronize(); ta = time.time() - t0
        GG.gptq(W, S, lov, hiv, Hinv, dead)                     # first call captures
        torch.cuda.synchronize(); t0 = time.time(); Qb = GG.gptq(W, S, lov, hiv, Hinv, dead); torch.cuda.synchronize(); tb = time.time() - t0
        same = torch.equal(Qa, Qb) and Qb.dtype == F32
        ok &= same
        print(f"{R}x{Cc} {tier}: identical={same} eager {ta*1000:.0f} ms graph {tb*1000:.0f} ms", flush=True)
print("PASS" if ok else "FAIL")
