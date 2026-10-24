"""Unit checks of tools/v41/torch_kernels/kernel.py against independent naive definitions."""
import sys, os, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch_kernels"))
import kernel as K
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

# fp4 rounding: every grid value round-trips; midpoints go to the even code
g = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -6], device=dev)
check("e2m1 grid exact", torch.equal(K._round_e2m1(g), g))
mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
check("e2m1 ties to even code", torch.equal(K._round_e2m1(mid), torch.tensor([0, 1, 1, 2, 2, 4, 4.], device=dev)))
# pow2 ceil
v = torch.tensor([1.0, 1.5, 0.25, 3.0, 1e-3], device=dev)
check("pow2_ceil", torch.equal(K._pow2_ceil(v), torch.tensor([1.0, 2.0, 0.25, 4.0, 2.0**-9], device=dev)))
# act_quant ue8m0: scale is a power of two >= amax/448, dequant error small
x = torch.randn(4, 256, device=dev, dtype=torch.bfloat16) * 3
q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
sf = s.float()
check("act_quant scale pow2", torch.all(torch.log2(sf) == torch.round(torch.log2(sf))))
deq = (q.float().unflatten(-1, (-1, 32)) * sf.unsqueeze(-1)).flatten(-2)
check("act_quant roundtrip rel err < 7%", ((deq - x.float()).norm() / x.float().norm()) < 0.07)
y = x.clone(); K.act_quant(y, 32, "ue8m0", torch.float8_e8m0fnu, True)
check("act_quant inplace == dequant", torch.allclose(y.float(), deq.to(torch.bfloat16).float()))
# fp4_act_quant inplace: values land on scale * grid
z = torch.randn(3, 64, device=dev, dtype=torch.bfloat16); z0 = z.clone()
K.fp4_act_quant(z, 32, True)
check("fp4_act_quant error bounded", ((z.float() - z0.float()).norm() / z0.float().norm()) < 0.25)
# fp8 gemm == dequant matmul; fp4 gemm == explicit unpack
N, Kd = 64, 128
w = (torch.randn(N, Kd, device=dev) * 0.05)
ws = torch.full((N // 32, Kd // 32), 2.0**-6, device=dev)
w8 = (w / ws.repeat_interleave(32, 0).repeat_interleave(32, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
a = torch.randn(5, Kd, device=dev, dtype=torch.bfloat16)
aq, as_ = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
ref = (aq.float().unflatten(-1, (-1, 32)) * as_.float().unsqueeze(-1)).flatten(-2) @ (w8.float() * 2.0**-6).t()
torch.set_default_dtype(torch.bfloat16)
out = K.fp8_gemm(aq, as_, w8, ws.to(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
check("fp8_gemm", torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2))
codes = torch.randint(0, 256, (N, Kd // 2), device=dev, dtype=torch.uint8)
fs = torch.randint(120, 130, (N, Kd // 32), device=dev, dtype=torch.uint8)
fsc = torch.ldexp(torch.ones(N, Kd // 32, device=dev, dtype=torch.float32), fs.int() - 127)
lut = K.FP4_LUT.to(dev); wf = torch.empty(N, Kd, device=dev, dtype=torch.float32)
for j in range(Kd):
    c = codes[:, j // 2]; wf[:, j] = lut[((c >> 4) if j % 2 else (c & 15)).long()] * fsc[:, j // 32]
out4 = K.fp4_gemm(aq, as_, codes.view(torch.float4_e2m1fn_x2), fs.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
ref4 = (aq.float().unflatten(-1, (-1, 32)) * as_.float().unsqueeze(-1)).flatten(-2) @ wf.t()
check("fp4_gemm (low nibble = even index)", torch.allclose(out4.float(), ref4, rtol=1e-2, atol=1e-1))
torch.set_default_dtype(torch.float32)
# sparse_attn with all indices == dense attention with a sink
b, m, h, d, n = 1, 7, 4, 64, 11
qq = torch.randn(b, m, h, d, device=dev, dtype=torch.bfloat16); kv = torch.randn(b, n, d, device=dev, dtype=torch.bfloat16)
sink = torch.randn(h, device=dev)
idx = torch.arange(n, device=dev).view(1, 1, n).expand(b, m, n).int().contiguous()
o = K.sparse_attn(qq, kv, sink, idx, d**-0.5)
sc = torch.einsum("bmhd,bnd->bmhn", qq.float(), kv.float()) * d**-0.5
full = torch.cat([sc, sink.view(1, 1, h, 1).expand(b, m, h, 1)], -1).softmax(-1)[..., :n]
ref = torch.einsum("bmhn,bnd->bmhd", full, kv.float())
check("sparse_attn == dense+sink", torch.allclose(o.float(), ref, atol=2e-2))
idx2 = idx.clone(); idx2[..., 5:] = -1
o2 = K.sparse_attn(qq, kv, sink, idx2, d**-0.5)
full2 = torch.cat([sc[..., :5], sink.view(1, 1, h, 1).expand(b, m, h, 1)], -1).softmax(-1)[..., :5]
check("sparse_attn masks -1", torch.allclose(o2.float(), torch.einsum("bmhn,bnd->bmhd", full2, kv[:, :5].float()), atol=2e-2))
# hc sinkhorn: comb ~ doubly stochastic
mix = torch.randn(2, 3, 24, device=dev)
pre, post, comb = K.hc_split_sinkhorn(mix, torch.rand(3, device=dev), torch.randn(24, device=dev))
check("sinkhorn doubly stochastic", torch.allclose(comb.sum(-1), torch.ones(2, 3, 4, device=dev), atol=1e-3)
      and torch.allclose(comb.sum(-2), torch.ones(2, 3, 4, device=dev), atol=1e-3))
check("post in (0,2), pre > 0", bool((post > 0).all() and (post < 2).all() and (pre > 0).all()))
print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
