"""DeepSeek's model.py self-test (small random model: prefill + decode + DSpark head) on our torch kernels.
Checks plumbing/shapes and finiteness, not numerics.  Device from argv[1] (default cpu)."""
import sys, os, torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "torch_kernels"), os.path.expanduser("~/AI/gsq-rco/dsv41-inference")]
import model as M
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_default_dtype(torch.bfloat16); torch.set_default_device(dev); torch.manual_seed(0)
args = M.ModelArgs(dspark_block_size=6, dspark_target_layer_ids=(3, 4), max_seq_len=512)
model = M.Transformer(args)
with torch.no_grad():   # torch.empty weights are garbage: give them small finite values so the run means something
    for n, p in model.named_parameters():
        if p.dtype in (torch.float8_e4m3fn,): p.copy_((torch.randn(p.shape, device=dev) * 0.5).to(p.dtype))
        elif p.dtype == torch.float8_e8m0fnu: p.copy_(torch.full(p.shape, 2.0**-4, device=dev).to(p.dtype))
        elif p.dtype == torch.float4_e2m1fn_x2: p.view(torch.uint8).copy_(torch.randint(0, 256, p.shape, device=dev, dtype=torch.uint8))
        elif p.is_floating_point(): p.copy_(torch.randn(p.shape, device=dev).to(p.dtype) * 0.02 + (1.0 if "norm" in n else 0.0))
x = torch.randint(0, args.vocab_size, (1, 80))
out, logits, mh = model(x[:, :64])
assert torch.isfinite(logits.float()).all(), "prefill logits not finite"
model.forward_spec(out, mh)
for i in range(64, 72):
    out, logits, mh = model(x[:, i:i + 1], i)
    assert torch.isfinite(logits.float()).all(), f"decode {i} not finite"
    r = model.forward_spec(out, mh, i); assert r is not None
print("SELFTEST OK", tuple(logits.shape))
