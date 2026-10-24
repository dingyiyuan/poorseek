"""Gate: v41_calib.py's split forward (attention half / expert-major MoE / hc_post) must equal DeepSeek's unmodified
Block.forward (+ engram before the block) on the same input.  Uses a smoke workdir made by
`V41_WORK=... v41_calib.py prep --n 4 --seq 256; init; layer 0 [1] --device cpu`.
  V41_WORK=<smoke> python3 test_calib_equiv.py L [device]"""
import os, sys
import numpy as np
import torch
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
import v41_calib as C  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
tokens = np.load(C.wpath("tokens.npy"))
N, S = tokens.shape
B = max(1, 4096 // S)   # the same per-batch rule as v41_calib layer (shared attention state is per batch)
torch.set_default_dtype(torch.bfloat16)
args = C.model_args(B, S)
layout = C.M.EngramLayout.from_args(args)
if L in args.engram_layer_ids:
    C.MemmapEngramEmbedding.SRC = (C.shard_file(C.weight_map()[f"layers.{L}.engram.embed.weight"]),
                                   f"layers.{L}.engram.embed.weight", f"layers.{L}.engram.embed.scale")
    C.M.ParallelEngramEmbedding = C.MemmapEngramEmbedding
with torch.device(dev):
    blk = C.M.Block(L, args, layout)
C.load_block(blk, L)
torch.set_default_device(dev)
shared = torch.load(C.wpath("shared", f"L{L:02d}.pt")) if os.path.exists(C.wpath("shared", f"L{L:02d}.pt")) else None
h_in, h_out = C.h_memmap(L, "r", N, S), C.h_memmap(L + 1, "r", N, S)
pre_in, pre_out = np.load(C.wpath("pre", f"L{L:02d}.npy")), np.load(C.wpath("pre", f"L{L + 1:02d}.npy"))
worst = 0.0
with torch.inference_mode():
    for bi, i0 in enumerate(range(0, N, B)):
        for k in C.SHARED:
            v = shared[bi][k] if shared else None
            setattr(C.M.shared_attn, k, None if v is None else v.to(dev))
        x = torch.from_numpy(np.asarray(h_in[i0:i0 + B])).to(dev).view(torch.bfloat16)
        if blk.engram is not None:
            hs = np.load(C.wpath("hashes.npy"))[i0:i0 + B, :, args.engram_layer_ids.index(L), :]
            x = blk.engram(x, torch.from_numpy(hs).to(dev), None)
        ref, ref_pre = blk(x, 0, torch.from_numpy(pre_in[i0:i0 + B]).to(dev), None)
        got = torch.from_numpy(np.asarray(h_out[i0:i0 + B])).view(torch.bfloat16).to(dev)
        rel = float((ref.float() - got.float()).norm() / ref.float().norm())
        exact = float((ref.view(torch.int16) == got.view(torch.int16)).float().mean())
        dpre = float((ref_pre - torch.from_numpy(pre_out[i0:i0 + B]).to(dev)).abs().max())
        print(f"layer {L} batch {bi}: rel diff {rel:.2e}, bit-exact {exact:.4%}, pre_mix max diff {dpre:.1e}")
        worst = max(worst, rel)
ok = worst < 2e-3
print("PASS" if ok else "FAIL", f"worst rel diff {worst:.2e}")
sys.exit(0 if ok else 1)
