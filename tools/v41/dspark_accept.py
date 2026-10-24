"""dspark_accept.py - offline DSpark (V4.1's mtp.* draft stack) acceptance on our engine's real trunk states (CPU).

    STRATA_DUMP_MAIN=main.bin build-ds4-gpu/ds41_generate ... --temp 0 -n N > gen.txt     # the engine side
    python3 tools/v41/dspark_accept.py main.bin gen.txt [--threads 16] [--max N]

main.bin: per engine step {int32 token, int32 pos, 3 x 5120 f32} = the step's input token and the HC-mean of the
stream ENTERING layers 37/38/39 (model.py:1264-1266 `main_hiddens.append(h.mean(dim=2))` before `layer(h,...)`;
v41_calib's `h/L37..L39` are the same quantity).  The first 128 records are the prompt tail (they fill DSpark's
sliding window); the rest are decode steps.  gen.txt: the generated ids, one per line (the memguard line skipped).

DeepSeek's own model.py (~/AI/gsq-rco/dsv41-inference) runs the draft on our torch kernels (tools/v41/torch_kernels),
weights straight from the FP8/FP4 checkpoint.  At each step j it calls forward_spec(next token, main_hidden_j, pos_j)
exactly like model.py's __main__ loop, and scores the 5 drafted tokens against what the engine actually produced
(greedy): accepted length = the matching prefix.  Also reports acceptance when the verify is cut at depth 1..5 and
the confidence head's calibration.

The checkpoint keeps `attn.wo_a` as FP8 (E8M0 32x32 block scales) while model.py builds that module in bf16 (its
comment: "convert.py dequantizes it to bf16"), so `load()` must apply the block scale itself - without it wo_a is
~512..2050x too large and every draft is garbage (measured 2.58 -> 0.35 accepted/step).  `--no-fp8-dequant` turns
the dequantization off to reproduce that failure.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.expanduser("~/AI/gsq-rco/dsv41-inference")
sys.path[:0] = [os.path.join(HERE, "torch_kernels"), REF, HERE]
import model as M  # noqa: E402
from v41_gguf import Source  # noqa: E402

def cache_weights():
    """model.py's fp8_gemm / fp4_gemm (our torch stand-ins) dequantize the whole weight to fp32 on every call - ~2G
    params per draft step.  FP8 e4m3 / FP4 e2m1 times a power-of-two scale are exact in bf16, and so is the FP8-quantized
    activation, so each weight is dequantized ONCE into a bf16 cache (keyed by storage address) and the GEMM runs in
    bf16 (fp32 accumulation): the same values, only the summation order differs."""
    import kernel as K
    cache = {}

    def deq(b, b_s, fp4, block_size):
        key = (b.data_ptr(), b_s.data_ptr())
        w = cache.get(key)
        if w is None:
            w = (K.dequant_fp4_weight(b, b_s) if fp4 else K.dequant_fp8_weight(b, b_s, block_size)).to(torch.bfloat16)
            cache[key] = w
        return w

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
        A = K._dequant_act(a, a_s, block_size).to(torch.bfloat16)
        return (A @ deq(b, b_s, False, block_size).t()).to(torch.get_default_dtype())

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
        A = K._dequant_act(a, a_s, act_block_size).to(torch.bfloat16)
        return (A @ deq(b, b_s, True, 0).t()).to(torch.get_default_dtype())

    M.fp8_gemm, M.fp4_gemm = fp8_gemm, fp4_gemm
    return cache


DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.float4_e2m1fn_x2,
      "BF16": torch.bfloat16, "F32": torch.float32}


def load(module, prefix, src, fp8_dequant=True):
    """Every parameter of `module` from checkpoint tensor prefix + its name (dtype-converted when they differ).

    FP8 tensors whose module parameter is NOT fp8 (model.py's `wo_a` is bf16: "convert.py dequantizes it to
    bf16") need the checkpoint's E8M0 block scale applied by hand.  `_dequant_fp8_weight`/convert.py's rule is
    `w = q * scale[row // bs, col // bs]` on a 32x32 or 128x128 grid; without it the value is off by 1/scale
    (measured 512..2050x for the DSpark blocks) and the draft stack outputs garbage.  Returns (missing, notes).
    """
    import kernel as K  # torch_kernels (put first on sys.path above): dequant_fp8_weight
    missing, notes = [], []
    for name, p in module.named_parameters():
        ck = prefix + name
        if ck not in src:
            missing.append(ck)
            continue
        dt, shape, _, _ = src.info(ck)
        t = torch.frombuffer(bytearray(src.read(ck)), dtype=torch.uint8).view(DT[dt]).reshape(shape)
        if dt == "F8_E4M3" and p.dtype != torch.float8_e4m3fn:
            sk = ck.rsplit(".", 1)[0] + ".scale"
            if sk in src:
                if fp8_dequant:
                    sdt, sshape, _, _ = src.info(sk)
                    s = torch.frombuffer(bytearray(src.read(sk)), dtype=torch.uint8).view(DT[sdt]).reshape(sshape)
                    bs = t.shape[0] // s.shape[0]
                    assert bs == t.shape[1] // s.shape[1] and bs in (32, 128), (ck, tuple(t.shape), tuple(s.shape))
                    t = K.dequant_fp8_weight(t, s, bs)          # fp32: q * e8m0 block scale
                    notes.append(f"dequant {ck} {tuple(t.shape)} x{tuple(s.shape)} bs{bs}")
                else:
                    notes.append(f"SCALE DROPPED {ck} (was off by 1/scale ~1e3)")
        if t.dtype != p.dtype:
            t = t.to(torch.float32).to(p.dtype)
        assert tuple(t.shape) == tuple(p.shape), (ck, tuple(t.shape), tuple(p.shape))
        p.data = t
    return missing, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_bin")
    ap.add_argument("gen_txt")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max", type=int, default=0, help="score at most N decode steps")
    ap.add_argument("--dump-drafts", default=None, metavar="FILE",
                    help="one line per scored step: 'j pos d0..d4 c0..c4' (compare with test_ds41_dspark --dump)")
    ap.add_argument("--teacher", default=None, metavar="WORKDIR",
                    help="main_hidden from the FP8 teacher instead: v41_calib.py prep-ids/init/layer work dir "
                         "(h/L37-39.bin = the streams ENTERING the target layers, tokens.npy); the records cover the "
                         "same positions as main_bin's (main_bin then only gives positions)")
    ap.add_argument("--teacher-top1", default=None, metavar="NPY",
                    help="v41_calib.py head --save-argmax output (stats/head_argmax_L40.npy) over the same tokens: also "
                         "report the first draft token vs the teacher's own next-token argmax, and how often our trunk's "
                         "token equals it (separates 'DSpark is weak' from 'our trunk's tokens differ from the teacher's')")
    ap.add_argument("--no-fp8-dequant", action="store_true",
                    help="compile-time bug reproduction: do NOT apply the E8M0 block scale of a checkpoint fp8 tensor "
                         "whose module parameter is bf16 (model.py's wo_a).  Off by 1/scale (~512..2050x on the DSpark "
                         "blocks) -> the old 0.35-0.44 numbers.  Default = dequantize (correct).")
    a = ap.parse_args()
    dump = open(a.dump_drafts, "w") if a.dump_drafts else None
    t1am = np.load(a.teacher_top1)[0] if a.teacher_top1 else None   # t1am[p] = teacher's argmax for token p+1
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.bfloat16)

    cfg = json.load(open(os.path.join(REF, "config.json")))
    fields = M.ModelArgs.__dataclass_fields__
    args = M.ModelArgs(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in fields})
    args.max_seq_len = 8192
    args.temperature = 0.0
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    D = args.dim

    wcache = cache_weights()
    src = Source()
    t0 = time.time()
    miss, notes = [], []
    with torch.no_grad():
        embed = M.ParallelEmbedding(args.vocab_size, D)
        head = M.ParallelHead(args.vocab_size, D, args.norm_eps, args.hc_eps)
        for mod, pre in ((embed, "embed."), (head, "head.")):
            m, n = load(mod, pre, src, not a.no_fp8_dequant)
            miss += m; notes += n
        mtp = []
        for i in range(args.n_mtp_layers):
            b = M.DSparkBlock(args.n_layers + i, args)
            m, n = load(b, f"mtp.{i}.", src, not a.no_fp8_dequant)
            miss += m; notes += n
            b.embed, b.head = embed, head
            mtp.append(b)
    if miss:
        raise SystemExit(f"checkpoint lacks {len(miss)} parameters, e.g. {miss[:5]}")
    if notes:
        print(f"fp8-with-scale for a non-fp8 parameter ({'dequantized' if not a.no_fp8_dequant else 'DROPPED'}):",
              flush=True)
        for n in notes:
            print("   ", n, flush=True)
    print(f"loaded DSpark stack ({args.n_mtp_layers} blocks, block size {args.dspark_block_size}) in {time.time() - t0:.0f} s",
          flush=True)

    rec = np.fromfile(a.main_bin, dtype=np.uint8)
    rs = 8 + 3 * D * 4
    assert rec.size % rs == 0, "main.bin size is not a whole number of records"
    rec = rec.reshape(-1, rs)
    toks = rec[:, 0:4].copy().view(np.int32)[:, 0]
    poss = rec[:, 4:8].copy().view(np.int32)[:, 0]
    hid = rec[:, 8:].copy().view(np.float32)
    eng = hid
    if a.teacher:   # replace the engine's states by the teacher's, same positions / tokens
        tt = np.load(os.path.join(a.teacher, "tokens.npy"))[0]
        S = len(tt)
        hs = []
        for L in args.dspark_target_layer_ids:
            hm = np.memmap(os.path.join(a.teacher, "h", f"L{L:02d}.bin"), dtype=np.int16, mode="r").reshape(S, args.hc_mult, D)
            hs.append((np.asarray(hm).view(np.uint16).astype(np.uint32) << 16).view(np.float32).mean(axis=1))   # bf16 bits -> f32, HC-mean
        th = np.concatenate(hs, axis=1)                                             # [S, 3D]
        assert np.all(toks == tt[poss]), "teacher tokens differ from the engine's at the dumped positions"
        hid = th[poss].astype(np.float32)
        print(f"teacher states from {a.teacher}: cosine vs engine per target layer: " + " ".join(
            f"{float(np.mean([np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30) for x, y in zip(hid[:, i*D:(i+1)*D], eng[:, i*D:(i+1)*D])])):.4f}"
            for i in range(3)), flush=True)
    gen = [int(x) for x in open(a.gen_txt).read().split() if x.lstrip("-").isdigit()]
    # the token after record j: record j+1's input; after the last record: the last generated id
    seq = list(toks) + [gen[-1]]
    n = len(toks)
    n_seed = min(128, n)
    BS = args.dspark_block_size
    print(f"{n} records ({n_seed} prompt-tail seed steps), positions {poss[0]}..{poss[-1]}, {len(gen)} generated", flush=True)

    acc_len, conf_rows, t1_rows = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for j in range(n):
            nxt = torch.tensor([[seq[j + 1]]])
            mh = torch.from_numpy(hid[j]).view(1, 1, -1).to(torch.bfloat16)
            pos = int(poss[j])
            h, main_x = mtp[0].forward_embed(mh, nxt[:, 0])
            pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
            for layer in mtp:
                h, pre_mix = layer(h, pos, pre_mix, main_x)
            r = mtp[-1].forward_head(h, pre_mix, nxt[:, 0])
            if j < n_seed or j + 1 + BS >= len(seq):
                if j % 16 == 0:
                    gb = sum(w.numel() for w in wcache.values()) * 2 / 1e9
                    print(f"  step {j}/{n} ({(time.time() - t0) / (j + 1):.2f} s/step, weight cache {gb:.1f} GB)", flush=True)
                continue
            out_ids, _, conf = r
            draft = out_ids[0, 1:].tolist()             # tokens for seq[j+2 .. j+1+BS]
            truth = seq[j + 2: j + 2 + BS]
            k = 0
            while k < len(truth) and draft[k] == truth[k]:
                k += 1
            acc_len.append(k)
            if dump:
                cc = conf[0].float().tolist()
                dump.write(f"{j} {pos} " + " ".join(str(int(d)) for d in draft) + " " + " ".join(f"{x:.4f}" for x in cc) + "\n")
            if t1am is not None and pos + 1 < len(t1am):   # the token at pos+2, as the teacher would pick it given ours up to pos+1
                tt1 = int(t1am[pos + 1])
                t1_rows.append((draft[0] == tt1, truth[0] == tt1, draft[0] == truth[0]))
            c = conf[0].float().tolist()
            conf_rows.append([(c[i], draft[i] == truth[i]) for i in range(len(truth))])
            if len(acc_len) % 32 == 0:
                print(f"  {len(acc_len)} drafts, mean accepted {np.mean(acc_len):.2f} ({(time.time() - t0) / len(acc_len):.2f} s/draft)",
                      flush=True)
            if a.max and len(acc_len) >= a.max:
                break

    L = np.array(acc_len)
    print(f"\nDSpark acceptance over {len(L)} decode steps (block {BS}, greedy, our GSQ trunk states):")
    print(f"  mean accepted draft tokens per step: {L.mean():.3f}  -> tokens per verify pass {1 + L.mean():.3f}")
    print("  distribution of accepted length: " + " ".join(f"{k}:{np.mean(L == k) * 100:.1f}%" for k in range(BS + 1)))
    for d in range(1, BS + 1):
        print(f"  verify depth {d}: tokens per pass {1 + np.minimum(L, d).mean():.3f}")
    if t1_rows:
        T = np.array(t1_rows, dtype=bool)   # columns: draft1 == teacher, trunk == teacher, draft1 == trunk
        agree = T[:, 1]
        print(f"  teacher top-1 ({len(T)} steps): first draft == teacher {T[:, 0].mean() * 100:.1f}% | our trunk's token == "
              f"teacher {agree.mean() * 100:.1f}% | first draft == trunk {T[:, 2].mean() * 100:.1f}%")
        if agree.any() and (~agree).any():
            print(f"    where trunk == teacher: first draft accepted {T[agree, 2].mean() * 100:.1f}%; "
                  f"where trunk != teacher: {T[~agree, 2].mean() * 100:.1f}% (draft took the teacher's token "
                  f"{T[~agree, 0].mean() * 100:.1f}%)")
    flat = [x for row in conf_rows for x in row]
    if flat:
        cs = np.array([c for c, _ in flat])
        ok = np.array([o for _, o in flat])
        qs = np.quantile(cs, [0.25, 0.5, 0.75])
        bins = np.digitize(cs, qs)
        print("  confidence quartiles -> per-token draft accuracy: " +
              " ".join(f"Q{b + 1} {ok[bins == b].mean() * 100:.0f}%" for b in range(4) if np.any(bins == b)))


if __name__ == "__main__":
    main()
