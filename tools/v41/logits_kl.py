"""logits_kl.py - compare two ds41_generate --dump-logits files (f32 [n_tokens, n_vocab], same prompt).

    python3 tools/v41/logits_kl.py A.logits B.logits [--ids prompt.i32] [--vocab 129280]

Prints mean / p99 / max KL(A || B) per token, top-1 agreement, and with --ids the perplexity of each over the prompt
(row t predicts ids[t+1]).  Streams in chunks of rows (the files are ~2 GB per 4K tokens).
"""
import argparse

import numpy as np


def logsoftmax(x):
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ids")
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--chunk", type=int, default=128)
    a = ap.parse_args()
    A = np.memmap(a.a, dtype=np.float32, mode="r").reshape(-1, a.vocab)
    B = np.memmap(a.b, dtype=np.float32, mode="r").reshape(-1, a.vocab)
    n = min(len(A), len(B))
    ids = np.fromfile(a.ids, dtype=np.int32) if a.ids else None
    kls, top1, nll_a, nll_b = [], 0, [], []
    for r0 in range(0, n, a.chunk):
        r1 = min(n, r0 + a.chunk)
        la = logsoftmax(np.asarray(A[r0:r1], dtype=np.float64))
        lb = logsoftmax(np.asarray(B[r0:r1], dtype=np.float64))
        kls.append((np.exp(la) * (la - lb)).sum(axis=1))
        top1 += int((la.argmax(1) == lb.argmax(1)).sum())
        if ids is not None:
            for t in range(r0, r1):
                if t + 1 < len(ids):
                    nll_a.append(-la[t - r0, ids[t + 1]])
                    nll_b.append(-lb[t - r0, ids[t + 1]])
    kl = np.concatenate(kls)
    print(f"{n} rows: KL mean {kl.mean():.3e}  p99 {np.quantile(kl, 0.99):.3e}  max {kl.max():.3e};  top-1 agreement {top1 / n:.4f}")
    if ids is not None:
        print(f"ppl A {np.exp(np.mean(nll_a)):.4f}  ppl B {np.exp(np.mean(nll_b)):.4f}  ({len(nll_a)} tokens)")


if __name__ == "__main__":
    main()
