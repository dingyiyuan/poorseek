#!/usr/bin/env python3
"""Compare a DSpark run's verify rows (STRATA_DSPARK_DUMP) against an independent full-path prefill of the same
stream (--dump-logits of prompt + the run's tokens).  Row `pos` of either predicts the token at pos+1.

Only rows whose INPUTS are on the emitted stream are comparable: in a pass that verifies [t, d0..d(k-1)] at
P+1.., rows t, d0..d(a-1) are (a = accepted drafts, from where the next pass starts); a re-fed row is too.

usage: vdump_cmp.py <run.vdump> <arb.logits> [nv=129280]
"""
import sys, struct
import numpy as np

vd, arb = sys.argv[1], sys.argv[2]
nv = int(sys.argv[3]) if len(sys.argv) > 3 else 129280
A = np.memmap(arb, dtype=np.float32, mode='r').reshape(-1, nv)

recs = []   # (pos, m, row, refeed, offset)
rb = 16 + 4 * nv
with open(vd, 'rb') as f:
    f.seek(0, 2); n = f.tell() // rb; f.seek(0)
    for i in range(n):
        pos, m, row, ref = struct.unpack('<4i', f.read(16))
        recs.append((pos, m, row, ref, i * rb + 16)); f.seek(4 * nv, 1)
V = np.memmap(vd, dtype=np.uint8, mode='r')

# group rows into passes (a pass = consecutive records with row 0..m-1)
passes, cur = [], []
for r in recs:
    if r[2] == 0 and cur: passes.append(cur); cur = []
    cur.append(r)
if cur: passes.append(cur)

def first_pos(p):   # P+1: the first non-re-fed row's position
    return p[1][0] if p[0][3] else p[0][0]

out = []
for pi, p in enumerate(passes):
    if pi + 1 >= len(passes): break
    acc = first_pos(passes[pi + 1]) - first_pos(p) - 1
    r0 = 1 if p[0][3] else 0
    for (pos, m, row, ref, off) in p:
        if row > r0 + acc: continue            # input is a rejected draft
        if pos >= A.shape[0]: continue
        v = np.frombuffer(V[off: off + 4 * nv].tobytes(), dtype=np.float32)
        a = A[pos]
        d = np.abs(v - a)
        av, aa = int(v.argmax()), int(a.argmax())
        top2 = np.sort(a)[-2:]
        out.append((pos, m, row, ref, float(d.max()), float(d.mean()), av == aa, float(top2[1] - top2[0]), av, aa))

if not out:
    print('no comparable rows'); sys.exit(1)
mx = np.array([o[4] for o in out]); mn = np.array([o[5] for o in out]); agree = np.array([o[6] for o in out])
print(f'{vd}: {len(out)} comparable rows; argmax agree {agree.mean()*100:.1f}%; max|dlogit| median {np.median(mx):.3f} '
      f'p90 {np.percentile(mx,90):.3f} max {mx.max():.3f}; mean|dlogit| median {np.median(mn):.4f}')
# by row offset within the pass (the multi-token signature: a bug that reads token 0's activation shows at row>0)
for r in sorted(set(o[2] for o in out)):
    sel = [o for o in out if o[2] == r]
    print(f'  row {r}: n={len(sel):3d}  max|d| median {np.median([o[4] for o in sel]):.3f}  argmax agree '
          f'{np.mean([o[6] for o in sel])*100:.1f}%')
worst = sorted(out, key=lambda o: -o[4])[:8]
print('  worst rows (pos, m, row, refeed, max|d|, prefill top-1 gap, run argmax, prefill argmax):')
for o in worst:
    print(f'    pos {o[0]} m {o[1]} row {o[2]} ref {o[3]}  max|d| {o[4]:.3f}  gap {o[7]:.3f}  {o[8]} vs {o[9]}')
