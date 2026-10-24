"""compare_layer_bytes.py QDIR_BIN QDIR_JSON GGUF LAYER [H_A H_B] - a v41_quant qdir run of one layer vs the same
layer written in place in the GGUF: every expert's gate/up/down bytes, and optionally the propagated hidden states."""
import json, sys, os, hashlib
qbin, qjson, gguf, L = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
lay = json.load(open(gguf + ".experts.json"))["layers"][L]
q = json.load(open(qjson))
assert q["gu_type"] == lay["gu_type"] and q["d_type"] == lay["d_type"], "tier types differ"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41_formats as F
import v41_quant as VQ   # the writer's own dims and sizes (C.D, FF)
D, FF, NE = VQ.C.D, VQ.FF, VQ.C.NE
size = [(F.nbytes(q["gu_type"][e], (D, FF)), F.nbytes(q["d_type"][e], (FF, D))) for e in range(NE)]
assert os.path.getsize(qbin) == sum(2 * g + d for g, d in size), "qdir file size != layout"
bad = []
with open(qbin, "rb") as fq, open(gguf, "rb") as fg:
    for e, (g, d) in enumerate(size):
        b = fq.read(2 * g + d)
        o = lay["off"][str(e)]
        for role, lo, n in (("gate", 0, g), ("up", g, g), ("down", 2 * g, d)):
            fg.seek(o[role])
            if fg.read(n) != b[lo:lo + n]:
                bad.append((e, role))
print(f"layer {L}: {NE} experts x 3 matrices compared, {len(bad)} differ", bad[:10])
ok = not bad
if len(sys.argv) > 6:
    ha, hb = sys.argv[5], sys.argv[6]
    same = os.path.getsize(ha) == os.path.getsize(hb)
    with open(ha, "rb") as a, open(hb, "rb") as b:
        while same:
            x, y = a.read(1 << 26), b.read(1 << 26)
            if x != y: same = False
            if not x: break
    print(f"propagated hidden states {os.path.basename(ha)}: identical={same}")
    ok &= same
print("PASS" if ok else "FAIL")
