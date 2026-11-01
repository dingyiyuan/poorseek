#!/usr/bin/env python3
"""engram_pack_patch.py - point a pack's Engram layers at a converted table (FP8 codes + E8M0 scales).

pack_gguf.py maps `engram_embd.weight` but knows neither a `.scale` tensor nor the FP8 dtypes, so a table taken from the FP8
checkpoint (tools/dsv41/engram_from_safetensors.py) is written into the pack here instead: the two `layers.<l>.engram.embed.weight`
lines are replaced with the FP8 rows and a `layers.<l>.engram.embed.scale` line follows, which is what makes ds4_model.cu
choose its fmt 3 (FP8 codes in one tensor, their E8M0 scales in another).

    python3 tools/dsv41/engram_pack_patch.py --pack /data/models/dsv41-iq2xxs-pack \\
        --engram /home/dyy/engram-mxfp8/engram-mxfp8.gguf --out /data/models/dsv41-iq2xxs-pack-real
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys


def load_pack_gguf():
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("pack_gguf", here / "pack_gguf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="the pack whose Engram lines are replaced")
    ap.add_argument("--engram", required=True, help="the GGUF written by engram_from_safetensors.py")
    ap.add_argument("--out", required=True, help="the new pack folder")
    ap.add_argument("--name", default=None, help="the name written into tensors.txt (default: the absolute path; the engine "
                                              "resolves names against model_dir, so a file beside the model is best)")
    a = ap.parse_args()
    pk = load_pack_gguf()
    pack = pathlib.Path(a.pack).expanduser()
    engram = pathlib.Path(a.engram).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    for name in ("ds4.txt", "engram.bin"):
        shutil.copy2(pack / name, out / name)
    if (out / "tokenizer").exists():
        shutil.rmtree(out / "tokenizer")
    shutil.copytree(pack / "tokenizer", out / "tokenizer")

    g = pk.Gguf(engram)
    tensors = {}
    for name, typ, dims, off in g.tensors:
        if "engram_embd" not in name:
            continue
        dname = pk.TYPES.get(typ, (None,))[0]
        n = 1
        for d in dims:
            n *= d
        blck, tsize = pk.TYPES[typ][1], pk.TYPES[typ][2]
        tensors[name] = (dname, g.data_start + off, n // blck * tsize, dims)   # the pack's offsets are absolute

    lines = (pack / "tensors.txt").read_text().splitlines()
    files, body = [], []
    for ln in lines:
        if ln.startswith("file "):
            files.append(ln)
        else:
            body.append(ln)
    # the engram file is the last one; its name is absolute so the pack works from anywhere
    idx = len(files)
    files.append("file %d %s" % (idx, a.name if a.name else str(engram)))

    table = {}
    for name, (dname, off, nbytes, dims) in tensors.items():   # blk.<l>.engram_embd.{weight,scale}
        import re
        m = re.match(r"blk\.(\d+)\.engram_embd\.(\w+)$", name)
        table[(int(m.group(1)), m.group(2))] = (dname, off, nbytes, dims)

    new_body, replaced = [], 0
    for ln in body:
        f = ln.split()
        if len(f) > 2 and f[0] == "t" and f[1].endswith(".engram.embed.weight"):
            layer = int(f[1].split(".")[1])
            w = table.get((layer, "weight"))
            sc = table.get((layer, "scale"))
            if w is None or sc is None:
                new_body.append(ln)
                continue
            new_body.append("t layers.%d.engram.embed.weight %s %d %d %d %d %d" %
                            (layer, w[0], idx, w[1], w[2], w[3][0], w[3][1]))
            new_body.append("t layers.%d.engram.embed.scale %s %d %d %d %d %d" %
                            (layer, sc[0], idx, sc[1], sc[2], sc[3][0], sc[3][1]))
            replaced += 1
        else:
            new_body.append(ln)
    if replaced == 0:
        raise SystemExit(f"{pack}/tensors.txt: no Engram weight line to replace")
    (out / "tensors.txt").write_text("\n".join(files + new_body) + "\n")
    print(f"{out}: {replaced} Engram layers now read %s (%s)" % (engram.name, ", ".join(sorted({v[0] for v in tensors.values()}))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
