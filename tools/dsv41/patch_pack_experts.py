#!/usr/bin/env python3
"""patch_pack_experts.py - build a test pack from an existing one with some experts re-pointed at a requantized file.

    python tools/dsv41/patch_pack_experts.py --pack PACK --out DIR --override FILE.bin --json FILE.json [--file-index 9]

Copies ds4.txt and tensors.txt, symlinks every other file the pack references, and rewrites the tensors the JSON lists
(type, file index, offset, bytes).  The engine then loads a pack whose experts are part of the new type - a per-layer
test bed, no big copy.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", required=True, help="the .bin expert_requant wrote")
    ap.add_argument("--json", required=True, help="its .json")
    ap.add_argument("--file-index", type=int, default=9)
    args = ap.parse_args()

    pack = pathlib.Path(args.pack)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    over = json.loads(pathlib.Path(args.json).read_text())

    # ds4.txt verbatim (the model_dir line keeps pointing at the shards)
    shutil.copyfile(pack / "ds4.txt", out / "ds4.txt")
    model_dir = None
    for line in (pack / "ds4.txt").read_text().splitlines():
        f = line.split()
        if len(f) >= 2 and f[0] == "model_dir":
            model_dir = f[1]
            break
    if model_dir is None:
        model_dir = str(pack)

    # tensors.txt: the file table, then the tensors (rewritten where the override has them)
    src = (pack / "tensors.txt").read_text().splitlines()
    files, tensors = [], []
    for line in src:
        f = line.split()
        if not f:
            continue
        if f[0] == "file":
            files.append((int(f[1]), f[2]))
        elif f[0] == "t":
            tensors.append(f)

    # symlink the referenced shards (and anything else next to the pack: engram, tokenizer)
    linked = set()
    for _, name in files:
        p = pathlib.Path(name)
        if not p.is_absolute():
            cand = pathlib.Path(model_dir) / name
            if not cand.exists():
                cand = pack / name
            p = cand
        if p.exists() and p.name not in linked:
            dst = out / p.name
            if not dst.exists():
                os.symlink(p.resolve(), dst)
            linked.add(p.name)
    for extra in ("engram.bin", "tokenizer", "chat_template.jinja", "tokenizer.json"):
        p = pack / extra
        if p.exists() and not (out / extra).exists():
            os.symlink(p.resolve(), out / extra)

    # the pack resolves every `file` name against ds4.txt's model_dir (an absolute one included: it is prefixed
    # too), so the override is named relative to it - the same form the engram stub line uses
    ov = pathlib.Path(args.override).resolve()
    name = os.path.relpath(ov, model_dir)
    files.append((args.file_index, name))
    n_done = 0
    lines = ["file %d %s" % (i, name) for i, name in sorted(files)]
    for t in tensors:
        name = t[1]
        if name in over:
            ty, off, nbytes = over[name][0], int(over[name][1]), int(over[name][2])
            t = [t[0], name, ty, str(args.file_index), str(off), str(nbytes)] + t[6:]
            n_done += 1
        lines.append(" ".join(t))
    (out / "tensors.txt").write_text("\n".join(lines) + "\n")
    print("%s: %d tensors re-pointed at %s (file %d), %d files linked" %
          (out, n_done, pathlib.Path(args.override).name, args.file_index, len(linked)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
