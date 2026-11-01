#!/usr/bin/env python3
"""engram_from_safetensors.py - the Engram table as a Strata pack reads it, copied out of the FP8 safetensors.

`strata-dsv41` reads an Engram layer as one of MXFP8 / Q6_K / IQ4_NL, or - when the pack also carries the layer's
`engram.embed.scale` - as FP8 codes plus their E8M0 scales ("an old pack", ds4_model.cu's fmt 3).  The FP8 checkpoint on
this PC has exactly that: `layers.{1,14}.engram.embed.weight` as F8_E4M3 and `...embed.scale` as F8_E8M0 (256 values a
row, 8 scale bytes a row).

So this is a BYTE COPY, not a requantisation (the Q4_K -> IQ4_NL path costs hours of CPU; the copy is disk-bound):

    python3 tools/dsv41/engram_from_safetensors.py \\
        --src /data/models/dsv41-abliterated-fp8 --out /home/dyy/engram-mxfp8/engram-mxfp8.gguf

The output is a GGUF with architecture "deepseek41-engram" and, a layer, `blk.<l>.engram_embd.weight` (type 32, one byte
a value) and `blk.<l>.engram_embd.scale` (type 34, one byte a 32 values).  pack_gguf.py does not know either name yet, so
the pack's tensors.txt is edited to point the two engram tensors at this file (tools/dsv41/engram_pack_patch.py).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import sys

GGUF_VERSION = 3
ALIGN = 32
F8_E4M3 = 32      # this fork of ggml has no FP8 type id (it was removed upstream); the pack records its own name
F8_E8M0 = 34


def s(f, text: str) -> None:
    b = text.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def safetensors_index(src: pathlib.Path) -> dict:
    """name -> (shard path, dtype, shape, data offset inside the shard's data section)."""
    out = {}
    for shard in sorted(src.glob("*.safetensors")):
        with shard.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            base = 8 + n
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            (b, e) = info["data_offsets"]
            out[name] = (shard, info["dtype"], tuple(info["shape"]), base + b, e - b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the FP8 safetensors folder")
    ap.add_argument("--out", required=True, help="the .gguf to write (needs ~203 GB free)")
    ap.add_argument("--layers", default="1,14")
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args()
    src = pathlib.Path(a.src).expanduser()
    out = pathlib.Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]

    idx = safetensors_index(src)
    rows_of, parts = {}, []
    for l in layers:
        w = f"layers.{l}.engram.embed.weight"
        sc = f"layers.{l}.engram.embed.scale"
        if w not in idx or sc not in idx:
            raise SystemExit(f"{src}: {w} / {sc} not in the checkpoint")
        rows = idx[w][2][0]
        rows_of[l] = rows
        parts.append((f"blk.{l}.engram_embd.weight", w, rows, rows * idx[w][2][1], F8_E4M3, idx[w][2]))
        parts.append((f"blk.{l}.engram_embd.scale", sc, rows, rows * idx[sc][2][1], F8_E8M0, idx[sc][2]))

    # header
    offs, off = [], 0
    for name, _, _, nbytes, _, _ in parts:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        offs.append(off)
        off += nbytes
    total = (off + ALIGN - 1) // ALIGN * ALIGN
    with out.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", GGUF_VERSION, len(parts), 1))
        s(f, "general.architecture")
        f.write(struct.pack("<I", 8))
        s(f, "deepseek41-engram")
        for (name, _, _, nbytes, typ, shape), o in zip(parts, offs):
            s(f, name)
            f.write(struct.pack("<I", len(shape)))
            for d in shape:                      # torch order: outermost first
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", typ))
            f.write(struct.pack("<Q", o))
        pos = f.tell()
        data_start = (pos + ALIGN - 1) // ALIGN * ALIGN
        f.write(b"\0" * (data_start - pos))

    # the bytes
    chunk = a.chunk_mb << 20
    buf = bytearray(chunk)
    view = memoryview(buf)
    with out.open("r+b") as fo:
        for (name, key, rows, nbytes, typ, _), o in zip(parts, offs):
            shard, _, _, base, _ = idx[key]
            done = 0
            with shard.open("rb") as fi:
                fi.seek(base)
                fo.seek(data_start + o)
                while done < nbytes:
                    n = min(chunk, nbytes - done)
                    got = fi.readinto(view[:n])
                    if not got:
                        raise SystemExit(f"{shard}: short read at {done} of {nbytes}")
                    fo.write(view[:got])
                    done += got
            print(f"  {name}: {nbytes / 2**30:.1f} GiB", flush=True)
    print(f"{out}: {total / 2**30:.1f} GiB apparent, {out.stat().st_size / 2**30:.1f} GiB written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
