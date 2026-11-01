#!/usr/bin/env python3
"""make_engram_stub.py - a sparse, all-zero Engram table set for a first light of the DeepSeek-V4.1 engine.

`strata-dsv41` reads the Engram rows per token and refuses a pack whose Engram layer has no `engram.embed.weight`
(ds4_model.cu: MXFP8, Q6_K or IQ4_NL only).  The tables are the 111-203 GB half of the model, and the shapes/metadata are
the only thing the engine validates - so for "does the whole engine path run on this card" a table of the right SHAPE
whose bytes are a hole in the file is enough: every row reads as zeros, the Engram adds nothing, and everything else
(attention, MoE routing, the CPU lane, sampling) is the real model.

    python3 tools/dsv41/make_engram_stub.py --out /data/models/dsv41-engram-stub [--rows 1:384006168 14:384016682]

Writes <out>/engram-q6k-stub.gguf: architecture "deepseek41-engram", one Q6_K `blk.<layer>.engram_embd.weight` per layer
of head_dim 256 (210 bytes a row), the tensor data a sparse hole (`f.truncate`), so the file's apparent size is the real
table's and its disk usage is the header's.  Pack it with `pack_gguf.py ... --engram <out>`.
"""
from __future__ import annotations

import argparse
import pathlib
import struct

Q6K_ROW_BYTES = 210      # 256 values a Q6_K super-block
HEAD_DIM = 256
GGUF_VERSION = 3
Q6K_TYPE = 14
ALIGN = 32


def s(f, text: str) -> None:
    b = text.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="engram-q6k-stub.gguf")
    ap.add_argument("--rows", nargs="+", default=["1:384006168", "14:384016682"],
                    help="layer:rows (the model's deepseek41.engram.num_embeddings)")
    a = ap.parse_args()
    layers = []
    for spec in a.rows:
        layer, rows = spec.split(":")
        layers.append((int(layer), int(rows)))
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / a.name

    infos = []
    off = 0
    for layer, rows in layers:
        off = (off + ALIGN - 1) // ALIGN * ALIGN
        infos.append((f"blk.{layer}.engram_embd.weight", rows, off))
        off += rows * Q6K_ROW_BYTES
    total = (off + ALIGN - 1) // ALIGN * ALIGN

    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", GGUF_VERSION, len(infos), 1))
        # metadata: the architecture is what makes pack_gguf's is_engram_file() take this set as the Engram tables
        s(f, "general.architecture")
        f.write(struct.pack("<I", 8))       # GGUF_METADATA_VALUE_TYPE_STRING
        s(f, "deepseek41-engram")
        for name, rows, toff in infos:
            s(f, name)
            f.write(struct.pack("<I", 2))                       # n_dims
            f.write(struct.pack("<QQ", HEAD_DIM, rows))          # dims, outermost first (torch order)
            f.write(struct.pack("<I", Q6K_TYPE))
            f.write(struct.pack("<Q", toff))                     # offset from the data section
        pos = f.tell()
        data_start = (pos + ALIGN - 1) // ALIGN * ALIGN
        f.write(b"\0" * (data_start - pos))
        # the rows are a hole: reading them gives zeros, the apparent size is the real table's
        f.truncate(data_start + total)
    st = path.stat()
    print("%s: %d layers, %d tensors, apparent %.1f GiB, on disk %.1f MiB" %
          (path, len(layers), len(infos), st.st_size / 2**30, st.st_blocks * 512 / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
