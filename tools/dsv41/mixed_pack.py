#!/usr/bin/env python3
# Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
"""tools/dsv41/mixed_pack.py - a *static mixed* pack: the first N layers' routed experts in MXFP4, straight from
DeepSeek's own FP4 checkpoint, and every other tensor exactly as the existing IQ2 pack has it.

    python tools/dsv41/mixed_pack.py build --layers 0-19 \
        --input /data/models/dsv41-abliterated-fp8 \
        --base-pack /data/models/dsv41-iq2xxs-pack-real \
        --out /data/models/dsv41-mixed-mxfp4-l0-19
    python tools/dsv41/mixed_pack.py verify   --pack /data/models/dsv41-mixed-mxfp4-l0-19
    python tools/dsv41/mixed_pack.py selftest --pack /data/models/dsv41-mixed-mxfp4-l0-19 \
        --input /data/models/dsv41-abliterated-fp8 --samples 32

WHY THIS IS EXACT (docs/PREFILL_FP8_DESIGN.md sec.3.3).  The checkpoint's routed experts are not FP8: config.json says
`expert_dtype: fp4`, and the tensors are `w1.weight` I8 [2304][2560] (two E2M1 codes a byte, the first in the low
nibble) plus `w1.scale` F8_E8M0 [2304][160] (one E8M0 exponent a 32 values).  ggml's MXFP4 block is the same 32 values
in 17 bytes: the E8M0 scale, then 16 bytes where value j is byte j's low nibble and value j+16 its high nibble, worth
kvalues[code] * 2^(e - 128) with kvalues = 2 x E2M1 (ds4_gguf.dp.hpp:10-11).  E2M1 * 2^(e-127) is DeepSeek's own
convention, so the scale byte and the code nibbles carry over unchanged: only the *interleaving* differs, which is what
drafter_pack.py's fp4_to_mxfp4() (already used for the DSpark drafter) rewrites.  selftest decodes both sides back and
demands bit equality.

WHY A MIXED LAYER SET IS LEGAL.  The engine reads the expert type *per layer* from tensors.txt
(ds4_model.dp.cpp:2274 expert_layout_mx); the "types must not mix" check (:2306-2312) is inside one layer only, and
`expert_type(lay, r)` is consulted per layer per role (ds4_gguf.dp.hpp:44-46).  So L0-19 MXFP4 next to L20-39 IQ1_M /
IQ2_XXS / IQ2_XS needs no engine change; `l.blob` (and so the VRAM slot size) is per layer as well.

WHAT IS COPIED AND WHAT IS ONLY REFERENCED.  The pack format resolves every `file <i> <name>` against ds4.txt's
model_dir, so a pack may point at files that live elsewhere - pack_gguf.py does exactly that (it never copies a GGUF).
Default: the base pack's experts for the layers we do not convert, its dense tensors and its 202 GB Engram tables are
*referenced* by relative path (read-only, byte ranges unchanged); only the new MXFP4 blob is written.  --copy-iq2
instead byte-copies those tensors into <out>/iq2-rest.bin (another ~69 GB) and rewrites their offsets.

WHAT IS WRITTEN.  <out>/ds4.txt (the base pack's, every key and its order kept, only model_dir replaced),
<out>/tensors.txt (the new expert tensors + the base pack's lines), <out>/experts-mxfp4-l<lo>-<hi>.bin (the MXFP4
blob), <out>/engram.bin and <out>/tokenizer/ (copied - the engine reads engram.bin from the pack dir itself).
expert-profile.bin is deliberately NOT copied: it is a heat profile for the old blob sizes, and the engine falls back
to no preload when it is absent.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from drafter_pack import E2M1, fp4_to_mxfp4      # noqa: E402  the exact FP4 -> ggml MXFP4 repack (+ its value table)
from pack_gguf import TYPES as GG_TYPES          # noqa: E402  ggml block type -> (name, values a block, bytes a block)

# name -> (values a block, bytes a block), the pack's own type table (pack_gguf.py's, the one the loaders agree with)
BY_NAME = {name: (vals, blk) for (name, vals, blk) in GG_TYPES.values()}
MXFP4 = "MXFP4"
ALIGN = 256          # the engine's align256 (ds4_model.dp.cpp: expert blob offsets and read_mx's blob layout)
ALIGN_COPY = 4096    # only for --copy-iq2 (a header-free flat file we lay out ourselves)
# the engine's role order for an expert's three matrices: gate, up, down (ds4_model.dp.cpp:2284 parts[3])
ROLES = ("w1", "w3", "w2")
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")

DEVICE_TYPES = {  # ds4_iq.cu's DS4_IQ_SWITCH + ds4_gguf.cu's MXFP4: what the kernels take as experts
    "MXFP4", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ1_S", "IQ1_M", "IQ4_NL",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",   # (Q5_K/Q6_K are in supported() but not in the switch: the pack-info
}                                             #  report flags them, see DESIGN sec.8 R5 - verify() warns about them)
DENSE_TYPES = {"BF16", "F32", "F16", "MXFP8", "Q8_0"} | DEVICE_TYPES


def align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


def row_bytes(dtype: str, k: int) -> int:
    """Bytes of a row of k values of a ggml block type (0: the type or the k is not a whole number of blocks)."""
    if dtype not in BY_NAME:
        return 0
    vals, blk = BY_NAME[dtype]
    return k // vals * blk if k % vals == 0 else 0


def part_bytes(dtype: str, n: int, k: int) -> int:
    rb = row_bytes(dtype, k)
    return n * rb if rb else 0


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def gb(n: float) -> str:
    return "%.2f GB" % (n / 1e9)


# ------------------------------------------------------------------ the inputs
class BasePack:
    """The existing pack: ds4.txt (key/value lines), tensors.txt (files + tensors), and where its files really are."""

    def __init__(self, d: str):
        self.dir = os.path.abspath(d)
        self.ds4_lines = (pathlib.Path(self.dir) / "ds4.txt").read_text().splitlines()
        self.kv: dict[str, str] = {}
        for line in self.ds4_lines:
            k, _, v = line.partition(" ")
            if k:
                self.kv[k] = v
        self.model_dir = self.kv.get("model_dir", "")
        if self.kv.get("format") != "gguf":
            sys.exit("%s: format %r - the experts are read in place (mmap) only for a gguf pack"
                     % (self.dir, self.kv.get("format")))
        self.files: list[str] = []
        self.tensors: list[tuple] = []   # (name, dtype, file, off, bytes, shape)
        for line in (pathlib.Path(self.dir) / "tensors.txt").read_text().splitlines():
            p = line.split()
            if not p:
                continue
            if p[0] == "file":
                i = int(p[1])
                while len(self.files) <= i:
                    self.files.append("")
                self.files[i] = p[2]
            elif p[0] == "t":
                self.tensors.append((p[1], p[2], int(p[3]), int(p[4]), int(p[5]), [int(x) for x in p[6:]]))
        self.by_name = {t[0]: t for t in self.tensors}
        # the absolute path of each file (the pack's model_dir + the name; a name may itself be ../something)
        self.abs = [os.path.abspath(os.path.join(self.model_dir, f)) for f in self.files]

    def geo(self, key: str, default: int = 0) -> int:
        return int(self.kv.get(key, default))


class Ckpt:
    """DeepSeek's own safetensors checkpoint: the index, and each shard's header on first use."""

    def __init__(self, d: str):
        self.dir = os.path.abspath(d)
        self.index = json.loads((pathlib.Path(self.dir) / "model.safetensors.index.json").read_text())["weight_map"]
        cfg = json.loads((pathlib.Path(self.dir) / "config.json").read_text())
        if cfg.get("quantization_config", {}).get("expert_dtype") != "fp4":
            sys.exit("%s: expert_dtype is %r, not fp4 - this tool converts FP4 experts only"
                     % (self.dir, cfg.get("quantization_config", {}).get("expert_dtype")))
        self._h: dict[str, tuple[dict, int]] = {}
        self._fd: dict[str, int] = {}

    def header(self, shard: str) -> tuple[dict, int]:
        if shard not in self._h:
            with open(os.path.join(self.dir, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self._h[shard] = (h, 8 + n)
        return self._h[shard]

    def fd(self, shard: str) -> int:
        if shard not in self._fd:
            self._fd[shard] = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
        return self._fd[shard]

    def get(self, name: str) -> tuple[str, int, dict]:
        shard = self.index[name]
        h, base = self.header(shard)
        return shard, base, h[name]

    def read(self, name: str) -> np.ndarray:
        shard, base, e = self.get(name)
        a, b = e["data_offsets"]
        return pread(self.fd(shard), base + a, b - a)

    def close(self) -> None:
        for fd in self._fd.values():
            os.close(fd)
        self._fd.clear()


def pread(fd: int, off: int, n: int) -> np.ndarray:
    """n bytes at off, as a fresh uint8 array (short reads looped)."""
    out = np.empty(n, np.uint8)
    mv, got = memoryview(out), 0
    while got < n:
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            raise OSError("short read at %d (%d of %d bytes)" % (off, got, n))
        got += k
    return out


def decode_mxfp4(blob: np.ndarray, rows: int, k: int) -> np.ndarray:
    """A pack's MXFP4 matrix -> float32, exactly as ds4_gguf's mx_block does: kvalues[code] * 2^(e - 128)."""
    nb = k // 32
    blk = blob.reshape(rows, nb, 17)
    codes = np.empty((rows, nb, 32), np.uint8)
    codes[..., :16] = blk[..., 1:] & 15      # value j   -> byte j's low nibble
    codes[..., 16:] = blk[..., 1:] >> 4      # value j+16 -> byte j's high nibble
    e = np.ldexp(np.float32(1.0), blk[..., 0].astype(np.int32) - 128)[..., None]
    return E2M1[codes] * e


def decode_fp4_source(w: np.ndarray, s: np.ndarray, rows: int, k: int) -> np.ndarray:
    """The checkpoint's I8 [rows][k/2] + E8M0 [rows][k/32] -> float32, by the same convention (E2M1 * 2^(e-127))."""
    nb = k // 32
    p = w.reshape(rows, nb, 16)
    codes = np.empty((rows, nb, 32), np.uint8)
    codes[..., 0::2] = p & 15                # value 2i in byte i's low nibble
    codes[..., 1::2] = p >> 4                # value 2i+1 in its high nibble
    e = np.ldexp(np.float32(1.0), s.reshape(rows, nb).astype(np.int32) - 128)[..., None]
    return E2M1[codes] * e


# ------------------------------------------------------------------ build
def build(a) -> int:
    t_start = time.time()
    layers = parse_layers(a.layers)
    out = os.path.abspath(a.out)
    base = BasePack(a.base_pack)
    ckpt = Ckpt(a.input)

    dim, ff, nexp, nl = base.geo("hidden_size"), base.geo("moe_intermediate_size"), \
        base.geo("n_routed_experts"), base.geo("num_hidden_layers")
    if not (dim and ff and nexp and nl):
        sys.exit("%s/ds4.txt: hidden_size / moe_intermediate_size / n_routed_experts / num_hidden_layers missing"
                 % base.dir)
    bad = [L for L in layers if not 0 <= L < nl]
    if bad:
        sys.exit("--layers %s: outside 0..%d" % (bad, nl - 1))

    # the MXFP4 geometry: role -> (rows, k) as expert_layout_mx computes them (n = r<2 ? ff : dim, k = r<2 ? dim : ff)
    role_nk = {"w1": (ff, dim), "w3": (ff, dim), "w2": (dim, ff)}
    sz = {r: part_bytes(MXFP4, *role_nk[r]) for r in ROLES}
    for r in ROLES:
        if sz[r] == 0:
            sys.exit("MXFP4: %s = [%d][%d] is not a whole number of 32-value blocks" % ((r,) + role_nk[r]))
        # (the engine's blob/stride arithmetic wants a 256-byte multiple; 6266880 already is one for 5120/2304)
        if sz[r] % ALIGN:
            sys.exit("%s: %d bytes is not %d-aligned" % (r, sz[r], ALIGN))
    stride = {r: align(sz[r]) for r in ROLES}
    per_layer = sum(nexp * stride[r] for r in ROLES)
    new_bytes = per_layer * len(layers)

    bin_name = "experts-mxfp4-l%d-%d.bin" % (layers[0], layers[-1]) if len(layers) > 1 \
        else "experts-mxfp4-l%d.bin" % layers[0]

    # ---- the tensors that stay as the base pack has them
    kept = [t for t in base.tensors
            if not ((m := EXPERT_RE.match(t[0])) and int(m.group(1)) in layers)]
    dropped = len(base.tensors) - len(kept)
    if dropped != len(layers) * nexp * 3:
        sys.exit("the base pack has %d expert tensors for layers %s, expected %d"
                 % (dropped, layers, len(layers) * nexp * 3))
    need_files = sorted({t[2] for t in kept})
    for fi in need_files:
        if not base.files[fi]:
            sys.exit("base pack: file %d is referenced but never declared" % fi)
        if not os.path.exists(base.abs[fi]):
            sys.exit("base pack: %s is missing" % base.abs[fi])

    # ---- disk preflight (the brief's hard rule: > 250 GB free before writing, and at least the blob + 20 GB headroom)
    probe = os.path.dirname(out) or "."
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe) or "."
    free = shutil.disk_usage(probe).free
    copy_bytes = sum(t[4] for t in kept) if a.copy_iq2 else 0
    need = new_bytes + copy_bytes + (1 << 30)
    print("disk: %s free, this build writes %s%s (blob %s%s)"
          % (gb(free), gb(need), "" if a.copy_iq2 else " (the rest is referenced, not copied)",
             gb(new_bytes), "" if not copy_bytes else " + %s copied" % gb(copy_bytes)), flush=True)
    if free < need + 20 * (1 << 30):
        sys.exit("not enough free space: %s free, %s needed + 20 GB headroom" % (gb(free), gb(need)))
    if free < 250 * (1 << 30):
        print("WARNING: %s free, the brief asks for > 250 GB before writing" % gb(free), flush=True)

    if os.path.isdir(out) and os.listdir(out) and not a.force:
        sys.exit("%s exists and is not empty (--force overwrites this tool's own files, nothing else)" % out)
    if os.path.exists(out) and not os.path.isdir(out):
        sys.exit("%s exists and is not a directory" % out)
    os.makedirs(out, exist_ok=True)

    # ---- the output file list: 0 = the new MXFP4 blob, then the base pack's files we still need
    files: list[str] = [bin_name]
    remap: dict[int, int] = {}
    if a.copy_iq2:
        files.append("iq2-rest.bin")
        remap = {fi: 1 for fi in need_files}
    else:
        for fi in need_files:
            files.append(os.path.relpath(base.abs[fi], out))
            remap[fi] = len(files) - 1
            if " " in files[-1]:
                sys.exit("%s: a space in a file name - tensors.txt is whitespace separated" % files[-1])
    for nm in files[1:]:
        print("  file -> %s" % nm, flush=True)

    # ---- offsets of the new blob (one region a layer and role, the experts ascending: the base GGUF's own shape)
    off = {}
    cursor = 0
    for L in layers:
        for r in ROLES:
            off[(L, r)] = cursor
            cursor += nexp * stride[r]
    if cursor != new_bytes:
        sys.exit("internal: planned %d bytes, laid out %d" % (new_bytes, cursor))

    if a.dry_run:
        print("dry run: %d layers x %d experts x 3 matrices of %s (%s a layer), blob %s -> %s"
              % (len(layers), nexp, ", ".join(str(sz[r]) for r in ROLES), gb(per_layer), gb(new_bytes),
                 os.path.join(out, bin_name)))
        for L in layers[:1]:
            for r in ROLES:
                print("  L%d %s: off %d stride %d rows %d k %d" % (L, r, off[(L, r)], stride[r],
                                                                   role_nk[r][0], role_nk[r][1]))
        return 0

    # ---- the MXFP4 blob
    bin_path = os.path.join(out, bin_name)
    print("%s: %d layers x %d experts, %s total" % (bin_name, len(layers), nexp, gb(new_bytes)), flush=True)
    done = 0
    t0 = time.time()
    fh = open(bin_path, "wb")
    try:
        for L in layers:
            tl = time.time()
            for r in ROLES:
                rows, k = role_nk[r]
                for e in range(nexp):
                    pre = "layers.%d.ffn.experts.%d.%s." % (L, e, r)
                    w = ckpt.read(pre + "weight")
                    s = ckpt.read(pre + "scale")
                    if w.size != rows * (k // 2) or s.size != rows * (k // 32):
                        sys.exit("%s%s: %d + %d bytes, expected %d + %d (rows %d, k %d)"
                                 % (pre, "weight", w.size, s.size, rows * k // 2, rows * k // 32, rows, k))
                    o = fp4_to_mxfp4(w, s, rows, k)
                    mv = memoryview(o.data)
                    while mv:
                        n = fh.write(mv)
                        mv = mv[n:]
                    done += o.nbytes
            dt = time.time() - tl
            el = time.time() - t0
            rate = done / max(el, 1e-9) / 1e9
            print("  L%-2d %s / %s  %5.1fs (%.2f GB/s, %.1f s elapsed, eta %.0f s)"
                  % (L, gb(done), gb(new_bytes), dt, rate, el,
                     (new_bytes - done) / 1e9 / max(rate, 1e-9)), flush=True)
    finally:
        fh.flush()
        os.fsync(fh.fileno())
        fh.close()
    got = os.path.getsize(bin_path)
    if got != new_bytes:
        sys.exit("%s: %d bytes written, expected %d" % (bin_path, got, new_bytes))

    # ---- --copy-iq2: the kept tensors' bytes into one flat file (aligned, offsets rewritten)
    if a.copy_iq2:
        print("copying the remaining tensors into iq2-rest.bin (%s)..." % gb(copy_bytes), flush=True)
        rest = os.path.join(out, "iq2-rest.bin")
        places: dict[str, tuple[int, int]] = {}
        wr = open(rest, "wb")
        cur = 0
        try:
            for i, (name, dt, fi, o, n, shape) in enumerate(kept):
                cur = align(cur, ALIGN_COPY)             # a flat file's own offsets: keep every tensor aligned
                b = _read_base(base, fi, o, n)
                wr.seek(cur)
                mv = memoryview(b.data)
                while mv:
                    k = wr.write(mv)
                    mv = mv[k:]
                places[name] = (cur, n)
                cur += n
                if (i + 1) % 5000 == 0:
                    print("  %d / %d tensors, %s" % (i + 1, len(kept), gb(cur)), flush=True)
        finally:
            wr.flush()
            os.fsync(wr.fileno())
            wr.close()
        for fd in getattr(base, "_fds", {}).values():
            os.close(fd)
        kept = [(name, dt, remap[fi], places[name][0], n, shape) for (name, dt, fi, o, n, shape) in kept]
        print("  iq2-rest.bin: %s" % gb(os.path.getsize(rest)), flush=True)
    else:
        kept = [(name, dt, remap[fi], o, n, shape) for (name, dt, fi, o, n, shape) in kept]

    # ---- the new expert tensors (name, dtype, file 0, off, bytes, shape) - torch order [out][in]
    fresh = []
    for L in layers:
        for r in ROLES:
            rows, k = role_nk[r]
            for e in range(nexp):
                fresh.append(("layers.%d.ffn.experts.%d.%s.weight" % (L, e, r), MXFP4, 0,
                              off[(L, r)] + e * stride[r], sz[r], [rows, k]))

    # ---- ds4.txt: the base pack's, key for key and in its order, only model_dir replaced
    with open(os.path.join(out, "ds4.txt"), "w") as f:
        for line in base.ds4_lines:
            k, _, v = line.partition(" ")
            f.write("model_dir %s\n" % out if k == "model_dir" else line + "\n")
    # every key still there (a dropped key changes the geometry the engine computes)
    got_kv = {}
    for line in open(os.path.join(out, "ds4.txt")).read().splitlines():
        k, _, v = line.partition(" ")
        got_kv[k] = v
    if set(got_kv) != set(base.kv):
        sys.exit("ds4.txt: keys changed: %s" % (set(base.kv) ^ set(got_kv)))

    # ---- tensors.txt
    all_t = kept + fresh
    with open(os.path.join(out, "tensors.txt"), "w") as f:
        for i, nm in enumerate(files):
            f.write("file %d %s\n" % (i, nm))
        for name, dt, fi, o, n, shape in all_t:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, o, n, " ".join(str(x) for x in shape)))

    # ---- the small files the engine reads from the pack dir itself
    for small in ("engram.bin",):
        src = os.path.join(base.dir, small)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, small))
    tok = os.path.join(base.dir, "tokenizer")
    if os.path.isdir(tok) and not os.path.isdir(os.path.join(out, "tokenizer")):
        shutil.copytree(tok, os.path.join(out, "tokenizer"))

    ckpt.close()
    print("\nbuilt %s in %.1f s" % (out, time.time() - t_start))
    print("  ds4.txt / tensors.txt / %s (%s) / engram.bin%s"
          % (bin_name, gb(new_bytes), " / tokenizer" if os.path.isdir(os.path.join(out, "tokenizer")) else ""))
    print("  %d tensors in %d files (the base pack's %d kept, %d replaced by MXFP4, %d new)"
          % (len(all_t), len(files), len(kept), dropped, len(fresh)))
    return verify(a.out, layers)


def _fd_of(base: BasePack, fi: int) -> int:
    if not hasattr(base, "_fds"):
        base._fds = {}
    if fi not in base._fds:
        base._fds[fi] = os.open(base.abs[fi], os.O_RDONLY)
    return base._fds[fi]


def _read_base(base: BasePack, fi: int, off: int, n: int) -> np.ndarray:
    return pread(_fd_of(base, fi), off, n)


# ------------------------------------------------------------------ verify
def verify(pack: str, expect_layers=None) -> int:
    """The checks the engine makes, offline: every tensor inside its file, every layer's experts where
    expert_layout_mx demands them (one file, one type, a uniform stride), the type map a layer and role, the blob.
    Same arithmetic as ds4_model.dp.cpp:2274-2337.  Returns 0 when the pack loads."""
    d = os.path.abspath(pack)
    kv: dict[str, str] = {}
    for line in open(os.path.join(d, "ds4.txt")).read().splitlines():
        k, _, v = line.partition(" ")
        if k:
            kv[k] = v
    md = kv.get("model_dir", "")
    dim, ff, nexp, nl = (int(kv.get("hidden_size", 0)), int(kv.get("moe_intermediate_size", 0)),
                         int(kv.get("n_routed_experts", 0)), int(kv.get("num_hidden_layers", 0)))
    files, tensors = [], {}
    for line in open(os.path.join(d, "tensors.txt")).read().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "file":
            while len(files) <= int(p[1]):
                files.append("")
            files[int(p[1])] = p[2]
        elif p[0] == "t":
            tensors[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]), [int(x) for x in p[6:]])
    problems: list[str] = []

    # every file present and big enough for the tensors that point into it
    need: dict[int, int] = {}
    for name, (dt, fi, off, n, shape) in tensors.items():
        need[fi] = max(need.get(fi, 0), off + n)
    sizes = {}
    for i, nm in enumerate(files):
        p = os.path.abspath(os.path.join(md, nm))
        if not os.path.exists(p):
            problems.append("file %d %s: missing (%s)" % (i, nm, p))
            sizes[i] = 0
        else:
            sizes[i] = os.path.getsize(p)
            if sizes[i] < need.get(i, 0):
                problems.append("file %d %s: %d bytes, the tensors need %d" % (i, nm, sizes[i], need.get(i, 0)))

    # the expert type map, layer by layer, role by role (the engine's parts order)
    role_nk = {"w1": (ff, dim), "w3": (ff, dim), "w2": (dim, ff)}
    map_txt, blobs = [], {}
    mixed = 0
    for L in range(nl):
        ks, blob, off = [], 0, 0
        for r in ROLES:
            rows, k = role_nk[r]
            ts = [(e, tensors.get("layers.%d.ffn.experts.%d.%s.weight" % (L, e, r)))
                  for e in range(nexp)]
            miss = [e for e, t in ts if t is None]
            if miss:
                problems.append("L%d %s: %d of %d experts missing (first %s)" % (L, r, len(miss), nexp, miss[:3]))
                ks.append("?")
                continue
            dts = {t[0] for _, t in ts}
            if len(dts) != 1:
                problems.append("L%d %s: types %s inside one layer" % (L, r, sorted(dts)))
                mixed += 1
            dt = ts[0][1][0]
            want = part_bytes(dt, rows, k)
            if want == 0:
                problems.append("L%d %s: type %s is not a whole number of blocks for [%d][%d]" % (L, r, dt, rows, k))
            for e, t in ts:
                if t[0] != dt or t[3] != want:
                    problems.append("L%d %s: expert %d is %s/%d bytes, the layer's others %s/%d"
                                    % (L, r, e, t[0], t[3], dt, want))
                    break
                if t[4] != [rows, k]:
                    problems.append("L%d %s: expert %d shape %s, expected [%d][%d]" % (L, r, e, t[4], rows, k))
                    break
            fl = {t[1] for _, t in ts}
            if len(fl) != 1:
                problems.append("L%d %s: experts in files %s (expert_layout_mx wants one)" % (L, r, sorted(fl)))
                ks.append(dt)
                continue
            st = {ts[e + 1][1][2] - ts[e][1][2] for e in range(nexp - 1)}
            if len(st) != 1:
                problems.append("L%d %s: expert offsets are not one stride (%d distinct)" % (L, r, len(st)))
            st0 = st.pop() if st else want
            for e in range(nexp):
                if ts[e][1][2] != ts[0][1][2] + e * st0:
                    problems.append("L%d %s: expert %d is not off0 + %d stride" % (L, r, e, e))
                    break
            if st0 < want:
                problems.append("L%d %s: stride %d < the matrix %d bytes" % (L, r, st0, want))
            ks.append(dt)
            off += align(want)          # + ExpertStore::kSlack when the RAM store is on
        blobs[L] = off
        map_txt.append("L%d %s blob %d" % (L, "/".join(ks), off))

    # what the device kernels do not take (they go to the CPU lane; MXFP4 mixed with an i-quant inside a layer would
    # be refused at load, and is the one mix the engine rejects)
    cpu_only = sorted({t[0] for name, t in tensors.items() if EXPERT_RE.match(name) and t[0] not in DEVICE_TYPES})
    warn_r5 = sorted({t[0] for name, t in tensors.items() if EXPERT_RE.match(name) and t[0] in ("Q5_K", "Q6_K")})
    # (the Engram tables are their own path - ds4_engram_io's MXFP8 row format, not gg::rows - so they are exempt)
    dense_bad = sorted({t[0] for name, t in tensors.items()
                        if not EXPERT_RE.match(name) and ".engram.embed." not in name and t[0] not in DENSE_TYPES})

    n_ex = sum(1 for name in tensors if EXPERT_RE.match(name))
    print("pack %s" % d)
    print("  %d tensors in %d files; %d expert tensors, %d other" % (len(tensors), len(files), n_ex,
                                                                     len(tensors) - n_ex))
    print("  model_dir %s, format %s, %d layers x %d experts, dim %d, ff %d"
          % (md, kv.get("format"), nl, nexp, dim, ff))
    # the per-layer type map, one line a layer
    print("  expert type map (gate/up/down, blob bytes):")
    for L in range(nl):
        print("    " + map_txt[L])
    if expect_layers is not None:
        for L in range(nl):
            want = "MXFP4" if L in expect_layers else None
            got = map_txt[L].split()[1].split("/")
            if want and got != [want] * 3:
                problems.append("L%d: expected MXFP4/×3, got %s" % (L, "/".join(got)))
            if not want and "MXFP4" in got:
                problems.append("L%d: MXFP4 outside the converted layers (%s)" % (L, "/".join(got)))

    types: dict[str, float] = {}
    for name, (dt, fi, off, n, shape) in tensors.items():
        types[dt] = types.get(dt, 0) + n
    print("  bytes a type: %s" % ", ".join("%s %s" % (k, gb(v)) for k, v in sorted(types.items())))
    print("  total referenced: %s in %s on this disk" % (gb(sum(types.values())),
                                                         gb(sum(sizes.values()))))
    print("  blobs: min %d max %d (the store's slot size follows the max)"
          % (min(blobs.values()), max(blobs.values())))
    if cpu_only:
        print("  no device kernel (CPU lane): %s" % ", ".join(cpu_only))
    if warn_r5:
        print("  WARNING: %s as an expert type: in supported() but not in DS4_IQ_SWITCH (DESIGN sec.8 R5)"
              % ", ".join(warn_r5))
    if dense_bad:
        problems.append("dense types the dense path does not read: %s" % ", ".join(dense_bad))
    for p in problems[:40]:
        print("  PROBLEM: %s" % p)
    if len(problems) > 40:
        print("  ... %d more problems" % (len(problems) - 40))
    print("  %s" % ("OK: this pack loads" if not problems else "%d PROBLEMS" % len(problems)))
    return 0 if not problems else 1


# ------------------------------------------------------------------ selftest
def selftest(a) -> int:
    """Bit equality: sample experts, decode them from the pack's blob and from the checkpoint, compare.
    The blob's blocks are read where the engine reads them (file, mx_off + e * mx_stride)."""
    d = os.path.abspath(a.pack)
    kv = {}
    for line in open(os.path.join(d, "ds4.txt")).read().splitlines():
        k, _, v = line.partition(" ")
        if k:
            kv[k] = v
    dim, ff, nexp, nl = (int(kv["hidden_size"]), int(kv["moe_intermediate_size"]),
                         int(kv["n_routed_experts"]), int(kv["num_hidden_layers"]))
    md = kv.get("model_dir", "")
    files, tensors = [], {}
    for line in open(os.path.join(d, "tensors.txt")).read().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "file":
            while len(files) <= int(p[1]):
                files.append("")
            files[int(p[1])] = p[2]
        elif p[0] == "t":
            tensors[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]))
    ckpt = Ckpt(a.input)
    role_nk = {"w1": (ff, dim), "w3": (ff, dim), "w2": (dim, ff)}
    mx_layers = [L for L in range(nl)
                 if all(tensors.get("layers.%d.ffn.experts.0.%s.weight" % (L, r), ("",))[0] == MXFP4 for r in ROLES)]
    if a.layers:
        only = set(parse_layers(a.layers))
        mx_layers = [L for L in mx_layers if L in only]
    if not mx_layers:
        sys.exit("no layer has MXFP4 experts")
    fds = {}

    def fd(i):
        if i not in fds:
            fds[i] = os.open(os.path.abspath(os.path.join(md, files[i])), os.O_RDONLY)
        return fds[i]

    rng = np.random.default_rng(a.seed)
    picks = []
    per = max(1, a.samples // len(mx_layers))
    for L in mx_layers:
        picks += [(L, int(e)) for e in rng.choice(nexp, size=min(per, nexp), replace=False)]
    picks = picks[:a.samples] if a.samples else picks
    worst_f32 = 0.0
    worst_f16 = 0.0
    unequal = 0
    n_val = 0
    for L, e in picks:
        for r in ROLES:
            rows, k = role_nk[r]
            nb = k // 32
            t0 = tensors["layers.%d.ffn.experts.0.%s.weight" % (L, r)]
            te = tensors["layers.%d.ffn.experts.%d.%s.weight" % (L, e, r)]
            stride = tensors["layers.%d.ffn.experts.1.%s.weight" % (L, r)][2] - t0[2]
            blob = pread(fd(te[1]), te[2], te[3])
            a32 = decode_mxfp4(blob, rows, k)
            w = ckpt.read("layers.%d.ffn.experts.%d.%s.weight" % (L, e, r))
            s = ckpt.read("layers.%d.ffn.experts.%d.%s.scale" % (L, e, r))
            b32 = decode_fp4_source(w, s, rows, k)
            n_val += a32.size
            bad = int(np.count_nonzero(a32 != b32))
            unequal += bad
            df32 = float(np.max(np.abs(a32 - b32))) if bad else 0.0
            a16, b16 = a32.astype(np.float16), b32.astype(np.float16)
            df16 = float(np.max(np.abs(a16.astype(np.float32) - b16.astype(np.float32))))
            worst_f32 = max(worst_f32, df32)
            worst_f16 = max(worst_f16, df16)
            if int(te[2]) != int(t0[2]) + e * stride:      # the engine's own arithmetic: off0 + e * stride
                print("  L%d e%d %s: offset %d is not off0 + %d * stride" % (L, e, r, te[2], e))
            if bad:
                print("  L%d e%d %s: %d of %d values differ, max |err| %g" % (L, e, r, bad, a32.size, df32))
    ckpt.close()
    print("selftest: %d experts x 3 matrices, %d values, layers %s" % (len(picks), n_val, mx_layers))
    print("  max |err| (F32): %g    max |err| (F16): %g    differing values: %d" % (worst_f32, worst_f16, unequal))
    ok = unequal == 0 and worst_f16 == 0.0
    print("  %s" % ("EXACT: the pack's MXFP4 is the checkpoint's FP4, bit for bit" if ok else "NOT EXACT"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write a mixed pack")
    b.add_argument("--input", required=True, help="DeepSeek's FP4 safetensors checkpoint")
    b.add_argument("--base-pack", required=True, help="the existing IQ2 pack (read only)")
    b.add_argument("--out", required=True, help="the new pack's directory")
    b.add_argument("--layers", default="0-19", help="the layers whose experts become MXFP4 (default 0-19)")
    b.add_argument("--copy-iq2", action="store_true",
                   help="byte-copy the remaining tensors into iq2-rest.bin instead of referencing the base pack")
    b.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    b.add_argument("--force", action="store_true", help="write into a non-empty --out (this tool's files only)")
    b.set_defaults(func=lambda a: build(a))

    v = sub.add_parser("verify", help="the engine's checks, offline")
    v.add_argument("--pack", required=True)
    v.add_argument("--expect-layers", default=None, help="the layers that must be MXFP4 (e.g. 0-19)")
    v.set_defaults(func=lambda a: verify(a.pack, parse_layers(a.expect_layers) if a.expect_layers else None))

    s = sub.add_parser("selftest", help="decode sampled experts from the pack and the checkpoint, compare")
    s.add_argument("--pack", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--layers", default=None, help="only these MXFP4 layers (default: all of them)")
    s.add_argument("--samples", type=int, default=32)
    s.add_argument("--seed", type=int, default=1234)
    s.set_defaults(func=selftest)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
