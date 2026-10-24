#!/usr/bin/env python3
"""tools/v41/ternary/mesh_pack.py - meshapplied's ternary experts (`ternary-g128-hadamard-gptq`) for the V4.1 engine.

Source: huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest (layer-LL.safetensors + manifest-layer-LL.json),
format spec github.com/meshapplied/goethe docs/pack.md, runtime ternary-kernels/ternary_runtime (rotate kernel in
csrc/ternary_moe.cu, reference unpack kernels.py:ternary_grouped_ref).  Built from the STOCK checkpoint; dealignai's
abliteration left the routed experts byte-identical (its card: "Routed MoE experts - untouched"), so they drop into
our GSQ unchanged.

The format, as the runtime implements it:
  - per expert keys layers.L.ffn.experts.E.{w1,w2,w3}.{weight,scale}; weight int32 [N, K/16] = 16 two-bit codes per
    word, element k in word k/16 at bits 2*(k%16), value = code - 1 in {-1,0,1}; scale fp16 [N, K/128]
  - the stored matrix is W R^T with R = blockdiag(H_b diag(s)) / sqrt(b): b = 1024 for K = 5120 (w1/w3), 256 for
    K = 2304 (w2); s = `hadamard.signs.<K>` (b values, the SAME vector for every block); H_b = natural-order
    Walsh-Hadamard (the in-place butterfly a+b / a-b).  The runtime feeds R x to the stored matrix.
  - experts listed as FP4 fallback in the manifest stay the base checkpoint's (no ternary copy).

Two products:
  derotated(L, e)   float32 W1, W3, W2 in OUR basis (W_eff = stored @ R): the screen compares these with our GSQ
                    experts on real FFN inputs (mesh_screen.py).
  ptq1_rows(L, e)   the codes + scales repacked LOSSLESSLY as PTQ1_0 rows (our block is exactly 128 trits x one fp16
                    scale): the engine would serve them as a new class "PTQ1_0 in the Hadamard basis" (the trits
                    stay rotated; the kernel rotates its activations - see MESH_PLAN.md).

    python3 mesh_pack.py check PACKDIR [--layer 5]     self-test on a real layer file (CPU, seconds)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import v41_formats as F  # noqa: E402

HIDDEN, INTER = 5120, 2304
FORMAT = "ternary-g128-hadamard-gptq"


def block_of(K: int) -> int:
    """b = min(1024, the largest power of two dividing K)."""
    return min(1024, K & -K)


def fwht_rows(x: np.ndarray, b: int) -> np.ndarray:
    """Natural-order Walsh-Hadamard over consecutive length-b blocks of the last axis, unnormalised (the butterfly of
    csrc/ternary_moe.cu rotate_kernel: pairs (i, i + h), h = 1, 2, 4, .., a+b / a-b)."""
    sh = x.shape
    y = np.array(x, dtype=np.float64).reshape(-1, sh[-1] // b, b)
    h = 1
    while h < b:
        y = y.reshape(y.shape[0], y.shape[1], b // (2 * h), 2, h)
        a, c = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + c, a - c
        y = y.reshape(y.shape[0], y.shape[1], b)
        h *= 2
    return y.reshape(sh)


def rotate(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """R x per row (the runtime's activation transform): FWHT(x * s) / sqrt(b), s repeated per block."""
    b = len(signs)
    xs = np.asarray(x, np.float64) * np.tile(signs, x.shape[-1] // b)
    return fwht_rows(xs, b) / np.sqrt(b)


def derotate_rows(w_stored: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """W_eff = stored @ R: per row block v -> (H v) * s / sqrt(b) (H symmetric), so W_eff x == stored (R x)."""
    b = len(signs)
    return (fwht_rows(w_stored, b) * np.tile(signs, w_stored.shape[-1] // b) / np.sqrt(b)).astype(np.float32)


def unpack_codes(words: np.ndarray) -> np.ndarray:
    """int32 [N, K/16] -> int8 [N, K] in {-1, 0, 1}."""
    w = words.astype(np.int64) & 0xFFFFFFFF
    shifts = (np.arange(16, dtype=np.int64) * 2)
    codes = ((w[..., None] >> shifts) & 3).astype(np.int8) - 1
    if (codes > 1).any():
        raise ValueError("code 3 (value +2) in a ternary pack")
    return codes.reshape(words.shape[0], -1)


class MeshLayer:
    """One layer-LL.safetensors (+ manifest).  Lazily reads experts."""

    def __init__(self, packdir: str, layer: int):
        from safetensors import safe_open
        self.layer = layer
        self.path = os.path.join(packdir, f"layer-{layer:02d}.safetensors")
        self.f = safe_open(self.path, framework="numpy")
        meta = self.f.metadata() or {}
        if meta.get("format", FORMAT) != FORMAT or str(meta.get("group", "128")) != "128":
            raise ValueError(f"{self.path}: format {meta.get('format')} group {meta.get('group')}")
        man = os.path.join(packdir, f"manifest-layer-{layer:02d}.json")
        self.manifest = json.load(open(man)) if os.path.exists(man) else {}
        keys = set(self.f.keys())
        self.signs = {K: self.f.get_tensor(f"hadamard.signs.{K}").astype(np.float64)
                      for K in (HIDDEN, INTER) if f"hadamard.signs.{K}" in keys}
        for K, s in self.signs.items():
            if len(s) != block_of(K) or not np.all(np.abs(s) == 1):
                raise ValueError(f"signs.{K}: {len(s)} values, expected +-1 x {block_of(K)}")
        self.ternary = sorted({int(k.split(".")[4]) for k in keys if k.startswith(f"layers.{layer}.ffn.experts.")})

    def has(self, e: int) -> bool:
        return e in self.ternary

    def raw(self, e: int, proj: str):
        p = f"layers.{self.layer}.ffn.experts.{e}.{proj}"
        return unpack_codes(self.f.get_tensor(p + ".weight")), self.f.get_tensor(p + ".scale").astype(np.float32)

    def stored(self, e: int, proj: str) -> np.ndarray:
        codes, sc = self.raw(e, proj)
        return codes.astype(np.float32) * np.repeat(sc, 128, axis=1)

    def derotated(self, e: int):
        """(W1, W3, W2) float32 in the model's own basis: W1/W3 [2304, 5120], W2 [5120, 2304]."""
        return tuple(derotate_rows(self.stored(e, p), self.signs[HIDDEN if p != "w2" else INTER])
                     for p in ("w1", "w3", "w2"))

    def ptq1_rows(self, e: int, proj: str) -> bytes:
        """The projection as PTQ1_0 rows (lossless: same trits, same fp16 scales, still in the rotated basis)."""
        codes, sc = self.raw(e, proj)
        return F.pack_ptq1_0(codes, sc.astype(np.float16)).tobytes()


def cmd_check(a):
    m = MeshLayer(a.packdir, a.layer)
    print(f"{m.path}: {len(m.ternary)} ternary experts; signs {sorted(m.signs)}; manifest keys {list(m.manifest)[:6]}")
    rng = np.random.default_rng(0)
    # 1. the rotation is orthonormal and derotate is its adjoint: W_eff x == stored (R x)
    for K in (HIDDEN, INTER):
        s = m.signs[K]
        x = rng.standard_normal((3, K))
        assert np.allclose(np.linalg.norm(rotate(x, s), axis=1), np.linalg.norm(x, axis=1)), "R not orthonormal"
        Ws = rng.standard_normal((4, K))
        assert np.allclose(derotate_rows(Ws, s) @ x.T, Ws @ rotate(x, s).T, atol=1e-6), "derotate != adjoint"
    # 2. on a real expert: PTQ1_0 repack decodes to exactly the stored values, and W_eff x == stored (R x)
    e = m.ternary[len(m.ternary) // 2]
    for proj in ("w1", "w3", "w2"):
        stored = m.stored(e, proj)
        rows = np.frombuffer(m.ptq1_rows(e, proj), np.uint8).reshape(stored.shape[0], -1)
        dec = np.stack([F.decode(F.PTQ1_0, r, stored.shape[1]) for r in rows[:64]])
        assert np.array_equal(dec, stored[:64]), f"{proj}: PTQ1_0 repack is not lossless"
        K = stored.shape[1]
        x = rng.standard_normal((2, K))
        W = derotate_rows(stored, m.signs[K])
        rel = np.abs(W @ x.T - stored @ rotate(x, m.signs[K]).T).max() / np.abs(stored @ rotate(x, m.signs[K]).T).max()
        assert rel < 1e-5, f"{proj}: derotated product differs (rel {rel:.2e})"
    print(f"OK: rotation orthonormal, derotate == adjoint, expert {e} w1/w3/w2 repack lossless + products match")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("packdir")
    c.add_argument("--layer", type=int, default=5)
    a = ap.parse_args()
    {"check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
