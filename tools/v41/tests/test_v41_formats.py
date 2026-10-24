"""test_v41_formats.py - v41_formats' encoders, bit for bit against the C references (fmt_ref.cpp: ggml's dequantizers
and quantize_row_q8_0_ref, Strata's ptq1_0_pack / ptq1_0_dequant), plus the GPTQ-output path (encode_tier).

    python3 tools/v41/tests/test_v41_formats.py        (CPU only; builds fmt_ref once into $V41_FMT_REF_DIR or /tmp)
"""
import os
import subprocess
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "v41"))
import v41_formats as F  # noqa: E402
import v41_gptq as G  # noqa: E402

GGML = os.path.join(REPO, "third_party", "llama.cpp", "ggml")
LIBS = [os.path.join(REPO, "..", "Strata-DS4-up", "build-gpu", "ggml", "src", "libggml-base.a"),
        os.path.join(REPO, "build-ds4-gpu", "ggml", "src", "libggml-base.a")]


def build_ref():
    out = os.path.join(os.environ.get("V41_FMT_REF_DIR", "/tmp"), f"v41_fmt_ref_{os.getuid()}")
    src = [os.path.join(HERE, "fmt_ref.cpp"), os.path.join(REPO, "src", "kernels", "cpu", "ptq1_0.cpp")]
    if os.path.exists(out) and all(os.path.getmtime(out) > os.path.getmtime(s) for s in src):
        return out
    lib = next(p for p in LIBS if os.path.exists(p))
    subprocess.run(["g++", "-O2", "-std=c++17", "-mavx2", "-mfma", "-mf16c", "-o", out, *src,
                    "-I", os.path.join(GGML, "include"), "-I", os.path.join(REPO, "include"), lib,
                    "-lm", "-lpthread", "-fopenmp"], check=True)
    return out


REF = None


def ref(*args, data=b""):
    return subprocess.run([REF, *map(str, args)], input=data, capture_output=True, check=True).stdout


class Formats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global REF
        REF = build_ref()
        cls.rng = np.random.default_rng(7)

    def check_deq(self, t, raw, n):
        want = np.frombuffer(ref("deq", t, n, data=raw.tobytes()), np.float32)
        got = F.decode(t, raw.tobytes(), n)
        np.testing.assert_array_equal(got, want)   # our decoder == the C one, bit for bit
        return want

    def test_q2_0(self):
        R, C = 3, 512
        codes = self.rng.integers(-1, 3, (R, C)).astype(np.int8)
        d = (self.rng.standard_normal((R, C // 64)) * 0.01).astype(np.float32)
        raw = F.pack_q2_0(codes, d)
        self.assertEqual(raw.shape, (R, C // 64 * 18))
        y = self.check_deq(F.Q2_0, raw, R * C)
        np.testing.assert_array_equal(y.reshape(R, C), codes * np.repeat(d.astype(np.float16).astype(np.float32), 64, 1))

    def test_ptq1_0(self):
        R, C = 2, 1024
        codes = self.rng.integers(-1, 2, (R, C)).astype(np.int8)
        d = (self.rng.standard_normal((R, C // 128)) * 0.01).astype(np.float32)
        raw = F.pack_ptq1_0(codes, d)
        want = np.frombuffer(ref("pack143", R * C, data=codes.tobytes() + d.tobytes()), np.uint8)
        np.testing.assert_array_equal(raw.reshape(-1), want)     # == Strata's ptq1_0_pack
        y = self.check_deq(F.PTQ1_0, raw, R * C)
        np.testing.assert_array_equal(y.reshape(R, C), codes * np.repeat(d.astype(np.float16).astype(np.float32), 128, 1))

    def test_q3_k(self):
        R, C = 2, 768
        codes = self.rng.integers(-4, 4, (R, C)).astype(np.int8)
        dsup = (np.abs(self.rng.standard_normal((R, C // 256))) * 1e-3).astype(np.float16).astype(np.float32)
        sc = self.rng.integers(-32, 32, (R, C // 16)).astype(np.int16)
        raw = F.pack_q3_k(codes, dsup, sc)
        y = self.check_deq(F.Q3_K, raw, R * C)
        want = codes * (np.repeat(dsup, 256, 1) * np.repeat(sc, 16, 1)).astype(np.float32)
        np.testing.assert_allclose(y.reshape(R, C), want, rtol=1e-6, atol=0)

    def test_q8_0(self):
        W = (self.rng.standard_normal((4, 256)) * 0.05).astype(np.float32)
        W[0, :32] = 0   # an all-zero block
        raw = F.encode_q8_0(W)
        want = np.frombuffer(ref("q8_0", W.size, data=W.tobytes()), np.uint8)
        np.testing.assert_array_equal(raw.reshape(-1), want)
        self.check_deq(F.Q8_0, raw, W.size)

    def test_gptq_output_every_tier(self):
        """encode_tier on v41_gptq's own output (GPTQ values + search_scales scales) reproduces the values exactly."""
        torch.manual_seed(0)
        W = torch.randn(8, 512) * 0.02
        X = torch.randn(600, 512)
        H = X.T @ X
        d = torch.diag(H)
        for tier in ("ptq1", "q2_0", "q3_k"):
            S = G.search_scales(W, d, tier)
            Hinv, dead = G.hinv_chol(H, 1.0)
            lo, hi, grp, _ = G.TIERS[tier]
            Se = G.expand_scales(S, tier, 512)
            Q = G.gptq(W.clone(), Se, torch.full((8, 1), float(lo)), torch.full((8, 1), float(hi)), Hinv, dead)
            raw = F.encode_tier(Q.numpy(), S.numpy(), tier)
            t = F.TIER_TYPE[tier]
            self.assertEqual(raw.shape[1], F.row_bytes(t, 512))
            y = self.check_deq(t, raw, W.numel()).reshape(8, 512)
            np.testing.assert_allclose(y, Q.numpy(), rtol=1e-6, atol=1e-12, err_msg=tier)

    def test_rtn_and_mismatch(self):
        W = (self.rng.standard_normal((4, 512)) * 0.02).astype(np.float32)
        for tier in ("ptq1", "q2_0", "q3_k"):
            raw = F.encode_rtn(W, tier)
            y = F.decode(F.TIER_TYPE[tier], raw.tobytes(), W.size)
            rel = np.linalg.norm(y - W.reshape(-1)) / np.linalg.norm(W)
            self.assertLess(rel, 0.6, tier)
        with self.assertRaises(ValueError):   # values that are not codes x scales
            F.encode_tier(W, np.full((4, 8), 0.01, np.float32), "q2_0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
