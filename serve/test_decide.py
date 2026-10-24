"""local (Indra's Mirror): /v1/decide's pure parts - the engine's SCORE line, the options' first tokens, the pick."""
import math
import unittest

from serve.server import ByteTokenizer, decide_option_tokens, decide_result, parse_score_line


class DecideTest(unittest.TestCase):
    def test_parse_score_line(self):
        got = parse_score_line("SCORE 12:-0.105361 40:-2.302585 top=12:-0.105361")
        self.assertEqual(got["ids"], {12: -0.105361, 40: -2.302585})
        self.assertEqual(got["top"], (12, -0.105361))

    def test_option_tokens_must_differ(self):
        tok = ByteTokenizer()
        self.assertEqual(len(decide_option_tokens(tok, ["WAIT", "RESPOND"])), 2)
        with self.assertRaises(ValueError):
            decide_option_tokens(tok, ["WAIT", "WAITING"])       # same first byte token
        with self.assertRaises(ValueError):
            decide_option_tokens(tok, ["only"])

    def test_bias_is_a_threshold(self):
        # p(WAIT) = 0.3, p(RESPOND) = 0.7: plain argmax says RESPOND; a log-odds bias of ln(0.7/0.3)+eps says WAIT
        scored = {"ids": {1: math.log(0.3), 2: math.log(0.7)}, "top": (2, math.log(0.7))}
        plain = decide_result(["WAIT", "RESPOND"], [1, 2], scored, None)
        self.assertEqual(plain["choice"], "RESPOND")
        self.assertAlmostEqual(plain["probs"]["WAIT"], 0.3, places=5)
        self.assertAlmostEqual(plain["coverage"], 1.0, places=5)
        biased = decide_result(["WAIT", "RESPOND"], [1, 2], scored, {"WAIT": math.log(0.7 / 0.3) + 1e-3})
        self.assertEqual(biased["choice"], "WAIT")
        with self.assertRaises(ValueError):
            decide_result(["WAIT", "RESPOND"], [1, 2], scored, {"NOPE": 1.0})
        with self.assertRaises(ValueError):
            decide_result(["WAIT", "RESPOND"], [1, 3], scored, None)   # no score for 3


if __name__ == "__main__":
    unittest.main()
