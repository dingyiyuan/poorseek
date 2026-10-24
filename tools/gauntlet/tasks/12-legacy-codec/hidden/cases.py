"""Hidden inputs for 12-legacy-codec: seeded random bytes of every length class plus structured edge cases."""
import random

_r = random.Random(20261016)
CASES = [b"", b"\x00", b"\xff", b"\xff\xff", b"\xff\xff\xff", b"\x00" * 255, b"\xff" * 255, bytes(range(256))[:255]]
CASES += [bytes([x]) * n for x in (0x41, 0x00, 0xFF, 0x5A) for n in (2, 3, 4, 129, 130, 131, 132, 200)]
CASES += [b"ab" * n for n in (1, 2, 50)] + [b"aab", b"aaab", b"abbbc", b"x" * 3 + b"\xff" * 2 + b"x" * 3]
CASES += [bytes(_r.choice((0x41, 0x42, 0xFF)) for _ in range(_r.randint(1, 255))) for _ in range(60)]   # runs + escapes
CASES += [bytes(_r.randrange(256) for _ in range(n)) for n in range(1, 64)]                         # every short length
CASES += [bytes(_r.randrange(256) for _ in range(_r.randint(0, 255))) for _ in range(250)]          # arbitrary
CASES += [b"".join(bytes([_r.randrange(256)]) * _r.randint(1, 6) for _ in range(40))[:255] for _ in range(60)]  # short runs
