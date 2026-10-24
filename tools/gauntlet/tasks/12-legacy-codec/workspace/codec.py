"""Reimplement the legacy licence-code generator.

`legacy_codec` (in this directory) is the only surviving build of a tool whose source is lost. It reads up to 255
raw bytes on stdin and prints one code line. Downstream systems verify codes by regenerating them, so the
reimplementation must be BIT-FOR-BIT identical for EVERY input of 0..255 bytes:

    encode(data: bytes) -> str     the line legacy_codec prints, without the trailing newline
                                   (more than 255 bytes: raise ValueError; the tool refuses them too)

Querying it:    subprocess.run(["./legacy_codec"], input=data, capture_output=True).stdout

Rules (the grade enforces them): the binary is deleted before grading, and codec.py must be pure computation -
it may import only struct, itertools, functools, math, collections, typing, operator, string, and may not use
open/exec/eval/compile/__import__/getattr or dunder attributes. A lookup table of queried outputs will not help:
the grading inputs are thousands of bytes of data you have not seen, of every length, including long runs of a
repeated byte and plenty of 0xFF.
"""


def encode(data):
    raise NotImplementedError
