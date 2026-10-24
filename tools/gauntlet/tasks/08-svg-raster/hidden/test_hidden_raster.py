import json
import os

import pytest

from raster import rasterize
from shapes import SHAPES

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_rasters.json")))


@pytest.mark.parametrize("name,d,w,h,rule", SHAPES)
def test_shape(name, d, w, h, rule):
    got = rasterize(d, w, h, rule)
    want = REF[name]
    if got != want:
        diff = sum(a != b for ga, wa in zip(got, want) for a, b in zip(ga, wa))
        pytest.fail(f"{name}: {diff} pixels differ\n" + "\n".join(got) + "\n--- want ---\n" + "\n".join(want))


@pytest.mark.parametrize("bad", [
    "M1 1 L2",                    # a parameter set missing numbers
    "L1 x",                       # a command, then a character where a number belongs
    "1 2 3",                      # a number with no command before it
    "M1 1 X2 2",                  # unknown command letter
    "M1 1 #2 2",                  # a character outside the grammar
    "M1 1 C1 2 3 4 5",            # a cubic with too few numbers
    "M1 1 Q2 2",                  # a quadratic with too few numbers
    "M1 1 A5 5 0 1 1 10",         # an arc with too few numbers
    "M1 1 A5 5 0 2 1 10 10",      # large-arc-flag is not 0 or 1
    "M1 1 A5 5 0 1 7 10 10",      # sweep-flag is not 0 or 1
])
def test_malformed(bad):
    with pytest.raises(ValueError):
        rasterize(bad, 8, 8)
