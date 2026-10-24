import json
import os

import pytest

from raytrace import render
from scenes import SCENE_A, SCENE_B

REF = json.load(open(os.path.join(os.path.dirname(__file__), "ref_images.json")))


def close(img, ref):
    assert len(img) == len(ref) and all(len(r) == len(q) for r, q in zip(img, ref)), "wrong image size"
    bad, total, err = 0, 0, 0
    for r, q in zip(img, ref):
        for a, b in zip(r, q):
            d = max(abs(int(a[i]) - int(b[i])) for i in range(3))
            err += d
            total += 1
            bad += d > 2
    return bad / total, err / total


@pytest.mark.parametrize("name,scene", [("A", SCENE_A), ("B", SCENE_B)])
def test_matches_reference(name, scene):
    img = render(scene, 64, 48)
    frac_bad, mean_err = close([[tuple(p) for p in row] for row in img], REF[name])
    assert frac_bad <= 0.01 and mean_err < 0.5, f"scene {name}: {frac_bad:.1%} pixels off by > 2, mean err {mean_err:.2f}"
