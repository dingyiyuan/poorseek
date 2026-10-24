from raytrace import render

SCENE = {"camera": {"pos": [0, 0, 0], "fov_deg": 60}, "spheres": [], "light": {"pos": [5, 5, 5], "intensity": 1.0},
         "ambient": 0.1}


def test_sky_only():
    img = render(SCENE, 4, 3)
    assert len(img) == 3 and all(len(r) == 4 for r in img)
    assert img[0][0] == (171, 205, 255)        # top-left: looking up -> bluer
    assert img[2][0][0] > img[0][0][0]         # lower rows are whiter
