"""A tiny deterministic ray tracer (reference solution)."""
import math


def _sub(a, b): return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
def _add(a, b): return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
def _mul(a, s): return [a[0] * s, a[1] * s, a[2] * s]
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _norm(a):
    n = math.sqrt(_dot(a, a))
    return [a[0] / n, a[1] / n, a[2] / n]


def _hit(scene, o, d):
    best = None
    for s in scene.get("spheres", []):
        oc = _sub(o, s["center"])
        b = _dot(oc, d)
        c = _dot(oc, oc) - s["radius"] ** 2
        disc = b * b - c
        if disc < 0:
            continue
        sq = math.sqrt(disc)
        for t in (-b - sq, -b + sq):
            if t > 1e-4:
                if best is None or t < best[0]:
                    p = _add(o, _mul(d, t))
                    best = (t, p, _mul(_sub(p, s["center"]), 1 / s["radius"]), s["color"], s.get("reflect", 0))
                break
    pl = scene.get("plane")
    if pl and abs(d[1]) > 1e-12:
        t = (pl["y"] - o[1]) / d[1]
        if t > 1e-4 and (best is None or t < best[0]):
            p = _add(o, _mul(d, t))
            cell = math.floor(p[0]) + math.floor(p[2])
            best = (t, p, [0.0, 1.0, 0.0], pl["colors"][cell % 2], pl.get("reflect", 0))
    return best


def _trace(scene, o, d, depth):
    h = _hit(scene, o, d)
    if h is None:
        t = 0.5 * (d[1] + 1)
        return [(1 - t) + t * 0.5, (1 - t) + t * 0.7, (1 - t) + t * 1.0]
    _, p, n, c, k = h
    lp = scene["light"]["pos"]
    to_l = _sub(lp, p)
    dist = math.sqrt(_dot(to_l, to_l))
    L = _mul(to_l, 1 / dist)
    diffuse = max(0.0, _dot(n, L)) * scene["light"]["intensity"]
    if diffuse > 0:
        sh = _hit(scene, _add(p, _mul(n, 1e-4)), L)
        if sh is not None and sh[0] < dist:
            diffuse = 0.0
    a = scene["ambient"]
    local = [c[0] * (a + diffuse), c[1] * (a + diffuse), c[2] * (a + diffuse)]
    if k > 0:
        if depth >= 3:
            refl = [0.0, 0.0, 0.0]
        else:
            r = _sub(d, _mul(n, 2 * _dot(d, n)))
            refl = _trace(scene, _add(p, _mul(n, 1e-4)), r, depth + 1)
        local = [local[i] * (1 - k) + refl[i] * k for i in range(3)]
    return local


def render(scene, width, height):
    cam = scene["camera"]
    scale = math.tan(math.radians(cam["fov_deg"]) / 2)
    aspect = width / height
    img = []
    for py in range(height):
        row = []
        for px in range(width):
            nx = (px + 0.5) / width * 2 - 1
            ny = 1 - (py + 0.5) / height * 2
            d = _norm([nx * scale * aspect, ny * scale, -1.0])
            col = _trace(scene, cam["pos"], d, 0)
            row.append(tuple(round(min(1.0, max(0.0, ch)) * 255) for ch in col))
        img.append(row)
    return img
