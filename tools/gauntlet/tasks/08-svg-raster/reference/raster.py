"""Rasterize a filled SVG path (reference)."""
import math
import re

_TOK = re.compile(r"[MmLlHhVvCcQqAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
N = 32


def _off(rel, pt, x, y):
    return (pt[0] + x, pt[1] + y) if rel else (x, y)


def _cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1])


def _quad(p0, p1, p2, t):
    u = 1.0 - t
    return (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])


def _angle(ux, uy, vx, vy):
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    a = math.acos(c)
    return a if (ux * vy - uy * vx) >= 0 else -a


def _arc(p0, rx, ry, rot, fA, fS, p1):
    """Flatten the SVG arc from p0 to p1: points at arc parameter p = 1/N .. N/N."""
    x1, y1 = p0
    x2, y2 = p1
    if x1 == x2 and y1 == y2:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0:
        return [p1]
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx + sp * dy
    y1p = -sp * dx + cp * dy
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = (1.0 if fA != fS else -1.0) * math.sqrt(max(0.0, num / den))
    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0
    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    t0 = _angle(1.0, 0.0, ux, uy)
    dth = _angle(ux, uy, vx, vy)
    if fS == 0 and dth > 0:
        dth -= 2 * math.pi
    elif fS == 1 and dth < 0:
        dth += 2 * math.pi
    pts = []
    for k in range(1, N + 1):
        th = t0 + (k / N) * dth
        cth, sth = math.cos(th), math.sin(th)
        pts.append((cx + rx * cth * cp - ry * sth * sp, cy + rx * cth * sp + ry * sth * cp))
    return pts


def _subpaths(d):
    toks = _TOK.findall(d)
    if re.sub(r"[\s,]", "", d) != "".join(toks):
        raise ValueError("unparseable path")
    subs, cur, start, pt, cmd, i = [], None, (0.0, 0.0), (0.0, 0.0), None, 0

    def num():
        nonlocal i
        if i >= len(toks) or toks[i].isalpha():
            raise ValueError("missing number")
        i += 1
        return float(toks[i - 1])

    def line_to(p):
        nonlocal cur, pt
        if cur is None:
            cur = [start]
        cur.append(p)
        pt = p

    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur is not None:
                    subs.append(cur)
                    cur = None
                pt = start
                continue
        elif cmd is None or cmd in "Zz":
            raise ValueError("number without a command")
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = num(), num()
            pt = (pt[0] + x, pt[1] + y) if rel else (x, y)
            if cur is not None:
                subs.append(cur)
            cur, start = [pt], pt
            cmd = "l" if rel else "L"
        elif c == "L":
            x, y = num(), num()
            line_to((pt[0] + x, pt[1] + y) if rel else (x, y))
        elif c == "H":
            x = num()
            line_to((pt[0] + x, pt[1]) if rel else (x, pt[1]))
        elif c == "V":
            y = num()
            line_to((pt[0], pt[1] + y) if rel else (pt[0], y))
        elif c == "C":
            x1, y1, x2, y2, x, y = num(), num(), num(), num(), num(), num()
            q = pt
            p1, p2, p3 = _off(rel, q, x1, y1), _off(rel, q, x2, y2), _off(rel, q, x, y)
            if cur is None:
                cur = [start]
            for k in range(1, N + 1):
                cur.append(_cubic(q, p1, p2, p3, k / N))
            pt = p3
        elif c == "Q":
            x1, y1, x, y = num(), num(), num(), num()
            q = pt
            p1, p2 = _off(rel, q, x1, y1), _off(rel, q, x, y)
            if cur is None:
                cur = [start]
            for k in range(1, N + 1):
                cur.append(_quad(q, p1, p2, k / N))
            pt = p2
        elif c == "A":
            rx, ry, rot, fa, fs, x, y = num(), num(), num(), num(), num(), num(), num()
            if fa not in (0.0, 1.0) or fs not in (0.0, 1.0):
                raise ValueError("arc flag")
            p2 = _off(rel, pt, x, y)
            if cur is None:
                cur = [start]
            cur.extend(_arc(pt, rx, ry, rot, fa, fs, p2))
            pt = p2
        else:
            raise ValueError(f"unknown command {cmd}")
    if cur is not None:
        subs.append(cur)
    return subs


def rasterize(d, width, height, fill_rule="nonzero"):
    if fill_rule not in ("nonzero", "evenodd"):
        raise ValueError("fill_rule")
    edges = []
    for poly in _subpaths(d):
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            if a[1] != b[1]:
                edges.append((a[0], a[1], b[0], b[1]))
    out = []
    for row in range(height):
        cy = row + 0.5
        line = []
        for col in range(width):
            cx = col + 0.5
            wind, cross = 0, 0
            for x0, y0, x1, y1 in edges:
                if (y0 <= cy < y1) or (y1 <= cy < y0):
                    xc = x0 + (cy - y0) * (x1 - x0) / (y1 - y0)
                    if xc > cx:
                        cross += 1
                        wind += 1 if y1 > y0 else -1
            inside = wind != 0 if fill_rule == "nonzero" else cross % 2 == 1
            line.append("#" if inside else ".")
        out.append("".join(line))
    return out
