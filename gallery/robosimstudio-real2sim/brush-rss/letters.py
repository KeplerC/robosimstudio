"""The letters R, S, S as brush strokes: one pen-down centreline per letter, and the mask the strokes should clean.

Letter coordinates: ``u`` along the line of text, ``v`` up the letter, in metres, origin at the letter box's lower
left. On the panel the text reads toward world +y and the letters stand toward world -x (the camera is on the +x
side), so ``world = (x_base - v, y0 + u)``.
"""

from __future__ import annotations

import numpy as np

H = 0.110
"""Letter height [m] (outer, as cleaned)."""
W = 0.014
"""Stroke width [m]: the brush's measured clean footprint (probe/width.py: 14 mm along both axes at a 4 mm press,
10 mm sole, 7 x 7 bristles, flare 0.09)."""
L = 0.068
"""Letter width [m] (outer)."""
GAP = 0.014
"""Space between letters [m]."""


def _arc(cx, cy, rx, ry, a0, a1, n=None):
    a0, a1 = np.radians(a0), np.radians(a1)
    if n is None:
        n = max(8, int(abs(a1 - a0) * max(rx, ry) / 0.002))
    t = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def _line(a, b, step=0.002):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return a + (b - a) * np.linspace(0, 1, n)[:, None]


def _join(*parts):
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p[1:] if np.allclose(out[-1][-1], p[0], atol=1e-6) else p)
    return np.concatenate(out)


def letter_R(h=H, w=W, ln=L):
    c = w / 2
    rb = (h - c - (0.45 * h)) / 2  # bowl centreline radius: the bowl's bottom bar at 45% of the height
    top = h - c
    mid = top - 2 * rb
    bx = ln - c - rb
    uj = c + 0.30 * (ln - 2 * c)
    return _join(
        _line((c, c), (c, top)),
        _line((c, top), (bx, top)),
        _arc(bx, top - rb, rb, rb, 90, -90),
        _line((bx, mid), (uj, mid)),
        _line((uj, mid), (ln - c, c)),
    )


def letter_S(h=H, w=W, ln=L, start=20.0, end=-160.0):
    c = w / 2
    rx = (ln - 2 * c) / 2
    ry = (h - 2 * c) / 4
    cu = (ln / 2, h - c - ry)
    cl = (ln / 2, c + ry)
    return _join(_arc(*cu, rx, ry, start, 270), _arc(*cl, rx, ry, 90, end))


def strokes_uv():
    """[(name, (N,2) u,v polyline)] for R, S, S, placed along the line."""
    out = []
    u0 = 0.0
    for name, fn in (("R", letter_R), ("S1", letter_S), ("S2", letter_S)):
        p = fn().copy()
        p[:, 0] += u0
        out.append((name, p))
        u0 += L + GAP
    return out


def text_width():
    return 3 * L + 2 * GAP


def to_world(uv, x_base, y0):
    uv = np.asarray(uv, float)
    return np.stack([x_base - uv[:, 1], y0 + uv[:, 0]], axis=1)


def mask(strokes_xy, cx, cy, half=W / 2):
    """Cells (centres ``cx`` along x, ``cy`` along y) a square ``2*half`` footprint, kept square to the world axes
    (the hand does not turn), covers when its centre sweeps the polylines: the letters as this brush draws them."""
    m = np.zeros((len(cy), len(cx)), dtype=bool)
    X, Y = np.meshgrid(cx, cy)
    for pts in strokes_xy:
        pts = np.asarray(pts, float)
        # sweep densely: resample to 0.5 mm
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        t = np.linspace(0, s[-1], max(2, int(s[-1] / 0.0005)))
        px = np.interp(t, s, pts[:, 0])
        py = np.interp(t, s, pts[:, 1])
        for x, y in zip(px, py):
            ix0, ix1 = np.searchsorted(cx, x - half), np.searchsorted(cx, x + half)
            iy0, iy1 = np.searchsorted(cy, y - half), np.searchsorted(cy, y + half)
            m[iy0:iy1, ix0:ix1] = True
    return m


def iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def lead_path(pts, d, corner_deg=25.0, step_deg=10.0):
    """The hand's path that drags a brush's trailing tips along ``pts``: each point moved ``d`` ahead along the
    stroke's tangent there; at a corner sharper than ``corner_deg`` the hand swings round the corner point
    (an arc of radius ``d``) while the tips stay on it. The first point is the touchdown itself (no lead yet)."""
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    seg_t = seg / np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
    out = [pts[0]]
    prev_t = None
    for i in range(len(pts)):
        t_in = seg_t[i - 1] if i > 0 else seg_t[0]
        t_out = seg_t[i] if i < len(seg_t) else seg_t[-1]
        ang = np.degrees(np.arctan2(t_in[0] * t_out[1] - t_in[1] * t_out[0], float(np.dot(t_in, t_out))))
        if abs(ang) > corner_deg or i == 0:
            a0 = np.arctan2(t_in[1], t_in[0])
            n = max(1, int(abs(ang) / step_deg))
            for k in range(n + 1):
                a = a0 + np.radians(ang) * k / n
                out.append(pts[i] + d * np.array([np.cos(a), np.sin(a)]))
        else:
            t = t_in + t_out
            t = t / max(np.linalg.norm(t), 1e-12)
            out.append(pts[i] + d * t)
    return np.asarray(out)
