"""Hand-observed blade boundaries, in photo pixels; geometry, never image textures.

Each boundary is projected onto a separately inferred 3D leaf plane. Interior
vertices are cupped and veined, and the mesh subsequently settles as cloth.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

OUTLINES = {
    "leaf_01": {
        "base": (1105, 520),
        "tip": (936, 599),
        "normal": (-0.55, 0.25, 0.80),
        "left": [(1087, 449), (1085, 485), (1050, 522), (1000, 567), (936, 599)],
        "right": [(1140, 533), (1115, 554), (1070, 571), (1010, 588), (936, 599)],
    },
    "leaf_04": {
        "notch": (1248, 360),
        "interior_hold": True,
        "base": (1286, 362),
        "tip": (1434, 391),
        "normal": (-0.70, 0.05, 0.71),
        "left": [(1262, 342), (1304, 337), (1350, 350), (1390, 370), (1434, 391)],
        "right": [(1233, 383), (1255, 416), (1310, 432), (1382, 422), (1434, 391)],
    },
    "leaf_05": {
        "base": (1312, 329),
        "tip": (1445, 351),
        "normal": (-0.6044734941, -0.5086500508, 0.6130961758),
        "left": [(1315, 289), (1344, 274), (1380, 290), (1409, 320), (1445, 351)],
        "right": [(1305, 351), (1320, 376), (1360, 386), (1410, 373), (1445, 351)],
    },
    "leaf_06": {
        "base": (1441, 456),
        "tip": (1518, 487),
        "normal": (-0.3012326212, -0.4183022254, 0.8569026527),
        "left": [(1430, 418), (1452, 422), (1483, 441), (1506, 466), (1518, 487)],
        "right": [(1421, 454), (1438, 473), (1464, 480), (1490, 484), (1518, 487)],
    },
    "leaf_07": {
        "base": (1436, 521),
        "tip": (1558, 537),
        "normal": (-0.47, -0.28, 0.84),
        "left": [(1430, 496), (1474, 499), (1510, 514), (1540, 528), (1558, 537)],
        "right": [(1418, 538), (1454, 557), (1498, 554), (1538, 544), (1558, 537)],
    },
    "leaf_09": {
        "base": (1354, 618),
        "tip": (1393, 683),
        "normal": (-0.60, -0.20, 0.77),
        "left": [(1335, 607), (1325, 631), (1338, 656), (1361, 671), (1393, 683)],
        "right": [(1378, 584), (1391, 602), (1388, 631), (1381, 660), (1393, 683)],
    },
    "spathe_01": {
        "base": (1225, 310),
        "tip": (1215, 220),
        "normal": (-0.87, 0.1, 0.48),
        "left": [(1140, 335), (1138, 305), (1158, 257), (1187, 226), (1215, 220)],
        "right": [(1301, 268), (1276, 276), (1269, 255), (1242, 223), (1215, 220)],
    },
    "spathe_02": {
        "base": (1346, 299),
        "tip": (1374, 246),
        "normal": (-0.65, -0.40, 0.65),
        "left": [(1319, 288), (1320, 272), (1340, 253), (1365, 246), (1374, 246)],
        "right": [(1399, 307), (1405, 295), (1410, 279), (1394, 254), (1374, 246)],
    },
    "spathe_03": {
        "base": (1218, 325),
        "tip": (1320, 293),
        "normal": (-0.40, 0.55, 0.73),
        "left": [(1189, 284), (1211, 270), (1243, 276), (1278, 287), (1320, 293)],
        "right": [(1281, 319), (1295, 309), (1306, 304), (1314, 300), (1320, 293)],
    },
    "spathe_04": {
        "notch": (1356, 359),
        "interior_hold": True,
        "base": (1356, 377),
        "tip": (1328, 401),
        "normal": (-0.85, 0.05, 0.525),
        "left": [(1282, 343), (1278, 365), (1292, 388), (1311, 398), (1328, 401)],
        "right": [(1414, 345), (1414, 366), (1388, 387), (1358, 392), (1328, 401)],
    },
}


def mesh(outline, base, eye, forward, right, up, focal, origin, spathe=False):
    n = np.asarray(outline["normal"], float)
    n /= np.linalg.norm(n)
    B = np.asarray(outline.get("notch", outline["base"]), float)
    T = np.asarray(outline["tip"], float)

    def point(pixel):
        ray = forward + (pixel[0] - 960) / focal * right - (pixel[1] - 640) / focal * up
        return eye + ray * np.dot(n, base + origin - eye) / np.dot(n, ray) - origin

    def edge(key, t):
        pts = np.asarray(outline[key], float)
        knots = np.linspace(0, 1, len(pts))
        return PchipInterpolator(knots, pts, axis=0)(t)

    length = np.linalg.norm(point(T) - base)
    if spathe:
        length = max(length, *(np.linalg.norm(point(p) - base) for p in outline["left"] + outline["right"]))
    vertices = []
    for _i, t in enumerate(np.linspace(0, 1, 11)):
        middle = B + (T - B) * t
        for u in np.linspace(-1, 1, 11):
            pixel = middle + (edge("left" if u < 0 else "right", t) - middle) * abs(u)
            p = point(pixel)
            # A depressed bowl for spathes, a raised midrib and ripples for leaves.
            cup = (-0.18 if spathe else 0.035) * length * (1 - u * u) * np.sin(np.pi * t)
            ripple = 0.018 * length * np.sin(4 * np.pi * t + u) * np.sin(np.pi * t) * (1 - abs(u))
            vertices.append(p + n * (cup + ripple))
    vertices = [*vertices[:110], vertices[115]]
    return np.asarray(vertices), n
