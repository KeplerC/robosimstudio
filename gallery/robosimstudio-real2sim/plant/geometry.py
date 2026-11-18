from dataclasses import dataclass, field

import numpy as np


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def shell(V: np.ndarray, F: np.ndarray, t: float):
    """A blade's surface thickened to a closed shell ``t`` thick: (vertices, faces), outward-wound, watertight when the
    surface is an edge-manifold disc (every interior edge in two faces)."""
    n = vertex_normals(V, F)
    nv = len(V)
    top, bot = V + 0.5 * t * n, V - 0.5 * t * n
    faces = [F, F[:, ::-1] + nv]
    edges = {}
    for f in F:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges[(a, b)] = edges.get((a, b), 0) + 1
    side = []
    for (a, b), _cnt in edges.items():
        if (b, a) not in edges:  # a boundary edge, directed as the top face winds it
            side += [(b, a, a + nv), (b, a + nv, b + nv)]
    if side:
        faces.append(np.asarray(side, np.int64))
    return np.concatenate([top, bot]), np.concatenate(faces).astype(np.int64)


@dataclass
class Leaf:
    name: str
    kind: str  # "blade" | "spathe"
    V: np.ndarray  # drawn surface [n,3] bench frame
    F: np.ndarray
    UV: np.ndarray
    tex: str
    color: tuple
    junction: np.ndarray  # where its stalk meets it [3]
    shell_V: np.ndarray = None
    shell_F: np.ndarray = None
    area: float = 0.0
    tex_c: str = ""
    tex_n: str = ""
    spadix: tuple | None = None  # (base, tip, radius) bench frame


@dataclass
class Stalk:
    name: str
    nodes: np.ndarray  # [links+1, 3], root first
    radius: float
    leaf: str


@dataclass
class Layout:
    params: dict
    leaves: list = field(default_factory=list)
    stalks: list = field(default_factory=list)
    pot: dict = field(default_factory=dict)
    stalk_color: tuple = (0.3, 0.32, 0.26)

    def leaf(self, name: str) -> Leaf:
        return next(lf for lf in self.leaves if lf.name == name)
