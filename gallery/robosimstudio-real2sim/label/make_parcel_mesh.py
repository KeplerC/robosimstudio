"""Bake the parcel's native UV atlas and mesh. Run after art.py.

The artwork is 100 x 120 mm; the kraft tile is 200 x 200 mm. Every face
uses 10.24 pixels/mm in the atlas, avoiding stretched side-face fibres.
The top image is NOT flipped; the cloth's independent UV convention is.
"""

from pathlib import Path

import numpy as np
from art import save_png
from PIL import Image

A = Path(__file__).resolve().parent / "assets"
W, H = 3072, 2048
TW, TH = 1024, 1229
top = Image.open(A / "print.png").resize((TW, TH), Image.Resampling.LANCZOS)
kraft = Image.open(A / "kraft.png").resize((2048, 2048), Image.Resampling.LANCZOS)
im = Image.new("RGB", (W, H), (190, 151, 105))
im.paste(top, (0, 0))
im.paste(kraft, (1024, 0))
save_png(im, A / "parcel_atlas.png")
save_png(Image.open(A / "label.png").transpose(Image.Transpose.FLIP_TOP_BOTTOM), A / "label_face.png")
faces = [
    [(-0.05, -0.06, 0.025), (0.05, -0.06, 0.025), (0.05, 0.06, 0.025), (-0.05, 0.06, 0.025)],
    [(-0.05, 0.06, -0.025), (0.05, 0.06, -0.025), (0.05, -0.06, -0.025), (-0.05, -0.06, -0.025)],
    [(0.05, -0.06, -0.025), (0.05, 0.06, -0.025), (0.05, 0.06, 0.025), (0.05, -0.06, 0.025)],
    [(-0.05, 0.06, -0.025), (-0.05, -0.06, -0.025), (-0.05, -0.06, 0.025), (-0.05, 0.06, 0.025)],
    [(-0.05, -0.06, -0.025), (0.05, -0.06, -0.025), (0.05, -0.06, 0.025), (-0.05, -0.06, 0.025)],
    [(0.05, 0.06, -0.025), (-0.05, 0.06, -0.025), (-0.05, 0.06, 0.025), (0.05, 0.06, 0.025)],
]
V = []
F = []
UV = []
for i, face in enumerate(faces):
    j = len(V)
    V += face
    F += [[j, j + 1, j + 2], [j, j + 2, j + 3]]
    if i == 0:
        a, b, c, d = 0.5 / W, (TW - 0.5) / W, 1 - (TH - 0.5) / H, 1 - 0.5 / H
    else:
        width = np.linalg.norm(np.array(face[1]) - face[0])
        height = np.linalg.norm(np.array(face[3]) - face[0])
        a = 1025 / W
        b = a + width / 0.2 * 2046 / W
        c = 1 / H
        d = c + height / 0.2 * 2046 / H
    UV += [(a, c), (b, c), (b, d), (a, d)]
np.savez(
    A / "parcel_mesh.npz", vertices=np.array(V, np.float32), faces=np.array(F, np.int32), uv=np.array(UV, np.float32)
)

save_png(Image.open(A / "parcel_atlas.png"), A / "parcel_atlas_r3.png")
save_png(Image.open(A / "label_face.png"), A / "label_face_r3.png")
