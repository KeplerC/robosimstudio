from pathlib import Path

import numpy as np
from art import save_png
from PIL import Image, ImageDraw

A = Path(__file__).resolve().parent / "assets"
rng = np.random.default_rng(24)
n = 1024
y, x = np.mgrid[:n, :n]
# Long grain, knots, worn grain and pale scoring; generated, not photographic.
v = np.zeros((n, n))
for k in range(5):
    small = Image.fromarray(np.uint8(rng.uniform(0, 255, (80 // (k + 1), 8 + k * 3))))
    layer = np.asarray(small.resize((n, n), Image.Resampling.BICUBIC), float) / 255 - 0.5
    v += layer * (0.20 / (k + 1))
warp = x + 8 * np.sin(y / 190) + 3 * np.sin(y / 65)
v += 0.035 * np.sin(warp * 0.4) + 0.018 * np.sin(warp * 1.3)
base = np.clip(np.array([0.45, 0.285, 0.145])[None, None, :] * (1 + v[:, :, None] * 2.2), 0, 1)
im = Image.fromarray(np.uint8(base * 255))
d = ImageDraw.Draw(im)
for _i in range(160):
    px, py = rng.integers(0, n, 2)
    length = int(rng.integers(4, 170))
    d.line(
        (px, py, px + int(rng.integers(-3, 4)), py + length),
        fill=tuple(int(c) for c in rng.choice([[115, 83, 49], [66, 43, 25], [143, 106, 66]])),
        width=int(rng.integers(1, 3)),
    )
save_png(im, A / "workbench_wood.png")
p = Image.new("RGB", (512, 512), (142, 106, 68))
d = ImageDraw.Draw(p)
for x in range(16, 512, 32):
    for y in range(16, 512, 32):
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(33, 28, 21))
        d.arc((x - 5, y - 5, x + 5, y + 5), 30, 175, fill=(193, 151, 99), width=1)
save_png(p, A / "pegboard.png")

save_png(Image.open(A / "pegboard.png"), A / "pegboard_r3.png")
