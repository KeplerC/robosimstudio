from art import save_png

"""Generate one square metre of worn workbench: grain, scratches and stains.

Run beside art.py. Colour and roughness are native material textures; no
photo or camera projection is involved. Seeded for reproducible rebuilding.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

A = Path(__file__).resolve().parent / "assets"
rng = np.random.default_rng(138)
n = 2048
y, x = np.mgrid[:n, :n]
v = np.zeros((n, n), np.float32)
for rows, cols, amount in [(24, 256, 0.08), (5, 1024, 0.035), (100, 30, 0.045)]:
    src = Image.fromarray(np.uint8(rng.uniform(0, 255, (rows, cols))))
    v += (np.asarray(src.resize((n, n), Image.Resampling.BICUBIC), np.float32) / 255 - 0.5) * amount
curve = 5 * np.sin(y / 240) + 2 * np.sin(y / 74)
v += 0.013 * np.sin((x + curve) * 0.44) + 0.007 * np.sin((x + curve) * 1.79)
for cx, cy in [(305, 680), (1220, 1480), (1780, 350)]:
    rad = np.sqrt((x - cx) ** 2 + ((y - cy) / 5) ** 2)
    v -= 0.07 * np.exp(-rad / 35)
    v += 0.018 * np.sin(rad * 0.65) * np.exp(-rad / 80)
base = np.stack([0.55 + v, 0.365 + v * 0.8, 0.21 + v * 0.55], axis=2)
for k in range(8):
    base[:, k * 256 : (k + 1) * 256] *= rng.uniform(0.9, 1.08)
# Grease/water stains, broad enough to remain visible in the close-up.
for _i in range(100):
    cx, cy = rng.integers(0, n, 2)
    sx, sy = rng.uniform(10, 65, 2)
    blob = np.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2)
    base -= blob[:, :, None] * rng.uniform(0.025, 0.10) * np.array([1.0, 0.85, 0.6])
im = Image.fromarray(np.uint8(np.clip(base, 0, 1) * 255))
d = ImageDraw.Draw(im)
for _i in range(750):
    px, py = rng.integers(0, n, 2)
    length = int(rng.integers(8, 100))
    dx = int(rng.integers(-length, length))
    colour = (184, 139, 88) if rng.random() < 0.55 else (93, 60, 34)
    d.line((px, py, px + dx, py + length), fill=colour, width=int(rng.integers(1, 4)))
for _i in range(180):
    px, py = rng.integers(0, n, 2)
    r = int(rng.integers(2, 6))
    d.ellipse((px - r, py - r // 2, px + r, py + r // 2), fill=(88, 57, 32))
for k in range(1, 8):
    d.line((k * 256, 0, k * 256, n), fill=(83, 52, 29), width=3)
    d.line((k * 256 + 4, 0, k * 256 + 4, n), fill=(174, 123, 72), width=2)
save_png(im, A / "wood_worn.png")
grey = np.asarray(im.convert("L"), np.float32)
rough = np.uint8(np.clip(210 + (grey - grey.mean()) * 0.5, 160, 250))
save_png(Image.fromarray(rough), A / "wood_roughness.png")

save_png(Image.open(A / "wood_worn.png"), A / "wood_worn_r3.png")
save_png(Image.open(A / "wood_roughness.png"), A / "wood_roughness_r3.png")
