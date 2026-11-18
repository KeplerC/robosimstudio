"""Drawn surface maps. No reference-frame pixels are used."""

from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).parent / "assets"
# Texture coordinates are metric box projection, shifted to the phone's centre.
im = np.full((512, 1024, 3), (216, 215, 203), np.uint8)
cv2.rectangle(im, (18, 20), (1005, 491), (13, 16, 19), -1)
for x, y in [(30, 32), (993, 32), (30, 480), (993, 480)]:
    cv2.circle(im, (x, y), 12, (13, 16, 19), -1)
cv2.rectangle(im, (970, 221), (989, 291), (5, 6, 7), -1)
cv2.circle(im, (981, 199), 5, (38, 42, 45), -1)
cv2.imwrite(str(root / "phone_surface.png"), im)
rng = np.random.default_rng(44)
im = np.full((1024, 512), 50, np.uint8)
for _k in range(1600):
    p = rng.integers([0, 0], [512, 1024])
    d = rng.normal([9, 18], [18, 35])
    q = np.clip(p + d, [0, 0], [511, 1023]).astype(int)
    cv2.line(im, tuple(p), tuple(q), int(rng.integers(85, 225)), int(rng.choice([1, 1, 1, 2])), cv2.LINE_AA)
for _k in range(60):
    p = rng.integers([0, 0], [512, 1024])
    cv2.ellipse(
        im,
        tuple(p),
        (int(rng.integers(2, 9)), int(rng.integers(8, 28))),
        float(rng.uniform(-60, 60)),
        0,
        300,
        int(rng.integers(90, 190)),
        1,
    )
cv2.imwrite(str(root / "film_roughness.png"), im)
