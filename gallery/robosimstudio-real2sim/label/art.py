#!/usr/bin/env python
"""The label shot's printed surfaces (display only), written to ``assets/``:

- ``label.png``: the shipping label's face, 1900 x 800 px over the sheet's 95 x 40 mm (20 px/mm). Image x runs from
  the label's free end (left) to its tab end (right); the last 300 px (the 15 mm turned up into the tab) stay blank.
  Image up is the sheet's +x, which lies away from a camera on the bench's +x side once the label is laid.
- ``kraft.png``: a seamless kraft-board tile for the parcel (box-projected, so it must look alike every way).
- ``spot.png``: the print on the parcel's top round the label's spot (a dashed box and PLACE LABEL HERE on kraft),
  as the camera sees it; ``print.png`` is it turned for the decal slab's UVs (``scene.DECAL``).

    python art.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
ART = HERE / "assets"
PX = 20  # px per mm
FONTS = ART / "fonts"
KRAFT = (190, 151, 105)


def save_png(image, path):
    """Atomic publish: a live texture reader must never see a partial PNG."""
    path = Path(path)
    temporary = path.with_name(path.stem + ".building.png")
    image.save(temporary)
    temporary.replace(path)


def font(name: str, size: int):
    return ImageFont.truetype(str(FONTS / name), size)


def fit(draw, text, fnt_name, box_w, box_h):
    """The largest size of ``fnt_name`` whose ``text`` fits box_w x box_h px (by its ink box)."""
    lo, hi = 8, 2000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        l, t, r, b = draw.textbbox((0, 0), text, font=font(fnt_name, mid))
        if r - l <= box_w and b - t <= box_h:
            lo = mid
        else:
            hi = mid
    return font(fnt_name, lo)


def text_in(draw, text, fnt, cx, cy, fill=(0, 0, 0)):
    l, t, r, b = draw.textbbox((0, 0), text, font=fnt)
    draw.text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text, font=fnt, fill=fill)


def label_png(path: Path) -> None:
    W, H = 95 * PX, 40 * PX
    vis = 80 * PX  # the flat part; past it the sheet turns up into the tab
    im = Image.new("RGB", (W, H), (247, 246, 240))
    d = ImageDraw.Draw(im)
    ink = (18, 18, 20)
    m, lw = 28, 12
    d.rectangle([m, m, vis - m, H - m], outline=ink, width=lw)
    # RSS at the free end (left), the ship-to block and barcode by the tab (right), where the standing tab may hide
    # some of it from a camera on that side.
    split = vis - 560
    d.line([(split, m), (split, H - m)], fill=ink, width=lw)
    pad = 60
    big = fit(d, "RSS", "lato/Lato-Black.ttf", split - m - 2 * pad, H - 2 * m - 2 * pad)
    text_in(d, "RSS", big, (m + split) / 2, H / 2, fill=ink)
    x0, x1 = split + 40, vis - m - 40
    d.text((x0, m + 30), "SHIP TO", font=font("lato/Lato-Black.ttf", 62), fill=ink)
    small = font("lato/Lato-Bold.ttf", 50)
    d.text((x0, m + 110), "R. S. STUDIO", font=small, fill=ink)
    d.text((x0, m + 170), "BENCH 0728", font=small, fill=ink)
    rng = np.random.default_rng(7)
    x, top, bot = x0, 420, 650
    while x < x1:
        w = int(rng.choice([6, 6, 10, 14, 20]))
        if x + w > x1:
            break
        d.rectangle([x, top, x + w - 1, bot], fill=ink)
        x += w + int(rng.choice([6, 8, 12, 16]))
    text_in(d, "1Z 58R 2026 1020", font("lato/Lato-Bold.ttf", 44), (x0 + x1) / 2, 705)
    # The turned-up end: plain stock.
    d.rectangle([vis, 0, W, H], fill=(247, 246, 240))
    save_png(im, path)


def kraft_png(path: Path, size: int = 1024, seed: int = 3) -> None:
    """Seamless kraft: low-frequency mottling plus fine fibres, wrapped so the tile repeats without a seam."""
    rng = np.random.default_rng(seed)

    def wrapped_noise(cells: int) -> np.ndarray:
        g = rng.normal(size=(cells, cells))
        f = np.fft.fft2(g)
        ky = np.fft.fftfreq(cells)[:, None]
        kx = np.fft.fftfreq(cells)[None, :]
        return f, np.hypot(kx, ky)

    f, k = wrapped_noise(size)
    low = np.real(np.fft.ifft2(f * np.exp(-((k / 0.01) ** 2))))
    f2, _ = wrapped_noise(size)
    mid = np.real(np.fft.ifft2(f2 * np.exp(-((k / 0.08) ** 2))))
    fib = rng.normal(size=(size, size))
    fib = np.real(
        np.fft.ifft2(
            np.fft.fft2(fib)
            * np.exp(-((np.fft.fftfreq(size)[None, :] / 0.35) ** 2 + (np.fft.fftfreq(size)[:, None] / 0.02) ** 2))
        )
    )

    def norm(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    v = 0.012 * norm(low) + 0.010 * norm(mid) + 0.012 * norm(fib)
    base = np.array(KRAFT, dtype=np.float64) / 255.0
    rgb = np.clip(base[None, None, :] * (1.0 + v[..., None]), 0, 1)
    save_png(Image.fromarray((rgb * 255).astype(np.uint8)), path)


def spot_png(path: Path, w_mm: float, h_mm: float, inner_w_mm: float, inner_h_mm: float) -> None:
    """The parcel-top print: kraft (as kraft.png, untiled) with a dashed box inner_w x inner_h and PLACE LABEL HERE.

    Image x is world +y (left to right as a camera on the bench's +x side sees it), image up is world -x."""
    W, H = int(w_mm * PX), int(h_mm * PX)
    k = Image.open(ART / "kraft.png").convert("RGB").resize((200 * PX, 200 * PX))
    im = k.crop((0, 0, W, H))
    d = ImageDraw.Draw(im)
    ink = (40, 28, 22)
    x0, y0 = (W - inner_w_mm * PX) / 2, (H - inner_h_mm * PX) / 2
    x1, y1 = W - x0, H - y0
    dash, gap, lw = 5 * PX, 3 * PX, int(1.0 * PX)

    def dashed(a, b):
        a, b = np.array(a, float), np.array(b, float)
        n = np.linalg.norm(b - a)
        u = (b - a) / n
        s = 0.0
        while s < n:
            e = min(s + dash, n)
            d.line([tuple(a + u * s), tuple(a + u * e)], fill=ink, width=lw)
            s = e + gap

    dashed((x0, y0), (x1, y0))
    dashed((x1, y0), (x1, y1))
    dashed((x1, y1), (x0, y1))
    dashed((x0, y1), (x0, y0))
    # Corner ticks, solid.
    t = 9 * PX
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
        d.line([(cx, cy), (cx + sx * t, cy)], fill=ink, width=int(1.6 * PX))
        d.line([(cx, cy), (cx, cy + sy * t)], fill=ink, width=int(1.6 * PX))
    f = fit(d, "PLACE LABEL HERE", "lato/Lato-Black.ttf", (x1 - x0) - 16 * PX, 9 * PX)
    text_in(d, "PLACE LABEL HERE", f, W / 2, H / 2, fill=ink)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    save_png(im, path)


def main() -> int:
    ART.mkdir(exist_ok=True)
    label_png(ART / "label.png")
    kraft_png(ART / "kraft.png")
    # The print covers the whole box top, 120 (y, image x) x 100 (x, image y) mm, as the camera sees it ...
    spot_png(ART / "spot.png", 120.0, 100.0, 88.0, 50.0)
    # ... and turned for the decal's box-projected UVs, which run u along the box's x and v along its y.
    save_png(Image.open(ART / "spot.png").rotate(90, expand=True), ART / "print.png")
    save_png(Image.open(ART / "kraft.png"), ART / "kraft_r3.png")
    print("wrote", sorted(p.name for p in ART.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
