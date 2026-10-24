# SPDX-License-Identifier: Apache-2.0
"""A photograph standing in for everything that never moves, through the camera that took it.

A twin of a fixed camera's clip can be made to look like the clip in two ways.
One is to model the room -- the wall, the curtain, the desk and its cables -- and
light it until it reads the same, which no renderer gets to the pixel. The other
is the one visual-matching benchmarks use: the clip already *is* a picture of
everything that does not move, so the render draws only what does -- the robot's
moving links, the things it handles, every deformable -- and every other pixel
comes from a clean plate of the clip (the clip with what moves taken out).
Through a calibrated camera the background is then the clip's own pixels, and
what is left to get right is the camera and what moves.

Which pixels are which is read off the tiled tracer's shape index: a pixel that
hit nothing, or a shape on the world or on a body held to the world by fixed
joints only (the bench, a fixture, a robot's base), shows the plate; a pixel
that hit anything else, a particle or a deformable's mesh shows the render.
Occlusion comes out right because the index is resolved in 3D: the near wall of
a static tray in front of a moving reel is a static pixel, so the plate's wall
is drawn over it.

The static geometry still does its physical job: it is the collider the robot
and the things touch. A collider that is not under its picture in the plate
shows as a thing resting on air or sunk into the table -- which is the reading
that tells a twin its camera or its layout is off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LABEL_MOVING",
    "LABEL_STATIC",
    "composite",
    "load_plate",
    "nearer",
    "shade",
    "shape_labels",
    "static_bodies",
    "static_shapes",
]

LABEL_MOVING = 1
"""The label a shape that can move takes in the composite's segmentation pass."""
LABEL_STATIC = 2
"""The label a shape on the world, or on a body held to it by fixed joints only, takes."""

_PLATES: dict[tuple[str, float, int, int], np.ndarray] = {}


def static_bodies(model: Any) -> np.ndarray:
    """Per body, whether it is held to the world by fixed joints only (see :func:`static_shapes`)."""
    import newton

    static_body = np.zeros(int(model.body_count), dtype=bool)
    if int(model.joint_count):
        jtype = np.asarray(model.joint_type.numpy())
        parent = np.asarray(model.joint_parent.numpy())
        child = np.asarray(model.joint_child.numpy())
        fixed = int(newton.JointType.FIXED)
        changed = True
        while changed:
            changed = False
            for j in range(len(jtype)):
                c = int(child[j])
                if 0 <= c < len(static_body) and not static_body[c] and int(jtype[j]) == fixed:
                    p = int(parent[j])
                    if p < 0 or static_body[p]:
                        static_body[c] = True
                        changed = True
    return static_body


def static_shapes(model: Any) -> np.ndarray:
    """Per shape, whether nothing in the simulation can move it.

    A shape on body ``-1`` is on the world. A body is static when the joint
    that makes it a child is fixed and its parent is the world or another
    static body; a free, revolute or prismatic joint anywhere up the chain makes
    it something that moves.
    """
    shape_body = np.asarray(model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else model.shape_body)
    static_body = static_bodies(model)
    body_static = np.concatenate([static_body, [True]])  # index -1 is the world
    return body_static[np.where(shape_body < 0, len(static_body), shape_body)]


def shape_labels(model: Any) -> dict[int, int]:
    """``shape -> label`` for the tiled tracer: :data:`LABEL_STATIC` or :data:`LABEL_MOVING`."""
    static = static_shapes(model)
    return {int(s): LABEL_STATIC if bool(static[s]) else LABEL_MOVING for s in range(len(static))}


def load_plate(path: str | Path, width: int, height: int) -> np.ndarray:
    """The plate as RGB ``uint8`` at ``width`` x ``height``; read again only when the file changes."""
    from PIL import Image

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no plate at {path.resolve()} (a CameraSpec's background is read beside scene.py)")
    key = (str(path.resolve()), path.stat().st_mtime, int(width), int(height))
    plate = _PLATES.get(key)
    if plate is None:
        with Image.open(path) as opened:
            img = opened.convert("RGB")
        if img.size != (width, height):
            w, h = img.size
            if abs(w / h - width / height) > 0.01:
                raise ValueError(
                    f"the plate {path.name} is {w}x{h} and the camera {width}x{height}: another shape of picture, "
                    "so it was not taken through this camera"
                )
            img = img.resize((int(width), int(height)), Image.LANCZOS)
        plate = np.asarray(img, dtype=np.uint8)
        _PLATES.clear()  # one plate per camera in use; an edited plate replaces its old copy
        _PLATES[key] = plate
    return plate


def shade(
    with_moving: np.ndarray, without_moving: np.ndarray, seg: np.ndarray | None = None, soften: float = 0.0
) -> np.ndarray:
    """How much light what moves takes off each pixel: ``with / without``, at most 1, softened.

    Two renders of the same camera through one light, differing only in
    whether the moving things are there (they are sent far below the floor
    for the second): where the first is darker, a moving thing's shadow falls.
    Debevec's differential rendering, reduced to the shadow term, which is the
    one a plate cannot have -- the clip's own shadows of what moves are where
    things were, not where they are. ``soften`` blurs it by that many pixels,
    since a room's light is broader than one lamp; the moving things' own
    pixels (``seg``) count as unshadowed, so the blur leaves no halo round them.
    """
    a = np.asarray(with_moving)[..., :3].astype(np.float32).mean(axis=-1)
    b = np.asarray(without_moving)[..., :3].astype(np.float32).mean(axis=-1)
    ratio = np.clip((a + 1.0) / (b + 1.0), 0.0, 1.0)
    if seg is not None:
        seg = np.asarray(seg).reshape(ratio.shape)
        ratio[(seg == LABEL_MOVING) | (seg < 0)] = 1.0
    if soften > 0:
        from scipy.ndimage import gaussian_filter

        ratio = gaussian_filter(ratio, sigma=soften / 2.0, truncate=2.0)
    return ratio


def nearer(
    seg: np.ndarray, depth: np.ndarray, static_depth: np.ndarray, grow: int = 2, slack: float = 0.004
) -> np.ndarray:
    """``seg`` with its moving pixels kept only where the render's ``depth`` is in front of the static scene's.

    ``seg`` comes off the colliders, ``depth`` off what the render drew: a
    rope's collider spheres are wider than its tube, and a visual mesh can
    overhang its collider. So the moving region is grown by ``grow`` pixels,
    and inside it a pixel is moving where the drawn surface is nearer than the
    static scene (``static_depth``, 0 where nothing static is hit) by more than
    ``slack`` metres plus a thousandth of the range -- under a rope lying on the
    bench, over the path tracer's own mat (0.75 mm). The rest is static.
    """
    from scipy.ndimage import binary_dilation

    seg = np.asarray(seg).reshape(np.shape(depth)[:2])
    moving = (seg == LABEL_MOVING) | (seg < 0)
    region = binary_dilation(moving, iterations=int(grow)) if grow > 0 else moving
    d = np.asarray(depth, dtype=np.float32).reshape(seg.shape)
    behind = np.asarray(static_depth, dtype=np.float32).reshape(seg.shape)
    behind = np.where(behind > 0, behind, np.inf)
    drawn = region & (d > 0) & (d < behind - (slack + 0.001 * d))
    out = np.where(moving & ~drawn, LABEL_STATIC, seg)
    return np.where(drawn & ~moving, LABEL_MOVING, out).astype(seg.dtype)


def composite(rgb: np.ndarray, seg: np.ndarray, plate: np.ndarray, shadow: np.ndarray | None = None) -> np.ndarray:
    """The render where ``seg`` says something that moves, the plate (darkened by ``shadow``) everywhere else."""
    rgb = np.asarray(rgb)[..., :3]
    seg = np.asarray(seg).reshape(rgb.shape[:2])
    moving = (seg == LABEL_MOVING) | (seg < 0)  # negative labels: particles and deformable meshes
    out = np.array(plate[..., :3], dtype=np.uint8, copy=True)
    if shadow is not None:
        out = np.clip(out.astype(np.float32) * shadow[..., None], 0, 255).astype(np.uint8)
    out[moving] = rgb[moving] if rgb.dtype == np.uint8 else np.clip(rgb[moving] * 255.0, 0, 255).astype(np.uint8)
    return out
