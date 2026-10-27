# SPDX-License-Identifier: Apache-2.0
"""A finalized Newton model of a few bodies and particles, on the CPU, for a fast test.

Twenty-nine fast modules reach for ``newton.ModelBuilder()`` to get something a
function under test can call ``eval_fk`` on, look a label up in, or count the
bodies of. :func:`tiny_model` is that model, said once: free bodies at stated
positions, each with a small box so it has a shape, loose particles, and as
many replicated worlds as asked for. Finalized on ``"cpu"``, so it never lands
on a GPU however the machine running it is equipped.
"""

from __future__ import annotations

from collections.abc import Sequence

import newton
import warp as wp

__all__ = ["tiny_model"]

HALF = 0.01
"""Half extent of each body's box [m]."""


def tiny_model(
    bodies: Sequence[tuple[str, Sequence[float]]] = (),
    *,
    particles: Sequence[Sequence[float]] = (),
    worlds: int = 1,
) -> newton.Model:
    """``bodies`` as ``(label, (x, y, z))`` free bodies with a box each, ``particles`` at rest, in ``worlds`` copies.

    Body, shape and particle ``i`` of world ``w`` sit at ``w * per_world + i``,
    which is the layout :class:`~robosimstudio.core.worlds.WorldMap` states.
    """
    template = newton.ModelBuilder()
    for label, position in bodies:
        body = template.add_body(xform=wp.transform(tuple(float(v) for v in position), wp.quat_identity()), label=label)
        template.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF)
    for position in particles:
        template.add_particle(tuple(float(v) for v in position), (0.0, 0.0, 0.0), 0.1)
    if worlds == 1:
        return template.finalize(device="cpu")
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    return builder.finalize(device="cpu")
