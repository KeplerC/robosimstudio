# SPDX-License-Identifier: Apache-2.0
"""What this task peels: an adhesive strip, and a protective film.

Both are bonded sheets over the same cohesive-zone model, which is why they are
one task's assets rather than two. The film subclasses the tape entity rather
than restating it -- a thinner sheet with its own measured constants, the same
bond arithmetic -- and that import is now intra-task rather than a reach into
another suite's package.

``layout`` is the third module and it is what the documents actually type:
the tape and the film, *placed* -- each declaration with the factory that puts
it on this suite's bench, under a name that says which layout it is.
"""

from __future__ import annotations

from . import (
    film,
    layout,
    tape,
)

__all__ = ["film", "layout", "tape"]
