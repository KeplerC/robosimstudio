# SPDX-License-Identifier: Apache-2.0
"""The scripted routines this suite's documents name.

Three of them, re-exported from ``solutions/`` for the reason that package's
docstring gives. Unlike ``tools``, these are resolved as *attributes* -- a
document names ``"routine:RoutineTunnel"`` and lowering fetches the class -- so
there is nothing to defer and the imports are ordinary.
"""

from __future__ import annotations

from .routine_decant import RoutineBoxDecant
from .routine_peel import RoutineFilmPeel
from .routine_prep import RoutineSurfacePrep
from .routine_tunnel import RoutineTunnel

__all__ = ["RoutineBoxDecant", "RoutineFilmPeel", "RoutineSurfacePrep", "RoutineTunnel"]
