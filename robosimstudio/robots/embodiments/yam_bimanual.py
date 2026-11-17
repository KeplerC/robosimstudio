# SPDX-License-Identifier: Apache-2.0
"""The YAM's old module name, kept so a scene that imports from it still runs; use :mod:`.yam`.

Registers nothing: ``YAM_BIMANUAL`` is :data:`.yam.YAM` itself, named ``yam``.
"""

from .yam import *  # noqa: F403
from .yam import FINGER_OPEN, YAM

YAM_BIMANUAL = YAM

__all__ = ["FINGER_OPEN", "YAM", "YAM_BIMANUAL"]
