# SPDX-License-Identifier: Apache-2.0
"""Getting Genesis initialised, once per process, without collateral damage.

Three things about ``gs.init`` are worth a module of their own.

**It is process-global and first-caller-wins**, like SuperDex's ``initialize``.
A second island in the same process gets whatever the first asked for, except
an IPC island asking for 64 after any island initialised float32 (a liquid's, or
an IPC island's that asked for 32), which is refused (below).

**It requires float64 for the two-way harvest, and this is not a preference.**
The IPC coupler's reaction is ``(k/dt^2) * m_link * (pos_ipc - pos_aim)``. At
equilibrium the physical force is fixed, so the displacement encoding it is
``dx ~ 1/(k*m)`` -- and at ordinary settings that is about one float32 ULP of a
world coordinate. Measured: at ``precision="32"`` the harvested force **collapses
to exactly zero** as the constraint stiffens or the mirror gets heavier, which is
precisely what you reach for when coupling feels loose. At ``precision="64"`` the
same sweep is flat across four orders of magnitude of stiffness and two of mass.
The cost is nothing measurable (57.2 vs 57.6 engine steps/s on the same scene) --
libuipc's barrier solve dominates and is double-precision regardless. Genesis's
own SAP-coupler documentation requires 64-bit for rigid-FEM coupling too.

**It mutates torch's process-global defaults.** ``gs.init`` calls
``torch.set_default_device`` and ``torch.set_default_dtype``; measured here,
the default device went ``cpu -> cuda:0``. This stack runs VLA policies and
LeRobot code in the same process, so the defaults are snapshotted and restored.
Without that, a policy silently allocates on a device nobody chose, weeks later.

Device selection is the launcher's job and never this module's: libuipc builds
its engine as ``Engine("cuda", workspace)`` with no device argument, so
``CUDA_VISIBLE_DEVICES`` (with ``QD_VISIBLE_DEVICE`` and ``EGL_DEVICE_ID``) must
already be set when this runs. Setting them here would be a silent no-op once a
CUDA context exists, which is the worst possible shape for that failure.
"""

from __future__ import annotations

import logging
import os
from typing import Any

__all__ = ["ensure_initialized", "require_genesis"]

log = logging.getLogger(__name__)

_ENGINE: dict[str, Any] = {"precision": None, "coupler": None}
"""Process-global engine state: the precision ``init`` actually ran with, and whose island asked for it."""


def require_genesis():
    """Import Genesis, or raise naming the extra. The only import site."""
    try:
        import genesis as gs
    except ImportError as error:
        raise ImportError(
            "the 'genesis' backend needs the genesis-world wheel (and libuipc for cloth and soft "
            "bodies): install the [genesis] extra in a virtual environment of its own, with this "
            "checkout on PYTHONPATH (the commands are in pyproject.toml's [genesis] comment). It "
            "ships for Linux/Windows x86 with an NVIDIA GPU only."
        ) from error
    return gs


def ensure_initialized(precision: str = "64", *, coupler: str = "ipc") -> Any:
    """Initialise Genesis once per process. Returns the module.

    Args:
        precision: ``"64"`` -- see the module docstring. ``"32"`` is accepted so
            a caller can measure the difference, and warned about loudly,
            because a scene that harvests wrenches at 32 reads zero rather than
            reading wrong, which is much harder to notice. An IPC island asking
            for 64 in a process an earlier island initialised at 32 is refused,
            and the refusal names whose island that was: a liquid's, or an IPC
            island's that asked for 32.
        coupler: ``"ipc"`` or ``"legacy"``: whose harvest the precision serves. The
            legacy coupler's reaction is a momentum change, which float32 reads
            (:mod:`robosimstudio.engines.genesis.sph`), so it asks for 32 without
            the warning.
    """
    gs = require_genesis()
    if _ENGINE["precision"] is not None:
        if str(precision) == "64" and _ENGINE["precision"] == "32" and coupler == "ipc":
            # An island built first in this process set float32 (a liquid's by choice, or an IPC
            # island measuring the difference), and an IPC island now wants 64: its harvest would read zero.
            earlier = (
                "a liquid island's, as rss.WATER_60HZ states"
                if _ENGINE.get("coupler") == "legacy"
                else "an IPC island that asked for 32"
            )
            raise RuntimeError(
                f"genesis was initialised at precision=32 by an earlier island in this process ({earlier}), and "
                "gs.init runs once a process: this IPC island, which asks for 64, would run its two-way harvest at "
                "float32, where it reads ZERO for a stiff constraint. Build the IPC scene in a process of its own."
            )
        elif str(precision) == "64" and _ENGINE["precision"] == "32":
            # A liquid island that asked for 64 gets 32: its harvest still reads, as a momentum change.
            log.warning(
                "genesis was initialised at precision=32 by an earlier island in this process, so this liquid "
                "island, which asks for 64, steps its SPH at float32: its reaction still reads (a momentum "
                "change), but not at the precision it asked for. Build it in a process of its own for float64."
            )
        elif str(precision) == "32" and coupler == "legacy" and _ENGINE["precision"] == "64":
            # An IPC island initialised the process at 64 first: the liquid is right, and slow.
            log.warning(
                "genesis was initialised at precision=64 by an earlier island in this process, so this liquid "
                "island steps its SPH at float64 too: right, and about 8x slower per substep (1.93 ms against "
                "0.25 ms for 1,728 particles, p2_cost.py). Build it in a process of its own for its own budget."
            )
        elif str(precision) != _ENGINE["precision"]:
            log.debug(
                "genesis already initialised at precision %s; ignoring %s",
                _ENGINE["precision"],
                precision,
            )
        return gs
    if str(precision) != "64" and coupler == "ipc":
        log.warning(
            "genesis initialised at precision=%s: the two-way wrench harvest is a difference of "
            "two nearly-equal world coordinates and reads ZERO at float32 for a stiff constraint "
            "or a heavy mirror. Use precision='64' unless you are deliberately measuring this.",
            precision,
        )
    elif str(precision) == "64" and coupler == "legacy":
        # backend("genesis") states no precision, and SolverGenesis's default is the IPC coupler's.
        log.warning(
            "genesis initialised at precision=64 for a liquid island: its SPH is right and about 8x slower per "
            "substep than at 32 (1.93 ms against 0.25 ms for 1,728 particles, p2_cost.py), and its reaction reads "
            "at 32. State backend('genesis', substeps=20, precision='32'), as rss.WATER_60HZ does, unless an IPC "
            "island shares the process."
        )
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        log.debug(
            "CUDA_VISIBLE_DEVICES is unset; libuipc takes whatever device Genesis initialises on, "
            "and this stack runs one GPU per process by convention"
        )

    # gs.init mutates torch's process-global defaults; put them back.
    try:
        import torch

        before = (torch.get_default_device(), torch.get_default_dtype())
    except ImportError:
        torch, before = None, None

    backend = getattr(gs, "gpu", None) or gs.cuda
    try:
        gs.init(backend=backend, precision=str(precision), logging_level="warning")
        _ENGINE["precision"] = str(precision)
        _ENGINE["coupler"] = str(coupler)
    finally:
        # Also when gs.init raises part-way: the defaults may already be changed.
        if torch is not None and before is not None:
            after = (torch.get_default_device(), torch.get_default_dtype())
            if after != before:
                torch.set_default_device(before[0])
                torch.set_default_dtype(before[1])
                log.debug("restored torch defaults %s that gs.init changed to %s", before, after)
    return gs
